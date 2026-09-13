"""Compare complete nano-vLLM scheduling/generation against the frozen SDPA path.

The reference run records greedy choices. The native run consumes those choices
to keep teacher-forced inputs identical, and independently verifies that its own
greedy choices match. This changes the sampler only inside the test process.
"""
import argparse
import atexit
import gc
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from bench_attention import sdpa_baseline

MODEL_KL_LIMIT = 1e-3  # nats, temperature=1, per request and decoding step


def reference_forward(self, q, k, v):
    from nanovllm.utils.context import get_context
    return sdpa_baseline(q, k, v, self.k_cache, self.v_cache, get_context())


def fp64_forward(self, q, k, v):
    from nanovllm.utils.context import get_context
    def high_precision(query, key, value, **kwargs):
        output = torch.nn.functional.scaled_dot_product_attention(
            query.cpu().double(), key.cpu().double(), value.cpu().double(), **kwargs)
        return output.to(device=query.device, dtype=query.dtype)
    return sdpa_baseline(q, k, v, self.k_cache, self.v_cache, get_context(), high_precision)


def run_case(model, settings, batches, expected=None, performance=False, diagnostic=False,
             high_precision=False, gold=None):
    from nanovllm import LLM, SamplingParams
    from nanovllm.layers.attention import Attention
    started = time.perf_counter()
    # After integration, native execution uses the real production forward method.
    forward = fp64_forward if high_precision else reference_forward if expected is None else Attention.forward
    override = patch.object(Attention, 'forward', forward)
    with override:
        llm = LLM(str(model), device='mps', **settings)
        try:
            counters = {'prefix_hits': 0, 'preemptions': 0, 'sample_calls': 0}
            original_allocate = llm.scheduler.block_manager.allocate
            def allocate(seq, num_cached_blocks):
                counters['prefix_hits'] += int(num_cached_blocks > 0)
                return original_allocate(seq, num_cached_blocks)
            llm.scheduler.block_manager.allocate = allocate
            original_preempt = llm.scheduler.preempt
            def preempt(seq):
                counters['preemptions'] += 1
                return original_preempt(seq)
            llm.scheduler.preempt = preempt
            traces = []
            errors = []
            sample_times = []
            def sample(logits, temperatures):
                choices = logits.argmax(dim=-1)
                if performance:
                    torch.mps.synchronize()
                    sample_times.append(time.perf_counter())
                    counters['sample_calls'] += 1
                    return choices
                actual = logits.detach().float().cpu()
                if not torch.isfinite(actual).all():
                    raise AssertionError('Model produced non-finite logits')
                sample_times.append(time.perf_counter())
                if expected is None:
                    traces.append((actual, choices.cpu()))
                    result = choices
                else:
                    index = counters['sample_calls']
                    ref, tokens = expected[index]
                    difference = actual-ref
                    ref_logp = ref.double().log_softmax(dim=-1)
                    actual_logp = actual.double().log_softmax(dim=-1)
                    ref_probability = ref_logp.exp()
                    kl = (ref_probability * (ref_logp-actual_logp)).sum(dim=-1)
                    error = {'max_abs': float(difference.abs().max()), 'mean_abs': float(difference.abs().mean()),
                             'relative_l2': float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(ref).clamp_min(1e-12)),
                             'kl_divergence_max': float(kl.max().clamp_min(0)),
                             'probability_error_max': float((ref_probability-actual_logp.exp()).abs().max()),
                             'greedy_matches': torch.equal(choices.cpu(), tokens)}
                    errors.append(error)
                    if high_precision:
                        traces.append((actual, tokens))
                    if gold is not None:
                        golden = gold[index][0]
                        native_rms = (actual-golden).square().mean(dim=-1).sqrt()
                        baseline_rms = (ref-golden).square().mean(dim=-1).sqrt()
                        # Compare two FP16 engines against independent FP64 attention,
                        # not against an arbitrary absolute logit budget near zero.
                        floor = torch.finfo(torch.float16).eps
                        ratio = native_rms / baseline_rms.clamp_min(floor)
                        error.update(native_gold_rms_max=float(native_rms.max()),
                                     sdpa_gold_rms_max=float(baseline_rms.max()),
                                     gold_error_ratio_max=float(ratio.max()))
                    if not diagnostic:
                        if not torch.isfinite(kl).all() or error['kl_divergence_max'] > MODEL_KL_LIMIT:
                            raise AssertionError(f'Prediction distribution differs at {index}: {error}')
                        if not error['greedy_matches']:
                            raise AssertionError(f'Greedy choices differ at sampler call {index}')
                    result = tokens.to(logits.device)
                counters['sample_calls'] += 1
                return result
            llm.model_runner.sampler.forward = sample
            outputs, timings = [], []
            for prompts, max_tokens in batches:
                torch.mps.synchronize()
                begin = time.perf_counter()
                sample_start = len(sample_times)
                result = llm.generate(prompts, SamplingParams(temperature=.6, max_tokens=max_tokens, ignore_eos=True), use_tqdm=False)
                torch.mps.synchronize()
                duration = time.perf_counter()-begin
                assert all(len(item['token_ids']) == max_tokens for item in result)
                emitted = sum(len(item['token_ids']) for item in result)
                calls = sample_times[sample_start:]
                timings.append({'elapsed_s': duration, 'tokens_per_s': emitted/duration,
                                'first_sample_s': calls[0]-begin,
                                'sample_intervals_s': [b-a for a,b in zip(calls,calls[1:])]})
                outputs.append([item['token_ids'] for item in result])
            if expected is not None and not performance:
                assert len(expected) == counters['sample_calls']
            metrics = {'counters': counters, 'timings': timings, 'logit_errors': errors,
                       'output_tokens': outputs, 'including_load_s': time.perf_counter()-started,
                       'allocated_bytes': torch.mps.current_allocated_memory(),
                       'driver_bytes': torch.mps.driver_allocated_memory()}
            return metrics, traces
        finally:
            atexit.unregister(llm.exit)
            llm.exit()
            del llm
            gc.collect()
            torch.mps.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--diagnostic', action='store_true', help='Collect errors without marking acceptance passed')
    parser.add_argument('--fp64-reference', action='store_true', help='Also measure a CPU FP64-attention trajectory; this is diagnostic, not a whole-model 2x error gate')
    opts = parser.parse_args()
    assert torch.backends.mps.is_available(), 'Real MPS hardware is required'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(opts.model)
    pattern = tokenizer.encode('Explain how a computer stores and retrieves information. ')
    def tokens(length, offset=0):
        return [pattern[(i+offset) % len(pattern)] for i in range(length)]
    base = dict(max_num_seqs=4, max_model_len=1024, max_num_batched_tokens=1024, num_kvcache_blocks=16)
    cases = [
        ('normal', base, [(['The capital of France is', 'Explain what an attention mechanism does.'], 16)]),
        ('cross-page', base, [([tokens(255)], 16)]),
        ('shared-prefix', base, [([tokens(520)], 8), ([tokens(512)+tokens(9,2), tokens(512)+tokens(13,4)], 16)]),
        ('chunked-prefill', {**base, 'max_num_batched_tokens': 128}, [([tokens(777)], 16)]),
        ('preemption', {**base, 'num_kvcache_blocks': 2, 'max_model_len': 512, 'max_num_batched_tokens': 512}, [([tokens(255), tokens(255, 3)], 16)]),
    ]
    if opts.performance:
        cases = [(f'performance-b{b}', {**base, 'max_model_len': 1536, 'max_num_batched_tokens': 4096, 'num_kvcache_blocks': 32},
                  [([tokens(1024, i) for i in range(b)], 128)]) for b in (1, 4)]
    if opts.case:
        cases = [case for case in cases if case[0] == opts.case]
        assert cases, 'Unknown case'
    report = {'torch': torch.__version__, 'model': opts.model.name,
              'attention_module': 'flash_attn_mps', 'cases': [],
              'model_criterion': 'Finite logits; per-request, per-step KL(P_SDPA || P_native) <= 0.001 nats at temperature 1.0; identical greedy choices. Kernel pointwise tolerances are independently enforced.',
              'timing_note': ('Greedy generation with identical synchronization; no logit readback/comparison in timed runs.' if opts.performance else
                              'Correctness runs include logit readback/comparison and are not performance claims. First sample is TTFT only for single-chunk prefill.')}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    for name, settings, batches in cases:
        print('REFERENCE', name, flush=True)
        reference, trace = run_case(opts.model, settings, batches, performance=opts.performance, diagnostic=opts.diagnostic)
        gold_metrics, gold_trace = None, None
        if opts.fp64_reference and not opts.performance:
            print('FP64 ATTENTION', name, flush=True)
            gold_metrics, gold_trace = run_case(opts.model, settings, batches, trace, diagnostic=True, high_precision=True)
        print('NATIVE', name, flush=True)
        native, _ = run_case(opts.model, settings, batches, trace, performance=opts.performance, diagnostic=opts.diagnostic, gold=gold_trace)
        assert reference['output_tokens'] == native['output_tokens']
        if name == 'shared-prefix':
            assert native['counters']['prefix_hits'] >= 2
        if name == 'preemption':
            assert native['counters']['preemptions'] > 0
        if name == 'chunked-prefill':
            assert native['counters']['sample_calls'] > 16
        report['cases'].append({'name': name, 'settings': settings, 'reference': reference,
                               'fp64_attention': gold_metrics, 'native': native, 'passed': not opts.diagnostic})
        opts.output.write_text(json.dumps(report, indent=2)+'\n')
        print('DIAGNOSTIC' if opts.diagnostic else 'PASS', name, native['counters'], flush=True)
        del trace, gold_trace
    report['passed'] = not opts.diagnostic
    opts.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
