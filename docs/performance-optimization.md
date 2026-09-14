# MPS Attention 性能优化 - main 分支验证

优化版本主要改善了单 token decode：最终完整测量中，B16 连续 decode 从发布版的 **5.815 ms 降到 1.753 ms**；混合长度分页 decode 从 **8.703 ms 降到 1.527 ms**。Qwen3-0.6B 的 B1/B4 生成吞吐分别提高约 **21.7% / 18.8%**。普通长 prefill 基本保持原有水平，部分场景仍慢于外部实现。

本报告验证 `main` 中基于提交 `336d1f5` 完成的性能优化。公开的 [v0.1.0](https://github.com/SuperMarioYL/flash-attn-mps/releases/tag/v0.1.0) 仍是优化前版本；优化源码已纳入主分支，尚未发行新的 Release。本报告描述本机 MPS 验证。

## 测量范围与门槛

环境为 Apple M1 Max、macOS 26.6.2、Python 3.12.13、PyTorch 2.14.0。禁用 MPS CPU fallback 和 MTL 自动 shim；发布版从已安装的非 editable wheel 加载，候选从源码加载，使用独立 Python 包别名。最终算子和模型报告都确认运行前后源码 SHA 未变。

每个算子实现预热 10 次，测量 50 次，执行 3 轮并轮换顺序。表中延迟是**三轮 p50 的中位数**，单位均为 **ms**，没有跨运行挑选最小值。全部输出检查有限性；Q≤128 时对所有 query 使用 CPU FP64 参考，长 prefill 检查每个 batch 的 15 个固定边界/内部 query。除 BF16 沿用既定 `atol=.02, rtol=.05` 外，其余比较统一为 `.003/.01`。

自动性能门槛只覆盖 **6 个 FP16 算子场景和 2 个 FP16 分页场景**：候选 p50 不超过发布版及有效外部对照中最小 p50 的 **1.10 倍**。7 个扩展功能场景用于检查正确性和性能表现。最终报告的 `formal_performance_target_verified=true` 具有这个明确范围，不能解释为所有功能、所有轮次或所有硬件均达到最优。[完整最终测量](results/optimization-formal-final.json)

- **算子 API**：固定输入 view、mask 和长度元数据准备在计时外；公共 API 内部的分配、复制仍计时，不包含 KV 写入或分页收集。
- **分页完整调用**：所有实现共用发布版 KV writer，计入实时 GPU 元数据、`index_select` 收集、长度分组或有效位 mask、输出整理。HF 必需的 packing 和同步也计入，没有使用前一轮已证明较差的 fancy-index 基线。
- 外部对照使用 HF Metal revision `761199956ba9baffbc93e0a3e08933668f06cf7a`、MTL 0.2.0 和 PyTorch SDPA。MTL 运行时实际后端已记录；不能通过相同数值门槛的实现不参与排名。

## 同一次完整运行的全部 15 个场景

除表内注明外，均为 FP16、Hq/Hkv=16/8、QK/V head dimension=128。分页大小为 256，物理页顺序打乱，无效尾部填 NaN。`候选/外部` 越小越快；外部栏仅在此次实际测量且数值通过的实现中取最小值。

| 范围与场景 | 候选 p50 | v0.1.0 p50 | 最佳已测外部 API | 外部 p50 | 候选/外部 |
| --- | ---: | ---: | --- | ---: | ---: |
| 算子：B1，Q=K=128 | 0.2992 | 0.3162 | SDPA | 0.3098 | 0.966 |
| 算子：B1，Q=K=2048 | 3.3356 | 3.3301 | HF Metal | 3.3173 | 1.006 |
| 算子：B4，Q=K=4096 | 47.5329 | 47.7189 | HF Metal | 46.9150 | 1.013 |
| 算子：B1，Q1/K8192 | 0.3288 | 1.6299 | SDPA | 0.3087 | 1.065 |
| 算子：B16，Q1/K8192 | 1.7534 | 5.8154 | SDPA | 3.2097 | 0.546 |
| 算子：B4，Q64/K576 | 0.4848 | 0.4544 | HF Metal | 0.4921 | 0.985 |
| 分页：B4，Q64/K576 | 0.6354 | 0.6338 | SDPA 批处理 + index_select | 1.3070 | 0.486 |
| 分页：B16，Q1，K4096/8192 交替 | 1.5271 | 8.7029 | SDPA 实时分组 + index_select | 5.0788 | 0.301 |
| BF16：B2，Q128/K512 | 0.4625 | 0.4604 | HF Metal | 0.4802 | 0.963 |
| FP32：B4，Q1/K2048 | 0.4680 | 0.9073 | SDPA | 0.4178 | **1.120** |
| DiffKV：B2，Q64/K512，Dv=64 | 0.6300 | 0.7972 | SDPA | 0.6779 | 0.929 |
| FP8 + descale：B4，Q1/K2048 | 0.4236 | 1.6327 | 无等价直接 API | — | — |
| window=(127,0)：B2，Q128/K1024 | 0.6067 | 1.7569 | SDPA | 0.8094 | 0.750 |
| sink：B4，Q1/K2048 | 0.3391 | 1.7845 | SDPA sink adapter | 2.1633 | 0.157 |
| softcap=3：B2，Q64/K512 | 0.4190 | 0.9648 | MTL auto，实际 v2 | 0.6750 | 0.621 |

有三个必须保留的限制：

1. **FP32 decode 仍比 SDPA 慢约 12.0%**，不属于上述 8 个 FP16 性能门槛场景。FP8 行只能说明相对 v0.1.0 的改善，不能宣称胜过等价外部算子。
2. B1/Q1/K8192 最终三轮的候选/SDPA 比值分别为 **1.223、1.072、1.064**；首轮慢 **22.3%**。总体中位数通过 1.10 门槛，不意味着每轮都通过。
3. B4/Q64/K576 连续算子相对发布版慢约 **6.7%**，尽管比此次 HF 测量略快。自动门槛包含发布版，不能仅用“最佳外部”掩盖这项回退。MTL v1 的 window 场景有 19/524288 个元素未通过固定容差，因此未排名；其最大绝对误差为 0.00867。

前一次同样完整的 [formal-all 测量](results/optimization-formal-all.json) 也保留：当时 B1/Q1/K8192 为 **0.3479 / 0.3117 ms（候选 / SDPA）**，比值 **1.116**，整体性能门槛未通过。随后调整了 softcap 快路径和 decode 内重复检查，再对全部 15 个场景重测。两次运行的源码指纹不同；本表全部来自后一次完整运行，不拼接各次最优数字。小幅延迟差异仍包含运行波动，不能全部归因于单项代码修改。

## 保留的实现与未采纳的尝试

**单 token decode 使用专门的原生分段归约。** [分发入口](../flash_attn_mps/_attention.py#L134) 在适用的 Q1 场景进入 [_decode.py](../flash_attn_mps/_decode.py#L40)，避免把一个 query 填进 32-row prefill tile。内核直接按 GPU block table 读取分页 KV，在允许的形状中共同计算一对 GQA query，使用 FP32 partial output/LSE 稳定合并；FP8 descale、sink、窗口和 softcap 在内核内处理。分片数作为运行参数传入，不因上下文增长反复生成 shader。[Metal 实现](../flash_attn_mps/kernels/attention_decode.metal)

归约设计参考 PyTorch v2.14 的 DecodeAttention.h 及其 MLX 来源，保留 [PyTorch BSD 许可](../LICENSES/PYTORCH-BSD.txt)和 [NOTICE](../NOTICE)。确定性、ALiBi 或专用 mask 等不适合该分支的组合继续由原有原生路径处理，未删除接口能力。

**通用 feature 路径只对稳定属性做编译期专门化，并跳过不可见窗口块。** dtype、维度、是否启用特性进入 shader cache key，序列长度和 tensor 内容仍由 GPU 元数据决定。单纯专门化曾使 window 从 1.745 ms 变慢到 2.102 ms；加入可见范围的整块跳过后，配对探针降到 0.618 ms。边界块仍逐行应用 mask，允许远处 prefix 的特殊 mask 不使用这项窗口裁剪。[实现](../flash_attn_mps/kernels/attention.metal#L108) · [完整探针](results/optimization-generic-tuning.json)

**softcap 接入已有 tiled 快路径。** 在自然尺度的 QK score 上先应用 `cap*tanh(score/cap)`，再做 mask，避免把已屏蔽的负无穷变成可参与 attention 的有限数。保留原来的 32×16 tile；配对探针中 generic 0.852 ms、该路径 0.430 ms，MTL auto 0.729 ms。[实现](../flash_attn_mps/kernels/attention_fast.metal#L371) · [探针](results/optimization-softcap-tuning.json)

**减少重复的主机工作。** [统一长度 offsets](../flash_attn_mps/interface.py#L13)按形状缓存只读元数据；私有 decode 层移除已由公共入口保证的重复检查，保留自己负责的约束和转换。19 个边界用例仍被公共入口正确拒绝。配对延迟改善只有数微秒，报告不把它描述成大幅算力提升。[检查与配对结果](results/optimization-decode-validation-paired.json)

没有保留缺乏稳定收益的修改：

- [prefill tile 扫描](results/optimization-prefill-tuning.json)：另外四种 tile 在 B1/2048、B4/4096 两种形状都未超过原 32×16 配置。
- [half QK fragment](results/optimization-prefill-half-qk.json)：配对改善接近噪声，且并非每轮更快，保留原实现。
- [decode vector4](results/optimization-decode-vector4-conclusion.json)：B1/B16 的加速约 0.997×/1.003×，已恢复实验前代码。
- [split cap 32→16](results/optimization-public-decode-split-cap.json)：没有在目标 B1/K8192 建立稳定收益，保留默认上限 32。

## 正确性与真实模型

**166 项真实 MPS 测试通过，0 failure、0 error、0 skip。** 包括新增 decode 路径以及既有缓存、布局、FP8、不同 QK/V 维度、mask、LSE/合并和公共接口测试。[JUnit 原始结果](results/optimization-native-tests-final.xml)

与冻结 SDPA 路径比较的五个真实 nano-vLLM 生命周期场景——普通生成、跨页 decode、共享 prefix、chunked prefill、抢占恢复——全部通过，greedy token 完全一致；最大逐请求/逐步 KL 为 **9.813×10⁻⁵**，低于 0.001。实际观测到共享 prefix 命中 2 次、抢占 1 次、chunked 场景 22 次 sample 调用。这些带 logits 回读的运行用于正确性，不作为性能结果。[生命周期数据](results/optimization-nano-lifecycle.json)

下面是另一组独立的 **发布版 v0.1.0 / 候选**端到端测量：本地 Qwen3-0.6B，每请求输入 1024 token、生成 128 token，B1/B4 各 3 轮交替顺序。每个引擎先用不同首 token 的请求预热 128 token，再测量正常请求，测量阶段 prefix 命中和抢占均为 0。性能阶段保留原生产 sampler（temperature=0.6）；正确性另用既有 teacher-forced helper，不在计时中读取 logits 或比较结果。[完整模型结果](results/optimization-nano-final.json)

| Batch | 发布版吞吐 token/s | 候选吞吐 token/s | 吞吐提升 | 发布版 / 候选 TTFT ms | 发布版 / 候选 decode 间隔 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.389 | 33.342 | 21.7% | 224.524 / 225.931 | 34.801 / 28.316 |
| 4 | 89.519 | 106.371 | 18.8% | 784.259 / 803.991 | 38.699 / 31.381 |

两个 batch 都在每一轮取得更高吞吐；**TTFT 没有改善，B4 反而增加约 2.5%**。这里 TTFT 是引擎内部首个采样 token 就绪的时间，不是 HTTP 客户端的流式首包。decode 间隔是相邻 sample 完成时间之差；总吞吐包含 scheduler、采样和最终 tokenizer decode。另行比较的 128 步 release/development logits，B1/B4 最大 KL 分别为 **6.369×10⁻⁵ / 8.464×10⁻⁵**，greedy 选择全部一致。

模型加载后的 resident 内存快照如下。三轮的测量前后数值均一致，两个版本也相同；**这不是峰值内存测量，不能据此宣称峰值内存更低**。

| Batch | 发布版 allocated MiB | 候选 allocated MiB | 两版本 driver MiB |
| --- | ---: | ---: | ---: |
| 1 | 2060.1265 | 2060.1265 | 2952.6250 |
| 4 | 2060.1267 | 2060.1267 | 2952.6250 |

## 分发包检查

本地候选 wheel 在独立 Python 3.12 / Torch 2.14 环境中非 editable 安装后，随包 **166 项测试全部通过**；安装后的每个 Python/Metal 文件均与最终算子测量的源码指纹一致。sdist 也完成了独立构建和安装，单 token decode 与 softcap prefill 对 CPU FP64 参考通过。Metal 源码、第三方许可证及测试资源均在包内。这些是本地候选包检查，未覆盖或替换 GitHub 的 v0.1.0 资产。[安装记录](results/optimization-installed-validation.json) · [随包测试](results/optimization-installed-tests.xml) · [sdist 检查](results/optimization-sdist-validation.json)

nano-vLLM 保持 `bb823b3e06983d71485a8e1f23715ebd87d98ef8` 基线上的原有五个本地修改文件。模型测量已核对其源码在运行前后不变；本地差异另外保存在忽略目录 `.artifacts/optimization/nano-local.patch`，没有公开推送。

## 复跑

以下是本机已使用的环境与本地模型路径；工作目录为 `/Users/yulei/workspace/flash-attn-mps`。benchmark 会分别加载源码与已安装的 v0.1.0，记录实际路径和源码 SHA。

```bash
MTLFLASHATTN_SHIM=off PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
  .artifacts/operator-comparison/env/bin/python benchmarks/bench_optimization.py \
  --output .artifacts/optimization/recheck-formal.json
```

```bash
MTLFLASHATTN_SHIM=off PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
  /Users/yulei/workspace/nano-vllm/.venv/bin/python benchmarks/bench_nanovllm_optimization.py \
  --model /Users/yulei/huggingface/Qwen3-0.6B \
  --output .artifacts/optimization/recheck-nano.json
```

```bash
MTLFLASHATTN_SHIM=off PYTORCH_ENABLE_MPS_FALLBACK=0 PYTHONPATH=. \
  /Users/yulei/workspace/nano-vllm/.venv/bin/python -m pytest -q tests \
  --junitxml=.artifacts/optimization/recheck-tests.xml
```

算子脚本可用 `--suite`、重复的 `--case` 和 `--quick` 筛选；模型脚本可用 `--mode correctness` / `--mode performance` 分开执行。快速或部分运行不会被标记为完整正式验证。原始 JSON/XML 均保留在 `docs/results/optimization-*`，包括前一次未通过性能门槛的完整运行。
