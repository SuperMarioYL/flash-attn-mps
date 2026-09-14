# 其他 MPS attention 实现的同机对照

本页保留优化前发布版 **v0.1.0** 的对照结果。当前 `main` 的优化结果及真实模型测量见[性能优化报告](performance-optimization.md)。

实测时间：2026-09-13。Apple M1 Max、macOS 26.6.2、Python 3.12、PyTorch 2.14.0。所有比较固定 FP16、Hq/Hkv=16/8、D=128；分页场景 page=256。使用独立环境，禁用 CPU fallback 和 mtlflashattn 的全局 import shim。自建库使用 GitHub 发布的 v0.1.0，未修改其实现。

**结论：自建库不能称为普遍最快。prefill 与 HF Metal 接近；连续 KV 的单 token decode 明显落后于 SDPA。包含分页处理后，缓存 prefill 是优势场景，但优化后的分组 SDPA 在混合分页 decode 上仍更快。**

## 1. 算子 API 延迟

输入 Q/K/V、布局 view、固定 mask 和 cu_seqlens 在计时前准备。无 KV 写入、无分页 gather；各库公开 API 内部自己执行的分配/复制仍计时。统一 token-major B,S,H,D 存储，同一数值输入；SDPA/Flex 使用预先准备的 heads-second view，HF/自建库使用 packed view。

等长 causal 用 SDPA is_causal=True；Q=1 decode 的全部已有 KV 都可见，使用 attn_mask=None、is_causal=False；真正的矩形 chunk 用右下对齐 mask。Flex 是 backend=inductor、fullgraph=True 的真实 Metal 编译路径。

每个实现每场景 10 次 warmup、50 次测量、3 轮轮换顺序。下表为三轮 p50 的中位数，单位 **ms，越小越好**。[同轮原始数据](results/comparison-operators.json)

| 场景 | SDPA | flash-attn-mps | HF Metal | MTL auto→v2 | compiled Flex |
| --- | ---: | ---: | ---: | ---: | ---: |
| 短 prefill，B1，Q=K=128 | 0.315 | 0.304 | 0.326 | 0.381 | 2.580 |
| prefill，B1，Q=K=2048 | 4.759 | 3.442 | 3.442 | 8.889 | 226.721 |
| prefill，B4，Q=K=4096 | 46.009 | 44.677 | 44.219 | 120.496 | 3158.684 |
| decode，B1，Q1，K8192 | 0.326 | 1.678 | 1.678 | 3.184 | 93.752 |
| decode，B16，Q1，K8192 | 3.125 | 5.902 | 5.936 | 13.783 | 365.505 |
| 连续 KV chunk，B4，Q64，K576 | 0.597 | 0.485 | 0.509 | 1.030 | 20.789 |

B1/K8192 的连续 KV decode，SDPA 约比自建库快 5.16 倍；B16 同形状约快 1.89 倍。长 prefill 三者 SDPA/HF/自建库的差距更小，不应把几个百分点当成跨环境的稳定优势。

在另一次相同协议的非 Flex 复测中加入了 MTL 的显式 v1 模式。原始记录完整保留，未从两轮里挑各库最快数值拼表。[补测数据](results/comparison-operators-common.json)

| 场景 | MTL auto→v2 | MTL v1 |
| --- | ---: | --- |
| 短 prefill，B1，Q=K=128 | 0.392 | 未过统一数值门槛，不排名 |
| prefill，B1，Q=K=2048 | 9.200 | 未过统一数值门槛，不排名 |
| prefill，B4，Q=K=4096 | 123.596 | 未过统一数值门槛，不排名 |
| decode，B1，Q1，K8192 | 3.528 | 9.549 |
| decode，B16，Q1，K8192 | 14.558 | 29.938 |
| 连续 KV chunk，B4，Q64，K576 | 1.021 | 1.553 |

正确性先于速度：所有输出检查 finite，Q≤128 时全查询对 CPU FP64 参考；长 prefill 检查每个 batch 的 15 个固定边界/内部查询，容差统一 atol=0.003、rtol=0.01。MTL v1 在三个方形 prefill 场景超出该门槛；没有降低门槛给它排名。auto→v2 在所有主测场景通过。

## 2. 包含分页整理的完整调用

计时从 GPU cache、block_table、lengths、slot_mapping 和本轮 Q/K/V 开始。所有实现共用相同的原生 store_kvcache，避免把不同 KV writer 的差异误算到 attention；必要的 gather、padding 清零、mask 更新、长度读取、分组及输出归并均计入。

先前使用高级索引 gather 的策略不是最优：逐位比较（包括 NaN）证明 index_select 与它等值；单 cache gather 分别快 2.67×、9.10×。在此基础上增加单批 SDPA 和按实时长度分组 SDPA，避免只与较差的整理路径比较。

下表来自同一轮配对补测，10 warmup / 50 samples / 3 rounds；单位 ms。[原始数据与 gather 微测](results/comparison-paths-supplement.json)

| 策略 | 缓存 prefill：B4 Q64 K576 | 混合 decode：B16，K4096/8192 交替 |
| --- | ---: | ---: |
| 自建库直接分页 | 0.672 | 8.738 |
| SDPA + index_select + 单批动态 mask | 1.344 | 32.958 |
| SDPA + index_select + 按长度分组 | 1.488 | 5.399 |

**缓存 prefill：自建库约快于最佳已测 SDPA 策略 2.00×。混合分页 decode：分组 SDPA 约快于同轮自建库 1.62×，三轮均胜出。**

同样把 index_select 改善用于第三方库后，结果如下；这是另一轮配对测量，因此单列其自建库控制组，不与上一表的控制组混算比值。[第三方补测](results/comparison-paths-third-party-supplement.json)

| 策略 | 缓存 prefill | 混合分页 decode |
| --- | ---: | ---: |
| 自建库，同轮控制组 | 0.664 | 9.360 |
| HF + index_select + boolean pack | 2.242 | 36.705 |
| MTL auto→v2 + index_select + grouping | 1.869 | 16.888 |
| MTL v1 + index_select + grouping | 2.455 | 35.299 |

初始 fancy-gather 的全部六策略结果也保留，包括 compiled Flex；它们代表具体适配方式，不能称为各库的最佳实现。[初始完整调用数据](results/comparison-paths.json)

## 3. 已核实的限制与解释

- HF 固定 kernel revision 为 761199956ba9baffbc93e0a3e08933668f06cf7a，使用 torch214-metal-aarch64-darwin 构建；支持本次变长和右下 causal，但没有原生分页/with_kvcache 接口。完整路径中，gather/pack 紧接该扩展曾触发 Metal encoder assertion；需要的 MPS synchronize 已计入计时，没有藏到计时外。
- mtlflashattn 0.2.0 的 auto 在这台 M1/macOS26.6.2 上实际选择 v2；同时实测了可公开选择的 v1，未按 README 对硬件的概括猜测路径。其 dense API 内部布局复制计入耗时；varlen 本身包含 Python 请求循环，分页须由调用方组织。
- Flex 数字只描述 PyTorch 2.14/MPS。生成代码确认实际执行 Metal MMA；QK 使用 MMA，但 PV 是逐 key、逐输出元素更新，未发现可用公开 kernel_options 切换成另一条 MPS PV-MMA 路径。不能推广为 CUDA FlexAttention 的结论。[编译证据记录](results/comparison-flex-mps-feasibility.json)
- mps-flash-attn 0.6.3 当前依赖 torch>=2.5,<2.14，因此没有在本次固定2.14环境强装后排名。
- 本次输入形状和 FP16 数据类型之外，不推断性能。首调用只报告进程内初始化/编译，不能称为清空整个 OS shader cache 的冷启动；没有用混合运行后的 resident memory 宣称峰值显存优势。

## 4. 对原发布结果的影响

此前 1.19–4.33× 的结果仍然是对冻结的旧 nano-vLLM 路径成立的测量；它包含旧路径的逐请求循环、元数据回读和 KV gather，不能据此声称裸算子或最佳现有组合全面领先。

后续最值得优化的是单 token decode，并应同时对照无多余 mask 的 SDPA，以及 index_select + 实时分组的完整分页策略。prefill 则应以 HF Metal 和当前 SDPA 为接近的性能基线。此次只补比较脚本与数据，没有改动库实现或已发布 v0.1.0。

## 复跑与来源

脚本：benchmarks/compare_operators.py、benchmarks/compare_paths.py。使用独立环境 torch==2.14.0、kernels==0.16.1、mtlflashattn==0.2.0、flash-attn-mps==0.1.0，进程启动前设置 MTLFLASHATTN_SHIM=off，禁用 PYTORCH_ENABLE_MPS_FALLBACK。

- [HF Metal kernel](https://huggingface.co/kernels/kernels-community/metal-flash-sdpa)
- [mtlflashattn 0.2.0](https://pypi.org/project/mtlflashattn/0.2.0/)
- [mps-flash-attn 0.6.3 metadata](https://pypi.org/pypi/mps-flash-attn/0.6.3/json)
- [PyTorch MPS Flex code generator](https://github.com/pytorch/pytorch/blob/v2.14.0/torch/_inductor/codegen/metal_flex_attention_template.py)
