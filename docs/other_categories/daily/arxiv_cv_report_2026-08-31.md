time: 20260831

# Arxiv Computer Vision Papers - 2026-08-31

## Executive Summary

## 今日研究主题

本日入选论文集中在自动驾驶与机器人感知、三维重建/压缩、多模态对齐，以及视觉基础模型的效率与缩放规律。共同趋势是：利用几何、采集状态或结构化先验提高跨视角/跨传感器鲁棒性，并通过更高效的模型设计降低大规模视觉模型的计算成本。

## 特别值得优先阅读

- **How Far Can 5,500 Hours of Driving Take You?**：基于 1M–9B 参数视频扩散模型和最多 5,500 小时驾驶数据拟合模型规模、训练暴露量与验证损失的缩放规律，为自动驾驶生成模型的训练预算分配提供定量依据。
- **GAAT**：将局部几何对应可靠性显式引入多模态 UAV 感知，在稀疏查询融合前估计 token/query 级几何先验，适合关注真实平台运动和跨传感器错位的研究。
- **uScenes**：发布同步 RGB 与三维多波束声纳数据，覆盖水下机器人在弱光和散射环境下的三维感知问题，具有明显的数据集基础设施价值。

## 新兴方向

1. 几何感知的稀疏跨模态融合与跨视角指代检测；
2. 面向 UAV、水下机器人和大范围测绘的真实世界多模态数据集；
3. 3D Gaussian Splatting 的非均匀量化与视觉模型结构化剪枝；
4. 通过注意力头语义专门化和缩放定律提升视觉生成/多模态模型的效率。

## 阅读建议

若重点关注自动驾驶和机器人，建议优先阅读驾驶视频扩散缩放规律、GAAT、uScenes 与 Contact-Guided Exploration；若关注视觉基础模型效率，则优先阅读 Ariadne Attention、Cut-ViT 和 3DGS 压缩论文。
## 2026-08-31 arXiv CV/RO 执行摘要

本期 10 篇论文呈现出一个清晰趋势：视觉与机器人系统正在把“更强的单一模型”推进为“几何、数据、算力和部署约束共同设计”的系统。代表性工作覆盖自动驾驶视频生成、UAV 多模态对齐与大规模测绘、鱼眼视觉适配、VLM 的高效注意力、3DGS 压缩、视觉基础模型剪枝，以及面向真实接触动力学的移动操作。共同方法是将可靠性或结构先验显式化，再以稀疏交互、坐标锚定、非均匀量化或任务特定子空间保持计算效率。

最值得优先阅读的工作包括：

- **How Far Can 5,500 Hours of Driving Take You?**：在约 5,500 小时驾驶数据上系统拟合模型规模、训练曝光和算力缩放律；结果显示固定模型时延长训练比扩大模型更有效，同时 9B 模型仍具有更低渐近损失，并在 nuScenes 上取得强视频生成结果。其对自动驾驶生成模型训练预算具有直接工程价值。
- **GeoFF3D**：将地理平移与重力方向直接作为前馈 3D 重建的坐标锚，并通过分层足迹聚合解决大规模 UAV 航迹的近共线退化和块间接缝；在 UAVScenes 上相对 π³X+SLRF 将 completeness 误差降低约 48%，是大规模测绘部署方向的重要进展。
- **GAAT**：在跨模态融合前估计局部对应可靠性，用稀疏查询替代不可靠的全 patch 交互，并配套 StateBench 诊断采集状态；在分割、检测、变化检测等任务上均有稳定收益，适合研究真实 UAV 多传感器系统。
- **Semantic Head Specialization / Ariadne Attention**：把 ViT 注意力头的前景/背景专化转化为可测量的 SHS-Index，并据此设计混合注意力；在约 6.5 倍较低注意力计算量下接近 full attention，说明可解释诊断信号能够指导 VLM 编码器架构。

其他论文补充了这一方向的不同环节。DEX 用轻量调制器把透视预训练特征迁移到鱼眼深度和开放词汇分割，避免标定与重采样；Cut-ViT 通过 Gram 锚定保持任务相关子空间，在高稀疏率下显著降低时间和显存；Non-Uniform Quantisation 将渲染重要性用于 3DGS 的加权合并与 Lloyd–Max 量化，在 V-PCC/G-PCC 上获得明显 BD-Rate 降低。uScenes 提供同步 RGB 与 3D 多波束声呐数据，但目前缺少逐帧密集标注；A-PAIR 将空地指代检测改为 pair-level identity-consistent 评测，暴露航拍召回仍是瓶颈；Multi-Critic PPO 通过接触引导探索改善非抓取移动操作，仿真成功率达 94.1%，但真机合计成功率为 69.0%，反映 sim-to-real 接触与本体定位误差仍需解决。

正在形成的研究方向包括：以局部几何可靠性控制多模态交互；把物理/重力/地理先验直接嵌入基础模型；用任务相关子空间和感知重要性进行模型与表示压缩；以及用 scaling law 指导受限数据条件下的生成模型训练。建议完整阅读顺序为 **GeoFF3D、GAAT、自动驾驶视频缩放律、Ariadne Attention、DEX**；若关注机器人真实部署，再加入 **Multi-Critic PPO、uScenes**。

---

## Table of Contents

1. [How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models](#2608.28404v1)
2. [GAAT: Geometry-Aware Alignment Transformer for Multimodal UAV Perception](#2608.27971v1)
3. [Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs](#2608.28383v1)
4. [uScenes: A Multimodal RGB and 3D Sonar Dataset for Underwater Robot Perception](#2608.27795v1)
5. [From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation](#2608.27860v1)
6. [A-PAIR: A Benchmark and Identity-Consistent Grounding Framework for Air-Ground Cross-View Referring Person Detection](#2608.27997v1)
7. [GeoFF3D: Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping](#2608.28288v1)
8. [Non-Uniform Quantisation for 3DGS Compression](#2608.28272v1)
9. [Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency](#2608.28205v1)
10. [Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL](#2608.28140v1)

---

## Papers

<a id='2608.28404v1'></a>
## [How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models](https://arxiv.org/abs/2608.28404v1)

**Authors:** Victor Besnier, Anh-Quan Cao, Elias Ramzi, Spyros Gidaris, Tuan-Hung Vu, Andrei Bursuc, Eloi Zablocki, Matthieu Cord

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

Video generation for autonomous driving cannot follow the web-scale route: driving data is expensive to collect, bound by privacy requirements, and cannot be scraped at will, so models must make the most of a fixed corpus. We present a systematic scaling-law study of video diffusion models trained from scratch on driving data: a family of models from 1M to 9B parameters, trained at different exposures on up to 5,500 hours of driving. Validation loss follows consistent power laws in both model size and training exposure, answering the questions that shape a training budget: whether compute is better spent on longer training or on a larger model, and whether more data is needed. Loss improves much faster with training exposure than with model size, making longer training the most effective way to improve a fixed model under limited compute. However, larger models continue to achieve lower asymptotic loss, so compute-optimal scaling still favors increasing model size when sufficient compute and data are available. Guided by these laws, we train a 9B-parameter model, to our knowledge the largest video diffusion model trained from scratch on driving data: it sets a new open-source state of the art for driving video generation, as measured on nuScenes. Our code and pretrained models are available at https://github.com/valeoai/VATIX. NATIX is separately releasing the underlying driving data in stages.

### 论文解读

#### 研究问题

自动驾驶视频生成无法像语言模型那样无限抓取互联网数据：驾驶视频采集昂贵，受隐私约束，而且许多团队只有固定语料和有限 GPU。本文研究一个非常实际的问题：在约 5,500 小时驾驶数据不变时，新增算力究竟应投入更大的模型，还是更长的训练？“更多数据”又应理解为更多独特片段，还是允许重复看到同一片段？

#### 核心方法

作者从头训练一组条件流匹配视频模型，而不是在封闭的互联网视频模型上做适配。视频被切成 2.5 秒、9 Hz 的前视片段，经 Wan 2.1 VAE 编码后，由 DiT 式时空 Transformer 在潜空间中学习从高斯噪声到真实视频的速度场。模型输入首帧，训练时使用 25 帧、320×416 分辨率；模型规模覆盖 1.6M 到 1.1B 参数，另训练一个 9B 模型。作者进行了超过 200 次运行，分别改变参数量、训练曝光量和总 TFLOPs，并统一拟合 (L(x)=L_0+A x^{-\alpha}) 的渐近幂律。

这里的训练曝光量是模型实际看到的样本总数，因此同一片段重复出现也会计入。数据集约含 6.3M 个独特训练片段，而实验曝光量为 10M–28M，相当于约 1.6–4.5 次遍历。对于可控生成，作者再利用 OccAny 提取并过滤自车伪轨迹，按 11 类前进、刹车、转弯和倒车动作均衡采样，将 25 个鸟瞰 waypoint 注入 Transformer 的自适应归一化层。

#### 关键发现

三条缩放规律都相当稳定，但下降速度不同。模型规模拟合的指数为 0.2125，而训练曝光指数平均约 0.74，说明在模型固定时，延长训练通常比增加参数更快降低验证损失；不过更大的模型仍有更低的渐近损失，所以算力充足时，最优配置会逐渐转向更大模型。数据限制实验也很有启发：在固定曝光下，把独特视频从 5,500 小时减少到 55 小时，12.5M 样本时损失只从 0.0939 增至 0.0980；只有缩到 5.5 小时、导致接近 2,000 次重复遍历时才显著恶化。作者据此认为，当前训练阶段真正的瓶颈还不是独特驾驶数据，而是训练预算和模型容量。

这些规律被用于规划 9B 模型。根据此前较小模型得到的曲线，约 (10^7) 个样本时预期验证损失为 0.0753；实际训练到 (1.2\times10^7) 个样本后达到 0.0781，相对误差仅 3.6%，即使预测范围比拟合中最大模型大 8 倍。9B 模型在 NATIX 单帧条件生成上的四项指标为 FID-Inception 4.91、FID-DINO 60.81、FVD-I3D 37.16 和 FVD-VideoMAE 75.86；加入轨迹微调后分别改善到 4.03、42.16、24.68 和 44.95，轨迹 ADE 为 3.84 米。在 nuScenes 微调评测中，9B 模型的 FID-Inception/FVD-I3D 达到 2.72/25.50（Vista split）和 3.61/31.52（Epona split），整体超过论文比较的开源方法。

#### 局限与意义

该结论最适合固定规模、数据昂贵的视频世界模型预算规划，尤其能帮助团队判断“先多训一段时间”是否比“立刻扩容”更划算。局限包括 1.1B 到 9B 之间缺少 3B–4B 中间模型，外推仍可能受尺度间隙影响；9B 为稳定训练在中途降低学习率，而拟合规律主要假设训练配方不变；冻结的 VAE 也未纳入缩放分析。此外，9B 与 1B 在 nuScenes 上的差距小于在预训练数据上的差距，部分视频指标甚至略差，提示轨迹微调的学习率和迭代数仍需针对大模型重新设计。总体而言，本文的价值在于把“重复曝光”纳入驾驶视频缩放律，并用可检验的经验曲线支撑 9B 级模型的训练决策。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28404v1)
- [arXiv](https://arxiv.org/abs/2608.28404v1)

---

<a id='2608.27971v1'></a>
## [GAAT: Geometry-Aware Alignment Transformer for Multimodal UAV Perception](https://arxiv.org/abs/2608.27971v1)

**Authors:** Jingpu Yang, Debin Tang, Yilin Sun, Fengxian Ji, Jiahua Zhu, Wenrui Ding, Yufeng Wang

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

Unmanned aerial vehicle (UAV) multimodal perception integrates visible (RGB), infrared (IR), synthetic aperture radar (SAR), and depth sensors for scene understanding under diverse conditions. However, differences in optics, resolution, and mounting often limit practical systems to global or image-center alignment. After tokenization, parallax, platform motion, and lens distortion can shift corresponding patch centers across modalities, weakening the spatial correspondence assumed by dense contrastive learning and cross-modal fusion. We propose GAAT (Geometry-Aware Alignment Transformer), an alignment-first pretrained model that estimates local correspondence reliability before cross-modal interaction. GAAT introduces syncPATC, which learns patch-center consistency under synchronized view transformations without correspondence annotations. It emits geometric priors, including token and query confidence, query centers, and sub-token offsets, that identify reliable local anchors across residual misalignment. Guided by these priors, MG-Sparse-MMA performs query-mediated sparse fusion over top-K_s reliable regions, replacing dense all-patch interaction with geometry-calibrated local updates. RA-QCGCL aligns pretraining supervision with this sparse query bottleneck through reliable patch-to-patch, patch-to-query, and query-to-query contrastive branches. We introduce UAVMeta and StateBench, which provide four acquisition-state scores derived from platform telemetry and image statistics: camera reliability, observation scale, viewpoint stability, and flight maneuver complexity. Extensive experiments across six downstream tasks demonstrate consistently superior transfer performance, establishing GAAT as a state-of-the-art multimodal foundation model for UAV perception. StateBench further enables a systematic diagnosis of real-world acquisition conditions.

### 论文解读

无人机常同时搭载 RGB 与红外相机，但两种传感器的视角、分辨率、镜头和安装位置不同。即使图像经过中心对齐，切成 patch 后，同一索引也可能对应不同物理区域；视差、斜视、飞行运动和采集时间差会进一步造成局部错位。这会让对比学习把错误区域当作正样本，也会让跨模态注意力把车辆与道路或背景混合。论文提出 GAAT（Geometry-Aware Alignment Transformer），核心思想是先估计局部对应是否可靠，再进行跨模态交互。

GAAT 使用两个独立的 Swin-V2-Base 编码器处理 RGB 和红外输入，并加入模态特定的图像线索：RGB 分支增强 Sobel 与 Laplacian 高频边缘，红外分支增强 Sobel 热边界和 5×5 局部热对比度。训练时将同一个仿射变换同步施加到两种模态，旋转范围为 ±20°、平移范围为 ±24 像素、缩放范围为 0.85–1.15、剪切范围为 ±8°。由于变换本身已知，模型可以在没有人工对应标注的情况下学习几何一致性。

其中，syncPATC 通过可学习 query 的注意力中心定位候选区域，把中心映射到变换后的视图，并在半径为 2 的局部邻域中依据特征相似度和空间距离寻找匹配。匹配分布越集中，置信度越高；再通过逆变换循环一致性约束 query。模块输出 token/query 置信度、局部中心以及子 token 偏移。训练采用由易到难的视差课程，逐步增加大幅视图变换的影响。随后，MG-Sparse-MMA 按可靠性选择 4 个 query，只在每个 query 周围的 3×3 邻域内做可变形采样和门控的双向 RGB–IR 融合，最后把更新散射回原 token 网格。因此跨模态交互从全局密集计算缩减为少量 query 的局部计算。

监督模块 RA-QCGCL 将图像边缘/热对比度证据与几何置信度合成为联合可靠性，只保留可靠性最高的 25% token，并设置三种互补的对比目标：可靠 patch-to-patch 保留空间锚定，patch-to-query 把局部 patch 对齐到语义 query，query-to-query 在 patch 对应较弱时仍提供跨模态语义约束。预训练同时使用 mask ratio 0.6 的模态不对称重建、RGB teacher 蒸馏和模态缺失补全。

实验在多个分割、检测、分类、变更检测、跟踪和热红外新视角重建任务上验证迁移能力。KUST4K 分割取得 82.14 mIoU 和 92.41 mAcc；在同域 UAVMeta 上，RGB–IR 分割达到 67.79±1.18 mIoU，检测达到 44.86±3.76 mAP。DroneVehicle 检测 mAP 为 56.59，比 DDQ-DETR 高 6.49；CDD 和 LEVIR-CD 的最佳 F1 分别为 97.85 和 95.96；两个 M3OT 序列的 pooled HOTA 为 51.05 和 49.79。对于采集状态预测，GAAT 的 MSPA-4D 为 52.69，并取得最低的飞行动作复杂度误差 12.66。消融结果总体支持三模块协同：去掉任一模块都会在若干空间敏感任务上退化，但不同任务的波动也表明收益并非简单、均匀叠加。

实际意义在于，GAAT 更适合低空、视角变化大、RGB–IR 仅粗对齐的无人机感知，尤其是分割、检测、变化分析和目标关联等依赖局部空间精度的任务。它不需要精确的跨相机标定，还能减少高分辨率下的无效跨模态计算。不过当前验证主要限于 RGB–IR 和同步仿射扰动，真实局部形变、极端遮挡及更多传感器的效果仍需验证；UAVMeta 也只有 2,575 对图像，数据规模和标注覆盖有限。因此，将方法扩展到 SAR、深度或多光谱时，需要重新设计模态线索和可靠性标定。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27971v1)
- [arXiv](https://arxiv.org/abs/2608.27971v1)

---

<a id='2608.28383v1'></a>
## [Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs](https://arxiv.org/abs/2608.28383v1)

**Authors:** Chenhong He, Lei Li, Shicheng Li, Hanglong Lv, Lingpeng Kong, Qi Liu, Tong Yang, Shuhuai Ren

**Published:** 2026-08-28

**Categories:** cs.CV, cs.CL

**Abstract:**

Hybrid attention dominates frontier LLMs, yet Vision Transformers (ViTs) in multimodal LLMs lack a satisfactory hybrid design, with no consensus on why certain attention patterns work better. To fill this gap, we study ViT attention heads and find they differentiate into object- and background-specialist roles, a pattern most pronounced under full attention; we call this Semantic Head Specialization (SHS). We propose SHS-Index to quantify this specialization, show that it distinguishes full-attention from chunk-window ViTs, and find that it strongly tracks downstream benchmark performance. We then identify three structural factors that shape SHS---window interaction, token serialization, and local softmax allocation---and use them as design principles for hybrid attention. Guided by these factors, we design Ariadne Attention, a hybrid that matches full attention on 22 image and video tasks at 6.5x less attention compute. Our findings establish head specialization as a measurable property for diagnosing and designing principled hybrid ViT attention at the multimodal-LLM scale.

### 论文解读

#### 研究问题

高分辨率视觉输入会让 ViT 的全局自注意力产生昂贵的二次计算，因此多模态大模型常用非重叠窗口降低成本。然而，窗口化视觉编码器往往不如全注意力稳定。本文追问：全注意力究竟保留了什么信息？能否把这种机制转化为可测量的设计原则？

#### 核心发现与方法

作者发现，全注意力 ViT 中不同注意力头会形成稳定的语义分工：有些头偏向图像前景物体，有些头偏向背景；分块窗口中的头则更容易出现窗口边界造成的网格模式，前景—背景区分较弱。为量化这种现象，作者提出 SHS-Index。具体而言，先用图像分割掩码把视觉 token 标为前景或背景，再统计每个 token 从所有可见查询收到的注意力总量，用 AUROC 判断一个 head 能否区分两类 token，同时不区分“偏前景”还是“偏背景”的方向，最后对图像、层和 head 求平均。

在相同训练条件下，全注意力的 SHS-Index 为 0.606，分块窗口为 0.577。对 16 个开源视觉编码器和视觉语言模型的分析也得到清晰分离：11 个全注意力模型均值约 0.631，5 个分块窗口模型均值为 0.585；同一个 Qwen2.5-VL 视觉编码器搭配 3B 到 72B 的语言模型时，指标变化小于 0.002，说明信号主要来自视觉注意力结构，而非语言模型规模。

#### Ariadne Attention 如何工作

作者通过受控改动定位了三个关键因素。第一，非重叠窗口限制信息跨区域传播，改用左右各 64 个 token 的重叠滑动窗口后，SHS-Index 从 0.577 提升到 0.588。第二，滑窗本质上处理一维序列，因此需要与图像空间相符的序列化；将原有的二维块排列改为行主序后，指标进一步升至 0.600。第三，局部 softmax 会迫使模型把概率分配给窗口内的无信息背景，于是加入每个 head 的可学习 sink bias，作为不对应任何视觉 token 的“虚拟吸收通道”。它在多数局部结构中改善专门化，但并非无条件有效。

Ariadne 将这些原则组合起来：每个八层模块包含四层行主序滑窗加 sink、三层列主序滑窗加 sink，以及一层全注意力；模块重复四次形成 32 层视觉编码器。行列交替让局部信息沿两个图像方向传播，全注意力层则提供全局锚点。窗口大小选择为每侧 64 个 token，总上下文 128。

#### 结果与意义

在 20 个图像任务上，全注意力、原始分块窗口和 Ariadne 的宏平均分别为 40.92、37.35 和 40.40，Ariadne 与全注意力只差 0.52 分；加入两个视频任务后，三者平均为 39.21、36.07 和 38.93。Ariadne 相对分块窗口在 CV-Bench、V*、DocVQA、OCRBench 上分别提升 6.1、11.0、9.8、8.9 分，说明它尤其适合跨区域搜索、文档布局理解和 OCR。另一方面，ChartXiv-RQ、ChartQA、PixMo-Count 分别下降 1.6、1.2、1.9 分，精确几何推理和实例计数仍需要额外的结构设计。

在八个受控配置中，8k 训练检查点的 SHS-Index 与 22 任务平均性能相关性为 Pearson r=0.858（p=0.006），但作者强调这只是跨架构的诊断信号，不是保证每项任务收益的因果预测器。效率方面，Ariadne 的注意力计算量为 1.68T，而全注意力为 11.00T，降低 6.5 倍；在 896²、4096 个 token 时，完整 ViT 前向时间从 123.5 ms 降至 106.8 ms，在 1792² 分辨率下端到端节省达到 39.4%。

#### 局限与适用场景

实验主要采用单随机种子、统一的较小 ViT 与 Qwen2-0.5B 语言模型，尚未覆盖更大语言骨干、更多训练设置或与各种轴向注意力方法的公平从零比较。SHS 也不能充分描述计数和精确几何任务。总体而言，这项工作最有价值的地方不是提出一个孤立的新算子，而是提供了一个可观测的“头分工”指标：在追求高分辨率、低延迟视觉编码时，可先用 SHS-Index 检查局部结构是否保留了前景与背景之间的互补表示，再据此选择窗口交互和 token 排列方式。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28383v1)
- [arXiv](https://arxiv.org/abs/2608.28383v1)

---

<a id='2608.27795v1'></a>
## [uScenes: A Multimodal RGB and 3D Sonar Dataset for Underwater Robot Perception](https://arxiv.org/abs/2608.27795v1)

**Authors:** Trung Tien Dong, Zhenqi Wu, Aditya Penumarti, Zi-Hao Zhang, Micaiah Bartlett, Jane Shin, Xiaomin Lin

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

Robust perception is essential for the deployment of autonomous underwater robots. However, optical cameras become unreliable under poor illumination and backscatter. Forward looking (2D) acoustic sensors remain effective under these conditions, but they measure range and bearing while leaving elevation unresolved, creating an ambiguity that prevents individual sonar returns from being localized in three dimensional (3D) space. This complicates the sensor use for 3D scene understanding and precise object detection. We introduce \textbf{uScenes}, a multimodal underwater dataset containing synchronized 3D multibeam sonar point clouds and RGB imagery. The dataset contains 110 scenes and 95,834 synchronized observation, representing 277.6 minutes of data collected across multiple field sessions. uScenes establishes a foundation for underwater sensor fusion, cross modal representation learning and 3D scene understanding. Code and datasets are given at https://github.com/era-research-lab/uScenes.

### 论文解读

#### 研究问题与数据价值

水下机器人在低照度、浑浊和悬浮颗粒环境中工作时，RGB 相机容易受到光衰减、颜色失真和散射影响；传统前视二维声呐虽能在这些条件下工作，却通常只提供距离与方位，缺少仰角信息，单个回波无法唯一确定三维位置。uScenes 针对这一缺口，发布了在移动水下机器人上采集的同步 RGB 图像与三维多波束声呐点云。数据覆盖 110 个场景、95,834 对同步观测和 277.6 分钟记录，并包含潜水员、鱼类、水下机器人、平台、管道、结构物和洞穴通道等内容。

#### 采集与时间配对

数据由 BlueROV2 搭载 DWE 水下相机和 Water Linked Sonar 3D-15 采集。相机分辨率为 1280×720，约 30 Hz；声呐以 256×64 的距离采样、90° 水平视场和 40° 垂直视场工作，约 6 Hz，每帧回波数量为 1 至 12,627，平均 6,805。两种传感器刚性安装，但中心存在约 120 mm 的横向和 31.71 mm 的垂向偏移。

由于采样频率不同，作者为每个声呐观测选择时间戳最近的相机图像，并设置 83 ms 的最大时间差。最终配对的绝对时间差均值为 13.2 ms，中位数为 11.2 ms，95% 的配对不超过 32.8 ms。这个策略对场景级关联和跨模态学习很有用，但不能自动保证声呐点与图像像素的空间对应。

#### 三维声呐表示

对距离采样位置，方法依据水平和垂直视场把采样坐标线性转换成两个角度，再由距离和球坐标关系生成三维点：
\(p=r[\cos\psi\cos\theta,\cos\psi\sin\theta,\sin\psi]^\top\)。数据同时保存点的位置与声学信号强度。信号强度先进行距离补偿，再在每个声呐帧内按第 99 百分位归一化到 [0,1]，因此它是相对响应，不能被当作跨帧一致的绝对反射率。

数据组织为 RGB JPEG、N×4 浮点声呐数组和 JSON Lines 元数据；数组每行包含三维坐标及归一化强度，清单记录时间戳、时间差、帧标识和回波数量。场景类别用于检索和划分数据，不是逐帧目标检测标注。

#### 光声标定思路

论文特别强调“时间同步不等于空间注册”。为估计声呐坐标到相机坐标的刚体变换，作者设计了尺寸已知的混凝土块、8×8 圆头针板和不同深度的平头螺丝板。图像中选择靶点中心，声呐点云中选择对应回波，利用水下相机投影模型最小化鲁棒重投影误差；EPnP 用于初始化，随后进行非线性细化。平头螺丝产生更孤立的声学响应，有利于可靠地建立对应关系。

#### 结果、意义与局限

本文的验证重点是数据规模、传感器协议、同步质量、点云构造和标定设计，而不是提出新的检测或分割网络，因此没有可比较的下游 mAP 或 mIoU。数据集的价值在于把野外移动平台的真实运动和能见度变化，与可直接表达三维几何的声呐回波及 RGB 外观结合起来，适合研究 RGB-声呐融合、跨模态表征学习、声学辅助目标感知和三维场景理解。

需要注意的是，当前数据来自单一机器人和单一淡水地点，跨平台、跨地域及海水泛化能力尚未得到验证；缺少密集二维/三维标注，难以直接形成监督检测基准；逐线扫描中的平台或目标运动可能造成点云畸变；声学强度不可进行跨帧绝对比较；相机的有效针孔模型对水下舷窗折射的刻画也仍有限。此外，使用者若要把声呐点投影到图像，必须先完成独立的空间标定。总体而言，uScenes 更像是一个面向未来水下多模态研究的几何与时间基础设施，而不是已经封装好的端到端 benchmark。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27795v1)
- [arXiv](https://arxiv.org/abs/2608.27795v1)

---

<a id='2608.27860v1'></a>
## [From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation](https://arxiv.org/abs/2608.27860v1)

**Authors:** Rit Gangopadhyay, Alex Wong

**Published:** 2026-08-28

**Categories:** cs.CV, cs.AI

**Abstract:**

Vision foundation models are capable of generalizing across 3-dimensional (3D) scenes with high-fidelity estimates; their empirical success can be attributed to training on large-scale datasets of perspective images. However, when transferred to wide field-of-view (FoV) images, such as those captured by fisheye cameras, they return erroneous outputs due to a covariate shift stemming from the radial distortion on the image pixels. We propose a method to generalize vision foundation models to fisheye cameras. The crux of our method lies in a set of learnable parameters, termed Distortion Extenders (DEX), that model the fisheye distortion coefficients and the distributional shift between fisheye and perspective images encoded in the latent space. By minimizing a self-supervised alignment loss, DEX transforms the latent embeddings of fisheye images to resemble those of perspective images to recover high-fidelity estimates. DEX is architecture- and task-agnostic: We demonstrate DEX on monocular depth estimation and open-vocabulary segmentation for convolution- and Transformer-based architectures, where we consistently improve over baselines across indoor and outdoor fisheye datasets. As a byproduct, the activations of DEX can also be decoded to distortion coefficients to support camera calibration. Code available at: https://github.com/Suchisrit/DEX.

### 论文解读

#### 研究问题

视觉基础模型通常由海量透视相机图像训练而成。当它们直接处理鱼眼图像时，镜头造成的强径向畸变会改变局部像素排列，进而造成视觉特征分布偏移。结果是深度图在边缘区域失真，开放词汇分割也难以保持物体边界。常见的解决方案是先依据相机内参去畸变，或转换到全景、球面、切平面和立方体等表示，但这会带来重采样伪影、视场损失、额外延迟和标定依赖。为每种相机重新训练模型又需要大量鱼眼数据，并可能损害原模型的通用性。

#### 核心方法

论文提出 Distortion Extenders（DEX），把适配过程放在特征空间而不是输入图像空间中。原有视觉骨干和任务解码器全部冻结，只在编码器的多个块后加入轻量模块。每个模块包含若干可学习的 Extender 向量，以及一个低秩投影。输入特征先计算自己与 Extender 的相似度，再用 softmax 得到权重，最后将这些向量的加权组合加回特征。不同输入区域可以选择不同组合，因此模块能够用少量参数表达从轻微到强烈、从中心到边缘的不同畸变修正。

训练不需要真实鱼眼图像或鱼眼标注。作者从校准的透视图像出发，随机采样鱼眼模型参数生成对应的畸变图像；冻结模型处理原图得到参考输出，同时让 DEX 处理畸变图像，再把输出逆变换回参考坐标系并进行自监督对齐。对于深度估计，DEX 将输出从传统的相机 z 深度扩展为欧氏距离 (R=\sqrt{X^2+Y^2+z^2})，更适合大视场镜头；对于开放词汇分割，则直接对齐图像嵌入，使其继续与文本嵌入匹配。部署时鱼眼图像可以直接输入，无需去畸变、额外标定或更换任务头。

#### 实验结果

作者用多种室内和室外透视数据合成训练样本，并在真实鱼眼数据上进行零样本测试。以 UniDepthV2 为骨干，ScanNet++ 上的 RMSE 从 0.329 降至 0.200，δ1 从 0.671 升至 0.872；在畸变更强的 KITTI-360 上，RMSE 从 7.093 降至 1.663，δ1 从 0.262 升至 0.842。DEX 也优于 LoRA 和 Calibration Tokens 等参数高效适配方式，在不同深度骨干上平均带来约 19% 的 RMSE 改善和约 15% 的 δ1 提升。

在 WoodScape 开放词汇分割实验中，LSeg 的 mIoU 由 0.321 提升到 0.362，SED 的 mIoU 达到 0.449；这说明特征对齐策略并不局限于回归任务。联合深度与分割的 PanopticDepth 同样获得提升。与先去畸变再推理相比，DEX 在 ScanNet++ 上把点云 Chamfer 距离从 0.360 降到 0.117、F1 从 0.317 提高到 0.841，在 KITTI-360 上也从 2.428/0.196 改善到 0.343/0.887。模块规模很小，在 UniDepthV2 上约增加 2.8 MB 存储和 0.3 ms 推理时间。

#### 局限与意义

DEX 的上限由冻结基础模型决定：如果模型对训练场景或目标域本来就不可靠，自监督参考信号也会带来相应误差。训练主要依赖合成畸变，因此极端镜头、真实镜头的非理想成像以及完全未见的相机仍需进一步验证。作者还发现，前阶畸变参数更容易从模块激活中恢复，后阶参数的相对误差较大，但这些误差对图像外观的影响有限。

这项工作的价值在于提供了一个简洁的相机域扩展接口：已有透视基础模型无需重训，只需学习少量特征调制参数，就能服务于混合相机的自动驾驶、机器人和空间感知系统。其思路也可迁移到其他成像域变化，只要能构造可逆的训练扰动并提供可靠的原模型输出作为参照。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27860v1)
- [arXiv](https://arxiv.org/abs/2608.27860v1)

---

<a id='2608.27997v1'></a>
## [A-PAIR: A Benchmark and Identity-Consistent Grounding Framework for Air-Ground Cross-View Referring Person Detection](https://arxiv.org/abs/2608.27997v1)

**Authors:** Zhoupeng Guo, Xinjie Yao, Yunqi Zhu, Zhihe Fan, Siqi Zhao, Jianjun Chen, Yichen Dong, Yan Fan, Pengfei Zhu

**Published:** 2026-08-28

**Categories:** cs.CV, cs.MM

**Abstract:**

Air-ground cross-view referring person detection is a necessary component in the language-to-perception-to-control chain of collective embodied intelligence, grounding a language command into the same physical target before ground and aerial agents can coordinate downstream actions. Existing referring expression comprehension and open-vocabulary grounding methods do not jointly account for cross-view identity consistency, making them insufficient for Air-Ground Cross-View Referring Person Detection (AGCV-RPD), which involves similar pedestrian distractors, weak aerial appearance cues, and cross-view identity consistency. To study this problem, we introduce Air-Ground Paired Identity-Aware Referring (A-PAIR), the first comprehensive AGCV-RPD benchmark, containing 22,137 cross-view referring samples. To construct A-PAIR efficiently, we propose Factorized Annotation and Referential Alignment (FARA), a semi-automatic annotation framework that generates factorized referring descriptions and identity-consistency supervision at reduced cost. We propose Identity-Consistent Referring Grounding (ICRG), a framework that combines factorized referential grounding, candidate-completeness supervision, and cross-view consistency calibration for joint air-ground pair selection. ICRG improves ground, aerial, and pair-level detection over strong baselines, increasing pair F1 from 16.65% to 22.28%. These results show that AGCV-RPD requires paired detection and identity-consistent reasoning.

### 论文解读

#### 研究问题

空地协同系统常需要把一句话同时交给地面机器人和无人机执行。例如，“戴彩色条纹头巾、在绿色公交车附近的人”应当在街景和俯视画面中定位同一个人。现有指代表达理解或开放词汇检测通常分别处理每张图，可能在两个视角得到两个都很像的行人，却无法保证他们是同一身份；而航拍中的行人又小且纹理弱，使语言中的服饰线索难以直接使用。论文将这一任务定义为空地跨视角指代行人检测，并要求两个预测框都达到 IoU 0.5，同时满足同一目标身份。

#### A-PAIR 基准与标注

论文从 G2APS 的配对观测构建 A-PAIR，包含 22,137 个跨视角指代样本，训练、验证和测试分别为 15,497、2,213 和 4,427 个。每个样本同时提供地面图、航拍图、身份标签以及两种语言描述：一种强调跨视角相对稳定的外观，另一种描述当前视角中的道路、车辆、树木或建筑关系。其半自动标注流程 FARA 先按场景和身份配对观测，再用视觉语言模型从多个地面裁剪中提炼服饰或携带物等稳定外观短语，并在两个视角的关键帧中生成空间短语。经过受约束的传播和自动质量检查后，数据还提供全图行人框和跨视角身份正负对，从而能够同时研究语言定位、候选召回和身份配对。

#### ICRG 如何工作

ICRG 不把两张图当作两个独立的 grounding 问题，而是直接为地面—航拍候选对打分。第一步使用基于 GroundingDINO 的定位器，分别输入外观描述和空间描述；两路候选取并集，使外观清楚时能精确定位，外观模糊或人群拥挤时仍可借助空间关系保留目标。第二步使用全人检测器枚举每个视角的行人，并与语言候选合并，避免小航拍目标因语言分支漏检而永久消失。第三步对每个地面候选和航拍候选裁剪，用 ResNet-50 特征的余弦相似度学习跨视角身份兼容性。最终对检测置信度、语言匹配度和身份相似度进行加权，遍历候选组合并选择最高分的一对；权重通过验证集网格搜索确定。

#### 结果、局限与意义

在图像不重叠的测试集上，ICRG 的地面和航拍实例 F1 分别为 35.17% 和 21.89%，Pair Acc 为 12.54%，Pair F1 为 22.28%；强基线 GroundingDINO-T 的 Pair Acc/F1 为 9.08%/16.65%。消融显示，单独加入外观—空间分解便将 Pair F1 从 3.60% 提高到 20.88%，候选完整性进一步提高到 21.82%，身份校准达到 22.28%。这表明最大的收益来自避免把不同语义线索过早混成一句话，同时也说明身份校准主要是在已有候选中纠正配对，而不能挽回完全漏检的目标。

论文的绝对配对准确率仍然不高，航拍小目标、相似行人和重复空间布局是主要障碍。数据主要来自配对帧，尚未覆盖更长时序、更拥挤的人群和更广泛环境。总体而言，A-PAIR 把语言理解、跨视角感知和身份推理统一到一个可测量的配对决策中，适合用于无人机—地面机器人协同搜索、跨摄像头语言检索以及多传感器具身交互。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27997v1)
- [arXiv](https://arxiv.org/abs/2608.27997v1)

---

<a id='2608.28288v1'></a>
## [GeoFF3D: Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping](https://arxiv.org/abs/2608.28288v1)

**Authors:** Xiang Yang, Yongli Wang, Yunsheng Zhang

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

Existing feed-forward 3D reconstruction methods typically process a bounded number of images and recover cameras and geometry in local or internally normalized frames. Extending them to large-scale UAV mapping requires scalable multi-chunk processing and reliable aggregation, while full Sim(3) alignment can become unstable for near collinear trajectories. We present GeoFF3D, which combines a coordinate-anchored model with a spatial large-scale reconstruction framework (SLRF). The model uses georeferenced camera translations and optional geometric priors to predict camera poses and dense point maps directly in a gravity-aligned Z-up metric frame. SLRF partitions images into spatially overlapping chunks, propagates shared-view priors, and aggregates local reconstructions hierarchically, while remaining applicable to different bounded-view models. Across nine aerial mapping blocks, GeoFF3D achieves the best average reconstruction quality, improving F@5 from 0.829 for Pi3X + SLRF to 0.877. On long UAVScenes sequences, it reaches 0.848, compared with 0.687 for Pi3X + SLRF and 0.451 for the strongest evaluated SLAM/streaming baseline. GeoFF3D reconstructs 2,000 images in approximately five minutes, demonstrating scalable and robust large-scale UAV reconstruction.The code is available at https://github.com/yanxian-ll/GeoFF3D.

### 论文解读

#### 研究问题与核心思路

无人机测绘常包含数百至数千张倾斜影像，而前馈三维重建模型通常只能一次处理有限视图，并把场景恢复到局部或内部归一化坐标系。将多个局部结果拼接起来时，近共线飞行轨迹尤其棘手：相机中心可以对齐，但由于轨迹对 roll 和 pitch 的约束弱，点云仍可能整体倾斜；不同块对同一影像预测的深度不一致，也会造成重复表面、垂直错位和接缝。GeoFF3D 的关键改变是让地理参考平移直接定义预测坐标系，而不是先在任意坐标系重建、再用完整 Sim(3) 事后配准。同时，它提出空间大规模重建框架 SLRF，把有限视图模型扩展到大范围、多航带和长序列。

#### 坐标锚定模型如何工作

模型将图像特征与可用的射线方向、深度观测、地理参考平移和旋转先验融合。平移先验提供近似的尺度和空间位置，模型则在重力对齐、Z-up 的度量坐标系中预测相机位姿、每像素射线、正深度和几何置信度。为保证数值稳定，平移和深度会围绕块中心进行归一化，推理结果再恢复到地理尺度；先验只是条件信号，并非直接复制成预测结果。训练同时监督相机坐标系几何、锚定世界坐标系几何、相机位姿和重力方向，并在高分辨率微调阶段随机丢弃或扰动先验，因此可以处理不完整、有噪声的测量。

#### SLRF 的大规模拼接流程

SLRF 不按拍摄时间切块，而是先根据 GNSS 位置和 IMU 姿态估计每幅图像的地面覆盖范围，再用自适应二叉树划分空间紧凑的块。每个块由核心视图和邻域接缝视图组成，核心视图负责最终几何，接缝视图提供跨块重叠。推理从最接近场景中心的块开始向外扩展；共享影像的深度被缓存并作为邻块先验，低置信度区域被过滤，重复结果优先采用核心视图。

聚合时，叶块使用允许尺度、水平旋转和平移变化、但保持重力方向的相似变换；兄弟块之间只估计残余水平旋转和平移，再沿空间树自底向上合并。这个设计把相机先验、共享深度和重力约束结合起来，减少了自由的 roll/pitch 对齐在细长航迹上的不稳定性。

#### 实验结果与意义

作者在 9 个有参考几何的航测块上比较 GeoFF3D、VGGT+SLRF 和 Pi3X+SLRF，并在 8 条 UAVScenes 长序列上加入多种 SLAM/流式重建方法。所有方法使用最长边 518 像素的输入；GeoFF3D 的块预算为 30 视图。航测块平均 Accuracy 和 Completeness 分别为 2.72 m、2.59 m，F@1 和 F@5 为 0.267、0.877，优于 Pi3X+SLRF 的 3.34 m、3.07 m、0.230 和 0.829。长序列上的优势更明显：GeoFF3D 的 Accuracy/Completeness 为 4.14/2.28 m，F@1/F@5 为 0.319/0.848；Pi3X+SLRF 的 F@5 为 0.687，最强的所比较 SLAM/流式基线为 0.451。

消融实验显示，世界坐标监督是建立锚定坐标系的关键，重力监督能降低相机 up 方向误差；空间足迹分块、重力保持聚合和共享深度传播分别改善全局误差与接缝一致性。块大小 30 在精度和资源间较平衡，处理 2000 张图像约需五分钟，峰值显存约 16 GiB。

#### 局限与适用场景

该方法适合拥有粗略 GNSS/IMU 参考、需要统一度量坐标和大范围稠密重建的无人机测绘、地形测量、灾害评估与基础设施巡检。它并不消除先验质量的影响：加倍位姿噪声会使误差上升，只使用平移而缺少旋转先验时退化更明显。方法也没有在线全局束调整，困难场景仍可能出现边界不连续、重复表面或残余块间误差。因此，若任务缺少可靠重力或地理参考，或要求严格的全局优化精度，仍需结合传统几何优化或后端校正。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28288v1)
- [arXiv](https://arxiv.org/abs/2608.28288v1)

---

<a id='2608.28272v1'></a>
## [Non-Uniform Quantisation for 3DGS Compression](https://arxiv.org/abs/2608.28272v1)

**Authors:** Bert Van hauwermeiren, Patrice Rondao Alface, Adrian Munteanu

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, yet its high bitrate requirements pose significant challenges for storage and transmission. To enable practical applications and ensure interoperability within the 3DGS ecosystem, standardised compression formats are essential. In this paper, we propose a novel non-uniform quantisation scheme specifically tailored for 3DGS models. Our approach adapts to the underlying data distribution by applying importance-weighted quantisation and eliminating post-voxelisation redundancy through importance weighted merging. Extensive evaluations on benchmark datasets demonstrate that our method achieves state-of-the-art compression performance. Furthermore, the proposed scheme is compatible with any point-cloud-based representation and is intended as a formal contribution to the upcoming MPEG 3DGS compression standardisation activities.

### 论文解读

#### 研究问题与动机

3D Gaussian Splatting（3DGS）用大量带有位置、尺度、旋转、不透明度和球谐系数的高斯表示场景，能够实时合成新视角，但单个场景可能包含数百万个高斯，存储和传输成本很高。面向互操作性的G-PCC与V-PCC标准通常采用均匀量化，默认所有高斯和所有参数同等重要，因而没有利用数据分布及视觉影响的差异。本文提出一套不改变原始渲染表示的非均匀压缩流程，目标是在标准点云编码器上获得更好的率失真平衡。

#### 核心方法

方法首先估计每个高斯对最终渲染的影响。光栅化过程中，一个高斯对像素的贡献由自身透明度及前方遮挡的透射率共同决定；作者将其对全部像素的贡献平方求和，作为重要性权重。为降低全场景渲染开销，再用一个仅含205个参数的小型两层MLP，根据尺度、不透明度和位置预测该权重。

位置和属性随后使用加权Lloyd-Max量化。每个量化区间的重建值不是普通平均，而是按高斯重要性计算加权质心，使视觉影响大的参数得到更小的误差。由于直接保存全部重建值会造成元数据开销，作者再对相邻重建值的差分进行第二次量化。采用4个差分重建值、2 bit的典型设置时，解码端只需保存首值、少量差分值和索引，即可递推恢复完整量化器。

位置量化形成体素后，同一体素内的高斯会被合并。作者重点采用重要性加权的参数平均，并使用适合四元数双覆盖性质的Markley平均处理旋转；同时比较协方差空间和Log-Euclidean平均。为防止外观差异过大的高斯被错误合并，算法结合球谐、协方差、不透明度和重要性构造相异度，仅合并低于阈值的基元。合并后的属性再进行非均匀量化，最后接入G-PCC或V-PCC的熵编码流程。

#### 实验结果与意义

作者在MPEG 3DGS公共测试条件下评估大尺度场景及物体/人物捕获，使用PSNR、SSIM、IVSSIM、LPIPS和BD-Rate衡量质量。在大尺度场景上，采用V-PCC时，相对均匀量化的RGB-PSNR BD-Rate为-28.27%，LPIPS BD-Rate为-30.01%；采用G-PCC时对应结果为-48.08%和-44.14%。消融实验表明，位置量化带来的收益最大；小型MLP预测的重要性接近直接渲染估计，却显著减少预处理成本；重要性加权参数平均也优于不合并、简单平均及协方差平均。

该方法的代价主要位于编码前处理：大尺度场景的预处理约需116秒，但V-PCC编码时间从约760秒降至约352秒，解码时间仍约4秒，适合训练完成后的离线压缩和实时播放。它还与现有剪枝互补，并可将重要性替换为语义权重，以优先保护关键对象。

#### 局限性与适用场景

该方法依赖小型MLP跨场景近似重要性的能力，且前处理并非实时；在物体数据的部分G-PCC设置中，自适应体素化仍可能具有更好的率失真表现。FlexGaussian在单一工作点可能达到更高质量，但不适合直接接入有损熵编码。总体而言，本文更适用于需要标准化、跨平台解码和离线编码的3DGS传输场景，也适合作为其他点云式splatting表示的通用压缩前端。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28272v1)
- [arXiv](https://arxiv.org/abs/2608.28272v1)

---

<a id='2608.28205v1'></a>
## [Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency](https://arxiv.org/abs/2608.28205v1)

**Authors:** Jianjian Yin, Liulei Li, Tao Chen, Yi Chen, Yazhou Yao, Wenguan Wang

**Published:** 2026-08-28

**Categories:** cs.CV

**Abstract:**

Pruning visual foundation models has attracted considerable attention. However, existing methods focus on rigid point-to-point token alignment on a single dataset for pruning, suffering from two limitations: i) robustness degradation, and ii) task-specificity deficiency. To address these limitations, we propose a task-specific pruning pipeline, named Cut-ViT. Specifically, we first construct gram anchoring matrices from both spatial and semantic perspectives, and perform the subspace decomposition to extract the corresponding subspace bases. Basis-agnostic and residual constraints are then adopted to align the gram subspaces between the native and pruned DINOv3 models along spatial and channel dimensions, enabling subnetworks to inherit robust feature representations of native DINOv3. Furthermore, we design spectral entropy adaptation, which quantifies the information density of feature manifolds along spatial and channel dimensions, thereby adapting the pruning objective to specific downstream tasks. Experiments show that Cut-ViT requires approximately one minute on a single A100 GPU to obtain subnetworks at various sparsity levels, using only 20.9% of the time and 45.5% of the GPU memory compared with previous methods, while achieving SOTA performance on six tasks across nine datasets.

### 论文解读

#### 研究问题

视觉基础模型具有很强的迁移能力，但模型规模和计算量限制了它们在边缘设备上的使用。现有训练无关的一次性结构化剪枝，通常用 MSE 等损失逐个匹配原模型与剪枝模型的 token，并在通用数据集上寻找统一子网络。这种做法容易拟合局部数值和噪声，也无法适应分割、匹配、检测等任务对空间细节的不同需求。Cut-ViT 的目标是在不重新训练基础模型的前提下，快速产生面向具体任务的子网络。

#### 核心方法

方法以 DINOv3 ViT-B/16 为编码器，让原模型作为冻结教师，与共享权重的剪枝模型处理采样图像和加噪图像。设特征为 (F\in\mathbb{R}^{L\times D})，首先计算空间 Gram 矩阵 (S=FF^\top) 和通道 Gram 矩阵 (C=F^\top F)。前者描述 token 之间的布局关系，后者描述通道承载的语义关系。对两个矩阵进行 SVD，保留主导子空间；实验中使用 192 个主成分，累计解释约 98.9% 的能量。

剪枝模型不必逐个复制教师的特征坐标，而是最小化

\[
1-\|U_p^\top U_t\|_F^2,
\]

其中 (U_p) 与 (U_t) 是剪枝模型和教师的子空间基。这个目标衡量两个子空间的几何重叠，对基向量的旋转不敏感，避免把等价的坐标变换误认为表示错误。方法还惩罚剪枝特征在教师子空间之外的能量，以抑制无关激活。随后根据目标函数对参数 mask 的梯度估计经验 Fisher 重要性，并用 xNES 搜索不同稀疏度的结构化子网络。

任务适配来自谱熵。作者把 Gram 矩阵的奇异值归一化为能量分布，分别计算空间熵和通道熵，再用二者的相对大小动态加权两类约束。这样，分类任务可以更多保留全局语义，而分割、视频分割和语义匹配能够获得更高权重的空间结构。剪枝校准只需目标数据中的 1000 个样本，不需要标签；分类使用 CLS 表示，其他任务同时利用空间与通道信息。

#### 实验结果与意义

作者在九个数据集、六类任务上评估 10%—30% 稀疏度的模型。20% 稀疏度时，DAVIS-2017 视频目标分割达到 65.0 的综合 J&F、63.2 的区域相似度和 66.8 的轮廓准确度，相比 SnapViT 分别提升 4.2、4.4 和 4.0 个百分点；ADE20K 语义分割 mIoU 为 50.0，提升 1.7 个百分点。30% 稀疏度时，COCO 检测 mAP 达到 48.6，较 SnapViT 提升 2.8 个百分点；FG3DCar 语义匹配 PCK@0.05 达到 70.8，提升 5.1 个百分点。ImageNet 在 10% 稀疏度下 Top-1 准确率为 88.6%，提升 1.8 个百分点。用 ADE20K 校准、在 VOC 上测试时，20% 稀疏度 mIoU 为 74.7，也优于对比方法，说明子空间约束有助于保留跨域鲁棒性。

效率方面，方法在单张 A100 上约 61 秒即可完成剪枝，需要 10.7 GB 显存；对比方法分别约需 292 秒和 23.5 GB。消融实验显示，基无关约束、残差约束、任务适配以及空间和通道两类 Gram 信息具有互补作用；在 ADE20K 上，完整方案的 mIoU 达到 50.0，而基础设置为 48.4。使用谱熵动态权重也持续优于固定权重。该结果表明，剪枝时保留特征关系的几何结构，比追求逐 token 数值一致更适合压缩视觉基础模型。

#### 局限与适用场景

Cut-ViT 仍依赖 Gram 子空间能够代表基础模型的鲁棒信息，迁移到更多模型和任务时需要重新验证。谱熵权重目前是启发式规则，检测等任务还需要额外训练解码器；极高稀疏度下的表现也缺少充分证据。总体而言，它适合需要快速、无需基础模型再训练、又希望针对目标域生成不同压缩率模型的部署场景，尤其适用于密集预测、视频分割和语义对应等不能丢失空间结构的任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28205v1)
- [arXiv](https://arxiv.org/abs/2608.28205v1)

---

<a id='2608.28140v1'></a>
## [Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL](https://arxiv.org/abs/2608.28140v1)

**Authors:** Simone Tolomei, Mayank Mittal, Franco Angelini, Manolo Garabini, Paolo Salaris, Marco Hutter

**Published:** 2026-08-28

**Categories:** cs.RO

**Abstract:**

Non-prehensile manipulation offers versatile skills for moving and rearranging heavy or bulky objects, particularly when combined with a mobile manipulation platform. However, both model-based and model-free approaches struggle with the complex hybrid dynamics and the sparsity of the contact in these tasks. To address these challenges, we propose a contact-guided exploration strategy implemented within a Multi-Critic Reinforcement Learning (RL) framework. A dedicated exploration critic is trained with a dense contact-seeking reward that guides the end-effector toward meaningful contact points; its influence is progressively decayed to recover a task-optimal policy. We obtain candidate interaction points from a general-purpose grasping algorithm, enabling the exploration mechanism to generalise across various object geometries. We evaluate the approach on multiple tasks, including box pushing, chair transportation, and a dishwasher opening task. Finally, we validate the chair transportation policy through extensive experiments on a quadrupedal mobile manipulator, demonstrating deployable non-prehensile manipulation in the real world.

### 论文解读

#### 研究问题

四足移动操作机器人可以推、拉和滑动物体，但非抓取操作依赖短暂的单侧接触与摩擦，机器人还要同时协调底盘稳定性、机械臂可达性和物体运动。传统强化学习很难随机找到“接触到哪里、从哪个方向施力”的有效行为，正则化项甚至会让策略为了动作平滑而完全避开物体。本文关注的核心问题是：不依赖专家演示，如何让策略先发现接触，再学会稳定搬运。

#### 核心方法

方法把 PPO 的价值估计拆成三个相互独立的部分：任务完成、接触探索和动作正则化。训练每个回合开始时，从物体上生成的候选交互点中随机选一个；探索奖励鼓励机械臂末端靠近该点。椅子使用通用抓取算法从网格中得到 25 个候选位置，以避开复杂非凸表面上不可达或物理效果差的点；箱子则在可见表面均匀取点。

策略网络采用带 256 个隐状态单元的 LSTM，输出机械臂关节目标、底盘平面速度、转向速度和底盘高度。底盘命令交给预训练的行走控制器执行，因而学习重点是全身协调和接触行为。三个价值分支分别计算优势，最终按权重混合。任务权重固定为 0.75；接触探索权重在训练步数 5,000 到 10,000 之间从 0.1 降到 0.01，同时把正则化权重从 0.15 提高到 0.24。这个安排让策略早期主动寻找接触，后期不再执着于某个接触点，而是根据运输目标和物理稳定性调整动作。

#### 主要结果

在 4,096 个并行仿真环境、五个随机种子下，椅子搬运成功率达到 94.1%，倾翻率为 4.4%，平均完成时间约 9.2 秒。没有探索奖励时成功率为 0%；把权重调度直接施加到单一标量奖励，虽然失触率降至 4.0%，但训练仍不稳定；固定权重的多 critic 能找到接触，却容易持续追踪接触点并增加倾翻。箱子推动成功率超过 90%。在洗碗机任务中，策略学会先利用把手启动开门，再根据门的运动转向门板继续推开，并将机械臂接近关节限位的时间比例降低 59%。

真机测试使用 ALMA 四足移动操作平台，在四种未参与训练的 IKEA 物体上共完成 40/58 次，成功率为 69.0%。其中折叠椅达到 3/3，三脚桌达到 2/4，说明策略并非只记忆标准椅子外形。它还可以跟踪实时移动的目标、搬运总质量 6.5 kg 的椅子，并在人工推扰导致失触后重新接近和恢复操作。

#### 局限与意义

仿真到真机仍有差距：为避免自碰撞而进行的侧向接近会产生较大的转向变化，进而放大里程计误差；真机物体位姿还依赖外部运动捕捉系统。候选点生成借用了抓取启发，对极端非凸、可动或接触区域很少的物体仍需验证，探索权重的固定调度也可能增加调参负担。总体而言，本文的价值不在于增加一个普通距离奖励，而在于将“发现接触”和“完成任务”分开学习，再用课程式权重把两者衔接起来。这种设计适合需要全身协调、接触稀疏且物体形状变化较大的推拉搬运任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28140v1)
- [arXiv](https://arxiv.org/abs/2608.28140v1)

---

