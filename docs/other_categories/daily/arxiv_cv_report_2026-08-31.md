time: 20260831

# Arxiv Computer Vision Papers - 2026-08-31

## Executive Summary

# 今日研究主题

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

**Scores:** relevance 3 / significance 3 / combined 6

**Abstract:**

Video generation for autonomous driving cannot follow the web-scale route: driving data is expensive to collect, bound by privacy requirements, and cannot be scraped at will, so models must make the most of a fixed corpus. We present a systematic scaling-law study of video diffusion models trained from scratch on driving data: a family of models from 1M to 9B parameters, trained at different exposures on up to 5,500 hours of driving. Validation loss follows consistent power laws in both model size and training exposure, answering the questions that shape a training budget: whether compute is better spent on longer training or on a larger model, and whether more data is needed. Loss improves much faster with training exposure than with model size, making longer training the most effective way to improve a fixed model under limited compute. However, larger models continue to achieve lower asymptotic loss, so compute-optimal scaling still favors increasing model size when sufficient compute and data are available. Guided by these laws, we train a 9B-parameter model, to our knowledge the largest video diffusion model trained from scratch on driving data: it sets a new open-source state of the art for driving video generation, as measured on nuScenes. Our code and pretrained models are available at https://github.com/valeoai/VATIX. NATIX is separately releasing the underlying driving data in stages.

**Analysis:**

# How Far Can 5,500 Hours of Driving Take You? A Scaling Law Analysis of Video Diffusion Models

- **ArXiv ID**: 2608.28404v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28404v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 19 页；`agent_pdfs/part-001.pdf` 19 页；`full_text.txt` 覆盖 1 个 part，页数一致，无缺失页或提取错误记录

## 问题定义

自动驾驶场景的视频生成无法依赖互联网级数据爬取：驾驶数据昂贵、受隐私约束且规模固定。论文在**固定约 5,500 小时驾驶视频**、有限算力预算下，回答三个工程问题：应优先扩大模型、延长训练，还是增加数据？能否用缩放律外推旗舰 9B 模型的验证损失？

## 方法与设计

### 生成范式与架构
- **条件流匹配（CFM）**：学习从噪声 \(x_0 \sim \mathcal{N}(0,I)\) 到数据 \(x_1\) 的速度场 \(u_\theta(x_t,t,c)\)。
  - 线性插值路径：\(x_t = (1-t)x_0 + t x_1\)
  - 目标速度：\(u(x_t,t) = x_1 - x_0\)
  - 损失：\(\mathcal{L} = \mathbb{E}_{x_0,x_1,t}\|u_\theta(x_t,t,c) - (x_1-x_0)\|^2\)
- **DiT 时空 Transformer**：基于 Wan 2.1 VAE 潜空间；每层含空间/时间自注意力 + AdaLN 调制；输入 25 帧、320×416 前视 RGB，潜空间 7×16×40×52。
- **模型族**：1.6M—1.1B（拟合用）至 **9B**（旗舰）；≤1B 用 DDP，9B 用 FSDP2。

### 缩放律形式（统一）
\[
\mathcal{L}(x) = L_0 + A \cdot x^{-\alpha}
\]
- **模型缩放** \(x=N\)：\(\mathcal{L}(N) = 0.0595 + 0.1041 \cdot N^{-0.2125}\)（72 次运行，10M 样本固定曝光）
- **训练曝光** \(x=D\)：各尺寸单独拟合，\(\alpha_D \approx 0.74\)（0.66—0.84）
- **算力缩放** \(x=C\)：\(\mathcal{L}(C) = 0.0522 + 0.5955 \cdot C^{-0.15443}\)

### 数据与训练协议
- **NATIX 驾驶数据**：约 5,500 小时、307K 分钟级视频、28 国；切 2.5s/9Hz 前视 clip；IID 划分 260,853/15,404/30,822（train/val/test）。
- 曝光 10M—28M 样本 ≈ **1.6—4.5 epoch**（全库重复）。
- 轨迹后训练：OccAny 伪轨迹 + 11 类 ego-action 平衡采样；1B/9B 在 nuScenes 上 25K iter 全参数微调。

### 架构示意（图 1，第 4—5 页）
Wan 编码潜变量 → 加噪 → N 个 Transformer 块（空间注意力、时间注意力、MLP，AdaLN 调制）→ 预测速度；时间步与 ego 轨迹 waypoint 经 MLP 注入。

## 数据集与实验协议

| 用途 | 数据 | 协议要点 |
|------|------|----------|
| 缩放律拟合 | NATIX 训练集 | >200 次超参网格（lr、batch、seed） |
| 生成质量 | NATIX 测试（2,000 视频，11 类 ego-action 平衡） | FID(Inception/DINOv2)、FVD(I3D/VideoMAE)、ADE |
| 对标 SOTA | nuScenes（Vista 5,369 / Epona 1,690 split） | 1 帧条件 2.5s@9Hz |

**数据限制消融**（图 4，第 8—9 页）：Base 模型在固定曝光下将唯一 footage 从 5,500h 缩至 5.5h；100× 缩减（5500→55h）损失几乎不变，仅极端 ~2000 epoch 时劣化。

## 定量结果

### 缩放律外推（9B）
| 指标 | 预测 | 实测（1.2×10⁷ 样本） | 相对误差 |
|------|------|------------------------|----------|
| 验证损失 | 0.0753 | 0.0781 | 3.6% |

### NATIX 上按尺寸缩放（表 2，第 12—13 页，单帧条件 2.5s）
| 模型 | FID_Inception↓ | FID_DINO↓ | FVD_I3D↓ | FVD_VideoMAE↓ | ADE↓ |
|------|----------------|-----------|----------|---------------|------|
| Tiny | 26.87 | 280.19 | 175.99 | 198.40 | — |
| Base | 10.66 | 132.51 | 68.43 | 116.94 | — |
| Large | 8.67 | 117.76 | 48.09 | 99.43 | — |
| 1B | 5.61 | 87.05 | 33.94 | 89.49 | — |
| 9B | 4.91 | 60.81 | 37.16 | 75.86 | — |
| 1B* | 4.41 | 63.93 | 32.86 | 49.28 | 3.92 |
| 9B* | 4.03 | 42.16 | 24.68 | 44.95 | 3.84 |

### nuScenes 对标（表 3，第 13 页）
| 方法 | FID_Inception↓ | FID_DINO↓ | FVD_I3D↓ | FVD_VideoMAE↓ |
|------|----------------|-----------|----------|---------------|
| Vista | 6.9 | — | 89.4 | — |
| GEM | 10.5 | — | 158.5 | — |
| Epona | 7.5 | — | 82.8 | — |
| **1B (ours) Vista** | **3.10** | 132.86 | 28.46 | 30.69 |
| **9B (ours) Vista** | **2.72** | 87.12 | 25.50 | 32.58 |
| **9B (ours) Epona** | **3.61** | 92.73 | 31.52 | 39.75 |

## 消融与局限

**主要发现**（结论，第 14—15 页）：
1. 训练曝光指数远陡于模型规模指数 → 固定模型下延长训练最有效；算力最优仍倾向更大模型。
2. 固定语料内重复数据在 <5 epoch 内近似新样本（flow matching 每步重采样噪声/时间步）。
3. 9B 外推 8× 仍准确至 3.6%。

**局限**（第 15 页）：
- 拟合最大 1.1B，与 9B 旗舰存在尺度间隙；缺少 3B—4B 中间点。
- 缩放律假设恒定学习率，9B 需在 3.2M 样本后降 lr（10⁻⁴→10⁻⁵）以稳定训练。
- 未联合建模冻结 VAE 的缩放行为。

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 模型架构（图 1） | 4—5 |
| 模型缩放曲线（图 2） | 6—7 |
| 训练曝光 per-model 曲线（图 3） | 7—8 |
| 数据限制消融（图 4） | 8—9 |
| 算力缩放包络（图 5） | 10 |
| 9B 训练拟合（图 6） | 11—12 |
| 视觉质量对比 5s rollout（图 7） | 14 |
| 模型尺寸表（表 1） | 4 |
| NATIX 指标表（表 2） | 12—13 |
| nuScenes 对标（表 3） | 13 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28404v1)
- [arXiv](https://arxiv.org/abs/2608.28404v1)

---

<a id='2608.27971v1'></a>
## [GAAT: Geometry-Aware Alignment Transformer for Multimodal UAV Perception](https://arxiv.org/abs/2608.27971v1)

**Authors:** Jingpu Yang, Debin Tang, Yilin Sun, Fengxian Ji, Jiahua Zhu, Wenrui Ding, Yufeng Wang

**Published:** 2026-08-28

**Categories:** cs.CV

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Unmanned aerial vehicle (UAV) multimodal perception integrates visible (RGB), infrared (IR), synthetic aperture radar (SAR), and depth sensors for scene understanding under diverse conditions. However, differences in optics, resolution, and mounting often limit practical systems to global or image-center alignment. After tokenization, parallax, platform motion, and lens distortion can shift corresponding patch centers across modalities, weakening the spatial correspondence assumed by dense contrastive learning and cross-modal fusion. We propose GAAT (Geometry-Aware Alignment Transformer), an alignment-first pretrained model that estimates local correspondence reliability before cross-modal interaction. GAAT introduces syncPATC, which learns patch-center consistency under synchronized view transformations without correspondence annotations. It emits geometric priors, including token and query confidence, query centers, and sub-token offsets, that identify reliable local anchors across residual misalignment. Guided by these priors, MG-Sparse-MMA performs query-mediated sparse fusion over top-K_s reliable regions, replacing dense all-patch interaction with geometry-calibrated local updates. RA-QCGCL aligns pretraining supervision with this sparse query bottleneck through reliable patch-to-patch, patch-to-query, and query-to-query contrastive branches. We introduce UAVMeta and StateBench, which provide four acquisition-state scores derived from platform telemetry and image statistics: camera reliability, observation scale, viewpoint stability, and flight maneuver complexity. Extensive experiments across six downstream tasks demonstrate consistently superior transfer performance, establishing GAAT as a state-of-the-art multimodal foundation model for UAV perception. StateBench further enables a systematic diagnosis of real-world acquisition conditions.

**Analysis:**

# GAAT: Geometry-Aware Alignment Transformer for Multimodal UAV Perception

- **ArXiv ID**: 2608.27971v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.27971v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 26 页；`agent_pdfs/part-001.pdf` 26 页；`full_text.txt` 单 part 全覆盖，无缺失页

## 问题定义

UAV 多模态感知（RGB + 红外等）在粗对齐后，**patch 中心仍可能错位**（视差、平台运动、镜头畸变），导致：
1. 对比学习正样本构造失效（同索引 token 非同一物理区域）；
2. 稠密跨模态融合引入语义不一致交互。

现有遥感/卫星基础模型多假设全局或图像中心对齐，未在融合前估计**局部对应可靠性**。

## 方法与设计

### 总体框架（图 4，第 7 页起）
三条模块共享同一几何先验通路：
1. **syncPATC**：同步视图变换下学习 patch 中心一致性，输出 token/query 置信度、query 中心、子 token 偏移。
2. **MG-Sparse-MMA**：按可靠性 Top-\(K_s\) 稀疏查询引导双向融合，复杂度从 \(O((HW)^2)\) 降至 \(O(K_s L_s)\)。
3. **RA-QCGCL**：可靠 patch-patch、patch-query、query-query 三分支对比学习，与稀疏瓶颈一致。

### UAVMeta 与 StateBench
- **UAVMeta**：2,575 对 RGB–IR（train/val/test = 1,715/572/288），采集级划分；LoFTR/SuperPoint+LightGlue 仿射粗对齐，保留局部位移。
- **四类采集状态分数**（式 1—4，第 5—6 页）：
  - CARS（相机可靠性）、OSGS（观测尺度/GSD）、VSS（视点稳定性）、FMCS（机动复杂度）
- **MSPA-4D**（式 5—6）：四分数预测准确度，带容差 \(\tau_j\) 与活动权重 \(r_j\)。

### 输入形式
配对图像 \((I^R, I^T)\) 经双编码器；syncPATC 对同步仿射变换 \(W_\theta\) 学习变换一致置信度；MG-Sparse-MMA 仅在几何校准邻域内更新。

## 数据集与下游协议

| 任务 | 主要基准 | 输入 |
|------|----------|------|
| 语义分割 | KUST4K、UAVMeta | RGB / IR / RGB–IR |
| 目标检测 | DroneVehicle、UAVMeta | 同上 |
| 场景分类 | AID、RESISC45、UAVMeta | RGB 或 RGB–IR |
| 变化检测 | CDD、LEVIR-CD | 双时相 |
| 多目标跟踪 | Drone 1 等 | RGB–IR |
| 新视角合成 | 单时 AM00 | RGB–IR + 遥测 |
| StateBench | UAVMeta | MSPA-4D |

## 定量结果（正文主表）

### 语义分割
| 基准 | 方法 | mIoU↑ | 备注 |
|------|------|-------|------|
| KUST4K | SGFNet | 81.07 | 基线最强之一 |
| KUST4K | **GAAT(C)** | **82.14** | mAcc 92.41（表 IV，~第 11 页） |
| UAVMeta | SegFormer-B5 | 63.52 | RGB-only |
| UAVMeta | **GAAT(C)** | **67.79±1.18** | FWIoU 78.58，Pixel Acc 87.44（表 V，第 12 页） |

### 目标检测
| 基准 | 方法 | mAP | mAP50 | mAP75 |
|------|------|-----|-------|-------|
| DroneVehicle | DDQ-DETR | 50.10 | 72.40 | 59.50 |
| DroneVehicle | **GAAT(D)** | **56.59** | **80.12** | **67.16**（表 VI） |
| UAVMeta | DDQ-DETR | 43.33±3.67 | 61.93 | 50.37 |
| UAVMeta | **GAAT(D)** | **44.86±3.76** | **62.64** | **52.00**（表 VII） |

### 场景分类
| 基准 | GAAT(G) Acc |
|------|-------------|
| AID (20%/50%) | 95.90 / 97.38 |
| RESISC45 (10%/20%) | 92.84 / 94.49 |
| UAVMeta mean Acc | 63.66±1.78；macro-F1 48.12±1.23（表 IX） |

### 变化检测（表 X）
| 基准 | GAAT(CF) F1 / IoU |
|------|-------------------|
| CDD | 97.85 / — |
| LEVIR-CD | 95.96 / 92.47 |

### 新视角合成（表 XI 区域，~第 12 页）
GAAT+ThermalGS：PSNR **26.92**，SSIM **0.89**，LPIPS **0.11**。

### StateBench
GAAT 在 MSPA-4D 聚合得分上领先代表性基线（正文摘要与消融 XIV 表）。

## 消融与局限

**模块消融**（表 XIV，附录）：移除 syncPATC / MG-Sparse-MMA / RA-QCGCL 均导致代表性任务下降；三模块共享可靠性信号的设计得到验证。

**局限**（讨论节）：
- 当前以 RGB–IR 为主实验，SAR/深度等模态在 schema 层支持但正文实验较少。
- 粗仿射对齐无法消除全部局部视差；极端机动下置信度估计仍可能失效。
- UAVMeta 规模（2.5K 对）相对 VisDrone 等仍偏小。

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 几何挑战示意（图 1—2） | 1—2 |
| 遥感基础模型对比（表 I） | 2—3 |
| UAVMeta 样本（图 3） | 4—5 |
| 数据集对比（表 II） | 4—5 |
| 框架总览（图 4） | 7 |
| KUST4K 分割（表 IV） | 11 |
| UAVMeta 分割/检测（表 V—VII） | 12—13 |
| 场景分类（表 VIII—IX） | 13 |
| 变化检测（表 X） | 13 |
| 定性对比（图 5—6 等） | 14+ |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27971v1)
- [arXiv](https://arxiv.org/abs/2608.27971v1)

---

<a id='2608.28383v1'></a>
## [Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs](https://arxiv.org/abs/2608.28383v1)

**Authors:** Chenhong He, Lei Li, Shicheng Li, Hanglong Lv, Lingpeng Kong, Qi Liu, Tong Yang, Shuhuai Ren

**Published:** 2026-08-28

**Categories:** cs.CV, cs.CL

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Hybrid attention dominates frontier LLMs, yet Vision Transformers (ViTs) in multimodal LLMs lack a satisfactory hybrid design, with no consensus on why certain attention patterns work better. To fill this gap, we study ViT attention heads and find they differentiate into object- and background-specialist roles, a pattern most pronounced under full attention; we call this Semantic Head Specialization (SHS). We propose SHS-Index to quantify this specialization, show that it distinguishes full-attention from chunk-window ViTs, and find that it strongly tracks downstream benchmark performance. We then identify three structural factors that shape SHS---window interaction, token serialization, and local softmax allocation---and use them as design principles for hybrid attention. Guided by these factors, we design Ariadne Attention, a hybrid that matches full attention on 22 image and video tasks at 6.5x less attention compute. Our findings establish head specialization as a measurable property for diagnosing and designing principled hybrid ViT attention at the multimodal-LLM scale.

**Analysis:**

# Semantic Head Specialization Guides Hybrid ViT Attention for Multimodal LLMs

- **ArXiv ID**: 2608.28383v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28383v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 22 页；`agent_pdfs/part-001.pdf` 22 页；提取完整

## 问题定义

多模态 LLM 中 ViT 编码器普遍采用 **chunk-window 局部注意力**（如 Qwen2.5-VL）以降算力，但与 **full attention** 相比下游质量仍有差距。论文追问：full attention 保留了什么？能否据此设计可解释、可度量的 hybrid ViT？

## 核心现象：Semantic Head Specialization (SHS)

- 在 matched 32 层 ViT 对照实验中，full attention 的注意力头分化为**前景物体头**与**背景头**；chunk-window 头呈现网格状窗口伪影、角色不分化（图 2，第 1—2 页）。
- **SHS-Index**（式 3—5）：对每个 head 用 COCO 分割 mask 计算 received attention 区分前景/背景的 AUROC，取 \(\max(\text{AUROC}, 1-\text{AUROC})\)，再对图像/层/头平均。
  - Full: **0.606** vs Chunk: **0.577**
  - 16 个开源 encoder/VLM：11 个 full-attention（均值 0.631）与 5 个 chunk（0.585）**无重叠**（图 1）

## 方法：三因素 → Ariadne Attention

### 因素 1：窗口隔离（图 3—5）
- Chunk-2D → **SWA-2D**：SHS 0.577→**0.588**；平均注意力距离 D: 10.69→11.80

### 因素 2：序列化顺序
- SWA-2D → **SWA-1D**（行主序）：0.588→**0.600**

### 因素 3：局部 softmax 分配
- 每 head **sink bias** \(s_{\text{aux}}\)（式 1）：吸收背景概率质量；Full+Sink 达 **0.611**（图 5，第 4 页）

### Ariadne 结构（图 7，§4.1）
- 每 8 层块：**4 层 row-major SWA+sink + 3 层 col-major SWA+sink + 1 层 full attention**；重复 4 次得 32 层。
- SWA 窗口 \(w=64\)（每侧 128 token）。

### 训练设置
- 32 层 ViT，16 heads，\(d_h=80\)；与 chunk 对照除注意力算子外完全一致；8k iteration 主 checkpoint；22 个下游 benchmark（20 图像 + 2 视频）。

## 定量结果

### 主结果（表 1，第 5—6 页；图 8）
| 方法 | 20 图像任务均分 | 22 任务均分（含视频） | 注意力 TFLOPs | 端到端 ViT 时间 @896² |
|------|----------------|----------------------|---------------|----------------------|
| Full Attention | **40.92** | 22.1 | 11.00T | 123.4 ms |
| Chunk Window | 38.93 | — | 1.68T | — |
| **Ariadne** | **40.40** | **24.3** | **1.68T（6.5×↓）** | **106.8 ms（13.5%↓）** |

与 full 差距仅 **0.52** 分（20 图像任务）。

### SHS 与 benchmark 相关性（图 6）
- 8 个诊断配置：SHS-Index 与 22 任务均分 Pearson **r=0.858**，p=0.006；置换检验 p=0.008。

### 配置扫描（表 3，附录）
| 配置 | SHS | Bench(20) |
|------|-----|-----------|
| chunk_2d | 0.577 | 37.72 |
| swa_sink_1d | 0.600 | 40.14 |
| **swa_sink_4r3c (Ariadne)** | **0.602** | **40.40** |
| full | 0.606 | 40.92 |

### 逐任务亮点（表 7，附录）
Ariadne 在 V*、DocVQA、OCRBench 等视觉中心任务上显著追回 chunk 差距（如 OCRBench +8.9 vs chunk）。

## 消融与局限

**消融**：
- 窗口尺寸 \(w\in\{32,64,128\}\)：w=64 在 SHS 与 benchmark 间最优（表 4）。
- 训练步数 6k/8k/10k：8k 为报告主 checkpoint（表 5—6）。

**局限**（§6，第 6 页）：
- SHS 与 benchmark 相关是**跨架构信号**，非因果预测器；几何/计数类任务仅部分被 SHS 捕获。
- 对照主要在自训练 32 层 ViT；开源异构模型仅用于 SHS 诊断，未统一重训。
- Ariadne 仍含 4/32 层 full attention，非纯局部算子。

## 重要图表页码

| 内容 | 页码 |
|------|------|
| SHS 开源模型分离（图 1） | 1 |
| 注意力热图对比（图 2） | 1—2 |
| Token 序列化（图 3—4） | 3—4 |
| SHS 结构消融（图 5—6） | 4—5 |
| Ariadne 块结构（图 7） | 5 |
| 算力-质量（图 8） | 5—6 |
| 主 benchmark 表（表 1） | 5—6 |
| 全配置表（表 3） | 附录 |
| 逐任务表（表 7） | 附录 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28383v1)
- [arXiv](https://arxiv.org/abs/2608.28383v1)

---

<a id='2608.27795v1'></a>
## [uScenes: A Multimodal RGB and 3D Sonar Dataset for Underwater Robot Perception](https://arxiv.org/abs/2608.27795v1)

**Authors:** Trung Tien Dong, Zhenqi Wu, Aditya Penumarti, Zi-Hao Zhang, Micaiah Bartlett, Jane Shin, Xiaomin Lin

**Published:** 2026-08-28

**Categories:** cs.CV

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Robust perception is essential for the deployment of autonomous underwater robots. However, optical cameras become unreliable under poor illumination and backscatter. Forward looking (2D) acoustic sensors remain effective under these conditions, but they measure range and bearing while leaving elevation unresolved, creating an ambiguity that prevents individual sonar returns from being localized in three dimensional (3D) space. This complicates the sensor use for 3D scene understanding and precise object detection. We introduce \textbf{uScenes}, a multimodal underwater dataset containing synchronized 3D multibeam sonar point clouds and RGB imagery. The dataset contains 110 scenes and 95,834 synchronized observation, representing 277.6 minutes of data collected across multiple field sessions. uScenes establishes a foundation for underwater sensor fusion, cross modal representation learning and 3D scene understanding. Code and datasets are given at https://github.com/era-research-lab/uScenes.

**Analysis:**

# uScenes: A Multimodal RGB and 3D Sonar Dataset for Underwater Robot Perception

- **ArXiv ID**: 2608.27795v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.27795v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 7 页；`agent_pdfs/part-001.pdf` 7 页；提取完整

## 问题定义

水下机器人感知中，光学相机在浑浊/弱光下失效，而传统前视 2D 声呐仅有距离-方位、**仰角未解析**，单回波无法唯一三维定位。现有多模态数据集多提供 2D 声呐图或激光点云，缺少**移动平台上同步的 RGB + 度量 3D 多波束声呐点云**。

## 数据集与方法

### 采集平台（图 3，表 II，第 3—4 页）
- **BlueROV2** + DWE RGB 相机 + Water Linked Sonar 3D-15
- 相机：1280×720 @~30 Hz；针孔 + 5 径向/2 切向畸变（\(f_x=922.26, f_y=920.86, c_x=694.06, c_y=387.33\)）
- 3D 声呐：256×64 range image，FOV 90°×40°，~6 Hz；每帧 1—12,627 点（均值 6,805）
- 传感器中心横向 120 mm、垂向 31.71 mm（总位移 ~124 mm）

### 规模（摘要 + 表 III）
| 指标 | 数值 |
|------|------|
| 场景数 | 110 |
| 同步观测对 | 95,834 |
| 总时长 | 277.6 分钟 |
| 采集地点 | 佛罗里达 Blue Grotto，7 次野外作业 |

### 场景类别（表 III，第 4 页）
| 类别 | 场景数 | 占比 | 配对数 |
|------|--------|------|--------|
| None（一般环境） | 46 | 41.8% | 42,957 |
| Human | 24 | 21.8% | 24,167 |
| Structure | 30 | 27.3% | 20,979 |
| ROV | 4 | 3.6% | 4,312 |
| Fish | 6 | 5.5% | 3,419 |

### 时间同步（式 1，第 3—4 页）
\[
i^* = \arg\min_i |t_{c,i} - t_s|,\quad |t_{c,i^*} - t_s| \le 83\text{ ms}
\]
- 均值时差 13.2 ms，中位 11.2 ms，95% ≤ 32.8 ms

### 3D 点云构造（式 2—4）
- 角度：\(\theta = \frac{u}{W-1}\phi_h - \frac{\phi_h}{2}\)，\(\psi = \frac{v}{H-1}\phi_v - \frac{\phi_v}{2}\)
- 笛卡尔：\(p_s = r[\cos\psi\cos\theta,\ \cos\psi\sin\theta,\ \sin\psi]^\top\)
- 数据集二进制横向符号与厂商坐标系相反，加载时用 \(\text{diag}(1,-1,1)\) 转换
- 信号强度：距离补偿 \(e^{ar^2}\) 后按帧 99 分位归一化至 [0,1]

### 数据组织
- RGB：JPEG；声呐：\(N\times4\) float（x,y,z,归一化强度）
- JSONL manifest：时间戳、帧 ID、点数等
- 场景级描述为粗粒度，**非逐帧检测标注**

### 光声标定（§III-E，图 4）
三类标定靶（混凝土块、圆头针板、平头螺丝板）；外参 \(p_c^j = R_{cs} p_s^j + t_{cs}\) 最小化重投影误差（式 5—6），EPnP 初始化 + 非线性优化。

## 与现有数据集对比（表 I，第 2—3 页）

uScenes 是唯一同时提供 **RGB + 同步 3D 声呐点云 + 移动平台野外采集** 的条目；RGBS50/HODOR/R-S9 为 2D 声呐视频；BrSPCD/SUOP 无同步 RGB。

## 实验与基准

本文为**数据集论文**，未报告学习算法 SOTA 数值。支持的研究方向：
- 跨模态表示学习、传感器融合
- 3D 场景理解、劣化光学条件下的目标感知
- 光声标定方法开发与评测

## 定量结果

论文的主要定量结果是数据规模与采集协议，而非下游模型精度：110 个场景、95,834 对同步观测、277.6 分钟数据；声呐每帧点数为 1—12,627（均值 6,805），时间同步误差均值 13.2 ms、中位数 11.2 ms，95% 不超过 32.8 ms。由于没有帧级检测标注或学习算法实验，不能从本文正文推出检测 mAP、分割 mIoU 等 SOTA 数值。

## 局限（§IV-C，第 5—6 页）

1. **单平台、单淡水场地**（Blue Grotto），未覆盖咸水/多地理区域
2. **无稠密目标标注**；场景描述非帧级 GT
3. 信号强度为帧内相对值，**不可跨帧比较绝对反射率**
4. 声呐逐线扫描，平台/目标运动可导致点云畸变
5. 数据集定位为**时间同步但未空间配准**；需投影的任务须先完成光声标定

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 野外采集（图 1） | 1 |
| 同步 RGB/声呐示例（图 2） | 2 |
| 平台与传感器（图 3，表 II） | 3—4 |
| 标定靶（图 4） | 4 |
| 场景统计（表 III） | 4 |
| 数据集对比（表 I） | 2—3 |
| 讨论与结论 | 5—6 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27795v1)
- [arXiv](https://arxiv.org/abs/2608.27795v1)

---

<a id='2608.27860v1'></a>
## [From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation](https://arxiv.org/abs/2608.27860v1)

**Authors:** Rit Gangopadhyay, Alex Wong

**Published:** 2026-08-28

**Categories:** cs.CV, cs.AI

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Vision foundation models are capable of generalizing across 3-dimensional (3D) scenes with high-fidelity estimates; their empirical success can be attributed to training on large-scale datasets of perspective images. However, when transferred to wide field-of-view (FoV) images, such as those captured by fisheye cameras, they return erroneous outputs due to a covariate shift stemming from the radial distortion on the image pixels. We propose a method to generalize vision foundation models to fisheye cameras. The crux of our method lies in a set of learnable parameters, termed Distortion Extenders (DEX), that model the fisheye distortion coefficients and the distributional shift between fisheye and perspective images encoded in the latent space. By minimizing a self-supervised alignment loss, DEX transforms the latent embeddings of fisheye images to resemble those of perspective images to recover high-fidelity estimates. DEX is architecture- and task-agnostic: We demonstrate DEX on monocular depth estimation and open-vocabulary segmentation for convolution- and Transformer-based architectures, where we consistently improve over baselines across indoor and outdoor fisheye datasets. As a byproduct, the activations of DEX can also be decoded to distortion coefficients to support camera calibration. Code available at: https://github.com/Suchisrit/DEX.

**Analysis:**

# From Perspective to Fisheye Depth Estimation and Open-Vocabulary Segmentation

- **ArXiv ID**: 2608.27860v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.27860v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 29 页；`agent_pdfs/part-001.pdf` 29 页；提取完整

## 问题定义

视觉基础模型在透视图像上训练，迁移到鱼眼/大 FoV 图像时因径向畸变导致**潜空间分布偏移**，深度估计与开放词汇分割质量崩溃（图 1，第 2 页）。传统方案依赖已知内参重投影/ERP/立方体贴图，引入重采样伪影与标定依赖。

## 方法：Distortion Extenders (DEX)

### 核心思想（§3，图 2，第 6—8 页）
- **冻结**透视预训练骨干 \(f,g\)，在每层插入轻量调制器 \(\theta=\{E^{(l)}, W^{(l)}\}_{l=1}^L\)
- 对透视图 \(I\) 合成鱼眼 \(I_f = T\circ I\)（KB 模型）；透视输出为监督，鱼眼经 DEX 后逆变换对齐比较

### 调制机制（式 1—2）
\[
S = EW,\quad A = \text{softmax}\left(\frac{XS^\top}{\sqrt{D}}\right),\quad X \leftarrow X + AE
\]
- \(W=W_1^\top W_2\) 低秩；\(E\in\mathbb{R}^{M\times D}\) 为 Extender“码本”凸组合

### 训练目标
- **深度**（式 3—6）：透视深度 \(z\) 转球面距离 \(R\)，对数 L1
- **开放词汇分割**（式 7）：在图像嵌入空间对齐，避免乘语言分支

### 推理
- 无需标定/重投影；鱼眼直接前向；额外存储与时间开销小（表 5）

## 数据集与协议

| 任务 | 训练 | 评测（零样本真实鱼眼） |
|------|------|------------------------|
| 单目深度 | 仅校准透视图（合成鱼眼） | ScanNet++（室内）、KITTI-360（室外） |
| OVS | Mix 50K（LSeg/SED） | WoodScape |
| 多任务 | PanopticDepth 输出作监督 | WoodScape 深度+分割 |

## 定量结果

### 单目深度（表 1，第 11—12 页）

**ScanNet++（UniDepthV2 + DEX，Mix 200K 训练）**
| 方法 | RMSE↓ | δ1↑ |
|------|-------|-----|
| Base | 0.329 | 0.671 |
| LoRA | 0.235 | 0.854 |
| Calibration Tokens | 0.223 | 0.841 |
| **DEX** | **0.200** | **0.872** |

**KITTI-360（UniDepthV2 + DEX）**
| 方法 | RMSE↓ | δ1↑ |
|------|-------|-----|
| Base | 7.093 | 0.262 |
| LoRA | 1.916 | 0.771 |
| Calibration Tokens | 1.788 | 0.763 |
| **DEX** | **1.663** | **0.842** |

相对 LoRA 平均约 **19% RMSE↓、15% δ1↑**（正文 §4.1）。

### 开放词汇分割 WoodScape（表 2）
| 模型 | 方法 | mIoU↑ | weighted IoU↑ |
|------|------|-------|---------------|
| LSeg | Calibration Tokens | 0.321 | 0.823 |
| LSeg | **DEX** | **0.362** | **0.838** |
| SED | **DEX** | **0.449** | 0.832 |

相对 Calibration Tokens mIoU 约 **+13%**。

### PanopticDepth 联合（表 3）
| 方法 | RMSE↓ | δ1↑ | mIoU↑ | wIoU↑ |
|------|-------|-----|-------|-------|
| 基线 | 7.273 | 0.267 | 0.160 | 0.300 |
| **+DEX** | **3.583** | **0.354** | **0.282** | **0.833** |

## 消融与局限

**消融**（表 4—6，§5）：
- 笛卡尔深度监督在室外大 FoV 显著劣化；低秩 \(W\) 必要
- t-SNE 显示 DEX 使鱼眼特征接近透视特征（图 5）
- Extender 权重可解码 KB 畸变系数（表 6）

**局限**（§6，附录）：
- 训练仅用合成鱼眼，极端畸变/新相机可能泛化不足
- 未在训练中需要真实鱼眼 GT
- 与 Calibration Tokens 相比优势因骨干与任务而异

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 失败案例（图 1） | 2 |
| DEX 管线（图 2） | 6—7 |
| 深度表（表 1） | 11—12 |
| 分割表（表 2—3） | 12—13 |
| 消融（表 4—5） | 13+ |
| 畸变系数解码（表 6） | 13+ |
| 讨论与局限（§6） | 14+ |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27860v1)
- [arXiv](https://arxiv.org/abs/2608.27860v1)

---

<a id='2608.27997v1'></a>
## [A-PAIR: A Benchmark and Identity-Consistent Grounding Framework for Air-Ground Cross-View Referring Person Detection](https://arxiv.org/abs/2608.27997v1)

**Authors:** Zhoupeng Guo, Xinjie Yao, Yunqi Zhu, Zhihe Fan, Siqi Zhao, Jianjun Chen, Yichen Dong, Yan Fan, Pengfei Zhu

**Published:** 2026-08-28

**Categories:** cs.CV, cs.MM

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Air-ground cross-view referring person detection is a necessary component in the language-to-perception-to-control chain of collective embodied intelligence, grounding a language command into the same physical target before ground and aerial agents can coordinate downstream actions. Existing referring expression comprehension and open-vocabulary grounding methods do not jointly account for cross-view identity consistency, making them insufficient for Air-Ground Cross-View Referring Person Detection (AGCV-RPD), which involves similar pedestrian distractors, weak aerial appearance cues, and cross-view identity consistency. To study this problem, we introduce Air-Ground Paired Identity-Aware Referring (A-PAIR), the first comprehensive AGCV-RPD benchmark, containing 22,137 cross-view referring samples. To construct A-PAIR efficiently, we propose Factorized Annotation and Referential Alignment (FARA), a semi-automatic annotation framework that generates factorized referring descriptions and identity-consistency supervision at reduced cost. We propose Identity-Consistent Referring Grounding (ICRG), a framework that combines factorized referential grounding, candidate-completeness supervision, and cross-view consistency calibration for joint air-ground pair selection. ICRG improves ground, aerial, and pair-level detection over strong baselines, increasing pair F1 from 16.65% to 22.28%. These results show that AGCV-RPD requires paired detection and identity-consistent reasoning.

**Analysis:**

# A-PAIR: A Benchmark and Identity-Consistent Grounding Framework for Air-Ground Cross-View Referring Person Detection

- **ArXiv ID**: 2608.27997v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.27997v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 9 页；`agent_pdfs/part-001.pdf` 9 页；提取完整

## 问题定义

**空地跨视角指代行人检测（AGCV-RPD）**：给定地面图、航拍图与自然语言描述，两视角各输出一个框，且必须指向**同一物理个体**。难点：
1. 相似行人干扰与空间歧义；
2. 航拍目标小、纹理弱；
3. 单视角各自合理 ≠ 跨视角身份一致。

## 基准 A-PAIR

### 数据来源与规模（表 2，第 3 页）
- 源自 G2APS：7,588 共享身份地面对；3,891 唯一航拍图
- **22,137** 跨视角指代样本（train/val/test = 15,497 / 2,213 / 4,427）
- 每样本含 appearance-full 表达式 \(e_f\) 与 scene-spatial 表达式 \(e_s\)
- 全图行人 COCO 标注：地面 32,016 / 航拍 34,108（训练）
- 身份一致性训练对：377,402（50,741 正样本）

### FARA 标注流水线（图 2，第 2—3 页）
1. 跨视角配对 + xywh 框
2. VLM 提取 pid 级稳定外观短语 \(e_f\)
3. 各视角独立空间关键帧短语，非关键帧复制最近关键帧
4. 质量检查 + 导出 benchmark

### 与相关基准对比（表 1，第 3 页）
A-PAIR 唯一同时具备：语言、航拍+地面、同身份框、**pair 级评测**。

## 方法：ICRG

### 任务正确性（式 5，第 4 页）
\[
\text{hit}_v = \mathbb{1}[\text{IoU}(\hat{b}_v, b_v^*) \ge 0.5],\quad \text{hit}_{\text{pair}} = \text{hit}_g \land \text{hit}_a
\]

### 三信号融合（图 3）
1. **Factorized Grounding** \(G\)：GroundingDINO+Swin，分别用 \(e_f,e_s\) 得 \(R_v^{e_f}\cup R_v^{e_s}\)
2. **Candidate Completeness** \(D\)：全行人检测器，\(C_v = R_v \cup P_v\)
3. **Cross-View Consistency** \(f\)：ResNet-50 嵌入余弦相似度 \(s^{id}\)

### 推理融合（式 11—12，算法 1）
\[
\text{score}(g_i,a_j) = w_{\det}s^{det}_{ij} + w_{ref}s^{ref}_{ij} + w_{id}s^{id}(g_i,a_j)
\]
权重在验证集网格搜索，非学习参数。

## 实验协议

- 单目标指代；IoU 阈值 \(\tau=0.5\)
- 指标：各视角 F1/Acc（instance & image 级）+ **Pair F1/Acc**（主指标）

## 定量结果

### 主结果（表 3，第 5—6 页，测试集百分比）

| 方法 | Ground F1_i | Aerial F1_i | Pair Acc | **Pair F1** |
|------|-------------|-------------|----------|-------------|
| TransVG | 2.53 | 0.05 | 0.00 | 0.00 |
| GDINO-T | 32.28 | 17.42 | 9.08 | 16.65 |
| RefDrone | 33.50 | 14.66 | 8.31 | 15.35 |
| **ICRG** | **35.17** | **21.89** | **12.54** | **22.28** |

相对最强基线 GDINO-T，**Pair F1 +5.63 百分点**（16.65→22.28）。

### 消融（表 4）
| Fact. | Cand. | Cons. | Pair F1 |
|-------|-------|-------|---------|
| — | — | — | 3.60 |
| ✓ | — | — | 20.88 |
| ✓ | ✓ | — | 21.82 |
| ✓ | ✓ | ✓ | **22.28** |

- Fact.：分解外观/空间语义，缓解相似行人歧义
- Cand.：主要提升航拍召回（F1_i 20.83→21.82）
- Cons.：在已有候选上重排，提升 pair 一致性

### 诊断（图 4）
双视角 IoU≥0.5 的 pair hit 仅占测试集 **12.54%**；航拍检测是主要瓶颈。

## 局限

- 依赖外部动捕估计目标位姿，限制野外部署
- Pair F1 绝对值仍低（22.28%），密集人群与极端俯视仍困难
- 融合权重固定于验证集，可能过拟合小验证集
- 未来：时序空地观测、更拥挤场景

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 任务示意（图 1） | 1 |
| FARA 流水线（图 2） | 2—3 |
| 基准对比（表 1—2） | 3 |
| ICRG 框架（图 3） | 4 |
| IoU 质量分解（图 4） | 5 |
| 主结果（表 3） | 5—6 |
| 消融（表 4） | 6 |
| Ground vs Pair 散点（图 5） | 6 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27997v1)
- [arXiv](https://arxiv.org/abs/2608.27997v1)

---

<a id='2608.28288v1'></a>
## [GeoFF3D: Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping](https://arxiv.org/abs/2608.28288v1)

**Authors:** Xiang Yang, Yongli Wang, Yunsheng Zhang

**Published:** 2026-08-28

**Categories:** cs.CV

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Existing feed-forward 3D reconstruction methods typically process a bounded number of images and recover cameras and geometry in local or internally normalized frames. Extending them to large-scale UAV mapping requires scalable multi-chunk processing and reliable aggregation, while full Sim(3) alignment can become unstable for near collinear trajectories. We present GeoFF3D, which combines a coordinate-anchored model with a spatial large-scale reconstruction framework (SLRF). The model uses georeferenced camera translations and optional geometric priors to predict camera poses and dense point maps directly in a gravity-aligned Z-up metric frame. SLRF partitions images into spatially overlapping chunks, propagates shared-view priors, and aggregates local reconstructions hierarchically, while remaining applicable to different bounded-view models. Across nine aerial mapping blocks, GeoFF3D achieves the best average reconstruction quality, improving F@5 from 0.829 for Pi3X + SLRF to 0.877. On long UAVScenes sequences, it reaches 0.848, compared with 0.687 for Pi3X + SLRF and 0.451 for the strongest evaluated SLAM/streaming baseline. GeoFF3D reconstructs 2,000 images in approximately five minutes, demonstrating scalable and robust large-scale UAV reconstruction.The code is available at https://github.com/yanxian-ll/GeoFF3D.

**Analysis:**

# GeoFF3D: Coordinate-Anchored Feed-Forward Reconstruction for Large-Scale UAV Mapping

- **ArXiv ID**: 2608.28288v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28288v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 13 页（含附录 A—B 至第 13 页）；`agent_pdfs/part-001.pdf` 13 页；提取完整

## 问题定义

前馈 3D 重建（DUSt3R/MASt3R/VGGT/π³ 等）通常处理有界视角数、**局部或归一化坐标系**。大规模 UAV 倾斜摄影需：
1. 分块处理与聚合；
2. 后验 Sim(3) 配准地理参考——在**近共线航迹**上 roll/pitch 约束弱，易整体倾斜；
3. 块间深度/坐标不一致导致重复面与接缝。

## 方法

### 坐标锚定前馈模型（§3.2，图 3）
- 输入：≤M 张图 + **地理参考平移**（可选旋转/内参/深度先验）
- 在 **重力对齐 Z-up 度量坐标系**直接预测相机位姿与稠密点图
- 块内归一化：平移中心 \(c_T\)、尺度 \(s_T\)；输出再反归一化
- 融合 token：\(F_i = F_i^I + m_i^R F_i^R + m_i^D F_i^D + m_i^t e_i^t + m_i^r e_i^r\)
- 深度头：射线 \(\hat{r}_i(u)\) + 正深度 \(\hat{d}_i(u)\) → 世界点（式 3.2）
- 训练：由 π³X 初始化；UAVFF3D + BlendedMVS；损失
  \[
  \mathcal{L} = 0.5\mathcal{L}_{local} + 1.0\mathcal{L}_{world} + 0.2\mathcal{L}_{pose} + 1.0\mathcal{L}_{grav}
  \]
  两阶段：224px×80 epoch → 518px×10 epoch + 先验 dropout

### SLRF 大规模框架（§3.3，图 4）
1. **足迹分块**：自适应二叉树，叶节点 ≤⌊0.7M⌋ core views + seam views，|S_k|≤M
2. **中心向外推理**：BFS；共享视图深度缓存传播为先验；置信度 0.25 分位滤波
3. **重力对齐层次聚合**：
   - 叶节点：**GA-Sim** \(T(x)=sR_z(\theta)x+t\)（保 roll/pitch）
   - 兄弟节点：**GA-Rigid** 仅 residual yaw+平移
   - 自底向上合并，仅融合 core-view 几何

## 数据集与基线

| 评测集 | 场景 | 图像数/场景 | 角色 |
|--------|------|-------------|------|
| UseGeo D1—D3 | 3 | 224—327 | 定量块 |
| UAVFF3D-Real | 6 | 387—1,177 | 定量块（与训练地理不相交） |
| UAVScenes | 8 序列 | 1,317—2,589（stride 3） | 长序列 |
| NPU-DroneMap | 12 | 285—648 | 定性 |

**基线**：VGGT+SLRF、π³X+SLRF（均用 full Sim(3) 对齐）；长序列另比 VGGT-SLAM、VGGT-SLAM 2.0、VGGT-Long、TTT3R、LingBot-Map。

**指标**：Accuracy↓、Completeness↓、F@1↑、F@5↑（有 GT 时）；评测前统一相机中心 Sim(3)+ICP（与块内协议分离）。

**实现**：518px 最长边；块预算 M=30；2×A100 训练，单 A100 推理。

## 定量结果

### 九块平均（表 1，第 7 页，单位：米 / 分数）

| 方法 | Acc↓ | Comp↓ | F@1↑ | F@5↑ |
|------|------|-------|------|------|
| VGGT+SLRF | 4.79 | 3.96 | 0.187 | 0.730 |
| π³X+SLRF | 3.34 | 3.07 | 0.230 | 0.829 |
| **GeoFF3D** | **2.72** | **2.59** | **0.267** | **0.877** |

相对 π³X+SLRF：Acc **-18.7%**，Comp **-15.8%**，F@5 **+0.048**。

### UAVScenes 八序列平均（表 2）

| 方法 | Acc↓ | Comp↓ | F@1↑ | F@5↑ |
|------|------|-------|------|------|
| LingBot-Map | 7.75 | 44.02 | 0.114 | 0.451 |
| π³X+SLRF | 6.05 | 4.40 | 0.181 | 0.687 |
| **GeoFF3D** | **4.14** | **2.28** | **0.319** | **0.848** |

相对 π³X+SLRF：Acc **-31.5%**，Comp **-48.2%**，F@5 **+0.161**。

### 效率（图 6，§4.3）
- **~2000 张图约 5 分钟**，峰值 GPU 内存 ~16 GiB；运行时间近线性增长

## 消融（§4.3，表 3—6）

| 消融 | 要点 |
|------|------|
| 去掉 \(\mathcal{L}_{world}\)（表 3b） | ATE/CD 显著恶化，说明平移锚定定义坐标系 |
| 去掉 \(\mathcal{L}_{grav}\)（表 3c） | GDE 上升；完整模型 GA-Sim 优于 Sim(3) 保重力 |
| SLRF vs 时序流水线（表 4） | 足迹层次分块降低 CD/ATE/接缝误差 |
| 位姿噪声 2×（表 5） | GeoFF3D 相对稳定；仅平移先验退化明显 |
| 块大小 M（表 6） | M=30 在 CD/ATE 与内存间最优 |

## 局限

- 依赖 GNSS/IMU 等地理先验；稀疏或噪声大时性能下降（表 5）
- 未做在线全局 BA；块间仍可能有残余误差
- UAVScenes 部分序列（如 Valley）GeoFF3D Acc 仍较高（12.7m）
- 附录定性显示边界/重复面问题在难块仍存在

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 共线航迹 Sim(3) 问题（图 1） | 1 |
| 多样场景重建（图 2） | 2 |
| 坐标锚定模型（图 3） | 3 |
| SLRF 流程（图 4） | 4—5 |
| 定性对比（图 5） | 6—7 |
| 九块定量（表 1） | 7 |
| UAVScenes 平均（表 2） | 7 |
| 模型/SLRF 消融（表 3—6） | 8 |
| 效率（图 6） | 8 |
| 逐序列结果（附录表 7—8） | 9—13 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28288v1)
- [arXiv](https://arxiv.org/abs/2608.28288v1)

---

<a id='2608.28272v1'></a>
## [Non-Uniform Quantisation for 3DGS Compression](https://arxiv.org/abs/2608.28272v1)

**Authors:** Bert Van hauwermeiren, Patrice Rondao Alface, Adrian Munteanu

**Published:** 2026-08-28

**Categories:** cs.CV

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, yet its high bitrate requirements pose significant challenges for storage and transmission. To enable practical applications and ensure interoperability within the 3DGS ecosystem, standardised compression formats are essential. In this paper, we propose a novel non-uniform quantisation scheme specifically tailored for 3DGS models. Our approach adapts to the underlying data distribution by applying importance-weighted quantisation and eliminating post-voxelisation redundancy through importance weighted merging. Extensive evaluations on benchmark datasets demonstrate that our method achieves state-of-the-art compression performance. Furthermore, the proposed scheme is compatible with any point-cloud-based representation and is intended as a formal contribution to the upcoming MPEG 3DGS compression standardisation activities.

**Analysis:**

# Non-Uniform Quantisation for 3DGS Compression

- **ArXiv ID**: 2608.28272v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28272v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 19 页；`agent_pdfs/part-001.pdf` 19 页；提取完整

## 问题定义

3D Gaussian Splatting (3DGS) 单场景可达百万级高斯，存储/传输比特率极高。MPEG 正将 G-PCC/V-PCC 扩展至 3DGS，但现有流程对几何与属性多采用**均匀量化**，未利用非均匀分布与感知重要性差异。

## 方法（图 1，§3）

### 1. 重要性加权（§3.1，式 2）
\[
w_g = \sum_p w_{p,g}^2,\quad w_{p,g}=\alpha_{p,g}\prod_{j<i}(1-\alpha_j)
\]
- 完整光栅化可得；实际用 **205 参数 MLP** 近似：\(\hat{w}_g = \text{MLP}(s_g, o_g, \mu_g)\)

### 2. 加权 Lloyd-Max 非均匀量化（§3.2）
最小化加权 MSE：
\[
D = \mathbb{E}[w(x)(x-Q(x))^2]
\]
重建值更新为 bin 内加权质心。为降元数据开销，对重建值差分 \(d_i=y_i-y_{i-1}\) **递归二次量化**（典型 K=4，2 bit）。

### 3. 体素化后加权合并（§3.3，图 2）
同体素高斯合并策略：
- 加权参数平均（四元数用 Markley 优化平均，式 5）
- 协方差空间平均 / Log-Euclidean 平均（式 6—7）
- **相异度阈值**（式 8）限制合并，避免高影响且视觉差异大的高斯合并

### 管线顺序
重要性估计 → 位置非均匀量化 → 加权合并 → 属性非均匀量化 → G-PCC/V-PCC 熵编码

## 实验设置（§4）

| 项目 | 配置 |
|------|------|
| 熵编码 | G-PCC (mpeg151)、V-PCC (mpeg152-gs-anchor, HEVC Main10) |
| 数据 | MPEG 3DGS CTC：MPEG Scenes（大场景）、MPEG Objects |
| 基线 | Uniform、Adaptive Voxelization [26]、FlexGaussian [24] |
| 指标 | RGB/YUV PSNR、SSIM、IVSSIM、LPIPS；**BD-Rate**（锚：uniform） |
| 硬件 | i9-13900 + RTX 4090 |
| 默认 | 场景体素 14 bit、物体 11 bit；二次量化 2 bit；合并=加权参数平均 |

## 定量结果

### 相对 Uniform 的 BD-Rate（表 2 节选，MPEG Scenes，14 bit 体素）

| Codec | RGB-PSNR | YUV-PSNR | YUV-SSIM | LPIPS |
|-------|----------|----------|----------|-------|
| V-PCC 提出方法 | **-28.27%** | **-28.04%** | **-27.65%** | **-30.01%** |
| G-PCC 提出方法 | **-48.08%** | **-40.78%** | **-59.72%** | **-44.14%** |

### 与 SOTA 对比（§4.2，图 3—4）
- 一致优于 uniform 与 Adaptive Voxelization（后者码率低但视觉劣化严重）
- FlexGaussian 单点率失真竞争力强，但**不兼容有损熵编码**、仅单工作点
- G-PCC 某些配置下绝对质量仍可能低于 FlexGaussian，但比特率显著更低

### 耗时（表 1，MPEG Scenes 示例）
| 方法 | 预处理(s) | 编码(s) V-PCC |
|------|-----------|---------------|
| Uniform | 11.77 | 759.82 |
| 提出方法 | 116.18 | **351.63** |

预处理增加，但编码因高斯数减少可更快；**解码时间与 uniform 相当**。

## 消融摘要

| 消融 | 结论（表 3—8） |
|------|----------------|
| 体素 bit 数（表 2） | 14 bit（场景）在率失真间最优；过低急剧劣化 |
| 权重方式（表 3） | MLP≈渲染权重，显著优于 uniform；渲染权重预处理慢 |
| 合并策略（表 5） | **加权参数平均**最优；不合并或简单平均更差 |
| 相异度准则（表 6） | 有准则：MPEG Scenes V-PCC splats 376,527 vs 不合并 495,486 |
| 二次量化 bit（表 7） | 2 bit 最优权衡 |
| 按属性（表 8） | **位置**非均匀量化收益最大 |

**产业采纳**：加权合并已纳入 **V-PCC Amd1 for GS**（§6—7）。

## 局限

- 预处理（~100s 级）高于 uniform；编码端非实时，但符合“训练后离线压缩”场景
- MLP 重要性需 leave-one-out 训练，跨场景泛化依赖小网络规则性
- Objects 数据集上 G-PCC 相对 Adaptive Voxelization 并非全面领先
- 未对协方差预变换做熵编码归一化（文中标注 out of scope）

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 管线总览（图 1） | 3—4 |
| 合并策略示意（图 2） | 7 |
| 定性对比（图 3） | 10—11 |
| RD 曲线（图 4） | 11—12 |
| 体素 bit RD（图 5） | 14 |
| 耗时（表 1） | 10 |
| BD-Rate 消融（表 2—8） | 13—17 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28272v1)
- [arXiv](https://arxiv.org/abs/2608.28272v1)

---

<a id='2608.28205v1'></a>
## [Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency](https://arxiv.org/abs/2608.28205v1)

**Authors:** Jianjian Yin, Liulei Li, Tao Chen, Yi Chen, Yazhou Yao, Wenguan Wang

**Published:** 2026-08-28

**Categories:** cs.CV

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Pruning visual foundation models has attracted considerable attention. However, existing methods focus on rigid point-to-point token alignment on a single dataset for pruning, suffering from two limitations: i) robustness degradation, and ii) task-specificity deficiency. To address these limitations, we propose a task-specific pruning pipeline, named Cut-ViT. Specifically, we first construct gram anchoring matrices from both spatial and semantic perspectives, and perform the subspace decomposition to extract the corresponding subspace bases. Basis-agnostic and residual constraints are then adopted to align the gram subspaces between the native and pruned DINOv3 models along spatial and channel dimensions, enabling subnetworks to inherit robust feature representations of native DINOv3. Furthermore, we design spectral entropy adaptation, which quantifies the information density of feature manifolds along spatial and channel dimensions, thereby adapting the pruning objective to specific downstream tasks. Experiments show that Cut-ViT requires approximately one minute on a single A100 GPU to obtain subnetworks at various sparsity levels, using only 20.9% of the time and 45.5% of the GPU memory compared with previous methods, while achieving SOTA performance on six tasks across nine datasets.

**Analysis:**

# Cut-ViT: Task-Specific Model Pruning via Gram Anchoring Subspace Consistency

- **ArXiv ID**: 2608.28205v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28205v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 20 页；`agent_pdfs/part-001.pdf` 20 页；提取完整

## 问题定义

视觉基础模型（DINOv3）剪枝中，现有 **one-shot pruning (OSP)** 存在：
1. **刚性 token 点对点对齐** → 记忆数值而非流形拓扑，鲁棒性下降；
2. **在 ImageNet 等通用集剪枝** → 边缘部署任务特异性差。

Cut-ViT 在**目标数据集**上，用 Gram 锚定子空间一致性指导 DINOv3-ViT-B/16 结构化剪枝，约 **61s / 单 A100** 得多稀疏度子网。

## 方法（图 2，§3）

### Gram 子空间分解
- 空间 Gram：\(S = FF^\top \in \mathbb{R}^{L\times L}\)（“Where”）
- 通道 Gram：\(C = F^\top F \in \mathbb{R}^{D\times D}\)（“What”）
- SVD 取 Top-K 基 \(U^S, U^C\)

### 基不变子空间一致性（式 5—10）
- **Basis-agnostic**：\( \mathcal{L}_{basis} = 1 - \|U_p^\top U_t\|_F^2 \)
- **Residual**：\( \mathcal{L}_{residual} = \|(I-U_tU_t^\top)F_p\|_F^2 \)

### 任务自适应：谱熵加权（式 11—14）
\[
w^{spatial} = \frac{H(S_t)}{H(S_t)+H(C_t)},\quad w^{channel} = \frac{H(C_t)}{H(S_t)+H(C_t)}
\]
分类任务空间熵低 → 更重视通道；分割/匹配等高空间频率任务相反。

### 剪枝流程
- 原生/剪枝 DINOv3 共享权重；噪声图与采样图对；反向传播得 Fisher 近似重要性 \(H\)；xNES 搜索二值 mask；稀疏度 10%—30%。

## 数据集与任务（§4）

| 任务 | 数据集 | 指标 |
|------|--------|------|
| 语义分割 | ADE20K, PASCAL VOC | mIoU |
| 目标检测 | COCO | mAP |
| 深度估计 | NYUv2 | ARel↓, δ1↑ |
| 视频目标分割 | DAVIS-2017 | J&F, J, F |
| 语义匹配 | FG3DCar, JODS, SBD | PCK@0.05 |
| 分类 | ImageNet | Top-1 Acc |

剪枝校准：N=1000 样本；SVD 主成分 K=192（CEVR≈98.9%）。

## 定量结果（相对训练无关 OSP 提升）

### DAVIS-2017 VOS（表 1，20% 稀疏度，(J&F)/J/F）
| 方法 | 数据集 | J&F / J / F |
|------|--------|-------------|
| SnapViT | ImageNet | 60.8 / 58.8 / 62.8 |
| **Cut-ViT** | DAVIS | **65.0 / 63.2 / 66.8**（↑4.2/4.4/4.0） |

### 语义匹配 FG3DCar（表 2，30% 稀疏度 PCK@0.05）
| SnapViT | 65.7 |
| **Cut-ViT** | **70.8**（**+5.1**） |

### COCO 检测 mAP（表 3）
| SnapViT 30% | 45.8 |
| **Cut-ViT 30%** | **48.6**（**+2.8**） |

### ADE20K 分割 mIoU（表 4）
| SnapViT 20% | 48.3 |
| **Cut-ViT 20%** | **50.0**（**+1.7**） |

### NYUv2 深度（表 5，20%：ARel/δ1）
| SnapViT | 5.6 / 98.5 |
| **Cut-ViT** | **3.8 / 99.3** |

### ImageNet 分类 Top-1（表 7，10%）
| SnapViT | 86.8 |
| **Cut-ViT** | **88.6**（+1.8） |

### 复杂度（表 8，ADE20K mIoU 50.0）
| 方法 | 剪枝时间(s) | GPU(GB) |
|------|-------------|---------|
| SnapViT | 292 | 23.5 |
| EA-ViT（训练式） | 13920 | 37.9 |
| **Cut-ViT** | **61** | **10.7** |

时间为 SnapViT 的 **20.9%**，显存 **45.5%**；相对 EA-ViT 时间 **0.44%**。

### OOD（表 6）
ADE20K 训练 → VOC 测试，20% 稀疏 mIoU：**74.7**（SnapViT 72.6，+1.8）

## 消融（§4.3，表 9—15）

| 组件 | ADE20K 20% mIoU |
|------|-----------------|
| 基线 L_ba | 48.4 |
| + basis-agnostic | 49.1 |
| + task-specific | 49.3 |
| + residual | 49.7 |
| + 双 Gram | **50.0** |

- 目标数据集剪枝 vs ImageNet（表 13）：Cut-ViT 在 VOS/深度/匹配上大幅领先
- 架构泛化（表 14）：SAM/DeiT/CLIP 上均优于 SNOWS/SnapViT
- MSE vs 基不变（表 15）：BI mAP 55.7 vs MSE 54.7

## 局限

- 仍依赖 DINOv3 特定 Gram 锚定结构，迁移到其他 VFM 需验证
- 谱熵权重为启发式，未学习化
- 检测等任务需额外 decoder 微调；剪枝仅针对 encoder
- 极高稀疏度（>30%）性能未充分展开

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 与现有方法对比（图 1） | 2 |
| Cut-ViT 管线（图 2） | 4—5 |
| Gram 空间/通道示意（图 3） | 5—6 |
| 主结果表（表 1—7） | 10—12 |
| 复杂度（表 8） | 12 |
| 组件消融（表 9—12） | 13 |
| 子空间对齐可视化（图 4—5） | 14—15 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28205v1)
- [arXiv](https://arxiv.org/abs/2608.28205v1)

---

<a id='2608.28140v1'></a>
## [Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL](https://arxiv.org/abs/2608.28140v1)

**Authors:** Simone Tolomei, Mayank Mittal, Franco Angelini, Manolo Garabini, Paolo Salaris, Marco Hutter

**Published:** 2026-08-28

**Categories:** cs.RO

**Scores:** relevance 3 / significance 2 / combined 5

**Abstract:**

Non-prehensile manipulation offers versatile skills for moving and rearranging heavy or bulky objects, particularly when combined with a mobile manipulation platform. However, both model-based and model-free approaches struggle with the complex hybrid dynamics and the sparsity of the contact in these tasks. To address these challenges, we propose a contact-guided exploration strategy implemented within a Multi-Critic Reinforcement Learning (RL) framework. A dedicated exploration critic is trained with a dense contact-seeking reward that guides the end-effector toward meaningful contact points; its influence is progressively decayed to recover a task-optimal policy. We obtain candidate interaction points from a general-purpose grasping algorithm, enabling the exploration mechanism to generalise across various object geometries. We evaluate the approach on multiple tasks, including box pushing, chair transportation, and a dishwasher opening task. Finally, we validate the chair transportation policy through extensive experiments on a quadrupedal mobile manipulator, demonstrating deployable non-prehensile manipulation in the real world.

**Analysis:**

# Contact-Guided Exploration for Non-Prehensile Locomanipulation with Multi-Critic RL

- **ArXiv ID**: 2608.28140v1
- **source**: `local_cli_pdf`
- **来源**: `pdfs/2608.28140v1.pdf`（local_cli_pdf）
- **页数验证**: 全文 8 页；`agent_pdfs/part-001.pdf` 8 页；提取完整

## 问题定义

四足移动操作臂执行**非抓取（non-prehensile）**任务（推、拉、滑移搬运椅子/箱子等）时，接触模式稀疏、混合动力学复杂，标准 RL 易陷入**回避接触**的局部最优。需在无专家演示下引导探索，并最终收敛到任务最优策略。

## 方法（图 2—3，§III）

### 分层控制
- 高层策略输出：臂关节目标 \(q_a^*\)、基座 SE(2) 速度 \((v_x^*,v_y^*,\omega_z^*)\)、基座高度 \(h^*\)
- 低层：**冻结**预训练四足 locomotion 策略跟踪基座命令；臂位姿作为其观测一部分

### 接触引导探索
- 用抓取算法 [27] 从物体 mesh 采样 **25 个候选接触点**（椅子）；每 episode 随机选 \(p_{target}\)
- 稠密奖励 \(r_{exp}\)：末端执行器逼近 \(p_{target}\)
- 箱子任务：可见表面均匀采样（类似 [9]）

### Multi-Critic PPO（§III-B）
奖励向量 \(r_t \in \mathbb{R}^3\)：
| 组 | 内容 | 权重 schedule |
|----|------|---------------|
| \(r_{task}\) | 物体到目标速度/位置跟踪 | \(w_{task}=0.75\) 固定 |
| \(r_{exp}\) | 接触点跟踪 | \(w_{exp}\): 0.1→0.01（5k—10k step 线性衰减） |
| \(r_{reg}\) | 动作平滑、钩高惩罚等 | \(w_{reg}\): 0.15→0.24 |

复合优势：\(A_t = \sum_h w_h A_t^h\)；三独立 value head + 共享 LSTM actor（hidden 256）。

### 训练细节（表 I，第 4—5 页）
- Isaac Lab，4096 并行环境；dt=0.005s，控制 0.02s
- 椅子：15 IKEA + 100 程序化随机椅子；质量 2—4 kg，摩擦随机
- 成功：物体距目标 <0.2 m；失败：位移<0.2m（Missed Contact）、倾角>35°（Tipover）、超时

## 实验

### 仿真任务
1. **椅子搬运**（主任务）
2. **箱子推动**
3. **洗碗机开门**（定性：把手→门板接触切换，图 9）

### 硬件
- **ALMA** 四足移动操作臂；机载本体感知 + 外部位姿 mocap
- IKEA 四物体：ADDE、SANDSBERG、VIHALS、LOVBACKEN（表 II）

## 定量结果（图 5，5 seeds 平均）

### 椅子搬运
| 方法 | Success | Missed Contact | Tipover | Timeout |
|------|---------|----------------|---------|---------|
| PPO | 低（Missed 9.1%） | 高 | — | 高 |
| PPO+WS | 改善 | 4.0% | — | — |
| Multi-Critic PPO（固定权重） | 接触好 | 低 | **Tipover 高** | — |
| **Multi-Critic PPO+WS（提出）** | **94.1%** | 低 | **4.4%** | 低 |

- 完成时间 ~9.2 s（与 PPO+WS ~9.1 s 相当）
- 无探索奖励：Success **0%**

### 箱子推动
提出方法 Success **>90%**，显著高于 PPO 系列（图 5a）

### 真机 IKEA（表 II）
| 物体 | 成功率 |
|------|--------|
| ADDE | 72.9% (27/37) |
| SANDSBERG | 57.1% (8/14) |
| VIHALS（折叠椅） | 100% (3/3) |
| LOVBACKEN（三脚桌） | 50% (2/4) |
| **合计** | **69.0% (40/58)** |

仿真 94.1% → 真机下降主因：侧向接近导致 yaw 剧烈、里程计漂移。

### 真机扩展实验
- **动态目标**：实时改目标位置可跟踪
- **6.5 kg 载荷**：超臂额定静载仍成功
- **扰动恢复**：人工推椅子后可重新挂钩
- 洗碗机：PPO 常“静态拉把手”；提出方法 **关节限位违规时间减少 59%**（§IV-D）

## 消融与局限

**消融逻辑**（§IV-A）：
- 标量 PPO：探索瓶颈
- 固定权重 Multi-Critic：过拟合接触、倾翻多
- **权重衰减 Multi-Critic**：兼顾发现接触与任务物理

**局限**（结论，第 7—8 页）：
- 三任务验证，更广域迁移待测
- 真机依赖**外部位姿 mocap**，限制野外部署
- 固定 critic 权重 schedule，未自适应课程
- 候选接触来自抓取启发，极端非凸几何可能失效

## 重要图表页码

| 内容 | 页码 |
|------|------|
| 系统示意（图 1） | 1 |
| 架构总览（图 2） | 3 |
| 接触候选（图 3） | 3 |
| 搬运策略定性（图 4） | 4 |
| 成功率/失败模式（图 5） | 4—5 |
| 奖励与超参（表 I） | 4—5 |
| 真机泛化（图 6—7） | 5—6 |
| 载荷/扰动（图 8） | 6 |
| 洗碗机接触序列（图 9） | 7 |
| 真机统计（表 II） | 5 |

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28140v1)
- [arXiv](https://arxiv.org/abs/2608.28140v1)

---

