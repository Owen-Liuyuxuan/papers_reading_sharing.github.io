time: 20260831

# Arxiv Computer Vision Papers - 2026-08-31

## Executive Summary

# 今日重点

本日入选论文集中于自动驾驶视频生成、UAV多模态几何对齐与大规模重建、视觉Transformer高效注意力/剪枝，以及水下机器人和空地协同感知。最值得关注的共同趋势是：在数据、算力或传感器对齐受限时，引入可测量的结构先验（缩放规律、几何可靠性、语义头专门化、子空间一致性）来提升系统效率与鲁棒性。GeoFF3D强调坐标锚定和分块聚合的工程可扩展性；GAAT与A-PAIR分别解决跨模态局部错位和跨视角身份一致性；uScenes提供水下RGB-声纳融合基础数据。Cut-ViT和Ariadne Attention则从表示子空间和注意力结构出发降低视觉基础模型成本。

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

# 2608.28404v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
自动驾驶视频生成不能照搬互联网规模化路线：驾驶数据采集昂贵且受隐私约束。作者从零训练1M—9B参数视频扩散模型，利用最多5500小时驾驶数据研究模型规模与训练暴露量的缩放规律。

## 动机与方法
核心痛点是数据固定、预算有限，不清楚应增加模型规模还是训练时长。作者系统改变参数量和数据重复暴露次数，观察验证损失的幂律关系，并据此选择计算最优配置。结论是固定模型下延长训练收益更快；数据与算力充足时扩大模型仍有更低渐近损失。最终训练9B模型并在nuScenes上取得开放式驾驶视频生成SOTA。

## 对比、实验与实用性
贡献在于把驾驶视频扩散的预算分配从经验调参转化为可量化缩放分析。适用于隐私敏感、数据规模受限的自动驾驶生成建模。代码和预训练模型由VATIX开源，底层驾驶数据分阶段发布；复现重点是控制模型规模、训练暴露量和算力预算。局限是摘要未给出跨数据域泛化或感知闭环收益。

## 总结
核心思想：用缩放规律优化驾驶视频训练预算。
速记：改变模型规模→改变训练暴露→拟合损失幂律→选择预算→训练9B模型。

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

# 2608.27971v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
GAAT面向RGB、红外、SAR和深度融合中的局部错位问题，先估计对应可靠性，再进行跨模态交互。它通过无标注同步变换学习patch中心一致性，产生token/query置信度、query中心和子token偏移，并以此指导稀疏融合与对比预训练。

## 动机与方法
不同光学特性、分辨率、安装位姿及视差会破坏密集对齐。syncPATC学习几何先验；MG-Sparse-MMA仅在top-K_s可靠区域进行query-mediated融合，避免所有patch两两交互；RA-QCGCL以patch-patch、patch-query和query-query分支匹配这一稀疏瓶颈。UAVMeta与StateBench再用遥测和图像统计构造相机可靠性、观察尺度、视角稳定性、机动复杂度四类状态分数。

## 实验与实用性
六项下游任务均显示更强迁移，StateBench支持按真实采集条件诊断。创新是“先几何可靠性、后跨模态融合”，适合多传感器UAV。摘要未说明具体训练超参和开源状态；主要风险是top-K稀疏策略对极端错位及新传感器的敏感性。

## 总结
核心思想：用局部几何可靠性约束多模态融合。
速记：同步变换学习先验→筛选可靠query→稀疏融合→多分支对比→跨任务迁移。

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

# 2608.28383v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
作者发现视觉Transformer注意力头会分化为目标专家和背景专家，这种语义头专门化在全注意力下最明显。提出SHS-Index量化该现象，并以窗口交互、token序列化和局部softmax分配为设计原则，构造Ariadne Attention。

## 方法与贡献
研究先比较全注意力与chunk-window ViT的头部语义行为，再验证SHS-Index与下游性能的相关性，最后将三个结构因素组合为混合注意力。Ariadne在22项图像/视频任务上匹配全注意力，同时将注意力计算降至约1/6.5。相比凭经验设计窗口，本文把可测量的语义专门化作为结构诊断指标和设计信号。

## 实验、适用性与局限
适合多模态LLM中的视觉编码器，尤其是需要降低高分辨率图像/视频成本的场景。摘要未披露具体剪枝/训练超参、代码状态及跨模型复现细节；SHS-Index的因果性仍主要由相关性支持。

## 总结
核心思想：以注意力头语义分化设计高效混合注意力。
速记：测量头专门化→分析结构因素→设计混合注意力→多任务验证→降低计算。

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

# 2608.27795v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
uScenes是面向水下机器人感知的多模态数据集，包含同步3D多波束声纳点云与RGB图像，共110个场景、95,834次同步观测、277.6分钟多次外场采集数据。

## 动机与方法
弱光和后向散射会使相机失效；二维前视声纳虽能提供距离和方位，却缺少俯仰，导致回波难以定位到3D。数据集通过时间同步的RGB与3D声纳互补这一缺陷，为跨模态表示学习、传感器融合和3D场景理解提供统一基准。

## 实验与实用性
本文主要贡献是数据资源而非单一模型。适用于水下目标检测、3D重建、声纳-RGB融合和恶劣照明研究。代码与数据集已在GitHub提供。复现重点是处理同步、声纳稀疏性与跨模态坐标标定；摘要未给出基准任务结果和覆盖环境的完整统计，外推到不同水体需谨慎。

## 总结
核心思想：用同步RGB与3D声纳补齐水下感知。
速记：外场采集→时间同步→声纳/RGB配准→构建多模态基准→支持融合任务。

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

# 2608.27997v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
A-PAIR研究空地跨视角指代表达行人检测，包含22,137个跨视角指代表达样本。FARA以半自动方式生成分解式描述与身份一致性监督；ICRG联合分解式指代定位、候选完整性监督和跨视角一致性校准。

## 动机与方法
传统指代表达理解和开放词汇定位没有联合处理空地身份一致性，难以应对相似行人、空中视角外观弱和同一实体跨视角对应。ICRG同时选择地面候选、空中候选及其配对，使语言指令落到同一物理目标。

## 实验与适用性
相对强基线，地面、空中和配对检测均提升，pair F1从16.65%升至22.28%。适合空地协同、语言到感知到控制链路。数据和半自动标注框架有助于扩展规模；摘要未说明开源实现与复杂指令的失败模式，身份配对质量仍可能成为瓶颈。

## 总结
核心思想：把语言定位与跨视角身份一致性联合建模。
速记：分解描述→生成候选→跨视角配对→一致性校准→评估pair F1。

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

# 2608.28288v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
GeoFF3D将带坐标锚定的前馈模型与空间大规模重建框架结合，用地理参考相机平移和可选几何先验，直接预测重力对齐、Z-up度量坐标中的相机姿态与稠密点图。

## 方法与实验
SLRF把图像划分为有空间重叠的chunk，传播共享视图先验，并分层聚合局部重建；因此可复用不同有界视图模型，并避免长近共线轨迹下完整Sim(3)对齐不稳定。九个航测块上平均F@5从0.829提升到0.877；长UAVScenes序列为0.848，优于0.687和0.451基线；2000张图约5分钟。代码已开源。

## 适用性与局限
适合大范围UAV测绘、需要真实坐标和快速重建的场景。关键迁移条件是可靠地理坐标、重叠chunk和共享视图。摘要未展开先验缺失、GNSS误差及复杂地形下的退化情况。

## 总结
核心思想：用地理坐标锚定可扩展三维重建。
速记：坐标锚定预测→空间分块→传播重叠先验→层级聚合→输出大范围模型。

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

# 2608.28205v1（摘要级分析）
来源：abstract（native PDF tool unavailable）

## 摘要翻译
Cut-ViT针对视觉基础模型进行任务特定剪枝。它从空间和语义两方面构造Gram锚定矩阵并分解子空间，以无关基约束和残差约束对齐原生与剪枝DINOv3的空间/通道Gram子空间，再用谱熵适应不同下游任务的信息密度。

## 方法与贡献
点对点token对齐往往只依赖单一数据集，导致鲁棒性下降且缺乏任务特异性。Cut-ViT通过子空间继承原模型的稳健表示，并按空间与通道流形信息密度调整剪枝目标，因此在不同稀疏率下获得任务定制子网络。

## 实验与实用性
单张A100约一分钟生成多种稀疏率子网，仅用前方法20.9%的时间和45.5%的显存，在9个数据集、6项任务上达到SOTA。适合部署端视觉编码器压缩；实现关键是Gram矩阵分解、残差约束和谱熵估计。摘要未说明剪枝后微调成本及极高稀疏率稳定性。

## 总结
核心思想：用子空间一致性实现任务定制视觉剪枝。
速记：构造Gram锚点→分解子空间→约束原/剪枝表示→谱熵调目标→生成任务子网。

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

**Links:**

- [PDF](https://arxiv.org/pdf/2608.28140v1)
- [arXiv](https://arxiv.org/abs/2608.28140v1)

---

