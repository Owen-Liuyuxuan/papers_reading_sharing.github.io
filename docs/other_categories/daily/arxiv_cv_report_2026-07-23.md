time: 20260723

# Arxiv Computer Vision Papers - 2026-07-23

## Executive Summary

# 每日Arxiv计算机视觉论文执行摘要（2026-07-22）

**1. 主要主题与趋势**

本期论文集中在三大方向：**具身智能与机器人操作**（5篇）、**生成式模型与新表示学习**（3篇）、**多模态感知与自动驾驶**（2篇）。具身智能领域尤为突出，涵盖了从视觉跟踪、语言引导抓取到单视频快速技能习得、零售人形机器人学习以及扩散策略加速等全链条。生成式方面，3D高斯泼溅（3DGS）的紧凑前馈化、身份保真视频生成以及立体匹配中的扩散流方法成为新亮点。多模态预训练中的采样策略和自动驾驶中的世界-动作联合建模也展现了细分领域进展。

**2. 特别重要/创新的论文**

- **《Robots Acquire Manipulation Skills in Seconds from a Single Human Video》**（论文9）：提出从单段人类视频中秒级习得机器人操作技能的方法，极大降低了机器人学习的数据门槛，具有显著的实用潜力。
- **《ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion》**（论文2）：实现前馈式紧凑3DGS，通过自适应token扩展绕过传统逐场景优化，有望推动实时3D重建。
- **《Vera: Identity-Faithful Human Subject-to-Video Generation》**（论文5）：在人物视频生成中严格保持身份一致性，解决了当前扩散模型常见的身份漂移问题。
- **《STEREOFLOW: Progressive Stereo Matching with StereoDiT and Transition Flow Matching》**（论文10）：将扩散Transformer与流匹配结合用于立体匹配，提出渐进式框架，可能革新传统立体视觉算法。

**3. 新兴研究方向与技术**

- **语言引导的实施例感知**：如ReferTrack（语言引导跟踪）和SeededGrasp（语言引导抓取，支持多机器人形态），表明语言正成为机器人感知与动作的通用接口。
- **从单视频到技能迁移**：论文9展示的“单视频零样本技能获取”是机器人学习领域的重要突破，可能催生大规模演示数据集外的快速部署方案。
- **扩散策略推理加速**：论文4提出进化缓存调度来加速扩散策略推理，针对机器人实时控制场景，或将成为扩散模型实用化的关键技巧。
- **可见光-红外预训练中的非均匀采样**：论文6指出不同patch对多模态预训练贡献不同，为跨模态基础模型训练提供了新思路。
- **自适应专家路由**：论文8的PerceptDrive将感知先验与自适应专家路由结合，代表了端到端自动驾驶中模块化与可解释性的折中方向。

**4. 建议全文阅读的论文**

- **论文9（单视频机器人技能习得）**：对于机器人学习和模仿学习领域的研究者，这是必读项，方法新颖且实用性强。
- **论文2（前馈3DGS）**：所有从事3D重建、新视图合成的研究人员应关注，可能改变3DGS部署的范式。
- **论文5（身份保真视频生成）**：对视频生成、数字人、社交媒体应用感兴趣者的首选。
- **论文10（立体匹配+扩散流）**：计算机视觉基础问题（立体匹配）与生成模型的融合，兼具理论意义和工程价值。
- **论文8（自主驾驶感知-动作建模）**：自动驾驶领域读者可了解最新端到端架构设计与专家路由策略。

---

## Table of Contents

1. [ReferTrack: Referring Then Tracking for Embodied Visual Tracking](#2607.20061v1)
2. [ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion](#2607.20417v1)
3. [Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids](#2607.20345v1)
4. [Evolving Cache Schedules for Fast Diffusion Policy Inference](#2607.20293v1)
5. [Vera: Identity-Faithful Human Subject-to-Video Generation](#2607.20247v1)
6. [Not All Patches are Equal: Sampling Matters for Visible-Infrared Pre-Training](#2607.20238v1)
7. [SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments](#2607.20207v1)
8. [PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving](#2607.20175v1)
9. [Robots Acquire Manipulation Skills in Seconds from a Single Human Video](#2607.20033v1)
10. [STEREOFLOW: Progressive Stereo Matching with StereoDiT and Transition Flow Matching](#2607.19986v1)

---

## Papers

<a id='2607.20061v1'></a>
## [ReferTrack: Referring Then Tracking for Embodied Visual Tracking](https://arxiv.org/abs/2607.20061v1)

**Authors:** Hanjing Ye, Tianle Zeng, Jiazhao Zhang, Shaoan Wang, Zibo Zhang, Weisi Situ, Yuchen Zhou, Yonggen Ling, Hong Zhang

**Published:** 2026-07-22

**Categories:** cs.RO

**Abstract:**

Embodied visual tracking (EVT) requires a mobile agent to continuously follow a specific target described in natural language using only onboard vision. While recent vision-language-action (VLA) policies unify target identification and trajectory planning, their chain-of-thought (CoT) reasoning often operates in abstract spatial latents that are difficult to supervise and weakly aligned with explicit image-space detections. To address this, we introduce ReferTrack, a referring-then-tracking paradigm that grounds EVT using a single forward-facing camera. Our model first selects the target from an indexed set of bounding boxes, then decodes tracking waypoints conditioned on this image-grounded decision. To preserve target motion cues over time, ReferTrack maintains a sliding-window queue of previously selected bounding boxes, injecting their geometric features into the visual history via temporal-viewpoint-bbox indicator (TVBI) tokens. We further enhance target identification by co-training on a custom Refer-QA dataset. On EVT-Bench, ReferTrack achieves state-of-the-art single-view performance with success rates of 89.4%, 73.3%, and 74.1% on the single-target, distracted, and ambiguity tracking splits, respectively -- matching or even surpassing several multi-camera baselines on identification-heavy tasks. Finally, real-world deployments on legged and humanoid robots validate its robust sim-to-real transfer capabilities. Code is available at https://github.com/MedlarTea/referTrack.

**Analysis:**

### 1. 摘要翻译
具身视觉跟踪（EVT）要求移动代理利用车载视觉，持续追踪自然语言所描述的特定目标。现有的视觉-语言-动作（VLA）策略虽统一了目标识别与轨迹规划，但其思维链（CoT）推理往往基于难以监督且与显式图像空间检测弱对齐的抽象空间潜变量。为此，我们引入了 ReferTrack，这是一种通过单一前视摄像头实现 EVT 接地的“先指认、后追踪”范式。我们的模型首先从索引边界框集合中选择目标，随后基于这一图像接地决策解码追踪航点。为保留目标运动线索，ReferTrack 维护了一个先前的边界框滑动窗口队列，通过时间-视点-边界框指示（TVBI）标记将几何特征注入视觉历史。此外，我们通过在定制的 Refer-QA 数据集上进行联合训练，进一步增强了目标识别能力。在 EVT-Bench 上，ReferTrack 在单目标、干扰和模糊跟踪基准中分别取得了 89.4%、73.3% 和 74.1% 的成功率，性能达到领先水平，在强身份识别任务中甚至匹配或超越了多个多摄像头基线。在腿足式和人形机器人上的实际部署验证了其稳健的仿真到现实迁移能力。代码已开源。

### 2. 方法动机分析
*   **驱动力**：解决现有 VLA 在 EVT 中因依赖“抽象空间潜变量”导致推理不透明、目标对齐差的问题。
*   **痛点**：端到端模型缺乏对图像空间的显式接地（Grounding），导致在复杂、拥挤环境中无法准确锁定目标或维持跟踪稳定性。
*   **研究假设**：通过将“目标识别”转化为“索引边界框选择”这一离散选择问题，可以利用现有的检测器输出，显著降低推理难度，并提供强可监督性。

### 3. 方法设计详解
*   **核心流程**：
    1.  **检测与索引**：对每帧图像进行目标检测，构建包含所有候选对象的“索引目录（Indexed Catalog）”。
    2.  **Refer-CoT 推理**：模型根据指令在目录中选出一个目标索引（或 `<NO_EXIST>`），实现“指认”。
    3.  **轨迹规划**：将选定的目标及其历史特征作为条件，通过 Action Head 生成导航航点。
    4.  **历史记忆更新**：将选定的边界框存入 FIFO 队列，通过 TVBI Token 注入视觉历史，强化时序连续性。
*   **关键公式意义**：$E_{TVBI}(t) = E_{TVI}(t) + P_{bbox}(b_t)$。此公式将目标几何特征（边界框坐标）与通用的时间视点特征相加，使模型能够感知特定目标的时空运动演变，而非仅仅依赖于通用的视觉特征。

### 4. 方法对比分析
*   **本质区别**：从“黑盒思维链”转向“显式指认（Referring）后导航（Tracking）”。其他方法倾向于在潜空间推理，ReferTrack 将其显式化为离散索引选择。
*   **创新贡献**：引入索引化候选目录（Candidate Catalog）和 TVBI 机制，通过 Refer-QA 联合训练，成功将静态视觉接地任务的能力迁移到了动态导航任务中。

### 5. 实验分析（精简版）
*   **关键结果**：在单摄像头设置下，在 EVT-Bench 的干扰跟踪（DT）上较最强基线提升 6.8 SR。
*   **主要优势**：极强的抗干扰能力，且对多摄像头依赖度低，仅靠单目即可实现媲美多目系统的表现。
*   **主要局限**：高度依赖前置检测器（如 YOLO11）的质量，如果检测器在极端遮挡下失效，模型性能会受影响。

### 6. 实用指南
*   **开源地址**：https://github.com/MedlarTea/referTrack
*   **实现要点**：
    *   **两阶段训练**：先对齐视觉投影器（Stage 1），再进行联合导航与 QA 的微调（Stage 2）。
    *   **超参数**：$\alpha=10$ 作为 loss 的权重缩放因子；使用 AdamW 优化器，学习率 2e-5 (LLM) 和 1e-4 (其他模块)。
    *   **迁移建议**：该方法非常适合任何需要“对象特定导航”的任务，只需替换检测器目录接口即可迁移。

### 7. 总结
*   **核心思想**：通过显式索引化目标指认，将空间接地与长程导航解耦。
*   **速记版pipeline**：1. 检测并编号场景中所有行人；2. 指令驱动选择特定行人编号；3. 记录轨迹坐标入队；4. 基于指认结果规划下一步动作。

**Key Findings:**

- To address this, we introduce ReferTrack, a referring-then-tracking paradigm that grounds EVT using a single forward-facing camera.
- On EVT-Bench, ReferTrack achieves state-of-the-art single-view performance with success rates of 89.4%, 73.3%, and 74.1% on the single-target, distracted, and ambiguity tracking splits, respectively -- matching or even surpassing several multi-camera baselines on identification-heavy tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20061v1)
- [arXiv](https://arxiv.org/abs/2607.20061v1)

---

<a id='2607.20417v1'></a>
## [ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion](https://arxiv.org/abs/2607.20417v1)

**Authors:** Cho In, Jeonghwan Cho, Mijin Yoo, Gim Hee Lee, Seon Joo Kim

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

3D Gaussian Splatting (3DGS) achieves high-quality novel-view synthesis by optimizing freely placed primitives in 3D and adaptively densifying them in under-reconstructed regions. However, this scene-adaptive capacity allocation is largely lost in existing feed-forward 3DGS methods, which commonly regress Gaussians at input pixels and lift them along camera rays. Such pixel-aligned formulations make the number and placement of primitives depend on image resolution and input viewpoints rather than scene complexity, resulting in dense and often redundant Gaussian sets. We present ATSplat, a feed-forward 3DGS framework that restores the adaptive allocation capability of 3DGS optimization through Adaptive 3D Tokens. ATSplat first lifts coarse patch-level depth and camera cues into sparse 3D anchor tokens, forming a compact scaffold of the scene. Each token is then regressed into local Gaussians with learnable 3D offsets, decoupling primitive placement from input image grids. An Adaptive Token Expansion module predicts a token-level uncertainty score, supervised by rendering error maps, and selectively expands high-uncertainty tokens through learnable expansion layers. This sparse-to-adaptive formulation enables ATSplat to concentrate primitives in challenging regions while maintaining a compact representation. Experiments on two representative datasets, RealEstate10K and DL3DV, show that ATSplat achieves state-of-the-art rendering quality while reducing the number of Gaussians by more than $5.7\times$ compared with dense feed-forward 3DGS methods. From 12 input images at $512 \times 960$ resolution, ATSplat completes reconstruction in less than a second using a single commercial GPU, and renders high-quality novel views at 1136 FPS ($512 \times 960$) with only 311K Gaussians.

**Analysis:**

这是一份针对《ATSplat: Compact Feed-forward 3D Gaussian Splatting with Adaptive Token Expansion》的深度分析报告：

### 1. 摘要翻译
ATSplat 提出了一种紧凑的前馈 3D 高斯泼溅（3DGS）框架，通过自适应 Token 扩展技术，恢复了 3DGS 在 per-scene 优化中所具备的场景自适应能力。与现有的像素对齐式前馈方法将高斯基元固定在输入图像网格上不同，ATSplat 从稀疏的 3D 锚点（anchor）Token 出发，通过解码器中的自适应 token 扩展模块，根据重构难度对挑战性区域进行动态加密。这种稀疏到自适应的分配策略有效地将表达能力集中在复杂场景中，从而以较少的高斯基元实现高质量的新视角合成。实验表明，ATSplat 在保持前馈模型高效率的同时，高斯基元数量减少了 5.7 倍以上，并实现了实时渲染。

### 2. 方法动机分析
- **核心动机**：传统 3DGS 依赖于 per-scene 优化，成本高；现有前馈 3DGS 方法（Pixel-aligned）简单地将高斯基元与像素一一对应，导致冗余度极高且无法随场景复杂度动态调整容量。
- **痛点**：像素对齐导致 primitive 数量完全取决于图像分辨率，在简单区域（如墙壁）浪费了大量计算资源，而在复杂区域（如细长结构）容量不足。
- **核心直觉**：重构容量的分配应基于“场景复杂度”而非“图像网格密度”。

### 3. 方法设计详解
- **流程总结**：
  1. **初始化**：多视图编码器提取 coarse  patch 特征并预测深度，将 patch 投影为稀疏 3D 锚点（anchor）。
  2. **解码与扩展**：图像到 3D 解码器通过交叉注意力融合细粒度信息。关键在于 **ATE (Adaptive Token Expansion) 模块**，它根据不确定性分数筛选并复制高难度区域的 token。
  3. **高斯生成**：每个 token 解码出局部高斯集，其中心点位移（offset）相对于锚点进行回归，实现 primitive 与输入网格的解耦。
- **算法解释**：
  - **ATE 模块**：通过辅助头预测 token 的不确定性分数，基于 2D 重构误差监督，实现无梯度下的“动态加密”。
  - **Anchor-offset 回归**：放弃了“沿着射线分布”的假设，允许高斯基元在 3D 空间中自由偏移，这是提升细节重构能力的关键。

### 4. 方法对比分析
- **本质区别**：从“基于网格的填充”转变为“基于场景内容的自适应撒点”。
- **创新点**：引入自适应 Token 扩展，将“不确定性”显式建模为重构容量分配的依据，实现了无需 per-scene 优化的动态密度控制。
- **适用场景**：高分辨率、高细节要求的新视角合成任务，尤其适合计算资源受限的实时渲染应用。

### 5. 实验分析
- **关键结论**：在 RealEstate10K 和 DL3DV 数据集上，相比同类前馈方法，在 PSNR/SSIM 指标持平或更优的前提下，高斯基元数减少了 5.7 倍以上。
- **主要优势**：极高的紧凑性、优秀的细节保持能力、极快的渲染速度（1136 FPS）。
- **局限**：目前缺乏针对已生成 Token 的修剪机制（Pruning），冗余 token 仍可能随层数增加。

### 6. 实用指南
- **开源地址**：[https://join16.github.io/page-atsplat](https://join16.github.io/page-atsplat)
- **实现细节**：建议使用 DINOv2 作为 encoder 骨干；ATE 模块的 token 选择比（selection ratio）是调优的关键超参数；训练时需要使用 MSE + Perceptual/D-SSIM 的联合损失。
- **迁移建议**：该“基于稀疏 token 动态加密”的架构可以很容易迁移到其他显式表示（如点云渲染或体素场），只需更换解码头的回归目标即可。

### 7. 总结
- **核心思想**：基于场景重构难度，动态自适应分配 3D 表达容量。
- **速记版 Pipeline**：
  1. 图像编码与稀疏 3D 锚点生成；
  2. 解码器基于预测难度动态分裂 token；
  3. 各 token 解码为局部偏移的高斯基元；
  4. 最终渲染高保真 3D 场景。

**Key Findings:**

- 3D Gaussian Splatting (3DGS) achieves high-quality novel-view synthesis by optimizing freely placed primitives in 3D and adaptively densifying them in under-reconstructed regions.
- We present ATSplat, a feed-forward 3DGS framework that restores the adaptive allocation capability of 3DGS optimization through Adaptive 3D Tokens.
- Experiments on two representative datasets, RealEstate10K and DL3DV, show that ATSplat achieves state-of-the-art rendering quality while reducing the number of Gaussians by more than $5.7\times$ compared with dense feed-forward 3DGS methods.
- From 12 input images at $512 \times 960$ resolution, ATSplat completes reconstruction in less than a second using a single commercial GPU, and renders high-quality novel views at 1136 FPS ($512 \times 960$) with only 311K Gaussians.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20417v1)
- [arXiv](https://arxiv.org/abs/2607.20417v1)

---

<a id='2607.20345v1'></a>
## [Closing the Lab-to-Store Gap: A Data-Efficient Post-Training and Experience-Driven Learning VLA Framework for Retail Humanoids](https://arxiv.org/abs/2607.20345v1)

**Authors:** Roger Sala Sisó, Tiago Silvério, Jakob Sand, Tran Nguyen Le

**Published:** 2026-07-22

**Categories:** cs.RO, cs.AI

**Abstract:**

Closing the gap between benchmark performance and reliable real-world operation remains a central challenge for Vision-Language-Action (VLA) humanoid robots, which must handle execution errors, distribution shifts, and environmental variability. This paper presents DEED (Data-Efficient Post-Training and Experience-Driven Learning), a systems-level approach evaluated on a supermarket chip-restocking task using a Unitree G1-Edu humanoid robot and the GR00T N1.6 foundation model. DEED comprises three key components: (1) a data-efficient post-training pipeline with control-frequency alignment, data curation, task-relevant visual highlighting, and reduced VLA dependence; (2) a real-world study of experience-driven refinement, adapted from RECAP via a text-based advantage prefix and a vision-language value function; and (3) a latent-space analysis tool for studying in- and out-of-distribution behavior. Our results suggest that bridging the lab-to-store gap is primarily a systems integration challenge rather than an architectural one: careful data design and targeted post-training can transform a policy that fails under naive fine-tuning into a competent real-world system using only a single GPU.

**Analysis:**

## 1. 摘要翻译
本文旨在解决视觉-语言-动作（VLA）人形机器人基准性能与可靠实际部署之间的巨大鸿沟，这些鸿沟主要源于执行错误、分布偏移和环境多变性。我们提出了DEED（数据高效后训练与经验驱动学习）框架，并在Unitree G1-Edu人形机器人执行超市补货任务中进行了评估。DEED包含三个关键部分：（1）一种数据高效的后训练流水线，结合了控制频率对齐、数据清洗、任务相关视觉高亮及降低对VLA依赖的策略；（2）一种适应于RECAP的经验驱动改进研究，采用文本优势前缀和视觉-语言值函数；（3）一种用于研究分布内/外行为的潜空间分析工具。结果表明，跨越“实验室到商场”的鸿沟主要是一个系统集成挑战，而非单纯的架构问题：通过仔细的数据设计和针对性的后训练，仅使用单个GPU即可将无法正常工作的策略转化为稳健的现实系统。

## 2. 方法动机分析
*   **驱动力**：作者试图解决VLA在受限的实验室基准测试到非受限的复杂真实环境部署之间的可靠性鸿沟。
*   **现有方法痛点**：预训练VLA模型在面对实际环境的执行噪声和分布偏移时极易失效，且离线策略缺乏从自身部署经验中持续改进的机制。
*   **研究假设**：机器人部署的可靠性更多依赖于系统工程决策（数据质量、控制频率对齐、任务感知）而非仅靠改进模型架构。

## 3. 方法设计详解
DEED框架分为两阶段：
1.  **数据高效（DE）后训练**：
    *   **频率对齐**：强制执行 $f_r = f_{ctrl} \le f_{cam}$ 且 $f_t \ge f_r$ 的层级，确保机器人感知和动作在时间语义上是一致的。
    *   **数据清洗规则**：剔除冗余或含噪数据，保留成功路径及有效的失败回复（Recovery），避免训练无效的“空操作（No-op）”。
    *   **任务相关视觉高亮**：利用IA-VLA通过掩码高亮任务目标（如空货架），减少VLA对背景噪声的关注。
    *   **实用Trick**：采用二值化手部控制器解决高频抓取控制难题；使用Butterworth滤波器平滑动作输出，减轻抖动。
2.  **经验驱动（ED）改进**：
    *   将RECAP方法适配至GR00T架构。通过引入“Advantage=True/False”文本前缀将优势信息注入策略。
    *   **值函数设计**：设计了一个独立于策略的MLP值函数，利用Eagle-3特征进行离散化优势评估。优势标签 $A^{(N)}_t$ 通过多步回报估算，赋予智能体对未来任务进度的感知。
3.  **潜空间分析工具**：
    *   将状态投影至潜空间并使用GMM建模。通过Mahalanobis距离检测实时状态是否偏移了训练分布（OOD detection），为诊断性能衰减提供定量依据。

## 4. 方法对比分析
*   **本质区别**：与通用RL不同，DEED强调利用系统层面的工程手段对模型进行“修剪”与“对齐”，而非单纯增加数据规模或修改主干网络。
*   **创新贡献**：成功将RECAP从端到端模型迁移至解耦架构；提供了一套完整的从工程预处理到部署监控的工程范式。
*   **适用场景**：适合中等规模数据集、计算资源有限且需要高可靠性的特定长周期任务（如工业补货）。

## 5. 实验分析（精简版）
*   **验证方法**：在超市补货真实环境下对比原始模型与DEED各阶段的表现。
*   **关键结果**：DEED将任务成功率从0%（原始）提升至32%（DE阶段），在进一步经验驱动后达到42%。但多轮精炼可能导致分布过拟合，需保持数据平衡。
*   **优势**：在单GPU上显著提高了模型在真实物理场景下的鲁棒性。
*   **局限**：模型在连续多次任务后容易产生分布偏移，导致性能随迭代衰减。

## 6. 实用指南
*   **实现细节**：频率设置至关重要；在微调时，保持预训练主干冻结，仅微调头部，能大幅降低训练成本。
*   **迁移可能**：该框架的潜空间监控工具和数据清洗规则具有极强的通用性，可直接应用于其他基于视觉的端到端机器人任务。

## 7. 总结
*   **核心思想**：通过系统工程优化与在线经验修正闭环，实现VLA高效部署。
*   **速记版pipeline**：
    1.  规范化控制频率与传感器采样；
    2.  清洗数据并加入视觉目标高亮；
    3.  微调策略并引入优势条件控制；
    4.  通过分布分析工具监控异常行为。

**Key Findings:**

- DEED comprises three key components: (1) a data-efficient post-training pipeline with control-frequency alignment, data curation, task-relevant visual highlighting, and reduced VLA dependence; (2) a real-world study of experience-driven refinement, adapted from RECAP via a text-based advantage prefix and a vision-language value function; and (3) a latent-space analysis tool for studying in- and out-of-distribution behavior.
- Our results suggest that bridging the lab-to-store gap is primarily a systems integration challenge rather than an architectural one: careful data design and targeted post-training can transform a policy that fails under naive fine-tuning into a competent real-world system using only a single GPU.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20345v1)
- [arXiv](https://arxiv.org/abs/2607.20345v1)

---

<a id='2607.20293v1'></a>
## [Evolving Cache Schedules for Fast Diffusion Policy Inference](https://arxiv.org/abs/2607.20293v1)

**Authors:** Siying Wang, Kangye Ji, Di Wang, Fei Cheng

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

Diffusion policies achieve strong visuomotor control by iteratively denoising action chunks, but repeated denoising makes real-time deployment computationally demanding. Cache-based methods reduce inference cost by reusing intermediate activations, but existing training-free schedules typically allocate computation uniformly across blocks, ignoring heterogeneous redundancy across blocks and leading to a suboptimal performance-efficiency trade-off. To bridge this gap, we introduce Evolving Cache Schedules (EVO), a training-free acceleration framework that globally schedules cache refreshes via evolutionary search. EVO represents each candidate as a complete schedule over the block-timestep lattice. Thus, redundant transformer computations during iterative denoising can be skipped through cache reuse while preserving closed-loop rollout performance. To make the search practical, EVO introduces redundancy-aware initialization, which seeds the population with promising schedules, and target-conditioned early stopping, which verifies and terminates once a desired performance target is reached. The offline-optimized schedule can be directly plugged into pretrained diffusion policies without retraining. Extensive manipulation benchmarks show that EVO preserves near-full performance while substantially reducing computation, achieving up to 8.05x action-generation speedup and reducing FLOPs from 15.77G to as low as 1.96G. Source code is available at https://github.com/pillom/EVO.

**Analysis:**

### 1. 摘要翻译
扩散策略通过迭代去噪动作块实现强大的视觉运动控制，但重复的去噪过程在实时部署中计算压力巨大。基于缓存的方法通过重用中间激活来降低推理成本，但现有的训练无关（training-free）调度通常在各层间均匀分配计算，忽略了层间异构的冗余度，导致性能与效率的权衡欠佳。为弥补这一缺口，我们引入了进化缓存调度（EVO），这是一种通过进化搜索全局规划缓存刷新的训练无关加速框架。EVO将每个候选方案表示为块-时间步（block-timestep）网格上的完整调度，从而在保持闭环部署性能的同时，通过缓存重用跳过迭代去噪中的冗余Transformer计算。为提升搜索实用性，EVO引入了冗余感知初始化（以有潜力的调度播种种群）和目标条件提前终止（验证并达到预期性能目标即停止）。优化后的离线调度可直接插入预训练的扩散策略中，无需重新训练。在机器人操作基准上的实验表明，EVO在保持近乎全精度性能的同时，大幅降低了计算量，实现了最高8.05倍的动作生成加速，并将FLOPs从15.77G降低至1.96G。

### 2. 方法动机分析
*   **驱动力**：旨在解决Transformer-based扩散策略在实时机器人控制中的高推理延迟问题，同时避免昂贵的模型再训练或蒸馏代价。
*   **现有方法痛点**：现有缓存方法（如EfficientVLA、BAC）要么采用固定间隔刷新，要么在各块间进行受限的局部均衡分配，忽略了不同Transformer块在不同时间步上的**计算敏感度差异**。
*   **研究假设**：Transformer不同层级和模块类型在迭代去噪过程中存在异构的冗余模式，通过全局的、针对闭环性能的调度搜索，能够比启发式固定规则更好地实现计算资源的帕累托最优分配。

### 3. 方法设计详解
*   **流程总结**：
    1.  **网格建模**：将策略推理建模为“块-时间步”的二维网格 $B \times T$。
    2.  **进化搜索**：
        *   **初始化**：结合“冗余感知采样”（基于激活值的特征相似度作为先验概率）与“随机采样”，使初始种群既具针对性又不失多样性。
        *   **搜索与评价**：以闭环仿真环境下的“成功率”为适应度函数（Fitness）。
        *   **算子**：包含Tournament选择、集合级交叉、随机变异及修复算子（确保每个候选集满足预设的缓存预算 $K$）。
    3.  **加速部署**：根据搜索出的最优调度表 $S^*$，在推理时选择性地刷新或重用缓存，无需修改原模型权重。
*   **关键机制**：
    *   **目标条件提前终止**：通过快速评估（Quick Evaluation）筛选有潜力的候选，仅对达标个体进行正式评估（Formal Evaluation），大幅减少了离线搜索成本。

### 4. 方法对比分析
*   **本质区别**：与现有方法最大的不同在于，它是“全局预算分配”而非“局部块内均衡”，且以闭环任务成功率直接作为搜索目标，而非特征相似度等代理指标。
*   **创新贡献**：提出了一种结合进化算法、冗余先验与任务性能反馈的自动缓存调度范式。
*   **适用场景**：适用于任何基于Transformer的离线或预训练扩散模型加速，特别是对延迟敏感的具身智能任务。

### 5. 实验分析（精简版）
*   **验证方法**：在RoboMimic、Push-T、Kitchen等主流机器人操作基准上进行了详尽测试。
*   **关键结论**：在保持Success Rate几乎无损（如0.78 vs 0.80）的前提下，将推理延迟降低了约3.5倍，FLOPs压缩约8倍。
*   **核心优势**：无需训练、计算资源需求可控、任务兼容性强。
*   **主要局限**：依赖闭环仿真环境进行进化搜索，如果环境构建困难或任务极度复杂，搜索本身的离线成本可能较高。

### 6. 实用指南
*   **开源情况**：已开源，参见 [https://github.com/pillom/EVO](https://github.com/pillom/EVO)。
*   **实现要点**：
    *   **超参数**：重点配置进化代数、种群规模及缓存预算（$M$值）。
    *   **初始化**：必须预先计算一次特征dissimilarity矩阵，以获得引导搜索的Prior。
*   **迁移建议**：通过替换适应度函数（计算环境成功率的函数），可直接应用于图像生成或其他Transformer推理加速任务。

### 7. 总结
*   **核心思想**：通过进化算法实现Transformer缓存资源的跨层级全局最优分配。
*   **速记版Pipeline**：
    1. 计算各层冗余度，初始化候选策略库；
    2. 评估各策略在闭环任务中的成功率；
    3. 通过进化算子演化最优缓存调度表；
    4. 部署时按表执行，跳过计算实现加速。

**Key Findings:**

- To bridge this gap, we introduce Evolving Cache Schedules (EVO), a training-free acceleration framework that globally schedules cache refreshes via evolutionary search.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20293v1)
- [arXiv](https://arxiv.org/abs/2607.20293v1)

---

<a id='2607.20247v1'></a>
## [Vera: Identity-Faithful Human Subject-to-Video Generation](https://arxiv.org/abs/2607.20247v1)

**Authors:** Yulong Xu, Xinyue Liu, Shujuan Li, huafeng shi, Yan Zhou, Jiwen Liu, Xintao Wang, Yu Shen Liu, Huaibo Huang

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

Subject-to-video (S2V) generation has made substantial progress in preserving reference subjects across diverse categories, yet generic subject consistency remains insufficient for human-centric generation. A video may appear globally consistent while identity-critical human details still drift across frames, poses, and interactions. This issue becomes more severe in multi-person scenarios, where incorrect identity-role binding leads to subject confusion, attribute swapping, and excessive copying of reference-specific appearance cues. We propose Vera, a unified human-centric S2V framework for single- and multi-person generation. We first construct a million-pair identity-aligned human image-video dataset through person-level cross-clip retrieval, providing explicit identity correspondence and diverse references. Built on this dataset, Vera introduces two complementary designs. Identity-Focal Masked Supervision (IFMS) strengthens identity-aware learning with spatially focused supervision while reducing interference from irrelevant artifacts. Reference-Aware Layer-wise Attention (RALA) regulates how video tokens interact with reference identity cues in the DiT backbone, preserving stable identity anchors and enhancing layer-aware identity readout. Extensive experiments demonstrate that Vera improves human identity consistency, multi-person subject binding, and motion naturalness, while reducing identity confusion and excessive reference-image copying.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《Vera: Identity-Faithful Human Subject-to-Video Generation》的分析如下：

### 1. 核心贡献总结
Vera 提出了一个统一的人像视频生成框架，旨在解决现有 Subject-to-Video（S2V）方法在人像身份保持（Identity Preservation）上的不足，特别是解决了多人物场景下常见的身份漂移、属性错配及身份混淆问题。该研究通过构建大规模身份对齐数据集，并在扩散模型（DiT）中引入精细化的注意力与监督机制，显著提升了生成视频中人物的身份一致性与动作自然度。

### 2. 关键创新与方法论
该论文的创新点主要集中在以下三个方面：
*   **数据集层面的突破**：构建了一个包含百万级身份对齐的人像视频数据集，通过“跨片段检索”技术，为模型提供了高质量的显式身份对应关系。
*   **Identity-Focal Masked Supervision (IFMS)**：引入空间聚焦的掩码监督机制，强制模型关注人体的核心特征区域，从而减少背景或无关伪影对身份特征学习的干扰。
*   **Reference-Aware Layer-wise Attention (RALA)**：对 DiT 主干网络的注意力机制进行解耦优化。通过在不同层级引导视频 Token 与参考身份特征的交互，实现了对身份锚点的稳定控制，并改善了对身份细节的“读出”能力，防止了模型对参考图像的机械复制（Over-copying）。

### 3. 对领域的潜在影响
*   **重塑视频生成标准**：现有的 S2V 模型往往以牺牲“身份真实性”来换取“运动多样性”，Vera 的出现可能将视频生成的研究重点从单纯的视觉质量转移到更具挑战性的“身份保真度”上。
*   **多人物生成范式的变革**：解决多人物场景下的身份错位（Identity-role binding）是目前视频生成的“圣杯”问题之一，Vera 的多人物处理框架为行业应用提供了可落地的技术路径。

### 4. 受益的相关领域与应用
*   **影视特效与虚拟数字人**：在电影制作、数字人合成中，保持演员在复杂动作和交互中的身份稳定性是核心需求。
*   **个性化短视频创作**：允许用户通过一张参考图生成带有自身形象、且能稳定进行各类复杂动作的 AI 视频。
*   **互动娱乐与游戏**：在 NPCs 或玩家化身生成中，确保多角色在复杂社交互动场景下不出现身份特征交换。

### 5. 可推断的局限性
尽管摘要表现出色，但基于现有技术框架，仍存在以下潜在局限：
*   **计算开销**：由于引入了层级化的注意力机制（RALA）和大规模的数据集训练，模型的推理效率（Inference Speed）和对算力的要求可能较高。
*   **极端动作泛化**：虽然 IFMS 增强了特征学习，但在剧烈形变、极度遮挡或罕见交互动作下，身份特征是否仍能保持，尚需进一步验证。
*   **长视频稳定性**：摘要未提及时间维度上的长期一致性（Long-term temporal consistency），在生成超长视频时，Identity Drift 是否会随时间累积仍是一个潜在挑战。

**专家点评：**
Vera 的趣味性在于它没有盲目追求“泛化性”，而是选择“深耕人像领域”。通过将视觉注意力机制与身份验证思想（Identity-aware）深度耦合，该论文有效地将计算机视觉中的“识别（Recognition）”逻辑引入了“生成（Generation）”任务，这正是目前生成式 AI 向高精度、可控性演进的重要趋势。

**Key Findings:**

- We propose Vera, a unified human-centric S2V framework for single- and multi-person generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20247v1)
- [arXiv](https://arxiv.org/abs/2607.20247v1)

---

<a id='2607.20238v1'></a>
## [Not All Patches are Equal: Sampling Matters for Visible-Infrared Pre-Training](https://arxiv.org/abs/2607.20238v1)

**Authors:** Qiwei Ma, Bin Deng, Junjie Zhu, Qiangjuan Huang, Puhong Duan, Ke Yang, Xudong Kang, Shutao Li

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

Visible-infrared (VIS-IR) alignment is a key pre-training task for robust multi-sensor perception. Most existing methods use uniform patch-wise contrastive learning, but this can be unreliable in VIS-IR data because imaging-physics differences make some spatially paired regions inherently less comparable, and aligning them with equal strength hinders representation learning and downstream transfer. In this paper, we revisit VIS-IR pre-training from a sampling perspective and propose Importance-Aware Sampling (IAS), which adjusts training emphasis based on patch reliability. Specifically, IAS (i) derives patch weights from infrared structural cues and uses them to reweight the contrastive objective; (ii) learns a soft importance mask with a lightweight sampler, optionally warm-started from the hand-crafted prior; and (iii) employs a patch curriculum learning strategy that gradually expands from high-reliability regions to harder patches. It is worth noting that IAS is plug-and-play and works with both patch-/correlation-level alignment (e.g., UNIV-style) and image-level contrastive baselines (e.g., ImageBind-style). Extensive experiments on multiple VIS-IR benchmarks demonstrate consistent improvements over strong baselines, including for IR semantic segmentation, IR object detection and VIS semantic segmentation and cross-modal retrieval task. Code will be released on https://github.com/KlayMa527/IAS.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《Not All Patches are Equal: Sampling Matters for Visible-Infrared Pre-Training》的分析如下：

### 1. 论文核心贡献总结
该论文指出在可见光-红外（VIS-IR）多模态预训练中，传统均匀采样忽视了模态间成像物理机制差异导致的局部区域不可比性。作者提出了一种**重要性感知采样（Importance-Aware Sampling, IAS）框架**，通过动态加权与课程学习策略，引导模型优先对齐高可靠性区域，从而显著提升了多模态表征的学习质量及下游任务的迁移表现。

### 2. 关键创新与方法论
该方法的核心在于打破了“全图同等对待”的范式，通过三个维度实施精细化采样：
*   **物理先验驱动的加权：** 利用红外图像的结构化线索（Structural Cues）评估局部补丁（patch）的对齐难度，从而动态调整对比学习目标的权重。
*   **轻量级采样器（Soft Importance Mask）：** 引入可学习的掩码机制，不仅能利用启发式先验，还能通过模型训练自适应调整对不同区域的关注度。
*   **补丁级课程学习（Patch Curriculum Learning）：** 遵循“从易到难”的原则，训练过程从高度对齐的特征区域逐渐扩展至模态差异显著的复杂区域，防止模型在早期阶段被噪声干扰。

### 3. 对计算机视觉领域的潜在影响
该论文的**“即插即用（Plug-and-Play）”特性**使其具有极高的实用价值：
*   **打破对比学习范式：** 它挑战了当前主流的多模态预训练（如 ImageBind, UNIV）中对Patch进行无差别对齐的“懒惰”做法，为模态间存在巨大分布差异（如热成像与RGB）的任务提供了新的优化范式。
*   **提升特征鲁棒性：** 通过显式地剔除或弱化“不可比”区域的噪声干扰，该方法有助于提升多模态模型在极端光照、恶劣天气等复杂环境下的感知准确性。

### 4. 相关领域与受益应用
这项研究将直接利好对可靠性要求极高的应用场景：
*   **自动驾驶：** 提升在全天候环境（尤其是夜间或强光反射）下对行人、车辆的检测与分割能力。
*   **红外遥感与目标监测：** 改善红外图像在缺乏纹理信息时的语义理解能力。
*   **医疗成像：** 若涉及多种成像模态的融合，此采样策略可有效处理不同模态间物理空间的不对齐问题。
*   **跨模态检索：** 优化VIS与IR模态间的嵌入空间一致性，提升检索精度。

### 5. 潜在局限性（基于摘要分析）
*   **计算开销的边际成本：** 虽然IAS是“轻量级”的，但引入额外的采样器和掩码计算可能会在训练阶段增加内存占用和训练时长。
*   **对红外先验的依赖度：** 论文提到IAS可从手工设计的先验“冷启动”，但如果红外成像质量极差或结构信息退化，该先验是否依然有效，或者是否会引入系统性偏差，值得商榷。
*   **泛化到其他异构模态的难度：** 该方法高度依赖“红外结构化线索”这一特定领域知识，将其直接迁移到如“深度图-RGB”或“CT-MRI”等其他模态配对任务时，可能需要重新设计重要性度量标准。

**专家总结：**
这篇论文的有趣之处在于它将**“注意力机制”的理念延伸到了数据采样层面**。在多模态预训练趋向于“暴力计算”的大模型时代，这种回归数据质量、关注“采样效率”的思路显得尤为务实且具有启发性，是解决跨模态对齐中“噪声不可比”这一顽疾的有力手段。

**Key Findings:**

- Extensive experiments on multiple VIS-IR benchmarks demonstrate consistent improvements over strong baselines, including for IR semantic segmentation, IR object detection and VIS semantic segmentation and cross-modal retrieval task.
- Code will be released on https://github.com/KlayMa527/IAS.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20238v1)
- [arXiv](https://arxiv.org/abs/2607.20238v1)

---

<a id='2607.20207v1'></a>
## [SeededGrasp: Language-Guided Grasping in Complex Scenes with Multiple Embodiments](https://arxiv.org/abs/2607.20207v1)

**Authors:** Yang Xu, Gurpreet Singh Mukker, Raymond Wang, Jasper Gerigk, Maria Attarian, Igor Gilitschenski

**Published:** 2026-07-22

**Categories:** cs.RO

**Abstract:**

Practical robotic grasping in complex scenes requires both 3D spatial reasoning and alignment with task-specific requirements. Vision-language models (VLMs) offer a natural way to specify these requirements using language, but existing approaches either use a VLM to predict the grasp directly with limited spatial awareness, or train the VLM together with the grasping model, which requires significantly more data and compute. These limitations impede performance and have prevented scaling to multiple embodiments in complex scenes. We address this by proposing SeededGrasp, a novel data-efficient framework that enables a VLM to predict a seed point to be used as conditioning for a subsequent lightweight grasp-generation model. Our architecture decouples high-level semantic reasoning from low-level geometric execution, enabling multi-embodiment support while bypassing the need for expensive end-to-end training. To enable training such models, we release the first multi-embodiment tabletop grasping dataset comprising over 2.5M grasps in cluttered scenes. Experimental results demonstrate that our approach outperforms existing baselines, achieving 72% success in simulation and 78% in real-world grasping experiments. See our project site for data and code: https://uoft-isl.github.io/seeded-grasp/

**Analysis:**

### 1. 摘要翻译
实际场景中的机器人抓取不仅需要3D空间推理，还需与任务需求对齐。视觉语言模型（VLM）为通过语言指定这些需求提供了自然的方式，但现有方法要么直接利用VLM进行抓取预测而缺乏空间感知，要么将VLM与抓取模型进行端到端联合训练，这消耗了大量的计算资源和数据。这些局限性阻碍了多形态机器人系统在复杂场景中的扩展。为此，我们提出了SeededGrasp，这是一种数据高效的框架，使VLM能够预测一个作为后续轻量级抓取生成模型条件（Conditioning）的“种子点”。我们的架构将高级语义推理与低级几何执行解耦，在支持多种机器人形态的同时避免了昂贵的端到端训练。为了训练该模型，我们发布了首个包含超过250万个复杂场景抓取数据的多形态桌面抓取数据集。实验表明，该方法优于现有基线，在仿真中达到了72%的成功率，在真实世界实验中达到了78%。

---

### 2. 方法动机分析
- **核心驱动力**：旨在构建一种能够桥接“自然语言任务意图”与“多机器人形态几何约束”的轻量化框架。
- **现有痛点**：
    1. **端到端训练成本高**：直接训练VLM处理抓取需要极其庞大的数据和计算资源。
    2. **泛化能力差**：以往方法大多绑定特定抓取器，难以在复杂场景中支持多种机器人形态。
    3. **空间感知不足**：仅靠BEV图像预测抓取无法解决3D场景中的深度对齐与遮挡问题。
- **核心假设**：通过VLM预测一个三维“种子点”作为中间表示，即可实现语义任务与几何执行的有效解耦。

---

### 3. 方法设计详解
- **Pipeline**：
    1. **种子点选择（Seed Point Selection）**：将场景BEV图像与指令输入预训练VLM，由VLM在图像平面输出一个像素坐标（y, x），并映射至3D点云，定义抓取中心。
    2. **点云编码（Point Cloud Encoding）**：分别处理场景点云（$p_c$）与机器人点云（$p_r$），通过Fourier编码与PCA估计局部特征。引入可学习的“机器人查询向量（Robot Query）”以编码不同末端执行器的形态差异。
    3. **抓取生成（Grasp Generation）**：基于条件流匹配（Flow Matching）模型，以种子点、机器人特征向量和场景特征为条件，通过ODE求解将噪声采样迭代转化为具体的抓取姿态（旋转、位移、关节角）。
- **关键设计**：
    - **流匹配（Flow Matching）**：相比传统GAN或VAE，流匹配在生成复杂多模态抓取分布时具有更好的收敛性和准确性。
    - **Classifier-Free Guidance**：在训练中随机丢弃条件信号，在推理时通过外推强化模型对语义意图的对齐程度。

---

### 4. 方法对比分析
- **本质区别**：本文不是直接从语言预测动作，而是将语言作为“定位器”，将几何动作生成作为一个“以点为条件的分布采样过程”。
- **创新点**：提出了一个解耦的模块化范式，通过“种子点”这一轻量级界面，实现了VLM与具体机器人抓取控制的松耦合。
- **适用场景**：适用于多种机器人末端执行器共享同一场景语义认知的大规模复杂桌面操作。

---

### 5. 实验分析
- **验证方法**：在610个场景、334个物体及3种不同形态抓取器（Franka, Robotiq, Allegro）上进行大规模评估。
- **关键结果**：在多形态复杂对象场景中，相比基线方法（如Geomatch）成功率提升约35%。
- **主要局限**：模型目前缺乏手臂动力学约束，未考虑动作规划，且过度依赖预定义的点云处理。

---

### 6. 实用指南
- **开源情况**：项目地址：[https://uoft-isl.github.io/seeded-grasp/](https://uoft-isl.github.io/seeded-grasp/)
- **关键细节**：
    - **训练技巧**：在Loss中对不同动作分量（位移、旋转、关节角）赋予不同的权重系数（0.4, 0.04, 0.8）；采用L1 Loss而非L2以提升精度。
    - **迁移建议**：若要迁移至新抓取器，只需微调该机器人的“查询向量”及对应的关节掩码（Joint Mask），无需重新训练整个语义识别模型。

---

### 7. 总结
- **核心思想**：利用VLM进行任务点位引导，通过流匹配实现形态感知的几何抓取生成。
- **速记版Pipeline**：
    1. VLM看图定位（找种子点）。
    2. 编码器注入抓取器特征。
    3. 扩散模型将噪声平滑转化为抓取动作。

**Key Findings:**

- We address this by proposing SeededGrasp, a novel data-efficient framework that enables a VLM to predict a seed point to be used as conditioning for a subsequent lightweight grasp-generation model.
- Experimental results demonstrate that our approach outperforms existing baselines, achieving 72% success in simulation and 78% in real-world grasping experiments.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20207v1)
- [arXiv](https://arxiv.org/abs/2607.20207v1)

---

<a id='2607.20175v1'></a>
## [PerceptDrive: Perception Prior World-Action Modeling with Adaptive Expert Routing for End-to-End Autonomous Driving](https://arxiv.org/abs/2607.20175v1)

**Authors:** Yushan Liu, Tianxiong Lv, Bohua Wang, Hangqi Fan, Chenxu Zhao, He Zheng, Xuchang Zhong, Yifan Xie, Congyang Zhao, Zhihao Liao, Leigang Luo, Yang Cai, Xiao-Ping Zhang, Wenbo Ding

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

Frozen perception foundation models encode rich geometric, semantic, and dynamic knowledge. Yet narrow conditioning interfaces may attenuate task-relevant cues, while static fusion cannot adjust expert contributions to each scene. We cast this challenge as the prior-to-plan transfer problem and introduce PerceptDrive, a perception prior world-action modeling framework with adaptive expert routing. PerceptDrive feeds teacher-distilled priors from a frozen, driving-adapted provider and dense observation latents from a frozen self-supervised video encoder into a trainable expert-routed world-action model. Expert-specific query branches process these signals, while a prior-retention objective anchors each branch to its prior. A router predicts soft gates from a shared scene representation and combines the expert conditions before trajectory generation. During training, privileged rule-based sub-metric estimates for branch-specific trajectory drafts provide soft-gate distillation targets. The predicted action-free future latent conditions a flow-matching actor. At inference, privileged components are absent; with one front-facing camera, PerceptDrive generates one trajectory per planning step without test-time scoring, reranking, or search. Experiments show that PerceptDrive achieves state-of-the-art performance with 90.4 PDMS on NAVSIM v1 and 90.2 EPDMS on NAVSIM v2, outperforming existing methods. Ablations confirm complementary gains from prior retention and scene-conditioned routing, alongside differential reliance on the three priors. These results demonstrate that preserving and adaptively routing perception priors improves direct planning without test-time candidate selection.

**Analysis:**

### 1. 摘要翻译
冻结的感知基础模型编码了丰富的几何、语义和动态知识，但狭窄的条件接口可能削弱任务相关线索，而静态融合无法根据具体场景调整专家贡献。我们将此挑战表述为“先验到规划”的迁移问题，并引入了 PerceptDrive，这是一种具有自适应专家路由的感知先验世界-行动建模框架。PerceptDrive 将来自冻结的驾驶适配提供程序的教师蒸馏先验和来自冻结自监督视频编码器的稠密观测潜变量输入到可训练的专家路由世界-行动模型中。专家特定的查询分支处理这些信号，而先验保留目标将每个分支锚定到其先验上。路由器从共享场景表示中预测软门，并在轨迹生成前组合专家条件。在训练期间，用于分支特定轨迹草案的特权规则化子指标估计提供软门蒸馏目标。预测的无行动未来潜变量调节流匹配执行器。在推理时，特权组件被移除；仅需一个前视摄像头，PerceptDrive 即可在每个规划步骤生成一条轨迹，无需测试时评分、重排序或搜索。实验表明，PerceptDrive 在 NAVSIM v1 上达到 90.4 PDMS，在 NAVSIM v2 上达到 90.2 EPDMS，性能优于现有方法。消融实验证实了先验保留和场景条件路由的互补增益，以及对三种先验的差异化依赖。

---

### 2. 方法动机分析
- **驱动力**：解决端到端自动驾驶中“先验到规划”的迁移问题，即如何有效地利用冻结的预训练基础模型知识，并将其转化为对当前场景高度自适应的决策能力。
- **现有方法痛点**：
    1. **信息压缩损耗**：现有查询式规划器将复杂感知信号压缩为紧凑的 Token，导致场景相关的重要先验信息被稀释。
    2. **静态融合局限**：传统的静态加权无法根据特定场景动态调整几何、语义和动态先验的相对重要性。
- **核心研究假设**：通过在查询接口引入“先验保留”机制，并结合场景自适应的“软路由”机制，可以将异构感知先验与底层规划深度解耦并动态融合，从而实现直接生成最优轨迹。

---

### 3. 方法设计详解
- **流程总结**：
    1. **感知提供器构建**：先对 InternVL3 进行驾驶 QA 微调，再通过三个专家分支分支（GEO, SEM, DYN）蒸馏冻结教师模型（VGGT, V-JEPA, Wan 2.1）的知识，得到固定的感知先验表示 $H_t$。
    2. **专家路由世界-行动模型（WAM）**：
        - **信息整合**：将 $H_t$ 和冻结的 V-JEPA 视频编码器输出的稠密潜变量 $F_t$ 送入查询银行。
        - **先验保留（Retention）**：通过“保留探针”将分支输出拉向对应的先验目标，防止分支表示塌陷。
        - **自适应路由**：利用 MLP 路由器根据场景向量 $s_t$ 预测软门（Soft-gate），动态组合三个专家分支的条件。
    3. **轨迹生成**：利用预测的“无行动未来潜变量”调节流匹配（Flow-matching）执行器，直接生成单条最优轨迹。
- **关键算法**：通过训练阶段引入特权规则化指标（Privileged Metrics）对路由器进行蒸馏，将评估器的能力嵌入门控逻辑，推理时彻底卸载特权监督。

---

### 4. 方法对比分析
- **本质区别**：与传统 Mixture-of-Experts（MoE）路由计算资源不同，PerceptDrive 路由的是“先验条件的融合权重”，且通过目标锚定机制保留了各专业知识的独立性。
- **创新贡献**：提出“先验保留目标”和“特权路由蒸馏”，将传统的测试时候选重排序过程转化为训练时的高效知识迁移。
- **适用场景**：适用于需要将大规模预训练先验知识精准注入特定规划任务的场景。

---

### 5. 实验分析
- **结论**：在 NAVSIM v1/v2 协议下均达到 State-of-the-Art，证明了该框架在复杂环境下的规划稳定性。
- **优势**：无需测试时候选评分或重排序，推理延迟低；具备极好的场景自适应性。
- **局限**：高度依赖预先构建的特权监督池（Offline pool），且对所选先验教师模型的质量有较强依赖。

---

### 6. 实用指南
- **复现关键**：需预先准备并固定高质量的感知先验教师模型；训练过程分为两阶段（先感知适配/蒸馏，后 WAM 优化）。
- **迁移性**：该“先验保留+路由蒸馏”架构极易迁移至机器人操作或其他多模态决策任务，只需更换对应的先验专家和评价指标。

---

### 7. 总结
- **核心思想**：通过先验保留与特权路由蒸馏，将异构先验动态转化为精准的规划决策。
- **速记版pipeline**：
    1. 构建感知专家分支；
    2. 使用辅助探针锁定先验特征；
    3. 根据场景自适应预测路由权重；
    4. 融合专家条件经流匹配生成轨迹。

**Key Findings:**

- Yet narrow conditioning interfaces may attenuate task-relevant cues, while static fusion cannot adjust expert contributions to each scene.
- Experiments show that PerceptDrive achieves state-of-the-art performance with 90.4 PDMS on NAVSIM v1 and 90.2 EPDMS on NAVSIM v2, outperforming existing methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20175v1)
- [arXiv](https://arxiv.org/abs/2607.20175v1)

---

<a id='2607.20033v1'></a>
## [Robots Acquire Manipulation Skills in Seconds from a Single Human Video](https://arxiv.org/abs/2607.20033v1)

**Authors:** Guangyan Chen, Meiling Wang, Te Cui, Zichen Zhou, Qi Shao, Shalfun Li, Hang Su, Roy Gan, Hao Wang, Mengyin Fu, Yi Yang, Yufeng Yue

**Published:** 2026-07-22

**Categories:** cs.RO

**Abstract:**

The ability to acquire skills rapidly and effortlessly while retaining those already mastered is essential for robots. However, current methods still rely on a cumbersome training-time loop that is costly and slow, while eroding skills already mastered. In this paper, we introduce HOST (Human-to-robot One-Shot Skill AcquisiTion), a framework that enables a robot to acquire skills in seconds from a single human video while retaining previously mastered skills. HOST resolves skill acquisition through a cascade of self-grounded prediction. It first estimates the robot's progress within the demonstrated task, then translates the upcoming progression into the robot's own future observations, and finally derives actions from these predicted observations. This cascade is trained on targets coupled to the video demonstration, obtained by mapping the robot trajectory and the video demonstration onto a shared task progress manifold, then redefining each target to align with the future progression of the video. HOST thereby enables the robot to actively follow the demonstrated procedure and adapt it to the robot's embodiment. HOST acquires novel skills at inference time from a single human video in an average of 29 seconds and achieves a 62% average success rate. It exceeds the zero-shot baseline by 45% while retaining previously mastered skills. HOST even exceeds the baseline fine-tuned on 50 robot demonstrations per task while requiring 50 times fewer demonstrations and acquiring each skill 507 times faster. Additional information about HOST is available on the project website.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为《Robots Acquire Manipulation Skills in Seconds from a Single Human Video》的论文进行了如下深度分析：

### 1. 论文核心贡献总结
该论文提出了一种名为 **HOST (Human-to-robot One-Shot Skill AcquisiTion)** 的框架，使机器人能够仅通过一段人类操作视频，在几十秒内快速习得新技能，同时克服了传统强化学习或模仿学习中“灾难性遗忘”的问题。这项研究实现了从“人类演示”到“机器人执行”的高效跨域映射，显著降低了机器人学习新任务的时间成本与数据依赖。

### 2. 关键创新与方法论
HOST 的核心在于一种**“自接地预测级联”（Cascade of self-grounded prediction）**机制，具体创新点如下：
*   **任务进度流形（Task Progress Manifold）：** 论文通过将人类演示视频与机器人的轨迹映射到一个共享的进度流形中，建立了跨模态的对齐机制。
*   **预测级联：** 算法分为三步走：首先估计任务进度，随后将人类演示的未来预期转化为机器人视角的观测（Future Observations），最后从预测出的观测中导出机器人的动作序列。
*   **训练范式：** 这种方法无需在测试时进行昂贵的微调（Fine-tuning），而是通过将目标对齐到视频的未来进程中，实现了对机器人本体（Embodiment）的动态适应。

### 3. 对计算机视觉领域的潜在影响
*   **数据效率革命：** 将机器人学习所需的演示数据从“几十次/几百次”压缩到“一次（One-Shot）”，这打破了机器人学中长期存在的数据采集瓶颈。
*   **通用性提升：** 该方法展示了计算机视觉在解析人类行为意图并将其转化为可操作指令方面的强大能力。通过引入“进度流形”，它为解决视频模仿学习中的长尾问题提供了新的数学框架。
*   **非侵入式学习：** 这种只需视频输入的方法，意味着机器人未来可以通过观察互联网上海量的视频资源（如 YouTube/TikTok 操作教程）自我进化，无需专门的实验室数据收集。

### 4. 相关领域与应用前景
*   **具身智能 (Embodied AI)：** 直接推动家用辅助机器人和工业协作机器人的快速部署，使其能够通过观察人类行为快速“上手”。
*   **视频理解与行为识别：** 论文中提到的“任务进度估计”是视频理解的高阶应用，对视频动作分割、步骤规划具有借鉴意义。
*   **少样本学习 (Few-Shot Learning)：** 为其他需要从极少量参考样本中进行策略泛化的领域（如自动驾驶模拟器交互、虚拟人动画制作）提供了参考方案。

### 5. 可推断的局限性
*   **环境适应性风险：** 虽然论文提到能够适应机器人本体，但如果人类视频场景与机器人实际环境在物理属性（如光照、物体刚性、遮挡）上存在巨大差异，预测的“未来观测”可能会出现漂移。
*   **复杂动力学建模难度：** 视频仅包含视觉信息，缺乏力觉和触觉反馈。对于涉及精密力控（如精细组装）的任务，仅靠视觉预测生成的动作可能存在精度不足的问题。
*   **对视频质量的依赖：** 算法的效果很大程度上取决于视频演示的清晰度、动作连贯性以及对任务进度的准确标注，若视频存在非关键噪声或歧义，可能会影响模型的预测鲁棒性。

**专家总结：**
这篇论文的精妙之处在于它绕过了传统的“通过试错学习动作”的重型范式，转而利用**视觉空间中的进度对齐**来实现动作预测。这种“将视觉转化为行动”的级联策略，是当前具身智能研究中极具潜力的方向，特别是它在保留旧技能的同时实现快速习得，对于构建可持续更新的通用机器人系统具有极高的研究价值。

**Key Findings:**

- In this paper, we introduce HOST (Human-to-robot One-Shot Skill AcquisiTion), a framework that enables a robot to acquire skills in seconds from a single human video while retaining previously mastered skills.
- HOST acquires novel skills at inference time from a single human video in an average of 29 seconds and achieves a 62% average success rate.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.20033v1)
- [arXiv](https://arxiv.org/abs/2607.20033v1)

---

<a id='2607.19986v1'></a>
## [STEREOFLOW: Progressive Stereo Matching with StereoDiT and Transition Flow Matching](https://arxiv.org/abs/2607.19986v1)

**Authors:** Hao Wang, Haoran Geng, Xiaotong Yang, Jing Tang, Songlin Wei, Linlong Lang, Yeying Jin, Zheng Zhu, Zhaoxin Fan, Biao Leng

**Published:** 2026-07-22

**Categories:** cs.CV

**Abstract:**

Stereo matching is a fundamental task in 3D reconstruction. Despite remarkable advances, the prevailing paradigms formulate stereo matching as a deterministic regression problem, collapsing the multimodal distribution modeling into a single-point estimation. This formulation suffers from a regression-to-mean bias, frequently struggling with ambiguous regions. In contrast, we introduce a prior-guided generative framework that integrates deterministic matching regression and generative distribution modeling within a complementary formulation. Built upon this formulation, we introduce StereoFlow through three key components: (i) a two-stage progressive cascade matching network that progressively produces multi-resolution stereo conditions with complementary matching cues; (ii) a pixel diffusion transformer (termed StereoDiT) with a frequency-decoupled architecture for modeling correspondence ambiguity; (iii) a few-step flow matching objective (termed Transition Flow Matching) for efficient optimization. In summary, \textsc{\textbf{StereoFlow}} achieves strong geometric consistency and rich fine-grained details in ill-posed, discontinuous regions and under zero-shot generalization. Extensive experiments demonstrate that the proposed StereoFlow establishes multiple state-of-the-art results across benchmarks, including Scene Flow, KITTI, ETH3D, and Middlebury.

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为 **《STEREOFLOW: Progressive Stereo Matching with StereoDiT and Transition Flow Matching》** 的论文分析如下：

### 1. 论文核心贡献总结
该论文提出了一种全新的立体匹配范式，将传统的确定性回归建模与生成式分布建模相结合，有效解决了现有方法在处理多模态视差分布时的“回归均值偏差”问题。通过引入渐进式级联网络和基于流匹配（Flow Matching）的扩散模型，该研究在处理遮挡、纹理缺失等模糊区域时展现了卓越的鲁棒性和细节恢复能力，并确立了多项基准测试的最优性能（SOTA）。

### 2. 关键创新点与方法论
*   **生成式与确定性结合的混合架构**：改变了以往单一的回归建模方式，利用生成式模型对复杂的视差概率分布进行建模，从而捕捉多模态分布特征。
*   **StereoDiT（像素级扩散 Transformer）**：核心创新在于引入了频率解耦的 Transformer 结构，能够通过扩散过程精细化处理对应关系的模糊性。
*   **Transition Flow Matching（过渡流匹配）**：引入了高效的流匹配目标函数，解决了扩散模型通常面临的采样速度慢的问题，实现了从噪声到高精度视差图的快速、高效推理。
*   **渐进式级联策略**：通过多分辨率条件输入，由粗到精地逐步细化视差，为生成模型提供了强有力的几何先验引导。

### 3. 对领域的潜在影响
*   **范式转变**：该研究标志着立体匹配任务正从单纯的“深度估计器”向“概率分布生成器”转型，这对于解决计算机视觉中长久以来的“病态区域（ill-posed regions）”估计问题具有里程碑意义。
*   **零样本泛化能力的提升**：通过生成式建模捕捉更深层的几何先验，使得模型在未见过的复杂场景（Zero-shot）中展现出更强的泛化潜力，这是工业界部署的核心需求。
*   **扩散模型在 3D 视觉的应用深化**：展示了 Flow Matching 技术在除了图像生成以外的低层视觉任务（如立体匹配）中的高效适用性，为其他相关任务（如光流估计、点云配准）提供了新思路。

### 4. 相关领域与受益应用
*   **自动驾驶**：在处理远处、弱纹理物体或复杂光照下的深度感知时，该技术能显著提升避障的准确性。
*   **机器人视觉/SLAM**：对于需要高精度几何重建的机器人导航任务，更细致的边界处理和遮挡恢复至关重要。
*   **三维建模与 AR/VR**：在从静态影像生成高精度 3D 资产（NeRF 或 3DGS 输入）过程中，高质量的初始视差图是后续渲染效果的保证。

### 5. 可推测的局限性
*   **推理延迟（Inference Latency）**：尽管使用了 Flow Matching 进行加速，但与单次前向传播的端到端回归模型相比，涉及扩散采样过程的方法在实时性上可能仍面临挑战，特别是在高分辨率场景下。
*   **算力资源消耗**：扩散模型架构（DiT）通常需要较大的内存和计算开销，这可能限制其在边缘设备（如移动端手机或嵌入式系统）上的直接部署。
*   **训练稳定性**：混合确定性回归与生成式建模的 Loss 函数平衡（平衡正则化约束与生成质量）通常非常敏感，模型调优可能具有较高的复杂性。

---
**专家点评：** 这篇论文的趣味性在于它巧妙地将**扩散模型（Diffusion Models）的生成能力**引入到了**经典的几何匹配问题**中。它不仅仅是堆砌组件，而是深刻地指出了“回归到均值（Regression-to-mean）”这一痛点，并用生成流匹配的方式优雅地化解了该矛盾。这是目前 CV 领域将生成式 AI 与经典几何任务深度融合的优秀范例。

**Key Findings:**

- In contrast, we introduce a prior-guided generative framework that integrates deterministic matching regression and generative distribution modeling within a complementary formulation.
- Built upon this formulation, we introduce StereoFlow through three key components: (i) a two-stage progressive cascade matching network that progressively produces multi-resolution stereo conditions with complementary matching cues; (ii) a pixel diffusion transformer (termed StereoDiT) with a frequency-decoupled architecture for modeling correspondence ambiguity; (iii) a few-step flow matching objective (termed Transition Flow Matching) for efficient optimization.
- Extensive experiments demonstrate that the proposed StereoFlow establishes multiple state-of-the-art results across benchmarks, including Scene Flow, KITTI, ETH3D, and Middlebury.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.19986v1)
- [arXiv](https://arxiv.org/abs/2607.19986v1)

---

