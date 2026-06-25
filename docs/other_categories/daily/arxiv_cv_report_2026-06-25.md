time: 20260625

# Arxiv Computer Vision Papers - 2026-06-25

## Executive Summary

## 每日计算机视觉与机器人研究报告：执行摘要（2026-06-24）

### 一、主要主题与趋势

本期论文呈现出三大核心主题：**视频生成与自回归扩散模型**、**机器人操作与导航**、以及**三维场景重建与SLAM**。视频生成方面，扩散蒸馏与自回归方法结合成为关注焦点（论文1）；机器人领域占据半壁江山，涵盖跨本体操作先验学习、力觉操控、人群导航、主动SLAM、上下文世界模型、扩散规划以及高速里程计等子方向；三维重建则聚焦于从稀疏二维锚点生成一致的三维高斯街道场景。整体趋势表明：**生成式模型向实时、交互式世界模型演进**，**机器人学习越来越强调跨本体迁移、意图感知和鲁棒性**，**SLAM系统追求高速与精度平衡**。

### 二、显著创新论文

- **论文1《Causal-rCM》**：提出统一的教师强制与自强制蒸馏配方，用于自回归扩散蒸馏，可生成流式视频并构建交互式世界模型。该方法有望大幅降低视频生成推理成本，并支持实时环境交互，是当前扩散模型落地的重要突破。
- **论文10《FORCE》**：提出视觉-语言-动作（VLA）模型的高效强化微调框架，通过价值校准预训练和自蒸馏，解决了VLA模型在真实机器人任务中微调成本高、样本效率低的问题，具有实用价值。
- **论文5《RoboAtlas》**：将上下文信息融入主动SLAM，实现机器人动态环境下的自主探索与定位，为长期自主作业提供新范式。

### 三、新兴研究方向

1. **自回归扩散蒸馏与流式生成**：论文1代表的方向将自回归建模与扩散蒸馏结合，有望成为视频生成和交互式世界模型的核心技术。
2. **跨本体机器人学习**：论文2学习动作先验以实现跨机械臂构型的技能迁移，减少对特定硬件的依赖，是通用机器人操作的关键。
3. **肌电信号（sEMG）驱动的力控操作**：论文3引入生物信号实现力觉操控，为人机协作和远程操作提供新模态。
4. **意图感知的密集人群导航**：论文4将行人意图建模到场景表示中，提升导航安全性，适用于服务机器人。
5. **上下文世界模型**：论文6探索在机器人控制中构建可推理的世界模型，而非单纯基于策略，为模型预测控制提供新思路。
6. **高速LiDAR-惯性里程计**：论文8面向高速自主系统，在精度和鲁棒性上取得平衡，支撑无人机、自动驾驶等场景。

### 四、推荐全文阅读的论文（按优先级排序）

1. **论文1《Causal-rCM》** —— 视频生成与世界模型领域的关键技术进展。
2. **论文10《FORCE》** —— 实用VLA微调方法，对机器人研究者极具参考价值。
3. **论文5《RoboAtlas》** —— 主动SLAM的上下文感知创新，适合SLAM与自主导航方向。
4. **论文7《G2DP》** —— 扩散规划结合时空网格引导，在复杂长程任务中有潜力。
5. **论文9《From Sparse and Imperfect 2D Anchors...》** —— 街道场景三维重建的突破性方法，对自动驾驶和城市建模有直接帮助。

其余论文（2、3、4、6、8）也具有专业价值，但可在特定子领域内详读。

---

## Table of Contents

1. [Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models](#2606.25473v1)
2. [Learning Action Priors for Cross-embodiment Robot Manipulation](#2606.26095v1)
3. [ForceBand: Learning Forceful Manipulation with sEMG](#2606.26093v1)
4. [Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations](#2606.26047v1)
5. [RoboAtlas: Contextual Active SLAM](#2606.26046v1)
6. [In-Context World Modeling for Robotic Control](#2606.26025v1)
7. [G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance](#2606.26017v1)
8. [FAR-LIO: Enabling High-Speed Autonomy through Fast, Accurate, and Robust LiDAR-Inertial Odometry](#2606.26010v1)
9. [From Sparse and Imperfect 2D Anchors to Consistent 3D Gaussian Street Scenes: Support-Aware Appearance](#2606.26007v1)
10. [FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation](#2606.26006v1)

---

## Papers

<a id='2606.25473v1'></a>
## [Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation in Streaming Video Generation and Interactive World Models](https://arxiv.org/abs/2606.25473v1)

**Authors:** Kaiwen Zheng, Guande He, Min Zhao, Jintao Zhang, Huayu Chen, Jianfei Chen, Chen-Hsuan Lin, Ming-Yu Liu, Jun Zhu, Qianli Ma

**Published:** 2026-06-24

**Categories:** cs.CV, cs.LG

**Abstract:**

Autoregressive video diffusion with causal diffusion transformers has emerged as a major paradigm for real-time streaming video generation and action-conditioned interactive world models. In this work, we extend rCM, an advanced diffusion distillation framework, to autoregressive video diffusion. The core philosophy of rCM lies in the complementarity between forward and reverse divergences, represented by consistency models (CMs) and distribution matching distillation (DMD), respectively, in diffusion distillation. This philosophy naturally carries over to the autoregressive setting, where teacher-forcing (TF) provides an offline, forward-divergence causal training paradigm, while self-forcing (SF) corresponds to an on-policy, reverse-divergence refinement.   Our contributions are: (1) through extensive experiments, we show that teacher-forcing CM is currently the best complement to self-forcing DMD as an initialization strategy (2) we present the first implementation of teacher-forcing-based continuous-time CMs (e.g., sCM/MeanFlow) for autoregressive video diffusion, enabled by our custom-mask FlashAttention-2 JVP kernel, achieving 10$\times$ faster convergence compared to discrete-time CMs (dCMs) (3) we introduce Causal-rCM, a leading, unified, and scalable algorithm-infrastructure open recipe for diffusion distillation and causal training (4) we achieve state-of-the-art streaming video generation performance in both frame-wise and chunk-wise settings, using only synthetic data for training.   Notably, our distilled 2-step causal Wan2.1-1.3B model achieves a VBench-T2V score of 84.63 with only 1 or 2 sampling steps. We further apply Causal-rCM to Cosmos 3, an advanced omnimodal world foundation model for physical AI with action-conditioned generation capability, enabling an interactive world model.

**Analysis:**

这是一份关于论文 **Causal-rCM: A Unified Teacher-Forcing and Self-Forcing Open Recipe for Autoregressive Diffusion Distillation** 的深度分析报告。

---

### 1. 摘要翻译
自回归视频扩散与因果扩散Transformer已成为实时流式视频生成和动作条件交互式世界模型的主流范式。本文将先进的扩散蒸馏框架 rCM 扩展至自回归视频扩散。其核心理念在于扩散蒸馏中正向（Forward）与反向（Reverse）散度间的互补性。在自回归设定下，教师强制（Teacher-Forcing, TF）提供离线、正向散度的因果训练范式，而自我强制（Self-Forcing, SF）则对应策略在线、反向散度的优化。本文主要贡献：(1) 实验证明 TF-CM 是 SF-DMD 最优的初始化策略；(2) 提出首个基于 JVP 的连续时间教师强制 CM 实现，收敛速度较离散时间 CM 快 10 倍；(3) 引入 Causal-rCM，一个统一、可扩展的算法-基础设施开源配方；(4) 仅使用合成数据即在流式视频生成中实现 SOTA 性能。Distilled 2-step 模型的 VBench-T2V 分数达到 84.63，并成功应用于 Cosmos 3 世界模型。

### 2. 方法动机分析
*   **驱动力**：解决现有自回归视频扩散在训练-推理间存在的“暴露偏差（Exposure Bias）”问题，同时追求极简、高效的蒸馏流程。
*   **痛点**：TF 训练稳定但存在训练-推理偏差；SF 虽解决了偏差但对初始化非常敏感，极易导致模式崩溃（Mode Collapse）。
*   **核心直觉**：借鉴 rCM 的“正反互补”哲学，TF 提供稳定的离线训练目标（模式覆盖），SF 提供在线分布匹配（模式寻求）。通过将二者串联蒸馏，结合高效的连续时间 CM 初始化，可获得最稳健的推理效果。

### 3. 方法设计详解
**三阶段 Pipeline**：
1.  **Stage 1 (初始化)**：将预训练的双向扩散模型通过 TF 转换为自回归因果模型。
2.  **Stage 2 (TF-CM)**：利用 TF-sCM 将阶段 1 的模型蒸馏为几步因果学生模型，提供结构化的先验初始化。
3.  **Stage 3 (SF-DMD)**：应用自我强制策略进行 DMD 在线微调，直接优化推理时的分布匹配，利用重放（Replayed）反向传播节省内存。

**关键技术细节**：
*   **TF-sCM 实现**：核心在于 JVP（雅可比-向量积）的计算。作者构建了自定义掩码的 FlashAttention-2 JVP Kernel，通过“稀疏矩形列表”方式高效处理因果掩码，实现连续时间切线目标的计算。
*   **Noisy Context**：推理时通过重复使用缓存的 KV 状态作为后续块的上下文，降低推理延迟。

### 4. 方法对比分析
*   **与 APT2/Causal Forcing++ 的本质区别**：Causal-rCM 首次提供了完整的 JVP-based sCM 在因果场景下的实现，相比于离散时间的 dCM，其收敛速度快 10 倍且质量更优；同时通过统一的 Infrastructure 设计（如 SAC × FlexAttention 的兼容性）解决了复杂功能耦合导致的冲突。

### 5. 实验分析
*   **验证方法**：基于 Wan2.1 1.3B/14B 模型，在 480p 分辨率视频生成任务上进行广泛消融实验。
*   **关键结论**：TF-sCM 是 SF-DMD 的最佳初始化方案。在 Frame-wise 设定下，1-step 或 2-step 模型性能超过了 4-step 模型（源于降低了自回归误差累积）。
*   **优势**：极高的采样效率（1-2 步即 SOTA），算法设计模块化且易于迁移。
*   **局限**：Frame-wise 训练仍存在相机漂移问题；SF-DMD 阶段训练稳定性对不同初始化方案有不同反馈。

### 6. 实用指南
*   **开源地址**：[https://github.com/NVlabs/rcm](https://github.com/NVlabs/rcm)
*   **实现要点**：必须使用作者提供的 JVP Kernel 以支持因果掩码下的连续时间导数计算；TF-sCM 阶段的切线预热（Tangent warmup）对稳定性至关重要。
*   **迁移建议**：对于其他 AR 任务（如动作建模、机器人控制），可直接替换其中的模型主干，只需维护因果注意力掩码即可。

### 7. 总结
*   **核心思想**：通过“TF 预训练（覆盖）+ SF 微调（匹配）”的联合配方实现高效蒸馏。
*   **速记版 Pipeline**：
    1.  TF 训练一个稳定的自回归基线。
    2.  利用 JVP 进行连续时间蒸馏，获取高质量切线目标。
    3.  利用 SF-DMD 进行在线策略微调。
    4.  通过 KV 缓存与自定义 Kernel 进行加速推理。

**Key Findings:**

- Our contributions are: (1) through extensive experiments, we show that teacher-forcing CM is currently the best complement to self-forcing DMD as an initialization strategy (2) we present the first implementation of teacher-forcing-based continuous-time CMs (e.g., sCM/MeanFlow) for autoregressive video diffusion, enabled by our custom-mask FlashAttention-2 JVP kernel, achieving 10$\times$ faster convergence compared to discrete-time CMs (dCMs) (3) we introduce Causal-rCM, a leading, unified, and scalable algorithm-infrastructure open recipe for diffusion distillation and causal training (4) we achieve state-of-the-art streaming video generation performance in both frame-wise and chunk-wise settings, using only synthetic data for training.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.25473v1)
- [arXiv](https://arxiv.org/abs/2606.25473v1)

---

<a id='2606.26095v1'></a>
## [Learning Action Priors for Cross-embodiment Robot Manipulation](https://arxiv.org/abs/2606.26095v1)

**Authors:** Dong Jing, Tianqi Zhang, Jiaqi Liu, Jinman Zhao, Zelong Sun, Li Erran Li, Zhiwu Lu, Mingyu Ding

**Published:** 2026-06-24

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Most Vision-Language-Action (VLA) models build on a Vision-Language Model (VLM) backbone by attaching an action module and optimizing the full policy jointly. This design inherits strong visual and linguistic priors from the VLM, but leaves the action module to learn physical motion almost from scratch. As a result, the policy lacks an explicit motion prior, forcing early optimization to simultaneously discover temporal action dynamics and cross-modal alignment, a challenge further amplified in cross-embodiment settings. In this work, we propose to pretrain the action module with motion priors before cross-modal VLA alignment. Specifically, we introduce a two-stage training framework that equips the action module with cross-embodiment temporal motion structure before VLA training begins. In Stage~1, a lightweight flow-matching-based encoder-decoder action module efficiently learns temporal motion structure solely from unconditioned action trajectories, without processing visual or language tokens. In Stage~2, this learned prior is transferred to VLA training through decoder reuse and early-stage latent distillation, aligning visual-language features with the action embedding space while still allowing end-to-end policy refinement. In addition, the trained encoder serves as a compact history compressor, summarizing state-action histories into a single temporal context token for history-aware modeling at negligible cost. Extensive experiments across 13 diverse cross-embodiment tasks on both simulated and real-world platforms validate the effectiveness of our approach. Compared with VLA training without action priors, our model achieves faster convergence, higher success rates, and substantially stronger performance on data-scarce real-world tasks. Moreover, scaling up the action data in Stage~1 yields a more generalizable action prior that directly improves downstream VLA performance.

**Analysis:**

## 1. 摘要翻译
大多数视觉-语言-动作（VLA）模型通过将动作模块附加到视觉-语言模型（VLM）骨干上来实现，这种设计虽然继承了VLM强大的视觉和语言先验，但动作模块却几乎从零开始学习物理运动。因此，策略缺乏明确的运动先验，导致早期训练必须同时发现时间动作动态和跨模态对齐，这在具有异构动作分布的跨具身设置中尤为困难。在本文中，我们提出在跨模态VLA对齐之前，预训练带有运动先验的动作模块。具体而言，我们引入了一个两阶段训练框架。第一阶段，一个轻量级的、基于流匹配的编码器-解码器动作模块仅从未标注的动作轨迹中学习时间运动结构。第二阶段，该先验被转移到VLA训练中，通过解码器复用和早期潜在蒸馏，在对齐视觉-语言特征与动作嵌入空间的同时，允许端到端策略微调。此外，训练好的编码器可作为紧凑的历史压缩器，以极低的成本将状态-动作历史总结为单个时间上下文标记。在13个跨具身任务上的实验表明，该方法实现了更快的收敛、更高的成功率以及在数据匮乏的现实世界任务中显著更强的性能。

## 2. 方法动机分析
- **核心动机**：解决VLA训练中“动作模块物理常识缺失”导致的不平衡问题。
- **现有痛点**：VLM预训练赋予了模型强大的语义理解能力，但动作模块通常随机初始化，迫使VLA在训练初期同时处理“学习动作规律”和“视觉-语言-动作对齐”两个高难度任务，导致梯度不稳定且收敛缓慢。
- **核心假设**：如果动作模块在接触视觉/语言指令前，已经通过大量动作数据学习到了通用的“运动动力学”结构，那么VLA训练过程将更高效且稳定。

## 3. 方法设计详解
### 流程总结
- **Stage 1 (动作先验学习)**：脱离视觉和语言。利用包含异构机器人数据的轨迹集，通过自监督重构任务训练一个Transformer编码器-解码器。输入是交替排列的状态序列和动作序列，目标是重构动作 chunk。
- **Stage 2 (VLA引导训练)**：冻结或微调动作解码器作为VLA的动作头。引入“早期潜在蒸馏”策略，强制VLM输出的特征对齐Stage 1训练好的“运动嵌入空间”。
- **历史压缩**：Stage 1的编码器被复用为历史处理器，将长历史轨迹压缩为一个紧凑的Latent Token送入VLM，增强时间上下文感知。

### 关键细节
- **流匹配(Flow-matching)**：不同于传统的Diffusion模型，采用流匹配作为动作生成目标，能更灵活地处理连续动作分布并降低推理时的计算复杂度。
- **潜在蒸馏损失(L_align)**：引入`||z' - sg(z)||^2`作为监督，利用停止梯度操作防止VLM破坏已学习的运动空间结构，且通过线性衰减权重`λ(k)`在训练后期放开限制。

## 4. 方法对比分析
- **本质区别**：本文采用“先动作，后语义”的解耦式两阶段训练，而非当前主流的端到端联合训练。
- **贡献度**：
    - 提出了“结构化动作先验”的概念，填补了机器人物理运动理解的空白。
    - 设计了高效的历史压缩机制，用一个token解决长序列动作依赖，显著缓解了计算负担。

## 5. 实验分析
- **有效性验证**：在LIBERO、RoboCasa及真实机器人平台（Franka）上进行验证。
- **关键结果**：在数据匮乏的“长尾”真实世界任务上成功率提升巨大（如Grasp Coke从5%提升至35%），整体平均成功率显著超过GR00T和π0.5。
- **优势**：训练收敛速度快（Stage 1成本极低），且在小样本任务中鲁棒性极强。
- **局限**：在极端复杂的长序列任务中，对历史窗口的长度仍有一定的依赖。

## 6. 实用指南
- **实现建议**：Stage 1的训练只需数小时，建议先在各类机器人数据集上跑通该环节。蒸馏阶段的`N_decay`参数需根据任务复杂度和总训练步数进行微调（本文设为5000）。
- **迁移性**：该方法本质上与骨干网络无关，完全可以移植到任意主流VLA架构（如OpenVLA等）中作为插件使用。

## 7. 总结
- **核心思想**：通过解耦运动学习与语义映射，为动作模块注入物理先验。
- **速记版pipeline**：
    1. 使用无监督动作数据预训练动作编码器-解码器。
    2. 将训练好的解码器作为VLA动作头。
    3. 利用编码器产生的结构化空间蒸馏VLM特征。
    4. 将历史动作序列压缩为token注入模型。

**Key Findings:**

- In this work, we propose to pretrain the action module with motion priors before cross-modal VLA alignment.
- Specifically, we introduce a two-stage training framework that equips the action module with cross-embodiment temporal motion structure before VLA training begins.
- Extensive experiments across 13 diverse cross-embodiment tasks on both simulated and real-world platforms validate the effectiveness of our approach.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26095v1)
- [arXiv](https://arxiv.org/abs/2606.26095v1)

---

<a id='2606.26093v1'></a>
## [ForceBand: Learning Forceful Manipulation with sEMG](https://arxiv.org/abs/2606.26093v1)

**Authors:** Botao He, Zhi Wang, Linna Kuang, Ishaan Ghosh, Jitendra Malik, Cornelia Fermuller, Tingfan Wu, Jiayuan Mao, Ruoshi Liu, Haozhi Qi, Yiannis Aloimonos

**Published:** 2026-06-24

**Categories:** cs.RO

**Abstract:**

Human demonstrations are a scalable data source for learning robot manipulation policies. However, common sources of human demonstration data, such as motion-capture trajectories and internet videos, capture mostly motion and appearance while missing the contact forces that are critical for force-sensitive manipulation. In this paper, we introduce ForceBand, a low-cost wrist-worn sEMG system that turns human muscle activity into force-enriched demonstrations. We first collect a 10-hour multimodal dataset containing egocentric video, sEMG, IMU, and fingertip force measurements across diverse actions and objects. Using this dataset, we pre-train an EMG2Force model that predicts per-finger forces from sEMG and IMU signals. After a short user-specific calibration, users can collect target-task demonstrations using only ForceBand and video; EMG2Force then labels these demonstrations with per-finger force traces, producing force-augmented demonstrations for robot policy learning. Experiments show that ForceBand recovers fine-grained fingertip interactions with over 50% lower force prediction error than vision-based baselines and achieves an 87% success rate on pick, squeeze, and place tasks that require object-specific force control across objects with diverse shapes, sizes, and weights. Project website: https://forceband-emg.github.io

**Analysis:**

以下是针对《ForceBand: Learning Forceful Manipulation with sEMG》的深度分析报告：

### 1. 摘要翻译
人类演示是学习机器人操作策略的可扩展数据源。然而，常见的演示数据源（如动作捕捉轨迹和互联网视频）主要捕捉运动和外观，却缺失了力敏感操作至关重要的接触力信息。本文介绍了ForceBand，这是一种低成本、手腕佩戴的sEMG（表面肌电）系统，可将人体肌肉活动转化为力增强演示。我们首先收集了一个10小时的多模态数据集，包含涵盖各种动作和物体的自我中心视频、sEMG、IMU和指尖力测量。利用该数据集，我们预训练了一个EMG2Force模型，从sEMG和IMU信号预测各手指的力。经过简短的用户特定校准后，用户仅需使用ForceBand和视频即可收集目标任务演示；EMG2Force随后用指尖力轨迹标记这些演示，生成用于机器人策略学习的力增强演示。实验表明，ForceBand恢复了细粒度的指尖交互，力预测误差比基于视觉的基线降低了50%以上，并在需要物体特定力控制的抓取、挤压和放置任务中达到了87%的成功率。

### 2. 方法动机分析
*   **驱动力**：机器人操纵不仅需要运动轨迹（在哪儿动），还需要精确的力控制（多大劲），以应对不同材质、重量和形状的物体。
*   **痛点**：纯视觉方法无法从外观中明确推断接触力；而直接的触觉手套（Tactile Gloves）会遮挡手指，干扰真实交互体验且成本高昂。
*   **核心直觉**：人体肌肉电信号（sEMG）与手指施加的力之间存在内在的生物力学联系，通过学习这种映射，可以在不干扰手指触感的情况下，从腕部远程“感知”交互力。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **数据采集与预处理**：佩戴包含sEMG与IMU的定制腕带，通过简短校准采集肌肉信号与指尖真实力数据。
    2.  **EMG2Force建模**：采用Transformer架构，输入包含时域信号与通过STFT生成的频域谱图（Spectrogram），输出为各手指的力轨迹。
    3.  **力增强演示生成**：利用训练好的模型，仅通过视频和腕带数据自动推断演示中的力轨迹。
    4.  **策略学习**：使用流匹配（Flow Matching）策略，将视频中的运动轨迹与推断出的力轨迹同步，训练能够输出动作和力协同控制的机器人策略。
*   **关键技术**：
    *   **肌肉感知布控（Muscle-aware placement）**：非均匀分布电极，重点覆盖控制拇指、食指、中指的特定前臂肌肉。
    *   **双域融合**：结合时域原始信号（捕捉动态反应）与频域谱图（捕捉肌肉激活模式），显著提升力预测精度。
    *   **强制力辅助动作预测**：在策略的动作空间中显式引入力目标 $f$，使机器人学会在抓取时动态调节力，而非仅仅执行位置命令。

### 4. 方法对比与创新
*   **本质区别**：与现有视觉估计方法（由外观推断力）不同，ForceBand通过生物信号（肌肉激活）直接解算力，解决了遮挡导致的力估计失效问题。
*   **创新点**：提出了“肌肉感知的电极布局”和“时频双域融合的力预测模型”，并将力作为策略学习的显式控制维度。

### 5. 实验简析
*   **结论**：在手部接触力回归任务中，ForceBand的误差比基于视觉的基线（如FEEL）降低了50%以上。在需要特定挤压力度的任务（如处理易碎品或重物）中，其表现远超仅依赖视觉和位置控制的基线。
*   **优势**：低成本、非侵入式、泛化性较好。
*   **局限**：目前的校准仍需依赖指尖力传感器，且对于极度复杂的环境视觉变化存在一定的力预测漂移。

### 6. 实用指南
*   **开源情况**：项目主页为 https://forceband-emg.github.io/。
*   **实现建议**：电极放置对精度影响极大（8通道配置比均匀分布提升18%的性能），实现时需严格对应目标肌肉。
*   **迁移路径**：该框架可直接迁移至任何需要“力控”的机器人手部操作任务，仅需更换针对目标任务的数据集进行微调（Fine-tuning）。

### 7. 总结
*   **核心思想**：通过腕部肌电信号远程解算手指交互力，实现力感知的机器人动作策略学习。
*   **速记版Pipeline**：
    1. 穿戴肌电腕带采集动作数据；
    2. 预训练模型将肌电转为手指力信号；
    3. 力信号与视频对齐形成强化演示；
    4. 机器人模型联合训练动作与力控制。

**Key Findings:**

- In this paper, we introduce ForceBand, a low-cost wrist-worn sEMG system that turns human muscle activity into force-enriched demonstrations.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26093v1)
- [arXiv](https://arxiv.org/abs/2606.26093v1)

---

<a id='2606.26047v1'></a>
## [Learning Robot Visual Navigation in Crowds via Intention-Aware Scene Representations](https://arxiv.org/abs/2606.26047v1)

**Authors:** Han Bao, Bingyi Xia, Hanjing Ye, Yu Zhan, Hao Cheng, Baozhi Jia, Wenjun Xu, Jiankun Wang

**Published:** 2026-06-24

**Categories:** cs.RO

**Abstract:**

Robot crowd navigation requires the ability to infer human intentions while accounting for the structural constraints of the environment. Currently, deep reinforcement learning (DRL) provides a promising method for learning navigation policies that understand human intentions. However, most of them rely on limited scene representations, treating pedestrians as simple 2D points and ignoring rich visual cues from both humans and the environment. To address this issue, we introduce iCrowdNav, a novel visual crowd navigation method with intention-aware scene representations, to encode behavioral and structural context from egocentric visual observations. Our method employs two key components: a spatio-temporal encoder for extracting occupancy features of the scene, and Intent-Interact Former (I$^2$ Former), an attention-based module that encodes human poses to infer pedestrians' motion intentions. These features are integrated into a compact state embedding that supports effective DRL policy training. Extensive experiments show that our method achieves superior performance over baselines, and real-world deployment demonstrates vision-based crowd navigation.

**Analysis:**

### 1. 摘要翻译
机器人人群导航需要推理人类意图并考虑环境结构约束。目前，深度强化学习（DRL）为学习理解人类意图的导航策略提供了有前景的方法。然而，现有方法大多依赖有限的场景表示，将行人简单地视为2D点，忽略了人类和环境丰富的视觉线索。为此，本文提出了iCrowdNav，一种采用意图感知场景表示的新型视觉导航方法，旨在从自我中心视觉观察中编码行为和结构上下文。我们的方法包含两个关键组件：用于提取场景占用特征的时空编码器，以及用于编码人体姿态以推理行人运动意图的意图-交互Transformer（I2Former）。这些特征被集成到支持高效DRL策略训练的紧凑状态嵌入中。大量实验表明，我们的方法优于基线模型，并在真实世界部署中展现了卓越的视觉导航能力。

---

### 2. 方法动机分析
*   **驱动力**：旨在填补“低维状态表示”与“复杂真实环境”之间的鸿沟，利用视觉感知中丰富的语义信息（如人体姿态、语义布局）来实现更具社交感知的导航。
*   **现有痛点**：现有方法将行人简化为2D位置/速度点，使用二进制占用图或激光雷达数据，完全丢失了人体动作（如转头、目光朝向）带来的意图预测线索。
*   **核心直觉**：人类的动作意图（如即将变向）隐藏在复杂的视觉特征中，通过融合“环境几何布局（BEV）”与“行人姿态（3D Pose）”，能够实现更具前瞻性的预测和避障。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **特征提取**：输入多帧RGB-D数据和机器人内部状态。
        *   **时空编码器**：采用Fiery架构，利用Lift-Splat技术将多视角特征提升至3D，并与历史帧BEV特征对齐，提取环境占用特征。
        *   **I2Former**：使用YOLO检测2D姿态，提升至3D坐标，通过IntentFormer模块利用自注意力机制建模人体关节关系，提取行为意图。
    2.  **特征融合**：将BEV占用特征、意图特征及机器人状态拼接，通过MLP融合为高维DRL状态表示。
    3.  **决策输出**：基于PPO（Proximal Policy Optimization）算法输出机器人的线/角速度。
*   **模型结构**：
    *   **IntentFormer**：处理行人个体意图（Self-Attention）。
    *   **InteractFormer**：处理机器人与行人间的交互（Cross-Attention，以机器人状态为Query）。

---

### 4. 方法对比分析
*   **本质区别**：从传统的“点位追踪”转变为“视觉感知+行为特征推理”。不仅“看”到了障碍物，还“推断”出了障碍物的潜在运动意图。
*   **创新贡献**：引入意图感知场景表示，将3D人体姿态和视觉BEV有效解耦又融合，解决了端到端黑盒导航训练中维度灾难与信息缺失的矛盾。
*   **适用场景**：人流密集、走廊狭窄、存在复杂动态障碍物（如商场、车站）的室内外场景。

---

### 5. 实验分析
*   **验证方法**：在Isaac Sim中搭建SocNav-Gym，设置不同密度和宽度环境，对比DRL-VO、SARL*-OM、ViNT等基线。
*   **关键结果**：在SR（成功率）上显著提升，且在TPZ（私人空间侵入时间）上表现最低，验证了其安全性。
*   **优势**：在 constrained（受限）空间中表现尤其出色，策略更具前瞻性，减少了对行人的干扰。
*   **局限**：在极端拥挤场景（超高密度）下，严重遮挡可能导致姿态提取不稳定，从而削弱意图推断能力。

---

### 6. 实用指南
*   **开源情况**：已开源，代码与附录见：[https://broln7.github.io/socialbev.io/](https://broln7.github.io/socialbev.io/)。
*   **实现细节**：需预训练RGB Backbone和时空编码器（建议利用nuScenes数据集），PPO训练阶段保持这些模块固定（Frozen），仅训练融合层及策略网络，以提升稳定性。
*   **迁移可能**：该框架的意图推理模块（I2Former）可独立迁移至行人轨迹预测任务中。

---

### 7. 总结
*   **核心思想**：利用多模态视觉感知整合环境结构与人类姿态意图。
*   **速记版pipeline**：
    1.  **看环境**：通过多视角相机把影像转为鸟瞰图，摸清障碍物方位。
    2.  **看姿态**：通过AI检测行人肢体动作，判断他们的移动方向。
    3.  **做决策**：融合两类信息，利用DRL模型给出既不撞人又能顺畅通行的指令。

**Key Findings:**

- To address this issue, we introduce iCrowdNav, a novel visual crowd navigation method with intention-aware scene representations, to encode behavioral and structural context from egocentric visual observations.
- Our method employs two key components: a spatio-temporal encoder for extracting occupancy features of the scene, and Intent-Interact Former (I$^2$ Former), an attention-based module that encodes human poses to infer pedestrians' motion intentions.
- Extensive experiments show that our method achieves superior performance over baselines, and real-world deployment demonstrates vision-based crowd navigation.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26047v1)
- [arXiv](https://arxiv.org/abs/2606.26047v1)

---

<a id='2606.26046v1'></a>
## [RoboAtlas: Contextual Active SLAM](https://arxiv.org/abs/2606.26046v1)

**Authors:** Alexander Schperberg, Shivam K. Panda, Abraham P. Vinod, M. K. Jawed, Stefano Di Cairano

**Published:** 2026-06-24

**Categories:** cs.RO, cs.CV

**Abstract:**

We present RoboAtlas, a contextual Active SLAM framework that adaptively balances geometric exploration and semantic reasoning using a scalable 3D semantic mapping system, OpenRoboVox. RoboAtlas integrates frontier exploration, global semantic-map reasoning, and egocentric VLM-based reasoning through a contextual multi-armed bandit that transitions from exploration to semantically guided navigation as scene understanding improves. We evaluate the system in simulation and on a Unitree Go2 robot in large-scale real-world environments exceeding 1800 m2 with approx. 30k mapped semantic instances, achieving a 100% task success rate. On the GOAT-Bench "Val Unseen" benchmark, RoboAtlas achieves state-of-the-art performance with highest reported success rate (SR) of 90.6%, using GPT-4o, improving over the strongest prior baseline by 17.8 percentage points in SR. Using the much smaller Qwen2.5-VL-7B model, it still achieves 88.8% SR, outperforming all baselines using GPT-4o in SR, and revealing the importance of the information gained by our semantic mapping framework over simply replacing the underlying foundation model. The results demonstrate that grounding foundation models with large-scale 3D semantic maps enables robust and efficient contextual Active SLAM.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **RoboAtlas** 的论文分析如下：

### 1. 主要贡献总结
RoboAtlas 提出了一种创新的“情境感知主动SLAM”（Contextual Active SLAM）框架，通过集成名为 OpenRoboVox 的 3D 语义地图系统，实现了几何探索与语义推理的动态平衡。该系统通过多臂老虎机（Multi-Armed Bandit）机制智能调度探索与导航策略，在复杂的大规模环境中展现了极高的任务成功率，并成功证明了精细化 3D 语义映射对提升基础模型（VLM）决策效能的关键作用。

### 2. 核心创新与方法论
*   **语义与几何的协同机制**：该研究不再将 SLAM 仅视为几何建图，而是将其转化为语义决策过程。系统通过“情境感知”平衡了早期的盲目探索与后期的目标导向导航。
*   **OpenRoboVox 系统**：这是一个可扩展的 3D 语义映射框架，为 VLM（视觉语言模型）提供了高质量、结构化的场景上下文，从而解决了 VLM 在缺乏空间意识时的盲区问题。
*   **自适应推理策略**：引入多臂老虎机框架来动态调节探索模式，使机器人能够根据环境理解程度（Semantic Mapping maturity）自动平滑过渡决策逻辑。
*   **模型与数据的解耦**：实验表明，使用较小的模型（Qwen2.5-VL-7B）配合高质量的 3D 语义上下文，能够超越直接使用更大模型（GPT-4o）但缺乏有效空间映射的基准，强调了“数据结构化”在具身智能中的重要性。

### 3. 对领域的潜在影响
*   **具身智能的新范式**：该论文证明了“语义地图 + VLM”是实现复杂环境下自主任务的最佳实践。它为“从大模型到强导航”的跨越提供了一条明确的技术路径。
*   **算力与效率的平衡**：研究揭示了即使在参数量较小的 VLM 场景下，通过构建高质量的 3D 语义地图，也能达成极高的任务成功率，这对未来低功耗机器人平台（如边缘端部署）具有极高的指导价值。
*   **基准测试的影响**：在 GOAT-Bench 上的显著提升（17.8% 的增长）为该任务设定了新的技术标杆，可能推动学术界在空间语义建模领域投入更多关注。

### 4. 受益的相关领域与应用
*   **服务机器人（Service Robotics）**：如养老护理、办公场景下的室内自主配送，这些场景对语义理解和长距离导航要求极高。
*   **灾后搜索与救援**：在未知、大规模且复杂的地形中，该系统能快速建立语义认知并寻找特定目标。
*   **工业自动化/数字孪生**：OpenRoboVox 的语义映射能力可直接应用于自动化工厂的环境数字化与巡检。
*   **人机协作**：能够理解“找那个蓝色的杯子”这类复杂指令的机器人，本质上得益于该文提出的语义地图架构。

### 5. 可推断的潜在限制
*   **计算资源需求**：尽管论文强调了语义映射的效能，但在 1800 平方米、3 万个语义实例的环境下，实时维护全局 3D 语义地图的计算开销（内存和算力）在移动端设备上可能仍存在瓶颈。
*   **推理延迟**：将 3D 地图输入 VLM 涉及到复杂的编码和查询过程，在高动态场景下，系统对实时性（Latency）的响应能力可能受到网络延迟或 GPU 算力的制约。
*   **泛化性挑战**：尽管在室内大规模环境表现优异，但未提及在极端室外环境、弱纹理环境或极度混乱动态环境下的鲁棒性，这些是传统 SLAM 依然面临的挑战。
*   **对语义分割的依赖**：语义地图的准确性严重依赖于底层语义分割模型的精度，如果 OpenRoboVox 在长尾分类或遮挡严重的情况下出现语义识别错误，将直接影响上层的导航逻辑。

**专家点评**：这篇论文的趣味性在于它清晰地回答了“我们是否真的需要参数量无限大的模型”这一问题。**RoboAtlas 证明了高质量的 3D 环境表征是连接视觉与行为的缺失环节**，这对于计算机视觉从单纯的“视觉识别”走向“空间行为决策”具有里程碑式的启示。

**Key Findings:**

- We present RoboAtlas, a contextual Active SLAM framework that adaptively balances geometric exploration and semantic reasoning using a scalable 3D semantic mapping system, OpenRoboVox.
- On the GOAT-Bench "Val Unseen" benchmark, RoboAtlas achieves state-of-the-art performance with highest reported success rate (SR) of 90.6%, using GPT-4o, improving over the strongest prior baseline by 17.8 percentage points in SR.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26046v1)
- [arXiv](https://arxiv.org/abs/2606.26046v1)

---

<a id='2606.26025v1'></a>
## [In-Context World Modeling for Robotic Control](https://arxiv.org/abs/2606.26025v1)

**Authors:** Siyin Wang, Junhao Shi, Senyu Fei, Zhaoyang Fu, Li Ji, Jingjing Gong, Xipeng Qiu

**Published:** 2026-06-24

**Categories:** cs.RO, cs.CV

**Abstract:**

Modern Vision-Language-Action (VLA) models often fail to generalize to novel setups, such as altered camera viewpoints or robot morphologies, because they are typically conditioned only on current observations and language instructions. By ignoring the underlying system configuration as a variable, these models implicitly assume a fixed execution context encountered during training, necessitating data-intensive fine-tuning for any new environment. In this work, we introduce In-Context World Modeling (ICWM), a framework that treats system identification as an in-context adaptation problem. ICWM enables robot policies to autonomously infer essential system variables from a short history of self-generated, task-agnostic interactions. Unlike traditional In-Context Learning that uses demonstrations to specify what task to perform, ICWM leverages the context window to understand how the system operates. By processing these interactions before task execution, the model implicitly captures the world dynamics of the current system, enabling adaptation to novel configurations without parameter updates. Extensive experiments in simulation and on real-world robot platforms demonstrate that ICWM significantly outperforms standard VLA baselines on novel camera viewpoints.

**Analysis:**

# 论文分析：In-Context World Modeling (ICWM) for Robotic Control

### 1. 摘要翻译
现代视觉-语言-动作（VLA）模型由于通常仅以当前观测和语言指令作为条件，导致其无法泛化到新颖配置（如变化的摄像机视角或机器人形态）。通过忽略潜在的系统配置作为变量，这些模型隐式地假设了训练期间固定的执行环境，从而导致在新环境中必须进行数据密集型的微调。本文提出了“上下文内世界建模”（ICWM），一种将系统辨识视为上下文内适应问题的框架。ICWM使机器人策略能够从一段简短的、自我生成的、任务无关的交互历史中自主推断出关键的系统变量。与使用演示来指定执行任务的传统上下文内学习（ICL）不同，ICWM利用上下文窗口来理解系统是如何运作的。通过在任务执行前处理这些交互，模型隐式地捕获了当前系统的世界动态，从而无需参数更新即可适应新颖配置。在仿真和真实世界机器人平台上的大量实验表明，ICWM在处理新视角时显著优于标准的VLA基线。

### 2. 方法动机分析
*   **驱动力**：解决VLA模型在部署时因环境（视角、形态）变化导致的“泛化崩溃”问题，将其从静态的映射函数转变为能够动态推断物理规律的自适应系统。
*   **现有方法痛点**：当前VLA模型将系统配置（$\psi$）视为训练集中的“隐式常数”而未被显式建模。当测试环境发生变化（如视角偏移）时，模型缺乏补偿空间畸变或动力学差异的机制，导致动作预测偏移。
*   **研究假设**：机器人通过短暂的、随机的探索行为（Self-Probing），可以收集足够的环境动力学信息。通过将这些“交互片段”作为上下文前缀（Context Prefix）输入给Transformer，模型能够利用Attention机制在任务执行前实现隐式的“系统辨识”。

### 3. 方法设计详解
*   **流程总结**：
    1.  **主动探索阶段（Active Probing）**：在执行任务前，机器人不执行任务动作，而是随机采样几个目标姿态并执行，记录期间的“初始图像-动作-结束图像”片段。
    2.  **交互前缀构建**：将记录的$N$个探索片段组合为上下文输入$T$。
    3.  **隐式配置推理**：将$T$作为Prompt前缀输入给预训练VLM。模型通过Attention机制处理$T$，构建出包含当前环境动力学信息的隐藏状态（即$\Psi(T)$）。
    4.  **任务执行**：在得到$\Psi(T)$的基础上，输入当前观测$o_t$和任务指令$l$，模型生成修正后的动作$a_t$。
*   **算法解释**：核心公式为 $a_t \sim \pi_\theta(a_t | \Psi(T), o_t, l)$。$\Psi$并未增加额外参数，而是复用了VLA主干的参数。通过这种方式，交互上下文起到了“系统自校准”的作用。

### 4. 方法对比分析
*   **本质区别**：现有的ICL侧重于“行为规范”（Behavior Specification，告诉模型做什么），而ICWM侧重于“系统辨识”（System Identification，让模型学会如何运作）。
*   **创新贡献**：提出了一种无需额外标注、无需任务演示的零样本（Zero-shot）环境适应范式。证明了随机运动足以提取出系统动力学特征，实现了“从探索中学习物理规律”。
*   **适用场景**：适用于机器人摄像机位置变动、夹具更换、甚至场景物体材质变换等需要快速适应的场景。

### 5. 实验分析（精简版）
*   **关键结论**：在仿真和实机中，ICWM对OOD（分布外）视角的适应性远超标准MV（多视角）训练，成功率提升显著。
*   **主要优势**：即插即用，无需梯度更新；支持长程任务，缓解了误差积累。
*   **主要局限**：对“探索阶段”的动作空间有一定要求，若探索未能覆盖足够的动力学空间，适应效果会打折扣。

### 6. 实用指南
*   **开源情况**：已发布。实现该框架的关键在于构建多样化的自我探索数据，并在预训练中加入交互片段作为上下文。
*   **实现细节**：建议采样5个片段（$N=5$），采用AdamW优化器进行训练。推理时可利用KV Cache存储$T$产生的隐藏状态，避免计算冗余。
*   **迁移可能**：该方法天然适用于任何基于Transformer的序列化机器人策略，可轻松移植到其他闭环控制系统中。

### 7. 总结
*   **核心思想**：利用随机交互作为上下文，让模型实时“感知”并适应物理环境差异。
*   **速记版pipeline**：
    1. 随机探索生成交互片段
    2. 将片段作为上下文输入
    3. 推断当前系统配置
    4. 执行校准后的任务动作

**Key Findings:**

- Modern Vision-Language-Action (VLA) models often fail to generalize to novel setups, such as altered camera viewpoints or robot morphologies, because they are typically conditioned only on current observations and language instructions.
- By ignoring the underlying system configuration as a variable, these models implicitly assume a fixed execution context encountered during training, necessitating data-intensive fine-tuning for any new environment.
- In this work, we introduce In-Context World Modeling (ICWM), a framework that treats system identification as an in-context adaptation problem.
- By processing these interactions before task execution, the model implicitly captures the world dynamics of the current system, enabling adaptation to novel configurations without parameter updates.
- Extensive experiments in simulation and on real-world robot platforms demonstrate that ICWM significantly outperforms standard VLA baselines on novel camera viewpoints.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26025v1)
- [arXiv](https://arxiv.org/abs/2606.26025v1)

---

<a id='2606.26017v1'></a>
## [G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance](https://arxiv.org/abs/2606.26017v1)

**Authors:** Hang Yu, Ye Jin, Alessandro Canevaro, Julian Schmidt, Julian Jordan, Peizheng Li, Marc Kaufeld, Silvan Lindner, Johannes Betz, Wilhelm Stork

**Published:** 2026-06-24

**Categories:** cs.RO

**Abstract:**

In autonomous driving, diffusion-based planners have emerged as a promising paradigm for robust motion planning in dense and interactive traffic, as they can effectively model diverse driving behaviors. However, their inherent stochasticity often requires explicit guidance during denoising to ensure safety and route adherence for robust closed-loop execution. Existing guidance typically relies on sparse, entity-centric geometric queries or post-hoc refinement, yielding limited situational awareness and fragile performance in interactive scenes. To address this issue, we propose G2DP (Grid-Guided Diffusion Planning), a diffusion-based planner that directly enforces dense environmental constraints through inference-time guidance. Specifically, G2DP constructs a differentiable spatio-temporal cost volume by fusing probabilistic future occupancy distributions with a route-progress map. By formulating this volume as a continuous safety energy functional, it injects dense gradients directly into the denoising loop, actively steering trajectory generation toward collision-free and progress-optimal regions. Extensive closed-loop evaluations show that G2DP achieves state-of-the-art performance on nuPlan, outperforming the strongest imitation-learning baseline by +7.2 points in reactive score. It further maintains top scores in zero-shot transfers to interPlan and DeepScenario benchmarks, with collision avoidance improving by +10.15 over the unguided approach on interPlan. These results demonstrate that spatio-temporal cost grids serve as an effective representation for robust guidance in diffusion-based planning.

**Analysis:**

以下是对论文《G2DP: Diffusion Planning with Spatio-Temporal Grid Guidance》的深度分析：

### 1. 摘要翻译
在自动驾驶中，基于扩散的规划器因其能对多模态驾驶行为进行建模而备受关注。然而，扩散过程固有的随机性需要显式引导，以确保闭环执行中的安全性和路径依从性。现有方法多依赖稀疏的几何查询或后处理，导致环境感知局限及交互场景下表现脆弱。为此，我们提出了G2DP（网格引导扩散规划），通过推理时引导，直接将密集的环境约束强制作用于生成过程。G2DP构建了一个可微的时空代价体（融合了概率性未来占据分布与路径进度图），并将其作为连续安全能量函数，通过梯度注入的方式，将轨迹实时引导至无碰撞且符合路径规划的最优区域。在nuPlan等基准测试中，G2DP表现出色，显著提升了交互场景下的安全与进度指标。

### 2. 方法动机分析
*   **驱动力**：解决基于扩散的生成式规划在复杂交互场景下“缺乏显式约束”导致的安全性与合理性问题。
*   **痛点**：现有工作多采用稀疏几何约束（如中心点距离），无法捕捉未来不确定性，且难以实现真正的“主动式”路径 steering（转向）。
*   **研究假设**：通过将鸟瞰图（BEV）下的密集占用概率和路径进度转化为可微的能量场，直接作用于扩散去噪过程，能实现更细腻、更主动的安全避障。

### 3. 方法设计详解
*   **核心Pipeline**：
    1.  **场景编码**：利用DiT（Diffusion Transformer）处理场景上下文（邻居车辆、地图）。
    2.  **密集代价体构建**：使用U-Net预测未来时空的占用概率图（Occupancy Grid），融合预设的路径进度引导图。
    3.  **能量函数定义**：基于Top-K聚合策略计算累积 trajectory cost，既避免了单点采样的噪声敏感，又保留了空间感知。
    4.  **梯度注入引导**：在去噪的最后阶段（Step 8-9），计算代价体对当前轨迹位置的梯度，通过“能量引导（Energy-based guidance）”修正去噪方向。
*   **关键算法**：公式 $\nabla_{x_t} \log p^*_t = \nabla_{x_t} \log p_t - \lambda_t \nabla_{x_t} E(x_t; C)$。这意味着在模型原本的学习分布中减去一个“能量梯度”，迫使轨迹偏离高代价（高风险）区域。

### 4. 方法对比分析
*   **本质区别**：与依赖“几何原语（点、线）”的稀疏引导不同，G2DP利用的是“密集概率场（BEV Grid）”，能更全面地覆盖车辆外廓的碰撞风险。
*   **创新贡献**：提出了一种可微的时空代价体构建方式，并证明了无需重训 backbone，仅靠推理时梯度引导即可大幅提升安全性能。
*   **适用场景**：高密度、高交互的城市道路场景，特别是在需要避让复杂动态障碍物（如行人、自行车）时表现最优。

### 5. 实验分析
*   **关键结论**：在nuPlan的Test14-hard（高难度）基准上，G2DP较最强基线提升明显；在interPlan零样本迁移实验中，碰撞避免指标提升达+10.15。
*   **主要优势**：提供了主动式避障能力，且生成的轨迹比后处理平滑得多。
*   **主要局限**：对计算资源有一定要求（需在去噪循环中计算梯度）；Footprint（足迹）近似处理较为简化。

### 6. 实用指南
*   **开源情况**：已开源，见 https://github.com/HangYuu/G2DP。
*   **关键点**：
    *   **超参数调节**：$\gamma=0.95$（占据 vs 进度）及 $\lambda_t=0.5$ 是关键。
    *   **窗口选择**：仅在去噪后期注入梯度（第8-9步），过早注入会干扰模型意图。
*   **迁移建议**：可直接迁移至任何基于Transformer的去噪规划框架中，只需确保你的感知模块能输出BEV网格图。

### 7. 总结
*   **核心思想**：通过时空代价体的梯度注入，引导扩散模型生成安全轨迹。
*   **速记版Pipeline**：
    1. 预测环境未来的碰撞风险网格；
    2. 定义一条理想的路径进度线；
    3. 将风险网格与进度线融合成一张“代价图”；
    4. 在去噪过程中，根据“代价图”的梯度修正车辆行驶方向，避开风险点。

**Key Findings:**

- To address this issue, we propose G2DP (Grid-Guided Diffusion Planning), a diffusion-based planner that directly enforces dense environmental constraints through inference-time guidance.
- Extensive closed-loop evaluations show that G2DP achieves state-of-the-art performance on nuPlan, outperforming the strongest imitation-learning baseline by +7.2 points in reactive score.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26017v1)
- [arXiv](https://arxiv.org/abs/2606.26017v1)

---

<a id='2606.26010v1'></a>
## [FAR-LIO: Enabling High-Speed Autonomy through Fast, Accurate, and Robust LiDAR-Inertial Odometry](https://arxiv.org/abs/2606.26010v1)

**Authors:** Maximilian Leitenstern, Marcel Weinmann, Patrick Haft, Tobias Lasser, Dominik Kulmer, Markus Lienkamp

**Published:** 2026-06-24

**Categories:** cs.RO

**Abstract:**

Robust and accurate odometry estimation is essential in modern robotics. In environments characterized by highly dynamic motion and sensor noise, odometry estimation becomes increasingly challenging. Autonomous racing combines both factors in an unstructured setting, where minimizing odometry latency is essential for stable closed-loop control. This paper introduces FAR-LIO, a highly optimized CUDA-accelerated LiDAR-inertial odometry framework developed for Fast, Accurate, and Robust performance. Our system leverages a novel CUDA-based voxel hashmap to enable parallelized nearest-neighbor search and efficient map updates. We employ a sparsity-aware Generalized Iterative Closest Point algorithm with adaptive thresholding on top of the CUDA-based voxel hashmap with adaptive density to achieve low-latency without compromising accuracy. An Extended Kalman Filter serves as a robust backend. It utilizes an upsampling and delay compensation strategy to fuse the LiDAR odometry with high-frequency IMU data, thereby ensuring a robust and smooth odometry output. We evaluate FAR-LIO across four different sensor setups, using both public datasets and data from two autonomous racecars driving at speeds of up to 250 km/h. FAR-LIO achieves an average 6.9% reduction in the positional error and 38.4% lower runtime compared to state-of-the-art baselines on target hardware using a single parameter set. This demonstrates its computational efficiency and broad applicability. To build upon our work, our code is available open-source on https://github.com/TUMFTM/FAR-LIO.

**Analysis:**

### 1. 摘要翻译
稳健且精确的里程计估计对于现代机器人技术至关重要。在具有高度动态运动和传感器噪声的环境中，里程计估计极具挑战性。自主赛车结合了上述两种因素，且在非结构化场景下，最小化里程计延迟对于稳定的闭环控制至关重要。本文介绍了 FAR-LIO，这是一种高度优化的 CUDA 加速激光雷达-惯性里程计框架，旨在实现快速、精确和稳健的性能。我们的系统利用一种新型的基于 CUDA 的体素哈希图，实现了并行化的近邻搜索和高效的地图更新。我们在该哈希图之上采用了一种稀疏感知（sparsity-aware）的广义迭代最近点（GICP）算法，并结合自适应密度和自适应阈值策略，在不牺牲精度的情况下实现了低延迟。扩展卡尔曼滤波（EKF）作为稳健后端，利用上采样和延迟补偿策略将激光雷达里程计与高频 IMU 数据融合，确保了输出的鲁棒性和平滑性。我们在四种不同的传感器设置下评估了 FAR-LIO，包括公共数据集和两辆自主赛车（时速高达 250 公里/小时）的实测数据。在单一参数集下，FAR-LIO 与当前最优基线相比，平均位置误差降低了 6.9%，运行时间缩短了 38.4%。这证明了其计算效率和广泛的适用性。

---

### 2. 方法动机分析
*   **驱动力**：在高动态、高速（250km/h）自主竞速场景下，现有算法往往因为计算延迟过高或对参数敏感而无法维持闭环控制。
*   **痛点**：现有 CPU 密集型 LIO 框架难以满足实时性要求；部分 GPU 加速方法未充分利用硬件并行潜力，或在处理极端动力学时鲁棒性不足。
*   **研究假设**：通过深度定制的 CUDA 并行数据结构（cuVoxelMap）和稀疏感知 GICP 算法，可以实现极端速度下的毫秒级里程计估计，同时利用 EKF 的延迟补偿机制消除非确定性时延。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **预处理与去畸变**：利用 EKF 历史运动信息，通过线性回归建模 LiDAR 运动，实现点云并行去畸变。
    2.  **cuVoxelMap 构建**：基于 `cuco::static_map` 实现 CUDA 优化的体素哈希图，处理点云的空间存储与快速查找。
    3.  **稀疏感知 GICP (SA-GICP)**：在局部子图内执行 kNN 搜索，通过自适应密度和健壮的 Cauchy 核函数计算变换矩阵。针对点云稀疏区域，退化为点对点匹配以维持鲁棒性。
    4.  **EKF 后端融合**：融合 IMU 与激光雷达测量值。核心创新在于引入“延迟补偿”机制，通过回溯 EKF 状态历史，将历史观测值重新对齐到当前时刻。
*   **关键公式**：论文引入的鲁棒核函数 $\rho(e)$ 动态调整权重 $\kappa$，显著提升了在非结构化复杂地形下的匹配收敛性。

---

### 4. 方法对比分析
*   **本质区别**：FAR-LIO 并非单纯在现有算法上添加 GPU 加速，而是重构了从底层数据结构（体素哈希）到前端注册（稀疏感知 GICP）再到后端融合（延迟补偿 EKF）的全流水线。
*   **创新贡献**：提出 **cuVoxelMap** 与 **ASMD (自适应子图密度)** 机制，解决了高速下点密度衰减导致的配准失效问题。
*   **适用场景**：极端高速、高噪声、强动态且对计算时延极其敏感的机器人与自动驾驶场景。

---

### 5. 实验分析
*   **关键结果**：在 250km/h 高速赛车数据上，保持单一参数集，平均运行时间仅为 19.23ms，且相对于主流基线误差显著下降。
*   **优势**：极低的平均延迟与确定性的计算分布，在极端动态环境下稳定性优于传统 CPU 框架。
*   **局限**：对 GPU 硬件（如 NVIDIA A5000）有较强的依赖，且极端长序列下的 pitch 漂移需进一步优化。

---

### 6. 实用指南
*   **开源地址**：[https://github.com/TUMFTM/FAR-LIO](https://github.com/TUMFTM/FAR-LIO)
*   **实现细节**：依赖 `cuCollections` 和 `Thrust` 库；需注意体素大小（v=4m）与点数上限（N=40）的设置，这是平衡精度与性能的关键超参数。
*   **迁移建议**：其核心的 `cuVoxelMap` 可直接移植到其他 SLAM 系统（如 LOAM 或 LeGO-LOAM）以替代传统的 kd-tree 实现。

---

### 7. 总结
*   **核心思想**：全 CUDA 并行流水线实现低延迟与高鲁棒性里程计。
*   **速记版 Pipeline**：
    1. 计算并去除运动畸变。
    2. 在 GPU 内存中完成哈希化点云关联。
    3. 执行稀疏感知配准计算位姿。
    4. 用 EKF 补偿时延并平滑输出。

**Key Findings:**

- Our system leverages a novel CUDA-based voxel hashmap to enable parallelized nearest-neighbor search and efficient map updates.
- FAR-LIO achieves an average 6.9% reduction in the positional error and 38.4% lower runtime compared to state-of-the-art baselines on target hardware using a single parameter set.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26010v1)
- [arXiv](https://arxiv.org/abs/2606.26010v1)

---

<a id='2606.26007v1'></a>
## [From Sparse and Imperfect 2D Anchors to Consistent 3D Gaussian Street Scenes: Support-Aware Appearance](https://arxiv.org/abs/2606.26007v1)

**Authors:** Long Cao, Zhongquan Wang, Jie Li, Yuhan Chen, Kefei Qian, Xiangfei Huang, Guofa Li

**Published:** 2026-06-24

**Categories:** cs.CV, cs.GR

**Abstract:**

Image priors can synthesize target conditions for 3D Gaussian street scenes, but independently edited views do not define a coherent 3D target. Direct fitting can propagate view-specific noise, while existing pipelines do not jointly handle imperfect sparse anchors and standard-rasterizer deployment. To address this gap, teacher-relative appearance residual distillation is introduced for appearance baking. A structured space for frequency decomposition, confidence estimation, and primitive-level lifting is formed by residuals between teacher anchors and original renders. The direct optimization signal is supplied by renderer-space matching, while primitive assignment is regularized by support-aware Gaussian-space aggregation. Supported detail is admitted and unsupported noise is suppressed through confidence-gated coarse-to-fine optimization, after which all residuals are baked into fixed-geometry spherical-harmonic coefficients. The teacher and auxiliary training modules are discarded at inference. Evaluation across Waymo street assets, Tanks and Temples scenes, and multiple target conditions shows a favorable overall balance of target alignment, content preservation, artifact suppression, and cross-view consistency over editing-based baselines. Ablations confirm the effectiveness of the main components. Code will be released at https://github.com/Cagares/Baking-for-3D-Gaussian.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为《From Sparse and Imperfect 2D Anchors to Consistent 3D Gaussian Street Scenes: Support-Aware Appearance》的论文进行了如下深度分析：

### 1. 核心贡献摘要
该论文提出了一种针对3D高斯泼溅（3D Gaussian Splatting, 3DGS）街道场景的“外观烘焙”（Appearance Baking）框架，旨在解决从稀疏且有缺陷的2D锚点合成一致性3D场景时的噪声传播问题。通过引入教师-学生蒸馏机制与支撑感知（Support-Aware）的优化策略，该方法能够在保持几何结构的同时，有效地将视图无关的细节烘焙至球谐函数系数中，实现了高质量、跨视图一致的场景重建。

### 2. 关键创新与方法论
*   **教师-学生残差蒸馏（Teacher-Relative Appearance Residual Distillation）：** 这是该方法的核心。通过计算教师锚点与原始渲染之间的残差，模型构建了一个结构化的空间，用于频率分解和置信度估计。这种方式将“内容”与“噪声”分离开来。
*   **支撑感知高斯聚合（Support-Aware Gaussian-space Aggregation）：** 该机制通过置信度门控（Confidence-gated）的方式，区分了“受支持的细节”（Supported detail）与“不受支持的噪声”（Unsupported noise）。这确保了只有具有高度一致性语义的特征才会被优化进3D模型。
*   **外观烘焙架构（Appearance Baking）：** 将学习到的残差信息最终烘焙到固定几何的球谐系数中。这种设计的一个巨大优势是**推理时计算代价极低**——因为教师模型和辅助模块在推理阶段被直接丢弃，仅保留标准的3DGS表示。

### 3. 对计算机视觉领域的潜在影响
*   **填补了编辑与重建的鸿沟：** 目前的3D生成或编辑模型（如基于Diffusion的视角合成）通常无法保证多视角几何一致性，该研究展示了如何利用这些不完美的先验去“提炼”出一个一致的3D场景。
*   **推动了隐式表征的应用落地：** 3DGS在城市级大规模场景中面临存储与一致性挑战，该方法提供了一种高效的正则化路径，使得3DGS在复杂室外环境下具备了更高的鲁棒性。

### 4. 受益的相关领域与应用
*   **自动驾驶仿真：** 在Waymo等数据集上的优异表现直接服务于自动驾驶的场景回放与虚拟测试，能够利用零散的采集数据生成高质量的数字孪生街道。
*   **虚拟现实（VR）与元宇宙资产生成：** 为从单张图像或稀疏视角构建大规模3D场景提供了一条低成本、高保真的技术路线。
*   **影视后期与场景编辑：** 该方法能够处理“不完美的编辑需求”，为艺术家在街道场景中进行一致性修改提供了技术保障。

### 5. 可推断的局限性
*   **对初始化质量的依赖：** 虽然文章声称能处理“不完美的锚点”，但如果初始化的稀疏锚点结构本身存在严重的几何偏差（例如深度错误或空洞），该方法在补全这些深度信息时可能仍会产生渲染模糊。
*   **固定几何的制约：** 由于最终将残差烘焙进固定几何的球谐系数，该方法可能难以处理大规模场景中动态变化的外观（如复杂的动态光影变化或反射变化），因为这些复杂效应可能超出了球谐函数（SH）的表示能力。
*   **复杂场景的训练开销：** 虽然推理速度很快，但引入教师模块和辅助训练过程意味着训练阶段的显存需求和计算负载较高，在大规模城市场景的端到端训练中可能面临工程瓶颈。

**专家点评：**
这篇论文的有趣之处在于它没有试图去改进底层渲染引擎，而是通过**“蒸馏式烘焙”**的策略处理了生成模型先验的非一致性。这种“离线训练-在线轻量化”的思路是当前3D视觉领域实现从实验室研究向工业应用落地的最佳实践方案之一。

**Key Findings:**

- Ablations confirm the effectiveness of the main components.
- Code will be released at https://github.com/Cagares/Baking-for-3D-Gaussian.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26007v1)
- [arXiv](https://arxiv.org/abs/2606.26007v1)

---

<a id='2606.26006v1'></a>
## [FORCE: Efficient VLA Reinforcement Fine-Tuning via Value-Calibrated Warm-up and Self-Distillation](https://arxiv.org/abs/2606.26006v1)

**Authors:** Shuyi Zhang, Yunfan Lou, Hongyang Cheng, Yichen Guo, Chuyao Fu, Yaoxu Lyu, Xiaojie Zhang, Haoran Li, Pengwei Wang, Zhongyuan Wang, Shanghang Zhang

**Published:** 2026-06-24

**Categories:** cs.RO, cs.AI

**Abstract:**

Vision-Language-Action (VLA) models are often constrained by the imitation ceiling imposed by sub-optimal data. While Reinforcement Learning (RL) fine-tuning can surpass this limit, it is notoriously sample inefficient. This challenge arises from two core issues: (1) catastrophic initial unlearning due to an unstable Q-function and (2) inefficient policy updates caused by low-quality exploration data, often forcing a reliance on costly human interventions. We introduce FORCE, a 3-stage framework that stabilizes fine-tuning by tackling both issues. FORCE first incorporates a Value-Calibrated Warm-Up phase, utilizing on-policy rollouts to mitigate the distributional shift of the Q-function. Subsequently, during the online stage, this calibrated Q-function acts as a filter for both the policy's own action proposals and expert data, ensuring only high-value actions are used for the policy update. We evaluate FORCE on various simulation and real-world tasks, and the result shows that FORCE achieves a 79% absolute improvement in success rates and outperform prior RL methods by 10%, while accelerating training by 32.5%. Critically, it mitigates the common success rate drop and achieves this robust performance without human intervention, marking a significant step towards deploying capable and autonomous robotic agents.

**Analysis:**

# 论文分析：FORCE - 高效VLA强化学习微调框架

### 1. 摘要翻译
视觉-语言-动作（VLA）模型往往受限于由次优数据导致的“模仿学习天花板”。虽然强化学习（RL）微调可以突破这一限制，但其样本效率极低。该挑战源于两个核心问题：(1) 由于Q函数不稳定导致的灾难性初始遗忘；(2) 由低质量探索数据导致的策略更新效率低下，往往迫使系统依赖代价高昂的人工干预。我们引入了FORCE，一个通过解决上述两个问题来稳定微调的3阶段框架。FORCE首先整合了“价值校准预热”阶段，利用在线策略推出（rollouts）来减轻Q函数的分布偏移。随后，在在线阶段，此校准后的Q函数作为策略动作建议和专家数据的过滤器，确保仅使用高价值动作进行策略更新。在多种仿真和现实世界任务中的实验表明，FORCE在成功率上实现了79%的绝对提升，超过现有RL方法10%，同时训练速度加快了32.5%。关键的是，它缓解了常见的成功率下降问题，且在无需人工干预的情况下实现了稳健的性能，标志着在部署能力强且自主的机器人代理方面迈出了重要一步。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决VLA模型从离线（模仿学习）到在线（强化学习）微调过程中的不稳定性和样本效率问题。
*   **现有方法痛点**：
    *   **初始遗忘（Initial Unlearning）**：离线预训练的Q函数在面对新在线数据时出现规模匹配失误，导致性能灾难性崩塌。
    *   **低效探索**：在线策略在探索未见区域时产生大量低价值动作，传统的RL方法难以区分这些高质量/低质量探索数据。
*   **研究假设**：通过在微调前显式校准Q函数以适配当前策略分布，并利用价值函数动态过滤探索数据，可以实现无需人工干预的平稳且高效的策略改进。

---

### 3. 方法设计详解
FORCE包含三个阶段：
1.  **离线RL预训练**：使用Cal-QL（Calibrated Q-Learning）在离线数据集上初始化策略和Critic，通过增加校准正则化器约束OOD（分布外）动作的Q值，防止高估。
2.  **分布预热（Distributional Warm-up）**：这是克服O2O（离线到在线）分布偏移的核心。作者收集少量在线Rollout与离线数据混合，在该混合集上继续校准Q函数。这使得Q函数的支持域扩展到包含当前策略的访问分布，从而在在线微调开始前“预稳定”了价值估计。
3.  **价值引导的策略自蒸馏（VGPD）**：在在线阶段，Critic成为动态过滤器。对于策略采样到的动作，计算其价值均值（Baseline），仅保留优于均值的动作用于更新策略。这类似于一种自适应课程学习，早期偏向模仿，后期自动切换为基于价值的自我改进。

---

### 4. 方法对比分析
*   **本质区别**：不同于常规RL直接微调，FORCE显式地引入了“分布预热”环节来桥接静态数据集与动态环境，且采用了基于优势截断的自动课程学习，而非传统的强策略梯度下降。
*   **创新贡献**：
    *   **分布预热机制**：理论上解决了离线到在线过渡的初始性能坍塌问题。
    *   **VGPD**：通过动态优势过滤将探索噪声降至最低，实现了无需人工干预的样本高效微调。
*   **适用场景**：高维连续动作空间的机器人操控任务，特别是对样本效率和自主性要求极高的场景。

---

### 5. 实验分析
*   **验证方法**：在6个ManiSkill仿真任务及现实世界中的Franka机器人上进行验证。
*   **关键结果**：在多个任务上达到接近100%的成功率，平均训练速度相比基线提升32.5%。
*   **主要优势**：性能稳健，彻底规避了训练初期的“策略崩塌”，无需人工干预。
*   **主要局限**：VGPD在每一步需要采样K个候选动作，增加了在线推断的计算开销。

---

### 6. 实用指南
*   **开源情况**：论文提到该框架，建议密切关注作者GitHub（文中提及OpenVLA及后续相关项目）。
*   **实现细节**：
    *   **关键参数**：$\alpha$ (校准约束), $\tau$ (蒸馏温度), $K$ (动作采样数)。
    *   **训练建议**：预热阶段的Rollout规模不宜过大，关键在于覆盖当前策略的起始分布；VGPD的Baseline计算需保证 Critic 梯度更新的稳定性。
*   **迁移可能**：该框架的价值校准和自蒸馏逻辑完全可以迁移到其他基于Transformer的视觉行为任务，尤其是那些依赖离线预训练的模型。

---

### 7. 总结
*   **核心思想**：通过价值函数预校准与自适应蒸馏，平滑离线到在线的策略进化。
*   **速记版pipeline**：
    1.  预训练：先在离线数据上练就基本功。
    2.  预热：混合在线数据修正价值预测，消除起步焦虑。
    3.  蒸馏：把好的动作挑出来学习，把坏的探索噪声丢掉。

**Key Findings:**

- We introduce FORCE, a 3-stage framework that stabilizes fine-tuning by tackling both issues.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.26006v1)
- [arXiv](https://arxiv.org/abs/2606.26006v1)

---

