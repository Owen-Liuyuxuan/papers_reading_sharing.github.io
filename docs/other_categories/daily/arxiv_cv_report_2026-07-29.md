time: 20260729

# Arxiv Computer Vision Papers - 2026-07-29

## Executive Summary

## 执行摘要

### 一、主要主题与趋势

本日收录的10篇论文呈现出以下核心趋势：

- **机器人学习与操作策略**占据主导地位（7篇），重点关注**可部署性、实时性与多模态感知**。从高保真数据（HiFi-UMI）、实时反应式策略（πR²）到视听融合（S2A2）与多臂协同（Tri-Manual），体现了从仿真到真实、从单一模态到多模态融合的演进。
- **多模态基础模型**持续深化，MODUS提出**解码器专用、任意模态到任意模态**的统一框架，与DC-WAM的世界-动作模型、SAM3D引导的视觉-语言-动作对齐形成互补，显示出对通用智能体范式的追求。
- **生成模型加速**方面，并行解码蒸馏（Parallel Decoding Distillation）为图像与视频生成提供了高效率方案。
- **综述类**论文（Instruction-based Image Editing）系统梳理了基于指令的图像编辑领域，为后续研究提供全景式参考。

### 二、关键创新论文

1. **πR² (Reactive Real-time Flow Policies)** — 提出实时反应式流策略，有望解决机器人策略部署中延迟与鲁棒性矛盾，对**闭环控制**意义重大。
2. **MODUS (Decoder-Only Any-to-Any Modeling)** — 以纯解码器架构实现跨任意模态的统一建模，挑战传统编码器-解码器范式，为多模态智能体提供简约而强大的基线。
3. **SAM3D-Guided Object-Centric Representation Alignment** — 将3D分割模型SAM3D引入视觉-语言-动作模型，通过物体中心表示对齐提升机器人操作的可泛化性，是**预训练视觉模型与机器人学习深度融合**的典型范例。
4. **Transformer Transformer** — 将机器人形态设计与运动策略联合建模，实现“形态-运动”协同优化，为**机器人共设计**提供了统一框架。

### 三、新兴研究方向与技术

- **动态中心的世界-动作模型** (DC-WAM)：强调从动态变化中学习世界模型，而非静态场景，可提升机器人对交互环境的预测能力。
- **声学空间信息用于模仿学习** (S2A2)：将音频模态引入操作学习，利用声音定位与材质属性，拓展了传统视觉-触觉之外的新感知通道。
- **并行蒸馏加速生成**：在扩散模型等生成框架中采用**分布式解码蒸馏**，有望在不牺牲质量的前提下大幅降低推理延迟，对视频生成等实时应用尤为关键。
- **多臂/多人协同模仿学习** (Tri-Manual)：探索三机械臂或人机共融场景下的运动学模仿，推动机器人从单体操作向社群协作发展。

### 四、推荐精读论文（按优先级排序）

1. **πR²** — 若你关注机器人控制中实时性与反应式策略的落地，这是首选。
2. **MODUS** — 对多模态模型架构感兴趣的读者不可错过，其解码器专用设计可能引领未来趋势。
3. **Instruction-based Image Editing (Survey)** — 为该子领域的新手或想系统了解进展的研究者提供全面梳理。
4. **SAM3D-Guided Object-Centric Alignment** — 结合近期SAM系列与VLA模型的研究热点，实用价值高。
5. **HiFi-UMI** — 若你侧重于数据驱动、可部署的机器人策略，本文所提出的高保真数据范式值得细读。

以上论文共同反映了计算机视觉与机器人学交叉领域的蓬勃活力：从静态感知向动态交互、从单模态向多模态、从离线训练向实时部署的全面跃进。建议结合自身研究方向选择性深入阅读。

---

## Table of Contents

1. [HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone](#2607.25895v1)
2. [$π\mathbf{R}^2$: Reactive Real-time Flow Policies](#2607.26055v1)
3. [S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information](#2607.26047v1)
4. [Parallel Decoding Distillation for Fast Image and Video Generation](#2607.26004v1)
5. [MODUS: Decoder-Only Any-to-Any Modeling of Diverse Modalities](#2607.25948v1)
6. [DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models](#2607.25918v1)
7. [SAM3D-Guided Object-Centric Representation Alignment for Vision-Language-Action Models](#2607.25912v1)
8. [Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design](#2607.25798v1)
9. [Tri-Manual Visuomotor Imitation Learning of Robot Policies](#2607.25731v1)
10. [Instruction-based Image Editing: A Survey on Data, Models, Evaluation, and Applications](#2607.25642v1)

---

## Papers

<a id='2607.25895v1'></a>
## [HiFi-UMI: Learning Deployable Manipulation Policies from High-Fidelity UMI Data Alone](https://arxiv.org/abs/2607.25895v1)

**Authors:** Simple AI,  :, Yuteng Wei, Jinming Ma, Jiawei Wang, Weitao Zhou, Yushen Zuo, Ke Rui, Minglei Li, Jinhao Zhang, Zhikang Pan, Xiang Wang, Haoran Jia, Huan Du, Zicheng Zeng, Jun Ma, Guiyu Qin, Di Zhang, Xiaofei Li

**Published:** 2026-07-28

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Learning deployable manipulation policies is bottlenecked by the scarcity of data that is both high-fidelity and scalable. Real-robot teleoperation is accurate but costly to scale; robot-free UMI capture scales readily, and current practice uses the resulting data mainly for pre-training, adding a small real-robot "anchor" at post-training. We ask whether raising the fidelity of robot-free UMI data, rather than shrinking the real-robot fraction, can remove that anchor. We present HiFi-UMI, a portable UMI data-production system co-designed for trajectory accuracy, inter-gripper relative pose, synchronization, and field of view: head-mounted offline stereo-inertial SLAM, native rather than reconstructed relative pose, a shared microsecond GPIO trigger, and two wide-angle cameras per hand covering ~200 degrees. It reaches 3 mm workspace-local end-effector accuracy without external tracking infrastructure. Using this corpus, we demonstrate zero-robot post-training: a policy post-trained solely on HiFi-UMI demonstrations deploys directly on a real robot and matches in-domain teleoperation across three backbones spanning the vision-language-action and world-action-model families, with success-rate differences of -2.5, +3.1, and -0.6 percentage points on StarVLA-QwenPI, OpenPI-pi_0.5, and LingBot-VA; the strongest policy reaches 85% on a precision insertion task, even though the teleoperation baseline is collected in the evaluation scene and no HiFi-UMI trajectory is. Pre-training on 4,000 hours from the same corpus lowers action error on ten unseen tasks by 41% and, on StarVLA-QwenPI, raises real-robot success by a further 18.1 percentage points. We open-source HiFi-UMI-2K, 2,000 hours of microsecond-synchronized, ultra-wide-FoV demonstrations, each automatically reconstructed and validated through simulation replay, as a large-scale, high-fidelity resource for the robot-learning community.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
**HiFi-UMI** 提出了一套高保真、可扩展的机器人操作数据采集系统，成功证明了可以通过提升“机器人离线（robot-free）”数据采集的精度，实现无需真实机器人微调（zero-robot post-training）的策略部署。该研究通过高质量的数据闭环，消除了传统机器人学习中对昂贵且难以扩展的实时遥操作（teleoperation）的依赖，并在多个主流策略模型架构上实现了与实时遥操作相当的性能。

### 2. 核心创新与方法论
该工作的核心在于**“数据质量驱动策略泛化”**，具体创新点包括：
*   **端到端的高精度采集方案**：摒弃了传统的外部追踪设施（如Vicon），采用头戴式立体惯性SLAM进行位姿估计，并引入微秒级GPIO触发器实现多视角视觉与动作的极高同步。
*   **硬件闭环设计**：通过优化手部双广角相机（覆盖约200度视场）和原生相对位姿感知，解决了机器人无关数据（robot-free data）中最棘手的精度退化问题。
*   **验证范式**：引入了“模拟重放（simulation replay）”机制对采集数据进行自动重构与有效性验证，确保了大规模数据集的鲁棒性。
*   **Zero-Robot策略架构**：证明了当数据保真度达到一定阈值（3mm工作空间局部精度），模型可以跨越“模拟/非实物数据”与“真实物理世界”之间的鸿沟，无需额外真机锚点即可直接部署。

### 3. 对领域的潜在影响
*   **打破数据瓶颈**：极大地降低了高质量机器人数据采集的成本，为机器人领域构建像 ImageNet 或 Common Crawl 那样的“大规模高质量数据集”提供了蓝图。
*   **重塑训练范式**：如果“机器人无关数据”的保真度足以支撑直接部署，那么机器人策略学习将从“数据稀缺/成本昂贵”转变为“数据工程/采集效率”的竞争，显著加速机器人学的规模化发展。
*   **视觉与动作的深度耦合**：该论文对于计算机视觉领域而言，强调了在复杂的动态手部操作场景下，视觉输入不仅需要“看清”，还需要极高的空间几何保真度（Spatial Fidelity），这对未来视觉模型在动作空间中的表征学习具有重要启示。

### 4. 受益的相关领域与应用
*   **具身智能 (Embodied AI)**：特别是端到端动作预测模型（如VLA模型）。
*   **复杂操作任务**：如文中所述的精密插入（precision insertion）任务，以及居家/生产线中的灵巧操作。
*   **大规模视觉预训练**：HiFi-UMI-2K 数据集作为通用机器人学习的预训练语料，将直接推动视觉-动作（Vision-to-Action）预训练模型的发展。
*   **远程操控与数字孪生**：该系统中使用的SLAM与自动重构技术可直接应用于增强现实（AR）或数字孪生数据的生产。

### 5. 可推断的局限性
*   **硬件部署约束**：尽管该系统是“便携式”的，但头戴式设备和外部同步硬件对操作者的舒适度及系统配置有一定要求，并非完全无需门槛。
*   **环境依赖性**：虽然通过高精度数据提升了泛化性，但在极端未知或与采集环境视觉特征差异巨大的场景下，其鲁棒性仍有待验证（尤其是光照变化或动态复杂干扰）。
*   **触觉缺失**：该方案高度依赖视觉保真度，但在涉及复杂力反馈（Force-feedback）或非视觉感知类任务（如盲操作、硬度感知）时，仅靠视觉数据可能存在上限。
*   **算力成本**：虽节省了机器人采集时间，但大规模数据的自动重构与验证对计算资源的需求依然巨大。

**总结：** 这篇论文是具身智能领域的一个重要里程碑，它有力地论证了“数据精度即策略上限”。通过将机器人学习从昂贵的物理实验中解放出来，HiFi-UMI 为构建通用操作模型打开了通往大规模数据时代的快速通道。

**Key Findings:**

- We present HiFi-UMI, a portable UMI data-production system co-designed for trajectory accuracy, inter-gripper relative pose, synchronization, and field of view: head-mounted offline stereo-inertial SLAM, native rather than reconstructed relative pose, a shared microsecond GPIO trigger, and two wide-angle cameras per hand covering ~200 degrees.
- Using this corpus, we demonstrate zero-robot post-training: a policy post-trained solely on HiFi-UMI demonstrations deploys directly on a real robot and matches in-domain teleoperation across three backbones spanning the vision-language-action and world-action-model families, with success-rate differences of -2.5, +3.1, and -0.6 percentage points on StarVLA-QwenPI, OpenPI-pi_0.5, and LingBot-VA; the strongest policy reaches 85% on a precision insertion task, even though the teleoperation baseline is collected in the evaluation scene and no HiFi-UMI trajectory is.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25895v1)
- [arXiv](https://arxiv.org/abs/2607.25895v1)

---

<a id='2607.26055v1'></a>
## [$π\mathbf{R}^2$: Reactive Real-time Flow Policies](https://arxiv.org/abs/2607.26055v1)

**Authors:** Sungjae Park, Shubham Tulsiani

**Published:** 2026-07-28

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Generalist manipulation policies increasingly take the form of action-chunking flow policies built on large pretrained backbones. Such chunks run open-loop, so the policy cannot react to sensory input arriving mid-execution, sacrificing \emph{reactivity}. Replanning more often would restore it, but the perception-to-action pipeline (a large backbone plus multiple denoising steps) is too slow: this \emph{latency} forbids frequent replanning and leaves committed actions stale, making such policies ill-suited for dynamic, closed-loop control. We present $π\mathbf{R}^2$, which makes these policies reactive and real-time while retaining large backbones, expressive multi-modal policies, and multi-action prediction. Built on the per-position noise schedule of diffusion forcing, $π\mathbf{R}^2$ contributes two ideas. First, it splits conditioning into a fast channel (proprioception, fresh every tick) and an asynchronously updated slow channel (vision-language features), so the policy reacts to proprioception within a chunk while tolerating stale vision. Second, a latency-adaptive flow schedule treats in-flight actions as inpainting conditioning and emits actions in one denoising step per call, letting one trained model adapt to varying hardware latency. Requiring minimal modification to existing architectures, $π\mathbf{R}^2$ can be finetuned from a pretrained policy: applied to GR00T-N1.7 on a real xArm6+XHand platform, it replans closed-loop roughly $4\times$ faster than the base policy (~$25$Hz on an A5000 GPU), acting on a fresh observation every $40$ms. Across simulation and real-world manipulation tasks, $π\mathbf{R}^2$ improves the success rate by up to $23\%$ in simulation and $30\%$ in the real world over the strongest baseline. Project page: https://pi-r2-flow.github.io/

**Analysis:**

### 1. 摘要翻译
通用操纵策略通常基于大型预训练主干网络，采用动作分块（Action Chunking）流策略。由于这类分块以开环方式运行，策略无法在执行中途对传感器输入做出反应，从而牺牲了响应性。虽然提高重规划频率可以恢复响应性，但感知到动作的管线（庞大的主干网络加多次去噪步骤）延迟过高，使得频繁重规划变得不可行。我们提出了 $\pi R^2$，它在保留大型主干网络、高表达力多模态策略和多动作预测能力的同时，实现了策略的实时响应。基于扩散强制（Diffusion Forcing）的按位置噪声调度，$\pi R^2$ 贡献了两点创新：首先，将条件输入拆分为快通道（实时本体感觉）和异步更新的慢通道（视觉-语言特征），使策略在执行分块期间能对本体感觉做出反应，同时容忍视觉信息的滞后；其次，开发了一种延迟自适应流调度方案，将执行中的动作视为修复（inpainting）条件，并在每次调用时仅执行一步去噪，使单一训练模型能够适应不同的硬件延迟。在真实的 xArm6+XHand 平台上，$\pi R^2$ 的闭环重规划速度比基础策略快约 4 倍，且在模拟和真实世界的操纵任务中，成功率分别比最强基线提高了 23% 和 30%。

---

### 2. 方法动机分析
*   **驱动力**：解决大型机器人基础模型（如 VLA）在动态操纵任务中因为推理延迟高而导致的响应性（Reactivity）不足的问题。
*   **痛点**：传统的动作分块策略强制模型以开环方式执行一段完整的动作，期间无法感知环境变化；而直接尝试高频重规划又会因主干网络（视觉编码器+VLM）处理图像的巨大算力需求导致系统产生不可接受的推理延迟。
*   **核心直觉**：感知与动作并非都需要同一频率。本体感觉（Proprioception）轻量且高频，适合局部快速纠偏；而视觉语言特征复杂且高延迟，适合提供全局任务指导。通过解耦两者，并在去噪过程中采用自适应噪声调度，可以实现低延迟的闭环控制。

---

### 3. 方法设计详解
*   **流程 pipeline**：
    1.  **特征解耦**：将输入观测分为 $o_{fast}$（本体感觉）和 $o_{slow}$（视觉-语言特征）。$o_{slow}$ 在后台异步更新，利用缓存的旧特征进行推理；$o_{fast}$ 在每次控制周期更新。
    2.  **延迟自适应调度（Staircase Schedule）**：在训练时采样 inference 延迟 $d$，构建三段式噪声调度曲线（前段 $d$ 为已执行动作，保持清洁用于修复；中间为线性降噪 ramp；后端为纯噪声）。
    3.  **单步推理**：动作头不再执行完整的 $K$ 步去噪，而是结合最新的 $o_{fast}$ 和滞后的 $o_{slow}$，在每次调用时仅执行**一步（One Euler step）**去噪，快速输出下一个动作。
*   **算法本质**：将动作预测从“从纯噪声生成完整 chunk”转化为“基于当前状态，不断对已知动作序列进行细化（Inpainting）”。

---

### 4. 方法对比分析
*   **本质区别**：与现有流模型（Flow Matching）不同，$\pi R^2$ 引入了异步处理机制和延迟自适应噪声 schedule，将原本“开环分块预测”变为了“高频闭环纠偏”。
*   **创新点**：
    *   **异步感知机制**：解耦感知通道，成功绕过 VLM 主干的计算瓶颈。
    *   **阶梯式噪声进度表**：通过随机采样延迟 $d$ 进行训练，使模型具备了对推理延迟的鲁棒性。
*   **适用场景**：高动态、接触密集型、需要频繁纠偏的机器人操纵任务。

---

### 5. 实验分析
*   **验证方法**：在 MuJoCo Leap Cube Reorientation 任务和真实世界 xArm6+XHand 平台上对比了基础策略、Naive Async、以及训练时 RTC（Train-Time RTC）等基线。
*   **结论**：在处理高延迟环境时，$\pi R^2$ 表现出更优的响应能力，成功率显著提升，且在面对意外接触（如接住掉落物）时，能够即时调整握力，避免过冲或掉落。
*   **优势**：极高的响应性（25Hz）；硬件适应性强；架构修改极小，可直接在现有预训练 VLA 上微调。

---

### 6. 实用指南
*   **实现细节**：
    *   **AdaLN 调制**：将 AdaLN 的条件参数改为按位置（per-position）调制，这对于处理分块动作极其关键。
    *   **推理循环**：必须实现一个独立的后台线程用于 VLM 异步推理，并将推理延迟 $d$ 通过滚窗（rolling window）实时测量反馈给模型。
*   **迁移建议**：该方法适用于任何基于 DiT 或 U-Net 架构的扩散/流匹配策略。迁移时，只需保留原有的 backbone，将动作头替换为本文提出的分块调度 head，并确保训练时采样了 $d$ 分布即可。

---

### 7. 总结
*   **核心思想**：通过多模态异步感知与自适应噪声调度，实现实时闭环机器人控制。
*   **速记版 Pipeline**：
    1.  **解耦输入**：视觉走慢通道异步更新，本体感觉走快通道实时更新。
    2.  **阶梯调度**：引入延迟 $d$ 进行训练，构建支持实时修正的噪声曲线。
    3.  **单步推演**：每次执行仅跑一次去噪，利用当前最新状态更新动作流。

**Key Findings:**

- We present $π\mathbf{R}^2$, which makes these policies reactive and real-time while retaining large backbones, expressive multi-modal policies, and multi-action prediction.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.26055v1)
- [arXiv](https://arxiv.org/abs/2607.26055v1)

---

<a id='2607.26047v1'></a>
## [S2A2: Audio-Visual Imitation Learning for Manipulation Tasks Using Acoustic Spatial Information](https://arxiv.org/abs/2607.26047v1)

**Authors:** Kaneyoshi Hiratsuka, Benjamin Yen, Ryosuke Kojima

**Published:** 2026-07-28

**Categories:** cs.RO

**Abstract:**

Acoustic information provides rich cues about object location, material properties, and changes caused by contact or motion. This paper introduces a new set of acoustic-aware manipulation tasks for imitation learning, in which robots must use auditory cues to determine manipulation targets. These tasks require sound source localization and identification for active exploration in robotic manipulation. Also, we propose a multimodal imitation learning framework, Spatial-Spectral Audio Action (S2A2), that integrates visual features with acoustic spatial and acoustic signal information for the acoustic-aware manipulation tasks. We implemented S2A2 models that integrates policies such as ACT, Diffusion Policy, VQ-BeT, and $π_0$, into our framework. Simulation experiments showed that the proposed method is the most effective for tasks requiring both position and timbre. Furthermore, real-robot experiments confirm the applicability of the proposed tasks and framework to real-world manipulation.

**Analysis:**

### 1. 摘要翻译
声学信息提供了关于物体位置、材料属性以及由接触或运动引起的变化的丰富线索。本文引入了一套用于模仿学习的“声学感知操作任务”新框架，要求机器人必须利用听觉提示来确定操作目标，这些任务涉及主动探索中的声源定位与识别。为此，我们提出了空间-光谱音频动作（S2A2）框架，它将视觉特征与声学空间及信号信息相结合。我们实现了将S2A2集成到ACT、Diffusion Policy、VQ-BeT和π0等策略中的模型。仿真实验表明，该方法在需要结合位置和音色信息的任务中最为有效。此外，真实机器人实验证实了该任务和框架在现实世界操作中的适用性。

### 2. 方法动机分析
*   **驱动力**：旨在解决传统视觉中心（vision-centric）操纵在照明变化、遮挡、物体外观难以分辨或内部状态不可见时的鲁棒性问题。
*   **现有痛点**：现有方法多依赖单通道音频或接触式传感器，忽略了声学信息中蕴含的“空间分布”线索，导致在复杂环境下无法有效进行目标选择和主动探索。
*   **核心直觉**：声音不仅包含“发生了什么”（频谱信号），还包含“在哪里发生”（空间定位）。将声学空间映射（空间信息）与谱图（信号信息）融合，能提供缺失的物理感知维度。

### 3. 方法设计详解
S2A2框架包含三个并行流水线：
*   **声学空间映射流水线 (Acoustic Spatial Map Pipeline)**：基于麦克风阵列，应用MUSIC方法估计源方向，结合距离衰减项，投影到224×224的2D网格上，构建声源存在似然图，由ResNet-18编码。
*   **光谱分析流水线 (Spectrogram Pipeline)**：基于声学空间图中的峰值进行源定位，利用Spotforming技术（NMF算法）从多个阵列信号中提取目标区域的清晰谱图，去除环境噪声与共向噪声，由ResNet-18编码。
*   **多模态策略 (Multimodal Policy)**：将上述两个编码特征与视觉编码特征、本体感知信息连接（Concat），输入到主流策略（如Diffusion Policy或ACT）中进行动作生成。
*   **关键公式意义**：$S_c(u)$ 将方位角概率分布映射到空间，引入距离衰减以抑制远场噪声；$G(f, \tau)$ 利用NMF分解去除侧瓣干扰，确保策略只关注目标声源特征。

### 4. 方法对比分析
*   **本质区别**：从传统的“视觉+声学信号”升级为“视觉+声学空间图+声学信号”，引入了空间几何维度的声学感知。
*   **创新贡献**：首次将多阵列麦克风的声学定位与操作模仿学习深度耦合。
*   **适用场景**：适用于视觉完全混淆但声学特征鲜明（如材质不同、声音位置不同）的复杂工业或家庭操作场景。

### 5. 实验分析（精简版）
*   **验证方法**：在Genesis物理仿真与Pyroomacoustics音频仿真集成的环境中进行，包括Localization（定位）、Identification（识别）、L&I（定位加识别）、Exploratory（主动探索）四类任务。
*   **关键结论**：在L&I这种需要时空双重信息的任务中，S2A2表现出显著优越性；若任务仅需单一模态，冗余的声学分支反而可能因数据限制导致性能下降。
*   **优势**：极大地解决了遮挡和视觉 indistinguishable（难以分辨）的问题。
*   **局限**：对策略架构的依赖性较强，且目前的模型在高维度、复杂环境下的泛化仍受限于训练数据量。

### 6. 实用指南
*   **开源情况**：核心算法逻辑已在论文中明确，涉及开源环境包括Genesis和Pyroomacoustics。
*   **实现细节**：
    *   **NMF迭代**：初始化需40步以收敛，后续步仅需15步以减少计算压力。
    *   **超参数**：麦克风阵列数量 $N=4$ 为性能平衡点；FPS建议设置在10以上。
*   **迁移可能**：可直接迁移至移动机器人导航或复杂环境下的多目标交互任务，只需修改空间投影的坐标系定义。

### 7. 总结
*   **核心思想**：融合声学空间概率图与特征谱图，实现多模态精准操作。
*   **速记版pipeline**：
    1.  麦克风阵列采集并利用MUSIC构建2D声学空间图。
    2.  根据空间图定位声源，使用Spotforming提取降噪光谱。
    3.  融合视觉、空间图与光谱信息特征。
    4.  输入策略网络输出控制指令。

**Key Findings:**

- This paper introduces a new set of acoustic-aware manipulation tasks for imitation learning, in which robots must use auditory cues to determine manipulation targets.
- Also, we propose a multimodal imitation learning framework, Spatial-Spectral Audio Action (S2A2), that integrates visual features with acoustic spatial and acoustic signal information for the acoustic-aware manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.26047v1)
- [arXiv](https://arxiv.org/abs/2607.26047v1)

---

<a id='2607.26004v1'></a>
## [Parallel Decoding Distillation for Fast Image and Video Generation](https://arxiv.org/abs/2607.26004v1)

**Authors:** Neta Shaul, Chao Liu, Arash Vahdat, Julius Berner

**Published:** 2026-07-28

**Categories:** cs.CV, cs.LG

**Abstract:**

Generation in video diffusion or flow models is computationally expensive due to the slow and iterative sampling process. Current state-of-the-art (SOTA) acceleration methods heavily rely on variational score distillation (VSD) and adversarial losses to distill diffusion models into few-step generators. Albeit achieving high-quality video generation, these training losses are notoriously hard to optimize and suffer from mode collapse, leading to loss of video diversity and lack of motion. In this paper, we introduce Parallel Decoding Distillation (PDD), a simplified and scalable trajectory-based distillation method for fast inference of diffusion and flow matching models. Our architecture and training procedure are compatible with any pre-trained model and support sampling with a varying number of function evaluations (NFE). PDD accelerates generation by predicting multiple denoising steps per network evaluation. Conceptually, it learns a representation of the mean velocity without regressing its derivative using JVPs or finite-difference approximations. Our method achieves SOTA performance with 4-8 NFE on LTX-2.3 Text-to-Video/Audio, Wan 14B Text-to-Video, and Qwen-Image Text-to-Image. Moreover, PDD presents a significant improvement in generated video diversity.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文《Parallel Decoding Distillation for Fast Image and Video Generation》的分析如下：

### 1. 主要贡献总结
该论文提出了一种名为**并行解码蒸馏（Parallel Decoding Distillation, PDD）**的新型轨迹蒸馏方法，旨在解决扩散模型与流匹配模型在图像和视频生成中采样迭代速度慢的问题。该方法通过在单次网络评估中预测多个去噪步骤，在保证生成质量的同时，实现了仅需 4-8 次函数评估（NFE）即可达到 SOTA 性能，并有效缓解了传统蒸馏方法中常见的视频多样性缺失和运动停滞问题。

### 2. 核心创新与方法论
*   **并行解码策略：** 不同于传统方法逐一迭代，PDD 允许模型在一个网络调用周期内预测轨迹上的多个点，从而大幅压缩推理时间。
*   **轨迹学习（Trajectory-based Learning）：** 该方法巧妙地学习了均值速度的表示，而非像传统蒸馏那样强制回归导数（这通常涉及复杂的雅可比向量积 JVP 或有限差分近似），这种设计使得优化过程更加稳定且易于扩展。
*   **架构通用性：** 这种训练过程与预训练模型解耦，具有极强的兼容性，适用于现有的扩散模型和流匹配模型（如 LTX-2.3, Wan 14B 等）。

### 3. 对领域的潜在影响
*   **打破生成速度与质量的权衡（Speed-Quality Trade-off）：** 长期以来，高画质生成依赖数百次迭代，而快速生成通常伴随严重的视觉伪影。PDD 提供了一条高效的路径，使高性能模型在消费级硬件上的实时部署成为可能。
*   **解决“模式崩溃”痛点：** 通过规避基于对抗损失（Adversarial Losses）的训练范式，PDD 解决了视频生成中常见的“运动僵化”问题，这对于提升生成视频的动态连贯性和现实感具有里程碑意义。

### 4. 受益的领域与应用
*   **实时视频生成应用：** 交互式数字媒体、游戏引擎中的实时动态生成，以及直播间的实时视觉合成。
*   **边缘计算与移动端 AI：** 由于大幅降低了 NFE（仅需 4-8 次），使得在算力受限的设备上运行大模型（如 Wan 14B）变得可行。
*   **生成式音频与多模态合成：** 论文提及该方法同样适用于音频生成，这暗示其算法框架在跨模态任务中具有良好的泛化能力。

### 5. 可推断的局限性
*   **精度瓶颈：** 虽然 4-8 NFE 已经非常高效，但在极低 NFE（如 1-2 NFE）下，模型是否仍能保持极高的细节保真度仍需观察，毕竟“并行解码”可能存在误差累积。
*   **对预训练模型的依赖：** 尽管方法通用，但蒸馏过程仍需依赖一个高质量的“教师模型”。若教师模型本身在特定领域（如长视频连贯性）表现不佳，PDD 可能难以超越该上限。
*   **训练成本与复杂性：** 尽管推理加速显著，但论文未详细提及在大模型上的蒸馏所需的显存消耗与训练时长，对于个人研究者而言，这种蒸馏过程可能依然是资源密集型的。

**专家总结：**
这篇论文的趣味性在于它绕过了目前主流（但难以收敛）的对抗性蒸馏技术，转向了更具稳定性的轨迹建模。在当前的视频生成大模型竞赛（Wan 14B, Sora-like models）中，这种能够**“既要速度又要多样性”**的方法论，极有可能成为未来工业级落地的主流方案。

**Key Findings:**

- Current state-of-the-art (SOTA) acceleration methods heavily rely on variational score distillation (VSD) and adversarial losses to distill diffusion models into few-step generators.
- In this paper, we introduce Parallel Decoding Distillation (PDD), a simplified and scalable trajectory-based distillation method for fast inference of diffusion and flow matching models.
- Our method achieves SOTA performance with 4-8 NFE on LTX-2.3 Text-to-Video/Audio, Wan 14B Text-to-Video, and Qwen-Image Text-to-Image.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.26004v1)
- [arXiv](https://arxiv.org/abs/2607.26004v1)

---

<a id='2607.25948v1'></a>
## [MODUS: Decoder-Only Any-to-Any Modeling of Diverse Modalities](https://arxiv.org/abs/2607.25948v1)

**Authors:** Mingqiao Ye, Zhaochong An, Zhitong Gao, Xian Liu, François Fleuret, Chuan Li, Amir Zadeh, Serge Belongie, Afshin Dehghan, Jesse Allardice, David Mizrahi, Oğuzhan Fatih Kar, Roman Bachmann, Amir Zamir

**Published:** 2026-07-28

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Any-to-any models predict any modality from any combination of others within a single network, a formulation used in multimodal vision and vision-language models, and increasingly in scientific domains such as ecology and astronomy. Existing any-to-any models are typically trained from scratch using encoder-decoder or diffusion architectures, impacting their performance and preventing them from using strong pre-trained decoder-only models as a prior. In this work, we investigate decoder-only any-to-any multimodal modeling, which treats all modalities symmetrically and supports arbitrary modalities as inputs and outputs without modality-specific heads, losses, or task pipelines. Because every modality is both an input and an output of the same model, the resulting model, named Modus, can support a range of applications, such as chained generation through intermediate modalities or cross-modal self-verification by scoring the model's own outputs with another generated modality. Modus demonstrates strong out-of-the-box performance and is competitive with specialist and multitask baselines using a single model across various benchmarks. All materials are open-sourced at https://modus-multimodal.epfl.ch/.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **MODUS** 的论文进行了深入分析。以下是针对该工作的专业评估：

### 1. 论文核心贡献总结
MODUS 提出了一种基于 **Decoder-Only（仅解码器）架构** 的通用“任意到任意”（Any-to-Any）多模态建模范式，打破了以往依赖特定编码器或扩散模型的限制。该研究证明了通过对称处理不同模态，可以在无需特定任务头（task-specific heads）或复杂管道的情况下，实现跨模态的统一生成、推理和自验证。

### 2. 关键创新与方法论
*   **架构范式转移（Decoder-Only Prior）**：这是最显著的突破。它利用了预训练的大型 Decoder-only 模型（如大语言模型架构）作为先验，而非从零开始训练复杂的 Encoder-Decoder 或扩散模型。这不仅利用了现有预训练模型强大的推理能力，还简化了训练目标。
*   **模态的对称性（Symmetric Treatment）**：MODUS 将所有模态视为地位平等的输入和输出，不再区分“提示词输入”和“生成输出”。这种对称性允许模型直接进行“模态序列生成”，即通过中间模态进行链式生成（Chained Generation）。
*   **去耦合化设计**：抛弃了模态特定的损失函数或任务流水线，通过统一的建模方式处理多模态交互，增强了系统的扩展性。

### 3. 对计算机视觉领域的影响
*   **统一多模态学习的范式化**：目前视觉与多模态模型往往针对特定任务（如 Image-to-Text 或 Text-to-Image）设计，MODUS 提供了一种“大一统”的架构思路，有望降低多模态系统的工程复杂度。
*   **解锁自验证能力**：论文提到的“交叉模态自验证”（Cross-modal self-verification）具有极大的潜力——模型可以利用生成出的模态（例如根据生成的图像再反向生成描述或评分）来评估自身输出的质量，这对于增强生成式 AI 的可靠性至关重要。

### 4. 相关领域与应用价值
*   **科学计算与跨域分析**：摘要中提到的生态学和天文学应用非常契合。在这些领域，数据往往不仅限于图像，还包含传感器读数、时间序列和文本，MODUS 的通用性可以极大简化跨领域数据的多模态融合分析。
*   **复杂推理任务**：对于需要多步推理的生成任务，MODUS 的链式生成特性能够让系统在不同模态间过渡，例如：音频 -> 视觉描述 -> 图像生成，实现更连贯的语义表达。
*   **模型高效微调**：由于基于成熟的 Decoder-only 框架，开发者可以更容易地利用现有的 PEFT（参数高效微调）技术在特定领域扩展模型能力。

### 5. 潜在局限性（推断）
*   **模态对齐难度**：虽然架构通用，但在不同模态间的语义对齐（Tokenization）上，Decoder-only 模型可能面临挑战，特别是处理极高分辨率图像或视频时的 token 序列长度问题。
*   **推理延迟与计算资源**：Decoder-only 模型在处理长序列（如高保真图像或视频流）时，由于 Transformer 的自注意力机制（Self-attention）二次复杂度，可能会遇到显存和推理效率的瓶颈。
*   **特定任务性能上限**：通常情况下，通用模型在处理极度专业化的领域（如医学影像精细诊断）时，可能难以达到针对性优化的小型专门模型（Specialist Model）的性能极限。

---
**专家点评：** 
MODUS 的吸引力在于它试图将“多模态”从“拼接组合式”升级为“原生统一式”。这不仅是对架构的优化，更是对“什么是多模态输入”这一本质问题的重新思考。如果该架构能成功扩展至高维视频或复杂科学数据，它极有可能成为下一代通用基础模型的重要候选架构。

**Key Findings:**

- Modus demonstrates strong out-of-the-box performance and is competitive with specialist and multitask baselines using a single model across various benchmarks.
- All materials are open-sourced at https://modus-multimodal.epfl.ch/.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25948v1)
- [arXiv](https://arxiv.org/abs/2607.25948v1)

---

<a id='2607.25918v1'></a>
## [DC-WAM: Dynamic-Centric Visual Supervision and Reasoning for World-Action Models](https://arxiv.org/abs/2607.25918v1)

**Authors:** Haoyuan Ji, Lingxiang Fan, Shang Su, Yinqiao Lu, Mengkai Shi, Jun Gao, Shuo Feng

**Published:** 2026-07-28

**Categories:** cs.RO

**Abstract:**

World-Action Models (WAMs) augment robot policies with future visual prediction, but it remains unclear what the visual modality should learn for control. While photorealistic future prediction provides dense supervision, it also incurs substantial computation and can allocate capacity to texture, illumination, and background variations that are only weakly related to action selection. Recent efficient WAM variants suggest that the main benefit of the video branch may not lie in the rendered future itself, but in the control-relevant visual representations induced during training. In this work, we revisit future video prediction from a dynamic-centric perspective and ask whether an existing RGB-based WAM can be redirected from appearance-dominated reconstruction toward interaction-induced visual dynamics without introducing additional modality-specific predictions or online inputs at deployment. We propose DC-WAM, a dynamic-centric WAM framework that redistributes supervision and computation in the RGB video branch. At the supervision level, DC-WAM combines temporal-difference flow matching with trajectory-guided weighting, emphasizing dense temporal changes and localized regions where the gripper, manipulated objects, and contact areas move. At the reasoning level, DynaRoute predicts token-wise dynamic relevance and converts it into an attention bias, guiding the model toward control-relevant future tokens. Experiments in simulation and on real-world manipulation tasks show that DC-WAM consistently improves policy performance, especially under out-of-distribution perturbations in lighting, object appearance, and background texture.

**Analysis:**

### 1. 摘要翻译
世界动作模型（WAMs）通过预测未来视觉演变来增强机器人策略，但目前尚不清楚视觉模态应学习何种信息以服务于控制。尽管照片级的未来预测能提供稠密监督，但它消耗了大量计算资源，且容易将容量分配给与动作选择相关性较弱的纹理、照明和背景变化。本文从动态中心（Dynamic-Centric）的角度重新审视未来视频预测，探讨是否可以将现有的RGB-WAM从外观主导的重建转向交互诱导的视觉动态，且无需在推理时引入额外的模态或输入。为此，我们提出了DC-WAM，一个重新分配RGB视频分支监督与计算的框架。在监督层面，DC-WAM结合了时间差分流匹配与轨迹引导加权，强调密集的局部变化及机械臂、物体与接触区域的运动。在推理层面，DynaRoute预测token级的动态相关性并将其转化为注意力偏置，引导模型关注与控制相关的未来token。实验表明，DC-WAM在模拟和真实操纵任务中均能一致提升策略性能，特别是在照明、物体外观和背景纹理等分布外（OOD）扰动下表现鲁棒。

### 2. 方法动机分析
*   **驱动力**：现有的WAM过度追求像素级的视觉逼真度（高PSNR），导致模型容量被“无意义”的环境变化占据，而非专注于动作控制的关键动态。
*   **痛点**：均匀的RGB未来预测将复杂的环境因素（如光影、纹理）与manipulation动态纠缠在一起，导致策略在遇到分布偏移（OOD）时极易失效。
*   **研究假设**：控制策略的提升主要源于训练阶段诱导出的“对动态敏感的表示”，而非推理时生成逼真的视频内容本身。

### 3. 方法设计详解
DC-WAM的核心在于将监督信号“动态化”，使模型学会聚焦于交互区域。
*   **流程总结**：
    1.  **离线轨迹采集**：使用CoTracker3和SAM离线生成交互区域的动态热图（Dynamic Map $M^*$）。
    2.  **动态感知监督（Training）**：
        *   **时间差分流匹配（$\mathcal{L}_{TD}^V$）**：通过计算相邻帧预测流的差值，抑制静态背景。
        *   **Tracker引导流匹配（$\mathcal{L}_{TrackFM}^V$）**：利用$M^*$对流匹配损失进行加权，强制模型专注于机械臂接触及物体运动区域。
    3.  **DynaRoute注意力引导（Reasoning）**：训练一个轻量化模块，预测每个token的动态相关性，并将其转换为Attention Bias，直接干预MoT（Mask-of-Tokens）注意力流。
    4.  **高效推理**：在推理时（$t_{init}=1$），仅运行一次视频分支构建Cache，随后完全关闭视频分支，仅利用Cache中的视觉Key-Value进行动作生成。
*   **算法本质**：$\mathcal{L}_{TrackFM}^V$ 是通过空间掩码进行的一种“注意力强制分配”，它告诉网络：哪些像素的预测误差更重要。

### 4. 方法对比分析
*   **本质区别**：DC-WAM不引入额外的视觉模态预测（如点轨迹或语义掩码），而是直接对原生的RGB视频预测过程进行“动态引导”。
*   **创新贡献**：提出将Tracker生成的动态图作为监督信号（$\mathcal{L}_{TrackFM}$）与推理时注意力偏置（DynaRoute）相结合，完美绕过了生成高质量视频的计算瓶颈。

### 5. 实验分析
*   **关键结论**：在LIBERO-Plus测试集（包含视角、照明、背景等OOD变化）中，DC-WAM的平均成功率领先基线7-9个百分点；且证实了PSNR与控制性能并不单调相关。
*   **局限**：方法依赖于离线轨迹生成，虽然训练后不再需要，但如果环境极为复杂导致tracker失效，则动态热图的质量会受影响。

### 6. 实用指南
*   **实现细节**：关键参数为 $\lambda=0.25$ 和 $\sigma_p=1.25$（用于Gaussian Rasterization）。在训练时务必确保Tracker的离线生成是针对训练视频集的，不要泄露测试环境信息。
*   **迁移建议**：该方法非常适合任何基于Video-to-Action的模型（如Transfomer-based WAMs），只需将现有的流匹配损失替换为加权损失，并增加一个轻量级的Attention Bias层。

### 7. 总结
*   **核心思想**：通过动态先验重塑视觉监督，将注意力强制聚焦于交互核心。
*   **速记版Pipeline**：
    1. 离线计算物体运动热图。
    2. 使用热图加权强化动作相关的流预测。
    3. 训练预测器生成动态相关性偏置。
    4. 推理时仅调用一次视频分支，实现高效动作生成。

**Key Findings:**

- We propose DC-WAM, a dynamic-centric WAM framework that redistributes supervision and computation in the RGB video branch.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25918v1)
- [arXiv](https://arxiv.org/abs/2607.25918v1)

---

<a id='2607.25912v1'></a>
## [SAM3D-Guided Object-Centric Representation Alignment for Vision-Language-Action Models](https://arxiv.org/abs/2607.25912v1)

**Authors:** Zonghe Liu, Shanyuan Jie, Xiaoquan Sun, Chen Cao, Zetian Xu, Zongsheng Liu, Jiayu Chen

**Published:** 2026-07-28

**Categories:** cs.RO, cs.AI

**Abstract:**

Vision-Language-Action (VLA) models have shown strong potential for general robot manipulation, but most existing models rely on 2D visual-language backbones and lack fine-grained 3D understanding of target objects, especially under occlusion, pose variation, scale changes, and precise spatial interaction. We propose an object-centric 3D representation alignment framework built upon $π_0$, using SAM3D as a frozen 3D teacher to provide target-object 3D priors during training. Specifically, we localize task-relevant objects with object recognition models, generate corresponding object masks, and use SAM3D to extract dense object-level 3D representations, which are aligned with intermediate visual features of $π_0$. This enables the policy to internalize target-object 3D information while preserving the original RGB-language-to-action inference pipeline without requiring depth, point clouds, masks, SAM3D, or additional 3D modules at test time. Simulation experiments show consistent improvements, achieving 99.1\% on LIBERO and an average length of 4.11 on CALVIN. Real-world experiments further demonstrate that our method is particularly effective in long-horizon manipulation scenarios where the robot must focus on different target objects across multiple subtasks.

**Analysis:**

## 1. 摘要翻译
视觉-语言-动作（VLA）模型在通用机器人操作领域展现出巨大潜力，但大多数现有模型依赖2D视觉-语言主干，缺乏对目标物体的细粒度3D理解，特别是在遮挡、姿态变化、尺度变化和精确空间交互场景下。本文提出了一种基于$\pi_0$的面向对象的3D表示对齐框架，利用SAM3D作为冻结的教师模型，在训练期间提供目标物体的3D先验。具体而言，我们利用目标识别模型定位任务相关物体，生成相应的掩码，并利用SAM3D提取密集的物体级3D表示，将其与$\pi_0$的中间视觉特征进行对齐。这使得策略能够内化目标物体的3D信息，同时在推理时保留了原始的RGB-语言到动作推理流程，无需在测试时使用深度图、点云、掩码、SAM3D或额外的3D模块。仿真实验显示出显著的性能提升，在LIBERO上达到99.1%的成功率，在CALVIN上达到4.11的平均任务长度。真实世界实验进一步证明了该方法在长跨度操作场景中的有效性。

## 2. 方法动机分析
*   **驱动力**：解决RGB基VLA模型在复杂动态环境中缺乏几何感知（形状、尺度、空间布局）的瓶颈，特别是长跨度任务中对物体的持续关注。
*   **现有痛点**：纯2D主干缺乏3D先验；现有3D增强方法往往需要昂贵的测试时输入（深度、点云）或改变原有的输入输出结构。
*   **研究假设**：通过在训练阶段引入冻结的强3D教师（SAM3D），可以将物体的3D几何知识“蒸馏”进VLA的通用视觉特征中，从而使策略在推理阶段仅用RGB图像就能获得“隐式的”3D理解能力。

## 3. 方法设计详解
*   **流程总结**：
    1.  **数据处理**：利用预训练检测器（Grounding DINO）和分割模型（SAM2）获取目标物体的Mask。
    2.  **教师特征提取**：将RGB图像与对应的Mask传入冻结的SAM3D模型，提取物体级的3D几何特征。
    3.  **空间对齐与映射**：将SAM3D输出的特征重塑并插值（Bilinear Interpolation），与VLA中间层的token空间对齐。
    4.  **表示对齐（Alignment）**：计算投影后的VLA特征与SAM3D特征之间的masked MSE loss（仅在目标区域计算），促使VLA特征编码3D先验。
    5.  **联合训练**：动作预测loss与对齐loss结合进行端到端优化。
*   **模型结构**：主体为$\pi_0$架构（SigLIP+Gemma），额外添加一个适配器（Adapter）用于映射，以及一个冻结的SAM3D teacher。
*   **关键公式**：$L = L_{\text{action}} + \alpha L_{\text{align}}$。其中，$L_{\text{align}}$ 对齐了归一化的教师特征与学生投影特征，保证在尺度差异下的稳健性。

## 4. 方法对比分析
*   **本质区别**：它是一种“训练时监督，推理时蒸馏”的框架，即通过对齐操作将3D信息内化到参数权重中，而非在推理时显式拼接3D输入。
*   **创新贡献**：提出了一种不改变模型推理接口的3D先验注入方法；引入了基于子任务的动态掩码监督，增强了长跨度任务的专注力。
*   **适用场景**：适用于需要精确空间操作、存在遮挡或物体位姿多变的工业或家庭服务机器人。

## 5. 实验分析
*   **验证方法**：在LIBERO（仿真）和CALVIN（仿真及真实世界）上进行多任务评估。
*   **关键结果**：在LIBERO-Long（长跨度任务）上表现优异（98.4%），在真实世界遮挡测试环境下平均成功率显著超过基线。
*   **优势**：测试阶段无额外计算开销；无需深度相机；对不同空间布局泛化能力强。
*   **局限**：依赖预处理环节的检测与分割准确度，若检测失败会导致错误的3D先验注入。

## 6. 实用指南
*   **开源情况**：论文目前未明确说明是否开源，但基于$\pi_0$和SAM3D，复现路径清晰。
*   **实现细节**：建议关注教师特征到学生token空间的重采样细节（插值维度对齐）；$\alpha$权重通常设为较小值以平衡任务目标。
*   **迁移可能**：该框架具有通用性，可轻松迁移至其他基于Transformer的VLA主干，只需替换对应的教师模型（如更换为更先进的几何感知模型）。

## 7. 总结
*   **核心思想**：通过3D教师模型对齐，将几何先验隐式融入VLA策略。
*   **速记版pipeline**：
    1. 获取目标物体的区域掩码；
    2. 用3D教师提取物体几何特征；
    3. 将几何特征对齐至VLA中间层；
    4. 训练策略同时学习动作与3D表示；
    5. 推理时舍弃教师和掩码。

**Key Findings:**

- We propose an object-centric 3D representation alignment framework built upon $π_0$, using SAM3D as a frozen 3D teacher to provide target-object 3D priors during training.
- Real-world experiments further demonstrate that our method is particularly effective in long-horizon manipulation scenarios where the robot must focus on different target objects across multiple subtasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25912v1)
- [arXiv](https://arxiv.org/abs/2607.25912v1)

---

<a id='2607.25798v1'></a>
## [Transformer Transformer: A Unified Model for Motion-Conditioned Robot Co-design](https://arxiv.org/abs/2607.25798v1)

**Authors:** Huy Ha, C. Karen Liu, Shuran Song

**Published:** 2026-07-28

**Categories:** cs.RO

**Abstract:**

An often overlooked factor of robot manipulation performance is the embodiment of the robot itself. Motivated by this problem, we study motion-conditioned robot co-design, where the goal is to generate complete robot designs that track target end-effector trajectories (from human demonstrations) while optimizing user-defined rewards. We introduce Transformer Transformer, a diffusion transformer trained on RoboTokens, a unified tokenization of robot embodiments, states, and actions. The same architecture can be used across embodiment spaces (e.g., wheeled bimanual, quadrupeds, humanoids) and use cases (embodiment generation, cross embodiment controller). Rather than overfitting to one reward function, Transformer Transformer is a dynamics model, whose reward-agnostic state and action predictions can be converted into reward-specific value predictions. These value predictions are used to steer embodiment diffusion towards high value robot designs, through a procedure we call Dynamics Self-Guidance. Experiments across multiple design spaces show zero-shot optimization of unseen rewards and trajectories, improving performance and runtime over the evolutionary baseline. Finally, we fabricated an optimized ALOHA design, which reduced tracking error by over 70% compared to the original design.

**Analysis:**

### 1. 摘要翻译
机器人操作性能往往忽视了一个关键因素——机器人本身的本体设计（embodiment）。针对此问题，本文研究了“运动条件下的机器人协同设计”，旨在生成能追踪人类演示的端点轨迹，同时优化用户定义奖励函数的完整机器人设计。我们引入了“Transformer Transformer”，这是一个基于RoboTokens（一种统一的机器人本体、状态和动作的标记化方案）训练的扩散Transformer。该架构可跨多种本体空间（如双臂轮式、四足、人形）及应用场景（设计生成、跨本体控制）通用。该模型通过一种称为“动力学自引导（Dynamics Self-Guidance）”的过程，将奖励无关的动力学预测转化为奖励相关的价值预测，从而引导本体扩散过程。实验表明，该模型在多种设计空间上实现了对未知奖励和轨迹的零样本优化，性能与运行效率均优于演化基线。最终，我们制造了一个优化的ALOHA机器人设计，其跟踪误差较原始设计降低了70%以上。

### 2. 方法动机分析
*   **驱动力**：现有的策略学习通常假设机器人本体是固定的，而忽视了本体设计对任务性能的决定性作用。作者旨在将“设计”与“控制”统一到一个可学习的框架中。
*   **现有方法痛点**：
    *   多数机器人设计优化依赖模拟器进行演化搜索，计算代价极其高昂。
    *   现有基于数据的生成方法通常过拟合于特定的任务或奖励函数。
    *   设计与控制往往是割裂的两个优化流程，缺乏统一的端到端学习与推理手段。
*   **研究假设**：如果能够将机器人本体、状态和动作统一表征为序列（RoboTokens），就可以利用扩散模型强大的生成与建模能力，通过统一的预测模型实现设计空间的探索和任务动力学的零样本优化。

### 3. 方法设计详解
*   **RoboToken表示**：将机器人拆解为五种本体token（链接、关节等）和状态/动作token。所有属性（几何、质量、惯性等）被投影为连续向量，实现异构机器人空间的同构化。
*   **Transformer Transformer架构**：
    *   采用DiT架构，通过mask建模方案，一个模型同时担任生成器、评估器（Critic）和控制器。
    *   **动力学自引导**：这是核心贡献。在推理时，模型预测动作的动力学，利用可微分的奖励函数梯度，通过反向传播指导扩散噪声的去除方向，从而直接“引导”模型生成高奖励的本体设计。
*   **流程pipeline**：
    1.  **输入**：目标末端执行器轨迹 + 奖励函数。
    2.  **并行扩散**：并行运行多个噪声采样过程，共同优化本体和动力学token。
    3.  **引导生成**：利用奖励函数的梯度或“零阶优化器”（Zeroth-Order Optimizer）对候选设计进行排序。
    4.  **验证控制**：将生成的最优本体传入同一个模型，模型直接预测动作序列以验证任务执行效果。

### 4. 方法对比分析
*   **本质区别**：本文将设计优化视为一个可学习的分布建模问题，而非单纯的仿真优化问题。它利用扩散模型的引导机制（Guidance），无需针对每种奖励重新训练模型。
*   **创新贡献**：
    *   提出了RoboToken统一表示，打通了不同类型机器人的设计空间。
    *   提出了“动力学自引导”，无需额外训练动力学模型即可实现零样本奖励优化。
*   **适用场景**：适用于需要自动探索机械结构（如连杆长度、关节数量、安装位置）以匹配特定动作任务的机器人协同设计场景。

### 5. 实验分析
*   **关键结论**：在固定臂、四足机器人、双臂移动操作三个空间中，模型表现出卓越的零样本泛化能力；在真实物理实验中，优化后的ALOHA设计在复杂洗碗任务中跟踪误差降低了73%。
*   **优势**：极高的采样效率（并行化推理比演化算法快几个数量级）；跨任务的通用性。
*   **局限**：对“训练分布外”的设计（如超出训练范围的过长连杆）泛化能力受限；对复杂几何形体的表示目前局限于原始几何体（Primitives）。

### 6. 实用指南
*   **开源**：项目主页为 [transformer-transformer.github.io](https://transformer-transformer.github.io)。
*   **关键实现**：必须做好“惯性分解”（Inertia Splitting）以维持物理一致性；使用DDIM采样器时，对于引导任务建议设置合适的η值（如1.0）以增强探索性。
*   **迁移建议**：若要迁移至新任务，只需定义新的奖励函数接口，并将新任务的机器人拓扑结构转化为RoboToken格式即可。

### 7. 总结
*   **核心思想**：统一表征与动力学引导，实现设计与控制协同生成。
*   **速记版pipeline**：
    1. 机器人结构序列化（RoboTokens）。
    2. 扩散模型生成候选设计方案。
    3. 利用奖励梯度反向引导采样。
    4. 筛选最优设计并执行验证。

**Key Findings:**

- We introduce Transformer Transformer, a diffusion transformer trained on RoboTokens, a unified tokenization of robot embodiments, states, and actions.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25798v1)
- [arXiv](https://arxiv.org/abs/2607.25798v1)

---

<a id='2607.25731v1'></a>
## [Tri-Manual Visuomotor Imitation Learning of Robot Policies](https://arxiv.org/abs/2607.25731v1)

**Authors:** James Zhao, Mingyuan Ba, Weiming Zhi

**Published:** 2026-07-28

**Categories:** cs.RO

**Abstract:**

Bimanual teleoperation provides an effective way to collect robot demonstrations, but it assumes that the operator and robot have matching numbers of simultaneous control channels. This assumption breaks for tri-manual systems: the robot can coordinate three arms concurrently, whereas a single operator can continuously control only two. Pairwise mode switching may therefore record otherwise independent motions sequentially, causing behaviour cloning to reproduce delays imposed by the interface rather than required by the task. We present TriManPolicy, a tri-manual imitation learning system that allows one operator to demonstrate behaviours for three arms. Its central component is Dependency-Aware Tri-Arm Scheduling (DATS). The key idea is to preserve the demonstrated arm motions while reconsidering when they occur. DATS retimes demonstrations offline by preserving local sensorimotor segments of fixed duration and repositioning them according to constraints on task order and arm usage that are reviewed by a human. The resulting data train a single synchronous policy for all three arms, while deployment requires neither the dependency graph nor the scheduler. Across six challenging tasks performed in the real world, policies trained on demonstrations retimed by DATS exhibit more efficient coordination while maintaining comparable observed task success. Offline analysis further shows that DATS changes the supervision across arms rather than merely removing idle periods.

**Analysis:**

### 1. 摘要翻译
双臂遥操作虽为机器人演示收集提供了有效途径，但通常假设操作者的实时控制通道数与机器人末端执行器数量相匹配。该假设在三臂系统中失效：机器人可同时协调三臂，而单操作者仅能控制两臂。这种“两臂控制三臂”的模式切换会导致原本独立的动作被强制串行化，使行为克隆模型不仅学习了任务逻辑，还复现了因接口限制产生的无效等待。本文提出 **TriManPolicy**，一种三臂模仿学习系统。其核心组件是“依赖感知三臂调度器”（DATS），它在保持演示动作局部时序完整性的前提下，根据任务依赖约束离线重构全局时序。该方法生成的重排数据训练出的同步策略，在执行时无需依赖图或调度器。实验表明，该方法在六项挑战性任务中实现了更高效的协同，且无需额外增加任务成功率的代价。

### 2. 方法动机分析
- **驱动力**：解决“操作接口的并行度”与“机器人本体的并行度”不匹配问题。
- **痛点**：传统的行为克隆（BC）会将操作者因切换模式产生的“停顿”视为任务的一部分，导致部署时机器人表现出不必要的延迟。
- **研究假设**：演示数据中的“局部动作（segment）”是有效的，但“全局时序（global timing）”是由受限的接口人为定义的。只要满足任务的依赖图约束，动作的时序可以重排。

### 3. 方法设计详解
- **流程总结**：
    1. **数据采集**：通过模式切换接口收集三臂任务，记录操作者对不同臂组（LR, LO, RO）的控制。
    2. **任务图构建**：利用VLM（视觉语言模型）辅助提取子任务序列、资源占用（Arms）及依赖关系（Predecessor relations）。
    3. **DATS调度（核心）**：将任务视作固定时长的片段，利用 CP-SAT 约束规划器进行重新排布。约束包括：① 必须遵循的依赖关系；② 任何时刻每只机械臂最多只能执行一个子任务。
    4. **重排数据训练**：将原始动作流映射到新时序，训练一个动作组块（Action-chunked）Transformer策略。
- **算法解释**：公式(8)是整个系统的灵魂。作者将Makespan（总工时）最小化作为目标函数，通过 `NoOverlap` 约束强迫机械臂在任务依赖允许的范围内尽可能同时工作。

### 4. 方法对比分析
- **本质区别**：不试图去拟合原始演示的时序，而是将“任务时序”作为一种可优化的约束问题，而非数据的固有属性。
- **创新贡献**：提出一种“数据驱动的调度重构”方案，将低效率的演示转化为高效的同步策略，摆脱了遥操作接口的物理限制。
- **适用场景**：适用于控制能力超过演示者控制接口维度的多臂机器人系统。

### 5. 实验分析
- **验证方法**：在六项真实世界的三臂 manipulation 任务中，对比“原始数据训练”与“DATS重排数据训练”的结果。
- **关键结果**：DATS策略在所有任务中均提升了执行效率（缩短了31.3%-49.9%的完成时间），且成功率持平或更高。
- **局限**：依赖于高质量的子任务图标注；若任务依赖关系设计不当，可能会产生不合逻辑的动作重排。

### 6. 实用指南
- **复现关键**：需要构建一个鲁棒的“子任务图（Subtask Graph）”。建议先利用简单的VLM工具标注关键节点，并使用 Google OR-Tools 的 CP-SAT solver 求解调度。
- **迁移建议**：该方法完全适用于其他受限于动作维度的领域，如四足机器人的多步态学习、工业装配的多工位调度。只需定义“任务片段的原子性”和“资源排他性约束”即可迁移。

### 7. 总结
- **核心思想**：通过离线约束调度，剥离演示数据的接口延迟，构建同步策略。
- **速记版pipeline**：
    1. 录制受限于切换模式的演示数据；
    2. 自动/手动构建子任务依赖图；
    3. 使用规划器重排片段以最小化总耗时；
    4. 使用重构后的数据训练同步控制策略。

**Key Findings:**

- We present TriManPolicy, a tri-manual imitation learning system that allows one operator to demonstrate behaviours for three arms.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25731v1)
- [arXiv](https://arxiv.org/abs/2607.25731v1)

---

<a id='2607.25642v1'></a>
## [Instruction-based Image Editing: A Survey on Data, Models, Evaluation, and Applications](https://arxiv.org/abs/2607.25642v1)

**Authors:** Xianghao Zang, Zijian Jiang, Jiarong Cheng, Qianrui Teng, Ying He, Yuxuan Mu, Chao Ban, Huayu Zhang, Lanxiang Zhou, Zerun Feng, Chi Zhang

**Published:** 2026-07-28

**Categories:** cs.CV, cs.CL

**Abstract:**

Instruction-based Image Editing (IIE) aims to transform a given image into a new one based on textual instructions. Advances in Large Language Models (LLMs) and Vision-Language Models (VLMs) have accelerated progress toward practical ``one-sentence image editing" systems. This survey presents a systematic taxonomy and comprehensive review of IIE research, structured around five core dimensions: (1) task definition and hierarchical categorization of editing operations, (2) methodologies for training data construction, (3) architectural evolution from GAN-based to diffusion and autoregressive paradigms, (4) standardized evaluation metrics and benchmark development, and (5) introduction of commercial solutions. Our analysis shows critical technological milestones across model generations. We further propose a Comprehensive, in-Depth, and Diagnostic benchmark for IIE task (CDD-IIE Bench), which can rigorously assess the multiple aspects of model performance. Through empirical comparisons of open-source solutions, we highlight their respective capabilities and limitations. Finally, we discuss future research directions to advance the field.

**Analysis:**

以下是对论文《Instruction-based Image Editing: A Survey on Data, Models, Evaluation, and Applications》的方法论分析与总结：

### 1. 摘要翻译
指令驱动图像编辑（IIE）旨在根据文本指令将源图像转换为新图像。大型语言模型（LLMs）和视觉-语言模型（VLMs）的进步加速了通向“一句话图像编辑”系统的进程。本综述提出了一个系统性的分类法，并围绕五个核心维度对IIE研究进行了全面综述：（1）任务定义与编辑操作的层级分类，（2）训练数据构建方法，（3）从GAN到扩散模型及自回归范式的架构演进，（4）标准化评估指标与基准测试的发展，（5）商业化方案介绍。分析揭示了各模型世代的关键技术里程碑。我们进一步提出了针对IIE任务的全面、深入且具诊断性的基准（CDD-IIE Bench），能够严谨评估模型性能。通过对开源方案的实证对比，我们突出了各自的能力与局限，并探讨了未来的研究方向。

### 2. 方法动机分析
- **驱动力**：解决图像编辑任务缺乏统一的定义、评价标准及针对复杂推理能力评估的空白。
- **现有痛点**：当前研究多局限于单步操作，缺乏对多步推理、空间理解和组合指令的评估；同时，传统的指标（如L1, CLIP-score）无法捕获编辑的语义细微差别，与人类主观感知存在严重偏差。
- **核心假设**：通过建立细粒度的层级任务分类与诊断性基准（CDD-IIE），能够从本质上发现模型在处理复杂语义和空间理解方面的瓶颈，进而推动“通用型”编辑模型的发展。

### 3. 方法设计详解（CDD-IIE Bench）
CDD-IIE Bench的设计并非简单的任务堆砌，而是构建了严谨的诊断体系：
- **层级架构**：将任务划分为“基本原子编辑 suite”和“高级组合编辑 suite”。
  - **原子操作**：覆盖对象添加、移除、属性修改、风格迁移等21个子任务，作为构建复杂编辑的“积木”。
  - **组合推理**：侧重并行指令、序列指令、空间调整（如位置、计数、尺寸）及隐式推理，旨在探测模型处理复杂逻辑的能力边界。
- **诊断逻辑**：采用了基于GPT-4o的评估架构，将评估维度解构为“指令遵循（Sadh）”、“编辑质量（Squa）”和“细节保留（Spres）”。这种三维评分机制允许研究人员精准识别模型失败的具体环节（例如，是因为指令没读懂，还是因为破坏了背景）。

### 4. 方法对比分析
- **本质区别**：与仅关注单一指标（如CLIP相关性）的基准不同，CDD-IIE引入了“诊断性”视角，通过专业的人类专家与VLM协同打分，明确区分了模型在“原子级”与“组合级”任务上的性能断层。
- **创新贡献**：首次系统化梳理了从GAN、扩散模型到自回归架构在IIE任务中的信号整合策略，并针对性地指出了当前模型在空间推理上的薄弱。

### 5. 实验分析
- **关键结论**：目前的顶尖模型（如Qwen-Image-Edit-2509）在原子编辑任务上表现优异，但在“空间理解与推理”任务上显著下降，证实了模型逻辑推理能力的匮乏。
- **局限性**：模型普遍缺乏健壮性，表现为在图像语义编辑上性能领先，但在低级的图像修复（Image Repair）任务上表现出显著的 bias，未能实现真正的多任务均衡。

### 6. 实用指南
- **开源/复现**：该基准数据及评价协议将开源。复现的关键在于构建符合IIE层级定义的1,353个对比样本。
- **迁移注意**：在评估自身模型时，需重点关注“细节保留”维度，防止模型为了提升指令匹配度而过度编辑（over-editing），导致源图像结构丢失。

### 7. 总结
- **核心思想**：通过分层诊断基准，揭示模型在组合推理与原子操作间的性能鸿沟，推动通用编辑范式。
- **速记版pipeline**：
  1. 定义21类原子及组合编辑任务。
  2. 精心策划包含1,353组图像-指令的基准样本。
  3. 引入三维诊断评分体系（遵循度、质量、保留度）。
  4. 利用LLM/VLM与专家人工进行多维度综合打分。
  5. 输出模型性能诊断报告，明确改进方向。

**Key Findings:**

- Instruction-based Image Editing (IIE) aims to transform a given image into a new one based on textual instructions.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.25642v1)
- [arXiv](https://arxiv.org/abs/2607.25642v1)

---

