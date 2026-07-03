time: 20260703

# Arxiv Computer Vision Papers - 2026-07-03

## Executive Summary

## 每日论文执行摘要（2026-07-02）

本日收录的10篇计算机视觉论文主要集中于**具身智能/机器人学习**（6篇）与**视觉生成/定位/鲁棒性**（4篇），反映了领域内对“从感知到行动”闭环系统的持续关注，以及数据效率与泛化能力的核心诉求。

### 一、主要主题与趋势

1. **具身AI的规模化与实用化**：多篇论文探索如何让视觉-语言-动作（VLA）模型在真实机器人上高效部署与泛化。例如，Embodied.cpp 提供了异构机器人上的轻量推理运行时，HEFT 实现了重型全尺寸人形机器人的遥操作，WorldSample 则结合世界模型进行闭环真实机器人强化学习。
2. **数据效率与任务无关预训练**：多篇工作强调**先学习通用运动先验**再学习具体任务（如“Learning to Move Before Learning to Do”），或通过混合动态数据增强空间泛化（The Moving Eye），以及利用自监督/数据增强替代昂贵标注（From SRA to Self-Flow）。
3. **多模态融合与鲁棒性**：视觉-触觉融合（VT-WAM）成为接触丰富操作的关键方案；同时，针对文本攻击的鲁棒性研究（Towards Robustness）展示了无训练概念定位的可行性。
4. **生成模型的加速**：表示分布匹配（Representation Distribution Matching）提出单步视觉生成方法，显著提升采样效率。

### 二、特别重要的创新性论文

- **VT-WAM（Visual-Tactile World Action Model）**：首次将视觉与触觉联合建模为世界动作模型，用于接触丰富的操作任务，填补了精细操控中触觉反馈缺失的空白。
- **WorldSample（闭环真实机器人RL）**：在真实机器人上实现基于世界模型的强化学习闭环，避免仿真到真实的迁移鸿沟，是具身智能落地的重要一步。
- **Representation Distribution Matching**：提出无需迭代采样的单步图像生成范式，有望大幅提升扩散模型的推理速度，对实时应用意义重大。
- **HEFT（重型全尺寸人形遥操作）**：解决了大载荷下人形机器人的运动引导与安全遥操作问题，为工业级应用提供了可行框架。

### 三、新兴研究方向与技术

- **无训练概念定位防御**：不依赖对抗训练即可定位并修正模型对文本攻击的脆弱点，为鲁棒性研究开辟了新路径（第4篇）。
- **描述符无关的全局视觉定位**：GeoMix 利用全局上下文与多检测器训练，绕过了传统描述子匹配的瓶颈，有望提升长期视觉定位的鲁棒性。
- **任务无关的VLA预训练**：先学习运动基元再学习任务具体策略（第6篇），借鉴了生物运动发育过程，可能成为VLA训练的范式转变。
- **触觉+视觉的世界模型**：VT-WAM 提出的多模态世界动作模型将推动机器人更精细的操作能力。

### 四、建议全文阅读的论文

1. **VT-WAM**（第2篇）——若从事机器人操控或多模态融合研究，此篇提供全新框架。
2. **WorldSample**（第7篇）——对真实机器人强化学习感兴趣者的必读文献，展示了闭环方案的可行性。
3. **Representation Distribution Matching**（第8篇）——关注生成模型效率的研究者不应错过，可能推动即时视觉生成应用。
4. **The Moving Eye**（第10篇）与 **Learning to Move Before Learning to Do**（第6篇）——分别从数据采集和预训练角度提升VLA泛化，互为补充。

以上摘要旨在帮助研究人员快速把握当日最具影响力的进展，建议根据自身方向选择性深入阅读。

---

## Table of Contents

1. [From SRA to Self-Flow: Data Augmentation or Self-Supervision?](#2607.02508v1)
2. [VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation](#2607.02503v1)
3. [Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots](#2607.02501v1)
4. [Towards Robustness against Typographic Attack with Training-free Concept Localization](#2607.02494v1)
5. [GeoMix: Descriptor-Free Visual Localization via Global Context and Multi-Detector Training](#2607.02486v1)
6. [Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs](#2607.02466v1)
7. [WorldSample: Closed-loop Real-robot RL with World Modelling](#2607.02431v1)
8. [Representation Distribution Matching for One-Step Visual Generation](#2607.02375v1)
9. [HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum](#2607.02332v1)
10. [The Moving Eye: Enhancing VLA Spatial Generalization via Hybrid Dynamic Data Collection](#2607.02322v1)

---

## Papers

<a id='2607.02508v1'></a>
## [From SRA to Self-Flow: Data Augmentation or Self-Supervision?](https://arxiv.org/abs/2607.02508v1)

**Authors:** Dengyang Jiang, Mengmeng Wang, Harry Yang, Jingdong Wang

**Published:** 2026-07-02

**Categories:** cs.CV

**Abstract:**

Representation alignment has become an effective way to accelerate diffusion transformer training and improve generation quality. Recent self-alignment methods, such as SRA and Self-Flow, further remove the dependency on external pretrained encoders by constructing alignment within the diffusion model itself. However, the mechanism behind the improvement from SRA to Self-Flow, dual-time scheduling, remains under-examined: Self-Flow attributes its gain to interactions between tokens at different noise levels, where cleaner tokens help infer noisier ones. In this work, we revisit this explanation and ask whether the gain instead comes from data augmentation along the noise dimension. To disentangle these factors, we introduce Attention Separation, which preserves the same dual-timestep input as Self-Flow while blocking attention between tokens assigned to different noise levels. Surprisingly, removing such interaction does not degrade performance and can even improve it, suggesting that the improvement from SRA to Self-Flow mainly comes from data augmentation. Furthermore,We show that Attention Separation itself provides an augmentation effect by splitting a single image into multiple effective training parts to expand the training data. Based on these observations, we combine self-representation alignment with dual-timestep and attention-separation augmentation, and demonstrate the effectiveness of this design on ImageNet.

**Analysis:**

### 1. 摘要翻译
表示对齐已成为加速扩散模型（DiT）训练及提升生成质量的有效途径。近期的自我对齐方法（如SRA和Self-Flow）通过在扩散模型内部构建对齐，消除了对外部预训练编码器的依赖。然而，Self-Flow在SRA基础上的改进机制——“双时间步调度”——的本质尚不明确：Self-Flow将其归因于不同噪声水平标记（token）间的交互。本研究重新审视了这一解释，提出该增益实际上源于噪声维度的“数据增强”。为解耦这些因素，我们引入了“注意力分离（Attention Separation）”机制，在保留双时间步输入的同时，阻断不同噪声水平标记间的注意力交互。实验表明，移除交互不仅不会损害性能，反而能进一步提升效果，证实了改进主要源于数据增强。此外，我们证明了注意力分离本身通过将单张图像切分为多个有效训练片段，发挥了数据增强的作用。基于此，我们将双时间步调度与注意力分离结合，在ImageNet上验证了该设计的有效性。

### 2. 方法动机分析
- **驱动力**：作者质疑现有方法（如Self-Flow）将改进单纯归因于“ cleaner tokens 帮助 noisy tokens 推理”的自我监督机制，试图探究其性能提升的根本原因。
- **现有方法痛点**：SRA和Self-Flow虽然有效，但对其改进机制的理解存在偏差，导致难以进一步优化训练范式。
- **研究假设**：双时间步调度并非增强了 token 间的交互，而是通过引入多样的噪声状态，实质上扩展了模型在训练中可见的噪声分布，起到了一种“数据增强”的效果。

### 3. 方法设计详解
- **核心逻辑**：通过控制变量法，解耦交互与增强。
- **流程pipeline**：
    1. **双时间步调度**：给定输入，将标记划分为高噪声水平（$t_1$）和低噪声水平（$t_2$）两组，输入同一个模型。
    2. **注意力分离（核心创新）**：修改自注意力计算中的掩码 $A^{sep}_{ij}$。对于第 $i$ 个 token，只有当它与第 $j$ 个 token 属于同一时间步组时，才允许计算注意力权重；跨组交互则被强制设为 $-\infty$。
    3. **多视角训练**：该操作将单张图像拆解为多个部分观察视图，模型在处理这些视图时，共享参数且同时优化去噪目标，变相增加了训练样本的多样性。
- **算法解释**：公式 (9) 通过引入 block-diagonal 掩码，将全局注意力矩阵转化为块对角矩阵。这不仅验证了交互不是性能提升的必要条件，还通过强制模型从局部视图恢复整体信息，增强了鲁棒性。

### 4. 方法对比分析
- **本质区别**：从传统的“依赖跨 token 交互提供监督”转向“视双时间步为数据增强手段”。
- **创新贡献**：提出注意力分离操作，不仅作为诊断工具验证了假设，还自身演化为一种简单且有效的图像分区数据增强策略。
- **适用场景**：适用于任何基于 DiT 的生成模型训练，尤其是当计算资源受限或需要进一步挖掘现有数据潜力时。

### 5. 实验分析
- **验证方法**：在 ImageNet 256×256 和 512×512 上，对比全注意力与注意力分离机制在相同噪声策略下的 FID/IS 表现。
- **关键结果**：在阻断跨噪声交互后，模型 FID 反而从 25.19 提升至 25.06（800K步），证明了交互并非必须。
- **主要优势**：无需引入额外计算开销，通过结构化注意力掩码实现了数据增强，性能优于 SRA 和 Self-Flow。
- **主要局限**：在较大的 mask ratio（如 0.5）下，可能会引入训练与推理阶段（全注意力）的模式不匹配，需混合全图样本训练来缓解。

### 6. 实用指南
- **开源情况**：代码已开源（github.com/vvvvvjdy/SRA）。
- **实现细节**：关键参数是 mask ratio ($\alpha$)，默认推荐 0.25。若采用较大掩码比例，建议在训练中混合 25% 的常规全图样本以对齐推理分布。
- **迁移可能**：可直接迁移至所有 Transformer 架构的扩散模型（如 DiT、SiT），作为一种通用的训练级数据增强手段。

### 7. 总结
- **核心思想**：利用注意力分离将噪声调度转化为高效的数据增强手段。
- **速记版pipeline**：
    1. 将图像切分为不同噪声水平的标记组。
    2. 对标记组应用注意力掩码，只允许同组交互。
    3. 共享参数同步训练所有标记组。
    4. 混合少量全图样本以平衡推理性能。

**Key Findings:**

- To disentangle these factors, we introduce Attention Separation, which preserves the same dual-timestep input as Self-Flow while blocking attention between tokens assigned to different noise levels.
- Furthermore,We show that Attention Separation itself provides an augmentation effect by splitting a single image into multiple effective training parts to expand the training data.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02508v1)
- [arXiv](https://arxiv.org/abs/2607.02508v1)

---

<a id='2607.02503v1'></a>
## [VT-WAM: Visual-Tactile World Action Model for Contact-Rich Manipulation](https://arxiv.org/abs/2607.02503v1)

**Authors:** Shuai Tian, Yupeng Zheng, Yuhang Zheng, Songen Gu, Yujie Zang, Yuxing Qin, Weize Li, Haoran Li, Wenchao Ding, Dongbin Zhao

**Published:** 2026-07-02

**Categories:** cs.RO

**Abstract:**

Contact-rich manipulation requires policies to react to local deformation, pressure, slip, and friction, yet these cues are temporally sparse and often invisible in visual observations. Existing visual-tactile policies usually feed tactile observations directly into action prediction, but rarely model tactile deformation dynamics during action generation. In this paper, we introduce VT-WAM, a Visual-Tactile World Action Model that jointly learns future visual prediction, tactile deformation prediction, and action prediction within a unified flow matching framework. In particular, VT-WAM introduces (1) Asymmetric Mixture-of-Transformers (MoT) attention to bridge a first-frame visual anchor with temporal tactile dynamics, and (2) contact-gated Action-Visual-Tactile Attention Guidance (AVTAG) to encourage action queries to rely on tactile evidence during contact phases. Across six real-world contact-rich manipulation tasks, VT-WAM achieves a 71.67% average success rate, outperforming Fast-WAM by 26.67% and OmniVTLA by 35.84%. Ablations demonstrate that modeling tactile deformation dynamics and guiding contact-phase tactile attention are both important for contact-rich tasks. Project website: https://vt-wam.github.io/.

**Analysis:**

### 1. 摘要翻译
接触丰富的操作需要策略能对局部形变、压力、滑动和摩擦做出反应，但这些线索在视觉观察中往往是时间稀疏且不可见的。现有的视觉-触觉策略通常将触觉观测直接馈入动作预测，却很少在动作生成过程中对触觉形变动力学进行建模。本文提出了 **VT-WAM**，一种视觉-触觉世界动作模型，在统一的流匹配框架内联合学习未来视觉预测、触觉形变预测和动作预测。具体而言，VT-WAM 引入了：(1) **非对称 Mixture-of-Transformers (MoT) 注意力机制**，将第一帧视觉锚点与时间触觉动力学关联；(2) **接触门控动作-视觉-触觉注意力引导 (AVTAG)**，鼓励动作查询在接触阶段依赖触觉证据。在六项现实世界接触丰富操作任务中，VT-WAM 平均成功率达到 71.67%，较 Fast-WAM 提升了 26.67%，较 OmniVTLA 提升了 35.84%。消融实验证明了对触觉形变动力学建模和对接触阶段触觉注意力进行引导的重要性。

### 2. 方法动机分析
*   **驱动力**：在接触丰富的机器人操作中，视觉信息在物体接触发生瞬间往往因遮挡或微小形变而变得不可靠，而触觉信号虽然在接触阶段至关重要，但其本身是稀疏且瞬时的。作者希望建立一种能将触觉变化显式地转化为动作预测动力学的模型。
*   **现有方法痛点**：现有方法（如 VLA 变体）要么将触觉仅仅作为额外的辅助输入，未能充分利用其动力学演变特性；要么在执行时仍依赖全序列视觉预测，导致推理延迟高且无法聚焦于接触瞬间的微小细节。
*   **研究假设**：通过将触觉形变建模为世界动力学的一部分，并在训练中强制动作预测关注触觉变化，能够显著提升模型在复杂环境下的细微纠偏能力。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **模态编码**：Wrist 相机视觉序列编码为 $X_v$（分为首帧锚点和未来帧），触觉形变序列编码为 $X_t$，动作片段为 $X_a$。
    2.  **非对称 MoT 注意力**：这是核心结构。它通过 Mask 机制实现非对称读取：动作查询 $X_a$ 能够读取触觉全序列（捕捉动力学）和第一帧视觉锚点（提供场景上下文），但屏蔽掉未来的视觉预测，从而在推理时实现“视觉缓存”模式（无需未来视觉预测）。
    3.  **接触门控 AVTAG**：引入训练阶段专有的 hinge ranking loss，当触觉反馈显著（接触阶段）时，如果动作模型对视觉的注意力权重超过了对触觉的权重，就会受到惩罚。这强制模型在接触发生时必须“听从”触觉指令。
    4.  **联合训练**：采用流匹配（Flow Matching）目标，同时优化视觉预测、触觉预测和动作预测三个分支。
*   **关键公式**：$L_{AVTAG} = \mathbb{E}_{r\in C} [\max(0, p_v(r) - p_t(r))]$，其中 $p_v$ 和 $p_t$ 分别为视觉和触觉的注意力权重，该公式直接抑制了动作查询的“视觉偏见”。

### 4. 方法对比分析
*   **本质区别**：VT-WAM 不仅仅将触觉作为一种输入特征，而是将其建模为一种**动态演化过程**，并在推理时通过“视觉锚点+触觉动力学”的混合模式，兼顾了全局场景一致性和局部接触纠偏。
*   **创新贡献**：提出“非对称”Transformer 注意力架构，巧妙解决了多模态融合中“视觉主导”导致触觉信息被淹没的问题。
*   **适用场景**：高精度、接触依赖的机器人任务（如插拔、擦拭、精密装配）。

### 5. 实验分析（精简版）
*   **关键结果**：在插入类任务（如插管）中，相比 OmniVTLA（38%），VT-WAM 的性能接近翻倍（61%+），证明了该方法在解决小物体对齐时的优越性。
*   **主要优势**：推理效率高（省去了未来视觉预测），在接触阶段鲁棒性极强。
*   **主要局限**：任务间的泛化能力未在文中大规模讨论，目前侧重于单个任务的深度精调。

### 6. 实用指南
*   **开源**：访问项目官网 https://vt-wam.github.io/。
*   **实现细节**：训练时需同步处理视觉、触觉和 proprioceptive 数据，采样率对齐至 30Hz；AVTAG 的 loss 权重 $\lambda_{AVTAG}$ 设为 0.05 即可平衡多目标。
*   **迁移建议**：如果你的任务涉及频繁接触（如软体操作），可直接套用 AVTAG 损失函数，它是纠正神经网络“视觉依赖”最有效的方法。

### 7. 总结
*   **核心思想**：通过非对称建模触觉动力学与接触引导，实现触觉驱动的精密操作。
*   **速记版 pipeline**：
    1. 分解视觉为首帧上下文和未来预测。
    2. 用非对称注意力让动作分支只看“第一眼视觉”和“持续触觉”。
    3. 用接触感知损失强迫模型在接触瞬间“多看触觉、少看视觉”。
    4. 训练联合流模型，推理时只运行触觉分支，实现低延迟高精度动作输出。

**Key Findings:**

- In this paper, we introduce VT-WAM, a Visual-Tactile World Action Model that jointly learns future visual prediction, tactile deformation prediction, and action prediction within a unified flow matching framework.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02503v1)
- [arXiv](https://arxiv.org/abs/2607.02503v1)

---

<a id='2607.02501v1'></a>
## [Embodied.cpp: A Portable Inference Runtime of Embodied AI Models on Heterogeneous Robots](https://arxiv.org/abs/2607.02501v1)

**Authors:** Ling Xu, Chuyu Han, Borui Li, Hao Wu, Shiqi Jiang, Ting Cao, Chuanyou Li, Sheng Zhong, Shuai Wang

**Published:** 2026-07-02

**Categories:** cs.RO, cs.CV, cs.OS

**Abstract:**

Embodied AI models now span vision-language-action (VLA) models and world-action models (WAMs), but practical deployment remains fragmented across model-specific Python stacks, backend assumptions, and robot-side glue code, especially on heterogeneous edge devices. Existing inference runtimes are designed mainly for request-response serving and therefore do not satisfy the runtime contract of embodied deployment: multi-rate execution inside closed-loop control, latency-first batch-1 inference on heterogeneous hardware, and extensible embodied interfaces beyond fixed token I/O. We present Embodied.cpp, a portable C++ inference runtime for embodied models. Based on an architectural analysis of representative VLA models and WAMs, Embodied.cpp captures a shared execution path and organizes it into five layers: input adapters, sequence builders, backbone execution, head plugins, and deployment adapters. The runtime provides modular multi-rate execution, latency-first fused inference, and extensible operator and I/O support, enabling deployment across heterogeneous devices, robots, and simulators through one backend abstraction. We evaluate Embodied.cpp on two VLA models, HY-VLA and pi0.5, and on a preliminary WAM benchmark using a LingBot-VA Transformer block. The VLA deployments achieve successful closed-loop execution with 100.0% and 91.0% task success rates, respectively. The WAM benchmark reduces block memory from 312.2 MiB to 88.1 MiB. These results show that Embodied.cpp improves deployment efficiency while preserving high accuracy across diverse embodied model architectures.

**Analysis:**

### 1. 摘要翻译
具身智能模型目前已涵盖视觉-语言-动作（VLA）模型和世界-动作模型（WAMs），但实际部署仍碎片化地分散在特定模型的Python栈、后端假设和机器人端“胶水代码”中，在异构边缘设备上尤为严重。现有推理运行时多为请求-响应式服务设计，无法满足具身部署的运行合同：即闭环控制内的多速率执行、异构硬件上的低延迟Batch-1推理，以及超出固定Token I/O的扩展具身接口。我们提出了 **Embodied.cpp**，这是一个用于具身模型的便携式C++推理运行时。基于对典型VLA和WAM模型的架构分析，Embodied.cpp 捕获了共享执行路径并将其组织为五个层级：输入适配器、序列构建器、骨干网执行、头部插件和部署适配器。该运行时提供模块化多速率执行、延迟优先的融合推理以及可扩展的算子和I/O支持，通过单一后端抽象实现了跨异构设备、机器人和模拟器的部署。我们在两种VLA模型（HY-VLA和pi0.5）及一个WAM基准测试上进行了评估。VLA部署分别达到了100.0%和91.0%的任务成功率，WAM基准测试将块内存占用从312.2 MiB降低至88.1 MiB。这些结果表明，Embodied.cpp在保持跨不同具身架构高精度的同时，显著提升了部署效率。

---

### 2. 方法动机分析
*   **驱动力**：旨在为具身AI模型构建一个统一的、跨平台的、满足闭环实时控制需求的高效运行时。
*   **现有方法痛点**：当前运行时（如llama.cpp, vLLM）面向云端“请求-响应”模式，无法适配具身智能特有的闭环控制，缺乏对多速率执行、异构硬件实时同步的支持，且部署时需大量定制化胶水代码。
*   **研究假设**：通过架构解耦（将模型拆分为共享基础骨干与特定的头部/插件），可以构建一个既有统一核心又具备高度扩展性的基础设施，满足不同具身模型（VLA/WAM）在资源受限边缘设备上的高效运行。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入层（Input Adapters）**：统一传感器流（相机、力传感器、IMU）及数据集格式。
    2.  **序列构建器（Sequence Builders）**：处理跨模态输入，构建模型执行所需的序列特征。
    3.  **骨干网执行（Backbone Execution）**：运行模型核心的Transformer块，利用后端抽象在CPU/GPU/NPU上执行。
    4.  **头部插件（Head Plugins）**：针对预测任务（动作生成、世界预测）的特定头层。
    5.  **部署适配器（Deployment Adapters）**：将推理输出适配至机器人底层控制器（如ROS, Apollo）。
*   **模型结构**：采用了“五层架构”，核心亮点是**模块化**。它将模型逻辑拆分为稳定的共享部分（骨干）和易变的任务部分（插件），使得更换模型架构无需重写底层运行时。
*   **算法解释**：核心逻辑在于**多速率执行**策略，它允许感知部分（低频）和动作控制部分（高频）在同一运行时内解耦运行，避免了传统同步推理产生的控制延迟。

---

### 4. 方法对比分析
*   **本质区别**：从“基于Token的服务”转向“基于闭环控制的实时推理”。
*   **创新贡献**：首次提出了涵盖VLA与WAM的统一运行时架构；引入了针对边缘设备优化的Batch-1融合执行技术。
*   **适用场景**：机器人控制、实时模拟器交互、边缘AI计算集群。

---

### 5. 实验分析（精简版）
*   **验证方法**：通过HY-VLA与pi0.5在机器人任务中的闭环成功率，以及LingBot-VA的内存优化效果进行验证。
*   **关键结果**：在保证精度前提下，内存占用减少约70%，成功实现复杂闭环控制。
*   **优势**：极佳的跨平台便携性（C++实现）和闭环部署能力。
*   **局限**：目前对于完全复杂的WAM闭环部署尚在实验阶段，主要基于微基准测试。

---

### 6. 实用指南
*   **开源情况**：已开源，项目地址：[https://github.com/SEU-PAISys/Embodied.cpp](https://github.com/SEU-PAISys/Embodied.cpp)
*   **实现细节**：关键在于模型权重的量化处理（如GGUF Q4_K），以及针对目标板卡（如Jetson）的算子融合配置。
*   **迁移可能**：非常适合迁移到新的具身模型，只需实现相应的`Head Plugin`和`Input Adapter`即可接入框架。

---

### 7. 总结
*   **核心思想**：具身推理需将“共享骨干”与“差异化插件”解耦，以适配实时闭环控制。
*   **速记版pipeline**：
    1. 统一接入传感器数据；
    2. 将模型逻辑拆分为基础算子和任务插件；
    3. 根据不同刷新频率调度感知与动作模块；
    4. 将推理结果无缝对接至硬件控制器。

**Key Findings:**

- We present Embodied.cpp, a portable C++ inference runtime for embodied models.
- These results show that Embodied.cpp improves deployment efficiency while preserving high accuracy across diverse embodied model architectures.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02501v1)
- [arXiv](https://arxiv.org/abs/2607.02501v1)

---

<a id='2607.02494v1'></a>
## [Towards Robustness against Typographic Attack with Training-free Concept Localization](https://arxiv.org/abs/2607.02494v1)

**Authors:** Bohan Liu, Wenqian Ye, Guangzhi Xiong, Zhenghao He, Sanchit Sinha, Aidong Zhang

**Published:** 2026-07-02

**Categories:** cs.CV, cs.CL

**Abstract:**

Models trained via Contrastive Language-Image Pretraining (CLIP) serve as the foundational vision encoders for most modern Large Vision Language Models (LVLMs). Despite their widespread adoption, CLIP models exhibit a critical yet underexplored failure mode: irrelevant text appearing within images confounds visual representations, biasing them toward lexical meaning rather than true visual semantics. This robustness issue, commonly described as a Typographic Attack (TA), exposes a vulnerability that poses a significant risk to safety-critical applications such as autonomous driving. To achieve interpretable and effective robustness against TA, we propose a novel, training-free mechanistic interpretability method. Our method provides sampling-based interpretations of hidden state representations and quantitatively attributes semantic versus lexical focus to individual attention heads. Through probabilistic analysis and circuit mining, we isolate specific Vision Transformer (ViT) components that disproportionately encode lexical information, thereby identifying the mechanistic source of TA. We further show that simple interventions applied directly to the identified circuits, without any additional training, can substantially improve robustness against Typographic Attacks in object classification. These interventions, such as selective adjustment of attention weights, also outperform both supervised and training-free defense methods. Our experiments demonstrate that applying the proposed intervention to the vision encoders of several state-of-the-art LVLMs yields substantial gains in Visual Question Answering accuracy under Typographic Attack interference on RIO-Bench. These results confirm both the efficacy and the generalizability of our mechanistic approach. Code is released at https://github.com/Liu-524/SamplingTAR.

**Analysis:**

### 1. 摘要翻译
由对比语言-图像预训练（CLIP）模型驱动的大型视觉语言模型（LVLMs）尽管得到了广泛应用，但存在一个关键却未被充分探索的失效模式：图像中的无关文本会混淆视觉表示，使模型偏向于词汇语义而非真实的视觉语义。这种稳健性问题被称为“印刷攻击（Typographic Attack, TA）”，对自动驾驶等安全关键应用构成了重大风险。为了实现对TA的可解释且有效的稳健性，我们提出了一种新颖的、无训练的机械可解释性方法。该方法提供了隐藏状态表示的基于采样的解释，并定量地将语义与词汇焦点归因于各个注意力头。通过概率分析和电路挖掘，我们隔离了特定视觉Transformer（ViT）中不成比例地编码词汇信息的组件，从而识别了TA的机械来源。我们进一步表明，直接应用于所识别电路的简单干预措施（无需额外训练）可以显著提高对象分类中对印刷攻击的稳健性。这些干预措施，如选择性调整注意力权重，优于现有的监督和无训练防御方法。我们的实验还证明，将该方法应用于多个最先进LVLMs的视觉编码器，在RIO-Bench的印刷攻击干扰下，能显著提升视觉问答（VQA）的准确性。这些结果证实了我们机械方法在有效性和通用性上的优势。

---

### 2. 方法动机分析
- **驱动力**：旨在解决LVLMs中CLIP视觉编码器被“印刷攻击”（即图像中的文本干扰语义）误导的稳健性问题。
- **痛点**：现有方法（如Prefix tuning）通常依赖有监督学习或昂贵的参数化字典学习，计算成本高且黑盒属性强，缺乏对模型内部“印刷阅读电路”的精确解构。
- **研究假设**：基于“线性表示假设（Linear Representation Hypothesis）”，作者认为神经网络的隐空间是多个概念向量的线性叠加，因此可以通过在隐空间进行随机采样，利用注意力头的自然分解特性来定位引起词汇偏见的“特定电路”。

---

### 3. 方法设计详解
- **核心Pipeline**：
  1. **随机采样（Stochastic Lottery）**：在MHSA（多头自注意力）子空间中采样随机向量，将其作为“概念探针”，利用线性探测原理筛选出与词汇干扰相关的向量。
  2. **信息流归因（Information Flow Attribution）**：利用梯度归因方法，将特定patch的注意力logit与伪概念向量关联，通过计算“词汇归因分数（nTAS）”量化不同注意力头对文本的敏感度。
  3. **机械干预（Mechanistic Intervention）**：一旦识别出高文本敏感度的注意力头（即有害电路），在推理阶段执行“注意力重加权（Reweighting）”或“零消融（Zero Ablation）”来屏蔽这些头的影响。
- **算法精要**：归因公式（公式5）通过计算logit的偏导，精确地将注意力激活与概念对齐结合，确保归因结果仅反映词汇特征而非通用语义。

---

### 4. 方法对比分析
- **本质区别**：无需训练，不依赖人工标注标签，直接通过模型内部的数学属性进行“电路挖掘”，且干预发生在推理阶段。
- **创新贡献**：将“Lottery Ticket Hypothesis”与“线性表示假设”结合，首次提出在MHSA bottleneck处进行无监督概念挖掘，极大地降低了计算复杂度。
- **适用场景**：适用于任何基于ViT架构的视觉编码器，尤其是对于那些对安全性要求极高的多模态部署场景。

---

### 5. 实验分析
- **验证方法**：在RTA-100、Disentangling、PAINT等攻击数据集上进行分类实验，并在RIO-Bench上对LVLMs进行VQA测试。
- **关键结果**：在不损失通用性能的前提下（精度波动<1%），显著降低了Text Confusion Rate（文本混淆率），在多个ViT变体上提升了6%以上的准确率。
- **优势/局限**：优势在于极快（单次提取<1分钟，零推理开销）；局限在于依赖于对“印刷电路”分布的假设，极端分布外数据可能导致部分失效。

---

### 6. 实用指南
- **开源地址**：https://github.com/Liu-524/SamplingTAR
- **关键细节**：
  - **超参数**：推荐设置扩展比（expansion ratio）为16，在该配置下模型趋于稳定。
  - **迁移建议**：对于没有 `<cls>` token的LVLM，可直接使用第一个视觉token作为近似的全局状态表示。
  - **实现逻辑**：重点在于计算梯度的ReLU投影，以滤除负贡献，确保定位准确。

---

### 7. 总结
- **核心思想**：通过随机采样挖掘视觉Transformer中的词汇敏感电路并进行推理阶段干预。
- **速记版Pipeline**：
  1. 在隐空间随机采样概念向量。
  2. 计算归因分数定位有害注意力头。
  3. 推理时抑制或屏蔽有害头输出。
  4. 恢复纯视觉语义处理能力。

**Key Findings:**

- To achieve interpretable and effective robustness against TA, we propose a novel, training-free mechanistic interpretability method.
- Our method provides sampling-based interpretations of hidden state representations and quantitatively attributes semantic versus lexical focus to individual attention heads.
- Our experiments demonstrate that applying the proposed intervention to the vision encoders of several state-of-the-art LVLMs yields substantial gains in Visual Question Answering accuracy under Typographic Attack interference on RIO-Bench.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02494v1)
- [arXiv](https://arxiv.org/abs/2607.02494v1)

---

<a id='2607.02486v1'></a>
## [GeoMix: Descriptor-Free Visual Localization via Global Context and Multi-Detector Training](https://arxiv.org/abs/2607.02486v1)

**Authors:** Yejun Zhang, Xinjue Wang, Zihan Wang, Esa Rahtu, Juho Kannala

**Published:** 2026-07-02

**Categories:** cs.CV

**Abstract:**

Descriptor-free visual localization eliminates high-dimensional descriptor storage, preserves scene privacy, and simplifies map maintenance, yet its accuracy still lags far behind descriptor-based pipelines. We identify this gap to insufficient geometric discriminability in geometry-only matching. Without visual appearance, current methods underutilize local geometry cues, lack the global context among keypoints, and overfit to a single keypoint detector. We further observe that descriptor-free matching naturally enables multi-detector training, as heterogeneous keypoints can be optimized in a shared geometry-only space without aligning descriptor spaces. Building on these insights, we propose GeoMix, a descriptor-free 2D-3D matching framework that strengthens geometric discriminability at three levels. Locally, directional and distance-aware embeddings enrich neighborhood aggregation with fine-grained spatial structure. Globally, learnable context nodes aggregate and redistribute scene-wide information via cross-attention to resolve ambiguities beyond local receptive fields. At the training level, Mix-Training exploits this detector-agnostic geometry space to learn representations across multiple keypoint detectors. Extensive experiments on MegaDepth, Cambridge Landmarks, 7Scenes, and Aachen Day-Night show that GeoMix sets a new state of the art among descriptor-free methods, reducing 75th-percentile rotation error by 89\% and translation error by up to 90\% over the previous best, while generalizing zero-shot to unseen detectors and narrowing the gap to descriptor-based pipelines. Code is available at $\href{https://github.com/YejunZhang/Geomix}{\text{this links}}$.

**Analysis:**

## 1. 摘要翻译
描述符无关的视觉定位技术消除了对高维描述符存储的需求，保护了场景隐私，并简化了地图维护工作，但其定位精度仍显著落后于基于描述符的管线。本文指出，这一精度差距源于几何匹配中几何可辨别能力的不足。在缺乏视觉外观信息的情况下，现有方法对局部几何线索利用不足，缺乏关键点间的全局上下文，且容易过拟合于单一关键点检测器。我们进一步发现，描述符无关的匹配天然支持多检测器训练，因为异构关键点可以在共享的几何空间中进行优化，无需对齐描述符空间。基于这些洞察，我们提出了GeoMix。

## 2. 方法动机分析
*   **驱动力**：解决描述符无关（Descriptor-Free）视觉定位中因缺乏视觉特征导致的精度较低问题。
*   **痛点**：现有方法不仅在局部邻域聚合上忽略了方位和度量信息，而且局部卷积缺乏全局感受野，无法区分几何相似的重复结构（如长廊、外墙）。此外，单一检测器训练导致网络具有较强的“检测器偏见”。
*   **研究假设**：几何空间是检测器无关的。通过引入多检测器联合训练（Mix-Training），可以强迫模型学习到更具鲁棒性的本质几何结构，而非特定检测器的响应模式。

## 3. 方法设计详解
GeoMix在描述符无关框架下通过三级策略增强几何判别力：
*   **局部几何（Local Geometry）**：通过双分支结构处理局部邻域。不仅使用传统的annular convolution（环形卷积），还引入了显式的**方位（Direction）和距离（Distance）几何嵌入**，通过拼接相对位移 $\Delta p_{ij}$ 及其单位方向，增强对局部空间拓扑的感知。
*   **全局上下文（Global Context Nodes）**：针对$O(N^2)$的全局自注意力计算成本问题，引入$N_g$个可学习的全局上下文节点。这些节点作为信息瓶颈，通过跨注意力（Cross-attention）机制聚合和分发场景全局信息，有效缓解局部感受野受限带来的歧义。
*   **多检测器混合训练（Mix-Training）**：这是本文的核心创新。在训练时，不再局限于单一检测器产生的关键点，而是将SIFT、SuperPoint、DISK等多检测器生成的关键点集分别输入模型，并共享同一套3D地图点进行训练。这起到了一种“几何正则化”的作用，使模型强制学习到与检测器无关的通用几何推理能力。

## 4. 方法对比分析
*   **本质区别**：与仅利用基本空间关系的方法不同，GeoMix通过显式注入度量几何信息（距离+方向）和学习全局上下文，提升了对单个匹配点的判别性。
*   **创新贡献**：提出了“多检测器混合训练”策略，利用描述符无关的特性，成功解决了视觉定位中长期存在的“检测器特定偏见”问题，实现了零样本跨检测器迁移。
*   **适用场景**：适用于需要低存储、高隐私保护，且对定位精度要求极高的移动端或嵌入式系统。

## 5. 实验分析
*   **验证方法**：在MegaDepth（训练）、Cambridge Landmarks、7Scenes及Aachen Day-Night数据集上进行验证。
*   **关键结论**：在MegaDepth上，GeoMix显著提升了定位精度，75th百分位旋转误差降低了89%，平移误差降低了90%。
*   **优势**：在保持紧凑模型尺寸（3.5M参数）和极低地图存储需求的同时，显著缩小了与描述符基准方法的精度差距。
*   **局限**：在高外点比率下，仅依赖几何线索仍存在辨别力上限，且目前3D地图构建依然依赖SIFT，存在源域偏见。

## 6. 实用指南
*   **开源情况**：代码已开源（https://github.com/YejunZhang/Geomix）。
*   **实现细节**：
    *   **超参数**：默认使用 $N_g=4$ 个全局上下文节点。
    *   **训练策略**：需要维护多个检测器生成训练样本，且需固定3D点云地图，仅改变查询端的关键点分布。
*   **迁移可能**：该框架易于迁移至其他基于图形神经网络的特征匹配任务，特别是那些存在多种特征提取器选择或需要跨模态匹配的场景。

## 7. 总结
*   **核心思想**：利用几何空间的通用性，通过多检测器混合训练打破特征提取器的束缚。
*   **速记版Pipeline**：
    1.  **输入处理**：提取2D关键点和3D点云，转换为统一的“方位向量”。
    2.  **特征编码**：引入方向与距离的显式几何嵌入，强化局部感知。
    3.  **全局交互**：通过少量全局节点进行跨点集通信，消除重复场景的歧义。
    4.  **混合训练**：轮流使用不同检测器生成的关键点进行训练，提升泛化能力。
    5.  **稳健匹配**：利用Sinkhorn算法与学习到的分类器过滤外点，估计最终位姿。

**Key Findings:**

- Building on these insights, we propose GeoMix, a descriptor-free 2D-3D matching framework that strengthens geometric discriminability at three levels.
- Extensive experiments on MegaDepth, Cambridge Landmarks, 7Scenes, and Aachen Day-Night show that GeoMix sets a new state of the art among descriptor-free methods, reducing 75th-percentile rotation error by 89\% and translation error by up to 90\% over the previous best, while generalizing zero-shot to unseen detectors and narrowing the gap to descriptor-based pipelines.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02486v1)
- [arXiv](https://arxiv.org/abs/2607.02486v1)

---

<a id='2607.02466v1'></a>
## [Learning to Move Before Learning to Do: Task-Agnostic pretraining for VLAs](https://arxiv.org/abs/2607.02466v1)

**Authors:** Junhao Shi, Siyin Wang, Xiaopeng Yu, Li Ji, Jingjing Gong, Xipeng Qiu

**Published:** 2026-07-02

**Categories:** cs.RO, cs.AI

**Abstract:**

Vision-Language-Action (VLA) models are fundamentally bottlenecked by the scarcity of expert demonstrations -- triplets of observations, instructions, and actions that are costly to collect at scale. We argue that this bottleneck stems from conflating two distinct learning objectives: acquiring physical competence (how to move) and acquiring semantic alignment (what to do). Crucially, only the latter requires language supervision. Building on this Decomposition Hypothesis, we propose Task-Agnostic Pretraining (TAP), a two-stage framework that first learns transferable motor priors from cheap, unlabeled interaction data -- including discarded off-task trajectories and autonomous robot play -- via a self-supervised Inverse Dynamics objective. A lightweight second stage then grounds these priors in language using minimal expert data. On the SIMPLER benchmark, TAP matches models trained on over 1M expert trajectories while using orders of magnitude less labeled data, yielding a 10% absolute gain over standard behavior cloning. On a real-world WidowX platform, TAP retains 25% success under camera perturbations where internet-scale baselines collapse to 0%, demonstrating that task-agnostic pretraining produces robust, transferable physical representations and offers a scalable path forward for Embodied AI.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作（VLA）模型目前受限于专家演示数据的匮乏，因为这些包含观察、指令和动作的三元组数据收集成本高昂。我们认为，这一瓶颈源于将两个不同的学习目标混为一谈：习得物理能力（如何移动）和习得语义对齐（做什么）。关键在于，只有后者需要语言监督。基于这一“分解假设”（Decomposition Hypothesis），我们提出了“任务不可知预训练”（TAP），这是一个两阶段框架：第一阶段通过自监督逆动力学目标，利用廉价的未标记交互数据（包括废弃的任务无关轨迹和自主机器人游戏）学习可迁移的运动先验；第二阶段则利用极少量的专家数据将这些先验与语言指令进行对齐。在SIMPLER基准测试中，TAP在仅使用极少量标注数据的情况下，性能匹配了百万级专家轨迹训练的模型，且在标准行为克隆基础上实现了10%的绝对提升。在WidowX机器人实验中，TAP在极端摄像机扰动下仍保持25%的成功率，而基线模型则完全失效。这证明了任务不可知预训练能产生鲁棒、可迁移的物理表征，为具身智能提供了可扩展的路径。

### 2. 方法动机分析
- **驱动力**：VLA模型受制于“数据墙”，即高质量专家数据收集成本高、不可扩展。
- **现有方法痛点**：传统方法盲目地将“物理运动”与“语言指令”绑定，浪费了海量无标注但包含丰富物理交互信息的机器人交互数据。
- **研究假设**：**分解假设**——动作生成可拆解为“物理形态感知（如何移动）”与“语义意图 grounding（做什么）”。前者不依赖语言，仅通过交互即可学得。

### 3. 方法设计详解
- **流程总结**：
  1. **数据准备**：收集“任务无关”轨迹（如其他任务的废弃数据）和“自主探索”数据（机器人随机运动）。
  2. **阶段一（TAP预训练）**：利用逆动力学目标（Inverse Dynamics），给定观察$o_t$和$o_{t+1}$，模型预测中间动作$a_t$。该过程强迫模型忽略静态背景，专注于末端执行器运动与物体位移，从而提取“物理常识”。
  3. **阶段二（任务对齐）**：冻结部分权重或进行轻量级微调，利用少量专家演示数据，将已有的物理先验映射到特定的语义指令上。
- **关键细节**：通过Voxel Grid下采样和Contact Heuristic（接触启发式）确保探索数据的有效性；将训练频率下采样至5Hz，避免传感器噪声掩盖真实物理位移。

### 4. 方法对比分析
- **本质区别**：TAP并非端到端地将语意与动作绑定，而是先进行“物理预热”，再进行“语义微调”。
- **创新贡献**：成功将“无用”的任务无关交互数据转化为高质量物理先验，证明了机器人无需专家监督也能习得底层控制能力。

### 5. 实验分析
- **验证方法**：在SIMPLER仿真环境和WidowX真实机器人上进行，涵盖标准、扰动、复杂场景。
- **关键结果**：TAP在仅需约30小时自主探索和极少专家数据的前提下，在复杂任务中超越了需百万级数据的基线模型，鲁棒性提升显著。
- **优势**：显著提升样本效率，具备极强的抗扰动能力和域迁移能力。
- **局限**：在高层语义逻辑推理上仍受限于VLM后端能力。

### 6. 实用指南
- **开源情况**：代码及模型已开源（https://github.com/sjh0354/Task-Agnostic-Pretrain）。
- **实现细节**：
  - 数据收集时需注意 elevation threshold 参数，避免机械臂仅在空中无用挥舞。
  - 第一阶段训练时需 freeze 视觉编码器，仅微调VLM主干及动作头。
- **迁移可能**：该框架天然适用于多具身（Multi-embodiment）系统，预训练的物理先验具备通用性，可快速迁移至不同尺寸/构型的机械臂。

### 7. 总结
- **核心思想**：通过逆动力学将“物理运动能力”与“语义理解”解耦进行分阶段学习。
- **速记版pipeline**：
  1. 机器人自主玩耍收集无标记轨迹。
  2. 通过预测相邻帧间的动作学习运动规律。
  3. 利用极少量专家指令对齐物理先验。

**Key Findings:**

- Building on this Decomposition Hypothesis, we propose Task-Agnostic Pretraining (TAP), a two-stage framework that first learns transferable motor priors from cheap, unlabeled interaction data -- including discarded off-task trajectories and autonomous robot play -- via a self-supervised Inverse Dynamics objective.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02466v1)
- [arXiv](https://arxiv.org/abs/2607.02466v1)

---

<a id='2607.02431v1'></a>
## [WorldSample: Closed-loop Real-robot RL with World Modelling](https://arxiv.org/abs/2607.02431v1)

**Authors:** Yuquan Xue, Le Xu, Zeyi Liu, Zhenyu Wu, Zhengyi Gu, Xinyang Song, Bofang Jia, Ziwei Wang

**Published:** 2026-07-02

**Categories:** cs.RO, cs.AI

**Abstract:**

Reinforcement learning (RL) can overcome the demonstration-coverage limitation of imitation learning (IL) by allowing robots to improve through trial-and-error interaction beyond the states observed in demonstrations. However, deploying RL on real robots remains constrained by high interaction costs, since each physical rollout is costly and reflects only one realized action-outcome path. To address this challenge, we propose WorldSample, a physically grounded data augmentation framework for real-robot RL that closes a real-synthetic loop between physical rollouts, world-model generation, and policy improvement. Grounded on real rollouts, WorldSample generates high-fidelity synthetic transitions through a post-trained world model, which greatly lowers the visual hallucination. Specifically, rather than simply using these transitions as real-world experience, WorldSample introduces Policy-Paced Learning (PPL) to regulate the training process through sample selection and scheduling, balancing useful augmentation against value overestimation and mitigating the hallucination-induced noise. Experiments on robot manipulation tasks involving contact-rich and precise tasks show that WorldSample improves policy success rate by 28% while reducing training steps by 59% compared with baselines. Furthermore, WorldSample improves world model visual fidelity by 19.4dB in PSNR and 0.47 in SSIM over demonstration-only post-training, validating the effectiveness of the real-synthetic loop for both policy and world model performance.

**Analysis:**

### 1. 摘要翻译
强化学习（RL）通过允许机器人超越演示数据中的状态进行反复尝试，克服了模仿学习（IL）的覆盖局限。然而，在真实机器人上部署RL受限于高昂的交互成本，因为物理 rollout 既昂贵又仅反映单一动作结果路径。为此，我们提出了 WorldSample，这是一个用于真实机器人RL的物理锚定数据增强框架，它闭合了物理 rollout、世界模型生成和策略改进之间的真实-合成循环。基于真实 rollout，WorldSample 通过预训练的世界模型生成高保真合成转换，极大降低了视觉幻觉。具体而言，WorldSample 不仅简单地将这些转换视为真实世界经验，还引入了策略步调学习（PPL）来通过样本选择和调度调节训练过程，在平衡有用增强与价值高估的同时，减轻幻觉引入的噪声。在涉及接触丰富和精密操作任务的机器人实验表明，与基线相比，WorldSample 将策略成功率提高了 28%，同时减少了 59% 的训练步骤。此外，WorldSample 将世界模型视觉保真度在 PSNR 上提高了 19.4dB，在 SSIM 上提高了 0.47，验证了真实-合成循环对策略和世界模型性能的有效性。

### 2. 方法动机分析
- **驱动力**：旨在解决真实世界机器人强化学习中样本效率低、物理交互昂贵且易导致硬件损坏的瓶颈。
- **痛点**：现有方法要么依赖专家演示（覆盖率低），要么单纯使用世界模型“做梦”（存在严重的视觉幻觉和动力学漂移，导致策略训练不稳定）。
- **研究假设**：通过“反事实轨迹生成”保持物理锚定，并引入“策略步调学习（PPL）”机制，可以安全地利用世界模型生成的合成数据，实现更高效的在线学习。

### 3. 方法设计详解
- **流程总结**：
  1. **数据收集**：从真实机器人获取 rollout，构建真实重放缓冲区。
  2. **反事实轨迹生成**：对真实 rollout 中的动作序列进行随机扰动（Counterfactual Sampling），输入世界模型生成对应的合成视频和奖励。
  3. **异步调度**：利用异步工作流并行生成数据，避免阻塞实时控制。
  4. **PPL 训练**：利用 Q-aware 样本选择（根据奖励模型区分成功/失败样本）和不确定性引导的调度（基于 actor 熵调整合成数据比例），将合成数据注入 RL 更新。
- **关键技术**：
  - **Counterfactual Trajectory Generation**：通过对真实动作序列施加 scale-randomized 局部扰动，确保生成数据在物理上可行。
  - **Q-aware Sample Selection**：利用奖励模型过滤数据，平衡正负样本，防止价值估计偏差。
  - **Uncertainty-Guided Scheduling**：公式 $\rho(H_t) = \rho_{\max} \cdot \frac{1}{1 + \kappa H_t^p}$，当策略在高不确定性（高熵）时减少合成数据，随着策略收敛增加权重。

### 4. 方法对比分析
- **本质区别**：WorldSample 将世界模型作为数据增强工具，而非单一的模拟器；其创新在于实现了“真实-合成”的闭环，动态调整策略对合成数据的信任度。
- **创新贡献**：引入 PPL 机制，将RL训练的稳定性与数据增强的采样效率解耦。

### 5. 实验分析（精简版）
- **验证方法**：在 Galaxea A1X 机器人上进行 5 类接触丰富或精密的操作任务（如插入、组装等）。
- **关键结论**：成功率平均提升 28%，训练步骤减少 59%。
- **优势**：显著提升了长周期任务的成功率，模型泛化能力强。
- **局限**：目前主要针对单任务，且高度依赖基础世界模型（如 Cosmos-Predict）的预训练质量。

### 6. 实用指南
- **开源情况**：项目主页已提供（https://xxreinsno.github.io/worldsample/）。
- **实现细节**：
  - 参数：$\xi=0.20, \sigma=0.05, \rho_{\max}=0.30$。
  - 核心逻辑：务必保证 world model 能随在线 rollout 同步更新（post-training），否则会产生累积误差。
- **迁移可能**：该框架易于迁移至任何基于 actor-critic 的机器人 RL 框架（如 SAC/RLPD）。

### 7. 总结
- **核心思想**：通过物理锚定生成与策略不确定性感知调度，安全扩展样本。
- **速记版pipeline**：
  1. 真实交互获取基础路径。
  2. 对路径进行局部扰动生成合成数据。
  3. 计算策略不确定性。
  4. 动态调整真实与合成数据的混合比例。
  5. 闭环更新世界模型。

**Key Findings:**

- To address this challenge, we propose WorldSample, a physically grounded data augmentation framework for real-robot RL that closes a real-synthetic loop between physical rollouts, world-model generation, and policy improvement.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02431v1)
- [arXiv](https://arxiv.org/abs/2607.02431v1)

---

<a id='2607.02375v1'></a>
## [Representation Distribution Matching for One-Step Visual Generation](https://arxiv.org/abs/2607.02375v1)

**Authors:** Lan Feng, Wuyang Li, Eloi Zablocki, Matthieu Cord, Alexandre Alahi

**Published:** 2026-07-02

**Categories:** cs.CV

**Abstract:**

We elucidate the design space of Representation Distribution Matching (RDM), our name for the paradigm that trains a one-step image generator by matching generated and reference feature distributions under frozen pretrained encoders. We identify two design axes, how the distributions are compared and the representations they are compared in, and controlled studies along them yield three findings. First, the classical MMD, which could not train convincing generators a decade ago, becomes a strong and scalable objective once estimated right. Second, the generated batch is then the operative variable, with an optimum above 2048, far beyond customary batch sizes. Third, any single representation can be gamed, driven below the real score while images stay visibly fake, so we match against a balanced battery of encoders and evaluate with SW_r14, a Sliced-Wasserstein distance over 14 encoders that is independent of the training loss and resists gaming. Combining the preferred choices yields improved RDM (iRDM): it sets the one-step state of the art on ImageNet at SW_r14 1.30, corroborated by PickScore, a human-preference proxy our objective never optimizes, which prefers it over the prior best one-step generator on 71.2% of matched samples. The same recipe post-trains the four-step FLUX.2 [klein] into a one-step generator, surpassing the four-step version on GenEval, 0.826 to 0.794, and on PickScore, 22.76 to 22.58, in 90 H200 GPU-hours. Project page: https://alan-lanfeng.github.io/rdm/.

**Analysis:**

### 1. 摘要翻译
我们阐明了表征分布匹配（RDM）的设计空间，这是一种通过在冻结的预训练编码器下匹配生成特征分布和参考特征分布来训练单步图像生成器的范式。我们确定了两个设计轴：分布如何比较以及在何种表征中进行比较，并基于此进行了受控研究，得出了三个结论：第一，曾被认为无法训练高质量生成器的经典最大均值差异（MMD），在正确估计后成为了一种强大且可扩展的目标函数；第二，生成的批次大小是关键变量，最佳值超过2048，远超传统批次大小；第三，单一表征容易被“刷分”（即生成看似逼真但实际虚假、在特定编码器上得分极高的图像），因此我们采用一组平衡的编码器进行匹配，并使用基于14个编码器的切片瓦瑟斯坦距离（$SW_{r14}$）进行评估，该指标独立于训练损失且难以被针对性破解。结合这些偏好选择，我们提出了改进的RDM（iRDM），在ImageNet上创下了$SW_{r14}=1.30$的单步生成SOTA，且PickScore优于现有模型。该配方还可将四步FLUX.2 [klein]后训练为单步生成器，在90个H200 GPU小时内超越了原四步版本的GenEval得分。

### 2. 方法动机分析
- **核心动机**：打破“多步采样”与“模型质量”的折中，直接实现高质量单步生成，并解决现有匹配方法中存在的指标可破解性及训练不稳定问题。
- **痛点分析**：
    1. **传统判别器/对抗训练**：不适用于单步生成，存在训练不稳定、难收敛问题。
    2. **单一指标/单编码器依赖**：生成器会针对性优化特定编码器特征，导致“奖励黑客”（Reward Hacking），生成图像在特定指标上分数极高，但实际质量极差。
    3. **估计偏差**：现有分布匹配方法对样本量和参考分布估计不足，导致生成分布未能覆盖真实数据分布。
- **核心直觉**：生成分布应在多个语义丰富的预训练特征空间中同时与真实数据分布达到一致，且匹配过程需使用严谨的统计估计（如Nystrom近似），而非简单的矩匹配或不稳定的小批次估计。

### 3. 方法设计详解
- **pipeline总结**：
    1. **参考构建**：使用Nystrom方法将128万张真实图像压缩为4096个 landmark，构建一个固定的、高质量的参考分布（Attraction）。
    2. **实时生成**：每步生成一个大批次（$N \ge 2048$）样本。
    3. **分布匹配**：在多个冻结的预训练编码器（如DINOv2, CLIP, SigLIP等）下计算损失。
    4. **动态平衡**：通过PID拉格朗日控制器动态调整各编码器权重，确保生成器不会在单一编码器上过拟合。
- **算法解释**：损失函数$L_\phi$包含两项：
    - **精确排斥（Repulsion）**：计算生成批次内样本间的精确核距离，防止模式坍塌，使生成样本均匀分布。
    - **Nystrom吸引（Attraction）**：计算生成样本与预先冻结的真实数据特征中心（均值嵌入）的距离，将生成样本拉向真实流形。

### 4. 方法对比分析
- **本质区别**：不再依赖对抗训练或蒸馏轨迹，而是将单步生成建模为纯粹的“多空间特征分布对齐”问题。
- **创新点**：
    - 提出了平衡的编码器电池（Encoder Battery）及拉格朗日权重平衡机制，从根本上解决了单一编码器匹配带来的游戏化（Gaming）问题。
    - 将MMD与Nystrom近似结合，实现了计算高效且估计精准的分布匹配。
- **适用场景**：适用于任何可被提取高维特征的图像生成任务，尤其适合需要大幅提升推理速度（单步）的场景。

### 5. 实验分析（精简版）
- **核心结论**：iRDM在ImageNet生成上达到$SW_{r14}=1.30$，且在对FLUX.2模型进行后训练时，PickScore和GenEval均超越了原四步Teacher模型。
- **优势**：无需在线教师，训练稳定，指标对齐度高，抗破解能力强。
- **局限**：对超大批次显存要求较高，需依赖梯度缓存（Gradient Caching）技术实现。

### 6. 实用指南
- **开源/复现**：项目主页已开放，关键实现需注意：保持大批量生成，预计算真实数据侧的Nystrom landmark。
- **实现细节**：建议使用$N \ge 5120$的批次，使用AdamW优化器，并确保采用Diverse Encoder集合，避免模型过拟合单个表征。
- **迁移性**：该框架天然适用于任何存在强大预训练编码器的任务，如视频、音频特征空间匹配。

### 7. 总结
- **核心思想**：通过多编码器空间分布匹配与动态权重平衡，实现高质量单步生成。
- **速记版pipeline**：
    1. 冻结参考分布（基于Nystrom landmark）；
    2. 生成海量新鲜样本；
    3. 在多个编码器空间并行对齐；
    4. 根据对齐效果动态加权修正。

**Key Findings:**

- The same recipe post-trains the four-step FLUX.2 [klein] into a one-step generator, surpassing the four-step version on GenEval, 0.826 to 0.794, and on PickScore, 22.76 to 22.58, in 90 H200 GPU-hours.
- Project page: https://alan-lanfeng.github.io/rdm/.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02375v1)
- [arXiv](https://arxiv.org/abs/2607.02375v1)

---

<a id='2607.02332v1'></a>
## [HEFT: Heavy-Payload Full-size Humanoid Teleoperation with Privileged Motion Guidance and Windowed Payload Curriculum](https://arxiv.org/abs/2607.02332v1)

**Authors:** Chenxin Liu, Qingzhou Lu, Guangxiao Yang, Xuanyang Shi, Chenghan Yang, Yanjiang Guo, Jianyu Chen

**Published:** 2026-07-02

**Categories:** cs.RO

**Abstract:**

General motion tracking and teleoperation offer a promising path to scalable humanoid skill acquisition, yet most existing frameworks are validated on compact platforms or without real payload interaction, leaving full-size humanoids with real payloads largely unexplored. Scaling to full-size humanoids introduces two compounding challenges: their larger inertia and tighter balance margins make tracking highly sensitive to noise, drift, and retargeting errors from commodity VR trackers, while their payload potential remains largely underutilized. We present HEFT, a heavy-payload full-size humanoid teleoperation framework that addresses both challenges. HEFT learns from deployable noisy VR references with physically plausible reconstructed references through Privileged Motion Guidance (PMG), and uses a Windowed Payload Curriculum (WPC) with expert-guided payload caps to acquire robust heavy-payload tracking. We deploy HEFT on L7, a 175cm, 65kg humanoid. The robot tracks motions including turns, forward/backward locomotion, and squats under payloads up to 24kg.

**Analysis:**

## 1. 摘要翻译

通用运动追踪和遥操作是实现人形机器人技能扩展的有效路径，但现有的框架多局限于小型平台或缺乏真实负载交互，全尺寸人形机器人在真实负载下的潜力尚未得到充分开发。扩展到全尺寸人形机器人带来了两个复合挑战：其巨大的惯性及狭窄的平衡裕度使得追踪极易受到来自通用VR跟踪器产生的噪声、漂移和重定向误差的影响，同时其负载潜力也未被充分利用。我们提出了HEFT，一个能够解决上述两个挑战的重载全尺寸人形机器人遥操作框架。HEFT通过特权运动引导（Privileged Motion Guidance, PMG）利用物理上合理的重构参考信息来学习可部署的噪声VR参考指令，并结合带有专家引导负载上限的窗口化负载课程（Windowed Payload Curriculum, WPC）来实现鲁棒的重载追踪。我们将HEFT部署在175cm、65kg的L7人形机器人上，实现了在高达24kg负载下，对转弯、前后移动及深蹲等全身运动的鲁棒追踪。

## 2. 方法动机分析

*   **驱动力**：解决全尺寸人形机器人在真实物理负载下，因惯性大、稳定性敏感，导致遥操作过程中对VR输入噪声难以容忍的问题。
*   **现有方法痛点**：现有的遥操作方法通常处理的是轻量化机器人，或忽略了负载带来的动力学影响；在线直接使用重构后的高质量运动数据会带来额外时延，损害遥操作的实时反馈闭环。
*   **研究假设**：通过将“带噪声的在线VR数据”与“高质量离线重构数据”进行解耦训练，并在任务中引入“与运动状态相关的负载约束”，可以实现对复杂全身运动的重载鲁棒控制。

## 3. 方法设计详解

*   **流程总结**：
    1.  **数据预处理**：构建原始VR输入与离线重构（利用RoHM去噪）之间的配对数据集。
    2.  **特权运动引导 (PMG)**：在训练时，Actor接收原始VR指令（模拟在线），Critic和奖赏函数则接收经过重构的物理一致参考指令，使策略学习“识别并忽略”追踪伪影。
    3.  **窗口化负载课程 (WPC)**：将完整运动切分为5秒片段，利用专家策略进行rollout搜索，确定每个片段能承受的最大负载上限，并在训练中动态采样负载。
    4.  **控制器蒸馏**：采用RMA结构的教师-学生架构。教师获取特权信息（模拟状态、负载信息、重构参考），训练出Latent空间；学生（Adapter）仅通过观测历史观测序列来预测该Latent，实现端侧零特权信息部署。

*   **算法解释**：
    *   **PMG**：利用异步监督机制，使模型在无需实时重构时，通过“特权信息”将噪声参考强制锚定到更优的基准运动空间。
    *   **WPC**：核心在于“因地制宜”，根据动作片段（如深蹲需更低负载）动态限制训练负载，避免盲目过载导致无法收敛。

## 4. 方法对比分析

*   **本质区别**：传统方法要么直接追踪噪声参考，要么对动作做过强的平滑处理。HEFT通过PMG区分了“操作者意图”和“传感器噪声”，实现了对原始参考的智能适应。
*   **创新贡献**：提出PMG特权引导范式与WPC窗口化课程，实现了在单一控制器下处理极重载（24kg）的多样化运动任务。
*   **适用场景**：高负载搬运、复杂物流、建筑场景下的全身人机遥操作。

## 5. 实验分析（精简版）

*   **验证方法**：在SEED与VR数据集上进行对比，并部署于L7全尺寸机器人。
*   **关键结论**：在20kg以上的高负载下，成功率显著优于TWIST2+FC等基线；在无负载动态运动中，PMG能够显著降低追踪漂移。
*   **局限**：模型无法显式建模抓取质量、物体形状及环境接触点，依赖离线重构作为引导。

## 6. 实用指南

*   **开源**：HEFT-homepage已公开。
*   **细节**：实现关键在于RoHM的离线数据去噪效果，以及Expert在预处理阶段对动作片段负载阈值的准确搜索。超参数适配方面，Adapter的监督权重（L_adapt）至关重要。
*   **迁移**：可迁移至其他具有强动力学耦合或高惯性的机器人平台，需替换对应的URDF模型及低层控制器接口。

## 7. 总结

*   **核心思想**：利用特权信息解耦噪声，通过分段负载调度提升重载鲁棒性。
*   **速记版Pipeline**：
    1. 离线数据去噪与重构；
    2. 分段扫描寻找最大动作负载；
    3. 训练特权教师处理噪声；
    4. 蒸馏适配器实现零特权部署。

**Key Findings:**

- We present HEFT, a heavy-payload full-size humanoid teleoperation framework that addresses both challenges.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02332v1)
- [arXiv](https://arxiv.org/abs/2607.02332v1)

---

<a id='2607.02322v1'></a>
## [The Moving Eye: Enhancing VLA Spatial Generalization via Hybrid Dynamic Data Collection](https://arxiv.org/abs/2607.02322v1)

**Authors:** Jincheng Tang, Yilong Zhu, Zhengyuan Xie, Jiang-Jiang Liu, Jiaxing Zhang

**Published:** 2026-07-02

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have shown remarkable promise in generalized robotic manipulation. However, their spatial generalization remains fragile. We argue that simply increasing the number of viewpoints is insufficient. Models often fall into the trap of Shortcut Learning, latching onto spurious correlations (e.g., fixed relative poses between objects or between the camera and robot base) rather than learning true spatial relationships. In this work, we propose a data-centric solution to enhance VLA spatial generalization. We utilize a dual-arm setup where one arm performs manipulation while the other serves as a mobile environmental camera. We systematically evaluate three data distribution patterns: Fixed, Multi-Fixed, and Moving Views. Our findings reveal that a hybrid strategy, combining continuous camera motion with diverse static viewpoints, yields the best performance by substantially reducing spurious correlations while maintaining training stability. Our experiments demonstrate that this strategy mitigates spurious correlations, enabling VLAs to generalize to unseen camera poses and object configurations where simply adding more static viewpoints fails. Crucially, we reveal that the susceptibility to shortcut learning and the struggle with spatial generalization are universal characteristics shared across diverse architectures. Consequently, all evaluated models (ACT, Diffusion, and VLA models including Pi0 and Gr00t) benefit significantly from our mixed data strategy.

**Analysis:**

## 1. 摘要翻译
视觉-语言-动作（VLA）模型在通用机器人操作中展现了巨大潜力，但其空间泛化能力仍然脆弱。单纯增加视角数量往往无法解决问题，模型易陷入“捷径学习”的陷阱，即通过记忆摄像头与机器人基座或物体之间的伪相关性（固定相对位姿）来完成任务，而非真正理解空间关系。本文提出一种数据驱动的解决方案以增强VLA的空间泛化能力，即利用双臂机器人设置，通过一臂操作、另一臂充当移动环境摄像头的模式，系统性评估并构建了“固定”、“多固定”与“移动”三种数据分布模式。研究发现，结合连续摄像头运动与多样化静态视角的混合策略效果最佳，能显著减少伪相关性并保持训练稳定性。实验证明，该策略使VLA模型能够泛化至未见的摄像头位姿和物体配置，优于单纯增加静态视角的传统做法。该策略对ACT、Diffusion、Pi0及Gr00t等多种架构均有显著提升。

---

## 2. 方法动机分析
*   **驱动力**：解决VLA模型在面对视点变化时泛化能力骤降的问题，将模型从“记住特定视角的像素模式”转变为“理解空间布局”。
*   **现有方法痛点**：当前通过增加静态视角来提升泛化的做法，依然受限于物体间的固定相对位姿（Object-Position Coupling）或相机基座的固定外观，模型倾向于利用这些“捷径”而非学习任务本身的空间几何。
*   **研究假设**：通过在训练中强制引入连续的动态视角（Moving View）作为正则化项，可以破坏由于固定视点产生的伪相关性，促使模型学习真正的空间关系。

---

## 3. 方法设计详解
*   **流程总结**：采用双臂协作机制，主臂执行 manipulation 任务，从臂携带摄像头以连续轨迹运动，构建包含“多固定视角（Multi-Fixed）”与“移动视角（Moving View）”的混合数据集。
*   **核心策略（Hybrid Strategy）**：
    1.  **分层视点采样**：混合使用Multi-Fixed（保证收敛稳定性）和Moving（作为空间不变性的正则化器）。
    2.  **多维多样性注入**：不仅在相机视点上多样化，还显式改变物体间的相对位置（如将笔架放在不同位置），从根本上打破物体间位置耦合。
    3.  **最优混合比例（Golden Ratio）**：实验得出Moving:Multi-Fixed = 1:3 是Gr00t模型的最优混合比例，既能避免单纯动态数据导致的高方差收敛困难，又能有效去除偏见。
*   **公式意义**：$D_{train} = \frac{k}{k+1}D_{MultiFixed} + \frac{1}{k+1}D_{Moving}$，通过调整参数 $k$（此处为3），平衡稳定性和泛化性。

---

## 4. 方法对比分析
*   **本质区别**：从单纯增加视角数量转变为“通过数据工程主动解耦”。该方法不仅关注“视角多”，更关注“破坏视角/物体间的相关性”。
*   **创新贡献**：提出了在物理机器人上进行大规模动态轨迹数据采集的系统方案，并明确了“移动视角”对“多固定视角”的正则化协同效应。
*   **适用场景**：适用于所有对空间几何要求敏感的机器人操作任务，尤其是需要应对非结构化环境或相机易受扰动场景。

---

## 5. 实验分析
*   **关键结果**：在Pen Pick-and-Place任务中，传统的Fixed-View模型在ID测试达85%，OOD测试降至43%；而使用本文混合数据策略后，ID/OOD均维持在83%-90%左右。
*   **优势**：极强的跨架构通用性，对现有流行的VLA模型（ACT, Diffusion, Pi0, Gr00t）均有效。
*   **局限**：目前的实验主要聚焦于桌面Pick-and-Place，对长时程、复杂接触任务的系统性验证仍有待深入。

---

## 6. 实用指南
*   **开源情况**：基于LeRobot生态，建议参考文中提到的SO-101 pipeline。
*   **实现细节**：
    *   对于Moving View，建议保持匀速运动（如0.05m/s），防止因过快的帧间视角跳变导致模型无法追踪动作。
    *   在构建数据集时，需确保“物体相对位置”的变化范围覆盖测试集可能遇到的空间区域。
*   **迁移可能**：该方法非常适合迁移到任何机器人 manipulation 任务，只需利用额外的摄像头arm收集少量辅助数据，即可增强现有静态策略。

---

## 7. 总结
*   **核心思想**：利用移动视角数据打破物理空间中的伪相关性，实现空间泛化。
*   **速记版pipeline**：
    1. 配置双臂机器人，一臂负责移动摄像头采集动态路径数据；
    2. 混合“多固定视角”与“移动视角”数据，比例设为3:1；
    3. 在采集过程中随机变换物体相对位置；
    4. 联合混合数据训练VLA模型。

**Key Findings:**

- In this work, we propose a data-centric solution to enhance VLA spatial generalization.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.02322v1)
- [arXiv](https://arxiv.org/abs/2607.02322v1)

---

