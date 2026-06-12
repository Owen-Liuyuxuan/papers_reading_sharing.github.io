time: 20260612

# Arxiv Computer Vision Papers - 2026-06-12

## Executive Summary

## 每日Arxiv计算机视觉论文执行摘要（2026-06-11）

### 一、主题与趋势总览
本期10篇论文高度集中在**机器人操作与技能泛化**领域，尤其是围绕**触觉感知、世界模型与动作策略的融合**展开。主要趋势包括：
- **基础模型向机器人操作迁移**：多篇工作尝试构建跨传感器、跨任务的通用基础策略（如触觉、灵巧操作）。
- **世界模型成为核心架构**：超过三篇论文直接研究世界模型（WEAVER、MaskWAM、RepWAM），强调利用视觉-动作联合表征提升操作泛化性与长时规划能力。
- **3D几何与生成模型结合**：World Tracing和Surflo分别探索像素级几何生成与一致表面流，推动可解释、可编辑的3D表示。
- **多模态与传感器融合**：触觉、视觉、激光雷达的融合策略在操作与定位任务中均得到深化。

### 二、重要与创新论文Highlight
1. **FTP-1**：首个跨多种触觉传感器的通用基础策略，实现接触丰富操作中的零样本迁移，对具身智能的硬件解耦意义重大。
2. **WEAVER**：提出更快、更长时域的世界模型，显著提升机器人操作任务的规划与执行效率，是当前SOTA世界模型的代表性改进。
3. **World Tracing**：突破“可见范围”限制，从单张图像生成像素对齐的完整3D几何，为AR/VR和场景重建提供新范式。
4. **Surflo**：引入全局状态的一致3D表面流模型，实现时序上稳定的表面变形与跟踪，对动态场景理解有重要价值。

### 三、新兴研究方向与技术
- **动作-世界联合建模**：MaskWAM和RepWAM统一掩码预测与视觉-动作分词器，将操作策略学习转化为自回归或掩码生成任务，有望简化机器人策略训练。
- **触觉基础模型**：FTP-1标志着触觉感知从专用模块向通用基础能力的转变。
- **流逆转指导**：Andy Tang等人的工作通过逆向流修正策略分布，提升机器人通用策略的安全性与鲁棒性。
- **异构传感器早期融合重排序**：Vilella-Cantos等人针对非结构化环境的长时地点识别，提出LiDAR早期融合与学习重排序，对自动驾驶和户外SLAM有启发。

### 四、建议精读论文（按优先级）
1. **FTP-1** — 触觉泛化突破，适合对机器人操作硬件泛化感兴趣的研究者。
2. **WEAVER** — 世界模型效率与长时性能标杆，值得所有机器人学习方向阅读。
3. **World Tracing** — 3D生成与几何理解前沿，适合计算机视觉与图形学交叉领域。
4. **Mana** — 铰接工具灵巧操作的系统级解决方案，对复杂操作任务具参考价值。
5. **Surflo** — 动态3D表面建模，适合处理非刚性场景的团队。

总体而言，本期论文凸显了**“感知-世界-动作”三者深度融合**的趋势，并展现出从专用模型向通用基础模型演进的清晰路径。建议重点关注触觉基础化、世界模型效率优化以及跨传感器泛化这三个方向。

---

## Table of Contents

1. [FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation](#2606.13102v1)
2. [Mana: Dexterous Manipulation of Articulated Tools](#2606.13677v1)
3. [Improving Robotic Generalist Policies via Flow Reversal Steering](#2606.13675v1)
4. [Modality Forcing for Scalable Spatial Generation](#2606.13676v1)
5. [RepWAM: World Action Modeling with Representation Visual-Action Tokenizers](#2606.13674v1)
6. [$\texttt{WEAVER}$, Better, Faster, Longer: An Effective World Model for Robotic Manipulation](#2606.13672v1)
7. [World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible](#2606.13652v1)
8. [Surflo: Consistent 3D Surface Flow Model with Global State](#2606.13644v1)
9. [MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models](#2606.13515v1)
10. [Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments](#2606.13503v1)

---

## Papers

<a id='2606.13102v1'></a>
## [FTP-1: A Generalist Foundation Tactile Policy Across Tactile Sensors for Contact-Rich Manipulation](https://arxiv.org/abs/2606.13102v1)

**Authors:** Chengbo Yuan, Zicheng Zhang, Mingjie Zhou, Wendi Chen, Yi Wang, Zhuoyang Liu, Dantong Niu, Shuo Wang, Hui Zhang, Wenkang Zhang, Yingdong Hu, Yuanqing Gong, Wanli Xing, Chuan Wen, Cewu Lu, Kaifeng Zhang, Yang Gao

**Published:** 2026-06-11

**Categories:** cs.RO

**Abstract:**

Despite the success of vision-based generalist robotic policies, existing tactile-based policies remain tied to fixed embodiments and sensor setups. This is because tactile signals are highly heterogeneous across hardware, making cross-sensor generalization difficult. We present FTP-1,the first generalist foundation tactile policy pretrained to acquire transferable tactile manipulation abilities across diverse sensors and embodiments. FTP-1 supports varied tactile inputs, including image-, array-, and state-based signals, by using heterogeneous encoders to project them into unified morphology-aware latent tokens that are jointly modeled by a shared tactile Transformer expert. Pretrained on around 3,000 hours of tactile manipulation data aggregated from 26 data sources, spanning human and robot demonstrations across 21 sensors, FTP-1 learns tactile skills that transfer beyond the sensors seen during pretraining. Across downstream finetuning experiments spanning 5 hardware configurations, FTP-1 improves contact-rich manipulation on seen sensor setups by +17.2% and, surprisingly, transfers to two previously unseen tactile-sensor setups, achieving a +31% gain in success rate. FTP-1 establishes the first unified foundation baseline for tactile manipulation, providing future tactile policies with a shared model-level starting point. Pretrained models, datasets, training code and more visualization at https://ftp1-policy.github.io.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **FTP-1** 这篇论文的分析如下：

### 1. 论文核心贡献总结
FTP-1 是首个跨传感器、跨具身（Embodiment-agnostic）的通用基础触觉策略模型，旨在解决触觉数据在不同硬件间高度异构导致的泛化难题。通过在 3,000 小时的多源数据上预训练，该模型实现了在触觉操作任务中对多种传感模式的统一建模，并在未见过的传感器配置上展现出了卓越的迁移能力。

### 2. 关键创新与方法论
*   **异构输入统一化（Unified Latent Tokenization）：** 这是最核心的创新。FTP-1 设计了专门的**异构编码器（Heterogeneous Encoders）**，能够将图像型（如 GelSight）、阵列型及状态型触觉信号映射为**统一的形态感知（Morphology-aware）潜在 Token**。
*   **共享触觉 Transformer 架构：** 利用统一的 Token 空间，模型通过一个共享的 Transformer 主干网络进行建模，从而打破了不同硬件的“信息孤岛”。
*   **大规模异构预训练：** 在 26 个数据源、21 种传感器组成的庞大数据集上进行预训练，这种“触觉领域的大规模预训练”逻辑借鉴了视觉大模型（如 ViT/MAE）的成功经验，通过海量数据压缩触觉感知的共性特征。

### 3. 对领域的潜在影响
*   **从“定制化”到“平台化”：** 触觉感知此前一直深受硬件依赖困扰（即换一个传感器就要重新训练模型）。FTP-1 证明了触觉也可以像视觉模型一样，构建一个通用的预训练基座（Foundation Baseline），这将大幅降低机器人触觉控制的开发门槛。
*   **推动触觉与视觉的语义对齐：** 该工作为未来构建“视触觉统一基础大模型”提供了关键的触觉模态支柱，对于实现真正具备灵巧操作能力的通用机器人至关重要。

### 4. 受益的相关领域与应用
*   **灵巧操作（Dexterous Manipulation）：** 在需要精密力控和物体交互的任务中（如医疗机器人手术、家务机器人整理），该方法能显著提升对未知物体的抓取和操作成功率。
*   **机器人硬件研发：** 对于触觉传感器厂商，FTP-1 提供了一个通用接口，使得新型传感器能够快速集成到现有策略中，加速硬件研发循环。
*   **人机交互（HCI）：** 能够理解复杂触觉信息的系统在 VR/AR 触觉反馈以及假肢控制方面具有巨大的应用潜力。

### 5. 可推断的局限性
*   **数据质量与分布差异：** 尽管拥有 3,000 小时数据，但触觉数据往往伴随极高的噪声和非结构化特征，模型在极度稀疏或极端环境下的稳定性尚待观察。
*   **推理延迟（Inference Latency）：** 使用 Transformer 架构处理实时触觉反馈可能面临延迟挑战，在需要高频闭环控制（如 500Hz 以上）的硬实时场景中，模型的计算开销可能是一个限制因素。
*   **触觉定义的泛化边界：** 尽管在传感器之间实现了迁移，但对于极其特殊的物理交互场景（如极高压力、极端剪切力），目前的统一表示空间是否足以捕获所有物理细节仍存在疑问。

---
**专家点评：**
FTP-1 最令我感兴趣的点在于它证明了**“触觉信息的 Token 化”**是可行的。长期以来，触觉数据的非几何特性（力、变形、振动）使其难以与视觉的像素空间兼容，FTP-1 的方法论实际上是为触觉建立了一套“通用特征表示层”，这在机器人感知领域具有里程碑意义。它标志着触觉研究从单纯的信号处理转向了基于大模型范式的表示学习。

**Key Findings:**

- We present FTP-1,the first generalist foundation tactile policy pretrained to acquire transferable tactile manipulation abilities across diverse sensors and embodiments.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13102v1)
- [arXiv](https://arxiv.org/abs/2606.13102v1)

---

<a id='2606.13677v1'></a>
## [Mana: Dexterous Manipulation of Articulated Tools](https://arxiv.org/abs/2606.13677v1)

**Authors:** Zhao-Heng Yin, Guanya Shi, Pieter Abbeel, C. Karen Liu

**Published:** 2026-06-11

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Articulated tool manipulation remains a major challenge in dexterous robotics due to the need to coordinate internal degrees of freedom and contact-rich interactions. While prior work has largely focused on rigid objects, articulated tool use remains underexplored because of its physical complexity and the difficulty of learning functional grasping and manipulation policies. We present Mana (Manipulation Animator), a general sim-to-real framework that reinterprets dexterous manipulation as an animation problem. Inspired by computer animation, Mana employs a coarse-to-fine pipeline that transforms procedurally-generated grasp keyframes into manipulation trajectories through motion planning and reinforcement learning. The data generation process is largely automatic, requiring only a few mouse clicks to specify functional affordances (<1 minute per tool). Across four articulated tools spanning different scales and joint types, Mana achieves zero-shot sim-to-real transfer for both grasping and in-hand manipulation, demonstrating a scalable approach to dexterous articulated tool use.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **Mana (Manipulation Animator)** 的论文分析如下：

### 1. 主要贡献总结
Mana 提出了一种通用的 Sim-to-Real 框架，专门用于解决灵巧手操作铰接式工具（Articulated Tools）这一难题。该研究通过将复杂的操控问题转化为类似计算机图形学的“动画生成问题”，实现了从简单的功能性标注到高水平操作策略的自动化流程，成功在大规模铰接工具集上实现了零样本（Zero-shot）模拟到现实世界的迁移。

### 2. 核心创新与方法论
*   **动画思维（Animation Paradigm）：** 借鉴了计算动画中“关键帧（Keyframes）到轨迹（Trajectories）”的处理思想，通过将粗粒度的功能性关键帧转化为精细的操作轨迹，降低了复杂操控的学习难度。
*   **粗到细（Coarse-to-Fine）的管道设计：** 结合了运动规划（Motion Planning）与强化学习（Reinforcement Learning），前者负责约束几何与物理可行性，后者负责优化灵巧操作的动力学细节。
*   **高效的数据生成流程：** 论文大幅降低了人工干预成本，通过极简的交互式标注（<1分钟/工具）即可定义工具的功能可供性（Affordance），从而自动生成海量的训练数据，解决了铰接工具数据匮乏的痛点。

### 3. 对领域的潜在影响
*   **改变灵巧操作的范式：** 传统方法往往试图直接从感知空间端到端映射到动作空间，而 Mana 证明了**引入计算图形学的“动画”逻辑**是解决高自由度、多接触点操作任务的有效捷径。
*   **加速 Sim-to-Real 进程：** 该框架的通用性和零样本迁移能力，为灵巧机器人在非结构化环境下的实用化提供了有力支撑。
*   **视觉与控制的深度耦合：** 在计算机视觉层面，该研究可能推动对物体“功能性可供性（Functional Affordance）”的理解，使模型不再仅仅识别几何形状，而是理解铰接结构的运动逻辑。

### 4. 受益的相关领域与应用
*   **人形机器人与服务机器人：** 直接提升机器人在厨房、实验室等环境下使用剪刀、钳子、扳手等铰接式工具的能力。
*   **数字孪生与动画合成：** 该框架中的动画生成逻辑可直接反哺于虚拟人生成，提升电影及游戏产业中虚拟角色交互的真实感。
*   **工业自动化：** 处理具有柔性或复杂铰接结构的精密零件装配。

### 5. 可推测的局限性
*   **感知层面的依赖：** 虽然摘要强调了操作的自动化，但 Sim-to-Real 的成功高度依赖于对铰接结构的准确感知（如关节轴向、运动范围估计），这可能成为真实世界复杂纹理或遮挡环境下的瓶颈。
*   **处理长程任务的能力：** Mana 目前侧重于“操作（Manipulation）”，对于多阶段任务序列（例如：拿起工具 -> 调整位置 -> 执行 -> 放回），其扩展性有待考量。
*   **触觉反馈的缺失（潜在）：** 摘要未提及触觉传感器的集成，在处理需要力反馈的精细操作时，仅靠视觉和位置控制可能在现实世界中遇到挑战。

**对计算机视觉领域的独特意义：**
这篇论文的有趣之处在于它将**视觉理解（功能性可供性）**与**几何运动规划**进行了巧妙的桥梁式连接。它挑战了“端到端黑盒”学习的倾向，展示了引入**几何先验（Geometric Priors）**和**经典动画原理**作为归纳偏置（Inductive Bias），能够极大地提升模型在复杂物理任务上的泛化能力。对于从事机器人感知研究的人员来说，Mana 提示我们：理解物体的“运动学骨架”远比仅获取其“点云形状”对操控更重要。

**Key Findings:**

- We present Mana (Manipulation Animator), a general sim-to-real framework that reinterprets dexterous manipulation as an animation problem.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13677v1)
- [arXiv](https://arxiv.org/abs/2606.13677v1)

---

<a id='2606.13675v1'></a>
## [Improving Robotic Generalist Policies via Flow Reversal Steering](https://arxiv.org/abs/2606.13675v1)

**Authors:** Andy Tang, William Chen, Andrew Wagenmaker, Chelsea Finn, Sergey Levine

**Published:** 2026-06-11

**Categories:** cs.RO

**Abstract:**

Generalist policies can learn a wide range of skills from diverse robot datasets. In order to solve or improve on challenging news tasks, we need a way to infer and invoke the appropriate actions from the policy's rich behavioral prior, especially when directly commanding the policy fails. We focus on flow matching generalists and propose Flow Reversal Steering (FRS): a method that takes suboptimal but ``reasonable'' actions, finds their latent noises by passing them through the flow policy in reverse, and maps them to nearby generalist action modes. We evaluate FRS across many simulated and real-world manipulation settings. First, FRS can turn coarse semantic guidance from humans or vision-language models (VLMs) into corresponding good robot actions, improving zero-shot control. These gains can be distilled with behavioral cloning by training an auxiliary policy to output noises that the generalist maps to good actions -- showing up to 95% absolute task success rate boosts in under a minute of training. Finally, FRS enables policy improvement by bootstrapping reinforcement learning with semantic knowledge, improving on several tasks that standard RL fails to improve on.

**Analysis:**

### 1. 摘要翻译
通用机器人策略可以通过多样的机器人数据集学习广泛的技能。为了解决具有挑战性的新任务，我们需要一种方法来从策略的丰富行为先验中推断并调用适当的动作，特别是在直接下达指令失效时。本文关注基于流匹配（Flow Matching）的通用策略，并提出了**流反转引导（Flow Reversal Steering, FRS）**：一种将次优但“合理”的动作通过流策略进行反向传递以找到对应的潜空间噪声，进而映射到邻近通用动作模态的方法。我们在多个仿真和真实世界操作场景中评估了FRS。首先，FRS能将人类或视觉语言模型（VLM）提供的粗略语义引导转化为高质量的机器人动作，从而提高零样本控制能力。这些增益可以通过行为克隆（BC）进行蒸馏，即训练一个辅助策略来输出噪声，使其被通用模型映射为有效动作——在不到一分钟的训练内，任务成功率最高提升了95%。最后，FRS通过利用语义知识引导强化学习（RL），在标准RL难以改进的任务上实现了策略提升。

### 2. 方法动机分析
- **驱动力**：通用策略模型（VLAs）虽然具备强大的行为先验，但在面对未见任务或复杂长程任务时，往往无法精确地将高层语义转化为低层动作。
- **现有方法痛点**：目前基于流匹配或扩散模型的策略，其噪声空间缺乏直观的结构，直接进行随机探索的代价极高；传统的基于RL的引导方法效率低下，且容易陷入局部最优。
- **研究假设**：通用模型训练数据中蕴含的“合理行为先验”可以通过流反转机制被显式地挖掘出来，通过语义先验指导流策略进行“反向重构”，可以快速将粗略的语义建议映射为精确的、“分布内”的动作。

### 3. 方法设计详解
FRS的核心在于利用流匹配策略的确定性逆映射能力：
1. **语义输入获取**：人类操作员或VLM根据任务场景，输出一个粗略的 Cartesian 动作（如“向右移动”）。
2. **流反转（Flow Reversal）**：将该粗略动作 $a_1$ 通过 ODE 的逆向积分过程：$a_{t-h} \leftarrow a_t - v_\theta(a_t, t | o) \cdot h$，反向映射回潜在空间噪声 $\hat{a}_0$。
3. **去噪与精细化（Flow Denoising）**：利用通用模型将得到的噪声 $\hat{a}_0$ 重新进行前向生成：$\hat{a}_1 \leftarrow \mu_\theta(\hat{a}_0, o)$。这一过程利用了通用模型本身的先验，将粗糙的输入转化为分布内的、高质量的机器人动作。
4. **策略提升（改进学习）**：
    - **零样本模式**：直接执行生成的 $\hat{a}_1$。
    - **DSBC（噪声空间行为克隆）**：将反转产生的“专家噪声”对作为数据，训练一个小的噪声预测网络 $\pi_\phi^{noise}(a_0|o)$。
    - **DSRL+FRS**：利用FRS生成的成功轨迹填充RL回放缓冲区，作为先验引导RL优化。

### 4. 方法对比分析
- **本质区别**：与现有扩散引导方法（如修改采样路径）不同，FRS通过明确的“逆向积分”挖掘噪声空间，将粗略的语义引导转化为模型能理解的细粒度噪声，而非依赖随机采样或高成本RL搜索。
- **创新贡献**：首次提出利用流策略的确定性逆向特性进行语义 steering，证明了语义先验不仅能指导动作，还能通过噪声空间的“专家轨迹”加速策略优化。
- **适用场景**：适用于具备预训练基础模型、但缺乏特定任务精细动作数据的通用操作场景。

### 5. 实验分析（精简版）
- **关键结果**：在LIBERO-90任务集上，FRS在11个基准成功率几乎为0的任务中，成功率显著提升；在真实世界DROID任务中，仅需10条FRS生成的成功轨迹即可通过DSBC实现性能质变。
- **优势**：极高的样本效率（只需不到1分钟训练），无需修改通用模型权重，通过简单的语义引导即可在困难任务上实现快速冷启动。
- **局限**：对基础模型的流场结构有依赖，对于非常复杂的语义歧义，仍然需要更强的 reasoner 支持。

### 6. 实用指南
- **开源情况**：作者已在 `flow-reversal-steering.github.io` 开源。
- **迁移注意**：在迁移至其他任务时，需要确保语义输入能转化为机器人动作空间的输入（Cartesian方向或末端执行器姿态）。
- **关键超参数**：实验表明集成步数（Integration steps）取10时，可以在动作保真度与分布内特性之间达到最佳平衡。

### 7. 总结
- **核心思想**：利用流模型的逆向属性，将语义引导转化为高效的潜在动作噪声。
- **速记版pipeline**：
    1. 获取高层语义建议（VLM/人类）。
    2. 执行流反转：将建议动作反向积分求噪声。
    3. 执行流前向去噪：得到分布内的细粒度动作。
    4. 离线/在线学习：利用产生的噪声轨迹优化策略。

**Key Findings:**

- In order to solve or improve on challenging news tasks, we need a way to infer and invoke the appropriate actions from the policy's rich behavioral prior, especially when directly commanding the policy fails.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13675v1)
- [arXiv](https://arxiv.org/abs/2606.13675v1)

---

<a id='2606.13676v1'></a>
## [Modality Forcing for Scalable Spatial Generation](https://arxiv.org/abs/2606.13676v1)

**Authors:** Bardienus Pieter Duisterhof, Deva Ramanan, Jeffrey Ichnowski, Justin Johnson, Keunhong Park

**Published:** 2026-06-11

**Categories:** cs.CV

**Abstract:**

Text-to-image (T2I) models contain rich spatial priors. Synthesizing photorealistic, cluttered scenes requires an understanding of geometry, including perspective and relative scale. Prior works adapt T2I models to leverage this prior for depth prediction, but they require dense depth data and involve complex recipes. We propose Modality Forcing, a simple, scalable post-training recipe for joint image-depth generation using a single DiT trained on sparse depth data. Modality Forcing enables conditional and joint generation of image and depth in any permutation by assigning separate noise levels per modality. Per-modality decoders let us train on sparse, real-world depth and achieve strong, generalizable depth prediction. We further show that Modality Forcing inherits the scalability of T2I pre-training: by training a set of T2I models from scratch (370M to 3.3B parameters), we find that larger models trained on more image data produce more accurate depth. Our strongest model is competitive with state-of-the-art monocular depth estimators and reduces AbsRel by 57% relative to existing joint image-depth generative models. These results provide strong evidence that image generation is a scalable pre-training objective for spatial perception. https://modality-forcing.github.io/

**Analysis:**

### 1. 摘要翻译
文本到图像（T2I）模型蕴含着丰富的空间先验。合成逼真的杂乱场景需要对几何结构（包括透视和相对尺度）的深刻理解。现有工作通过利用此类先验进行深度预测，但通常需要稠密的深度数据，且流程复杂。我们提出了“模态强制”（Modality Forcing），这是一种简单、可扩展的后训练方案，利用单一扩散变换器（DiT）在稀疏深度数据上实现图像与深度的联合生成。通过为不同模态分配独立的噪声水平，模态强制实现了任意组合的条件生成与联合生成。通过分模态解码器，模型能够学习真实的稀疏深度并取得优异的泛化性能。此外，我们发现该方法继承了T2I预训练的可扩展性：通过从头训练一系列T2I模型（370M到3.3B参数），我们发现更大的模型配合更多数据能产生更精确的深度图。我们的最强模型在单目深度估计任务上表现出色，且在联合图像-深度生成任务中，将相对绝对误差（AbsRel）降低了57%。这些结果强力证明了图像生成是空间感知的一种可扩展预训练目标。

---

### 2. 方法动机分析
- **驱动力**：作者旨在证明T2I模型中隐含的丰富空间先验可以通过简单的后训练扩展，从而统一多种空间生成与感知任务。
- **痛点**：现有方法严重依赖密集的RGB-D标注，这在真实世界中难以获取；此外，现有的多模态适配方式（如通过Vae微调或复杂适配器）往往会破坏T2I模型原本强大的生成能力。
- **研究假设**：通过在扩散过程中引入“分模态噪声调度”，模型能够学习模态间的联合分布，且这种能力随模型规模增加呈现可预测的增长趋势。

---

### 3. 方法设计详解
- **核心Pipeline**：
  1. **分模态噪声控制**：这是该方法的精髓。在扩散过程中，RGB图像和深度图各自维护独立的噪声水平（$t_{rgb}, t_{depth}$）。通过在训练时随机采样这些时间步，使模型学会从纯噪声到信号的任意重构。
  2. **稀疏像素空间建模**：不同于传统的先将深度映射到潜在空间（Latent Space），该方法直接在像素空间对深度进行扩散处理，从而能天然地处理稀疏的真实标注数据（未标注区域填充高斯噪声）。
  3. **分模态 timestep 嵌入**：RGB和深度拥有独立的 timestep embedder，并通过“跨流混合模块（Cross-stream mixing）”实现模态间的交叉调制，使模型能在特定模态条件受限时进行补全。
  4. **自蒸馏机制**：为了防止在后训练过程中破坏T2I原有的生成能力，引入了Ldist损失，在训练时强制学生模型的RGB预测轨迹向原始冻结的T2I教师模型对齐。

---

### 4. 方法对比分析
- **本质区别**：不使用额外的大型适配器，而是通过改变噪声调度策略，使单一 DiT 主干同时具备生成与感知能力。
- **创新点**：
    - **模态级扩散调度**：将“控制模态”与“生成模态”的边界模糊化。
    - **自蒸馏方案**：确保了空间感知能力的提升不会以牺牲图像生成质量为代价。
- **适用场景**：适用于资源有限、需要从稀疏数据提取深度信息或实现图到深度/深度到图双向转换的场景。

---

### 5. 实验分析
- **验证方法**：在多项公开数据集（NYUv2, KITTI, ETH3D等）上进行 affine-invariant 深度评估，并与领域内的强基线（如Depth Pro, MoGe-2, JointDiT）进行对比。
- **关键结论**：深度预测精度确实随着T2I底座规模（参数量）和预训练数据量的增加而提升，证明了空间感知是T2I模型的内在属性。
- **局限**：目前的缩放研究上限在3.3B参数，且对深度指令的遵循能力（D2I任务）相比于图像生成仍有提升空间。

---

### 6. 实用指南
- **开源信息**：论文虽提及HuggingFace Demo，但使用时需关注是否开源了权重。
- **关键细节**：$p_{i2d}=0.2, p_{d2i}=0.2$ 的采样概率设置是聚焦于全去噪情况的关键；自蒸馏的权重系数 $\lambda_{hi} > \lambda_{lo}$ 是保持性能的关键超参数。
- **迁移建议**：该思路完全可以扩展至法向量图（Normals）、语义分割图或 3D 点云的联合生成，仅需更换对应的 tokenizer 和 detokenizer。

---

### 7. 总结
- **核心思想**：利用模态独立噪声控制将T2I模型改造为多模态通用空间生成器。
- **速记版pipeline**：
    1. 给RGB和深度分别加不同程度的噪声；
    2. 使用单一DiT对两者进行联合去噪；
    3. 利用自蒸馏防止模型原有的图像能力退化；
    4. 推理时通过调节噪声水平切换任务（I2D/D2I/Joint）。

**Key Findings:**

- We propose Modality Forcing, a simple, scalable post-training recipe for joint image-depth generation using a single DiT trained on sparse depth data.
- Our strongest model is competitive with state-of-the-art monocular depth estimators and reduces AbsRel by 57% relative to existing joint image-depth generative models.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13676v1)
- [arXiv](https://arxiv.org/abs/2606.13676v1)

---

<a id='2606.13674v1'></a>
## [RepWAM: World Action Modeling with Representation Visual-Action Tokenizers](https://arxiv.org/abs/2606.13674v1)

**Authors:** Junke Wang, Qihang Zhang, Shuai Yang, Yiming Luo, Yujun Shen, Zuxuan Wu, Yu-Gang Jiang, Yinghao Xu

**Published:** 2026-06-11

**Categories:** cs.CV

**Abstract:**

This work presents RepWAM, a representation-centric world action model (WAM) built on representation visual-action tokenizers. Existing WAMs typically inherit reconstruction-oriented video tokenizers from pretrained video generation models. Although these tokenizers preserve visual fidelity, pixel reconstruction alone provides limited guidance for learning instruction-following dynamics that connect future prediction with robot control. To address this, we explore a semantic visual-action latent space for representation-centric world action modeling. Specifically, we train a representation visual-action tokenizer that maps visual inputs into aligned visual and latent action tokens. We then pretrain our WAM to jointly model future visual states and the latent actions that connect them under language instructions, followed by adaptation to real robot trajectories for closed-loop manipulation. Experiments on real-world manipulation tasks and simulation benchmarks show that RepWAM delivers strong performance across diverse manipulation settings, while ablations highlight the value of semantic visual-action tokenization over reconstruction-oriented alternatives. These results establish representation visual-action tokenization as a promising foundation for world action models and a step toward generalist robot policies. Code and weights will be available at https://github.com/wdrink/RepWAM.

**Analysis:**

以下是对论文《RepWAM: World Action Modeling with Representation Visual-Action Tokenizers》的深度分析：

### 1. 摘要翻译
本文提出了RepWAM（表征世界动作模型），这是一个基于“表征视觉-动作分词器”的以表征为中心的世界动作模型（WAM）。现有的WAM通常继承自预训练视频生成模型中的重构导向分词器，尽管其视觉保真度高，但纯像素重构对学习机器人控制相关的指令遵循动力学指导有限。为此，我们探索了一个用于以表征为中心的世界动作建模的语义视觉-动作潜空间。具体而言，我们训练了一个表征视觉-动作分词器，将视觉输入映射为对齐的视觉和潜动作标记。随后，我们预训练WAM以联合建模语言指令下的未来视觉状态和潜动作，并将其适配到真实机器人轨迹进行闭环操作。在真实世界操作任务和仿真基准上的实验表明，RepWAM在多种设置下均表现优异，消融实验进一步证实了语义视觉-动作分词相比重构导向方案的优势。这些结果确立了表征视觉-动作分词作为世界动作模型基础的潜力，是迈向通用机器人策略的一步。

### 2. 方法动机分析
*   **驱动力**：作者认为现有WAM的瓶颈在于“表征偏差”，即通用视觉生成的潜空间（关注像素重构）与机器人控制所需的语义空间（关注对象、交互、动力学）存在本质割裂。
*   **痛点**：1. 视觉侧，基于像素重构的Tokenizer分配了过多潜变量容量给背景噪声；2. 动作侧，潜空间与物理动作空间解耦，强依赖逆动力学模型（IDM）强行桥接，导致结构性脱节。
*   **核心直觉**：通过引入一个预训练视觉基础模型（Foundation Model）引导的语义分词器，将“视觉状态”与“动作转换”在同一语义空间内进行联合建模。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **RepViTok（视觉-动作分词器）**：
        *   将视觉观测编码为“语义视觉 Token”。利用教师模型（视觉基础模型）通过特征对齐损失（Alignment Loss）进行语义引导。
        *   基于语义视觉Token，进一步学习“潜动作 Token”。这是一种基于流（Transport Map）的编码：IDM将连续两帧的视觉变化压缩为潜动作Token，FDM（向前动力学模型）通过类似光流的Transport机制重构下一帧。
    2.  **Causal World Action Model（WAM）**：
        *   采用因果扩散Transformer架构。
        *   将语言指令、视觉Token、动作Token组成“Chunk”进行联合预测，通过流匹配（Flow Matching）目标函数进行训练。
    3.  **适配（Adaptation）**：将预训练的通用模型通过微调适配到特定机器人的真实轨迹数据中。
*   **关键公式意义**：公式(3)实现了软传输操作（Transport Map），即不再预测像素改变，而是预测“状态转移”，这使得潜动作具有跨任务的迁移性。

### 4. 方法对比分析
*   **本质区别**：传统方法先分词后处理，RepWAM则是“以动作为导向”地进行视觉分词，动作是视觉状态的转换函数而非外部标签。
*   **创新贡献**：提出将潜动作作为连接前后视觉状态的“转换桥梁”，实现了视觉空间与动作空间的语义深度融合。
*   **适用场景**：高自由度、长时序的复杂操作任务（如文中涉及的长视野拖拽、精密插入）。

### 5. 实验分析
*   **结论**：在RoboTwin 2.0任务中，RepWAM-5B优于π0.5和Motus，消融实验显示“两阶段训练”比联合训练（Joint Pred）能获得更好的动态建模能力。
*   **优势**：在不依赖预训练视频生成模型的情况下，仅通过语义对齐即可达到甚至超越重构导向模型的效果，且大幅减少了对CFG（分类器无关引导）的依赖。
*   **局限**：对长视野任务的性能提升明显，但对极其基础的感知任务（Pick the fruit）在不同模型规模下提升趋于饱和。

### 6. 实用指南
*   **开源信息**：已开源，代码与主页见 `https://github.com/wdrink/RepWAM`。
*   **实现建议**：在训练中，语义对齐权重 `λ_align` 是平衡“生成重构质量”与“语义表达”的关键参数；建议使用 `Muon` 优化器，该方法在Transformer参数效率上表现良好。
*   **迁移迁移**：方法可迁移至任何需要多模态对齐的长短序列动作控制场景。

### 7. 总结
*   **核心思想**：将视觉和动作统一映射至预训练的语义空间中建模。
*   **速记版 Pipeline**：
    1.  **语义编码**：用视觉基础模型约束视觉Token。
    2.  **动作诱导**：通过视觉变化提取潜动作Token。
    3.  **联合建模**：用Transformer同时生成视觉与动作序列。
    4.  **适配微调**：在真实机器人数据上执行闭环控制。

**Key Findings:**

- These results establish representation visual-action tokenization as a promising foundation for world action models and a step toward generalist robot policies.
- Code and weights will be available at https://github.com/wdrink/RepWAM.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13674v1)
- [arXiv](https://arxiv.org/abs/2606.13674v1)

---

<a id='2606.13672v1'></a>
## [$\texttt{WEAVER}$, Better, Faster, Longer: An Effective World Model for Robotic Manipulation](https://arxiv.org/abs/2606.13672v1)

**Authors:** Arnav Kumar Jain, Yilin Wu, Jesse Farebrother, Gokul Swamy, Andrea Bajcsy

**Published:** 2026-06-11

**Categories:** cs.RO

**Abstract:**

The potential impacts of world models (WMs, i.e., learned simulators) on robotics are far-reaching -- policy evaluation, policy improvement, and test-time planning -- all with limited real-world interaction. To unlock these downstream capabilities, a WM needs to jointly satisfy three desiderata: $\textit{(i)}$ fidelity (i.e., producing simulated trajectories that correlate with reality), $\textit{(ii)}$ consistency (i.e., producing simulated trajectories that are coherent over long horizons), and $\textit{(iii)}$ efficiency (i.e., producing simulated trajectories quickly). We propose $\texttt{WEAVER}$ (World Estimation Across Views for Embodied Reasoning): a WM architecture that simultaneously achieves all three desiderata, providing state-of-the-art results on robotic manipulation tasks. $\texttt{WEAVER}$ is a multi-view WM trained to predict future latents and reward values via a flow-matching loss. We distill the key design decisions across model architecture, memory, and prediction objectives required to unlock the kinds of long-horizon dynamic manipulation tasks that have confounded prior world modeling approaches. We apply $\texttt{WEAVER}$ in robotic hardware, demonstrating its effectiveness at policy evaluation ($ρ$=0.870 correlation with real-world success rate), policy improvement (real-world success rate improvement of $38\%$ on top of the $π_{0.5}$ robot foundation model), and test-time planning (real-world success rate improvement of $14\%$ with a $5-10\times$ speedup over prior WMs). $\texttt{WEAVER}$ also demonstrates better performance than prior WMs when evaluated on out-of-distribution scenarios. Code, models, and videos at: https://arnavkj1995.github.io/WEAVER/ .

**Analysis:**

以下是对论文 **WEAVER: An Effective World Model for Robotic Manipulation** 的深度分析：

### 1. 摘要翻译
世界模型（学习到的模拟器）在机器人领域具有深远影响，包括策略评估、策略改进和测试时规划，且所需真实交互极少。为实现这些功能，世界模型需同时满足三点核心期望：(i) 高保真度（模拟轨迹与现实相关联），(ii) 一致性（长时程轨迹预测连贯），以及 (iii) 高效率（快速生成模拟轨迹）。我们提出了 WEAVER（World Estimation Across Views for Embodied Reasoning），一种能够同时满足这三点核心期望的世界模型架构，并在机器人操作任务上提供了最先进的结果。WEAVER 是一个多视图世界模型，通过流匹配（flow-matching）损失预测未来的潜变量和奖励值。我们精炼了模型架构、记忆和预测目标方面的关键设计决策，以解锁以前的世界模型方法无法解决的长程动态操作任务。我们在机器人硬件上应用了 WEAVER，证明了其在策略评估、策略改进和测试时规划方面的有效性。代码、模型和视频详见：https://arnavkj1995.github.io/WEAVER/。

### 2. 方法动机分析
*   **驱动力**：在机器人操作中，世界模型通常面临“ fidelity-efficiency-consistency”的博弈。现有模型（如 Dreamer-v4 或 Ctrl-World）要么因训练从零开始导致 OOD 鲁棒性差，要么因推理速度慢而无法进行实时的测试时规划。
*   **痛点**：长时程操作任务（如 Pour Beans）涉及复杂的接触动力学，现有模型往往难以在保持连贯性的同时进行快速推理。
*   **核心假设**：结合预训练视觉编码器（利用冻结特征保证 OOD 鲁棒性）、多视图记忆机制、流匹配（Flow Matching）目标及基于 Advantage 的测试时策略筛选，可以实现高保真度与实时推理速度的统一。

### 3. 方法设计详解
*   **pipeline 概览**：
    1.  **编码与输入**：使用冻结的 Stable Diffusion 3 VAE 编码器将多视图 RGB 图像转为 latent tokens，并与 proprioceptive state（本体感受状态）连接。
    2.  **动态模型（Dynamics Model）**：基于一个包含空间注意力与因果时间注意力的 2D Transformer，以记忆（memory）和短时历史（history）为条件，通过流匹配（Flow Matching）目标 autoregressively 生成 $h$ 步 latent 轨迹。
    3.  **价值评估**：引入轻量级奖励头（Reward Head）和 Critic 网络直接在潜空间进行评估，无需解码出图像或调用缓慢的 VLM 作为 judge。
    4.  **推理加速**：利用 KV caching 减少前向推理开销，通过余弦噪声调度器（Cosine Schedule）优化采样质量，并通过 ReFlow 蒸馏实现少步数生成。
*   **核心模块**：
    *   **Sparse Memory & Short-term History**：通过存储稀疏的远期 latents 配合近期的连续历史，在减少显存占用的同时保持长程一致性。
    *   **Diffusion Forcing**：利用不同时间步采样噪声，提升模型的训练稳定性与生成的一致性。

### 4. 方法对比分析
*   **核心区别**：与 Ctrl-World 等相比，WEAVER 采用了更轻量高效的流匹配目标和预训练编码器，摆脱了对外部 VLM 的强依赖，实现了 5-10 倍的 inference 加速。
*   **创新贡献**：将“Diffusion Forcing”和“Flow Matching”引入机器人世界模型，平衡了生成质量与速度，且引入了轻量级 Reward/Critic head 实现高效的 Best-of-N 规划。

### 5. 实验分析
*   **验证方法**：在 DROID 数据集和 5 个实际硬件操作任务上进行测试。
*   **关键结论**：在策略评估中达到 $\rho=0.870$ 的相关性，使 π0.5 策略的成功率提升 38%，并大幅降低了规划延迟。
*   **局限性**：仍受限于视觉传感的局部观测（未引入触觉），在处理复杂颗粒物（如 Beans）时偶尔会出现长时程误差累积。

### 6. 实用指南
*   **开源情况**：已开源（项目主页提供代码）。
*   **实现建议**：训练关键点在于使用预训练的编码器（无需微调）以保证表示能力；使用 SPRINT blocks 可以有效丢弃 patch tokens 提升效率。
*   **迁移建议**：该架构可以直接适配任何基于视觉的机器人控制任务，只需更换对应的 Reward 标定逻辑即可。

### 7. 总结
*   **核心思想**：利用流匹配目标与预训练特征实现高效、连贯的长程世界模型预测。
*   **速记版 Pipeline**：
    1.  多视图图像与本体状态编码。
    2.  利用记忆与历史的 Transformer 生成未来轨迹。
    3.  潜空间内通过 Reward/Critic Head 直接评价。
    4.  筛选最优动作轨迹执行或蒸馏提升策略。

**Key Findings:**

- We propose $\texttt{WEAVER}$ (World Estimation Across Views for Embodied Reasoning): a WM architecture that simultaneously achieves all three desiderata, providing state-of-the-art results on robotic manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13672v1)
- [arXiv](https://arxiv.org/abs/2606.13672v1)

---

<a id='2606.13652v1'></a>
## [World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible](https://arxiv.org/abs/2606.13652v1)

**Authors:** Hao Zhang, Mohamed El Banani, Jen-Hao Cheng, Paul Zhang, Yi Hua, Ben Mildenhall, Christoph Lassner, Narendra Ahuja, Gengshan Yang

**Published:** 2026-06-11

**Categories:** cs.CV, cs.GR

**Abstract:**

Image-to-3D methods often trade off faithfulness and completeness: depth estimators are anchored to input pixels but stop at the visible surface, while image-to-3D models generate complete shapes that are often misaligned with the input. We introduce World Tracing, a generative pixel-aligned geometry representation that predicts 3D points aligned with observed pixels while completing geometry beyond the visible surface. For each input pixel, World Tracing predicts an ordered stack of camera-space 3D points, where the first layer represents the visible surface and subsequent layers represent front-to-back intersections with occluded surfaces. We instantiate this representation with a world-tracing diffusion transformer, WT-DiT, which treats multiple geometry layers as separate denoising tokens coupled through factorized and global attention. WT-DiT is trained with pixel-space flow matching and a mixed noise schedule that balances visible-surface reconstruction with occluded-geometry generation. World Tracing achieves strong performance on visible-surface reconstruction and complete geometry generation across object, scene, and dynamic benchmarks, outperforming both depth predictors and image-to-3D generators. It also preserves 2D-to-3D correspondence, enabling text-driven 3D scene editing, geometry-conditioned novel-view video synthesis, and training-free integration with textured-mesh generators.

**Analysis:**

这是一份关于《World Tracing: Generative Pixel-Aligned Geometry Beyond the Visible》的深度技术分析。

---

### 1. 摘要翻译
图像到3D模型的方法通常在真实性（Faithfulness）和完整性（Completeness）之间做权衡：深度估计器锚定于输入像素，但仅能捕获可见表面；而现有的图像到3D模型生成的完整形状往往与输入未对齐。我们引入“世界追踪”（World Tracing），一种生成的像素对齐几何表示，它在预测与观测像素对齐的3D点的同时，完成了可见表面之外的几何结构。对于每个输入像素，World Tracing 预测一个有序的相机空间3D点栈，其中第一层代表可见表面，后续层代表与遮挡表面的前向交叉。我们使用世界追踪扩散变换器（WT-DiT）实例化该表示，它将多个几何层视为通过因子化和全局注意力耦合的独立去噪令牌。

### 2. 方法动机分析
*   **驱动力**：解决现有3D生成中“完全几何”与“像素对齐”不可兼得的矛盾，实现一个既能忠实重构可见表面，又能合理补全遮挡部分的统一框架。
*   **现有方法痛点**：
    *   **深度预测**：限于可见表面，缺乏遮挡部分的几何补全能力。
    *   **3D生成（Canonical-frame）**：通常生成在规范坐标系下的物体，丢失了输入图像的像素对齐关系，下游任务（如编辑、插入）难以实现。
*   **研究假设**：几何补全应视为像素射线上的“多层有序交叉”问题，通过在输入像素网格上直接回归这些层，可以同时保留视觉特征与3D空间一致性。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **输入处理**：输入RGBA图像，利用预训练的MoGe编码器提取特征。
    2.  **几何表示**：将几何建模为 $L$ 层（$L=6$）的点图（Pointmap），每个像素对应一条射线上的 $L$ 个深度交叉点。
    3.  **WT-DiT 网络**：
        *   **Tokenization**：将几何层视为 Patch 化的 Token，与提取的图像特征融合。
        *   **三向注意力（Three-way Attention）**：
            *   *Layer-wise*：在同一层内做2D空间注意力。
            *   *Ray-wise*：在同一像素的 $L$ 层间做注意力，保证前向深度顺序。
            *   *Global*：获取全局场景上下文。
        *   **输出**：经线性投影还原为完整的 XYZ 点栈。
    4.  **深度填充（Depth-filling）**：通过前向填充（Forward-filling）策略，将无交点的层填补为最近的有效层，将该任务转化为密集的XYZ回归，规避了预测稀疏掩码带来的分类不平衡问题。

### 4. 方法对比分析
*   **本质区别**：不使用掩码头，通过“前向填充”将稀疏的遮挡问题转化为密集的回归问题，维持了与输入图像的严格像素对应。
*   **创新贡献**：提出了一种统一的相机空间多层几何张量表示，实现了高保真可见表面与合理遮挡补全的有机结合。
*   **适用场景**：单图3D重构、场景编辑、3D物体插入、 novel-view 视频合成。

### 5. 实验分析（精简版）
*   **验证方法**：在公开 benchmark（如3D-FRONT、DAVIS）上，与单目深度预测器和现有的生成式3D模型（如TRELLIS）对比。
*   **关键结论**：在保持可见表面高保真度（MAE/RMSE领先）的同时，在 Chamfer Distance 等完整几何指标上也显著优于现有基线。
*   **优势**：训练高效，支持多 regimes（物体、场景、动态）的统一建模。
*   **局限**：对极度多孔（highly perforated）物体的几何表达受固定层数（$L=6$）限制。

### 6. 实用指南
*   **开源情况**：提供 Project Page，后续可关注代码更新。
*   **迁移细节**：其核心在于“多层回归”和“深度填充”，若迁移到其他任务，建议优先复用该几何表达方式，无需显式预测遮挡掩码。
*   **训练建议**：必须使用训练噪声课程（Training noise curriculum），区分可见层与隐藏层的采样概率。

### 7. 总结
*   **核心思想**：将几何视为像素射线上的有序多层回归。
*   **速记版 Pipeline**：
    1. 特征编码：提取图像视觉特征。
    2. 分层点堆：预测射线上的6层深度点。
    3. 深度填充：处理缺失几何，构建密集回归目标。
    4. 多向协同：利用三向注意力确保层间及全局一致性。
    5. 逆向重构：结合相机参数还原3D空间点云。

**Key Findings:**

- We introduce World Tracing, a generative pixel-aligned geometry representation that predicts 3D points aligned with observed pixels while completing geometry beyond the visible surface.
- It also preserves 2D-to-3D correspondence, enabling text-driven 3D scene editing, geometry-conditioned novel-view video synthesis, and training-free integration with textured-mesh generators.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13652v1)
- [arXiv](https://arxiv.org/abs/2606.13652v1)

---

<a id='2606.13644v1'></a>
## [Surflo: Consistent 3D Surface Flow Model with Global State](https://arxiv.org/abs/2606.13644v1)

**Authors:** Antoine Guédon, Shu Nakamura, Nicolas Dufour, Jiahui Lei, Ko Nishino, Angjoo Kanazawa

**Published:** 2026-06-11

**Categories:** cs.CV

**Abstract:**

Geometry is invariant to viewpoint, which makes any collection of images a redundant encoding of a single 3D state. Existing feed-forward reconstruction models fail to exploit this: per-view methods emit overlapping, unaligned pointmaps that grow linearly with input count, while global-latent methods commit to a fixed, low-resolution output. We introduce Surflo, which compresses a variable number of unposed RGB views into K latent tokens-one global state-and decodes oriented 3D surface points by independently transporting them from noise onto the surface via flow matching. This frees the output from any fixed grid or token budget: the same latent yields from a few thousand to a million points in a single forward pass. To suppress the local inconsistencies inherent to independent per-point decoding, an inference-time guidance term correlates nearby points by injecting a photometric gradient during ODE integration. Surflo matches or surpasses feed-forward baselines on surface metrics, runs an order of magnitude faster than optimization-based methods that require hundreds of views, and is the only feed-forward approach to combine a global latent with arbitrary-resolution decoding.

**Analysis:**

以下是对《Surflo: Consistent 3D Surface Flow Model with Global State》的深度分析：

### 1. 摘要翻译
几何结构对视角具有不变性，这意味着任何图像集合都是对单一3D状态的冗余编码。现有的前馈重建模型未能有效利用这一点：基于单视角的方法会生成重叠、未对齐的点云，且其计算量随视角增加呈线性增长；而基于全局隐变量的方法则受限于固定的低分辨率输出。我们引入了Surflo，它将任意数量的未校准RGB视图压缩为一组 $K$ 个潜在标记（即“全局状态”），并通过流匹配（flow matching）将查询点从噪声独立输运至3D表面。该方法打破了固定网格或标记预算的限制，单次前向传播即可生成从几千到上百万个点。为解决独立点解码带来的不一致性，我们提出了一种推理时引导机制，通过在ODE积分过程中注入光度梯度来关联相邻点。Surflo在表面重建指标上匹配或超越了现有前馈基线，运行速度比依赖数百张图像的优化方法快一个数量级，是目前唯一能够实现全局潜变量与任意分辨率解码相结合的前馈模型。

### 2. 方法动机分析
- **驱动力**：利用3D几何的视角不变性，构建一种视角无关的全局场景表示，解决传统方法在处理多视图时产生的冗余计算和不一致问题。
- **痛点**：
    - **前馈法（Per-view）**：点云随输入视角线性增长，且由于缺乏全局一致性，融合为mesh时会出现伪影和重叠。
    - **全局隐变量法**：通常输出受限于固定的点云预算或网格分辨率，灵活性差。
- **假设**：通过将多视角信息压缩为统一的全局潜变量（$z$），并利用流匹配（Flow Matching）实现从噪声到表面的动态输运，可以解耦输入视角数量与重建分辨率。

### 3. 方法设计详解
- **流程pipeline**：
    1. **全局编码 (Encoder)**：使用冻结的VGGT backbone提取 patch token，结合3D位置编码，通过Perceiver架构将其压缩为固定 $K=128$ 个 latent tokens ($z$)。
    2. **流匹配解码 (Decoder)**：对于任意查询点 $x_t \in \mathbb{R}^3 \times \mathbb{S}^2$，根据潜在状态 $z$ 和当前时间 $t$，预测其向表面移动的“速度”向量 $v_\theta$。
    3. **引导式积分 (Guided ODE)**：在ODE求解的最后阶段（$t \ge 0.95$），引入基于渲染的损失函数（Render Loss），通过反向传播调整点的位置，增强空间一致性。
- **核心逻辑**：将重建建模为：从噪声分布出发，依据全局状态定义的“流场”，将点“推”向真实物体表面的过程。

### 4. 方法对比分析
- **本质区别**：Surflo实现了**输入视角可变（Encoder）**与**输出点数/分辨率可变（Decoder）**的完全解耦。
- **创新贡献**：
    - 引入了流匹配架构来处理非固定分辨率的3D重建。
    - 提出了“通信引导（Communication via Guidance）”机制，通过微分渲染梯度在推理时动态修正独立采样点之间的不一致性。
    - 辅助贡献了包含10.5K个场景的带Watertight网格的DL3DV数据集。

### 5. 实验分析
- **验证方法**：在DL3DV、Tanks & Temples等4个数据集上对比了基于点映射、基于固定潜变量及优化类方法。
- **关键结论**：在Chamfer Distance（CD）和F1-score上显著优于其它前馈模型，且推理速度比Gaussian Wrapping等优化法快10倍以上。
- **局限**：对极少数视角下的复杂场景鲁棒性依赖于预训练backbone的性能；生成过程仍需ODE积分（尽管比优化快，但并非完全瞬时）。

### 6. 实用指南
- **开源情况**：作者承诺随论文发布代码。
- **实现细节**：关键超参数为 $K=128$，$M=32$（梯度下降步数），需注意ODE求解器的步长设置。
- **迁移可能**：该架构（Encoder+流匹配Decoder）可直接迁移至医疗图像3D重建、机器人三维环境感知等需要灵活分辨率重建的任务。

### 7. 总结
- **核心思想**：通过全局隐空间流场映射，实现任意视角输入的灵活几何重建。
- **速记版pipeline**：
    1. 冻结骨干网提取多视角特征；
    2. 通过注意力机制压缩为固定潜变量；
    3. 流匹配器将查询点向表面输运；
    4. 推理阶段引入渲染梯度修正位置。

**Key Findings:**

- We introduce Surflo, which compresses a variable number of unposed RGB views into K latent tokens-one global state-and decodes oriented 3D surface points by independently transporting them from noise onto the surface via flow matching.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13644v1)
- [arXiv](https://arxiv.org/abs/2606.13644v1)

---

<a id='2606.13515v1'></a>
## [MaskWAM: Unifying Mask Prompting and Prediction for World-Action Models](https://arxiv.org/abs/2606.13515v1)

**Authors:** Hanyang Yu, Haitao Lin, Jingbo Zhang, Wenyao Zhang, Chenghao Gu, Heng Li, Ping Tan

**Published:** 2026-06-11

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

World Action Models (WAMs) present a promising paradigm for robotic control via video prediction. However, current WAMs suffer from fundamental spatial bottlenecks: standard text inputs introduce referential ambiguity in cluttered scenes, while unstructured RGB predictions lack semantic grounding and remain biased by task-irrelevant backgrounds. To overcome these limitations, we introduce MaskWAM, an object-centric world-action model. By jointly integrating masks as both explicit inputs and predictions via a unified Mixture of Transformers (MoT), MaskWAM unlocks robust policy generalization. This design provides two key benefits: (1) predicting future masks yields object-centric semantic supervision that suppresses visual noise, significantly enhancing even standard text-conditioned WAMs; and (2) coupling this predictive supervision with first-frame visual prompts, such as target object masks, establishes a precise spatial anchor that substantially reduces language ambiguity. Crucially, as WAMs are inherently vision-driven architectures, direct mask conditioning yields substantially stronger guidance than text alone, establishing a precise and robust paradigm for manipulating unseen objects. Evaluations on LIBERO, RoboTwin, and real-world tasks demonstrate that MaskWAM significantly outperforms baselines in both language-clear and language-ambiguous tasks.

**Analysis:**

### 1. 摘要翻译
世界动作模型（WAMs）为机器人控制提供了一种基于视频预测的新兴范式。然而，当前的WAMs在空间表现上存在瓶颈：标准文本输入在复杂场景中存在指代歧义，而无结构化的RGB预测缺乏语义基础，常受任务无关背景的干扰。为此，我们引入了MaskWAM，一种以对象为中心的世界动作模型。通过利用统一的变换器混合（MoT）架构将掩码作为显式输入和预测目标，MaskWAM实现了稳健的策略泛化。该设计具有两大优势：(1) 通过预测未来掩码提供对象级语义监督，有效抑制视觉噪声；(2) 将预测监督与第一帧视觉提示（如目标对象掩码）耦合，建立了精确的空间锚点，显著降低了语言指代歧义。实验表明，MaskWAM在LIBERO、RoboTwin及真实世界任务中均显著优于现有基线。

### 2. 方法动机分析
*   **驱动力**：解决WAMs在处理复杂空间信息和精细任务时的“定位模糊”问题，提升策略对目标对象的专注度。
*   **现有方法痛点**：现有的基于RGB的视频预测往往忽略了任务相关的核心对象（目标与背景纠缠），且文本指令在描述复杂空间关系或区分相似物体时极其匮乏（缺乏精确的空间锚点）。
*   **研究假设**：如果模型能够显式地预测并编码任务相关目标的掩码，并辅以视觉掩码作为提示，模型就能获得更强烈的空间归纳偏置，从而实现更精准的控制。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入处理**：当前帧RGB $I_0$、状态 $s_0$、文本指令 $\ell$ 以及可选的初始掩码 $M_0$ 被送入系统。
    2.  **编码与融合**：利用冻结的视频VAE编码RGB和掩码（将掩码渲染为RGB格式以兼容网络），并在潜在空间将两者按通道拼接。
    3.  **统一预测（MoT）**：采用 Mixture of Transformers (MoT) 架构，联合预测未来RGB、未来掩码和动作 chunk。
    4.  **解耦流匹配**：训练过程中采用两套解耦的噪声调度器 $\tau_v$（视觉）和 $\tau_a$（动作），确保时空对齐的同时优化策略。
*   **关键公式与逻辑**：
    *   $z = [z_v; z_m]$：将RGB潜在向量与掩码潜在向量在通道维度拼接。通过预测掩码，模型被迫从像素空间转向结构化的对象空间，充当了“空间正则化器”。
*   **训练策略**：引入“掩码丢弃（Mask Dropout）”策略，训练时以50%概率置零初始掩码，使模型既能处理有提示的复杂场景，也能在纯文本引导下运行。

### 4. 方法对比分析
*   **本质区别**：从传统的“单纯视频预测”转变为“视频+掩码协同预测”。将掩码从被动辅助信息提升为与RGB地位对等的预测目标。
*   **创新贡献**：提出了一种统一的掩码提示与预测框架，证明了在世界模型中引入显式空间表示优于单纯的坐标文本提示。
*   **适用场景**：高精度抓取、多物体区分、复杂背景下的杂乱环境操作。

### 5. 实验分析
*   **关键结论**：在LIBERO benchmark上取得98.4% SOTA成功率；在真实机器人实验中，面对语言歧义任务（如从16个红化妆品中抓取目标），表现优于基线33.2%。
*   **主要优势**：极强的抗干扰能力（背景噪声）和零样本泛化能力（面对未见对象和照明变化）。
*   **主要局限**：高度依赖Mask提取（SAM3等），且目前在大规模多任务预训练上的计算开销仍有待优化。

### 6. 实用指南
*   **实现细节**：VAE保持冻结，仅微调扩散骨干；掩码输入采用渲染后的3通道图像。
*   **迁移可能**：掩码预测模块可轻松插入现有的Video-Diffusion策略模型中作为附加分支，适用于任何需要精确定位的机器人操作任务。

### 7. 总结
*   **核心思想**：通过联合预测未来掩码与RGB，显式增强模型对任务关键区域的空间感知。
*   **速记版pipeline**：
    1. 渲染掩码作为视觉锚点；
    2. 将RGB和掩码拼接编码；
    3. 联合预测未来视频帧与掩码；
    4. 基于解耦噪声调度优化控制策略。

**Key Findings:**

- To overcome these limitations, we introduce MaskWAM, an object-centric world-action model.
- Evaluations on LIBERO, RoboTwin, and real-world tasks demonstrate that MaskWAM significantly outperforms baselines in both language-clear and language-ambiguous tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13515v1)
- [arXiv](https://arxiv.org/abs/2606.13515v1)

---

<a id='2606.13503v1'></a>
## [Heterogeneous LiDAR Early Fusion and Learned Re-Ranking Strategy for Robust Long-Term Place Recognition in Unstructured Environments](https://arxiv.org/abs/2606.13503v1)

**Authors:** Judith Vilella-Cantos, Juan José Cabrera, Mónica Ballesta, David Valiente, Luis Payá

**Published:** 2026-06-11

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Robust localization in unstructured environments, such as agricultural fields, is a critical challenge for autonomous systems. LiDAR sensors provide detailed 3D information about the environment and are invariant to lighting conditions. For this reason, LiDAR-based place recognition methods have gained significant attention. In this paper, we propose MinkUNeXt-VINE++, a novel approach that combines early fusion of heterogeneous LiDAR data from two sensors (Livox Mid-360 and Velodyne VLP-16) and a learned re-ranking strategy in inference time. This fusion leverages the strengths of each sensor to provide a more comprehensive representation of the environment. Additionally, the re-ranking approach is particularly important in repetitive environments, such as vineyards, as finding true positives is a major challenge. We evaluated our approach using the TEMPO-VINE dataset, which provides heterogeneous LiDAR data in vineyard environments across different phenological stages. Our results demonstrate that MinkUNeXt-VINE++ significantly improves place recognition performance compared to single-sensor approaches and state-of-the-art methods. MinkUNeXt-VINE++ achieves a 20% improvement in the Recall@1 metric compared to single-sensor approaches, and +30% including re-ranking. The code of our method is publicly available for reproduction.

**Analysis:**

## 1. 摘要翻译
本文针对农业环境（如葡萄园）中自主系统在长期定位方面所面临的严峻挑战，提出了一种名为 **MinkUNeXt-VINE++** 的新颖方法。该方法结合了两种异构LiDAR传感器（Livox Mid-360和Velodyne VLP-16）的早期融合策略，并引入了推理阶段的自学习重排序策略。融合策略有效利用了各传感器的优势，构建了更全面的环境表征；重排序策略则在重复性环境（如葡萄园）中显著提升了寻找真阳性的能力。在TEMPO-VINE数据集上的实验结果表明，该方法在Recall@1指标上较单传感器方法提高了20%，加上重排序后提升了30%，显著优于现有最先进（SOTA）方法。

## 2. 方法动机分析
- **驱动力**：在非结构化农业环境中，LiDAR点云缺乏显著几何特征，且环境高度动态（植被生长/天气），导致传统的单传感器PLACE识别方法性能较差。
- **现有痛点**：单传感器难以同时兼顾近距离细节与远距离覆盖。且在重复性高的环境中，初级检索阶段易产生大量误报，难以精确定位。
- **研究假设**：通过融合异构LiDAR数据（优势互补）并增加一个轻量级的后处理重排序头（基于学习的置信度精调），可以显著提升PLACE识别的鲁棒性和精度。

## 3. 方法设计详解
### 流程总结
1. **异构融合**：
   - 对两传感器点云进行下采样至统一分辨率，并转换至共同坐标系。
   - 设定10米距离阈值：近距离使用Livox（细节丰富），远距离使用Velodyne（扫描模式更均匀，适合长距离）。
   - 组合筛选后的点云，形成一个均匀、密集的融合点云。
2. **基础特征提取**：将融合点云送入预训练的MinkUNeXt-VINE主干网络获取全局描述子。
3. **学习型重排序（Re-ranking）**：
   - 输入：主干网络检索出的Top-K候选的查询特征（$q$）与数据库描述子（$d$）。
   - MLP结构：包含3层线性层与ReLU激活函数，输入维度384（$q, d$拼接）。
   - 训练：使用二元交叉熵（BCE）损失函数，以真实的匹配关系作为标签，优化重排序评分。

### 算法与设计关键
- **多描述子输入**：文中对比了仅输入$(q, d)$以及加入element-wise product $(q * d)$和difference $(q - d)$的效果。虽然包含差异项在某些极端案例有提升，但综合计算开销（FLOPs/MACs），最终选定$(q, d)$作为最优配置。

## 4. 方法对比分析
- **本质区别**：与传统依赖单一LiDAR的方法不同，它采用早期物理级融合，在数据预处理阶段即利用传感器互补性，而非简单的特征层级拼接。
- **创新贡献**：
  1. 首个针对非结构化环境异构LiDAR的早期融合框架。
  2. 极度轻量化的重排序头，在不增加显著计算负担前提下实现Recall@1的巨大跃升。
- **适用场景**：对LiDAR点云密度有要求、环境具有高度重复性（如农场、隧道、长廊）的定位场景。

## 5. 实验分析
- **关键结论**：在TEMPO-VINE数据集上，融合策略使Recall@1提升约34.56%，叠加重排序后进一步提升。证明该方法在处理植被变化和外观高度相似环境时具有显著鲁棒性。
- **主要优势**：计算效率高，适合实时部署；对环境季节性变化（cross-season）有良好的泛化能力。
- **局限性**：在极度退化或与训练数据差异巨大的全新场景中，仍存在性能衰减。

## 6. 实用指南
- **开源情况**：已开源，参考 https://github.com/JudithV/MinkUNeXt-VINE_plusplus。
- **迁移建议**：
  - 此方法迁移到其他任务的关键在于**传感器外参标定**以及针对特定场景的**距离阈值重调**。
  - 重排序头可以直接作为现有任何PLACE识别方法的“外挂插件”，只需预训练一个MLP即可。

## 7. 总结
- **核心思想**：融合多传感器硬件优势，配合轻量级特征重排序头实现精准定位。
- **速记版pipeline**：
  1. 传感器物理融合（按距离阈值筛选）。
  2. 主干网络特征检索。
  3. 轻量化MLP重排序评分。

**Key Findings:**

- In this paper, we propose MinkUNeXt-VINE++, a novel approach that combines early fusion of heterogeneous LiDAR data from two sensors (Livox Mid-360 and Velodyne VLP-16) and a learned re-ranking strategy in inference time.
- We evaluated our approach using the TEMPO-VINE dataset, which provides heterogeneous LiDAR data in vineyard environments across different phenological stages.
- Our results demonstrate that MinkUNeXt-VINE++ significantly improves place recognition performance compared to single-sensor approaches and state-of-the-art methods.
- The code of our method is publicly available for reproduction.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.13503v1)
- [arXiv](https://arxiv.org/abs/2606.13503v1)

---

