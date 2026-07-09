time: 20260709

# Arxiv Computer Vision Papers - 2026-07-09

## Executive Summary

以下是对2026年7月8日Arxiv计算机视觉领域10篇论文的执行摘要，旨在帮助研究人员快速把握最新发展。

## 1. 主要主题与趋势

本期论文高度集中于**具身智能与机器人操作**，同时涵盖自动驾驶、3D场景理解与交互生成。核心趋势包括：

- **基础模型向触觉与多模态融合**：从视觉语言模型扩展到触觉（TouchWorld）和机器人状态（GeoProp），强调感知与执行的深度耦合。
- **仿真与数据合成驱动**：多篇论文利用世界引擎（EmbodiedGen V2）、CARLA仿真（CARLA-GS）生成极端/角落案例，解决真实数据稀缺问题。
- **可组合性与即时适应性**：运动生成（Compositional Motion Generation）和扩散策略（PriGo）强调从演示中解耦分解，并在测试时快速调整。
- **不确定性建模与鲁棒性**：PUF在3D场景图生成中引入不确定性感知融合，提升在线系统的可靠性。
- **大规模预训练在具身任务中迁移**：采用混合专家（MoE）视频预训练（Scaling Mixture-of-Experts）为通用操作提供视觉表征。

## 2. 特别突出的创新性论文

- **TouchWorld**：首个预测性与反应性结合的触觉基础模型，直接应用于灵巧操作，将触觉融入具身智能基座，具有里程碑意义。
- **CARLA-GS**：在自动驾驶角落案例合成中解耦表征、推理与物理仿真，为安全评估提供可控、多样化的长尾场景生成框架。
- **EmbodiedGen V2**：提出智能体化的3D世界引擎，能自动生成仿真就绪的场景与交互逻辑，显著降低具身AI训练的环境构建成本。
- **PUF (Plug-and-Play Uncertainty-Aware Fusion)**：在线3D场景图生成中首次引入即插即用的不确定性融合，提升动态环境下语义与拓扑关系的鲁棒性。
- **PriGo**：在测试时利用原始引导（primitive guidance）动态调整扩散/流策略，实现自适应机器人操作，为闭环策略提供了轻量级解决方案。

## 3. 新兴研究方向与技术

- **触觉基础模型**：将触觉作为与视觉、语言并列的感知模态，用于精确操作与物理交互，预计将成为下一个研究热点。
- **混合专家视频预训练**：在具身智能中使用MoE架构扩展视频预训练规模，平衡计算效率与表征能力，推动通用操作智能体。
- **VR+LLM人机交互**：结合虚拟现实与大语言模型驱动的人形机器人，探索沉浸式社会交互的新范式，开启“具身聊天”方向。
- **测试时引导（Test-Time Guidance）**：不修改模型权重，而在推理时通过先验引导调整策略输出，适应新任务或新环境，适合部署在资源受限平台。
- **不确定性驱动的在线3D场景理解**：将不确定性作为融合权重而非后处理手段，使系统更适应实时、变化的环境。

## 4. 建议全文阅读的论文

- **TouchWorld**：如果你关注灵巧操作或触觉感知，这是必读论文，其预测+反应框架极具启发性。
- **CARLA-GS**：对于自动驾驶安全场景生成的研究人员，该文提供了清晰的可解耦架构与物理仿真验证。
- **EmbodiedGen V2**：具身AI研究者应关注其世界引擎设计，尤其是如何自动化生成训练环境。
- **PUF**：从事在线3D场景理解或机器人导航的读者，其不确定性融合方法可直接借鉴。
- **PriGo**：对扩散策略在机器人操作中的实际应用感兴趣者，此文展示了测试时调优的简洁高效性。

---

总体而言，本期论文标志着**从感知到执行、从静态到动态、从专用到通用**的明显转变，触觉、不确定性建模与即时适应性将是未来数月的热点方向。

---

## Table of Contents

1. [TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation](#2607.07287v1)
2. [GeoProp: Grounding Robot State in Vision for Generalist Manipulation](#2607.07101v1)
3. [Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence](#2607.07675v1)
4. [CARLA-GS: Decoupling Representation, Reasoning, and Physics Simulation for Autonomous Driving Corner-Case Synthesis](#2607.07601v1)
5. [EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI](#2607.07459v1)
6. [Immersive Social Interaction with VR and LLM-Assisted Humanoids](#2607.07430v1)
7. [Behavior Foundations for Quadruped Robots: ABot-C0 Technical Report](#2607.07370v1)
8. [PUF: Plug-and-Play Uncertainty-Aware Fusion for Online 3D Scene Graph Generation](#2607.07170v1)
9. [Compositional Motion Generation from Demonstration with Object-Centric Neural Fields](#2607.07129v1)
10. [PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation](#2607.07076v1)

---

## Papers

<a id='2607.07287v1'></a>
## [TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation](https://arxiv.org/abs/2607.07287v1)

**Authors:** Jianyi Zhou, Feiyang Hong, Yunhao Li, Yicheng Zhao, Yongjue Cen, Zirui Liu, Jiakang Huang, Zirui Chen, Ruiyang Zhang, Weizhuo Zhu, Xuhua Song, Shuo Yang

**Published:** 2026-07-08

**Categories:** cs.RO

**Abstract:**

Dexterous manipulation in everyday environments requires both anticipation and reaction: a robot must predict how contact should evolve while rapidly correcting local errors caused by slip, misalignment, unstable grasping, or force mismatch. Vision and language provide semantic and geometric guidance, but they cannot reliably reveal hidden contact states such as force, slip, and contact stability. Although tactile sensing exposes these physical cues, most existing policies treat touch as a low-frequency observation stream within a monolithic action model, coupling slow task reasoning, action generation, and fast contact feedback in a single loop. We introduce TouchWorld, a predictive-and-reactive tactile foundation model for dexterous manipulation. TouchWorld uses a hierarchical policy that separates vision-language subtask planning, tactile world-model prediction, visuo-tactile goal-conditioned action generation, and high-frequency tactile residual refinement. A High-Level Planning Layer produces executable subtasks and predicts tactile subgoals; a Visuo-Tactile Goal-Conditioned Policy generates nominal action chunks; and a Tactile-Conditioned Refinement Policy performs online residual correction using recent tactile and proprioceptive feedback. By using touch as both a predictive contact reference and a fast feedback signal, TouchWorld preserves the semantic generalization of vision-language-action policies while improving local contact adaptation. Across six long-horizon and contact-rich dexterous manipulation tasks, TouchWorld achieves 65.0% success in the clean setting and 53.7% success under human perturbations, outperforming the strongest baseline by 15.7 and 18.5 percentage points, respectively.

**Analysis:**

以下是针对《TouchWorld: A Predictive and Reactive Tactile Foundation Model for Dexterous Manipulation》的深度技术分析：

### 1. 摘要翻译
灵巧操作需要预期（预测接触演变）与反应（纠正滑动、错位等局部误差）并重。视觉和语言虽能提供语义指引，但无法可靠揭示力、滑移及接触稳定性等隐藏状态。现有方法通常将触觉作为低频观测流耦合进单一动作模型中，导致语义推理、动作生成与快速触觉反馈在同一环路中竞争资源，难以适应接触密集型操作。本文提出 TouchWorld，一个分层触觉基础模型。它通过三个层级解耦任务：高层规划层负责语义拆解与触觉目标预测；视觉-触觉目标条件策略生成名义动作块；触觉调节细化策略进行高频局部残差修正。实验表明，TouchWorld 在六项长期、接触密集型任务中表现显著优于基线。

### 2. 方法动机分析
*   **驱动力**：触觉本质上具有多尺度特征：语义层（任务完成）慢，局部交互（滑动、接触）极快。
*   **现有方法痛点**：单体策略模型将上述不同频率的反馈强行塞入一个控制环路，导致模型在处理“语义逻辑”的同时无法及时响应“局部接触突变”，引起灵巧操作中的失稳。
*   **研究假设**：通过显式的时间尺度解耦（Separation of Time Scales），将任务规划、动作生成与局部残差细化拆分，能够同时获得语义泛化能力与高频触觉鲁棒性。

### 3. 方法设计详解
TouchWorld 采用三阶段多频率控制架构：
*   **高层规划层 (1 Hz)**：
    *   **Subtask Planner**：基于 Qwen3-VL，将长任务指令拆解为可执行子任务。
    *   **Tactile World Model**：利用 Wan2.2-TI2V 架构，预测当前子任务对应的“未来视觉-触觉目标（Subgoals）”。这为下层策略提供了“接触参考基准”。
*   **视觉-触觉目标条件策略层 (10 Hz)**：
    *   接收任务状态与预测的 tactile subgoals，使用扩散 Transformer 生成**名义动作块 (Nominal Action Chunks)**。它处理的是全局语义和几何规划，不直接负责微小误差。
*   **触觉调节细化层 (30 Hz)**：
    *   **Tactile Residual Transformer (TRT)**：这是论文的核心创新点。它接收名义动作块与高频触觉流，实时输出**残差 (Residual)**。动作执行 = 名义动作 + 残差修正。该层仅在“触觉敏感子空间”内操作，不改变整体任务目标。

### 4. 方法对比分析
*   **本质区别**：与传统将触觉作为“额外输入向量”的 VLA 模型不同，TouchWorld 将触觉引入了预测（通过 World Model 生成目标）和反馈（通过 Refinement Layer 进行局部修正）两个独立循环。
*   **创新贡献**：提出了一种“残差化触觉控制”范式，有效规避了触觉数据高采样率与 VLA 推理低效率的冲突。

### 5. 实验分析（精简版）
*   **关键结论**：在人类干扰环境下，TouchWorld 成功率比最强基线提升 18.5%，证明了高频残差细化对鲁棒性的决定性作用。
*   **主要优势**：实现了语义与物理交互的解耦，对滑动和接触突变具有极强的抗干扰性。
*   **主要局限**：感知模型深度依赖于特定的传感布局（如触觉手套），跨硬件平台的迁移成本较高。

### 6. 实用指南
*   **实现要点**：残差细化层应只在局部触觉敏感的自由度（如手部关节、手腕姿态）上进行修正，其余维度直接取自名义策略。
*   **训练细节**：四阶段分级训练（Planner -> World Model -> VLA Policy -> Refinement Layer）是关键，不可端到端硬训练，以免破坏预训练权重。
*   **迁移建议**：若迁移至新机器人，重点在于如何将传感器读数归一化为统一的“图像格式（Tactile Image）”，以便复用现有的视觉骨干网络。

### 7. 总结
*   **核心思想**：通过分层解耦与残差修正，实现长程语义规划与高频接触鲁棒控制的统一。
*   **速记版 Pipeline**：
    1.  **宏观规划**：根据指令生成子任务。
    2.  **目标预测**：预测下一步的期望触觉反馈。
    3.  **名义动作**：生成一段平滑的动作序列。
    4.  **实时修正**：根据瞬时触觉误差对名义动作执行局部加减（修正）。

**Key Findings:**

- We introduce TouchWorld, a predictive-and-reactive tactile foundation model for dexterous manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07287v1)
- [arXiv](https://arxiv.org/abs/2607.07287v1)

---

<a id='2607.07101v1'></a>
## [GeoProp: Grounding Robot State in Vision for Generalist Manipulation](https://arxiv.org/abs/2607.07101v1)

**Authors:** Guoyang Zhao, Quanhao Qian, Gongjie Zhang, Wenhao Li, Jiuniu Wang, Xiaowei Lu, Deli Zhao, Ran Xu

**Published:** 2026-07-08

**Categories:** cs.RO, cs.AI

**Abstract:**

Proprioception is fundamental to robotic manipulation, yet standard fusion methods often treat it as an isolated vector lacking explicit alignment with visual tokens. Without a direct correspondence between 3D kinematics and 2D feature maps, manipulation policies struggle to ground the robot's state within the scene, frequently underperforming even vision-only baselines. To address this, we introduce GeoProp, a lightweight, plug-and-play adapter that aligns proprioception with vision through explicit geometric grounding and spatial feature sampling. GeoProp projects the robot state onto the image plane to sample localized visual features, constructing a grounded state token. It then injects state-derived spatial priors into the corresponding visual features via FiLM modulation. To capture motion intent, GeoProp further samples features at a short-horizon predicted coordinate derived from recent kinematics, providing look-ahead visual context. Across 67 tasks, GeoProp improves Diffusion Policy by 8.7% on 63 simulation tasks and pi_0 by 4.0% on the RoboTwin subset, and yields a 10.6% average gain across both policy families in the real world, while adding only 2-3% to the parameter count. These results demonstrate that GeoProp is a simple yet high-impact inductive bias for generalist embodied policies. Project page: https://alibaba-damo-academy.github.io/GeoProp/.

**Analysis:**

### 1. 摘要翻译
本体感受对于机器人操作至关重要，然而现有的融合方法通常将其视为一个孤立的向量，缺乏与视觉特征的显式对齐。由于三维运动学和二维特征图之间缺乏直接对应关系，操控策略难以将机器人状态定位到场景中，其性能往往甚至不及纯视觉基线。为此，我们引入了 **GeoProp**，这是一种轻量级的即插即用适配器，通过显式的几何定位和空间特征采样，将本体感受与视觉对齐。GeoProp 将机器人状态投影到图像平面，以采样局部视觉特征，构建“接地（grounded）”状态标记；随后，通过 FiLM 调制将状态衍生的空间先验注入相应的视觉特征。为捕捉运动意图，GeoProp 进一步在根据近期运动学预测的短视界坐标处采样特征，提供前瞻性视觉上下文。在 67 项任务中，GeoProp 使 Diffusion Policy 在 63 项仿真任务上性能提升 8.7%，在 RoboTwin 子集上使 π0 性能提升 4.0%，并在真实世界中实现了 10.6% 的平均增益，且参数量仅增加 2–3%。这些结果表明，GeoProp 是通用具身策略的一种简单且高影响力的归纳偏置。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决视觉与本体感受之间在表示空间上的“结构性解耦”问题，即如何让策略显式理解“我在视觉空间中处于什么位置”。
*   **现有方法痛点**：传统方法将本体感受视为一个全局、无关联的特征向量，通过简单的拼接（concatenation）或深层交叉注意力（cross-attention）进行融合。这种方法强迫策略在训练中“隐式”学习三维运动学与二维图像像素的对应关系，容易导致模态对齐失败或产生虚假关联。
*   **研究假设**：通过引入显式的几何投影将本体感受“扎根（ground）”于视觉特征图，可以直接消除模态对齐的搜索空间，从而提升操作精度。

---

### 3. 方法设计详解
GeoProp 包含三个核心组件：
1.  **几何投影与特征采样**：
    *   利用相机内参 $K$ 和外参 $(R, t)$，将末端执行器的 3D 坐标 $\mathbf{r}_t$ 投影至图像平面获得 $\mathbf{q}_t$，并映射到特征图的连续坐标 $\bar{\mathbf{q}}_t$。
    *   通过双线性采样提取该位置的特征作为“接地状态标记（grounded state token）”，直接关联交互点位。
2.  **空间对齐特征调制 (FiLM)**：
    *   在特征聚合（FPN）之前，利用本体感受状态 $\mathbf{p}_t$ 生成通道维度的调制参数 $\gamma, \beta$。
    *   仅对投影点及其周围的特征单元执行 FiLM 操作，使视觉表示在该局部区域主动“感知”到机器人状态，而非全局散布。
3.  **预测运动学采样**：
    *   通过多项式外推最近的轨迹窗口，预测下一时刻的末端坐标 $\hat{\mathbf{r}}_{t+1}$。
    *   在该预判位置采样未受调制原始特征图，获得前瞻性的运动意图标记。

---

### 4. 方法对比分析
*   **本质区别**：从传统的“隐式对齐”转向“显式几何投影”，强调本体感受必须在空间位置上与视觉像素“共定位（co-located）”。
*   **创新贡献**：提出了一种与骨干网络架构无关（framework-agnostic）的几何适配器，通过局部特征调制和轨迹前瞻，为现有策略注入了高效的几何归纳偏置。
*   **适用场景**：特别适用于需要高精度视觉-运动对齐的任务，如精细零件抓取、组装等。

---

### 5. 实验分析
*   **验证方法**：对比了无本体感受基线、原始融合基线与 GeoProp，分别在 Diffusion Policy 和 π0 (VLA) 上进行了仿真与真实世界测试。
*   **关键结果**：GeoProp 在极低参数开销（2-3%）下，显著提升了在精细操作任务中的成功率（例如 MetaWorld 中的 Basketball 任务提升 24+ 个点）。
*   **局限性**：对相机标定敏感；若物体遮挡了末端执行器，投影点可能产生歧义；假设运动轨迹局部平滑。

---

### 6. 实用指南
*   **开源情况**：项目主页已提供（https://alibaba-damo-academy.github.io/GeoProp/）。
*   **实现细节**：
    *   采样模块需要 FPN（特征金字塔）以统一多尺度特征分辨率。
    *   调制参数生成器采用两层 MLP，初始层设为零初始化，以保证训练初期适配器不会破坏原有策略的特征提取。
    *   训练时确保相机参数准确，否则需引入增强手段（如模拟标定漂移）。

---

### 7. 总结
*   **核心思想**：通过几何投影，实现本体感受与视觉空间在局部特征层面的显式对齐。
*   **速记版pipeline**：
    1.  投影机器人位置到图片坐标；
    2.  在投影点处采样视觉特征；
    3.  用机器人状态调制特征；
    4.  拼接预测的未来轨迹特征；
    5.  将所有信息送入策略网络。

**Key Findings:**

- To address this, we introduce GeoProp, a lightweight, plug-and-play adapter that aligns proprioception with vision through explicit geometric grounding and spatial feature sampling.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07101v1)
- [arXiv](https://arxiv.org/abs/2607.07101v1)

---

<a id='2607.07675v1'></a>
## [Scaling Mixture-of-Experts Video Pretraining for Embodied Intelligence](https://arxiv.org/abs/2607.07675v1)

**Authors:** Shuailei Ma, Jiaqi Liao, Xinyang Wang, Jingjing Wang, Chaoran Feng, Zijing Hu, Chong Bao, Zichen Xi, Yuqi Gan, Weisen Wang, Yanhong Zeng, Qin Zhao, Zifan Shi, Wei Wu, Hao Ouyang, Qiuyu Wang, Shangzhan Zhang, Jiahao Shao, Yipengjing Sun, Liangxiao Hu, Lunke Pan, Nan Xue, Kecheng Zheng, Yinghao Xu, Xing Zhu, Yujun Shen, Ka Leong Cheng

**Published:** 2026-07-08

**Categories:** cs.CV

**Abstract:**

Despite the recent promise in robot control, video generative models suffer from a domain mismatch due to their primary focus on content creation. For example, their design inherently prioritizes visual fidelity and creativity over computational efficiency and physical realism. In this work, we present LingBot-Video, a DiT-based video pretraining paradigm specifically tailored for embodied intelligence. From the architecture perspective, we adopt the Mixture-of-Experts (MoE), instead of dense, framework to achieve a better trade-off between modeling capacity and inference efficiency, and manage to scale it up from scratch. From the data perspective, we construct a data profiling engine that augments standard internet videos with extensive robot-oriented footage, encompassing manipulation, navigation, and egocentric perspectives, to equip the base model with an intrinsic understanding of actions and world dynamics. From the training perspective, we develop a multi-dimensional reward system to enforce the alignment regarding physical rationality and task completion, going beyond standard criteria such as aesthetics, prompt-following, and motion consistency. Comprehensive evaluations validate its performance and efficiency as a video foundation model. We contribute LingBot-Video as the inaugural large-scale, open-source MoE video foundation model to the community, in a pioneering effort to bridge digital creativity and physical actuation.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **LingBot-Video** 的论文摘要分析如下：

### 1. 核心贡献总结
该论文提出了 **LingBot-Video**，这是一个专为具身智能（Embodied Intelligence）设计的基于 DiT（Diffusion Transformer）架构的视频预训练范式。通过引入 **MoE（混合专家模型）架构**与**机器人导向的数据分析引擎**，该模型有效克服了传统视频生成模型在机器人控制任务中“追求视觉美感而忽视物理真实性与计算效率”的领域错配问题。

### 2. 关键创新与方法论
*   **架构革新 (MoE Scaling)：** 与传统的密集型（Dense）模型不同，LingBot-Video 采用了 MoE 架构。这在提升模型参数规模以增强对复杂环境理解能力的同时，大幅优化了推理效率，实现了模型容量与计算成本的平衡。
*   **数据工程 (Robot-Centric Profiling)：** 论文设计了一套专门的数据分析与增强引擎。它不仅利用通用互联网视频，还深度整合了机器人操纵、导航及第一人称视角（Egocentric）的专属数据，强化了模型对动作执行和世界物理动态的内隐理解。
*   **训练范式 (Multi-dimensional Reward)：** 引入了一套多维度奖励系统，将物理合理性（Physical Rationality）和任务完成度作为核心指标，替代了仅关注美学或运动一致性的传统训练目标，确保生成的内容具备实际的可执行性。

### 3. 对领域的潜在影响
*   **弥合数字创意与物理控制的鸿沟：** 该研究标志着视频生成模型从“内容创作”向“物理行为预演”的范式转移，为具身智能领域提供了一个具备通用物理先验的基座模型。
*   **算力成本的优化：** 通过在大规模 MoE 上的成功实践，为社区提供了一种在计算资源受限的情况下仍能利用海量视频数据进行大模型训练的范例。
*   **标准化基准：** 作为首个开源的 MoE 视频基座模型，它可能成为具身智能领域评估世界模型（World Models）性能的新标杆。

### 4. 受益的相关领域与应用
*   **机器人模仿学习 (Imitation Learning)：** 通过视频生成进行轨迹规划或策略预训练，解决小样本学习难题。
*   **数字孪生与仿真环境构建：** 利用该模型快速生成符合物理法则的训练场景，用于强化学习训练。
*   **人机交互 (HRI)：** 提升机器人对人类意图的理解能力，使其生成的动作轨迹更符合人类的物理直觉。
*   **自动驾驶：** 预训练模型对世界动态的深刻理解可直接迁移至复杂的导航任务中。

### 5. 可推断的局限性
*   **Sim-to-Real 鸿沟的挑战：** 尽管模型增强了物理合理性，但从视频空间生成的物理逻辑与真实世界的传感器反馈之间仍可能存在本质差异，即“逻辑合理”不等于“物理精确”。
*   **推理延迟：** 虽然 MoE 优化了计算效率，但在高性能实时机器人控制系统中，部署这种大规模模型依然存在严苛的延迟挑战。
*   **长程一致性：** 视频生成模型在处理长周期、复杂操作序列时，如何保证任务的长期逻辑连贯性（Long-horizon consistency）仍是一个悬而未决的问题。

---

**专家点评：**
这篇论文的有趣之处在于它敏锐地捕捉到了当前视频生成模型（如 Sora 风格）在“具身智能”场景下的痛点——**即生成结果仅能“看起来真实”，而无法“执行真实”**。通过从架构、数据、评估机制的全链路重构，LingBot-Video 试图构建一个能够理解物理规律的世界模型，这对于将 AI 从屏幕推向物理空间具有里程碑式的意义。

**Key Findings:**

- In this work, we present LingBot-Video, a DiT-based video pretraining paradigm specifically tailored for embodied intelligence.
- From the training perspective, we develop a multi-dimensional reward system to enforce the alignment regarding physical rationality and task completion, going beyond standard criteria such as aesthetics, prompt-following, and motion consistency.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07675v1)
- [arXiv](https://arxiv.org/abs/2607.07675v1)

---

<a id='2607.07601v1'></a>
## [CARLA-GS: Decoupling Representation, Reasoning, and Physics Simulation for Autonomous Driving Corner-Case Synthesis](https://arxiv.org/abs/2607.07601v1)

**Authors:** Kaicong Huang, Meng Ma, Ruimin Ke

**Published:** 2026-07-08

**Categories:** cs.RO, cs.AI

**Abstract:**

Safety evaluation for autonomous driving is dominated by rare, safety-critical interactions, motivating simulators that can deliberately synthesize corner cases with photorealistic observations. Corner-case generation is inherently a multi-source problem spanning visual representation, scene reasoning, and vehicle trajectory generation and control. Prior knowledge- and model-based approaches typically focus on scene or trajectory components in isolation, while diffusion-based methods attempt end-to-end generation but still struggle to ensure spatiotemporal consistency and physical realism. To unify these aspects within a single framework, we propose CARLA-GS, a modular corner-case synthesis pipeline that decouples visual representation, semantic reasoning, and physics-based execution while maintaining tight cross-module coupling. Starting from real driving data, we reconstruct an editable gaussian scene with additional geometry-consistent constraints. A multi-agent LLM then performs scene-level reasoning to identify risky interactions and generate intent-level waypoint trajectories, while the low-level motion control is delegated to CARLA, where a PID controller ensures kinematic and dynamic feasibility. The simulated vehicle states are finally re-projected into the gaussian scene for ego-centric rendering. This design enables high-level semantic reasoning, low-level physically executable motion, and photorealistic corner-case generation within a unified pipeline. Experiments on the Waymo Open Dataset show, both quantitatively and qualitatively, that our framework enables controllable corner-case generation and produces photorealistic, spatiotemporally consistent videos aligned with semantic intent and physically feasible motion.

**Analysis:**

### 1. 摘要翻译
自动驾驶的安全评估受限于罕见的极端场景（corner cases），这促使我们需要能够生成具有真实感观测结果的模拟器。corner-case的生成本质上是一个跨越视觉表示、场景推理和车辆轨迹生成的跨领域问题。现有方法通常将场景或轨迹生成孤立处理，而基于扩散模型的方法在尝试端到端生成时，难以保证时空一致性和物理真实性。为解决上述问题，我们提出了CARLA-GS，这是一个模块化框架，通过在保持跨模块强耦合的同时，将视觉表示、语义推理和物理仿真进行解耦。该方法基于真实驾驶数据重建可编辑的Gaussian场景，并引入几何一致性约束。利用多智能体大语言模型（LLM）进行场景级推理，以识别风险交互并生成意图层面的路点轨迹，随后委托CARLA执行低层运动控制，确保运动的运动学和动力学可行性。最后，将模拟出的车辆状态回投至Gaussian场景中进行自我中心视角渲染。实验表明，该框架实现了受控的corner-case生成，并能产生与语义意图和物理可行轨迹对齐的、时空一致的真实感视频。

---

### 2. 方法动机分析
*   **驱动力**：在“稀有事件诅咒”下，如何高效、可控地合成具有物理一致性和高视觉保真度的仿真场景，以提升自动驾驶系统的鲁棒性。
*   **现有痛点**：NeRF类方法计算成本高，扩散模型难以保证时空一致性且缺乏可控性； vanilla 3DGS在场景编辑和车辆动力学方面缺乏几何约束和动作语义支持。
*   **研究假设**：通过模块化解耦（视觉表示由GS负责、意图生成由LLM负责、物理执行由CARLA负责），可以实现对复杂驾驶场景的高保真合成与精准控制。

---

### 3. 方法设计详解
**Pipeline流程：**
1.  **场景重建与优化**：以Street Gaussians为基础，引入Flattening loss（优化各向异性缩放）和Normal/Geometry loss（基于深度图和法线约束），提升场景表面几何一致性。
2.  **多智能体LLM推理**：构建两层代理（Agent）：
    *   `<Zone>`代理：根据行为库（如“紧急制动”）分析场景，识别威胁区域和目标车辆，输出结构化决策元组。
    *   `<Trajectory>`代理：根据当前状态和目标点，生成符合意图的参考路点序列。
3.  **CARLA物理仿真**：将LLM生成的参考点输入CARLA，通过PID控制器执行低层运动，得到符合动力学约束的轨迹，输出精确的车辆pose（位置与朝向）。
4.  **回投与资产替换**：将仿真轨迹映射回3DGS场景。对于近场遮挡或重建不佳的车辆，利用SAM-3D重建的高质量资产进行替换。

**关键算法解释：**
*   **几何一致性损失**：通过比较渲染的法线与深度衍生出的参考法线，并应用平滑权重，解决了GS在稀疏视角下容易出现的表面“坑洼”问题。
*   **状态回投（Re-projection）**：利用3DGS的显式特性，直接更新车辆Gaussian primitives的姿态，实现仿真与渲染的无缝集成。

---

### 4. 方法对比分析
*   **本质区别**：与端到端生成模型不同，该方法通过显式解耦，将LLM的“规划”与传统模拟器的“动力学”结合，实现了语义可控与物理真实的平衡。
*   **创新点**：首次将LLM的推理能力与GS场景表示及CARLA仿真引擎进行深度协同，解决了生成场景中“行为不合理”与“渲染不真实”的两难。
*   **适用场景**：适用于自动驾驶仿真平台，特别是需要针对特定交通参与者进行定制化风险测试的场景。

---

### 5. 实验分析
*   **关键结论**：在Waymo数据集上，CARLA-GS在“Zone Hit”准确率和“Success”（TTC < 1s）指标上显著优于规则驱动和随机扰动基线。
*   **优势**：具有极强的意图可控性，能生成高时空一致性的仿真视频。
*   **局限**：重建环节（3DGS）训练耗时较长（约2小时/100帧），对计算资源要求较高。

---

### 6. 实用指南
*   **实现细节**：
    *   `Flattening loss`权重的设定对于平滑表面至关重要。
    *   PID控制器需在场景起始帧初始化，以减少进入corner-case时的动作抖动。
    *   SAM-3D在处理车辆资产替换时，需精确对齐bounding box以确保几何一致性。
*   **迁移建议**：可将LLM代理替换为特定任务的策略模型（如强化学习策略），以实现更广泛的交互生成。

---

### 7. 总结
*   **核心思想**：模块化协同：GS负责视觉，LLM负责意图，CARLA负责物理现实。
*   **速记版pipeline**：
    1. 三维重建场景并平滑几何表面。
    2. 机器人大脑（LLM）规划威胁方案。
    3. 物理引擎（CARLA）计算运动轨迹。
    4. 将虚拟车辆状态回填至场景并渲染。

**Key Findings:**

- To unify these aspects within a single framework, we propose CARLA-GS, a modular corner-case synthesis pipeline that decouples visual representation, semantic reasoning, and physics-based execution while maintaining tight cross-module coupling.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07601v1)
- [arXiv](https://arxiv.org/abs/2607.07601v1)

---

<a id='2607.07459v1'></a>
## [EmbodiedGen V2: An Agentic, Simulation-Ready 3D World Engine for Embodied AI](https://arxiv.org/abs/2607.07459v1)

**Authors:** Xinjie Wang, Liu Liu, Taojun Ding, Andrew Choi, Chaodong Huang, Mengao Zhao, Ziang Li, Jackson Jiang, Chunlei Yu, Shengxiang Liu, Wei Xu, Zhizhong Su

**Published:** 2026-07-08

**Categories:** cs.RO, cs.CV

**Abstract:**

We present EmbodiedGen V2, a generative 3D world engine for building executable sim-ready environments for embodied intelligence. Sim-ready 3D asset generation has advanced rapidly, yet assembling such assets into policy-ready task environments remains largely manual, limiting scalable closed-loop learning. EmbodiedGen V2 addresses this gap through a unified sim-ready representation that connects cross-simulator assets, interaction affordances, task-driven worlds, large-scale multi-room scenes, and stateful Vibe Coding into a generative, editable, and reusable simulation pipeline. The generated environments support manipulation, navigation, mobile manipulation, cross-simulator deployment, and embodied policy training. In evaluation, the asset pipeline achieves 96.5% human acceptance and 98.6% collision success, and 83.3% of task-driven worlds are directly usable for downstream simulation without manual modification. Online reinforcement learning with generated environments further improves simulation success from 9.7% to 79.8%, and transfers to real robots with task success increasing from 21.7% to 75.0%. These results establish EmbodiedGen V2 as scalable simulation infrastructure for training, evaluating, and deploying embodied policies.

**Analysis:**

作为计算机视觉与具身智能（Embodied AI）领域的专家，我对 **EmbodiedGen V2** 的分析如下：

### 1. 核心贡献总结
EmbodiedGen V2 提出了一套生成式 3D 世界引擎，旨在解决具身智能领域中“从资产生成到可执行任务环境”的自动化鸿沟。它通过统一的表征架构，实现了从单一资产到大规模、多房间、具有交互能力的仿真环境的自动化构建，显著提升了具身智能策略的训练效率与真实世界迁移效果。

### 2. 关键创新与方法论
*   **统一的仿真就绪（Sim-ready）表示：** 这是该论文的核心创新点。它不仅仅生成视觉上的 3D 模型，还通过“交互可供性（Affordance）”映射、状态逻辑与“Vibe Coding（氛围代码）”的结合，赋予了生成内容物理仿真属性。
*   **闭环流程自动化：** 将跨模拟器资产整合、交互逻辑定义、任务场景生成与策略训练串联成一个流水线，消除了传统手动构建环境的大量冗余工作，解决了具身智能学习在规模化上的瓶颈。
*   **高可靠性的生成管线：** 在碰撞检测、环境可用性及人类验收率上达到了极高的指标，证明了其生成的环境在“物理正确性”上足以替代部分人工设计的基准测试集。

### 3. 对领域的潜在影响
*   **打破“模拟器孤岛”：** 该工具通过跨模拟器部署能力，极大地降低了算法在不同物理引擎（如 Isaac Sim, Mujoco, PyBullet 等）之间切换的适配成本。
*   **从“数据驱动”转向“环境驱动”：** 以往研究多关注视觉数据的扩增，而 EmbodiedGen V2 推动了“仿真环境的规模化生成”，这意味着我们可以通过程序化生成成千上万个差异化的家庭或工业场景来训练机器人，类似于大语言模型利用海量文本进行预训练，这为实现“具身智能的大模型”提供了基础设施。

### 4. 相关领域与受益应用
*   **具身基础模型（Embodied Foundation Models）：** 为需要海量环境交互训练的通用机器人策略（如 RT-2, Gato 等）提供数据生成的“兵工厂”。
*   **家庭服务机器人（Household Robotics）：** 由于该引擎支持多房间场景，对室内移动操作（Mobile Manipulation）类任务的训练具有直接推动作用。
*   **数字孪生与虚拟制片：** 其生成的可交互 3D 资产和逻辑代码，亦可延伸至虚拟现实（VR）或数字孪生领域，用于构建具备智能反馈的交互式环境。
*   **Sim-to-Real 迁移：** 论文中提到的 21.7% 到 75.0% 的迁移率提升，直接证明了其生成的环境在分布对齐（Distribution Alignment）上的卓越表现。

### 5. 可推断的局限性
*   **语义理解的鲁棒性：** 尽管提出了“Vibe Coding”，但对于极其复杂的非结构化物理交互或长程因果逻辑，生成的环境可能仍存在逻辑死角。
*   **物理仿真精度上限：** 自动化生成的物体属性（如摩擦系数、惯性张量等）可能在微观物理特性上与真实世界存在偏差，可能限制对高度精细操作（如穿针引线）的训练。
*   **计算资源开销：** 这种规模化生成引擎背后的模型推理成本（尤其是大规模场景的渲染与物理检查）可能非常高，对于中小型科研团队而言具有较高的算力门槛。

---

**专家点评：**
这篇论文之所以对计算机视觉领域具有重要意义，是因为它触及了具身智能发展的**核心痛点——“环境数据荒”**。计算机视觉研究者通常擅长处理静态图像或视频，但 EmbodiedGen V2 将视觉内容置于物理语境下，标志着视觉研究正从“看”向“用”进行彻底的范式转变。这种自动生成“可执行、可交互物理世界”的能力，是通往通用机器人（General-purpose Robots）不可或缺的基石。

**Key Findings:**

- We present EmbodiedGen V2, a generative 3D world engine for building executable sim-ready environments for embodied intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07459v1)
- [arXiv](https://arxiv.org/abs/2607.07459v1)

---

<a id='2607.07430v1'></a>
## [Immersive Social Interaction with VR and LLM-Assisted Humanoids](https://arxiv.org/abs/2607.07430v1)

**Authors:** Niraj Pudasaini, Geeta Chandra Raju Bethala, Pranav Doma, Anthony Tzes, Yi Fang

**Published:** 2026-07-08

**Categories:** cs.RO, eess.SY

**Abstract:**

Humanoid robots can extend human presence to remote, constrained, or hazardous environments, but existing teleoperation interfaces often require physically demanding motion tracking or cognitively demanding low-level control. This paper presents an immersive teleoperation framework that integrates voice-controlled locomotion, VR-based manipulation, and bidirectional social interaction for whole-body humanoid control. Using Apple Vision Pro, the operator receives egocentric visual feedback, issues natural-language locomotion commands, and teleoperates the robot's arms and dexterous hands through wrist and finger tracking. An LLM-assisted voice-control module converts spoken instructions into high-level locomotion commands, while the manipulation module retargets human hand motions to the robot through inverse kinematics and PD control. The system also records multimodal data, including egocentric RGB observations, voice/text commands, joint states, hand motions, and eye-gaze signals, supporting future imitation learning and autonomy. We evaluate the framework on a Unitree H1 humanoid equipped with dexterous hands in manipulation and social interaction tasks. Results show that novice users can successfully operate the system after brief familiarization, achieving 80\% success in object manipulation and 70\% success in a social cube-passing task. These results demonstrate the potential of immersive, language-assisted teleoperation as an accessible interface for humanoid interaction, remote assistance, and multimodal data collection.

**Analysis:**

以下是对该论文的方法分析：

### 1. 摘要翻译
本文介绍了一种多功能的类人机器人远程操作系统，该系统利用语音指令进行移动控制，通过VR手势跟踪实现操控。该框架不仅实现了机器人全身的实时操控，还支持多模态数据采集，可用于训练机器人执行更复杂的任务。该系统为类人机器人与人类的自然互动及协作提供了新方案，对于居家养老及高危环境任务具有重要意义。

### 2. 方法动机分析
*   **驱动力**：解决老龄化社会带来的居家照护与社会互动需求，同时兼顾高危环境下的远程作业。
*   **现有痛点**：传统遥操作系统在复杂非结构化环境下的移动控制能力弱，且人机接口不够友好，操作者需同时管理多个关节，认知负荷大。
*   **研究假设**：通过将“高层语义语音指令”与“底层直观肢体映射”相结合，可以大幅降低操作复杂度，实现类人机器人的自然、全身交互。

### 3. 方法设计详解
本系统由三个核心模块构成：
*   **语音引导移动控制（Voice-controlled Locomotion）**：
    *   **流程**：Vision Pro采集自中心视角图像 $\rightarrow$ Deepgram进行实时语音转文字 $\rightarrow$ GPT-4将指令解析为 `move(x, y)` 等高层语义命令 $\rightarrow$ 调用预训练的深度强化学习策略执行移动。
    *   **创新机制**：针对GPT-4可能出现的解析误判，引入了**指令确认机制**（System-in-the-loop），若语义不确定则请求人工确认，保证执行安全。
*   **VR操控（Manipulation）**：
    *   采用 VisionPro Teleop 将 Apple Vision Pro 的手部姿态（SE(3)）数据传至服务端。
    *   使用 **Pinocchio 逆运动学算法**计算机器人各关节角度，通过PD控制器驱动机械臂和灵巧手，实现动作的实时映射。
*   **社交交互（Social Interaction）**：
    *   建立基于ROS 1的双向音频链路，实现环境感知与对话同步。

### 4. 方法对比分析
*   **本质区别**：区别于单纯的模仿学习或纯肢体遥控，该方法实现了**语义级（语音）与操作级（VR）的混合控制**，能够同时处理导航、交互与复杂的 manipulation 任务。
*   **创新贡献**：引入了“大模型验证机制”来降低语音交互的失败率，并提出了一种易于部署的全身交互范式。
*   **适用场景**：适用于养老陪伴、高危搜救等需要操作者保持高水平 situational awareness（情境感知）的场景。

### 5. 实验分析
*   **验证方法**：通过“捡拾瓶子”和“传递立方体”两项任务，对比新手与专家在成功率和耗时上的差异。
*   **关键结论**：新手在实验中的表现虽慢于专家，但仍能达到 0.7-0.8 的成功率，验证了系统的易上手性。
*   **优势**：交互直观，多模态数据录制（图像+语音+姿态）价值高。
*   **局限**：目前依赖 egocentric（自中心）视觉，对复杂环境的全向感知能力有待提升（作者计划后续增加腰部摄像头）。

### 6. 实用指南
*   **实现细节**：系统强依赖于 Apple Vision Pro 的手势捕获能力，服务端需部署 Pinocchio 引擎。注意在实验中，GPT-4的视觉模式默认关闭以优化推理延迟。
*   **迁移可能**：该框架的“语音转控制逻辑”可直接迁移至其他足式机器人（如四足机器人），只需更换对应的控制接口。

### 7. 总结
*   **核心思想**：语音语义指令与VR手势映射融合的全身遥操。
*   **速记版pipeline**：
    1.  佩戴VR头显获取视角并采集语音/动作；
    2.  大语言模型将语音解析为标准移动命令；
    3.  通过逆运动学将肢体动作映射到机器人；
    4.  实时双向音频同步确保人机社交感。

**Key Findings:**

- Results show that novice users can successfully operate the system after brief familiarization, achieving 80\% success in object manipulation and 70\% success in a social cube-passing task.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07430v1)
- [arXiv](https://arxiv.org/abs/2607.07430v1)

---

<a id='2607.07370v1'></a>
## [Behavior Foundations for Quadruped Robots: ABot-C0 Technical Report](https://arxiv.org/abs/2607.07370v1)

**Authors:** Xufeng Zhao, Fuzhi Yang, Jianhui Chen, Li Gao, Zhang Meng, Jie Gao, Yao Zheng, Wenyu Liu, Menglin Yang, Minqi Gu, Yaru Zhao, Honglin Han, Shihui Su, Zixiao Tang, Liu Liu, Mu Xu, Yang Cai, Wenbin Tang

**Published:** 2026-07-08

**Categories:** cs.RO, cs.AI, cs.HC, cs.LG

**Abstract:**

In embodied intelligence systems, the motion controller serves as the critical bridge between semantic reasoning and physical execution. Humanoid control has progressed rapidly through large-scale human motion-capture data and motion-tracking paradigm. However, producing quadruped robots motion corpora with scalability and physical feasibility faces more fundamental obstacles: animal motion data is scarce, and cross-embodiment retargeting remains fragile. We present ABot-C0, a generalist motion-control system for quadruped robots that establishes three complementary behavior foundations: a scalable multi-source motion-data pipeline, robust policy learning across motion tracking, locomotion, and scene interaction, and a unified deployment stack for reliable real-world operation. Fundamentally, we construct a data pyramid through conditional video-generation synthesis, annotated motion capture, teleoperation and human design, producing 16,074 physically feasible motion clips as the data foundation for various motion learning demands. We then train a Flow-Matching generalist policy that demonstrates for the first time quadruped motion tracking scaling law that its performance improves consistently as training scales up, with zero-shot capability to track unseen motions. Then, we push a step further for robust all-terrain traversal locomotion by adopting a three-stage privileged-to-perceptive framework with temporal LiDAR memory and terrain-predictive supervision. Collectively, these components form a motion generalist that coordinates multi-policy execution, smooth behavior transitions, energy-efficient control, and safety mechanisms for real-world deployment. Extensive experiments on urban-terrain autonomous navigation and companion-style multimodal interaction demonstrate that quadruped robots move beyond single-function demos toward product-level behavioral intelligence.

**Analysis:**

这份分析报告针对《Behavior Foundations for Quadruped Robots: ABot-C0 Technical Report》进行深度解析。

### 1. 摘要翻译
在具身智能系统中，运动控制器是语义推理与物理执行之间的关键桥梁。尽管人形机器人控制因大规模动作捕捉数据而快速进步，但四足机器人面临数据稀缺及跨形态迁移困难等基础障碍。我们提出了 ABot-C0，这是一个四足机器人通用运动控制系统，建立了三个互补的行为基础：可扩展的多源运动数据流水线、跨运动追踪/运动/场景交互的鲁棒策略学习，以及用于真实世界可靠部署的统一架构。通过数据金字塔技术，我们合成了 16,074 个物理可行的动作片段。我们训练了一种流匹配（Flow-Matching）通用策略，首次验证了四足运动追踪的缩放定律（Scaling Law），并实现了对未见动作的零样本追踪。此外，通过引入三阶段“特权到感知”框架和时序 LiDAR 记忆，我们提升了全地形移动的鲁棒性。实验证明，该系统使四足机器人超越了单功能演示，迈向了产品级的行为智能。

### 2. 方法动机分析
- **核心驱动力**：旨在填补四足机器人与人形机器人在行为基础模型（BFM）研究上的差距，构建一个统一、可扩展的通用控制系统。
- **现有痛点**：1. 运动数据稀缺且难以获取；2. 跨形态 retargeting（重定向）脆弱；3. 现有方法多为特定任务设计的孤立技能，缺乏通用性和鲁棒性。
- **研究假设**：通过“规模化数据+通用策略模型+统一部署栈”，可以像大语言模型一样，通过增加数据规模和模型计算量，涌现出具备通用适应能力的机器人行为控制器。

### 3. 方法设计详解
ABot-C0 由三个核心模块构成：
1.  **数据引擎（Data Engine）**：利用 Teleoperation、Artist Design、MoCap 和 Video-to-Motion 生成技术。关键创新在于**一致性视频生成流水线**（Wan2.2 微调 + Identity Consistency Loss），确保生成的动作片段保持机器人身份的一致性，并通过多重过滤（语义、几何、物理）确保训练数据质量。
2.  **流匹配通用策略（Flow-Matching Policy）**：
    *   **专家到通用策略蒸馏**：先训练多个特定动作的专家（PPO），再通过 DAgger 蒸馏到统一的流匹配策略。
    *   **MCRC（Manifold-Calibrated Reference Conditioning）**：引入一个参考窗口 VAE，将复杂的参考动作编码为低维流形空间 latent $z$，作为策略的条件输入，显著提升追踪精度。
3.  **全地形移动与感知**：采用“特权到感知”的三阶段训练。第一阶段教师策略接触全部特权信息；第二阶段将感知能力（LiDAR）蒸馏给学生策略；第三阶段在噪声环境下微调，确保从模拟到现实的鲁棒迁移。

### 4. 方法对比分析
- **本质区别**：不同于传统的“任务-策略”解耦设计，ABot-C0 强调“通用基础+统一执行栈”。它通过流匹配解决了多模态动作分布问题，通过流形校准解决了参考条件下的误差问题。
- **创新点**：构建了首个四足机器人动作数据缩放定律，验证了模型性能随数据量增长的确定性提升。
- **适用场景**：复杂室内外环境下的长时任务、人机协作、需要高保真动作还原的场景。

### 5. 实验分析（精简版）
- **关键结果**：在 7,076 个动作数据规模下，缩放趋势显著；通过 MCRC 增强的策略在 unseen 任务上 MPJPE 显著降低；全地形机器人导航成功率达 83.2%。
- **优势**：动作复现精度高，对非结构化地形适应性强，通过 NP3O 约束实现了无违规安全控制。
- **局限**：目前的感知（云端计算）与控制（本地计算）仍存在物理分离，需进一步优化端到端的一体化感知-动作架构。

### 6. 实用指南
- **开源情况**：文中提到的代码及基准数据主要见于相关的技术报告索引，建议参考文中提到的《Quadfm》数据集（[14]）。
- **关键细节**：
    *   **MCRC VAE**：训练时需包含关节姿态、root-relative 身体姿态和速度信息，历史窗口设为 20 帧。
    *   **流匹配 ODE 步数**：推理时 $D=5$ 为性能平衡点。
- **迁移可能**：MCRC 和 Specialist-to-Generalist 训练范式可直接迁移至机械臂或其他多足机器人系统。

### 7. 总结
- **核心思想**：通过规模化数据与流形约束策略，构建具身智能行为模型。
- **速记版pipeline**：
    1. 生成/筛选大规模高质动作；
    2. 将专家动作蒸馏至统一流匹配策略；
    3. 利用VAE提取动作流形信息作为条件；
    4. 训练感知-控制鲁棒策略并部署。

**Key Findings:**

- We present ABot-C0, a generalist motion-control system for quadruped robots that establishes three complementary behavior foundations: a scalable multi-source motion-data pipeline, robust policy learning across motion tracking, locomotion, and scene interaction, and a unified deployment stack for reliable real-world operation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07370v1)
- [arXiv](https://arxiv.org/abs/2607.07370v1)

---

<a id='2607.07170v1'></a>
## [PUF: Plug-and-Play Uncertainty-Aware Fusion for Online 3D Scene Graph Generation](https://arxiv.org/abs/2607.07170v1)

**Authors:** Yi Yang, Myrna Castillo, Bodo Rosenhahn, Michael Ying Yang

**Published:** 2026-07-08

**Categories:** cs.CV

**Abstract:**

Online 3D scene graph generation builds a persistent, structured representation of a scene by incrementally fusing 2D observations into a global 3D graph. Existing online methods treat this fusion as a fully deterministic pipeline, where we identify three sources of uncertainty that are overlooked: observation, 2D model, and 3D representation. We propose PUF: a Plug-and-play, Uncertainty-aware, and training-free Fusion framework. Scene graph node association is reformulated as a probabilistic likelihood over semantic and spatial factors, replacing binary accept/reject gates. Dirichlet evidence accumulation distributes class and relationship evidence across plausible candidates proportional to association likelihood. An optional class-conditional prior completes edges for sparsely or never co-observed object pairs. We instantiate PUF with both a 3D Gaussian and a 3D voxel backend and observe consistent improvements, demonstrating its ability to generalize across different representations. Experiments on the 3DSSG and ReplicaSSG benchmarks show that our method substantially outperforms existing approaches while maintaining real-time latency. These results establish uncertainty-aware fusion as a principled and effective paradigm for online 3D scene understanding. The source code is publicly available at https://github.com/yyyyangyi/PUF.

**Analysis:**

## 1. 摘要翻译
在线3D场景图生成旨在通过将2D观察结果增量融合为全局3D图来构建持久的结构化场景表示。现有在线方法通常采用完全确定性的融合流程，而我们发现了其中被忽视的三种不确定性来源：观察、2D模型和3D表示。为此，我们提出了PUF：一个即插即用（Plug-and-play）、不确定性感知且无需训练的融合框架。它将场景图节点关联重新表述为关于语义和空间因素的概率似然，取代了二进制的接受/拒绝门控；通过狄利克雷（Dirichlet）证据累积，将类别和关系证据根据关联似然分布到各候选目标。此外，一个可选的类条件先验能够补充稀疏或未共同观察到的对象对的边缘信息。我们在3D高斯和3D体素后端上实例化了PUF，并在3DSSG和ReplicaSSG基准测试中观察到了一致的性能提升，同时保持了实时延迟，证明了不确定性感知融合在在线3D场景理解中的有效性。

## 2. 方法动机分析
- **驱动力**：现有的在线3D场景图生成方法往往在每一帧处理中进行硬性分类和关联（即全有或全无的决定），这丢失了许多有价值的概率信息。
- **痛点**：
    1. **观察不确定性**：物体可能被截断，且关联只在两端物体同时被观察到时才准确。
    2. **模型不确定性**：2D模型输出的软分布（Soft distribution）被强行归约（Argmax），造成语义信息丢失。
    3. **表示不确定性**：3D重投影（如将2D框转为3D）本身带有深度模糊，硬阈值判定忽略了这种位置模糊性。
- **研究假设**：通过保留并传播不确定性，用概率化的关联与证据累积代替确定性的硬截断，能显著提高场景图生成的稳健性和对长尾边缘的覆盖能力。

## 3. 方法设计详解
- **核心Pipeline**：
    1. **软输出提取**：利用2D SGG模型（RT-DETR+EGTR）的完整Softmax输出（包含类别分布$p^c$和关系分布$p^r$）。
    2. **3D表示与lifting**：将2D检测结果提升为3D（高斯或体素表示）。
    3. **概率关联**：不再设置固定阈值。通过结合空间因素（高斯重叠度或体素包含率）和语义因素（Dirichlet后验均值的JSD散度），计算观察与现有全局节点的联合似然。
    4. **狄利克雷证据累积**：使用Dirichlet分布维护节点语义和边关系。将关联似然作为权重，将2D信息平滑累积到全局节点，而非简单的覆盖。
    5. **关系先验补全**：对于长期未被观察到的关系对，引入类条件先验（基于训练集统计）进行平滑补全。
- **关键公式解析**：
    - **联合似然 $L$**：空间因子（连续度量）与语义因子（JSD散度指数）相乘，使得不确定的几何位置 attenuate（削弱）而非直接 veto（否决）关联。
    - **Dirichlet更新**：通过加权累积 $\boldsymbol{\alpha}_k \leftarrow \boldsymbol{\alpha}_k + \beta \cdot \hat{p}_{obs}^c$，将多帧观察的证据融合到同一个Dirichlet参数中。

## 4. 方法对比分析
- **本质区别**：与现有确定性融合方法不同，PUF引入了贝叶斯框架处理增量学习中的不确定性。
- **创新贡献**：
    1. **全链路不确定性感知**：实现了从2D inference到3D fusion的完整不确定性传递。
    2. **训练即插即用**：该方法完全在推理阶段运行，无需对现有2D模型进行再训练。
- **适用场景**：实时在线机器人导航、动态环境场景重构。

## 5. 实验分析
- **验证方法**：在3DSSG和ReplicaSSG基准上，分别对比了Gaussian和Voxel两种后端。
- **关键结果**：在3DSSG上，PUF-Gaussian在关系Recall@1上相较最强在线基线FROSS提升了18.1个百分点，且维持在15ms/帧的实时效率。
- **局限性**：仍依赖2D检测器性能；对于某些完全从未出现过的长尾类别（如“connected to”），单纯依靠统计先验可能仍存在一定困难。

## 6. 实用指南
- **开源**：代码已公开（https://github.com/yyyyangyi/PUF）。
- **实现注意**：
    - 需保证2D模型输出Softmax完整概率分布而非硬类别标签。
    - 超参数 $\sigma_{se}$ (语义带宽) 和 $\lambda_{birth}$ (新节点生成概率) 对不同数据集表现敏感，建议使用网格搜索确定。
- **迁移性**：该框架是表示无关的，理论上可以轻松作为后端插入任何能输出物体检测与关系分类概率的SGG框架。

## 7. 总结
- **核心思想**：通过Dirichlet分布进行概率化的不确定性证据累积。
- **速记版Pipeline**：
    1. 获取2D模型软性类别与关系概率。
    2. 计算观察与全局节点间的概率关联得分。
    3. 将证据按权重加权累积至Dirichlet参数。
    4. 利用类条件统计先验补全缺失的边缘关系。

**Key Findings:**

- We propose PUF: a Plug-and-play, Uncertainty-aware, and training-free Fusion framework.
- Experiments on the 3DSSG and ReplicaSSG benchmarks show that our method substantially outperforms existing approaches while maintaining real-time latency.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07170v1)
- [arXiv](https://arxiv.org/abs/2607.07170v1)

---

<a id='2607.07129v1'></a>
## [Compositional Motion Generation from Demonstration with Object-Centric Neural Fields](https://arxiv.org/abs/2607.07129v1)

**Authors:** Ahmet Ercan Tekden, Yasemin Bekiroglu

**Published:** 2026-07-08

**Categories:** cs.RO

**Abstract:**

Compositionality, by organizing complex behavior as combinations of simpler elements, enables robot learning that is scalable and data efficient. Leveraging this principle, we propose a generative learning-from-demonstration framework that enables compositional modeling of robotic behavior by connecting perception and motion through shared object-level representations. We render scenes from object-centric neural representations that integrate canonical neural fields with latent-conditioned deformations, capturing positional and geometric variations in a smooth, consistent, and interpretable way. For motion generation, a temporal mixture-of-experts (MoE) employs a gating mechanism to combine object-conditioned movement primitives over time, producing complete trajectories. This spatial-temporal compositionality maintains the data efficiency of movement primitives while grounding motion in visual structure, enabling systematic generalization across diverse scene configurations. In simulation, long-horizon manipulation tasks are successfully completed using the proposed model, which requires significantly less training data than other image-based baselines. Real-world experiments further demonstrate the method's robustness to noise, its ability to generalize at the category level through language-based segmentation models, and its capacity to operate directly on 3D scene representations.

**Analysis:**

### 1. 摘要翻译
组合性通过将复杂行为组织为简单元素的组合，实现了可扩展且数据高效的机器人学习。基于此原则，我们提出了一种生成式示教学习（LfD）框架，通过共享的对象级表示将感知与运动连接起来，从而实现机器人行为的组合建模。我们利用结合了隐变量条件变形的典型神经场，以平滑、一致且可解释的方式捕获位置和几何变化，从而从以对象为中心的神经表示中渲染场景。在运动生成方面，我们采用了一种时间混合专家（MoE）架构，通过门控机制在时间维度上组合对象条件下的运动基元，生成完整的轨迹。这种时空组合性在保持运动基元数据高效性的同时，将运动锚定在视觉结构中，实现了跨不同场景配置的系统泛化。仿真实验表明，该模型在完成长时程操作任务时，相比其他基于图像的基线方法，显著减少了所需的训练数据。真实机器人实验进一步验证了该方法的噪声鲁棒性、通过语言分割模型实现类别级泛化的能力，以及直接在三维场景表示上操作的能力。

---

### 2. 方法动机分析
*   **驱动力**：解决机器人示教学习中“长时程任务”难以通过少量数据泛化的问题，将“场景感知”与“运动规划”解耦为可组合的模块。
*   **现有痛点**：
    *   **单体化问题**：将复杂动作视作单一序列忽略了其由“抓取、移动、放置”等子任务构成的本质组合结构。
    *   **特征工程限制**：传统运动基元（MPs）高度依赖手绘特征，而直接学习图像特征又极度消耗数据。
*   **研究假设**：通过以对象为中心的神经表征（object-centric representation）作为桥梁，可以将场景变化（空间）与运动执行（时间）进行组合建模。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **场景建模（Spatial MoE）**：将图像分解为多个物体和一个背景的神经场。每个物体使用“典型神经场 + 隐变量条件变形”表示，并通过空间Softmax生成物体掩码。
    2.  **潜空间规范化（Latent Relabeling）**：为了确保泛化，利用边缘潜在向量构成的凸集，通过Latent Search（算法1）将任意演示场景的隐变量映射为这些边缘向量的凸组合。
    3.  **运动建模（Motion MoE）**：利用时间门控机制，根据当前时间片$t$决定各物体专家（Expert）的权重，通过FiLM（Feature-wise Linear Modulation）将感知端的潜空间向量转换为运动生成器的调节参数。
*   **算法关键**：
    *   **空间MoE**：通过物体间的重要性采样防止专家collapse（坍缩）。
    *   **时间门控**：允许在轨迹的不同阶段由不同物体的主导贡献生成动作，实现 sequential compositionality。

---

### 4. 方法对比分析
*   **本质区别**：不同于Diffusion Policy等端到端黑盒策略，该方法显式地将“物体识别（空间）”与“动作序列（时间）”分层建模，且引入了基于神经场的场景重构作为辅助监督，从而大幅降低数据需求。
*   **适用场景**：具备明确对象交互的Manipulation任务，特别是存在多物体、需要长时程规划或在类别级进行泛化的任务。

---

### 5. 实验分析
*   **验证方法**：在Wall Avoidance、Incline Pick-and-Place、Cup/Cube Stacking任务上进行对比，评价指标为轨迹均方根误差（MED）和成功率。
*   **关键结论**：在少量数据（10-30个演示）下，该方法优于CNMP和Diffusion Policy；在数据量提升后，性能优势依然保持稳定。
*   **优势**：极高的数据效率，且生成的动作具备良好的可解释性。
*   **局限**：对“掩码分割”的准确性有一定依赖；若物体高度相似且遮挡严重，潜空间搜索可能出现模糊性。

---

### 6. 实用指南
*   **开源情况**：项目主页：https://fzaero.github.io/compositional/
*   **实现建议**：
    *   **预处理**：必须确保背景在演示中保持一致，且摄像机视角固定。
    *   **训练策略**：采用先训练背景，再训练物体，最后进行联合训练的“三阶段策略”，有助于提升收敛稳定性。
    *   **迁移**：该架构可轻松迁移至任何利用物体属性作为任务条件的策略生成模型中。

---

### 7. 总结
*   **核心思想**：利用对象级神经场实现场景可解释组合，驱动时序动作生成。
*   **速记版Pipeline**：
    1. 训练物体神经场与掩码模型（分解场景）；
    2. 提取物体潜空间特征并进行凸组合规范化；
    3. 利用门控权重动态调用物体专家；
    4. 通过FiLM调节运动生成网络输出轨迹。

**Key Findings:**

- Leveraging this principle, we propose a generative learning-from-demonstration framework that enables compositional modeling of robotic behavior by connecting perception and motion through shared object-level representations.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07129v1)
- [arXiv](https://arxiv.org/abs/2607.07129v1)

---

<a id='2607.07076v1'></a>
## [PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation](https://arxiv.org/abs/2607.07076v1)

**Authors:** Zezeng Li, Enda Xiang, Thuy Tran, Di Huang, Momath Thiam, Liming Chen

**Published:** 2026-07-08

**Categories:** cs.RO

**Abstract:**

Imitation learning has enabled remarkable progress in robotic manipulation, especially with diffusion and flow-based policies that generate complex visuomotor behaviors directly from demonstrations. Yet, despite their strong performance, these policies often fail to generalize across tasks and environments. A key reason is that existing policies tend to imitate superficial action correlations rather than the underlying intent. Inspired by the compositional structure of human behaviors, we propose PriGo, a primitive-guided test-time adaptive framework for robust robotic manipulation. PriGo introduces PANet, a lightweight primitive prediction module that infers primitive distributions directly from observations. We further propose a differentiable primitive guidance mechanism that refines generated actions during inference, steering trajectories toward semantically consistent behaviors. Unlike prior primitive-conditioned approaches, PriGo operates entirely at test time and can be seamlessly integrated into pretrained diffusion and flow policies without retraining. Extensive experiments on LIBERO, CALVIN, SIMPLER, and real-world robotic tasks demonstrate that PriGo consistently improves robustness, long-horizon execution, and generalization ability across both diffusion and flow-based policies.

**Analysis:**

这是一份针对论文《PriGo: Test-Time Primitive Guidance to Diffusion and Flow Policies for Adaptive Robotic Manipulation》的深入分析。

### 1. 摘要翻译
模仿学习在机器人操作领域取得了显著进步，特别是能够从演示中直接生成复杂视觉运动行为的扩散模型（Diffusion）和流策略（Flow policies）。然而，尽管这些策略表现强劲，它们往往难以跨任务和环境泛化。一个核心原因是现有策略倾向于模仿表面的动作关联，而非底层的行为意图。受人类行为组合结构的启发，我们提出了 **PriGo**，一个用于鲁棒机器人操作的基元引导（Primitive-guided）测试时自适应框架。PriGo 引入了 **PANet**，一个直接从观测中推断基元分布的轻量级基元预测模块。此外，我们提出了一种**可微分基元引导机制**，在推理过程中通过梯度优化修正生成动作，将轨迹导向语义一致的行为。与以往需训练的条件策略不同，PriGo 完全在测试时运行，无需重新训练即可无缝集成到预训练的扩散和流策略中。实验表明，PriGo 在 LIBERO、CALVIN、SIMPLER 和真实机器人任务中均显著提升了性能。

### 2. 方法动机分析
*   **驱动力**：试图弥合“低层动作生成”与“高层行为意图”之间的鸿沟，利用人类行为的组合性来约束策略，使其在分布偏移（Distribution shifts）下更鲁棒。
*   **现有方法痛点**：现有策略主要学习观测与动作间的相关性（correlation），在面对未见环境或扰动时，容易产生局部看似正确但整体结构失败的动作（如“先拉后转”的死锁）。
*   **研究假设**：通过在推理时引入“操作基元（Primitive）”作为结构化先验，通过梯度引导动作生成，可以将无约束的轨迹映射到语义一致的任务行为上。

### 3. 方法设计详解
*   **流程总结**：
    1.  **PANet 预测**：利用 DINOv2 和 T5 作为编码器，PANet 根据当前观测 $O$ 预测概率基元分布 $\hat{y}^k$。
    2.  **动作映射**：将策略生成的动作 $a^k$ 通过定义好的软分类函数（Soft Classification）转换为动作对应的基元分布 $p^k$。
    3.  **梯度引导**：计算 $L_{PG}$（交叉熵损失，即预测基元与当前动作分布的差异），通过梯度 $\nabla_a L_{PG}$ 对扩散过程或流匹配的推理进行修正，使得生成的动作更符合预期的基元结构。
*   **核心算法逻辑**：
    *   **定义基元集**：$\{idle, grasp, release, push, pull, rotation, push+rotation, pull+rotation\}$。
    *   **可微分引导**：不同于硬约束（hard constraint），PriGo 定义了软基元分布。在推理的每一步（去噪或流演进），利用 $\eta \nabla_a L_{PG}$ 来“微调”动作，确保动作序列的逻辑链条（如先抓后拉）保持正确。

### 4. 方法对比分析
*   **本质区别**：它是一个**测试时（Test-time）插件**。不需要重新训练预训练好的策略模型（如 Diffusion Policy），而是通过在推理推理流中引入基元先验来实现即时适应。
*   **创新贡献**：将复杂的动作生成解耦为“策略预测”和“基元约束”。其“可微分基元引导机制”允许在不损失扩散模型生成能力的前提下，引入结构化语义约束。
*   **适用场景**：适用于任何基于扩散或流匹配的策略，尤其在需要长周期（Long-horizon）任务和高泛化需求的场景中表现突出。

### 5. 实验分析
*   **验证方法**：在 LIBERO、SIMPLER 和 CALVIN 等主流基准上，分别集成到 π0、SmolVLA、DP 和 CogACT 等模型中进行测试。
*   **关键结果**：在长周期任务中，PriGo 平均提升 3–5 个点（SIMPLER 甚至达到 5-7 点）。真实机器人实验中，成功率提升 26 个百分点。
*   **优势**：真正的即插即用（plug-and-play），推理开销可控（15% 左右），显著提升长序列执行的鲁棒性。
*   **局限**：性能提升上限受限于预训练基座模型；PANet 的泛化能力仍依赖训练数据覆盖度。

### 6. 实用指南
*   **开源情况**：代码已开源，参考论文提到的 GitHub 项目（或搜索 PriGo）。
*   **迁移与实现**：
    *   只需保留现有的策略模型。
    *   训练一个轻量级的 PANet，无需大规模算力。
    *   在推理循环中插入引导项 $\nabla_a L_{PG}$，调整步长 $\eta$ 即可。
*   **关键超参数**：$\tau_{trans}, \tau_{rot}, \tau_w$（自动标注阈值）和 $\eta$（梯度步长）。实验证明该方法对阈值不敏感，易于配置。

### 7. 总结
*   **核心思想**：通过推理时的可微分基元引导，强制策略生成结构化的语义动作。
*   **速记版 Pipeline**：
    1.  通过观测输入预测任务应采取的基元（如“抓取”）。
    2.  计算当前动作偏离目标基元的梯度损失。
    3.  利用该梯度微调扩散模型的去噪步骤。
    4.  输出符合语义要求的修正动作。

**Key Findings:**

- Inspired by the compositional structure of human behaviors, we propose PriGo, a primitive-guided test-time adaptive framework for robust robotic manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.07076v1)
- [arXiv](https://arxiv.org/abs/2607.07076v1)

---

