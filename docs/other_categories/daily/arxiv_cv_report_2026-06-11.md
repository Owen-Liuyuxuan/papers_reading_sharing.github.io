time: 20260611

# Arxiv Computer Vision Papers - 2026-06-11

## Executive Summary

以下是对2026年6月10日Arxiv计算机视觉领域10篇论文的每日报告执行摘要，旨在帮助研究人员快速把握重要进展。

---

### 一、主要主题与趋势

本批论文高度聚焦于**具身智能与视觉-语言-动作（VLA）模型的实用化**，呈现三大核心趋势：

1. **VLA模型的效率与协作**：多篇工作围绕VLA的推理效率（测试时计算分配、Token路由）、异步解耦（DAM-VLA）以及多体协作（CHORUS）展开，旨在降低部署成本并提升可扩展性。
2. **从次优/真实世界数据中学习**：Ambient Diffusion Policy、UniIntervene等探索从次优数据、真实世界干预数据中进行模仿学习或强化学习，降低对完美演示数据的依赖。
3. **几何与动作先验的注入**：World Pilot、VLGA、Fourier Features等通过引入世界模型先验、几何信息或频域特征，增强VLA模型在自动驾驶、机器人操作等任务中的精度与鲁棒性。

此外，目标检测/分割的推理加速（Turbo-Inference）仍是一个持续热点。

---

### 二、特别重要或创新的论文

- ***DIRECT: When and Where Should You Allocate Test-Time Compute in Embodied Planners?***  
  首次系统研究具身规划器中测试时计算的最佳分配策略（时间步与空间区域），为资源受限的机器人推理提供实用指导，具有较高工程价值。

- ***Reroute, Don‘t Remove: Recoverable Visual Token Routing for Vision-Language Models***  
  提出可恢复的视觉Token路由机制，在减少冗余Token的同时保留关键信息，相比直接丢弃更灵活，有望提升VLMs的推理效率与准确性。

- ***Fourier Features Let Agents Learn High Precision Policies with Imitation Learning***  
  引入傅里叶特征作为输入编码，使简单模仿学习达到高精度策略，方法简洁有效，对低维控制任务具有广泛适用性。

- ***World Pilot: Steering Vision-Language-Action Models with World-Action Priors***  
  将世界模型先验显式注入VLA模型，提升决策的可解释性与长期规划能力，是VLA与World Model结合的重要尝试。

---

### 三、新兴研究方向或技术

- **去中心化多体VLA协作**（CHORUS）：单一VLA策略即可支持多个异构机器人的分布式协作，为多机器人系统提供了轻量化方案。
- **解耦异步多模态融合**（DAM-VLA）：将视觉、语言、动作编码解耦为异步处理管道，提升VLA模型对不同模态延迟的容忍度。
- **智能体主动干预**（UniIntervene）：在真实世界强化学习中，利用Agentic Intervention策略智能地介入低效探索，减少样本复杂度。
- **视觉-语言-几何-动作统一建模**（VLGA）：将几何信息显式纳入自动驾驶VLA框架，强化空间理解能力。

---

### 四、建议全文阅读的论文

1. **DIRECT**：若你从事具身规划或机器人推理效率研究，此文提供了直接的基准方法与分配策略。
2. **Reroute, Don't Remove**：对VLMs的Token效率优化感兴趣者必读，方法新颖且实用。
3. **Fourier Features**：简单但极具启发的技术，适合所有从事模仿学习或策略学习的读者。
4. **World Pilot**：关注VLA与World Model结合的前沿读者，此文展示了具体的实现路径。
5. **UniIntervene**：致力于真实世界强化学习部署的研究者，此文提供了可操作的干预框架。

---

**总结**：本期论文标志着VLA模型正从“能否工作”走向“如何高效、鲁棒、可协作地工作”。测试时计算分配、Token路由、几何先验与扩散策略等技术的交叉融合，预示着具身智能将在未来一年内迎来更落地的实践。

---

## Table of Contents

1. [DIRECT: When and Where Should You Allocate Test-Time Compute in Embodied Planners?](#2606.12402v1)
2. [Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics](#2606.12365v1)
3. [CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy](#2606.12352v1)
4. [DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model](#2606.12105v1)
5. [Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models](#2606.12412v1)
6. [World Pilot: Steering Vision-Language-Action Models with World-Action Priors](#2606.12403v1)
7. [VLGA: Vision-Language-Geometry-Action Models for Autonomous Driving](#2606.12396v1)
8. [UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning](#2606.12372v1)
9. [A Turbo-Inference Strategy for Object Detection and Instance Segmentation](#2606.12371v1)
10. [Fourier Features Let Agents Learn High Precision Policies with Imitation Learning](#2606.12334v1)

---

## Papers

<a id='2606.12402v1'></a>
## [DIRECT: When and Where Should You Allocate Test-Time Compute in Embodied Planners?](https://arxiv.org/abs/2606.12402v1)

**Authors:** Jadelynn Dao, Milan Ganai, Yasmina Abukhadra, Ajay Sridhar, Mozhgan Nasr Azadani, Katie Luo, Clark Barrett, Jiajun Wu, Chelsea Finn, Marco Pavone

**Published:** 2026-06-10

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Vision-Language Models (VLMs) are increasingly deployed as high-level planners for embodied agents, with an emerging strategy of scaling test-time compute to improve capability. However, we observe that doing so increases latency, token usage, and FLOPs while yielding uneven, often diminishing gains in downstream success, limiting where embodied agents can be deployed. We argue that choosing when and where to spend test-time compute is central to bringing frontier performance to the real world. We introduce DIRECT, a routing framework that uses multimodal scene context to allocate compute per prompt, improving the success--cost Pareto frontier over fixed model selection. Across three dominant scaling axes, namely chain-of-thought depth, model size, and memory history, our experiments on VLABench and RoboMME show that test-time compute is not a uniform lever: different axes yield qualitatively distinct capability gains. We validate these insights on a physical Franka arm in a DROID setup spanning zero-shot manipulation and long-horizon chaining, where our router matches or exceeds a stronger model's success rate at up to 65% lower average latency. Ultimately, our results show that naively scaling test-time compute is wasteful, and that DIRECT can provide frontier-level embodied planning in robotic systems at a fraction of the cost. Project page can be found at jadee-dao.github.io/direct/.

**Analysis:**

### 1. 摘要翻译
视觉语言模型（VLMs）正日益成为具身智能体的高级规划器，一种新兴的提升能力策略是增加推理时的计算投入（test-time compute）。然而，我们观察到这种策略在增加延迟、Token消耗和FLOPs的同时，下游任务的成功率往往收益不均甚至递减，限制了具身智能体的部署空间。我们认为，决策“何时”及“何地”投入推理计算对于将前沿性能带入现实世界至关重要。为此，我们引入了 DIRECT（具身计算权衡的动态推理路由框架），这是一个利用多模态场景上下文来分配计算资源的路由框架，改善了相较于固定模型选择的成功率-成本帕累托前沿。我们在 VLABench 和 RoboMME 上进行实验，并验证了其在真实的 Franka 机械臂（DROID 环境）上的表现。结果显示，与固定基线相比，我们的路由框架能在显著降低延迟（最高降低 65%）的同时，匹配甚至超过最强模型的成功率。最终，我们的结果表明盲目扩大推理计算是浪费的，DIRECT 能够以极低的成本提供前沿级的具身规划能力。

---

### 2. 方法动机分析
- **驱动力**：作者试图解决“推理时计算成本”与“任务实际难度”之间的错配问题，目标是构建一种能够根据任务自适应分配计算资源（如思维链深度、模型参数量、历史记忆长度）的智能路由方案。
- **现有方法痛点**：现有方法往往“一刀切”地对所有任务应用高昂的计算资源，造成严重的资源浪费且显著增加了延迟，导致部署上的不可行。
- **研究假设**：测试时计算并不是一个均匀的性能杠杆（Uniform Lever）；不同类型的任务对计算的收益模式不同（如语义约束任务适合 CoT，历史相关任务适合记忆优化），因此存在一种基于输入上下文的轻量级路由策略，能实现更优的成本-效益平衡。

---

### 3. 方法设计详解
- **核心 pipeline**：
    1. **数据矩阵构建**：预先收集任务集 $D_{train}$ 在不同计算配置下的成功率矩阵 $Q$ 和成本矩阵 $C$。
    2. **轻量级特征编码**：通过冻结的 SigLIP 视觉编码器和 BGE-M3 文本编码器，将任务的场景图像 $I$ 和指令 $\ell$ 融合为向量 $\phi(x)$。
    3. **路由策略推断**：利用一个极其轻量的路由模型 $r(\phi(x))$（如 MLP 或线性层）预测当前任务在不同配置下的质量分 $q$ 和成本 $c$。
    4. **动态决策**：根据效用函数 $U(q, c)$ 计算出当前任务的最佳计算配置，并将任务分发给对应的 VLM Planner。
- **模型结构与算法**：路由模型实质上是一个回归器，用于预测质量和成本。其核心公式为 $k_i^* = \arg \max_k U(q_{i,k}, c_{i,k})$。通过将分类/回归目标设定为最大化特定效用函数，路由决策可以在保持极低推理延迟（~20-50ms）的前提下完成。

---

### 4. 方法对比分析
- **本质区别**：与现有针对大语言模型（LLM）路由（如 FrugalGPT）的方法不同，DIRECT 显式整合了**视觉场景上下文**。它不仅仅基于文本难度路由，还能够感知物理场景的复杂度和 trajectory（轨迹）可行性。
- **创新贡献**：首次在具身智能领域系统性地探索了“思维链、模型参数量、历史记忆”三大维度的计算权衡，并证明了路由策略是实现“前沿性能、低成本部署”的关键桥梁。

---

### 5. 实验分析
- **验证方法**：在 VLABench 和 RoboMME 模拟数据集及 Franka 真实机器人 DROID 环境上进行验证。
- **关键结论**：DIRECT 能够成功将复杂的、耗时的模型应用到“真正需要”的任务中，而在简单任务中降级使用“快速模型”，从而在几乎所有实验配置下均实现了对单一固定模型方案（无论是大模型还是小模型）的帕累托超越。
- **优缺点**：**优势**在于泛化性强、延迟极低；**局限**在于需要离线预训练以收集 $Q$ 和 $C$ 矩阵，且不具备动态扩充配置的能力（pool 是固定的）。

---

### 6. 实用指南
- **开源情况**：项目主页 `jadee-dao.github.io/direct/`，包含代码与实验配置。
- **实现细节**：路由器的推理计算量（~20ms）应远小于 planner 的推理时间；特征融合方式（如 `normalize_concat`）对路由准确度有显著影响，建议优先尝试线性层或两层 MLP。
- **迁移可能**：非常容易迁移。只需要为你的 Planner 集合构建一个简单的“成功率-成本”基准测试，即可训练对应的路由层。

---

### 7. 总结
- **核心思想**：具身规划任务的计算需求是异构的，通过轻量化上下文感知路由实现最优计算配置匹配。
- **速记版 pipeline**：
    1. 测算备选模型在任务集上的表现与成本。
    2. 训练轻量路由层，学会根据视觉语言输入预测模型能力。
    3. 部署时，由路由器实时分发任务给最合适的模型。

**Key Findings:**

- We introduce DIRECT, a routing framework that uses multimodal scene context to allocate compute per prompt, improving the success--cost Pareto frontier over fixed model selection.
- Ultimately, our results show that naively scaling test-time compute is wasteful, and that DIRECT can provide frontier-level embodied planning in robotic systems at a fraction of the cost.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12402v1)
- [arXiv](https://arxiv.org/abs/2606.12402v1)

---

<a id='2606.12365v1'></a>
## [Ambient Diffusion Policy: Imitation Learning from Suboptimal Data in Robotics](https://arxiv.org/abs/2606.12365v1)

**Authors:** Adam Wei, Nicholas Pfaff, Thomas Cohn, Arif Kerem Dayı, Constantinos Daskalakis, Giannis Daras, Russ Tedrake

**Published:** 2026-06-10

**Categories:** cs.RO, cs.AI

**Abstract:**

We propose Ambient Diffusion Policy, a simple and principled method for imitation learning from suboptimal data in robotics. High-quality, task-specific robot data is expensive and time-consuming to collect, while suboptimal datasets with lower-quality or out-of-distribution demonstrations are abundant. Existing methods that co-train on both data sources in robotics often fail to separate the meaningful and the harmful features in the suboptimal samples. In contrast, our method extracts only the useful features by introducing a new axis to co-training in robotics: noise-dependent data usage. Ambient Diffusion Policy restricts the contribution of suboptimal data during training to only the high and low diffusion times. To rigorously justify our approach, we first observe that robot action data exhibits a spectral power law. This induces two important properties on the optimal Diffusion Policy that we exploit: a global-to-local hierarchy and locality. We theoretically formalize this discussion using a simplified model. Our experiments validate Ambient Diffusion Policy on four types of suboptimal action data (noisy trajectories, sim-to-real gap, task mismatch, and large-scale data mixtures) across six tasks. The results show that it effectively learns from arbitrary sources of suboptimal data. Notably, it outperforms existing co-training baselines by up to 33% when scaled to Open X-Embodiment - a large dataset with heterogeneous data quality and unstructured distribution shifts. Overall, Ambient Diffusion Policy increases the utility of suboptimal demonstrations and expands the set of usable data sources in robotics.

**Analysis:**

### 1. 摘要翻译
我们提出了“Ambient Diffusion Policy”，这是一种简单且具有原则性的机器人模仿学习方法，旨在利用次优数据进行训练。高质量的特定任务机器人数据收集昂贵且耗时，而低质量或分布外（OOD）的数据却非常丰富。机器人领域的现有协同训练（co-training）方法往往无法区分次优样本中“有益”与“有害”的特征。相比之下，我们的方法通过引入一种新的协同训练维度——**噪声依赖数据使用（noise-dependent data usage）**——提取了有用的特征。Ambient Diffusion Policy 仅在特定的高噪声和低噪声阶段使用次优数据，这种策略由机器人动作数据表现出的“谱幂律（spectral power law）”所诱导的“全局到局部（global-to-local）”层级性和“局部性（locality）”性质所支撑。实验表明，该方法在处理噪声轨迹、模拟到真实（sim-to-real）差距、任务失配及大规模数据混合等多种次优数据源上表现优异，在 Open X-Embodiment 大规模数据集上相比基线提升达 33%。

---

### 2. 方法动机分析
- **核心动机**：如何从海量且廉价的“脏数据”中挖掘价值，而不引入干扰性能的有害偏差。
- **痛点**：传统协同训练方法（Dataset Re-weighting）本质上是将所有数据混合训练，导致模型在处理高质量分布时被次优分布拉偏，产生“有害”偏差。
- **研究假设**：机器人动作数据遵循谱幂律，这意味着高频成分对应局部微小动作（易受扰动影响），低频成分对应全局规划（具有跨分布的一致性）。利用扩散模型在不同噪声尺度下的生成特性，可以实现对数据特征的智能筛选。

---

### 3. 方法设计详解
- **核心策略**：通过限制次优数据 $D_q$ 仅在特定的扩散时间 $t$ 范围内参与训练。
- **流程pipeline**：
    1. **数据标记（Phase 1）**：使用分类器 $c_\phi(A_t, t)$ 判定 $t_{min}$，即次优分布 $q$ 与目标分布 $p$ 在该噪声水平下无法区分的阈值。
    2. **噪声掩码机制**：利用谱幂律，将 Gaussian 噪声视为“高频遮罩”。在 $t > t_{min}$ 时，$q$ 和 $p$ 的高频差异被噪声掩盖，此时引入次优数据以学习“全局规划”；在 $t < t_{max}$ 时，利用“局部性”性质，仅在相关感受野内学习精细动作。
    3. **训练（Phase 2）**：修改 Diffusion Policy 的数据采样器，确保次优样本仅在 $t \in [0, t_{max}) \cup (t_{min}, T]$ 区间采样。
- **关键公式含义**：$d_{TV}(p_t, q_t) \le d_{TV}(p, q) \cdot \frac{D}{2\sigma_t}$，证明了随噪声增加，分布差异呈指数级衰减，为次优数据在特定噪声水平下的“无害化”提供了理论保证。

---

### 4. 方法对比分析
- **本质区别**：从传统的“加权混合”转向“时域筛选”，利用扩散模型的频域特性进行细粒度数据使用。
- **创新贡献**：将 Ambient Diffusion 框架引入机器人领域，建立了动作数据统计特性与扩散模型学习层级之间的理论联系。
- **适用场景**：适用于拥有大量非专家演示、仿真数据或跨任务数据的模仿学习场景。

---

### 5. 实验分析
- **关键结果**：在 7-DoF 机器人实验中，Ambient 策略在保持与高质量数据相同成功率的同时，平均加速度显著下降，验证了其在抑制噪声、提升平滑性方面的能力。
- **主要优势**：通用性强，无需修改模型架构，仅改变采样逻辑即可提升性能。
- **主要局限**：计算开销较数据过滤（data filtering）更大（因为保留了更多数据参与训练）；超参数 $t_{min}$ 与 $t_{max}$ 的精确估计在某些极端分布下仍需调优。

---

### 6. 实用指南
- **开源情况**：已开源，项目网站：[ambient-diffusion-policy.github.io](https://ambient-diffusion-policy.github.io/)
- **实现细节**：建议优先使用分类器进行 $t_{min}$ 标注；若资源有限，可先进行小规模的 $t_{min}$ 超参数扫描（Sweep）。
- **迁移方法**：若要迁移至新任务，只需评估其谱幂律特征，若符合幂律分布，即可直接应用该方法。

---

### 7. 总结
- **核心思想**：通过噪声掩码技术，让扩散模型在不同噪声尺度下选择性学习数据的全局一致性与局部平滑性。
- **速记版pipeline**：
    1. 训练分类器区分目标与次优数据；
    2. 计算分布差异阈值 $t_{min}$；
    3. 采样数据时根据 $t$ 决定是否采用次优样本；
    4. 执行标准去噪训练。

**Key Findings:**

- We propose Ambient Diffusion Policy, a simple and principled method for imitation learning from suboptimal data in robotics.
- In contrast, our method extracts only the useful features by introducing a new axis to co-training in robotics: noise-dependent data usage.
- Ambient Diffusion Policy restricts the contribution of suboptimal data during training to only the high and low diffusion times.
- To rigorously justify our approach, we first observe that robot action data exhibits a spectral power law.
- The results show that it effectively learns from arbitrary sources of suboptimal data.
- Notably, it outperforms existing co-training baselines by up to 33% when scaled to Open X-Embodiment - a large dataset with heterogeneous data quality and unstructured distribution shifts.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12365v1)
- [arXiv](https://arxiv.org/abs/2606.12365v1)

---

<a id='2606.12352v1'></a>
## [CHORUS: Decentralized Multi-Embodiment Collaboration with One VLA Policy](https://arxiv.org/abs/2606.12352v1)

**Authors:** Ria Doshi, Tian Gao, Annie Chen, Chelsea Finn, Jeannette Bohg

**Published:** 2026-06-10

**Categories:** cs.RO, cs.AI

**Abstract:**

Multi-robot collaboration allows robots to efficiently take on a wide range of tasks, from moving a couch through a doorway to assembling structures on a construction site. However, achieving such coordination in mobile multi-robot settings remains challenging: centralized methods conditioned on the combined observations of a team scale poorly with team size, and decentralized methods that train one policy per robot often require explicit alignment procedures or information sharing at inference time to overcome partial observability. Our key insight is that the visuomotor priors of pretrained vision-language-action (VLA) models should enable reactive, decentralized collaboration from each robot's local observations alone, without these inference-time assumptions. We propose CHORUS, a framework that adapts a single VLA backbone to control diverse, multi-robot teams. At inference time, each robot runs an independent copy of CHORUS, conditioned only on its own observations and a robot-identifying prompt. In real-world experiments including mobile tape measurement, library book handovers, and laundry basket lifting, CHORUS achieves a 64% point improvement over decentralized, from-scratch models, improves reactivity to teammate behavior by 40% points, and outperforms centralized baselines. Together, these results show that a shared VLA backbone is capable of achieving decentralized multi-robot collaboration, without per-robot policies or inter-robot communication at inference.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **CHORUS** 这篇论文的分析如下：

### 1. 核心贡献摘要
该论文提出了一种名为 **CHORUS** 的新型框架，旨在通过单一的预训练视觉-语言-动作（VLA）模型实现多机器人协作，且无需中央协调器或推理时的实时通信。通过利用 VLA 模型强大的预训练视觉运动先验，每个机器人仅依靠局部观测和身份提示即可实现高效的去中心化协作。

### 2. 关键创新与方法论
*   **单一 VLA 骨干模型（Unified VLA Backbone）：** 核心创新在于跳出了“针对每个机器人训练策略”或“集中式状态聚合”的传统范式，而是将所有机器人的行为统一到一个共享的 VLA 模型中。
*   **去中心化推断（Decentralized Inference）：** 机器人无需在推理时共享观测信息或进行复杂的对齐，仅需通过一个“机器人标识符（Robot-identifying prompt）”即可在该模型中切换角色，实现“各司其职”的协作。
*   **利用视觉运动先验：** 该方法论证了大型视觉-语言模型中隐含的知识不仅能处理单机任务，还能在无需明确训练协调协议的情况下，通过观察队友行为产生“涌现”的协作能力（Reactive collaboration）。

### 3. 对计算机视觉领域的潜在影响
该研究极具重要性，因为它挑战了多机器人协同必须依赖“强实时通信”或“集中式大模型”的假设：
*   **范式转移：** 它展示了 VLA 不仅仅是语义理解工具，还可以作为**具身智能的协同基座**。
*   **计算效率与扩展性：** 通过去除推理时的通信依赖，极大提升了多机器人系统的鲁棒性和可扩展性，为解决“多智能体系统（MAS）在复杂环境下通信延迟”的问题提供了一种纯视觉驱动的路径。

### 4. 受益的相关领域与应用
*   **协作机器人（Cobots）：** 如文中提到的搬运重物、协作组装等仓储和物流任务。
*   **服务机器人：** 在家居或图书馆等非结构化环境中，多个异构机器人通过自然语言指令进行协同作业。
*   **受限环境下的机器人集群：** 在地下、水下或空间探索等通信受限场景中，去中心化协作具有极高的实用价值。
*   **人机协作（HRC）：** 该技术可扩展至机器人与人类队友的实时动态配合中，因为人类也可以被视为系统中的“另一个智能体”。

### 5. 可推断的局限性
*   **训练数据的规模与多样性：** 尽管实现了去中心化，但该方法依赖于 VLA 预训练模型及针对多智能体协作场景的微调，训练数据中包含多样化队友行为的难度较大。
*   **隐式协作的局限性：** 论文虽然展示了优异的反应性，但纯粹基于视觉的推断在处理超长时序、复杂依赖的协同任务时，是否会因为缺乏显式通信机制而出现“决策冲突”或“死锁”现象，仍有待观察。
*   **身份区分的鲁棒性：** 仅依靠“机器人标识符” prompt 来区分职能，在机器人数量激增（如百台规模）或队友职能互换的情况下，模型的泛化能力和策略稳定性可能受到限制。

**总结评价：** CHORUS 是一项将生成式预训练模型与具身机器人学深度融合的标杆性工作。它证明了在大模型时代，**“通过视觉观测进行感知与反应”比“通过通信进行显式指令对齐”更具落地潜力和鲁棒性**，是机器人迈向更复杂协作环境的重要一步。

**Key Findings:**

- We propose CHORUS, a framework that adapts a single VLA backbone to control diverse, multi-robot teams.
- In real-world experiments including mobile tape measurement, library book handovers, and laundry basket lifting, CHORUS achieves a 64% point improvement over decentralized, from-scratch models, improves reactivity to teammate behavior by 40% points, and outperforms centralized baselines.
- Together, these results show that a shared VLA backbone is capable of achieving decentralized multi-robot collaboration, without per-robot policies or inter-robot communication at inference.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12352v1)
- [arXiv](https://arxiv.org/abs/2606.12352v1)

---

<a id='2606.12105v1'></a>
## [DAM-VLA: Decoupled Asynchronous Multimodal Vision Language Action model](https://arxiv.org/abs/2606.12105v1)

**Authors:** Pankhuri Vanjani, Zhuoyue Li, Jakub Suliga, Moritz Reuss, Gianluca Geraci, Xinkai Jiang, Rudolf Lioutikov

**Published:** 2026-06-10

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Vision-language-action (VLA) models inherit a shared synchronous clock from vision-language pretraining, processing every input at one rate. This is misaligned with physical interaction, where a high-frequency modality changes at hundreds of hertz, vision evolves more slowly, and language stays constant across an episode. A synchronous VLA oversamples slow modalities, undersamples fast ones, and caps action generation at the lowest effective frequency. We hypothesize that decoupling temporal processing per modality, letting each update and retain information at its own sensor rate, yields stronger representations and more robust control. We present DAM-VLA, which maintains per-modality latent buffers refreshed at sensor rates and read continuously by the action head, integrating new high-frequency modalities through gated cross-attention that leaves the pretrained backbone intact. Across seven contact-rich real-world manipulation tasks, DAM-VLA more than doubles the average success rate of the strongest synchronous baseline (95.2\% vs.\ 40.95\%) while sustaining smooth, reactive 100\,Hz control. Project website: \href{https://intuitive-robots.github.io/DAM-VLA/}{intuitive-robots.github.io/DAM-VLA/}

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作（VLA）模型继承了视觉-语言预训练的同步时钟，即以单一速率处理所有输入。这与物理交互相冲突，因为高频模态（如力矩）在数百赫兹变化，而视觉演变较慢，语言则贯穿始终。同步VLA过度采样慢速模态、欠采样快速模态，并将动作生成限制在最低频率。我们假设：按模态解耦时间处理，让每个模态根据传感器速率更新并保留信息，能产生更强的表示和更鲁棒的控制。为此，我们提出了**解耦异步多模态视觉-语言-动作模型（DAM-VLA）**。它维护按模态划分的潜变量缓冲区，该缓冲区以传感器速率刷新，并由动作头连续读取；通过门控交叉注意力机制集成高频模态，从而保持预训练主干完整。在七项接触密集型真实机器人任务中，DAM-VLA的平均成功率是基线的两倍多（95.2% 对 40.95%），同时支持平滑、反应灵敏的 100 Hz 控制。

### 2. 方法动机分析
- **驱动力**：解决VLA模型在机器人控制中的“时钟失配”问题。
- **痛点**：
  1. **冗余计算**：高昂的VLM编码器重复处理语义相同的图像。
  2. **跨模态速率失配**：同步时钟导致欠采样高频传感器（丢失瞬时接触）和过采样低频传感器（浪费算力）。
  3. **动作延迟**：策略被绑在最慢模态的更新周期上，导致控制响应迟钝。
- **研究假设**：通过将各模态时间处理解耦，保留各自的传感器速率与时间视野，能构建反映真实物理结构而非投影结构的鲁棒表示。

### 3. 方法设计详解
- **核心Pipeline**：
  1. **独立异步缓冲**：视觉以低频（25Hz）更新，力矩/本体感知以高频（100Hz）持续采样。每个模态拥有自己的潜变量缓冲区（Latent Buffer）。
  2. **持续读取机制**：动作头在每个控制周期（100Hz）读取所有缓冲区，即便是视觉缓存，若未更新则继续读取历史Token，从而解耦编码与决策周期。
  3. **门控交叉注意力（GCA）**：
     - **视觉记忆路径**：使用零初始化的残差连接，由压缩的视觉Token序列调节动作。
     - **输入依赖门（Input-Dependent Gate）**：针对力矩传感器，引入门控机制学习其在何时是“信息丰富”的（如接触瞬间），避免噪声污染预训练特征。
- **模型结构**：基于X-VLA backbone，通过GCA在Transformer层间插入模态信息，确保不破坏预训练权重。
- **算法精髓**：公式 (2) 和 (3) 展示了纯相加的增量更新（Additive Delta），保证了条件调节路径的独立性。

### 4. 方法对比分析
- **本质区别**：从传统的“同步Bundle输入”转变为“异步流式读取”。
- **创新贡献**：提出了一种既能利用预训练模型语义，又能实现实时、高响应接触控制的异步架构，通过门控机制有效消除了多模态冲突。
- **适用场景**：高频接触式操作（如插拔、擦拭、精密装配）。

### 5. 实验分析
- **关键结论**：在7项任务中，DAM-VLA达到95.2%的成功率，相较于Naive高频扩展（X-VLA100）的21.9%有巨大提升。
- **主要优势**：实现了低延迟、高流畅的控制，显著减少了犹豫和冗余运动。
- **主要局限**：视觉编码仍固定周期更新，而非事件触发，导致在剧烈场景切换时存在感知滞后。

### 6. 实用指南
- **开源情况**：官方提供了项目页面：`intuitive-robots.github.io/DAM-VLA/`。
- **实现细节**：
  - **超参数**：Force/Proprioception使用100Hz，Visual stride S=8（历史帧窗口）。
  - **训练技巧**：确保训练期间对齐时间轴，推理时保持buffer平稳。
- **迁移可能**：该架构对传感器类型解耦，可轻松接入触觉传感器或激光雷达等不同频率模态。

### 7. 总结
- **核心思想**：按模态原生频率异步解耦，通过门控跨模态路径实现实时决策。
- **速记版pipeline**：
  1. 传感器原生速率采样。
  2. 独立缓冲存储语义/状态信息。
  3. 动作头实时读取缓存。
  4. 门控机制动态注入修正信号。

**Key Findings:**

- We present DAM-VLA, which maintains per-modality latent buffers refreshed at sensor rates and read continuously by the action head, integrating new high-frequency modalities through gated cross-attention that leaves the pretrained backbone intact.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12105v1)
- [arXiv](https://arxiv.org/abs/2606.12105v1)

---

<a id='2606.12412v1'></a>
## [Reroute, Don't Remove: Recoverable Visual Token Routing for Vision-Language Models](https://arxiv.org/abs/2606.12412v1)

**Authors:** Cheng-Yu Yang, Shao-Yuan Lo, Yu-Lun Liu

**Published:** 2026-06-10

**Categories:** cs.CV, cs.AI

**Abstract:**

Vision-language models (VLMs) project images into hundreds to thousands of visual tokens, making decoder inference expensive in both attention computation and KV-cache memory. Existing visual-token reduction methods largely follow a rank-and-remove paradigm: they score visual tokens, keep a compact subset, and permanently discard the rest. We show that this irreversible action is fragile because visual-token importance changes across decoder depth; tokens ranked low at one stage may become relevant in later layers, especially for grounding-sensitive queries. We propose Reroute, a training-free plug-in that replaces removal with recoverable routing. At each routing stage, selected vision tokens pass through decoder blocks, while deferred tokens bypass the stage and re-enter the candidate pool at the next routing decision. Reroute reuses existing attention-score ranking rules and stage-wise schedules, preserving the theoretical TFLOPs and KV-cache budget class of the pruning method it augments. Across FastV, PDrop, and Nüwa variants on LLaVA-1.5 and Qwen backbones, reroute improves grounding under aggressive token reduction while maintaining general VQA performance. These results suggest that VLM token reduction should not be viewed only as irreversible pruning, but also as recoverable routing. The code can be found here: https://github.com/elmma/mllm-reroute/

**Analysis:**

### 1. 摘要翻译
视觉语言模型（VLMs）将图像映射为数百至数千个视觉Token，导致解码器在注意力计算和KV缓存内存方面开销巨大。现有的视觉Token缩减方法主要遵循“排名并移除”（rank-and-remove）范式：对Token评分、保留一小部分，并永久丢弃其余部分。我们发现这种不可逆的操作是脆弱的，因为视觉Token的重要性会随解码器深度变化；在浅层排名靠后的Token在深层可能变得重要，尤其是在需要精细定位（grounding）的查询中。我们提出了 **Reroute**，一个无需训练的插件，它用“可恢复路由”（recoverable routing）取代了“移除”。在每个路由阶段，被选中的视觉Token通过解码器块，而延迟（deferred）的Token绕过当前阶段，并重新进入下一阶段的候补池中进行筛选。Reroute复用了现有的注意力分数排名规则和分阶段调度，在保持原剪枝方法理论算力（TFLOPs）和KV缓存预算的同时，提升了模型性能。实验表明，Reroute在LLaVA-1.5和Qwen骨干网络上，不仅维持了通用视觉问答（VQA）性能，还显著改善了高压缩率下的视觉定位能力。

### 2. 方法动机分析
*   **驱动力**：作者质疑“Token重要性在解码全过程中恒定”这一隐性假设。研究发现，浅层解码器关注点发散，目标相关区域往往在深层才显现。
*   **现有方法痛点**：传统剪枝属于“一次性裁决”，一旦Token被丢弃，后续即便该Token变得关键也无法恢复，导致在高压缩率下出现严重的性能坍塌。
*   **研究假设**：通过引入“可恢复性”，将不可逆的剪枝转化为可动态调整的路由过程，可以利用Token随深度变化的动态重要性，实现更高效的算力分配。

### 3. 方法设计详解
*   **流程总结**：
    1.  **分阶段排名**：解码器被分为$S$个阶段，每阶段根据当前的文本-视觉注意力权重对所有候选Token进行打分。
    2.  **Top-K选择**：根据预设的保留率$r_i$，选出Top-K个Token进行标准的解码器注意力与FFN计算。
    3.  **动态旁路（Bypass）**：未选中的Token（deferred tokens）不会被删除，而是通过残差连接直接进入下一阶段。
    4.  **重新入选**：在后续阶段，这些延迟Token重新加入候选池，与全量或部分剩余Token再次竞争入选资格。
*   **关键公式与设计**：
    *   模型在阶段$i$的输出更新为：$h_j^{\ell+1} = \text{Block}_\ell(H_A^\ell)_j$（若$j \in A_i$）或直接保持原值（若$j \in V_{def}$）。
    *   **无需训练**：Reroute直接复用原剪枝方法已有的注意力打分逻辑，无需额外参数，这使其成为即插即用的轻量化插件。

### 4. 方法对比分析
*   **本质区别**：从“不可逆的删除”变为“可恢复的延迟”，实现了类似Mixture-of-Depth（MoD）的动态计算，但不需要显式训练路由门控。
*   **创新点**：将剪枝视为一种退化的路由策略，通过简单的残差旁路机制打破了“一旦移除，永久消失”的约束。
*   **适用场景**：极度追求推理速度、显存占用，同时对视觉定位精度有高要求的场景（如密集目标检测、细粒度图像理解）。

### 5. 实验分析（精简版）
*   **验证方法**：在LLaVA-1.5、Qwen2.5-VL、Qwen3.5-VL等多个基线上，对FastV、PDrop等剪枝方法进行增强测试。
*   **关键结果**：在88.9%的极致高压缩率下，Reroute将基线方法的定位IoU从不到0.4显著提升至0.8以上。
*   **优势**：在保持原推理预算（TFLOPs/KV缓存）的前提下，显著提升了模型对关键视觉细节的捕捉能力。
*   **局限**：推理延迟在实践中依赖于高效的Tensor切片和Gather/Scatter算子实现。

### 6. 实用指南
*   **开源情况**：已开源，GitHub地址：`https://github.com/elmma/mllm-reroute/`。
*   **迁移建议**：该方法非常适合任何基于“打分+丢弃”的视觉Token剪枝架构。迁移时只需将“删除”操作改为“残差旁路”，并确保后续层能够处理动态变化的Token序列长度。
*   **实现细节**：建议在解码器早期和中期设置多个分布均匀的路由检查点。

### 7. 总结
*   **核心思想**：变“一次性剔除”为“动态竞争入选”，实现视觉Token的可恢复性路由。
*   **速记版Pipeline**：
    1. 计算Token重要度；
    2. 选出Top-K执行深层计算；
    3. 未选中的Token通过旁路保留；
    4. 下一阶段它们重新参与竞争。

**Key Findings:**

- We show that this irreversible action is fragile because visual-token importance changes across decoder depth; tokens ranked low at one stage may become relevant in later layers, especially for grounding-sensitive queries.
- We propose Reroute, a training-free plug-in that replaces removal with recoverable routing.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12412v1)
- [arXiv](https://arxiv.org/abs/2606.12412v1)

---

<a id='2606.12403v1'></a>
## [World Pilot: Steering Vision-Language-Action Models with World-Action Priors](https://arxiv.org/abs/2606.12403v1)

**Authors:** Zefu Lin, Rongxu Cui, Junjia Xu, Xiaojuan Jin, Wenling Li, Lue Fan, Zhaoxiang Zhang

**Published:** 2026-06-10

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models inherit semantic grounding from large-scale pretraining and perform competently across in-distribution manipulation tasks. This grounding, however, is built on static image-text pairs, whereas manipulation is a continuous, contact-rich process whose dynamics this pretraining cannot capture. We present World Pilot, a VLA framework that augments the policy with priors from a World-Action Model (WAM), routed into the decision chain through two complementary pathways. Latent Steering conditions the perception layer on a scene-evolution latent, and Action Steering supplies an anticipated trajectory as a motion prior to the action generator. Together the two priors equip the VLA with an anticipated view of the scene and a trajectory-level motion hint alongside its semantic conditioning, and the scene-evolution prior remains effective even when supplied by a video-pretrained world model that has not been action-post-trained. World Pilot attains a state-of-the-art Total success rate of 84.7% on the LIBERO-Plus zero-shot OOD benchmark and the highest success rate on every real-robot setting across four manipulation tasks, with the largest margins under shifts in viewpoint, geometry, deformable state, and pose. Project Website: https://world-pilot.github.io/

**Analysis:**

这是一份关于论文《World Pilot: Steering Vision-Language-Action Models with World-Action Priors》的深度技术分析。

---

### 1. 摘要翻译
视觉-语言-动作（VLA）模型继承了大规模预训练带来的语义基础，并在分布内操作任务上表现出色。然而，这种基础建立在静态图像-文本对之上，而操作过程本质上是连续且充满接触的，预训练无法捕捉其动力学规律。我们提出了“World Pilot”，一个通过World-Action模型（WAM）提供的先验信息增强策略的VLA框架。这些先验信息通过两条互补路径引入决策链：*Latent Steering（潜空间引导）*将场景演化潜变量注入感知层；*Action Steering（动作引导）*向动作生成器提供预期的轨迹作为运动先验。两项先验共同为VLA赋予了对场景的预期视角和运动提示。World Pilot在LIBERO-Plus零样本OOD基准上达到了84.7%的SOTA成功率，并在所有真实机器人操作任务中表现最优，特别是在视点、几何结构、变形状态及位姿发生偏移时优势显著。

### 2. 方法动机分析
*   **驱动力**：旨在弥补VLA模型在处理动态操作任务时对物理世界演化缺乏理解的缺陷，利用预训练视频模型（WAM）的动力学知识来指导动作生成。
*   **痛点**：传统VLA基于静态图像-文本对训练，导致其对视点变化、物体形变等分布外（OOD）场景极其脆弱。
*   **研究假设**：视频模型不仅能通过压缩的潜变量提供场景演化信息，还能提供粗略的轨迹假设，将这两者以非侵入式（Additive）方式注入VLA，能显著提升策略的泛化性和鲁棒性。

### 3. 方法设计详解
*   **核心模块**：
    1.  **World-Action Model (WAM)**：一个冻结的预训练模型，输入观测，输出场景演化潜变量 $Z_t^w$ 和粗略的动作轨迹 $A_t^w$。
    2.  **Latent Steering**：将 $Z_t^w$ 通过动态编码器映射为未来场景令牌 $D_t^w$。VLM的隐藏状态 $H_t$ 通过Cross-Attention与 $D_t^w$ 交互，注入时空动态感知。
    3.  **Action Steering**：将 $A_t^w$ 压缩为单个前缀令牌 $s_t^w$，直接拼接到流匹配（flow-matching）动作生成器的输入序列中，引导生成过程朝着正确的运动趋势演化。
*   **算法逻辑**：两步注入均为加性更新（Residual Update），不改变原VLA的token结构。通过保持WAM冻结，仅微调VLA，有效实现了“动力学知识通过先验注入”的解耦架构。

### 4. 方法对比分析
*   **本质区别**：不同于以往将WAM作为环境仿真器（如DreamVLA）或预测未来像素（如VISTA），World Pilot仅抽取**高层潜变量与轨迹先验**，避开了像素级生成带来的噪声干扰。
*   **创新贡献**：设计了针对“感知侧（Latent Steering）”和“生成侧（Action Steering）”的双路径引导机制，且验证了即使没有经过动作微调的通用视频模型，其先验依然有效。
*   **适用场景**：适用于需要长程规划、对环境变换敏感的机器人操作任务。

### 5. 实验分析
*   **验证方法**：在LIBERO-Plus OOD基准（涵盖7种扰动）及真实机器人任务（堆叠、折叠等）上进行评估。
*   **关键结论**：在LIBERO-Plus上总成功率达84.7%，较最强基线提升2.6个点；在真实机器人实验中，在OOD环境下的性能衰减明显小于传统VLA。
*   **核心优势**：极强的OOD鲁棒性；模块化设计，WAM可插拔。
*   **主要局限**：推理时需额外运行一次WAM，增加了计算开销；对高频实时控制存在限制。

### 6. 实用指南
*   **开源情况**：项目主页已公布（https://world-pilot.github.io/）。
*   **实现建议**：注意将WAM的输出通过Dropout（实验中为0.3）进行正则化，以防止VLA过分依赖先验信号而忽略了自身的语义感知。
*   **迁移迁移**：方法可直接迁移至任何基于VLM的动作生成架构，只需将先验注入到VLM的Cross-Attention层和动作生成器的输入层即可。

### 7. 总结
*   **核心思想**：通过双路径注入视频世界的动力学先验，引导VLA进行更符合物理规律的操作。
*   **速记版Pipeline**：
    1.  视频模型预判场景演化与动作走向。
    2.  将演化信息注入视觉特征中增强理解。
    3.  将轨迹先验压缩为令牌引导生成。
    4.  冻结视频模型，仅微调策略执行器。

**Key Findings:**

- We present World Pilot, a VLA framework that augments the policy with priors from a World-Action Model (WAM), routed into the decision chain through two complementary pathways.
- World Pilot attains a state-of-the-art Total success rate of 84.7% on the LIBERO-Plus zero-shot OOD benchmark and the highest success rate on every real-robot setting across four manipulation tasks, with the largest margins under shifts in viewpoint, geometry, deformable state, and pose.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12403v1)
- [arXiv](https://arxiv.org/abs/2606.12403v1)

---

<a id='2606.12396v1'></a>
## [VLGA: Vision-Language-Geometry-Action Models for Autonomous Driving](https://arxiv.org/abs/2606.12396v1)

**Authors:** Jin Yao, Dhruva Dixith Kurra, Tom Lampo, Zezhou Cheng, Danhua Guo, Burhan Yaman

**Published:** 2026-06-10

**Categories:** cs.CV, cs.RO

**Abstract:**

Vision-language-action (VLA) models can describe scenes and reason about them in language, yet still struggle to ground their actions in the dense 3D world around them. Existing approaches either inject features from a frozen 3D foundation model without an objective that ensures the policy uses them, or constrain geometry with sparse box and map losses that provide no dense spatial signal. We introduce VLGA, the first vision-language-action model supervised to reconstruct the dense 3D world it drives through. VLGA introduces geometry as a fourth modality alongside vision, language, and action through a dedicated expert supervised by a per-pixel pointmap regression loss against LiDAR. Extensive experiments conducted on challenging nuScenes and Bench2Drive datasets for open-loop and closed-loop evaluations, respectively, show the superiority of VLGA over counterpart VLA methods. In particular, on open-loop nuScenes, VLGA sets a new state of the art among VLA methods without ego status, with the lowest L2 (0.50\,m average) and 3-second collision rate (0.18\%). On closed-loop Bench2Drive, VLGA attains the state-of-the-art driving score of 79.08, +0.71 over the strongest prior VLA, at comparable efficiency and comfort.

**Analysis:**

## 1. 摘要翻译
视觉-语言-动作（VLA）模型虽能理解和推理场景，但难以将动作锚定在稠密的3D现实世界中。现有方法要么注入来自冻结3D基础模型的特征且缺乏确保策略使用该特征的客观目标，要么仅依赖稀疏的方框和地图损失来约束几何，缺乏稠密的空间信号。我们引入了VLGA，这是首个通过监督重建其行驶的稠密3D世界来训练的VLA模型。VLGA通过一个专门的专家引入“几何”作为继视觉、语言和动作之后的第四种模态，并通过针对LiDAR的逐像素点图回归损失进行监督。在nuScenes和Bench2Drive数据集上的广泛实验表明，VLGA优于同类VLA方法。特别是在开放环的nuScenes上，VLGA在无需自我状态（ego status）的情况下达到了VLA方法的最优水平，具有最低的L2平均误差（0.50米）和3秒碰撞率（0.18%）。在闭环的Bench2Drive上，VLGA达到了79.08的驾驶评分，比最强的现有VLA高出+0.71，且具备相当的效率和舒适度。

## 2. 方法动机分析
*   **驱动力**：VLA模型在语义理解上表现出色，但在处理需要连续空间精确度的驾驶规划任务时，往往因缺乏对场景结构的深刻理解而产生偏差。
*   **痛点**：
    *   **稀疏感知**：仅输出3D框或地图点，无法提供细粒度的空间地面真实性。
    *   **注入式特征**：将3D特征融入LLM中缺乏显式的几何监督，模型可能忽略这些特征。
    *   **几何优先路径**：虽然实现了稠密几何，但通常牺牲了语言推理能力。
*   **研究假设**：通过在VLA架构中加入一个参数独立的几何模态（Expert），并配合显式的稠密3D重建监督，可以使模型在保持语言推理能力的同时，显著提升规划的安全性与空间精确度。

## 3. 方法设计详解
*   **架构概览**：VLGA采用“混合专家（MoT）”架构，包含四个Expert：理解（U）、感知（P）、几何（G）、动作（A）。
*   **流程细节**：
    1.  **输入处理**：多视图相机输入通过视觉编码器与预训练的几何骨干网络（DVGT-2）。
    2.  **几何专家构建**：几何骨干网络产生稠密特征，通过投影器（Projector）映射至MoT token空间，作为G专家的输入。
    3.  ** masked joint attention**：各专家通过联合注意力机制进行交互，其中动作专家A显式地融合了来自G专家的空间信息。
    4.  **监督训练**：核心创新在于引入了一个轻量级Transformer解码器D，专门用于将几何token重构成逐像素的3D点图，并使用LiDAR数据进行监督，该解码器仅在训练时使用。
*   **算法关键**：采用了置信度加权的回归损失（$\mathcal{L}_{\text{pmap}}$），结合了距离回归与不确定性估计，有效监督几何流的准确性。

## 4. 方法对比分析
*   **本质区别**：与现有VLA（如UniDriveVLA）的主要区别在于它不是简单地让VLM去处理3D特征，而是通过显式的“几何重建损失”强制模型“学习”稠密的3D空间表示。
*   **创新点**：
    1.  **几何作为第四模态**：参数独立且专门化。
    2.  **两阶段训练策略**：先通过几何任务预热，再进行联合任务微调，避免了多任务干扰。

## 5. 实验分析
*   **关键结论**：在nuScenes上，无需ego状态输入的情况下，L2误差和碰撞率均刷新最优纪录；在闭环Bench2Drive上，驾驶评分提升至79.08。
*   **优势**：在长视距碰撞率和需要精细偏移控制的场景下（如“礼让”、“紧急制动”）表现尤为卓越。
*   **局限**：推理成本较高，主要源于庞大的VLM底座，暂未优化至边缘计算水平。

## 6. 实用指南
*   **开源信息**：项目主页：[yaojin17.github.io/VLGA](https://yaojin17.github.io/VLGA)。
*   **迁移与实现**：核心在于几何特征的“投影”以及针对点图的回归监督。该方法可迁移至任何基于MoT架构的自动驾驶模型，只需替换或增加一个独立Geometry Expert模块即可。
*   **注意**：训练分为几何和联合两个阶段，必须先冻结非几何模块以“温育”几何专家的权重，否则多模态联合训练早期会导致性能崩坏。

## 7. 总结
*   **核心思想**：通过显式几何重建监督，为VLA模型注入“稠密空间感知力”。
*   **速记版Pipeline**：
    1. 提取多视图稠密视觉特征。
    2. 通过独立专家处理几何信息。
    3. 联合注意力层交互专家特征。
    4. 用LiDAR回归损失监督空间重建。
    5. 动作专家依据几何感知输出轨迹。

**Key Findings:**

- We introduce VLGA, the first vision-language-action model supervised to reconstruct the dense 3D world it drives through.
- In particular, on open-loop nuScenes, VLGA sets a new state of the art among VLA methods without ego status, with the lowest L2 (0.50\,m average) and 3-second collision rate (0.18\%).
- On closed-loop Bench2Drive, VLGA attains the state-of-the-art driving score of 79.08, +0.71 over the strongest prior VLA, at comparable efficiency and comfort.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12396v1)
- [arXiv](https://arxiv.org/abs/2606.12396v1)

---

<a id='2606.12372v1'></a>
## [UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning](https://arxiv.org/abs/2606.12372v1)

**Authors:** Haoyuan Deng, Yitong Gao, Yudong Lin, Haichao Liu, Zhenyu Wu, Ziwei Wang

**Published:** 2026-06-10

**Categories:** cs.RO, cs.LG

**Abstract:**

Human-in-the-loop reinforcement learning (HiL-RL) has emerged as an effective paradigm for real-world robotic manipulation, enabling online policy improvement with human guidance. However, current HiL-RL frameworks remain intervention-intensive, relying on frequent human corrections to redirect the policy out of unproductive exploration, which incurs high labor cost and limits real-world scalability. To address this, we propose UniIntervene, an agentic intervention model that detects unproductive exploration and autonomously recovers the policy toward high-value states, taking over the bulk of interventions from human operators. Specifically, UniIntervene first performs future-conditioned action-value estimation, predicting the latent consequence of the current action and evaluating its induced value, which provides a more stable progress signal. Building on this, a temporal value-risk critic aggregates recent value dynamics and triggers intervention when the estimated value exhibits sustained stagnation or degradation. When intervention is required, UniIntervene retrieves a high-value recovery target from a memory of past intervention episodes and produces executable corrective actions through a goal-conditioned recovery policy. In this way, UniIntervene turns intervention from passive human correction into a value-aware recovery process for efficient real-world RL. Extensive experiments on diverse real-world manipulation tasks demonstrate that UniIntervene improves the average success rate by 8.6% while reducing human interventions by 57% relative to state-of-the-art HiL-RL baselines.

**Analysis:**

以下是对论文《UniIntervene: Agentic Intervention for Efficient Real-World Reinforcement Learning》的深度分析。

### 1. 摘要翻译
人机协同强化学习（HiL-RL）是现实世界机器人操作的一种有效范式。然而，现有的HiL-RL框架仍高度依赖人工介入来纠正低效的探索行为，这不仅产生了高昂的劳动成本，也限制了其实际扩展性。为此，我们提出了UniIntervene，这是一种代理干预模型，能够自主检测低效探索，并将策略恢复至高价值状态，从而承担了原本由人工执行的大部分干预任务。具体而言，UniIntervene通过未来条件动作价值估计，预测当前动作的潜在后果，提供更稳定的进展信号；在此基础上，时序价值-风险评论员（temporal value-risk critic）聚合近期价值动态，在检测到停滞或降级时触发干预。一旦触发，模型从过往干预片段库中检索高价值恢复目标，并通过目标条件恢复策略产生可执行的纠正动作。实验表明，该方法在各类现实操作任务中将平均成功率提高了8.6%，并将人为干预减少了57%。

### 2. 方法动机分析
- **驱动力**：旨在解决HiL-RL中“过度依赖人工”的瓶颈，使机器人能够自主识别并修复“非安全但无进展”的探索状态。
- **现有方法痛点**：传统方法要么将人视为“标签提供者”，无法自主判断何时干预；要么仅具备防御性的安全约束（如防碰撞），无法识别虽无危险但进度停滞的低效行为。
- **研究假设**：通过预测未来状态并结合时序价值趋势，可以量化地检测出系统是否进入了“productive”与“stagnant”的临界点。

### 3. 方法设计详解
- **pipeline总结**：
  1. **未来价值估计**：利用Qwen-VL作为基座，预测当前动作后的latent状态，并将其映射为价值分数。
  2. **时序价值-风险检测（TVR）**：利用滑窗机制，计算近期价值增长的短缺量，一旦低于期望阈值即视为“停滞”，触发干预。
  3. **记忆引导的恢复**：从离线的高价值成功经验库中检索语义最接近的“目标状态”，作为纠正指引。
  4. **目标条件动作生成**：通过基于目标状态的策略，利用行为克隆（BC）生成纠正动作片段，将机器人拉回正确的轨道。
- **核心逻辑**：将干预决策从“被动响应”转变为“内部自主的价值监控”。

### 4. 方法对比分析
- **本质区别**：与传统的基于故障检测或安全规则不同，UniIntervene是基于**进展价值（Progress Value）**的动态监控。
- **创新贡献**：
    1. **时序价值-风险指标**：不仅看当前价值，更看价值变化趋势，有效过滤了短时波动。
    2. **记忆库机制**：不仅检测到错误，还通过检索实现了“精准修复”，避免了盲目试错。

### 5. 实验分析
- **主要结论**：在5项真实操作任务中，成功率领先，且干预频率显著下降。
- **核心优势**：在接触密集（contact-rich）任务中表现出极强的鲁棒性，因为它能区分正常的对齐摩擦与真正的停滞。
- **局限性**：依赖于离线数据分布，若遇到训练集从未覆盖的新奇错误，仍需人类介入。

### 6. 实用指南
- **实现细节**：
    - 需要预先构建Proxy Value Function（用成功/失败轨迹训练）。
    - 触发机制（TVR）的窗口长度（$K=8$）和阈值需针对具体任务微调。
- **迁移建议**：该方法逻辑通用，可直接迁移至任何具备离线成功演示数据的机器人控制任务。只需更换V-JEPA2 encoder以适配不同传感器模态，并重构对应任务的检索记忆库。

### 7. 总结
- **核心思想**：通过预测未来价值趋势自主检测低效，并利用记忆指导实现无人干预恢复。
- **速记版pipeline**：
    1. **监控**：实时估计未来动作价值。
    2. **检测**：计算滑动窗口内价值停滞趋势。
    3. **检索**：在记忆库匹配高价值修复目标。
    4. **恢复**：执行目标导向的纠正策略。

**Key Findings:**

- To address this, we propose UniIntervene, an agentic intervention model that detects unproductive exploration and autonomously recovers the policy toward high-value states, taking over the bulk of interventions from human operators.
- Extensive experiments on diverse real-world manipulation tasks demonstrate that UniIntervene improves the average success rate by 8.6% while reducing human interventions by 57% relative to state-of-the-art HiL-RL baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12372v1)
- [arXiv](https://arxiv.org/abs/2606.12372v1)

---

<a id='2606.12371v1'></a>
## [A Turbo-Inference Strategy for Object Detection and Instance Segmentation](https://arxiv.org/abs/2606.12371v1)

**Authors:** Zhen Zhao, Gang Zhang, Xiaolin Hu, Liang Tang

**Published:** 2026-06-10

**Categories:** cs.CV

**Abstract:**

Object detection and instance segmentation tasks are closely related. Existing top-down instance segmentation methods usually follow a detect-then-segment paradigm, where an initial detector is used to recognize and localize objects with bounding boxes, followed by the segmentation of an instance mask within each bounding box. In such methods, the detection accuracy directly influences the subsequent segmentation performance. However, previous research has seldom explored the impact of the instance segmentation task on object detection. In this paper, we present a turbo-inference strategy for the top-down methods that leverages the complementary information between detection and segmentation tasks iteratively. Specifically we design two modules: turbo-detection head and turbo-segmentation head, which facilitate communication between the tasks. The two modules form a closed loop that interlaces the detection and segmentation results without retraining the model. Comprehensive experiments on the COCO, iFLYTEK, and Cityscapes datasets demonstrate that our method substantially enhances both detection and segmentation accuracies with a certain increase in computational cost. The proposed method represents a tradeoff between prediction accuracy and inference speed. Codes are available at https://github.com/zhaozhen2333/Turbo-Learning.git.

**Analysis:**

### 1. 摘要翻译
物体检测与实例分割任务紧密相关。现有的自顶向下（top-down）实例分割方法通常采用“先检测后分割”范式，即首先利用检测器进行定位，随后在检测框内执行分割。在此类方法中，检测精度直接影响分割性能。然而，过往研究较少探讨实例分割任务对物体检测的反向影响。本文提出了一种用于自顶向下方法的“涡轮推理（Turbo-Inference）”策略，通过迭代利用检测与分割任务间的互补信息。具体而言，我们设计了两个模块：涡轮检测头和涡轮分割头，实现了任务间的闭环通信，且无需重新训练模型。在COCO、iFLYTEK和Cityscapes数据集上的实验证明，该方法在保持推理速度可控的前提下，显著提升了检测与分割精度。代码已开源：https://github.com/zhaozhen2333/Turbo-Learning.git。

### 2. 方法动机分析
- **驱动力**：利用分割掩码（mask）中蕴含的高精度空间定位信息，反哺并修正粗糙的检测框（bounding box）和置信度，实现检测与分割的协同增益。
- **现有方法痛点**：当前“先检测后分割”范式是单向的，即检测引导分割，但分割输出的高质量空间信息却被浪费；此外，置信度仅由检测器决定，缺乏对掩码质量的综合考量，导致冗余检测框难以被有效剔除。
- **研究假设**：如果将分割结果反馈给检测器进行迭代，模型可以利用掩码的像素级精度进一步微调边界框，同时通过评估掩码的质量（不确定性）优化分类置信度，从而形成闭环优化。

### 3. 方法设计详解
- **流程总结**：
  1. **初始化**：通过基线模型获得检测框 $B$ 和类别置信度 $S$。
  2. **初次分割**：由Vanilla-Seg head预测掩码 $M$。
  3. **涡轮检测头（Turbo-Det）**：
     - **Box Refinement**：将掩码 $M$ 通过双线性插值映射回原图，利用前景像素点计算新的极值坐标，重定义更贴合目标的边界框 $B_{ref}$。
     - **Maskness**：根据掩码的置信度分布计算不确定性得分（$U_{mask}, U_{bbox}$），将初始置信度 $S$ 乘以该得分，过滤冗余低质量框。
  4. **涡轮分割头（Turbo-Seg）**：基于修正后的 $B_{ref}$ 重新提取RoI特征，生成更精准的掩码 $M_{ref}$。
  5. **迭代**：上述过程可重复多次（如Stage 3, 4, 5, 6），进一步收敛。

- **算法关键点**：$U_{mask}$ 衡量了像素在前景/背景区分上的模糊程度。作者通过公式计算不确定性，将 mask 的高质量特征转化为置信度修正量，实现对检测框的有效“过滤”。

### 4. 方法对比分析
- **本质区别**：与传统Cascade R-CNN等仅在检测阶段迭代不同，本文提出了检测与分割之间的跨任务“交叉级联”架构。
- **创新贡献**：无需重新训练模型，作为一个即插即用的推理插件（Turbo-inference），它赋予了传统模型“动态调整”的能力。
- **适用场景**：适用于所有基于“先检测后分割”范式的Top-down实例分割框架。

### 5. 实验分析
- **关键结果**：在Mask R-CNN基线上，该策略实现了1.1%的Box AP提升和1.3%的Mask AP提升。
- **主要优势**：显著提升检测与分割精度，尤其是对边界框定位的校正效果明显。
- **主要局限**：推理时间随迭代次数增加，在FPS上有一定损失；对某些动态卷积架构（如QueryInst）的适配存在轻微性能波动。

### 6. 实用指南
- **开源地址**：https://github.com/zhaozhen2333/Turbo-Learning.git
- **实现细节**：需注意阈值 $S_B$（控制掩码二值化）和 $S_M$（控制置信度过滤）的调节。实验表明，$S_B=0.23$ 是较优设置。
- **迁移可能**：该策略极易迁移至其他检测任务，只需确保网络结构支持在推理阶段输入额外的掩码反馈信息。

### 7. 总结
- **核心思想**：通过掩码信息反哺检测器，实现检测与分割的闭环迭代优化。
- **速记版pipeline**：
  1. 检测器产生初始框和掩码。
  2. 用掩码的极值点修剪框。
  3. 用掩码的置信度滤除烂框。
  4. 基于新框重做更精细的分割。

**Key Findings:**

- In this paper, we present a turbo-inference strategy for the top-down methods that leverages the complementary information between detection and segmentation tasks iteratively.
- Comprehensive experiments on the COCO, iFLYTEK, and Cityscapes datasets demonstrate that our method substantially enhances both detection and segmentation accuracies with a certain increase in computational cost.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12371v1)
- [arXiv](https://arxiv.org/abs/2606.12371v1)

---

<a id='2606.12334v1'></a>
## [Fourier Features Let Agents Learn High Precision Policies with Imitation Learning](https://arxiv.org/abs/2606.12334v1)

**Authors:** Balázs Gyenes, Emiliyan Gospodinov, Jan Frieling, Enrico Krohmer, Nicolas Schreiber, Xiaogang Jia, Niklas Freymuth, Gerhard Neumann

**Published:** 2026-06-10

**Categories:** cs.LG, cs.RO

**Abstract:**

High-precision robotic manipulation requires fine-grained spatial reasoning that is often difficult to achieve with RGB-only policies due to depth ambiguity and perspective scale issues. Policies that leverage 3D information directly, such as those based on point clouds, offer a stronger geometric prior over purely image-based ones, yet their performance remains highly task-dependent. We hypothesize that this discrepancy may be due to the spectral bias of neural networks towards learning low frequency functions, which especially affects architectures conditioned on slow-moving Cartesian features. We thus propose to map point clouds from Cartesian space into high-dimensional Fourier space, effectively equipping the point cloud encoder with direct access to high-frequency features. We experimentally validate the use of Fourier features on challenging manipulation tasks from the RoboCasa and ManiSkill3 benchmarks and on a real robot setup. Despite their simplicity, we find that Fourier features provide significant benefits across diverse encoder architectures and benchmarks and are robust across hyperparameters. Our results indicate that Fourier features let policies leverage geometric details more effectively than Cartesian features, showing their potential as a general-purpose tool for point cloud-based imitation learning. We provide source code and videos on our project page: https://fourier-il.github.io/fourier-il

**Analysis:**

### 1. 摘要翻译
高精度机器人操作需要细粒度的空间推理，由于深度模糊和透视缩放问题，这在仅使用RGB信息的策略中通常难以实现。直接利用点云等3D信息的策略提供了比纯图像策略更强的几何先验，但其性能仍高度依赖任务。我们假设这种差异可能是由于神经网络对学习低频函数的频谱偏差（Spectral Bias）造成的，这尤其影响了以慢变笛卡尔特征为条件的架构。因此，我们提出将点云从笛卡尔空间映射到高维傅里叶空间，从而有效地使点云编码器能够直接访问高频特征。我们在RoboCasa和ManiSkill3基准测试以及真实机器人设置中对傅里叶特征的应用进行了实验验证。尽管傅里叶特征非常简单，但我们发现它们在各种编码器架构和基准测试中都提供了显著的收益，并且在超参数上具有鲁棒性。我们的结果表明，傅里叶特征使策略能够比笛卡尔特征更有效地利用几何细节，展示了其作为点云模仿学习通用工具的潜力。

---

### 2. 方法动机分析
*   **驱动力**：作者试图解决3D点云模仿学习在执行高精度任务（如插拔）时性能受限的问题。
*   **现有痛点**：基于点云的策略通常直接将原始笛卡尔坐标（XYZ）输入神经网络。由于神经网络存在“频谱偏差”，倾向于优先学习低频函数，导致模型难以捕捉场景中微小的几何差异（高频特征），进而无法在需要细粒度动作的任务上表现出色。
*   **核心直觉**：通过傅里叶特征映射（Fourier Feature Mapping），将低维坐标映射到高维空间，通过引入不同频率的三角函数，显式地为神经网络注入高频信号，从而“强制”模型关注几何空间中的高频细节，克服频谱偏差。

---

### 3. 方法设计详解
*   **核心Pipeline**：
    1.  **输入处理**：将原始点云坐标 $p = (x, y, z)$ 归一化。
    2.  **傅里叶映射**：对每个点的坐标应用轴对齐的映射函数 $\gamma(x) = [\sin(2\pi x/\lambda_k), \cos(2\pi x/\lambda_k)]^T$，其中 $\lambda_k$ 为设定的波长。
    3.  **特征融合**：将得到的 $2L$ 维傅里叶特征向量作为点云编码器（如PointPatch, PointTransformer等）的初始输入。
    4.  **扩散模型推理**：编码后的特征进入扩散策略（Diffusion Policy）进行去噪，最终输出动作序列。
*   **关键公式意义**：$\gamma(x)$ 将坐标空间变换到频率空间。通过选择一组 log-spaced 的波长 $\lambda_k$，模型可以同时获取全局位置（长波长）和局部几何细节（短波长）的表征，从而解决传统MLP无法区分空间中极其接近点的问题。

---

### 4. 方法对比分析
*   **本质区别**：本文并未修改原有的PointNet或Transformer骨干网络，而是通过**输入层的预处理（傅里叶特征映射）**在不改变模型容量的情况下改变了其感应偏置（Inductive Bias）。
*   **创新贡献**：提出了一种普适的、非参数化的数据增强方式，验证了它对各种主流点云编码器（PointPatch, DP3, PCM等）均有显著且一致的性能提升。
*   **适用场景**：所有涉及高精度空间操纵、且使用坐标作为输入的点云神经网络模型。

---

### 5. 实验分析（精简版）
*   **验证方法**：在RoboCasa（模拟）、ManiSkill3（模拟）和4项真实世界操纵任务上验证。
*   **关键结果**：成功率平均提升显著（如RoboCasa提升20%，真实世界任务从14.8%提至40.2%），且在剔除高频信息（增加扰动）后，傅里叶特征依然优于基线，说明其同时也优化了学习动力学。
*   **优势**：简单易实现、无需额外训练、对超参数鲁棒。
*   **局限**：在某些本身任务极简单的场景（如ManiSkill3部分任务）改进空间较小。

---

### 6. 实用指南
*   **开源情况**：已开源，详情请见：[https://fourier-il.github.io/fourier-il](https://fourier-il.github.io/fourier-il)。
*   **实现建议**：
    *   **频率带设置**：建议使用 $L=16$ 个频率带，$\lambda_{max}=4.0m, \lambda_{min}=2.0cm$。
    *   **坐标归一化**：必须确保点云被限定在 $[\lambda_{max}/2, \lambda_{max}/2]$ 范围内。
    *   **迁移**：该方法极易迁移，只需在点云Encoder输入层前加一个映射模块，计算图基本无需变动。

---

### 7. 总结
*   **核心思想**：通过傅里叶投影打破神经网络的低频频谱偏置，捕获几何高频细节。
*   **速记版pipeline**：
    1.  提取点云坐标；
    2.  应用傅里叶映射将坐标映射至高维频率空间；
    3.  将高维特征作为Encoder输入；
    4.  执行后续去噪策略预测动作。

**Key Findings:**

- Our results indicate that Fourier features let policies leverage geometric details more effectively than Cartesian features, showing their potential as a general-purpose tool for point cloud-based imitation learning.
- We provide source code and videos on our project page: https://fourier-il.github.io/fourier-il

**Links:**

- [PDF](https://arxiv.org/pdf/2606.12334v1)
- [arXiv](https://arxiv.org/abs/2606.12334v1)

---

