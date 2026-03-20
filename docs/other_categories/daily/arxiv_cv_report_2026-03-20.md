time: 20260320

# Arxiv Computer Vision Papers - 2026-03-20

## Executive Summary

## **Arxiv 计算机视觉领域论文日报执行摘要（2026-03-19）**

---

### **1. 核心主题与趋势**

今日的论文集体现了计算机视觉研究的三个显著交叉趋势：

*   **多模态感知与具身智能的深度融合**：超过一半的论文（如 **OmniVTA**、**Articulated-Body Dynamics Network**、**NavTrust**、**DriveTok**）聚焦于如何让AI系统（尤其是机器人或自动驾驶代理）通过整合视觉、触觉、动力学、语言等多模态信息，来理解和交互于复杂的3D物理世界。这标志着研究重心正从“被动理解”转向“主动交互与决策”。
*   **3D场景理解的生成式先验**：一个突出的技术趋势是利用强大的生成模型（如扩散模型）作为隐式的3D世界知识库，来提升下游的感知与理解任务。**Generation Models Know Space** 和 **DriveTok** 是此方向的典型代表，表明“生成即理解”的范式正在扩展。
*   **基础模型架构的反思与优化**：研究社区正在深入审视现有主流架构（如Vision Transformer）的局限，并探索更高效的替代方案。**Do VLMs Need Vision Transformers?** 和 **FASTER** 分别从基础视觉编码器和实时视觉-语言-动作模型的角度，致力于提升模型的效率、速度或可解释性。

### **2. 重点与创新性论文**

*   **最具范式创新性**：**《Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding》** 提出了一种颠覆性的思路：无需显式3D标注，直接利用预训练的大规模图像/视频生成模型中所蕴含的丰富3D几何与场景先验，来显著提升单目3D场景理解任务的性能。这为利用生成式AI红利解决传统感知难题开辟了新路径。
*   **最具系统性与实用性**：**《OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation》** 系统性地整合视觉与触觉感知，构建世界模型以解决接触丰富的机器人操作难题。其多模态融合方法对实现灵巧、鲁棒的机器人操控具有重要的实用价值。
*   **最具批判性与启发性**：**《Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders》** 直接挑战当前视觉-语言模型的架构基石。它系统评估了状态空间模型作为ViT替代方案的潜力，可能引发关于视觉编码器基础架构的新一轮讨论与探索。

### **3. 新兴研究方向**

1.  **触觉-视觉融合的世界建模**：将触觉反馈正式、深度地整合进视觉世界模型，用于预测物理交互结果，是机器人学的一个新兴前沿（见 **OmniVTA**）。
2.  **动力学启发的机器人学习先验**：将刚体/铰接体动力学作为结构化先验嵌入神经网络（**Articulated-Body Dynamics Network**），为数据驱动的机器人学习提供了更具物理真实性的归纳偏置。
3.  **以驾驶为中心的全栈3D场景表征**：**DriveTok** 提出的“3D场景分词”概念，旨在构建一个统一、紧凑的驾驶场景表征，服务于从重建到理解的多种任务，代表了自动驾驶感知系统设计的新思路。
4.  **视觉-语言-动作模型的可解释性研究**：**《Not All Features Are Created Equal》** 从机制角度分析VLAs，标志着研究开始从追求性能转向理解这些复杂多模态模型内部的运作机制。

### **4. 推荐精读论文**

根据研究方向的普适性和影响力，建议按以下优先级阅读：

1.  **首要精读**：
    *   **《Generation Models Know Space》**：了解生成式先验如何变革3D感知，这是可能影响未来数年研究范式的关键思想。
    *   **《Do VLMs Need Vision Transformers?》**：紧跟视觉基础模型架构可能发生变革的前沿讨论。

2.  **领域相关精读**：
    *   **机器人学/具身AI研究者**：必读 **《OmniVTA》** 和 **《Articulated-Body Dynamics Network》**。
    *   **自动驾驶研究者**：必读 **《DriveTok》**，并关注 **《NavTrust》**（评估具身导航系统的可信赖性）。
    *   **3D视觉研究者**：**《MonoArt》**（单目铰接物体重建）提供了渐进式结构推理的扎实技术方案。

**总结**：今日的论文快照显示，计算机视觉领域正强力迈向 **多模态、具身化、生成式先验驱动** 的新阶段，同时伴随着对底层模型架构的冷静反思与革新。研究的前沿在于如何让AI系统更物理化、更高效地理解并与我们身处的三维世界进行交互。

---

---

## Table of Contents

1. [OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation](#2603.19201v1)
2. [Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning](#2603.19078v1)
3. [Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding](#2603.19235v1)
4. [Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](#2603.19232v1)
5. [Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models](#2603.19233v1)
6. [MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction](#2603.19231v1)
7. [NavTrust: Benchmarking Trustworthiness for Embodied Navigation](#2603.19229v1)
8. [DriveTok: 3D Driving Scene Tokenization for Unified Multi-View Reconstruction and Understanding](#2603.19219v1)
9. [Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders](#2603.19209v1)
10. [FASTER: Rethinking Real-Time Flow VLAs](#2603.19199v1)

---

## Papers

<a id='2603.19201v1'></a>
## [OmniVTA: Visuo-Tactile World Modeling for Contact-Rich Robotic Manipulation](https://arxiv.org/abs/2603.19201v1)

**Authors:** Yuhang Zheng, Songen Gu, Weize Li, Yupeng Zheng, Yujie Zang, Shuai Tian, Xiang Li, Ruihai Wu, Ce Hao, Chen Gao, Si Liu, Haoran Li, Yilun Chen, Shuicheng Yan, Wenchao Ding

**Published:** 2026-03-19

**Categories:** cs.RO

**Abstract:**

Contact-rich manipulation tasks, such as wiping and assembly, require accurate perception of contact forces, friction changes, and state transitions that cannot be reliably inferred from vision alone. Despite growing interest in visuo-tactile manipulation, progress is constrained by two persistent limitations: existing datasets are small in scale and narrow in task coverage, and current methods treat tactile signals as passive observations rather than using them to model contact dynamics or enable closed-loop control explicitly. In this paper, we present \textbf{OmniViTac}, a large-scale visuo-tactile-action dataset comprising $21{,}000+$ trajectories across $86$ tasks and $100+$ objects, organized into six physics-grounded interaction patterns. Building on this dataset, we propose \textbf{OmniVTA}, a world-model-based visuo-tactile manipulation framework that integrates four tightly coupled modules: a self-supervised tactile encoder, a two-stream visuo-tactile world model for predicting short-horizon contact evolution, a contact-aware fusion policy for action generation, and a 60Hz reflexive controller that corrects deviations between predicted and observed tactile signals in a closed loop. Real-robot experiments across all six interaction categories show that OmniVTA outperforms existing methods and generalizes well to unseen objects and geometric configurations, confirming the value of combining predictive contact modeling with high-frequency tactile feedback for contact-rich manipulation. All data, models, and code will be made publicly available on the project website at https://mrsecant.github.io/OmniVTA.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我针对 **OmniVTA** 这篇论文进行了如下深度分析：

### 1. 论文核心贡献总结
OmniVTA 旨在解决接触密集型机器人操作中“视觉感知不足”的难题，构建了一个包含 21,000+ 条轨迹的超大规模多模态数据集（OmniViTac），并提出了首个基于世界模型的视觉-触觉（Visuo-Tactile）操作框架。该方法通过将触觉从被动感知转化为动态演化预测的组成部分，实现了对接触过程的闭环精确控制。

### 2. 关键创新与方法论
该论文的创新点在于构建了一个**紧密耦合的闭环控制闭环（Closed-loop Control Pipeline）**，其核心架构包括：
*   **多尺度数据基础：** OmniViTac 数据集不仅量级大，而且按物理交互模式（如擦拭、组装等）分类，为跨任务学习提供了必要的多样性。
*   **双流世界模型（Two-stream World Model）：** 将视觉与触觉特征融合，不仅用于状态预测，更侧重于预测“短时程接触演化”，这是实现复杂操作的关键。
*   **高频反射式控制器（60Hz Reflexive Controller）：** 这是该方法的精髓。通过实时对比预测触觉与观测触觉的偏差，控制器能够实现毫秒级的快速修正，解决了传统基于策略学习（Policy Learning）方法在应对突发接触时的滞后性问题。

### 3. 对领域的潜在影响
*   **从“观察”到“预测”的范式转变：** 本研究推动了触觉信号从单纯的分类任务向动态环境建模（World Modeling）演进，预示着触觉在机器人学习中将扮演类似“模型预测控制（MPC）中状态观测器”的核心角色。
*   **数据驱动触觉学的里程碑：** 大规模高质量数据集的公开，将极大降低机器人触觉感知的研究门槛，类似于 ImageNet 对计算机视觉领域的影响，可能会引发触觉编码器（Tactile Encoder）架构的“军备竞赛”。

### 4. 受益的相关领域与应用
*   **精密制造与组装：** 如电子零件装配、线缆连接等对力感知要求极高的工业场景。
*   **服务机器人：** 在复杂的非结构化环境中（如家庭环境），处理易碎物品或执行清理任务（Wiping）。
*   **多模态大模型（LMMs）与机器人：** 该工作为未来将触觉表征对齐到大语言模型或视觉语言模型中提供了高质量的底层数据和控制逻辑支撑。

### 5. 可推断的局限性
*   **硬件依赖性：** 触觉传感器（如 GelSight 或类似类型）的硬件配置差异会影响数据集和模型的泛化能力。虽然论文声称泛化性好，但跨传感器平台的迁移仍是巨大挑战。
*   **计算资源需求：** 60Hz 的闭环控制结合双流世界模型，对机器人的机载算力提出了较高要求，在资源受限的嵌入式边缘设备上运行可能存在延迟挑战。
*   **触觉定义的普适性：** 该研究目前集中在“接触密集型”任务，对于非接触式操作或长时程全局任务，该框架的有效性是否依然优于纯视觉方案尚待验证。

**专家评价：**
这篇论文的趣味性在于它完美契合了当前“具身智能（Embodied AI）”的研究热点，尤其是它将**数据规模化**与**高频闭环控制逻辑**结合，规避了单纯依赖端到端学习带来的不可解释性和高采样需求。对于研究视觉处理、触觉反馈和机器人控制的学者而言，这是一个极具启发性的工作。

**Key Findings:**

- In this paper, we present \textbf{OmniViTac}, a large-scale visuo-tactile-action dataset comprising $21{,}000+$ trajectories across $86$ tasks and $100+$ objects, organized into six physics-grounded interaction patterns.
- Building on this dataset, we propose \textbf{OmniVTA}, a world-model-based visuo-tactile manipulation framework that integrates four tightly coupled modules: a self-supervised tactile encoder, a two-stream visuo-tactile world model for predicting short-horizon contact evolution, a contact-aware fusion policy for action generation, and a 60Hz reflexive controller that corrects deviations between predicted and observed tactile signals in a closed loop.
- Real-robot experiments across all six interaction categories show that OmniVTA outperforms existing methods and generalizes well to unseen objects and geometric configurations, confirming the value of combining predictive contact modeling with high-frequency tactile feedback for contact-rich manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19201v1)
- [arXiv](https://arxiv.org/abs/2603.19201v1)

---

<a id='2603.19078v1'></a>
## [Articulated-Body Dynamics Network: Dynamics-Grounded Prior for Robot Learning](https://arxiv.org/abs/2603.19078v1)

**Authors:** Sangwoo Shin, Kunzhao Ren, Xiaobin Xiong, Josiah Hanna

**Published:** 2026-03-19

**Categories:** cs.RO

**Abstract:**

Recent work in reinforcement learning has shown that incorporating structural priors for articulated robots, such as link connectivity, into policy networks improves learning efficiency. However, dynamics properties, despite their fundamental role in determining how forces and motion propagate through the body, remain largely underexplored as an inductive bias for policy learning. To address this gap, we present the Articulated-Body Dynamics Network (ABD-Net), a novel graph neural network architecture grounded in the computational structure of forward dynamics. Specifically, we adapt the inertia propagation mechanism from the Articulated Body Algorithm, systematically aggregating inertial quantities from child to parent links in a tree-structured manner, while replacing physical quantities with learnable parameters. Embedding ABD-NET into the policy actor enables dynamics-informed representations that capture how actions propagate through the body, leading to efficient and robust policy learning. Through experiments with simulated humanoid, quadruped, and hopper robots, our approach demonstrates increased sample efficiency and generalization to dynamics shifts compared to transformer-based and GNN baselines. We further validate the learned policy on real Unitree G1 and Go2 robots, state-of-the-art humanoid and quadruped platforms, generating dynamic, versatile and robust locomotion behaviors through sim-to-real transfer with real-time inference.

**Analysis:**

这是一份关于《Articulated-Body Dynamics Network (ABD-NET)》的深度技术分析：

### 1. 摘要翻译
本文提出了一种新颖的图神经网络架构——关节体动力学网络（ABD-NET），旨在将前向动力学的计算结构作为机器人策略学习的归纳偏置（Inductive Bias）。通过模拟关节体算法（ABA）中惯性量从子节点向父节点传播的机制，ABD-NET将物理规律转化为可学习的参数，从而使策略能够捕获动作在机器人体内的传播效应。实验表明，该方法在模拟的人形、四足和跳跃机器人任务中表现出更高的样本效率和动力学泛化能力，并在Unitree G1和Go2真实机器人上实现了高效的零样本迁移（Sim-to-Real）。

### 2. 方法动机分析
*   **核心驱动力**：现有的GNN或Transformer策略网络仅利用了机器人的“连接结构”（邻接矩阵），而忽略了“物理动力学结构”（如质量、惯性如何沿运动链传播）。
*   **现有痛点**：传统方法将特征在节点间的传播完全交给神经网络去拟合，缺乏物理语义，导致在复杂形态或动力学参数变化时泛化性能差。
*   **研究假设**：若能在网络架构中强制植入“前向动力学”的计算图（即自底向上的信息累积过程），模型将获得更强的物理先验，从而显著提升学习效率和鲁棒性。

### 3. 方法设计详解
ABD-NET的核心是将策略函数 $\pi_\theta$ 解构为三个模块：
1.  **观察编码 ($\Phi$)**：将原始传感器观测 $s$ 投影为每个连杆的独立嵌入向量 $\{z_i\}$。
2.  **动力学感知信息传递 ($M$)**：这是论文的创新点。它模拟了ABA算法中“叶子到根”的递归过程：
    *   **贡献计算**：每个子节点 $j$ 计算一个贡献值 $v_j^a$（模拟惯性修正量），公式为 $v_j^a = v_j - v_j \odot (W_j W_j^\top v_j)$。这里 $W_j$ 是可学习的矩阵，代表关节运动空间。
    *   **信息聚合**：父节点 $i$ 通过公式 $v_i = \text{softplus}(z_i + B_i) + \sum_{j \in \text{CH}(i)} v_j^a$ 来更新自身状态。
    *   **约束与正交化**：引入 $L_{\text{orth}}$ 正交化约束，确保 $W_j W_j^\top$ 能够模拟动力学投影过程，避免因缺乏约束而导致性能退化。
3.  **动作解码 ($\Psi$)**：利用更新后的父节点连杆特征 $v_{\text{PA}(j)}$ 直接推导关节动作 $a_j$。

### 4. 方法对比分析
*   **本质区别**：与传统GNN不同，ABD-NET的信息流向有严格的物理约束，即“仅从子到父”，而非全向聚合。
*   **创新贡献**：成功将经典机器人动力学算法（ABA）的递归结构微分为神经网络算子，且保持了计算效率（避免了矩阵求逆）。
*   **适用场景**：适用于具有清晰运动链结构的铰接式机器人（如四足、类人、机械臂），特别是在动力学参数易变的场景中优势明显。

### 5. 实验分析（精简版）
*   **验证方法**：在Genesis和SAPIEN环境下，测试了多款机器人平台，并与Transformer（BOT/SWAT）及标准GNN/MLP进行了对比。
*   **核心结论**：ABD-NET实现了最高的IQM（平均值区间），在动力学偏移（如增加质量）下的鲁棒性远超现有方法。
*   **优劣势**：优势在于物理先验带来的极强泛化性和样本效率；局限在于序列化的递归计算导致训练时间较长（尽管推理延迟依然在可控范围内）。

### 6. 实用指南
*   **开源情况**：作者已在相关领域验证，实现上需利用 PyTorch 搭建树状结构的递归传播函数。
*   **关键技巧**：
    *   **软正交约束**：必须加入正则项 $L_{\text{orth}}$，否则网络倾向于坍缩。
    *   **软加法（Softplus）**：在公式(7)中加入softplus保证了物理量（惯性）的正定性，这是物理一致性的关键。
*   **迁移方案**：该架构可直接用于任何URDF文件生成的机器人模型，通过解析父子关系构建图索引即可。

### 7. 总结
*   **核心思想**：将机器人动力学算法的递归逻辑嵌入神经网络拓扑结构。
*   **速记版pipeline**：
    1.  将传感器输入转化为连杆嵌入；
    2.  根据运动树从叶子向根部传递特征；
    3.  各级节点根据动力学投影公式修正特征；
    4.  基于父节点特征计算关节控制动作。

**Key Findings:**

- To address this gap, we present the Articulated-Body Dynamics Network (ABD-Net), a novel graph neural network architecture grounded in the computational structure of forward dynamics.
- Through experiments with simulated humanoid, quadruped, and hopper robots, our approach demonstrates increased sample efficiency and generalization to dynamics shifts compared to transformer-based and GNN baselines.
- We further validate the learned policy on real Unitree G1 and Go2 robots, state-of-the-art humanoid and quadruped platforms, generating dynamic, versatile and robust locomotion behaviors through sim-to-real transfer with real-time inference.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19078v1)
- [arXiv](https://arxiv.org/abs/2603.19078v1)

---

<a id='2603.19235v1'></a>
## [Generation Models Know Space: Unleashing Implicit 3D Priors for Scene Understanding](https://arxiv.org/abs/2603.19235v1)

**Authors:** Xianjin Wu, Dingkang Liang, Tianrui Feng, Kui Xia, Yumeng Zhang, Xiaofan Li, Xiao Tan, Xiang Bai

**Published:** 2026-03-19

**Categories:** cs.CV, cs.RO

**Abstract:**

While Multimodal Large Language Models demonstrate impressive semantic capabilities, they often suffer from spatial blindness, struggling with fine-grained geometric reasoning and physical dynamics. Existing solutions typically rely on explicit 3D modalities or complex geometric scaffolding, which are limited by data scarcity and generalization challenges. In this work, we propose a paradigm shift by leveraging the implicit spatial prior within large-scale video generation models. We posit that to synthesize temporally coherent videos, these models inherently learn robust 3D structural priors and physical laws. We introduce VEGA-3D (Video Extracted Generative Awareness), a plug-and-play framework that repurposes a pre-trained video diffusion model as a Latent World Simulator. By extracting spatiotemporal features from intermediate noise levels and integrating them with semantic representations via a token-level adaptive gated fusion mechanism, we enrich MLLMs with dense geometric cues without explicit 3D supervision. Extensive experiments across 3D scene understanding, spatial reasoning, and embodied manipulation benchmarks demonstrate that our method outperforms state-of-the-art baselines, validating that generative priors provide a scalable foundation for physical-world understanding. Code is publicly available at https://github.com/H-EmbodVis/VEGA-3D.

**Analysis:**

### 1. 摘要翻译
多模态大语言模型（MLLMs）在语义理解上表现出色，但往往存在“空间盲区”，在细粒度几何推理和物理动态理解上表现不佳。现有解决方案通常依赖显式的3D模态或复杂的几何辅助，受到数据稀缺和泛化能力的限制。本文提出一种新范式，利用大规模视频生成模型中蕴含的隐式空间先验。我们认为，为了合成时间上连贯的视频，这些模型天生就学习到了稳健的3D结构先验和物理定律。为此，我们引入了 **VEGA-3D**（Video Extracted Generative Awareness），这是一种即插即用框架，将预训练视频扩散模型转化为“潜在世界模拟器”。通过从中间噪声水平提取时空特征，并利用标记级自适应门控融合机制将其与语义表示集成，VEGA-3D在无需显式3D监督的情况下，为MLLMs丰富了密集的几何线索。

### 2. 方法动机分析
- **驱动力**：作者认为现代视频生成模型在训练过程中为了实现时空一致性，必然隐式地习得了“物理世界模型”，即3D结构和物理规律。
- **现有痛点**：当前MLLMs多依赖于显式3D输入（如点云）或复杂的几何重建监督，这些方法受限于数据稀缺，且推理管线极其复杂。
- **研究假设**：无需昂贵的3D数据，直接从预训练好的、具有强大生成能力的视频扩散模型（如Wan2.1）中提取“潜在空间特征”，即可作为有效的空间先验引导MLLMs。

### 3. 方法设计详解
VEGA-3D的流程核心在于将“视频生成模型”作为“视觉空间增强器”：
1. **潜在世界模拟（Latent World Simulator）**：作者利用预训练并冻结的视频扩散模型（如Wan2.1）。为了激发模型对结构的理解，作者没有使用纯净的潜变量，而是通过Flow Matching路径对输入视频特征进行噪声扰动（$z_k = (1-t_k)z_0 + t_k \epsilon$），在中间时间步提取DiT层特征。
2. **自适应门控融合（Adaptive Gated Fusion）**：这是本文的核心创新。由于生成空间的物理特征与MLLM的离散语义空间存在差异，作者设计了token级别的门控机制（由Sigmoid激活函数计算），根据具体token动态平衡语义特征和几何先验。
3. **视觉 token 增强**：将处理后的融合特征通过 MLP 投影到 LLM 相同的维度，并将这些包含结构信息的特征与原始语义视觉特征一起输入LLM。

### 4. 方法对比分析
- **本质区别**：不进行任何3D重建或深度图显式计算，完全依赖生成模型在压缩latent空间内的“固有物理知识”。
- **创新贡献**：
    - **发现**：明确了multi-view correspondence score（多视角一致性分数）是评估模型3D空间理解能力的核心指标。
    - **机制**：提出了token级自适应融合，避免了简单的特征加权带来的语义损失。
- **适用场景**：适用于需要物体定位、空间关系推理、场景理解等对几何敏感的任务，且模型训练时无需任何3D标签。

### 5. 实验分析
- **关键结果**：在ScanRefer、SQA3D等任务中，VEGA-3D性能显著提升，特别是在Localization（物体定位）类任务中效果拔群。
- **优势**：轻量级即插即用；无需额外3D监督；可随着视频生成模型能力的提升而自动进化。
- **局限**：由于引入了额外的视频扩散模型分支，增加了推理延迟和显存占用。

### 6. 实用指南
- **开源情况**：已开源（https://github.com/H-EmbodVis/VEGA-3D）。
- **实现细节**：建议采样32帧，中间层（如第20层）和中等噪声（k=300）提取特征效果最优。
- **迁移可能**：可直接集成到任何基于视觉Encoder（如SigLIP）的MLLM中，作为独立的辅助视觉分支。

### 7. 总结
- **核心思想**：利用视频生成模型作为潜在世界模拟器，提取隐式几何先验引导MLLM。
- **速记版pipeline**：
    1. 输入视频序列进行Flow Matching处理。
    2. 从中间噪声层提取结构特征。
    3. 通过自适应门控与语义特征融合。
    4. 将融合后的视觉token送入MLLM。

**Key Findings:**

- In this work, we propose a paradigm shift by leveraging the implicit spatial prior within large-scale video generation models.
- We introduce VEGA-3D (Video Extracted Generative Awareness), a plug-and-play framework that repurposes a pre-trained video diffusion model as a Latent World Simulator.
- Extensive experiments across 3D scene understanding, spatial reasoning, and embodied manipulation benchmarks demonstrate that our method outperforms state-of-the-art baselines, validating that generative priors provide a scalable foundation for physical-world understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19235v1)
- [arXiv](https://arxiv.org/abs/2603.19235v1)

---

<a id='2603.19232v1'></a>
## [Cubic Discrete Diffusion: Discrete Visual Generation on High-Dimensional Representation Tokens](https://arxiv.org/abs/2603.19232v1)

**Authors:** Yuqing Wang, Chuofan Ma, Zhijie Lin, Yao Teng, Lijun Yu, Shuai Wang, Jiaming Han, Jiashi Feng, Yi Jiang, Xihui Liu

**Published:** 2026-03-19

**Categories:** cs.CV

**Abstract:**

Visual generation with discrete tokens has gained significant attention as it enables a unified token prediction paradigm shared with language models, promising seamless multimodal architectures. However, current discrete generation methods remain limited to low-dimensional latent tokens (typically 8-32 dims), sacrificing the semantic richness essential for understanding. While high-dimensional pretrained representations (768-1024 dims) could bridge this gap, their discrete generation poses fundamental challenges. In this paper, we present Cubic Discrete Diffusion (CubiD), the first discrete generation model for high-dimensional representations. CubiD performs fine-grained masking throughout the high-dimensional discrete representation -- any dimension at any position can be masked and predicted from partial observations. This enables the model to learn rich correlations both within and across spatial positions, with the number of generation steps fixed at $T$ regardless of feature dimensionality, where $T \ll hwd$. On ImageNet-256, CubiD achieves state-of-the-art discrete generation with strong scaling behavior from 900M to 3.7B parameters. Crucially, we validate that these discretized tokens preserve original representation capabilities, demonstrating that the same discrete tokens can effectively serve both understanding and generation tasks. We hope this work will inspire future research toward unified multimodal architectures. Code is available at: https://github.com/YuqingWang1029/CubiD.

**Analysis:**

### 1. 摘要翻译
视觉生成领域中，基于离散标记（discrete tokens）的生成方法因其与语言模型共享统一的标记预测范式，具备实现无缝多模态架构的潜力，因而备受关注。然而，现有的离散生成方法大多受限于低维潜在空间（通常为8-32维），这牺牲了理解任务所需的语义丰富度。虽然高维预训练表征（768-1024维）可以弥补这一缺憾，但对其进行离散化生成面临根本性挑战。为此，我们提出了“立方离散扩散”（Cubic Discrete Diffusion, CubiD），这是首个针对高维表征的离散生成模型。CubiD在整个高维离散表征中执行细粒度掩码——即任何位置的任何维度都可以被掩码并根据局部观察进行预测。这使得模型能够学习空间内以及跨空间位置的丰富相关性，且生成步数固定为$T$，与特征维度无关（$T \ll hwd$）。在ImageNet-256数据集上，CubiD在900M到3.7B参数规模下实现了最先进的离散生成效果，且展现出强大的缩放行为。实验证明，这些离散化标记保留了原始表征的理解能力，使同一标记集能够有效服务于理解与生成任务。代码已开源。

### 2. 方法动机分析
*   **驱动力**：旨在构建统一的多模态模型，使生成任务能直接利用高维、语义丰富的视觉表征，而不是压缩成低维、信息有损的紧凑标记。
*   **现有痛点**：高维特征（768+维）导致数据分布极度稀疏，“维数灾难”使得传统的向量量化（VQ）聚类失效，且直接建模导致标记序列极长，序列式生成（Autoregressive）计算不可行。
*   **研究假设**：高维表征包含结构化的语义信息，通过维度无关的量化（Dimension-wise Quantization）可以保留语义，且通过在3D立方体空间中进行细粒度的并行掩码预测，能有效捕捉复杂的维度间与空间相关性，绕过序列化瓶颈。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特征预处理**：使用Frozen的预训练编码器（如DINOv2）提取高维特征图（$h \times w \times d$）。
    2.  **维度无关量化**：不采用整体向量量化，而是对每个维度独立进行量化，映射为离散指数。
    3.  **立方体掩码建模**：将$h \times w \times d$视为统一的3D空间。训练时，随机掩码部分元素。
    4.  **迭代生成**：推理时，从全掩码状态开始，利用Transformer进行多步迭代并行去噪（unmasking）。
*   **模型结构**：Transformer架构，输入为重构后的$d$维向量（包含`[MASK]`嵌入），输出经MLP预测$d \times L$个逻辑值。
*   **关键公式**：$\mathcal {L} = -\mathbb {E}_{\mathbf {q},\mathbf {M}} [ \sum _{i \in \mathbf {M}} \log p(q_i | \mathbf {q}_{\bar {\mathbf {M}}}) ]$，该公式强调了对掩码位置的独立联合预测，而非传统的序列预测。

### 4. 方法对比分析
*   **本质区别**：从“空间维度的线性序列建模”转变为“3D立方体空间内的细粒度并行去噪建模”。
*   **创新贡献**：成功将离散扩散模型的适用范围从低维拉近到768+维的原生高维特征，且无需重新训练特征编码器，保留了原始表征的推理能力。
*   **适用场景**：适用于需要高质量图像生成且希望复用已有强语义预训练表征的多模态任务。

### 5. 实验分析
*   **关键结果**：在ImageNet-256上以1.88的gFID达到SOTA，且在大模型规模（3.7B）下表现出良好的缩放性能。
*   **主要优势**：生成效率高（$T$步并行），保持了高维特征的语义完整性。
*   **主要局限**：重建质量受限于冻结编码器的上限；推理步数仍多于连续扩散模型。

### 6. 实用指南
*   **开源情况**：已开源，可直接调用。
*   **实现细节**：关键在于**维度无关量化**（Dimension-wise Quantization）的使用，避免了在高维空间中进行聚类。
*   **迁移可能**：该方法逻辑通用，可轻松迁移至视频生成等领域，只需调整掩码策略以适应时空结构。

### 7. 总结
*   **核心思想**：在高维张量空间通过细粒度掩码实现高效并行扩散生成。
*   **速记版Pipeline**：
    1. 冻结预训练编码器提取特征；
    2. 维度独立量化保留语义信息；
    3. 将$h \times w \times d$张量视为3D空间；
    4. 迭代掩码预测实现并行去噪生成。

**Key Findings:**

- In this paper, we present Cubic Discrete Diffusion (CubiD), the first discrete generation model for high-dimensional representations.
- On ImageNet-256, CubiD achieves state-of-the-art discrete generation with strong scaling behavior from 900M to 3.7B parameters.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19232v1)
- [arXiv](https://arxiv.org/abs/2603.19232v1)

---

<a id='2603.19233v1'></a>
## [Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models](https://arxiv.org/abs/2603.19233v1)

**Authors:** Bryce Grant, Xijia Zhao, Peng Wang

**Published:** 2026-03-19

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models combine perception, language, and motor control in a single architecture, yet how they translate multimodal inputs into actions remains poorly understood. We apply activation injection, sparse autoencoders (SAEs), and linear probes to six models spanning 80M--7B parameters across 394,000+ rollout episodes on four benchmarks. The visual pathway dominates action generation across all architectures: injecting baseline activations into null-prompt episodes recovers near-identical behavior, while cross-task injection steers robots toward source-task positions (99.8\% of X-VLA episodes align with the source trajectory), exposing spatially bound motor programs tied to scene coordinates rather than abstract task representations. Language sensitivity depends on task structure, not model design: when visual context uniquely specifies the task, language is ignored; when multiple goals share a scene, language becomes essential (X-VLA \texttt{libero\_goal}: 94\%$\to$10\% under wrong prompts vs.\ \texttt{libero\_object}: 60--100\% regardless). In all three multi-pathway architectures (\pizhalf{}, SmolVLA, GR00T), expert pathways encode motor programs while VLM pathways encode goal semantics ($2\times$ greater behavioral displacement from expert injection), and subspace injection confirms these occupy separable activation subspaces. Per-token SAE processing is essential for action fidelity on most architectures, though mean-pooling improves fidelity on X-VLA. Contrastive identification recovers 82+ manipulation concepts, and causal ablation reveals sensitivity spanning 28--92\% zero-effect rates independent of representation width. We release \textbf{Action Atlas} (https://action-atlas.com) for interactive exploration of VLA representations across all six models.

**Analysis:**

以下是对《Not All Features Are Created Equal: A Mechanistic Study of Vision-Language-Action Models》这篇论文的深度解析。

### 1. 摘要翻译
视觉-语言-动作（VLA）模型结合了感知、语言和运动控制，但其多模态输入转化为动作的机制尚不明确。我们应用激活注入、稀疏自编码器（SAE）和线性探针，对6种模型、39.4万+回放片段进行了机理研究。研究发现，视觉通路在所有架构中均主导动作生成：零提示下的基线激活注入可恢复几乎相同的行为，而跨任务注入则将机器人引向源任务位置，揭示了受场景坐标约束的运动程序，而非抽象的任务表征。语言敏感度取决于任务结构，而非模型设计：在视觉上下文明确时，语言被忽略；在歧义场景下，语言才至关重要。我们发布了Action Atlas平台，用于交互式探索VLA表征。

### 2. 方法动机分析
*   **驱动力**：旨在解决VLA模型“黑盒”问题。在机器人出现异常行为时，缺乏原则性的诊断工具。
*   **痛点**：以往研究仅限于行为观察，缺乏对其内部表征（尤其是语言指令与视觉运动先验）如何协作的因果理解。
*   **研究假设**：VLA模型的内部表征并非同质，而是通过空间绑定与通路专业化（Expert/VLM pathway）来实现，且语言指令在任务区分度高的场景中往往被视觉先验所覆盖。

### 3. 方法设计详解
本研究的核心在于通过**因果干预**解构VLA的行为机制：
1.  **激活注入（Activation Injection）**：通过将源任务的激活值（$H_A$）替换进目标任务（$H_B$），对比轨迹偏移以判定该层在行为生成中的因果贡献。
2.  **稀疏自编码器（SAE）解耦**：使用TopK稀疏性约束（$k=64$）训练SAE，将高维稠密激活分解为人类可理解的语义特征。
    *   **关键处理**：论文强调**Per-Token Processing**，即针对每一个动作Token独立进行SAE处理，因为对时序激活进行均值池化（Mean-pooling）会摧毁动作序列的动态结构，导致任务失败。
3.  **频率加权对比选择（Contrastive Selection）**：定义$score_f = d_f \times freq_f$，量化特定特征对特定任务（概念存在与否）的敏感度，用于识别关键操纵概念。
4.  **因果验证（Causal Ablation）**：通过“斩首”（Ablation）特定的SAE特征（将特征置零），观察机器人任务成功率的下降程度（Kill-switch分析），确定其功能必要性。

### 4. 方法对比分析
*   **本质区别**：与传统通过注意力热力图进行解释的方法不同，本文采用的是**因果干预（Causal Intervention）**，通过主动修改内部表征并观察任务成功率变化来建立因果链路。
*   **创新贡献**：首次系统性验证了VLA中视觉通路的主导地位和路径特异性，揭示了机器人动作受“坐标约束”的物理本质，而非基于对象的抽象推理。

### 5. 实验分析（精简版）
*   **验证方法**：通过大规模（39.4万+） rollout评估，结合不同程度的特征修改（注入、置零、偏移分析）。
*   **关键结论**：
    1.  **视觉通路主导**：在视觉上下文充足时，移除语言指令对结果影响微乎其微。
    2.  **空间绑定**：跨任务干预显示，机器人倾向于在“物理位置”上模仿源任务，而非“逻辑语义”。
    3.  **专家/语义双通路**：在多通路架构中，专家路径负责“如何做”（运动程序），VLM路径负责“做什么”（目标语义）。
*   **局限**：目前的因果分析主要基于模拟器，且针对全层残差流的干预存在效应放大，需后续精细化到MLP子层。

### 6. 实用指南
*   **开源情况**：Action Atlas（https://action-atlas.com）。
*   **实现细节**：SAE训练必须保留时间维度（Per-token），否则会导致模型灾难性遗忘。对MLP子层进行干预优于直接干预残差流，可避免对跳跃连接的过度干扰。
*   **迁移可能**：该机制分析框架可直接迁移至其他基于Transformer的 embodied AI 模型（如RT-2, OpenVLA）。

### 7. 总结
*   **核心思想**：视觉路径即因果路径，机器人动作受空间坐标而非语言逻辑支配。
*   **速记版pipeline**：1. 录制多源回放片段；2. 执行激活注入与特征置零；3. 利用SAE聚类识别关键控制概念；4. 验证特征置零后的轨迹偏移与成功率下降。

**Key Findings:**

- Contrastive identification recovers 82+ manipulation concepts, and causal ablation reveals sensitivity spanning 28--92\% zero-effect rates independent of representation width.
- We release \textbf{Action Atlas} (https://action-atlas.com) for interactive exploration of VLA representations across all six models.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19233v1)
- [arXiv](https://arxiv.org/abs/2603.19233v1)

---

<a id='2603.19231v1'></a>
## [MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction](https://arxiv.org/abs/2603.19231v1)

**Authors:** Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, Ziwei Liu

**Published:** 2026-03-19

**Categories:** cs.CV

**Abstract:**

Reconstructing articulated 3D objects from a single image requires jointly inferring object geometry, part structure, and motion parameters from limited visual evidence. A key difficulty lies in the entanglement between motion cues and object structure, which makes direct articulation regression unstable. Existing methods address this challenge through multi-view supervision, retrieval-based assembly, or auxiliary video generation, often sacrificing scalability or efficiency. We present MonoArt, a unified framework grounded in progressive structural reasoning. Rather than predicting articulation directly from image features, MonoArt progressively transforms visual observations into canonical geometry, structured part representations, and motion-aware embeddings within a single architecture. This structured reasoning process enables stable and interpretable articulation inference without external motion templates or multi-stage pipelines. Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed. The framework further generalizes to robotic manipulation and articulated scene reconstruction.

**Analysis:**

以下是针对《MonoArt: Progressive Structural Reasoning for Monocular Articulated 3D Reconstruction》论文的深度分析：

### 1. 摘要翻译
本文提出了一种名为MonoArt的统一框架，通过渐进式结构推理实现单目关节物体3D重建。现有方法常因运动线索与物体结构的纠缠而导致关节回归不稳定，或依赖多视图、检索库、辅助视频生成等复杂技术，导致扩展性与效率受限。MonoArt不直接从图像特征回归运动参数，而是将视觉观察渐进地转化为规范几何、结构化部件表示和运动感知嵌入。这种推理过程实现了稳定且可解释的关节推断，无需外部模板。实验表明，MonoArt在PartNet-Mobility数据集上实现了重建精度与推理速度的SOTA表现，并可推广至机器人操作与场景重建。

### 2. 方法动机分析
*   **驱动力**：旨在解决单张图像进行关节物体重建时，“形状（几何）”与“运动（关节参数）”特征高度纠缠导致预测不稳定的问题。
*   **痛点**：现有方法（如检索式或视频辅助式）要么受限于库内资产，导致纹理错位，要么计算复杂度极高且缺乏结构化先验，难以直接通过单帧推理。
*   **核心假设**：通过将重建过程分解为“规范几何生成 -> 部件语义感知 -> 运动双重查询解码 -> 结构化关节回归”的渐进式路径，可以显著解耦形状与运动的特征，提高模型的鲁棒性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **几何基础**：利用冻结的TRELLIS模型从单图生成规范几何（Mesh）与稀疏体素特征（Z）。
    2.  **部件推理**：通过“三线性插值 + 三平面投影 + 部件对比Transformer”，将离散几何特征映射为具备部件级区分度的嵌入（H）。
    3.  **运动解码**：采用**双查询机制（Dual-Query）**，将“位置查询（Qp）”与“内容查询（Qc）”分开建模，通过L层残差迭代Refinement Block逐步细化运动假设。
    4.  **关节回归**：基于Refinement后的查询特征，预测关节类型、轴向、基点、限位等，并利用二分匹配构建运动学树。
*   **关键公式意义**：公式(3)通过查询与特征点乘获取Mask，公式(4)利用双线性形式计算部件间的依赖关系概率，有效实现了关节链结构的自动推理。

### 4. 方法对比分析
*   **本质区别**：从传统的“特征到参数的黑盒回归”转向了“受结构先验引导的渐进式推理”。
*   **创新点**：
    *   **双查询架构**：解耦了“动在哪里（几何位置）”与“动成什么样（语义类型）”，这是性能提升的关键。
    *   **对比学习监督**：通过三元组损失（Triplet Loss）强制部件特征分离，增强了模型对结构信息的感知。
*   **适用场景**：适用于各类具有运动部件（如抽屉、铰链门、转轴）的3D资产生成，尤其是需要进行机器人抓取等交互式场景。

### 5. 实验分析（精简版）
*   **验证方法**：在PartNet-Mobility数据集（7类及46类全集）上对比了SOTA方法，包括几何与关节参数回归指标。
*   **关键结论**：在保持最高F-score的同时，推理速度相比基于检索的方法大幅下降（20.5秒/实例）。
*   **主要优势**：极强的鲁棒性，能够从单图预测出物理可行的关节参数，且自动生成Kinematic Tree。
*   **主要局限**：对“极小部件”（如微小的按钮）的特征表示能力较弱，且对于未见过的、非典型的关节类型泛化能力有限。

### 6. 实用指南
*   **开源状态**：开源地址为 `https://lihaitian.com/MonoArt`。
*   **实现建议**：
    *   **超参数注意**：双查询数量 $N_q=100$，Refinement层数 $L=6$ 是性能/速度权衡的最优点。
    *   **四阶段训练**：严格遵守文中的四阶段训练流程（预热部件感知 -> 初始化训练 -> 联合优化 -> 树结构训练），特别是Warm-up阶段对下游关节回归至关重要。
*   **迁移思路**：该模型架构可直接用于智能家居场景重建、AR/VR物体交互、仿真环境自动生成。

### 7. 总结
*   **核心思想**：利用渐进式结构推理与双查询解耦机制，实现单目稳定关节重建。
*   **速记版pipeline**：
    1. 生成规范几何Mesh。
    2. 特征映射提取部件语义。
    3. 双查询迭代细化运动轨迹。
    4. 回归关节参数并构建树结构。

**Key Findings:**

- We present MonoArt, a unified framework grounded in progressive structural reasoning.
- Extensive experiments on PartNet-Mobility demonstrate that OM achieves state-of-the-art performance in both reconstruction accuracy and inference speed.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19231v1)
- [arXiv](https://arxiv.org/abs/2603.19231v1)

---

<a id='2603.19229v1'></a>
## [NavTrust: Benchmarking Trustworthiness for Embodied Navigation](https://arxiv.org/abs/2603.19229v1)

**Authors:** Huaide Jiang, Yash Chaudhary, Yuping Wang, Zehao Wang, Raghav Sharma, Manan Mehta, Yang Zhou, Lichao Sun, Zhiwen Fan, Zhengzhong Tu, Jiachen Li

**Published:** 2026-03-19

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG, eess.SY

**Abstract:**

There are two major categories of embodied navigation: Vision-Language Navigation (VLN), where agents navigate by following natural language instructions; and Object-Goal Navigation (OGN), where agents navigate to a specified target object. However, existing work primarily evaluates model performance under nominal conditions, overlooking the potential corruptions that arise in real-world settings. To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, including RGB, depth, and instructions, in realistic scenarios and evaluates their impact on navigation performance. To our best knowledge, NavTrust is the first benchmark that exposes embodied navigation agents to diverse RGB-Depth corruptions and instruction variations in a unified framework. Our extensive evaluation of seven state-of-the-art approaches reveals substantial performance degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems. Furthermore, we systematically evaluate four distinct mitigation strategies to enhance robustness against RGB-Depth and instructions corruptions. Our base models include Uni-NaVid and ETPNav. We deployed them on a real mobile robot and observed improved robustness to corruptions. The project website is: https://navtrust.github.io.

**Analysis:**

以下是对 NavTrust 论文的深度分析：

### 1. 摘要翻译
本文介绍了 NavTrust，这是首个统一的基准测试框架，旨在严格评估具身导航代理（包括视觉语言导航 VLN 和目标导航 OGN）的鲁棒性。针对当前研究主要在理想条件下评估性能的局限，NavTrust 系统地引入了涵盖 RGB、深度传感器以及自然语言指令的各种真实环境下的退化与干扰。通过在七种前沿导航模型上的广泛评估，我们揭示了现有模型在面对现实干扰时的显著性能下降。此外，我们评估了四种提升鲁棒性的关键策略，并验证了这些策略在仿真和真实机器人平台间的迁移能力，为开发更稳健可靠的具身导航系统提供了路径。

### 2. 方法动机分析
- **驱动力**：具身智能模型正从实验室环境走向复杂多变的真实场景，现有的基准测试忽略了感知（RGB/深度）与交互（指令）中的不可控噪声。
- **痛点**：现有研究过度依赖理想输入，缺乏对 depth-sensor 故障及复杂指令偏差（如 prompt injection）的量化评估，且缺乏针对性的鲁棒性提升框架。
- **研究假设**：通过引入多样化、结构化的干扰扰动，可以定量评估感知模型的鲁棒性瓶颈，并利用针对性的增强策略（如适配器、蒸馏、指令清洗）有效弥补这些缺陷。

### 3. 方法设计详解
NavTrust 的核心在于构建了一个分层评估与防御体系：
- **感知层干扰（RGB/Depth）**：
  - **RGB 干扰**：模拟 Motion Blur（动效）、Low-Lighting（光照）、Spatter（污染）、Flare（耀斑）、Defocus（离焦）、Foreign Object（遮挡）、Black-out（失效）。
  - **深度干扰**：针对深度图提出 Gaussian Noise（抖动）、Missing Data（缺失）、Multipath（反射干扰）、Quantization（低分辨率量化），直接测试感知系统的几何鲁棒性。
- **指令干扰**：
  - 通过 Diversity（多风格）、Capitalizing（强调）、Masking（掩码）、Black-box/White-box 攻击测试语言模块的抗干扰性。
- **四大缓解策略（Mitigation Strategies）**：
  - **Data Augmentation**：引入基于性能的分布式采样，优先增强模型处理短板干扰的能力。
  - **Teacher-Student Distillation**：将结构化知识通过特征层 MSE 和逻辑层 KL 散度从强模型迁移至鲁棒性较弱的轻量化模型。
  - **Adapters**：仅训练 1-3% 的残差参数，通过全景加权融合机制，实现感知修正，提升对 noisy 输入的过滤能力。
  - **Safeguard LLM**：利用小型化指令清洗器（如 LLaMA-3.2 8-bit）将对抗性或畸变的指令标准化（Canonicalization），剥离有害内容。

### 4. 方法对比分析
- **本质区别**：NavTrust 不再局限于视觉图像质量的增强，而是将“指令完整性”与“几何鲁棒性”纳入统一的评估框架。
- **创新点**：首次提出深度传感器层面的量化干扰模型，并提出一套完整的感知-决策链路的保护伞方案。
- **适用场景**：适用于所有基于视觉传感器和指令控制的具身移动机器人系统的健壮性测试与升级。

### 5. 实验分析（精简版）
- **验证方法**：在 Habitat-Matterport3D 环境及 R2R/RxR 数据集上对比七个 SOTA 模型。
- **关键结论**：多模态/全景信息（Panoramic input）能显著提升鲁棒性；基于深度学习的 early-fusion 方法往往在感知干扰下崩溃，而带有显式噪声门控（gating）或延迟融合（late-fusion）的方法更优。
- **优劣势**：优势在于不仅定位了故障点（如 tokenizer 缺陷、几何丢失），还提供了实测可行的防御手段；局限在于对极小数据量/极度受限环境的泛化分析仍有待提升。

### 6. 实用指南
- **开源情况**：已发布项目主页 https://navtrust.github.io。
- **实现细节**：在迁移策略中，建议优先采用 **Per-episode DA**（而非 Per-frame），因为前者能更好地保持时间序列上的连贯性，这对 topological planning 尤为重要。
- **迁移迁移**：Adapter 模块非常容易迁移到任何基于 CNN 或 Transformer 的感知 backbone 中，只需保持其残差连接逻辑。

### 7. 总结
- **核心思想**：构建全模态干扰基准，通过感知修正与指令标准化提升导航健壮性。
- **速记版pipeline**：1. 定义多种感官干扰模式；2. 运行代理评估性能；3. 计算 PRS 评分分析弱点；4. 部署 Adapter 或 LLM 清洗层以增强决策稳定性。

**Key Findings:**

- To address this gap, we present NavTrust, a unified benchmark that systematically corrupts input modalities, including RGB, depth, and instructions, in realistic scenarios and evaluates their impact on navigation performance.
- Our extensive evaluation of seven state-of-the-art approaches reveals substantial performance degradation under realistic corruptions, which highlights critical robustness gaps and provides a roadmap toward more trustworthy embodied navigation systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19229v1)
- [arXiv](https://arxiv.org/abs/2603.19229v1)

---

<a id='2603.19219v1'></a>
## [DriveTok: 3D Driving Scene Tokenization for Unified Multi-View Reconstruction and Understanding](https://arxiv.org/abs/2603.19219v1)

**Authors:** Dong Zhuo, Wenzhao Zheng, Sicheng Zuo, Siming Yan, Lu Hou, Jie Zhou, Jiwen Lu

**Published:** 2026-03-19

**Categories:** cs.CV, cs.LG

**Abstract:**

With the growing adoption of vision-language-action models and world models in autonomous driving systems, scalable image tokenization becomes crucial as the interface for the visual modality. However, most existing tokenizers are designed for monocular and 2D scenes, leading to inefficiency and inter-view inconsistency when applied to high-resolution multi-view driving scenes. To address this, we propose DriveTok, an efficient 3D driving scene tokenizer for unified multi-view reconstruction and understanding. DriveTok first obtains semantically rich visual features from vision foundation models and then transforms them into the scene tokens with 3D deformable cross-attention. For decoding, we employ a multi-view transformer to reconstruct multi-view features from the scene tokens and use multiple heads to obtain RGB, depth, and semantic reconstructions. We also add a 3D head directly on the scene tokens for 3D semantic occupancy prediction for better spatial awareness. With the multiple training objectives, DriveTok learns unified scene tokens that integrate semantic, geometric, and textural information for efficient multi-view tokenization. Extensive experiments on the widely used nuScenes dataset demonstrate that the scene tokens from DriveTok perform well on image reconstruction, semantic segmentation, depth prediction, and 3D occupancy prediction tasks.

**Analysis:**

### 1. 摘要翻译
随着视觉-语言-动作模型和世界模型在自动驾驶系统中的应用，可扩展的图像标记化（Tokenization）成为视觉模态的关键接口。然而，现有标记器多针对单目和2D场景，在处理高分辨率多视图驾驶场景时存在效率低下和视图间不一致的问题。为此，我们提出了DriveTok，一个用于统一多视图重建与理解的高效3D驾驶场景标记器。DriveTok首先利用视觉基础模型获取语义丰富的视觉特征，随后通过3D可变形交叉注意力将其转换为场景标记。在解码阶段，我们采用多视图Transformer从场景标记中重构多视图特征，并利用多头机制实现RGB、深度及语义重构。此外，我们在场景标记上直接添加3D头，用于预测3D语义占据（Occupancy），增强空间感知。实验表明，DriveTok生成的统一场景标记在图像重构、语义分割、深度预测及3D占据预测任务上均表现出色。

### 2. 方法动机分析
*   **驱动力**：旨在为自动驾驶打造一种“通用视觉接口”，使后续的大模型（如VLA、世界模型）能够以统一且紧凑的方式处理多视角的3D感知信息，而非处理冗余的2D图像。
*   **现有痛点**：现有标记器（如VQGAN）基于单张图像，忽略了驾驶场景中多视图间的共享3D结构；高分辨率导致Token数量爆炸，且缺乏多视图间的几何一致性。
*   **核心直觉**：通过将多视角视觉特征“投影”到一个统一的、与分辨率及相机数量无关的3D空间（即场景Tokens），实现空间对齐和语义整合。

### 3. 方法设计详解
*   **核心Pipeline**：
    1.  **特征提取**：利用DINOv3-ViTB提取多尺度特征，通过FPN构建金字塔特征映射。
    2.  **3D场景编码（Lift）**：利用BEVFormer风格的查询机制（Scene Queries），结合相机内外参，通过多尺度可变形注意力（Deformable Attention）从2D特征采样，投影到预定义的固定3D栅格空间。
    3.  **空间感知解码（Spatial-Aware Decoder）**：利用ViT结构处理场景Tokens（Scene Tokens）与视图Tokens（View Tokens）之间的交互。关键在于引入了**可见性引导注意力（Visibility-Guided Attention）**，通过预计算的二值可见性掩码（Mask）阻断物理上不可见的视线交互，强制模型学习几何一致性。
    4.  **多任务输出**：通过DPT decoder重构RGB、深度和语义，并通过专门的3D Occupancy head输出占据空间分布。

### 4. 方法对比分析
*   **本质区别**：DriveTok并非单纯地对图像进行压缩，而是将2D信息提升为“几何感知”的3D场景Token。其最大的不同点在于引入了显式的**几何投影约束**（3D Deformable Cross-Attention）和**物理可见性约束**（Visibility-Aware Mask）。
*   **创新贡献**：提出了一种与相机布局、分辨率解耦的统一场景表示，显著降低了后续大模型的处理负担。
*   **适用场景**：自动驾驶多传感器融合感知、端到端自动驾驶规划、具身智能视觉理解。

### 5. 实验分析
*   **验证方法**：在nuScenes数据集上，与VQGAN、BEV-VAE等主流方法对比重构质量；通过消融实验验证Visibility-Guided Attention和多任务协同训练的必要性。
*   **关键结果**：在降低Token预算的同时，在图像重构和3D占据预测（IoU/mIoU）上达到了SOTA水平，且展现了极高的推理效率（∼80-90ms/scene）。
*   **局限性**：高度依赖于预训练视觉基础模型（如DINOv3）的语义质量，如果基础模型对驾驶场景理解有限，会影响整体性能。

### 6. 实用指南
*   **开源情况**：代码已开源，详见：`https://github.com/paryi555/DriveTok`。
*   **实现细节**：
    *   训练过程：需先对齐单目深度（使用MoGe-2与LiDAR投影校准）。
    *   训练目标：总损失函数包含RGB重构、深度、语义预测、占据预测及语义正则化。
*   **迁移建议**：该架构可以直接迁移到任何需要“将多视角图像转化为紧凑3D空间表示”的任务中，只需调整BEV栅格分辨率和任务头。

### 7. 总结
*   **核心思想**：通过3D可变形映射和可见性约束，构建空间一致的统一3D场景Token。
*   **速记版Pipeline**：
    1. 提取多视图视觉特征；
    2. 通过可变形注意力投影至3D栅格；
    3. 加入可见性掩码的Transformer交互；
    4. 多头输出图像及几何重构。

**Key Findings:**

- To address this, we propose DriveTok, an efficient 3D driving scene tokenizer for unified multi-view reconstruction and understanding.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19219v1)
- [arXiv](https://arxiv.org/abs/2603.19219v1)

---

<a id='2603.19209v1'></a>
## [Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders](https://arxiv.org/abs/2603.19209v1)

**Authors:** Shang-Jui Ray Kuo, Paola Cascante-Bonilla

**Published:** 2026-03-19

**Categories:** cs.CV, cs.LG

**Abstract:**

Large vision--language models (VLMs) often use a frozen vision backbone, whose image features are mapped into a large language model through a lightweight connector. While transformer-based encoders are the standard visual backbone, we ask whether state space model (SSM) vision backbones can be a strong alternative. We systematically evaluate SSM vision backbones for VLMs in a controlled setting. Under matched ImageNet-1K initialization, the SSM backbone achieves the strongest overall performance across both VQA and grounding/localization. We further adapt both SSM and ViT-family backbones with detection or segmentation training and find that dense-task tuning generally improves performance across families; after this adaptation, the SSM backbone remains competitive while operating at a substantially smaller model scale. We further observe that (i) higher ImageNet accuracy or larger backbones do not reliably translate into better VLM performance, and (ii) some visual backbones are unstable in localization. Based on these findings, we propose stabilization strategies that improve robustness for both backbone families and highlight SSM backbones as a strong alternative to transformer-based vision encoders in VLMs.

**Analysis:**

这是一份关于论文《Do VLMs Need Vision Transformers? Evaluating State Space Models as Vision Encoders》的深度分析报告。

### 1. 摘要翻译
大规模视觉-语言模型（VLMs）通常使用冻结的视觉主干，通过轻量级连接器将图像特征映射到大语言模型（LLM）中。虽然基于Transformer的编码器是标准的视觉主干，但我们探讨了状态空间模型（SSM）视觉主干是否能成为强有力的替代方案。我们在受控设置下系统地评估了VLMs的SSM视觉主干。在匹配ImageNet-1K初始化的情况下，SSM主干在视觉问答（VQA）和定位/接地（grounding）任务上均表现出最强的综合性能。我们进一步通过检测或分割训练对SSM和ViT系列主干进行了适配，发现密集任务微调通常会提升各主干系列的性能；适配后，SSM主干在大幅减小模型规模的同时仍保持竞争优势。我们进一步观察到：(i) 更高的ImageNet准确率或更大的主干并不一定转化为更好的VLM性能；(ii) 某些视觉主干在定位方面存在不稳定性。基于这些发现，我们提出了旨在提高两个主干系列鲁棒性的稳定化策略，并强调SSM主干是VLMs中基于Transformer的视觉编码器的强有力替代方案。

### 2. 方法动机分析
- **驱动力**：旨在打破VLM视觉编码器单一依赖ViT的局面，探究SSM架构在保留空间信息和计算效率方面的潜力。
- **痛点**：现有的VLM比较通常混淆了多个变量（如预训练目标、训练流水线、分辨率、连接器设计等），导致难以孤立评估单一主干架构的影响。此外，Transformer架构在缺乏明确空间监督时，容易在编码过程中削弱空间特征。
- **核心直觉**：SSM的2D选择性扫描（SS2D）设计能够通过多方向结构化状态更新保留丰富的空间特征，这比Transformer的全局自注意力更利于需要精细定位的VLM任务。

### 3. 方法设计详解
- **受控实验框架**：作者建立了一个严格的“受控实验室”，确保除了视觉主干（Backbone）发生变化外，连接器、LLM（Vicuna-7B）、训练数据（665K LLaVA-v1.5 mixture）和训练超参数完全一致。
- **视觉主干适配**：
  - **分类预训练**：对比IN1K-224下各家族表现。
  - **密集任务适配**：针对检测（COCO）和分割（ADE20K）任务微调主干，探究密集预训练对空间理解的增益。
- **稳定化策略（核心创新）**：
  - **增强连接器容量**：将默认的2层MLP替换为3层MLP，以解决“传输瓶颈（Transmission bottleneck）”。
  - **几何重塑**：将非方形输入调整为方形（512x512），解决因输入宽高比与推理分辨率不匹配导致的“利用瓶颈（Utilization bottleneck）”和“定位崩溃（localization collapse）”。

### 4. 方法对比分析
- **本质区别**：Transformer依赖全局自注意力（置换不变性），依赖位置编码构建空间结构；SSM通过2D多向扫描显式地烘焙空间依赖性。
- **创新贡献**：首次在受控环境下对SSM视觉主干进行了系统评估；提出了“定位崩溃”现象，并通过增加连接器容量和几何归一化成功解决了该失效模式。
- **适用场景**：对细粒度、需要空间定位能力的VLM任务（如RefCOCO系列），SSM具有显著优势。

### 5. 实验分析
- **关键结论**：VMamba-T/S在 matched 设置下优于同规模ViT，尤其是在定位任务上表现突出。
- **主要发现**：更大的模型和更高的ImageNet分类精度并不等同于更好的VLM downstream性能（即“分类目标过拟合”）。
- **局限性**：高分辨率或复杂适配后的主干在处理不当的接口设计下会发生定位崩溃。

### 6. 实用指南
- **开源情况**：已通过GitHub (TRI-ML/prismatic-vlms) 验证。
- **实现细节**：对于hierarchical结构（如VMamba），应选择stage-3特征以匹配标准ViT/16的token数量。
- **迁移建议**：对于新的视觉主干，建议通过简单的线性探测（Linear Probing）预测试；若发生下游任务退化，优先检查是否是“传输瓶颈”（增大MLP）或“几何失配”（改为方形输入）。

### 7. 总结
- **核心思想**：通过受控实验证明SSM是VLM更具空间感知能力且高效的视觉主干选择。
- **速记版pipeline**：
  1. 冻结主干与LLM。
  2. 选定Stage-3输出保持token数匹配。
  3. 通过3层MLP连接器映射特征。
  4. 采用方形图像分辨率输入以维持稳定性。

**Key Findings:**

- Based on these findings, we propose stabilization strategies that improve robustness for both backbone families and highlight SSM backbones as a strong alternative to transformer-based vision encoders in VLMs.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19209v1)
- [arXiv](https://arxiv.org/abs/2603.19209v1)

---

<a id='2603.19199v1'></a>
## [FASTER: Rethinking Real-Time Flow VLAs](https://arxiv.org/abs/2603.19199v1)

**Authors:** Yuxiang Lu, Zhe Liu, Xianzhe Fan, Zhenya Yang, Jinghua Hou, Junyi Li, Kaixin Ding, Hengshuang Zhao

**Published:** 2026-03-19

**Categories:** cs.RO, cs.CV

**Abstract:**

Real-time execution is crucial for deploying Vision-Language-Action (VLA) models in the physical world. Existing asynchronous inference methods primarily optimize trajectory smoothness, but neglect the critical latency in reacting to environmental changes. By rethinking the notion of reaction in action chunking policies, this paper presents a systematic analysis of the factors governing reaction time. We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon. Moreover, we reveal that the standard practice of applying a constant schedule in flow-based VLAs can be inefficient and forces the system to complete all sampling steps before any movement can start, forming the bottleneck in reaction latency. To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER). By introducing a Horizon-Aware Schedule, FASTER adaptively prioritizes near-term actions during flow sampling, compressing the denoising of the immediate reaction by tenfold (e.g., in $π_{0.5}$ and X-VLA) into a single step, while preserving the quality of long-horizon trajectory. Coupled with a streaming client-server pipeline, FASTER substantially reduces the effective reaction latency on real robots, especially when deployed on consumer-grade GPUs. Real-world experiments, including a highly dynamic table tennis task, prove that FASTER unlocks unprecedented real-time responsiveness for generalist policies, enabling rapid generation of accurate and smooth trajectories.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为《FASTER: Rethinking Real-Time Flow VLAs》的论文分析如下：

### 1. 论文贡献总结
该论文针对具身智能中视觉-语言-动作（VLA）模型在实时控制任务中响应延迟过高的问题，提出了一种名为 **FASTER** 的框架。通过对流匹配（Flow-based）模型采样过程的重新设计，论文成功打破了“必须完成所有去噪步骤才能输出动作”的性能瓶颈，实现了在保证长程轨迹精度的同时，将即时反应延迟压缩至单步计算。

### 2. 核心创新点与方法论
*   **反应时间量化模型**：作者从理论上证明了反应时间由“首个动作时间（TTFA）”与“执行视野（execution horizon）”共同决定，并指出传统的均匀采样在实时交互中是低效的。
*   **Horizon-Aware Schedule（视野感知调度）**：这是本论文的核心创新。通过改变流匹配模型在去噪过程中的采样权重，FASTER 优先计算并输出近期的动作。
*   **非对称采样策略**：FASTER 将靠近时间轴起点的动作去噪步数大幅缩减（至单步），而保持对长程轨迹的平滑性约束。这种“即时响应先行”的逻辑重塑了生成式 VLA 的推理流程。
*   **流式计算架构**：结合流式客户端-服务器管线，进一步降低了端到端系统延迟，使其在消费级硬件上即可实现高频实时控制。

### 3. 潜在领域影响
*   **生成式具身智能的“实时化”突破**：传统基于扩散或流匹配的 VLA 模型通常因为推理耗时太长而难以部署在动态环境（如乒乓球、高速抓取）中，FASTER 为这一类大模型落地提供了通用范式。
*   **挑战“推断即瓶颈”的现状**：该方法论证明了模型不必通过堆叠算力来解决响应速度问题，通过优化调度逻辑即可获得数量级的性能提升，这对资源受限的边缘设备部署具有重要参考意义。

### 4. 受益的相关领域与应用
*   **动态环境下的机器人控制**：乒乓球、快速避障、甚至人机协作等对毫秒级响应要求极高的场景。
*   **端侧多模态模型部署**：对于需要在移动机器人或无人机上运行的视觉大模型，该方案显著降低了对昂贵算力的依赖。
*   **生成式控制理论**：为扩散模型和流匹配模型在“顺序决策（Sequential Decision Making）”中的应用提供了新的优化策略，未来可能延伸至自动驾驶和无人机轨迹规划。

### 5. 可推断的局限性
*   **长短程精度权衡（Trade-off）**：虽然 FASTER 压缩了即时反应步数，但这种非对称采样是否会导致模型在面对极度复杂、需要深度预测的轨迹时，出现长程动作的抖动或漂移，仍需验证。
*   **对特定模型结构的依赖**：该方法主要针对流匹配（Flow-based）架构设计，对于基于 Transformer 预测的离散动作或高斯分布建模的传统 VLA 模型，其迁移成本尚不可知。
*   **动态环境的边界**：在处理极高频的突发环境改变时，仅依赖“单步去噪”的即时反应在极端物理场景下的可靠性（Robustness）还有待更长时间尺度的压力测试。

**专家总结：**
这篇论文的趣味性在于它并没有试图去“加速”采样过程本身，而是通过**重新审视控制系统的时间维度**，利用生成式模型的特性实现了响应策略的优化。这种“不增加算力但通过调度逻辑优化实现质变”的研究思路，是当前大模型从实验环境走向物理世界部署的关键技术方向。

**Key Findings:**

- We show that reaction time follows a uniform distribution determined jointly by the Time to First Action (TTFA) and the execution horizon.
- To overcome this issue, we propose Fast Action Sampling for ImmediaTE Reaction (FASTER).

**Links:**

- [PDF](https://arxiv.org/pdf/2603.19199v1)
- [arXiv](https://arxiv.org/abs/2603.19199v1)

---

