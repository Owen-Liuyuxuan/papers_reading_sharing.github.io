time: 20260317

# Arxiv Computer Vision Papers - 2026-03-17

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要 (2026-03-16)**

**1. 核心主题与趋势**

今日的论文集清晰地反映了当前计算机视觉研究的三大前沿融合趋势：

*   **具身智能与机器人操作成为焦点**：超过一半的论文（1, 2, 3, 7, 8, 9）直接围绕机器人或智能体在物理世界中的感知、推理与行动展开。研究重点正从**被动感知**转向**主动交互与闭环控制**。
*   **视觉-语言-动作（VLA）模型的深化**：研究不再满足于简单的“看图说话”，而是追求更扎实的**基础表征**（论文4）、更真实的**世界模拟与落地**（论文1, 8）以及更复杂的**推理能力**（论文2, 7）。核心是让模型理解物理规律并做出可执行的决策。
*   **三维视觉与物理真实感**：多篇论文致力于构建更逼真、更具物理一致性的三维世界，服务于人机交互（论文5）、人体重建（论文6）和场景编辑（论文10）。这体现了对**仿真质量**和**数字孪生**基础技术的持续投入。

**2. 重点与创新性论文**

*   **最具影响力的基准论文**：**《RealVLG-R1》**（论文1）提出了一个大规模、真实世界的视觉-语言 grounding 基准，专门针对机器人任务。其实景数据和复杂任务设定有望成为评估和推动 VLA 模型实际应用能力的新标准。
*   **最具方法创新性的论文**：**《From Passive Observer to Active Critic》**（论文7）引入“过程推理”概念，通过强化学习让智能体学会批判性地评估自身行动序列的合理性，这为提升机器人操作的可靠性和可解释性提供了新思路。**《HSImul3R》**（论文5）将物理仿真循环引入人体-场景交互重建，显著提升了重建结果的物理合理性与仿真可用性。
*   **高效化应用的亮点**：**《Fast SAM 3D Body》**（论文6）专注于**实时**全身人体网格恢复，是 Segment Anything Model (SAM) 系列在三维人体领域的高效化落地，对AR/VR、动画制作有直接应用价值。

**3. 新兴研究方向**

*   **协作式空间推理**：论文2探索多智能体间的协作空间理解，预示着从单一智能体向**群体智能协同**的研究扩展。
*   **全景可供性预测**：论文9提出的“全景可供性预测”任务，要求模型一次性理解整个场景中所有可能的交互点，这是对传统物体级可供性分析的**场景级升级**，对机器人自主探索至关重要。
*   **世界模型与真实城市对接**：论文8尝试将世界仿真模型“锚定”在真实大都市中，标志着**大规模、高保真数字孪生**构建从技术演示走向与具体现实场景结合的关键一步。

**4. 推荐精读论文**

根据研究价值与影响力，建议优先阅读以下三篇：

1.  **《RealVLG-R1》**：**必读**。了解下一代机器人视觉-语言任务基准的设计思路、数据规模和评估挑战，把握领域发展方向。
2.  **《From Passive Observer to Active Critic》**：**推荐阅读**。学习如何将高级推理（过程批判）与低级控制（强化学习）相结合，以提升机器人操作智能的新范式。
3.  **《HSImul3R》**：**推荐阅读（尤其关注三维重建与仿真）**。作为物理引导重建的典范，其“仿真即用”的设计理念和物理在环方法对生成高质量仿真数据具有重要参考价值。

**总结**：今日的论文表明，计算机视觉的核心驱动力正紧密围绕 **“让AI系统在物理世界中有效行动”** 这一目标。研究呈现出**基准驱动**（RealVLG-R1）、**认知深化**（主动批判、协作推理）和**技术夯实**（物理仿真、高效模型）并行的特点。机器人学、强化学习与三维视觉的交叉融合是当前最活跃的创新地带。

---

## Table of Contents

1. [RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](#2603.14880v1)
2. [Ego to World: Collaborative Spatial Reasoning in Embodied Systems via Reinforcement Learning](#2603.14811v1)
3. [Towards Generalizable Robotic Manipulation in Dynamic Environments](#2603.15620v1)
4. [Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models](#2603.15618v1)
5. [HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions](#2603.15612v1)
6. [Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery](#2603.15603v1)
7. [From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation](#2603.15600v1)
8. [Grounding World Simulation Models in a Real-World Metropolis](#2603.15583v1)
9. [Panoramic Affordance Prediction](#2603.15558v1)
10. [Learning Latent Proxies for Controllable Single-Image Relighting](#2603.15555v1)

---

## Papers

<a id='2603.14880v1'></a>
## [RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation](https://arxiv.org/abs/2603.14880v1)

**Authors:** Linfei Li, Lin Zhang, Ying Shen

**Published:** 2026-03-16

**Categories:** cs.CV

**Abstract:**

Visual-language grounding aims to establish semantic correspondences between natural language and visual entities, enabling models to accurately identify and localize target objects based on textual instructions. Existing VLG approaches focus on coarse-grained, object-level localization, while traditional robotic grasping methods rely predominantly on geometric cues and lack language guidance, which limits their applicability in language-driven manipulation scenarios. To address these limitations, we propose the RealVLG framework, which integrates the RealVLG-11B dataset and the RealVLG-R1 model to unify real-world visual-language grounding and grasping tasks. RealVLG-11B dataset provides multi-granularity annotations including bounding boxes, segmentation masks, grasp poses, contact points, and human-verified fine-grained language descriptions, covering approximately 165,000 images, over 800 object instances, 1.3 million segmentation, detection, and language annotations, and roughly 11 billion grasping examples. Building on this dataset, RealVLG-R1 employs Reinforcement Fine-tuning on pretrained large-scale vision-language models to predict bounding boxes, segmentation masks, grasp poses, and contact points in a unified manner given natural language instructions. Experimental results demonstrate that RealVLG supports zero-shot perception and manipulation in real-world unseen environments, establishing a unified semantic-visual multimodal benchmark that provides a comprehensive data and evaluation platform for language-driven robotic perception and grasping policy learning. All data and code are publicly available at https://github.com/lif314/RealVLG-R1.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文《RealVLG-R1: A Large-Scale Real-World Visual-Language Grounding Benchmark for Robotic Perception and Manipulation》的分析如下：

### 1. 论文核心贡献总结
该论文提出了 **RealVLG** 框架，旨在打破传统视觉语言定位（VLG）与机器人抓取任务之间的壁垒。通过构建包含大规模多粒度标注（从检测、分割到抓取姿态）的 **RealVLG-11B** 数据集，并结合 **RealVLG-R1** 模型，实现了从自然语言指令到精确感知及物理操作（抓取）的端到端统一建模。

### 2. 关键创新与方法论
*   **多粒度任务统一：** 该方法不仅关注语义层面的定位（Bounding Boxes/Masks），还深入到操作层面（Grasp poses/Contact points），将“语义感知”与“机器人动作空间”在模型输出端进行了对齐。
*   **超大规模合成与真实数据融合：** 110亿（11 Billion）级别的抓取示例量级是其核心亮点。这暗示了该研究可能采用了高效的自动化标注或仿真到真实（Sim-to-Real）的数据生成技术，用于增强预训练模型在真实物理环境中的泛化能力。
*   **强化微调（Reinforcement Fine-tuning）：** 与传统的监督学习不同，论文采用强化学习策略对大规模预训练多模态模型（LVM/VLM）进行微调，使其在复杂环境下不仅能“识别”目标，还能“优化”抓取策略。

### 3. 对领域的潜在影响
*   **基准范式转变：** RealVLG-11B 提供了一个统一的、大规模的真实世界基准，有望改变目前机器人感知和控制任务往往处于独立开发轨道的现状。
*   **提升模型的物理常识：** 通过引入千万级的抓取数据，该研究推动了多模态大模型从“看图说话”向“动手操作”跨越，是迈向具身智能（Embodied AI）的重要一步。
*   **零样本（Zero-shot）能力的验证：** 其在未知环境下的表现证明了该框架在实际部署中的鲁棒性，为工业机器人和家庭辅助机器人的语言指令操控提供了高可信度的参考。

### 4. 获益的相关领域与应用
*   **具身智能与机器人学：** 直接助力服务机器人（如家庭管家）实现复杂任务的语言理解与执行。
*   **工业自动化：** 在非结构化环境中（如物流分拣、精密零件拆解）实现更灵活的抓取任务。
*   **多模态大模型开发：** 为视觉-动作对齐（Vision-Action Alignment）的研究人员提供了优质的训练基座和评估平台。
*   **人机协作（HRC）：** 强化了机器人对人类模糊指令的理解能力，使人机交互更加自然高效。

### 5. 可推断的局限性
*   **计算资源需求：** 尽管模型在实验中表现出色，但基于“强化微调”的大规模预训练模型在推理侧（Inference）的实时性可能面临挑战，尤其是需要部署在边缘计算设备上时。
*   **泛化能力的边界：** 摘要提到“110亿抓取示例”，这通常依赖于仿真生成或自动标注，在极端复杂或未见过的物理材质/光照条件下，模型可能会出现性能断层（Domain Gap）。
*   **动作序列的连贯性：** 论文侧重于单步抓取（Grasping），对于涉及长链条决策（Long-horizon tasks，如“把杯子拿过来并放在桌子上”）的复杂操作序列，目前的端到端架构可能仍有待进一步的逻辑规划扩展。

**专家总结：**
RealVLG-R1 论文的趣味性在于它试图**用“定位”的逻辑去解决“交互”的问题**。在当前大模型热潮下，将视觉定位的高精度语义能力与机器人操作的物理约束相结合，是解决具身智能“最后一公里”问题的必经之路。该项目的开源性质将使其成为该领域重要的基石级资源。

**Key Findings:**

- To address these limitations, we propose the RealVLG framework, which integrates the RealVLG-11B dataset and the RealVLG-R1 model to unify real-world visual-language grounding and grasping tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.14880v1)
- [arXiv](https://arxiv.org/abs/2603.14880v1)

---

<a id='2603.14811v1'></a>
## [Ego to World: Collaborative Spatial Reasoning in Embodied Systems via Reinforcement Learning](https://arxiv.org/abs/2603.14811v1)

**Authors:** Heng Zhou, Li Kang, Yiran Qin, Xiufeng Song, Ao Yu, Zilu Zhang, Haoming Song, Kaixin Xu, Yuchen Fan, Dongzhan Zhou, Xiaohong Liu, Ruimao Zhang, Philip Torr, Lei Bai, Zhenfei Yin

**Published:** 2026-03-16

**Categories:** cs.RO, cs.CV

**Abstract:**

Understanding the world from distributed, partial viewpoints is a fundamental challenge for embodied multi-agent systems. Each agent perceives the environment through an ego-centric view that is often limited by occlusion and ambiguity. To study this problem, we introduce the Ego-to-World (E2W) benchmark, which evaluates a vision-language model's ability to fuse heterogeneous viewpoints across three tasks: (i) global counting, (ii) relational location reasoning, and (iii) action-oriented grasping that requires predicting view-specific image coordinates. To address this setting, we propose CoRL, a two-stage framework that combines Chain-of-Thought supervised fine-tuning with reinforcement learning using Group-Relative Policy Optimization. Its core component, the Cross-View Spatial Reward (CVSR), provides dense task-aligned feedback by linking reasoning steps to visual evidence, ensuring coherent cross-view entity resolution, and guiding the model toward correct final predictions. Experiments on E2W show that CoRL consistently surpasses strong proprietary and open-source baselines on both reasoning and perception-grounding metrics, while ablations further confirm the necessity of each CVSR component. Beyond that, CoRL generalizes to external spatial reasoning benchmarks and enables effective real-world multi-robot manipulation with calibrated multi-camera rigs, demonstrating cross-view localization and successful grasp-and-place execution. Together, E2W and CoRL provide a principled foundation for learning world-centric scene understanding from distributed, ego-centric observations, advancing collaborative embodied AI.

**Analysis:**

### 1. 摘要翻译
理解分布式、局部视角下的世界是具身多智能体系统面临的根本挑战。每个智能体通过受遮挡和歧义影响的自我中心视角感知环境。为研究此问题，我们引入了Ego-to-World (E2W)基准测试，旨在评估视觉语言模型（VLM）融合异构视角的能力，涵盖三项任务：(i) 全局计数，(ii) 关系定位推理，(iii) 需要预测特定视角图像坐标的面向动作的抓取。为应对此挑战，我们提出了CoRL框架，这是一种结合思维链（CoT）监督微调与基于组相对策略优化（GRPO）强化学习的两阶段框架。其核心组件——跨视角空间奖励（CVSR），通过将推理步骤与视觉证据关联，提供密集且任务对齐的反馈，确保跨视角实体解析的一致性，引导模型达成正确预测。实验表明，CoRL在推理和感知对齐指标上均显著超越了现有模型，且在真实世界的机器人抓取任务中表现出色。

### 2. 方法动机分析
*   **驱动力**：打破单一视角感知的局限，实现多智能体协作下的全局场景理解与精准动作规划。
*   **现有痛点**：当前VLMs多局限于单一视角，忽略了多视角下跨智能体的信息融合；或简单假设存在“上帝视角”的全局状态，脱离了真实具身系统分布式的本质，导致遮挡和歧义无法有效解决。
*   **研究假设**：通过在强化学习训练中引入显式的空间奖励信号，可以引导VLM自动学习跨视角的实体对齐和空间关系推理，从而提升协作感知能力。

### 3. 方法设计详解
CoRL框架采用 **SFT + RL** 两阶段训练 pipeline：
*   **阶段一（SFT）**：利用包含思维链（CoT）的标注数据进行监督微调，赋予模型基本的空间 reasoning 和指令遵循能力，作为RL的冷启动起点。
*   **阶段二（RL/GRPO）**：使用组相对策略优化（GRPO）。对于每个输入，采样 $G$ 个候选回答，计算其针对CVSR的奖励并归一化，通过调整策略概率比实现稳定优化。
*   **核心模块：CVSR（跨视角空间奖励）**
    1.  **Grounding Reward（定位奖励）**：通过匈牙利匹配算法计算预测框与真实框的IoU，强制模型输出精准的空间坐标。
    2.  **Overlap Reward（重叠奖励）**：惩罚或奖励模型对跨视角相同实例的计数一致性，迫使模型识别冗余视角并建立跨空间锚点。
    3.  **Answer Reward（回答奖励）**：针对QA任务使用精确匹配，针对抓取任务使用距离加权惩罚，确保最终输出的正确性。

### 4. 方法对比分析
*   **本质区别**：与现有模型（如COMBO）假设全局状态不同，CoRL处理原始分布式观测，并引入CVSR奖励函数，将“跨视角一致性”作为学习的目标。
*   **创新贡献**：
    *   **E2W基准**：填补了多视角/多智能体协作空间理解的测试空白。
    *   **CVSR函数**：将空间几何约束转化为可微的奖励信号，有效桥接了语言模型与具身物理空间。
*   **适用场景**：多机器人协作、V2X感知、复杂场景下的精准操作。

### 5. 实验分析（精简版）
*   **关键结果**：在E2W-Bench上，CoRL-7B在空间推理和抓取任务中均达到了最高分，显著超过了未经SFT或直接RL训练的基线模型。
*   **主要优势**：极强的跨视角解析能力，能处理部分遮挡；在真实世界机器人抓取中，CoRL展示了对sim-to-real差异的鲁棒性。
*   **主要局限**：目前的架构是集中式的（所有视角传入一个VLM），随着智能体数量增加，通信开销和推理延迟会成为瓶颈；且目前仅支持同步静止帧处理。

### 6. 实用指南
*   **开源情况**：代码及训练细节已通过HuggingFace开源（参考原文链接）。
*   **实现细节**：关键超参数为 $\epsilon=0.2$ (GRPO裁剪阈值)，$\beta=0.04$ (KL惩罚)，以及CVSR中 `w_ans=0.7`, `w_ground=0.1`, `w_overlap=0.2` 的权重设置。
*   **迁移可能**：该奖励设计思想可迁移至任何涉及多传感器输入处理的多模态大模型，特别是需要实体空间一致性的任务。

### 7. 总结
*   **核心思想**：通过引入跨视角空间一致性奖励，引导VLM实现多智能体协同感知。
*   **速记版pipeline**：
    1. 生成思维链引导的监督微调；
    2. 针对特定任务采样多个候选响应；
    3. 利用空间一致性（框对齐、重复检查）计算反馈；
    4. 通过组相对优化更新模型策略。

**Key Findings:**

- To study this problem, we introduce the Ego-to-World (E2W) benchmark, which evaluates a vision-language model's ability to fuse heterogeneous viewpoints across three tasks: (i) global counting, (ii) relational location reasoning, and (iii) action-oriented grasping that requires predicting view-specific image coordinates.
- To address this setting, we propose CoRL, a two-stage framework that combines Chain-of-Thought supervised fine-tuning with reinforcement learning using Group-Relative Policy Optimization.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.14811v1)
- [arXiv](https://arxiv.org/abs/2603.14811v1)

---

<a id='2603.15620v1'></a>
## [Towards Generalizable Robotic Manipulation in Dynamic Environments](https://arxiv.org/abs/2603.15620v1)

**Authors:** Heng Fang, Shangru Li, Shuhan Wang, Xuanyang Xi, Dingkang Liang, Xiang Bai

**Published:** 2026-03-16

**Categories:** cs.CV, cs.RO

**Abstract:**

Vision-Language-Action (VLA) models excel in static manipulation but struggle in dynamic environments with moving targets. This performance gap primarily stems from a scarcity of dynamic manipulation datasets and the reliance of mainstream VLAs on single-frame observations, restricting their spatiotemporal reasoning capabilities. To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite. Through comprehensive experiments, we systematically evaluate existing VLAs on dynamic tasks, explore effective training strategies for dynamic awareness, and validate the generalizability of dynamic data. Furthermore, we propose PUMA, a dynamics-aware VLA architecture. By integrating scene-centric historical optical flow and specialized world queries to implicitly forecast object-centric future states, PUMA couples history-aware perception with short-horizon prediction. Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines. Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks. All code and data are available at https://github.com/H-EmbodVis/DOMINO.

**Analysis:**

这是一份针对论文《Towards Generalizable Robotic Manipulation in Dynamic Environments》的深度分析报告。

---

### 1. 摘要翻译
现有的视觉-语言-动作（VLA）模型在静态操作中表现出色，但在处理移动目标的动态环境时表现不佳。这种性能差距主要源于缺乏动态操作数据集，以及主流模型依赖单帧观测而限制了时空推理能力。为此，我们引入了 DOMINO，一个用于泛化动态操作的大规模数据集和基准测试，包含35个具有层级复杂性的任务、超过11万条专家轨迹及多维度评估套件。我们系统评估了现有 VLA 模型，并提出了一种具有动态感知能力的 VLA 架构——PUMA。通过整合场景中心的历史光流和专门的世界查询（world queries）来隐式预测物体中心的未来状态，PUMA 将历史感知与短程预测相结合。实验结果表明，PUMA 实现了最先进的性能，成功率比基线提高了 6.3%。此外，在动态数据上进行训练能促进稳健时空表示的形成，并可迁移至静态任务。

### 2. 方法动机分析
*   **驱动力**：旨在解决机器人动态操作中“时空精度不足”和“环境预测缺失”的关键问题，推动 Embodied AI 从静态场景向复杂动态场景泛化。
*   **现有痛点**：当前 VLA 模型严重依赖“单帧观测”，导致其无法捕捉物体的物理动态规律，进而无法在与动态物体交互时进行准确的闭环控制。
*   **研究假设**：动态环境下的成功交互依赖于对“历史背景的捕获”以及“对物体未来轨迹的 anticipatory（预期性）建模”，而不仅仅是当前视角的静态匹配。

### 3. 方法设计详解
PUMA 的核心架构基于 Qwen3-VL，通过引入时空编码与辅助预测模块实现动态感知：
1.  **场景中心历史动态编码（Scene-Centric Encoding）**：摒弃直接堆叠原始帧的低效做法，采样 $h$ 个历史帧并计算**光流图（Optical Flow Map）**。光流能显式提取物体的移动趋势，作为 dense 运动提示辅助策略学习。
2.  **双查询机制（Dual-Query Mechanism）**：
    *   **Action Query**：负责解码动作序列（Action Chunking），保证闭环控制的连续性。
    *   **World Query**：利用学习到的 $N$ 个世界查询向量，显式聚合时空上下文。
3.  **对象中心未来预测（Future Prediction）**：通过 GroundingDINO 和 SAM2 提取目标物体的掩码，利用 DINO 编码器计算目标物体在未来帧的特征作为监督信号。该模块仅在训练时起作用，通过相似度损失函数 $\mathcal{L}_{world}$ 迫使策略层学习到物体未来运动的隐式表征，提升对动态目标的预测能力。

### 4. 方法对比分析
*   **本质区别**：传统方法侧重于对静态场景特征的提取，PUMA 则是将**显式运动流（光流）**与**物体轨迹的预测性监督（World Query）**作为核心驱动。
*   **创新贡献**：提出了一种无需在推理时增加计算开销的 auxiliary-prediction 训练机制，实现了从“反应式”动作到“ anticipatory（预测性）”闭环控制的跨越。
*   **适用场景**：高动态、移动目标的抓取、跟随或协作任务。

### 5. 实验分析
*   **核心结论**：在 DOMINO 基准测试中，PUMA 的平均成功率（17.20%）明显优于现有 SOTA 模型（如 OpenVLA-OFT 的 10.90%）。
*   **优势**：通过融入历史光流，PUMA 有效解决了 baseline 模型因盲目跟踪导致的动作抖动和时序不一致问题。
*   **局限**：模型对极端随机且突变的物理环境（Level 3 难度）依然存在较大的挑战，依赖准确的光流计算可能受到遮挡环境的限制。

### 6. 实用指南
*   **开源情况**：代码和数据集已通过 GitHub (https://github.com/H-EmbodVis/DOMINO) 公开。
*   **实现细节**：在训练阶段，利用预训练好的 GroundingDINO 和 SAM2 进行伪标注，无需额外人工交互。建议关注 $\lambda$ 超参数，它调节了预测任务对策略学习的影响权重。
*   **迁移建议**：该架构设计的“光流编码 + 动作/世界查询”模块可直接迁移到任何基于 Transformer 的 Embodied Agent 中，尤其是需要处理非平稳目标的长视距任务。

### 7. 总结
*   **核心思想**：通过时空光流感知与预测监督，赋能 VLA 模型动态物体轨迹anticipation能力。
*   **速记版 Pipeline**：
    1.  提取多视角历史帧，计算光流显式表征运动趋势；
    2.  利用 Grounding 模型锁定目标，在训练中实施未来状态预测监督；
    3.  通过双查询机制整合历史动态与当前动作，输出连续闭环控制；
    4.  训练结束后剔除预测模块，仅使用动作策略进行推理。

**Key Findings:**

- To address this, we introduce DOMINO, a large-scale dataset and benchmark for generalizable dynamic manipulation, featuring 35 tasks with hierarchical complexities, over 110K expert trajectories, and a multi-dimensional evaluation suite.
- Furthermore, we propose PUMA, a dynamics-aware VLA architecture.
- Results demonstrate that PUMA achieves state-of-the-art performance, yielding a 6.3% absolute improvement in success rate over baselines.
- Moreover, we show that training on dynamic data fosters robust spatiotemporal representations that transfer to static tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15620v1)
- [arXiv](https://arxiv.org/abs/2603.15620v1)

---

<a id='2603.15618v1'></a>
## [Look Before Acting: Enhancing Vision Foundation Representations for Vision-Language-Action Models](https://arxiv.org/abs/2603.15618v1)

**Authors:** Yulin Luo, Hao Chen, Zhuangzhe Wu, Bowen Sui, Jiaming Liu, Chenyang Gu, Zhuoyang Liu, Qiuxuan Feng, Jiale Yu, Shuo Gu, Peng Jia, Pheng-Ann Heng, Shanghang Zhang

**Published:** 2026-03-16

**Categories:** cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for robotic manipulation, in which reliable action prediction critically depends on accurately interpreting and integrating visual observations conditioned on language instructions. Although recent works have sought to enhance the visual capabilities of VLA models, most approaches treat the LLM backbone as a black box, providing limited insight into how visual information is grounded into action generation. Therefore, we perform a systematic analysis of multiple VLA models across different action-generation paradigms and observe that sensitivity to visual tokens progressively decreases in deeper layers during action generation. Motivated by this observation, we propose \textbf{DeepVision-VLA}, built on a \textbf{Vision-Language Mixture-of-Transformers (VL-MoT)} framework. This framework enables shared attention between the vision foundation model and the VLA backbone, injecting multi-level visual features from the vision expert into deeper layers of the VLA backbone to enhance visual representations for precise and complex manipulation. In addition, we introduce \textbf{Action-Guided Visual Pruning (AGVP)}, which leverages shallow-layer attention to prune irrelevant visual tokens while preserving task-relevant ones, reinforcing critical visual cues for manipulation with minimal computational overhead. DeepVision-VLA outperforms prior state-of-the-art methods by 9.0\% and 7.5\% on simulated and real-world tasks, respectively, providing new insights for the design of visually enhanced VLA models.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Look Before Acting》的论文分析如下：

### 1. 主要贡献总结
该论文针对视觉-语言-动作（VLA）模型在深层网络中视觉敏感度下降的问题，提出了 **DeepVision-VLA** 架构。通过引入视觉专家模型与VLA骨干网络的深度融合，以及基于动作导向的视觉令牌裁剪（AGVP）策略，该研究显著提升了机器人对复杂任务的视觉感知与精细操作能力，在模拟和真实场景中均取得了显著的性能提升。

### 2. 关键创新点与方法论
*   **深层视觉敏感度退化分析**：论文通过实验发现，VLA模型在动作生成过程中，深层网络对输入视觉信息的响应逐渐减弱，这限制了模型处理复杂精细操作的能力。
*   **Vision-Language Mixture-of-Transformers (VL-MoT)**：这是一项核心架构创新。它打破了传统VLA将视觉编码器视为“黑盒”的范式，通过在视觉专家模型与VLA主干之间建立共享注意力机制，将多层级的视觉特征直接注入到VLA的深层中，增强了特征表征的深度与连贯性。
*   **Action-Guided Visual Pruning (AGVP)**：该方法引入了一种高效的注意力机制，在浅层即实现视觉令牌的裁剪。通过剔除无关信息、保留任务关键线索，在降低计算开销的同时，强化了对决策至关重要的环境特征的关注。

### 3. 对该领域的潜在影响
*   **重塑VLA模型设计范式**：该论文挑战了现有的将LLM简单作为动作预测头（Head）的“黑盒”做法，提倡一种“视觉感官深度参与”的架构设计，这对未来构建端到端具身智能模型具有重要的指导意义。
*   **计算效率与精度平衡**：AGVP提供了一种优雅的解决方案，展示了如何在不显著增加计算负载的情况下，通过精细化视觉注意力分布提升机器人性能，这对边缘侧机器人部署至关重要。

### 4. 相关领域与应用价值
*   **具身智能与机器人操作**：直接受益于此研究，尤其是处理复杂几何关系（如抓取、装配）或动态环境的任务。
*   **视觉表示学习**：研究中关于“深层视觉信息衰减”的发现，为大型多模态模型（LMM）的视觉-语言对齐机制提供了理论补位。
*   **自动驾驶与无人系统**：需要高精度视觉感知与即时决策的场景（如无人机避障、自动泊车）均可借鉴此结构设计。

### 5. 可推断的潜在局限性
*   **视觉-语言模态偏移**：虽然强化了视觉表示，但如何确保视觉特征的细粒度注入不干扰语言指令的语义理解（即所谓的“视觉噪声干扰”），是一个需要权衡的难点。
*   **对预训练视觉模型的依赖**：VL-MoT架构的性能很大程度上依赖于所选视觉基础模型（Foundation Model）的质量。如果视觉底座在特定领域（如工业精密零件）表现不佳，该方法的增益可能受限。
*   **动作生成的时序泛化性**：摘要中未提及该方法在处理长时序任务中的表现，视觉特征的增强是否会引入过拟合风险（即在训练场景表现好但跨场景泛化能力受限），仍需进一步验证。

**专家总结**：这篇论文极具价值之处在于它并没有单纯地堆叠参数量，而是深入剖析了VLA模型内部的表征机理。这种“深入底层机制分析+针对性架构优化”的方法论，是当前计算机视觉领域从“大模型炼金术”向“科学建模”转型的典型代表作。

**Key Findings:**

- Motivated by this observation, we propose \textbf{DeepVision-VLA}, built on a \textbf{Vision-Language Mixture-of-Transformers (VL-MoT)} framework.
- In addition, we introduce \textbf{Action-Guided Visual Pruning (AGVP)}, which leverages shallow-layer attention to prune irrelevant visual tokens while preserving task-relevant ones, reinforcing critical visual cues for manipulation with minimal computational overhead.
- DeepVision-VLA outperforms prior state-of-the-art methods by 9.0\% and 7.5\% on simulated and real-world tasks, respectively, providing new insights for the design of visually enhanced VLA models.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15618v1)
- [arXiv](https://arxiv.org/abs/2603.15618v1)

---

<a id='2603.15612v1'></a>
## [HSImul3R: Physics-in-the-Loop Reconstruction of Simulation-Ready Human-Scene Interactions](https://arxiv.org/abs/2603.15612v1)

**Authors:** Yukang Cao, Haozhe Xie, Fangzhou Hong, Long Zhuo, Zhaoxi Chen, Liang Pan, Ziwei Liu

**Published:** 2026-03-16

**Categories:** cs.CV, cs.RO

**Abstract:**

We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos. Existing methods suffer from a perception-simulation gap: visually plausible reconstructions often violate physical constraints, leading to instability in physics engines and failure in embodied AI applications. To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry. In the forward direction, we employ Scene-targeted Reinforcement Learning to optimize human motion under dual supervision of motion fidelity and contact stability. In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry. We further present HSIBench, a new benchmark with diverse objects and interaction scenarios. Extensive experiments demonstrate that HSImul3R produces the first stable, simulation-ready HSI reconstructions and can be directly deployed to real-world humanoid robots.

**Analysis:**

这是一份针对论文 **HSImul3R** 的深度技术分析。

### 1. 摘要翻译
我们提出了 HSImul3R，一个用于从稀疏视角图像或单目视频中，实现交互式 3D 人-场景重构（HSI）的统一框架。现有方法存在“感知-仿真鸿沟”：重构虽然视觉上合理，但往往违反物理约束，导致在仿真引擎中不稳定，难以用于具身智能。为解决此问题，我们引入了物理驱动的闭环优化管线，将物理仿真器作为活跃监督者，共同优化人体动力学与场景几何。在“前向”阶段，采用场景目标强化学习来优化人体运动，确保运动的逼真度与接触稳定性；在“反向”阶段，提出“直接仿真奖励优化”（DSRO），通过重力稳定性和交互成功的仿真反馈，迭代细化场景几何。我们还推出了包含多样化场景的基准数据集 HSIBench。实验证明，HSImul3R 生成了首批稳定、可仿真的 HSI 重构结果，并可直接部署于真实人形机器人。

### 2. 方法动机分析
*   **驱动力**：解决现有的 3D 重构结果在物理仿真环境中“一碰即碎”或“相互穿模”的问题，弥合视觉表象与物理本质的鸿沟。
*   **现有痛点**：当前 HSI 重构过分关注 2D 视觉对齐，忽视了物理耦合。重构的物体往往存在结构缺失（如桌椅缺腿）或表面瑕疵，无法承载真实的人体交互。
*   **研究假设**：如果将物理引擎作为“优化器”反馈的一部分，动态调整场景几何与人体运动，就能生成既符合视觉感知、又具备物理稳定性的交互数据。

### 3. 方法设计详解
HSImul3R 的核心是**闭环双向优化管线**：
1.  **初始重建与对齐**：利用 DUSt3R 恢复场景，SAM2+4DHumans 提取人体运动，通过联合 bundle adjustment 将两者统一到同一坐标系。
2.  **前向优化 (物理约束运动)**：基于强化学习，引入 $\ell_{scene}$ 损失函数，强制人体接触点与场景表面保持物理距离，修正运动轨迹，避免穿模或飘移。
3.  **反向优化 (DSRO)**：这是核心贡献。该模块直接利用仿真引擎的反馈（物体能否在重力下站立、交互是否成功）作为梯度监督，反向更新 MIDI 生成模型的权重，优化物体的几何结构。如果物体不稳定，DSRO 会驱动生成模型生成更稳固的几何形态。

### 4. 方法对比分析
*   **本质区别**：传统方法是“重建 -> 仿真”的单向流；本文引入了“仿真反馈 -> 更新重建”的反馈回路。
*   **创新贡献**：**DSRO 算法**首次将物理仿真结果转化为一种可微分的奖励，通过重构与仿真的闭环，赋予了模型“感知物理稳定性”的能力。

### 5. 实验分析
*   **验证方法**：在 HSIBench 数据集上，通过“稳定性-HSI”指标对比基线 HSfM 和其他变体。
*   **关键结论**：在 Easy、Medium、Hard 三个难度等级下，HSImul3R 的稳定性均远超现有基线。
*   **局限**：对极复杂交互或多物体场景的处理仍有失败率，且依赖于基础 MIDI 模型的先验知识，对完全偏离分布的数据泛化力有限。

### 6. 实用指南
*   **开源情况**：已开源 [https://yukangcao.github.io/HSImul3R/](https://yukangcao.github.io/HSImul3R/)。
*   **关键点**：DSRO 训练需较大的显存与时间（1800 步，4x A100 GPU），LoRA 微调是提升效率的关键。
*   **迁移建议**：该闭环优化思路可直接迁移到任何需要“物理一致性”的交互场景（如工业机械臂抓取、人机协作轨迹生成）。

### 7. 总结
*   **核心思想**：利用物理仿真反馈构建闭环，实现物理鲁棒的 3D 人-场景交互重构。
*   **速记版 Pipeline**：
    1.  基础 3D 场景与人体动作重构；
    2.  强化学习优化动作，确保人与物体的物理贴合；
    3.  仿真反馈评估几何稳定性；
    4.  DSRO 修正物体几何，达成物理最优重建。

**Key Findings:**

- We present HSImul3R, a unified framework for simulation-ready 3D reconstruction of human-scene interactions (HSI) from casual captures, including sparse-view images and monocular videos.
- To bridge this gap, we introduce a physically-grounded bi-directional optimization pipeline that treats the physics simulator as an active supervisor to jointly refine human dynamics and scene geometry.
- In the reverse direction, we propose Direct Simulation Reward Optimization, which leverages simulation feedback on gravitational stability and interaction success to refine scene geometry.
- We further present HSIBench, a new benchmark with diverse objects and interaction scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15612v1)
- [arXiv](https://arxiv.org/abs/2603.15612v1)

---

<a id='2603.15603v1'></a>
## [Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery](https://arxiv.org/abs/2603.15603v1)

**Authors:** Timing Yang, Sicheng He, Hongyi Jing, Jiawei Yang, Zhijian Liu, Chuhang Zou, Yue Wang

**Published:** 2026-03-16

**Categories:** cs.CV

**Abstract:**

SAM 3D Body (3DB) achieves state-of-the-art accuracy in monocular 3D human mesh recovery, yet its inference latency of several seconds per image precludes real-time application. We present Fast SAM 3D Body, a training-free acceleration framework that reformulates the 3DB inference pathway to achieve interactive rates. By decoupling serial spatial dependencies and applying architecture-aware pruning, we enable parallelized multi-crop feature extraction and streamlined transformer decoding. Moreover, to extract the joint-level kinematics (SMPL) compatible with existing humanoid control and policy learning frameworks, we replace the iterative mesh fitting with a direct feedforward mapping, accelerating this specific conversion by over 10,000x. Overall, our framework delivers up to a 10.9x end-to-end speedup while maintaining on-par reconstruction fidelity, even surpassing 3DB on benchmarks such as LSPET. We demonstrate its utility by deploying Fast SAM 3D Body in a vision-only teleoperation system that-unlike methods reliant on wearable IMUs-enables real-time humanoid control and the direct collection of manipulation policies from a single RGB stream.

**Analysis:**

以下是对论文《Fast SAM 3D Body: Accelerating SAM 3D Body for Real-Time Full-Body Human Mesh Recovery》的深度分析：

### 1. 摘要翻译
SAM 3D Body (3DB) 在单目 3D 人体网格重建任务中达到了最先进的精度，但其每张图像几秒的推理延迟限制了其实时应用。我们提出了 Fast SAM 3D Body，这是一个无需训练的加速框架，通过重构 3DB 的推理路径来实现交互式速度。通过解耦串行空间依赖关系并应用架构感知剪枝，我们实现了并行化的多裁剪（multi-crop）特征提取和精简的 Transformer 解码。此外，为了提取与现有类人机器人控制和策略学习框架兼容的关节级运动学（SMPL），我们用直接的前馈映射取代了迭代网格拟合，将此特定转换速度提升了 10,000 倍以上。

### 2. 方法动机分析
*   **驱动力**：打破“高精度模型=高延迟”的魔咒，将顶尖的 3D 人体网格重建模型推向机器人实时控制、远程操作等落地场景。
*   **现有痛点**：3DB 架构存在严重的“复合延迟”：级联式的多阶段检测、串行的手工特征裁剪以及极其耗时的迭代 MHR 到 SMPL 拟合（这是主要的性能瓶颈）。
*   **研究假设**：通过消除模型推理过程中的动态数据依赖（将动态图转为静态图）并利用轻量化前馈网络替代复杂的迭代优化，可以在几乎不损失精度的前提下实现数量级加速。

### 3. 方法设计详解
*   **流程总结**：
    1.  **空间依赖解耦**：引入轻量级 pose prior，利用单阶段检测器一次性输出人体及四肢（手部）的粗略 BBox。
    2.  **批处理优化**：将人体和手部的裁剪特征提取合并为一个单一的 GPU 并行批处理操作，彻底消除原本的 CPU-GPU 往返开销。
    3.  **计算感知解码**：剪枝冗余查询，去除多余的自精炼步骤，通过 `torch.compile` 或 TensorRT 将推理路径重构为确定性的静态计算图。
    4.  **神经运动学映射**：训练一个 3 层 MLP ($f_\omega$)，直接将 MHR 空间的特征投影到 SMPL 关节空间，取代昂贵的迭代优化。
*   **模型结构**：保持原有的 3DB 编码器-解码器架构，通过“手术式”修改其执行流而非改动模型权重（Training-free）。
*   **算法解释**：关键创新在于 $f_\omega (\mathbf{x}) = \hat{\boldsymbol{\Theta}}_{\mathrm{smpl}}$，它学习了 MHR 到 SMPL 的映射函数，将原本涉及数百步优化的过程转化为单次矩阵乘法。

### 4. 方法对比分析
*   **本质区别**：与现有重构 ViT 架构的加速方法不同，本方法关注于“系统级优化”和“推理路径重构”，保留了原模型鲁棒的预训练权重。
*   **创新贡献**：解耦 spatial dependency 和替换 iterative solver 的思路非常直观且有效，实现了对 SOTA 模型的“无损”压缩。
*   **适用场景**：所有需要高精度网格重建且对实时性有严格要求（如机器人遥操作、AR 交互）的任务。

### 5. 实验分析
*   **验证方法**：在 3DPW、EMDB 等标准数据集上，对比了 Oracle（GT BBox）和 Automatic（预测 BBox）两种协议下的速度与精度。
*   **关键结果**：在 NVIDIA RTX 5090 上达到了约 65ms/帧的实时速度，实现了 8-11 倍的端到端加速，精度与原版 3DB 持平。
*   **局限性**：尽管做到了 Training-free，但神经投影模块需要基于 3DB 的输出进行少量的离线数据拟合训练（虽不影响主模型，但有额外的处理步骤）。

### 6. 实用指南
*   **开源情况**：已开源，GitHub 地址：`yangtiming/Fast-SAM-3D-Body`。
*   **实现要点**：关键在于将原本动态生成的 BBox 计算提前，并利用 CUDA Graph 或 TensorRT 固定计算图。迁移时可重点参考其 `f_\omega` 的 MLP 设计。

### 7. 总结
*   **核心思想**：通过推理路径静态化与关键模块轻量化映射，实现模型计算性能的极致压缩。
*   **速记版pipeline**：
    1.  单阶段检测产生所有区域 BBox；
    2.  多区域一次性并行处理特征；
    3.  移除冗余层，固定计算执行流；
    4.  利用 MLP 一步完成 topology 转换。

**Key Findings:**

- SAM 3D Body (3DB) achieves state-of-the-art accuracy in monocular 3D human mesh recovery, yet its inference latency of several seconds per image precludes real-time application.
- We present Fast SAM 3D Body, a training-free acceleration framework that reformulates the 3DB inference pathway to achieve interactive rates.
- We demonstrate its utility by deploying Fast SAM 3D Body in a vision-only teleoperation system that-unlike methods reliant on wearable IMUs-enables real-time humanoid control and the direct collection of manipulation policies from a single RGB stream.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15603v1)
- [arXiv](https://arxiv.org/abs/2603.15603v1)

---

<a id='2603.15600v1'></a>
## [From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation](https://arxiv.org/abs/2603.15600v1)

**Authors:** Yibin Liu, Yaxing Lyu, Daqi Gao, Zhixuan Liang, Weiliang Tang, Shilong Mu, Xiaokang Yang, Yao Mu

**Published:** 2026-03-16

**Categories:** cs.RO, cs.AI, cs.CL, cs.CV

**Abstract:**

Accurate process supervision remains a critical challenge for long-horizon robotic manipulation. A primary bottleneck is that current video MLLMs, trained primarily under a Supervised Fine-Tuning (SFT) paradigm, function as passive "Observers" that recognize ongoing events rather than evaluating the current state relative to the final task goal. In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics". We leverage outcome-based Reinforcement Learning to incentivize explicit Chain-of-Thought generation for progress estimation. Furthermore, our architecture constructs a structured temporal input by explicitly anchoring the video sequence between initial and current state images. Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance. Quantitatively, our 7B model achieves a 50% reduction in the mean absolute error of specialized reasoning baselines, demonstrating significant relative accuracy improvements over 72B-scale general MLLMs. Furthermore, PRIMO R1 exhibits strong zero-shot generalization on difficult failure detection tasks. We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%.

**Analysis:**

以下是对论文《From Passive Observer to Active Critic: Reinforcement Learning Elicits Process Reasoning for Robotic Manipulation》的深入分析：

### 1. 摘要翻译
准确的过程监督对于长周期机器人操作仍是一项关键挑战。一个主要瓶颈是，目前的视频多模态大模型（MLLM）主要在监督微调（SFT）范式下训练，仅充当识别正在发生事件的“观察者”，而非根据最终任务目标评估当前状态的“评论家”。在本文中，我们引入了PRIMO R1（过程推理引导监控），这是一个7B参数的框架，将视频MLLM转化为主动的“评论家”。我们利用基于结果的强化学习来激励模型生成显式的思维链（CoT）以进行进度估计。此外，我们的架构通过在初始和当前状态图像之间显式锚定视频序列，构建了结构化的时间输入。依托所提出的PRIMO数据集和基准，在多样的域内环境和域外真实人形机器人场景下的广泛实验表明，PRIMO R1达到了最先进的性能。定量上，我们的7B模型将专业推理基线的平均绝对误差降低了50%，在准确性上显著优于72B规模的通用MLLM。此外，PRIMO R1在困难的故障检测任务中表现出强大的零样本泛化能力，在RoboFail基准上达到了67.0%的准确率，超越了OpenAI o1等闭源模型。

### 2. 方法动机分析
*   **驱动力**：旨在填补视频MLLM在机器人操作任务中“知其然（描述行为）但不知其所以然（缺乏定量评估）”的鸿沟，将单纯的视频描述者升级为具备逻辑评估能力的监督者。
*   **现有痛点**：当前模型基于SFT，本质是回归或分类器，导致其在处理长周期任务时，容易因视觉表面的相似性产生“幻觉”，且无法解释预测结果，对未见物体或失败案例的泛化能力差。
*   **研究假设**：通过强化学习（RL）激励模型显式生成思维链（CoT），强制模型在预测进度前进行因果推理，能有效对齐连续视觉轨迹与离散逻辑任务目标。

### 3. 方法设计详解
*   **模型结构**：PRIMO R1的核心在于其“三元输入”结构（初始状态 $I_{init}$ + 视频序列 $V_{seq}$ + 当前状态 $I_{curr}$）和“分步推理”机制。
*   **流程总结**：
    1.  **输入构造**：输入包含初始/当前状态和过程视频，强制模型对比目标状态。
    2.  **思维链生成（RL环节）**：模型在输出进度分值前，必须先生成 `<think>` 标签中的内容。包括：
        *   `Planning`：任务拆解。
        *   `Observation`：对当前时间点的细粒度描述。
        *   `Reasoning`：基于计划与观察的因果推断。
    3.  **强化学习（GRPO）优化**：使用GRPO算法，通过基于结果的奖励函数（格式奖励+进度准确度奖励）更新策略，强制模型学习高质量推理。
*   **算法关键**：GRPO（组相对策略优化）避免了训练独立的Value Head，通过对一组采样的输出求平均分来估计Baseline，在7B模型上实现了资源高效的强化学习。

### 4. 方法对比分析
*   **本质区别**：从“直接回归任务进度”范式切换为“通过推理链验证进度”范式。
*   **创新贡献**：引入了结构化输入（Boundary Anchoring）和基于结果的思维链强化学习（CoT RL），极大提升了模型在跨域环境下的稳健性。
*   **适用场景**：高复杂度、长周期的机器人操作任务进度监测，以及故障检测。

### 5. 实验分析
*   **关键结论**：在四个环境中（AgiBot, Behavior, RoboTwin, Real Humanoid），7B的PRIMO R1性能显著超过了72B通用模型，且在Sim-to-Real的零样本迁移任务中保持高准确率。
*   **主要优势**：极低的平均绝对误差（MAE），强大的因果推理和故障检测能力，计算效率高。
*   **主要局限**：对长视频的理解仍依赖于基础LLM的上下文窗口限制。

### 6. 实用指南
*   **实现细节**：建议在微调时冻结部分视觉塔，使用Flash Attention-2优化内存。重点关注GRPO中的奖励函数设计（线性衰减的精度奖励）。
*   **迁移可能**：该架构可以轻松迁移到任何需要“过程监控”的任务（如工业生产线质检、复杂手工操作评估）。

### 7. 总结
*   **核心思想**：通过强化学习激励模型进行因果推理，将观察者升级为具备逻辑判断力的评论家。
*   **速记版Pipeline**：
    1. 锚定起始与终止状态；
    2. 生成分解后的任务执行计划；
    3. 对比过程视频进行推理；
    4. 强化学习微调推理链路。

**Key Findings:**

- In this paper, we introduce PRIMO R1 (Process Reasoning Induced Monitoring), a 7B framework that transforms video MLLMs into active "Critics".
- Supported by the proposed PRIMO Dataset and Benchmark, extensive experiments across diverse in-domain environments and out-of-domain real-world humanoid scenarios demonstrate that PRIMO R1 achieves state-of-the-art performance.
- We establish state-of-the-art performance on RoboFail benchmark with 67.0% accuracy, surpassing closed-source models like OpenAI o1 by 6.0%.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15600v1)
- [arXiv](https://arxiv.org/abs/2603.15600v1)

---

<a id='2603.15583v1'></a>
## [Grounding World Simulation Models in a Real-World Metropolis](https://arxiv.org/abs/2603.15583v1)

**Authors:** Junyoung Seo, Hyunwook Choi, Minkyung Kwon, Jinhyeok Choi, Siyoon Jin, Gayoung Lee, Junho Kim, JoungBin Lee, Geonmo Gu, Dongyoon Han, Sangdoo Yun, Seungryong Kim, Jin-Hwa Kim

**Published:** 2026-03-16

**Categories:** cs.CV

**Abstract:**

What if a world simulation model could render not an imagined environment but a city that actually exists? Prior generative world models synthesize visually plausible yet artificial environments by imagining all content. We present Seoul World Model (SWM), a city-scale world model grounded in the real city of Seoul. SWM anchors autoregressive video generation through retrieval-augmented conditioning on nearby street-view images. However, this design introduces several challenges, including temporal misalignment between retrieved references and the dynamic target scene, limited trajectory diversity and data sparsity from vehicle-mounted captures at sparse intervals. We address these challenges through cross-temporal pairing, a large-scale synthetic dataset enabling diverse camera trajectories, and a view interpolation pipeline that synthesizes coherent training videos from sparse street-view images. We further introduce a Virtual Lookahead Sink to stabilize long-horizon generation by continuously re-grounding each chunk to a retrieved image at a future location. We evaluate SWM against recent video world models across three cities: Seoul, Busan, and Ann Arbor. SWM outperforms existing methods in generating spatially faithful, temporally consistent, long-horizon videos grounded in actual urban environments over trajectories reaching hundreds of meters, while supporting diverse camera movements and text-prompted scenario variations.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文《Grounding World Simulation Models in a Real-World Metropolis》的分析如下：

### 1. 论文核心贡献摘要
该论文提出了 **Seoul World Model (SWM)**，这是首个基于真实城市地理空间的城市级世界模拟模型。通过结合检索增强生成（Retrieval-Augmented Conditioning）与一系列跨时空数据处理技术，SWM 实现了在真实城市尺度上生成具有高度空间保真度和长时间序列一致性的视频，成功跨越了从“幻觉式生成”到“地理接地（Geospatial Grounding）”的鸿沟。

### 2. 关键创新与方法论
该工作的核心在于解决“现实数据稀疏”与“生成连续性”之间的矛盾，主要创新点包括：
*   **检索增强式条件化（Retrieval-Augmented Conditioning）：** 将实时定位与街道视图图像检索作为生成过程的锚点，确保生成的环境是现实中存在的，而非模型主观臆造的。
*   **跨时空配对与视图插值（Cross-temporal Pairing & View Interpolation）：** 针对车载采集数据的时空稀疏性，通过视图插值技术合成连贯的训练视频，填补了现实轨迹数据分布的空缺。
*   **虚拟前瞻汇聚（Virtual Lookahead Sink）：** 这是为了解决长序列生成中的“漂移”问题（Drift），通过持续将生成片段重定位（Re-grounding）到未来位置的参考图像上，显著提升了长时间跨度下地理环境的一致性。

### 3. 对该领域的潜在影响
*   **从“生成式幻觉”向“生成式真实”的范式转换：** 此前的世界模型多关注视觉物理的合理性（Plausible），而 SWM 强调的是地理的客观性（Factual），这为生成式 AI 落地到高精度地图和地理空间计算提供了新思路。
*   **数据合成的闭环：** 该模型能够生成具备多样性摄像机轨迹的城市视频，这为自动驾驶的模拟器（Simulation）注入了强有力的合成数据流，减少了对昂贵实地路测数据的依赖。

### 4. 受益的关联领域与应用
*   **自动驾驶与机器人（Autonomous Driving/Robotics）：** 极大地增强了自动驾驶模拟器的场景覆盖范围，尤其是长尾场景（Long-tail scenarios）的生成与演练。
*   **城市规划与数字孪生（Urban Planning/Digital Twin）：** 可用于模拟城市基础设施改动后的视觉效果，为政策制定提供沉浸式的直观参考。
*   **增强现实（AR）与元宇宙（Metaverse）：** 将现实世界地理信息转化为可生成内容，为基于地理位置的 AR 体验提供了高保真的背景渲染能力。

### 5. 潜在局限性（基于摘要的推论）
*   **对实时性与算力的挑战：** 虽然模型效果优越，但结合检索系统和长序列重定位的复杂架构，在推理速度上可能面临极大的算力压力，能否实现“实时互动生成”仍需观察。
*   **极端动态环境的处理能力：** 尽管引入了跨时空配对，但面对城市中极为复杂的突发事件（如严重的交通事故、施工封路等），模型可能仍依赖于检索到的原始静态地标，其对动态场景实时变化的理解深度可能受限。
*   **检索系统的边界：** 模型的性能高度依赖于已存储的街道视图图像数据的质量与覆盖密度；在缺乏街景覆盖的偏远地区或未开发区域，该模型的扩展性（Generalization）可能面临挑战。

**总结评价：** 这篇论文的独特之处在于它将“生成式模型”与“地理空间检索”进行了深度集成，标志着世界模型开始进入“精细化、地理定位化”的工业应用新阶段。

**Key Findings:**

- We present Seoul World Model (SWM), a city-scale world model grounded in the real city of Seoul.
- SWM outperforms existing methods in generating spatially faithful, temporally consistent, long-horizon videos grounded in actual urban environments over trajectories reaching hundreds of meters, while supporting diverse camera movements and text-prompted scenario variations.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15583v1)
- [arXiv](https://arxiv.org/abs/2603.15583v1)

---

<a id='2603.15558v1'></a>
## [Panoramic Affordance Prediction](https://arxiv.org/abs/2603.15558v1)

**Authors:** Zixin Zhang, Chenfei Liao, Hongfei Zhang, Harold Haodong Chen, Kanghao Chen, Zichen Wen, Litao Guo, Bin Ren, Xu Zheng, Yinchuan Li, Xuming Hu, Nicu Sebe, Ying-Cong Chen

**Published:** 2026-03-16

**Categories:** cs.CV, cs.RO

**Abstract:**

Affordance prediction serves as a critical bridge between perception and action in embodied AI. However, existing research is confined to pinhole camera models, which suffer from narrow Fields of View (FoV) and fragmented observations, often missing critical holistic environmental context. In this paper, we present the first exploration into Panoramic Affordance Prediction, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding. To facilitate this novel task, we first introduce PAP-12K, a large-scale benchmark dataset containing over 1,000 ultra-high-resolution (12k, 11904 x 5952) panoramic images with over 12k carefully annotated QA pairs and affordance masks. Furthermore, we propose PAP, a training-free, coarse-to-fine pipeline inspired by the human foveal visual system to tackle the ultra-high resolution and severe distortion inherent in panoramic images. PAP employs recursive visual routing via grid prompting to progressively locate targets, applies an adaptive gaze mechanism to rectify local geometric distortions, and utilizes a cascaded grounding pipeline to extract precise instance-level masks. Experimental results on PAP-12K reveal that existing affordance prediction methods designed for standard perspective images suffer severe performance degradation and fail due to the unique challenges of panoramic vision. In contrast, PAP framework effectively overcomes these obstacles, significantly outperforming state-of-the-art baselines and highlighting the immense potential of panoramic perception for robust embodied intelligence.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于“全景可供性预测（Panoramic Affordance Prediction）”的论文分析如下：

### 1. 主要贡献总结
该论文首次将可供性预测（Affordance Prediction）从传统的针孔相机视角扩展至360度全景领域，填补了具身智能在全局环境感知方面的空白。研究者不仅构建了大规模、超高分辨率的全景基准数据集 **PAP-12K**，还提出了一种无需训练（training-free）的“粗到细”计算流水线 **PAP**，有效解决了全景图像特有的超高分辨率计算负担及几何畸变问题。

### 2. 核心创新与方法论
该工作的核心在于模拟人类的“中央凹视觉（foveal vision）”机制来处理全景图像，其关键创新点包括：
*   **递归视觉路由（Recursive Visual Routing）**：利用网格提示（grid prompting）逐步定位目标，避免了在处理超高分辨率（12k）全景图时因缩放导致的信息丢失。
*   **自适应注视机制（Adaptive Gaze Mechanism）**：专门用于纠正全景投影中常见的几何畸变，确保在提取局部特征时保持物体形状的准确性。
*   **级联接地流水线（Cascaded Grounding Pipeline）**：通过层级化处理，实现了从宏观场景到实例级掩码（instance-level masks）的精准提取。
*   **Training-free设计**：该流程不依赖繁重的模型微调，展现了良好的通用性和部署潜力。

### 3. 对该领域的潜在影响
*   **范式转移**：该研究挑战了传统具身智能局限于“窄视角”的现状，推动了从局部感知向全方位、全局感知范式的转变。
*   **数据集基准**：PAP-12K 的发布为全景视觉感知任务提供了一个高质量的评估标准，有望催生更多基于全景图的下游任务研究。
*   **计算效率路径**：通过“粗到细”而非暴力处理全景图的思路，为未来处理海量高分辨率图像数据提供了一种高效的范例。

### 4. 受益的相关领域与应用
*   **具身智能与机器人导航**：移动机器人（如室内服务机器人）若能拥有全景可供性感知能力，将极大提升其在复杂环境中的路径规划与交互决策能力，避免“盲区”隐患。
*   **自动驾驶**：全景感知能帮助车载系统更好地理解车辆四周的动态环境，特别是在交叉路口等复杂场景下的交互意图识别。
*   **VR/AR 内容生成与交互**：增强现实应用可以利用全景可供性来更自然地在虚拟世界中植入可交互对象。
*   **虚拟场景理解**：对于元宇宙或数字孪生系统，全局化的语义与功能理解是实现高度沉浸式交互的基础。

### 5. 可推断的潜在局限性
*   **实时性挑战**：尽管采用了“粗到细”的方法，但处理 12k 超高分辨率图像仍面临较高的计算时延，在实时性要求极高的机器人系统中可能需要进一步的硬件优化（如FPGA或边缘加速）。
*   **跨模态泛化**：摘要中未提及该方法在动态视频序列（Temporal consistency）中的表现，若应用于机器人，需解决全景视频帧间的一致性问题。
*   **依赖先验知识**：如果其级联接地流水线高度依赖预训练大模型（如CLIP或SAM），则在垂直行业领域（如工业、医疗）可能面临领域分布偏移（Domain Shift）的挑战。

**总结评价：**
这篇论文的趣味性在于它巧妙地将**人类的视觉注意力机制（中央凹）与现代视觉接地（Visual Grounding）技术**结合，完美契合了全景图像“信息量大但畸变重”的特性。对于推动下一代具身智能系统实现“全知视角”的感知具有里程碑式的意义。

**Key Findings:**

- In this paper, we present the first exploration into Panoramic Affordance Prediction, utilizing 360-degree imagery to capture global spatial relationships and holistic scene understanding.
- To facilitate this novel task, we first introduce PAP-12K, a large-scale benchmark dataset containing over 1,000 ultra-high-resolution (12k, 11904 x 5952) panoramic images with over 12k carefully annotated QA pairs and affordance masks.
- Furthermore, we propose PAP, a training-free, coarse-to-fine pipeline inspired by the human foveal visual system to tackle the ultra-high resolution and severe distortion inherent in panoramic images.
- In contrast, PAP framework effectively overcomes these obstacles, significantly outperforming state-of-the-art baselines and highlighting the immense potential of panoramic perception for robust embodied intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15558v1)
- [arXiv](https://arxiv.org/abs/2603.15558v1)

---

<a id='2603.15555v1'></a>
## [Learning Latent Proxies for Controllable Single-Image Relighting](https://arxiv.org/abs/2603.15555v1)

**Authors:** Haoze Zheng, Zihao Wang, Xianfeng Wu, Yajing Bai, Yexin Liu, Yun Li, Xiaogang Xu, Harry Yang

**Published:** 2026-03-16

**Categories:** cs.CV

**Abstract:**

Single-image relighting is highly under-constrained: small illumination changes can produce large, nonlinear variations in shading, shadows, and specularities, while geometry and materials remain unobserved. Existing diffusion-based approaches either rely on intrinsic or G-buffer pipelines that require dense and fragile supervision, or operate purely in latent space without physical grounding, making fine-grained control of direction, intensity, and color unreliable. We observe that a full intrinsic decomposition is unnecessary and redundant for accurate relighting. Instead, sparse but physically meaningful cues, indicating where illumination should change and how materials should respond, are sufficient to guide a diffusion model. Based on this insight, we introduce LightCtrl that integrates physical priors at two levels: a few-shot latent proxy encoder that extracts compact material-geometry cues from limited PBR supervision, and a lighting-aware mask that identifies sensitive illumination regions and steers the denoiser toward shading relevant pixels. To compensate for scarce PBR data, we refine the proxy branch using a DPO-based objective that enforces physical consistency in the predicted cues. We also present ScaLight, a large-scale object-level dataset with systematically varied illumination and complete camera-light metadata, enabling physically consistent and controllable training. Across object and scene level benchmarks, our method achieves photometrically faithful relighting with accurate continuous control, surpassing prior diffusion and intrinsic-based baselines, including gains of up to +2.4 dB PSNR and 35% lower RMSE under controlled lighting shifts.

**Analysis:**

### 1. 摘要翻译
单图重光照（Relighting）是一个高度欠约束的问题：微小的照明变化会导致阴影、高光和漫反射的剧烈非线性变化，而几何和材质信息在单张RGB图中往往不可见。现有的基于扩散模型的方法要么依赖于需要高昂且脆弱的监督信息的“内在（intrinsic）”或“G-buffer”流水线，要么完全在隐空间（latent space）操作而缺乏物理约束，导致方向、强度和色彩的细粒度控制不可靠。我们观察到，要实现精确的重光照，全场景的内在分解是不必要且冗余的。相反，仅需少量具有物理意义的线索来指示照明变化和材质响应，就足以引导扩散模型。基于此，我们提出了 **LightCtrl**：它在两个层面集成了物理先验，即一个从有限PBR监督中提取紧凑材质-几何线索的少样本隐式代理编码器（latent proxy encoder），以及一个识别敏感照明区域并引导去噪器关注阴影相关像素的“照明感知掩码（lighting-aware mask）”。为了弥补PBR数据的稀缺，我们利用基于DPO（直接偏好优化）的目标函数对代理分支进行精修，强化了预测线索的物理一致性。此外，我们构建了 **ScaLight**，这是一个包含系统性照明变化和完整相机-光源元数据的大规模物体级数据集。在物体和场景级基准测试中，我们的方法实现了光度准确的重光照和精确的连续控制，在受控照明变化下，PSNR提升高达2.4 dB，RMSE降低了35%，超越了现有的扩散模型和基于内在分解的基线方法。

### 2. 方法动机分析
*   **驱动力**：旨在解决单图重光照中“物理一致性”与“细粒度可控性”难以兼得的问题。
*   **现有痛点**：基于物理的Pipeline（如Intrinsic/G-buffer）监督代价极高且难以泛化；纯隐空间扩散模型（如IC-Light）虽效果好，但由于缺乏物理约束，对光照方向、强度等参数的精确调整难以实现。
*   **研究假设**：无需完整的物理内在分解，通过从稀疏监督中学习“紧凑的隐式几何/材质线索”并结合“空间选择性掩码”，足以约束扩散模型实现精准且物理合理的重光照。

### 3. 方法设计详解
*   **Pipeline流程**：
    1.  **输入与编码**：输入单张RGB图像 $x_s$ 和目标光照条件 $\Delta \ell$。
    2.  **隐式代理提取 (Implicit PBR Encoder)**：利用少样本监督学习一个编码器，输出包含反照率、法线、粗糙度等物理参数的紧凑特征 $t_{phys}$。
    3.  **照明感知掩码预测 (Mask Predictor)**：根据输入图像和光照变化 $\Delta \ell$ 预测一个空间掩码 $M_\theta$，用于动态调节去噪过程中的注意力分布。
    4.  **扩散融合**：将表征图像信息的 $t_{img}$、光照的 $t_{light}$ 以及物理代理 $t_{phys}$ 通过Cross-attention机制注入到U-Net中。
    5.  **DPO精修**：针对编码器部分，通过对比Ground Truth PBR和模型预测的质量差异，利用DPO Loss强制模型纠正 artifacts（如将阴影刻画进材质），提升物理稳定性。
*   **算法核心**：利用 $M_\theta$ 调制 Cross-attention，使模型在照明变化敏感区域（如阴影边界）聚焦于物理引导，在不变区域保持原始材质特征。

### 4. 方法对比分析
*   **本质区别**：从传统的“全场景内在参数化”转向“由稀疏先验引导的扩散生成”，通过引入DPO处理稀疏数据，成功避开了对大规模密集物理监督的需求。
*   **创新贡献**：提出轻量级隐式代理编码器，通过DPO强化物理鲁棒性；构建了ScaLight大数据集。
*   **适用场景**：单物体或室内场景的精确光照编辑、材质一致性要求高的工业场景。

### 5. 实验分析（精简版）
*   **验证方法**：在ScaLight（受控合成）及MIIW（真实场景）上进行量化（RMSE/SSIM/PSNR）及用户主观意向调查。
*   **关键结论**：在物体级重光照中，User Preference Rate达到81.45%，远超基线。
*   **优势**：在连续光照参数变化下表现极佳，光影重建非常稳定。
*   **局限**：对极复杂的长程全局阴影（global cast-shadow）处理存在误差；高频几何细节在强对比环境下偶有平滑现象。

### 6. 实用指南
*   **开源**：论文及数据集ScaLight已公布。
*   **实现建议**：
    *   **DPO的使用**：这是提升物理鲁棒性的关键，确保负样本（rejected sample）取自模型当前时刻的预测，这样最能修正错误偏好。
    *   **超参数**：$\alpha$ 控制亮度差异的容忍度，建议结合具体数据集调整。
*   **迁移**：可迁移至室内设计渲染、产品拍摄后期等需要高保真光源编辑的任务。

### 7. 总结
*   **核心思想**：通过稀疏物理先验引导的生成，实现精准、可控的单图重光照。
*   **速记版pipeline**：1. 提取物理属性隐特征；2. 预测变化敏感区域掩码；3. 将光照、图像、物理特征注入扩散模型；4. 使用对比损失修正物理偏差。

**Key Findings:**

- Based on this insight, we introduce LightCtrl that integrates physical priors at two levels: a few-shot latent proxy encoder that extracts compact material-geometry cues from limited PBR supervision, and a lighting-aware mask that identifies sensitive illumination regions and steers the denoiser toward shading relevant pixels.
- Across object and scene level benchmarks, our method achieves photometrically faithful relighting with accurate continuous control, surpassing prior diffusion and intrinsic-based baselines, including gains of up to +2.4 dB PSNR and 35% lower RMSE under controlled lighting shifts.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.15555v1)
- [arXiv](https://arxiv.org/abs/2603.15555v1)

---

