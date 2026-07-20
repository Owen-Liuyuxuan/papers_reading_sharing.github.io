time: 20260720

# Arxiv Computer Vision Papers - 2026-07-20

## Executive Summary

## 每日计算机视觉论文执行摘要 (2026-07-17)

### 1. 主题与趋势概述

本日10篇论文覆盖了从底层感知到高层推理、从单一模态到多模态融合的广泛方向，呈现出以下核心趋势：

- **多模态深度融合**：视觉正与触觉 (VTLoc)、音频 (Audio-Visual Flamingo)、激光雷达 (CLIFE) 等传感器进行交叉，以解决单一模态的局限性。
- **视频理解向长时、复杂、交互式演进**：多篇工作聚焦超长视频的问答 (Searching Videos as Trees) 与开放域理解 (Audio-Visual Flamingo)，并引入自校正、树搜索等推理机制。
- **机器人感知与操控的具身化**：从灵巧手 (Handroid) 到全身遥操作 (Let the Body Follow)，以及主动学习 (Embodied Active Learning)，强调实体机器人在真实环境中的交互与适应能力。
- **高效与轻量化**：DPNeXt、CLIFE 等工作在保持性能的同时追求边缘设备部署或计算效率，契合实际应用需求。
- **零样本与鲁棒泛化**：PIXIE 针对工业装配缺陷物体进行零样本六自由度位姿估计，体现了对开放世界变化的适应追求。

### 2. 特别重要的创新论文

- **MotionForesight**：开创性地将预训练视频模型重新用于**未来3D场景流预测**，将动态场景理解从“回顾”拓展到“预测”，有望推动自动驾驶、机器人规划等任务。
- **Audio-Visual Flamingo**：将多模态大模型能力扩展到**音频-视觉融合的长视频理解**，实现开放域问答，为视频AI提供更强的感知与推理基础。
- **PIXIE**：针对有装配缺陷的未知物体，提出**零样本、纹理不变**的6D姿态估计框架，对工业自动化中柔性生产具有重要实用价值。
- **Searching Videos as Trees**：提出**自校正智能体**结合树型搜索策略解决长视频定位问答，显著提升复杂推理中的准确性与可解释性。

### 3. 新兴研究方向与技术

- **触觉-视觉点云融合定位** (VTLoc)：将触觉传感器信息注册到视觉点云中，实现精确接触定位，为机器人精细操作提供新传感范式。
- **全身耦合遥操作** (Let the Body Follow)：采用自我中心控制实现人-机器人全身运动映射，使非专业用户也能直观操控机器人，拓展远程操作应用场景。
- **边缘部署的路侧感知** (CLIFE)：针对弱势道路使用者 (VRU)，设计轻量级相机-激光雷达融合框架，推动自动驾驶中基础设施端的实时感知。
- **多任务密集预测的轻量ViT框架** (DPNeXt)：通过多尺度特征融合与高效架构设计，在ViT上实现多个密集预测任务 (如深度、分割) 的并行处理，兼顾精度与效率。
- **有限预算下的具身主动学习** (Embodied Active Learning)：研究机器人在导航与标注预算双重约束下如何主动选择场景进行目标检测学习，是具身智能与少样本学习的交叉点。

### 4. 建议全文阅读的论文

| 论文 | 推荐理由 |
|------|----------|
| **MotionForesight** | 视频模型重用于未来预测，思路新颖，潜力大 |
| **Audio-Visual Flamingo** | 开放域长视频音频视觉理解，代表多模态前沿 |
| **PIXIE** | 工业应用价值显著，零样本鲁棒性方法具参考性 |
| **Searching Videos as Trees** | 自校正推理机制在长视频QA中表现突出，可推广至其他任务 |
| **DPNeXt** | 轻量ViT多任务框架，对资源受限场景的研究者具启发 |

这些论文分别在理论创新、实际应用或方法工程上有突出贡献，值得详细阅读以把握领域最新进展。

---

## Table of Contents

1. [MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction](#2607.16192v1)
2. [Searching Videos as Trees: Self-Correcting Agents for Grounded Long Video QA](#2607.16189v1)
3. [Handroid: Bridging Dexterous Hand and Humanoid](#2607.16187v1)
4. [CLIFE: Camera-LiDAR Fusion Framework for Edge-Deployable Roadside VRU Perception](#2607.16154v1)
5. [VTLoc: Learning-based Tactile Contact Localization in Visual Point Clouds](#2607.16146v1)
6. [Audio-Visual Flamingo: Open Audio-Visual Intelligence for Long and Complex Videos](#2607.16107v1)
7. [Let the Body Follow: Coupled Egocentric Control for Whole-Body Robot Teleoperation](#2607.16095v1)
8. [PIXIE: A Zero-Shot texture-invariant 6D pose estimation framework for unseen objects with assembly defects](#2607.16015v1)
9. [DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction](#2607.16012v1)
10. [Embodied Active Learning under Limited Annotation and Navigation Budget for Object Detection](#2607.15974v1)

---

## Papers

<a id='2607.16192v1'></a>
## [MotionForesight: Re-purposing Video Models for Future 3D Scene-Flow Prediction](https://arxiv.org/abs/2607.16192v1)

**Authors:** Homanga Bharadhwaj, Yash Jangir

**Published:** 2026-07-17

**Categories:** cs.CV

**Abstract:**

Humans can infer how objects are likely to move from passive observation: a cup may be lifted, a drawer may slide, and a lid may rotate shut. Such predictions expose the physical consequences of interaction needed to act in the real world. We study how to learn this anticipation from ordinary monocular videos of human-object interaction. Given a short observed video context, MotionForesight predicts future 3D trajectories for points on the manipulated object. This casts interaction prediction as object-centered 3D motion forecasting without any assumptions on the object properties. Our key insight is that video prediction models already encode rich priors about how objects move during human interactions. We redirect these priors from pixel prediction toward future 3D scene flow. We start from a dense 3D tracker built on a pretrained video model, generate pseudo-ground-truth tracks from complete clips, and train the forecaster using only the observed frames. We replace future RGB and geometry with learned mask latents and train a lightweight adapter to turn the retrospective tracking representation into a forward predictor, while freezing the large video and tracking components. Using just 40k human videos and no auxiliary inputs such as language, MotionForesight generalizes across diverse out-of-distribution objects, environments, viewpoints, and interactions. It also outperforms substantially larger models that use over a million training videos. These results show that we can efficiently re-purpose video priors into explicit geometric forecasts for embodied intelligence. https://motionforesight.github.io/

**Analysis:**

## 论文方法分析与总结：MotionForesight

### 1. 摘要翻译
人类可以通过被动观察推断物体的运动趋势（如提起杯子、抽屉滑动）。这种预判对于理解物理因果关系至关重要。本文研究如何从普通单目人-物交互视频中学习这种预期。给定一段短视频上下文，MotionForesight 预测被操作物体上各点的未来 3D 轨迹。其核心见解是：视频预测模型已内化了物体在交互中如何运动的丰富先验知识。我们将这些先验知识从像素预测重定向到未来 3D 场景流预测。该模型基于预训练视频模型构建密集 3D 追踪器，通过遮蔽未来帧并在仅有的观测帧上训练轻量级适配器，实现了对未来几何形态的显式预测。实验证明，该模型仅需 40K 视频训练，且无需语言或动作标签，即在分布外物体、环境和交互中展现出优越的泛化能力。

### 2. 方法动机分析
*   **驱动力**：在具身智能领域，系统需要理解物体运动的物理因果，而不仅仅是生成视觉效果。
*   **痛点**：现有的视频生成模型往往侧重于生成像素（RGB），这不仅计算量巨大，且生成的视觉内容往往不具备明确的几何一致性（即不保证 3D 轨迹的准确性）。此外，现有的 3D 轨迹预测方法往往过度依赖语言指令或特定的动作标签。
*   **核心直觉**：预训练的大规模视频模型（Video DiT）在学习“生成视频”的过程中，实际上已经隐式编码了物体运动、接触、支点等物理规律，可以直接将其“重定向”为 3D 几何轨迹预测。

### 3. 方法设计详解
*   **流程总结**：
    1.  **预处理**：从原始视频中提取包含交互过程的固定长度片段（22帧），通过 `Segment-Anything` 提取物体掩码，结合 `DepthAnything3` 和相机参数计算出参考系下的密集 3D 点云追踪伪标签。
    2.  **上下文编码**：将观测到的 RGB 帧与点图（Pointmaps）通过 VAE 编码器映射为视觉-几何联合潜变量。
    3.  **未来预测（掩码机制）**：将未来时刻的输入替换为学习到的“掩码潜变量”（Learned Mask Latents）。
    4.  **适配与预测**：通过轻量级的 LoRA 适配器，利用 Video DiT 的冻结先验，将观测序列与查询向量（Query）映射为残差轨迹潜变量。
    5.  **解码**：通过冻结的 Track Decoder 将残差转换为 3D 轨迹坐标。
*   **模型结构**：采用了 **TrackCraft3R** 架构作为基础。Video DiT 被冻结，仅微调一个 32 秩（Rank-32）的 LoRA 适配器及输出头，极大地降低了训练成本。
*   **关键公式**：$\hat{X}_t(q) = X_0(q) + D_{track}(\hat{z}^\Delta_t)(q)$，模型预测的是当前点相对于参考点（初始时刻）的 3D 偏移量，而非绝对位置。

### 4. 方法对比分析
*   **本质区别**：与生成像素的 Video-to-Video 模型不同，它直接输出 metric 3D 轨迹空间，且不需要任何语言或动作标签作为条件。
*   **创新贡献**：提出了一种将“Retrospective Tracking”（回顾式追踪）转化为“Prospective Forecasting”（前瞻式预测）的通用架构，通过遮蔽未来与训练轻量级适配器，实现了视频先验的几何重定向。
*   **适用场景**：适用于机器人操作中的物体运动预测、具身智能的行为预判任务。

### 5. 实验分析
*   **结论**：在 SSv2 数据集和手机实拍（OOD）场景上，MotionForesight 的 ADE 和 FDE 指标均优于使用数百万数据训练的模型（如 MolmoMotion）。
*   **核心优势**：几何显式性更强，具备极强的跨分布泛化能力，即使数据量仅有 40K 也能实现高性能。
*   **局限**：目前是确定性预测（单一输出），无法完全覆盖现实中的多种可能性（多峰分布）。

### 6. 实用指南
*   **实现细节**：关键在于数据预处理（伪标签提取的质量决定上限）。坐标系统一化到最后一帧是非常重要的技巧（减小相机自身运动带来的噪声）。
*   **迁移建议**：如果要在自己的任务中复用，核心在于构建一个可靠的 3D 伪标签追踪 pipeline（如使用 CoTracker/TrackCraft3R），然后冻结 Video DiT 骨干网络进行适配。

### 7. 总结
*   **核心思想**：利用视频生成模型的先验实现零标签的 3D 几何轨迹预测。
*   **速记版 Pipeline**：
    1. 获取视频并计算伪 3D 轨迹；
    2. 将观测帧编码为几何潜变量；
    3. 用掩码向量表示未来未知时刻；
    4. 训练小适配器预测残差轨迹；
    5. 解码得到物体 3D 运动流。

**Key Findings:**

- It also outperforms substantially larger models that use over a million training videos.
- These results show that we can efficiently re-purpose video priors into explicit geometric forecasts for embodied intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16192v1)
- [arXiv](https://arxiv.org/abs/2607.16192v1)

---

<a id='2607.16189v1'></a>
## [Searching Videos as Trees: Self-Correcting Agents for Grounded Long Video QA](https://arxiv.org/abs/2607.16189v1)

**Authors:** Ce Zhang, Ziyang Wang, Yulu Pan, Oluwatumininu Oguntola, Pranav Wagh, Qiyu Wu, Hiromi Wakaki, Mohit Bansal, Gedas Bertasius

**Published:** 2026-07-17

**Categories:** cs.CV

**Abstract:**

Grounded long-video question answering (Grounded LVQA) requires answering a question about a long video while localizing the short evidence interval that supports the answer. Recent agentic methods frame this task as multi-turn exploration with a single crop_video(start, end) action, which supports coarse-to-fine narrowing but provides no primitive for fine-to-coarse backtracking. As a result, these agents typically converge prematurely and cannot recover from an early mistake. We propose VideoTreeSearch (VTS), a framework that casts grounded LVQA as iterative self-correcting search over an adaptive temporal tree. VTS constructs a non-uniform tree from visual scene boundaries so that each node corresponds to a semantically coherent segment, and trains an agent to navigate the tree through four discrete operations: zoom_in, zoom_out, shift, and answer. These operations expose backtracking and recovery as explicit, learnable primitives rather than implicit behaviors. To train this navigation, we introduce a trajectory synthesis pipeline that produces multi-step paths through the tree, including deliberate detours into incorrect branches followed by recovery. We use these trajectories for supervised fine-tuning, followed by reinforcement learning with grounding and answer-accuracy rewards. On three Grounded LVQA benchmarks (CG-Bench, Haystack-LVBench, Haystack-Ego4D), VTS outperforms the strongest prior agentic methods by +12.5 mIoU on CG-Bench and +7.4 T-F1 on Haystack-Ego4D. The learned policy also transfers to general long-video QA, surpassing all prior agentic baselines on Video-MME, MLVU, and LVBench by up to +7.1 accuracy points. Ablations confirm that self-correcting hierarchical search is the central mechanism behind these gains: removing either adaptive descent or explicit backtracking substantially degrades performance. Code is available at https://github.com/CeeZh/VTS.

**Analysis:**

### 1. 摘要翻译
接地长视频问答（Grounded LVQA）要求在回答问题的同时定位支持答案的短视频区间。现有的代理（Agentic）方法多采用单步`crop_video(start, end)`操作进行多轮探索，这种方式虽能实现由粗到细的缩窄，但缺乏从细到粗的“回溯”原语。因此，此类智能体往往因过早收敛而无法从早期错误中恢复。我们提出了**VideoTreeSearch (VTS)**，该框架将接地LVQA建模为自适应时间树上的迭代自纠正搜索。VTS基于视觉场景边界构建非均匀树，使每个节点对应语义连贯的片段；并训练智能体通过`zoom_in`（进入子片段）、`zoom_out`（回溯父节点）、`shift`（横向移动到兄弟节点）和`answer`（提交答案）四种离散操作进行导航。这些操作将定位与回溯转化为显式的可学习原语。我们引入轨迹合成流水线，通过包含故意进入错误分支及随后恢复的路径来训练智能体。实验表明，VTS在多个Grounded LVQA基准测试中大幅超越了现有最优方法。消融实验证实，自纠正层级搜索是性能提升的核心机制。

### 2. 方法动机分析
*   **驱动力**：解决长视频中由于错误定位导致无法纠正的顽疾，让模型具备像人类一样“试错、复盘、重新搜索”的显式能力。
*   **现有方法痛点**：
    *   **动作不对称**：连续裁剪（crop）只能向下钻取，无法向上回溯，一旦剪错了区间便无法挽回。
    *   **搜索空间平坦且无组织**：缺乏基于语义的层次化结构，导致模型在处理长视频时难以获得一致的训练信号。
*   **研究假设**：通过将视频组织成基于语义边界的树，并将“回溯”动作作为显式学习目标，模型能实现更精准的长视频定位与纠正。

### 3. 方法设计详解
*   **自适应时间树构建**：
    *   **逻辑**：根据CLIP特征的余弦距离变化（$\delta_i = 1 - \cos(e_i, e_{i+1})$）确定场景切换边界。
    *   **细节**：采用自适应阈值 $\tau = \text{mean}(\delta) + k \cdot \text{std}(\delta)$ 进行切分。递归构建树，直到段落长度低于64秒。这种“懒加载”式构建确保了搜索空间与任务相关性高。
*   **树接地动作空间**：
    *   **四元原语**：`zoom_in(c)`、`zoom_out()`、`shift(s)`、`answer`。将搜索任务彻底从连续坐标回归转化为图搜索决策。
*   **训练策略**：
    *   **轨迹合成（Trajectory Synthesis）**：合成包含“进入错误分支 -> 识别错误 -> 回溯 -> 找到正确路径”的轨迹，强制模型学习反思行为。
    *   **选择性监督**：仅对正确动作（正确的zoom、必要的backtrack和shift）计算损失，掩码掉导致错误的无效路径。
    *   **强化学习**：在SFT基础上，利用任务成功奖励（Rfmt, RIoU, Racc）进行微调。

### 4. 方法对比分析
*   **本质区别**：从“基于回归的连续裁剪（黑盒搜索）”转向“基于语义结构的离散动作决策（显式搜索）”。
*   **创新贡献**：引入了显式的`zoom_out`和`shift`作为可训练的回溯原语，实现了真正的自纠正闭环。
*   **适用场景**：极长视频（分钟至小时级）中的细粒度定位与问答任务。

### 5. 实验分析
*   **验证方法**：在CG-Bench、Haystack-LVBench和Haystack-Ego4D三个基准上对比了多种智能体。
*   **关键结果**：在CG-Bench上，VTS比最强基线LongVT高出+12.5 mIoU；在Haystack-Ego4D上，T-F1提升+7.4。
*   **核心优势**：显式回溯能力使得智能体在约60%的轨迹中能成功进行自我纠正。
*   **局限**：动作空间目前只支持定位单个证据区间，无法解决证据分布在多个分散区间的问题。

### 6. 实用指南
*   **开源情况**：代码已开源至 https://github.com/CeeZh/VTS。
*   **实现细节**：
    *   训练过程需使用ms-swift框架，分SFT和RL两个阶段。
    *   `k=1.5`是自适应切分的关键超参数。
    *   建议数据预处理中过滤掉gt超过视频30%长度的“覆盖范围过大”样本，以增强模型对局部证据的捕捉能力。

### 7. 总结
*   **核心思想**：通过语义树与显式导航原语，赋予视频QA智能体回溯纠错能力。
*   **速记版pipeline**：
    1. 计算视觉差异，按语义逻辑构建时间树。
    2. 将长视频定位转化为层级化的树遍历动作。
    3. 合成“犯错-回溯-寻找”路径，进行监督微调。
    4. 通过奖励驱动强化学习，优化导航策略。

**Key Findings:**

- We propose VideoTreeSearch (VTS), a framework that casts grounded LVQA as iterative self-correcting search over an adaptive temporal tree.
- To train this navigation, we introduce a trajectory synthesis pipeline that produces multi-step paths through the tree, including deliberate detours into incorrect branches followed by recovery.
- On three Grounded LVQA benchmarks (CG-Bench, Haystack-LVBench, Haystack-Ego4D), VTS outperforms the strongest prior agentic methods by +12.5 mIoU on CG-Bench and +7.4 T-F1 on Haystack-Ego4D.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16189v1)
- [arXiv](https://arxiv.org/abs/2607.16189v1)

---

<a id='2607.16187v1'></a>
## [Handroid: Bridging Dexterous Hand and Humanoid](https://arxiv.org/abs/2607.16187v1)

**Authors:** Ruogu Li, Chenyang Ma, Sikai Li, Zhenyu Wei, Yunchao Yao, Haochen Shi, C. Karen Liu, Shuran Song, Mingyu Ding

**Published:** 2026-07-17

**Categories:** cs.RO

**Abstract:**

Dexterous hands and humanoid robots are typically developed as distinct embodiments: the former enable contact-rich manipulation at the object scale, whereas the latter provide mobility and whole-body interaction in human-centered environments. We introduce \textbf{Handroid}, a desktop-scale dual-embodiment robot that integrates both capabilities within a single reconfigurable platform. Handroid reuses one 27-DoF electromechanical body as either a dexterous hand or a desktop humanoid, measuring 0.33 m in height and 2.05 kg in weight. In the dexterous hand embodiment, 20 DoFs form an anthropomorphic hand closely matching the kinematic structure of the human hand. In the humanoid embodiment, the same articulated modules are reconfigured into a humanoid with a head, arms, and legs, including a 12-DoF lower-limb structure for locomotion and whole-body motion. Handroid further provides a unified control and learning framework supporting hand teleoperation, dexterous grasping, in-hand manipulation, humanoid locomotion, gait generation, and interactive motion authoring. We validate the platform through real-world dexterous manipulation, reinforcement-learning-based locomotion, keyframe motion deployment, and a long-horizon task involving embodiment reconfiguration, locomotion, docking, and dexterous pick-and-place. These results position Handroid as a compact and reproducible platform for advancing morphology-reconfigurable robotics and cross-embodiment robot learning.

**Analysis:**

### 1. 摘要翻译
灵巧手和人形机器人通常作为独立的实体开发：前者实现物体尺度的接触丰富操作，而后者在以人为中心的环境中提供移动性和全身交互。我们介绍了**Handroid**，一个桌面级的双实体机器人，在一个可重构的单一平台中集成了这两种能力。Handroid复用同一个27自由度（DoF）机电本体，既可作为灵巧手，也可作为桌面人形机器人，高0.33米，重2.05公斤。在灵巧手实体模式下，20个自由度构成了与人手运动结构高度匹配的人形手；在人形机器人实体模式下，相同的铰接模块被重新配置为具有头部、手臂和腿部的人形，包括用于移动和全身运动的12自由度下肢结构。Handroid还提供了一个统一的控制和学习框架，支持手部遥控、灵巧抓取、手内操作、人形移动、步态生成和交互式运动创作。我们通过真实世界的灵巧操作、基于强化学习的移动、关键帧运动部署以及涉及实体配置重构、移动、对接和灵巧取放的长程任务验证了该平台。这些结果使Handroid成为推动形态可重构机器人和跨实体机器人学习的紧凑且可复现的平台。

### 2. 方法动机分析
*   **驱动力**：核心动机在于打破灵巧手与人形机器人形态的固化边界，证明“形态复用”在硬件上是可行的，旨在通过一套机电系统实现接触丰富的操作与全身移动能力的统一。
*   **现有痛点**：当前机器人系统呈现“二分法”，灵巧手多固定于机械臂，缺乏移动性；人形机器人虽具备移动性，但手部灵巧度不足。这种分离导致了机器人设计和学习的割裂。
*   **研究假设**：机器人实体本质上是连接在中心主体上的铰接运动链集合。通过共享结构件、感测和驱动系统，可以利用可重构的硬件架构，实现在单一物理系统上统一 manipulation（操作）和 mobility（移动）的学习。

### 3. 方法设计详解
*   **硬件架构与可重构设计**：Handroid采用27个DoF的模块化设计。通过集成在Base（模块VI）中的**齿轮齿条传动机构**，实现重构：当从人形转变为灵巧手时，对应的肢体模块（手臂）通过线性导轨平移至指定位置，变为手指（食指和小指）。该设计无需更换硬件，仅需位置调整。
*   **电气与控制系统**：基于ESP32-S3的统一主控板，集成IMU、功率管理（支持140W PD充电）和TTL总线驱动器。
*   **统一学习栈**：
    1.  **遥控与模仿**：基于Apple Vision Pro捕捉手部关键点，通过AnyTeleop框架完成手部动作映射。
    2.  **灵巧操作**：使用基于扩散策略（Diffusion Policy）的策略网络，以点云和本体感觉作为输入，预测动作块。
    3.  **人形移动**：采用**双层策略**，一是基于ZMP的参考轨迹追踪（Tracking Policy），二是参考自由（Reference-free）的RL速度控制策略（Velocity Policy），根据指令速度输出关节目标。
    4.  **运动编辑**：利用Viser Keyframe Editor，通过关键帧插值快速生成运动，并可导出作为参考轨迹或直接执行。

### 4. 方法对比分析
*   **本质区别**：不同于传统的固定形态机器人，Handroid是首个将灵巧手与人形形态在桌面尺度上进行物理可重构的系统。
*   **创新贡献**：提出了一种基于机电模块映射的morphology-reconfigurable机制；实现了一套通用的软硬件栈，使得在灵巧手上的操作策略与人形上的运动策略可以在同一底层逻辑下开发。
*   **适用场景**：极佳的桌面级机器人实验室科研平台，特别适合研究“跨形态迁移学习”和“复杂长程作业任务”。

### 5. 实验分析
*   **关键结论**：在灵巧抓取任务中取得72%的成功率；在长程任务（移动、避障、自主重构、自主对接、精密取放）中表现出稳定的跨形态协作能力。
*   **优势**：硬件极其紧凑，极大降低了研究复杂机器人行为的硬件成本和维护门槛；实现了感知-驱动的完全统一。
*   **局限**：目前的重构过程需要人工干预或部分外部指令，且由于电缆存在，电池寿命和移动范围受限，未来需向完全无线化发展。

### 6. 实用指南
*   **开源情况**：作者提供了开源支持（访问handroid.org获取详情）。
*   **迁移建议**：其核心思想“将不同形态抽象为关节链”可迁移至大尺寸人形机器人设计。若要在自己的机器人上复用，应重点参考其基于rack-and-pinion（齿轮齿条）的机械重构机制和基于统一position-target接口的控制逻辑。

### 7. 总结
*   **核心思想**：通过形态可重构硬件实现单一平台对灵巧操作与全身移动的统一。
*   **速记版pipeline**：
    1.  **形态切换**：通过齿轮齿条传动进行物理重构。
    2.  **统一感控**：利用通用通信协议与本体感测接口。
    3.  **策略学习**：训练任务导向的行为策略（Diffusion/RL）。
    4.  **部署执行**：通过关键帧或运动策略实时驱动电机。

**Key Findings:**

- We introduce \textbf{Handroid}, a desktop-scale dual-embodiment robot that integrates both capabilities within a single reconfigurable platform.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16187v1)
- [arXiv](https://arxiv.org/abs/2607.16187v1)

---

<a id='2607.16154v1'></a>
## [CLIFE: Camera-LiDAR Fusion Framework for Edge-Deployable Roadside VRU Perception](https://arxiv.org/abs/2607.16154v1)

**Authors:** Tam Bang, Hoang H. Nguyen, Lei Cheng, Lihao Guo, Siyang Cao, Hussam Abubakr, Tianya Zhang, Austin Harris, Mina Sartipi

**Published:** 2026-07-17

**Categories:** cs.CV, eess.SY

**Abstract:**

Reliable roadside perception of vulnerable road users (VRUs) remains challenging under occlusions, variable lighting, and diverse weather conditions, particularly under strict edge-computing and latency constraints. Existing multi-sensor fusion systems rely on cloud or server-grade infrastructure, creating a deployment gap at real-world intersections. We present CLIFE, an edge-native camera-LiDAR fusion framework that integrates targetless online calibration and lightweight late-fusion tracking entirely on a single embedded device, without cloud offloading. CLIFE adaptively refines camera-LiDAR alignment on demand and performs multi-sensor fusion and track association with O(N log N) per-frame cost. We deploy CLIFE across 12 signalized intersections in Chattanooga and conduct an in-depth evaluation at a representative intersection using synchronized camera-LiDAR data that spans diverse daytime, nighttime, and weather conditions. Our experiments demonstrate that the fusion architecture substantially enhances the perceptual range and robustness of the individual sensors under varied environmental and traffic conditions. The late-fusion core operates at 53.2 FPS on the Jetson AGX Thor, ensuring high throughput for real-time intersection-scale applications. By centering perception at the edge, CLIFE provides a deployable foundation for downstream safety applications, while reducing bandwidth and calibration overhead for agencies operating multi-intersection corridors.

**Analysis:**

### 1. 摘要翻译
可靠的道路脆弱使用者（VRU）感知在遮挡、多变光照和复杂天气条件下仍面临挑战，特别是在严格的边缘计算和延迟限制下。现有的多传感器融合系统往往依赖云端或服务器级基础设施，导致其难以在现实交通路口部署。本文提出了 **CLIFE**，一个边缘原生的相机-激光雷达（Camera-LiDAR）融合框架，它在单台嵌入式设备上实现了无标定在线标定和轻量级延迟融合跟踪，无需云端卸载。CLIFE 可按需自适应优化传感器对齐，并以 $O(N \log N)$ 的单帧复杂度执行多传感器融合与关联。我们在查塔努加市的12个信号交叉路口部署了CLIFE，并在代表性路口进行了多场景评估。实验表明，该融合架构显著增强了传感器在多变环境下的感知范围和鲁棒性。其延迟融合核心在 Jetson AGX Thor 上运行速度达 53.2 FPS，为实时交叉路口级应用提供了可部署的基石，同时降低了运维带宽和标定负担。

### 2. 方法动机分析
*   **驱动力**：解决现实交通路口中，因传感器遮挡、环境干扰及边缘设备算力有限而导致的VRU（行人、自行车等）感知不可靠问题。
*   **现有方法痛点**：
    *   **架构依赖**：现有方案多依赖服务器级后端，部署成本高且带宽压力大。
    *   **维护困难**：传统外参标定依赖人工、标定板或车道封闭，难以应对环境引起的安装漂移。
    *   **单模态局限**：相机易受光照干扰，激光雷达缺乏细粒度语义区分。
*   **研究假设**：通过边缘原生的在线目标无依赖（Targetless）标定与轻量级关联策略，能在单台嵌入式计算单元上实现与服务器端相媲美的实时鲁棒感知。

### 3. 方法设计详解
CLIFE 包含两个阶段，运行于 NVIDIA Jetson AGX Thor：
*   **阶段1（Camera-LiDAR 标定）**：采用无标定在线 routine，通过匹配跨模态检测结果（基于空间、外观和语义特征），估计地面单应性矩阵 $H \in \mathbb{R}^{3 \times 3}$，将激光雷达点云投影至图像平面。该模块为“按需触发”，当检测到 mount 漂移时自动运行。
*   **阶段2（延迟融合跟踪）**：
    *   **多传感器融合（Algorithm 1）**：通过已知的 $H$ 将 LiDAR 检测投影到图像，利用 KD-tree 在 2D 平面搜索空间半径 $r_s$ 内的匹配对象，并执行贪婪算法进行一对一配对。
    *   **属性融合**：匹配后的对象，其分类和置信度优先取自相机（语义强），位置信息取自相机，而速度向量取自 LiDAR（空间精度高）。
    *   **多传感器跟踪**：利用 FIFO 缓冲区累计最近 $K$ 帧信息。即使单一传感器失效（如遮挡），系统也能维持跟踪 ID，直到满足终止条件（超过 $\tau_{miss}$ 帧或双传感器同时丢失）。

### 4. 方法对比分析
*   **本质区别**：与传统“紧耦合”或“服务器端处理”不同，CLIFE 是完全的“边缘侧-延迟融合”方案，实现了标定与感知的全自动闭环。
*   **创新贡献**：提出了一种无需人工干预的在线标定机制，并优化了单设备处理多路流的能力，实现了从单点感知到交叉路口级感知的平滑跨越。
*   **适用场景**：实时性要求高、需长期无人值守的智慧路口基础设施。

### 5. 实验分析（精简版）
*   **验证方法**：在查塔努加市12个路口部署，测试了晴天、阴天、小雨等条件下的感知指标。
*   **关键结果**：在晴天条件下，相比相机单模态，融合方案将 MOTA 从 67.7 提升至 78.6；53.2 FPS 的处理速度证明了其在边缘计算设备上的高吞吐能力。
*   **优势**：在低光照、雨天及遮挡场景下表现出极强的鲁棒性；支持按需重标定，维护成本极低。
*   **局限**：对初始的单应性矩阵校准仍有依赖，且上游模块（如检测器）的误差会随延迟融合传播。

### 6. 实用指南
*   **实现细节**：关键在于使用 TensorRT 对 YOLOv11 等神经网络进行编译，并避免全帧 warp，仅在检测坐标空间进行坐标转换。
*   **迁移建议**：该架构模块解耦清晰，感知模型（YOLOv11）、标定模块（CalibRefine）和跟踪模块均可独立替换。对于其他任务（如V2X），可直接复用该“标定+延迟融合”框架。

### 7. 总结
*   **核心思想**：边缘侧自适应标定与轻量级延迟融合驱动的鲁棒VRU感知。
*   **速记版pipeline**：
    1.  同步传感器数据流。
    2.  触发无标定 routine 生成转换矩阵。
    3.  融合投影 LiDAR 检测至图像空间。
    4.  基于贪婪匹配与加权属性合并生成结果。
    5.  多帧 buffer 维护长时一致性 ID。

**Key Findings:**

- We present CLIFE, an edge-native camera-LiDAR fusion framework that integrates targetless online calibration and lightweight late-fusion tracking entirely on a single embedded device, without cloud offloading.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16154v1)
- [arXiv](https://arxiv.org/abs/2607.16154v1)

---

<a id='2607.16146v1'></a>
## [VTLoc: Learning-based Tactile Contact Localization in Visual Point Clouds](https://arxiv.org/abs/2607.16146v1)

**Authors:** Zhiyuan Wu, Zhuo Chen, Shan Luo

**Published:** 2026-07-17

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision and touch are complementary modalities essential for robotic perception and manipulation. While vision provides global object context, touch offers precise local information at contact points. Integrating these modalities for contact localization, i.e., predicting the location of touch on an object's surface, poses significant challenges due to the need for accurate spatial alignment between tactile data and visual geometry. To address this challenge, we propose VTLoc, a novel visual-tactile framework that localizes contact points from tactile readings using a 3D point cloud as visual input. VTLoc introduces two key components: a geometric multi-modal alignment module, which reconstructs a pseudo-point cloud from fused visual-tactile features and aligns it with the visual point cloud to enforce spatial consistencies across modalities; and an iterative localizing updater, which iteratively refines the predicted contact location using fused visual-tactile features. Evaluated on a new benchmark of 100 real-world objects, VTLoc improves single-touch contact localization by reducing local-to-global correspondence ambiguity.

**Analysis:**

### 1. 摘要翻译
视觉和触觉是机器人感知与操作中互补的核心模态。虽然视觉提供全局上下文，但触觉在接触点提供精确的局部信息。如何将两者有效融合以实现接触点定位（预测物体表面接触位置）存在巨大挑战，主要源于触觉数据与视觉几何之间精准的空间对齐难题。为此，我们提出了VTLoc，一种利用3D点云作为视觉输入，通过触觉感知进行接触点定位的视觉-触觉框架。VTLoc引入了两个关键模块：一是几何多模态对齐（GMA）模块，它通过融合视觉-触觉特征重建伪点云，并将其与原始视觉点云对齐，从而强化模态间的空间一致性；二是迭代定位更新器（ILU），通过融合特征对预测的接触位置进行迭代细化。在包含100个真实物体的新数据集上，VTLoc通过减少局部与全局对应的歧义性，显著提升了单次触觉接触定位的性能。

---

### 2. 方法动机分析
- **核心动机**：人类感知物体时并非一次性定型，而是通过“感知-重构-验证-精细化”的闭环过程。作者希望在模型中复现这一认知过程，以解决触觉定位中的空间不确定性。
- **现有痛点**：以往方法多采用简单的特征串联（Concatenation），缺乏对空间物理一致性的显式约束，且依赖于大规模离线采样，缺乏泛化能力。
- **核心直觉**：如果模型能通过视觉和触觉特征显式重构出物体的几何信息（Pseudo-point cloud），就能迫使编码器学到具有物理意义的空间表征。

---

### 3. 方法设计详解
**处理流程：**
1. **编码阶段**：利用ResNet-18处理触觉图像（$F^t$），利用PointNet++处理3D点云（$F^p$）。
2. **GMA模块（对齐器）**：将$F^t$和$F^p$融合，通过解码器生成一个“伪点云”($P^d$)。通过计算$P^d$与原点云$P$的Chamfer Distance（倒角距离），强制网络学习对齐的几何语义。
3. **ILU模块（精细化器）**：采用GRU作为递归结构，将初步接触预测值 $\hat{c}_0$ 作为初始状态，在迭代中不断计算补偿量 $\Delta \hat{c}$，逐步向真实的接触点收敛。
4. **概率生成**：最终预测结果作为基准，在离散化的候选点集中进行欧氏距离匹配，并通过加权距离生成3D热力图。

**关键公式解析：**
- **Chamfer Distance (Eq. 17)**：衡量重构伪点云与真实物体的几何偏离度，作为训练目标，强制特征表示具备几何一致性。
- **迭代更新 (Eq. 5)**：$\hat{c}_{k+1} = \hat{c}_k + \Delta \hat{c}$，这是模仿人类微调触碰位置的过程，防止模型陷入局部最优。

---

### 4. 方法对比分析
- **本质区别**：从“特征层面融合”转变为“物理空间几何对齐”。
- **创新贡献**：引入了基于几何重构的约束，使模型不仅在学匹配，还在学空间解构。ILU设计类似于视觉任务中的光流迭代优化，对触觉领域是创新。
- **适用场景**：适用于RGB-D相机可获取物体几何点云的工业机器人操作环境。

---

### 5. 实验分析
- **验证方法**：基于ObjectFolder Real数据集构建了100个物体的基准，分为“非均匀”和“均匀”几何曲面。
- **关键结论**：在非均匀表面（有明显特征），VTLoc性能提升极显著（ND指标提升近15%）。但在均匀表面（对称性强），几何法线信息的引入有时反而是噪声，需关注几何语义一致性。
- **主要优势**：不仅定位精度高，还通过3D热力图提供了极强的可解释性。

---

### 6. 实用指南
- **开源情况**：代码已开源至 [https://georgewuzy.github.io/vtloc-website/](https://georgewuzy.github.io/vtloc-website/)。
- **实现细节**：$\lambda$ 取1；迭代次数$N=16$是性能与效率的最佳平衡点。数据预处理需特别注意法线（Normal）的计算质量，若物体过于平滑，法线信息会失效。
- **迁移可能**：可直接迁移至机器人手眼协调或柔性物体操作，只需替换对应的触觉传感器编码器（如处理MEMS阵列数据）。

---

### 7. 总结
- **核心思想**：通过几何对齐与迭代回归模拟人类精细定位触点的认知过程。
- **速记版pipeline**：
    1. 视觉与触觉特征融合编码；
    2. 重构伪点云约束几何一致性；
    3. GRU迭代修正接触点坐标；
    4. 匹配生成3D概率分布。

**Key Findings:**

- To address this challenge, we propose VTLoc, a novel visual-tactile framework that localizes contact points from tactile readings using a 3D point cloud as visual input.
- Evaluated on a new benchmark of 100 real-world objects, VTLoc improves single-touch contact localization by reducing local-to-global correspondence ambiguity.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16146v1)
- [arXiv](https://arxiv.org/abs/2607.16146v1)

---

<a id='2607.16107v1'></a>
## [Audio-Visual Flamingo: Open Audio-Visual Intelligence for Long and Complex Videos](https://arxiv.org/abs/2607.16107v1)

**Authors:** Sreyan Ghosh, Arushi Goel, Kaousheik Jayakumar, Lasha Koroshinadze, Nishit Anand, Siddharth Gururani, Hanrong Ye, Pritam Biswas, Yuanhang Su, Ehsan Hosseini-Asl, Sang-gil Lee, Zhifeng Kong, Jaehyeon Kim, Sungwon Kim, S Sakshi, Ramani Duraiswami, Dinesh Manocha, Andrew Tao, Mohammad Shoeybi, Bryan Catanzaro, Ming-Yu Liu, Wei Ping

**Published:** 2026-07-17

**Categories:** eess.AS, cs.CV

**Abstract:**

We present Audio-Visual Flamingo (AV-Flamingo), a fully open state-of-the-art audio-visual large language model (AV-LLM) for joint understanding and reasoning over audio, images, and long-form videos. Unlike prior AV-LLMs that primarily focus on short clips, AV-Flamingo is designed for understanding and reasoning over long and complex real-world (audio-visual) videos. To support this, we make three key contributions: (i) Audio-Visual-Skills, a large-scale collection of real-world videos with ~7M caption and question-answer training instances designed to emphasize temporal, compositional, and cross-modal audio-visual reasoning; (ii) a novel three-stage curriculum that progressively trains the model from short-range perception to long-horizon multi-event reasoning; and (iii) Temporal Audio-Visual Interleaved Chain-of-Thought, a reasoning framework that explicitly grounds intermediate reasoning steps to timestamps in long audio-visual streams, improving temporal alignment and interpretability. Extensive experiments across 15+ audio-visual, omni-modal, audio, and vision benchmarks show that AV-Flamingo outperforms similarly sized open models by clear margins and remains highly competitive with, and in some cases surpasses, much larger open-weight and closed models, particularly on long and complex real-world audio-visual understanding and reasoning tasks. Beyond benchmark performance, AV-Flamingo exhibits strong real-world utility and transfers well to unseen tasks, highlighting its robustness and generalization ability.

**Analysis:**

### 1. 摘要翻译
我们提出了 Nemotron-Labs-Audio-Visual Flamingo (AV-Flamingo)，这是一个完全开源的先进音频-视觉大语言模型 (AV-LLM)，旨在实现对音频、图像和长视频的联合理解与推理。与主要关注短视频的现有 AV-LLM 不同，AV-Flamingo 专门为长且复杂的现实视频理解与推理而设计。为此，我们做出了三项主要贡献：(i) **Audio-Visual-Skills**，一个包含约 700 万条标注及问答实例的大规模真实视频数据集，旨在强调时间、组成及跨模态的音视频推理；(ii) **三阶段课程训练**，使模型从短程感知逐步进化到长跨度多事件推理；(iii) **时间音频-视觉交织思维链 (TAVIT)**，一种将中间推理步骤显式定位于长音视频流时间戳的推理框架，提升了时间对齐能力与可解释性。在 15 个以上基准测试中的实验表明，AV-Flamingo 在长视频理解任务中显著超越了同等规模的开源模型，并在某些情况下优于更大的闭源模型。

---

### 2. 方法动机分析
*   **驱动力**：人类对世界的感知是连续的音视频流，但现有视频理解模型往往将音频和视频分开处理，或仅能处理短片段，无法应对电影、讲座等长视频中的复杂推理。
*   **现有痛点**：现有 AV-LLM 缺乏高质量配对的音视频训练数据；模型存在“视觉偏见”，深层网络倾向于忽略潜在的音频特征；缺乏对长视频时间维度的显式建模。
*   **研究假设**：通过显式的跨模态对齐（时间交织）和时间戳标注的思维链（Chain-of-Thought），模型可以克服长期依赖带来的性能退化，实现精准的长视频多模态推理。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入编码**：SigLip 提取视觉特征，AF-Whisper（滑动窗口机制）提取长音频特征。
    2.  **跨模态对齐**：通过动态 S2 模块进行压缩，利用时间轴上的“时间交织”技术，将视频和音频块交替排列，以便 LLM 处理。
    3.  **时间标记**：注入“约束旋转时间嵌入 (CRTE)”，让模型感知绝对时间位置。
    4.  **推理与输出**：Qwen2.5-7B 作为骨干网，支持基于 GRPO 的强化学习，以生成带有时间戳的思维链（TAVIT）。
*   **算法解释**：GRPO (Group Relative Policy Optimization) 是一种无需显式评价函数的强化学习策略，通过比较同一输入下的多个候选输出的奖励均值来估计优势，有效降低了训练对计算资源的依赖，并提升了推理逻辑的严谨性。

---

### 4. 方法对比分析
*   **本质区别**：引入了显式的音视频时间戳 grounded 机制。不同于普通模型“先看后说”，AV-Flamingo 要求模型在思维链中明确指出“发生了什么事”以及“发生在什么时间”。
*   **创新贡献**：提出 **AV-Skills** 训练集，填补了长视频逻辑推理数据的空白；提出了 **TAVIT**，实现了音视频与文本的深度逻辑映射。
*   **适用场景**：极长视频的理解、复杂事件序列重组、多模态逻辑推断。

---

### 5. 实验分析
*   **验证方法**：在 WorldSense、DailyOmni、Video-MME 等 15 个基准测试上进行评估。
*   **关键结论**：在长视频推理指标上（如 Video-MME），AV-Flamingo (AVF-Think) 表现最优，显著优于传统的 NVILA 和 OmniVinci 架构。
*   **优势**：在长距离依赖推理和复杂音视频事件对齐上具有显著的鲁棒性。
*   **局限**：数据多来自公开互联网，存在潜在偏见；处理极高密度、超长跨度信息时仍有一定挑战。

---

### 6. 实用指南
*   **开源情况**：已开源模型权重、代码及相关训练技巧。
*   **实现要点**：使用混合序列并行 (Hybrid Sequence Parallelism) 配合 Ulysses 和 Ring-Attention，以解决长上下文下的内存瓶颈；在训练时必须包含音视频配对的 QA 数据以强化逻辑对齐。
*   **迁移建议**：可将 TAVIT 推理框架迁移至其他长视频分析任务（如监控视频分析、医疗手术视频记录），只需按照文中提到的 Schema 构建 metadata。

---

### 7. 总结
*   **核心思想**：通过时间轴上的音视频交织与带时间戳的思维链，实现长视频逻辑推理。
*   **速记版pipeline**：
    1. 视频音频多模态联合编码；
    2. 时间轴交织对齐与时间嵌入；
    3. LLM 骨干网长序列推理；
    4. 强化学习引导思维链（带时间戳）输出。

**Key Findings:**

- We present Audio-Visual Flamingo (AV-Flamingo), a fully open state-of-the-art audio-visual large language model (AV-LLM) for joint understanding and reasoning over audio, images, and long-form videos.
- To support this, we make three key contributions: (i) Audio-Visual-Skills, a large-scale collection of real-world videos with ~7M caption and question-answer training instances designed to emphasize temporal, compositional, and cross-modal audio-visual reasoning; (ii) a novel three-stage curriculum that progressively trains the model from short-range perception to long-horizon multi-event reasoning; and (iii) Temporal Audio-Visual Interleaved Chain-of-Thought, a reasoning framework that explicitly grounds intermediate reasoning steps to timestamps in long audio-visual streams, improving temporal alignment and interpretability.
- Extensive experiments across 15+ audio-visual, omni-modal, audio, and vision benchmarks show that AV-Flamingo outperforms similarly sized open models by clear margins and remains highly competitive with, and in some cases surpasses, much larger open-weight and closed models, particularly on long and complex real-world audio-visual understanding and reasoning tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16107v1)
- [arXiv](https://arxiv.org/abs/2607.16107v1)

---

<a id='2607.16095v1'></a>
## [Let the Body Follow: Coupled Egocentric Control for Whole-Body Robot Teleoperation](https://arxiv.org/abs/2607.16095v1)

**Authors:** Tsung-Chi Lin, Yichen Xie, Chien-Ming Huang

**Published:** 2026-07-17

**Categories:** cs.RO

**Abstract:**

Whole-body teleoperation requires users to coordinate perception, manipulation, posture, and mobility across multiple robot components. This coordination is difficult because users must simultaneously control the robot's head, arms, torso, and base while maintaining task awareness and avoiding kinematic or environmental constraints. In this paper, we propose coupled egocentric control, a body-following teleoperation approach in which the robot's torso and base automatically respond to the operator's head and arm motions. Rather than requiring explicit touchpad commands for every torso or base adjustment, the system lets users focus on gaze and hand control: head pitch adjusts torso height, head yaw drives base rotation, end-effector height adjusts torso motion, and end-effector workspace boundaries trigger base translation. We evaluate this approach in a user study on whole-body teleoperation of a TIAGo mobile manipulator for home-care-inspired tasks. Compared with a baseline hybrid interface, coupled egocentric control improves object manipulation efficiency, reduces button-based control effort and arm singularities, lowers mental demand and overall workload, and increases ease of use, ease of learning, confidence, and user preference for torso and base control.

**Analysis:**

### 1. 摘要翻译
全身遥操作（Whole-body teleoperation）需要用户在多个机器人组件（感知、操作、姿态、移动）之间保持协调。这种协调由于用户需在保持任务感知的同时控制头部、手臂、躯干和底盘而变得十分困难。在本文中，我们提出了“耦合自我中心控制”（coupled egocentric control），这是一种身体跟随式遥操作方法，使机器人的躯干和底盘能够自动响应操作员的头部和手臂动作。该系统无需针对躯干或底盘调整进行显式的触摸板命令，而是让用户专注于注视点和手部控制：头部俯仰调整躯干高度，头部偏航驱动底盘旋转，末端执行器高度调整躯干运动，末端执行器的工作空间边界触发底盘平移。我们通过对TIAGo移动操作机器人的居家护理任务用户研究评估了该方法。与基线混合界面相比，耦合自我中心控制提高了物体操作效率，减少了基于按钮的控制工作量和手臂奇异点，降低了心理负荷和整体工作负担，并提高了易用性、易学性、用户信心及对躯干和底盘控制的偏好。

---

### 2. 方法动机分析
- **驱动力**：解决遥操作中“认知负荷过载”问题。用户在操作复杂高自由度机器人时，必须分神手动调节基座和躯干，导致任务流中断。
- **痛点**：现有混合界面（Hybrid Interfaces）虽然结合了自由形式控制（感知/操作）和约束控制（姿态/移动），但用户仍需频繁在不同控制通道间切换关注点。
- **研究假设**：机器人的动作应作为操作员感知意图（头部运动）和操作意图（手臂运动）的延伸，通过自动耦合实现“身体随人动”。

---

### 3. 方法设计详解
- **核心逻辑**：将头部和手臂输入映射为支持性反馈，通过预定义的阈值函数 $\Gamma(s; \tau, u)$ 自动触发躯干和底盘动作。
- **感知中心耦合（Perception-Centered）**：
  - **躯干高度**：当头部俯仰角超过阈值时，自动升降躯干以扩展视野。
  - **底盘旋转**：当头部偏航角超过阈值时，底盘自动旋转以扩展视野。
- **操作中心耦合（Manipulation-Centered）**：
  - **躯干高度**：当末端执行器接近垂直工作空间边界时，自动升降躯干以延展操作范围。
  - **底盘平移**：当末端执行器达到平移边界时，底盘自动移动以保持手臂处于可操作区域。
- **饱和处理与优先级**：通过式 (6) 综合多种控制信号，并使用式 (7) 的优先级逻辑（后退 > 侧向 > 前进）确保运动的稳定性和可预测性。

---

### 4. 方法对比分析
- **本质区别**：从“手动-辅助”模式转变为“意图-跟随”模式，将姿态和移动控制从“任务”降级为“自动后台进程”。
- **创新点**：提出了基于操作员 egocentric 信号（视线、手势）与机器人机体运动的自动耦合策略，有效缓解了全身操作的协调负担。
- **适用场景**：复杂、杂乱环境下的精细化移动操作，如居家辅助、多层货架存取。

---

### 5. 实验分析
- **验证方法**：对比基线（手持控制器+手动触摸板）与耦合控制，通过Phase I（基础任务）和Phase II（复合清理任务）评估效率、控制努力和心理负荷。
- **关键结果**：耦合控制在操作精细度上显著优于基线，操作时间减少，按钮使用次数降低至基线的五分之一，且在复杂任务中用户表现出更强的偏好。
- **局限**：某些特定情况下（如后退动作）用户仍偏好手动控制，说明完全自治并非总是最佳。

---

### 6. 实用指南
- **迁移性**：该方法不依赖特定硬件，易于迁移至其他具有升降躯干和移动底盘的平台。
- **实现建议**：需重点调试阈值 $\tau$ 和速度缩放因子 $\alpha_M$，以防止由于系统过度响应导致用户的眩晕或不可控感。同时应保留“可中断性”，即当用户通过控制器介入时，自动耦合应立即被覆盖。

---

### 7. 总结
- **核心思想**：将机器人的躯干与基座变为用户意图的自动跟随者。
- **速记版pipeline**：
  1. 监测头部角度与末端执行器位置。
  2. 若超出预设阈值，计算相应的补偿速度。
  3. 通过优先级逻辑仲裁多重运动请求。
  4. 平滑执行躯干升降与底盘位姿变换。

**Key Findings:**

- In this paper, we propose coupled egocentric control, a body-following teleoperation approach in which the robot's torso and base automatically respond to the operator's head and arm motions.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16095v1)
- [arXiv](https://arxiv.org/abs/2607.16095v1)

---

<a id='2607.16015v1'></a>
## [PIXIE: A Zero-Shot texture-invariant 6D pose estimation framework for unseen objects with assembly defects](https://arxiv.org/abs/2607.16015v1)

**Authors:** Leon Jungemeyer, Alejandro Magaña, Gautham Mohan, Matthias Karl, Daniel Werdehausen

**Published:** 2026-07-17

**Categories:** cs.CV, cs.RO

**Abstract:**

6D pose estimation remains a key challenge in robotics and computer vision, particularly in industrial environments. The deployment of currently available data-driven methods is often limited by resource-intensive data pipelines, reliance on textured 3D models, and sensitivity to geometric deviations caused by damages or assembly defects. We present PIXIE, a zero-shot framework that estimates the 6D pose of an object from an RGB image using only an untextured 3D model. Synthetic depth and normal maps are rendered from sampled reference viewpoints and matched to the query image via a pretrained cross-modality feature matcher. Matched keypoints are back-projected to obtain 2D--3D correspondences for PnP-based pose estimation. Relying exclusively on geometry makes the method inherently robust to lighting and texture variation, while correspondence filtering handles geometric deviations between the model and physical object. We evaluate on widely-used public benchmarks, reporting state-of-the-art results on texture-less objects without object-specific training, and introduce a novel dataset with assembly defects, texture variations, and occlusion to demonstrate real-world applicability.

**Analysis:**

### 1. 摘要翻译
6D姿态估计在机器人技术和计算机视觉中仍是一项核心挑战，特别是在工业环境中。当前基于数据驱动的方法受限于资源密集型数据流水线、对纹理3D模型的依赖，以及对损伤或装配缺陷导致的几何偏差的敏感性。本文提出了PIXIE，一个仅利用无纹理3D模型即可从RGB图像估计物体6D姿态的零样本框架。该方法通过预训练的跨模态特征匹配器，将从采样参考视点渲染的合成深度图和法线图与查询图像进行匹配。匹配的关键点被反投影以获得2D-3D对应关系，从而进行PnP姿态估计。由于仅依赖几何信息，该方法天生具备对光照和纹理变化的鲁棒性，同时通过对应关系过滤处理模型与实际物体间的几何偏差。我们在广泛使用的公共基准上进行了评估，在无纹理物体上报告了最先进的结果，并引入了一个包含装配缺陷、纹理变化和遮挡的新数据集，以展示其现实世界的适用性。

### 2. 方法动机分析
- **驱动力**：工业生产中零部件频繁更换且缺乏高质量纹理模型，同时制造公差或装配缺陷会导致真实物体与CAD模型存在几何差异。
- **现有痛点**：现有方法严重依赖对特定物体训练（需要昂贵标注）或高度依赖物体纹理外观。当光照变化或物体表面出现缺陷时，基于外观的特征描述符往往失效。
- **核心直觉**：几何形状是物体本质的、不随纹理和光照改变的特征。通过绕过外观（RGB色彩），直接在几何表示（深度/法线图）上进行跨模态匹配，可以实现本质上的零样本泛化。

### 3. 方法设计详解
- **流程总结**：
  1. **离线几何参考生成**：利用无纹理CAD模型，从采样视点渲染深度图和法线图。
  2. **跨模态特征匹配**：将渲染的几何图与输入RGB图像输入通用跨模态匹配器（如MINIMA），获取2D-2D匹配对。
  3. **2D-3D投影与姿态求解**：利用已知的渲染相机参数，将深度图像素反投影到物体坐标系，结合匹配点通过PnP+RANSAC求解初步姿态。
  4. **迭代精炼**：基于初始位姿，在预计算的局部视角池中选择最优匹配视点，再次进行匹配与PnP，提高精度。
- **关键技术**：
  - **几何编码**：将深度与法线归一化并施加Colormap，使其适配预训练的图像骨干网络，实现“图像与几何”的同态匹配。
  - **反投影**：通过$P^{cam} = B(u, v, D(u, v), K)$实现从图像空间到物体3D空间的直接映射，消除了对训练的依赖。
  - **位姿精炼**：利用视点之间的角距离进行选择，规避了深度学习方法中常见的显式回归精炼过程，保持了框架的纯几何属性。

### 4. 方法对比分析
- **本质区别**：不学习物体的外观特征，而是利用神经网络处理“几何转RGB”的匹配问题。
- **创新贡献**：首次提出全流程无训练、纯几何驱动的6D姿态估计；定义了处理模型偏差的几何相似度量。
- **适用场景**：工业巡检中存在形变、表面磨损、甚至更换了材料/颜色的零部件姿态估计。

### 5. 实验分析（精简版）
- **验证方法**：在BOP基准数据集（T-LESS等）及自建含缺陷乐高积木数据集上验证。
- **关键结论**：在无纹理物体上超越了所有需要训练的方法；在面对缺陷部件时，其精度显著优于其他零样本方法。
- **优劣势**：优势在于泛化性极强，不受光照与纹理影响；局限在于对高度对称的物体（如纯球体）难以获取有效的几何关键点。

### 6. 实用指南
- **开源情况**：数据集已开源：`ju-leon.github.io/PIXIE-dataset`。
- **实现细节**：建议使用已有的成熟特征匹配器（如MINIMA或LightGlue），渲染视角需覆盖物体全空间（建议使用Fibonacci点集采样）。
- **迁移建议**：该方法无需训练，迁移到新零件只需提供STL/OBJ模型，工作流极短。

### 7. 总结
- **核心思想**：几何为骨，特征为媒，实现零样本姿态感知。
- **速记版pipeline**：
  1. 渲染CAD模型的深度/法线图作为参考。
  2. 使用通用匹配器匹配真实图片。
  3. 将匹配点反投影计算3D坐标。
  4. 通过几何对齐求解初步位姿。
  5. 迭代匹配邻近视角精炼结果。

**Key Findings:**

- We present PIXIE, a zero-shot framework that estimates the 6D pose of an object from an RGB image using only an untextured 3D model.
- We evaluate on widely-used public benchmarks, reporting state-of-the-art results on texture-less objects without object-specific training, and introduce a novel dataset with assembly defects, texture variations, and occlusion to demonstrate real-world applicability.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16015v1)
- [arXiv](https://arxiv.org/abs/2607.16015v1)

---

<a id='2607.16012v1'></a>
## [DPNeXt: A Lightweight Multi-Scale Feature Fusion Framework for Efficient ViT-Based Multi-Task Dense Prediction](https://arxiv.org/abs/2607.16012v1)

**Authors:** Jehun Kang, Jungha Wang, Youngjun Hwang, David Hyunchul Shim

**Published:** 2026-07-17

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Multi-Task Learning (MTL) in robotics perception systems supports comprehensive 3D spatial scene understanding by integrating semantic segmentation and depth estimation. While Vision Foundation Models (VFMs) are increasingly adopted as robust feature encoders, existing decoding strategies present a critical bottleneck. To address this, we propose DPNeXt, a streamlined multi-scale feature fusion decoder and efficient alternative to the standard Dense Prediction Transformer (DPT). DPNeXt uses dual depthwise separable inverted bottlenecks to improve frozen VFM utilization through fusion-centric decoding and independent task modularization. To further mitigate negative inductive transfer between tasks, we introduce the Multi-Task Boundary Guidance (MTBG) strategy. Unlike prior boundary-aware methods that add fusion modules or gating, MTBG applies symmetric boundary-focused supervision to encourage geometric consistency without extra annotation or inference cost. Experiments on Cityscapes show that DPNeXt-S outperforms prior state-of-the-art (SOTA) MTL models, while DPNeXt-B further improves the overall performance and achieves the best results among the compared methods. On NYUv2, DPNeXt-B also achieves the best semantic segmentation and depth estimation results among the compared methods while requiring substantially fewer trainable parameters than prior large-scale MTL models. Compared with the standard DPT, DPNeXt-S reduces trainable parameters by 78.6% and achieves the fastest inference speed among the compared models on resource-constrained laptop hardware. The source code, model checkpoints, and a demo video will be made available at https://github.com/kangjehun/DPNeXt.

**Analysis:**

### 1. 摘要翻译
多任务学习（MTL）在机器人感知系统中通过整合语义分割和深度估计来提供全面的3D空间场景理解。尽管视觉基础模型（VFMs）正日益被采用作为鲁棒的特征编码器，但现有的解码策略构成了关键瓶颈。为解决这一问题，我们提出了DPNeXt，一个精简的多尺度特征融合解码器，作为标准密集预测Transformer（DPT）的高效替代方案。DPNeXt利用双深度可分离反向瓶颈结构，通过融合中心解码和独立任务模块化，优化了冻结VFM的利用率。为进一步缓解任务间的负面归纳迁移，我们引入了多任务边界引导（MTBG）策略。与现有的增加融合模块或门控的边界感知方法不同，MTBG应用对称的边界聚焦监督，在无需额外标注或增加推理成本的前提下鼓励几何一致性。在Cityscapes上的实验表明，DPNeXt-S超越了现有的SOTA MTL模型，而DPNeXt-B进一步提升了整体性能并取得最优结果。在NYUv2上，DPNeXt-B同样实现了最优的语义分割和深度估计性能，且相比先前的大规模MTL模型，所需的训练参数显著减少。相比标准DPT，DPNeXt-S的训练参数减少了78.6%，并在资源受限的笔记本硬件上实现了最快的推理速度。

### 2. 方法动机分析
- **核心动机**：在VFM作为特征提取“基石”的背景下，解决如何高效解码VFM输出特征以平衡多任务间的精度与推理速度的矛盾。
- **现有痛点**：现有DPT类解码器存在“人工通道扩展”导致的冗余计算；现有任务平衡策略（如辅助分支）增加了推理延迟，且Cross-task模块引入了负面迁移和结构耦合。
- **核心直觉**：通过纯卷积的“融合中心”解码架构（摒弃人工扩展）和训练时引入边界引导（推理时删除）来实现零延迟的任务一致性优化。

### 3. 方法设计详解
- **Pipeline**：
    1.  **特征提取**：利用冻结的DINOv2-Reg提取四个尺度的特征图。
    2.  **多尺度重组（IPA）**：通过Isotropic Projection Adapter（IPA）将各层通道统一为256，消除冗余的通道扩展。
    3.  **特征融合（DDSIF）**：设计Dual Depthwise Separable Inverted Fusion块，递归地从深层向浅层融合特征。其本质是使用深度可分离卷积进行空间过滤，并配合点积卷积（Pointwise Convolution）完成通道扩展（r=2）。
    4.  **任务预测**：独立且模块化的任务头（Head）对融合后的特征进行处理。
    5.  **边界引导（MTBG）**：在训练中加入边界监督（Canny+Dilation获得标签），通过辅助损失鼓励模型在边缘处保持结构几何一致，推理时彻底丢弃此路径。
- **公式意义**：$L_{bound}$ 通过动态权重调节类别不平衡，将边界预测约束在语义一致性范围内。

### 4. 方法对比分析
- **本质区别**：DPNeXt拒绝在解码器中进行大参数量的通道扩张，完全依赖纯卷积进行高效多尺度信息流传递，而非Transformer机制。
- **创新点**：
    - **DDSIF模块**：将ResNet-style的残差融合单元轻量化，实现了硬件友好的推理。
    - **零推理代价的MTBG**：通过将复杂结构仅放在训练阶段，规避了推理时的性能损耗。
- **适用场景**：边缘计算平台、资源受限的机器人嵌入式设备，或任何需要高性能密集预测的实时场景。

### 5. 实验分析
- **关键结论**：在Cityscapes和NYUv2上，DPNeXt-B实现了SOTA，且DPNeXt-S在推理速度上显著领先（51.02 FPS），参数量降低78.6%。
- **优势**：极佳的参数-精度平衡，无特殊算子依赖（Native PyTorch/ONNX支持）。
- **局限**：对超大分辨率输入的处理主要依赖分片（patch-based）推理，未深入探讨非规则场景下的鲁棒性。

### 6. 实用指南
- **开源情况**：已开源，代码见 `https://github.com/kangjehun/DPNeXt`。
- **关键细节**：训练时需注意loss权重调整，特别是MTBG的比例；针对NYUv2需额外引入表面法线任务（Surface Normal Loss）。
- **迁移建议**：可将IPA+DDSIF模块直接替换掉现有DPT类结构的Decoder部分，通过预训练的DINOv2 backbone快速Fine-tune即可。

### 7. 总结
- **核心思想**：利用轻量级纯卷积解码与边界辅助训练，实现高效的多任务密集预测。
- **速记版Pipeline**：
    1. 冻结预训练视觉大模型提取特征。
    2. 通过适配器统一特征维度。
    3. 使用深度可分离卷积逐层融合。
    4. 训练时引入边界一致性约束。
    5. 推理时仅保留主任务预测分支。

**Key Findings:**

- To address this, we propose DPNeXt, a streamlined multi-scale feature fusion decoder and efficient alternative to the standard Dense Prediction Transformer (DPT).
- To further mitigate negative inductive transfer between tasks, we introduce the Multi-Task Boundary Guidance (MTBG) strategy.
- Experiments on Cityscapes show that DPNeXt-S outperforms prior state-of-the-art (SOTA) MTL models, while DPNeXt-B further improves the overall performance and achieves the best results among the compared methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.16012v1)
- [arXiv](https://arxiv.org/abs/2607.16012v1)

---

<a id='2607.15974v1'></a>
## [Embodied Active Learning under Limited Annotation and Navigation Budget for Object Detection](https://arxiv.org/abs/2607.15974v1)

**Authors:** Hadrien Crassous, Mohamed Yassine Kabouri, Minahil Raza, Joni Pajarinen, Riad Akrour

**Published:** 2026-07-17

**Categories:** cs.RO, cs.CV

**Abstract:**

This paper studies how to adapt a computer vision object detector to an unknown environment under both a robot navigation time and annotation budget constraint. Our approach selects informative robot trajectories and image samples to retrain the detector, explicitly targeting its failure cases. Formally, the approach is an embodied variant of batch active learning, where at each round an agent has a limited navigation budget to collect candidate samples and a limited annotation budget for the most relevant images. We leverage spatial consistency to identify images with inconsistent labels, which are likely to provide the greatest improvement to the vision model. We evaluate the approach using different active learning objectives on large scenes from the AI2-THOR simulator and on a real-world setup using a Boston Dynamics Spot robot with the real-time object detector YOLOv5. Through comparison against several baselines, our experimental results show that spatial inconsistency helps guide the agent and select relevant images without external supervision, achieving the highest detection accuracy at the end of the adaptation process under the same budget. The open-source project can be found at https://mkabouri.github.io/embodied-active-learning-od

**Analysis:**

以下是对该论文的方法分析与总结：

### 1. 摘要翻译
本文研究了如何在受限的机器人导航时间和标注预算下，将计算机视觉目标检测器适应到未知环境中。我们的方法通过选择具有信息量的机器人轨迹和图像样本来对检测器进行再训练，明确针对其失效场景。在形式上，该方法是批量主动学习（Batch Active Learning）的一种具体化变体，即在每一轮中，智能体拥有有限的导航预算来收集候选样本，并拥有有限的标注预算来标记最相关的图像。我们利用空间一致性来识别具有不一致标签的图像，这些图像极有可能为视觉模型提供最大的改进。通过在AI2-THOR模拟器的大型场景和使用Boston Dynamics Spot机器人的真实世界设置中进行评估，实验结果表明，空间一致性有助于在无需外部监督的情况下引导智能体并选择相关图像，从而在相同预算下实现了最高的检测精度。

### 2. 方法动机分析
*   **驱动力**：在真实世界中，机器人必须自主适应新环境，但标注数据极昂贵且耗时，同时机器人移动探索也受限于时间/电池预算。
*   **现有方法痛点**：传统主动学习通常是离线的，缺乏对机器人探索行为的有效引导；单纯依靠熵或模型置信度进行采样，容易忽略场景中的空间关系和时序变化导致的失效。
*   **研究假设**：通过比较连续观测帧之间的预测结果，可以量化“预测差异（Prediction Discrepancy）”，这种不一致性是衡量模型在该状态下是否“困惑”的有效代理指标，无需人工标注即可作为导航探索的内在奖励和数据采样的启发式准则。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **主动探索**：利用基于Random Forest和UCB的Bayesian Optimization（BO）规划器，将“预测差异”作为内在奖励，引导机器人前往模型最易出错的区域进行采集。
    2.  **数据采集**：机器人沿规划路径移动，获取RGB帧及其对应的检测器预测结果，并存入Buffer。
    3.  **不一致性评估**：计算连续帧之间预测物体的对称差（Symmetric Difference），得到差异分$\Delta$。
    4.  **样本选择**：采用两阶段策略，先选出$\Delta$最高的候选样本，再通过聚类采样（结合Class Conditioned Matching Similarity, CCMS）保证选出的样本多样性。
    5.  **迭代优化**：将选中的样本提交给Oracle（模拟器API或视觉大模型）获取真实标签，再进行模型微调（Retrain），更新Buffer中的预测结果。
*   **关键公式**：$\Delta(\theta, I_s, I_{s'}, a) = |C^s \cup C^{s'}| - |C^s \cap C^{s'}|$。该公式衡量了两个连续视角的检测框类别列表的差异，差异越大，说明模型在该处预测越不稳定。

### 4. 方法对比分析
*   **本质区别**：传统方法多关注单帧的预测不确定性，而本文的核心在于**“时序-空间一致性”**，将机器人轨迹规划与数据采样深度耦合。
*   **创新贡献**：提出了一种基于预测差异的轻量级衡量指标，不仅用于数据回放的排序，还直接驱动了好奇心驱动的机器人导航策略。
*   **适用场景**：适用于资源受限（Annotation/Navigation Budget）下的机器人场景适应任务。

### 5. 实验分析（精简版）
*   **验证方法**：在AI2-THOR模拟环境及真实Boston Dynamics Spot机器人上进行实验。
*   **关键结果**：在相同预算下，Prediction Discrepancy策略在mAP50指标上显著优于Random、Entropy及Count基线。
*   **优势**：无需外部监督即可自主发现难例，不仅提升了检测精度，还保证了数据分布的多样性。
*   **局限**：对检测器的漏检（Missing）非常敏感，但在计算量和硬件部署上仍有一定的开销。

### 6. 实用指南
*   **开源情况**：已开源，GitHub地址：`https://mkabouri.github.io/embodied-active-learning-od/`。
*   **实现细节**：在真实实验中，使用OWLv2作为Oracle，标注耗时约4.5秒/图，这限制了全量自动化的效率。迁移至其他任务时，核心是寻找一个能反映“模型困惑度”的度量函数（如文中提出的预测差异）。

### 7. 总结
*   **核心思想**：利用多视角预测的一致性差异，引导机器人主动探索并筛选难例。
*   **速记版 Pipeline**：
    1. 移动采集图像；
    2. 比较前后帧差异，标出难例；
    3. 对难例进行人类/模型标注；
    4. 重新训练检测模型；
    5. 重复迭代提升性能。

**Key Findings:**

- Our approach selects informative robot trajectories and image samples to retrain the detector, explicitly targeting its failure cases.
- Through comparison against several baselines, our experimental results show that spatial inconsistency helps guide the agent and select relevant images without external supervision, achieving the highest detection accuracy at the end of the adaptation process under the same budget.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15974v1)
- [arXiv](https://arxiv.org/abs/2607.15974v1)

---

