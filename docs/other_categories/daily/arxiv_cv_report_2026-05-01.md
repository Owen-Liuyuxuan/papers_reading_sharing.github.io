time: 20260501

# Arxiv Computer Vision Papers - 2026-05-01

## Executive Summary

以下是为2026年4月30日Arxiv计算机视觉论文日报撰写的执行摘要：

---

### **执行摘要：2026-04-30 计算机视觉前沿论文速览**

本日报收录了10篇于2026年4月30日发布的论文，整体呈现三大核心趋势：**多模态智能体的基础模型化**、**世界模型向统一与可控方向演进**、以及**3D重建与生成技术的泛化性突破**。以下为主要发现与推荐。

---

#### **一、主要主题与趋势**

1. **多模态模型向“原生智能体”进化**：多篇论文（如GLM-5V-Turbo、PRTS、LaST-R1）不再仅仅关注视觉理解，而是致力于将视觉、语言与行动（Action）深度融合，构建能自主推理、规划并执行任务的智能体。这标志着从“感知”到“决策-行动”闭环的范式转变。

2. **世界模型走向统一与可控**：HERMES++与PhyCo分别从驾驶场景和通用运动生成角度，推动世界模型从单一预测向“理解+生成+可控”的一体化框架发展。这说明该领域正致力于弥合仿真与真实世界的鸿沟。

3. **3D视觉的泛化与数据效率**：以Generalizable Sparse-View 3D Reconstruction和3D-ReGen为代表，研究聚焦于在极少或无约束输入下（如稀疏视角、非受控图像）实现高质量3D重建，这是降低数据采集成本、推动3D落地关键。

---

#### **二、特别重要的创新论文**

- **GLM-5V-Turbo** (论文2)：提出面向多模态智能体的原生基础模型，直接支持视觉、语言与行动的统一表征与推理，而非后拼接。其架构设计可能成为下一代多模态模型的基石。
- **HERMES++** (论文3)：统一驾驶世界模型，同时支持3D场景理解（感知）与生成（规划/仿真）。若能有效融合，将对自动驾驶仿真与决策有重大影响。
- **Representation Fréchet Loss** (论文6)：提出一种新的生成模型损失函数，直接对比特征分布（而非像素或语义），有望从原理上提升视觉生成质量与多样性，值得实验验证。

---

#### **三、新兴研究方向与技术**

- **物理先验的可控生成**：PhyCo (论文8) 首次将物理规律（如动量、接触力）作为可学习的可控先验融入运动生成，使生成动作既符合物理规律又能被用户精准操控。这是生成模型从“视觉合理”走向“物理合理”的重要一步。
- **黑盒蒸馏强化学习**：PRISM (论文10) 提出在无需访问教师模型内部参数的情况下，通过在线策略蒸馏实现多模态强化学习的预对齐。该方法对实际部署（如API限制场景）具有实用价值。
- **基于对比表示的原始推理**：PRTS (论文1) 利用对比学习将低级视觉特征映射为可推理的“原语”，为符号规划与视觉感知的桥梁提供了新思路。

---

#### **四、推荐精读论文（优先级排序）**

1. **GLM-5V-Turbo** — 若您关注多模态智能体或基础模型架构设计，不可错过。
2. **HERMES++** — 对自动驾驶、世界模型或仿真环境构建的研究者重点推荐。
3. **Representation Fréchet Loss** — 对生成模型（GAN/扩散）理论有深入兴趣者必读，可能影响未来损失函数设计。
4. **PhyCo** — 如果您从事人体运动、机器人或物理仿真生成工作，这篇论文提供的新范式极具启发性。
5. **3D-ReGen** — 对3D重建与生成统一框架感兴趣的读者，其“再生”（Regeneration）概念值得关注。

---

如需进一步获取任一论文的详细技术分析或文献对比，欢迎随时提出。

---

## Table of Contents

1. [PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations](#2604.27472v1)
2. [GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](#2604.26752v1)
3. [HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation](#2604.28196v1)
4. [Generalizable Sparse-View 3D Reconstruction from Unconstrained Images](#2604.28193v1)
5. [LaST-R1: Reinforcing Action via Adaptive Physical Latent Reasoning for VLA Models](#2604.28192v1)
6. [Representation Fréchet Loss for Visual Generation](#2604.28190v1)
7. [Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling](#2604.28185v1)
8. [PhyCo: Learning Controllable Physical Priors for Generative Motion](#2604.28169v1)
9. [3D-ReGen: A Unified 3D Geometry Regeneration Framework](#2604.28134v1)
10. [PRISM: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning](#2604.28123v1)

---

## Papers

<a id='2604.27472v1'></a>
## [PRTS: A Primitive Reasoning and Tasking System via Contrastive Representations](https://arxiv.org/abs/2604.27472v1)

**Authors:** Yang Zhang, Jiangyuan Zhao, Chenyou Fan, Fangzheng Yan, Tian Li, Haitong Tang, Sen Fu, Xuan'er Wu, Qizhen Weng, Weinan Zhang, Xiu Li, Chi Zhang, Chenjia Bai, Xuelong Li

**Published:** 2026-04-30

**Categories:** cs.AI, cs.LG, cs.RO

**Abstract:**

Vision-Language-Action (VLA) models advance robotic control via strong visual-linguistic priors. However, existing VLAs predominantly frame pretraining as supervised behavior cloning, overlooking the fundamental nature of robot learning as a goal-reaching process that requires understanding temporal task progress. We present \textbf{PRTS} (\textbf{P}rimitive \textbf{R}easoning and \textbf{T}asking \textbf{S}ystem), a VLA foundation model that reformulates pretraining through Goal-Conditioned Reinforcement Learning. By treating language instructions as goals and employing contrastive reinforcement learning, PRTS learns a unified embedding space where the inner product of state-action and goal embeddings approximates the log-discounted goal occupancy, the probability of reaching the language-specified goal from the current state-action, quantitatively assessing physical feasibility beyond static semantic matching. PRTS draws this dense goal-reachability supervision directly from offline trajectories without reward annotations, and folds it into the VLM backbone via a role-aware causal mask, incurring negligible overhead over vanilla behavior cloning. This paradigm endows the high-level reasoning system with intrinsic goal reachability awareness, bridging semantic reasoning and temporal task progress, and further benefits goal-conditioned action prediction. Pretrained on 167B tokens of diverse manipulation and embodied-reasoning data, PRTS reaches state-of-the-art performance on LIBERO, LIBERO-Pro, LIBERO-Plus, SimplerEnv, and a real-world suite of 14 complex tasks, with particularly substantial gains on long-horizon, contact-rich, and zero-shot novel-instruction settings, confirming that injecting goal-reachability awareness significantly improves both execution success and long-horizon planning of general-purpose robotic foundation policies.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **PRTS (Primitive Reasoning and Tasking System)** 的论文分析如下：

### 1. 主要贡献总结
PRTS 提出了一种通过“目标条件强化学习（Goal-Conditioned RL）”范式重构 VLA 模型预训练的新方法，旨在解决传统行为克隆（Behavior Cloning）忽略任务时序逻辑的问题。该系统通过对比学习构建了一个统一的嵌入空间，能够量化评估从当前状态达到目标指令的物理可行性，从而将高层语义推理与底层时序任务进度有效融合。

### 2. 关键创新与方法论
*   **对比强化学习范式**：不同于传统的监督学习，PRTS 将语言指令视为“目标”，通过对比学习将状态-动作与目标嵌入对齐。其核心是使两者的内积逼近“对数折现目标占有率（log-discounted goal occupancy）”，即从当前状态达到目标的可达性概率。
*   **无奖励标注的可达性监督**：该模型直接从离线轨迹中挖掘目标可达性信息，无需额外的奖励标注。这使得模型在缺乏专家奖励信号的情况下，能利用海量非结构化数据学习任务的时序规律。
*   **角色感知因果掩码（Role-aware Causal Mask）**：在 VLM 主干网络中引入特定掩码，将可达性监督无缝融入推理架构，在保持模型轻量化和高效性的同时，提升了对复杂任务的推理能力。

### 3. 对领域的潜在影响
*   **从“模仿”到“推理”的范式转变**：该研究标志着机器人基础模型从简单的动作复现（Behavior Cloning）向具备时序感知与逻辑推理的智能体演进，为解决长序列任务（Long-horizon tasks）提供了可解释的数学框架。
*   **提升物理一致性**：通过将“物理可达性”引入 embedding 空间，模型不再仅仅做语义匹配，而是能够判断动作在物理环境中的可行性，这是提升机器人与真实世界交互鲁棒性的关键。

### 4. 受益的相关领域与应用
*   **具身智能 (Embodied AI)**：特别是长序列、复杂操作场景（如家具装配、复杂工具使用）。
*   **机器人操作规划 (Manipulation Planning)**：该模型对接触密集（Contact-rich）和零样本指令（Zero-shot novel instructions）的良好表现，使其在柔性制造和家庭服务机器人领域具有广阔前景。
*   **多模态大模型 (MLLMs)**：为将视觉-语言模型从被动的感知转向主动的交互控制提供了重要的方法论借鉴。

### 5. 可推断的局限性
*   **对数据质量的依赖**：虽然利用离线轨迹无需奖励标注，但该方法依然高度依赖于 offline 数据集的覆盖范围；如果轨迹数据缺乏多样性或存在偏差，模型在未见过的物理环境下可能难以准确评估可达性。
*   **计算复杂度的权衡**：尽管文中提到“开销可忽略”，但在极大规模的实时推理任务中，计算目标可达性得分仍可能对模型的推理延迟（Latency）产生潜在影响，特别是如果需要在多目标规划中频繁调用该函数。
*   **语义与物理的割裂风险**：虽然模型试图统一两者，但在极端复杂的非结构化动态环境中，语义层面的语言指令（如“把那个清理干净”）到物理层面的像素级动作序列映射，可能仍存在解释鸿沟。

### 专家点评：
这篇论文的趣味性在于它敏锐地捕捉到了当前 VLA 模型的一个根本缺陷：**行为克隆虽然学会了“做什么”，但没学会“为什么做”以及“怎么判断任务进展”**。通过将 RL 中的可达性概念转化为对比表征空间的算子，PRTS 巧妙地为视觉模型注入了“物理常识”，这是通向高阶通用机器人智能的关键一步。

**Key Findings:**

- We present \textbf{PRTS} (\textbf{P}rimitive \textbf{R}easoning and \textbf{T}asking \textbf{S}ystem), a VLA foundation model that reformulates pretraining through Goal-Conditioned Reinforcement Learning.
- Pretrained on 167B tokens of diverse manipulation and embodied-reasoning data, PRTS reaches state-of-the-art performance on LIBERO, LIBERO-Pro, LIBERO-Plus, SimplerEnv, and a real-world suite of 14 complex tasks, with particularly substantial gains on long-horizon, contact-rich, and zero-shot novel-instruction settings, confirming that injecting goal-reachability awareness significantly improves both execution success and long-horizon planning of general-purpose robotic foundation policies.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.27472v1)
- [arXiv](https://arxiv.org/abs/2604.27472v1)

---

<a id='2604.26752v1'></a>
## [GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](https://arxiv.org/abs/2604.26752v1)

**Authors:** GLM-V Team,  :, Wenyi Hong, Xiaotao Gu, Ziyang Pan, Zhen Yang, Yuting Wang, Yue Wang, Yuanchang Yue, Yu Wang, Yanling Wang, Yan Wang, Xijun Liu, Wenmeng Yu, Weihan Wang, Wei Li, Shuaiqi Duan, Sheng Yang, Ruiliang Lv, Mingdao Liu, Lihang Pan, Ke Ning, Junhui Ji, Jinjiang Wang, Jing Chen, Jiazheng Xu, Jiale Zhu, Jiale Cheng, Ji Qi, Guobing Gan, Guo Wang, Cong Yao, Zijun Dou, Zihao Zhou, Zihan Wang, Zhiqi Ge, Zhijie Li, Zhenyu Hou, Zhao Xue, Zehui Wang, Zehai He, Yusen Liu, Yukuo Cen, Yuchen Li, Yuan Wang, Yijian Lu, Yanzi Wang, Yadong Xue, Xinyu Zhang, Xinyu Liu, Wenkai Li, Tianyu Tong, Tianshu Zhang, Shengdong Yan, Qinkai Zheng, Mingde Xu, Licheng Bao, Jiaxing Xu, Jiaxin Fan, Jiawen Qian, Jiali Chen, Jiahui Lin, Haozhi Zheng, Haoran Wang, Haochen Li, Fan Yang, Dan Zhang, Chuangxin Zhao, Chengcheng Wu, Boyan Shi, Bowei Jia, Baoxu Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

We present GLM-5V-Turbo, a step toward native foundation models for multimodal agents. As foundation models are increasingly deployed in real environments, agentic capability depends not only on language reasoning, but also on the ability to perceive, interpret, and act over heterogeneous contexts such as images, videos, webpages, documents, GUIs. GLM-5V-Turbo is built around this objective: multimodal perception is integrated as a core component of reasoning, planning, tool use, and execution, rather than as an auxiliary interface to a language model. This report summarizes the main improvements behind GLM-5V-Turbo across model design, multimodal training, reinforcement learning, toolchain expansion, and integration with agent frameworks. These developments lead to strong performance in multimodal coding, visual tool use, and framework-based agentic tasks, while preserving competitive text-only coding capability. More importantly, our development process offers practical insights for building multimodal agents, highlighting the central role of multimodal perception, hierarchical optimization, and reliable end-to-end verification.

**Analysis:**

这是一份关于 GLM-5V-Turbo 论文的方法分析报告。

### 1. 摘要翻译
我们推出了 GLM-5V-Turbo，这是迈向多模态智能体原生基础模型的重要一步。随着基础模型在现实环境中的广泛部署，其智能体能力不仅依赖于语言推理，还需具备对图像、视频、网页、文档及 GUI 等异构上下文的感知、解释和行动能力。GLM-5V-Turbo 将多模态感知作为推理、规划、工具使用和执行的核心组件，而非仅仅作为语言模型的辅助接口。本报告总结了 GLM-5V-Turbo 在模型设计、多模态训练、强化学习、工具链扩展及智能体框架集成方面的改进。这些进展赋予了模型在多模态编码、视觉工具使用和框架化任务中的卓越表现，同时保留了极具竞争力的纯文本编码能力。

### 2. 方法动机分析
- **驱动力**：旨在构建“原生”多模态智能体，使感知不再是附加模块，而是决策链的内生部分。
- **痛点**：现有模型多将视觉作为辅助输入，缺乏对异构视觉环境（如 GUI、复杂文档）的细粒度理解与长期动态规划能力；且端到端训练导致多任务间产生干扰。
- **核心假设**：通过分层优化（而非单一端到端训练）以及更紧密的视觉-语言集成，能更有效地构建稳健的智能体，且通过明确的策略模式共享，可实现各领域能力的共同提升。

### 3. 方法设计详解
- **CogViT 视觉编码器**：采用两阶段预训练。阶段一通过蒸馏（利用 SigLIP2 的语义能力和 DINOv3 的纹理能力）进行掩码图像建模；阶段二进行对比学习，引入 NaFlex 实现变长输入处理，并使用 QK-Norm 增强大规模训练稳定性。
- **多模态多 Token 预测（MMTP）**：在 MTP 基础上，GLM-5V-Turbo 放弃了直接传递视觉 Embedding 的方案，改为采用 `<|image|>` 特殊占位符。这大大降低了流水线并行通信负载，且在 0.5B 规模实验中证明能获得更低的训练损失和更稳定的收敛。
- **联合多任务 RL 优化**：在超过 30 个任务类别上进行协同训练，利用 RL 对感知、推理和智能体任务进行分层优化。
- **基础设施 redesign**：实现了（1）统一的 VLM RL Gym 环境；（2）解耦的推理、奖励评估与 batch 构建流程；（3）针对视觉输入内存优化的拓扑感知数据加载机制。

### 4. 方法对比分析
- **本质区别**：从“工具调用”范式转向“原生视觉决策”范式，将视觉感知深入融合至预训练和 RL 阶段。
- **创新贡献**：提出 MMTP 设计以优化系统效率；利用多任务协同 RL 减少跨领域负面干扰；提出了 ImageMining 基准。
- **适用场景**：复杂 UI 导航、网页自动生成、深度研究辅助、跨文档与视觉的长期规划任务。

### 5. 实验分析
- **关键结果**：在 AndroidWorld (75.7)、OSWorld (62.3) 等 GUI 任务及 Design2Code (94.8) 等编码任务上表现优异，且在保持原有的强大文本编码能力基础上，不仅未退化，在部分基准上还有提升。
- **局限**：在 RL 未覆盖的领域，能力可能发生退化；长期记忆仍受限于上下文窗口与计算开销。

### 6. 实用指南
- **开源/资源**：模型及相关技能可在 [chat.z.ai](https://chat.z.ai/) 体验，工具链开源于 [GitHub (zai-org/GLM-skills)](https://github.com/zai-org/GLM-skills)。
- **迁移建议**：分层优化策略适用于资源受限下的智能体构建，建议先解决底层感知/动作的准确性，再进行高层长序列训练。
- **关键细节**：训练过程中引入“批判数据（critic data）”有助于纠正感知幻觉，值得在工业应用中参考。

### 7. 总结
- **核心思想**：视觉感知与智能体逻辑的深度原生融合，分层优化以实现稳健决策。
- **速记版 pipeline**：
  1. 通过 CogViT 提取多尺度视觉特征。
  2. 使用 `<|image|>` 占位符将视觉信息高效注入 MTP 架构。
  3. 执行多任务分层协同强化学习以对齐策略。
  4. 整合外部框架（如 Claude Code）进行长序列、多步的实际环境交互。

**Key Findings:**

- We present GLM-5V-Turbo, a step toward native foundation models for multimodal agents.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26752v1)
- [arXiv](https://arxiv.org/abs/2604.26752v1)

---

<a id='2604.28196v1'></a>
## [HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation](https://arxiv.org/abs/2604.28196v1)

**Authors:** Xin Zhou, Dingkang Liang, Xiwu Chen, Feiyang Tan, Dingyuan Zhang, Hengshuang Zhao, Xiang Bai

**Published:** 2026-04-30

**Categories:** cs.CV

**Abstract:**

Driving world models serve as a pivotal technology for autonomous driving by simulating environmental dynamics. However, existing approaches predominantly focus on future scene generation, often overlooking comprehensive 3D scene understanding. Conversely, while Large Language Models (LLMs) demonstrate impressive reasoning capabilities, they lack the capacity to predict future geometric evolution, creating a significant disparity between semantic interpretation and physical simulation. To bridge this gap, we propose HERMES++, a unified driving world model that integrates 3D scene understanding and future geometry prediction within a single framework. Our approach addresses the distinct requirements of these tasks through synergistic designs. First, a BEV representation consolidates multi-view spatial information into a structure compatible with LLMs. Second, we introduce LLM-enhanced world queries to facilitate knowledge transfer from the understanding branch. Third, a Current-to-Future Link is designed to bridge the temporal gap, conditioning geometric evolution on semantic context. Finally, to enforce structural integrity, we employ a Joint Geometric Optimization strategy that integrates explicit geometric constraints with implicit latent regularization to align internal representations with geometry-aware priors. Extensive evaluations on multiple benchmarks validate the effectiveness of our method. HERMES++ achieves strong performance, outperforming specialist approaches in both future point cloud prediction and 3D scene understanding tasks. The model and code will be publicly released at https://github.com/H-EmbodVis/HERMESV2.

**Analysis:**

这是一份关于论文《HERMES++: Toward a Unified Driving World Model for 3D Scene Understanding and Generation》的深度方法分析。

### 1. 摘要翻译
自动驾驶世界模型是模拟环境动态的关键技术。然而，现有方法多侧重于未来场景生成，忽视了全面的3D场景理解；而大语言模型（LLM）虽具备强大的推理能力，却缺乏对未来几何演变的预测能力，导致语义理解与物理模拟脱节。为弥合这一鸿沟，本文提出了HERMES++，一个将3D场景理解与未来几何预测统一在单一框架下的驾驶世界模型。该模型通过三个协同设计解决这些任务的不同需求：首先，BEV表示将多视图空间信息整合为LLM兼容的结构；其次，引入LLM增强的世界查询以实现从理解分支到生成分支的知识迁移；第三，设计“当前到未来”链接以桥接时间间隙，并将几何演变条件化于语义上下文之上。此外，采用联合几何优化策略，通过显式几何约束与隐式潜空间正则化确保结构完整性。实验表明，HERMES++在未来点云预测和3D场景理解任务上均优于专家模型。

### 2. 方法动机分析
- **驱动力**：构建一个既能“看懂”当前环境语义，又能“预测”未来几何演变的统一智能体。
- **痛点**：现有生成模型无法解释场景背后的因果关系；现有语义理解模型（LLM/VLM）无法预测物理演变。
- **研究假设**：通过统一的BEV表征作为空间锚点，利用大模型作为推理中枢，并引入语义引导的几何生成机制，可以消除语义理解与物理预测之间的鸿沟。

### 3. 方法设计详解
- **Pipeline**：
    1. **视觉编码与表示**：多视图图像经视觉编码器提取特征，通过空间交叉注意力转化为BEV表示，并下采样为LLM兼容的视觉Token。
    2. **理解与推理**：LLM接收视觉Token与用户指令，输出自然语言回复，并利用因果注意力整合语义上下文至“世界查询（World Queries）”。
    3. **知识迁移**：世界查询（包含语义与先验）被注入至“当前到未来（Current-to-Future Link）”模块。
    4. **几何生成**：该模块结合文本注入（Textual Injection）与自适应自我运动调节（Ego Modulation），预测未来时序的BEV特征。
    5. **几何渲染**：通过可微的渲染器（Render）将BEV特征转换为3D点云，并辅以联合几何优化进行训练。
- **核心模块**：
    - **Current-to-Future Link**：这是核心桥梁，通过交叉注意力将LLM产生的语义信息映射到未来的空间演变中。
    - **Joint Geometric Optimization**：双重约束，显式约束保证点云坐标准确，隐式正则化（Lcos/Lgram）保证特征分布的结构一致性。

### 4. 方法对比分析
- **本质区别**：HERMES++不是简单堆砌多任务模型，而是通过“世界查询”将语义推理与空间几何演变动态链接。
- **创新点**：引入显式（几何约束）与隐式（特征Gram矩阵正则化）联合优化，解决生成中的结构塌陷问题；设计Ego Modulation有效解耦了运动与场景动态。

### 5. 实验分析
- **关键结论**：在NuScenes数据集上，HERMES++在3s未来点云预测的Chamfer Distance（CD）指标上比DriveX降低了8.2%；在场景理解（CIDEr指标）上比Omni-Q提升了9.2%。
- **优势**：无需额外的检测/地图监督，仅通过统一架构实现高精度感知与预测。
- **局限**：对极长跨度的未来预测稳定性仍有优化空间。

### 6. 实用指南
- **开源情况**：代码已发布：[https://github.com/H-EmbodVis/HERMESV2](https://github.com/H-EmbodVis/HERMESV2)。
- **实现细节**：建议使用OpenCLIP ConvNeXt-L作为骨干网络；训练分为三个阶段：预训练几何编码器、模态对齐（LLM投影）、联合端到端精调。
- **迁移可能**：该框架的“世界查询+当前到未来”链接设计非常适合需要“语义推理引导物理仿真”的机器人控制任务。

### 7. 总结
- **核心思想**：通过语义引导的BEV世界查询，实现场景理解与物理几何预测的深度融合。
- **速记版pipeline**：
    1. **空间转换**：把多视角图片变成统一的鸟瞰图(BEV)。
    2. **语义推理**：大模型解读BEV并回答任务指令。
    3. **查询生成**：根据语义提取关键任务信息(世界查询)。
    4. **几何推演**：根据语义引导预测未来的BEV空间。
    5. **结构优化**：利用隐式几何约束强化预测的准确性。

**Key Findings:**

- To bridge this gap, we propose HERMES++, a unified driving world model that integrates 3D scene understanding and future geometry prediction within a single framework.
- Our approach addresses the distinct requirements of these tasks through synergistic designs.
- First, a BEV representation consolidates multi-view spatial information into a structure compatible with LLMs. Second, we introduce LLM-enhanced world queries to facilitate knowledge transfer from the understanding branch.
- Extensive evaluations on multiple benchmarks validate the effectiveness of our method.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28196v1)
- [arXiv](https://arxiv.org/abs/2604.28196v1)

---

<a id='2604.28193v1'></a>
## [Generalizable Sparse-View 3D Reconstruction from Unconstrained Images](https://arxiv.org/abs/2604.28193v1)

**Authors:** Vinayak Gupta, Chih-Hao Lin, Shenlong Wang, Anand Bhattad, Jia-Bin Huang

**Published:** 2026-04-30

**Categories:** cs.CV

**Abstract:**

Reconstructing 3D scenes from sparse, unposed images remains challenging under real-world conditions with varying illumination and transient occlusions. Existing methods rely on scene-specific optimization using appearance embeddings or dynamic masks, which requires extensive per-scene training and fails under sparse views. Moreover, evaluations on limited scenes raise questions about generalization. We present GenWildSplat, a feed-forward framework for sparse-view outdoor reconstruction that requires no per-scene optimization. Given unposed internet images, GenWildSplat predicts depth, camera parameters, and 3D Gaussians in a canonical space using learned geometric priors. An appearance adapter modulates appearance for target lighting conditions, while semantic segmentation handles transient objects. Through curriculum learning on synthetic and real data, GenWildSplat generalizes across diverse illumination and occlusion patterns. Evaluations on PhotoTourism and MegaScenes benchmark demonstrate state-of-the-art feed-forward rendering quality, achieving real-time inference without test-time optimization

**Analysis:**

### 1. 摘要翻译
重构稀疏、未配准图像中的3D场景，在光照变化和瞬态遮挡等复杂现实条件下依然充满挑战。现有方法往往依赖于针对特定场景的优化（如外观嵌入或动态掩码），这不仅需要繁琐的逐场景训练，在极度稀疏的视图下也容易失效。此外，现有研究缺乏对真实环境泛化能力的深入评估。我们提出了 **GenWildSplat**，这是一个针对户外场景的稀疏视图重构前馈框架，无需逐场景优化。该框架利用学习到的几何先验，在规范空间中预测深度、相机参数及3D高斯分布。通过外观适配器（Appearance Adapter）根据目标光照条件调节颜色，并利用语义分割处理瞬态物体。通过在合成数据和真实数据上的课程学习，GenWildSplat在多样的光照与遮挡模式下展现出强大的泛化能力。在PhotoTourism和MegaScenes基准测试中，该方法实现了最先进的前馈渲染质量，且无需测试时优化，可实现实时推理。

### 2. 方法动机分析
- **驱动力**：实现一种“即插即用”的户外3D重构系统，摆脱传统方法对测试时昂贵优化（Optimization-based）的依赖，实现真正的前馈推理。
- **痛点**：现有方法（如WildGaussians, NexusSplats）依赖逐场景优化，导致推理速度极慢；且在稀疏视角下，几何与光照解耦失败，产生严重的几何伪影和颜色漂移。
- **研究假设**：通过在大规模合成和现实数据集上引入课程学习（Curriculum Learning），模型可以解耦场景的静态几何结构与动态光照变化，从而在推理阶段仅需前馈一次即可推断出鲁棒的3D场景。

### 3. 方法设计详解
- **核心流程**：
  1. **特征提取**：利用VGGT Transformer骨干网络，将稀疏的多视图输入映射为多尺度特征图。
  2. **头部分支**：通过三个头部网络分别预测深度图、相机参数（内参与外参）以及每像素的3D高斯属性（旋转、缩放、不透明度、颜色）。
  3. **外观解耦与转换**：引入**外观适配器（Appearance Adapter）**。首先由光编码器（Light Encoder）从单张图中提取压缩的“光照码”（Light Code），随后MLP将canonical（规范）空间下的3D高斯颜色调制为目标光照下的颜色。
  4. **Occlusion Handling**：利用预训练的语义分割网络（YOLOv8）生成瞬态物体遮挡掩码，通过掩码加权损失函数（Masked Loss）引导模型忽略瞬态区域，专注于静态结构。
- **算法精髓**：Canonical Space思想，即模型在训练时强迫学习一个统一的几何底座，通过可变的光照参数（Light Code）对该底座进行“渲染渲染映射”，而非通过过拟合不同视角来掩盖问题。

### 4. 方法对比分析
- **本质区别**：传统方法将外观作为可优化的变量嵌入场景，而GenWildSplat将外观视为输入图像的函数，通过网络前馈计算，实现了显式的光照控制。
- **创新贡献**：
    - 提出了外观适配器，实现了不同场景间的跨场景照明转移。
    - 引入三阶段课程学习（单场景 -> 多场景 -> 合成遮挡），解决了极度稀疏输入下几何模型易坍塌的问题。
- **适用场景**：稀疏视角户外场景（2-6张图），对实时性有要求的AR/VR与导航应用。

### 5. 实验分析
- **关键结论**：在MegaScenes基准上，GenWildSplat在指标（PSNR/SSIM）和视觉质量上显著优于现有前馈方法，且耗时从数小时压缩至3秒。
- **优势**：渲染速度极快，场景泛化性强，渲染一致性好。
- **局限**：对严重超出训练分布的场景（如室内复杂环境）仍有几何缺失；无法模拟精细的动态投影遮挡（cast shadows）。

### 6. 实用指南
- **开源情况**：项目主页已公布（https://genwildsplat.github.io/）。
- **训练细节**：训练需分为三阶段，对DL3DV数据集进行光照增强。关键超参数为外观适配器的MLP层数及Perceptual Loss权重（$\lambda = 0.05$）。
- **迁移可能**：该框架可以模块化迁移至其他前馈3D重构网络（如MVSplat），只需替换特征提取器或加入适配器分支即可。

### 7. 总结
- **核心思想**：利用课程学习构建几何先验，通过前馈适配器实现光照与遮挡的实时动态控制。
- **速记版Pipeline**：
  1. 输入稀疏图提取几何与外观特征；
  2. 预测场景深度与canonical高斯参数；
  3. 通过语义分割排除瞬态干扰；
  4. 使用光照编码器调制高斯颜色；
  5. 可微分栅格化渲染最终图像。

**Key Findings:**

- We present GenWildSplat, a feed-forward framework for sparse-view outdoor reconstruction that requires no per-scene optimization.
- Evaluations on PhotoTourism and MegaScenes benchmark demonstrate state-of-the-art feed-forward rendering quality, achieving real-time inference without test-time optimization

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28193v1)
- [arXiv](https://arxiv.org/abs/2604.28193v1)

---

<a id='2604.28192v1'></a>
## [LaST-R1: Reinforcing Action via Adaptive Physical Latent Reasoning for VLA Models](https://arxiv.org/abs/2604.28192v1)

**Authors:** Hao Chen, Jiaming Liu, Zhonghao Yan, Nuowei Han, Renrui Zhang, Chenyang Gu, Jialin Gao, Ziyu Guo, Siyuan Qian, Yinxi Wang, Peng Jia, Chi-Wing Fu, Shanghang Zhang, Pheng-Ann Heng

**Published:** 2026-04-30

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have increasingly incorporated reasoning mechanisms for complex robotic manipulation. However, existing approaches share a critical limitation: whether employing explicit linguistic reasoning that suffers from latency and discretization, or utilizing more expressive continuous latent reasoning, they are predominantly confined to static imitation learning that limits adaptability and generalization. While online reinforcement learning (RL) has been introduced to VLAs to enable trial-and-error exploration, current methods exclusively optimize the vanilla action space, bypassing the underlying physical reasoning process. In this paper, we present \textbf{LaST-R1}, a unified VLA framework that integrates latent Chain-of-Thought (CoT) reasoning over physical dynamics prior to action execution, along with a tailored RL post-training paradigm. Specifically, we propose \textbf{Latent-to-Action Policy Optimization (LAPO)}, a novel RL algorithm that jointly optimizes the latent reasoning process and the action generation. By bridging reasoning and control, LAPO improves the representation of physical world modeling and enhances robustness in interactive environments. Furthermore, an \textbf{adaptive latent CoT mechanism} is introduced to allow the policy to dynamically adjust its reasoning horizon based on environment complexity. Extensive experiments show that LaST-R1 achieves a near-perfect 99.8\% average success rate on the LIBERO benchmark with only one-shot supervised warm-up, significantly improving convergence speed and performance over prior state-of-the-art methods. In real-world deployments, LAPO post-training yields up to a 44\% improvement over the initial warm-up policy across four complex tasks, including both single-arm and dual-arm settings. Finally, LaST-R1 demonstrates strong generalization across simulated and real-world environments.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作（VLA）模型在机器人复杂操控任务中已广泛采用推理机制。然而，现有方法存在严重局限：无论是基于延迟高、离散化的显式语言推理，还是基于连续隐空间的推理，大都局限于静态模仿学习，限制了适应性和泛化能力。虽然在线强化学习（RL）已被引入VLA以实现试错探索，但现有方法仅优化原始动作空间，忽略了底层的物理推理过程。本文提出了LaST-R1，这是一个统一的VLA框架，在动作执行前集成了关于物理动态的隐式思维链（CoT）推理，并配套了针对性的RL后训练范式。具体而言，我们提出了“隐动作策略优化”（LAPO），一种联合优化隐推理过程与动作生成的新型RL算法。通过桥接推理与控制，LAPO改善了物理世界建模表示，增强了交互环境中的鲁棒性。此外，引入自适应隐式CoT机制，使策略能根据环境复杂度动态调整推理视野。实验表明，LaST-R1在LIBERO基准测试中仅需单次监督预热，平均成功率即达到接近完美的99.8%，显著优于现有前沿方法。在真实世界部署中，LAPO后训练在四项复杂任务中较初始预热策略提升了高达44%的成功率，并展现了在仿真与真实环境间的强大泛化能力。

---

### 2. 方法动机分析
*   **驱动力**：解决VLA模型在复杂交互中“只知执行、不知规划”的物理认知缺陷。
*   **现有方法痛点**：
    1.  **模仿学习受限**：依赖静态数据集，缺乏在线环境交互，难以应对未见过的场景（OOD）。
    2.  **动作空间优化盲目**：现有的RL仅优化动作，跳过了“推理-规划-动作”的完整链条，导致模型对物理规律缺乏深层理解。
    3.  **推理开销与僵化**：显式推理（如语言）带来严重延迟，固定的推理步长无法适应任务复杂度。
*   **研究假设**：通过在动作执行前引入隐式物理推理，并利用RL同时对推理链（Latent）和动作进行联合优化，可以显著提升机器人的物理建模能力和泛化鲁棒性。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **预计算隐式先验**：使用DINOv3提取视觉观察中的`<CLS>` token，结合top-k选择，将其作为表征物理环境的“认知锚点”。
    2.  **LaST-R1架构**：基于Qwen3-VL-4B，先进行 autoregressive 隐式Token生成（建模未来状态），后进行并行化动作解码。
    3.  **LAPO训练（核心）**：引入`<latent_end>`特殊Token，将RL回报信号同时反向传播给推理阶段（Latent）和执行阶段（Action）。
    4.  **自适应推理**：根据任务复杂度，模型自主决策何时生成`<latent_end>`，从而动态截断推理过程。
*   **算法解释**：
    *   **LAPO目标函数**：引入了 step-level 的联合似然比，允许 reward 同时引导 Latent Space 的偏移和 Action 的分布。通过 Clip-Surrogate 优化，确保推理过程向“带来高回报”的方向演化。
    *   **自适应机制**：将 `<latent_end>` 视为一个概率阈值事件（p > 0.99），配合特定的策略 Loss 惩罚，迫使模型在简单任务上“早停”，在复杂任务上“多思考”。

---

### 4. 方法对比分析
*   **本质区别**：与仅优化Action的Vanilla RL不同，LAPO优化的是整个“推理序列+动作生成”联合路径。
*   **创新贡献**：首次提出在VLA的RL后训练中将隐式推理Token视为决策变量，并设计了自适应推理长度的动态机制。
*   **适用场景**：高动态、高复杂度的机器人操作任务（如长序列抓取、双臂协作、对物理环境敏感的任务）。

---

### 5. 实验分析（精简版）
*   **关键结论**：在LIBERO基准上达到了99.8%的平均成功率，验证了“推理-动作”联合优化的有效性。
*   **优势**：在线RL带来显著的OOD泛化性能提升，不仅克服了模仿学习的数据依赖，还显著降低了执行步数。
*   **局限**：对推理Token的依赖增加了模型训练的复杂性，且在高频率物理交互中对算力的要求略高于纯动作策略。

---

### 6. 实用指南
*   **开源情况**：已发布项目主页（siriyep.github.io/last-r1/）。
*   **实现细节**：
    *   训练使用bf16混合精度，推荐AdamW优化器。
    *   关键超参数：推理Token最大长度Nmax=8，候选位置M=4。
    *   注意Critic-to-Actor的更新比例（论文建议2:1）。
*   **迁移建议**：若要迁移，关键在于复用其联合Loss的架构设计，即确保Reward不仅约束动作，还能约束Latent Token的语义聚类。

---

### 7. 总结
*   **核心思想**：通过RL优化隐式推理与动作的联合概率，实现机器人“思考-执行”的闭环迭代。
*   **速记版pipeline**：1. 视觉特征提取预计算；2. 隐式推理Token自回归生成；3. 联合优化推理与动作；4. 依据回报动态调整推理时长。

**Key Findings:**

- In this paper, we present \textbf{LaST-R1}, a unified VLA framework that integrates latent Chain-of-Thought (CoT) reasoning over physical dynamics prior to action execution, along with a tailored RL post-training paradigm.
- Specifically, we propose \textbf{Latent-to-Action Policy Optimization (LAPO)}, a novel RL algorithm that jointly optimizes the latent reasoning process and the action generation.
- Extensive experiments show that LaST-R1 achieves a near-perfect 99.8\% average success rate on the LIBERO benchmark with only one-shot supervised warm-up, significantly improving convergence speed and performance over prior state-of-the-art methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28192v1)
- [arXiv](https://arxiv.org/abs/2604.28192v1)

---

<a id='2604.28190v1'></a>
## [Representation Fréchet Loss for Visual Generation](https://arxiv.org/abs/2604.28190v1)

**Authors:** Jiawei Yang, Zhengyang Geng, Xuan Ju, Yonglong Tian, Yue Wang

**Published:** 2026-04-30

**Categories:** cs.CV

**Abstract:**

We show that Fréchet Distance (FD), long considered impractical as a training objective, can in fact be effectively optimized in the representation space. Our idea is simple: decouple the population size for FD estimation (e.g., 50k) from the batch size for gradient computation (e.g., 1024). We term this approach FD-loss. Optimizing FD-loss reveals several surprising findings. First, post-training a base generator with FD-loss in different representation spaces consistently improves visual quality. Under the Inception feature space, a one-step generator achieves0.72 FID on ImageNet 256x256. Second, the same FD-loss repurposes multi-step generators into strong one-step generators without teacher distillation, adversarial training or per-sample targets. Third, FID can misrank visual quality: modern representations can yield better samples despite worse Inception FID. This motivates FDr$^k$, a multi-representation metric. We hope this work will encourage further exploration of distributional distances in diverse representation spaces as both training objectives and evaluation metrics for generative models.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Representation Fréchet Loss for Visual Generation》的论文进行了如下分析：

### 1. 论文核心贡献总结
该论文提出了一种名为 **FD-loss** 的方法，首次证明了长期被视为“仅能作为评价指标”的 Fréchet Distance (FD) 可以转化为有效的生成模型训练目标。通过将 FD 的分布估计样本量（population size）与梯度计算的批次大小（batch size）解耦，该方法不仅提升了生成模型的视觉质量，还使得多步生成模型能够直接转化为高性能的单步生成器，无需复杂的蒸馏或对抗训练。

### 2. 关键创新与方法论
*   **解耦策略 (Decoupling Strategy)：** 传统意义上，计算 Fréchet Distance 需要较大的样本集来拟合分布，这在小 Batch 下无法实现。作者引入的解耦思想允许在小 Batch 上计算梯度，同时通过维护更大规模的分布统计信息来保持分布估计的准确性，从而使梯度下降成为可能。
*   **训练目标的重新定义：** 将原本的“评估指标”转换为“损失函数”。这意味着生成模型在训练过程中不再仅仅依赖于逐像素的损失（如 MSE 或感知损失），而是直接优化其在特征空间（如 Inception 空间）中的分布表现。
*   **提出 FDr$^k$ 指标：** 针对单一 Inception 空间评价指标可能失效的问题，提出了基于多表征（multi-representation）的评价体系，揭示了现代深度特征在评价生成质量上的局限性。

### 3. 对计算机视觉领域的潜在影响
*   **简化生成模型训练范式：** 该方法通过直接优化分布距离，省去了传统的对抗训练（GAN 的不稳定博弈）或繁琐的蒸馏过程（Distillation），极大地简化了将多步模型（如扩散模型）转化为高效单步模型的过程。
*   **视觉质量的新标杆：** 在 ImageNet 256x256 上实现 0.72 的极低 FID，证明了该方法在生成性能上的统治力，为生成式模型的发展提供了新的技术路径。
*   **重新定义评价指标：** 指出 FID 的局限性并提出 FDr$^k$，促使学界重新审视“如何准确度量图像生成质量”，可能会改变当前生成模型 benchmarks 的评价标准。

### 4. 相关领域与受益应用
*   **扩散模型（Diffusion Models）：** 特别是模型蒸馏（Distillation）与加速推理领域，该方法提供了一种直接获得单步高效模型的方案。
*   **图像生成与修复：** 对生成质量要求极高的任务（如高保真图像合成）。
*   **生成模型评估工具开发：** 任何需要高质量自动化评估指标的生成任务（如视频生成、音频生成）。
*   **表征学习：** 由于该方法依赖于特征空间，未来可以结合自监督学习表征（如 DINOv2）进一步提升生成表现。

### 5. 可推断的潜在局限性
*   **对特征空间的依赖：** 训练效果极大地受限于所选用的表征空间。如果表征空间本身存在偏差，FD-loss 可能会导致模型产生“针对特定表征优化”的伪影，即“过拟合”了特征提取器。
*   **计算开销：** 虽然解耦降低了单步计算量，但在特征空间中维护大规模样本的分布统计量依然需要额外的显存和计算资源。
*   **对分布多样性的捕捉：** 尽管 FD 关注分布整体，但在极其复杂的模态分布下，仅优化 FD 是否会丢失掉部分长尾分布的细节，仍需进一步验证。

**总结建议：** 这篇论文的趣味性在于它打破了“训练”与“评估”的界限，用极其简洁的数学思路解决了复杂的问题，极有可能成为未来生成式模型训练架构中的重要组件。

**Key Findings:**

- We show that Fréchet Distance (FD), long considered impractical as a training objective, can in fact be effectively optimized in the representation space.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28190v1)
- [arXiv](https://arxiv.org/abs/2604.28190v1)

---

<a id='2604.28185v1'></a>
## [Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling](https://arxiv.org/abs/2604.28185v1)

**Authors:** Keming Wu, Zuhao Yang, Kaichen Zhang, Shizun Wang, Haowei Zhu, Sicong Leng, Zhongyu Yang, Qijie Wang, Sudong Wang, Ziting Wang, Zili Wang, Hui Zhang, Haonan Wang, Hang Zhou, Yifan Pu, Xingxuan Li, Fangneng Zhan, Bo Li, Lidong Bing, Yuxin Song, Ziwei Liu, Wenhu Chen, Jingdong Wang, Xinchao Wang, Xiaojuan Qi, Shijian Lu, Bin Wang

**Published:** 2026-04-30

**Categories:** cs.CV

**Abstract:**

Recent visual generation models have made major progress in photorealism, typography, instruction following, and interactive editing, yet they still struggle with spatial reasoning, persistent state, long-horizon consistency, and causal understanding. We argue that the field should move beyond appearance synthesis toward intelligent visual generation: plausible visuals grounded in structure, dynamics, domain knowledge, and causal relations. To frame this shift, we introduce a five-level taxonomy: Atomic Generation, Conditional Generation, In-Context Generation, Agentic Generation, and World-Modeling Generation, progressing from passive renderers to interactive, agentic, world-aware generators. We analyze key technical drivers, including flow matching, unified understanding-and-generation models, improved visual representations, post-training, reward modeling, data curation, synthetic data distillation, and sampling acceleration. We further show that current evaluations often overestimate progress by emphasizing perceptual quality while missing structural, temporal, and causal failures. By combining benchmark review, in-the-wild stress tests, and expert-constrained case studies, this roadmap offers a capability-centered lens for understanding, evaluating, and advancing the next generation of intelligent visual generation systems.

**Analysis:**

作为计算机视觉领域的专家，我针对这篇题为《Visual Generation in the New Era: An Evolution from Atomic Mapping to Agentic World Modeling》的综述性论文进行了深度分析：

### 1. 核心贡献总结
该论文系统性地总结了视觉生成模型从“像素合成”向“智能世界建模”演进的范式转移，提出了一个衡量视觉生成能力的五级分类法（从原子生成到世界模型生成）。它旨在通过定义这一演进路线，指出当前模型在因果推理、长时一致性与物理规律方面存在的不足，并为未来构建具备自主性与世界理解力的生成系统提供了战略框架。

### 2. 关键创新与方法论
*   **五级分类框架 (Five-level Taxonomy)**：这是本文最核心的创新。它不仅是一个分类法，更是一个评价模型智能水平的坐标系：
    *   **Atomic (原子级)**：单纯的视觉映射。
    *   **Conditional (条件级)**：基于输入约束的生成。
    *   **In-Context (上下文级)**：利用跨模态上下文关联。
    *   **Agentic (代理级)**：具备交互能力与任务目标导向。
    *   **World-Modeling (世界模型级)**：具备物理规律、因果理解与持久状态记忆。
*   **重新定义评估范式**：论文尖锐地指出，目前的评估过于侧重“视觉感知质量（Perceptual Quality）”，而忽略了“结构、时序与因果逻辑”的失效。它提倡一种“以能力为中心”的综合评测方案。

### 3. 对领域的潜在影响
*   **学术导向的修正**：该论文极有可能将视觉生成的研究重心从单纯追求“美学质量”和“训练规模”转向“物理合理性”和“因果可解释性”。
*   **技术路线的整合**：它将扩散模型（Flow Matching）、大模型架构（Unified models）与世界模型（World Models）的研究孤岛串联起来，为构建下一代具身智能（Embodied AI）提供了理论基础。

### 4. 受益的相关领域与应用
*   **具身智能与机器人 (Embodied AI & Robotics)**：这是直接受益者。机器人需要基于视觉世界模型进行动作预测和因果推理，而非简单的图像生成。
*   **自动驾驶 (Autonomous Driving)**：仿真环境的构建依赖于对复杂交通场景中时序演变和因果交互的模拟。
*   **电影与游戏工业**：从“画面生成”向“场景演化”的跨越，将彻底改变长视频创作的生产工作流，实现高度一致性的长时内容生成。
*   **科学计算与仿真**：在物理实验的数字孪生中，生成模型需要满足物理守恒定律，这与论文提倡的方向高度契合。

### 5. 可推断的局限性
*   **定义的模糊边界**：虽然提出了五级分类，但在实际操作中，区分“Agentic”与“World-Modeling”在具体架构实现上可能存在交叉点，模型如何界定其“智能边界”尚需更严苛的测试。
*   **合成数据依赖的悖论**：论文提到“合成数据蒸馏”是技术驱动力之一，但如果模型自身的“世界建模”能力不足，通过合成数据训练出的模型可能陷入“错误反馈闭环（Model Collapse）”，如何平衡合成数据的质量与真实物理真理（Ground Truth）的验证是其潜在的技术瓶颈。
*   **计算成本的挑战**：向世界模型演进意味着更高的推理代价，论文虽然提到了采样加速，但实现这一演进所需的算力增长是否会超过产业界的可承受范围，是该路线图落地的一大现实隐忧。

**专家点评**：
这篇论文的意义在于它**为视觉生成领域提供了一个“成年礼”式的反思**。它标志着视觉生成正走出“唯美学论”的初级阶段，开始向理解现实物理世界、具备逻辑推理能力的深水区迈进。对于正在投入大模型研究的团队而言，这篇论文不仅是研究地图，更是一份关于“智能本质”的路线规划书。

**Key Findings:**

- To frame this shift, we introduce a five-level taxonomy: Atomic Generation, Conditional Generation, In-Context Generation, Agentic Generation, and World-Modeling Generation, progressing from passive renderers to interactive, agentic, world-aware generators.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28185v1)
- [arXiv](https://arxiv.org/abs/2604.28185v1)

---

<a id='2604.28169v1'></a>
## [PhyCo: Learning Controllable Physical Priors for Generative Motion](https://arxiv.org/abs/2604.28169v1)

**Authors:** Sriram Narayanan, Ziyu Jiang, Srinivasa Narasimhan, Manmohan Chandraker

**Published:** 2026-04-30

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Modern video diffusion models excel at appearance synthesis but still struggle with physical consistency: objects drift, collisions lack realistic rebound, and material responses seldom match their underlying properties. We present PhyCo, a framework that introduces continuous, interpretable, and physically grounded control into video generation. Our approach integrates three key components: (i) a large-scale dataset of over 100K photorealistic simulation videos where friction, restitution, deformation, and force are systematically varied across diverse scenarios; (ii) physics-supervised fine-tuning of a pretrained diffusion model using a ControlNet conditioned on pixel-aligned physical property maps; and (iii) VLM-guided reward optimization, where a fine-tuned vision-language model evaluates generated videos with targeted physics queries and provides differentiable feedback. This combination enables a generative model to produce physically consistent and controllable outputs through variations in physical attributes-without any simulator or geometry reconstruction at inference. On the Physics-IQ benchmark, PhyCo significantly improves physical realism over strong baselines, and human studies confirm clearer and more faithful control over physical attributes. Our results demonstrate a scalable path toward physically consistent, controllable generative video models that generalize beyond synthetic training environments.

**Analysis:**

## 1. 摘要翻译

现代视频扩散模型在外观合成方面表现出色，但在物理一致性上仍存在不足：物体漂移、碰撞缺乏真实回弹、材料响应难以与其物理属性匹配。为此，我们提出了PhyCo，一个为视频生成引入连续、可解释且具有物理基础控制的框架。我们的方法包含三个核心组件：(i) 一个包含10万多个真实模拟视频的大规模数据集，涵盖摩擦力、恢复系数、变形和外力等多种物理属性的系统性变化；(ii) 使用ControlNet对预训练扩散模型进行物理监督微调，并以像素对齐的物理属性图为条件；(iii) 基于视觉语言模型（VLM）的奖励优化，通过精调后的VLM对生成视频进行物理相关性提问，并提供可微分反馈。该方法无需在推理阶段进行物理模拟或几何重建，即可生成物理一致且可控的视频。在Physics-IQ基准测试中，PhyCo在物理真实性方面显著优于强基线模型，人类评估也证实了其对物理属性更清晰、更忠实的控制。

## 2. 方法动机分析

- **驱动力**：旨在填补通用视频生成模型在“物理常识”和“精确物理控制”上的空白，使其能根据人类指定的物理参数（如“该物体非常软”或“在粗糙表面滑动”）生成动态视频。
- **现有方法痛点**：现有方法要么依赖复杂的显式物理引擎（如PhysGen、WonderPlay），推理成本高且泛化性差；要么仅依靠语言提示（如ForcePrompting）进行模糊引导，缺乏对物理属性（摩擦、变形等）的定量、连续控制。
- **研究假设**：通过在具备明确物理标注的大规模合成数据上进行监督学习，并在生成过程中显式注入空间对齐的物理属性图，可以使扩散模型内化物理规律，从而在无模拟器辅助的情况下实现高保真的物理控制。

## 3. 方法设计详解

- **流程总结**：
  1. **数据构建**：利用Kubric和PyBullet生成10万+视频，涵盖物体运动的多样化物理场景，并将物理参数（摩擦、变形等）编码为空间对齐的属性图。
  2. **物理监督微调 (Stage 1)**：固定预训练的Cosmos-Predict2模型，通过ControlNet架构注入物理属性图。属性图被分为摩擦/恢复系数、变形参数、外力矢量三组，通过Tokenizer编码后作为条件输入。
  3. **VLM奖励优化 (Stage 2)**：利用精调后的Qwen2.5-VL作为物理审查器，针对生成视频提出物理属性相关的问题（如“摩擦力是否在指定范围内”），计算逻辑差异并将其转化为可微分奖励，反向传播微调ControlNet，强化物理约束。
- **模型结构**：采用“冻结底座+可训练ControlNet适配器”模式。ControlNet分支负责将抽象的物理参数场映射到隐空间，实现对运动轨迹和物体形态的干预。
- **算法解释**：VLM Loss通过计算正确答案与错误答案的Logit差值（Binary Cross-Entropy），强迫生成模型输出更符合物理常识的轨迹。

## 4. 方法对比分析

- **本质区别**：与显式模拟（依赖求解器）和隐式先验（依赖纯文本）不同，PhyCo采用了“显式空间条件输入 + VLM逻辑反馈”的混合监督策略，平衡了控制精度与生成速度。
- **创新贡献**：提出了将物理属性（F, R, D, E）作为空间对齐热力图进行条件注入的架构，以及一套针对物理逻辑的VLM微调奖励机制。
- **适用场景**：适用于需要精确物体动力学控制的视频生成任务，如科学模拟、复杂交互动画、交互式视频合成。

## 5. 实验分析

- **验证方法**：在Physics-IQ benchmark上进行定量对比；通过2AFC人类偏好测试评估物理真实性；通过定向力测试验证泛化能力。
- **关键结果**：在Physics-IQ五大领域（固体、流体、光学、磁力、热力）上得分显著领先；力方向控制误差（15.2°）远低于基线（40.5°）。
- **优势**：无需测试时模拟，泛化性强，可实现多属性组合控制（Compositionality）。
- **局限**：对复杂的多物体强耦合交互（如复杂的流固耦合）物理模型仍是近似；不能完全保证严苛的守恒定律（如动量守恒）。

## 6. 实用指南

- **开源情况**：代码和视频结果见 [phyco-video.github.io](https://phyco-video.github.io)。
- **实现细节**：建议使用高FPS（24FPS）进行训练以减少闪烁伪影；VLM监督部分计算资源消耗大，建议先对视频下采样处理。
- **迁移可能**：该框架的ControlNet + VLM奖励机制可直接迁移至其他视频Backbone（如Sora, CogVideoX），只需重构适配器的输入维度。

## 7. 总结

- **核心思想**：通过物理属性图条件化视频扩散模型并结合VLM反馈进行强化。
- **速记版pipeline**：
  1. 使用物理引擎创建带属性标注的合成视频集。
  2. 训练ControlNet将物理场映射为视频生成条件。
  3. 利用VLM对视频进行物理逻辑打分。
  4. 反向传播物理反馈以优化生成视频质量。

**Key Findings:**

- We present PhyCo, a framework that introduces continuous, interpretable, and physically grounded control into video generation.
- Our approach integrates three key components: (i) a large-scale dataset of over 100K photorealistic simulation videos where friction, restitution, deformation, and force are systematically varied across diverse scenarios; (ii) physics-supervised fine-tuning of a pretrained diffusion model using a ControlNet conditioned on pixel-aligned physical property maps; and (iii) VLM-guided reward optimization, where a fine-tuned vision-language model evaluates generated videos with targeted physics queries and provides differentiable feedback.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28169v1)
- [arXiv](https://arxiv.org/abs/2604.28169v1)

---

<a id='2604.28134v1'></a>
## [3D-ReGen: A Unified 3D Geometry Regeneration Framework](https://arxiv.org/abs/2604.28134v1)

**Authors:** Geon Yeong Park, Roman Shapovalov, Rakesh Ranjan, Jong Chul Ye, Andrea Vedaldi, Thu Nguyen-Phuoc

**Published:** 2026-04-30

**Categories:** cs.CV

**Abstract:**

We consider the problem of regenerating 3D objects from 2D images and initial 3D shapes. Most 3D generators operate in a one-shot fashion, converting text or images to a 3D object with limited controllability. We introduce instead 3D-ReGen, a 3D regenerator that is conditioned on an initial 3D shape. This conceptually simple formulation allows us to support numerous useful tasks, including 3D enhancement, reconstruction, and editing. 3D-ReGen uses a new conditioning mechanism based on VecSet, which allows the regenerator to update or improve the input geometry with consistent fine-grained details. 3D-ReGen learns a widely applicable regeneration prior from off-the-shelf 3D datasets via self-supervised pretext tasks and augmentations, without additional annotations. We evaluate both the geometric consistency and fine-grained quality of 3D-ReGen, achieving state-of-the-art performance in controllable 3D generation across several tasks.

**Analysis:**

作为计算机视觉领域的专家，我针对 **3D-ReGen** 这篇论文的摘要进行了深度解析。以下是详细分析：

### 1. 核心贡献总结
**3D-ReGen** 提出了一种创新的统一 3D 几何再生框架，通过引入初始 3D 形状作为调节（Conditioning）输入，解决了传统“一键式”生成方法中缺乏可控性的问题。该框架实现了 3D 模型的增强、重建与编辑，并能够通过自监督学习从现有大规模 3D 数据集中提取通用的几何再生先验，显著提升了生成结果的几何一致性与细节表现。

### 2. 关键创新与方法论
*   **基于 VecSet 的调节机制：** 这是该论文的核心技术突破。不同于传统的体素（Voxel）或隐式场（Implicit Field）表示，VecSet 似乎提供了一种更灵活的几何特征嵌入方式，使模型能够有效地“感知”并基于原始形状进行精细化修补或升级，同时保持几何结构的连贯性。
*   **自监督再生先验（Self-supervised Prior）：** 该方法无需额外的标注数据，通过设计预训练任务（pretext tasks）和数据增强，让模型学习如何将“低质量/不完整”的 3D 数据重构为“高质量/完整”的几何体。这种范式将 3D 生成问题从“从头生成”转变为“对已有几何的迭代优化”。

### 3. 对领域的潜在影响
*   **从“生成式”转向“编辑/优化式”：** 目前 3D 生成领域多聚焦于生成外观，但缺乏对现有 3D 资产的“精修”能力。3D-ReGen 的出现标志着 3D 内容创作流程（Pipeline）的升级，即 3D 资产可以像 2D 图像编辑一样进行微调和迭代。
*   **打破生成与重建的边界：** 该研究模糊了“生成模型”与“重建模型”的界限，利用统一的再生框架处理从稀疏点云到高质量 Mesh 的跨越，极大地提升了 3D 数据处理的通用性。

### 4. 相关应用领域
*   **游戏开发与工业设计：** 游戏美术师可以利用此工具快速将草图或低模（Low-poly）转换为精细的高模（High-poly），大幅降低建模成本。
*   **数字孪生与医疗成像：** 在处理传感器扫描得到的噪声数据或不完整重建时，3D-ReGen 可作为强大的降噪与补全工具。
*   **AR/VR 内容生产：** 为实时扫描的真实世界物体提供自动化的细节补偿和拓扑优化。

### 5. 可推断的局限性
*   **对初始形状的依赖性：** 尽管增强了可控性，但如果初始 3D 形状的几何拓扑错误严重（例如完全错误的连通性），模型是否能纠正该误差是一个挑战。
*   **计算资源与推理速度：** 作为一个具备“再生”能力的框架，其在处理复杂几何结构时，对比轻量级的前馈网络（Feed-forward networks），可能在推理延迟（Latency）方面存在开销。
*   **纹理与材质处理：** 摘要侧重于“几何再生（Geometry Regeneration）”，尚不明确该模型在处理复杂纹理映射（Texture Mapping）或物理材质属性（PBR）上的表现能力，这通常是几何生成之外的另一个难点。

---
**专家点评：**
3D-ReGen 的研究重点切中了当前 3D AIGC 领域最大的痛点——**缺乏对生成过程的精细干预能力**。通过将 3D 几何问题建模为“再生”过程，该论文提供了一种极具潜力的架构路径，特别是在处理大规模工业级数据应用时，这种利用自监督学习获取先验的方法具有极高的学术研究价值与商业落地前景。

**Key Findings:**

- We introduce instead 3D-ReGen, a 3D regenerator that is conditioned on an initial 3D shape.
- 3D-ReGen uses a new conditioning mechanism based on VecSet, which allows the regenerator to update or improve the input geometry with consistent fine-grained details.
- We evaluate both the geometric consistency and fine-grained quality of 3D-ReGen, achieving state-of-the-art performance in controllable 3D generation across several tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28134v1)
- [arXiv](https://arxiv.org/abs/2604.28134v1)

---

<a id='2604.28123v1'></a>
## [PRISM: Pre-alignment via Black-box On-policy Distillation for Multimodal Reinforcement Learning](https://arxiv.org/abs/2604.28123v1)

**Authors:** Sudong Wang, Weiquan Huang, Xiaomin Yu, Zuhao Yang, Hehai Lin, Keming Wu, Chaojun Xiao, Chen Chen, Wenxuan Wang, Beier Zhu, Yunjian Zhang, Chengwei Qin

**Published:** 2026-04-30

**Categories:** cs.CV, cs.AI, cs.CL

**Abstract:**

The standard post-training recipe for large multimodal models (LMMs) applies supervised fine-tuning (SFT) on curated demonstrations followed by reinforcement learning with verifiable rewards (RLVR). However, SFT introduces distributional drift that neither preserves the model's original capabilities nor faithfully matches the supervision distribution. This problem is further amplified in multimodal reasoning, where perception errors and reasoning failures follow distinct drift patterns that compound during subsequent RL. We introduce PRISM, a three-stage pipeline that mitigates this drift by inserting an explicit distribution-alignment stage between SFT and RLVR. Building on the principle of on-policy distillation (OPD), PRISM casts alignment as a black-box, response-level adversarial game between the policy and a Mixture-of-Experts (MoE) discriminator with dedicated perception and reasoning experts, providing disentangled corrective signals that steer the policy toward the supervision distribution without requiring access to teacher logits. While 1.26M public demonstrations suffice for broad SFT initialization, distribution alignment demands higher-fidelity supervision; we therefore curate 113K additional demonstrations from Gemini 3 Flash, featuring dense visual grounding and step-by-step reasoning on the hardest unsolved problems. Experiments on Qwen3-VL show that PRISM consistently improves downstream RLVR performance across multiple RL algorithms (GRPO, DAPO, GSPO) and diverse multimodal benchmarks, improving average accuracy by +4.4 and +6.0 points over the SFT-to-RLVR baseline on 4B and 8B, respectively. Our code, data, and model checkpoints are publicly available at https://github.com/XIAO4579/PRISM.

**Analysis:**

### 1. 摘要翻译
大型多模态模型（LMMs）的标准训练流程是：先进行监督微调（SFT），随后通过基于验证奖励的强化学习（RLVR）进行优化。然而，SFT会导致分布漂移，这既未能保留模型原始能力，也无法准确匹配监督分布。这种问题在多模态推理中尤为严重，因为感知错误和推理失误遵循不同的漂移模式。我们提出了**PRISM**，这是一种在SFT和RLVR之间插入显式分布对齐阶段的三阶段训练流水线。基于在线策略蒸馏（OPD）原理，PRISM将对齐转化为一种黑盒、响应级的对抗博弈，即策略模型与混合专家（MoE）判别器之间的博弈。该判别器具备专门的感知和推理专家，能够在无需教师模型对数（logits）的情况下，提供解耦的纠正信号，将策略引导回监督分布。实验表明，PRISM在Qwen3-VL模型上显著提升了下游RLVR性能，平均准确率较SFT→RLVR基线提升了4.4（4B）和6.0（8B）个百分点。

---

### 2. 方法动机分析
- **驱动力**：SFT过程中的分布漂移会改变基础模型的初始分布，导致后续RLVR在错误的起点上开始优化，且多模态任务中感知与推理的漂移表现各异，需针对性修复。
- **痛点**：传统的SFT通过词元级（token-level） imitation训练，未能区分“处理过程”与“结果”，且SFT后的分布偏离导致模型不再保留原始的强推理能力。
- **研究假设**：在进入强化学习前，引入一个基于对抗机制的显式分布对齐阶段，可以修复因SFT产生的异质性分布漂移，从而提供更优的初始化，解锁后续RLVR的潜力。

---

### 3. 方法设计详解
- **pipeline流程**：
  1. **冷启动SFT**：在高质量数据集（1.37M样本，包含113K curated）上进行传统SFT，获得初始策略。
  2. **分布对齐阶段**：关键创新点。使用MoE判别器与策略模型进行对抗训练。策略模型生成样本，MoE判别器根据“感知准确度”和“推理一致性”提供反馈，通过Bradley-Terry损失进行对抗优化。
  3. **RLVR阶段**：在对齐后的策略基础上，使用确定性 verifiable reward 进行最终强化学习。
- **模型结构**：MoE判别器由两部分组成：
  - **感知专家 ($D_v$)**：专门评估视觉描述与输入图像的对齐程度。
  - **推理专家 ($D_r$)**：评估思维链（CoT）推理的逻辑一致性和推导正确性。
  - 最终奖励：$r = \alpha D_v + (1-\alpha) D_r$。
- **算法解释**：将对齐看作一个Minimax博弈，判别器最大化监督样本与生成样本之间的得分差距，策略模型最小化这一差距。通过消除KL正则项，允许策略模型彻底重塑其输出分布以匹配监督参考。

---

### 4. 方法对比分析
- **本质区别**：与现有方法相比，PRISM将“分布对齐”作为一个独立的、明确的中间阶段，而非在RL流程中内嵌或单纯通过数据增强实现。
- **创新贡献**：引入了黑盒对抗式的MoE判别器，无需教师模型logits，且能对感知和推理进行解耦纠正。
- **适用场景**：适用于所有具有结构化输出（如思维链+结论）的多模态推理任务。

---

### 5. 实验分析
- **验证方法**：在Qwen3-VL (4B/8B) 上，通过MathVista, MathVerse, MMMU等多个多模态推理基准进行测试。
- **结论**：PRISM在所有算法（GRPO/DAPO/GSPO）上均表现出显著的性能提升，且发现SFT数据规模与对齐阶段呈互补关系。
- **优势/局限**：优势在于显著提升了RL起点质量；局限是增加了额外的计算开销和对结构化输出格式的依赖。

---

### 6. 实用指南
- **开源情况**：代码、数据及模型权重均开源（Github: XIAO4579/PRISM）。
- **实现细节**： alignment阶段使用1e-6学习率，MoE权重 $\alpha=0.5$，不使用KL正则。核心在于判别器必须先进行 warm-start 训练。
- **迁移可能**：可直接迁移至纯文本逻辑推理或长文档处理，仅需更换判别器专家配置（如移除感知专家，增加事实性检查专家）。

---

### 7. 总结
- **核心思想**：通过解耦的MoE判别器在RL前显式修复SFT导致的分布漂移。
- **速记版pipeline**：
  1. **SFT**：基础能力引导。
  2. **MoE对抗对齐**：修复感知与推理漂移。
  3. **难度过滤**：筛选合适的RL训练集。
  4. **RLVR**：策略最终优化。

**Key Findings:**

- We introduce PRISM, a three-stage pipeline that mitigates this drift by inserting an explicit distribution-alignment stage between SFT and RLVR.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.28123v1)
- [arXiv](https://arxiv.org/abs/2604.28123v1)

---

