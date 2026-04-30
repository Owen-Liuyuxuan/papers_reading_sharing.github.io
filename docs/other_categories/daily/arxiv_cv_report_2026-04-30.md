time: 20260430

# Arxiv Computer Vision Papers - 2026-04-30

## Executive Summary

以下是针对2026年4月29日arXiv计算机视觉领域10篇论文的执行摘要，旨在为忙碌的研究人员快速提炼关键进展与趋势。

---

### 一、主要主题与趋势观察

本日论文呈现三大核心趋势：
1. **多模态与具身智能的深度融合**：多篇工作聚焦于将视觉-语言模型（VLM）或基础模型与具身智能（如机器人操作、导航）结合，强调从“感知”向“行动”的跨越（如 #1, #2, #7, #10）。
2. **世界模型与空间推理的兴起**：世界模型（World Model）被用于增强模型对动态场景的理解与规划能力，尤其在视频预测、空间时序推理方面（#3, #7, #10）。
3. **基础模型的应用与蒸馏**：视觉基础模型（如ViT、VFM）被广泛用作骨干或教师，通过知识蒸馏、因果推理或特征对齐等技术，适配到特定下游任务（如域泛化 #8、AI生成图像检测 #9、边缘检测 #6）。

### 二、特别重要或创新的论文

- **#1 “GLM-5V-Turbo”**：声称是首个为多模态智能体（Multimodal Agents）设计的原生基础模型。其“原生”设计（而非简单拼接视觉与语言模块）可能奠定下一代多模态交互的范式，价值极高。
- **#3 “World2VLM”**：创新地将世界模型中的“想象”（即预测的未来状态）蒸馏进VLM，用于动态空间推理。这巧妙绕过了VLM在时间序列推理上的固有局限，思路新颖。
- **#10 “Unified 4D World Action Modeling”**：提出统一建模4D（3D+时间）世界与动作，并引入异步去噪策略，有效解决了视频先验与动作生成间的对齐问题，对机器人控制与自动驾驶具参考意义。

### 三、新兴研究方向与技术

- **分层/层次化规划用于零样本导航**：如 #2 “Three-Step Nav” 将全局-局部导航分层，结合语言指令与视觉反馈，推动零样本导航迈向实用。
- **图神经网络赋能多模态对齐**：如 #4 使用图结构进行语义校准，应对无人机RGB-T图像之间的不对齐问题，提示图网络仍然是解决模态鸿沟的有效工具。
- **打破刚性的3D异常检测**：如 #5 首次关注“可动关节物体”的3D异常检测，挑战了传统假设物体刚性的设定，对工业检测与自动驾驶场景中的动态目标更具现实意义。
- **AI生成图像检测轻量级化**：如 #9 提出利用Vision Foundation Model（VFM）中间层patch token特征进行检测，而非仅用最后分类头，为低算力场景提供了新的高效方案。

### 四、建议优先精读的论文

1. **#1 (GLM-5V-Turbo)**：对多模态基础模型、智能体系统研究者必读，可能定义未来方向。
2. **#3 (World2VLM)**：对VLM改进、世界模型应用、动态场景理解感兴趣的读者首选。
3. **#5 (Articulated 3D Anomaly Detection)**：对3D视觉、异常检测、非刚性物体分析方向的研究者，是重要的差异化工作。
4. **#10 (Unified 4D World Action Modeling)**：对机器人学习、视频到动作生成、时序建模有需求的读者，该文提出的统一框架值得关注。
5. **#7 (STARRY)**：专注于机器人操作的动作-时空世界模型，实用性较强，适合具身智能研究者。

---

**总结**：本日论文池反映出计算机视觉正快速向“具身化”、“因果推理”和“动态世界建模”前进。基础模型的多模态原生设计（#1）与世界模型的蒸馏技巧（#3）是最大的跨领域亮点，值得深入研读。

---

## Table of Contents

1. [GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](#2604.26752v1)
2. [Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation](#2604.26946v1)
3. [World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning](#2604.26934v1)
4. [Graph-based Semantic Calibration Network for Unaligned UAV RGBT Image Semantic Segmentation and A Large-scale Benchmark](#2604.26893v1)
5. [Breaking the Rigid Prior: Towards Articulated 3D Anomaly Detection](#2604.26868v1)
6. [Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation](#2604.26857v1)
7. [STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation](#2604.26848v1)
8. [Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization](#2604.26820v1)
9. [TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection](#2604.26772v1)
10. [Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising](#2604.26694v1)

---

## Papers

<a id='2604.26752v1'></a>
## [GLM-5V-Turbo: Toward a Native Foundation Model for Multimodal Agents](https://arxiv.org/abs/2604.26752v1)

**Authors:** GLM-V Team,  :, Wenyi Hong, Xiaotao Gu, Ziyang Pan, Zhen Yang, Yuting Wang, Yue Wang, Yuanchang Yue, Yu Wang, Yanling Wang, Yan Wang, Xijun Liu, Wenmeng Yu, Weihan Wang, Wei Li, Shuaiqi Duan, Sheng Yang, Ruiliang Lv, Mingdao Liu, Lihang Pan, Ke Ning, Junhui Ji, Jinjiang Wang, Jing Chen, Jiazheng Xu, Jiale Zhu, Jiale Cheng, Ji Qi, Guobing Gan, Guo Wang, Cong Yao, Zijun Dou, Zihao Zhou, Zihan Wang, Zhiqi Ge, Zhijie Li, Zhenyu Hou, Zhao Xue, Zehui Wang, Zehai He, Yusen Liu, Yukuo Cen, Yuchen Li, Yuan Wang, Yijian Lu, Yanzi Wang, Yadong Xue, Xinyu Zhang, Xinyu Liu, Wenkai Li, Tianyu Tong, Tianshu Zhang, Shengdong Yan, Qinkai Zheng, Mingde Xu, Licheng Bao, Jiaxing Xu, Jiaxin Fan, Jiawen Qian, Jiali Chen, Jiahui Lin, Haozhi Zheng, Haoran Wang, Haochen Li, Fan Yang, Dan Zhang, Chuangxin Zhao, Chengcheng Wu, Boyan Shi, Bowei Jia, Baoxu Wang, Peng Zhang, Debing Liu, Bin Xu, Juanzi Li, Minlie Huang, Yuxiao Dong, Jie Tang

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

We present GLM-5V-Turbo, a step toward native foundation models for multimodal agents. As foundation models are increasingly deployed in real environments, agentic capability depends not only on language reasoning, but also on the ability to perceive, interpret, and act over heterogeneous contexts such as images, videos, webpages, documents, GUIs. GLM-5V-Turbo is built around this objective: multimodal perception is integrated as a core component of reasoning, planning, tool use, and execution, rather than as an auxiliary interface to a language model. This report summarizes the main improvements behind GLM-5V-Turbo across model design, multimodal training, reinforcement learning, toolchain expansion, and integration with agent frameworks. These developments lead to strong performance in multimodal coding, visual tool use, and framework-based agentic tasks, while preserving competitive text-only coding capability. More importantly, our development process offers practical insights for building multimodal agents, highlighting the central role of multimodal perception, hierarchical optimization, and reliable end-to-end verification.

**Analysis:**

### 1. 摘要翻译
我们推出了GLM-5V-Turbo，这是迈向多模态智能体原生基础模型的重要一步。随着基础模型在现实环境中的广泛部署，智能体能力不仅依赖于语言推理，还取决于感知、解释及操作异构上下文（如图像、视频、网页、文档、GUI）的能力。GLM-5V-Turbo的核心设计理念是将多模态感知整合为推理、规划、工具使用和执行的有机组成部分，而非仅仅作为语言模型的辅助接口。本报告总结了GLM-5V-Turbo在模型设计、多模态训练、强化学习、工具链扩展及智能体框架集成方面的改进，在多模态编码、视觉工具使用和智能体任务上表现出强大性能，同时保留了极具竞争力的纯文本编码能力。更重要的是，该开发流程强调了多模态感知、分层优化和可靠的端到端验证在构建智能体中的核心作用。

### 2. 方法动机分析
- **驱动力**：旨在构建真正具备“感知-规划-执行”闭环的通用智能体，解决现有模型在处理复杂视觉环境（如GUI、动态网页）时感知与执行脱节的问题。
- **痛点**：现有模型多将视觉作为辅助，多模态特征与逻辑推理耦合度低；且针对长程智能体任务的端到端训练往往伴随着严重的跨域干扰与训练不稳定性。
- **研究假设**：通过原生多模态设计与分层强化学习（Hierarchical RL），能够实现感知能力与高级推理能力的深度协同，从而提升模型在复杂现实任务中的泛化边界。

### 3. 方法设计详解
- **核心组件**：
  - **CogViT 视觉编码器**：采用两阶段训练，第一阶段通过蒸馏SigLIP2（语义）与DINOv3（纹理）实现高质量表示；第二阶段引入NaFlex机制处理变分辨率图像，并结合大规模多语言数据进行对比对齐。
  - **多模态多Token预测 (MMTP)**：这是本文的一大亮点。相比将视觉Embedding直接传入头部，GLM-5V-Turbo采用特殊`<|image|>` Token占位符设计。该设计极大降低了跨设备通信开销，通过轻量级Head避免了视觉与文本特征分布不一致带来的优化压力，显著提升了训练稳定性。
- **流程总结**：
  1. **感知**：CogViT处理多模态输入，生成视觉特征。
  2. **决策**：基于MMTP将视觉特征与语言特征统一在Embedding空间。
  3. **协同**：利用“分层优化”策略，将训练任务解耦，分阶段进行感知、单步决策及长程轨迹预测的RL优化。
  4. **执行**：通过官方提供的“Official Skills”与外部框架（Claude Code/AutoClaw）交互，实现复杂动作。

### 4. 方法对比分析
- **本质区别**：从传统的“语言模型+视觉适配器”转向“原生感知智能体”。MMTP设计在系统效率与性能之间找到了平衡点。
- **创新贡献**：引入分层协作RL优化，有效缓解了跨任务干扰，并提出了ImageMining benchmark，量化了智能体“深度视觉搜索”的能力。

### 5. 实验分析
- **关键结论**：在多模态编码（Design2Code: 94.8）、GUI交互（OSWorld: 62.3）等任务上表现优异；在保持高水平多模态能力的同时，CC-RepoExploration等纯文本基准测试甚至有所超越。
- **局限**：在极长程任务中，依然面临上下文窗口压力和视觉信息丢失问题。

### 6. 实用指南
- **开源/生态**：模型及相关工具链集成在 Z.ai 环境中，官方Skill可通过 [ClawHub](https://clawhub.ai) 获取。
- **迁移建议**：对于需要处理复杂视觉UI的任务，建议参考其“分层RL”思想，先在感知/单步动作任务上训练，再扩展至复杂长程逻辑，避免单一模型“直接对齐”导致的训练震荡。

### 7. 总结
- **核心思想**：通过原生感知与多任务分层协同强化学习，打造具备真实世界行动力的多模态智能体。
- **速记版pipeline**：
  1. 视觉输入通过CogViT提取特征。
  2. 使用`<|image|>`占位符将特征与语言空间对齐。
  3. 通过多任务分层RL优化感知与行动。
  4. 结合工具调用执行具体网页/UI操作。

**Key Findings:**

- We present GLM-5V-Turbo, a step toward native foundation models for multimodal agents.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26752v1)
- [arXiv](https://arxiv.org/abs/2604.26752v1)

---

<a id='2604.26946v1'></a>
## [Three-Step Nav: A Hierarchical Global-Local Planner for Zero-Shot Vision-and-Language Navigation](https://arxiv.org/abs/2604.26946v1)

**Authors:** Wanrong Zheng, Yunhao Ge, Laurent Itti

**Published:** 2026-04-29

**Categories:** cs.CV, cs.RO

**Abstract:**

Breakthrough progress in vision-based navigation through unknown environments has been achieved by using multimodal large language models (MLLMs). These models can plan a sequence of motions by evaluating the current view at each time step against the task and goal given to the agent. However, current zero-shot Vision-and-Language Navigation (VLN) agents powered by MLLMs still tend to drift off course, halt prematurely, and achieve low overall success rates. We propose Three-Step Nav to counteract these failures with a three-view protocol: First, "look forward" to extract global landmarks and sketch a coarse plan. Then, "look now" to align the current visual observation with the next sub-goal for fine-grained guidance. Finally, "look backward" audits the entire trajectory to correct accumulated drift before stopping. Requiring no gradient updates or task-specific fine-tuning, our planner drops into existing VLN pipelines with minimal overhead. Three-Step Nav achieves state-of-the-art zero-shot performance on the R2R-CE and RxR-CE dataset. Our code is available at https://github.com/ZoeyZheng0/3-step-Nav.

**Analysis:**

这份论文提出了 **Three-Step Nav**，一种用于零样本视觉-语言导航（VLN-CE）的分层全局-局部规划框架。以下是对该方法的深度分析：

### 1. 摘要翻译
通过多模态大语言模型（MLLMs），视觉导航在未知环境中取得了显著进展。然而，现有的零样本VLN代理在面对长路径时，往往因为上下文窗口限制及漂移问题导致偏离航线或过早停止。为此，我们提出了 Three-Step Nav，一种包含“向前看（look forward）”、“现在看（look now）”和“向后看（look backward）”的三阶段协议。该方法首先通过向前看提取全局路标并规划粗略路径；其次通过现在看将当前视觉观察与子目标对齐；最后通过向后看审计已执行路径，在停止前修正累积漂移。该方案无需梯度更新或特定任务微调，在R2R-CE和RxR-CE数据集上取得了零样本导航的SOTA性能。

### 2. 方法动机分析
- **驱动力**：旨在解决连续空间导航中，因缺乏显式路径规划及长程依赖导致的性能瓶颈，实现无需训练的高鲁棒性零样本导航。
- **痛点**：现有局部思维链（CoT）方法仅依赖当前观测，极易在长任务中产生“幻觉”或被环境中无关物体干扰，且缺乏对已执行轨迹的自我纠错机制。
- **核心假设**：导航应是一个包含“全局规划-细粒度执行-事后审计”的闭环迭代过程，通过引入轨迹层面的全局验证，能显著抑制累积误差。

### 3. 方法设计详解
- **pipeline（三阶段机制）**：
  1. **Look Forward (全局规划)**：在运动前，利用MLLM将完整指令分解为带全局路标的原子子指令序列，构建出“路线图”。
  2. **Look Now (局部执行)**：在当前子目标下，结合视觉观察、候选视点及历史记录，利用MLLM预测最优运动方向（动作空间采样），并判断是否达到局部子目标距离阈值。
  3. **Look Backward (轨迹审计)**：在子任务完成后，对已执行路径进行紧凑的文本回放，向MLLM询问：“轨迹是否满足指令？”或“是否存在遗漏路标？”。
- **Adaptive Judge（元技能模块）**：引入四个关键技能：
    - *Continue*：确认满足，继续下一目标。
    - *Stay*：信号模糊，保持原地重试。
    - *Backtrack*： audit失败，回滚至最近可靠节点。
    - *Look-around*：高不确定时，探测周边节点后重评价。

### 4. 方法对比分析
- **本质区别**：从传统的“贪婪式单步推理”转变为“分层式的轨迹可控推理”。它不仅关注“下一步往哪走”，更关注“之前的走法是否符合原意”。
- **创新点**：引入了 Look Backward 审计机制和四个自主修正元技能（Backtrack/Look-around等），这使得LLM在无需权重更新的情况下具备了“后悔”和“反思”能力。

### 5. 实验分析
- **验证方法**：在连续环境下的R2R-CE和RxR-CE数据集进行零样本测试。
- **关键结果**：相比最佳基线，NE（导航误差）降低约15%，SPL（路径加权成功率）提升12%。
- **局限**：对底层MLLM的推理能力依赖极强，计算延迟较高；暂不支持处理动态障碍物。

### 6. 实用指南
- **开源情况**：代码已开源（github.com/ZoeyZheng0/3-step-Nav）。
- **实现建议**：该方法模块化程度极高，可直接插入任何现有基于LLM的导航流。关键在于设计准确的Prompt，让模型能准确理解“回放路径”的语义描述。
- **迁移可能**：可轻松迁移至具身智能中的长序列操作任务（如长程任务拆解、烹饪机器人），只需替换对应任务的“路标”定义即可。

### 7. 总结
- **核心思想**：通过分层规划与轨迹审计实现零样本的自我校正导航。
- **速记版pipeline**：
    1. 拆解指令为全局路标图；
    2. 基于当前视点执行微小步进；
    3. 事后审计路径以防偏移；
    4. 依据不确定性触发回溯或巡视。

**Key Findings:**

- We propose Three-Step Nav to counteract these failures with a three-view protocol: First, "look forward" to extract global landmarks and sketch a coarse plan.
- Three-Step Nav achieves state-of-the-art zero-shot performance on the R2R-CE and RxR-CE dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26946v1)
- [arXiv](https://arxiv.org/abs/2604.26946v1)

---

<a id='2604.26934v1'></a>
## [World2VLM: Distilling World Model Imagination into VLMs for Dynamic Spatial Reasoning](https://arxiv.org/abs/2604.26934v1)

**Authors:** Wanyue Zhang, Wenxiang Wu, Wang Xu, Jiaxin Luo, Helu Zhi, Yibin Huang, Shuo Ren, Zitao Liu, Jiajun Zhang

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

Vision-language models (VLMs) have shown strong performance on static visual understanding, yet they still struggle with dynamic spatial reasoning that requires imagining how scenes evolve under egocentric motion. Recent efforts address this limitation either by scaling spatial supervision with synthetic data or by coupling VLMs with world models at inference time. However, the former often lacks explicit modeling of motion-conditioned state transitions, while the latter incurs substantial computational overhead. In this work, we propose World2VLM, a training framework that distills spatial imagination from a generative world model into a vision-language model. Given an initial observation and a parameterized camera trajectory, we use a view-consistent world model to synthesize geometrically aligned future views and derive structured supervision for both forward (action-to-outcome) and inverse (outcome-to-action) spatial reasoning. We post-train the VLM with a two-stage recipe on a compact dataset generated by this pipeline and evaluate it on multiple spatial reasoning benchmarks. World2VLM delivers consistent improvements over the base model across diverse benchmarks, including SAT-Real, SAT-Synthesized, VSI-Bench, and MindCube. It also outperforms the test-time world-model-coupled methods while eliminating the need for expensive inference-time generation. Our results suggest that world models can serve not only as inference-time tools, but also as effective training-time teachers, enabling VLMs to internalize spatial imagination in a scalable and efficient manner.

**Analysis:**

### 1. 摘要翻译
视觉语言模型（VLMs）在静态视觉理解方面表现出色，但在处理需要想象场景在自中心运动下如何演变的动态空间推理任务时仍显吃力。现有方案要么通过合成数据扩展空间监督，要么在推理阶段耦合世界模型，前者缺乏对运动条件下的状态转换的显式建模，后者则带来了巨大的计算开销。为此，我们提出了 **World2VLM**，一种通过将生成式世界模型的空间想象力“蒸馏”到视觉语言模型中的训练框架。给定初始观测和参数化的相机轨迹，我们利用视图一致的世界模型合成几何对齐的未来视图，并衍生出用于前向（动作到结果）和逆向（结果到动作）空间推理的结构化监督。我们在一个由该流水线生成的紧凑数据集上对VLM进行了两阶段微调，并在多个空间推理基准上进行了评估。World2VLM在无需昂贵的推理时生成的条件下，在各项基准测试中均持续优于基准模型及推理时耦合世界模型的方法。

### 2. 方法动机分析
*   **驱动力**：解决VLM在空间理解中的“静态陷阱”，使其具备真正的“空间想象力”，即理解动作导致的视角变化和场景演变。
*   **痛点**：现有方法要么仅仅是在静态图上做功，缺乏对动作后果的物理预测；要么在推理时实时调用世界模型（如MindJourney），导致部署成本极大且不可控。
*   **研究假设**：空间想象能力可以通过离线蒸馏的方式“内化”到VLM参数中，使其在无需外部世界模型参与的情况下，具备根据初始观测和动作预测未来状态的能力。

### 3. 方法设计详解
World2VLM通过三个核心模块实现空间能力的内化：
1.  **世界模型引导的转换构建（Transition Construction）**：使用预训练的世界模型（如SVC或HY-WorldPlay）作为离线教师。给定锚点观测 $s_t$ 和相机动作 $a_t$，合成几何一致的未来视图 $s_{t+1}$，并通过检测器获取对象元数据，形成带语义的动作-视图对。
2.  **双向空间监督流水线（Bidirectional Task Suite）**：
    *   **逆向推理（Inverse）**：给定 $s_t$ 和 $s_{t+1}$，推理动作 $a_t$（如距离、方向、序列预测）。
    *   **前向预测（Forward）**：给定 $s_t$ 和 $a_t$，预测 $s_{t+1}$ 的后果（如物体边界框变换、可见性判断、身份一致性）。
    *   这种“双向约束”强制模型从解释过去（逆向）和预测未来（前向）两个维度建立空间映射。
3.  **两阶段训练策略（Two-Stage Training）**：
    *   **SFT阶段**：利用生成的监督数据进行参数高效微调（LoRA）。
    *   **GRPO阶段**：通过组相对策略优化（GRPO）进行强化学习，引入格式规范、数值精度、空间逻辑和轨迹一致性奖励，进一步强化模型的结构化输出能力。

### 4. 方法对比分析
*   **本质区别**：将世界模型的使用从“推理时实时生成（Test-time）”转变为“训练时教师指导（Train-time Distillation）”。
*   **创新贡献**：提出了一套“前向+逆向”的双向空间监督框架，成功将复杂环境的物理规律蒸馏进VLM的权重中。
*   **适用场景**：适用于机器人导航、 embodied AI、以及需要理解相机视角变换和对象追踪的视觉任务。

### 5. 实验分析
*   **验证方法**：在SAT-Real、SAT-Synth、VSI-Bench、MindCube四个基准上进行对比。
*   **关键结果**：World2VLM显著超越了基准Qwen2.5-VL，并在所有类别上不仅打败了不加蒸馏的模型，也打败了推理时调用世界模型的基线。
*   **优势**：极高的部署效率（无需推理时调用生成模型）；通过RL强化了数值敏感度和逻辑连贯性。
*   **局限**：模型效果受限于教师模型（世界模型）的生成质量，若教师生成存在伪影，则会引入噪声监督。

### 6. 实用指南
*   **开源情况**：代码和数据已开源，可见论文主页或GitHub仓库。
*   **实现细节**：GRPO阶段奖励设计是核心，通过定义 `rfmt` (格式)、 `rsem` (正确性)、 `rnum` (数值精度) 等 reward term，可有效约束模型的回答输出。
*   **迁移可能**：该框架是模型无关的，可迁移至任何需要空间推理的VLM，只需替换教师生成器，并将任务转化为对应的 QA 模板。

### 7. 总结
*   **核心思想**：离线蒸馏物理世界规律，使VLM无需推理即具空间预判能力。
*   **速记版Pipeline**：
    1. 选锚点图片与相机动作，用世界模型离线生成未来视图。
    2. 将视图对转化成“预测动作”和“预测后果”的训练问答对。
    3. 进行监督微调（SFT）建立空间映射。
    4. 通过强化学习（GRPO）规范化输出格式并校准空间数值预测。

**Key Findings:**

- In this work, we propose World2VLM, a training framework that distills spatial imagination from a generative world model into a vision-language model.
- It also outperforms the test-time world-model-coupled methods while eliminating the need for expensive inference-time generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26934v1)
- [arXiv](https://arxiv.org/abs/2604.26934v1)

---

<a id='2604.26893v1'></a>
## [Graph-based Semantic Calibration Network for Unaligned UAV RGBT Image Semantic Segmentation and A Large-scale Benchmark](https://arxiv.org/abs/2604.26893v1)

**Authors:** Fangqiang Fan, Zhicheng Zhao, Xiaoliang Ma, Chenglong Li, Jin Tang

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

Fine-grained RGBT image semantic segmentation is crucial for all-weather unmanned aerial vehicle (UAV) scene understanding. However, UAV RGBT semantic segmentation faces two coupled challenges: cross-modal spatial misalignment caused by sensor parallax and platform vibration, and severe semantic confusion among fine-grained ground objects under top-down aerial views. To address these issues, we propose a Graph-based Semantic Calibration Network (GSCNet) for unaligned UAV RGBT image semantic segmentation. Specifically, we design a Feature Decoupling and Alignment Module (FDAM) that decouples each modality into shared structural and private perceptual components and performs deformable alignment in the shared subspace, enabling robust spatial correction with reduced modality appearance interference. Moreover, we propose a Semantic Graph Calibration Module (SGCM) that explicitly encodes the hierarchical taxonomy and co-occurrence regularities among ground-object categories in UAV scenes into a structured category graph, and incorporates these priors into graph-attention reasoning to calibrate predictions of visually similar and rare categories.In addition, we construct the Unaligned RGB-Thermal Fine-grained (URTF) benchmark, to the best of our knowledge, the largest and most fine-grained benchmark for unaligned UAV RGBT image semantic segmentation, containing over 25,000 image pairs across 61 categories with realistic cross-modal misalignment. Extensive experiments on URTF demonstrate that GSCNet significantly outperforms state-of-the-art methods, with notable gains on fine-grained categories. The dataset is available at https://github.com/mmic-lcl/Datasets-and-benchmark-code.

**Analysis:**

### 1. 摘要翻译
精细化RGB-T图像语义分割对于全天候无人机（UAV）场景理解至关重要。然而，无人机RGBT语义分割面临两个相互耦合的挑战：传感器视差与平台振动导致的跨模态空间错位，以及俯视视角下细粒度地物间的严重语义混淆。针对这些问题，本文提出了用于非对齐无人机RGBT语义分割的图语义校准网络（GSCNet）。我们设计了特征解耦与对齐模块（FDAM），将各模态解耦为共享结构组件和私有感知组件，并在共享子空间进行可变形对齐，以减少模态外观差异的影响；同时提出了语义图校准模块（SGCM），通过结构化类别图显式编码分类层级和共现规律，并利用图注意力推理来校准视觉相似及稀疏类别的预测。此外，我们构建了目前最大、最精细的非对齐无人机RGBT基准数据集URTF。实验表明，GSCNet显著优于现有前沿方法，特别是在细粒度类别上表现优异。

### 2. 方法动机分析
*   **驱动力**：解决无人机实际部署中无法实现像素级对齐的严峻现实问题，并应对由俯视视角引发的细粒度类别语义混淆。
*   **现有痛点**：传统方法假设模态间已对齐，或简单进行特征融合，导致跨模态空间错位引入“鬼影”和边界模糊；且现有的特征融合方式无法处理长尾分布中稀疏类别（如电线杆、路灯）的识别困境。
*   **研究假设**：通过“先解耦后对齐”策略，可以在结构一致的共享子空间修正错位；通过“类间关系图推理”，可以利用层次与共现先验弥补局部特征对稀疏类别的判别力不足。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **特征解耦与对齐（FDAM）**：通过非对称解耦（AFD）将RGB/T特征分解为共享结构特征（结构信息稳定）和私有感知特征（纹理/辐射差异信息）。在共享空间进行光照自适应对齐（IAA），通过预测偏移量进行可变形卷积补偿，并将对齐后的几何变换应用到私有分支。
    2.  **特征融合**：将对齐后的共享特征与对应的私有特征进行压缩与级联。
    3.  **语义图校准（SGCM）**：构建包含层级与共现先验的类别关系图。利用 base logits 对输入图像进行池化得到类别节点特征，通过图注意力网络（GAT）进行推理，将校准后的图logits（$L_g$）与基础logits（$L_0$）进行加权融合。
*   **关键点**：IAA模块中的光照路由器（$\lambda$）动态调整RGB和热成像在对齐时的权重，解决了昼夜模态重要性差异问题。

### 4. 方法对比分析
*   **本质区别**：从传统的“隐式融合”转变为“显式空间校准+语义图推理”的联合框架。
*   **创新贡献**：
    1.  FDAM首次实现了在解耦后的共享空间进行几何对齐，有效规避了模态差异对偏移预测的干扰。
    2.  SGCM将层次先验（taxonomy）与共现先验（co-occurrence）引入GAT，为长尾稀疏类别提供了额外的语义上下文支撑。
*   **适用场景**：适用于存在视角差、运动模糊、多模态异质性明显的复杂无人机遥感环境。

### 5. 实验分析
*   **结论**：在URTF上，GSCNet的mIoU达到71.04%，显著超过次优方法（68.31%）。在Tail-16（稀疏类别）上提升尤为明显（60.17% vs 55.54%）。
*   **优势**：极强的抗错位能力和处理细粒度长尾类别的推理鲁棒性。
*   **局限**：模型参数量（160M）和FPS（16.6）对超轻量级无人机端侧推理仍有挑战。

### 6. 实用指南
*   **开源地址**：https://github.com/mmic-lcl/Datasets-and-benchmark-code
*   **实现建议**：在微调时，应重点关注解耦权重（$\lambda_{dis}=0.1$）和融合系数（$\gamma=0.85$）的调节。建议在使用新数据集时，重新构建基于该数据集的共现先验矩阵（$A_C$）。

### 7. 总结
*   **核心思想**：通过结构化解耦与图注意力推理，实现异构模态的高鲁棒性对齐与语义校准。
*   **速记版pipeline**：
    1.  特征拆分为共享结构与私有细节；
    2.  在共享空间执行光照自适应的空间对齐；
    3.  利用层次结构图通过注意力机制修正分类决策。

**Key Findings:**

- To address these issues, we propose a Graph-based Semantic Calibration Network (GSCNet) for unaligned UAV RGBT image semantic segmentation.
- Moreover, we propose a Semantic Graph Calibration Module (SGCM) that explicitly encodes the hierarchical taxonomy and co-occurrence regularities among ground-object categories in UAV scenes into a structured category graph, and incorporates these priors into graph-attention reasoning to calibrate predictions of visually similar and rare categories.In addition, we construct the Unaligned RGB-Thermal Fine-grained (URTF) benchmark, to the best of our knowledge, the largest and most fine-grained benchmark for unaligned UAV RGBT image semantic segmentation, containing over 25,000 image pairs across 61 categories with realistic cross-modal misalignment.
- Extensive experiments on URTF demonstrate that GSCNet significantly outperforms state-of-the-art methods, with notable gains on fine-grained categories.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26893v1)
- [arXiv](https://arxiv.org/abs/2604.26893v1)

---

<a id='2604.26868v1'></a>
## [Breaking the Rigid Prior: Towards Articulated 3D Anomaly Detection](https://arxiv.org/abs/2604.26868v1)

**Authors:** Jinye Gan, Bozhong Zheng, Xiaohao Xu, Junye Ren, Zixuan Zhang, Na Ni, Yingna Wu

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

Existing 3D anomaly detection methods are built on a rigid prior: normal geometry is pose-invariant and can be canonicalized through registration or alignment. This prior does not hold for articulated objects with hinge or sliding joints, where valid pose changes induce structured geometric variations that cannot be collapsed to a single canonical template, causing pose-induced deformations to be misidentified as anomalies while true structural defects are obscured. No existing benchmark addresses this challenge. We introduce ArtiAD, the first large-scale benchmark for articulated 3D anomaly detection, comprising 15,229 point clouds across 39 object categories with dense joint-angle variations and six structural anomaly types. Each sample is annotated with its joint configuration and part-level motion labels, enabling explicit disentanglement of pose-induced geometry from structural defects. ArtiAD also provides a seen/unseen articulation split to evaluate both interpolation and extrapolation to novel joint configurations. We propose Shape-Pose-Aware Signed Distance Field (SPA-SDF), a baseline that replaces the rigid prior with a continuous pose-conditioned implicit field, factorized into an articulation-independent structural prior and a Fourier-encoded joint embedding. At inference, the articulation state is recovered by minimizing reconstruction energy, and anomalies are identified as point-wise deviations from the learned manifold. SPA-SDF achieves 0.884 object-level AUROC on seen configurations and 0.874 on unseen configurations, substantially outperforming all rigid-based baselines. Our code and benchmark will be publicly released to facilitate future research.

**Analysis:**

以下是对《Breaking the Rigid Prior: Towards Articulated 3D Anomaly Detection》一文的深度分析。

### 1. 摘要翻译
现有的3D异常检测方法依赖于“刚性先验”（Rigid Prior）：即正常几何体是姿态不变的，可以通过标准化处理坍缩至单个模板。然而，该先验在具有铰链或滑动关节的铰接物体上失效，因为姿态变化会引起结构化几何变动，导致姿态诱导的形变被误判为异常，而真实的结构缺陷被掩盖。为此，我们提出了ArtiAD，首个针对铰接式3D异常检测的大规模基准，包含39个类别、15,229个点云及多种结构异常类型。此外，我们提出了一种基于姿态条件隐式场的SPA-SDF基准方法，将结构先验与关节配置解耦。在推理时，通过最小化重构能量来恢复关节状态。实验表明，SPA-SDF在可见和未见关节配置上均大幅优于传统方法。

### 2. 方法动机分析
*   **驱动力**：铰接物体在不同关节角度下呈现截然不同的几何形态，且均为正常（valid）。现有的刚性对齐或预训练方法无法处理这种“由运动产生的非刚性几何变化”。
*   **痛点**：现有方法将“关节运动导致的正常几何变动”视为需要对齐的噪声，这在铰接场景下是错误的。对齐操作会掩盖真实缺陷，或因强行对齐导致正常的运动被误报。
*   **核心直觉**：正常几何形态应当表示为关节配置的函数，而非固定模板。因此，需要学习一个“姿态条件下的正态分布”。

### 3. 方法设计详解
*   **pipeline总结**：
    1.  **输入与编码**：将点云 $X$ 及关节状态 $\psi$ 输入。 $\psi$ 通过多频傅里叶特征编码， $X$ 的坐标通过位置编码（PE）。
    2.  **结构编码（Stage 1）**：利用形状嵌入网络将点云坐标和形状潜码 $\phi$ 映射为与关节无关的结构化嵌入 $h_s$（通过Skip connection保留细节）。
    3.  **姿态条件解码（Stage 2）**：将结构特征 $h_s$ 与关节嵌入 $e_\psi$ 融合，预测符号距离函数（SDF）值 $\hat{s}$。
    4.  **训练监督**：仅使用正常数据进行SDF重构，并辅以部件分类正则化（Part-Aware Regularization）以增强几何一致性。
    5.  **异常推理**：未知关节状态 $\psi^*$ 通过最小化重构能量（SDF误差）反求。计算异常分数为重构残差，残差高处即为潜在异常。
*   **关键公式意义**：$f(x; \phi, \gamma(\psi))$，该设计将形状表示分解为类级别的结构潜码（不变部分）和傅里叶编码的关节状态（动态部分），实现了解耦。

### 4. 方法对比分析
*   **本质区别**：从“静态匹配”转向“动态生成”。现有方法试图将所有物体对齐到一致，SPA-SDF则动态预测在给定姿态下的正常几何应该是怎样的。
*   **创新贡献**：提出了首个铰接式3D异常检测数据集；提出了将关节先验作为条件注入Implicit Field的框架。
*   **适用场景**：所有具有明确 kinematic 结构（旋转、平移）的工业零部件检测。

### 5. 实验分析
*   **关键结论**：在所有39个类别上，SPA-SDF在seen和unseen配置下的Obj-AUROC均达到了0.87+，大幅领先基于刚性对齐的方法（如PatchCore, PO3AD）。
*   **主要优势**：极强的姿态归纳与推断能力，有效减少了铰接运动带来的虚警。
*   **主要局限**：目前的关节状态恢复依赖于exhaustive grid search（暴力网格搜索），计算量大且难以扩展到多自由度（multi-DoF）系统。

### 6. 实用指南
*   **关键实现**：多频傅里叶特征（$L=16$）是处理周期性运动的关键，不可省略。训练阶段的“部件正则化”对保持物体各部分结构完整性至关重要。
*   **迁移建议**：该框架可直接迁移至任何具有运动学约束的物体建模任务（如机器人抓取检测、动态物体点云补全）。

### 7. 总结
*   **核心思想**：将刚性对齐替换为连续的姿态条件隐式场。
*   **速记版pipeline**：
    1.  对关节角度进行傅里叶特征编码。
    2.  将输入点云解耦为结构特征和姿态特征。
    3.  利用解码器预测当前姿态下的SDF值。
    4.  通过最小化重构误差自动拟合测试样本的关节状态。
    5.  计算SDF重构残差作为异常得分。

**Key Findings:**

- We introduce ArtiAD, the first large-scale benchmark for articulated 3D anomaly detection, comprising 15,229 point clouds across 39 object categories with dense joint-angle variations and six structural anomaly types.
- ArtiAD also provides a seen/unseen articulation split to evaluate both interpolation and extrapolation to novel joint configurations.
- We propose Shape-Pose-Aware Signed Distance Field (SPA-SDF), a baseline that replaces the rigid prior with a continuous pose-conditioned implicit field, factorized into an articulation-independent structural prior and a Fourier-encoded joint embedding.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26868v1)
- [arXiv](https://arxiv.org/abs/2604.26868v1)

---

<a id='2604.26857v1'></a>
## [Edge AI for Automotive Vulnerable Road User Safety: Deployable Detection via Knowledge Distillation](https://arxiv.org/abs/2604.26857v1)

**Authors:** Akshay Karjol, Darrin M. Hanna

**Published:** 2026-04-29

**Categories:** cs.CV, cs.LG, cs.RO, eess.IV

**Abstract:**

Deploying accurate object detection for Vulnerable Road User (VRU) safety on edge hardware requires balancing model capacity against computational constraints. Large models achieve high accuracy but fail under INT8 quantization required for edge deployment, while small models sacrifice detection performance. This paper presents a knowledge distillation (KD) framework that trains a compact YOLOv8-S student (11.2M parameters) to mimic a YOLOv8-L teacher (43.7M parameters), achieving 3.9x compression while preserving quantization robustness. We evaluate on full-scale BDD100K (70K training images) with Post-Training Quantization to INT8. The teacher suffers catastrophic degradation under INT8 (-23% mAP), while the KD student retains accuracy (-5.6% mAP). Analysis reveals that KD transfers precision calibration rather than raw detection capacity: the KD student achieves 0.748 precision versus 0.653 for direct training at INT8, a 14.5% gain at equivalent recall, reducing false alarms by 44% versus the collapsed teacher. At INT8, the KD student exceeds the teacher's FP32 precision (0.748 vs. 0.718) in a model 3.9x smaller. These findings establish knowledge distillation as a requirement for deploying accurate, safety-critical VRU detection on edge hardware.

**Analysis:**

### 1. 摘要翻译
为边缘硬件部署高精度的易受伤害道路使用者（VRU）检测模型，需要在模型容量与计算约束之间取得平衡。大型模型精度高，但在边缘部署所需的INT8量化下表现不佳；小型模型则牺牲了检测性能。本文提出了一种知识蒸馏（KD）框架，训练紧凑型YOLOv8-S学生模型（11.2M参数）模仿YOLOv8-L教师模型（43.7M参数），在保持量化鲁棒性的同时实现3.9倍的压缩。实验在BDD100K数据集上进行。研究发现，教师模型在INT8下会发生灾难性性能退化（mAP下降23%），而蒸馏后的学生模型能显著保持精度（mAP仅下降5.6%）。分析表明，KD并非提升了原始检测容量，而是传递了精度校准能力：在INT8下，KD学生模型比直接训练的模型精确度提升了14.5%，误报率降低了44%。这些发现确立了知识蒸馏对于边缘计算中安全关键型VRU检测的必要性。

### 2. 方法动机分析
*   **驱动力**：解决边缘计算环境下，模型因INT8量化导致的性能“灾难性崩溃”问题，确保VRU安全检测在资源受限设备上的可靠性。
*   **现有痛点**：大规模模型在量化后性能大幅衰减，而小模型由于参数量小，对稀有类别的识别能力不足，二者均无法兼顾精度与部署需求。
*   **研究假设**：知识蒸馏不仅能传递特征表示，还能传递教师模型的“信心校准”（Confidence Calibration），这种校准信息在量化后依然具有鲁棒性。

### 3. 方法设计详解
*   **流程总结**：
    1.  **冻结教师**：使用预训练的YOLOv8-L作为教师，冻结权重。
    2.  **蒸馏训练**：使用YOLOv8-S作为学生，联合优化检测任务损失（$L_{task}$）、逻辑蒸馏损失（$L_{logit}$）和特征蒸馏损失（$L_{feat}$）。
    3.  **量化评估**：通过TensorRT对训练好的学生模型进行INT8后训练量化（PTQ）。
*   **关键公式**：$L_{total} = \alpha \cdot L_{task} + \beta \cdot L_{logit} + \gamma \cdot L_{feat}$。
    *   **任务降噪（Task-dampening）**：作者核心发现是$\alpha=0.5$（下调任务损失权重）能增强学生对教师软标签分布的依赖，这是校准传递的关键。
    *   **逻辑蒸馏**：利用温度系数$T=10$的KL散度，将教师的概率分布软化，提取类间相似性。
    *   **特征蒸馏**：采用L2距离对齐学生和教师的中间层特征图。

### 4. 方法对比分析
*   **本质区别**：本文并未简单追求提升mAP，而是关注量化后的“部署鲁棒性”和“精度校准”。
*   **创新贡献**：首次证明了KD在物体检测任务中，其核心价值在于改善学生模型量化后的信心分布（即减少误报），而非单纯提升模型容量。
*   **适用场景**：适用于任何要求“低误报率”且需部署在边缘端（如Jetson平台）的实时安防或自动驾驶检测系统。

### 5. 实验分析
*   **关键结论**：在INT8条件下，KD学生模型通过校准，实现了比老师FP32版本更高的精确度（0.748 vs 0.718），且误报率显著下降。
*   **主要优势**：极高的计算效率（3.9×压缩）和卓越的量化适应性，极大地降低了安全系统中的误报风险。
*   **主要局限**：KD无法跨越架构带来的性能上限；在大规模数据集上KD产生的额外训练开销较大。

### 6. 实用指南
*   **实现细节**：
    *   关键超参数：$\alpha=0.5, \beta=0.3, \gamma=0.02, T=10$。
    *   量化工具：使用TensorRT的PTQ，建议使用1024张验证集样本进行校准。
*   **迁移建议**：该方法逻辑通用，可直接迁移至YOLOv10或其它检测器。若模型架构差异大，可进一步引入关系型蒸馏损失。

### 7. 总结
*   **核心思想**：通过蒸馏传递信心校准，构建量化鲁棒的小型检测模型。
*   **速记版Pipeline**：
    1. 用大模型作为教师模型冻结权重；
    2. 设置较低的检测任务权重，强迫学生模型专注学习教师的概率输出；
    3. 加入温度因子软化教师预测分布；
    4. 对蒸馏后的轻量模型进行INT8量化部署。

**Key Findings:**

- 0.718) in a model 3.9x smaller.
- These findings establish knowledge distillation as a requirement for deploying accurate, safety-critical VRU detection on edge hardware.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26857v1)
- [arXiv](https://arxiv.org/abs/2604.26857v1)

---

<a id='2604.26848v1'></a>
## [STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation](https://arxiv.org/abs/2604.26848v1)

**Authors:** Yuxuan Tian, Yurun Jin, Bin Yu, Yukun Shi, Hao Wu, Chi Harold Liu, Kai Chen, Cong Huang

**Published:** 2026-04-29

**Categories:** cs.RO

**Abstract:**

Robotic manipulation critically requires reasoning about future spatial-temporal interactions, yet existing VLA policies and world-model-enhanced policies do not fully model action-relevant spatial-temporal interaction structure. We propose STARRY, a world-model-enhanced action-generation policy that aligns spatial-temporal prediction with action generation. STARRY jointly denoises future spatial-temporal latents and action sequences, and introduces Geometry-Aware Selective Attention Modulation to convert predicted depth and end-effector geometry into token-aligned weights for selective action-attention modulation. On RoboTwin 2.0, STARRY achieves 93.82% / 93.30% average success under Clean and Randomized settings. Real-world experiments further improve average success from 42.5% to 70.8% over $π_{0.5}$, demonstrating the effectiveness of action-centric spatial-temporal world modeling for spatial-temporally demanding robotic action generation.

**Analysis:**

这是一份关于论文《STARRY: Spatial-Temporal Action-Centric World Modeling for Robotic Manipulation》的深度分析。

---

### 1. 摘要翻译
机器人操作需要对未来的时空交互进行推理。现有的视觉-语言-动作（VLA）策略和基于世界模型的策略，并未充分建模与动作相关的时空交互结构。为此，我们提出了STARRY，这是一种世界模型增强的动作生成策略，旨在将时空预测与动作生成对齐。STARRY联合去噪未来的时空潜变量和动作序列，并引入了“几何感知选择性注意力调制”（GASAM），将预测的深度和末端执行器几何信息转换为令牌对齐的权重，用于选择性地调制动作注意力。在RoboTwin 2.0基准上，STARRY在常规和随机设置下分别取得了93.82%和93.30%的平均成功率。真实环境实验将平均成功率从42.5%提升至70.8%，验证了面向动作的几何感知时空世界建模在机器人动作生成中的有效性。

### 2. 方法动机分析
- **驱动力**：解决VLA模型在复杂、空间受限的操作任务中缺乏前瞻性和几何推理能力的问题。
- **痛点**：现有世界模型倾向于优化“视觉保真度”而非“控制效用”，导致预测未来虽视觉连贯，但忽略了抓取、碰撞检测等关键空间约束；且通用表征难以区分决策关键区域与背景。
- **研究假设**：机器人世界模型应具备“动作中心（Action-centric）”和“几何扎根（Geometry-grounded）”属性，即模型不仅要预测环境变化，还需明确预测哪些区域对执行特定动作至关重要。

### 3. 方法设计详解
STARRY由四个核心模块组成：理解专家、ST世界模型、几何专家、动作专家。
- **Pipeline**：
    1.  **输入表征**：将多视角RGB-D图像与轨迹投影结合，构成统一的时空表征。
    2.  **联合建模**：利用扩散模型，在同一时域内同步生成未来的时空潜变量（$z_{t+1:t+H}$）和动作序列（$a_{t+1:t+H}$）。
    3.  **几何预测（核心）**：几何专家预测未来的深度图和末端执行器位置，计算场景点与末端执行器的3D距离。
    4.  **GASAM调制**：将距离映射为“几何感知权重”，通过数学手段将这些权重注入到“动作-视频”交叉注意力机制中。
- **算法精解**：公式(7)是精髓。它在Softmax前引入了对几何权重的对数项（$\lambda \log(w + \epsilon)$），通过偏置注意力矩阵，强迫模型将关注点聚集在几何上与末端执行器靠近的区域（如物体把手、接触面）。

### 4. 方法对比分析
- **本质区别**：传统模型关注“未来会发生什么”，STARRY关注“为了完成动作，几何上哪些位置最关键”。
- **创新贡献**：提出GASAM模块，填补了视觉特征与物理空间控制之间的鸿沟，实现了显式的几何引导，而非隐式的特征提取。
- **适用场景**：适用于需要精确末端位姿、复杂接触及多阶段执行的机器人操作（如手眼协同、器皿挂放等）。

### 5. 实验分析（精简版）
- **验证方法**：在RoboTwin 2.0模拟基准及ARX R5真实机器人平台上进行多阶段任务测试。
- **关键结果**：在模拟中，相比LingBot-VA，平均成功率有显著提升；真实世界中，在更难的“阶段2”任务中，成功率提升幅度高达31.7%。
- **优势**：极强的多阶段任务长时序连贯性，明确的几何约束减少了控制偏离。
- **局限**：对预测深度和几何信息的依赖较重，若环境视觉极度模糊，几何先验的有效性会降低。

### 6. 实用指南
- **迁移建议**：本方法非常适合集成到现有的基于扩散的策略网络中。
- **实现细节**：注意几何专家（Geometry Expert）的训练分为独立训练和后期联合微调两个阶段，以避免直接修改扩散模型目标导致的不稳定性。
- **超参数注意**：$\lambda$ 控制调制强度，需要针对具体任务的几何敏感度进行调整。

### 7. 总结
- **核心思想**：通过预测几何关系并显式调制注意力，将世界模型引导向动作相关的物理关键区。
- **速记版Pipeline**：
    1. 输入融合（视频+几何+轨迹）；
    2. 扩散式联合预测（未来动态+动作）；
    3. 几何推理（估计末端与环境的3D距离）；
    4. 动作注意力偏置（根据几何权重高亮决策区域）；
    5. 动作执行。

**Key Findings:**

- We propose STARRY, a world-model-enhanced action-generation policy that aligns spatial-temporal prediction with action generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26848v1)
- [arXiv](https://arxiv.org/abs/2604.26848v1)

---

<a id='2604.26820v1'></a>
## [Bridge: Basis-Driven Causal Inference Marries VFMs for Domain Generalization](https://arxiv.org/abs/2604.26820v1)

**Authors:** Mingbo Hong, Feng Liu, Caroline Gevaert, George Vosselman, Hao Cheng

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

Detectors often suffer from degraded performance, primarily due to the distributional gap between the source and target domains. This issue is especially evident in single-source domains with limited data, as models tend to rely on confounders (e.g., illumination, co-occurrence, and style) from the source domain, leading to spurious correlations that hinder generalization. To this end, this paper proposes a novel Basis-driven framework for domain generalization, namely \textbf{\textit{Bridge}}, that incorporates causal inference into object detection. By learning the low-rank bases for front-door adjustment, \textbf{\textit{Bridge}} blocks confounders' effects to mitigate spurious correlations, while simultaneously refining representations by filtering redundant and task-irrelevant components. \textbf{\textit{Bridge}} can be seamlessly integrated with both discriminative (e.g., DINOv2/3, SAM) and generative (e.g., Stable Diffusion) Vision Foundation Models (VFMs). Extensive experiments across multiple domain generalization object detection datasets, i.e., Cross-Camera, Adverse Weather, Real-to-Artistic, Diverse Weather Datasets, and Diverse Weather DroneVehicle (our newly augmented real-world UAV-based benchmark), underscore the superiority of our proposed method over previous state-of-the-art approaches. The project page is available at: https://mingbohong.github.io/Bridge/.

**Analysis:**

### 1. 摘要翻译
目标检测器常因源域与目标域之间的分布差异导致性能退化，特别是在单源域数据有限的情况下，模型倾向于依赖光照、共现模式和风格等混淆因子（confounders），从而产生阻碍泛化的伪相关性。为此，本文提出了一种名为 **Bridge** 的新型基于基的学习（Basis-driven）框架，将因果推理引入目标检测。通过学习用于前门调整（front-door adjustment）的低秩基向量，Bridge能够有效阻断混淆因子的影响，减轻伪相关性，同时通过过滤冗余和任务无关的成分来提炼表征。Bridge可无缝集成于判别式（如DINOv2/3、SAM）及生成式（如Stable Diffusion）视觉基础模型（VFMs）中。在多项领域泛化（DG）目标检测基准上的实验证明了其优越性。

### 2. 方法动机分析
- **驱动力**：解决单源域训练中，模型因过拟合源域特征（如光照、背景）而产生的伪相关性问题，提升在未知分布（OOD）目标域的鲁棒性。
- **现有方法痛点**：现有DG方法多依赖人工设计的特定混淆因子或复杂后处理，缺乏灵活性；且未充分利用VFM强大的预训练表征能力。
- **研究假设**：通过因果推理中的“前门调整”机制，利用可学习的低秩基向量（Basis）作为中介变量（Mediator）的近似，可以有效解耦混淆因子，捕捉真正的因果特征。

### 3. 方法设计详解
- **核心逻辑**：利用“前门调整”公式 $P(Y | do(X)) = \sum_{M} P(M|X) \sum_{X'} P(Y|X', M) P(X')$，通过中介变量 $M$ 阻断 $X \to Z \to Y$ 的路径。由于闭式期望难以计算，作者提出了 **Causal Basis Block (CBB)**。
- **流程总结**：
  1. **特征提取**：利用冻结的VFM提取多尺度特征。
  2. **样本查询（Sample Queries）**：引入可学习的查询向量 $Q_s$，模拟DETR中的Object Queries，从训练集中聚合全局分布信息。
  3. **加权表征**：通过Softmax计算样本权重，对输入特征进行重加权，突出一般化信息。
  4. **低秩投影**：引入一组可学习基向量 $B$，将特征投影至低秩子空间，以此过滤冗余，仅保留最代表性的成分。
  5. **特征聚合**：将期望特征（Causal）与mediator特征（Task-specific）合并，完成最终校准。

### 4. 方法对比分析
- **本质区别**：无需显式建模混淆因子（Z），而是通过基学习（Basis Learning）隐式地从数据中解构出因果表征。
- **创新贡献**：提出CBB模块，将前门调整与词典学习（Dictionary Learning）思想结合，实现端到端的因果校准，且对各种VFM具有“插件式”的兼容性。
- **适用场景**：单源域、小样本、存在显著域偏移的目标检测任务。

### 5. 实验分析
- **关键结论**：在五个DG目标检测基准（如BDD100K、FoggyCityscapes等）上，Bridge均取得最优结果。特别是针对极端低光照环境（Extreme-Dark），性能显著优于基线。
- **主要优势**：框架轻量、可插拔、无需昂贵的重训练（冻结VFM骨干），且在不同检测器（Faster R-CNN, Sparse R-CNN等）上表现稳定。
- **局限**：对极小样本或极高噪声场景的性能依赖于基向量 $B$ 的表征能力，且目前仍需通过NWGM近似来规避计算量问题。

### 6. 实用指南
- **开源地址**：[https://mingbohong.github.io/Bridge/](https://mingbohong.github.io/Bridge/)
- **实现细节**：建议在多尺度特征层插入CBB；基向量个数 $K$ 的设置需根据VFM容量调整（强骨干用小比例，弱骨干用大比例）。
- **迁移可能**：该因果框架极易迁移至分类或分割任务中，只需将输入特征替换为任务特定的特征图。

### 7. 总结
- **核心思想**：利用低秩基学习实现前门调整，阻断伪相关性。
- **速记版pipeline**：
  1. 冻结预训练骨干提取特征；
  2. 使用查询向量聚合样本分布；
  3. 通过低秩投影筛选本质特征；
  4. 结合中介信息输出校准表征。

**Key Findings:**

- To this end, this paper proposes a novel Basis-driven framework for domain generalization, namely \textbf{\textit{Bridge}}, that incorporates causal inference into object detection.
- Extensive experiments across multiple domain generalization object detection datasets, i.e., Cross-Camera, Adverse Weather, Real-to-Artistic, Diverse Weather Datasets, and Diverse Weather DroneVehicle (our newly augmented real-world UAV-based benchmark), underscore the superiority of our proposed method over previous state-of-the-art approaches.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26820v1)
- [arXiv](https://arxiv.org/abs/2604.26820v1)

---

<a id='2604.26772v1'></a>
## [TAP into the Patch Tokens: Leveraging Vision Foundation Model Features for AI-Generated Image Detection](https://arxiv.org/abs/2604.26772v1)

**Authors:** Ahmed Abdullah, Nikolas Ebert, Oliver Wasenmüller

**Published:** 2026-04-29

**Categories:** cs.CV

**Abstract:**

Recent methods demonstrate that large-scale pretrained models, such as CLIP vision transformers, effectively detect AI-generated images (AIGIs) from unseen generative models when used as feature extractors. Many state-of-the-art methods for AI-generated image detection build upon the original CLIP-ViT to enhance this generalization. Since CLIP's release, numerous vision foundation models (VFMs) have emerged, incorporating architectural improvements and different training paradigms. Despite these advances, their potential for AIGI detection and AI image forensics remains largely unexplored. In this work, we present a comprehensive benchmark across multiple VFM families, covering diverse pretraining objectives, input resolutions, and model scales. We systematically evaluate their out-of-the-box performance for detecting fully-generated AI-images and AI-inpainted images, and discover that the best model outperforms the original CLIP by more than 12% in accuracy, beating established approaches in the process. To fully leverage the features of a modern VFM, we propose a simple redesign of the classifier head by utilizing tunable attention pooling (TAP), which aggregates output tokens into a refined global representation. Integrating TAP with the latest VFMs yields substantial performance gains across several AIGI detection benchmarks, establishing a new state-of-the-art on two challenging benchmarks for in-the-wild detection of AI-generated and -inpainted images.

**Analysis:**

这是一篇关于AI生成图像检测（AIGI Detection）的高水平技术论文。以下是深度分析：

### 1. 摘要翻译
现有方法主要依赖CLIP等大型预训练视觉模型作为特征提取器来识别AI生成图像（AIGI）。虽然CLIP在通用特征提取方面表现优异，但其性能对于不断演进的生成模型泛化能力仍然有限。本文提出了一套全面的评估体系，涵盖不同架构、分辨率及预训练目标的视觉基础模型（VFM）。研究发现，最佳模型在准确率上超越传统CLIP超过12%。为充分挖掘VFM潜能，本文提出了一种简单且高效的“可调注意力池化（TAP）”头设计，通过整合输出的补丁（patch）标记与cls标记，实现了对局部生成伪影的敏感捕获。TAP与最新VFM的结合在多个基准测试中达到了新的SOTA水平。

### 2. 方法动机分析
- **驱动力**：旨在突破传统方法仅依赖全局特征（cls token）的局限，提升对各种新型生成模型（如SDXL, Flux）的泛化检测能力。
- **痛点**：现有方法将ViT输出的cls token视为完整语义表示，丢弃了含有丰富空间细节的patch tokens。而AIGI/inpainted图像的篡改往往是局部性的，缺乏全局一致性，cls token无法有效捕捉这些精细的局部伪影。
- **研究假设**：通过引入TAP层，强制模型“关注”蕴含生成痕迹的局部patch tokens，能够比单纯依赖分类token提取更具判别力的特征空间。

### 3. 方法设计详解
- **Pipeline**：
  1. **特征提取**：输入图像经冻结的VFM编码器，输出patch tokens与cls token序列。
  2. **可调注意力池化 (TAP)**：引入一个可学习的query向量 $q$，通过多头交叉注意力机制（MHA）与输入的patch tokens交互。此过程可看作是模型自适应地学习提取图像中对分类最关键的局部信息。
  3. **特征Refine**：输出的池化向量 $z$ 经过带有残差连接的MLP层进行进一步语义提炼，生成最终的全局聚合表示 $gpl$。
  4. **分类**：将 $cls$ 与 $gpl$ 拼接，输入全连接分类头输出检测结果。
- **算法解释**：TAP的本质是一个可学习的投影操作。相较于简单的平均池化（丢弃局部空间结构）或仅用cls token（忽略局部细节），TAP通过注意力机制动态聚焦于异常区域，降低了维度，同时保留了局部上下文信息。

### 4. 方法对比分析
- **本质区别**：从传统的“固定特征提取”转变为“任务导向的局部特征聚合”。
- **创新点**：TAP模块在不引入复杂辅助网络（如专门的频率编码器）的情况下，仅通过轻量级参数即可显着提升性能。
- **适用场景**：适用于所有基于ViT架构的视觉模型，特别是在检测具有局部伪影（Inpainting、局部修改）的图像时效果最优。

### 5. 实验分析
- **验证方法**：在GenImage、Chameleon、OpenSDI三大数据集上进行评估。
- **结论**：PE-Core-ViT-G结合TAP后在OpenSDI上性能领先，相比仅使用cls token方法，mean F1提升显著。
- **优势**：轻量级设计，极易集成；优异的跨生成模型泛化性能。
- **局限**：在极端高分辨率或极具挑战性的新型生成模型（如Midjourney）上，可能存在一定程度的过拟合风险，需通过调整训练迭代次数来维持泛化性。

### 6. 实用指南
- **开源/复现**：方法高度模块化。核心在于将预训练ViT的最后一层替换为TAP层。
- **实现细节**：建议使用AdamW优化器，学习率设为1e-4，batch size为128。图像增强（JPEG压缩、高斯模糊）是处理生成伪影的关键。
- **迁移能力**：TAP是一个通用插件，不仅限于AIGI检测，任何需要捕捉局部细节的分类任务（如医疗影像病变检测、伪造文档识别）均可直接嵌入。

### 7. 总结
- **核心思想**：利用TAP挖掘Transformer的局部补丁特征，显著提升检测判别力。
- **速记版pipeline**：
  1. 冻结VFM backbone；
  2. patch tokens经TAP交互学习；
  3. 拼接cls标记与TAP输出的gpl向量；
  4. 最终全连接层二分类。

**Key Findings:**

- Many state-of-the-art methods for AI-generated image detection build upon the original CLIP-ViT to enhance this generalization.
- In this work, we present a comprehensive benchmark across multiple VFM families, covering diverse pretraining objectives, input resolutions, and model scales.
- We systematically evaluate their out-of-the-box performance for detecting fully-generated AI-images and AI-inpainted images, and discover that the best model outperforms the original CLIP by more than 12% in accuracy, beating established approaches in the process.
- To fully leverage the features of a modern VFM, we propose a simple redesign of the classifier head by utilizing tunable attention pooling (TAP), which aggregates output tokens into a refined global representation.
- Integrating TAP with the latest VFMs yields substantial performance gains across several AIGI detection benchmarks, establishing a new state-of-the-art on two challenging benchmarks for in-the-wild detection of AI-generated and -inpainted images.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26772v1)
- [arXiv](https://arxiv.org/abs/2604.26772v1)

---

<a id='2604.26694v1'></a>
## [Unified 4D World Action Modeling from Video Priors with Asynchronous Denoising](https://arxiv.org/abs/2604.26694v1)

**Authors:** Jun Guo, Qiwei Li, Peiyan Li, Zilong Chen, Nan Sun, Yifei Su, Heyun Wang, Yuan Zhang, Xinghang Li, Huaping Liu

**Published:** 2026-04-29

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework, addressing the critical limitations of prior unified world models (e.g., UWM) that only model 2D pixel-space and fail to balance action efficiency and world modeling quality. To leverage the strong visual priors of pretrained video diffusion models, X-WAM imagines the future world by predicting multi-view RGB-D videos, and obtains spatial information efficiently through a lightweight structural adaptation: replicating the final few blocks of the pretrained Diffusion Transformer into a dedicated depth prediction branch for the reconstruction of future spatial information. Moreover, we propose Asynchronous Noise Sampling (ANS) to jointly optimize generation quality and action decoding efficiency. ANS applies a specialized asynchronous denoising schedule during inference, which rapidly decodes actions with fewer steps to enable efficient real-time execution, while dedicating the full sequence of steps to generate high-fidelity video. Rather than entirely decoupling the timesteps during training, ANS samples from their joint distribution to align with the inference distribution. Pretrained on over 5,800 hours of robotic data, X-WAM achieves 79.2% and 90.7% average success rate on RoboCasa and RoboTwin 2.0 benchmarks, while producing high-fidelity 4D reconstruction and generation surpassing existing methods in both visual and geometric metrics.

**Analysis:**

### 1. 摘要翻译
我们提出了 X-WAM，一个统一的 4D 世界模型。它在单一框架内实现了实时机器人动作执行与高保真 4D 世界合成（视频+3D 重建），解决了现有统一世界模型仅建模 2D 像素空间且难以平衡动作效率与建模质量的局限。为利用预训练视频扩散模型的强视觉先验，X-WAM 通过预测多视角 RGB-D 视频来构想未来世界，并利用轻量级结构自适应获取空间信息：通过复制预训练扩散 Transformer 的末端层构建专门的深度预测分支。此外，我们提出了异步噪声采样（ANS）以联合优化生成质量与动作解码效率。ANS 在推理阶段采用专门的异步去噪调度，以较少的步数快速解码动作以实现实时执行，同时投入完整的序列步骤生成高保真视频。通过在 5800 多小时的机器人数据上进行预训练，X-WAM 在 RoboCasa 和 RoboTwin 2.0 基准测试中实现了 79.2% 和 90.7% 的平均成功率，并在视觉和几何指标上均超越了现有方法。

### 2. 方法动机分析
- **驱动力**：旨在将机器人控制（策略模型）与环境预测（世界模型）真正统一，并赋予其 3D 几何认知能力。
- **痛点**：现有统一模型多局限于 2D 像素空间，缺乏物理感知；同时存在“模态失配”问题：高质量视频生成需要大量去噪步数，而低维动作预测只需少量步数，同步去噪会导致推理延迟过高或性能折损。
- **核心直觉**：通过“单侧注意力”的轻量级结构注入 3D 深度信息，并利用动作对噪声的鲁棒性，解耦视频与动作的去噪频率，实现推理加速与质量保障的平衡。

### 3. 方法设计详解
- **模型结构**：基于 Wan2.2-5B 预训练模型，X-WAM 将 RGB 视频、深度图、动作和状态视为一个联合序列。
  - **轻量级深度分支**：为了不破坏预训练权重，作者没有增加全序列长度，而是通过“单侧注意力（Unilateral Attention）”机制，复制预训练 DiT 的末端 $M$ 层作为深度分支，仅在去噪过程中读取主分支特征，而不反向修改主分支。
  - **模态统一**：将动作、状态、RGB、深度统一编码进 Latent Space 进行 denoising，通过 learnable view embeddings 处理多视角信息。
- **算法（异步噪声采样 ANS）**：
  - **推理策略**：动作分支与视频分支采用不同的步数预算 ($T_a < T_O$)。动作在 $T_a$ 步后即达到无噪状态并输出控制指令，随后动作分支作为“干净的上下文”参与剩余视频的去噪。
  - **训练策略**：设计联合噪声分布，通过 Beta 分布控制视频与动作的去噪水平，确保训练过程中视频与动作的噪声匹配推理时的异步状态，避免“样本分布偏移”。

### 4. 方法对比分析
- **本质区别**：从传统的“先生成后规划”或“单纯的端到端动作预测”转变为“以动作预测为核心的异步并行 4D 动力学模拟”。
- **创新贡献**：
  1. **单侧注意力架构**：高效且无损地引入了 3D 空间结构。
  2. **异步噪声采样**：从训练分布层面直接解决了动作/视频去噪步数的模态失配问题。
- **适用场景**：高自由度、需要长时程规划且对实时性有严格要求的复杂机器人操作任务。

### 5. 实验分析
- **验证方法**：在 RoboCasa 和 RoboTwin 2.0 上进行策略成功率测试，并在模拟器内对比 PSNR、LPIPS、Chamfer Distance 等重构指标。
- **关键结果**：在保证 1033ms 低延迟的同时，SR 均值达到 79.2%（RoboCasa），显著超过 Cosmos Policy 等基线。
- **优势/局限**：优势在于统一框架下的高性能重构与控制，局限在于上下文长度固定，缺乏长时程历史记忆，导致复杂场景下可能存在短视问题。

### 6. 实用指南
- **开源/复现**：项目主页已公布 (https://sharinka0715.github.io/X-WAM/)。复现关键在于 Wan2.2-5B 的微调及 UniPC 调度器的正确配置。
- **迁移建议**：深度分支的“单侧注意力”设计非常适合任何基于 Transformer 的多模态生成任务（如将图像扩散模型适配为视频/深度预测模型）。
- **注意**：训练阶段的 Beta 分布采样概率 $p=0.5$ 是实现异步性能的关键超参数。

### 7. 总结
- **核心思想**：利用异步去噪调度与单侧结构适配，实现视频生成与动作控制的实时统一。
- **速记版 pipeline**：
  1. 将输入投射到统一 Latent Space。
  2. 主分支处理视频，深度分支通过单侧注意力注入空间特征。
  3. 联合采样不同噪声等级的视频与动作进行训练。
  4. 推理时先完成动作解码用于即时控制，再完成剩余视频去噪。

**Key Findings:**

- We propose X-WAM, a Unified 4D World Model that unifies real-time robotic action execution and high-fidelity 4D world synthesis (video + 3D reconstruction) in a single framework, addressing the critical limitations of prior unified world models (e.g., UWM) that only model 2D pixel-space and fail to balance action efficiency and world modeling quality.
- Moreover, we propose Asynchronous Noise Sampling (ANS) to jointly optimize generation quality and action decoding efficiency.

**Links:**

- [PDF](https://arxiv.org/pdf/2604.26694v1)
- [arXiv](https://arxiv.org/abs/2604.26694v1)

---

