time: 20261027

# Arxiv Computer Vision Papers - 2026-10-27

## Executive Summary

## 今日执行摘要

今日筛选出两篇值得完整阅读的工作，分别聚焦交互式多模态助手评估与视觉语言导航解释。共同趋势是从单次静态视觉判断转向利用历史、跟踪阶段变化，并用可检验的过程或结果支撑结论。

InteractiveWorld 将“助手给建议、人类执行、环境反馈”的多轮协作构造成 120 项状态可验证任务。12 个多模态模型中最佳整体完成率为 41.7%，长程指导和规划任务分别只有 20% 和 25%，显示持续记忆、清晰沟通和根据反馈重规划仍是短板。其贡献在于补足静态视频问答与直接机器人控制之间的评估空缺；当前结果仍局限于模拟环境及小规模真人执行者检查。

视觉语言导航归因论文则针对解释随子目标变化不明显的问题，把选定动作的梯度、跨帧视觉证据和指令阶段结合起来，并以阶段加权历史融合与指令引导滤波改善解释。在 R2R/NaVILA 上，F1 从 DEX-AR 的 11.21 提高到 36.93，消融与阶段敏感度分析支持两个模块的作用。该方法适合用于导航策略调试，但解释质量指标依赖模拟语义标注，真实机器人证据仍为定性展示。

建议优先通读两篇：InteractiveWorld 适合关注交互评测、用户代理和状态型 benchmark 的读者；阶段感知归因工作适合研究 VLN/VLA 可解释性、长时程导航诊断及历史证据融合的读者。二者都提示，下一步需要在更真实的执行反馈、长程记忆和可审计评估上继续推进。

---

## Table of Contents

1. [Explaining Navigation Decisions: Stage-Aware Visual Attribution for Vision-Language Navigation](#2610.34028v1)
2. [InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants](#2610.33966v1)

---

## Papers

<a id='2610.34028v1'></a>
## [Explaining Navigation Decisions: Stage-Aware Visual Attribution for Vision-Language Navigation](https://arxiv.org/abs/2610.34028v1)

**Authors:** Yiyuan Yao, Xinpeng Ding, Minglei Chen, Fei Gao, Lihuo He, Jie Yuan, Aijun Li, Nannan Wang, Xinbo Gao

**Published:** 2026-10-25

**Categories:** cs.CV

**Abstract:**

As Vision-Language Models (VLMs) increasingly enable robots to follow language instructions, understanding their navigation decisions becomes important for reliable deployment. Existing explainability methods overlook a key challenge in Vision-Language Navigation (VLN): evidence for the current action may be distributed across historical and current observations and associated with different instruction stages. We identify instruction-stage insensitivity, where existing methods produce similar explanations despite changes across different navigation stages. To address this issue, we propose a stage-aware attribution framework for VLN. We first construct action-conditioned temporal instruction--visual attributions across observations. These attributions are projected into instruction space to estimate stage associations, which guide the aggregation of relevant historical evidence for refining the current explanation. An instruction-guided spatial filter further suppresses isolated responses while preserving compact, instruction-relevant regions. Experiments across multiple VLN policies and benchmarks show improved attribution quality over existing methods. We further demonstrate that our method acts as a diagnostic tool for analyzing failure cases, and validate its applicability to real-world navigation on a Unitree G1 robot.

### 论文解读
#### 摘要翻译
本文研究如何解释视觉语言导航模型的动作。一个动作可能依赖当前画面和历史观察，而历史画面又对应指令中的不同阶段；直接叠加视觉显著图容易混淆已完成与当前子目标。作者提出阶段感知归因方法，在保留动作相关视觉证据的同时，按指令阶段筛选历史信息，并用指令与图像区域的关联抑制孤立噪声。
#### 方法动机分析
传统视觉解释方法常为单帧或生成词定位证据，但导航是连续决策：走过桌球台后再去电视旁停下，当前动作应更多依赖电视和沙发，而非旧画面中的桌球台。若忽略时间和子指令变化，解释可能随阶段改变仍几乎不变。本文的核心假设是，模型注意力与选中动作的梯度可提供动作相关线索，而每帧归因在指令词上的分布可近似表示其阶段。
#### 方法设计详解
输入是完整导航指令、当前视图、历史视图和策略选定动作。首先对每帧提取指令词到视觉 patch 的注意力，并乘以动作 logit 的梯度，得到动作条件归因矩阵。接着把矩阵投影到指令词维度，计算归一化位置期望作为软阶段标记；历史帧按其与当前阶段的距离加权，用来重标当前帧中各指令词的归因，同时保留最低响应，避免压掉新证据。最后分别计算 patch 对词、词对 patch 的归因集中度，调节空间滤波：强匹配区域尽量保留，弱关联的孤立响应则加强抑制。实现中使用策略第 8–15 层和全部注意力头；NaVILA、NaVid、StereoNav 输入分辨率分别为 384、224、448 像素见方。
#### 方法对比分析
与 Grad-CAM 或按生成词解释的 TAM、DEX-AR 相比，该方法把“当前导航动作”“历史视角”和“指令阶段”共同纳入解释。Stage-Guided Temporal Refinement 处理跨阶段干扰，Instruction-Guided Spatial Filter 则避免常规空间滤波误删紧凑但相关的区域。方法解释策略已经选出的动作，不负责改进动作规划本身。
#### 实验分析（精简版）
在 R2R 的 NaVILA 评测中，本文的 corrected ΔNLL、Precision、F1 分别为 31.98、43.70、36.93；DEX-AR 分别为 14.54、9.12、11.21。R2R 消融中，两模块同时开启时 F1 为 36.93，无模块时为 32.52；另一个阶段敏感度指标显示跨阶段热图相似度从 96.8% 降至 57.4%。实验涵盖 R2R、RxR 与三种策略。真实 Unitree G1 展示是定性结果；语义对齐依赖模拟场景标注和 CLIP 相关度，不能当作解释正确或安全的证明。
#### 实用指南
论文给出层数、分辨率及 τH=0.25、δ=0.7、滤波参数 k=3 等设置，但未说明公开代码。复现需确保不同基线使用同一策略输入，并按动作选择梯度；迁移到移动机器人或自动驾驶时，可把指令阶段换成子任务/路线阶段，但必须针对新传感器与策略重新验证解释和人工可读性。
#### 总结
核心思想：让解释对齐当前子目标
1. 计算选定动作对应的跨帧词—区域归因。
2. 用指令词分布估计每帧阶段。
3. 按阶段相关性融合历史证据并修正当前帧。
4. 保留词—区域匹配强的证据，过滤孤立响应。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.34028v1)
- [arXiv](https://arxiv.org/abs/2610.34028v1)

---

<a id='2610.33966v1'></a>
## [InteractiveWorld: A Verifiable Human-Centric Environment for Building Better Interactive MLLM Assistants](https://arxiv.org/abs/2610.33966v1)

**Authors:** Qinsi Wang, Wei Wen, Chong Zhou, Saksham Suri, Zechun Liu, Lemeng Wu, Hai Helen Li, Yiran Chen, Raghuraman Krishnamoorthi, Qi Qian

**Published:** 2026-10-25

**Categories:** cs.HC, cs.CV

**Abstract:**

People have long dreamed of an assistant that is always there to guide them and remind what they forget, from new dishes to lost keys. Powered by multimodal large language models (MLLMs), egocentric AI assistants on wearable devices are bringing this dream within reach: they see what the user sees and give instructions for user's physical work when necessary. However, testing such assistance is challenging and usually requires real people. The expensive and rare real interaction data leaves a key question unanswered: Can today's MLLMs truly help people in physical world? To answer this question, we propose InteractiveWorld, the first verifiable environment for closed-loop interaction between human and AI assistant. Unlike existing robot-centric simulators, InteractiveWorld lets the assistant guide a human agent who observes the scene, understands instructions, and acts independently. It thus realistically simulates how AI assistant collaborates with human: the assistant instructs, the human acts, and the environment executes actions and updates what they see next. Building on it, we introduce InteractiveBench, the first dynamic benchmark for egocentric AI assistants, with 120 multi-turn tasks across six categories. In each task, the assistant guides the human in real time, and every instruction shapes the outcome, which is scored by verifiable, state-based rewards. Experiments with 12 frontier MLLMs show they are still far from reliable assistants. Even the strongest model, Claude Fable 5.1, achieves at most 20% and 25% on long-horizon Guidance and Planning tasks, respectively. Overall, our work offers a verifiable platform and a rigorous evaluation set, laying a solid foundation for research on interactive MLLM assistants that truly help people in the physical world.

### 论文解读
#### 摘要翻译
作者提出 InteractiveWorld，一个让多模态助手在模拟日常环境中指导人类行动的交互平台，并据此构建 InteractiveBench。助手观看第一人称画面、用自然语言给建议；独立的人类代理理解建议并执行动作，模拟器再更新场景。120 项多轮任务覆盖指导、规划、无障碍、决策、预防和寻物。对 12 个模型的测试显示，当前助手仍难以稳定帮助人完成长流程。
#### 方法动机分析
现有评估的痛点是，许多第一人称基准只让模型回答固定视频中的问题，回答不会改变后续画面；真实帮助却要求模型观察人的反应、记住之前发生的事，并根据新状态调整建议。机器人模拟器虽能闭环执行，却通常让模型直接控制机器人，未评估“助手说、人来做”的合作过程。作者因此把人类执行者、自然语言建议和可核验的环境状态接入同一循环。
#### 方法设计详解
InteractiveWorld 的执行流程分为三层：物理层用 Isaac Sim、人体模型和物体资产维持场景；人类动作层将高层目标转为可碰撞检查的导航、姿态和操作；语义层提供 36 种动作接口，把助手语言交给人类代理执行。每个动作后环境生成新的第一人称视图，助手可继续建议或结束。InteractiveBench 在六类场景中设计 120 个任务，每类 20 个；任务需要多轮观察、记忆和沟通，成功依据物体位置、开关状态等模拟器状态自动判断，而不是只比对文字答案。主实验固定 Gemini 3.1 Pro 作为执行者，助手每轮最多接收 32 帧。
#### 方法对比分析
与静态视频问答相比，交互输出会改变世界与下一次观察；与机器人控制基准相比，助手不直接操作身体，而须让执行者理解并完成建议。核心区别是评估对象从单轮回答或机器人动作转为“助手说、人来做”的合作过程，因此能测长期跟踪、指令清晰度与适时干预，也让任务成败可复核。它评估的是人机助手能力，不是机器人运动规划本身。
#### 实验分析（精简版）
12 个模型中，Claude Fable 5.1 的整体完成率最高，为 41.7%；其 Guidance 与 Planning 完成率分别只有 20% 和 25%。同一人类代理在 Oracle 助手下达到 100%，没有助手时为 0%，说明任务需要有效指导。错误分析认为视觉识别和记忆问题占主要部分。实验还发现成功任务常伴随多余动作，部分类别的动作成本达到 Oracle 的数倍。真人执行者对照仅含 3 人、36 项任务；结果仍来自模拟环境，不能证明真实部署可靠。
#### 实用指南
论文列出代码仓库与项目页，可用于复现模拟环境及任务协议。复现时应固定任务包、提示词、执行者模型和画面采样规则，并检查动作后的真实状态；只报告问答准确率会漏掉闭环失败。迁移到机器人或可穿戴助手时，可保留多轮反馈和状态评分，再补充真实传感误差、安全约束与用户差异。
#### 总结
核心思想：让建议改变场景
1. 助手观察第一人称画面并接收任务。
2. 助手用语言指导人类代理。
3. 代理执行动作，模拟器更新状态和画面。
4. 用最终环境状态验证任务是否完成。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.33966v1)
- [arXiv](https://arxiv.org/abs/2610.33966v1)

---

