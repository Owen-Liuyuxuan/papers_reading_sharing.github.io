time: 20261127

# Arxiv Computer Vision Papers - 2026-11-27

## Executive Summary

## 2026-11-27 ArXiv 论文执行摘要

本期纳入 10 篇论文，研究重心落在视觉与具身智能的结合：如何把视觉表征转化为可执行动作，如何让代码策略和基础模型在真实平台上更可靠，以及如何以更有效的数据与校准方法提升生成和适配能力。整体趋势是从单纯追求模型规模或离线指标，转向显式处理机器人形态、物理可行性、运行时验证和部署约束；与此同时，多项工作仍以仿真或小规模实机实验为主，跨平台泛化与重复试验稳定性仍需进一步检验。

### 主要进展

- **策略可靠性开始成为方法设计的核心。** *Toward Agentic Robotics* 根据模型能力路由策略代码与 VLM 救援，并以回归门控迭代；在部分基准上相对基策略有明显提升，但标准 LIBERO 已接近饱和，实机验证仅涉及两个任务。*RegHarness* 则把试内重置重放、验证器与任务审计，以及试间技能修订纳入代码策略流程。其组合系统在 LIBERO-PRO 上达到 87.8%（158/180），但开发状态参与技能迭代、不同配置预算也有变化，增益来源仍需谨慎解释。两篇论文共同提示：可复用技能、按能力选择工具以及有约束的策略演化，可能是提升代码式机器人控制的重要抓手。
- **视觉表征与动作生成的连接方式更加多样。** *FreshVLA* 从无动作视频构造快慢分层的连续潜在动作，在 LIBERO、CALVIN 和四项 Flexiv 桌面任务上报告较强结果；*HAVM* 把演示库的软动作检索作为残差校正锚点，在稀疏、多样的人类演示场景中取得显著改善，但在密集演示或生成数据上收益减弱或反转。*VT-CoRe* 通过跨帧对应与潜在轨迹约束学习视频特征，对模拟模仿学习有积极结果；其机器人证据限于仿真中的冻结特征行为克隆。
- **物理可执行数据与机器人形态规范化受到关注。** *DexEngine* 将人类手物交互数据转为灵巧手全身轨迹，并显示物理修正数据在模拟任务微调中优于纯运动学回放；但强化学习生成成本高，固定躯干的可达性筛选也排除了约三分之二候选片段。*HoloBrain-0.5* 通过共享 3D 运动学图统一不同本体的动作语义，跨本体预训练和少样本实机结果具有参考价值；不过零样本迁移没有覆盖全新机器人，实机成绩也未报告重复试验方差。
- **部署前诊断和几何线索值得重视。** *TRIAGE* 将传感契约、目标硬件延迟门控、短日志探针和闭环验收组成固定平台模型筛选流程；小规模车辆路线结果显示，仅按离线分数选模型可能遗漏关键的时延问题。*Who Is Ego?* 则将相机佩戴者识别表述为跨视角几何问题，并利用候选掩码进行无标签测试时适配；其数据集规模和未见设定结果突出，但非人类真实成对数据仍不足，部分测试依赖仿真。
- **生成模型校准出现免搜索思路。** *Guidance as Variance Matching* 用校准样本估计引导强度，使引导输出方差匹配数据方差，并在多类生成任务中复现已发布尺度。它提供了可迁移的校准原则，但任务指标改善并不一致，且机器人扩散策略上的效果尚未实测。

### 值得优先阅读全文

1. **Toward Agentic Robotics**：适合关注 VLM、VLA 与代码策略系统的读者。论文对多种 harness 设计做了较大规模受控比较，并报告真实机器人实验；阅读全文可重点核查路由机制、回归门控与 API/GPU 成本之间的权衡。
2. **FreshVLA**：适合研究视频预训练与机器人动作生成的读者。无动作视频如何形成具有时间尺度分工的潜在动作，是本期较清晰的新训练方向；建议重点看消融设计及真实任务覆盖范围。
3. **DexEngine**：适合关注人类演示扩展、灵巧操作和 sim-to-real 的读者。其物理修正数据与纯运动学回放的对照直接关系到合成演示是否可用于策略训练，也揭示了可达性和生成成本的实际瓶颈。
4. **TRIAGE**：适合在固定车辆或机器人平台部署基础模型的团队。该工作把传感、时延、分布偏移和闭环验收放进同一筛选流程，提供了采集本地训练数据前诊断适配瓶颈的实用框架。
5. **HAVM**：适合少样本模仿学习研究者。软检索基础优于单个最近原型的结果值得细读，同时其对演示分布的敏感性提示应关注目标数据是否稀疏且多样。

本期较值得持续跟踪的方向包括：以视觉或视频表征驱动分层动作生成、将机器人可达性和接触物理纳入示范合成、基于几何结构的跨视角与跨本体泛化，以及在目标硬件上开展早期可行性筛选和闭环评估。后续工作若能扩大真实机器人任务、平台与重复试验规模，将有助于判断这些方法的增益能否稳定迁移到实际部署。

---

## Table of Contents

1. [Toward Agentic Robotics: A Large-Scale Study of Harness Design for Robot-Use Agents](#2611.34778v1)
2. [Hetero-Associative Visuomotor Memory: Soft Recall as a Base for Learned Residuals in Few-Demonstration Imitation](#2611.34014v1)
3. [DexEngine: Scaling Human Demonstrations into Physically Executable Whole-Robot Data](#2611.33579v1)
4. [FreshVLA: Frequency-Regularized Time Scale Hierarchical Latent Actions for Vision-Language-Action Models](#2611.33635v1)
5. [RegHarness: Regulating What Code-as-Policy Agents Keep from Steps to Skills](#2611.33739v1)
6. [Triage Before Fine-Tuning: Feasibility-Gated Selection and Monitored Adaptation of Embodied Foundation Models for a Fixed Physical Platform](#2611.33772v1)
7. [Variation-aware Temporal Correspondence for Self-Supervised Video Representation Learning](#2611.34220v1)
8. [HoloBrain-0.5 Technical Report](#2611.34449v1)
9. [Who Is Ego? Scaling Cross-View Camera Wearer Identification and Segmentation beyond Humans](#2611.34489v1)
10. [Guidance as Variance Matching](#2611.34801v1)

---

## Papers

<a id='2611.34778v1'></a>
## [Toward Agentic Robotics: A Large-Scale Study of Harness Design for Robot-Use Agents](https://arxiv.org/abs/2611.34778v1)

**Authors:** Yuxiang Lai, Jike Zhong, Yu-Jhe Li, Lan Wang, Chengzhi Liu, Youbao Tang, Ming Li, Xi Xiao, Xiaofeng Yang, Liang Zhao, Zheda Mai, Ruixi Fan

**Published:** 2026-11-25

**Categories:** cs.RO

**Abstract:**

Recent vision-language models (VLMs) can act as robot-use agents that understand diverse scenes and control robots through policy code and tools. However, they predict 3D gripper poses poorly even with extended reasoning, struggling with precise manipulation. Vision-language-action (VLA) and world-action (WAM) models are the opposite: they execute fast, smooth control on trained tasks, yet fail under domain shifts or when tasks require long-horizon memory and reasoning. Agentic robots therefore combine VLM reasoning and perception with precise VLA/WAM control through a policy harness. Yet harness design spans thousands of configurations, shifting success by tens of points with the same models. Moreover, testing even one design takes thousands of rollouts at high API and GPU cost, so prior works often commit to a single design chosen from few alternatives. To guide harness design, we evaluate over 470,000 rollouts across eight simulation benchmarks, six VLMs, and 15 VLA/WAM checkpoints. Seven controlled studies show effective harnesses must adapt to both VLM and VLA/WAM capabilities. Our findings specify what structure the VLM's policy code needs, where VLA manipulation needs help from tools and policy code, and when to involve the VLM at test time. The reference recipe X-Harness shows that these findings compose without weight updates, lifting base policies up to $+42.8$ points across memory, reasoning, and domain shifts.

### 论文解读

#### 摘要翻译

视觉语言模型能理解场景并写控制代码，但难以准确预测三维抓手位姿；视觉语言动作模型控制流畅，却容易在分布偏移、长时记忆与推理任务中失效。作者系统评估连接两者的策略 harness：在八个仿真基准、267项任务、六个视觉语言模型和15个动作策略上完成七类设计研究，共489,380次 rollout。研究总结出按模型能力组织策略、工具和调用时机的规则，并提出无需更新模型权重的 X-Harness，在多类任务上提升基础策略表现。

#### 方法动机分析

瓶颈不只在模型本身，也在如何把高层理解交给低层控制。直接让 VLM 输出精确位姿容易失败，逐步询问 VLM 又会让机器人等待。作者的核心假设是，不存在适用所有任务的固定封装；应根据 VLM 编码能力、VLA 已掌握的技能和任务的感知需求，动态选择执行路径。

#### 方法设计详解

输入是任务指令、完整视频、空间约定及视觉工具输出；处理流程为 VLM 编写策略代码 → 闭环代码读取每步图像/机器人状态 → VLA 执行语言子目标或三维工具调用运动原语 → 失败保护器按需请求 VLM，输出为任务完成或安全恢复。Harness 由上下文、感知/动作工具、策略代码、修订门控和运行时调用层级五个模块构成。策略可以是自由 Python，也可以填入规则、行为树、PDDL 等模板。训练集阶段每项任务通常使用约10个训练 episode 收集失败日志，最多修订八轮；只有成功提升且回归有限的修改通过门控。推理阶段分为 Offline（零 VLM 调用）、Semi-online（失败触发救援）和 Full-online（每个子目标调用）。每个设置至少测试20个 episode，策略权重冻结；不确定性以任务和 episode 两层配对 bootstrap 估计。

#### 方法对比分析

与逐步 VLM 规划器相比，X-Harness 把反复推理编译为本地闭环代码，让常规控制不等待远端模型。它也不同于一律由 VLA 执行的固定层级：已掌握的技能委托给 VLA，空间偏移或记忆短板则增加逻辑、感知工具或原语。实验发现弱编码模型受益于模板，强模型反而更适合自由代码，因此代码形式也应按能力选择。

#### 实验分析（精简版）

七项受控研究覆盖八个基准的267项任务。Opus 版 X-Harness 在 LIBERO-PRO 达95.4%，相同协议下 π0.5 基线为58.6%，提升36.8个百分点；VLABench 为68.3%，基线25.5%，提升42.8点。标准 LIBERO 上基线已达97.2%，harness 为96.1%，说明增益集中于基础策略的短板。作者还报告新种子、视觉扰动及双臂 YAM 实机结果。实机仅涵盖双箱记忆和计数舀豆两个任务，主体证据仍来自仿真，不能据此推断广泛部署能力。

#### 实用指南

作者公开代码、评测套件和 rollout 日志，见 https://github.com/google-research/x-as-policies。复现时需匹配基准接口、冻结策略、训练/测试种子划分、视觉工具和回归门控；论文常规演化为约10个训练 episode、最多八轮。全研究资源开销很大：超过28,000 H100小时、42B VLM token及12万美元 API 标价。迁移时可先针对目标任务比较离线闭环策略与失败触发救援，检查延迟、深度误差和退化任务，再扩展候选模型。

#### 总结

核心思想：按能力分工，失败再求助

1. 给模型完整历史和工具感知结果。
2. 将高层决策写成可反馈的闭环策略。
3. 已掌握动作交给 VLA，短板由工具和逻辑补足。
4. 用训练失败门控修订，测试时只在异常调用 VLM。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34778v1)
- [arXiv](https://arxiv.org/abs/2611.34778v1)

---

<a id='2611.34014v1'></a>
## [Hetero-Associative Visuomotor Memory: Soft Recall as a Base for Learned Residuals in Few-Demonstration Imitation](https://arxiv.org/abs/2611.34014v1)

**Authors:** Runcong Wang

**Published:** 2026-11-25

**Categories:** cs.RO, cs.LG

**Abstract:**

Retrieve-and-correct policies typically refine the nearest retrieved demonstration chunk, the   readout that works best when executed directly. We study what a demonstration memory should   supply to the corrector instead and find that a smooth, softmax-weighted recall over the whole   memory is a much better base. On ALOHA cube transfer (40 human demonstrations), soft recall is   no better to execute alone than the nearest chunk; however, with keys, candidates, and training   fixed, correcting it raises success by 22.4 points on fresh resets with new encoders and by 7.4   against a closed-loop-tuned top-1 corrector. On robomimic Square, it leads by 7.9 points with   the two better operators and by 6.7 on a label-free draw of equal size. This gain holds on   sparse, varied demonstrations under an action-aware key and leave-one-demonstration-out   training, and fades or slightly reverses on dense, scripted, or generated data. Building on this   finding, our hetero-associative visuomotor memory (HAVM, 14M parameters) trains offline in about   five minutes, including a learned retrieval temperature $β$, and runs in 1.1 ms. Across six   fresh ALOHA encoders, it reaches 76.4% within 24 s, comparable to a 264M Diffusion Policy   trained for 6.7 h (78.8%) and ahead of ACT (71.7%) and fine-tuned VLAs (65.2-75.2%), while   leading every 40-demonstration baseline within the benchmark's 8 s limit (74.6%; +6.5 over   Diffusion Policy, +15.1 over ACT). On LIBERO, one 40-task memory reaches 92.2% and loses far   less than SmolVLA to rewording; on a real SO-101 arm, it sustains 30 Hz control without   stalling. Unlike pretrained foundation policies aimed at zero-shot generalization to unseen   tasks and workspaces, HAVM trades broad out-of-distribution generalization for rapid adaptation   to the available demonstrations.

### 论文解读

#### 摘要翻译
本文研究检索策略给校正器什么动作基础。稀疏、多样示范下，全库软召回虽不适合单独执行，却比最近邻更易校正。作者据此提出HAVM：14M参数，约5分钟离线训练，毫秒级推理；它适合快速适配已有演示覆盖的任务。

#### 方法动机分析
最近邻可复用一条真实轨迹，但查询稍变就可能切换示范，令校正目标跳变。全库加权平均本身会模糊动作，却可能形成更平滑的校正锚点。论文假设：在示范少、操作者行为有差异时，残差头更容易修正平滑原型，而非追随最近邻的突变。

#### 方法设计详解
输入相机图像、机器人状态和可选指令，输出未来动作块。ResNet-18提取图像特征，状态经MLP编码，融合为128维键；双视图对比损失保持表征稳定，未来动作相似度损失让相近动作对应的帧靠近。记忆逐帧存储键和后续H步动作。查询按wᵢ=softmaxᵢ(βkᵢᵀq)对全库加权，基础动作为A_base=ΣwᵢVᵢ；β控制从最近邻到多演示原型。

校正头读取基础块、top-4候选块和观察token，经两层注意力输出残差：A_hat=A_base+σ⊙r。候选保留局部细节，零初始化输出使初始策略等于原始召回。训练某帧时整段移除其所属示范，避免记忆中重叠动作块泄漏；编码器和库冻结，只用离线示范拟合头部。这使头部学习跨演示的校正规律，而非记住查询自身动作。ALOHA设置H=50、执行25步后重规划，50Hz；约14M参数，训练约5分钟，GPU调用1.1ms。

#### 方法对比分析
ALT等方法直接执行top-1，HAVM则把检索视为可校正的基线；相比邻居提议均值，它对全库软召回并学习温度，再由动作感知注意力头细化。相比ACT、扩散策略和微调VLA，它用显式演示记忆换取更快适配与推理，但依赖演示覆盖，不追求广泛零样本泛化。

#### 实验分析（精简版）
ALOHA人类演示控制实验中，软基础单独成功率14.5%，top-1为60.5%；加注意力校正后分别83.5%和42.7%。六个新编码器、600个新reset上，24秒成功率76.4%，Diffusion Policy为78.8%；8秒限制下HAVM为74.6%，高6.5个百分点。LIBERO四套40任务达到92.2%。实机仅10对试验；示范工作空间边缘失败明显，编码器跨种子也有较大波动。

#### 实用指南
论文以ResNet-18、AdamW和RTX 5090实现。复现重点是动作感知键、整段示范留出和β闭环选择。少量、多样的人类示范可优先试软基础；密集、脚本化或生成数据应比较top-1。β学习仅在ALOHA验证，且在MimicGen未通过预设检查，部署需闭环调参。迁移到移动机器人需另加覆盖和风险约束。

#### 总结
核心思想：平滑记忆托住残差

1. 演示逐帧编码并存储未来动作。
2. 全库软召回，结合局部候选与观察。
3. 留出所属示范，离线训练残差头。
4. 执行动作前缀并反复重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34014v1)
- [arXiv](https://arxiv.org/abs/2611.34014v1)

---

<a id='2611.33579v1'></a>
## [DexEngine: Scaling Human Demonstrations into Physically Executable Whole-Robot Data](https://arxiv.org/abs/2611.33579v1)

**Authors:** Wongyun Yu, Honggeun Kim, Donguk Lee, Minchan Kim, Kyoung Whan Choe, Minsu Cho

**Published:** 2026-11-25

**Categories:** cs.RO, cs.AI

**Abstract:**

A growing body of recent work addresses the retargeting of human hand-object demonstrations to robotic hands. To train whole-robot policies, however, the retargeted motions must be kinematically feasible and physically executable, and be produced efficiently at scale. We present DexEngine, a simulation pipeline that turns human manipulation data into train-ready, physically executable arm-hand trajectories. DexEngine first places each clip's hand-object motion where the arms can reach it, finds an arm motion that follows the human hands, and then corrects the motion in physics simulation with residual reinforcement learning. From nine human hand-object datasets, we generate about 5,200 clips for each of four dexterous hands, augmented into 65,000 trajectories (116 h) and 325,000 episodes (578 h) in total. Even with arms attached, DexEngine tracks objects more accurately than prior floating-hand methods and runs over 10x faster than RL-based ones. Policies trained on our data far outperform those trained on the same clips without physics correction or with the arms attached after a floating-hand correction, showing that fine-tuning data need physics correction. Mid-training on our data further improves them at equal fine-tuning budget, in simulation and on a real robot.

### 论文解读
#### 摘要翻译
DexEngine 将人类手物交互数据转化为可训练的全身机器人轨迹。它先把动作放到机械臂可达的位置，再生成臂部运动参考，最后用仿真中的残差强化学习修正接触与物体运动。作者汇集九个数据集、四种灵巧手，构建约 6.5 万条轨迹和 32.5 万个训练 episode；报告了相对浮动手方法更准确、比已有 RL 管线更快的跟踪，并展示数据对仿真及真机策略训练的帮助。
#### 方法动机分析
既有运动学重定向虽能生成动作标签，却不保证接触力足以推动物体；不少物理重定向只控制悬空手，忽略机械臂的可达性、关节限制和碰撞；加入机械臂的 RL 又往往速度慢或覆盖数据少。本文的核心假设是：先找到全身可行的运动参考，再围绕它做物理校正，能避免“手能跟上、手臂到不了”的系统性失败，并让一个策略批量处理多个片段。
#### 方法设计详解
输入为按片段整理的人手腕、手指和物体轨迹，以及物体网格。首先将手姿态重定向到机器人手，搜索场景平移和偏航。预计算的可达性地图筛掉多数位置候选；对剩余候选做连续臂 IK，通过动态规划选取稀疏帧的关节分支，再从粗路径初始化逐帧 IK，检查跟踪误差、关节跳变和碰撞，并按速度限制重定时。物理阶段用 PPO 训练腕部位姿与手指关节的残差策略；随机起点时把手臂复位到同一参考分支，避免重复 IK 引起的分支跳变。每个策略最多处理 64 个片段，训练使用 4,096 个并行环境、1,600 次迭代。接受的轨迹记录关节状态与目标、物体位姿、逐链接接触力、双目 RGB-D 和语言指令；随后通过重新仿真改变场景位姿，并改变渲染外观。
#### 方法对比分析
与纯运动学方案相比，DexEngine 的输出经过接触物理校正；与先校正浮动手再挂接机械臂相比，它把可达性约束前置；与逐片段 RL 相比，策略在片段束间共享。关键不是单独改进 IK 或 RL，而是把场景放置、连续臂参考和物理残差校正组合成可扩展数据引擎。适合固定基座、可获得手物轨迹且需要全身动作监督的灵巧操作系统。
#### 实验分析（精简版）
在同一组 64 个 TACO 片段和 SharpA 手上，带臂且每 64 段共享策略的方案达到 0.90 ADD-AUC、84.4% MP-SR，每片段约 3.9 GPU 分钟；单片段策略虽达 0.97 ADD-AUC，却耗时 290 分钟。三任务模拟中，GR00T 使用物理校正数据微调的平均成功率为训练/留出位姿 65%/50%，运动学回放数据仅 6%/5%。真机策略实验仅有三个任务片段、每条件五次试验，结果说明 mid-training 有潜力，但外推范围有限。
#### 实用指南
论文提供数据集与代码链接，部分受源数据许可限制的内容需由代码重建。复现重点是匹配机器人手模型、机械臂运动学、碰撞几何和接触参数；同时保留连续 IK 分支及统一重置参考。作者以 LeRobot v2.1 组织输出，视频为 30 Hz、控制为 50 Hz。迁移到其他机械臂时，需重建可达性地图并重新评估固定基座可行率；迁移到自动驾驶可借鉴“先施加执行器/动力学约束，再做残差修正”的数据生成思路，但本文未验证车辆控制。
#### 总结
先保证全身可达，再用物理修正人类动作。
1. 清理并切分人类手物轨迹。
2. 搜索可达场景，求连续臂部参考。
3. 通过残差强化学习修正接触，记录并扩增训练数据。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.33579v1)
- [arXiv](https://arxiv.org/abs/2611.33579v1)

---

<a id='2611.33635v1'></a>
## [FreshVLA: Frequency-Regularized Time Scale Hierarchical Latent Actions for Vision-Language-Action Models](https://arxiv.org/abs/2611.33635v1)

**Authors:** Jiaming Liang, Yifeng Zhan, Rongyi He, Mu Li, Jiaqi Fan, Zhiyuan Zhang, Qiwei Liang, Zhengjie Zhang, Boyang Cai, Ningyuan Zhang, Xingjian Jing, Qiang Nie

**Published:** 2026-11-25

**Categories:** cs.RO

**Abstract:**

Learning reusable latent actions from action-free videos offers an effective route to improving Vision-Language-Action (VLA) policies. However, reconstruction-oriented latent actions do not necessarily benefit policy learning: as they may encode task-irrelevant static appearance with scene identity, or entangle slowly varying task progress with fine-grained execution dynamics in limited discrete code. We propose FreshVLA, a VLA framework built on a frequency-guided, dynamics-sensitive learning paradigm for acquiring temporally hierarchical continuous latent actions. FreshVLA uses a high-pass prior to guide visual transition encoding toward dynamical regions, thereby reducing reliance on static appearance cues. It further organizes latent actions into coarse-grained task progress and fine-grained execution dynamics through cross-timescale sampling and frequency-domain regularization, enhancing their capacity to represent long-horizon behaviors. During policy training, we freeze the pretrained latent action encoder and transfer its representations to robot control through joint flow-matching of latent and robot actions. FreshVLA achieves 98.2% average success on LIBERO and 98.0% on LIBERO-Long, exceeding GR00T-N1.6 on the latter by 3.6%. It also reaches an average sequence length of 4.52 on CALVIN and strong overall success rate of 80.4% in the zero-shot LIBERO-PLUS evaluation. Furthermore, FreshVLA exhibits greater robustness to visual perturbations and a more distinct division of roles between two branches in modeling low- and high-frequency motion, respectively. In 35-step action regression, it achieves a 51.3-percentage-point higher relative error reduction than its hierarchical discrete counterpart. Real-world experiments demonstrate that FreshVLA improves the average success rate over $π_{0.5}$ by 7.5%. The FreshVLA project page is available at https://freshvla.github.io/.

### 论文解读
#### 摘要翻译
FreshVLA 从无动作视频学习潜在动作，用于视觉语言动作策略。重建式表征可能记住静态场景线索，或把缓慢任务进度与快速执行细节挤在同一离散代码中。FreshVLA 用时间高通先验关注视觉变化区域，以跨时间尺度结构和频域正则化形成连续层级潜在动作；策略训练时冻结编码器，并联合建模潜在动作与机器人动作。模型在仿真基准、扰动评测和真实操作中取得有竞争力的结果。

#### 方法动机分析
动作标注昂贵且具身间坐标约定不同，无动作视频可扩展视觉变化学习。但能预测下一帧不等于学到控制：背景可能成为捷径，长时操作又兼有阶段进度与局部修正。FreshVLA 假设变化区域更贴近操控，快慢时间结构应由不同分支表达。它用稳定后的固定第三人称视频，避免腕部相机自运动混淆交互变化。

#### 方法设计详解
第一阶段从 36 帧视频窗口提取帧特征：冻结 DINOv3 输出 patch token，DynaAdapter 汇聚帧级表征并预测 patch 注意力。算法对每个 RGB 像素的时间序列做 FFT，保留较高频分量后逆变换，以变化幅度构造注意力目标；模型通过 KL 对齐关注动态区域。逐帧特征差分提供细粒度转移，时间锚点间差分提供阶段转移。阶段编码器提取慢变任务进度并上采样；从逐帧转移中减去它，再由执行分支编码剩余局部动态，组合成连续潜在动作。DCT 约束阶段分支拟合低频成分，组合表示覆盖各频率；共享编码器处理正向和反向转移，并由 Transformer 预测邻帧特征。

第二阶段固定视频预训练所得编码器，将其潜在动作作为策略学习目标。视觉语言模型读取当前观察、指令及机器人状态；DiT 专家从噪声出发，同时对潜在动作流与机器人动作流做 flow matching。推理时两路仍联合去噪，但只执行机器人动作，潜在流提供内部结构约束。实验配置使用 Qwen3.5-2B、7 维动作、12 步动作块；预训练采用 36 帧输入和 6 个子阶段。

#### 方法对比分析
离散 LAM 用有限码本表达帧间变化；FreshVLA 使用连续表征，显式分出阶段进度与执行残差。FAST/MINT 对带标签动作块做频域建模，FreshVLA 则从无动作视频学习潜在转移，作为策略训练目标。它适合长时操作且有无动作视频的场景；下游策略仍需机器人动作数据。

#### 实验分析（精简版）
在 LIBERO 四套任务上，2B FreshVLA 平均成功率为 98.2%，LIBERO-Long 为 98.0%，较 GR00T-N1.6 高 3.6 个百分点；CALVIN 平均连续完成 4.52 个子任务。零样本 LIBERO-PLUS 七类扰动宏平均为 80.4%。真实实验使用 6 自由度 Flexiv 机械臂和 RealSense D435，在四项桌面任务上每项收集 30 条轨迹，平均成功率 84.2%，较 π0.5 高 7.5 个百分点。消融显示，去掉频率正则后 LIBERO-Long 从 98.0% 降至 93.0%。证据覆盖多个仿真基准并含真实机器人验证，但真实任务和硬件覆盖有限，未报告统计区间；扰动类别上也非全面领先。

#### 实用指南
论文提供项目主页（freshvla.github.io）。预训练语料为 227,894 条轨迹、约 2,787 万帧，来自多种真实与仿真数据。Stage I 用 AdamW 训练 120,000 步、batch 128、学习率 1e-4，使用 8 张 A100。复现需关注固定视角、视频稳定、突变帧过滤、36 帧窗口和 6 个子阶段。迁移到腕载相机或自动驾驶时，自运动会破坏“高频即交互变化”的对应，需补充运动补偿或任务相关目标。

#### 总结
核心思想：按视觉变化与时间尺度分解潜在动作。

1. 从视频提取帧级特征，并用高通变化先验引导注意力。
2. 将慢速阶段进度与快速执行残差编码为连续层级潜在动作。
3. 用频域约束和双向预测训练编码器，再冻结它作为策略目标。
4. 联合去噪潜在动作和机器人动作，最终执行机器人动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.33635v1)
- [arXiv](https://arxiv.org/abs/2611.33635v1)

---

<a id='2611.33739v1'></a>
## [RegHarness: Regulating What Code-as-Policy Agents Keep from Steps to Skills](https://arxiv.org/abs/2611.33739v1)

**Authors:** Shengming Liu, Yunyi Gao, Xinyi Wang, Jianyong Sun, Shuai Wang, Yi Deng, Li Chenye, Mingcai Zhou, Sim Kuan Goh

**Published:** 2026-11-25

**Categories:** cs.RO

**Abstract:**

Code-as-Policy agents generate, execute, and revise robot programs, but often lack explicit mechanisms to govern what persists across executions: which actions are retained within a trial and which code modifications carry over to subsequent runs. This leaves two critical aspects of agent behavior insufficiently controlled: step-level execution and cross-run adaptation. We present RegHarness, a two-level harness that governs both forms of persistence for the RATs agent team, without relying on a vision-language-action (VLA) policy or demonstrations. Within each trial, run-time governance evaluates each proposed step from a reset scene, replaying the committed steps, while a fixed-rule controller commits the step only when both RATs' verifier and a task-long LLM auditor approve it. Between runs, a coding agent uses logged failures to develop and test revisions to skills, primitives, and rules, which remain disabled by default until explicitly enabled by the authors. Using a single low-cost model across all run-time roles, RegHarness achieves an 87.8% success rate (158/180) on paired LIBERO-PRO trials, compared with 39.4% for RATs, and a 72.0% success rate on MolmoSpaces trials. A $2\times2$ ablation study of RegHarness's two levels shows that its skill refinement alone increases RATs' successful trials from 71 to 135, whereas run-time governance alone achieves 105 and RATs' native stepwise reset-and-replay mode achieves only 46. These findings suggest that, for Code-as-Policy agents operating with low-cost models, task success depends not only on how generated programs are executed, but also on the quality of the reusable skills they invoke and how those skills evolve across runs.

### 论文解读
#### 摘要翻译
代码策略机器人会生成程序、执行并按失败修订，但通常没有机制明确规定一次试验中哪些步骤能够保留，以及代码修改怎样进入后续运行。RegHarness给RATs增加两层治理：试内从重置场景测试每个候选步骤，只有验证器与任务级审计器都同意才提交；试间由编码代理依据失败记录修改技能、基础操作和规则，改动默认关闭并由作者启用。系统不依赖VLA策略或示范。低成本模型下，它在LIBERO-PRO达到87.8%，在MolmoSpaces达到72.0%；消融显示技能修订贡献大于单独的运行时治理。

#### 方法动机分析
生成整段机器人程序后再检查，会让失败动作改变现场，后续重试也可能改写已成功的步骤。重置有助于重复试验，却会抹去故障现场，所以系统要保留步骤记录与审计上下文。作者假设：执行干净、验证器通过且审计器认可，步骤才能提交；长期能力则来自经测试且由作者批准的技能修订。reset本身不等于治理：强制提交仍可能保留错误步骤。

#### 方法设计详解
输入为任务文本、保存的初始仿真场景、技能/API文档和当前已提交步骤；输出为最终执行计划与跨运行更新的技能库。单次试验最多4个计划，每步最多4次尝试。流程是：planner生成计划，审计器接受、改写或拒绝；coder逐步写代码，静态检查与代码审计筛除错误；执行器重置场景、重放已提交前缀并运行草稿；verifier评估步骤，step audit结合日志、前后帧和预算提出建议；controller按规则决定提交、重试、重规划或终止。只有执行成功、verifier明确通过且auditor同意，步骤才提交。任务检查由控制器读取，不向模型开放。计划整体未达标时，final audit将失败原因交给下一计划。跨试验阶段，Claude Opus 5.5分析失败日志，提出默认关闭的skill、primitive或rule开关；开关通过单测、关闭时字节等价、对抗审查和泄漏审计后，再由作者启用。关键直觉是把“发生了什么”与“什么可以留下”分开，并以长时审计上下文跨过物理重置。

#### 方法对比分析
RATs整段执行后再验证并重跑；新方案逐步执行和提交。RATs已含逐步reset-and-replay模式，但仅按verifier提交，尝试耗尽时会强制保留未验证步骤；该模式结果低于RATs。它不只增加LLM审计，还修改共享技能库，联合调节试内执行与跨试验代码演化。它适合可重置仿真、能记录执行轨迹、操作可拆为步骤且技能跨任务复用的场景；依赖自动重置的部分不适用于未经改造的真机流程。

#### 实验分析（精简版）
在LIBERO-PRO初始状态0–2的180个配对试验中，RegHarness为158/180（87.8%），RATs为71/180（39.4%）；仅技能修订的RATs达到135/180（75.0%），仅运行时治理为105/180（58.3%），RATs原生逐步重放模式为46/180（25.6%）。系统在0–4五个状态共260/300（86.7%）。但状态0参与开发，占配对数据三分之一；配置间还同时变化提示、预算、时限和规划规则，因此消融不能当作纯组件因果分解。MolmoSpaces为同一批场景的重复开发评估。真机Kinova Gen3结果为28/45对21/45，差异不显著（p=0.20）；reset-and-replay仅限仿真。

#### 实用指南
论文说明每任务最多4个计划、每步最多4次尝试，执行模型为deepseek-flash高思考强度；审计器与代码模型使用同一低成本模型，离线技能修订使用Claude Opus 5.5。复现重点是固定初始状态和预算，明确区分任务成功检查与模型可见信息，并逐次记录重放、审计建议及最终提交；测试时应把开发状态与留出状态分开。文中基于RATs代码修改。迁移时可借鉴版本化技能、确定性提交门槛和作者审核；真机需用安全恢复替代仿真重置，并重新校准夹持阈值与传感器证据。

#### 总结
核心思想：治理步骤提交与技能演化
1. 任务拆成程序步骤，审计计划与代码。
2. 重置场景、重放已提交前缀，测试当前步骤并按规则提交或重试。
3. 汇总失败，离线修改默认关闭的技能开关，经测试与人工批准后用于下一轮。
4. 用配对基准、留出状态和硬件试验分别检验收益与边界。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.33739v1)
- [arXiv](https://arxiv.org/abs/2611.33739v1)

---

<a id='2611.33772v1'></a>
## [Triage Before Fine-Tuning: Feasibility-Gated Selection and Monitored Adaptation of Embodied Foundation Models for a Fixed Physical Platform](https://arxiv.org/abs/2611.33772v1)

**Authors:** Zilin Huang, Zhengyang Wan, Zihao Sheng, Boyue Wang, Ruolin Yang, Yuhao Luo, Sikai Chen

**Published:** 2026-11-25

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Embodied foundation models are increasingly deployed by selection: a practitioner picks a public checkpoint by benchmark rank and fine-tunes it on local data. The platform is fixed before the model is chosen, with little or no local data. We ask what such a checkpoint loses on a fixed platform and which part of the loss local data can buy back. We split the gap by what can repair it. A sensing gap, information the sensors do not capture, cannot be closed with their data; a latency gap, output arriving after the control loop needs it, belongs to the loop, not the weights; only a distribution gap is fine-tuning's to repair. TRIAGE turns this into feasibility-gated selection and monitored adaptation in five stages: gates on the sensor contract and on end-to-end latency that no score can override, probes on ten minutes of local logs, a ranking by target-relevant score among the candidates that pass, and fine-tuning that is monitored and checked in closed loop before release. On driving benchmarks, a deadline below a leading closed-loop model's latency makes it collide on 80 to 90% of routes, and losing field of view moves plans 2.2 to 5.7 times more than losing resolution at equal pixels. Local frames close 75% of a stability gap but raise perturbation sensitivity at small budgets, and picking by score violates a gate in two of four deployment profiles. On a full-scale vehicle, the factory camera failed the sensing gate and an added camera repaired it. The latency gate removed the two highest-scoring candidates, neither of which completed a closed-loop trip. The selected model passed acceptance in 4 of 5 closed-loop trips, and the closed-loop check withheld a fine-tuned model that had improved offline. TRIAGE lets organizations that cannot train foundation models from scratch find out whether a public checkpoint can run on the robot or vehicle they own, and what to fix if not.

### 论文解读
#### 摘要翻译
具身基础模型常通过选择公开检查点并用本地数据微调来部署，但实际平台的传感器、算力和控制接口早已固定，起步时往往缺少本地数据。本文把模型与执行接口视为整体，区分三类差距：传感器未捕获的信息属于感知差距，输出赶不上控制周期属于延迟差距，只有训练域不同属于分布差距。作者提出 TRIAGE：先检查传感契约与端到端时限，再用短日志探测，之后仅在可行模型中按目标分数排序，微调后还须监控并通过闭环验收。

#### 方法动机分析
排行榜回答模型在特定基准上的能力，却不保证它能读取目标车辆的相机，也不保证结果在控制器需要时到达。若缺少视野，更多同类数据无法补出从未观测的信息；若命令迟到，改进权重也不会改变计算路径或旧输出的消费方式。作者的核心假设是：部署差距应先按可修复原因定位，再决定是否买数据、换传感器或改执行架构。

#### 方法设计详解
TRIAGE把候选检查点和传感/计算/控制接口组成配置，并按五步运行。第一，核对模型实际消费的相机等通道；缺失视角不能由分数补偿，可转换的输入需作为新配置。第二，在目标硬件和实际运行负载下测从最新图像到规划结果可供控制器使用的第95百分位时延，并与预先设定的控制期限比较；只测推理时间不足以证明通过。第三，在约十分钟本地日志上测时间稳定性，以及模型对亮度、压缩、视场和分辨率变化的响应，并用图像盲基线、转弯方向检查和运行噪声校验结果。第四，只在通过两道门的候选中按目标相关基准分数排序，同时保留挑战者。第五，用本地数据适配时监控跟踪误差、扰动敏感性和图像依赖，最后在闭环中验收模型及转换接口。感知差距要改传感器或模型输入；延迟差距要改算力或控制消费方式；分布差距才交给数据适配。

#### 方法对比分析
与直接按榜单选模型或用迁移估计器排序不同，TRIAGE先判断模型是否能接入平台并按时工作，再排序。低成本探针有助于发现盲模型、异常响应和模型层级，但实验发现其不足以稳定地区分同一层级的候选，因此作者仍采用可行集内的基准排序。与把所有域差异视为微调问题相比，该框架明确给出失败后应改的系统部件，也把闭环验收列为发布条件。

#### 实验分析（精简版）
在驾驶与操作模型中，像素预算相同时，裁去视场导致的计划/动作变化更大：驾驶模型为降分辨率的2.2–5.7倍，六个操作模型为1.6–2.4倍。CARLA中，SimLingo面对500 ms命令期限时仍继续行驶，但80–90%的路线发生碰撞；旧图像配合新车身状态则接近基线。全尺寸车辆上，原厂相机未满足视场要求；添加广角视图后，时延门控排除两个高分模型，DriveVLA-W0在143 m路线中完成4/5次。样本量很小，物理验证只覆盖单车短路线；Bridge V2结果为离线推理，不能证明机器人实机表现。

#### 实用指南
作者称将以开放许可证发布代码、探针定义和逐运行结果，模型权重仍遵循各自许可。车辆微调示例用2,667个本地样本训练五轮、三个随机种子，每次约16分钟/单GPU。复现时需锁定评估器版本，按日志划分训练与留出集，明确端到端计时起止点，并根据控制器是保持命令还是跟踪旧规划确定期限。对自动驾驶系统，可将图像采集、重投影、编码、推理、解码、轨迹发布和控制器消费纳入实测；这属于基于本文原则的迁移建议。

#### 总结
核心思想：先判可行，再做适配。
1. 核对模型需要的传感器输入。
2. 在目标硬件上测端到端时延并设门槛。
3. 用短日志诊断输入响应与分布差异。
4. 可行模型按目标分数选择，微调后闭环验收。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.33772v1)
- [arXiv](https://arxiv.org/abs/2611.33772v1)

---

<a id='2611.34220v1'></a>
## [Variation-aware Temporal Correspondence for Self-Supervised Video Representation Learning](https://arxiv.org/abs/2611.34220v1)

**Authors:** Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, Qingming Huang

**Published:** 2026-11-25

**Categories:** cs.CV

**Abstract:**

Self-supervised video representation learning has achieved remarkable progress by leveraging large-scale unlabeled video data. Recent methods commonly extend Masked Image Modeling (MIM) to Masked Video Modeling (MVM), where video representations are learned by capturing semantic information from temporal contexts during masked content recovery. Unlike static images, videos exhibit inherent temporal variation, as visual objects continuously change their positions, appearances, and motion states across frames. Therefore, MVM needs to preserve semantic consistency while capturing temporal changes of corresponding regions, which raises two central challenges: 1) how to establish reliable temporal correspondence across frames, and 2) how to formulate an appropriate representation learning objective for modeling both semantic consistency and temporal variation. To address these challenges, we propose Variation-aware Temporal Correspondence for video Representation learning (VT-CoRe) framework. For challenge 1), VT-CoRe introduces a sandwich sampling strategy to construct two-sided temporal references and a Patch Registration Module (PRM) to establish reliable patch-level correspondence. For challenge 2), VT-CoRe imposes latent-space semantic alignment on registered representations through a self-distillation architecture and further proposes temporal variation modeling to constrain representation trajectories over time. Experiments on 8 downstream tasks across 16 benchmarks demonstrate the effectiveness and applicability of VT-CoRe across pixel-level, frame-level, video-level, and vision-based robotic learning scenarios. Complementary representation analysis shows that VT-CoRe learns a more structured latent space for variation-aware temporal correspondence learning.

### 论文解读

#### 摘要翻译

本文提出 VT-CoRe，一种自监督视频表征学习框架。它通过前后双侧采样和 Patch Registration Module 建立跨帧 patch 对应，再用自蒸馏保持对应区域的语义，并以时间变化约束刻画表示轨迹。作者在 8 类任务、16 个基准上评估，覆盖像素、帧、视频和视觉机器人学习。

#### 方法动机分析

图像掩码建模只需从空间上下文恢复缺失区域；视频中的物体会移动、变形或改变状态，单侧参考帧可能容许多种目标解释，像素重建也容易偏重纹理。作者的核心假设是：短时间内同一视觉区域具有稳定语义身份，同时其潜在表示应连续变化，而非被压成完全相同的静态特征。

#### 方法设计详解

输入视频以 2 FPS 取帧。每次选择当前帧，并按时间偏移比例 0.10–0.20 取过去和未来帧；遮挡当前帧部分 patch。ViT 编码三帧后，PRM 用当前帧 patch 作 query、辅助帧特征作 key/value，以交叉注意力将前后特征注册到当前帧的空间布局，再用自注意力聚合上下文。前后匹配置信度相乘，选出排名前 60% 的位置作为可靠对应。

随后，学生分支注册后的过去、未来特征分别匹配 EMA 教师对当前帧产生的语义目标，并约束两侧预测一致。另一条变化损失先将三态投影到 256 维空间，以不等间隔有限差分近似一阶、二阶导数，形成二阶轨迹残差。模型设置 3 组可学习系数，以软分配选择适合每个视频片段的局部变化模式。该损失在训练进度 60% 后逐步启用，避免早期错误对应影响轨迹学习。

#### 方法对比分析

相比密集采样的像素重建，VT-CoRe在潜在语义空间监督；相比仅用过去帧预测当前帧，它用过去与未来共同约束插值；相比只拉近邻帧特征，它保留不同时间状态的变化方向。关键组合是“先注册并筛选对应，再在对应轨迹上分别约束语义与变化”。

#### 实验分析（精简版）

在 Kinetics-400 预训练的 ViT-B/16 上，DAVIS 视频分割 J&F 为 67.0，DINOv2 为 64.9；VIP 人体部件 mIoU 为 40.6，对照 38.2。模拟 CortexBench 的冻结特征模仿学习四域平均分为 65.8，优于比较方法至少 3.7 点。消融支持置信筛选和投影变化约束的作用。局限是机器人证据仅来自仿真行为克隆，且对照方法的预训练数据和计算预算不完全一致。

#### 实用指南

复现可从 ViT-B/16、2 FPS、Kinetics-400、batch size 128 起步；像素任务训练 200 epoch，其他任务 400 epoch。变化项的默认设置为 3 组系数、256 维投影和 0.6 patch 保留率。需要重点监测遮挡、长间隔和大幅运动下 PRM 的对应可靠度。论文未给出代码发布信息，也没有真实机器人验证。迁移到机器人视觉可将其作为视觉骨干预训练或跟踪特征方案，再在实机采集数据上单独验证闭环效果。

#### 总结

核心思想：对应区域语义稳定，表示轨迹可变。

1. 遮挡当前帧，采样过去与未来帧。
2. 注册两侧 patch 到当前坐标，并筛选高置信对应。
3. 用教师蒸馏对齐语义，再以二阶残差约束时间变化。
4. 将学得的视频特征迁移到分割、理解和模拟视觉控制任务。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34220v1)
- [arXiv](https://arxiv.org/abs/2611.34220v1)

---

<a id='2611.34449v1'></a>
## [HoloBrain-0.5 Technical Report](https://arxiv.org/abs/2611.34449v1)

**Authors:** Xuewu Lin, Jiahao Wang, Jiawei Li, Mengdi Li, Zhengmao Sun, Yiwei Jin, Hongyu Xie, Zhengyang Feng, Kun Wu, Zihao Chen, Lianzhou Zhang, Lichao Huang, Zhizhong Su

**Published:** 2026-11-25

**Categories:** cs.RO

**Abstract:**

We present HoloBrain-0.5, a family of vision-language-action (VLA) models for manipulation across robot embodiments. The policies plan in a shared action space using robot state, kinematic structure, and calibrated camera observations. We standardize joint axes, coordinate frames, and tool-center points in robot descriptions (URDFs) and encode their kinematic graphs in a shared 3D representation. HoloBrain-0.5-Continuous generates action chunks by diffusion, while HoloBrain-0.5-Discrete uses residual finite scalar quantization (FSQ) and delayed autoregression. We pretrain both policies on over 10,000 hours of real-robot, simulation, and human-manipulation data spanning 15 embodiment categories.   Across six closed-loop simulation settings, the discrete policy converges faster and reaches a higher mean success rate than the continuous policy (56.77% vs. 49.71%). On seven real-robot tasks, it achieves a mean progress score of 87.76% without task-specific post-training. After post-training, the continuous policy outperforms $π_{0.5}$ on the seven-task benchmark (94.52% vs. 86.25%) and an 11-task dual-arm benchmark (79.07% vs. 69.41%). The discrete-initialized continuous policy transfers five tasks zero-shot from Piper to Piper-X, scoring 63.28% on average. With target-robot demonstrations, the continuous policy scores 89.2% on Franka Panda and 92.94% on a mobile dual-arm platform. Although pretraining includes no Unitree G1 data, post-training with limited G1 demonstrations yields a mean progress score of 70.24% across three humanoid tasks, compared with 11.00% for $π_{0.5}$.

### 论文解读
#### 摘要翻译
HoloBrain-0.5 是面向多种机器人形态的视觉—语言—动作模型。它把机器人状态、运动学结构与经标定的相机观测映射到共享动作空间，并提供连续扩散和离散动作两种策略。训练数据超过一万小时，涵盖实机、仿真与人手操作。离散策略在六种仿真设置上平均成功率为 56.77%，连续策略为 49.71%；实机任务后训练后，连续策略在七任务上达到 94.52%，高于 π0.5 的 86.25%。

#### 方法动机分析
跨本体数据不仅外观不同，关节数、轴向约定、坐标系、TCP 和相机布局也不一致，因此相同动作数字未必有相同物理含义。作者的驱动力是把这些差异显式写入标准化运动学图，并用相机标定把视觉和机器人基座联系起来。核心假设是：共享几何语义能让不同机器人数据共用动作表示，减少策略从数据中猜测本体对应关系的负担。

#### 方法设计详解
输入包括多相机图像、语言指令、机器人数值状态、标准化 URDF 和相机内外参，输出是一段未来动作。首先把转动或移动关节的运动轴规范到局部 +z，并补偿子链接变换，保证关节零位、连杆几何和旧轨迹的物理效果不变；再选关节子链接作为 6 自由度动作锚点，加入 TCP、夹爪及关节图距离，得到节点数可变的 3D 表示。连续分支使用 Qwen3.5-2B 骨干和扩散动作解码器；图像、文本、时序关节三路注意力并行计算，再由查询相关路由融合。离散分支先以虚拟关节聚合运动链轨迹，再通过三阶段残差有限标量量化生成共享代码，tokenizer 解码器根据当前状态重建动作。代码按虚拟关节、时间块和残差级别的对角顺序延迟自回归，兼顾并行预测和已生成代码依赖。离散 tokenizer 的训练设置为动作块长 64 帧，每 4 帧一时间块，共 16 块；隐藏宽度为 384，三层编码器和三层残差量化。策略预训练评估到 300k 训练步。

#### 方法对比分析
与把固定关节向量直接拼接或量化的基线相比，本文的创新贡献是用标准化 URDF 与图距离为不同拓扑提供共享几何语义，再以虚拟关节把变长轨迹压到共享代码空间。连续扩散适于细粒度动作生成；离散表示在预训练时更快收敛，但少样本后训练可能不稳。论文结果显示两者优势依赖训练阶段，不能据此认定某一种表示处处占优。

#### 实验分析（精简版）
预训练使用 10,006.5 小时、15 类本体数据。跨三种模拟器的六设置闭环评测中，离散策略平均成功率 56.77%，连续策略 49.71%。七项实机任务各用 200 条示教后，连续策略平均进度分 94.52%，对比 π0.5 的 86.25%。另有 Piper 到 Piper-X 的五项任务组合零样本迁移，平均进度 63.28%；但目标机器人在其它任务上已见过。实机分数按任务里程碑加权，未报告跨重复运行方差，且不同评测的后训练预算不一。

#### 实用指南
作者提供 HoloBrain 代码、UniURDF 与模型权重入口。复现重点是准确的 URDF 轴和 TCP 规范、可靠相机标定、关节轨迹变换校验，以及数据频率和尺度统一。迁移到移动机器人时可将底盘自由度纳入图；用于自动驾驶则只是设计启发，车辆动力学、安全约束和长时序规划都需要重新建模。当前策略主要依据当前观测，记忆、触觉和高层语义监督仍待加入。

#### 总结
核心思想：先统一本体几何，再共享动作学习
1. 规范 URDF、关节轴、TCP 与相机坐标。
2. 把各机器人轨迹编码为同一 3D 关节图。
3. 用扩散或延迟自回归策略预测动作片段。
4. 通过仿真、实机和跨本体实验检验迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34449v1)
- [arXiv](https://arxiv.org/abs/2611.34449v1)

---

<a id='2611.34489v1'></a>
## [Who Is Ego? Scaling Cross-View Camera Wearer Identification and Segmentation beyond Humans](https://arxiv.org/abs/2611.34489v1)

**Authors:** Yifan Jin, Zheng Zhang, Tianwen Qian, Dong Wang, Bo Zhao, Yanwei Fu, Yuqian Fu

**Published:** 2026-11-25

**Categories:** cs.CV, cs.RO

**Abstract:**

A first-person (ego) image reveals what a camera sees, but not who or what carries it. Given a synchronized third-person (exo) image, camera wearer identification asks who is ego: which candidate carries the ego camera? We extend this task beyond humans to robots, vehicles, animals, and drones, and broaden its spatial scope (where is ego) from selecting a provided candidate mask to wearer segmentation with absence prediction and mutual camera localization across ego-exo views. We introduce FindEgo-1M, comprising 1.16M real and simulated ego-exo pairs across 12 wearer types, 17 domains, and 111 test settings. Our approach exploits a geometric relation: the ego camera's projection into the exo image, i.e., the epipole, lies on or near its wearer. We introduce EgoRay-ZS, the first method to turn a frozen feed-forward 3D reconstruction model, VGGT-$Ω$, into a training-free solution for wearer identification and segmentation. Its geometric readout ranks candidate masks or segmentation proposals by angular offset and distance to the ego camera, while the relative pose provides mutual camera directions. We then develop the full EgoRay model through pose-supervised training on FindEgo-1M and label-free test-time adaptation constrained by candidate masks. Without task-specific training, EgoRay-ZS achieves 78.0% identification accuracy on TF2023's original labels, versus 52.9% reported for the dataset-trained PEN. On FindEgo-1M, EgoRay achieves state-of-the-art chance-corrected accuracy of 87.1 and 77.7 on seen and unseen settings. On matched queries, EgoRay also outperforms GPT-6 Astra by 31.6 and 21.3 points on seen and unseen settings, respectively, while running $18\times$ faster. Data, code, and models will be released.

### 论文解读

#### 摘要翻译
第一视角图像显示相机所见，却不显示谁携带相机。本文将同步ego/exo图像中的佩戴者识别扩展到机器人、车辆、动物和无人机，并研究无需给定候选掩码的佩戴者分割、缺席预测及双向相机定位。作者提出FindEgo-1M（116万真实与仿真图像对、12类佩戴者、17个域、111种测试设定）和几何方法EgoRay。冻结3D重建模型VGGT-Ω构成零样本EgoRay-ZS；位姿监督训练与无标签测试时适配得到EgoRay。零样本模型在TF2023原始标签上准确率78.0%，高于PEN已报告的52.9%；EgoRay在FindEgo-1M已见/未见识别κ-Acc为87.1/77.7。

#### 方法动机分析
既有方法依赖外观、运动或人群匹配，易被相似个体混淆，也局限于人体。本文基于“相机由佩戴者承载”：ego相机投影到exo图像的epipole应在佩戴者附近。由此将身份判断变成几何与掩码对应，并可扩展到定位、分割。前提是两图同步，且图像对能提供足够准确的相对位姿。

#### 方法设计详解
输入同步ego/exo图像，VGGT-Ω预测相对位姿、内参和exo深度，进而确定ego相机在exo视图中的方向。对每个候选掩码，逐像素计算其视线与预测方向的夹角，取最小角偏差；再将掩码深度反投影到3D，估计候选到相机的归一化距离。最终分数是在掩码像素上取最小的“距离＋3倍正弦角偏差”，最低分候选即佩戴者；掩码内最小值允许相机装在身体不同部位。为减轻模型对输入顺序的敏感性，交换图像次序推理后平均结果。

分割时由SAM 3按类别词生成exo候选，再按同一几何分数排序；若最佳候选角偏差超过阈值，或ego相机预测在exo相机后方，则判缺席。相对位姿还提供两个方向的相机定位。训练版对VGGT-Ω的rank-64 LoRA（1.65%参数）用真实位姿监督方向、旋转、焦距及候选排序。测试时每集再用无标签候选掩码适配200步，只要求预测投影落在某个候选上；旋转和焦距锚定项限制漂移。25条查询可保留约三分之二适配收益。

#### 方法对比分析
EgoRay将3D模型的相对位姿转成显式候选排序，无需零样本识别头。比单用方向的方法多了距离线索，可区分近似共线目标，但远处小目标的深度误差会削弱效果。相较多模态模型，它更依赖可恢复的跨视角几何，在相似机器人或无人机任务有优势；共享场景信息少时仍可能失败。

#### 实验分析（精简版）
完整测试中，EgoRay已见/未见识别κ-Acc为87.1/77.7，零样本为60.5/58.7；未见佩戴者L3为55.9，最强其他3D模型46.4。未见分割gIoU@20为55.2，互定位MA@5为50.6。TF2023原始标签协议下零样本准确率78.0%，PEN报告值为52.9%。非人类测试多数依赖仿真；水下低共享视野、密集动物仍难，且适配会降低部分场景的缺席判断准确性。

#### 实用指南
作者称数据、代码和模型将发布。复现需VGGT-Ω、SAM 3和图像对；训练默认rank-64 LoRA，适配每集200步，25条无标签查询可保留大部分适配收益。迁移到机器人或自动驾驶可尝试以相机投影约束关联车辆或机器人，但须验证同步、鱼眼畸变、遮挡、外参变化及缺席阈值，并单独统计误报与定位误差。

#### 总结
核心思想：用相机几何寻找佩戴者

1. 输入同步ego/exo图像。
2. 预测位姿与相机投影。
3. 按角度和距离为候选掩码排序。
4. 以位姿监督、无标签候选约束适配并输出结果。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34489v1)
- [arXiv](https://arxiv.org/abs/2611.34489v1)

---

<a id='2611.34801v1'></a>
## [Guidance as Variance Matching](https://arxiv.org/abs/2611.34801v1)

**Authors:** Ali Backour, Artem Lukoianov, Frank Permenter, Tommi Jaakkola, Justin Solomon, Vincent Sitzmann

**Published:** 2026-11-25

**Categories:** cs.CV, cs.LG

**Abstract:**

Guidance repels the prediction of a diffusion model away from a weaker reference prediction. It substantially improves sample quality, but its mechanism remains poorly understood. We first find that, at the scales used in practice, guidance does not improve the prediction of the posterior mean (the model's training target); it makes the prediction worse, while samples improve. Second, posterior-mean predictions have less variance than the data, and we find that guidance restores it. We introduce variance-matched guidance: a closed-form rule that picks the guidance scale by matching the variance of the guided predictions to that of the data. Without tuning, it recovers the hand-picked scales of eleven widely used models across image, video, audio, motion, and world generation; to our knowledge, no previous explanation of guidance predicts them. Applied per noise level, it yields a schedule without tuned hyperparameters that users prefer to hand-crafted ones on SD 3.5 Medium. Third, we offer an interpretation of the mechanism behind guidance: each model makes a random error in predicting the posterior mean, and guidance rescales the difference of these errors into a sample with the posterior's covariance, which improves sampling. This view also suggests two new hypotheses that our experiments support: guidance can work with negative scales, and guiding a model with its own checkpoint needs a larger scale than guiding it with an independently trained model.

### 论文解读

#### 摘要翻译

扩散模型的 guidance 让预测远离较弱参考模型，常能提升生成质量，但其机制和尺度选择长期依赖经验。本文发现，常用 guidance 会让去噪预测更偏离训练目标后验均值，却提升最终样本；后验均值本身的方差低于数据，而 guidance 能补回这部分方差。作者据此提出方差匹配 guidance（VMG），用闭式规则匹配 guided 预测与数据的方差。该规则无需调参即可复现图像、视频、音频、动作及世界生成领域 11 个模型的已发布尺度；逐噪声级应用后得到无需手调的 schedule，在 SD 3.5 Medium 的偏好比较中胜过手工方案。论文还把 guidance 解释为放大两个模型预测后验均值时产生的随机误差，并验证负尺度 guidance 及 checkpoint 配对需要更大尺度两项推论。

#### 方法动机分析

传统 CFG 常被解释为增强条件信号，autoguidance 则常被解释为抵消弱模型的系统误差，但都难解释为何不同模型需要不同的大尺度。作者指出，generation quality 与单步去噪误差不是同一目标：实验中 guidance 甚至让后验均值预测误差增加。因此，尺度应从最终采样需要的统计性质来确定。核心假设是后验均值丢掉了 x₀|xₜ 的不确定性，造成方差收缩；两种模型的随机预测误差近似沿着后验协方差的方向，适当外推便恢复这种不确定性。

#### 方法设计详解

给定弱模型 D₀ 和主模型 D₁，标准 guidance 为 D_w(xₜ)=D₀(xₜ)+w[D₁(xₜ)−D₀(xₜ)]。VMG 在少量校准样本 x₀ 上按采样噪声级加噪，分别计算两模型预测，并估计模型差值 Δ、两模型输出的协方差以及真实数据方差。把 Var(D_w) 设为 Var(x₀)，得到关于 w 的二次方程，取论文定义下的正根作为尺度；对每个噪声级单独求解即得 schedule，聚合各级统计量则得到常数尺度。该过程只需已有模型前向计算，不训练新模型，也不需要先生成样本再对 FID 等指标网格搜索。直觉上，模型预测是后验均值的带噪估计，弱强模型之差携带估计误差；沿此差值外推，既不改变均值目标，又可放大误差协方差来恢复后验所缺的方差。

#### 方法对比分析

与“锐化条件分布”的解释不同，VMG 也适用于没有条件差异的 autoguidance，并强调预测误差的协方差结构。与“消除共有模型误差”不同，它认为两模型差异提供了随机方向，方差匹配会确定放大倍数。与传统尺度网格搜索相比，VMG 直接从校准数据求尺度，且不依赖选择 FID、DINOv2 等特定质量指标；与手工 noise schedule 相比，它逐噪声级自动拟合。代价是理论依赖模型差值与后验协方差对齐；条件信息可能改变方向，因此 CFG 不一定满足这一理想假设。

#### 实验分析（精简版）

在 11 个跨模态模型上，预测尺度与论文报告的实际尺度 Spearman 相关系数为 0.96；例如 SD 1.5 得到 7.44，发布值为 7.5。SD 3.5 Medium 用 Euler 40 步、COCO 图像校准后，VMG 的 FID 为 20.13，发布尺度结果为 20.58；DINOv2 distance 由 242.4 降到 224.0。41 人、1,458 票的非正式偏好研究也更偏好 VMG 常数和 schedule，但只测试 SD 3.5 Medium。跨域并非全指标改善：CogVideoX 的 FVD、AudioLDM2 的 FAD 变差；偏好研究也不是大规模或独立复现。

#### 实用指南

论文提供了项目网页 ottogin.github.io/vmg；算法不需要训练，需访问少量代表性校准数据和弱/强模型。复现 SD 3.5 Medium 时，推理设置为 Euler 采样 40 步；作者以 5,705 张 COCO 校准图、40 个噪声级拟合，评估 10,000 个 held-out prompt。约 256 张校准图即可把常数尺度的标准差控制在 0.06。迁移到扩散策略或机器人动作生成在结构上可行，但需要以动作质量、任务成功率和安全指标重新标定，论文未提供机器人实验。

#### 总结

核心思想：匹配方差以免手调 guidance

1. 少量真实样本按采样噪声级加噪。
2. 计算弱、强去噪器的预测差与输出方差。
3. 解二次方程，使 guided 输出方差匹配数据方差。
4. 得到常数尺度或逐噪声级 schedule，再用于采样。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.34801v1)
- [arXiv](https://arxiv.org/abs/2611.34801v1)

---

