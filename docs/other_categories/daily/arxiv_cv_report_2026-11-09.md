time: 20261109

# Arxiv Computer Vision Papers - 2026-11-09

## Executive Summary

## 今日观察：从视觉表征走向可执行的具身系统

这十篇论文呈现出两条相互交织的主线：一是让视觉与语言模型更可靠地理解空间、动作和物理接触；二是让机器人系统在数据、算力或反馈有限时仍能高效适应。研究不再只比较静态识别分数，而是将证据获取、动作选择、接触转移、仿真评估和在线反馈放进完整流程中检验。

在具身控制方面，ActPrune根据动作相关性压缩视觉输入，在多个仿真任务与四项真实机器人任务上报告最高1.61倍加速和最高59.5%的FLOPs削减；FATHOM让世界模型为候选轨迹打分，在真实AUV跟踪中降低了名义与困难条件下的预测误差；PRISM把预先学习的动作原语路由到新任务，改善了RoboCasa365上的原子与复合任务表现。HOC2R则以手物接触轨迹为中间表示，将人类动作迁移到机器人，接触状态与物体位姿结果均优于其比较基线。这些工作共同提示：结构化中间表示和任务相关计算，可能比单纯扩大模型更直接地改善具身系统的效率与可控性。

另一组论文关注视觉证据与物理规律。毫米波人体状态感知研究显示，增加第二个视角可将平均误差从97毫米降至65毫米，强调传感器布置对机器人感知的实际价值。GeoJeV用空间概率引导高分辨率遥感图像中的证据采集，平均准确率由43.91%提升到49.23%，同时增加约1.12秒处理时间，反映精度与搜索成本的取舍。MotionCore以潜在神经模拟器支持带刚体动力学的视频生成，基准结果显示出有希望的物理一致性信号，但实验范围和种子数仍要求谨慎解读。Lodos-v2通过原始变量的精确摩擦接触求解改善速度，在不同MuJoCo模式中相对既有求解器表现差异显著，说明加速收益依赖问题表示与求解设置。

模型行为与评估也受到关注。视觉语言动作模型的目标表征编辑研究发现，否定指令并不总能被模型稳定处理，并展示了对内部目标层进行干预的可能性；证据来自受控的模拟场景，距离广泛部署结论仍有差距。Flex-RSI提出递归自我改进评测，发现分实例反馈对开源模型帮助明显，而严格的物理与精确任务仍然困难；其排行榜适合作为特定基准下的比较，不应等同于通用能力排序。

建议优先完整阅读FATHOM、PRISM与HOC2R，以比较世界模型评分、动作原语和接触轨迹三种不同的具身中间层；关注ActPrune与GeoJeV，可了解任务相关压缩和空间证据采集如何交换计算成本与性能；若关心评估可信度，则可将Flex-RSI、MotionCore与目标表征编辑研究并读，重点查看任务范围、基线、种子和失败条件。总体趋势是把模型能力放进明确的决策链与可核验的物理或任务指标中，同时更认真地报告限制和适用条件。

---

## Table of Contents

1. [What Limits mmWave Human-State Sensing for Robot Perception?](#2611.08863v1)
2. [PRISM: Primitive Routing via In-context Skill Mixing for Retentive VLA Adaptation](#2611.08917v1)
3. [Localizing and Editing the Goal Representation of a Vision-Language-Action Model](#2611.08895v1)
4. [FATHOM: World-Model-Guided Diffusion Action-Trajectory Selection for Sparse-Marker AUV Tracking](#2611.08898v1)
5. [HOC2R: Hand-Object Contact Trajectories for Human-to-Robot Motion Transfer](#2611.08920v1)
6. [ActPrune: Action-Grounded Visual Token Pruning for Efficient Vision-Language-Action Models](#2611.08827v1)
7. [Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents](#2611.08946v1)
8. [GeoJeV: Unified Spatial Probability for Training-Free Efficient Evidence Acquisition in Ultra-High-Resolution Remote Sensing](#2611.08879v1)
9. [MotionCore: A Latent Neural Simulator for Video Generation with Rigid-Body Dynamics](#2611.08904v1)
10. [Lodos-v2: An Efficient Primal Formulation of the Exact Frictional Contact Problem](#2611.08868v1)

---

## Papers

<a id='2611.08863v1'></a>
## [What Limits mmWave Human-State Sensing for Robot Perception?](https://arxiv.org/abs/2611.08863v1)

**Authors:** Shuntian Zheng, Guangming Wang, Hongyang He, Shuai He, Yu Guan

**Published:** 2026-11-06

**Categories:** cs.RO, cs.HC

**Abstract:**

Assistive robots and search-and-rescue platforms require reliable human state sensing in smoke, darkness, and privacy-sensitive environments, where cameras and LiDAR may be unreliable Millimeter-wave radar offers an alternative that works without ambient light, penetrates smoke, and does not capture visual appearance. However, the compact arrays that fit robot size, power, and cost budgets have limited apertures, producing joint-localization errors on the order of decimeters at a few meters, large enough to compromise the safety margins and tracking accuracy a robot depends on. Existing pipelines reduce pose error through frame fusion, multiple viewpoints, return selection, or stronger estimators, but examine one technique at a time, without relating its effect to a shared measurement limit or separating gains from added information from gains from estimator completion. We address this gap with a framework that models the human and environment as one time-varying scattering field and states a Cramer-Rao type error floor for joint localization as a function of bandwidth, aperture, signal-to-noise ratio, fused frame count, phase coherence, and visibility of retained returns. We then use existing frame fusion, multi viewpoint, return selection, and pose estimation methods as matched probes of this floor rather than as competing pipelines. Under the evaluated conditions, experiments on three datasets and a live deployment show these strategies move joint localization error in the predicted directions, for example lowering mean per-joint position error from about 97 to 65 mm with a second viewpoint, while a stronger estimator mainly lowers overall error by completing joints the measurement cannot resolve. These results give the robotics community a principled basis for deciding whether to prioritize observation design or estimator capacity when deploying reliable mmWave human sensing.

### 论文解读

#### 摘要翻译
What Limits mmWave Human-State Sensing for Robot Perception?由Shuntian Zheng、Guangming Wang、Hongyang He提出。三个数据集和现场部署显示，加入第二视角后平均关节位置误差约从97毫米降至65毫米。较强估计器主要补全不可见关节；这一结论应与观测本身提供的新信息区分。

#### 方法动机分析
毫米波可在黑暗、烟尘和隐私场景感知人体，但紧凑阵列孔径有限，关节定位误差可能危及安全。已有工作多单独测试融合、视角或估计器，难以判断改善来自真实新增信息还是模型补全。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
论文把人体与环境表为时变散射场，推导关节定位的 Cramér–Rao 型误差下界，显式纳入带宽、阵列孔径、SNR、帧数、相干性及可见回波。随后将多帧、多视角、回波筛选和姿态估计作为匹配探针，与该界比较；实验跨三个数据集并含现场部署。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
贡献是诊断框架而非新姿态网络：若增加视角降低误差，说明观测设计有效；若更强估计器主要补全不可见关节，则不能说测量分辨率提高。适用于传感器预算和部署方案选择。

#### 实验分析（精简版）
三个数据集和现场部署显示，加入第二视角后平均关节位置误差约从97毫米降至65毫米。较强估计器主要补全不可见关节；这一结论应与观测本身提供的新信息区分。 第二视角使平均定位误差降低约32毫米。

#### 实用指南
关注硬件的带宽、孔径与相干条件，以及回波保留策略。复现需统一人体动作、距离、视角和帧融合设置，分开报告可见关节定位与补全效果。自动驾驶可借鉴此法判断雷达多视角是否真正增加可观测性，但这属于方法迁移推断。  

#### 总结
核心思想：先诊断传感信息是否充足。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08863v1)
- [arXiv](https://arxiv.org/abs/2611.08863v1)

---

<a id='2611.08917v1'></a>
## [PRISM: Primitive Routing via In-context Skill Mixing for Retentive VLA Adaptation](https://arxiv.org/abs/2611.08917v1)

**Authors:** Jiachen Tao, Gengyu Zhang, Haoxuan Wang, Bing Liu, Yan Yan

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models promise a single language-conditioned policy adaptable to many manipulation behaviors, yet deployment additionally requires skill retention and long-horizon composition: a policy must retain previously acquired skills and reuse them inside composite tasks whose behavior changes across phases. The standard recipe absorbs every new task into one shared update, which provides neither a parameter-level locus for reusable skills nor a mechanism for invoking each skill along the trajectory. We observe that this missing structure is latent rather than absent: per-skill low-rank updates of a pretrained VLA occupy a robustly distinguishable, low-dimensional subspace, and a single composite-task LoRA concentrates roughly half of its Frobenius energy in this subspace, against a near-zero matched random-rotation control. Building on this geometry, we introduce PRISM, which represents reusable skills as parameter-level primitives: each skill family is stored as a dedicated low-rank action-expert adapter in a primitive bank, primitives are augmented with skill-specific low-rank residuals and composed at inference by a per-chunk VLM-conditioned router that reads the prefix representation to track task progress, and the router is supervised by per-chunk labels from a deterministic, task-conditioned, action-derived oracle. During composite training, the vision encoder, the language LoRA, the skill residuals, the router, and the shared action projections and time MLP are updated; every primitive is held frozen, so the base per-skill action-expert adapters are preserved by construction. On RoboCasa365, PRISM achieves 56.43% atomic-skill success and reaches 20.00% on long-horizon composite tasks, while a standard $π_{0.5}$-LoRA two-stage fine-tune reaches 17.86% atomic and 17.50% composite success. We additionally validate PRISM on an AgileX PIPER real-robot setup.

### 论文解读

#### 摘要翻译
PRISM: Primitive Routing via In-context Skill Mixing for Retentive VLA Adaptation由Jiachen Tao、Gengyu Zhang、Haoxuan Wang提出。RoboCasa365 上 PRISM 复合训练后原子技能成功率56.43%、复合任务20.00%；标准 π0.5-LoRA 两阶段微调为17.86%和17.50%，分别相差38.57与2.50个百分点。真机使用 AgileX PIPER 6-DoF机械臂，两个原子技能组成复合任务，并进行匹配起始状态试验。 

#### 方法动机分析
VLA 顺序微调新技能可能遗忘旧技能，而已学会的原子技能也常在长时程复合任务中被迫重新端到端学习。核心问题是如何保存可复用技能，并让策略在动作块级别随任务进度调用它们。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
PRISM 将每个技能族存成独立低秩 action-expert adapter primitive bank；复合训练时冻结 primitive，学习技能专属低秩 residual、共享模块与 VLM-prefix 条件 router。router 每个 action chunk 输出软混合权重，确定性、任务条件、由动作生成的 oracle 产生逐块监督标签。组合在激活空间完成，避免 LoRA 因子跨技能相乘产生交叉项。评测含 RoboCasa365 原子/复合任务及 AgileX PIPER 真机。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 冻结技能基元是保留旧技能的结构性约束，逐块路由则允许同一长任务中途切换。

#### 方法对比分析
标准共享 LoRA 直接吸收新行为，可能覆盖旧参数；PRISM 把技能基底冻结并通过 residual 和软路由组合。与硬 one-hot 分支相比，软激活空间组合更适合技能切换；性能依赖 router 输入、oracle 标签质量和技能库覆盖。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
RoboCasa365 上 PRISM 复合训练后原子技能成功率56.43%、复合任务20.00%；标准 π0.5-LoRA 两阶段微调为17.86%和17.50%，分别相差38.57与2.50个百分点。真机使用 AgileX PIPER 6-DoF机械臂，两个原子技能组成复合任务，并进行匹配起始状态试验。 真机验证表明组合流程能运行，但样本规模小于仿真评测。 相较标准 LoRA，复合任务提升2.50个百分点。


#### 实用指南
论文需关注每个技能数据、primitive 冻结策略、逐 action-chunk router 标签、成功率按任务宏平均的定义。真机展示支持可行性，但任务仅两个原子技能及一个组合链，不能等同开放世界泛化。训练阶段需逐技能建立 adapter bank，并在复合任务训练时冻结基元，另训练 residual 和 router。 训练配置需固定技能训练集与逐块标签；复现时核对基元冻结和 residual/router 更新范围。


#### 总结
核心思想：冻结技能基元并按进度软路由。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08917v1)
- [arXiv](https://arxiv.org/abs/2611.08917v1)

---

<a id='2611.08895v1'></a>
## [Localizing and Editing the Goal Representation of a Vision-Language-Action Model](https://arxiv.org/abs/2611.08895v1)

**Authors:** Sudhanshu Srivastava, Faisal Aqlan

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Operators can state safety constraints for robots as prohibitions such as "do not put the bowl on the stove". We study whether OpenVLA follows such prohibitions and where it represents the goal it executes. The 10 tasks of LIBEROGoal share one scene, so we can label the goal that each rollout executes. We combine closed-loop rollouts with activation patching, linear probes, activation steering and runtime monitors. On LIBERO-Goal, negation had no detectable effect on which task OpenVLA executed. The model executed the forbidden task in 79% of negated rollouts and completed 78% of plain tasks. A goal probe gives similar read-outs under "not X" and "X". Activation patching shows that the first action depends mostly on the instruction tokens before layer 12 and on the last prompt token from layer 16 onward. A fixed steering vector at the instruction tokens at layer 8 redirected 12 of 18 rollouts. Fixed vectors and prefill patches at layers 12 to 20 redirected none of 158. From layer 16, only patches at every decoded action token redirected the robot (15 of 18 rollouts at layer 24). When the instruction changes during a rollout, the layer-32 goal read-out detects non-compliance better than failure detectors, and an external probe on image and instruction features performs comparably. These results support enforcing prohibitions outside the instruction and monitoring goal read-outs against the specification. Goal edits in OpenVLA should act before layer 12 or patch every action token.

### 论文解读

#### 摘要翻译
Localizing and Editing the Goal Representation of a Vision-Language-Action Model由Sudhanshu Srivastava、Faisal Aqlan提出。在 LIBERO-Goal 上，否定语句未明显改变 OpenVLA 执行目标：79% 的否定 rollout 仍执行禁用任务，普通任务完成率为 78%。指令 token 第8层固定 steering 重定向18次中的12次；第12至20层固定向量未能重定向158次尝试，而第24层对每个解码动作 token patch 成功15/18。证据来自单一模拟场景和 OpenVLA。 

#### 方法动机分析
机器人安全约束往往以否定句表达，例如“不要把碗放到炉子上”。关键挑战是确认 VLA 实际执行的目标，而非仅判断动作是否失败；若否定词对策略没有作用，机器人可能熟练完成被禁止的行为。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
作者在 LIBERO-Goal 的同一场景十个任务上标注每个 rollout 真正执行的目标，结合闭环行为测试、线性 probe、activation patching、steering 和运行时监控。探针读出目标；patching 检查指令 token 与动作 token 在层间的作用；steering 尝试重定向目标；在线监控则比对目标读出和安全规格。实验使用 OpenVLA，分析层位置及 prefill 与逐动作 token 干预。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 这组分析以逐步动作目标判定为核心，说明离线定位与在线监控服务于不同安全环节。

#### 方法对比分析
它与常见失败检测器不同，直接监视“机器人在做哪个目标”，因此能发现成功执行但违反禁令的行为。否定任务上的内部编辑也有严格位置依赖，不能把激活修改视为通用安全补丁。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
在 LIBERO-Goal 上，否定语句未明显改变 OpenVLA 执行目标：79% 的否定 rollout 仍执行禁用任务，普通任务完成率为 78%。指令 token 第8层固定 steering 重定向18次中的12次；第12至20层固定向量未能重定向158次尝试，而第24层对每个解码动作 token patch 成功15/18。证据来自单一模拟场景和 OpenVLA。 结果支持外部规格监控，而非单靠自然语言禁令。

#### 实用指南
论文发表于 CoRL 2026 workshop，适合做 VLA 安全审计与解释性复现。复现时需逐步检查目标 predicate、指令 token 位置、动作 token patch 时机，并对不同模型、任务、禁令句式重复验证。外部运行时约束比依赖模型遵守否定词更可靠；这是论文结果支持的设计方向。推理时的逐动作检查应先在仿真中校准，再评估对新 VLA 的适用性。

#### 总结
核心思想：监视执行目标而非只看失败。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08895v1)
- [arXiv](https://arxiv.org/abs/2611.08895v1)

---

<a id='2611.08898v1'></a>
## [FATHOM: World-Model-Guided Diffusion Action-Trajectory Selection for Sparse-Marker AUV Tracking](https://arxiv.org/abs/2611.08898v1)

**Authors:** Xiaorui Wang, Zeyu Sha, Mingyang Yang, David Chen Wang, Feitian Zhang

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Reliable relative tracking is fundamental to cooperative autonomous underwater vehicles (AUVs), yet remains challenging under nonlinear vehicle dynamics and degraded underwater visual conditions. Active light markers improve robustness to appearance variation, but provide deliberately sparse observations that can become incomplete or corrupted by occlusion, reflections, and detection noise during dynamic motion. To address the resulting partial-observation control problem, this paper proposes FATHOM, a world-model-guided diffusion action-trajectory selection framework for sparse-marker AUV tracking. FATHOM first integrates structured marker observations, inertial measurements, action history, and temporal dynamics into a predictive tracking belief using a multimodal task-supervised recurrent state-space model. A diffusion action prior then generates multiple expert-like multi-thruster trajectories, whose future tracking consequences are predicted through the learned latent dynamics. Rather than directly executing a generated trajectory, FATHOM selects the most favorable candidate under an explicit receding-horizon objective and continually updates the decision from onboard observations. Extensive offline analyses verify the learned predictive belief, multi-step world-model prediction, diffusion trajectory generation, and predictive selection. More importantly, comprehensive real-AUV experiments, including strong pose-based comparisons and component ablations, demonstrate robust autonomous tracking across nominal and challenging motions with naturally occurring partial marker visibility, while maintaining real-time onboard operation without motion-capture feedback.

### 论文解读

#### 摘要翻译
FATHOM: World-Model-Guided Diffusion Action-Trajectory Selection for Sparse-Marker AUV Tracking由Xiaorui Wang、Zeyu Sha、Mingyang Yang提出。真实 AUV 对比中，加入世界模型预测选择后，名义运动跟踪 NRMSE 从0.097降至0.084（13.4%），挑战运动从0.190降至0.124；论文报告标称10 Hz运行，K=16候选、10步预测时选择延迟73.83 ms。自然部分可见灯标时（至多3个），NRMSE为0.163。 

#### 方法动机分析
水下相对跟踪面对非线性动力学、反光与遮挡，主动灯标虽稳健却故意稀疏，单帧往往不足以估计运动状态。直接回归一个动作序列也难表达同一状态下多种可行的多推进器控制。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
FATHOM 由多模态任务监督循环状态空间模型融合灯标观测、惯性测量、动作历史与时序动态，形成预测跟踪信念；扩散动作先验采样多条专家式多推进器轨迹；潜在世界模型预测每条轨迹的未来跟踪状态；滚动时域代价选择候选，只执行首个控制命令，下一次观测后再更新信念并重选。实验包含离线预测评估及无动作捕捉反馈的真实 AUV 部署。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
相较显式位姿估计加反馈控制，它保留与任务相关的历史和动作条件预测；相较单一行为克隆序列，它用扩散先验保留多模态候选，再由预测代价筛选。世界模型误差仍会造成候选排序错误，预测 horizon 与采样数存在计算权衡。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
真实 AUV 对比中，加入世界模型预测选择后，名义运动跟踪 NRMSE 从0.097降至0.084（13.4%），挑战运动从0.190降至0.124；论文报告标称10 Hz运行，K=16候选、10步预测时选择延迟73.83 ms。自然部分可见灯标时（至多3个），NRMSE为0.163。 这提供了可核对的实时控制条件。 真实水池中的结果还报告了部分标记缺失条件。 真实水池实验提供了闭环控制证据。


#### 实用指南
复现应关注水池与真实 AUV 的标定、灯标可见数、IMU同步、控制频率以及世界模型误差随 horizon 累积。论文提供了真实水下验证，是工程亮点；迁移到地面机器人可借鉴“候选动作加模型预测筛选”，但动力学和传感器需重新训练。控制推理以名义10 Hz运行，采样K=16条候选且预测10步；复现应保留相同预算。

#### 总结
核心思想：用世界模型筛选扩散动作轨迹。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08898v1)
- [arXiv](https://arxiv.org/abs/2611.08898v1)

---

<a id='2611.08920v1'></a>
## [HOC2R: Hand-Object Contact Trajectories for Human-to-Robot Motion Transfer](https://arxiv.org/abs/2611.08920v1)

**Authors:** Hongkun Yu, Francisco M. López, Francisco Cruz

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Human videos contain abundant object-manipulation demonstrations, but they do not directly provide motion cues that a robot can follow. We present HOC2R, a Hand-Object Contact trajectory extractor for Robot motion that combines frozen DINOv3 and SAM 2.1 features with learned heads for hand-object association and contact localization. HOC2R supports two modes: count-aware automation that detects, counts, and localizes visible contacts in untrimmed videos, and fixed-two localization for two-finger grippers when contact frames are known. On a 30-task point-annotated benchmark, count-aware automation improves contact-count accuracy from 58.1% to 71.4% and reduces mean localization distance from 19.81 to 10.46 pixels relative to an adapted VRB2 baseline. On seven unseen RGB-D rollouts, a five-task fixed-two checkpoint reduces mean localization distance from 67.67 to 27.02 pixels. Across six robot evaluation sequences, the 30-task fixed-two checkpoint produces four feasible trajectories and succeeds on three tasks, compared with two feasible trajectories and one successful task for each adapted baseline. HOC2R is competitive in the evaluated single-hand scenes, robust to the hand-color shift, and strongest in the evaluated multi-hand scenes. These results support HOC2R as a practical pseudo-labeling method for extracting robot-usable contact trajectories from human demonstrations.

### 论文解读

#### 摘要翻译
HOC2R: Hand-Object Contact Trajectories for Human-to-Robot Motion Transfer由Hongkun Yu、Francisco M. López、Francisco Cruz提出。30任务点标注基准中，count-aware 接触计数准确率由适配 VRB2 的58.1%升至71.4%，平均定位误差由19.81降至10.46像素。7个未见 RGB-D rollout 中固定双点误差由67.67降至27.02像素；6段机器人评测序列中，HOC2R 产生4条可行轨迹并完成3项任务，对照各为2条可行、1项成功。 

#### 方法动机分析
人类视频有丰富操作示范，却缺少机器人可直接跟随的接触点轨迹。手指小、遮挡和多手多物体会造成定位及关联错误；仅有框和分割掩码还不能决定夹爪应在哪里接触。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
HOC2R 冻结 DINOv3 与 SAM 2.1 的视觉特征，训练 proposal head 提出手物交互框，heatmap head 从整图特征估计接触概率，再用 proposal-box touch score 对热图候选重排。count-aware 模式自动判断接触帧及可见接触数；fixed-two 模式假设活动接触片段已知，定位二指夹爪的两个点。点经深度和彩色相机标定提升到机器人坐标，再经几何安全检查生成轨迹。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 固定双点定位与自动计数并非同一评测假设：前者输入活动接触片段，后者还要判断何时存在接触。

#### 方法对比分析
区别于只检测手和物体的方案，HOC2R 直接追踪可供机器人使用的接触几何。自动计数适合未剪辑视频；固定双点模式在域偏移时更稳健，但依赖接触帧已知。它输出伪标签/候选轨迹，不是完整示范到动作策略。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
30任务点标注基准中，count-aware 接触计数准确率由适配 VRB2 的58.1%升至71.4%，平均定位误差由19.81降至10.46像素。7个未见 RGB-D rollout 中固定双点误差由67.67降至27.02像素；6段机器人评测序列中，HOC2R 产生4条可行轨迹并完成3项任务，对照各为2条可行、1项成功。 这说明接触定位的提升能够传递到一部分机器人轨迹，但仍有轨迹被几何安全检查拒绝。

#### 实用指南
复现要核对接触点标注、相机内外参、深度反投影和安全几何拒绝标准。多手场景优势明显，但单手域外表现、接触帧切分和从轨迹到控制的误差仍限制结论。适合作为人类视频机器人学习的接触伪标签前端。推理链路依赖 RGB-D 深度与相机标定，建议先离线验证像素误差再做坐标变换。

#### 总结
核心思想：从人类视频提取机器人接触轨迹。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08920v1)
- [arXiv](https://arxiv.org/abs/2611.08920v1)

---

<a id='2611.08827v1'></a>
## [ActPrune: Action-Grounded Visual Token Pruning for Efficient Vision-Language-Action Models](https://arxiv.org/abs/2611.08827v1)

**Authors:** Sungwon Moon, Minji Kim, Hyemin Boo, Daeun Song, Jiyoung Lee

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Vision-language-action (VLA) models perform versatile manipulation tasks from language instructions, but their inference cost limits how fast a robot can respond in closed-loop control. Visual token pruning reduces this cost; however, prior methods estimate token redundancy from encoded visual embeddings or attention, only after the visual encoder has processed the entire observation. We propose \ours, a training-free token pruning method that moves the pruning decision ahead of the vision encoder by grounding it in a geometric prior derived from the robot's own action. The action chunk predicted at the previous control step is already available before the current observation is encoded, and it specifies the end-effector and camera motion between consecutive observations. The same action affects the two standard camera views asymmetrically: in the fixed third-person view, \ours retains patches along the projected end-effector trajectory together with semantic anchors, while in the wrist view, it prunes patches that remain unchanged after compensating for the action-induced camera motion. Extensive experiments on two simulation environments (LIBERO and VLABench) and four real-robot tasks demonstrate that \ours achieves up to 1.61 $\times$ speedup and 59.5\% FLOPs reduction with negligible degradation in success rates.

### 论文解读

#### 摘要翻译
ActPrune: Action-Grounded Visual Token Pruning for Efficient Vision-Language-Action Models由Sungwon Moon、Minji Kim、Hyemin Boo提出。LIBERO、VLABench 两个模拟器和四项真实机器人任务评测中，最高加速为1.61倍，FLOPs最多减少59.5%，成功率退化很小。读者应把它理解为任务中的最佳报告值，并逐项检查速度和成功率，而非假定全部工作负载都达到同一收益。

#### 方法动机分析
标准 token pruning 多在视觉编码后依据注意力或嵌入筛选，无法省掉视觉编码器本身。作者假设，动作块已给出末端执行器和腕部相机的运动线索，足以预测下一帧哪些区域会变、哪些视角值得保留。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
输入当前双视角图像、语言指令和上一动作块；在固定相机投影末端轨迹并保留语义锚点，在腕部视角补偿相机运动后识别静态冗余 patch；只把保留 token 送进视觉编码器，再由原 VLA 预测新动作。规则无需额外训练，依赖可用的动作与相机几何。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
与编码后 pruning 相比，它把决策前移到视觉主干之前，并使用控制信号而非模型注意力。适合具备相机标定、动作历史和双视角输入的操控系统；单目或动作不可预测场景需重新设计几何先验。

#### 实验分析（精简版）
LIBERO、VLABench 两个模拟器和四项真实机器人任务评测中，最高加速为1.61倍，FLOPs最多减少59.5%，成功率退化很小。读者应把它理解为任务中的最佳报告值，并逐项检查速度和成功率，而非假定全部工作负载都达到同一收益。

#### 实用指南
实现重点是动作到图像位移的投影、腕部相机运动补偿、语义锚点以及与视觉编码器接口的 token 顺序。代码与配置可复现时应核对逐任务成功率、相机同步和 FLOPs/延迟测量硬件。可迁移到移动操作中基于规划动作做视觉预算分配。  

#### 总结
核心思想：动作先验的视觉 token 预剪枝。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08827v1)
- [arXiv](https://arxiv.org/abs/2611.08827v1)

---

<a id='2611.08946v1'></a>
## [Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents](https://arxiv.org/abs/2611.08946v1)

**Authors:** Jingdong Zhang, Jionghao Wang, Zhengming Yu, Yancheng Cai, Wei Chow, Chris Choy, Zhiyang Dou, Hang Hua, Jiachen Li, Zhixuan Liu, Chen Wang, Songlin Wei, Siyuan Yang, Yu Yuan, Xiaohang Zhan, Yucheng Zhang, Shuhong Zheng, Xin Li, Lingjie Liu, Wojciech Matusik, Zhengzhong Tu, Wenping Wang

**Published:** 2026-11-06

**Categories:** cs.AI, cs.CV, cs.RO

**Abstract:**

Frontier language models are increasingly deployed as agents, and how well an agent performs depends not only on its model but also on its harness: the prompts, tools, memory, and control loop through which the model acts. When an agent is applied to a new problem, a natural question is whether it can use the feedback of the task to improve the way it works, and not merely its answer. Most existing benchmarks keep the agent's harness fixed and score the solution it produces, or target large-scale model training. We introduce Flex-RSI, a benchmark for the recursive self-improvement of agents on vision and robotics tasks under a fixed budget. In each round, an executor improves the task solution, which is kept only if it scores higher on held-out validation data, and an improver then revises the harness, including the executor and itself. Every run starts from the same general-purpose harness, and the final solution is scored on hidden test data. Evaluating closed-source and open-weight models on tasks spanning perception, reasoning, robot action, 3D reconstruction, visual creation, and infrastructure, we find that self-improvement helps most when the agent builds its own solution, that feedback recording only success or failure holds back open-weight models, and that open-weight models fall furthest behind where outputs must pass strict checks of physical stability or exact correctness. Paired runs further show that editing the harness itself contributes to the gains.

### 论文解读

#### 摘要翻译
Flex-RSI: Benchmarking Recursive Self-Improvement of Frontier Agents由Jingdong Zhang、Jionghao Wang、Zhengming Yu提出。基准覆盖20项任务。论文发现逐例连续分数反馈时，开源模型构造的任务解法更能改善；仅记录成功或失败时改善有限。物理稳定性和精确正确任务仍是难点。

#### 方法动机分析
单次提示或模型分数无法说明智能体能否从失败中持续改进；任务、反馈粒度和验证选择会改变结论。核心假设是固定预算和评估规则下，可比较模型对可执行解法及其改进器的优化能力。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
每轮包含执行器改任务解法、隔离改动、验证集评分、接受/拒绝、改进器更新工具流程。多轮后在隐藏测试集评估。20项任务覆盖机器人、推理、生成、重建、感知和基础设施，包含两组执行器设置及冻结改进器对照。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
不同于只测最终答案的基准，它评估迭代过程与工具改进；不同于开放式自我修改，限制可改路径、预算、数据与验证规则。分数受任务类型、反馈粒度、验证集选择与代理评估器影响。 相较固定 harness 的一次性评测，本文明确比较多轮解法优化与流程优化。相较传统固定评测，创新在于同时允许执行器改任务解法、改进器改 harness，并以相同预算观察迭代。

#### 实验分析（精简版）
基准覆盖20项任务。论文发现逐例连续分数反馈时，开源模型构造的任务解法更能改善；仅记录成功或失败时改善有限。物理稳定性和精确正确任务仍是难点。 相较固定 harness 的一次性评测，20项任务在多轮预算下比较执行器和改进器。

#### 实用指南
复现必须遵守数据隔离、预算、提交与测试协议，特别检查验证集自适应过拟合。应用于机器人需加入安全恢复与仿真真机差异评估。复现需按任务预算执行验证和停止规则，避免把验证集提升误当成隐藏测试泛化。

#### 总结
核心思想：用协议衡量递归改进能力。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08946v1)
- [arXiv](https://arxiv.org/abs/2611.08946v1)

---

<a id='2611.08879v1'></a>
## [GeoJeV: Unified Spatial Probability for Training-Free Efficient Evidence Acquisition in Ultra-High-Resolution Remote Sensing](https://arxiv.org/abs/2611.08879v1)

**Authors:** Xinyi Niu, Lingfeng Chen, Jiehua Zhang, Panhe Hu, Li Liu, Zhen Liu, Yongxiang Liu

**Published:** 2026-11-06

**Categories:** cs.CV, eess.SP

**Abstract:**

Ultra-high-resolution (UHR) remote sensing imagery provides rich spatial details but poses challenges for accurate and efficient multimodal understanding. Existing evidence acquisition methods selectively explore relevant image regions, yet largely rely on generative multimodal large language models (MLLMs) for intermediate search control. However, reinforcement learning from human feedback (RLHF) primarily aligns these models with human-preferred responses rather than explicitly optimizing region-level relevance judgments, while repeated autoregressive decoding introduces additional computational overhead. To address these limitations, we propose GeoJeV, a training-free method that formulates evidence acquisition as spatial probability-driven decision making. At each evaluated region, a frozen JeV model produces a probability distribution over candidate regions. Specifically, Region Rank prioritizes candidate regions, Sparse Branch adaptively determines branching width through sparse probability support, and Depth Gate controls refinement depth based on branch sparsity and distribution concentration. By coordinating search priority, branching width, and refinement depth through a shared distribution, GeoJeV reduces reliance on RLHF-aligned generation and eliminates intermediate decoding. Experiments on LRS-GRO, MME-RealWorld-RS, and LRS-VQA achieve accuracies of 60.59%, 52.79%, and 34.32%, respectively. Across the three benchmarks, GeoJeV improves the average accuracy of Qwen3-VL-8B by +5.32pp with only 1.12 additional seconds per question, while achieving a 7.8-fold inference speedup over the recent method.

### 论文解读

#### 摘要翻译
GeoJeV: Unified Spatial Probability for Training-Free Efficient Evidence Acquisition in Ultra-High-Resolution Remote Sensing由Xinyi Niu、Lingfeng Chen、Jiehua Zhang提出。Qwen3-VL-8B 在三基准平均准确率由43.91%升到49.23%，延迟从10.47秒增至11.59秒；相对 ZoomSearch 平均90.06秒约快7.8倍。计数和状态类别仍可能下降。 

#### 方法动机分析
整图缩放会损失小目标细节，逐块调用生成式多模态模型又耗时。作者假设冻结决策模型对候选区域的概率可直接驱动搜索，无需为中间步骤生成文字。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
把图像递归划成网格，JeV 为每个候选块输出概率。Region Rank 决定优先级；Sparse Branch 对概率取稀疏支持来定保留分支；Depth Gate 比较分布与均匀/集中锚点决定是否继续细分。将选定原图裁块按空间关系拼成证据画布，交冻结回答模型作答。默认配置使用3×3网格、温度2、最多6块、最多10次 JeV 前向。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
统一概率控制三个搜索决策，相对生成式搜索避免中间自回归解码；相对固定 Top-k 可自适应宽度。对局部目标有效，但分散对象/全局上下文问题存在收益回退。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
Qwen3-VL-8B 在三基准平均准确率由43.91%升到49.23%，延迟从10.47秒增至11.59秒；相对 ZoomSearch 平均90.06秒约快7.8倍。计数和状态类别仍可能下降。 相比直接回答平均增加1.12秒，换取5.32个百分点准确率提升。 跨 LRS-GRO、MME-RealWorld-RS 与 LRS-VQA 的平均值按基准等权计算。 三项基准按等权计算平均准确率。


#### 实用指南
实现依赖 JeV 与回答模型冻结，复现应核对裁块最小尺寸、动态分辨率、画布拼接与端到端计时。可迁移到地图、病理或高分辨率工业质检，但区域概率的校准需重新评估。推理配置为3×3网格、最多6个证据块、最多10次 JeV 前向，最小裁块边长224像素。

#### 总结
核心思想：核心机制与任务反馈结合。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08879v1)
- [arXiv](https://arxiv.org/abs/2611.08879v1)

---

<a id='2611.08904v1'></a>
## [MotionCore: A Latent Neural Simulator for Video Generation with Rigid-Body Dynamics](https://arxiv.org/abs/2611.08904v1)

**Authors:** Zheng Wang, Zhengwei Gong, Yuanzhi Liang, Chi Zhang

**Published:** 2026-11-06

**Categories:** cs.CV

**Abstract:**

Video generation models can produce highly realistic videos while generating physically inconsistent object motion. This limitation suggests that visual modeling alone may not provide a transferable representation of physical dynamics. We introduce MotionCore, a framework for learning physical dynamics via a learnable partial differential equation solver and focuses on rigid-body motion. We introduce a latent neural simulator learning to predict the evolution of object-level physical states from observed motion. Its predictions are then converted into spatially aligned latent controls for a controllable video diffusion model. The generator serves as a visual interface: it provides appearance and scene detail, while the learned dynamics model specifies the physical trajectory. To preserve this trajectory through the interface, MotionCore combines a diffusion-matching objective with a low-capacity decodability constraint on object identity and position. We evaluate MotionCore on public physical-consistency benchmarks and on paired data with known trajectories. Results indicate that MotionCore improves physical consistency and trajectory obedience in the evaluated rigid-body settings, while maintaining comparable visual quality. A controlled ablation protocol compares learned dynamics with an analytic prior and studies the choice of latent evolution processor.

### 论文解读

#### 摘要翻译
MotionCore: A Latent Neural Simulator for Video Generation with Rigid-Body Dynamics由Zheng Wang、Zhengwei Gong、Yuanzhi Liang提出。严格 Physics-IQ 子集上，27个场景、三个视角的 R2 球模型得分28.14±0.41，ST-IoU为0.3734±0.0117；四轮扩散种子而非独立训练。共享潜在槽小规模试验使用单种子、50次更新，Full 的 ADE/Req=0.5181，解析先验为0.5204，差距很小，不能证明收敛或统计显著。 

#### 方法动机分析
视频生成模型能够合成逼真画面，却可能违反刚体碰撞、滚动和支撑规律。论文假设把物体级物理状态和视觉合成分开建模，可以提供可控的运动轨迹，而不要求视频扩散主干自己学习完整动力学。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
MotionCore 先从观测历史估计物体状态，再用潜在神经模拟器预测刚体状态演化与接触；预测结果经空间对齐的潜在控制适配器送入 Wan 2.1-VACE 视频扩散模型。扩散匹配目标与低容量可解码约束共同保持物体身份和位置可恢复。分阶段训练模拟器和视频适配器，生成主干保持冻结。评测使用 Physics-IQ-Verified 刚体子集81段五秒视频及 RigidBench 100段两秒视频。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。

#### 方法对比分析
相对只按文本和首帧生成的视频基线，MotionCore额外使用允许的观测历史估计初始物理状态，并显式预测轨迹；视觉生成器仍负责外观。它不是端到端物理仿真的证明，控制缓存中的接触和支撑失败也会限制物理解释。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
严格 Physics-IQ 子集上，27个场景、三个视角的 R2 球模型得分28.14±0.41，ST-IoU为0.3734±0.0117；四轮扩散种子而非独立训练。共享潜在槽小规模试验使用单种子、50次更新，Full 的 ADE/Req=0.5181，解析先验为0.5204，差距很小，不能证明收敛或统计显著。  严格子集的 ST-IoU 为0.3734±0.0117，四轮是扩散种子而非独立训练。 主结果的四轮是扩散种子轮次，不是独立训练重复。 基准平均得分达到28.14分，低于 MiniMax-H3 的39.36分；物理指标与通用视频质量不能混为一谈。

#### 实用指南
实现使用 Wan 2.1-VACE-1.3B，832×480、30步采样、16 fps，实验使用8张 H100。复现要区分主结果、RigidBench诊断和小规模处理器试验，不把选出的可视化样例当成对照。迁移到机器人仿真需验证长时稳定性、状态估计误差和真实接触。模型在8张H100上训练，生成使用832×480分辨率、30步和16 fps；这些设置决定复现成本。 推理视频采用832×480、30步采样；训练和生成资源需求较高。 推理视频采样使用832×480、30步、16 fps；实验训练使用8张GPU。


#### 总结
核心思想：先预测物理状态再生成视频。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08904v1)
- [arXiv](https://arxiv.org/abs/2611.08904v1)

---

<a id='2611.08868v1'></a>
## [Lodos-v2: An Efficient Primal Formulation of the Exact Frictional Contact Problem](https://arxiv.org/abs/2611.08868v1)

**Authors:** Onur Beker, Andreas René Geist, Georg Martius

**Published:** 2026-11-06

**Categories:** cs.RO

**Abstract:**

Lodos-v2 is a GPU-based simulation framework for massively parallel rigid-body dynamics with contact. It is specifically designed to achieve high throughput in scenes involving collisions among large numbers of detailed meshes. Its predecessor, Lodos-v1, had two main limitations. First, it solved the standard convex relaxation of cone-complementarity-based contact dynamics, thereby inheriting well-known physical artifacts, including lift-off and artificial compliance in the normal direction, as well as creep in the tangential direction. Second, its contact-group reparameterization scheme, which rescales spring-damper forces using a single factor per collision pair, could not be integrated with other constraints required by a full-fledged simulator, such as joint limits and welds. Lodos-v2 addresses both limitations through two main contributions. First, it introduces an efficient algorithm for solving the exact frictional contact problem by sequentially solving a cascade of convex relaxations in primal form within an outer proximal minimization loop. Second, it extends the contact-group reparameterization scheme of Lodos-v1 to the primal formulation of cone-complementarity-based contact dynamics, enabling its integration with the other constraints of a full-fledged simulator. Experiments demonstrate that these contributions result in throughput improvements over established rigid-body simulation frameworks. The source code is available at https://github.com/martius-lab/lodos.

### 论文解读

#### 摘要翻译
Lodos-v2: An Efficient Primal Formulation of the Exact Frictional Contact Problem由Onur Beker、Andreas René Geist、Georg Martius提出。RTX 4080 上 MuJoCo 模式对 MJWarp 的 M8 螺母场景报告37.35倍吞吐；精确模式相对 Kamino 的最高205.5倍来自特定对照，且部分用原始网格与凸分解的非同等表示。 

#### 方法动机分析
传统凸松弛可能产生抬离、法向柔顺和切向蠕变；旧接触组重参数化又不易与关节限位、焊接等非接触约束共存。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。 因此，阅读结果时要把论文真正解决的变量与仍由数据、模型或平台决定的部分分开；动机成立，并不自动意味着方法适用于所有任务。

#### 方法设计详解
Lodos-v2 在外层近端最小化中迭代凸子问题，用 de Saxcé 修正与 Newton 更新求精确摩擦接触；在原始加速度形式中利用接触组重参数化降低计算规模，并使用 Woodbury 行空间因子化处理 Hessian。另提供 MuJoCo 模式。GPU 批量运行，比较 MJWarp 与 Kamino。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 从 input 到 output 的关键是维持各模块之间明确的信息接口，并核对每一步是否使用了论文所允许的输入。理解消融时应区分“组件带来的变化”和“整体系统同时变化”的结果。 精确模式和 MuJoCo 兼容模式使用不同的物理问题与参考参数，报告速度时要分开比较。

#### 方法对比分析
精确接触强调物理一致性，MuJoCo 模式强调兼容和吞吐；逐接触、锥和射线参数化在精度与变量数之间权衡。对接触密集场景有效，不代表一般机器人任务整体训练速度已提高。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。 这项差异也说明了方法的适用边界：当输入条件、对象分布或评估目标发生变化时，应重新做对照，而不宜只迁移论文中的结论。

#### 实验分析（精简版）
RTX 4080 上 MuJoCo 模式对 MJWarp 的 M8 螺母场景报告37.35倍吞吐；精确模式相对 Kamino 的最高205.5倍来自特定对照，且部分用原始网格与凸分解的非同等表示。 基准在单张 RTX 4080 上进行批量吞吐测量，不能直接等价为端到端机器人训练提速。 论文把每个场景的最高批量吞吐作为表中比较值。 最高批量吞吐基于每场景独立随机初态测得。


#### 实用指南
代码公开；复现需使用相同碰撞表示、随机初态、批大小和迭代容差，并区分精确与 MuJoCo 两种模式。机器人仿真可直接关注精确摩擦与接触丰富任务，需测量端到端学习收益。  

#### 总结
核心思想：核心机制与任务反馈结合。

1. 建立任务输入与中间表示。
2. 执行论文提出的核心筛选或规划。
3. 输出结果并按匹配指标验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.08868v1)
- [arXiv](https://arxiv.org/abs/2611.08868v1)

---

