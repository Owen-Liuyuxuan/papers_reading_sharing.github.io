time: 20261112

# Arxiv Computer Vision Papers - 2026-11-12

## Executive Summary

## 本期导读

本期从 138 篇 CS.CV/CS.RO 新投稿中选出 10 篇完整阅读。研究呈现出三条相互关联的主线：一是将风险与不确定性纳入规划决策，二是让视觉和动作模型显式表达物体交互及未来状态，三是把多模态感知或生成能力做成可复用的统一表征。

自动驾驶方向里，**Route-Time Conflict Maps**把伙伴车辆的意图映射到沿路线展开的时空冲突图，并以共同优先级处理让行；**RiSE-Drive**则从候选轨迹的长时结果分布中考虑低尾风险。两者分别处理多车协同和单车候选排序，实验主要来自仿真基准，部署安全仍需实车验证。更一般的规划方法 **Learn-To-Propagate** 用 Wasserstein 不确定集逐阶段传播非线性系统状态，在五个基准上报告全部成功；其形式化条件、采样要求和耗时边界值得与工程需求一起阅读。

具身学习方面，**LaST0.5**通过交互潜变量和进度预测衔接世界模型与动作生成，在双臂实物任务上优于所比较基线，但真实任务范围有限。**WAM is capable in BEHAVIOR-1K**提供大规模挑战赛系统经验；**LEASH**把共享安全策略放进任务、动作和提交三道审查门，并用仿真及小规模实机基准评测，突出安全与人机协作之间仍未解决的权衡。

感知与表征方向覆盖不同尺度：**GEMOR-GS**结合外参精化、全局高斯地图和在线重定位；**HiPFuse**以高度分区的点云—BEV双路融合争取边缘设备吞吐；**UGG**统一 RGB、稀疏/稠密深度与相机位姿条件；**UniFlow-RS**把六类遥感任务统一为潜空间生成问题。它们共同显示，统一模型的价值需要与特定任务精度、边界细节和数据协议并列考察。

建议优先通读 **LEASH**（安全评测与闭环保护）、**LaST0.5**（交互表征与动作生成）、**RiSE-Drive**（风险敏感的候选评价）和 **GEMOR-GS**（在线/离线地图系统设计）。若关注形式化安全规划，可再读 **Learn-To-Propagate**；关注端侧部署则先看 **HiPFuse**。阅读时应核对基准规模、实机试验次数和具体评测设置，避免将单一指标直接外推为实际部署表现。

---

## Table of Contents

1. [Route-Time Conflict Maps for Cooperative End-to-End Autonomous Driving](#2611.13494v1)
2. [GEMOR-GS: Global Extrinsic-calibration-refined Mapping and Online Relocalization for 3D Gaussian-Splatting SLAM](#2611.13528v1)
3. [RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving](#2611.13790v1)
4. [WAM is capable in BEHAVIOR-1K: the 2nd-place solution to the 2026 BEHAVIOR Challenge](#2611.13818v1)
5. [HiPFuse: Real-Time LiDAR Semantic Segmentation for Robotic Perception on Edge Devices](#2611.14025v1)
6. [LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning](#2611.14048v1)
7. [UGG: Unified Generative Geometry Model](#2611.14440v1)
8. [UniFlow-RS: Unified Generative Prediction for Heterogeneous Remote-Sensing Tasks](#2611.14281v1)
9. [Learn-To-Propagate: Distributionally Robust Motion Planning for Nonlinear Uncertain Systems](#2611.13863v1)
10. [LEASH: Stay Safe in Agentic Robotics](#2611.14115v1)

---

## Papers

<a id='2611.13494v1'></a>
## [Route-Time Conflict Maps for Cooperative End-to-End Autonomous Driving](https://arxiv.org/abs/2611.13494v1)

**Authors:** Haohua Que, Mingkai Liu, Tianle Zhu, Zhipeng Bao, Yiping Li, Handong Yao

**Published:** 2026-11-10

**Categories:** cs.RO

**Abstract:**

Existing cooperative end-to-end driving works at Class A (status-sharing) of the SAE J3216 classes of cooperative driving automation (CDA): vehicles exchange what they perceive, not what they intend, so each plans alone when to go, which limits cooperation and leads to deadlock when both vehicles yield or to collisions when both go. We present When2Go, a cooperative end-to-end driving system that learns when to go by sharing intent under CDA Class B or higher. Each vehicle broadcasts its LiDAR detections with forecast motion as instance tokens, together with its plan and intended trajectory. The receiver draws the fused detections, forecasts and partner plans into a route-time conflict map, the occupancy along its route over the planning horizon, on which a planner scores candidate speed profiles. Where two intended trajectories conflict, an order gate, a precedence network distilled from a right-of-way rule, decides who goes first; both vehicles evaluate it on the same messages and so reach the same order. When2Go imitates the rule-based expert of V2Xverse, which sees the simulator's ground truth; stops that When2Go cannot observe within its 2 s planning horizon are relabelled, so that it does not learn to wait without an observable cause. For our runs only, the benchmarks' PID controller steers by When2Go's waypoints instead of the route, and its forced push of long-waiting vehicles is off. With one model, When2Go sets the state of the art on three closed-loop CARLA benchmarks: 97.8 and 90.7 driving score on InterDrive without and with traffic (best published 90.1 and 78.9), 70.0 on the public V2Xverse protocol (best of sixteen rerun methods 59.1) and 88.2 on MDrive (best rerun 85.1), with 95.0% of its vehicles collision-free (87.7%).

### 论文解读

#### 摘要翻译
Route-Time Conflict Maps for Cooperative End-to-End Autonomous Driving提出：When2Go把多车感知、运动预测、车辆自身计划与伙伴意图编码成沿本车路线展开的时空冲突图，再对64条候选速度曲线评分。冲突轨迹交叉时，学习到的优先级门控决定谁先行；双方用相同消息得出同一让行次序。模型模仿可见全知状态的规则专家，并把规划视野以外的等待原因标成未知，避免学会无因停车。 论文报告的范围是在 CARLA 的 InterDrive、V2Xverse、MDrive 闭环评测中，Driving Score 分别为 97.8/90.7、70.0、88.2；对应基线最高值为 90.1/78.9、59.1、85.1。车辆无碰撞比例 95.0%，对照为 87.7%。方法使用单一18M参数模型，GPU侧每车处理约0.1秒。实验关闭 benchmark 的强制推动车辆机制，且报告的 PID 控制器按预测航点转向。

#### 方法动机分析
已有协同驾驶多处于只共享状态的 SAE J3216 Class A：各车仍独立决定何时通行，容易同时停等或同时前进。本文假设通过共享未来计划及一致的路权判定，可将协同从信息融合推进到意图协调；同时，模型只能依据其可观察的两秒规划视野行动。

#### 方法设计详解
When2Go把多车感知、运动预测、车辆自身计划与伙伴意图编码成沿本车路线展开的时空冲突图，再对64条候选速度曲线评分。冲突轨迹交叉时，学习到的优先级门控决定谁先行；双方用相同消息得出同一让行次序。模型模仿可见全知状态的规则专家，并把规划视野以外的等待原因标成未知，避免学会无因停车。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。论文给出 CARLA 多基准实现及复现实验配置，训练目标来自规则专家，推理同时融合对象 token、运动预测、路线几何与伙伴轨迹。迁移前应重建通信时延/丢包、规划视野和优先权约束，并将感知误差与伙伴策略变化纳入测试。

#### 方法对比分析
区别于只融合伙伴感知的端到端驾驶，本文把伙伴计划显式变成路线时间占用，再与候选自车速度曲线共同评估。其新意在冲突表示与可学习的共同优先级，而不只是扩大感知输入。适合多车交互、V2X和闭环仿真；当前结果依赖可通信且一致的伙伴计划。

#### 实验分析（精简版）
在 CARLA 的 InterDrive、V2Xverse、MDrive 闭环评测中，Driving Score 分别为 97.8/90.7、70.0、88.2；对应基线最高值为 90.1/78.9、59.1、85.1。车辆无碰撞比例 95.0%，对照为 87.7%。方法使用单一18M参数模型，GPU侧每车处理约0.1秒。实验关闭 benchmark 的强制推动车辆机制，且报告的 PID 控制器按预测航点转向。 这是模拟闭环证据，没有真实道路或实车验证。实验明确依赖理想通信设置，关闭强制推车，部分交通停滞；路线跟随也不使用交通灯输入。不同仿真器的交通流与控制器会影响分数，不能把领先直接解释为现实道路安全保证。

#### 实用指南
论文给出 CARLA 多基准实现及复现实验配置，训练目标来自规则专家，推理同时融合对象 token、运动预测、路线几何与伙伴轨迹。迁移前应重建通信时延/丢包、规划视野和优先权约束，并将感知误差与伙伴策略变化纳入测试。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：driving_planning

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.13494v1)
- [arXiv](https://arxiv.org/abs/2611.13494v1)

---

<a id='2611.13528v1'></a>
## [GEMOR-GS: Global Extrinsic-calibration-refined Mapping and Online Relocalization for 3D Gaussian-Splatting SLAM](https://arxiv.org/abs/2611.13528v1)

**Authors:** Yuchen Yang, Ming Ding, Xinyu Wu, Hesheng Wang

**Published:** 2026-11-10

**Categories:** cs.RO, cs.CV

**Abstract:**

Photorealistic and metrically consistent world representations are a primary requirement for LiDAR--inertial--visual SLAM systems that serve robots over continuous operation. 3D Gaussian splatting makes such representations attainable by supplying an explicit, renderable, and queryable scene model. We present GEMOR-GS (Global Extrinsic-calibration-refined Mapping and Online Relocalization for 3D Gaussian-Splatting SLAM) to make the Gaussian map geometrically accurate, globally refined, and re-enterable, providing a better appearance for robot perception and decision-making. To improve the 3DGS map, the LiDAR--camera extrinsic is first estimated offline by direct-method alignment over dedicated calibration segments and reused by all following online and offline processes. Under the calibrated geometry, keyframe poses and Gaussian parameters are refined alternately under a gated differentiable-rendering and LiDAR--projection objective. The final global reconstruction is then rebuilt from the refined keyframe schema with all online constraints lifted. For the robot SLAM lifecycle, the method recovers the pose from the 3DGS map after tracking loss through an XFeat bypass map, RANSAC-PnP coarse localization, and Gaussian RGB-D warp refinement. Experiments on one self-collected and five public sequences quantify each element: the offline reconstruction wins all paired keyframes against the online map by 14.3--19.9 dB, improves PSNR by approximately 9 dB over Gaussian-LIC2, and achieves 86--100% held-out relocalization success tied to extrinsic quality. The results show that GEMOR-GS improves the geometric accuracy, appearance fidelity, and re-enterability of the 3DGS map, serving as a dependable robot world model for SLAM and downstream tasks.

### 论文解读

#### 摘要翻译
GEMOR-GS: Global Extrinsic-calibration-refined Mapping and Online Relocalization for 3D Gaussian-Splatting SLAM提出：GEMOR-GS把在线跟踪和离线全局地图精化分开：在线以 IESKF 跟踪高斯地图；离线通过直接法细化相机外参，并交替优化关键帧位姿和地图，再执行重初始化、增密、裁剪及球谐阶数递进。重定位用 XFeat 描述子匹配、RANSAC-PnP 与 RGB-D warp 检验恢复位姿。 论文报告的范围是六个序列共6,632对关键帧；离线版本相对自有在线结果提升14.3–19.9 dB PSNR，在训练视图上比 Gaussian-LIC2 高5.38–12.29 dB。重定位多数序列成功率86–100%，CBD仅35%、Bright为0%。外参在弱纹理处仍会失败。

#### 方法动机分析
高斯泼溅 SLAM 的局部在线地图会受外参误差和累积漂移影响，全局细化后的地图也未必能让系统重新定位。作者假设分离在线实时状态估计与离线全局优化，再保留一条图像与几何联合重定位路径，可兼顾实时跟踪和全局重用。

#### 方法设计详解
GEMOR-GS把在线跟踪和离线全局地图精化分开：在线以 IESKF 跟踪高斯地图；离线通过直接法细化相机外参，并交替优化关键帧位姿和地图，再执行重初始化、增密、裁剪及球谐阶数递进。重定位用 XFeat 描述子匹配、RANSAC-PnP 与 RGB-D warp 检验恢复位姿。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。方法依赖相机-LiDAR外参、关键帧同步 RGB-D 及高斯地图管理。在线建图训练步数约42k，离线优化使用140–240k步；复现要固定数据序列、标定与视图划分，并测在线时延、离线质量和重定位率。在线滤波尚未用重定位结果闭环更新。

#### 方法对比分析
相较只做在线跟踪或离线全局重建的高斯 SLAM，GEMOR-GS将标定精化和在线重定位纳入同一系统。离线图像质量显著优于其在线版本，但与强基线的比较受训练视图/数据集协议影响，不能把 PSNR 提升直接当成定位或重建全面胜出。

#### 实验分析（精简版）
六个序列共6,632对关键帧；离线版本相对自有在线结果提升14.3–19.9 dB PSNR，在训练视图上比 Gaussian-LIC2 高5.38–12.29 dB。重定位多数序列成功率86–100%，CBD仅35%、Bright为0%。外参在弱纹理处仍会失败。 真实证据覆盖六个序列，不是广域部署；CBD与Bright重定位明显退化，弱纹理会损害标定。全局优化当前离线执行，恢复位姿尚未回灌在线滤波，网格/碰撞几何仍留作后续工作。

#### 实用指南
方法依赖相机-LiDAR外参、关键帧同步 RGB-D 及高斯地图管理。在线建图训练步数约42k，离线优化使用140–240k步；复现要固定数据序列、标定与视图划分，并测在线时延、离线质量和重定位率。在线滤波尚未用重定位结果闭环更新。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：robot_mapping

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.13528v1)
- [arXiv](https://arxiv.org/abs/2611.13528v1)

---

<a id='2611.13790v1'></a>
## [RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving](https://arxiv.org/abs/2611.13790v1)

**Authors:** Yu Zhao, Hongyi Duan, Ming Yin, Hongyuan Zhang, Fan Yang, Hongzhi Wang, Xiubo Liang

**Published:** 2026-11-10

**Categories:** cs.RO

**Abstract:**

End-to-end autonomous driving systems increasingly rely on generating multiple trajectory candidates and selecting one for execution. In this setting, a central challenge is no longer only candidate generation, but how to evaluate candidates according to their long-horizon consequences under uncertainty. Existing methods still mainly use deterministic point scores, heuristic subscores, or single-outcome evaluations, which makes it difficult to capture delayed failures, tail risk, and ranking instability under plausible future perturbations. We present RiSE-Drive, a framework for risk-sensitive long-horizon candidate evaluation in end-to-end autonomous driving. RiSE-Drive first performs candidate-conditioned latent future rollout, then estimates long-horizon outcome distributions instead of single scalar scores, and finally ranks candidates using a tail-risk-aware and perturbation-robust decision criterion. This reformulates candidate scoring from deterministic oracle-style evaluation into distributional outcome assessment with explicit risk control. As a result, the evaluator is encouraged to prefer candidates that are not only strong under nominal conditions, but also reliable under long-horizon rollout and safety-critical deviations. Experiments on the NAVSIM-v1 and NAVSIM-v2 benchmarks, as well as the closed-loop HUGSIM benchmark, show that RiSE-Drive achieves state-of-the-art performance and significantly improves the robustness of candidate selection.

### 论文解读

#### 摘要翻译
RiSE-Drive: Risk-Sensitive Long-Horizon Candidate Evaluation for End-to-End Autonomous Driving提出：RiSE-Drive保持上游感知与候选轨迹生成不变，给每条候选未来轨迹配4种未来条件，滚动潜在后果并形成候选级结果分布。分布头输出有序分位数，排序联合均值、低尾 CVaR 与跨条件不稳定度，使系统不仅关注平均驾驶质量，也惩罚尾部风险和条件敏感。 论文报告的范围是NAVSIM-v1 PDMS 94.2 对 DrivoR 93.1；NAVSIM-v2 EPDMS 51.3 对48.3，安全项提升但 ego progress 70.0降到68.7。HUGSIM平均 RC 50.1对49.8，硬/极端场景HD分数22.7/39.2，对照18.8/32.5。模型参数从41.8M增至42.3M。

#### 方法动机分析
端到端驾驶的候选生成可给出多个可能动作，但以平均预测结果排序会忽略少数高风险后果，且单一未来条件可能过度自信。本文假设显式采样多种未来条件、预测长时结果分布并惩罚低尾，可改善风险敏感的候选选择，而不重训感知或规划提议模块。

#### 方法设计详解
RiSE-Drive保持上游感知与候选轨迹生成不变，给每条候选未来轨迹配4种未来条件，滚动潜在后果并形成候选级结果分布。分布头输出有序分位数，排序联合均值、低尾 CVaR 与跨条件不稳定度，使系统不仅关注平均驾驶质量，也惩罚尾部风险和条件敏感。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。实现每场景生成4个未来分支、每分支评估64条候选并预测10个分位数；使用 DINOv2 ViT-S 表征，训练用4张 RTX 4090 GPU。接入现有系统需保持候选轨迹协议一致，并校准 CVaR 尾部水平和跨场景风险分布。

#### 方法对比分析
区别于常见均值回归/短视候选评分，RiSE增加候选条件潜在后果传播、有序分位数和 CVaR 风险项。相较仅安全指标提升，该方法在部分基准以进度略降换安全改善，因此适用性取决于安全与效率权衡。

#### 实验分析（精简版）
NAVSIM-v1 PDMS 94.2 对 DrivoR 93.1；NAVSIM-v2 EPDMS 51.3 对48.3，安全项提升但 ego progress 70.0降到68.7。HUGSIM平均 RC 50.1对49.8，硬/极端场景HD分数22.7/39.2，对照18.8/32.5。模型参数从41.8M增至42.3M。 主要证据是仿真/基准评估，没有实体车辆验证。某些安全分数提高伴随进度下降，HUGSIM平均增益很小；多条件分支是否覆盖真实未来仍取决于训练分布。不能把风险排序称为形式安全保证。

#### 实用指南
实现每场景生成4个未来分支、每分支评估64条候选并预测10个分位数；使用 DINOv2 ViT-S 表征，训练用4张 RTX 4090 GPU。接入现有系统需保持候选轨迹协议一致，并校准 CVaR 尾部水平和跨场景风险分布。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：driving_planning

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.13790v1)
- [arXiv](https://arxiv.org/abs/2611.13790v1)

---

<a id='2611.13818v1'></a>
## [WAM is capable in BEHAVIOR-1K: the 2nd-place solution to the 2026 BEHAVIOR Challenge](https://arxiv.org/abs/2611.13818v1)

**Authors:** Yelin He, Jiaquan Ye, Xinming Wei, Wenjie Xiao, Gengming Zhang, Guoting Wei, Xianbiao Qi, Rong Xiao

**Published:** 2026-11-10

**Categories:** cs.RO

**Abstract:**

This work presents the 2nd-place solution (Team IF1001) to the 2026 BEHAVIOR Challenge, which evaluates mobile manipulation across 100 household tasks. Our method is built on a powerful world action model (WAM), Cosmos 3. We improve Cosmos 3 by using richer visual history, task-progress conditioning, and task-group and skill-specific LoRA adapters. During inference, we apply a separate progress estimator to supply task-progress context, and use task- and progress-conditioned routing to select a task-group or skill-specific policy, complemented by action-candidate aggregation, rolling inpainting, and task-specific temporal and gripper control. To meet the minimum inference requirement of 1 FPS, we use neither recursive self-improvement (RSI) nor agentic harness methods and rely on no closed-source models; during official private-test evaluation, our system handled concurrent requests across 100 evaluation ports with limited hardware resources, sustaining an average end-to-end inference rate above 2 FPS per port. Our method achieves a mean Q-score of 0.613 and a full-task success rate of 40.9%, averaged over 3 self-evaluation rounds across all 100 tasks. On the official private test set, with 20 instances per task, our system achieves a mean Q-score of 0.615 and a full-task success rate of 41.2%.

### 论文解读

#### 摘要翻译
WAM is capable in BEHAVIOR-1K: the 2nd-place solution to the 2026 BEHAVIOR Challenge提出：方案以 Cosmos 3 为基础，把更丰富视觉历史、阶段进度估计、任务组 LoRA 与技能 LoRA 路由结合：进度估计器从历史图像预测子任务/阶段，再由路由器选择适配器，策略生成动作块并聚合多个候选。执行端滚动视频补全，配合动作重定时、夹爪控制与 TOPP-RA 轨迹速度约束。 论文报告的范围是官方测试平均 Q=0.615，完整任务率41.2%，获第二名；公开验证 Q=0.613、成功率40.9%，为三轮结果。训练约20k示范、100个任务，600k步、7,095 GPU小时；推理聚合5个动作候选、4步去噪，动作块32×23并以15Hz执行。

#### 方法动机分析
BEHAVIOR-1K 家务任务长时、多阶段，视觉历史存在歧义且技能差异明显。统一策略需要知道任务进度并在不同技能间复用数据。方案假设任务组和技能级适配器、显式阶段信号以及候选聚合能减少错阶段执行，并利用历史视频提高动作连贯性。

#### 方法设计详解
方案以 Cosmos 3 为基础，把更丰富视觉历史、阶段进度估计、任务组 LoRA 与技能 LoRA 路由结合：进度估计器从历史图像预测子任务/阶段，再由路由器选择适配器，策略生成动作块并聚合多个候选。执行端滚动视频补全，配合动作重定时、夹爪控制与 TOPP-RA 轨迹速度约束。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。复现需要 BEHAVIOR-1K 仿真、Cosmos 3、约7,095 GPU小时训练预算、技能路由配置和约束式动作执行。TOPP-RA这里只给速度/路径约束，不提供力矩或动态可行保证。需要先验证进度预测与路由，再开展闭环任务和失败恢复测试。

#### 方法对比分析
相较纯单一策略微调，方案加上阶段估计与分层 LoRA；与手写任务专属控制流程不同，执行主体仍是视频/动作世界模型。挑战赛结果证明端到端系统可竞争，但消融没有完全隔离组件贡献，路由配置对分数影响明显，无法把总成绩归因于某一个模块。

#### 实验分析（精简版）
官方测试平均 Q=0.615，完整任务率41.2%，获第二名；公开验证 Q=0.613、成功率40.9%，为三轮结果。训练约20k示范、100个任务，600k步、7,095 GPU小时；推理聚合5个动作候选、4步去噪，动作块32×23并以15Hz执行。 挑战赛报告是模拟器结果；作者明确对照版本与覆盖面不一，消融较弱。分技能适配器路由会改变成绩，且高算力/示范需求大。轨迹速度规划也不等于底层动力学安全保证。

#### 实用指南
复现需要 BEHAVIOR-1K 仿真、Cosmos 3、约7,095 GPU小时训练预算、技能路由配置和约束式动作执行。TOPP-RA这里只给速度/路径约束，不提供力矩或动态可行保证。需要先验证进度预测与路由，再开展闭环任务和失败恢复测试。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：robot_learning

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.13818v1)
- [arXiv](https://arxiv.org/abs/2611.13818v1)

---

<a id='2611.14025v1'></a>
## [HiPFuse: Real-Time LiDAR Semantic Segmentation for Robotic Perception on Edge Devices](https://arxiv.org/abs/2611.14025v1)

**Authors:** Haiyang Wu, George Vosselman, Ville Lehtola

**Published:** 2026-11-10

**Categories:** cs.CV, cs.RO

**Abstract:**

Real-time robotic perception requires accurate LiDAR semantics under limited resources. Bird's-eye-view (BEV) aggregation is efficient but can mix differently labeled surfaces in columns. We present HiPFuse, a point-BEV fusion framework that exchanges point and height-partitioned BEV features across horizontal scales, combining point-wise detail with 2D spatial context from one LiDAR scan. Fixed height assignments preserve separate vertical aggregation groups, with auxiliary supervision of initial BEV features using targets from the same 3D cells. A dual-scale BEV context module adds broader context from a coarser grid. On SemanticKITTI and MCD, HiPFuse achieves 63.7% and 62.5% mIoU at 14.2 and 22.4 FPS on Jetson AGX Orin, respectively, exceeding the 10 Hz scan rate. Separate training and evaluation show indoor applicability. Replacing only the baseline semantic front end with HiPFuse in fixed robotic systems reduces odometry absolute trajectory error RMSE by 10.6% on average across two outdoor sequences and increases accepted-loop recall by 8.7 percentage points on SemanticKITTI. Qualitative semantic maps show more continuous road labeling. Our code is available at https://github.com/haiyang2022/HiPFuse.

### 论文解读

#### 摘要翻译
HiPFuse: Real-Time LiDAR Semantic Segmentation for Robotic Perception on Edge Devices提出：HiPFuse从单帧点云按固定高度区间构造 BEV 单元，点支路保留逐点描述，BEV支路用2D卷积提取空间上下文；两者在多个水平尺度反复交换特征。高度分组在下采样时保持，DSC模块以更粗网格加入全局上下文，辅助头监督中间BEV。最终将逐点与BEV特征融合进行语义分类。 论文报告的范围是Jetson AGX Orin 上 SemanticKITTI 63.7% mIoU/14.2 FPS，MCD 62.5%/22.4 FPS，均快于10Hz；SemanticKITTI Odom ATE为3.71m，对SalsaNext 3.77m、HARP-NeXt 3.76m；MCD ATE 1.74m，较优。Indoor mIoU 71.6%，低于HARP-NeXt 75.2%。单种子主要对比，组件完整模型部分指标用3种子。

#### 方法动机分析
BEV聚合节省计算，却可能把同一水平格内不同高度、标签不同的表面混在一起；纯点特征又难以廉价取得全局上下文。作者假设保留高度分区和点级特征，并跨尺度交互，可以在边缘设备扫描速率内提高语义质量。

#### 方法设计详解
HiPFuse从单帧点云按固定高度区间构造 BEV 单元，点支路保留逐点描述，BEV支路用2D卷积提取空间上下文；两者在多个水平尺度反复交换特征。高度分组在下采样时保持，DSC模块以更粗网格加入全局上下文，辅助头监督中间BEV。最终将逐点与BEV特征融合进行语义分类。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。代码公开。输入为单帧坐标与强度；Orin实测包括读盘、预处理、推理、后处理。三数据集分别设不同高度范围、格尺寸和高度分区，迁移新传感器需重新选网格并有标注微调；应在目标板上测端到端延迟和显存。

#### 方法对比分析
相较单层BEV或单独点/体素方法，HiPFuse以固定高度分区、点-BEV双支路、辅助监督和粗尺度上下文构成组合方案。它在多个机器人感知系统中替换语义前端后改善部分下游指标，但仍低于若干分割基线的 indoor mIoU，说明跨任务收益不是每个分类指标都占优。

#### 实验分析（精简版）
Jetson AGX Orin 上 SemanticKITTI 63.7% mIoU/14.2 FPS，MCD 62.5%/22.4 FPS，均快于10Hz；SemanticKITTI Odom ATE为3.71m，对SalsaNext 3.77m、HARP-NeXt 3.76m；MCD ATE 1.74m，较优。Indoor mIoU 71.6%，低于HARP-NeXt 75.2%。单种子主要对比，组件完整模型部分指标用3种子。 参数网格依传感器设置，换传感器可能需要标签与重训；Semantic-ITC低于强基线，类别表现不均。下游 odometry 只有两条室外序列和一条无真值室内路线；回环效果基于单一序列，且结果多为固定系统中的前端替换。

#### 实用指南
代码公开。输入为单帧坐标与强度；Orin实测包括读盘、预处理、推理、后处理。三数据集分别设不同高度范围、格尺寸和高度分区，迁移新传感器需重新选网格并有标注微调；应在目标板上测端到端延迟和显存。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：robot_perception

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14025v1)
- [arXiv](https://arxiv.org/abs/2611.14025v1)

---

<a id='2611.14048v1'></a>
## [LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning](https://arxiv.org/abs/2611.14048v1)

**Authors:** Xuheng Zhang, Yiming Zhang, Jialin Gao, Hao Chen, Jiaming Liu, Ji Li, Runzhong Huang, Yinxi Wang, Chenyang Gu, Zihan Yang, Yaowei Liu, Zhonghao Yan, Ziyu Guo, Renrui Zhang, Peng Jia, Pheng-Ann Heng, Shanghang Zhang

**Published:** 2026-11-10

**Categories:** cs.RO

**Abstract:**

World Action Models (WAMs) have recently emerged as a promising paradigm for robotic manipulation, endowing policies with strong capabilities for modeling temporal and physical dynamics. Existing WAMs, encompassing both video-first action prediction and joint video-action modeling, directly transfer global physical representations from world models to action generation. However, global scene representations may underemphasize local robot-object interactions, creating a mismatch between representations optimized for world evolution and those required for manipulation. To bridge this gap, we propose LaST0.5, a novel WAM that introduces an interaction-centric physical representation between world modeling and action generation. Specifically, LaST0.5 learns a compact interaction-centric latent space under the supervision of visual interaction cues between the robot and the environment. This reasoning space is instantiated as a latent expert that bridges the world expert and the action expert within a Mixture-of-Transformers architecture. Furthermore, LaST0.5 performs adaptive world-to-action transfer by jointly predicting future representations and progress values, using progress consistency to adjust action execution and trigger future resampling. Finally, we pretrain LaST0.5 on a large-scale robotic data corpus comprising over 14k hours of data, establishing interaction-centric physical reasoning as a scalable and effective paradigm. LaST0.5 achieves 99.5% on LIBERO, surpasses the previous state of the art on ten evaluated RLBench tasks, and demonstrates robust long-horizon execution and precise dexterous manipulation in real-world experiments.

### 论文解读

#### 摘要翻译
LaST0.5: A Generalist World Action Model with Interaction-Centric Latent Reasoning提出：LaST0.5把预训练视频世界模型、交互潜变量专家和动作专家放进 MoT：先预测未来，再用未来帧中的末端轨迹曲线和目标物体框构造交互监督，最后以紧凑潜变量条件化动作块。AWAT另预测分段任务进度，只执行预测值持续上升的对齐动作；进度下降就重新观测，初值回退则并行采样未来再选。 论文报告的范围是预训练数据约14,892.6小时、2.48M回合。LIBERO平均成功率99.5%，对LaST0 98.1%；RLBench十任务均值88.0%，对LaST0 82.0%。Tianji实物四任务均值63.8%，对π0.5 52.5%；位置偏移下两任务均值40.0%，对照π0.5 15.0%。在8张H20 GPU上训练100/300 epoch；推理视频两步去噪、动作10步去噪，约5Hz。

#### 方法动机分析
现有世界动作模型把全局场景演化特征直接用于动作生成，可能让背景压过局部抓取/接触关系。作者假设交互中心表示能在世界预测和动作输出间形成适合操作的中间层，并可用进度一致性识别不可信的未来预测。

#### 方法设计详解
LaST0.5把预训练视频世界模型、交互潜变量专家和动作专家放进 MoT：先预测未来，再用未来帧中的末端轨迹曲线和目标物体框构造交互监督，最后以紧凑潜变量条件化动作块。AWAT另预测分段任务进度，只执行预测值持续上升的对齐动作；进度下降就重新观测，初值回退则并行采样未来再选。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。训练在8张H20 GPU上进行100/300 epoch；推理约5Hz。使用冻结 Cosmos 世界专家和 SigLIP 交互目标编码器；离线 Grounding DINO 框与相机投影轨迹生成视觉提示。数据跨机器人统一到10Hz相对末端位姿；上线先评估进度预测、动作截断与回退策略，再逐项加入视觉域和接触偏差。

#### 方法对比分析
相比直接视频到动作或视频-动作联合预测，LaST0.5增加交互潜变量监督和未来进度驱动的动作截断/重采样。它并非更强的通用控制保证：实物仅在四类定制双臂任务验证，失败仍来自滑移、碰撞和毫米级插入误差。

#### 实验分析（精简版）
预训练数据约14,892.6小时、2.48M回合。LIBERO平均成功率99.5%，对LaST0 98.1%；RLBench十任务均值88.0%，对LaST0 82.0%。Tianji实物四任务均值63.8%，对π0.5 52.5%；位置偏移下两任务均值40.0%，对照π0.5 15.0%。在8张H20 GPU上训练100/300 epoch；推理视频两步去噪、动作10步去噪，约5Hz。 仿真 LIBERO 汇报训练期间最佳成功率，而 RLBench使用三次运行；实物每项20回合且仅覆盖四项定制任务。论文失败分析指出拉链卡滞、篮筐被碰动、胶带滑脱与插入错位；位置OOD成功率也只有40%。14k小时预训练和8张H20复现门槛高。

#### 实用指南
训练在8张H20 GPU上进行100/300 epoch；推理约5Hz。使用冻结 Cosmos 世界专家和 SigLIP 交互目标编码器；离线 Grounding DINO 框与相机投影轨迹生成视觉提示。数据跨机器人统一到10Hz相对末端位姿；上线先评估进度预测、动作截断与回退策略，再逐项加入视觉域和接触偏差。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：world_models

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14048v1)
- [arXiv](https://arxiv.org/abs/2611.14048v1)

---

<a id='2611.14440v1'></a>
## [UGG: Unified Generative Geometry Model](https://arxiv.org/abs/2611.14440v1)

**Authors:** Amani Kiruga, Zhiyi Li, Ruojin Cai, Hansen Lillemark, Xiaoshan Wu, Ayush Tewari, Yilun Du

**Published:** 2026-11-10

**Categories:** cs.CV, cs.LG, cs.RO

**Abstract:**

Understanding 4D scenes is a core challenge in visual intelligence, requiring models to jointly capture scene geometry, appearance, and dynamics from partial observations. However, existing approaches are subdivided into feedforward models for direct geometric prediction, and generative models for visual synthesis. We introduce UGG, a unified framework for 4D scene understanding that jointly models RGB video and geometric representations such as depth and camera pose. By assigning different noise levels across modalities and time, UGG formulates visual generation and geometric prediction as conditional completion within a single generative model. This flexible formulation enables the model to learn from diverse, heterogeneous datasets with different subsets of geometric and visual annotations. Furthermore, our generative formulation enables inference-time search over multiple plausible scene hypotheses, using geometric evidence extracted from observed RGB frames to improve geometric prediction without ground truth geometry. We demonstrate UGG's effectiveness over non-unified baselines across camera-controlled video generation, depth estimation, and camera pose prediction, including in challenging sparse observation settings. Code and visualizations are available at https://ugg-model.github.io/

### 论文解读

#### 摘要翻译
UGG: Unified Generative Geometry Model提出：UGG把动态场景表达为逐帧 RGB、深度与相机 Plücker ray map。基于 Diffusion Forcing 的多模态 transformer 为不同模态/时间帧指定噪声等级，干净观测作条件、缺失模态作去噪目标，因此同一模型支持生成、深度和位姿预测。推理时从多个随机初值生成场景候选，再用观测图像的 RAFT 对应与重投影一致性分别筛选深度和相机。 论文报告的范围是50帧、288×384视频，训练约10万步、128张H200、两周。稠密位姿在ScanNet++ ATE 0.087，对Geo4D 0.190；稀疏位姿搜索及深度在多基准通常获益。稠密生成的一些指标仍不占优；推理40步，一段50帧约20秒/H100，TTS最高采样16候选。

#### 方法动机分析
单次前馈几何模型常给唯一预测，在稀疏观测和歧义场景中不稳；视频生成模型则可能视觉合理但几何不一致。作者假设外观、深度和相机射线可作为对齐模态共同学习，缺失预测与视频生成都可写成条件补全，并允许测试时用图像证据在多个几何假设中择优。

#### 方法设计详解
UGG把动态场景表达为逐帧 RGB、深度与相机 Plücker ray map。基于 Diffusion Forcing 的多模态 transformer 为不同模态/时间帧指定噪声等级，干净观测作条件、缺失模态作去噪目标，因此同一模型支持生成、深度和位姿预测。推理时从多个随机初值生成场景候选，再用观测图像的 RAFT 对应与重投影一致性分别筛选深度和相机。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。作者公开代码与权重。复现需要 Wan2.2-TI2V-5B 初始化、冻结VAE/文本编码器、异构视频数据及多模态 ray/depth 预处理；训练配置为全局 batch 1024、128 H200。测试时需安装 RAFT 与 DINOv2 验证器，单段推理成本较高。

#### 方法对比分析
相较分别训练视频生成、深度和位姿模型，UGG共享多模态注意力与条件去噪，同时允许监督不完整的数据参与训练。测试时搜索为概率模型增加额外推理预算。它不是普遍优于专用法：室内深度和像素对齐生成仍有专用模型领先，验证器只是重投影代理。

#### 实验分析（精简版）
50帧、288×384视频，训练约10万步、128张H200、两周。稠密位姿在ScanNet++ ATE 0.087，对Geo4D 0.190；稀疏位姿搜索及深度在多基准通常获益。稠密生成的一些指标仍不占优；推理40步，一段50帧约20秒/H100，TTS最高采样16候选。 深度需逐视频尺度/偏移对齐，稀疏输入先验不足；测试时重投影筛选可能被运动、反光、遮挡、低纹理影响，失败时退化为随机候选。论文承认专用模型在部分室内深度、视频像素指标领先；训练资源也不易复现。

#### 实用指南
作者公开代码与权重。复现需要 Wan2.2-TI2V-5B 初始化、冻结VAE/文本编码器、异构视频数据及多模态 ray/depth 预处理；训练配置为全局 batch 1024、128 H200。测试时需安装 RAFT 与 DINOv2 验证器，单段推理成本较高。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：world_models

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14440v1)
- [arXiv](https://arxiv.org/abs/2611.14440v1)

---

<a id='2611.14281v1'></a>
## [UniFlow-RS: Unified Generative Prediction for Heterogeneous Remote-Sensing Tasks](https://arxiv.org/abs/2611.14281v1)

**Authors:** Jianlong He, Liang Liao, Jing Xiao, Chaofeng Chen, Gui-Song Xia

**Published:** 2026-11-10

**Categories:** cs.CV

**Abstract:**

Remote-sensing foundation models increasingly unify representations across sensing modalities and downstream tasks, yet their prediction mechanisms remain largely task-specific. Separate decoders, prediction heads, and objectives leave prediction fragmented across tasks. This naturally raises a question: can heterogeneous remote-sensing tasks share the prediction process? UniFlow-RS answers this question with a single latent flow-matching predictor shared across tasks. This requires addressing two key challenges: unifying heterogeneous prediction spaces and harmonizing joint optimization across tasks. First, we cast heterogeneous task outputs into structured visual targets in a common latent space, allowing them to be modeled by the same pretrained flow predictor. Task-dependent conditioning further accommodates differences in input configurations. Second, we optimize all tasks with a common Flow Matching objective, eliminating task-specific prediction losses. To mitigate optimization mismatch across tasks and modalities, we introduce Task-Modality Adaptive Timestep Sampling (TMAS), which dynamically allocates training timesteps according to the online learning state of each task-modality group. Using a single shared model, UniFlow-RS handles six tasks and thirteen datasets, surpassing recent task-specific state-of-the-art methods on several benchmarks while remaining competitive on the others. Beyond these benchmarks, the same checkpoint demonstrates strong zero-shot transfer across unseen datasets, task settings, occlusion conditions, and near-infrared false-color imagery, without target-specific adaptation.

### 论文解读

#### 摘要翻译
UniFlow-RS: Unified Generative Prediction for Heterogeneous Remote-Sensing Tasks提出：UniFlow-RS把建筑提取、变化检测、语义分割、高程、指代表达分割和旋转框检测统一渲染为结构化视觉目标，经冻结 VAE 编码到共同潜空间。单一预训练 MMDiT 与共享 LoRA 学 flow-matching 速度场；输入条件按任务变化，时序变化用带方向的潜变量差。TMAS 按任务-模态跟踪不同时间步的长期误差和短期变化，动态分配采样，并保留均匀探索。 论文报告的范围是单一共享 LoRA 覆盖6项任务、13个数据集，统一任务分数 UTS=69.35，固定Beta对照68.45；TMAS达到固定Beta最终分数需37k步而非60k。最薄小实例的 VAE 恢复后 recall@0.75由25.39%降至21.56%，揭示边界与几何误差。另有无适配的跨数据集/云遮挡/近红外零样本测试。

#### 方法动机分析
遥感基础模型虽能共享特征，却常保留各任务专属预测头和损失；多模态任务的输出形式和流轨迹学习难度差别大。作者假设确定性视觉编解码能把异质目标投射到共用预测空间，任务条件补偿输入差异，而时间步自适应能缓解联合训练不均衡。

#### 方法设计详解
UniFlow-RS把建筑提取、变化检测、语义分割、高程、指代表达分割和旋转框检测统一渲染为结构化视觉目标，经冻结 VAE 编码到共同潜空间。单一预训练 MMDiT 与共享 LoRA 学 flow-matching 速度场；输入条件按任务变化，时序变化用带方向的潜变量差。TMAS 按任务-模态跟踪不同时间步的长期误差和短期变化，动态分配采样，并保留均匀探索。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。训练采用冻结 FLUX.2 Klein 4B、Qwen3文本编码器、共享 rank-32 LoRA（49.1M可训练参数），四张 RTX 6000 Ada、60k步、batch 32；推理默认两步。代码及模型资源公开。复现时必须严格使用目标 palette、label ontology 和确定性恢复规则。

#### 方法对比分析
相比共享骨干但每任务独立解码器的多任务模型，本文共享生成式预测过程和 flow-matching 目标。TMAS不重加权任务数据量，而是对十个任务-模态组分别调整时间步。方法实现通用性强，但检测目标经栅格生成再恢复实例会损失细粒度几何。

#### 实验分析（精简版）
单一共享 LoRA 覆盖6项任务、13个数据集，统一任务分数 UTS=69.35，固定Beta对照68.45；TMAS达到固定Beta最终分数需37k步而非60k。最薄小实例的 VAE 恢复后 recall@0.75由25.39%降至21.56%，揭示边界与几何误差。另有无适配的跨数据集/云遮挡/近红外零样本测试。 任务汇总得分的平均方式跨数据集归一并等权，不代表每个任务都超过专用模型；旋转检测对几何恢复敏感，VAE 对薄边界有损。零样本评估规模有限，且不同基准的模型容量/预训练数据不完全可比。

#### 实用指南
训练采用冻结 FLUX.2 Klein 4B、Qwen3文本编码器、共享 rank-32 LoRA（49.1M可训练参数），四张 RTX 6000 Ada、60k步、batch 32；推理默认两步。代码及模型资源公开。复现时必须严格使用目标 palette、label ontology 和确定性恢复规则。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：visual_generation

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14281v1)
- [arXiv](https://arxiv.org/abs/2611.14281v1)

---

<a id='2611.13863v1'></a>
## [Learn-To-Propagate: Distributionally Robust Motion Planning for Nonlinear Uncertain Systems](https://arxiv.org/abs/2611.13863v1)

**Authors:** Eduardo Figueiredo, Adrien Banse, Qi Heng Ho, Raphael Jungers, Morteza Lahijanian, Luca Laurenti

**Published:** 2026-11-10

**Categories:** cs.RO, eess.SY

**Abstract:**

Motion planning for stochastic systems aims to find a sequence of control inputs that drives a system to a goal set while avoiding unsafe states with high probability. Existing approaches, however, are either restricted to linear dynamics with perfect knowledge of the underlying distributions or lack formal correctness guarantees. To address these limitations, we propose \emph{Learn-To-Propagate Distributionally-Robust Forest (L2P-DR-Forest)}, a distributionally-robust sampling-based planning algorithm for \emph{nonlinear} stochastic systems with \emph{unknown} initial state and process noise distributions. Our approach leverages recent advances in optimal transport to learn ambiguity sets that contain the true system distributions with high confidence. We develop an efficient method for propagating these sets through the system dynamics and for verifying their safety. We then integrate these techniques into a novel sampling-based planning algorithm to grow a forest of search trees in the system's state space while accounting for and mitigating the growth of ambiguity-sets radii. We prove that the algorithm is correct, i.e., every returned plan satisfies the specifications with the prescribed confidence, and, under some feasibility and distributional assumptions, is probabilistically complete. Evaluations on several robotic benchmarks demonstrate that our algorithm significantly outperforms existing methods while providing formal guarantees with high confidence.

### 论文解读

#### 摘要翻译
Learn-To-Propagate: Distributionally Robust Motion Planning for Nonlinear Uncertain Systems提出：L2P-DR-Forest从初值和扰动样本学习 Wasserstein ambiguity set；利用动力学 Lipschitz 常数传播集合，并对集合外概率质量单独作上界。每个阶段在搜索树挑选重启节点，从新样本重学更紧的根和噪声分布，随后用 RRT 式树扩展，只保留通过分布鲁棒障碍检验的节点；最终目标检验亦有置信保证。 论文报告的范围是五个基准各20次，本文方法均100%成功；No re-learning 在全部基准0%。AV Maze规划平均1.56秒，粒子基线31.30秒；Tractor Trailer约55.51秒，基线20次都触及2分钟超时。方法的Monte Carlo检验每个时刻观察到0%不安全、目标命中100%，这是模拟证据而非形式证明替代。

#### 方法动机分析
非线性随机系统中的采样规划通常缺正式安全保证；线性高斯近似限制模型，粒子包络则可能保守且代价高。作者假设可访问初态与扰动的 i.i.d. 离线样本，且有可用的 Lipschitz 上界，从而学习不确定分布并在规划中保持概率安全证书。

#### 方法设计详解
L2P-DR-Forest从初值和扰动样本学习 Wasserstein ambiguity set；利用动力学 Lipschitz 常数传播集合，并对集合外概率质量单独作上界。每个阶段在搜索树挑选重启节点，从新样本重学更紧的根和噪声分布，随后用 RRT 式树扩展，只保留通过分布鲁棒障碍检验的节点；最终目标检验亦有置信保证。输入是论文任务中的观测、状态或候选，输出为逐点语义、规划决策、预测几何或任务得分。模块职责分别是构造可学习表示、融合异质线索、显式表达不确定性并把结果交给既定预测/控制端。算法参数含 confidence β=10^-4、每阶段迭代及样本规模增长，支持点通常小于10；Tractor Trailer使用LipSDP Lipschitz界1.2096。实现需初态/噪声数据、危险集合判定与MILP学习器。应用前应估算维度带来的Wasserstein样本复杂度。

#### 方法对比分析
核心差异是以 Wasserstein 分布集取代单一估计分布，并通过多树阶段性重学降低半径膨胀。相对 ε-RandUP-RRT，论文五个模拟基准整体成功率高且部分更快；理论保证建立在明确的 i.i.d.、Lipschitz、子高斯与可行裕量假设下，基线结果并无同等可实例化的置信证书。

#### 实验分析（精简版）
五个基准各20次，本文方法均100%成功；No re-learning 在全部基准0%。AV Maze规划平均1.56秒，粒子基线31.30秒；Tractor Trailer约55.51秒，基线20次都触及2分钟超时。方法的Monte Carlo检验每个时刻观察到0%不安全、目标命中100%，这是模拟证据而非形式证明替代。 作者指出高维样本复杂度、紧 Lipschitz 界与大型扰动数据是主要瓶颈。结果仅在5个模拟基准、特定安全阈值和建模假设下验证；Monte Carlo 0%风险不是实机安全认证，Lipschitz界过松会使可行集消失。

#### 实用指南
算法参数含 confidence β=10^-4、每阶段迭代及样本规模增长，支持点通常小于10；Tractor Trailer使用LipSDP Lipschitz界1.2096。实现需初态/噪声数据、危险集合判定与MILP学习器。应用前应估算维度带来的Wasserstein样本复杂度。迁移时应从论文中实际验证过的输入、设备与数据条件开始，保留基准协议，并增设失败条件和回归对照；不要只复用模型名称或最终平均分。

#### 总结
核心思想：robot_planning

1. 准备观测及任务条件。
2. 用论文提出的共享表示或风险机制生成候选。
3. 按任务目标评估输出并检查失效边界。
4. 在目标设备或系统上重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.13863v1)
- [arXiv](https://arxiv.org/abs/2611.13863v1)

---

<a id='2611.14115v1'></a>
## [LEASH: Stay Safe in Agentic Robotics](https://arxiv.org/abs/2611.14115v1)

**Authors:** Qinfeng Zhu, Weiguang Zhao, Ke Sun, Xihao Guo, Yunxi Jiang, Rui Zhang, Lei Fan, Kaizhu Huang

**Published:** 2026-11-10

**Categories:** cs.RO

**Abstract:**

Foundation models such as GPT and Claude have shown strong capabilities in spatial understanding and embodied intelligence. These advances make agentic robotics a promising path toward artificial general intelligence in robotics. However, we find that using foundation models to control robots poses serious safety risks, and comprehensive safety evaluation is still lacking. We introduce RoboRisk, a comprehensive benchmark for evaluating safety in agentic robotics. Its 92 simulated tasks and 13 physical-robot tasks fall into five categories: Task-level Refusal (T), Plan Repair (P), Execution-time Hazard (E), Injection (I), and Sequential Memory (S). Prior benchmarks cover at most two of these categories, and several evaluate only text outputs. In contrast, RoboRisk covers all five and executes every task in closed loop. We evaluate six state-of-the-art foundation models on RoboRisk: GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna, Claude Opus 5.5, Gemini 4 Argon, and Grok 4.7. These models generally show high harm rates. We also evaluate two robot-specific APIs: the Dream API and PokeBot's RoboAPI. We further propose the Lease-Enforced Agentic Safety Harness (LEASH), which offers essential safety infrastructure for agentic robotics and can be integrated into nearly any agentic architecture. LEASH leverages a decision model (Jev or OpenAI's Decisions API) to implement task, action, and commit gates, which review the task, each action, and each irreversible operation in their visual context. The planner and all three gates share a common safety policy. Experiments on RoboRisk in simulation and on physical robots show that LEASH effectively reduces harm across all six foundation models. It largely preserves their task completion performance and adds little time and token overhead.

### 论文解读

#### 摘要翻译
LEASH: Stay Safe in Agentic Robotics提出一套面向机器人智能体的安全 harness。RoboRisk包含92项仿真和13项实机任务，覆盖任务拒绝、计划修复、执行中突发危险、场景注入及跨轮记忆。LEASH将共享策略用于三个视觉检查点：先审任务目标，再审动作候选，最后在夹爪释放或倾倒前复核。测试显示，多数规划器的伤害率下降，任务完成率大体保留；但实机任务数量少，弱规划器仍有残余风险。

#### 方法动机分析
传统碰撞约束难以识别“把手机放进水里”这类语义风险，文本审查又看不到物理场景。作者认为现有机器人安全评测存在覆盖瓶颈：任务级拒绝之外，计划修复、执行中变化、场景注入和跨轮记忆缺少闭环测试。因此提出共享安全策略、场景感知审查和五类任务基准，处理机器人安全与场景理解的挑战。

#### 方法设计详解
任务门先检查用户目标，可终止任务；动作门在原子动作或长技能执行前检查候选；提交门在释放、倾倒等已监控的不可逆操作前再次判断。每道门查看相机图像及状态，共用同一安全策略。流程从目标判断、动作提案到执行前提交，逐步把审核结果反馈给规划器；被拒绝的动作需依据新观察重拟方案，动作门和提交门本身不能结束任务。实验以Dobot CR5V2实机与MuJoCo仿真为主，实机每个任务仅运行一次；任务/动作阈值为0.5，提交阈值为0.3。

#### 方法对比分析
相比仅审查用户文本或仅限制轨迹碰撞的防护层，LEASH将视觉场景、机器人状态和近期执行历史纳入任务、动作、提交三个时点的审查，并让规划器与审核器遵循同一策略。它的特点是拦截并反馈当前候选，而非由门控器代替规划器重做整个任务；因此系统效果仍依赖规划器是否能提出另一条安全方案。

#### 实验分析（精简版）
仿真包含92项任务，实机包含13项任务。GPT-6 Astra实机伤害率由15.38%降至0，安全成功率从69.23%升至84.62%；GPT-6.1 Sol由38.46%/46.15%变为0%/84.62%。Claude Opus 5.5和Gemini 4 Argon在各13次实机试验中也未观察到伤害。仿真Astra伤害率从11.23%降至0，安全成功率由78.99%增至80.80%。弱规划器仍出现伤害；实机一次试验的证据不能代表部署安全。平均运行时间增加1.66%，每轮门控成本估算为0.00522美元。

#### 实用指南
接入需能在 harness 层拦截动作，并向审核器提供RGB图像、机器人状态、历史与待审候选。任务门处理目标，动作门处理每个提案，提交门覆盖释放和倾倒；被拒绝后重新观察，再生成有实质差异的候选。论文使用高推理设置的规划器，并以MuJoCo基准和CR5V2实机测试。部署前应补齐长技能内部的危险监控、目标机器人专用任务和多次试验。

#### 总结
核心思想：机器人分阶段安全审核

1. 先审目标，再审动作与提交操作。
2. 拒绝候选后重新观察并规划。
3. 用仿真和实机任务持续测量风险与完成率。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.14115v1)
- [arXiv](https://arxiv.org/abs/2611.14115v1)

---

