time: 20261103

# Arxiv Computer Vision Papers - 2026-11-03

## Executive Summary

## 今日主题概览

今日十篇论文共同关注具身系统如何从更多样的数据和不完整观测中，生成可控、可验证的动作。连续动作建模有两条不同路径：ODE Policy直接学习动作变化动力学，TF²-Steering则让触觉预测与实时接触反馈共同修正视觉动作专家。两者都把执行中的时间结构纳入策略，但证据分别覆盖多种机器人任务与有限的精细接触设置。

世界模型方向从数据预训练延伸到遮挡场景规划。OpenNWM利用无动作导航视频学习潜在动作，再与真实控制对齐；SCOUT把隐藏物体几何、动作条件预测和树搜索连成货架整理流程，在500次模拟rollout中域内成功率为80.8%，高于最强基线的72.8%，真实验证为20次中成功12次。部署场景仍受数据来源、固定视角和真实样本量约束。

执行与系统层同样突出。PTS在两种冻结VLA的LIBERO设置下提高吞吐并维持成功率，Robot Use Agent Harness则用统一工具适配和资源感知流水线重叠推理与动作。两者分别解决动作块衔接和代理运行时问题，适合关注真实系统吞吐的读者；真实机器人安全性仍需独立验证。Policy Coverage提出用pass@k区分单次可靠性与多次采样覆盖，是评估生成式策略时值得细读的指标工作。

感知与迁移方向涵盖透明物体深度恢复、高光谱预训练和LiDAR开放类别适配。UA-Depth在八个零样本数据集及小规模真实抓取评估中联合利用深度与不确定性；SpecUltra显式编码波长结构；LiDAR类别发现把几何原型、最优传输和二维视觉语义结合，在十个迁移划分中的九个取得最佳整体mIoU。优先阅读全文可从 ODE Policy、PTS、OpenNWM、SCOUT 与 UA-Depth 入手，分别了解动作表示、执行系统、无标签视频预训练、遮挡规划和不确定性感知的设计与边界。

---

## Table of Contents

1. [UA-Depth: Uncertainty-Aware Depth Restoration for Material-Agnostic Object Manipulation](#2611.01645v1)
2. [Leveraging 2D Foundation Priors for Generalized Category Discovery under Test-Time Adaptation in 3D LiDAR Segmentation](#2611.01682v1)
3. [ODE Policy: Robot Learning via Action Dynamics, Not Actions](#2611.01729v1)
4. [Democratizing Robot-Use Agents: A Unified Harness with Open Models](#2611.01560v1)
5. [Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining](#2611.01565v1)
6. [SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts](#2611.01600v1)
7. [OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining](#2611.01844v1)
8. [$TF^2$-Steering: Action Refinement with Tactile Forecasts and Feedback via Velocity-Field Steering](#2611.01483v1)
9. [SpecUltra: A Unified Foundation Model for Heterogeneous Spectral Images](#2611.01396v1)
10. [Same Path, Less Time: Training-Free Fast Execution of Vision-Language-Action Models under Inference Latency](#2611.01390v1)

---

## Papers

<a id='2611.01645v1'></a>
## [UA-Depth: Uncertainty-Aware Depth Restoration for Material-Agnostic Object Manipulation](https://arxiv.org/abs/2611.01645v1)

**Authors:** Rui Cao, Jia-Hui Pan, Yin Zhang, Qiwei Meng, Jiapeng Li, Gang Wang, Runsong Zhu, Wei Chen, Chi-Wing Fu, Jiangliu Wang, Yun-Hui Liu

**Published:** 2026-11-01

**Categories:** cs.CV, cs.RO

**Abstract:**

Robotic manipulation requires accurate metric depth, yet commodity RGB-D cameras often fail on non-Lambertian objects such as transparent, reflective, and translucent materials, producing holes, structured noise, and biased depth. Most restoration methods produce only a single depth map, treating noisy and reliable regions uniformly for downstream tasks while ignoring regional uncertainty. We present UA-Depth, an uncertainty-aware depth restoration framework that iteratively co-refines depth and pixel-wise uncertainty, and uses uncertainty-gated features to suppress unreliable cues for stable updates. To strengthen zero-shot behavior, UA-Depth leverages a frozen depth foundation model as a prior to initialize missing or unreliable depth, and applies two-stage MAD-weighted log-depth scale normalization to reduce outlier bias in relative-to-metric alignment. We introduce an eight-dataset zero-shot benchmark across diverse scenes, materials, and sensors. Across these benchmarks, UA-Depth achieves the best mean rank among representative baselines, with particularly strong gains on datasets dominated by transparent and reflective objects. Beyond depth metrics, we demonstrate downstream improvements in category-level 6D pose estimation and 6D grasp synthesis by exploiting the predicted uncertainty. \emph{Code, models, and data for training and evaluation will be released upon publication.}

### 论文解读

#### 摘要翻译
本文提出不仅补全深度，还估计哪里不可靠，并让下游抓取和位姿模块按不确定性抑制风险区域。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
透明、反光等非朗伯材料会让RGB-D产生空洞、噪声和系统偏差；只输出一张深度图会把可靠和不可靠像素同等使用。作者假设深度与像素不确定性应反复协同更新，且冻结深度基础模型可为缺失区域提供零样本先验。

#### 方法设计详解
输入RGB-D深度和颜色，先以冻结Depth Anything V2先验填补缺失结构，并通过两阶段MAD加权对数深度归一化抑制离群尺度偏差。迭代网络同时维护深度和像素不确定性：不确定性门控融合视觉特征，降低噪声区域对下一轮更新的影响；输出深度及置信度图交给6D位姿估计、抓取合成等模块，作为权重而非仅做视觉展示。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
相比只估一个深度值，UA-Depth显式提供不确定度并在迭代中用于特征门控；相较一般相对深度先验，它加入鲁棒尺度对齐以面向度量深度。适合透明与反光物体抓取前感知，但基础模型先验和真实传感器域仍影响结果，置信度并不自动保证下游策略安全。

#### 实验分析（精简版）
八个零样本数据集平均排名1.4，论文报告领先代表性基线；真实匹配场景20个中，原始深度抓取成功率54.42%，UA-Depth为76.65%，碰撞率从88%降至9%。不确定性重加权变体反而降至71.08%，说明权重使用方式会影响收益。真实评估为固定场景少量试验，不能代表所有物体与抓取策略，也不能替代长期在线稳定性评估。

#### 实用指南
复现应严格按零样本划分核对基础模型训练暴露，报告恢复深度指标和下游抓取/碰撞指标；验证不确定性校准而非只看视觉图。模型约349M参数、单图约225.6ms，24 GPU日训练量意味着部署资源需评估。真实系统应在固定夹具与扰动场景都测，尤其检查透明薄片、镜面曲面及不确定区的抓取失败模式。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01645v1。

#### 总结
核心思想：以不确定性指导深度修复

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01645v1)
- [arXiv](https://arxiv.org/abs/2611.01645v1)

---

<a id='2611.01682v1'></a>
## [Leveraging 2D Foundation Priors for Generalized Category Discovery under Test-Time Adaptation in 3D LiDAR Segmentation](https://arxiv.org/abs/2611.01682v1)

**Authors:** Hanqi Chen, Chaoqi Chen, Tian Ji, Hui Huang

**Published:** 2026-11-01

**Categories:** cs.CV, cs.LG

**Abstract:**

In autonomous driving, 3D LiDAR semantic segmentation is expected to adapt to evolving environments and recognize previously unseen categories, making generalized category discovery under test-time adaptation (GCD-TTA) an important problem. However, existing works treat GCD and TTA separately, while the limited semantic information in 3D point clouds makes GCD-TTA under domain shift more prone to feature collapse when combining GCD with TTA. To address this issue, we propose a two-stage framework for GCD-TTA in 3D LiDAR semantic segmentation. Specifically, in the first stage, we adaptively aggregate local 3D features according to geometric density variations and obtain reliable pseudo labels under a global class-wise prototype consistency constraint. In the second stage, we cast category discovery as an optimal transport assignment process for globally consistent grouping of novel categories. Meanwhile, semantic features from 2D vision foundation models are fused with 3D point cloud features to enrich 3D semantic information and promote a more separable feature space. Extensive experiments on Synth4D to SemanticKITTI, SynLiDAR to SemanticKITTI, and Synth4D to nuScenes demonstrate the effectiveness of our framework. Under a unified evaluation protocol, our method achieves the best overall mIoU on 9 out of 10 benchmark splits and improves the overall mIoU by up to 12.21 points over the strongest re-implemented baseline.

### 论文解读

#### 摘要翻译
本文提出在3D点云目标域做类别发现适配，同时借用2D视觉基础模型语义避免新类聚类坍塌。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
自动驾驶环境会随地区、天气和传感器变化，点云分布偏移时还可能遇到训练未见类别。传统域适配假设标签集合固定，类别发现又容易把未知类与已知类混淆。作者假设几何密度自适应、已知类原型约束与2D语义先验可在测试时协同稳定表示。

#### 方法设计详解
第一阶段按局部几何密度自适应聚合点特征，以全局已知类原型一致性筛选伪标签；融合图像侧基础模型语义，提升稀疏点云特征可分性。第二阶段把未见点分类别发现转为最优传输分配，Sinkhorn匹配与过聚类提供全局一致分组；置信门控的2D语义融合后进行双向自训练，交替更新已知/新类别结构。流程在目标域测试数据上适配，需明确每批数据顺序和在线更新预算。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
与传统只适配已知类别的TTA不同，方法同时发现未知类；与单独聚类未知点相比，已知原型和2D语义减少类别混淆。它面向点云域偏移和开放类别，依赖图像-点云关联及预先知道新类别数量；已知类性能可能下降，不能直接视为完全开放世界识别。

#### 实验分析（精简版）
在Synth4D→SemanticKITTI、SynLiDAR→SemanticKITTI、Synth4D→nuScenes等十个迁移划分中，整体mIoU有九个划分优于比较方法，最高提升12.21点mIoU。结果覆盖合成到真实及真实到真实迁移；新类别数量由协议预先给定，且个别已知类可能退化，部署前应单独查看已知类与新类指标。

#### 实用指南
复现需保证源域/目标域拆分、相机投影和类别映射一致，核对目标数据可否按序流式访问，并记录每批适配时间与遗忘。2D基础模型的图像语义可迁移到有配对相机的激光雷达，但夜间、遮挡和投影稀疏会削弱其帮助。上线前需明确未知类数量来源、已知类性能下限和回滚机制；论文的十项split结果不等同持续道路运行安全保证。实验采用batch size=1的在线单遍适配设定，迁移时需核验目标流访问与更新时延。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01682v1。

#### 总结
核心思想：结合几何与语义发现新类

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01682v1)
- [arXiv](https://arxiv.org/abs/2611.01682v1)

---

<a id='2611.01729v1'></a>
## [ODE Policy: Robot Learning via Action Dynamics, Not Actions](https://arxiv.org/abs/2611.01729v1)

**Authors:** Jindou Jia, Jingliang Li, Kuangji Zuo, Tuo An, Yuxuan Hu, Mengfei Zhao, Bofan Lyu, Jiaqi Bai, Daoyuan Zhu, Hengyu Dai, Junpeng Zhang, Jianfei Yang

**Published:** 2026-11-01

**Categories:** cs.RO

**Abstract:**

Most robot policies, whether trained by imitation learning (IL) or reinforcement learning (RL), map observations to action values at discrete control steps and leave the continuity of physical motion to be learned implicitly. We propose ODE policy, which learns the dynamics of the action rather than the action itself. A neural ordinary differential equation (ODE) learns the vector field of the action by matching the integrated actions to the targets over short action chunks in IL and single control steps in RL. A lightweight diversity head captures multi-modal behavior by randomizing the direction of the field. At deployment, each action is integrated from the previous one, so the executed trajectory is continuous by construction. We evaluate it as a visuomotor policy, as the action head of vision-language-action (VLA) and world action models (WAMs), and as the actor of model-based RL against 14 baselines on unimanual, bimanual, dexterous, locomotion, and loco-manipulation tasks. In IL, against flow matching, it reaches 5.9$\times$ the success, 5.4$\times$ the generalization, and 3.9$\times$ the mode balance after one epoch, and generates actions 10.8$\times$ faster. It mostly moves more smoothly than human demonstrations. On 14 real-world tasks across single-arm, dual-arm, dexterous, and humanoid robots, it achieves the highest success rate. In RL, it further speeds up SuperMimic for humanoid motion tracking and smooths its actions. Efficient in learning, data, parameters, and execution, ODE policy offers a promising path to scaling robot learning.

### 论文解读

#### 摘要翻译
本文提出让动作连续性成为策略本身要学习和维持的对象，而不是把离散控制点交给底层执行器事后平滑。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
常见模仿学习或强化学习策略直接回归离散动作；即使动作块彼此不协调，学习目标也未显式惩罚不连续。作者的核心假设是：若策略输出动作随时间变化的向量场，并从前一动作连续积分，运动轨迹就能自然连贯，同时保留数据中的多模态。

#### 方法设计详解
输入观测后，策略先估计动作空间的神经常微分方程向量场；训练时在模仿学习中对短动作块积分、在强化学习中对单步动作积分，以积分终点监督目标动作。轻量多样性头随机化向量场方向，表示多种可行行为。部署时以上一时刻动作作初值进行数值积分，得到新动作，再由控制器执行；同一动作头可用于视觉运动策略、VLA/WAM动作头和模型式强化学习。论文还比较求解步数、积分时间和向量场参数，讨论速度与轨迹平滑的折衷。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
相较直接预测动作值或流匹配生成整段动作，ODE策略把学习目标改成动作变化率，且生成可沿时间连续推进；相较只在执行端滤波，它将连续性纳入可学习表示。创新不等于保证真实动力学安全，向量场平滑仍需结合控制频率和执行器约束。

#### 实验分析（精简版）
作者在单臂、双臂、灵巧手、人形行走及移动操作等模拟任务中与14个基线比较，并报告14项真实机器人任务结果。模仿学习中，文中报告首个训练轮次对比流匹配成功率约5.9倍、泛化约5.4倍、模式均衡约3.9倍，动作生成快约10.8倍；此类比值依赖所选基线和早期训练设置。真实结果覆盖多种平台，但不是统一硬件、统一任务或大样本可靠性证明。

#### 实用指南
复现时先确认动作维度、时间步长、ODE求解器与积分步数，再用相同数据和计算预算对比离散策略及流匹配；同时报告成功率、轨迹平滑、模式覆盖与延迟。论文展示跨策略角色的复用，但不同机械臂、灵巧手和人形平台仍需动作尺度与控制接口适配；不要把连续曲线直接等同安全执行。 复现时应保持数据协议、基线配置和评价指标一致；模仿学习主比较特别报告训练1 epoch后的表现，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01729v1。

#### 总结
核心思想：以动作动力学生成连续轨迹

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01729v1)
- [arXiv](https://arxiv.org/abs/2611.01729v1)

---

<a id='2611.01560v1'></a>
## [Democratizing Robot-Use Agents: A Unified Harness with Open Models](https://arxiv.org/abs/2611.01560v1)

**Authors:** Jiarui Yuan, Yuxin Zuo, Xiaoyu Huang, Wenbo Zhang, Kwanyoung Park, Junlin Yang, Tianxiang Zhang, Ruoqu Chen, Yutao Ouyang, Bingxiang He, Xingtai Lv, Hongjin Wu, Jiajun Liu, Jiahao Zhao, Tianwei Luo, Haozhan Li, Cheng Qian, Jiaxin Ge, Huan-ang Gao, Kaisen Yang, Jiaqi Deng, Qixin Xu, Siqiao Huang, Bowen Zhou, Mengdi Xu, Zhiyuan Liu, Pieter Abbeel, Ning Ding

**Published:** 2026-11-01

**Categories:** cs.RO

**Abstract:**

Robot-use Agents (RUAs) have shown promising zero-shot manipulation capabilities, but reliance on frontier, proprietary models limits research into further training. To democratize RUA research, we introduce uni-robot-agent, a unified harness for competitive open-weight robot use that supports further training. Without privileged inputs, an out-of-the-box Qwen3.8-Flash-Next achieves a RoboDojo Score of 31.39, surpassing the published GPT-6 Astra-RoboProbe result of 28.97 and many VLA/WAM baselines, and performs competitively on 19 real-world tasks. Our analysis across model generations, families, and sizes links effective robot use to four general capabilities, suggesting that future RUA training should develop them jointly.

### 论文解读

#### 摘要翻译
本文提出把机器人代理的工具、环境适配、动作调度和反馈做成可组合接口，让推理能够与安全有序的物理执行重叠。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
语言模型调用机器人时，环境API彼此不同，推理等待会浪费执行时间，而并发动作若不管理共享资源可能相互冲突。核心挑战在于兼顾推理效率与物理动作顺序。模拟器还常提供真实机器人拿不到的深度图或完成判定，导致仿真成绩难迁移。作者以统一工具接口、模块化适配、按资源串行化及现实可用观测为设计原则。

#### 方法设计详解
代理层把模型请求交给编排层，编排层调度工具并维护任务句柄；共享工具通过环境适配器转换到各机器人或模拟器的原生API，形成代理、编排、工具、服务和后端五层。后台调用允许模型继续独立推理，但依赖结果的决策仍等待；占用同一资源的工具按提交顺序执行，互不冲突的观察可并行。两阶段流水线使推理与执行重叠，状态记录和请求去重避免重复动作，必要时清理尚未启动的队列。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
相比每种环境为每个工具单独实现一套接口，共享工具加适配器减少组合复杂度；相比串行的“思考—执行—再思考”，后台任务和资源调度实现安全并行。它是系统/评测框架贡献，不是单个更强策略。需要关注读写状态一致性、取消语义与基线传感器条件，否则流水线速度或模型对比会失真。

#### 实验分析（精简版）
RoboDojo上Qwen3.8 Flash加框架得分31.39，较同表RoboProbe提升2.42分（28.97）；真实评测中Qwen得分49.50，π0.5为47.75。真实打分包含各能力维度部分分数；RoboDojo同一模型配置相较RoboProbe高2.42分。代理接收RGB加深度，而部分基线只有RGB，故结果不能视作严格同输入胜出。部分比较使用有限种子或布局；系统级代理评分也不等同任务全完成率。

#### 实用指南
复现先定义工具参数、单位、状态快照和错误恢复协议，再用同一机器人/传感器配置评估串行与流水线模式；记录取消请求是否已产生物理副作用。将框架迁往新平台通常只需实现适配器，但要重新验证资源互斥和状态一致性。论文对接口设计有参考价值，部署价值取决于真实机器人的故障处理和安全互锁是否到位。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01560v1。

#### 总结
核心思想：模块化工具并行执行

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01560v1)
- [arXiv](https://arxiv.org/abs/2611.01560v1)

---

<a id='2611.01565v1'></a>
## [Beyond Validation Loss: Coverage and Reliability Dynamics in Robot Policy Pretraining](https://arxiv.org/abs/2611.01565v1)

**Authors:** Zhanyi Sun, Max Simchowitz, Shuran Song

**Published:** 2026-11-01

**Categories:** cs.LG, cs.RO

**Abstract:**

In modern robot learning, generative robot policies are first pretrained through supervised learning from demonstrations, then refined through reinforcement learning, supervised finetuning, or other forms of posttraining. Yet conventional pretraining metrics, such as validation loss and average task success, provide an incomplete account of which policies support both effective direct deployment and subsequent posttraining. In this work, we study closed-loop pass@k, which evaluates the probability that k rollouts from the same initial condition attain at least one success. Thus, pass@1 measures mean task success; large-k pass@k measures whether success remains reachable under repeated sampling. Across a wide range of posttraining recipes, we show that large-k pass@k reveals posttraining differences not captured by pass@1. In particular, checkpoints with similar pass@1 but different large-k pass@k have different posttraining performance. Moreover, we show pass@k peaks at different checkpoints during supervised pretraining, revealing a coverage-reliability tradeoff over the course of training. Based on these observations, we propose two practical interventions: (1) a purely offline checkpoint-selection criterion that balances coverage and reliability without closed-loop evaluation, and (2) a generative data augmentation procedure that expands supervision beyond the finite demonstration set, preserving large-k pass@k while maintaining or improving pass@1 and raising the observed coverage-reliability Pareto frontier. Together, our results motivate evaluation, checkpoint selection, and training procedures that explicitly account for posttraining. Blog post, visualizations, and code are available at https://zhanyisun.github.io/robot-policy-learning-dynamics/.

### 论文解读

#### 摘要翻译
本文提出把固定起点下的单次策略可靠性与多次采样可覆盖的成功空间分开评估，并用误差分布指导检查点选择。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
模仿策略常在平均成功率上挑检查点，但同一模型对多次采样的覆盖能力可能不同；固定少数初始状态也掩盖策略“偶尔成功”和“多数都能做”的差别。作者假设pass@k曲线可拆分典型可靠性与最优采样覆盖，并且验证集低噪声去噪误差分位数能反映闭环覆盖的变化。

#### 方法设计详解
先在固定起点和多个随机种子上重复执行策略，记录每个起点的k次采样是否至少一次成功；pass@1描述一次执行可靠性，大k pass@k近似可采样成功覆盖。接着按验证误差分位数而非单一均值选择检查点，观察误差尾部和训练进程对覆盖的提示。JointAug训练联合观测—动作生成器，以联合样本增强典型策略和best-of-k表现；论文还检查DICE-RL及OGPO中的checkpoint选择。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
不同于只报告平均成功率，pass@k明确区分可靠性与可恢复/覆盖潜力；误差分位数补充传统单一loss监控。JointAug不是仅给动作噪声，而是联合建模观测和动作。适用于固定起点可重复评测的行为克隆研究，但大k指标受采样预算、随机种子与起点分布影响，不能代替实际在线可靠性。

#### 实验分析（精简版）
论文在MimicGen多任务、RoboCasa 65个任务及Push-T等设置比较，并报告JointAug能同时改善pass@1与大k覆盖。一个held-out状态上best-of-64动作MSE从0.44×10^-3（ep800）到0.18×10^-3（ep50），说明单次可靠性与多样本覆盖会呈不同趋势。真实Push-T只覆盖20个初始状态；主要证据仍来自模拟器固定起始配置，不能泛化到任意部署分布。

#### 实用指南
实现需固定起点、明确每个状态采样次数k，完整报告pass@1与多k曲线，避免只展示best-of-k；验证集应与训练起点隔离。检查点筛选要保留误差分位数并核对其与闭环成功关系。JointAug适用于可生成成对观测动作数据的策略训练；迁移到真实机器人前需在新场景分布和实际采样预算下重新测覆盖，不应以多次重试掩盖单次失败风险。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01565v1。

#### 总结
核心思想：分开衡量策略可靠与覆盖

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01565v1)
- [arXiv](https://arxiv.org/abs/2611.01565v1)

---

<a id='2611.01600v1'></a>
## [SCOUT: Planning under Occlusion via Object-Centric World Model Rollouts](https://arxiv.org/abs/2611.01600v1)

**Authors:** Amir Hossain Raj, Xuesu Xiao

**Published:** 2026-11-01

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Robots manipulating in cluttered spaces must often plan from partial views where foreground objects occlude the target, surrounding geometry, and usable free space. Reactive policies often miss these hidden constraints, while exhaustive search becomes computationally prohibitive in dense clutter. We present SCOUT, an occlusion-aware planning framework that bridges high-level deliberation with grounded physical rollouts. SCOUT leverages an object-centric scene representation to reconstruct hidden geometry and derive a relational planning state capturing blocker hierarchies and accessibility corridors. To navigate the vast action space in clutter, an LLM proposes structured action candidates which are then rigorously evaluated via Monte Carlo Tree Search (MCTS) using a learned action-conditioned world model. Evaluations on 'Reveal' and 'Placement' tasks show that SCOUT outperforms Vision-Language-Action (VLA) policies, embodied agents, and 3D world models, with significant gains in long-horizon tasks requiring reasoning over occluded geometry. We further validate the framework's robustness through real-world robotic manipulation in a cluttered shelf environment.

### 论文解读

#### 摘要翻译
本文提出先从多视角RGB-D重建可见与隐藏物体关系，再用世界模型预测和搜索评估长程整理动作。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
遮挡下仅靠当前像素反应容易漏掉目标后方障碍和可达通道；穷举长序列动作又会带来庞大搜索成本。作者假设显式恢复对象级隐藏几何和阻挡关系，可以缩小规划状态，再用语言模型提议动作、由学习到的世界模型筛选，而非直接相信语言模型计划。

#### 方法设计详解
GHOST将多视角RGB-D、相机位姿和SAM3D对象先验融合成对象槽；可见槽描述观测目标，固定数量隐藏槽作为潜在对象查询，经slot transformer交互后生成占据体、槽状态及关系图。关系规划状态编码目标、遮挡者和可达性。LLM产生结构化候选动作，动作条件世界模型预测各对象位移、是否移动及物理有效性；MCTS用预测状态滚动评估候选，选择短动作序列执行后重新规划。每次动作只执行有限原语，搜索深度受12步预算约束；规划推理每回合最多搜索T=12个动作原语。输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
其创新贡献是把对象级遮挡关系、动作条件世界模型和MCTS整合为分层规划流程。相较纯VLA按图像直接输出动作，SCOUT显式推理阻挡层级；相较只做世界模型预测，它以搜索检验候选序列；语言模型负责提议而不是最终裁决。依赖对象槽、几何先验和状态预测准确性，隐藏物体重建错误会级联到规划。

#### 实验分析（精简版）
仿真货架任务500次rollout、3个随机种子，初始视图协议下域内平均成功率80.8%，最强基线72.8%；新物体75.4%对67.6%，新视角76.8%对69.3%。真实机器人20次实验成功12次（60%）。仿真显示收益，但真实样本少且固定初始观察；作者失败分解包含隐藏场景重建、转移预测、几何可行性和搜索误差。

#### 实用指南
复现应统一原语执行器、动作预算、初始相机视角与模拟器成功判据，并分别测槽重建、转移预测和规划误差。真实迁移需要重新标定深度、物体先验和机械臂原语；若能执行后重新观察，闭环可能减少初始视图错误，但会改变实验协议和成本。适合研究杂乱场景下高层规划，不能替代低层碰撞检测。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01600v1。

#### 总结
核心思想：显式建模遮挡后再规划

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01600v1)
- [arXiv](https://arxiv.org/abs/2611.01600v1)

---

<a id='2611.01844v1'></a>
## [OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining](https://arxiv.org/abs/2611.01844v1)

**Authors:** Dujun Nie, Ruiqi Song, Zixuan Wang, Hanqing Wang, Yifan Han, Baiyong Ding, Wenzhao Zheng, Hangbin Wu, Long Chen

**Published:** 2026-11-01

**Categories:** cs.RO, cs.AI

**Abstract:**

Navigation world models predict future observations conditioned on navigation actions to support planning, but their generalization across real-world environments remains constrained by the limited coverage of action-labeled data. Action-free navigation videos provide diverse visual experience, yet lack the recorded actions required for action-conditioned training. We introduce OpenNWM, a generalist navigation world model with latent action pretraining. It learns universal latent actions from large-scale action-free videos through visual prediction pretraining, and further aligns them with physical motions via action-conditioned post-training. To support video pretraining, we curate NavAnywhere, a large-scale action-free navigation video dataset comprising 17.5 million visual observations from 15 heterogeneous sources. It captures navigation by ground robots, humans, and drones, with diverse viewpoints and motion patterns across five broad scene types: residential areas, public and commercial spaces, urban settings, parks, and natural landscapes. Experiments demonstrate improved visual prediction on in-domain and out-of-domain benchmarks, together with improved navigation performance. These results show that latent-action pretraining enables diverse, action-free videos to improve dynamics learning and generalization for open-world navigation.

### 论文解读

#### 摘要翻译
本文提出先从没有动作标签的视频中学会视觉变化所对应的潜在动作，再把潜在动作对齐到真实导航控制。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
导航世界模型需要动作条件转移，但公开视频量大且通常没有机器人动作标签；仅用少量带动作数据训练，场景和运动模式覆盖不足。作者假设视频预测可提供通用运动表征，随后再以物理动作监督对齐，不必要求所有预训练视频具备机器人控制标注。

#### 方法设计详解
第一阶段收集NavAnywhere多源导航视频并学习潜在动作编码：视频帧进入视觉预测模型，潜在动作作为帧间运动条件，以未来观测预测目标约束其携带运动信息。第二阶段使用带控制动作的导航数据，将潜在动作与机器人的实际运动对应起来并训练动作条件世界模型。规划时模型据动作预测未来观测/状态，供导航策略评估候选运动。数据包含地面机器人、人类与无人机等不同视角和五类场景；多源规模带来覆盖，也引入来源分布差异。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
相比只在动作标注轨迹上训练，潜在动作预训练吸收无标签视频；相比纯视频预测，它再用真实控制动作做物理对齐，避免把视觉变化直接当可执行命令。贡献主要是数据和两阶段训练范式的组合，适用场景是导航世界模型预训练；不能仅凭视觉预测改善推断任意机器人具备可迁移控制。

#### 实验分析（精简版）
NavAnywhere训练集含17.5 million条视觉观测、15个异质来源；论文在域内及域外视觉预测基准和导航任务评估，报告潜在动作预训练提高预测与导航表现。数据规模达到17.5 million条视觉观测，说明覆盖广但不直接代表同等数量的独立场景或无偏分布；训练源由Ego4D等视频构成且来源比例不均，跨平台迁移仍受相机、速度和运动主体差异影响。

#### 实用指南
NavAnywhere预训练集含17.5 million条视觉观测。复现需核实视频去重、来源切分及测试环境泄漏，保持预训练与目标导航数据严格分离；分别报告预测质量和闭环导航成功，不能用前者替代后者。迁移到新底盘时要重新校准潜在动作到控制量的对齐阶段，检查视角、步幅、转向和动态约束；论文的数据资源有助于研究预训练，但不意味着所有数据已具备一致授权或可直接下发机器人。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01844v1。

#### 总结
核心思想：无标签视频预训练导航动力学

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01844v1)
- [arXiv](https://arxiv.org/abs/2611.01844v1)

---

<a id='2611.01483v1'></a>
## [$TF^2$-Steering: Action Refinement with Tactile Forecasts and Feedback via Velocity-Field Steering](https://arxiv.org/abs/2611.01483v1)

**Authors:** Yiming Chen, Xiaofei Wei, Chun Gu, Wei Xiao, Peidong Liu, Li Zhang

**Published:** 2026-11-01

**Categories:** cs.RO

**Abstract:**

Tactile sensing plays a crucial role in everyday human interaction. Consider inserting a plug into a tight socket, vision alone provides limited guidance during the final insertion stage, and we instead rely on both anticipated contact and realized tactile feedback to ajust the insertion motion. The same capability is equally important for robots, particularly in dexterous manipulation tasks. Yet many contemporary robot policies remain vision-dominant. This visual bias, compounded by limited tactile data and sparse contact events, can cause directly incorporated tactile signals to be ignored or even compromise the policy's pretrained capabilities, yielding degrading performance as observed in our experiments. We present $TF^2$-Steering, which refines actions via learned residual steering of the flow-matching action expert's velocity field, using a tactile mixture-of-experts architecture to integrate both tactile forecasts and real-time tactile feedback. Together with specific designed three stages training recipe, $TF^2$-Steering enables tactile signals to correct and, when necessary, redirect the flow-matching action-generation process, while building on the strong action-generation capacity of the vision-based foundation policy. Extensive experiments across six simulated and six real-world manipulation tasks demonstrate the effectiveness of $TF^2$-Steering. It outperforms the strongest evaluated baselines by 10.7 percentage points in average simulation success rate and 9.1 percentage points in average real-world progress success rate. Project page: https://yiming-cc.github.io/tf2-steering/.

### 论文解读

#### 摘要翻译
本文提出用接触的预测和真实触觉共同修正动作生成器的速度场，让触觉能够改变计划而非只做末端补偿。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
视觉主导的基础策略在插入、扭转等接触任务中缺少力学状态；把触觉图像直接拼入输入可能被忽略，或扰乱预训练视觉能力。另一个痛点是只用已发生的接触反馈来修正已生成动作，难以在动作生成前预判将要遇到的接触。作者假设预期触觉与实时反馈应共同作用于同一动作修正过程。

#### 方法设计详解
以预训练视觉动作专家生成流匹配速度场，原视觉专家保持独立；触觉专家读取触觉历史和预测触觉，输出残差速度修正，可在生成过程中校正或重定向动作。触觉世界模型依据可用上下文预测未来接触状态，动作块开始前生成预测并在块执行期间保持缓存；执行时高频触觉反馈不断进入专家。训练分三阶段，逐步学习触觉预测与速度场残差，以免少量触觉数据覆盖原策略能力。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
与直接拼接触觉输入相比，残差式速度场修正保留预训练动作专家；与仅在推理时反馈修正相比，额外的触觉预测提供动作块级前瞻。贡献是预测和反馈共用流场 steering，适合接触丰富的精细操作；需要配套触觉传感器、数据和特定训练流程，通用性不能从有限任务直接外推。

#### 实验分析（精简版）
UniVTAC六项模拟任务平均成功率68.3%，高于该实验表中FTP-1的57.7%；六项真实任务每项收集100条示范并进行20次试验，平均进度成功率71.5%，高于T-Rex的62.4%；六项仿真任务平均成功率为68.3%。真实指标允许部分完成，不等同全任务完成率；测试用双DoF UR5e机械臂及22DoF Sharpa手，任务和硬件范围仍有限。

#### 实用指南
复现须准备视觉、预测触觉与实时触觉的时间同步数据，真实任务按每项100条示范进行任务微调，按三阶段训练，并单独消融只反馈、只预测、直接输入触觉等方案。真实部署要先验证传感器漂移、接触频率和动作安全边界，再测量阶段性进度与完整完成率。向其他机械手迁移时，触觉阵列形态和接触动力学都需重新适配；论文项目页公开，但实际代码、数据发布状态应另行核实。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01483v1。

#### 总结
核心思想：触觉预测反馈共同引导动作

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01483v1)
- [arXiv](https://arxiv.org/abs/2611.01483v1)

---

<a id='2611.01396v1'></a>
## [SpecUltra: A Unified Foundation Model for Heterogeneous Spectral Images](https://arxiv.org/abs/2611.01396v1)

**Authors:** Weili Kong, Baisen Liu, Xiaojun Bi, Jiaming Pei

**Published:** 2026-11-01

**Categories:** cs.CV

**Abstract:**

Spectral imagery, including hyperspectral image (HSI) and multispectral image (MSI), exhibits substantial heterogeneity in band numbers, wavelengths, spectral resolutions, hindering unified modeling and cross-sensor transfer. From the perspective of imaging physics, these heterogeneous observations can be regarded as discrete measurements of an underlying continuous spectrum acquired through different sampling processes. Motivated by this perspective, we propose SpecUltra, a unified foundation model for heterogeneous spectral images. Specifically, SpecUltra employs 3D convolutional tokenization and wavelength-aware 3D RoPE to encode relative wavelength and spatial relationships, enabling fine-grained spatial-spectral modeling within a shared coordinate system. We further construct MultiHyper-200K, a large-scale multi-sensor HSI dataset, and pretrain SpecUltra using auxiliary-token-assisted mask image modeling (MIM), where a level token indicates the radiometric product type and register tokens absorb sensor- and acquisition-dependent variations. To bridge the spectral-granularity gap between HSI and MSI, we introduce Spectral Kernel Transformation (SKT), which transfers pretrained knowledge in parameter space by constructing MSI embedding kernels from fine-grained HSI-pretrained kernel slices. This design preserves pretrained spectral priors while maintaining compatibility with the pretrained backbone. Experiments on HSI and MSI classification and segmentation benchmarks demonstrate state-of-the-art or competitive performance across disparate sensors and downstream tasks.

### 论文解读

#### 摘要翻译
本文提出把高光谱立方体的波段邻接结构显式编码进视觉Transformer，并把已有多光谱卷积核映射到高光谱输入。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
高光谱数据同时有空间和波长维度，直接把波段当普通通道或重复单波段位置编码，难以保留波段间隔与局部谱形；现有多光谱预训练权重也不能直接用于高光谱。作者假设波长感知的位置编码和谱核迁移可同时改善结构表达与初始化。

#### 方法设计详解
输入多波段影像后，空间切块与光谱维组合为三维token；位置编码由空间位置和实际波长共同构成3D RoPE，使不同传感器波段关系进入注意力。迁移模块将多光谱模型卷积核沿高光谱波段切片并对齐，生成光谱核token；模型通过层级token/寄存器组织不同谱域信息，再输出分类、分割或回归任务特征。作者建立MultiHyper-200K，整合六类传感器来源的200,498幅影像用于预训练。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
相较将HSI简单压为RGB或普通多通道，方法保留波长维次序与距离；相较从头训练，以多光谱卷积核切片提供可迁移初始化。主要贡献是统一光谱结构表示和预训练迁移，不意味着真实辐射标定被模型自动解决。适用于高光谱分类、分割和场景理解，跨传感器部署仍须检查波段响应与空间分辨率。

#### 实验分析（精简版）
论文在Qingpu、SegMunich、HRSSC和EuroSAT等四类下游基准/任务上比较，多数报告结果领先或具竞争力；采用五次运行报告波动。Qingpu微调划分按空间区域分离，降低邻近像素泄漏风险。基准数量和传感器类型仍有限，部分辐射产品经过经验尺度处理，不宜把结果推广为所有波段采样方案均有效。

#### 实用指南
训练集规模为200,498幅MultiHyper-200K影像。复现需保留传感器实际波长、统一空间尺度，并按空间块而非随机像素拆分训练测试；检查每种传感器可用波段映射和波长缺测处理。论文描述MultiHyper-200K及模型设计，但代码和模型是否已公开需以项目渠道核实。向新传感器迁移可沿用波长编码思路，仍要重新验证谱响应、辐射单位、预训练来源及下游标签定义。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01396v1。

#### 总结
核心思想：以波长结构编码高光谱

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01396v1)
- [arXiv](https://arxiv.org/abs/2611.01396v1)

---

<a id='2611.01390v1'></a>
## [Same Path, Less Time: Training-Free Fast Execution of Vision-Language-Action Models under Inference Latency](https://arxiv.org/abs/2611.01390v1)

**Authors:** Fuyi Chen

**Published:** 2026-11-01

**Categories:** cs.RO, cs.LG

**Abstract:**

Vision-language-action (VLA) models predict chunks of future actions, yet deployed robots waste time in two ways that are independent of model capability: they halt while the next chunk is inferred, and they move at the speed of slow, cautious demonstrations. Existing remedies remove one loss each, and stacking them fails, because asynchrony and speed-up each change the correspondence between planned progress and executed motion. We present Path-preserving Time Scaling (PTS), a training-free execution layer for frozen action-chunking policies built on one principle: the robot's motion is a time reparameterization of a reference path, and the path changes only in response to new observations. A path layer decodes every replan deterministically in one step and enters it at the progress consumed during inference. A time layer sums the planned increments, keeps the arm at 1x while the gripper moves, and caps the rate by the actuator's linear range and by chunk feasibility under latency. We show that misaligned entry and stochastic sampling add a bias and a variance to the continuation error, and derive a break-even success rate that states when speeding up pays off. On LIBERO with two frozen VLAs under injected inference latency, PTS raises task throughput over the strongest deployable baseline by 33% for pi0.5 and 104% for SmolVLA while matching or improving the success rate, is best on all six model-suite combinations, and exceeds zero-latency 1x execution. Against asynchronous 1x execution with the same one-step decoder, the time layer alone adds 28% and 38%. On LIBERO-Dynamic it improves on stop-and-wait execution by 38% and is comparable to asynchronous 1x execution.

### 论文解读

#### 摘要翻译
本文提出让异步推理期间机器人继续沿原路径前进，并把“走哪条路”与“走多快”分开控制。作者围绕这一目标构建方法，并在论文所述基准与任务中评估效果。

#### 方法动机分析
VLA输出未来动作块，但等待下一次推理会让机器人停顿；单纯加快动作又会改变路径并破坏重规划衔接。作者的假设是，把执行轨迹视为参考路径的时间重参数化，就能让新计划只在观测更新时改变路径，速度变化则单独受硬件与动作块可行性约束。

#### 方法设计详解
策略给出动作块后，路径层用确定性的一步解码将其转为参考路径；推理进行期间，机器人沿当前路径继续执行，并记录已经走过的进度。新动作块返回后，系统从与已消耗进度对齐的位置接入，避免重头播放或随机采样造成偏差。时间层累计动作增量，在夹爪运动时保持机械臂1倍速度，并根据执行器线性范围和动作块可行性限制加速倍率。作者进一步推导误差偏差、方差及盈亏平衡成功率，用于判断加速是否值得。 输入、表示、关键模块与输出形成一条完整流程；其中各模块分别承担表征、约束或决策职责，不能把最终提升简单归因于单一组件。

#### 方法对比分析
与停止等待相比，PTS让物理执行和策略推理重叠；与异步但1倍速方法相比，时间层再提高吞吐。它不改策略权重，适合冻结的动作块VLA与有明显推理延迟的机械臂。其路径连续性依赖动作块解码、关节插值和执行器边界符合设定，不能自动解决策略本身预测错误。

#### 实验分析（精简版）
在LIBERO两种冻结VLA及注入推理延迟条件下，PTS较最强可部署基线吞吐分别提高33%（π0.5）和104%（SmolVLA），成功率保持或改善；六个模型/套件组合均最佳。LIBERO-Dynamic相较停止等待执行提高38%。实验为模拟器评估，没有真实机器人结果，因此真实控制时延与机械约束下的收益仍需验证。

#### 实用指南
实现前测量推理时延和控制周期，按关节速度范围、夹爪并行动作及动作块可行性设置时间倍率；复现应同步记录吞吐、成功率和路径偏差，并比较停止等待、异步1倍速及完整PTS。该层无需再训练，但迁移到真实机器人必须验证轨迹插值、重规划进度定义及急停逻辑；吞吐提升不应以碰撞风险或任务完成率下降换取。 复现时应保持数据协议、基线配置和评价指标一致，并将论文直接证据与跨设备应用推断分开。

链接：https://arxiv.org/abs/2611.01390v1。

#### 总结
核心思想：路径不变而执行提速

1. 明确输入与目标场景。
2. 构造专用表示并执行关键模块。
3. 用匹配基线和任务指标验证输出。
4. 检查失败条件后再迁移部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.01390v1)
- [arXiv](https://arxiv.org/abs/2611.01390v1)

---

