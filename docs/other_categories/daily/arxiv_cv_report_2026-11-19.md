time: 20261119

# Arxiv Computer Vision Papers - 2026-11-19

## Executive Summary

## 2026-11-19 计算机视觉论文执行摘要

今日收录的 10 篇论文呈现出一个清晰趋势：视觉研究正从单次识别走向可供行动与规划使用的结构化预测。几何与运动重建、具身多机器人协作、世界模型校正和生成式动作策略是主要方向；与此同时，像素级推理和跨设备 3D Gaussian Splatting 展示了模型形态与系统部署上的新探索。多篇工作不只报告离线指标，也评估长时序、闭环执行、推理成本或真实机器人表现，但真实部署证据的规模仍普遍有限。

**值得重点关注的工作**包括 π⁴、PhyVGGT 与 JointGV：π⁴把稠密几何、相机、尺度和运动统一到单次前向 4D 感知中，并在长路线评估中优于所比方法；PhyVGGT借助静止基座几何，把单目关节估计转化为有限候选轴推断，在所述基准上超过使用 RGB-D 和真实位姿的比较方法；JointGV则以场景图动态校正视觉世界模型，在仿真和真实倒立摆记录上都降低了预测误差，并改善了仿真控制结果。三者共同指向一种值得追踪的路线：用显式几何或结构约束，提高视觉预测对长时规划的可用性。

机器人研究的另一条主线是协作与执行可靠性。Heave-Ho将多 humanoid 的物理协作、离散技能规划和语音同步纳入统一基准，在仿真任务上显著超过所比基线，并展示了有限的实机搬运结果；RoboTeam系统评测多机械臂视觉控制，揭示成功率、耗时、token 成本和视觉定位精度之间的权衡。关于真实机器人数据质量的研究则指出，示范中的盲区处理、绕行和控制并行化会直接影响学到的行为。整体看，评估重点正在扩展到协作、失败恢复、数据采集过程和资源成本，而不只是单任务成功率。

生成式方法也在从感知走向动作与规划。APM通过让教师目标锚定在学生实际到达的去噪状态，改善少步双臂策略蒸馏，并报告了真实平台上的加速结果；FIRE-Drive联合改进扩散规划的候选初始化、风险评分与次要模式监督，不过其基准提升有限，闭环比较条件也不完全一致。Pixel is All You Need 展示纯像素模型在受控合成任务中通过逐步推理大幅超过直接作答，但开放视觉理解和真实控制尚未得到验证。Cross3DGS则以显式硬件约束和数值校验支持工作站到移动设备的统一训练，提示可移植视觉计算仍是重要系统方向。

**建议优先阅读全文：**关注 4D 几何与机器人感知的读者可先读 π⁴ 和 PhyVGGT；研究世界模型或长时规划的读者可读 JointGV；关注真实多机器人协作可读 Heave-Ho，并结合 RoboTeam 理解评测与成本问题；研究生成式策略可读 APM 与 FIRE-Drive。若关注数据工程和部署，可读机器人数据质量研究及 Cross3DGS。解读这些结果时，应留意各论文自己指出的边界：不少实机实验样本有限，若干结论来自仿真或受控合成任务，跨场景与真实环境泛化仍需进一步验证。

---

## Table of Contents

1. [$π^4$: A Foundation Model for 4D Scene Geometry and Motion](#2611.23144v1)
2. [RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time](#2611.22138v1)
3. [PhyVGGT: Feed-Forward Articulated Object Reconstruction from Monocular Videos](#2611.22244v1)
4. [Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration](#2611.22275v1)
5. [FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning](#2611.22297v1)
6. [Hidden Variables Behind Real-Robot Data Quality: Lessons from Hundreds of Thousands of Hours](#2611.22388v1)
7. [Pixel is All You Need](#2611.22820v1)
8. [Cross3DGS: Unified Vulkan-Based 3D Gaussian Splatting Training from Workstations to Mobile Devices](#2611.22240v1)
9. [JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling](#2611.22543v1)
10. [Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching](#2611.22592v1)

---

## Papers

<a id='2611.23144v1'></a>
## [$π^4$: A Foundation Model for 4D Scene Geometry and Motion](https://arxiv.org/abs/2611.23144v1)

**Authors:** Yifan Wang, Weiguang Zhao, Ni Yang, Haoyu Guo, Junting Dong, Weinan Zhang, Chunhua Shen

**Published:** 2026-11-17

**Categories:** cs.CV

**Abstract:**

$π^4$ is a unified foundation model for 4D scene perception. From diverse, uncalibrated RGB observations in the wild---a single image, an unordered collection, or a video sequence---it recovers dense geometry, camera poses, and metric scale, together with 2D/3D point tracks and moving-object masks for multi-image inputs. It couples a geometry branch with a geometry-conditioned motion branch, which predicts image-space correspondences that are lifted into 3D using reconstructed target-view geometry. Across diverse benchmarks, $π^4$ achieves the best overall average rank in geometry evaluation, alongside competitive performance in single- and multiview 2D/3D point tracking and moving-object segmentation. With a shared backbone, the full geometry-and-motion system contains approximately 0.5B parameters---less than half the size of most baselines. It produces joint reconstructions and tracks in one forward pass, including large-scale scenes with thousands of input images despite training with contexts of at most 64 views. We further present the data and training recipe underlying these capabilities, together with empirical studies of key design choices. Code and models are available at https://github.com/yyfz/Pi4.

### 论文解读

#### 摘要翻译
π⁴从单张图、无序图像集或视频恢复稠密几何、相机位姿和度量尺度，多图输入还可预测2D/3D轨迹与移动物体掩码。几何分支重建场景，运动分支预测图像对应，再借目标视图几何提升为3D。完整系统约5亿参数，可一次前向处理数千张图像。

#### 方法动机分析
传统重建、相机估计和点跟踪常由不同模型完成，误差容易相互影响，这是统一建模的动机。直接回归三维轨迹也是痛点，因为要同时估计目标点、相机位姿和尺度。作者的核心假设是先建立场景几何，再预测2D对应，最后从目标图像几何取出3D位置。

#### 方法设计详解
输入→输出流程：未经标定、可乱序的RGB图像经共享DINOv3编码器提取特征，交给几何和运动两个循环Transformer。二者交替执行帧内及跨视图注意力，并共享循环步权重。几何分支不设参考帧，预测局部点图、相机位姿、置信度和共享度量尺度；射线乘深度形成点图，再由稀疏3D U-Net细化。位姿和尺度把局部点变换到度量坐标：X=m(RP+t)。运动分支用空间自适应归一化读取几何特征，预测查询像素到目标视图的2D对应和可见性，再在目标点图采样得到3D轨迹，因此可见端点位于预测表面。掩码头结合冻结SAM 2特征预测运动区域。训练先完成几何，再冻结几何训练运动；最终几何阶段使用64张A800 GPU，图像长边512像素，每GPU最多64视图。

#### 方法对比分析
相较直接回归三维端点或独立跟踪深度，π⁴先预测二维对应再用场景几何提升坐标，降低运动头负担；同时支持无序多视图并提供重建、相机和运动掩码。这一创新适合几何已能可靠恢复、而跨视图轨迹需要与表面一致的场景。轨迹仍会继承目标视图几何错误。

#### 实验分析（精简版）
16个几何设置的综合平均名次为3.00，略胜Pi3X的3.10。Oxford Spires六条361–800帧路线单次处理时，ATE为0.7337米、Chamfer距离为1.1573米，比DVLT的1.3885米和1.5238米分别低47.2%与24.1%。运动分割均值J&F为85.04，列第二。尺度误差均值20.11%，好于MapAnything但逊于Pi3X。弱纹理背景下模型仍会混淆相机与物体运动，超长序列全局一致性也不足。

#### 实用指南
论文提供代码和模型，训练数据混合公开集、合成场景、视频伪标签与GameRealm；几何约30万、运动约10万更新，训练最多64视图。复现时应留意训练上下文与数千图测试规模不同。机器人或自动驾驶迁移需验证绝对尺度、远距离位姿和长时漂移；论文没有闭环部署实验。

#### 总结
核心思想：用重建几何提升二维运动。
1. 输入RGB视图，恢复点图、相机和尺度。
2. 几何条件化运动分支预测二维对应。
3. 在目标点图采样，得到三维轨迹和动态掩码。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.23144v1)
- [arXiv](https://arxiv.org/abs/2611.23144v1)

---

<a id='2611.22138v1'></a>
## [RoboTeam: Benchmarking Vision-Based Multi-Agent Control of Many Robot Arms in Real Time](https://arxiv.org/abs/2611.22138v1)

**Authors:** Changxuan Fan, Zhongwei Xie, Boyi Zhang, Shaoli Hu, Chang Xu, Xiyu Ren, Yufei Li, Qiao Xiao, Huihao Jing, Kawai Chung, Wenbin Hu, Hongyu Luo, Lingyun Xie, Yisen Gao, Chunyang Li, Haoyu Huang, Wuganjing Song, Yuxuan Liu, Chiu Yeung Tang, Fangneng Zhan, Xi Yang

**Published:** 2026-11-17

**Categories:** cs.RO, cs.AI

**Abstract:**

Multimodal agents can now control robot arms by calling tools and writing code. Existing benchmarks fall short in three ways: most test one or two arms on short tasks, many pause the world while the agent thinks, and many give it ground-truth object positions. We present RoboTeam, a simulated benchmark where agents control 4 to 16 robot arms in real time from camera images alone. Its 36 tasks are long jobs that the arms must do together, such as passing timbers or building a picture from hundreds of cubes. We start from the most flexible agent system, the adaptive system, where one agent decides when to fork delegates. With it, gpt-6.1-sol and opus-5.5 complete 28 and 27 tasks, and four other models at most 3. But at a high price: a typical task takes gpt-6.1-sol 3.5 hours and 218 million input tokens, and a third of its steps fail at least once and must be retried. We then compare it with three fixed systems: single-agent, hierarchical, and decentralized. All four complete a similar number of tasks but differ in time and tokens. The hierarchical and decentralized systems are faster but use up to 1.9 times the tokens, and the single agent uses fewer tokens but takes 1.6 times as long, so the adaptive system lies between them. Even so, no model or system is yet fast, cheap, and reliable at once. RoboTeam measures these limits where earlier benchmarks cannot: long multi-arm jobs, camera images only, and a world that never pauses.

### 论文解读
#### 摘要翻译
多模态代理可以调用工具和编写代码控制机械臂，但现有基准常只测少量机械臂和短任务，有的思考时暂停仿真，有的提供物体坐标。RoboTeam在持续运行的仿真中，用相机图像控制4至16条机械臂完成36项长时协作任务。gpt-6.1-sol与opus-5.5分别完成28和27项，其他四个模型最多3项；代价是任务耗时数小时、输入token巨大且步骤常需重试。不同代理组织方式在耗时与token间折衷，尚无系统能兼顾速度、成本和可靠性。

#### 方法动机分析
真实机器人不会因模型思考而停摆，多臂必须共享空间并相互配合。痛点是现有测试暂停世界或提供真值坐标，掩盖等待与感知误差。核心假设：慢速代理仍须在不停摆世界仅凭视觉协调长任务，因此评测应记录失败和成本，而非只看成功。

#### 方法设计详解
基准包含运输部件、拼方块图案、传递长木料、协作转动密码旋钮四类任务；每类有三档难度与三种团队规模，共36项。MuJoCo以2毫秒物理步长运行，代理只能获取多相机RGB图像和机械臂状态，不能读取物体坐标、深度或接触信息。代理经受限工具移动机械臂、操作夹爪并检查动作；代码工具可用于图像测量，沙箱隔离模拟器和隐藏状态。实验使用六个模型、两个代理框架，比较单代理、层级、去中心化和按需分叉工作者的自适应系统。评价任务完成、步骤与抓取失败、时间、token和消息；模型使用30万token上下文，另测试128×80低分辨率。

#### 方法对比分析
不同于短时单臂或提供真值状态的基准，RoboTeam同时测协作、视觉定位和长时成本；不同于高频VLA策略，它关注慢速推理代理在世界持续变化时能否完成复杂工作。自适应架构可自行决定何时并行。本文贡献是评测基准，不是控制算法。

#### 实验分析（精简版）
自适应设置下gpt-6.1-sol完成28/36项，opus-5.5完成27项，其他四个模型最多3项。gpt-6.1-sol单项中位耗时3.48小时、总输入27.01B token，步骤失败率中位数34%。层级和去中心化更快，却分别多用约1.9和1.7倍token；单代理约慢1.6倍。真值坐标Oracle完成36项，移除代码工具或降低分辨率后仅完成13或14项，说明图像定位是瓶颈。多数条件只运行一次，28与29项的差别不可靠；结果仅来自仿真。

#### 实用指南
作者计划公开场景、运行时、工具、评测脚本与运行记录。复现需对齐MuJoCo、相机画质、机械臂规模、实时条件和预算。失败率、等待时间与通信指标可用于装配或多机器人评测；实机误差、安全与损坏风险仍待验证。

#### 总结
核心思想：实时评测多臂视觉协作

1. 构造四类多臂长任务。
2. 仅给图像和机械臂状态，调用工具操作。
3. 比较代理架构，并统计完成、失败与成本。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22138v1)
- [arXiv](https://arxiv.org/abs/2611.22138v1)

---

<a id='2611.22244v1'></a>
## [PhyVGGT: Feed-Forward Articulated Object Reconstruction from Monocular Videos](https://arxiv.org/abs/2611.22244v1)

**Authors:** Qi Shao, Zhihao Cao, Shuhao Zhai, Anh Nguyen, Hong Liang, Baoru Huang, Liang Li

**Published:** 2026-11-17

**Categories:** cs.CV

**Abstract:**

Reconstructing an articulated object with its 3D movable parts, joint parameters and unseen articulation states from an uncalibrated monocular video is essential for robot manipulation and interactive digital twins. Existing per-scene optimisation methods require minutes to hours and external depth or camera poses, whereas feed-forward 3D foundation models reconstruct dense point maps in one pass without segmenting movable parts or inferring joints. Even on these monocular point maps, regressing or fitting continuous joints directly fails on casual videos, since cross-frame attention blurs moving part boundaries in deep layers, underestimates motion along the viewing ray and leaves continuous axes ill-conditioned under small opening angles. To address these challenges, we propose PhyVGGT, a single-pass framework on a frozen 3D foundation model that anchors moving-part matching and closed-form joint estimation to the undistorted geometry of the stationary base. First, our Part and Motion Decoder fuses intermediate per-frame tokens with global tokens and references part queries to a static-base query to segment 3D parts, align point maps on the static base and predict learned Bingham priors. Second, our Kinematic Inference Layer fits an orthogonal Manhattan frame to static surface normals, selects the joint axis in closed form under a Bingham posterior combining pairwise motion scatter with the learned priors, and solves a linear system for the hinge pivot to drive unseen states. Trained solely on synthetic data, PhyVGGT achieves state-of-the-art monocular performance across five benchmarks, outperforming FAMOS and monocular ArtiPoint, surpassing posed RGB-D ArtiPoint in median error, and running in 6.8 s per 48-frame video against hours of per-scene optimisation in Articulat3D. Project page: https://phyvggt.github.io.

### 论文解读
#### 摘要翻译
PhyVGGT从未标定单目视频一次前向恢复三维可动部件、关节参数和未见姿态。逐场景优化依赖深度和相机位姿且耗时；通用几何模型会模糊运动边界、低估视线方向位移。方法以静止基座为锚，仅用合成数据训练。

#### 方法动机分析
VGGT提供几何与跟踪，却没有显式建模部件运动；跨帧融合模糊边界，深度误差及小开合角使连续轴估计不稳定。论文假设家具、电器的静止基座较可靠，平面法向可形成正交候选轴。这样可把受噪连续回归改写为有限候选比较，同时保留真实帧间运动来判定哪个方向确实在动。

#### 方法设计详解
输入未标定RGB视频。冻结VGGT提取多层逐帧/全局token、点图和跟踪特征，DINOv2补充图像patch特征。Part and Motion Decoder维护背景、基座和活动部件查询：逐帧注意力定位边界，跨帧注意力汇集运动；活动部件查询显式参考基座查询，掩码约束特征匹配，得到三维分割和跨帧对应点。

先用静止基座对应点做加权Procrustes对齐，抵消手持相机漂移，再从匹配部件估计帧间旋转和平移。Kinematic Inference Layer从基座静态区域的法向以四阶对齐求正交曼哈顿框架，将运动散布矩阵与网络预测的Bingham先验结合，对三条候选轴做softmax选轴。关节类型根据网络分类和运动残差判断；铰链中心由带边界先验的3×3正定线性系统求解。最后恢复逐帧关节状态，并用正向运动学生成未见姿态。训练使用冻结VGGT-1B与PartNet-Mobility合成视频，分两阶段；普通推理取12帧、504×504图像，长序列测试取48帧。

#### 方法对比分析
相较直接回归，方法在基座坐标系中从有限正交方向选轴并融合运动与学习先验，无需逐场景优化。适用前提是可见基座具有可靠近正交平面，不适合直接假设任意物体都服从曼哈顿结构。

#### 实验分析（精简版）
评测涵盖Arti4D、D3D-HOI、RBO真实视频及PartNet-Mobility合成划分。Arti4D上5°关节准确率49.3%、中位轴误差4.79°，优于使用RGB-D和真实位姿的ArtiPoint（42.2%、6.30°）。48帧Articulat3D-Sim上方法用6.8秒、误差2.20°；Articulat3D用2.6–11.9小时、误差2.63°，但获得深度、位姿、掩码和运动树。RBO铰链点中位误差8.76厘米；被遮挡部件与质量、摩擦等无法从被动RGB恢复。

#### 实用指南
论文报告在A100上测时，项目页为 phyvggt.github.io；正文未明确说明代码许可。复现需实现两阶段合成训练和静态基座法向筛选。迁移到机器人可用桌面或机身作稳定参照，并融合触觉；非正交结构需放宽轴候选假设。

#### 总结
核心思想：静止基座约束关节
1. 单目视频提取点图、特征并分割部件。
2. 对齐静止基座，以运动和正交候选估计关节。
3. 用运动学生成未见姿态。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22244v1)
- [arXiv](https://arxiv.org/abs/2611.22244v1)

---

<a id='2611.22275v1'></a>
## [Heave-Ho: Decentralized Long-Horizon Multi-Humanoid Collaboration](https://arxiv.org/abs/2611.22275v1)

**Authors:** Ting-Hsuan Chen, Siheng Zhao, Han Jiang, Qi Pan, Louison Lu, Jiaheng Luo, Henghui Bao, Yiyu Chen, Quan Nguyen, Yue Wang

**Published:** 2026-11-17

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Collaboration promises to extend humanoid capabilities beyond what individual robots can achieve. However, realizing this potential requires coordinated whole-body interactions, coherent long-horizon planning, and precise synchronization across robots. We introduce Heave-Ho, a decentralized framework for long-horizon collaborative multi-humanoid loco-manipulation. At the low level, we develop a multi-agent post-training framework that adapts a humanoid foundation tracking model to physically coupled interactions between humanoids, enabling coordinated decentralized control from local observations alone. At the high level, we train a vision-language planner and design a speech-based leader-follower coordination mechanism to synchronize decentralized execution without network communication. Trained with two identical humanoids, our system scales to teams of up to ten robots and generalizes to heterogeneous teams without retraining. To benchmark these capabilities, we introduce HH-Bench, comprising 12 tasks involving teams of 2 to 10 humanoids, including heterogeneous teams. Systematic evaluations show that Heave-Ho consistently outperforms baseline methods, while real-world experiments demonstrate robust decentralized collaboration across challenging tasks.

### 论文解读
#### 摘要翻译
Heave-Ho 面向多台人形机器人的长程协作。它以多智能体后训练适配人形动作跟踪器，并用视觉语言规划器和语音同步协调团队技能，无须机器人间网络消息。作者提出含12项任务、2至10机设定的 HH-Bench；双机训练策略扩展到更多机器人及异构团队。

#### 方法动机分析
共同搬运使伙伴动作经物体传力，影响彼此平衡；集中式控制要传输全队观测，各机独立规划又可能动作不一致或不同步。本文假设共享物体能传递协作所需的物理反馈，再由领导者广播团队技能并统一启动时刻。

#### 方法设计详解
低层每机读取本体感知、短时参考动作和物体规格。冻结 SONIC 跟踪器先产生关节目标，共享残差策略再修正动作以学习抓握和分担载荷。训练用共享演员与集中式 critic，部署时演员只看本机输入；采用特权教师蒸馏局部观测学生，再以 MAPPO 微调，并随机化物体、摩擦、外力和执行延迟。技能连接空手站立 S0 与持物站立 S1 两种枢纽姿态，便于组合行走、转向、升举和放下。
高层以领导者头部相机图像和任务文本为输入，LoRA 微调的 Qwen3-VL-2B 每次输出一项技能。任务每阶段结束后再决策。领导者说出技能，跟随者用本机麦克风识别，并都将动作排至下一个共同时间槽启动。规划器每任务用24段真实示范或1000段仿真轨迹训练；RTX 4090推理耗时52.5毫秒、占5.3 GiB。

#### 方法对比分析
集中式方案需全队通信，独立规划器无法保证动作和时刻一致。Heave-Ho 用共享物体提供物理耦合，以共享演员保持固定单机接口，再用单一领导者和时间槽统一决策与启动。代价是任务依赖有限技能库，适合能拆解成搬运、转向、交接等步骤的场景。

#### 实验分析（精简版）
HH-Bench有12项任务。低层在每任务3个种子、每种子512个团队的仿真评测中平均成功率94.41%，高于 Multi-HDMI 的28.56%。高层五项任务各10次，本文成功43/50，零样本 GPT-6 (Astra) 为21/50；本文决策52.5毫秒，GPT-6 API平均6.46秒，约慢120倍。真实异构桌搬运成功37/40；扩展到10机的证据仅来自仿真桌搬运。真实试验每项10次且初始状态较规整，不能和初值更广的仿真成功率直接比较。

#### 实用指南
规划器训练2轮、有效 batch size 16，约1.4亿可训练参数；低层基于 Isaac 仿真训练。复现需设计共享技能、枢纽姿态、载荷规格和动力学随机化；新任务规划器还需人工遥操作示范。迁移到移动操作或人机协作时，可借鉴共享技能及启动时隙，但须重新适配动力学和安全约束。语音同步要求可靠拾音，噪声会带来限制。

#### 总结
核心思想：共享技能让多机无网协作
1. 观察任务并选团队技能。
2. 领导者播报，各机对齐启动时刻。
3. 本地共享控制器协同完成动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22275v1)
- [arXiv](https://arxiv.org/abs/2611.22275v1)

---

<a id='2611.22297v1'></a>
## [FIRE-Drive: Feasibility-Informed Risk Evaluation for End-to-End Diffusion Planning](https://arxiv.org/abs/2611.22297v1)

**Authors:** Qifeng Xie, Kai-Yuan Feng, Wenkang Wan, Yue Wu, Maoguo Gong, Kai Sheng

**Published:** 2026-11-17

**Categories:** cs.RO, cs.AI

**Abstract:**

Diffusion-based end-to-end autonomous driving models have shown strong potential in generating multimodal planning trajectories. However, their performance still depends heavily on trajectory initialization, mode evaluation, and supervision quality. Weakly constrained noisy initialization may introduce infeasible proposals and increase the burden of denoising. Moreover, selecting the optimal mode remains challenging when candidate trajectories are geometrically similar but differ in safety, interaction risk, and progress. To address these issues, we propose FIRE-Drive, a feasibility-informed risk evaluation framework for end-to-end autonomous driving. FIRE-Drive improves diffusion-based planning through dynamic feasible anchor initialization, candidate-specific bird's-eye-view risk evidence, and compact high-quality candidate supervision with lightweight refinement. This design aligns the initial support, risk-aware scoring, and multimodal supervision of trajectory modes under a unified planning objective. It enables the diffusion decoder to focus on refining plausible trajectories while the scorer distinguishes modes using local risk evidence. Experiments on NAVSIM v1 show that FIRE-Drive achieves a Predictive Driver Model Score of 91.5 with a ResNet34 backbone. Additional closed-loop evaluation on 220 Bench2Drive routes yields a Driving Score of 71.43 and a Success Rate of 42.73%. Component ablations and a scorer comparison under matched candidate inputs support the contributions of feasible initialization, auxiliary supervision, and risk-informed selection. Our code is available at https://github.com/Latte00/FIRE-drive.

### 论文解读

#### 摘要翻译

FIRE-Drive面向端到端自动驾驶中的扩散式规划，针对初始轨迹可能不可行、相似候选难以按风险区分、单条示范难以覆盖多种行为的问题，提出可行锚点初始化、候选相关的鸟瞰图风险证据，以及经过筛选的多模态候选监督与轻量修正。NAVSIM v1上，ResNet34模型取得91.5的PDMS；在220条Bench2Drive闭环路线中，驾驶得分71.43、成功率42.73%。消融和固定候选输入的评分器比较支持各组件的作用。

#### 方法动机分析

扩散生成可提供多条驾驶轨迹，但一般噪声或固定锚点可能在障碍物里，也可能落到车辆短时间无法到达的区域；只比较轨迹整体几何形状，也难发现相近路线的碰撞暴露差异。作者假设让起点符合道路语义与车辆可达性，并给评分器局部风险证据和筛选后的替代监督，可改善生成与选择。

#### 方法设计详解

模型输入前左、前方、前右相机图像、LiDAR和自车状态，经场景编码得到BEV特征。每个模式以车辆运动先验为中心预测高斯锚点，并结合道路/中心线语义与障碍占用构造可行区域；再用简化自行车模型估计车辆可达位置。两者交集用于投影前三步锚点（Tc=3），之后由两层扩散解码器直接更新轨迹。模型输出64个模式，轻量修正器将位置调整限制在每步±1米、航向±0.1弧度。风险评分分支沿各候选轨迹读取BEV特征，把模式和自车状态融入查询；注意力同时依赖特征匹配和空间风险偏置。训练时，碰撞或越界事件生成带高斯扩散的局部目标，以KL损失教会注意力关注风险位置。额外监督由日志未来做超车、加速变形而来，每场景最多缓存5条候选；通过几何检查且同场景评估分高于日志未来的轨迹才保留，用来匹配次要模式。无合格候选的场景仍保留原有监督。主实验使用ResNet34、AdamW学习率2×10⁻⁴、batch size 64、RTX 5090混合精度，报告第40轮模型。

#### 方法对比分析

与无约束噪声初始化相比，它用场景语义和车辆可达性共同限制起始锚点；与只把质量分数回归给候选的评分器相比，它在特征聚合时加入候选路径附近的风险信息。与在线奖励优化不同，辅助轨迹离线生成、筛选和缓存，重复训练时直接复用。关键贡献是把初始化支持、替代模式监督和风险选择纳入同一规划目标。其相对优势可能更适合多模态驾驶规划中“起点是否可行”和“近似轨迹谁更安全”同时重要的场景。

#### 实验分析（精简版）

在NAVSIM v1、ResNet34设置下，PDMS为91.5，高于DiffusionDriveV2的91.2和DiffusionDrive的88.1；Bench2Drive 220条闭环路线得分71.43、成功率42.73%。固定候选输入的评分器比较中，完整模型PDMS 91.0，去掉风险上下文为90.3，支持风险证据对选择有帮助。限制是NAVSIM强基线上的提升有限；Bench2Drive使用不同文献配置，非统一条件的因果比较。证据来自基准与仿真，未做实车验证。候选池只覆盖部分场景，热图也不等于罕见危险统计。

#### 实用指南

作者公开了代码仓库，可从ResNet34配置开始复现。关键实现是锚点只投影前三步，64模式、两层解码器和一次修正；缓存的辅助候选仅在通过几何和质量筛选后参与训练，推理时不需要这些缓存。迁移到机器人规划，可把道路掩码换为无碰撞区、车辆可达掩码换为机器人动力学可达区，再沿候选构造风险查询；论文未验证。还需补足时序信息和紧急避障候选。

#### 总结

先筛起点，再按路径看风险

1. 图像、激光雷达和车辆状态编码成场景特征。
2. 道路语义与车辆可达区取交集，约束初始锚点。
3. 扩散解码并修正，产生多条候选轨迹。
4. 风险注意力评分选出最终计划；筛选过的备选轨迹补充训练。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22297v1)
- [arXiv](https://arxiv.org/abs/2611.22297v1)

---

<a id='2611.22388v1'></a>
## [Hidden Variables Behind Real-Robot Data Quality: Lessons from Hundreds of Thousands of Hours](https://arxiv.org/abs/2611.22388v1)

**Authors:** Maniformer Real-World Manipulation Team, Che Tsang Calvin, Xingyu Chen, Bowen Jiang, Naixiang Gao, Xiongfeng Cai, Enlin Gu, Ruiying Li, Peize Li, Mianrui Zhu, Haomiao Zhang, Xin He, Minhui Yu, Jun Zhao, Zihao Zhu, Yang Chen, Jingmei Jin, Keru Zhou, Chen Wang, Chang Zhou, Zhiwen Hou, Di Yang, Wenjun Sun, Fan Yang, Liyan Zhang, Yi Lu, Haojun Ma, Jun Zhu, Pengxin Zha, Zhaosheng Wang, Xin Zhao, Xindong He, Chiming Liu, Jianheng Song, Jiayao Ma, Maoqing Yao, Guanghui Ren, Qian Wang, Wei Shan, Yang Yang, Li Liu, Wenhao Wang

**Published:** 2026-11-17

**Categories:** cs.RO

**Abstract:**

As real-robot data collection scales, collection-system design shapes both operator efficiency and the behaviors learned by robot policies. Workspace limitations, control-interface friction, and latency can introduce pauses, detours, and corrective motions into otherwise successful demonstrations. We investigate these effects across embodiment, teleoperation control, and data consistency, combining a kinematic study of 25 dual-arm robots, controlled teleoperation experiments, and real-robot evaluation of fine-tuned $π_{0.5}$ policies. The embodiment study shows that positional reachability overstates orientation-aware task coverage and that modest shoulder motion can substantially expand the effective workspace. Teleoperation experiments show that parallelizing body-part control reduces task completion time by 43-53%, while compliant control improves success and reduces peak contact forces on constrained and contact-rich tasks. Policy-learning experiments show that demonstrations must capture the visual information underlying action choices: active-search demonstrations improve arm-selection accuracy from chance (50%) to 100%. Calibration experiments reveal that mixed-calibration training data remain usable in our tested configurations when an inference-aligned subset is retained, whereas inference-time miscalibration causes all 30 trials to fail. Policies also retain approximately 90% of the extra path length in detour-contaminated demonstrations, showing how inefficient operator motions persist in learned behavior. Together, these findings connect collection-system design to demonstration quality and downstream policy execution. They motivate evaluating effective workspace, control efficiency, observation completeness, calibration alignment, and motion quality before scaling collection, and addressing system-induced compensations at their source.

### 论文解读
#### 摘要翻译
大规模真实机器人数据采集时，机器人形态、遥操作界面、延迟和标定都会影响操作者效率及策略学到的动作。作者结合25种双臂机器人的运动学分析、受控遥操作实验和微调π0.5的实机评测，发现位置可达性高估了任务覆盖；并行控制缩短任务时间，顺应控制改善接触任务；示范若缺少决策所需视觉信息、推理时标定不一致或包含绕行，策略也会继承相应问题。

#### 方法动机分析
成功完成一次任务，不代表记录下来的轨迹适合学习。操作者可能靠机器人相机看不到的线索选动作，也可能因视野延迟、控制模式切换或工作空间不足反复纠正。论文的假设是：这些系统因素会改变示范中的观察—动作对应，最终改变策略执行。

#### 方法设计详解
工作分三条线。第一，作者从25种机器人URDF提取运动学模型，在工作空间采样位置和工具方向，用逆运动学求解可行姿态，比较形态、腰部运动和工具长度对方向覆盖及关节余量的影响。第二，通过代表性任务比较不同遥操作配置：将身体各部位交给独立输入或自动模块并行控制，另测试绝对/相对映射、位置/力控制、延迟和外设。延迟模型将网络、控制处理、必要身体部位中最慢响应及视频反馈相加。第三，微调π0.5并在实机任务中检验三类数据问题：动作发生前缺少视觉线索、训练与推理机器人关节零位不同、示范里掺入暂停和绕行。绕行实验每种比例训练100条示范，比例为0%、20%、50%、80%。

#### 方法对比分析
与只统计成功率或末端可达范围相比，论文把质量判断延伸到动作依据、方向覆盖、路径冗余和推理一致性。它不是提出新策略网络，而是用跨环节实验建立“采集系统—示范内容—策略行为”的诊断框架。该框架适合规划真实机器人数据采集与验收流程。

#### 实验分析（精简版）
在人形与带升降移动双臂的肩部切片中，位置可达率分别为55%和53%，但方向覆盖只有41%和33%；桌面双臂为91%和75%。控制通道拆分后，地面取物和擦拭耗时分别减少53%和43%。盲区任务中，加入主动搜索后选臂准确率由约50%升至100%；80%绕行示范训练的策略保留约90%的额外路径，完成耗时增加约29%。但数据实验只覆盖少数任务和一种后训练策略，作者也指出部分实验为单次训练，外推范围有限。

#### 实用指南
复现需具备机器人URDF和逆运动学、可控遥操作配置、实体机器人与π0.5后训练流程，并按任务记录视频、关节状态、动作及时间。落地时先用代表性任务检查方向覆盖和全链路反馈延迟，再确认示范包含决策所需观察、推理端校准与采集端一致，并检查路径长度和停顿。论文未在本文中提供完整开源复现包或训练超参数；结果来自作者采集系统，迁移到其他平台时需重新测量。

#### 总结
核心思想：采集系统决定策略行为

1. 检查机器人能否以任务所需姿态到位。
2. 让操作者并行、低延迟且顺应地控制。
3. 记录支撑每个动作的视觉线索并对齐标定。
4. 以路径效率和实机策略表现验收示范。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22388v1)
- [arXiv](https://arxiv.org/abs/2611.22388v1)

---

<a id='2611.22820v1'></a>
## [Pixel is All You Need](https://arxiv.org/abs/2611.22820v1)

**Authors:** Jie Chu, Haoting Shi, Fucheng Zhang, Jiawei Liu, Junwei Liao, Ying Wen, Muning Wen

**Published:** 2026-11-17

**Categories:** cs.CV

**Abstract:**

Text, images, audio, touch, and even robot actions can, in principle, all be rendered onto a single canvas, suggesting an ambitious goal of unifying all modalities within vision. Language provides the natural first test. To subsume it, a visual model must reason directly over text in pixels. Across a wide range of image, video, and unified multimodal generators, however, we find that their visual generation modules cannot perform symbolic reasoning over text on the canvas. They succeed only when an underlying text module or prompt rewriter already supplies the answer, effectively reducing the generator to a mere renderer, a limitation that post-training does little to resolve. To test whether this failure stems from pixels themselves or from current training paradigms, we train a family of purely pixel-space generative models from scratch without any text pathway. With images as their sole input and output, these models read questions, write intermediate steps, and produce answers directly on the canvas. Remarkably, they exhibit hallmark behaviors of large language models. Performance improves predictably with scale following a power law. Writing intermediate steps frame by frame substantially outperforms single-frame answering, echoing chain-of-thought, while reinforcement-learning post-training further strengthens reasoning. Moreover, sharing one canvas allows the same model to reason across modalities, a capability that text-only language models lack. To the best of our knowledge, this is the first systematic study of these phenomena in purely pixel-space generative models free of text tokens.

### 论文解读

#### 摘要翻译
作者研究能否只用像素完成读题、推理和作答。对现有图像、视频及统一多模态生成器的测试显示，它们往往依赖隐藏的文本思考或提示改写：文本模块先算出答案，图像模块再把答案画出来。研究者据此从头训练没有文本通路的像素模型，在画布上逐帧写出中间步骤。实验报告了随模型规模提升的误差规律、逐帧推理相对直接作答的优势，以及强化学习后训练带来的提升，并展示文字与图形任务可在同一画布中衔接。

#### 方法动机分析
图像生成器逐渐能生成清晰文字，也能遵循复杂编辑指令，但文字可读不等于模型能读懂问题并自己计算。作者用“答案是否先出现在文本提示中”诊断当前管线，再训练纯视觉模型，以区分像素媒介的限制和训练方式的限制。核心假设是：画布若能保留当前状态和中间结果，模型就可以把它作为工作记忆，连续完成符号计算和图形操作。

#### 方法设计详解
任务统一画成方形灰底画布：问题写在上方，答案留在下方，图形元素放在中间。模型是直接作用于原始像素的 flow-matching transformer，没有 VAE、文本编码器或 tokenizer。条件画布与加噪目标画布拼接后切成 576 个 8×8 图块，模型预测干净图像并用速度损失训练。研究比较两种输出方式：直接作答一次生成最终画布；逐帧方式学习从第 t 帧生成第 t+1 帧，推理时把新帧原样反馈，直到答案框出现内容。第二种方法让模型把算式、计数、曲线描摹等中间状态写在画面上。实验包括表达式、乘法、文字题、事实问答，以及迷宫、曲线追踪、计数后涂色等共十类任务。模型从 7.6M 到 458M 参数；主对照训练 40k 步，规模实验训练 132k 步，基础设置为 batch 256、AdamW、学习率 1e−4 和 bf16。

#### 方法对比分析
与提示改写后再绘图的系统不同，本文模型没有文本答案通道，必须直接读取画面并通过画面更新完成计算。与一次输出终稿相比，逐帧策略把中间过程变成可见状态，也可将视觉操作与文字记录交替起来。它与语言模型的链式思考相似，但计算载体是不断改写的像素画布；与常见统一多模态模型相比，统一发生在像素空间而非文本、图像 token 的共同序列中。适合研究可验证的视觉符号任务与跨模态状态更新，暂不能据此视为开放域通用推理器。

#### 实验分析（精简版）
纯像素模型的逐步误差随参数规模呈幂律变化：文本任务从 7.6M 的 0.174 降至 130M 的 0.064；只用不超过 130M 的模型拟合后，对 458M 留出模型的误差预测偏差为 2%。同为 130M、训练 40k 步时，逐帧文字题准确率为 78.6%，直接作答为 1.2%；在 Compute Then Paint 上逐帧为 98.1%，直接作答即使训练 160k 步也为 31.1%。GRPO 让 130M 文字题从 23.7% 提升至 84.6%，同预算最优示范 SFT 为 39.0%。优势来自受控合成任务，题目数量固定且视觉布局规整，尚未证明开放图像或真实环境泛化。

#### 实用指南
论文提供项目页与代码仓库： https://github.com/bxzzcj/pixel-is-all-you-need 。复现时需固定网格字体、自动生成的问题分布及程序化答案读取；测试集每项含 1,024 个问题。逐帧推理的采样步数为文字任务 30 步、跨模态任务 10 步，规模实验每种参数规模使用五个种子。迁移到机器人规划时，可尝试把对象、路径或动作候选画在共同状态图上并逐步修订；这属于可检验的研究设想，需在有噪声的视觉输入和仿真控制任务中单独评估。

#### 总结
核心思想：让画布承载推理

1. 把问题、图形和空白答案区画到同一画布。
2. 纯像素模型读取画布，逐帧写出计算或视觉操作。
3. 将新画布回馈为下一步状态，直到完成答案。
4. 用规模、逐帧对照及强化学习实验检验性能变化。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22820v1)
- [arXiv](https://arxiv.org/abs/2611.22820v1)

---

<a id='2611.22240v1'></a>
## [Cross3DGS: Unified Vulkan-Based 3D Gaussian Splatting Training from Workstations to Mobile Devices](https://arxiv.org/abs/2611.22240v1)

**Authors:** Hiroyuki Sato, Yusuke Takimoto

**Published:** 2026-11-17

**Categories:** cs.CV, cs.GR

**Abstract:**

Scenes captured with handheld and wearable cameras are still shipped to a CUDA workstation for 3D Gaussian Splatting (3DGS) training. Training on the device that captured them, or on its companion phone or laptop, requires more than translating kernels: GPUs and drivers differ in parallel features, memory limits, and command submission, and these differences decide whether a training schedule executes and returns the correct update. We present Cross3DGS, a cross-vendor, cross-platform Vulkan trainer that keeps one optimizer and chooses its execution schedule per device in two steps: rules on device limits, workload size, and observed driver failures exclude the schedules known not to run correctly, and a fixed preference order selects one of the remaining feasible schedules. The same core completes standard-quality training (30k updates, one million Gaussians) on every configuration we measured, spanning NVIDIA, Intel, Arm, Qualcomm, AMD, and Apple GPUs and including Android devices and an iPhone, whereas the existing portable trainers VkSplat and Brush do not run on some of the mobile GPUs. Averaged over 11 scenes on a workstation, Cross3DGS has higher quality than VkSplat with no significant difference in training time, and Brush reports about three times the training time at comparable quality. The final quality of Cross3DGS is nearly identical across devices. A mobile application turns a handheld capture into a viewable model entirely on the device.

### 论文解读
#### 摘要翻译
Cross3DGS 是一种基于 Vulkan 的跨平台 3D Gaussian Splatting 训练器。它让同一优化器根据设备能力、训练数据规模和已知驱动问题选择执行方案，在工作站、笔记本、手机和平板上训练。论文在八种配置上完成标准质量训练，并展示了从手机拍摄到重建、训练和查看的端侧应用。

#### 方法动机分析
把 GPU 算子改写成可移植着色器，并不能保证整套训练都能运行。不同设备的子组宽度、共享内存、浮点原子操作和单个存储缓冲区上限不同，某些驱动还会报告支持却执行失败。传统按厂商编写后端会重复维护，单一固定方案则可能在部分设备失效。作者的核心假设是：可以把这些差异描述成明确的可行性条件，再在可行方案里按固定偏好选一个，无需每台设备在线试跑全部方案。

#### 方法设计详解
一次训练更新包含投影、分块排序、透明度合成、损失与梯度计算及 Adam 更新。Cross3DGS 用六类实现选项组织执行 schedule：梯度计算的像素/高斯分配与梯度合并方式、命令提交、训练图像是否常驻、基数排序算法、投影和 Adam 是否融合、以及高斯对的间接或直接派发。组合成 256 种候选方案。

设备启动后，第一步根据设备限制筛选，例如 tiled 梯度需要至少 8 车道子组和 48 KiB 共享内存；native atomic 需要浮点原子加。工作负载规则会在图像或 SH 参数超过存储缓冲区上限时改成逐图流入或分片；驱动规则会移除已证实会出错的选项。第二步按固定优先级依次比较各选项，选出一个 schedule，并记录下来用于全部训练更新。没有可行方案就停止训练。

不同 schedule 的浮点累加次序不同，故作者离线比较梯度和 Adam 更新，而非只比较最终图像分数。每种可行梯度方式在设备上与参考实现对比，再将选定 schedule 与 CPU 双精度有限差分结果核验。训练设置为 30,000 次更新、SH degree 3、最多一百万个 Gaussian，使用 MCMC 式细化；训练损失由 0.8 L1 与 0.2(1−SSIM) 组成。Android 与 iOS 应用把相机采集、结构光束法平差、优化和自由视角查看连成端侧流程。

#### 方法对比分析
VkSplat 与 Brush 是最接近的可移植基线，但它们在部分手机 GPU 上不支持或无法完成标准训练。Cross3DGS 的区别是把多种执行选项放在同一目录中，以设备、负载和驱动证据筛选，而不是仅针对单个梯度内核做限制检查或在目标机计时搜索。固定偏好让运行更可预测，也可能错过设备上的最快方案，因此它并非自动调优器。该设计适合需要同一训练逻辑跨多类 GPU 运行、且能维护设备规则的离线重建场景。

#### 实验分析（精简版）
在八种 NVIDIA、Intel、Arm、Qualcomm、AMD 和 Apple 配置上，Cross3DGS 都完成标准质量训练；VkSplat 完成四种，Brush 完成六种。两个跨设备质量测试场景上，最终 PSNR 的最大跨度分别为 0.019 dB 和 0.063 dB。工作站 11 场景比较中，Cross3DGS 相比 VkSplat 平均提高 0.074 dB，训练时间差异无显著证据；Brush 的训练时间约为三倍。局限是完整跨设备质量只测两景，基线对比只在一台工作站，驱动规则也依赖有限设备与版本。

#### 实用指南
作者公开了训练器、Android/iOS 应用及基准脚本。复现应固定 Vulkan 驱动与设备信息，按论文的 30k 更新、SH3、百万 Gaussian 配置记录实际缓冲区与执行状态，并核对单步梯度和更新误差。手机标准训练耗时约数小时，电量、温度与厂商功耗策略会改变速度；应用演示模式则用 3k 更新和约 25 万容量。迁移到其他 GPU 时，应先重新验证规则和数值一致性，不能只依据同一厂商标签套用驱动例外。

#### 总结
核心思想：规则筛选可移植训练方案

1. 输入多视角图像并初始化 Gaussian 场景。
2. 从设备能力、数据规模和驱动记录筛掉不可行执行方案。
3. 按固定优先级运行训练，并离线验证更新数值。
4. 在设备上完成重建并交互查看结果。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22240v1)
- [arXiv](https://arxiv.org/abs/2611.22240v1)

---

<a id='2611.22543v1'></a>
## [JointGV: Structure-Aware Correction of Visual World Models with Parallel Graph Dynamics for Long-Horizon Modeling](https://arxiv.org/abs/2611.22543v1)

**Authors:** Tian Lyu, Yichen Jiang, Ziyuan Cheng, Yunda Yan, Xiangyu Pan, Junjie Jiang, Jialin Yu, Boyu Ning, Pietro Liò, Fan Mo

**Published:** 2026-11-17

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Visual world models predict future visual states conditioned on actions, but recursive errors in entity poses and spatial relations limit their long-horizon prediction for robot planning and control. We ask whether a separately predicted relational state can correct this drift without restructuring the visual model. This is challenging because graph nodes and visual states share neither elements nor a feature space. We propose Joint Graph--Visual Dynamics Correction (JointGV), a plug-in based on a cross-granularity dynamics hypothesis: despite their different representations, graph-node and visual-state changes describe the same physical transition. JointGV rolls out a task-conditioned scene graph alongside the visual world model, couples their predicted changes, and combines the resulting graph-motion context with graph-predicted node positions to compute a bounded residual. Graph-learning and correction gradients do not propagate into the original visual dynamics. We attach JointGV to DINO-WM (spatial tokens) and DreamerV3 (a global latent state), evaluating both on DMC Cheetah Run and Walker Walk across three seeds and extending the evaluation to contact-rich manipulation with DMC Finger Turn Hard. Across 32-step rollouts, JointGV improves structural and visual-prediction accuracy, reducing joint-position error by 11--28\%. For spatial tokens, learned graph--visual change coupling gives lower error than broadcasting the same mean graph-motion context to every token, particularly when non-adjacent parts appear close together in the image. On real-robot inverted-pendulum sequences, JointGV reduces DINO-WM's 32-step pole-angle error from $12.56^\circ$ to $6.69^\circ$ and improves DreamerV3 and LeWorldModel as well. In simulated inverted-pendulum control, an actor trained with JointGV-corrected rollouts completes 46 of 50 episodes, compared with none using the original rollouts.

### 论文解读

#### 摘要翻译
视觉世界模型根据动作预测未来画面，但递归预测会积累物体位置和空间关系错误，影响机器人长程规划。本文提出JointGV：在原视觉世界模型旁并行预测任务相关场景图，以图节点和视觉状态在同一动作下产生的变化作为共同物理信号，形成有界残差来修正视觉预测。方法接入DINO-WM空间token和DreamerV3全局latent，在仿真、真实机器人记录和控制任务中评估。

#### 方法动机分析
规划需要反复把预测状态送回模型，细小误差因此逐步放大。已有办法会改变视觉状态表示、只在视觉latent内部修残差，或另建对象中心模型；它们没有解决“独立预测的图状态如何校正元素与特征空间均不对应的视觉状态”。JointGV假设两者在同一动作下的变化代表同一物理转移，借此避开绝对特征的一一映射。

#### 方法设计详解
输入为当前RGB图像、动作和已观测的短历史。首先，RelTR中间特征经任务角色查询绑定成场景图节点，节点含特征、图像位置和关系；节点标签仅用于训练监督，推理时从RGB构图。其次，历史感知图动力学根据图状态与动作递归预测下一图，并以多步self-fed训练模拟推理过程。与此同时，原世界模型独立预测视觉下一状态及变化。对DINO-WM，将视觉token变化和图节点变化投影到共享空间，用Sinkhorn最优传输软匹配，形成token级图运动上下文；再利用图预测的节点位置形成空间上下文。对DreamerV3全局状态，则以全局变化为查询，对节点变化做注意力池化。最后，校正头结合视觉预测、两类图上下文和门控，产生逐维有界残差并加回原预测。训练校正时冻结原视觉模型，避免图支路损失反向改变它。DMC实现使用8个图槽、64维节点、3层图消息传递和4帧图历史；图动力学及校正训练8步，batch size为16，学习率为1e-4。

#### 方法对比分析
与把图状态直接作为世界模型状态的对象中心方法不同，JointGV保留既有视觉状态，将图作为外部修正信号；与视觉残差相比，修正依据显式预测的图变化和节点位置。空间token和全局latent分别采用OT匹配与注意力池化，因此同一原则可适配不同视觉表示。其代价是需要任务定义的节点角色及相应监督，并额外运行图观察器和图动力学。

#### 实验分析（精简版）
DINO-WM与DreamerV3在Cheetah Run、Walker Walk的四种组合均以三种子评估，32步平均关节位置误差降低11.1%–28.0%；DINO-WM Walker Walk从10.740降至7.682像素。接触操作Finger Turn Hard上误差下降27.6%。真实倒立摆记录的开环预测中，DINO-WM杆角误差从12.564°降至6.689°；仿真策略完成46/50个六秒回合，原模型策略0/50。实体机器人仅展示短时控制优势，尚无稳定六秒平衡；节点预测失败时校正也会变弱。

#### 实用指南
论文提供实现代码。复现需准备任务角色节点和关系的监督数据；训练可从冻结视觉主干、构造图观察器与多步self-fed图动力学开始，再训练变化对齐和有界校正。作者报告主DMC实验用24GB RTX 5090 D v2，另有RTX 4090及22GB RTX 2080 Ti运行种子。迁移时应先验证图节点定位及递归误差，再比较原模型、均值图运动、OT/注意力耦合与视觉残差，并监测低置信度图状态下的退化。增加推理成本和任务角色监督需求需要纳入部署预算。

#### 总结
核心思想：用图变化校正视觉漂移。
1. 从图像提取任务角色节点与关系。
2. 根据历史和动作递归预测场景图。
3. 把图节点变化与视觉变化对齐并汇聚位置线索。
4. 生成有界残差，修正视觉模型的下一状态。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22543v1)
- [arXiv](https://arxiv.org/abs/2611.22543v1)

---

<a id='2611.22592v1'></a>
## [Distilling World Action Models for Efficient Bimanual Manipulation via Anchored Path Matching](https://arxiv.org/abs/2611.22592v1)

**Authors:** Jialong Liu, Junjie He, Yandu Chen, Bin Zhou, Heng Su, Peijun Tang, Jianan Wang, Changhao Chen

**Published:** 2026-11-17

**Categories:** cs.CV, cs.RO

**Abstract:**

World Action Models (WAMs) jointly generate future video and robot actions through iterative denoising. Few-step distillation reduces the number of model evaluations, but its coarse updates can move the student away from the teacher's trajectory. We propose Anchored Path Matching (APM), which supervises endpoint and short-transition predictions from states reached by the student. From each state, we continue the teacher to zero noise and use its endpoint and first intermediate state as targets. Modality-routed adapters learn to match both targets from the same starting state. When the two gradients conflict, anchor-priority updates preserve the endpoint gradient and remove the opposing component of the local gradient within each modality. One checkpoint supports 1-, 2-, and 4-step joint generation. The sampling budget can be chosen at deployment without retraining or teacher queries. On RoboTwin 2.0, we train only on Clean demonstrations and evaluate directly on the domain-randomized Random setting. At one step, APM improves closed-loop success over continued base distillation at matched inference budgets by 2.0 percentage points on Clean and 1.4 on Random. Video peak signal-to-noise ratio (PSNR) rises from 30.169 to 32.317 dB, with gains in structural similarity (SSIM) and learned perceptual image patch similarity (LPIPS). On two real-robot tasks, APM improves success over the one-step teacher and reduces control-cycle latency by $21.3\times$ relative to the eight-step teacher.Project page: https://jialonglong.github.io/Anchored-Path-Matching/.

### 论文解读

#### 摘要翻译
世界动作模型（WAM）通过迭代去噪共同生成未来视频与机器人动作。少步蒸馏虽能减少模型调用，但粗糙更新可能使学生偏离教师轨迹。Anchored Path Matching（APM）从学生实际到达的状态出发，同时监督最终输出和短程转移：教师继续去噪到终点，并把首个中间状态也作为目标；模态适配器学习匹配两者。若两种目标的梯度冲突，方法优先保留终点梯度。一个检查点支持 1、2、4 步推理。在 RoboTwin 2.0 上仅用 Clean 演示训练，直接测试域随机化 Random 条件；一步相对继续基础蒸馏的成功率在 Clean/Random 分别提高 2.0/1.4 个百分点，视频 PSNR 由 30.169 升至 32.317 dB。两个真实机器人任务也显示了一步策略的潜力。

#### 方法动机分析
WAM 要预测“接下来会看到什么”和“接下来会做什么”，但每次去噪调用都增加时延。直接把多步教师压缩成少步学生时，学生的中间状态会偏离教师常见轨迹；在教师状态上学到的局部变化，未必适用于学生遇到的状态，而只监督最终结果又缺少局部引导。APM 假设，在学生状态上同时取得教师终点与短程转移，能兼顾完整任务结果和局部运动；当两者更新方向相反，应让完整结果优先。

#### 方法设计详解
输入包括当前观测、语言指令、机器人状态和视频/动作噪声潜变量；输出是未来视频与动作潜变量。首先以 LingBot-VA 为底座，用 AnyFlow 式有限区间蒸馏得到可跨步长调用的转移模型。第二阶段冻结底座，训练视频和动作各自路由的 LoRA 适配器。训练时从学生 rollout 选取一个已到达的状态并停止对前缀反传；教师从该状态沿八步 Euler 网格去噪。其最终状态是 Restart 锚点，首个中间状态是 Local-Field 目标。学生从相同起点分别预测终点和中间状态，以均方误差拟合。更新时分别计算两组梯度：先保留终点梯度；若局部梯度与终点梯度相反，就减去局部梯度在终点方向上的冲突分量，再加上基础蒸馏梯度。视频和动作分别做此操作。训练总计 9,000 步（Stage I 4,000、Stage II 5,000），batch size 为 32。部署时同一组权重可递归执行 1、2 或 4 次视频/动作联合去噪，选择步数无需重新训练或调用教师。

#### 方法对比分析
相较只做有限步转移蒸馏，APM 加入从学生所处状态构造的完整输出目标；相较只做终点蒸馏，又保留短程动态信息。作者认为，AnyStep-WAM 的预算对齐转移和预览恢复、RealtimeWAM 的动作端终点监督与 APM 互补；APM 的特点是两个目标来自同一次教师续跑、起点相同，并用单向梯度投影处理冲突。该策略适用于同时关心未来视觉预测与动作控制、且推理预算可变的生成式策略。

#### 实验分析（精简版）
RoboTwin 2.0 包含 50 个双臂操作任务，每个 Clean/Random 条件各评估 2,500 个 episode；仅用 Clean 训练，Random 不做适应。一步时 APM 平均成功率为 80.9%，相对同预算继续蒸馏的 79.2% 提高 1.7 个百分点；四步提高 1.6 点，两步仅提高 0.1 点。视频一步 PSNR 从 Continued 的 30.169 升至 32.317 dB，但 APM 对 Flash-WAM 的 SSIM、LPIPS 并非全面占优。真实 Astribot S1 上两项任务各 24 次，一步 APM 延迟 203.1 ms，八步教师为 4,323.6 ms（21.3 倍）；PENHOLDER 成功率仍低于完整教师。物理证据局限于单平台、两任务和较小试验数。

#### 实用指南
论文给出了代码和 Hugging Face 模型链接。复现需 LingBot-VA 底座、AnyFlow 式 Stage I、教师八步续跑以及 Stage II 模态适配器训练；适配器的 FFN rank 为 64、temporal rank 为 128，batch size 32。公平比较应固定训练预算与参数量，并分别报告各推理步数、Clean/随机化条件和视频指标；真实机器人还应统计控制周期时延及分任务失败。迁移到其他机器人或时序决策模型在概念上可行，但需验证新模态下的梯度分组、教师可靠性和安全边界，论文尚未覆盖更长预测跨度或多平台。

#### 总结
**学生状态双目标蒸馏**
1. 先把多步 WAM 蒸馏成可变步长的基础转移模型。
2. 从学生自己到达的状态取样，而非只沿教师轨迹取样。
3. 教师一次续跑产生最终锚点和局部转移目标。
4. 终点优先消除梯度冲突，训练后按算力选 1、2、4 步生成。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.22592v1)
- [arXiv](https://arxiv.org/abs/2611.22592v1)

---

