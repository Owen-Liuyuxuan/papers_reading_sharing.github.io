time: 20260909

# Arxiv Computer Vision Papers - 2026-09-09

## Table of Contents

1. [LightSplat: Real-Time High-Fidelity 3D Gaussian SLAM with Loop Closure](#2609.07274v1)
2. [Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body Model Predictive Control](#2609.07544v1)

---

## Papers

<a id='2609.07274v1'></a>
## [LightSplat: Real-Time High-Fidelity 3D Gaussian SLAM with Loop Closure](https://arxiv.org/abs/2609.07274v1)

**Authors:** Junze Bao, Ye Gao, Yiming Huang, Xiaolong Yu, Chen Dong, Qing Gao, Wei Wang, Jinhu Lü

**Published:** 2026-09-07

**Categories:** cs.RO, cs.CV

**Abstract:**

SLAM systems based on 3D Gaussian Splatting (3DGS) have recently demonstrated promising reconstruction accuracy for dense 3D scene representations. However, current 3DGS systems struggle to meet the strict demands of real-world deployments due to severe limitations in operational performance and map adaptability. To this end, we propose LightSplat, a hybrid-representation RGB-D SLAM framework. It synergizes local sparse features for robust and fast tracking with a dual-thread backend that progressively constructs dense Gaussian submaps. Crucially, we enable online loop closure through feature-accelerated 3DGS registration, refining overall map consistency through pose graph optimization. Ultimately, LightSplat achieves the online reconstruction of high-fidelity Gaussian map. Extensive experiments on multiple datasets and real-world robotic platform demonstrate that our method achieves near state-of-the-art reconstruction quality and the capability to accommodate practical camera motions, maintaining an average framerate of 8 FPS. Overall, LightSplat provides an efficient and robust foundation for deploying high-fidelity 3DGS in real-world environments.

### 论文解读

#### 摘要翻译
LightSplat 是一种混合表示的 RGB-D SLAM 框架：用局部稀疏特征实现快速稳健跟踪，用双线程后端逐步构建稠密 3D 高斯子图，并通过特征加速的 3DGS 配准在线闭环，再以位姿图优化保持全局一致性。它在多数据集和真实机器人平台上实现高保真在线重建，平均帧率达到 8 FPS。

#### 方法动机分析
纯光度驱动的 3DGS-SLAM 在快速相机运动时收敛范围有限，连续梯度位姿优化也带来延迟；许多方法只有视觉里程计，长期运行会累计漂移。LightSplat 的思路是让稀疏特征负责可靠的几何定位，让高斯负责细节表达，并把两者统一到关键帧和子图坐标中。

#### 方法设计详解
每个 RGB-D 帧先由 SuperPoint 提取特征、LightGlue 匹配，RANSAC 清除离群点；结合深度反投影得到稀疏 3D 地标，以重投影误差估计位姿，并在滑动窗口做局部 BA。达到位移、旋转或点数衰减条件后生成关键帧。地图由稀疏点云锚点和稠密高斯子图组成，高斯维护均值、协方差、不透明度与颜色。主线程快速建图，细化线程并发优化；损失结合 L1+SSIM 颜色、L1 深度和各向同性正则。闭环使用预训练 NetVLAD 检索候选，先 PnP 初始化，再按渲染残差自适应：残差低于 τ=0.25 时直接采用几何约束，否则对重叠视图细化配准并做加权旋转平均。PGO 的修正同步变换关键帧与高斯几何属性，免于重新优化颜色。实验使用 i7-13700K、RTX 4080 和 640×480 输入。

#### 方法对比分析
相较纯光度方法，稀疏特征扩大了快速运动下的跟踪能力；相较无闭环的 3DGS 里程计，NetVLAD、快速注册和 PGO 能抑制长期漂移；相较松散的追踪/建图解耦方案，双线程仍共享关键帧—子图坐标关系。其贡献重点是混合表示、并发细化与残差自适应注册，基础特征网络和 PGO 则是标准组件的组合。

#### 实验分析（精简版）
论文在 TUM RGB-D、ScanNet、Replica 和真实移动机器人上验证跟踪、重建与实时性，并与 LoopSplat、SplaTAM 等方法对比。在 TUM RGB-D 上，平均 ATE-RMSE 为 3.33 cm，优于 LoopSplat 的 3.46 cm 和 SplaTAM 的 5.48 cm。Replica 上 PSNR/SSIM/LPIPS 为 35.57/0.980/0.124，平均约 8 FPS；追踪延迟 0.04 s/帧、建图延迟 0.68 s/帧，闭环约 0.5 s/iter，而 LoopSplat 为 13 s/iter。消融中，稀疏图令速度从 5.0 增至 10.1 FPS，闭环使特定序列 ATE 降至 2.02 cm，说明稀疏几何和闭环分别贡献了效率与全局精度。代价是地图扩大后配准变慢、缺少全局 BA，受限视角仍可能有悬浮伪影；这些结果主要覆盖 RGB-D 室内场景，不能直接推断极大规模或无深度输入环境的表现。

#### 实用指南
复现需准备 RGB-D 标定与同步、SuperPoint/LightGlue/NetVLAD、PnP/RANSAC、3DGS 渲染和位姿图优化。推理时使用 640×480 输入，论文给出 α=0.8、τ=0.25 及 i7-13700K + RTX 4080 配置，但未提供可确认的官方代码或完整依赖链接，不能假定开箱即用。迁移到新机器人或数据集时，应重新处理深度尺度、特征检索阈值、高斯细化和闭环候选设置，并同时评估 ATE、渲染质量和实时吞吐。

#### 总结
核心思想：**稀疏几何驱动高斯闭环建图**
1. 特征匹配和深度锚点稳住位姿；
2. 关键帧 BA 优化局部结构；
3. 双线程生成、细化高斯子图；
4. 残差自适应闭环并用 PGO 校正全图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.07274v1)
- [arXiv](https://arxiv.org/abs/2609.07274v1)

---

<a id='2609.07544v1'></a>
## [Anti-Gravity Walking by a Flying Humanoid Robot via Thrust-Rate Input Whole-Body Model Predictive Control](https://arxiv.org/abs/2609.07544v1)

**Authors:** Kazuki Sugihara, Kei Okada

**Published:** 2026-09-07

**Categories:** cs.RO

**Abstract:**

Flying humanoids are expected to perform tasks in diverse environments, while their existing locomotion is mainly limited to aerial flight and ground walking. The capability to move in complex three-dimensional space can greatly expand their application range. For such walking motion on ceilings and similar anti-gravity environments, whole-body MPC is effective. However, the discontinuous changes in dynamic structure accompanying contact switching during walking can induce thrust spikes, resulting in control instability. Therefore, in this work, we propose and implement a real-time whole-body MPC framework for anti-gravity bipedal walking. First, we formulate whole-body MPC using the time derivative of thrust, namely thrust-rate, as the control input. This formulation guarantees continuity of the thrust trajectory during contact switching while preserving the sparse structure of the optimal control problem for fast computation. Second, we address the lack of natural support forces in anti-gravity environments. We introduce lower bounds on the foot-normal component of the contact force, and smoothly transfer them during the doublesupport phase. Finally, we implement the proposed framework and demonstrate anti-gravity walking by a flying humanoid through simulation and a hardware experiment. To the best of our knowledge, this is the first demonstration of multi-contact whole-body MPC for a transformable aerial robot and walking by a flying humanoid beyond the ground.

### 论文解读
#### 摘要翻译
本文提出实时全身模型预测控制（WB-MPC），让飞行人形机器人在天花板等“反重力”环境行走。方法以推力的时间导数（推力率）而非推力本身作为控制输入，保证推力连续；同时设置脚部法向接触力下界并平滑转移负载。仿真和硬件实验验证了多接触全身控制。

#### 方法动机分析
天花板行走要求旋翼持续把机器人压向顶面，才能依靠摩擦移动。接触切换时，直接把推力作为输入会出现无法由硬件跟踪的尖峰；另加速率约束又可能破坏优化问题的稀疏性。论文的核心假设是把推力纳入状态、用推力率驱动它的连续演化，并用法向力下界维持贴面安全。

#### 方法设计详解
输入是当前构型q、速度v、各旋翼推力λ以及接触模式；它们组成增广状态x=[q,v,λ]。控制模块输出推力率dλ/dt和关节力矩τ，再由全身动力学模块计算下一状态，并输出下一时刻的关节命令与旋翼推力。动力学同时考虑惯性、重力、旋翼雅可比、关节力矩和接触力；离散更新为λ下一时刻=λ当前+推力率×Δt，使预测轨迹中的推力天然连续。接触模块施加摩擦约束和法向力下界Fmin；双支撑期间，负载转移模块以进度ρ在上一支撑模式和下一模式之间插值，把压力逐步从旧支撑脚转给新支撑脚。优化模块采用扩展Crocoddyl的BoxFDDP，热启动滚动求解1秒、40节点预测窗；控制周期100 Hz，离散步长0.025秒，仿真和硬件分别使用4、8线程。

#### 方法对比分析
基线直接优化推力，接触拓扑变化时容易振荡。本文把连续性写入状态转移，并将法向力下界与负载转移结合起来；这比单纯事后限制推力变化更适合实时稀疏求解。该机制尤其适用于需要持续压紧表面、频繁切换接触的飞行/移动机器人。

#### 实验分析（精简版）
MuJoCo仿真中，机器人质量1.6 kg、最大推力20 N、摩擦系数0.7。轨迹优化的直接推力基线出现约3 N推力尖峰、约5 N接触力尖峰并需26次迭代；推力率方案消除尖峰，16次迭代收敛。Fmin=5 N时平均求解6.69 ms，91.5%低于10 ms；硬件平均5.47 ms，在约8.5秒内沿天花板走0.15 m、完成4步，姿态误差小于0.1 rad。Fmin=0或直接推力输入会造成接触不稳并脱离。局限是高减速舵机限制关节力控制，且尚无终端等式约束。

#### 实用指南
作者称在开源Crocoddyl上扩展了任意旋翼分布和推力率状态增广，但未提供独立代码链接。复现应采用1秒/40节点、25 ms步长、100 Hz控制和BoxFDDP热启动，并实现推力、力矩盒约束、摩擦约束及Fmin插值。迁移到其他机器人时需重建质量惯量、旋翼/接触雅可比、推力上限和摩擦参数，并重新调节跟踪权重与Fmin。

#### 总结
核心思想：推力率让反重力行走平滑可控。
1. 增广状态记录当前推力。
2. 推力率与力矩共同驱动全身动力学。
3. 法向力下界和插值实现平滑换脚。
4. BoxFDDP滚动求解并实时执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.07544v1)
- [arXiv](https://arxiv.org/abs/2609.07544v1)

---

