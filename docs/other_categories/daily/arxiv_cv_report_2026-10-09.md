time: 20261009

# Arxiv Computer Vision Papers - 2026-10-09

## Table of Contents

1. [RAGNAROK: Radar-Aided Gravity-Normalized Alignment for Robust Open Keyframe-based Radar-Visual-Kinematic-Inertial SLAM](#2610.11531v1)
2. [WARP-VLA: Wrist-Camera Adaptation for View-Robust Policy Execution in Vision-Language-Action Models](#2610.11508v1)
3. [Distributed Relative Localization Based on Ultra-WideBand and LiDAR for Multi-robot with Limited Communication](#2610.11141v1)
4. [Distributed Relative Localization for Homogeneous Multi-Robot Systems through UWB Ranging and Limited Communications](#2610.11308v1)
5. [SimVLA: Zero-Shot Sim-to-Real VLA Learning for Mobile Manipulation](#2610.11248v1)
6. [DAMP: Humanoid Locomotion via Denoised Belief Learning and Adversarial Motion Priors](#2610.11505v1)
7. [USDCraft: Geometrically Grounded Programmatic Modeling of Articulated 3D Assets for Simulation](#2610.11322v1)
8. [CRISP: Fixing Flying Pixels in Latent LiDAR Generation via Diffusion Decoding](#2610.11376v1)
9. [CoCam4D: Geometry-Aware Cooperative 4D Perception for Camera-Only Autonomous Driving](#2610.11577v1)

---

## Papers

<a id='2610.11531v1'></a>
## [RAGNAROK: Radar-Aided Gravity-Normalized Alignment for Robust Open Keyframe-based Radar-Visual-Kinematic-Inertial SLAM](https://arxiv.org/abs/2610.11531v1)

**Authors:** Hanjun Kim, Chiyun Noh, Sangwoo Jung, Jaehyung Jung, Simon Boche, Cedric Le Gentil, Stefan Leutenegger, Ayoung Kim

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Legged robots offer superior mobility in unstructured environments, but reliable operation in such conditions requires robust state estimation. To address the vulnerability of proprioceptive estimators in rough terrain, recent methods have incorporated radar to provide velocity measurements. However, their limited yaw observability still leads to drift, and failure-aware fusion for adverse environments remains underexplored. In this letter, we present RAGNAROK, the first radar-visual-kinematic-inertial SLAM designed for robust operation in challenging environments. It integrates slip- and rolling-contact-aware leg velocity estimation, a kinematics-aware radar factor, and degradation-aware image enhancement. We further incorporate a B-spline-based radar-aided proprioceptive backbone, adaptive weighting, and online extrinsic calibration. Extensive experiments on public and self-collected datasets demonstrate that RAGNAROK achieves robust performance under challenging conditions and outperforms state-of-the-art baselines. The source code and dataset are available at https://github.com/hanjun815/RAGNAROK.

### 论文解读

#### 摘要翻译

足式机器人在非结构化环境中具有优越的机动性，但在这些条件下可靠运行需要鲁棒的状态估计。针对本体感知估计器在崎岖地形中的脆弱性，近期方法引入雷达提供速度测量。然而，其有限的偏航角可观测性仍会导致漂移，而且面向恶劣环境的故障感知融合尚未得到充分探索。

本文提出 RAGNAROK，首个面向挑战性环境鲁棒运行而设计的雷达—视觉—运动学—惯性 SLAM 系统。它集成了感知打滑与滚动接触的腿部速度估计、运动学感知雷达因子，以及退化感知图像增强。系统进一步结合基于 B 样条的雷达辅助本体感知主干、自适应加权和在线外参标定。在公开数据集和自采数据集上的大量实验表明，RAGNAROK 在挑战性条件下具有鲁棒表现，并优于最先进的基线。源代码和数据集地址为：https://github.com/hanjun815/RAGNAROK。

#### 方法动机分析

**核心问题不是传感器不足，而是不同传感器会以不同方式失效。**

- **腿部运动学**：足端打滑、接触冲击和圆足滚动会破坏“支撑足静止”的假设；多条腿同时打滑时，单纯比较腿间速度也无法发现共同偏差。
- **雷达与惯性**：雷达提供直接速度约束，局部重力改善横滚和俯仰估计，但重力不能约束偏航，长期水平漂移仍然存在。
- **视觉**：能够提供相对姿态、回环和地图约束，却容易在黑暗、眩光、重复纹理及剧烈运动下退化。
- **固定权重融合**：可能让退化观测持续污染状态；直接丢弃观测又会损失尚可利用的信息。

因此，作者的设计是：**先构建能在视觉失效时持续工作的雷达辅助本体感知主干，再依据视觉可靠性动态引入视觉约束。**

核心假设包括：雷达中仍有足够的静态目标；腿部接触信息具有一定可信度；共同打滑主要位于局部支撑平面；部分时段仍能获得可用于匹配和几何验证的视觉特征。系统缓解的是相对偏航漂移，并非在无外部参考时获得绝对航向可观测性。

#### 方法设计详解

##### 1. 输入与双层状态表示

完整系统输入为双雷达、双目图像、IMU，以及关节位置、关节速度、关节力矩或驱动力信息、接触状态和接触法向。

系统分为两层：

- **雷达辅助本体感知层**：用连续时间 B 样条表示姿态 \(R(t)\)、机体系速度 \(v(t)\) 和局部重力 \(g(t)\)，同时估计二维打滑偏置与传感器外参。
- **外感知 SLAM 层**：在相机时间戳上估计离散位姿、速度、惯性偏置，并维护路标、关键帧和子地图。

连续时间表示的作用是：在任意相机时间戳查询运动，避免直接拼接不同频率的传感器输出。

##### 2. 腿部因子：分别处理滚动、局部失效和共同打滑

**输入 → 接触点运动学 → 可靠性加权速度 → 腿部残差。**

作者不直接把足心当作静止接触点，而是根据足半径 \(r_s\) 和地形法向 \(n_s\) 构造支撑点：

\[
p_{C_s}=p_{F_s}-r_sn_s,\\qquad
J_{c,s}=J_{p,s}+[r_sn_s]_\\times J_{\\omega,s}.
\]

这样能够显式补偿圆足滚动造成的接触速度。每条支撑腿给出：

\[
v_{B,s}=-J_{c,s}\\dot\\alpha_s+[p_{C_s}]_\\times\\omega(t).
\]

其中角速度来自优化后的姿态样条导数，而不是直接代入原始陀螺仪读数。

随后通过协方差膨胀降低不可靠腿的权重：

\[
\\Sigma_s=\\Sigma_{0,s}+
\\left(\\frac{w_\\tau}{\\tau_s+\\epsilon}
+w_e\\Delta e_s+w_dd_s\\right)I.
\]

三项分别对应刚触地、关节作用力突变和腿间速度不一致。该机制适合处理个别腿的异常。

对于**多条腿一致打滑**，作者另行估计二维偏置 \(b_v\)，并用支撑平面的切向基 \(T\) 将其映射为三维速度 \(Tb_v\)。这将“腿间不一致”和“共同切向滑动”分开建模，避免前者检测不到后者。

##### 3. 雷达因子：按运动方向传播测角误差

**雷达目标位置与多普勒速度 → 视线方向及其协方差 → 加权径向速度残差。**

两个雷达采用正交安装，其中一个绕前向轴旋转 \(90^\circ\)，以互补其不对称角分辨率。预测的径向速度同时考虑机体平移和安装杆臂产生的转动速度：

\[
r_R=-d^\\top R_{IR}^\\top
\\bigl(v+\\omega\\times p_{IR}\\bigr)-\\tilde v_m.
\]

关键新机制是将方向不确定性投影到当前速度方向：

\[
\\sigma_d^2=\\sigma_{vd}^2+
v_R^\\top\\Sigma_d^Iv_R.
\]

直观上，**同一个雷达点，在不同运动方向和速度下，对多普勒预测造成的误差不同**，因此不能只使用固定雷达噪声。优化按 \(1/\\sigma_d^2\) 加权，并通过 Cauchy 鲁棒损失抑制动态目标等异常值。

雷达、腿部、陀螺仪、局部重力及相关先验共同约束样条；外参通过带先验的在线优化更新。

##### 4. 图像增强：优化可用特征，而非只优化亮度

**原始图像 → AGCWD/CLAHE 增强与融合 → 特征感知效用 → 贝叶斯优化。**

作者融合自适应伽马校正 AGCWD 与 CLAHE，融合比例由增强结果的平均亮度和直方图熵决定。贝叶斯优化调整：

- AGCWD 的加权参数 \(\\eta\)；
- CLAHE 的裁剪阈值 \(\\kappa\)。

优化目标为：

\[
U_{\\mathrm{SFA\\text{-}EWG}}
=U_{\\mathrm{P\\text{-}EWG}}
(1+\\lambda_fQ_fQ_s).
\]

其中：

- \(U_{\\mathrm{P\\text{-}EWG}}\) 衡量熵加权梯度，并惩罚饱和区域，保证分数非负；
- \(Q_f\) 衡量 BRISK 特征数量；
- \(Q_s\) 衡量特征空间分布均匀性。

这使增强目标与 SLAM 的特征需求对齐，而不只是让图像“更亮”。仅当原始图像效用变化超过阈值时重新优化，否则复用参数。

##### 5. 本体感知辅助视觉与自适应后端

增强图像进入视觉前端后：

1. 用局部重力统一 BRISK 描述子方向；
2. 用样条运动初始化相对位姿；
3. 通过该运动约束形成的极线几何剔除不一致匹配；
4. 双目及跨关键帧三角化生成路标；
5. 在近静止阶段抑制冗余关键帧。

后端基于 OKVIS2-X，新增本体感知相对位姿和速度因子，与重投影、IMU、位姿图及地图对齐约束共同优化。

视觉质量 \(Q_i\) 由跟踪点数量及其图像覆盖率构成；不足 8 个点时置零。本体感知信息矩阵按

\[
\\alpha_i=\\frac{1}{\\max(Q_i,0.1)}
\]

放大，单就这一项最多提高至 10 倍。进入视觉退化模式后，系统进一步提高本体感知权重，并降低视觉及地图相关约束权重。

回环候选由 DBoW2 分数与雷达极坐标 BEV 图像的 FFT 互相关分数联合排序，最后仍需通过 3D–2D RANSAC 几何验证。输出包括机器人轨迹、路标和由预测深度融合形成的局部占据地图。

**训练边界**：主体是在线非线性优化系统，不是端到端学习模型。深度预测及融合模块沿用后端体系；本文未给出独立训练配方。

#### 方法对比分析

| 对比对象 | 本质差异及解决的问题 |
|---|---|
| GaRLILEO | 继承连续时间局部重力估计，但新增接触与雷达可靠性建模，并接入视觉回环和建图，缓解仅靠重力与速度约束的长期偏航漂移。 |
| Co-RaL | 不通过接触坐标系的时间运动再预积分处理滚动，而是在地形法向定义的支撑点直接建立解析速度约束。 |
| Pronto、MUSE 等融合方法 | 从较固定的融合策略转向腿部可靠性、雷达状态相关噪声和视觉质量驱动的多层自适应加权。 |
| OKVIS2-X | 保留其标准视觉惯性与地图后端，新增本体感知运动因子，并让重力和运动先验进入特征提取、匹配筛选及回环候选排序。 |

**机制创新**主要在接触建模、两类打滑处理、运动相关雷达协方差和特征感知增强目标。B 样条、BRISK、DBoW2、CLAHE、贝叶斯优化及因子图本身属于已有组件；贡献在于针对退化问题的改造与耦合。

适合具备雷达、视觉和腿部状态接口的足式机器人，尤其是夜间、楼梯、崎岖地面及混合光照场景。对于永久无有效视觉特征的场景，系统仍可依靠本体感知主干运行，但视觉回环与建图能力会受限。

#### 实验分析（精简版）

**验证协议**：自采数据包含 14 条室内、室外及混合场景序列，覆盖黑暗、眩光、楼梯、山地和重复走廊。真值由 PALoc 结合 Leica RTC360 地面激光扫描地图生成，初始位姿通过 ICP 精化。报告平移和旋转绝对位姿误差 RMSE；室外结果明确使用 5 次运行的中位数。另在缺少视觉的 Co-RaL、GaRLILEO 数据集上测试 RKI 子系统，并进行组件消融。

**最有支持力的两个结论：**

1. **完整融合提高平均精度，但视觉并非始终有益。**  
   室外平均平移误差：RAGNAROK-RKI 为 **1.079 m**，完整 Odom 为 **0.481 m**，SLAM 为 **0.387 m**。但室内及混合场景中，RKI 与 Odom 分别为 **0.275 m**、**0.276 m**，几乎持平。证据支持“视觉在有信息时改善精度，并增加回环与地图能力”，而非“增加视觉必然更准”。

2. **退化管理比单纯增加传感器更关键。** 三条消融序列上，完整 Odom 平均平移误差为 **0.272 m**；去掉图像增强后为 **0.675 m**，去掉自适应权重后为 **2.220 m**，对应论文报告的约 **59.7%** 和 **87.7%** 降幅。不过这些比例来自有限序列，不能直接视为所有环境的平均收益。

**效率与局限**：在 i7-14700 和 RTX 4080 16 GB 上，Terrace 平均处理时间为 **49.5 ms/帧**，可持续处理 15 Hz 图像；端到端延迟为 **105.5 ms**，不能与单帧处理时间混淆。实验证据尚未覆盖低功耗机载平台或不同机器人平台的广泛迁移。两个估计层共享部分惯性信息，论文采取降权措施缓解过度自信，但未提供严格的跨层相关性建模或一致性证明。

#### 实用指南

- **代码与数据**：论文声明源代码和数据集已在 https://github.com/hanjun815/RAGNAROK 发布；仓库具体内容、许可证及深度网络权重是否齐全，论文未说明。
- **优先复现路径**：先运行 RKI 子系统，确认雷达径向速度符号、坐标变换和腿部运动学正确，再接入视觉与地图后端。Co-RaL、GaRLILEO 只能验证无视觉配置，不能替代完整 SLAM 评测。
- **必要输入处理**：需要统一时间戳、相机标定、雷达及机体到 IMU 的外参初值，并提供足半径、接触法向、关节速度和接触状态。在线外参优化不能替代合理的初始标定。
- **已知设置**：正交双雷达；BRISK 特征；Cauchy 雷达鲁棒损失；视觉少于 8 个跟踪点时质量置零；\(Q_{\min}=0.1\)；图像效用变化触发贝叶斯优化。
- **缺失配置**：样条阶数与节点间隔、窗口大小、雷达噪声参数、腿部可靠性权重、增强搜索范围与预算、退化阈值及增益的完整数值，论文未说明。
- **偏置处理需保持一致**：本体感知层不使用加速度计约束速度，因此将加速度计偏置设为零；外感知层保留 IMU 因子并将加速度计偏置建模为随机游走。两层均忽略陀螺仪偏置，这可能影响长时运行。
- **迁移到其他机器人**：必须替换正向运动学、接触点雅可比、足形和接触法向接口，并重新设定可靠性权重。更换雷达需更新测角噪声和安装外参。主体无需重新训练；更换相机或环境后，深度网络是否需要重训，论文未说明。
- **评估注意**：区分单雷达单目 RVI 与双雷达双目完整配置；同时报告失败率、轨迹误差、吞吐率及端到端延迟。论文未完整交代轨迹对齐方式和失败判定阈值，复现时应明确记录。

#### 总结

核心思想：以退化感知融合互补运动约束
1. 在地形支撑点建立滚动接触约束，用腿部可靠性和切平面偏置分别处理局部异常与共同打滑。
2. 按当前运动传播正交双雷达的测角不确定性，联合优化速度、姿态及局部重力样条。
3. 以特征数量和空间均匀性优化图像增强，用重力与连续时间运动辅助描述子和匹配筛选。
4. 将样条运动注入视觉 SLAM，根据视觉可靠性调整融合权重，以雷达辅助回环排序并完成轨迹与地图优化。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11531v1)
- [arXiv](https://arxiv.org/abs/2610.11531v1)

---

<a id='2610.11508v1'></a>
## [WARP-VLA: Wrist-Camera Adaptation for View-Robust Policy Execution in Vision-Language-Action Models](https://arxiv.org/abs/2610.11508v1)

**Authors:** Junmyeong Lee, Dongmin Shin, Min-Gyu Park, Wooseok Jeon, Inho Chang, Hae-Gon Jeon

**Published:** 2026-10-08

**Categories:** cs.RO, cs.CV

**Abstract:**

Despite recent advances in Vision-Language-Action models (VLAs) for robotic manipulation, their performance remains sensitive to changes in camera configuration. The problem becomes more evident in cross-setup deployment, as reproducing the exact camera pose used for training is nearly impossible. Unlike fixed external views, wrist views are more challenging because the camera moves with the robot, causing even small mounting variations to alter fine-grained geometric cues. To address this, we propose WARP-VLA, a camera-view robust VLA for diverse wrist camera configurations. WARP-VLA adopts a Mixture-of-Experts (MoE) architecture where individual experts learn view-specific feature transformations, and a router combines them based on implicit view information. This allows the policy to be deployed without requiring camera extrinsic parameters as additional input. Through experiments on the LIBERO benchmark, WARP-VLA improves the average success rate of pi-0.5 from 39.2% to 78.3% under wrist-view perturbations. The real-robot experiments further show that the feature-level adaptation learned in simulation successfully transfers to diverse deployment settings. To facilitate reproducibility and future research, we release our wrist viewpoint robustness benchmark and a plug-and-play implementation.

### 论文解读

#### 摘要翻译

尽管近年来视觉–语言–动作模型（Vision-Language-Action Models，VLAs）取得了进展，其性能仍对相机配置变化敏感。这一问题在跨配置部署中更加明显，因为几乎不可能精确复现训练时使用的相机位姿。与固定外部视角不同，腕部视角更具挑战性：相机会随机器人运动，即使微小的安装差异也会改变细粒度几何线索。

为此，我们提出 WARP-VLA，一种适用于多种腕部相机配置、具有视角鲁棒性的 VLA。WARP-VLA 采用混合专家（Mixture-of-Experts，MoE）架构，各专家学习视角特定的特征变换，路由器根据隐式视角信息组合专家。这使策略部署时无需将相机外参作为额外输入。在 LIBERO 基准实验中，WARP-VLA 将 π0.5 在腕部视角扰动下的平均成功率从 39.2% 提高到 78.3%。真实机器人实验进一步表明，在仿真中学习的特征级适配能够成功迁移到多种部署配置。为促进可复现性与后续研究，我们发布腕部视角鲁棒性基准和即插即用实现。

#### 方法动机分析

- **核心问题**：腕部相机安装位置改变后，物体在图像中的位置与夹爪实际位置之间的对应关系发生变化。策略可能仍将物体对准相机，却没有对准夹爪。
- **现有方法痛点**：相机条件化依赖准确标定；新视角合成需要额外生成模型，且大幅视角变化时重建可能不可靠。仅加入几何先验，也不等于解决腕部安装变化。
- **核心假设**：冻结视觉编码器的特征包含足够的隐式视角信息；不同安装配置造成的特征偏移，可以由共享变换与少量视角专用修正共同表示。
- **设计目标与边界**：不重训动作策略、不在推理时输入相机参数，而是在特征空间恢复策略熟悉的标准视角表示。它解决的是**可观测内容的视角偏移**，不能恢复完全遮挡或移出视野的信息，也不保证超出训练相机分布后的鲁棒性。

#### 方法设计详解

**1. 构造配对视角数据与标准锚点**

使用 LIBERO 的 2,000 条示范，覆盖四个任务套件中的 40 个任务。保留每条轨迹的标准腕部图像，并在十种扰动相机配置下重新渲染，其他条件不变，从而获得同一场景状态的标准／扰动观测对。

训练扰动范围为：

- 半圆平移区域：\(r\leq6\) cm，\(\theta\in[0^\circ,180^\circ]\)；
- 倾斜范围：\(\tau\in[0^\circ,10^\circ]\)。

冻结图像编码器，提取标准特征 \(X^c\) 和扰动特征 \(X^p\)，均为 \(S\times D\) 个特征元素。将基础策略微调数据中的所有标准腕部特征取平均，得到伪标准锚点 \(A\)。

**锚点不是当前场景的标准图像**，而是跨场景平均的参考表示，旨在保留相机配置相关结构、减弱任务内容差异。

**2. 从特征统计量隐式判断视角变化**

路由器输入为：

\[
z=\Phi_{\mathrm{stat}}(X^p-A)\oplus\Phi_{\mathrm{stat}}(A),
\]

其中

\[
\Phi_{\mathrm{stat}}(Y)=
\mu(Y)\oplus\sigma(Y)\oplus m_x(Y)\oplus m_y(Y).
\]

均值和标准差描述通道响应分布，两个一阶空间矩描述响应在图像水平／垂直方向上的分布。加入空间统计，是为了避免全局池化丢失与相机偏移有关的位置线索。

\(z\in\mathbb{R}^{8D}\) 经 LayerNorm 和 MLP 输出专家 logits。训练时，作者还随机混合全局锚点与任务级标准锚点，降低路由器对单一锚点的依赖。

**3. 因子化路由组合视角专家**

共设置 15 个专家，分成两个组：

- 13 个平移专家：覆盖半圆平移空间中的不同半径与角度区域；
- 2 个倾斜专家：覆盖不同倾斜范围。

每组单独 softmax，再将组内概率乘以 \(1/2\)：

\[
w_e=\frac{1}{G}
\frac{\exp(\ell_e)}
{\sum_{j\in\mathcal E_{g(e)}}\exp(\ell_j)},\qquad G=2.
\]

这样平移与倾斜专家不会争夺同一个概率预算，并可组合表示两类扰动。论文描述的是加权混合，未说明使用 top-k 稀疏专家选择。

**4. 共享交叉注意力与专家低秩修正**

每个专家以扰动特征为 query、锚点为 key/value，输出标准视角特征估计：

\[
F_e(X^p,A)=\operatorname{Transformer}_e(X^p,A).
\]

为避免复制完整 Transformer，各专家共享基础投影，只增加低秩残差：

\[
W_{e,m}^{l}=W_m^l+A_{e,m}^lB_{e,m}^l.
\]

共享部分学习不同视角共通的恢复规律，低秩部分学习局部视角修正。低秩因子的后一项初始化为零，使所有专家从相同变换起步。

最终输出为：

\[
\hat X^c=\sum_e w_eF_e(X^p,A).
\]

适配器置于图像编码器与动作策略之间，只替换腕部图像 tokens，保持其形状；视觉编码器与动作策略均冻结。

**5. 以重建质量和位姿先验分配训练责任**

先独立计算每个专家的标准特征重建误差：

\[
d_e=\frac{1}{SD}\|F_e(X^p,A)-X^c\|_F^2.
\]

再结合组内路由概率 \(g_e=Gw_e\)、训练时已知相机扰动产生的软掩码 \(T_e\)，计算专家责任：

\[
h_e\propto g_eT_e\exp\left(-\frac{d_e}{2t^2}\right),
\]

并在每组内归一化。直观上，**位姿区域合适、当前恢复更准确、路由概率更高**的专家承担更多监督责任；重叠位姿区间让相邻专家在边界共享样本。

总目标包括：

\[
L_e=\frac1G\mathbb E\left[\sum_eh_ed_e\right],\qquad
L_g=-\frac1G\mathbb E\left[\sum_eh_e\log g_e\right].
\]

责任 \(h_e\) 停止梯度传播。重建损失监督各专家，而非只监督混合输出；路由损失让预测权重逼近责任分布。

**推理时**只需当前腕部图像与预计算锚点，不需要标准视角配对图像、相机位姿先验或外参。

#### 方法对比分析

| 对比对象 | 本质差异与解决的痛点 |
|---|---|
| KYC 等相机条件化方法 | 不输入显式相机几何，而从视觉特征统计量估计专家权重，减少部署标定需求 |
| AnyCamVLA 等新视角合成方法 | 不生成标准视角图像，直接恢复冻结策略使用的特征，绕开像素级重建 |
| 几何增强 VLA | 不以增强一般空间理解为主要目标，而专门对齐扰动腕部视角与标准策略表示 |
| 单一适配器 | 用多个低秩专家表示异质视角变化，再按观测动态组合，避免所有配置共用一个固定变换 |

**真正的新机制**是锚点引导的统计路由、平移／倾斜分组专家，以及结合位姿先验和重建误差的责任分配。Transformer、LoRA 和 MoE 本身是已有组件，贡献在于如何围绕腕部视角恢复组织它们。

该方法适合已有可靠标准视角策略、可获取标准示范并能生成配对多视角训练数据的场景。“无需标定”仅指推理阶段；训练仍使用已知相机扰动作为监督。

#### 实验分析（精简版）

**验证协议**

LIBERO 的小／中／大扰动分别采用 2／4／6 cm 平移及 3°／6°／9°倾斜。每级包含七个方位角与两个倾斜取值，共 14 种配置；40 个任务合计每级 560 次评估。LIBERO-Plus 每级评估 10,000 次，直接复用 LIBERO 训练的适配器。

真实实验使用 xArm6、Robotiq 2F-85、两台 D435 外部相机与一台 D405 腕部相机。标准配置及九种扰动配置下，每个任务进行五次测试，共十个任务。

**最重要的两个结论**

1. **特征适配显著缓解腕部视角偏移，但多专家贡献应与整体适配收益区分。**  
   LIBERO 上，π0.5 的扰动平均成功率从 **39.2% 提升至 78.3%**，高于 AnyCamVLA 的 **72.3%**。单一共享适配器已达 **74.8%**，均匀专家权重为 **74.5%**：说明主要收益来自特征恢复，多专家与动态路由进一步贡献约 3.5–3.8 个百分点。

2. **迁移有效，但真实部署提升更有限。**  
   LIBERO-Plus 上，π0.5 从 **28.8% 提升至 56.5%**，AnyCamVLA 为 **54.7%**。真实机器人扰动平均成功率从 **36.7% 提升至 48.7%**，AnyCamVLA 为 **40.7%**。真实部署保持适配器权重不变，但重新计算了真实示范的标准锚点，并非完全无需目标域数据。

**证据边界**：大扰动下仍有明显失败；遮挡和夹爪不可见可能导致提前松爪。论文未报告置信区间、推理延迟或显存开销，因此不足以判断统计稳定性与实时部署成本。真实实验的平移和倾斜分别测试，对复合扰动的证据较有限。

#### 实用指南

- **开源状态**：摘要宣称发布基准和即插即用实现，但提供文本未给出可核验链接；代码、权重、数据的实际公开可用状态无法确认。
- **复现关键**：在同一轨迹状态下生成标准／扰动视角配对，冻结编码器提取特征，并使用基础策略标准微调集计算锚点。仅改变腕部安装配置，避免混入其他域变化。
- **主要配置**：
  - 共享网络：8 个 ViT block，隐藏维度 2,048；
  - 注意力模块：256 维、4 个头；
  - 15 个专家，低秩 rank 为 16；
  - 路由输入 16,384 维，MLP 隐藏维度 2,048；
  - Adam，学习率 \(3\times10^{-4}\)，batch size 4,096，训练 10,000 步；
  - 8 张 NVIDIA RTX PRO 6000 GPU，约 30 小时。
- **未完整说明的细节**：温度 \(t\)、软位姿掩码的具体参数、锚点混合比例、部分预处理与软件版本未明确给出，严格复现需补足这些设定。
- **迁移新骨干**：论文为每个 VLA 骨干分别训练适配器，不能把“即插即用”理解为一套权重直接兼容任意特征空间。
- **迁移新机器人／任务**：先保证基础策略具备标准配置下的任务能力，再用目标域标准示范重算锚点。论文展示了固定仿真适配器的迁移；对超出训练安装范围、不同相机成像条件或显著遮挡变化，是否需要重训及效果如何，论文未说明。
- **评估注意**：保持任务、初始化与非腕部观测一致，分别报告标准视角及不同扰动等级；不能仅用扰动平均值掩盖大扰动失败。

#### 总结

核心思想：以隐式视角专家恢复标准特征
1. 将标准腕部特征跨场景平均为锚点，为扰动特征提供固定参考。
2. 提取残差与锚点的通道统计及空间矩，分别预测平移和倾斜专家权重。
3. 用共享交叉注意力与专家低秩残差恢复标准表示，通过位姿先验和重建质量引导专家分工。
4. 推理时混合专家输出并替换腕部 tokens，让冻结策略无需相机外参即可执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11508v1)
- [arXiv](https://arxiv.org/abs/2610.11508v1)

---

<a id='2610.11141v1'></a>
## [Distributed Relative Localization Based on Ultra-WideBand and LiDAR for Multi-robot with Limited Communication](https://arxiv.org/abs/2610.11141v1)

**Authors:** Zhiqiang Cao, Ran Liu, Billy Pik Lik Lau, Chau Yuen, U-Xuan Tan

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Relative localization is crucial for a multi-robot system to collaboratively perform tasks, such as exploration and formation. However, this is highly challenging for homogeneous robots with similar appearance in GPS-denied and communication-limited environments. In this paper, we propose a fully distributed relative position estimation approach for a team of robots based on onboard UWB and LiDAR sensors, in which LiDAR is utilized to obtain the position of anonymous objects in Line-of-Sight (LOS), and UWB is used for ranging between robots. We construct two graphs, namely UWB connection graph and LiDAR connection graph, to represent the spatial relationship among objects (robots and obstacles) based on UWB and LiDAR measurements. Identification and relative position estimation are formulated as a common subgraph matching problem. A falsely-matched robot identification approach is designed to recognize the falsely-matched results caused by obstacle blockage in LiDAR field of view. These robots are then localized by leveraging the well-matched robots and the UWB ranging measurements in the UWB connection graph. We conducted experiments to evaluate the performance of our approach. The results show that the proposed approach is capable of achieving satisfactory positioning accuracy for a team of robots in a distributed manner with only exchanging limited information.

### 论文解读

#### 摘要翻译

相对定位对于多机器人系统协同执行探索、编队等任务至关重要。然而，在 GPS 不可用、通信受限的环境中，为外观相似的同构机器人实现相对定位极具挑战。本文提出一种基于机载超宽带（UWB）与激光雷达（LiDAR）的全分布式机器人团队相对位置估计方法。其中，LiDAR 用于获取视距（LOS）条件下匿名物体的位置，UWB 用于机器人之间的测距。我们构建 UWB 连接图与 LiDAR 连接图，根据两类传感器测量表示物体（机器人和障碍物）之间的空间关系，并将身份识别与相对位置估计表述为公共子图匹配问题。针对障碍物遮挡 LiDAR 视野所导致的错误匹配，我们设计了错误匹配机器人识别方法；随后，利用正确匹配的机器人以及 UWB 连接图中的测距数据，对这些机器人进行定位。实验结果表明，该方法仅需交换少量信息，即可通过分布式方式为机器人团队提供令人满意的定位精度。

#### 方法动机分析

**核心矛盾是身份、位置和通信成本无法由单一传感器同时解决：**

- LiDAR 能输出局部坐标系中的目标位置，却难以区分外观相似的机器人，也无法观测被遮挡目标。
- UWB 提供唯一节点身份及机器人间距离，但单次距离没有方位信息，不能直接确定相对位置。
- 交换图像、点云或连续里程计可辅助关联，但依赖较好的通信条件；多 UWB 节点方案又增加机器人尺寸和硬件负担。

本文的设计理由是：**不用外观识别身份，而用多个目标之间的距离结构识别身份。** 每台机器人只交换少量 UWB 距离，通过匹配两个传感器描述的空间关系，将有身份的 UWB 节点与有位置的 LiDAR 簇关联。

关键假设与边界：

- 有足够多的可见机器人，且几何结构具有区分度；高度对称布局可能出现关联歧义。
- UWB 图需要足够丰富的边，通信过少会同时破坏匹配、校验和后续定位。
- 最大且距离最一致的公共子图被假定为可信基准，而不是经过独立验证的真值。
- 输出是机器人在本机坐标系中的**相对位置**，不是完整相对姿态；现有验证主要面向二维地面机器人。

#### 方法设计详解

**总体路径：本机 LiDAR 扫描、UWB 测距与邻居共享距离 → 双图构建 → 公共子图关联 → 错配筛查 → 遮挡目标定位。** 这是几何推理系统，没有端到端训练目标。

##### 1. 双图构建：分离身份与几何位置

- **UWB 图**：顶点是带唯一身份的机器人，边权来自本机测距和邻居广播的测距。通信缺失会造成边缺失。
- **LiDAR 图**：采用已有自适应聚类算法提取机器人及障碍物，删除明显大于机器人的簇。保留簇及本机作为顶点，利用簇位置计算两两距离，构成完全图。

LiDAR 簇身份是匿名的，且遮挡机器人可能不在图中；UWB 图则不包含普通障碍物。因此，不能要求两图整体一致。

##### 2. 公共子图匹配：利用距离结构建立身份关联

从本机在两图中的已知对应关系开始，执行深度优先递归搜索：

1. 为每个未匹配顶点维护标签序列，记录它到已匹配顶点的距离；缺失边记为无穷大。
2. 若一对候选顶点的标签序列逐元素满足
   
   |a_t-b_t|<\\lambda,
   
   就将其加入当前公共子图。
3. 更新其余顶点的标签序列，继续扩展；不能扩展时保存结果并回溯。

这里的 \\(\\lambda\\) 容忍 UWB 测距与 LiDAR 几何距离之间的误差。过小会漏配，过大则增加误配和搜索开销。

从候选公共子图中，先选顶点数最大的，再按距离不一致程度选最优：
\\[
D(M)=\\frac{1}{|M|}
\\sqrt{\\sum_{(v_i,u_l),(v_j,u_k)\\in M}(r_{ij}-d_{lk})^2}.
\\]

该分数越小，说明对应机器人之间的 UWB 距离与 LiDAR 簇间距离越一致。得到可信度最高的子图 \\(M^*\\) 后，再从其他子图补充未覆盖的零散对应 \\(O\\)，形成关联集合。

##### 3. 错配识别：用可信子图检验零散关联

将 \\(M^*\\) 中的机器人作为基准。对 \\(O\\) 中每个候选机器人，比较：

- 根据 LiDAR 位置计算的到基准机器人的距离；
- UWB 图中对应的实测距离。

归一化欧氏距离差超过阈值 \\(\\epsilon\\)，则拒绝该关联。这用于识别“被遮挡机器人错误对应到障碍物”等情况。

**边界在于：该机制主要验证零散关联，不会独立纠正 \\(M^*\\) 自身的错误。**

##### 4. 未估计机器人定位：从错误关联转为测距约束

对于未匹配或被拒绝的机器人 \\(j\\)，利用正确匹配机器人的位置作为锚点，求解：
\\[
\\hat{x}_{ij}
=\\arg\\min_x\\sum_{k\\in K}
\\left(r_{kj}-\\|x-\\hat{x}_{ik}\\|\\right)^2.
\\]

这是标准非线性最小二乘测距定位：寻找一个位置，使其到各锚点的距离尽量符合 UWB 测量。论文要求锚点集合大小大于三，但未给出退化几何的处理方案。

**贡献分工：**自适应聚类和最小二乘是已有组件；主要机制是容许部分重叠的双图身份关联，以及“可信子图校验—拒绝错配—测距恢复”的闭环。

#### 方法对比分析

| 最接近的方法类型 | 本文的本质差异 |
|---|---|
| UWB 距离与单个 LiDAR 簇距离匹配 | 从单距离比较改为多顶点距离结构匹配，提高等距目标之间的区分能力，并补充遮挡目标定位 |
| LiDAR 轨迹与邻机里程计匹配 | 不依赖连续轨迹匹配和里程计共享，但转而依赖当前几何结构及 UWB 图连通性 |
| 多 UWB 节点相对定位 | 每机器人只需一个 UWB 节点，由 LiDAR 提供局部位置几何；代价是依赖足够多的可见目标 |
| 视觉或 LiDAR 协同定位 | 不交换图像、特征或点云，仅共享测距，但输出及观测条件也更受限制 |

创新不在于首次组合 UWB 与 LiDAR，而在于将**匿名目标识别转化为部分距离图匹配**，并将匹配结果进一步用于错误检查和遮挡目标恢复。

适合通信预算有限、存在部分遮挡、外观难区分的机器人团队；不宜据此直接推断其适用于极少可见锚点、严重 UWB 非视距偏差、高度对称布局或高速大规模集群。

#### 实验分析（精简版）

**验证协议：**实物实验使用 8 台 Turtlebot2，在约 \\(6\\times17\\) 米室内环境中布置障碍物，主要报告其他机器人相对 Robot1 的平均位置误差与解算率，约 130 次试验。另有 6–12 台机器人的仿真和单机器人以约 \\(0.1\\) 米/秒运动的动态仿真。比较主要是模块消融、共享比例、通信顺序及团队规模，**没有报告与已有定位算法的直接定量对照**。

最重要的两项证据：

1. **校验与重定位解决了仅匹配的明显失败。**  
   在完整共享的模块消融中，仅匹配时，遮挡的 Robot3 无解，Robot4 平均误差为 4.81 米。加入错配筛查和测距重定位后，两者分别达到 0.31 米和 0.50 米，解算率均为 100%。

2. **有限共享可以接近完整共享，但存在最低信息需求。**  
   共享比例为 71% 时，每机器人共享 10 字节距离载荷，可见机器人误差为 0.16–0.29 米，两个遮挡机器人分别为 0.31 米和 0.53 米，全部解算率为 100%。完整共享需要 14 字节，遮挡机器人误差为 0.31 米和 0.50 米，收益较小。但共享比例仅 14% 时，一个遮挡机器人的误差达到 20.58 米，说明“有解”并不意味着定位正确。

**主要优势：**实验支持低载荷下的身份关联与遮挡恢复。  
**主要局限：**实物场景规模有限，动态测试简单；仿真采用零均值高斯 UWB 噪声，不能充分代表严重非视距偏差。计算时间也随规模明显上升：6 台时为 0.107 秒，12 台时为 1.725 秒，因此尚不能据此宣称具备大规模实时性。

#### 实用指南

- **开源状态：**论文未提供本方法代码、实验数据或模型权重的明确公开链接，是否开源论文未说明。
- **硬件与依赖：**实物系统使用 RPLIDAR A3、Nooploop Linktrack 和 ROS；计算设备为 Intel Xeon W-10855M、32 GB 内存。UWB 采样频率为 50 Hz，这不是定位算法输出频率。
- **预处理：**执行 LiDAR 自适应聚类、剔除大簇，维护本机与匿名簇的局部坐标。簇尺寸阈值、机器人中心提取、传感器外参标定及时间同步细节，论文未说明。
- **推荐复现设置：**采用 \\(\\lambda=1.0\\) 米、\\(\\epsilon=0.2\\)，使用论文规定的固定顺序共享距离；8 机器人实验的主要设置为约 71% 共享比例。
- **通信核算：**每个距离用 2 字节表示，分辨率为 0.004 米。报告的是共享距离载荷，不是包含协议头、测距交互和重传的总带宽；广播频率论文未说明。
- **计算预算：**8 机器人、71% 共享条件下，匹配至重定位模块平均约 0.372 秒；LiDAR 聚类另需约 12 毫秒。复现时不能默认系统能跟随 50 Hz UWB 数据实时输出。
- **训练与求解细节：**核心方法不需要训练。非线性优化的初始化、停止条件，以及零散对应冲突的完整处理规则，论文未充分说明；缺失边与标签比较的实现也应单独核对。
- **评估注意事项：**同时报告位置误差、解算率和运行时间；不能把解算率当作正确匹配率。实物真值获取方式及误差统计的完整细节，论文未说明。
- **迁移建议：**更换机器人后，需要重新检查聚类尺寸过滤、中心位置估计和传感器标定，并根据噪声调整两个阈值。迁移至三维需要替换目标提取与位置表示；若引入学习式检测器，其训练属于新增工作，不是论文已有流程。

#### 总结

核心思想：以双图匹配连接身份与位置
1. 用带身份的 UWB 距离和匿名 LiDAR 簇分别构图，保留两者不完整重叠的空间关系。
2. 从本机已知对应出发搜索公共子图，以规模优先、距离一致性次优选出可信关联，并补充零散对应。
3. 将可信子图作为基准，通过跨传感器距离差拒绝可疑零散匹配。
4. 利用正确关联的机器人作锚点，以 UWB 非线性最小二乘恢复遮挡或未匹配机器人的相对位置。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11141v1)
- [arXiv](https://arxiv.org/abs/2610.11141v1)

---

<a id='2610.11308v1'></a>
## [Distributed Relative Localization for Homogeneous Multi-Robot Systems through UWB Ranging and Limited Communications](https://arxiv.org/abs/2610.11308v1)

**Authors:** Zhiqiang Cao, Ran Liu, Billy Pik Lik Lau, Chau Yuen, U-Xuan Tan

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Accurate and reliable relative localization is crucial for multi-robot applications like exploration, search, and rescue missions. LiDAR-based solutions offer high accuracy in localizing surrounding objects; however, distinguishing homogeneous robots with similar appearances remains challenging due to the lack of distinctive identification features. In this paper, we propose a fully distributed relative pose estimation approach by integrating LiDAR, UWB, and odometry measurements, allowing each robot to accurately and continuously localize its teammates without external infrastructure. Specifically, potential anonymous teammate robot clusters from LiDAR scans are tracked by a dynamic tracker. We then identify teammate robots from these tracked anonymous clusters using a joint matching strategy, ensuring reliable data association between robots and clusters. Finally, by combining the corresponding LiDAR observations, UWB ranging, and odometry measurements, each robot precisely localizes others while minimizing data exchange. The system requires only odometry data exchange through onboard UWB, eliminating the need for additional communication infrastructure like WiFi routers or mesh networks. Extensive simulation and real-world experiments demonstrate the effectiveness and reliability of the proposed relative localization approach.

### 论文解读

#### 摘要翻译

准确、可靠的相对定位对于探索、搜索与救援等多机器人应用至关重要。基于 LiDAR 的方案能够高精度地定位周围物体，但由于缺少独特的身份识别特征，区分外观相似的同构机器人仍然具有挑战性。本文提出一种融合 LiDAR、超宽带（UWB）和里程计测量的完全分布式相对位姿估计方法，使每个机器人无需外部基础设施即可准确、持续地定位队友。具体而言，首先使用动态跟踪器跟踪 LiDAR 扫描中潜在的匿名队友机器人点云簇；随后通过联合匹配策略，从这些被跟踪的匿名簇中识别队友机器人，确保机器人与点云簇之间可靠的数据关联；最后结合对应的 LiDAR 观测、UWB 测距和里程计测量，在尽量减少数据交换的同时精确定位其他机器人。系统仅需通过机载 UWB 交换里程计数据，无需 WiFi 路由器或网状网络等额外通信基础设施。大量仿真和真实环境实验验证了该相对定位方法的有效性与可靠性。

#### 方法动机分析

**核心矛盾是：LiDAR 能提供相对位置，却不知道“这是谁”；UWB 知道节点身份，却只能提供距离。** 同构机器人外观相似，单靠点云难以关联身份；仅靠 UWB 与里程计，又面临初始相对位姿未知和非线性优化初始化困难。

现有方案的具体不足包括：

- 共享点云、图像特征进行匹配，通信与计算成本较高。
- 给机器人安装反光材料、编码灯板等标记，会增加平台改造要求。
- 直接共享自定位结果，需要已知初始坐标系关系，而且容易受到里程计漂移影响。
- 单纯距离匹配或轨迹形状匹配，各自存在歧义；LiDAR 还会因遮挡而失去观测。

本文的思路是：**先用运动轨迹与身份绑定的距离序列共同识别队友，再将该关联产生的坐标变换用于融合优化初始化。**

问题边界也很明确：方法针对二维运动，依赖可用的初期 LiDAR 观测和足够运动；不是任意静止编队下都能立即恢复完整相对位姿。UWB 对遮挡更有韧性，但并不意味着没有非视距偏差。

#### 方法设计详解

系统无训练阶段。每个机器人独立运行相同流程，以自身初始位置建立固定参考系，最终输出其他机器人的二维相对位置与朝向。

**1. 输入与预处理：减少错误测量和候选对象**

输入为本机 LiDAR、轮式里程计、带身份的 UWB 距离，以及其他机器人通过 UWB 广播的里程计。

- **UWB 清洗：**在短时间窗口内检查测距方差，用有效数据的均值初始化低通滤波器。后续只有与上一滤波值之差满足最大运动速度约束的测量才更新；长时间未更新则重新初始化。
- **LiDAR 候选筛选：**采用现有自适应聚类算法生成点云簇。若某簇到本机的距离与所有队友 UWB 距离均相差超过 **1.0 m**，则剔除；再依据包围盒尺寸排除明显大于机器人的对象。

输出仍是匿名候选簇，而非已识别的机器人。

**2. 动态跟踪：将匿名检测转为运动序列**

为每个候选簇维护独立跟踪器，状态包含位置与朝向。簇位置由本机里程计转换到本机固定参考系；初始朝向随机设置，随后通过随机速度模型预测。

新检测与预测位置按最近欧氏距离关联，并设置距离门限。关联成功则更新，否则继续预测；连续过久未获观测便终止跟踪。由此获得可供身份识别的匿名轨迹。

**3. 联合身份识别：同时比较“怎么走”和“离我多远”**

首先保留最近一段轨迹，并均匀稀疏采样。对每个“队友里程计轨迹—匿名簇轨迹”候选对，用 **Kabsch/SVD 刚体配准**估计二维旋转和平移，使两条轨迹对齐，并修正可能出现的反射。

随后计算两种归一化动态时间规整（DTW）代价：

- \(T_{j,m}\)：对齐后的队友里程计轨迹与匿名簇轨迹的位置差异。
- \(D_{j,m}\)：队友 UWB 距离序列与匿名簇到观察机器人距离序列的差异。

联合评分为：

\[
J_{j,m}=T_{j,m}D_{j,m}.
\]

对队友 \(j\)，选择评分最小的候选簇；仅当最小评分不超过阈值 \(\lambda\) 时接受，实验采用 **\(\lambda=0.005\)**。

直观上，轨迹形状用于区分运动行为，距离历史用于补充空间位置约束。需要注意，**乘积评分并不等价于两个分数分别通过严格门限**；论文也未明确给出跨队友的一对一全局分配机制。

**4. 队友跟踪：关联结果同时提供观测和优化初值**

只有队友已被识别，且其一段时间内累计位移超过 **2.5 m**，才启动对应队友跟踪器。该条件旨在提高轨迹配准旋转估计的可靠性，但不是完整的可观测性证明。

跟踪器采用滑动窗口位姿图优化：

\[
\min_X
\sum \rho(e_{\mathrm{odom},i}^{2})
+\sum \rho(e_{\mathrm{odom},j}^{2})
+\sum \rho(e_{\mathrm{LiDAR}}^{2})
+\sum \rho(e_{\mathrm{UWB}}^{2}).
\]

约束分别来自双方里程计增量、已识别队友的 LiDAR 相对位置、带身份的 UWB 距离。每轮通过最新轨迹配准变换，将队友里程计转换到本机参考系，作为优化初值，避免反复随机采样初始化。

采用 g2o 和 Huber 鲁棒核；里程计、LiDAR、UWB 的核阈值分别为 **0.025 m、0.5 m、1.0 m**。信息矩阵由固定标准差构造：里程计 **0.005 m、0.005 rad**，LiDAR **0.1 m**，UWB **0.2 m**。

初始化之后，短时遮挡导致 LiDAR 约束缺失时，仍可使用 UWB 与里程计继续估计。最终将队友位姿转换到本机当前坐标系，输出相对位姿。LiDAR 和 UWB 都不直接提供相对朝向约束，朝向主要依赖运动及里程计。

**5. 通信：只传低维里程计**

每条广播为 **8 Bytes**：首尾标记各 1 Byte，\(x,y,\theta\) 各 2 Bytes。量化分辨率约为 **0.004**，位置编码范围为 **−127.996～127.996 m**，广播频率为 **20 Hz**。

#### 方法对比分析

| 对比方向 | 本文的本质差异 | 解决的问题与边界 |
|---|---|---|
| Swarm-LIO 等轨迹匹配方法 | 用 UWB 距离历史补充轨迹形状，不依赖反光标记 | 降低形状相似导致的误关联，但仍需足够轨迹观测 |
| 作者此前的 LiDAR–UWB 子图匹配 | 从瞬时几何关系转向时间序列联合匹配 | 缓解几何对称歧义；完全相同运动与距离历史下的识别能力未证明 |
| UWB＋里程计相对定位 | 利用 LiDAR 关联产生配准初值与位置约束 | 减轻随机初始化负担，但增加 LiDAR 感知要求 |
| Omni-Swarm 等多源融合系统 | 仅交换压缩里程计，无需共享视觉检测或地图特征 | 适合带宽受限平台，代价是观测信息较少且当前仅支持二维 |

主要新机制是**联合身份评分，以及身份关联—轨迹配准—优化初始化之间的衔接**。聚类、DTW、Kabsch、位姿图优化和 Huber 核本身都是已有组件；8 Bytes 广播属于面向部署的工程设计。

#### 实验分析（精简版）

真实实验使用三台 TurtleBot2，配备 Nooploop LinkTrack UWB、Velodyne VLP-16 和轮式里程计；LiDAR 仅使用单条扫描通道。机器人最大速度为 **0.2 m/s**，使用基于已有占据栅格地图的 AMCL 作为参考真值。另在含行人与障碍物的环境测试，并用 Gazebo 将团队规模扩展到十台。

**最有支撑力的两个结论：**

1. **联合匹配确实减少了身份误关联。**从机器人 1 的视角看，距离匹配对机器人 2、3 分别产生 **80、259** 次错误关联；纯轨迹匹配分别产生 **0、279** 次。联合匹配均为 **0** 次，同时保留 **1579、806** 次正确关联。证据支持两种信息的互补性，但不代表所有场景都不会误识别。

2. **多源约束提高了定位精度，且通信负担较小。**对机器人 3，仅里程计约束、加入 UWB、再加入 LiDAR 时，平均平移误差依次为 **0.31、0.21、0.14 m**，旋转误差为 **4.72°、3.34°、3.20°**。完整系统在结构化实验中的总体平均误差为 **0.126 m、2.275°**，三机器人广播数据率为 **0.48 KB/s**。但 SLAM Toolbox 的旋转精度更好，符合本文缺少显式朝向约束的限制。

主要局限是：真实验证仅有三台低速机器人；十台规模只经过仿真；参考真值不是独立运动捕捉系统；长期完全遮挡、严重 UWB 非视距偏差和退化运动下的性能边界尚不充分。通信数字按应用层消息计算，不应直接视为含测距协议与链路开销的无线总带宽。

#### 实用指南

- **开放资源：**论文未说明代码、模型或实验数据的开源地址；提及附加视频，但所给文本未提供明确链接。该方法无需训练或预训练模型。
- **运行环境：**论文使用 ROS Noetic、g2o，并在 E5-2620 v4 CPU 上评估。传感器频率为 LiDAR **10 Hz**、UWB 测距 **50 Hz**、本机里程计 **50 Hz**、里程计广播 **20 Hz**。完整约束下，每个队友跟踪器约耗时 **50 ms**。
- **优先复现参数：**候选距离门限 **1.0 m**、联合评分门限 **0.005**、初始化累计位移 **2.5 m**，以及上述鲁棒核与噪声标准差。
- **未充分说明的细节：**轨迹窗口长度、采样数、动态跟踪关联门限、终止步数、UWB 滤波系数与时间阈值、时间对齐及传感器外参处理，论文未说明完整可复现设置。
- **公式实现需核查：**配准伪代码对 \(s+1\) 个样本的质心却写为除以 \(s\)，且其平移公式与正文使用的变换方向存在表述不一致。实现时应统一源、目标坐标系，并用已知刚体变换验证。优化还需明确参考位姿固定方式与角度残差；正文公式并未完整展开。
- **迁移要求：**更换机器人时，应调整尺寸筛选、运动模型、速度门限、噪声和传感器外参；扩大场地时需修改位置编码范围。无需重训，但需要重新标定与调参。三维迁移还需重构状态、配准和约束，不能仅替换里程计传感器；论文尚未验证三维版本。

#### 总结

核心思想：联合轨迹与测距识别队友
1. 用带身份的 UWB 距离筛选 LiDAR 匿名簇，并将连续检测组织为本机参考系中的轨迹。
2. 配准队友里程计与候选轨迹，将轨迹 DTW 和距离 DTW 的乘积用于身份关联。
3. 在累计运动充分后，用关联得到的坐标变换初始化队友位姿图，融合双方里程计、LiDAR 和 UWB。
4. 仅通过 UWB 广播压缩里程计，在初始化后的遮挡阶段继续依靠测距与运动信息输出相对位姿。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11308v1)
- [arXiv](https://arxiv.org/abs/2610.11308v1)

---

<a id='2610.11248v1'></a>
## [SimVLA: Zero-Shot Sim-to-Real VLA Learning for Mobile Manipulation](https://arxiv.org/abs/2610.11248v1)

**Authors:** Kyoungin Baik, Youngwoon Lee

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Large-scale, diverse datasets have driven the success of LLMs and VLMs. But VLAs for robotics remain limited by the cost and complexity of real-world data collection. While simulation offers a scalable alternative, its potential for sim-to-real VLA learning in mobile manipulation remains largely underexplored. We introduce SimVLA, an end-to-end framework that trains VLAs entirely on synthetic simulation data without teleoperation for mobile manipulation. SimVLA is first pre-trained on two complementary simulation-derived datasets: SimAction, a large-scale robot action dataset spanning 35 diverse mobile manipulation tasks, generated by composing atomic skills, and SimVQA, which leverages privileged simulator state to provide spatial, geometric, and subtask-level visual-language supervision. We further post-train SimVLA on a mixture of SimAction and SimDeploy, a dataset collected from policy rollouts across diverse simulated environments. We evaluate SimVLA on tasks including restocking, pouring, and cleaning, and show zero-shot transfer to real-world mobile manipulation, including real home environments. SimVLA outperforms policies trained on 50 in-domain real-world demonstrations, suggesting that simulation can enable scalable sim-to-real mobile manipulation. We further demonstrate the value of multiple complementary forms of supervision for effectively leveraging simulation in VLA training.

### 论文解读

#### 摘要翻译

大规模、多样化的数据集推动了 LLM 和 VLM 的成功，但用于机器人的 VLA 仍受到真实世界数据采集成本与复杂性的限制。仿真提供了一种可扩展的替代方案，但其在移动操作中的仿真到现实学习潜力仍未充分探索。

SimVLA 是一个完全使用合成仿真数据、无需遥操作即可训练移动操作 VLA 的端到端框架。它先在两个互补数据集上预训练：SimAction 通过组合原子技能生成覆盖 35 种移动操作任务的大规模机器人动作数据；SimVQA 利用仿真器特权状态提供空间、几何和子任务级视觉语言监督。随后，模型在 SimAction 与多样化仿真环境策略轨迹组成的 SimDeploy 混合数据上后训练。实验覆盖补货、倾倒和清洁，并展示了向真实移动操作（包括真实家庭环境）的零样本迁移。

#### 方法动机分析

移动操作同时受部分可观测性、底盘与双臂耦合、长时序分布偏移和真实示范覆盖不足的限制。仅增加动作轨迹并不能保证策略学会空间理解或处理自身执行误差。

方法的核心假设是：仿真应同时提供三类互补信号——SimAction 补足动作技能，SimVQA 直接监督空间与任务语义，SimDeploy 覆盖策略执行后产生的真实部署状态。问题边界主要是可程序化、可校准的厨房类刚体移动操作；论文未验证柔性物体、难模拟接触或任意家庭任务。

#### 方法设计详解

输入是场景布局、物体、机器人、语言任务、视觉观测和本体感知，输出是双臂末端姿态与底盘速度等连续动作。

1. **可迁移仿真环境。** Scene Synthesizer 生成多种厨房布局，Isaac Lab 随机化家具、物体、柜体和材质，并使用 Objaverse 几何实例。混合系统辨识同时约束末端位置、末端旋转和关节误差：\(J=e_p/\tau_p+e_r/\tau_r+e_j/\tau_j\)。候选仿真并行评估 4096 个环境，误差阈值分别为 0.03 m、0.2 rad 和 0.0025 rad²。系统辨识需要真实机器人校准序列，但策略训练不使用真实任务示范。
2. **SimAction。** 将任务拆成搜索、接近、抓取、搬运和放置等原子技能，由技能 API、BODex 抓取姿态和 cuRobo 无碰撞规划生成约 20 万条、2778 小时轨迹，覆盖 35 个任务和 100 个程序化房屋。搜索轨迹生成时可用目标真值；部署时策略只从视觉中复现搜索行为。
3. **SimVQA 与 SimDeploy。** 仿真特权状态被转成约 100 万条空间关系、几何属性、机器人状态和子任务问答。预训练策略再在 RoboCasa、MolmoSpaces 和 SceneSmith 等环境执行，得到约 2.2 万条、460 小时 SimDeploy 轨迹，使训练覆盖视角变化、底盘停靠误差和重试状态。
4. **VLA 训练与推理。** 模型采用约 3B 参数、SigLIP 视觉编码器、Gemma 语言骨干和 flow-matching 动作专家，输入三路 320×240 图像、指令和本体感知。预训练联合 flow、离散动作 token 和 VQA 目标；动作专家到 VLM 骨干的梯度被截断。后训练在 SimAction 与 SimDeploy 混合数据上优化 flow matching，并允许动作梯度更新骨干。推理不调用技能 API、规划器或仿真真值；动作 chunk 长度、推理频率和混合比例论文未说明。

#### 方法对比分析

与 MimicGen、MoMaGen 等围绕少量示范做增广的方法相比，SimVLA 从人工定义技能和规划器直接生成多样轨迹；与只扩大动作数据的合成方法相比，它还加入特权 VQA 和策略执行数据；与 π0.5 相比，主要变化在数据与监督组织，而非网络骨干。

贡献集中在“动作技能 + 空间语义 + 部署状态”的互补仿真数据框架。VLM、flow matching、FAST、运动规划、程序化场景和域随机化属于标准或已有组件。该方法适合可构建仿真资产、定义技能并校准动力学的移动操作任务。

#### 实验分析（精简版）

实验在 Anubis 双臂全向移动机器人上进行，覆盖入水槽、入抽屉、倾倒等任务，并比较每任务使用 50 或 200 条真实示范微调的 π0.5。指标是预定义子目标的完成比例（task progress），不是整项任务成功率。

SimVLA 平均任务进度为 **54.4%**，优于每任务 50 条真实示范的 π0.5；200 条示范的 π0.5 在域内模拟厨房为 **67.8%**，但在茶水间和真实家庭分别降至 **21.3%** 和 **29%**，说明域内示范数量不等于域外覆盖。MolmoSpaces 中，SimVLA 与 MolmoB0T 的拾取成功率分别为 **71.0%** 和 **论文未说明**；主要局限是整体完成仍不可靠，且实验集中于可模拟的刚体厨房操作。

#### 实用指南

论文摘要和当前阅读结果未给出可直接使用的代码、模型或数据发布链接，开源状态应视为论文未说明。复现至少需要 Scene Synthesizer、Isaac Lab、机器人模型与物体资产、技能定义、BODex/curobo 规划链、真实机器人校准序列，以及 SimAction、SimVQA 和 SimDeploy 的对应生成流程。应保留 320×240 三路视觉、语言和本体感知输入设定，并注意预训练阶段的梯度截断与后训练阶段的联合数据。迁移到其他机器人或任务时，需要替换机器人动力学、传感器和技能/仿真资产并重新生成数据、系统辨识和训练；动作 chunk、推理频率、采样比例和完整超参数论文未说明。

#### 总结

核心思想：用互补仿真监督学移动操作
1. 用系统辨识和程序化场景生成多样移动操作轨迹。
2. 用特权状态问答补足视觉空间与子任务语义。
3. 用策略在仿真中的执行轨迹覆盖部署时的误差状态。
4. 以多阶段 VLA 训练把三类信号迁移到真实移动操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11248v1)
- [arXiv](https://arxiv.org/abs/2610.11248v1)

---

<a id='2610.11505v1'></a>
## [DAMP: Humanoid Locomotion via Denoised Belief Learning and Adversarial Motion Priors](https://arxiv.org/abs/2610.11505v1)

**Authors:** Puying Shen, Wenhao Cui, Huaxing Huang, Bangyu Qin, Shengtao Li, Ziyang Dong, Guoteng Zhang

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Humanoid robots possess the structural capability to traverse complex terrains. However, achieving stable t raversal without relying on perceived information remains challenging, particularly in complex environments. This paper introduces DAMP, a reinforcement learning framework aimed at achieving robust and naturalistic humanoid locomotion over challenging terrains, with the assumption that no perceived information is available. The framework leverages recurrent neural networks to capture temporal dependencies and implicitly infer privileged and other task-relevant latent information. By aligning the learned representations with the task objective, the method enables robust and goal-consistent policy learning. This end-to-end framework achieves transfer learning from simulation to real-world environments, demonstrating the proposed method's robustness and generalization capabilities. The video of the real-world demonstration can be found at the following link: https://youtu.be/AkI7TZB2DDM.

### 论文解读

#### 摘要翻译

人形机器人具备穿越复杂地形的结构能力，但在没有感知信息的情况下实现稳定穿越仍然困难，尤其是在复杂环境中。本文提出 DAMP，一种面向挑战性地形鲁棒、自然行走的强化学习框架，假设没有外部感知信息。该框架利用循环神经网络捕获时间依赖，并隐式推断特权信息及其他与任务相关的潜变量。通过使学习到的表示与任务目标对齐，DAMP 实现了稳健且目标一致的策略学习。该端到端框架支持从仿真到真实环境的迁移，并在真实世界实验中展示了鲁棒性和泛化能力。

#### 方法动机分析

现有无感知人形行走方法主要依赖历史观测记忆或任务奖励：前者未必能形成可解释、可迁移的状态估计，后者容易在复杂接触和地形变化下产生不稳定动作。加入深度相机或 LiDAR 可以获得更多信息，但会增加计算负担，并引入传感器漂移；直接模仿动作又依赖高质量演示，且泛化范围有限。

DAMP 的核心假设是：部署时的本体感知序列虽然不完整，但其中包含足以推断速度、落脚状态和动力学条件的时间线索；训练时可访问的特权状态可以作为潜变量学习的监督，而不必在部署时保留外部感知。方法还假设演示动作先验能够提供有利于自然步态和训练稳定性的约束。

#### 方法设计详解

##### 输入、策略与特权监督

策略输入仅包含本体感知及指令：
`o_t={c_t,omega_t,g_t,	heta_t,dot	heta_t,a_{t-1}}`，其中包括期望线速度/偏航角速度、IMU 角速度、机体坐标系重力方向、关节位置和速度以及上一时刻动作。LSTM 编码观测历史得到隐变量 `z_t`，MLP actor 输出关节位置目标，再由 500 Hz PD 控制器生成力矩：
`\tau_t=k_p(a_t-\theta_t)-k_d\dot\theta_t`。

训练时 critic 额外获得机体线速度、质量、摩擦与恢复系数、关节增益、执行器强度、足端接触状态和 1.2 m × 0.8 m 地形高度扫描等特权状态 `s_t`。它只用于更准确的价值估计，不作为部署策略的输入。

##### Denoised World Learning

DWL 由 LSTM encoder 和 MLP decoder 组成。环境中注入噪声后，encoder 从历史本体观测提取 `z_t`，decoder 重建与特权状态同维的估计 `\tilde{s}_t`。训练目标为
`L_{denoise}=\|\tilde{s}_t-s_t\|_2^2+\lambda_r\|z_t\|_1`：第一项使潜表示包含有用的动力学/地形信息，L1 项促使表示稀疏，信息瓶颈则过滤观测噪声和任务无关变化。部署时丢弃 decoder 和特权监督，只保留 encoder 与 actor，因此这不是依赖外部传感器的在线感知模块。

##### AMP、接触奖励与平滑约束

DAMP 将 Adversarial Motion Prior 与任务奖励联合训练。判别器接收连续隐式状态转移，区分专家演示与策略生成转移；由判别器得到风格奖励，并与任务奖励组合：
`r_t=\alpha r_{task}+\beta r_{style}`。相比直接跟踪参考轨迹，AMP 约束的是动作转移的风格，帮助保持自然步态和训练多样性。

任务奖励同时考虑线速度和偏航速度跟踪、姿态稳定、足端零速度接触、接触力、腾空时间、默认姿态、能耗、动作平滑、碰撞和关节限位。作者用左右足接触的 XOR 及双足接触持续时间构造接触奖励，以鼓励自然的单支撑/双支撑转换，而非依赖 no-fly 惩罚或预设步态周期。另加入 Lipschitz 梯度惩罚
`L_{GP}=\mathbb{E}\|\nabla_s\log\pi(a|s)\|_2^2`，限制邻近状态产生剧烈不同动作，提高控制平滑性。整体采用 PPO 联合优化 actor、critic、DWL 和 AMP。

训练使用 IsaacGym，包含逐级增加难度的地形课程：8–20 cm 台阶、不规则鹅卵石、最高 15° 斜坡和 20 cm 路缘；前进速度采样范围为 0–0.8 m/s，水平角速度为 −1–1 rad/s。域随机化覆盖关节增益、执行器强度、质量、质心、摩擦、恢复系数、推力和扰动时机等。

#### 方法对比分析

- **相对仅用 RNN/历史记忆的方法**：DWL 用特权状态重建约束潜变量，使历史信息被组织为与任务相关的隐式状态，而非仅依赖黑盒记忆。
- **相对 HIM 等结合 AMP 的方法**：DAMP 增加显式去噪世界学习、接触奖励和 Lipschitz 平滑约束；AMP 负责自然动作先验，DWL 负责在无感知部署条件下恢复有用状态。
- **相对外感知方法**：DAMP 不使用深度相机或 LiDAR，避免额外计算与传感器漂移，但其能力边界由本体观测可推断的信息决定。
- **相对轨迹模仿**：判别器比较状态转移分布，不要求逐帧追踪参考轨迹，因而更适合同时满足速度指令和地形适应。

新机制主要是特权监督下的潜变量去噪、AMP 与任务目标的统一优化以及策略梯度平滑；LSTM、PPO、PD 控制、域随机化和 IsaacGym 属于标准组件。方法适用于有 IMU、关节编码器和足端接触/动力学反馈的人形机器人；若任务需要直接识别远处障碍物，则无外部感知的设定会成为限制。

#### 实验分析（精简版）

实验对象为 18 自由度、1.2 m 高、33 kg 的 Noetix N2，策略控制频率为 50 Hz。仿真比较 DAMP、去掉 DWL 去噪损失的版本、去掉 AMP 与风格奖励的版本，以及 HIM 基线；指标是课程训练中达到的平均地形等级。

在台阶、斜坡、离散崎岖地形上，DAMP 平均等级分别为 **5.76、5.84、5.89**；去掉去噪损失后降至 **3.38、3.36、3.78**，去掉 AMP 与风格奖励后为 **4.74、4.89、4.32**。这支持两点：DWL 对复杂地形适应最关键，AMP 则改善自然性、稳定性和训练收敛。论文还展示了线速度潜变量估计在启动瞬态误差较大、稳定行走后误差减小的现象，但未给出统一的误差汇总数字。

真实实验每种地形和难度进行 20 次测试，覆盖台阶、坡道和离散崎岖地形；N2 还完成了 World Humanoid Robot Games 2025 的 100 m 障碍赛并穿越赛场 A–J 类地形。论文未在正文给出各真实地形的完整成功率表。局限是方法仍依赖仿真中可获得的特权状态和演示动作进行训练，跨机器人、低算力平台及更大范围地形的迁移证据有限。

#### 实用指南

- **代码与数据**：论文提供真实演示视频链接（YouTube），但未明确给出代码、训练模型或数据集仓库；开源状态应视为论文未说明。
- **复现输入**：部署端需要指令速度、IMU 角速度和重力方向、关节位置/速度、上一动作，并保持训练与真实机器人的关节映射、控制频率和 PD 接口一致。
- **训练配置**：使用 IsaacGym、PPO、LSTM actor、特权 critic、DWL decoder、AMP 判别器和域随机化；论文报告 4090 GPU 上 7000 次迭代约 5 小时。奖励权重和随机化范围应按表中设定复现，缺失的 PPO、网络尺寸、时间步长及 `\lambda_r` 等完整超参数论文未说明。
- **迁移要点**：需重新校准机器人动力学、关节增益、执行器强度、质量/质心、摩擦和接触阈值，并按目标机器人调整动作空间与 PD 参数；若本体传感器配置或可观测性变化，DWL encoder 和 actor 需要重训。
- **评估注意**：应同时报告课程地形等级、逐地形成功率、速度跟踪、动作平滑和真实部署失败模式；不要把去掉 AMP、去掉 DWL 与 HIM 的结果混为同一种消融。

#### 总结

核心思想：用去噪潜变量稳健行走
1. 用 LSTM 汇总本体观测，并以特权状态重建约束潜变量。
2. 将去噪潜变量送入 actor，输出关节目标，由 PD 控制器执行。
3. 用 AMP 风格奖励、接触奖励和 Lipschitz 惩罚共同约束自然且平滑的动作。
4. 通过 PPO、地形课程和域随机化训练策略，再仅保留本体感知策略迁移到真实人形机器人。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11505v1)
- [arXiv](https://arxiv.org/abs/2610.11505v1)

---

<a id='2610.11322v1'></a>
## [USDCraft: Geometrically Grounded Programmatic Modeling of Articulated 3D Assets for Simulation](https://arxiv.org/abs/2610.11322v1)

**Authors:** Chuanrui Zhang, Zaijia Yang, Duomin Wang, Lu Shi, Daquan Zhou, Ruihua Zhang, Ziwei Wang

**Published:** 2026-10-08

**Categories:** cs.RO

**Abstract:**

Geometrically faithful and functional articulated 3D assets are essential for real-to-sim robot manipulation, where policies trained in simulation must transfer to physical objects. Recent mesh-based methods learn to infer articulation from annotated 3D assets, but deployment remains challenging when real-world objects fall outside the training distribution or their meshes are incomplete or corrupted. To address these limitations, we formulate articulated asset reconstruction as programmatic modeling grounded in partial geometric evidence and introduce USDCraft, a framework in which a pretrained LLM writes and revises executable programs for simulation-ready articulated assets without task-specific training. We propose source geometry analysis, which converts the source mesh into a metric textual description that distinguishes observed surface from unknown space, and iterative geometric rechecking, which re-encodes each candidate in the same representation so that discrepancies point to program edits while unobserved regions remain open to completion. Visual feedback and physical authoring guidance complete the modeling process, which produces articulated USD assets with explicit physical properties that load into Isaac Sim without manual adjustment. Experiments demonstrate leading articulation recovery on two benchmarks and validate USDCraft's effectiveness for real-to-sim-to-real robot manipulation.

### 论文解读

#### 摘要翻译

几何忠实且功能完整的关节式 3D 资产对于真实到仿真的机器人操作至关重要，因为在仿真中训练的策略必须迁移到物理对象。近期基于网格的方法从带标注的 3D 资产中学习推断关节结构，但当真实对象超出训练分布，或其网格不完整、受损时，部署仍然困难。

为解决这些限制，我们将关节式资产重建表述为由局部几何证据约束的程序化建模，并提出 USDCraft：一个无需任务专用训练、由预训练 LLM 编写和修改可执行程序以构建仿真就绪关节式资产的框架。我们提出源几何分析，将源网格转换为区分已观测表面与未知空间的公制文本描述；以及迭代几何复查，用相同表示重新编码每个候选资产，使差异能够指向程序修改，同时允许补全未观测区域。视觉反馈与物理属性编写指导共同完善建模过程，最终生成具有显式物理属性、无需手动调整即可载入 Isaac Sim 的关节式 USD 资产。实验在两个基准上展示了领先的关节结构恢复表现，并验证了 USDCraft 在真实—仿真—真实机器人操作中的有效性。

#### 方法动机分析

**核心矛盾是：既要尊重实测几何，又不能被残缺网格限制。**

- **网格分割路线**只能划分已有表面，难以补出扫描遗漏的零件、内部结构和功能开口；融合在一起的部件也可能无法正确分离。任务专用训练还带来类别和机构的分布外泛化问题。
- **程序化生成路线**能利用 LLM 的物体常识重建结构，但仅凭图像容易错估尺寸和部件位置。资产即使能够仿真，也可能让策略在真实对象上抓错把手或按错开关。

USDCraft 的核心假设是：**将可靠几何转成 LLM 可读、可定位的测量证据，并将候选误差返回同一表示，能够让程序编辑同时获得结构灵活性与实例几何精度。**

其问题边界是部分可观测条件下的刚体关节资产重建，不是从静态证据唯一恢复真实动力学。不可测的质量、摩擦等参数仍属于建模假设；未观测空间也不能直接认定为空或实心。

#### 方法设计详解

##### 1. 输入与证据整理

重建输入为**源网格、参考图像和可选文本要求**：

- 网格约束尺寸与部件位置。
- 图像提供物体身份、可见部件和可能的运动方式。
- 建模代理据此形成部件及连接、运动计划。

源几何统一到公制、Z 向上的坐标系。若输入疑似经过归一化且实际尺度未知，则由图像估计尺度假设，并在后续建模中固定。没有源网格时，系统转为文本或图像条件生成，跳过几何测量模块。

##### 2. 源几何分析：把网格变成可查询的测量文本

系统通过三轴射线交点、网格顶点和三角形重心采样表面，包括外壳内部已观测到的表面。

- 精细网格最长轴为 **192 个单元**。
- 文本摘要采用沿 Z 排序的 XY 切片；根据上下文预算，从最长轴 **32、24、16 个单元**中选择最细可容纳的表示，并压缩重复切片。
- 同时输出边界、凸起、开口和内部构件的局部公制摘要。
- 代理可进一步查询指定区域的边界、点分布、表面方向，或选择视角查看源网格。

关键不是一般体素化，而是**只记录表面支持，所有未采样单元均标为未知**。因此，扫描缺口不会被误当作自由空间，闭合壳体内部也不会被误当作实心材料。自动检测的结构仅作为测量锚点，其功能由代理判断。

##### 3. 程序化构建与编译

单个预训练 LLM 维护可执行资产程序，自主选择工具和修订顺序，而非遵循固定次数的流水线。

程序显式描述：

- 刚体部件及尺寸、位置；
- 关节类型、轴线、原点和运动限位；
- 质量、接触摩擦、关节摩擦及驱动；
- 材质、纹理和贴花。

几何采用 **CadQuery 参数化实体建模**与 **SDF 建模**：前者适合精确机械零件，后者适合有机形状和光滑过渡。编译器生成网格和碰撞几何，导出 USD，并支持 URDF 等输出。

物理属性直接使用 PhysX 参数。编写指导来自独立的 Isaac Sim 测试与故障分析，例如保持手动机构被动、检查正常操作过程中的运动间隙。这是工程约束，不是从观测中学习物理参数。

##### 4. 迭代几何复查：将误差定位到程序参数

候选资产在参考姿态下重新编码到源网格的同一坐标网格。设源表面单元集合为 \(S_M\)，候选为 \(S_t\)：

\[
r_t=\frac{|S_M\cap S_t|}{|S_M|},\qquad
D_t^-=S_M\setminus S_t,\qquad
D_t^+=S_t\setminus S_M.
\]

- \(r_t\)：候选覆盖了多少已观测源表面。
- \(D_t^-\)：候选遗漏的实测表面，是直接修正信号。
- \(D_t^+\)：候选在源未知区域新增的表面，可能是错误，也可能是合理补全，不能统一惩罚。

进一步使用六个轴对齐方向的首表面深度图检查面偏移、轮廓和开口位置。针对选定源区域与候选部件，令深度差为 \(\delta(u)=d_t(u)-d_M(u)\)，在共同观测区域 \(\Omega\) 上计算：

\[
b=\operatorname{median}_{u\in\Omega}\delta(u),\qquad
e_{\mathrm{profile}}=Q_{0.9}\bigl(|\delta(u)-b|\bigr).
\]

这里 \(b\) 表示沿视线的整体位置偏移，\(e_{\mathrm{profile}}\) 表示去除偏移后的形状误差。两者帮助代理区分“应移动部件”还是“应修改轮廓”，而无需真实部件分割标注。

这些量是**工具反馈，而非用于梯度训练的损失函数**。代理修改程序后重新编译、复查相关区域。

##### 5. 视觉反馈与最终输出

代理查看带材质和按部件着色的渲染，并检查不同关节姿态，识别几何覆盖指标不能判断的缺件、不合理装配、外观偏差及运动干涉。

最终输出为可编辑程序及具备关节、碰撞和物理属性的仿真资产。整个建模过程不进行任务专用微调。

#### 方法对比分析

| 对比路线 | 本质差异 | USDCraft 解决的问题 |
|---|---|---|
| Particulate、SIMART 等网格分割方法 | 从划分已有表面改为程序化重建 | 分离融合部件、补全缺失几何 |
| Articraft 等图像条件程序生成 | 从视觉猜测改为实测几何约束 | 减少尺寸、位置及关节锚点偏差 |
| Procedura 等装配检查路线 | 从检查装配合理性扩展到检查目标实例一致性 | “能动”不等于“与真实对象一致” |
| Mini Workflow 通用工具代理 | 提供专用测量表示和同表示误差反馈 | 降低代理临时编写测量代码的负担 |

**主要新机制**是未知空间感知的源几何文本表示，以及与之配套的非对称误差反馈和位置—形状误差分解。CadQuery、SDF、USD、PhysX 和视觉渲染是标准组件；贡献在于将它们组织成可测量、可编辑的闭环。

方法尤其适合存在缺面、部件融合但仍保有可靠几何锚点的扫描或生成网格。若要求逐表面精确复制完整网格，重新建模则可能不如直接分割保真。

#### 实验分析（精简版）

**协议。** USDCraft-bench 共 100 个对象，包含 60 个代理生成资产及 40 个生成网格与真实扫描扩展对象；Lightwheel 包含 243 个人工建模对象。输入移除部件标签、关节元数据和原始程序。评估覆盖部件 F1、静态及运动后的几何误差、关节方向和位置误差，并进行同骨干工具对照及逐项消融。

**结论一：专用几何反馈明显优于仅增强通用代理能力。**  
在 USDCraft-bench 上，同为 Astra 骨干，USDCraft 相比 Mini Workflow：

- 部件 F1：**74.480% → 84.830%**；
- 静态 gIoU：**0.512 → 0.694**；
- 关节角误差：**21.141° → 6.833°**；
- 平均耗时：**3.0 → 4.2 分钟/资产**。

三次运行的 F1 分别为 \(74.48\pm1.00\) 与 \(84.83\pm0.88\)，增益大于运行波动。独立移除源几何分析后，USDCraft 的 F1 降至 **81.975%**、静态 gIoU 降至 **0.616**，支持测量证据对规划和几何准确性的作用。

**结论二：更准确的交互几何对应更小的迁移损失。**  
每个可交互的方法—任务配置使用 200 条仿真示范训练 Diffusion Policy。USDCraft–Astra 在开抽屉、压烤面包机拨杆、开启烤面包机上的真实成功率分别为 **90%、85%、85%**，较仿真下降 **10、5、0 个百分点**。Mini Workflow–Astra 对应为 **0%、60%、70%**；论文将抽屉失败归因于把手位置错误造成的不安全接触。

**证据边界。** 下游仅覆盖三项操作，所给内容未说明成功率的试验次数和置信区间。Lightwheel 上并非全部指标领先：分割方法的整体表面误差更低，Instruct-Particulate 的 mIoU 更高。生成质量采用模型评审，不等同于人工评估或全面物理验证。细密网格结构仍难重建，柔性体动力学尚未验证。

#### 实用指南

- **开放状态：**论文介绍了 USDCraft-10k，含 10,000 个资产、超过 500 类物体，70% 来自文本条件、30% 来自图像条件。但所给文本未提供可核验的项目、代码或数据下载地址，是否公开可用，论文未说明。
- **核心依赖：**预训练多模态 LLM、CadQuery、SDF 网格化、USD/PhysX、Isaac Sim，以及源网格测量、候选渲染和自动编译工具。作者使用的专用 SDK 与完整工具接口能否获取，论文未说明。
- **关键复现设置：**统一公制与 Z-up 坐标；固定未知尺度假设；保留已观测内部表面；将未观测区域标为未知；候选与源共用采样网格。论文使用 Sol/high 与 Astra/low 等配置，但未在所给内容中说明完整停止条件、工具预算及硬件。
- **评估注意：**主基准对预测与参考分别居中并缩放到单位最长边，因此移除了全局尺度和位置误差；机器人部署应同时检查使用参考坐标系的评估结果。ArtLLM 的结果排除了 28 个重建失败案例，不能忽略这一协议差异。
- **下游训练：**Diffusion Policy 使用每配置 200 条仿真示范，训练 80 epochs、batch size 128；这属于操作策略训练，并非资产建模 LLM 的训练。
- **迁移建议：**换对象类别主要调整部件语义、关节惯例和物理编写指导，无需任务专用微调。换机器人需适配运动学、动作与观测接口、场景标定，并重新收集或适配示范、训练操作策略；论文未验证跨机器人直接迁移。换仿真器则需重新核验物理参数、碰撞及关节语义。

#### 总结

核心思想：以实测几何闭环修订资产程序
1. 将部分网格编码为带公制锚点的轴向切片，明确区分已观测表面与未知空间。
2. 联合图像部件计划和局部测量，编写可补全部件、显式定义关节及物理属性的资产程序。
3. 在同一网格上复查候选，对遗漏表面与未知区域新增表面区别处理，并分解位置与形状误差。
4. 根据几何及多姿态视觉反馈局部修订、重新编译，输出用于仿真和策略迁移的关节资产。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11322v1)
- [arXiv](https://arxiv.org/abs/2610.11322v1)

---

<a id='2610.11376v1'></a>
## [CRISP: Fixing Flying Pixels in Latent LiDAR Generation via Diffusion Decoding](https://arxiv.org/abs/2610.11376v1)

**Authors:** Andrea Ceron, Michael Schmidt, Alvaro Marcos-Ramiro, Sebastian Schmidt, Benjamin Busam

**Published:** 2026-10-08

**Categories:** cs.CV, cs.AI

**Abstract:**

Latent LiDAR pipelines suffer from flying pixels: convolutional VAEs blur sharp radial depth discontinuities, yielding edge depths that back-project to points floating between surfaces. We identify this as a major, directly correctable decoder bottleneck and introduce CRISP: a pixel-space diffusion decoder with a backbone-agnostic latent adapter, DiT-based denoiser, and support mask predictor. CRISP replaces video-VAE and LiDAR-native decoders alike while keeping the encoder and latent generator fixed. Across KITTI-360, SemanticKITTI, and nuScenes, replacing only the decoder reduces FSVD/FPVD by 50.5% on average across frozen backbones; for generic video VAEs, the reductions reach 71%/74%. On the LiDAR-native LiDM backbone, FRID drops by 71%, with the largest gains at depth discontinuities. In a pretrained LiDM world model, the same zero-shot replacement improves FSVD by 15.5%, narrowing the sim-to-real gap.

### 论文解读

#### 摘要翻译

潜在空间 LiDAR 流水线存在飞点（flying pixels）问题：卷积 VAE 会模糊尖锐的径向深度不连续边界，使边缘深度反投影为悬浮在表面之间的点。论文提出 CRISP：包含骨干无关潜变量适配器、基于 DiT 的去噪器和有效回波掩码预测器的像素空间扩散解码器。CRISP 可替换视频 VAE 和 LiDAR 原生解码器，同时保持编码器和潜变量生成器不变。

在 KITTI-360、SemanticKITTI 和 nuScenes 上，仅替换解码器即可使冻结骨干上的 FSVD/FPVD 平均降低 50.5%；对于通用视频 VAE，降幅分别达到 71%/74%。在 LiDAR 原生骨干 LiDM 上，FRID 降低 71%，收益最大出现在深度不连续处。预训练 LiDM 世界模型中的零样本替换使 FSVD 改善 15.5%。代码和检查点将通过项目主页发布。

#### 方法动机分析

问题的核心不是潜变量生成不好，而是卷积解码器把已有几何边界抹平：当前景与背景距离差异大时，它会输出两者之间的插值深度，反投影后形成不属于任何表面的飞点。圆周填充只处理方位角接缝，LiDM 的窄卷积仍可能跨越扫描行边界；同时，LiDAR 的空像素代表真实缺失回波，不能把所有像素当成连续深度。

论文的核心假设是：冻结潜变量仍携带足够的边界信息，换用合适的解码器即可恢复几何，因此把问题限定在解码端而非重训整个生成模型。

#### 方法设计详解

输入是柱面/球面投影的单通道 LiDAR 距离图和有效回波掩码 (m)。距离做对数归一化，空像素设为 (-1)。KITTI family 使用 (64\times1024)、1–56 m 和 ([-25^\circ,+3^\circ])；nuScenes 使用 (32\times1024)、2–45 m。预训练编码器得到 (z=E_\omega(\tilde y))，默认冻结；RGB 视频 VAE 基线将深度复制为三通道，而 CRISP 输出单通道。

1. **潜变量适配。** 通过 (	au=A_\psi(z)=W_s\,\mathrm{Flat}(W_cz)) 统一 SVD、Wan2.1、LiDM 的潜变量形状。骨干无关指接口可适配，不表示同一未经重训的解码器可跨骨干通用。
2. **像素空间扩散。** 对归一化距离图加噪声，DiT 根据 (x_t,\tau,t) 预测速度并还原深度。潜变量在 patch embedding 后和网络中部双重注入，使纯噪声先获得场景布局，再恢复局部边界。解码器含 24 个 Transformer 块、1024 维隐藏层、16 个注意力头，默认 5 步 Euler 采样。
3. **有效回波预测。** 三级 U-Net 同时读取 (z) 和预测深度，估计有效概率 (pi_\phi=\operatorname{sigmoid}(M_\phi(z,\hat y_d)))，再采样 Bernoulli 掩码；有效位置保留深度，无回波位置恢复 (-1)。
4. **边界优先训练。** 深度分支结合边界加权速度回归、无权回归、方向一/二阶差分、多尺度梯度和低噪声 (L_1)。边界权重先对有效回波做归一化滤波，避免把有效回波—空像素误判成真实边界。掩码分支使用 focal BCE、Dice、边缘/Laplacian、困难像素挖掘和深监督。

#### 方法对比分析

与 LiDM、RangeLDM 等主要改潜变量建模或沿用原解码器的方法不同，CRISP 直接替换解码阶段，默认不改变潜变量生成器；与 Pixel-Perfect Depth 不同，它不依赖干净 RGB，而依靠冻结潜变量并显式预测缺失回波；与 L3DR 的解码后修正不同，它在伪影产生的解码阶段处理问题；与 RAE 不同，它保留已有潜空间而无需重训生成器。

主要贡献是任务驱动的机制组合：冻结实验定位解码瓶颈；早期/中期潜变量融合为纯噪声解码提供布局；连续深度和离散回波支持分开恢复。它最适合已有距离图潜空间模型；非规则扫描、强时序需求和多回波传感器不能直接假定同样有效。

#### 实验分析（精简版）

SVD、Wan2.1 在 nuScenes 评估，LiDM 在 KITTI family 评估，并包含冻结解码器替换、视频 VAE 适配和冻结 LiDM 世界模型的零样本替换。冻结 SVD 的 FSVD 从 156.487 降至 40.244，整体 CD 从 1.33107 降至 0.63967；LiDM 边界 CD 从 7.913 降至 4.052，FRID 从 2.2351 降至 0.6389。冻结 LiDM 世界模型中 FSVD 从 37.70 降至 31.871、边界 CD 从 96.41 降至 79.55，但边界 F-score 从 0.0486 降至 0.0389，说明并非所有指标都改善。

主要局限是 CRISP 约 516M 参数，为被替换解码器的 7–60 倍，缺少容量匹配的非扩散对照；适配后的 SVD/Wan 部分分布指标回退；没有检测、分割或规划的直接收益，不预测 remission，逐帧解码也可能产生高频噪声和细节闪烁。

#### 实用指南

论文仅承诺未来发布代码和检查点，当前无法确认已公开，也未给出可核验仓库地址。复现时需严格保持投影、深度截断、对数归一化和空值编码；KITTI 与 nuScenes 的归一化公式不能混用。KITTI family 使用 4,376 个 25 帧训练片段，经五种增强形成每轮 21,880 个片段；实际仍逐帧编码和解码。冻结实验须锁定编码器，生成实验还须锁定潜变量生成器和采样潜变量。

默认 5 步采样；论文在 8×H200、`torch.compile`、batch 25 条件下报告 SVD/Wan/LiDM 解码耗时约 3.92/3.84/23.06 ms/帧，这不能直接视为机器人端单帧延迟。完整学习率、优化器、总步数和部分损失权重未说明。更换骨干需匹配潜变量形状并重训适配器和解码分支；更换传感器需重新定义投影、束数、视场、距离范围和回波掩码。

#### 总结

核心思想：冻结潜空间，重做几何解码
1. 将不同骨干的冻结潜变量适配为统一条件 token，不改变已有潜变量生成器。
2. 在像素扩散的起始与中部注入潜变量，先恢复场景布局，再细化深度边界。
3. 用有效区域回归、边界加权和方向差分约束，抑制跨表面插值形成的飞点。
4. 联合潜变量与预测深度估计回波支持，将稠密深度合成为稀疏 LiDAR 输出。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11376v1)
- [arXiv](https://arxiv.org/abs/2610.11376v1)

---

<a id='2610.11577v1'></a>
## [CoCam4D: Geometry-Aware Cooperative 4D Perception for Camera-Only Autonomous Driving](https://arxiv.org/abs/2610.11577v1)

**Authors:** Soham Pahari, Sudip Das, Arindam Das, Ujjwal Bhattacharya

**Published:** 2026-10-08

**Categories:** cs.RO, cs.CV

**Abstract:**

Autonomous vehicles often suffer from limited perception due to occlusions, blind spots, limited sensor range, and the complex nature of surrounding environments. Multi-agent collaborative perception (CP) addresses these challenges by allowing vehicles to share sensory information and reconstruct the scene cooperatively. However, camera-only perception remains fundamentally limited by the uncertainty of distance-dependent monocular depth estimation. We propose CoCam4D, a Bayesian framework for collaborative perception that explicitly models geometric uncertainty. It uses a VGGT-based feedforward network to generate 3D Gaussian scene representations with associated uncertainty estimates, enabling multiple vehicles or agents to efficiently combine their observations. By sharing compact Gaussian primitives, reliable observations from one agent can reduce the depth uncertainty of another without requiring LiDAR sensors. To support real-world deployment, we introduce Dynamic Object Primitives (DOPs), a compact 35-byte representation designed for efficient C-V2X communication. Extensive experiments show that our proposed method consistently outperforms recent vision-only methods, achieving improvements of 11.48% on OPV2V+ and 10.62% on DAIR-V2X-C, demonstrating the potential of geometrically grounded collaborative perception for LiDAR-free autonomous driving.

### 论文解读

#### 摘要翻译

自动驾驶车辆常因遮挡、盲区、有限的传感器探测范围及复杂环境而受到感知能力限制。多智能体协同感知（CP）通过使车辆无缝共享传感信息并重建场景来应对这些挑战。然而，纯相机感知仍从根本上受限于随距离变化的单目深度不确定性。

我们提出 CoCam4D，一个显式建模几何不确定性的协同感知贝叶斯框架。该方法利用基于 VGGT 的前馈网络，生成带不确定性估计的三维高斯场景表示，帮助多个车辆或智能体高效融合各自观测。通过共享紧凑的高斯基元，一个智能体的可靠观测能够缓解另一个智能体的深度不确定性，而无须使用 LiDAR 传感器数据。

为支持真实部署，我们引入动态对象基元（Dynamic Object Primitives，DOPs），一种用于高效 C-V2X 通信的紧凑 35 字节表示。大量实验表明，该方法持续优于近期纯视觉方法，在 OPV2V+ 和 DAIR-V2X-C 上分别取得 11.48% 和 10.62% 的提升，为无 LiDAR 自动驾驶提供了一条具有潜力、以几何为基础的研究方向。

#### 方法动机分析

**核心矛盾是：相机便宜且信息丰富，但远距离三维定位不可靠；多车可以补充视角，却不应把所有观测当作同等可信。**

从针孔投影关系 \(z=fH/h\) 出发，在物体高度和像素测量误差等条件固定时，深度标准差随距离近似平方增长：

\[
\sigma_z\propto z^2/f,\qquad \sigma_z^2\propto z^4/f^2.
\]

因此，增加单车网络容量并不能消除投影本身的歧义。论文认为，现有协同方法主要交换确定性的 BEV 特征，依赖学习式注意力处理跨车融合，缺少显式的、随距离和观察方向变化的置信度模型。

CoCam4D 的核心假设是：

- 同一对象可被不同车辆从互补方向观测，一车的不确定方向能由另一车补足。
- 深度误差能够转化为对象级观测协方差，并用于信息滤波。
- 对应关系、坐标变换和时间对齐足够准确。
- 各车误差近似独立且服从高斯分布，才能成立其最优融合论证。

其目标同时覆盖检测、运动估计、通信效率和延迟鲁棒性；但**视角重复、共同遮挡、相关误差和错误匹配**都可能破坏上述收益，车辆数量本身并非充分条件。

#### 方法设计详解

##### 1. 多帧多相机输入 → 共享几何特征

每车输入 \(F\) 帧、每帧 \(C\) 路相机图像。冻结的 ViT 提取图像块 token，再通过交替的图像内注意力与跨图像全局注意力聚合信息，最后拼接原始特征和注意力特征，供六个预测头使用：

- 相机头：预测内外参。
- 深度头：预测逐像素尺度化深度。
- 不确定性头：估计深度误差。
- 高斯头：预测三维高斯的尺度、旋转、透明度、颜色和寿命。
- 运动头：预测前后向三维速度及动态概率。
- 天空头：以半球上的高斯表示无穷远背景。

论文同时声明使用数据集提供的相机标定，但**预测标定与已知标定之间如何配合未交代清楚**。

##### 2. 深度误差 → 带方向性的三维高斯

不确定性头使用深度预测的绝对误差作为监督：

\[
\mathcal L_{\rm unc}
=\left\|\hat\Sigma-\operatorname{sg}(|D-D^*|)\right\|_1.
\]

其中 \(D^*\) 为真实深度，停止梯度避免该目标反向改变深度头。该设计不增加独立的不确定性标注，但仍需要真实深度，不能简单等同于完全自监督。

深度通过相机投影逆变换得到三维中心：

\[
\mu=R^\top(K^{-1}\tilde uD-t).
\]

随后用预测不确定性扩大深度方向的高斯尺度：

\[
s_z=s_z^{\rm base}(1+\beta\hat\Sigma).
\]

直观上，模糊的深度预测对应沿视线拉长的空间分布，而不是一个过度自信的三维点。需注意：**表示物体形状的高斯协方差与估计位置误差的观测协方差并非同一概念**；正文给出了连接思路，但未完整展开对象级传播和校准细节。

##### 3. 跨帧运动 → 动静分解与渲染监督

运动头加入时间编码，使源帧 token 关注相邻帧，预测速度和动态概率。以 \(p_{\rm dyn}>0.5\) 划分动态区域，其余归入静态场景。

训练时，目标帧的动态内容只能由其他帧的动态高斯经速度变换后渲染出来，不能直接使用该帧自身的动态高斯。这样，正确重建目标帧就要求预测合理运动，无须显式光流监督。高斯寿命通过时间衰减降低陈旧表示的影响。

总损失组合 RGB 重建、透明度、动态掩码、寿命及不确定性项。各项具体权重和多数定义未在所给正文展开。

**术语存在疑点：**论文将运动注意力称为“因果”，但公式允许访问 \(f_s-1\) 和 \(f_s+1\)，并非严格的仅看过去；在线运行是否引入未来帧等待，论文未说明。

##### 4. 动态高斯 → 紧凑 DOP 消息

对动态掩码做连通域聚类：

- 删除少于 20 个像素的区域。
- 拆分三维范围超过 15 m 的区域。
- 从保留区域提取中心、平均速度、由速度得到的航向及对象尺寸。

每个对象编码为：

\[
\mathcal P_k=(\mu_k,v_k,\theta_k,d_k,R_k).
\]

其中 \(R_k\) 表示各向异性观测误差，深度方差遵循 \(z^4/f^2\) 的缩放规律。量化后约 35 字节：位置 12、速度 6、航向 2、尺寸 6、对角协方差 9 字节。

这不是传输完整对象表面，而是传输**对象状态及其置信度**。动态掩码如何覆盖静止车辆、低速对象航向如何稳定估计，正文未完整说明。

##### 5. 跨车消息 → 对齐、匹配与信息融合

接收端依次执行：

1. 使用 GPS/IMU 提供的相对位姿变换对象中心，并按 \(R'=URU^\top\) 旋转误差椭球。
2. 用 \(\mu_{\rm comp}=\mu+v\Delta t\) 补偿通信延迟，同时增加过程噪声。
3. 用协方差归一化的马氏距离构造匹配代价，执行匈牙利匹配，并以 \(7.815\) 为三维卡方门限。
4. 对匹配对象相加信息矩阵与信息向量。

无先验、直接观测位置时：

\[
R_{\rm fused}=\left(\sum_jR_j^{-1}\right)^{-1},\qquad
\mu_{\rm fused}=R_{\rm fused}\sum_jR_j^{-1}\mu_j.
\]

因此，可信方向获得高权重，不确定方向贡献较小。该融合不需要跨车联合训练，但其最优性仅成立于所假设的独立高斯误差模型。

论文还合并匹配对象的动态高斯以补全形状；这需要额外高斯信息，**不能仅由 35 字节 DOP 恢复**。正文也未明确这些动态高斯的完整传输协议。文字称匹配使用速度，但给出的匹配公式只包含位置。

#### 方法对比分析

| 最接近的方法类别 | CoCam4D 的关键差异 |
|---|---|
| CoCa3D 等相机协同方法 | 从共享 BEV 特征转向共享对象状态及几何置信度，用信息滤波替代学习式融合 |
| VOGS-CP 等高斯协同表示 | 不仅使用高斯作为紧凑表示，还引入速度、深度误差传播和解析融合 |
| VGGT、动态前馈重建方法 | 从单车几何重建扩展到可通信、可匹配、可融合的对象级状态 |
| 协同 Kalman 跟踪方法 | 强调相机投影导致的距离依赖、各向异性误差，而非仅对检测框指定简化协方差 |

真正值得关注的是**“像素深度误差—空间不确定性—对象消息—融合权重”这条连接链**。信息滤波、匈牙利匹配、逆投影与连通域聚类本身都是标准组件；创新主要在于将其组织为相机协同感知机制，而不是提出新的贝叶斯估计公式。

适用场景是有时间同步、可靠定位和互补视角的车车或车路协同。迁移到缺少绝对定位、误差高度相关或动态对象密集粘连的场景，需要额外机制。

#### 实验分析（精简版）

实验包括 OPV2V+ 仿真、DAIR-V2X-C 真实车路协同数据，以及可控制车辆数量的 CARLA 密度扫描。基线覆盖 BEVFormer、CenterPoint、CoCa3D、Where2comm、V2X-ViT 和 VOGS-CP；消融采用三个协同车辆。

**结论一：报告结果支持检测与通信效率优势。**

- OPV2V+、两车、0–50 m 的 BEV AP@0.5：CoCam4D 为 **57.3**，CoCa3D 为 **51.4**，增加 **5.9 个百分点**，即约 **11.5% 相对提升**。
- 对应通信量为 **58 对 1240 KB/s**，约减少至 \(1/21\)。
- DAIR-V2X-C 的 AP@0.5 为 **42.7 对 38.6**，增加 **4.1 个百分点**，约 **10.6% 相对提升**。论文称未做域特定微调。

**结论二：消融支持不确定性与融合机制的重要性，但“五车胜单 LiDAR”只适用于报告配置。**

- 完整模型 AP@0.5 为 **60.4**；移除不确定性头降至 **55.2**，改用 BEV 融合降至 **53.7**。
- CARLA 中五车达到 AP@0.5 **63.7**、AP@0.7 **40.2**、AMOTA **47.8**，对应单车 LiDAR 为 **62.3、38.6、45.7**。这是多车系统与单车参考的比较，并非等硬件成本比较。

**主要证据边界：**

- 未提供不确定性校准曲线，无法确认预测误差是否确实对应有效概率协方差。
- 论文提出重建、渲染和端到端延迟指标，但所给正文没有相应完整结果，也未报告方差或置信区间。
- 约 1400 字节/帧的预算在 10 Hz 下约为 14 KB/s；完整系统报告 58 KB/s，并另含高斯缓存开销。两者不是同一通信口径，缓存更新、协议开销及实际容量约束仍需厘清。
- “因果”定义、深度监督来源及动态高斯传输细节存在缺口，限制了对实时性、无 LiDAR 训练和部署可行性的判断。

#### 实用指南

**资源状态：**所给论文未提供项目、代码或模型权重链接，开源状态论文未说明。实验数据集有名称和参考文献，但未给出本文使用版本与完整划分。

复现至少需要：

- **输入预处理：**组织同步多帧多相机序列，统一内外参、世界坐标、相对车辆位姿和消息时间戳；训练深度与动态相关目标所需标签来源需进一步确认。
- **模型实现：**冻结 ViT，接入 VGGT 式多尺度解码、六个预测头及可微高斯渲染器，再实现对象聚类、量化和信息融合。
- **已知设定：**动态阈值 0.5、最小区域 20 像素、15 m 拆分阈值、匹配门限 7.815、10 Hz 通信及约 35 字节 DOP。
- **关键缺失：**输入分辨率、帧数、训练周期、优化器、学习率、损失权重、过程噪声、协方差比例常数、量化范围和硬件资源，论文未说明。实验配置写四相机，感知示例描述六相机，实际协议需核实。
- **评估注意：**区分 OPV2V+ 与 CARLA，区分不同距离范围及 BEV/3D AP；分别统计 DOP、静态缓存和动态高斯流量。论文声称无需 LiDAR 监督，但损失使用真实深度，必须追溯其来源。

迁移到其他机器人时，信息融合公式可保留，但需替换相机模型和坐标变换，重新校准深度误差及运动噪声，调整对象尺寸与聚类规则，并针对新外观和运动分布微调预测头。对于共享定位误差或重复转发消息，还需避免将相关观测误当成独立信息重复累加。

#### 总结

核心思想：以几何不确定性融合多车视觉
1. 从多帧相机输入预测深度、误差和速度，将不可靠深度表示为沿视线扩展的三维高斯。
2. 通过跨帧动态高斯变换与渲染一致性约束运动，并用寿命衰减降低陈旧表示影响。
3. 将动态对象压缩为携带位置、速度和观测协方差的约 35 字节 DOP。
4. 对跨车 DOP 做位姿对齐、延迟补偿和匹配，再以信息滤波融合互补方向的可靠观测。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.11577v1)
- [arXiv](https://arxiv.org/abs/2610.11577v1)

---

