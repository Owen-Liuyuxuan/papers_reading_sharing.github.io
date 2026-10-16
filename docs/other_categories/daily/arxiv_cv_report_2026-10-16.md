time: 20261016

# Arxiv Computer Vision Papers - 2026-10-16

## Executive Summary

## 本期摘要

本期入选论文集中在机器人感知、规划与操作的可部署性：一类工作通过缓存或异步系统缩短在线计算，SpaceCache复用连续点云帧的深层特征，LIN-SLAM把高频几何里程计与神经隐式建图解耦；另一类工作在接触和不确定性下改进规划，LEAP与CQDC-Warp分别探索接触约束轨迹求解和GPU并行可微模拟，A-SLS则把在线学习对未来安全边界的影响纳入控制。

学习型系统也在把更丰富的任务信息接入真实动作。VLA-Δ以物理状态变化监督辅助动作生成，MoCaP从被动人类视频提取场景和接近经验，LICOA将底盘与投影镜面统一为光学动作；TGrasp6D用任务语义引导抓取，SOMA展示四足机身推物。它们共同体现出从单一感知或策略指标走向完整输入—决策—执行闭环的趋势，同时也暴露出对传感器、外部追踪、场景范围与初始化条件的依赖。

若关注实时性和算力约束，可先读SpaceCache、LIN-SLAM与A-SLS；若研究接触规划，可比较LEAP和CQDC-Warp的求解假设；若关注跨任务迁移和真实机器人评估，VLA-Δ、MoCaP与LICOA提供了不同的数据和接口设计。阅读时应区分仿真吞吐、真实闭环成功率及小规模硬件演示所支持的结论范围。

---

## Table of Contents

1. [SpaceCache: Intermediate Feature Reuse for Efficient Semantic Segmentation on Point Cloud Streams](#2610.21207v1)
2. [CQDC-Warp: GPU-Accelerated Differentiable Quasi-Dynamic Simulation for Contact-Rich Manipulation](#2610.21349v1)
3. [LIN-SLAM: Lightweight LiDAR--Inertial Neural SLAM for Real-Time Onboard Navigation](#2610.21199v1)
4. [SOMA: Stack-of-Objects Manipulation using a Quadruped Robot](#2610.20968v1)
5. [TGrasp6D: Task-Informed Vision-Language Priors and Pose Refinement enable Zero-Shot Task-Oriented 6DoF Grasping](#2610.20901v1)
6. [VLA-$Δ$: Learning Physical State Transition for Vision-Language-Action Models](#2610.21306v1)
7. [LICOA: Learning Instruction-Conditioned Optical Actions for Mobile Spatial Augmented Reality](#2610.21282v1)
8. [MoCaP: Mobile Code-as-Policies for Zero-Shot Mobile Manipulation from Passive Human Videos](#2610.20998v1)
9. [LEAP: Gradient Flow Augmented Lagrangian Trajectory Optimization for Contact Constrained Robots](#2610.20936v1)
10. [Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning](#2610.20925v1)

---

## Papers

<a id='2610.21207v1'></a>
## [SpaceCache: Intermediate Feature Reuse for Efficient Semantic Segmentation on Point Cloud Streams](https://arxiv.org/abs/2610.21207v1)

**Authors:** Dongheon Yang, Beomseok Kang, Jae-Joon Kim

**Published:** 2026-10-15

**Categories:** cs.CV

**Abstract:**

Point-cloud streams are central to autonomous driving and robotics, yet existing segmentation networks process every frame from scratch, although consecutive frames largely overlap. We observe that the intermediate features of point-cloud transformers remain semantically consistent across frames for the same spatial region, especially in deep stages. Building on this, we present SpaceCache, a training-free feature cache that reuses these features across frames. Tokens of previously observed regions skip a sequence of transformer blocks and take their cached features, only the remaining tokens are computed, and the full token layout is restored for dense prediction. Lightweight cache management keeps reused features reliable by limiting each entry's age and evicting low-confidence entries. Unlike token merging, SpaceCache selects tokens at negligible cost and runs its cache operations once per encoder stage. On both outdoor and indoor benchmarks, such as nuScenes, Waymo, and ScanNet, with two transformer backbones of Utonia and LitePT-B, SpaceCache achieves up to 1.65x speedup with less than 1-point mIoU drop and consistently outperforms token-merging methods.

### 论文解读

#### 摘要翻译
SpaceCache在点云流的相邻帧之间复用空间区域的中间特征。它在深层Transformer中跳过缓存命中token的部分计算，并通过置信度和寿命管理避免长期复用过时特征，不需要额外训练。

#### 方法动机分析
点云连续帧高度重叠，但常见分割网络每帧都从头计算，浪费深层特征中重复的空间信息。直接合并token可能破坏密集预测布局，也不能保证复用特征仍代表当前语义。作者假设深层同一区域特征更稳定，可用空间索引、语义置信度和有限缓存寿命筛掉不可靠复用。

#### 方法设计详解
每帧先将token坐标映射到全局量化空间区域。Mask Gate查询缓存并标记命中与未命中；命中token跳过选定深层block，未命中token正常计算。Reuse Gate把旧特征与新计算特征合并，恢复完整token布局供解码器密集预测。Commit Gate根据预测类别置信度及年龄更新条目，低置信或过期缓存被移除。默认置信阈值0.85，室外寿命L=8、室内L=4；缓存按stage独立运行。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：它无需训练，主要实现成本是全局坐标索引、稀疏token gather/scatter、类别置信度和缓存淘汰。需报告含缓存操作的端到端延迟、mIoU变化和显存，而不仅是主干FLOPs；作者提供项目代码链接。迁移到激光雷达驾驶栈应根据帧间运动和动态目标密度调寿命，并监控长时间运行的缓存内存。

#### 方法对比分析
创新贡献是按空间区域复用特征并恢复密集token布局；相比token合并，它不改变输出网格结构，相比单纯缓存则加入语义置信和寿命门控，适合帧间重叠较高的连续输入。

#### 实验分析（精简版）
在Waymo的LitePT-B设置中，stage 3与4复用带来1.65倍加速，mIoU从72.82降至72.11，下降0.71点；Utonia代表场景延迟由700.6ms降至526.5ms，缓存约69MiB。实验覆盖nuScenes、Waymo和ScanNet及两个骨干，但缓存管理额外开销和场景动态性影响实际收益。 延迟收益来自主干计算减少，缓存查询和重排本身也有开销；动态目标比例变化时，需重新检查精度损失是否仍在可接受范围。

#### 实用指南
它无需训练，主要实现成本是全局坐标索引、稀疏token gather/scatter、类别置信度和缓存淘汰。需报告含缓存操作的端到端延迟、mIoU变化和显存，而不仅是主干FLOPs；作者提供项目代码链接。迁移到激光雷达驾驶栈应根据帧间运动和动态目标密度调寿命，并监控长时间运行的缓存内存。

#### 总结
核心思想：复用稳定空间区域的深层特征。
1. 坐标映射到缓存区域。
2. 命中token跳过深层计算并复原布局。
3. 置信度与寿命控制缓存提交。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.21207v1)
- [arXiv](https://arxiv.org/abs/2610.21207v1)

---

<a id='2610.21349v1'></a>
## [CQDC-Warp: GPU-Accelerated Differentiable Quasi-Dynamic Simulation for Contact-Rich Manipulation](https://arxiv.org/abs/2610.21349v1)

**Authors:** Yanchen Ding, Jiayi Chen, Jianfeng Xu, Wei Zhao, Li Yi, He Wang

**Published:** 2026-10-15

**Categories:** cs.RO

**Abstract:**

Efficient synthesis of contact-rich manipulation trajectories is essential for downstream robot learning. Gradient-based planning offers a promising approach, but jointly optimizing motion and contact decisions remains challenging. Meanwhile, many existing planners run on CPUs with limited throughput. We present CQDC-Warp, a GPU-accelerated differentiable quasi-dynamic simulator for batched model predictive control (MPC). We use it to study contact-guided trajectory optimization, in which intended robot--object contacts are specified before planning and encouraged through soft costs, while the simulated dynamics determine contact geometry and frictional response without a prescribed contact-mode schedule. Across $17{,}408$ trials on the $17$-object Allegro in-hand rotation benchmark, our framework achieves $79.8\%$ success, compared with $73.7\%$ for CFree and $65.2\%$ for IMPACT, and reduces mean trajectory-generation time by factors of $27.0$ and $50.2$, respectively. On a single NVIDIA RTX 4090 with $2{,}048$ parallel worlds, forward and backward throughput reach $2.06\times10^5$ and $2.25\times10^5$ world-steps per second, respectively. Further experiments cover manipulation with LEAP and Shadow hands, environmental contact exploitation, bimanual manipulation, whole-body control, and real-world trajectory replay on a Sharpa hand.

### 论文解读

#### 摘要翻译
CQDC-Warp提出面向接触丰富操作的GPU并行可微准动力学模拟器，将意图接触作为软目标而非固定接触模式表，再用模拟动力学决定接触几何和摩擦响应，支持批量MPC轨迹生成。

#### 方法动机分析
接触丰富操作需要联合优化动作与接触，但现有可微规划器可能忽略关节—基座耦合、在CPU上逐轨迹运行，或要求预先指定完整接触模式。作者的假设是只需告诉规划器希望哪些部位接触，接触力学仍可由模拟器求解；GPU批处理则能以大量并行候选补偿局部优化对初值的依赖。

#### 方法设计详解
系统使用准静态/准动力学接触模型，将接触求解写成凸优化问题，并保留完整质量矩阵中的关节—基座耦合；针对接触几何使用witness-pair代理计算梯度，再对隐式接触解求导。用户指定期望机器人—物体接触，以软代价鼓励而不硬编码接触模式，接触位置和摩擦由动力学响应决定。Warp GPU上批量运行多个环境与MPC候选，梯度更新轨迹并滚动执行。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：论文在NVIDIA RTX 4090上评估，主要实现依赖Warp并行化、接触凸程序和隐式梯度；复现须单独核对模拟器步长、接触正则化、初始化与MPC预算。报告中的高吞吐是仿真world-step速率，不是机器人控制频率。迁移到真实硬件需把模拟轨迹放入安全控制器，并先测量模型偏差。

#### 方法对比分析
核心区别是规划者指定接触意图，而不是预写完整接触模式；相比固定接触序列优化更灵活，相比黑箱策略则可直接利用接触模型梯度，适用于接触主导的轨迹搜索。

#### 实验分析（精简版）
Allegro手17,408次试验中成功率79.8%，CFree为73.7%、IMPACT为65.2%；平均轨迹生成时间分别快27.0倍和50.2倍。RTX 4090上并行2048个世界，前向/反向吞吐约2.06×10^5/2.25×10^5 world-steps/s。Shadow手真实实验是轨迹回放，非大规模实物成功率；方法依赖调参和较好的接触初始化。 吞吐测试测量的是GPU模拟步，不代表同等数量的真实机器人动作；成功率也受初始化与代价权重影响，应连同条件一起报告。

#### 实用指南
论文在NVIDIA RTX 4090上评估，主要实现依赖Warp并行化、接触凸程序和隐式梯度；复现须单独核对模拟器步长、接触正则化、初始化与MPC预算。报告中的高吞吐是仿真world-step速率，不是机器人控制频率。迁移到真实硬件需把模拟轨迹放入安全控制器，并先测量模型偏差。

#### 总结
核心思想：用可微接触模拟并行优化操作轨迹。
1. 指定期望接触与任务目标。
2. GPU求解接触动力学并反传梯度。
3. 批量MPC筛选并执行轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.21349v1)
- [arXiv](https://arxiv.org/abs/2610.21349v1)

---

<a id='2610.21199v1'></a>
## [LIN-SLAM: Lightweight LiDAR--Inertial Neural SLAM for Real-Time Onboard Navigation](https://arxiv.org/abs/2610.21199v1)

**Authors:** Shiqi Che, Yijie Liu, Shuo Wang, Wenxin Liu

**Published:** 2026-10-15

**Categories:** cs.RO

**Abstract:**

Geometric light detection and ranging (LiDAR)--inertial simultaneous localization and mapping (SLAM) provides efficient state estimation but lacks the continuous map representation and global refinement capabilities of neural implicit SLAM, while neural implicit methods remain computationally expensive. We propose Lightweight LiDAR--Inertial Neural SLAM (LIN-SLAM), a hybrid method that couples high-rate geometric LiDAR--inertial estimation with neural implicit tracking, continual mapping, and global refinement for accurate real-time onboard SLAM. Experiments on twelve representative sequences from three datasets spanning 32-, 64-, and 128-channel LiDAR configurations show that LIN-SLAM achieves the lowest overall mean absolute trajectory error (ATE) among the evaluated state-of-the-art geometric and neural implicit methods, while substantially reducing runtime compared with existing neural implicit SLAM methods. On a Unitree Go2 equipped with an NVIDIA Jetson Orin NX and a 16-channel LiDAR, LIN-SLAM sustains real-time localization and neural mapping over a 1.47-km indoor trajectory through challenging repetitive corridor environments, with an endpoint closure error of only 0.141~m. To the best of our knowledge, LIN-SLAM is the first LiDAR--inertial SLAM system with a neural implicit backend to achieve real-time online operation on a mobile robotic platform while maintaining state-of-the-art localization accuracy. The code will be released upon acceptance.

### 论文解读

#### 摘要翻译
LIN-SLAM把高频几何LiDAR-惯性里程计与异步神经隐式地图结合，试图同时获得低延迟定位、连续表面表达和全局优化能力，并在Jetson Orin NX移动机器人上运行。

#### 方法动机分析
几何LiDAR-惯性SLAM高效，却难提供连续神经表面与全局细化；神经隐式SLAM地图表达丰富，但常因优化与建图耗时无法在线部署。作者的核心假设是让轻量前端持续提供高频位姿，再把选定关键帧异步送入神经后端，并用回环优化校正地图，使两类方法的优势互补。

#### 方法设计详解
前端用滚动四分之一圈LiDAR子扫与IMU预测，执行GICP匹配和迭代误差状态滤波，输出高频局部里程计及协方差；关键帧接口传递运动补偿后的完整扫描、时间区间和末端位姿。异步后端以相对运动初始化神经跟踪，更新隐式SDF与神经点图，再通过位姿图优化处理回环；全局校正变换返回前端。固定容量回放缓存、局部窗口与预分配内存控制Jetson上的资源。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：硬件复现需要匹配LiDAR转速、IMU频率、时间同步和GPU内存，并将前端实时率与后端队列积压分开监测。论文称代码将在接收后发布。迁移前应按传感器线数重设关键帧与局部窗口，并在有外部真值的长轨迹上验证ATE，而不能只报告闭环误差。

#### 方法对比分析
核心创新是以异步后端补充地图表达，同时不让神经优化阻塞高频定位；相比纯神经方法，它更重视前端实时性，相比纯几何方法则增加连续表面和全局细化。

#### 实验分析（精简版）
在NCD Stereo、Multi-Cam与M2DGR的12条序列上，论文报告三组平均ATE RMSE分别为0.221、0.086、0.638。Go2搭载Jetson Orin NX和16线LiDAR运行，1.47km走廊路线端点闭合误差0.141m，平均更新时间84.9ms；另一地下车库路线为0.073m与88.4ms。端点闭合只检验首尾一致性，不等于全轨迹有外部真值的绝对精度。 这些结果支持前后端解耦的实时可行性，但三组数据集的误差统计与机载端点闭合是不同指标，应分别解读。

#### 实用指南
硬件复现需要匹配LiDAR转速、IMU频率、时间同步和GPU内存，并将前端实时率与后端队列积压分开监测。论文称代码将在接收后发布。迁移前应按传感器线数重设关键帧与局部窗口，并在有外部真值的长轨迹上验证ATE，而不能只报告闭环误差。

#### 总结
核心思想：几何前端保实时，神经后端补地图。
1. LiDAR与IMU完成高频里程计。
2. 关键帧异步更新隐式地图。
3. 回环优化校正全局轨迹和地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.21199v1)
- [arXiv](https://arxiv.org/abs/2610.21199v1)

---

<a id='2610.20968v1'></a>
## [SOMA: Stack-of-Objects Manipulation using a Quadruped Robot](https://arxiv.org/abs/2610.20968v1)

**Authors:** Stergios E. Bachoumas, Panagiotis Artemiadis

**Published:** 2026-10-14

**Categories:** cs.RO, cs.AI

**Abstract:**

Executing heavy-duty physical manipulation in unpredictable environments remains a major challenge for quadrupedal robots lacking dedicated manipulators. This paper presents SOMA, a hierarchical reinforcement learning framework for whole-body non-prehensile manipulation of multi-object stacks using a quadruped robot. Unlike traditional methods restricted to two-dimensional planar pushing, SOMA integrates explicit robot base-height control, unlocking full three-dimensional pushing strategies. This capability enables the quadruped to manipulate complex payload stacks as a unified system while preventing relative sliding or tipping of upper layers. The architecture factorizes control into a high-level manipulation policy using a Simplified Structured State Space Sequence (S5) encoder to reason over long-horizon temporal history and predict bounded target tracking commands via Beta distributions, and a low-level locomotion policy that dynamically adapts leg stance configurations. In extensive simulation benchmarks, SOMA achieves a $90.3\%$ success rate for single-object reposing and up to $90.1\%$ for dynamic two-box stacks under stringent error margins ($0.05\text{ m}$ position error, $1^\circ$ orientation error). Finally, we validate zero-shot sim-to-real transfer on a Unitree Go1 quadruped robot across diverse payload masses and geometries, demonstrating active base-height adjustments to maintain traction and balance during heavy pushing tasks.

### 论文解读

#### 摘要翻译
SOMA让没有专用机械臂的四足机器人用身体和腿部进行非抓取式推物，处理多个相互独立物体组成的堆叠载荷。分层强化学习策略联合控制基座高度与步态，尝试在推动中维持堆叠稳定。

#### 方法动机分析
四足平台有较强负载和地形适应能力，但缺少末端执行器；以往全身推物往往限制在平面，并依赖固定基座姿态。作者假设显式调节机身高度、配合历史感知策略，可以改变接触高度和推力方向，从而在三维空间重置堆叠物体而不过度滑移或倾倒。

#### 方法设计详解
策略分为两个时间尺度：低层S5步态控制器以50Hz输出腿部动作，约10万参数；高层S5操控器以10Hz读取机器人与物体状态和历史，输出有界的前后/侧向速度、偏航和基座高度目标。训练采用PPO并分三阶段逐步增加操控难度，再将低层运动与高层推物策略组合。高层通过机身高度调节接触点，借助全身动作接近、推送和重新摆放物体。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练阶段和运行设置包括：仿真训练约1小时20分，使用NVIDIA A5500 GPU；训练前需匹配机器人动力学、地面摩擦、物体质量和堆叠接触。复现应报告每个任务的试验数与成功阈值，不应仅凭示例视频判断泛化。迁移到搜救搬运可尝试推运大箱体，但需增加本体感知、碰撞保护和载荷滑移监控。

#### 方法对比分析
核心区别是将基座高度作为显式操控变量，而不是只调平面推力或依赖机械臂末端；这一创新适用于大载荷粗操作，精细抓取场景仍更适合带末端执行器的平台。

#### 实验分析（精简版）
仿真单物体重定位成功率90.3%；两箱动态堆叠最高90.1%，成功判据为位置误差小于0.05m且方向误差小于1°。实机Unitree Go1测试不同载荷与几何形状，论文展示多种配置，但代表性配置只有少量重复。仿真成绩不能替代更大规模的实机成功率，实验使用外部OptiTrack追踪也限制了自主感知结论。

#### 实用指南
仿真训练约1小时20分，使用NVIDIA A5500 GPU；训练前需匹配机器人动力学、地面摩擦、物体质量和堆叠接触。复现应报告每个任务的试验数与成功阈值，不应仅凭示例视频判断泛化。迁移到搜救搬运可尝试推运大箱体，但需增加本体感知、碰撞保护和载荷滑移监控。

#### 总结
核心思想：用机身高度拓展四足推物维度。
1. 观察物体与机器人状态历史。
2. 高层给出速度和机身高度目标。
3. 低层步态执行并稳定堆叠。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.20968v1)
- [arXiv](https://arxiv.org/abs/2610.20968v1)

---

<a id='2610.20901v1'></a>
## [TGrasp6D: Task-Informed Vision-Language Priors and Pose Refinement enable Zero-Shot Task-Oriented 6DoF Grasping](https://arxiv.org/abs/2610.20901v1)

**Authors:** Manav Kulshrestha, Apoorva Vashisth, Damon Conover, Aniket Bera

**Published:** 2026-10-14

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Task-Oriented Grasping (TOG) has emerged as an important skill for purposeful robotic manipulation. Given the task of using a knife to "slice bread," a robot must choose to grasp from the handle, whereas for "hand[ing] the knife over," it is preferable for the handle to remain unobstructed for safe handoff. Current approaches for TOG either focus on 2D planar grasping, remain limited by their reliance on task-agnostic grasp generators, or are constrained by reliance on finite, manually-annotated, task-specific grasp datasets that cannot cover all possible grasp-task pairs. To that end, we present TGrasp6D, a vision-language based approach for generating task-oriented 6DoF grasps in a zero-shot manner. Given a task and a single RGB-D image of an associated object, TGrasp6D reasons about the appropriate semantic affordance for the given task and object geometry to produce an initial task- informed prior. This generated prior is further refined via 6DoF pose optimization for collision avoidance while preserving task consistency. Our approach achieves state-of-the-art success rate of 94% on the TaskGrasp dataset, with the best baseline at 74%. We also deploy our approach on a Franka Research 3 robot to demonstrate zero-shot generalization to real world objects

### 论文解读

#### 摘要翻译
论文研究如何让机器人按自然语言任务选择六自由度抓取，而不是只追求几何上容易抓取。TGrasp6D先用视觉语言推理定位任务相关部位，再结合深度几何采样并优化抓取姿态，在避碰的同时尽量保留任务语义。

#### 方法动机分析
普通抓取器通常回答“哪里容易抓”，却不回答“怎样抓才能完成当前用途”：递刀时应留出刀柄，切面包时则要握住刀柄。有限的任务标注数据也难覆盖新物体和新任务。作者的核心假设是，通用视觉语言模型可从图像与指令推断语义部位和抓取轴，而RGB-D几何优化能把这种粗先验变成可行的6DoF姿态。

#### 方法设计详解
输入是一张RGB-D图像和任务文本。第一步，视觉语言模型识别目标及任务相关affordance，并估计其位置与合适抓取轴；第二步，将图像平面的affordance经相机标定提升到三维物体坐标；第三步，以该语义先验为中心采样候选抓取姿态；第四步，对姿态做批量几何优化，惩罚碰撞并约束任务一致性，输出可执行的抓取位姿。关键不是让语言模型直接生成关节动作，而是用它限定几何搜索的语义方向。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：论文使用单帧RGB-D作为输入，依赖相机内外参、物体点云与候选姿态优化；复现时需固定视觉语言模型版本、提示词、候选采样和碰撞阈值，否则模型方差会混入比较。文中没有证明长时闭环抓取，也未报告统一推理成本。应把模型产生的affordance当作提议，再用机器人自身的运动学、碰撞和力控模块验证。代码公开状态需以项目仓库后续更新为准。

#### 方法对比分析
它的创新贡献是把语言模型输出限定在“语义先验”层，再由几何优化负责执行可行性；相比直接让模型输出抓取姿态，这种分工更容易接入现有机器人规划栈。

#### 实验分析（精简版）
在TaskGrasp基准上，作者报告成功率94.5%，FoundationGrasp为73.7%；实机Franka Research 3测试5个物体、10种物体—任务组合，每组10次，TGrasp6D成功93/100次，对照为41/100次。去掉姿态精化后基准结果降至89.4%，说明先验与几何细化均有作用。实机规模仍小，且单RGB-D视角不能覆盖遮挡、接触力和闭环重规划问题。

#### 实用指南
论文使用单帧RGB-D作为输入，依赖相机内外参、物体点云与候选姿态优化；复现时需固定视觉语言模型版本、提示词、候选采样和碰撞阈值，否则模型方差会混入比较。文中没有证明长时闭环抓取，也未报告统一推理成本。应把模型产生的affordance当作提议，再用机器人自身的运动学、碰撞和力控模块验证。代码公开状态需以项目仓库后续更新为准。

#### 总结
核心思想：用任务语义引导几何抓取优化。
1. 图像与指令提取任务相关部位。
2. 将部位和抓取轴提升到三维并采样。
3. 优化避碰且符合任务的6DoF姿态。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.20901v1)
- [arXiv](https://arxiv.org/abs/2610.20901v1)

---

<a id='2610.21306v1'></a>
## [VLA-$Δ$: Learning Physical State Transition for Vision-Language-Action Models](https://arxiv.org/abs/2610.21306v1)

**Authors:** Qi Lu, Jiarun Li, Ce Zhang, Miao Pan, Yi Pan, Xu Wang, Wenqi Zhang, Xuhong Zhang

**Published:** 2026-10-15

**Categories:** cs.CV

**Abstract:**

Vision-language-action (VLA) models transfer semantic knowledge acquired through vision-language pretraining to robot control, achieving strong performance across various benchmarks. However, most VLAs directly map observations to demonstrated actions without explicit supervision of intermediate physical state changes, making them prone to overfitting observation-action correspondences. Some methods introduce intermediate reasoning or future-state prediction in language, visual, or latent spaces to guide action generation, but do not explicitly supervise the physical state transition itself. We therefore propose VLA-$Δ$, a general framework that uses physical changes between current and future states as explicit learning targets. To support this objective, we construct a large-scale pretraining dataset and derive physical state-transition annotations covering task-relevant semantics and visual motion changes, providing more direct supervision for continuous action generation. During inference, the model generates actions directly from current observations without additional inference overhead. VLA-$Δ$ achieves performance competitive with the state of the art on LIBERO and RoboTwin, and outperforms $π_{0.5}$ in real-robot experiments. Our code is available at https://github.com/ZJU-OmniAI/VLA-Delta

### 论文解读

#### 摘要翻译
VLA-Δ通过Physical State Difference Queries学习动作前后物理状态变化，再用该表示调节连续动作生成。训练阶段联合结构化状态槽、光流和未来状态差异文本监督，部署时只保留查询表征与动作生成通路。

#### 方法动机分析
直接从图像、语言映射动作，容易忽略物体运动、接触变化和任务约束；显式生成未来视频或文字又可能增加推理负担。作者假设可学习查询能够在预训练中承载紧凑、任务相关的状态转移信息，并将其作为动作模型的中间表征，而不必在部署时逐项解码辅助监督。

#### 方法设计详解
多视角图像、指令与机器人状态输入Qwen3-VL-2B骨干；可学习的Physical State Difference Queries从隐藏状态提取转移表示。预训练同时监督九类结构化槽位、稠密光流和描述未来状态变化的文本，三者分别覆盖语义变量、细粒度视觉运动和任务相关约束；动作专家以扩散Transformer生成动作流。下游微调只优化动作目标，推理时不解码辅助标签。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：预训练语料由多来源机器人视频自动生成，规模约484小时、2400万帧；复现要统一相机视角、机器人状态与动作时间戳，并检查槽位本体和未来文本质量。模型包含Qwen3-VL-2B与九层扩散动作专家。部署虽不额外解码监督目标，仍需满足骨干推理时延；迁移需重新审计标注偏差与任务安全约束。

#### 方法对比分析
创新贡献是用紧凑查询承载状态转移监督；相比直接VLA，它增加物理变化中间目标，相比预测完整未来视频，它减少部署时需要生成的内容并保留连续动作输出。

#### 实验分析（精简版）
PiPER双臂实机五类任务共200次测试，平均成功率68.0%；作者还报告LIBERO 98.4%及RoboTwin clean/random约90%。实机任务各40次，结果涉及不同基准和策略比较，不能将跨数据集数字直接横向排名。标注抽查300例，各字段有效率约93%–99.33%，这是格式审计而非语义精度评估。

#### 实用指南
预训练语料由多来源机器人视频自动生成，规模约484小时、2400万帧；复现要统一相机视角、机器人状态与动作时间戳，并检查槽位本体和未来文本质量。模型包含Qwen3-VL-2B与九层扩散动作专家。部署虽不额外解码监督目标，仍需满足骨干推理时延；迁移需重新审计标注偏差与任务安全约束。

#### 总结
核心思想：用状态变化查询指导动作学习。
1. 图像、语言与状态进入VLA骨干。
2. 查询接受槽位、光流和文本联合监督。
3. 动作专家生成连续控制序列。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.21306v1)
- [arXiv](https://arxiv.org/abs/2610.21306v1)

---

<a id='2610.21282v1'></a>
## [LICOA: Learning Instruction-Conditioned Optical Actions for Mobile Spatial Augmented Reality](https://arxiv.org/abs/2610.21282v1)

**Authors:** Yuchen Deng, Shunkai Zhou, Ruixuan Zhu, Haibin Ling, Bingyao Huang

**Published:** 2026-10-15

**Categories:** cs.RO, cs.CV

**Abstract:**

Spatial augmented reality (SAR) augments physical environments with projected content, but most systems assume that the projector pose, target region, or motion path has been fixed or specified. We instead make projected light an embodied action: given a language instruction and current visual observations, a system must decide how to move and where and at what scale to place the projection. We introduce LICOA to realize this capability. Its differential-drive chassis provides room-scale approach and repositioning, while a steerable mirror with two degrees of freedom (2DoF) and projection scale provide rapid local control of the beam. A shared interface maps synchronized external and onboard RGB cameras with robot control state to physical optical actions across pretrained action models. Because instruction-conditioned optical action lacks dedicated training data and a common evaluation protocol, we construct a matched simulation embodiment and a corpus with 150 tasks and 7,500 rendered episodes, then evaluate models trained only on this corpus in previously unseen real-world environments. Across 480 complete closed-loop real-world execution trials (rollouts) on 12 tasks, the best adapted action model achieves a 57.50% success rate and 40.64% success weighted by path length, compared with 8.33% and 5.15% for the same architecture trained from scratch. A further 150 real-world ablation rollouts support the need for both camera views and coordinated control of the chassis and mirror. These results establish a physical platform, shared perception and optical action interface, corpus, and real-world evaluation protocol for treating projected light as an embodied action.

### 论文解读

#### 摘要翻译
LICOA把投影光束看作可学习的物理动作：机器人底盘负责房间尺度移动，二维可转镜负责快速调整光束方向和投影尺度。论文构造统一视觉—动作接口与仿真训练集，再在新真实环境测试预训练动作模型。

#### 方法动机分析
传统空间增强现实系统的痛点是通常假定投影仪位置、目标区域或光路预先固定，难以按自然语言在陌生环境中决定“移动到哪里、把光投到哪里”。作者认为需要同时学习底盘移动与局部光学控制，并以外部和机载相机的互补观察支撑定位与投影。

#### 方法设计详解
动作接口接收指令、同步外部/机载RGB图像及机器人状态，输出底盘速度、镜面二维转角、投影缩放和终止信号。作者构建匹配的仿真机器人与150项任务、7500段渲染轨迹，并将同一接口接入π0.5连续流匹配、π0-FAST自回归及Fast-WAM模型。模型只在仿真数据训练，随后在真实陌生环境闭环执行；消融分别移除相机视角或底盘/镜面动作能力。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：训练语料为仿真渲染，实验用H200训练30k步、batch size 32，动作块长16；复现必须同步相机、镜面角度与投影尺度，并校准外部相机坐标。迁移到仓库指示或共享空间投影时需考虑光照、反光材质、移动行人和投影安全，当前论文主要验证室内桌面与地面任务。

#### 方法对比分析
方法创新是把底盘移动和投影光学控制视为同一策略的动作；相比固定投影或只转镜面的方法，它能联合选择站位和落点，但适用场景仍受相机和光学硬件约束。

#### 实验分析（精简版）
最佳适配π0.5在12项真实任务、每项10次的120次完整试验中成功69次，成功率57.5%；同架构从头训练为10/120，即8.33%。另有150次真实消融显示同时使用两种相机及底盘与镜面动作更有效。人类评估成功判据和特定室内设置限制了结果外推。

#### 实用指南
训练语料为仿真渲染，实验用H200训练30k步、batch size 32，动作块长16；复现必须同步相机、镜面角度与投影尺度，并校准外部相机坐标。迁移到仓库指示或共享空间投影时需考虑光照、反光材质、移动行人和投影安全，当前论文主要验证室内桌面与地面任务。

#### 总结
核心思想：把投影光束作为移动机器人的动作。
1. 视觉与指令编码为统一状态。
2. 策略联合预测底盘和镜面动作。
3. 在真实环境闭环调整投影。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.21282v1)
- [arXiv](https://arxiv.org/abs/2610.21282v1)

---

<a id='2610.20998v1'></a>
## [MoCaP: Mobile Code-as-Policies for Zero-Shot Mobile Manipulation from Passive Human Videos](https://arxiv.org/abs/2610.20998v1)

**Authors:** Yiyang Cai, Tatsuya Kamijo, Ninghao Lu, Zijin Hu, Sherry Yang

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Passive human videos of everyday activity offer robots access to a massive amount of human experience without requiring a demonstration for each task. However, using this experience requires recovering knowledge from unstructured video recordings and adapting it to a different body (i.e., the robot). To overcome these challenges, we present mobile code-as-policy (MoCaP), an agentic framework that turns passive human video recordings into guidance for an end-to-end mobile manipulation policy. MoCaP extracts spatial memory and cues of how people prepare to interact with objects, which inform navigation and the search for a manipulation-ready configuration. Guided by passive human experience and current observations, MoCaP uses a closed-loop code-as-policy to coordinate planning and tool use primitives to solve motions under the robot's geometry and kinematic constraints. On a real mobile manipulator, MoCaP achieves 80% success on each of two mobile manipulation tasks in an indoor environment whereas existing approaches cannot succeed at the task. Video-derived spatial memory improves object-navigation success from 70% to 100%, and human approach prior raises the rate of finding a reachable base pose from 0% to 100% at an unfavorable start location. Videos and code are available on the project website: https://mocap-7.github.io/MoCAP-page/.

### 论文解读

#### 摘要翻译
MoCaP从日常被动人类视频提取可复用空间记忆与人类操作前的接近方向，再结合当前机器人观测和运动原语生成闭环移动操作程序。作者在真实移动操作平台上测试视频经验能否帮助导航和找到可操作的基座位置。

#### 方法动机分析
机器人通常需要逐任务示教，而普通人类视频并不包含机器人坐标或明确的动作标签；仅到达物体附近也不代表底盘处于可抓取姿态。作者假设视频能提供“在哪里找”和“人会从哪一侧接近”的弱先验，再由机器人实时几何与运动学检查将其转化为本体可行的操作。

#### 方法设计详解
离线阶段从被动视频估计场景结构、物体位置关系和人开始交互前的站位，形成空间记忆及接近方向；在线收到任务后，启动顾问结合任务与初始视图定位目标，再将人体视角中的准备姿势投影为机器人接近先验。代码策略调用导航、可见目标移动、抓取姿态采样、机械臂规划等机器人原语；执行中检查碰撞和可达性，依据反馈重新规划。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练阶段和运行设置包括：系统使用YOR移动操作平台、RGB-D感知与现成导航/抓取原语；在线推理按任务请求生成程序，底层原语执行后再读取反馈；复现关键在视频相机位姿、场景坐标对齐、人体准备帧检测和机器人可达性验证。论文提供视频和代码链接。迁移到仓储拣选可复用同一货架区域的经验，但动态布局、遮挡或人的行为差异会使记忆先验失效，需在线核验。

#### 方法对比分析
创新贡献是把视频中的人类接近姿态转换成机器人本体可达性约束；相比直接复制人的轨迹，它强调几何可行性筛选，适用于有导航和抓取原语的移动操作系统。

#### 实验分析（精简版）
真实平台上抓箱与抓罐各20次，成功率均为80%；视频空间记忆将物体导航成功率从70%提高到100%。在不利初始位置，人体接近先验找到可达基座姿态5/5，朴素近距离对接0/5。样本量小、仅一个室内环境，且基础机器人原语仍承担执行难题，不能推出普遍零样本能力。

#### 实用指南
系统使用YOR移动操作平台、RGB-D感知与现成导航/抓取原语；在线推理按任务请求生成程序，底层原语执行后再读取反馈；复现关键在视频相机位姿、场景坐标对齐、人体准备帧检测和机器人可达性验证。论文提供视频和代码链接。迁移到仓储拣选可复用同一货架区域的经验，但动态布局、遮挡或人的行为差异会使记忆先验失效，需在线核验。

#### 总结
核心思想：从人类视频提取可行接近经验。
1. 离线整理场景记忆和操作前站位。
2. 在线匹配任务并生成接近先验。
3. 调用机器人原语，依据反馈重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.20998v1)
- [arXiv](https://arxiv.org/abs/2610.20998v1)

---

<a id='2610.20936v1'></a>
## [LEAP: Gradient Flow Augmented Lagrangian Trajectory Optimization for Contact Constrained Robots](https://arxiv.org/abs/2610.20936v1)

**Authors:** César E. Ramos Chuquiure, Ryan Tsai, Yutong Zhou, Zihang Huang, Vansh Thakur, Akshunn Trivedi, Zachary Lin, Pearl Lin, Ruikai Yang, Challen Enninful Adu, Ram Vasudevan

**Published:** 2026-10-14

**Categories:** cs.RO

**Abstract:**

Generating dynamically feasible motions for legged and manipulating robots requires solving a large, nonlinear, non-convex optimal control problem (OCP) in which contact wrenches must satisfy friction-cone constraints throughout the motion, not only at isolated instants. Existing solvers are sensitive to the initial guess and scale poorly with the number of joints. This paper introduces LEAP, which transcribes the OCP with pseudospectral collocation and, instead of a Newton or SQP solve, integrates a primal--dual gradient flow of a Powell--Hestenes--Rockafellar augmented Lagrangian to compute a solution. Because the flow is stiff, this paper designs a linearly implicit integrator with a Gauss--Newton metric, momentum, and a merit-based step controller that keeps the augmented Lagrangian monotone. Across locomotion benchmarks on five robot platforms, from planar hoppers to 32-joint humanoids, LEAP is the only method to solve every tested problem, achieves substantial speedups over Crocoddyl and Aligator on several benchmarks, and reaches trackable references at smaller compute budgets. The resulting trajectories are validated by tracking them in a high-fidelity simulator and on hardware. We will make the code public upon final acceptance of this paper.

### 论文解读

#### 摘要翻译
LEAP面向含接触约束的机器人轨迹优化，将最优控制问题离散为伪谱节点上的约束问题，再用带稳定化设计的原始—对偶梯度流求解增广拉格朗日。目标是在多机器人、多接触任务中提升可行性与计算效率。

#### 方法动机分析
接触丰富运动同时包含非线性动力学、摩擦锥约束和接触模式切换，Newton/SQP类方法对初值敏感，规模增大后也可能难以收敛。作者的假设是，将连续时间轨迹用高阶伪谱节点表达，并沿增广拉格朗日的梯度流推进，可以更稳健地获得可行参考轨迹；关键难点是梯度流刚性和单调下降。

#### 方法设计详解
先由用户给定接触序列，建立混合动力学最优控制问题；用Chebyshev–Gauss–Lobatto伪谱节点离散状态、控制和接触力，并以Bezier控制点约束保证节点间摩擦锥可行。随后构造Powell–Hestenes–Rockafellar增广拉格朗日，以原始—对偶梯度流更新轨迹和乘子；针对刚性采用线性隐式积分器、Gauss–Newton度量与动量，再由merit函数步长控制维持增广目标单调。最后把优化轨迹交给高保真仿真或硬件跟踪。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的推理时延与运行设置包括：复现需要精确动力学、接触序列、摩擦参数和稳定的轨迹尺度化；需同时报告优化时间、可行性残差和闭环跟踪误差。论文称最终接收后公开代码，当前复现依赖代码发布情况。迁移到机械臂接触操作时可沿用约束优化框架，但必须重建接触模型并验证Bezier可行性界与真实摩擦偏差。

#### 方法对比分析
方法创新贡献是以梯度流和单调merit控制处理接触约束；相比局部Newton/SQP求解器，它强调稳定推进和较宽的初始化范围，但接触序列仍由外部给定。该方法的适用场景是接触安排已知、而数值求解鲁棒性仍是瓶颈的规划任务。

#### 实验分析（精简版）
论文在五类机器人、十个基准问题上测试，LEAP解决全部10项，Crocoddyl为7项、Aligator为8项；多个基准中更快达到可跟踪精度。实机Unitree Go2完成跳格轨迹，跟踪关节位置RMS误差为0.0682 rad。结果覆盖仿真和一项硬件任务，但不等同于对任意复杂接触环境都保证全局最优。

#### 实用指南
复现需要精确动力学、接触序列、摩擦参数和稳定的轨迹尺度化；需同时报告优化时间、可行性残差和闭环跟踪误差。论文称最终接收后公开代码，当前复现依赖代码发布情况。迁移到机械臂接触操作时可沿用约束优化框架，但必须重建接触模型并验证Bezier可行性界与真实摩擦偏差。

#### 总结
核心思想：用稳定梯度流求接触可行轨迹。
1. 指定接触序列并离散动力学。
2. 构建增广拉格朗日及摩擦约束。
3. 稳定积分求解，再跟踪验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.20936v1)
- [arXiv](https://arxiv.org/abs/2610.20936v1)

---

<a id='2610.20925v1'></a>
## [Planning to Learn: Real-Time Robust Adaptive Control on the GPU with Guaranteed Future Learning](https://arxiv.org/abs/2610.20925v1)

**Authors:** Anutam Srinivasan, Siddhant Panse, Glen Chou

**Published:** 2026-10-14

**Categories:** cs.RO, cs.LG, eess.SY, math.OC

**Abstract:**

Robust adaptive control is vital for keeping robots safe while actively reducing uncertainty in unfamiliar settings. However, traditional methods are often conservative since they fail to account for guaranteed future parameter learning during planning and struggle to scale to high-dimensional systems. In this work, we present A-SLS, a learning-aware robust adaptive model predictive control (RAMPC) framework based on system-level synthesis. A-SLS forecasts future parameter uncertainty and preserves correlations between process disturbances, parameter error, and estimate updates to construct tight, guaranteed reachable sets that account for future information gain. We also develop a GPU-accelerated A-SLS solver to enable real-time performance. Across nonlinear robotic systems, A-SLS reduces reachable-tube conservativeness, maintains real-time performance, and scales to whole-body control for a quadruped with 61 states, 8 uncertain parameters, and 12 control inputs. We further demonstrate the approach on hardware, where a physical Crazyflie drone learns an uncertain dynamics parameter online to stabilize on the fly.

### 论文解读

#### 摘要翻译
论文提出A-SLS，在鲁棒模型预测控制中显式预测未来在线辨识会怎样缩小参数不确定性，并把这一学习收益纳入安全规划。方法用系统级综合表示闭环响应，通过GPU求解器支持高维机器人控制。

#### 方法动机分析
传统鲁棒控制常把未知动力学参数当作始终不变的最坏情况不确定性，因此安全但保守；普通自适应控制虽会学习，却难在规划时保证学习期间的约束安全。作者假设参数估计更新与过程扰动之间的相关性可以被集合传播保留下来，从而可信地预测未来可达管束收缩，并在安全边界内主动利用信息。

#### 方法设计详解
A-SLS把系统状态和待估物理参数共同纳入预测。求解器在system-level synthesis变量上优化控制策略，同时传播扰动集合、参数误差和估计更新的相关性；在线辨识使用截断伪逆更新，并以zonotope等集合表示避免把相关误差粗略独立化。每个MPC周期据未来估计精度计算可达管束，再检查鲁棒约束。GPU并行线性代数让这一过程覆盖四足机器人的61维状态、8个阻尼参数和12路控制输入。这个处理流程把感知、状态表示、优化和执行分开，便于逐段检查误差来源；部署时仍需用目标平台的真实传感器与动力学验证模型假设。论文的训练/运行设置包括：复现重点是区分随机扰动与可学习参数、正确维护估计协方差和集合相关性，并在GPU上测完整控制周期而非只测矩阵核。论文展示Crazyflie和仿真四足，不意味着已对其他硬件验证。迁移前需检查执行器饱和、传感噪声、参数可辨识性及计算最坏时延；可先在低维系统验证集合传播，再扩到全身控制。

#### 方法对比分析
创新点在于把“控制会带来信息”写进未来不确定性传播；相比固定最坏情况集合，它能随在线辨识收紧约束，但相较无约束自适应策略仍保留鲁棒安全检查。

#### 实验分析（精简版）
四足仿真中，A-SLS在40次试验中成功38次，基线SLS成功24次；中位计算时间16.1ms，满足32ms控制周期。Crazyflie硬件实验中位计算时间8.15ms，低于50ms周期，展示了约2m飞行中的在线推力参数学习。硬件实验次数和场景有限，不能据此推断任意未知动力学下均有安全保证。

#### 实用指南
复现重点是区分随机扰动与可学习参数、正确维护估计协方差和集合相关性，并在GPU上测完整控制周期而非只测矩阵核。论文展示Crazyflie和仿真四足，不意味着已对其他硬件验证。迁移前需检查执行器饱和、传感噪声、参数可辨识性及计算最坏时延；可先在低维系统验证集合传播，再扩到全身控制。

#### 总结
核心思想：把未来学习纳入安全鲁棒规划。
1. 建模状态、扰动与未知参数。
2. 预测估计更新及不确定管束变化。
3. 在鲁棒约束下优化并执行控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.20925v1)
- [arXiv](https://arxiv.org/abs/2610.20925v1)

---

