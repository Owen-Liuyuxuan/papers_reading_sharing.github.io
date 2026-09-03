time: 20260903

# Arxiv Computer Vision Papers - 2026-09-03

## Table of Contents

1. [Towards Zero-Shot Transfer Across Embodiments For Driving VLAs](#2609.02341v1)
2. [Contact-Constrained Lower-Limb Joint-Offset Calibration for Humanoid Robots](#2609.02306v1)
3. [WildFab: Multi-Axis 3D Printing from Models in the Wild](#2609.02413v1)
4. [Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis](#2609.02219v1)
5. [KSG-Net: Key-Sparse and Global-Context Learning for Maritime 3D Ship Detection](#2609.02077v1)
6. [InsightSeg: Reusing Correction Insights for Guideline-Consistent Segmentation](#2609.02002v1)
7. [MS-MEM: Multi-Skill Manipulation-Enhanced Mapping via Uncertainty- and Disturbance-Aware Action Selection](#2609.02493v1)
8. [A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care](#2609.02402v1)

---

## Papers

<a id='2609.02341v1'></a>
## [Towards Zero-Shot Transfer Across Embodiments For Driving VLAs](https://arxiv.org/abs/2609.02341v1)

**Authors:** Caio Azevedo, Stefano Sabatini, Sascha Hornauer, Fabien Moutarde

**Published:** 2026-09-02

**Categories:** cs.CV

**Abstract:**

Vision-Language-Action models (VLAs) have shown strong potential in autonomous driving by leveraging multimodal pretraining for instruction following, visual reasoning, and scene-level generalization. In robotic manipulation, scaling VLA fine-tuning across multiple robot setups--especially when unifying representations across embodiments--has been shown to improve in-dataset performance and cross-embodiment generalization; in autonomous driving, however, VLAs remain largely trained on individual datasets and are rarely evaluated for zero-shot transfer to unseen datasets and camera rigs; furthermore naively adding more datasets to the training data does not necessarily lead to better performance within seen embodiments. To address these problems, we study multi-dataset training for the driving task and BEV-Forcing, an auxiliary objective that transfers ground-plane object-layout information from a specialized Bird's-Eye-View model into the VLA backbone. By encouraging the model to represent object position through a shared BEV spatial interface, we show that an auxiliary task such as BEV-Forcing can improve both in-distribution and out-of-distribution performance when training on a small number of camera rigs. As the number of training embodiments increases, however, the benefits of the auxiliary task are reduced; we present this as evidence that new techniques in the literature may see their benefits diminish when simply scaling up training diversity, which motivates presenting results taking into account data scaling.

### 论文解读

#### 研究问题与动机

视觉-语言-动作模型（VLA）正在被用于自动驾驶：它们可以理解图像和语言，再输出车辆未来轨迹。但这类模型常在固定相机布局上训练，换到不同的相机安装位置、视角或新数据集时，空间关系容易失效。也就是说，模型可能认识“前方有车”，却不能稳定判断车辆在地面上的准确位置，这是一种不同于语义泛化的几何迁移问题。

论文提出 BEV-Forcing，目标是在训练阶段把空间结构明确写入 VLA 的视觉表征，同时不增加部署时的推理负担。作者用跨 embodiment 的数据来检验模型是否真的学会了可迁移的几何，而不是记住某一套传感器配置。

这个问题对端到端驾驶尤其重要：不同车辆的相机高度、水平视场和相对朝向都会改变图像中的投影关系。若模型只把驾驶数据当作动作模仿样本，增加语言推理能力也未必能修复轨迹偏移。因此，作者把“是否能在未见过的传感器布局上规划”作为比单一数据集分数更严格的泛化检验，并将地面占用布局作为连接视觉输入与轨迹输出的中间几何证据。

#### 核心方法

方法为 VLA 增加一个训练期的鸟瞰图（BEV）占用预测任务。SimpleBEV 教师模型根据多视角图像生成占用网格，VLA 最后一层的图像特征则通过轻量 BEV head 预测相应的地面平面布局。这个 head 使用随机初始化的查询对图像 hidden states 做 cross-attention，再输出每个网格是否被道路参与者占据的结果；占用预测采用 binary cross-entropy。

训练时，标准的轨迹 next-token prediction loss 与 BEV loss 加权相加。动作以文本 waypoints 表示，轨迹频率为 1 Hz，并用 cubic spline 重采样。实验 backbone 是 Qwen 3.5 2B，采用 LoRA rank 16 微调，学习率 `1e-4`、cosine decay、global batch size 64，在 4 张 A100 GPU 上训练。推理阶段直接移除 BEV head，所以辅助任务不会带来额外部署成本。

训练数据包括 Waymo WOD-E2E、NAVSIM、nuScenes，以及用于维持视觉问答和语言能力的 nuScenes-QA。不同数据集被整理成统一样式，每个 planning 样本使用历史轨迹、三幅相机图像和车辆意图。这样的设计让辅助目标作用于共享图像特征，而不是额外依赖某个特定数据集的传感器格式；同时，低容量的 head 使 backbone 必须承担空间编码，而不是把任务完全交给一个强大的附加模块。

#### 实验结果

在只用 Waymo WOD-E2E 训练、再迁移到 Physical AI 数据的设置中，BEV-Forcing 将 ADE 从约 2.05 m 降至约 1.84 m，改善 10.1%。这说明显式的地面布局监督可以帮助模型适应训练中未出现的相机设置。

在 KITScenes LongTail 上，训练组合为 Waymo 加 nuScenes 并使用 BEV task 的模型取得 MMS 5.15、L2 误差 2.48 m；Gemini 3 Pro 的对应结果为 MMS 4.61、L2 2.99 m，Alpamayo 1.5 的 MMS 为 4.31。消融实验中，轻量 cross-attention head 的 ADE 为 2.376 m，完整 Transformer block 为 2.441 m；BEV-Forcing 的 ADE 为 2.319 m，也优于 Spatial Forcing 的 2.341 m。

这些结果覆盖了绝对质量和相对改进两个层面。KITScenes 上，本文模型的 MMS达到5.15，而 Gemini 3 Pro 只有4.61，L2误差也由2.99 m降低至2.48 m；在结构消融中，cross-attention方案的 ADE达到2.376 m，完整 Transformer 方案为2.441 m，表明增加 head 容量并没有带来收益。BEV-Forcing 相比 Spatial Forcing 的 ADE由2.341 m降低至2.319 m，虽然差距不大，却支持直接预测占用布局比重建中间 BEV 特征更合适的判断。

#### 局限性与意义

当训练数据包含更多 embodiment（例如再加入 NAVSIM 和 nuScenes）时，BEV 辅助任务的相对收益会减小，说明足够多样的相机数据也能促成几何泛化。此外，驾驶数据扩展可能削弱语言推理，nuScenes-QA 共训对保持语义能力很重要。WOD-E2E 上，加入多源数据、问答和 BEV task 的模型 RFS 为 7.902、ADE 为 2.938 m，表明跨域泛化与单一数据集上的最佳规划分数并非同一目标。

该方法依赖 SimpleBEV 教师，教师误差可能影响占用监督；论文也主要验证相机 rig 的变化，尚不能保证对天气、城市环境或交通规则变化同样稳健。尽管如此，BEV-Forcing 提供了一种低成本的几何正则化思路：训练时显式要求模型理解地面布局，部署时保持原有开销，并让端到端驾驶模型的跨相机鲁棒性可以被更直接地检验。

还应注意，本文并未声称 BEV 监督能解决所有驾驶安全问题。占用图主要描述地面上的空间布局，不能完整表达意图、规则、遮挡后的不确定性或长尾交互；SimpleBEV 作为教师也可能把自身偏差传给学生。未来可以研究更可靠的几何标签、时序占用和不确定性建模，并在更多车辆、镜头和真实道路条件下测试。就目前证据而言，这项工作最重要的意义是把跨相机泛化从模糊的“模型看起来更聪明”转化为可训练、可比较的空间表征问题。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02341v1)
- [arXiv](https://arxiv.org/abs/2609.02341v1)

---

<a id='2609.02306v1'></a>
## [Contact-Constrained Lower-Limb Joint-Offset Calibration for Humanoid Robots](https://arxiv.org/abs/2609.02306v1)

**Authors:** Kaixiang Lu, Haiyu Lan, Chunxiao Qiao, You Li, Chengyuan Luo, Enyu Li, Peiwen Lin, Chuang Wang

**Published:** 2026-09-02

**Categories:** cs.RO

**Abstract:**

Accurate joint encoder offsets are essential for kinematic consistency in humanoid lower limbs, yet existing calibration methods typically require external motion-capture systems or fiducial targets. We present a self-contained calibration framework exploiting only onboard joint encoders and a pelvis-mounted IMU during static double-support contact. The inter-foot transform from forward kinematics must stay constant when both feet are fixed; minimizing its posture-dependent dispersion yields a nonlinear least-squares problem over the 12-dimensional offset vector. A Hessian eigenstructure analysis shows that parallel pitch axes induce a rotational coupling. Orientation residuals then observe only the pitch-offset sum, while translation and posture diversity set the remaining numerical observability. For the A3 pitch-to-roll-to-yaw ordering, hip-roll and hip-yaw excitation reduce hip-pitch coupling. A standing-posture knee prior then anchors the remaining weak pitch-chain decomposition. Simulation and real-machine injection tests show consistent recovery, and on held-out recordings calibration reduces foot-height RMS residuals from 4.26 to 2.20 mm on A3 and from 8.03 to 1.43 mm on A2. An independent LiDAR-inertial reference checks the pitch-coupled channel. Removing an injected pitch offset moves the leg-odometry vertical drift back toward the LiDAR trajectory. A few static double-support stances thus provide contact-consistent corrections for well-excited directions. Individual offsets in the weak pitch chain remain prior-dependent.

### 论文解读

#### 研究问题与动机

人形机器人依赖准确的关节编码器零偏来计算脚的位置和姿态。哪怕每个关节只有不到一度的误差，沿着腿部串联后也可能造成厘米级的脚端误差，进而影响平衡、落脚和接触规划。现有方法常借助动作捕捉、激光跟踪器或外部标记，难以在真实机器人上反复使用；头部相机也不一定能看到脚。

这篇论文利用机器人自身的接触状态解决这个问题：当两只脚稳定踩在地面上、机器人改变下蹲或弓步姿态时，两脚之间的相对位姿应该保持不变。因此，如果模型预测的双足相对位姿随姿态明显变化，就可以把这种变化用于反推关节偏置。系统只需要腿部编码器和安装在骨盆上的 IMU，不需要外部视觉或测量设备。

#### 核心方法

作者把左右腿的 12 个固定关节偏置加入编码器读数，再用机器人的 URDF 做正向运动学。对每个双支撑片段，计算左脚坐标系中的右脚变换，并在 SE(3) 上求这些变换的平均值；所有帧相对该平均值的平移和旋转偏差共同构成接触残差。骨盆 IMU 提供重力方向，方法进一步要求两只脚等高、脚底法向与地面垂直，从而形成平面残差。

这些残差与偏置正则项组成非线性最小二乘问题。算法交替更新双足相对位姿均值和 12 维偏置，通常 3–4 次外层迭代就能收敛。数据采集采用正常深蹲、左弓步、右弓步和宽蹲四类双支撑姿态，每个片段保留约 450 个姿态。一个重要的可观测性结论是，髋俯仰、膝和踝俯仰的平行转轴会让姿态信息主要只能确定三者偏置之和，而不能可靠区分每一项。A3 通过髋横滚和髋偏航激励打破部分平行关系，再用约 5 秒的直腿站立膝先验选择剩余分解；A2 则使用更强正则，因此单个俯仰偏置更依赖先验。

#### 实验结果

在 MuJoCo A3 仿真中，作者注入一组已知的多关节偏置后重新估计，12 个关节的 RMS 误差为 0.120°，最大误差为 0.274°，最大误差出现在髋俯仰关节，符合其弱耦合方向的理论分析。受控的同几何仿真只改变近端关节顺序：R-Y-P 顺序的雅可比条件数为 348.90、信息矩阵条件数为 121728.69；P-R-Y 顺序分别为 90.69 和 8224.49，说明排序会显著影响数值可观测性。

真实机器人注入测试中的配对一致性误差 RMS 为 A2 的 0.012°、A3 的 0.061°。在未参与优化的平地记录上，A3 的脚高 RMS 残差从 4.26 mm 降到 2.20 mm，A2 从 8.03 mm 降到 1.43 mm。A3 还用独立 LiDAR-惯性里程计检查俯仰—竖直通道：16 段留出行走数据的竖直误差由 319±258 mm 降至 238±137 mm，下降 25.5%；总三维误差由 576±274 mm 降至 494±263 mm，下降 14.2%。此外，独立光学双足参考中的相对朝向误差从 5.223° 降至 4.085°，改善 21.8%。这些结果覆盖仿真注入、真实注入、留出平地一致性和外部传感器交叉检查，证据链较完整。

#### 局限性与意义

最重要的限制是：看起来能拟合数据，不等于每个关节偏置都被数据独立观测。特别是 A2 的髋俯仰—膝—踝俯仰链，正则化只能稳定地选出一个解，不能创造缺失的信息；改变膝先验中心会明显改变单关节估计，而俯仰偏置之和相对稳定。因此实际使用时应报告耦合的 pitch sum，并把机械零点、URDF/CAD 误差和正则权重作为结果的一部分。

方法还假设双脚没有滑动且接触面近似共面，IMU 倾角误差、接触不稳、模型几何误差和姿态激励不足都会造成偏差。LiDAR 交叉实验只验证俯仰—竖直通道，并不能证明所有关节绝对零点准确；论文也没有声称代码或数据已开源。与最接近的脚底高度方差方法在相同仿真上并非总是赢家。

尽管如此，这项工作的实用价值很明确：几段静态双支撑数据就能改善接触一致性，适合工厂出厂检查、维护后的现场重标定和行走前诊断。对新机器人迁移时，首先应设计能激励横滚、偏航和不同腿长姿态的双支撑动作，随后用独立平地或 LiDAR 数据验证；对于平行俯仰链，则应把“可观测的组合量”和“依赖先验的个体量”分开解释。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02306v1)
- [arXiv](https://arxiv.org/abs/2609.02306v1)

---

<a id='2609.02413v1'></a>
## [WildFab: Multi-Axis 3D Printing from Models in the Wild](https://arxiv.org/abs/2609.02413v1)

**Authors:** Jiasheng Qu, Zhikai Shen, Chenyu Xu, Hailin Sun, Chengkai Dai, Yuhu Guo, Junpeng Wang, Yeung Yam, Guoxin Fang

**Published:** 2026-09-02

**Categories:** cs.GR, cs.RO

**Abstract:**

Multi-axis 3D printing enables support-free fabrication and improved part quality, but robustly processing real-world geometries remains challenging. Models from design workflows or direct data acquisition often contain solid--shell combinations and non-manifold structures. Handling such models in the wild typically requires time-consuming geometry repair, which may alter the intended geometry. In this work, we present WildFab, a computational framework for multi-axis 3D printing that directly computes spatial toolpath and global collision-free motion from input models. Our pipeline builds on a hybrid query representation that combines a neural unsigned distance field (UDF) with a regularized generalized winding number field (reg-GWN). The UDF supplies differentiable surface-distance and direction queries, while the reg-GWN resolves near-surface ambiguity in the fitted UDF by providing reliable surface localization and a solid-void indicator. Based on this representation, we introduce a high-precision spatial toolpath computation algorithm that iteratively projects points between optimized guidance-field level sets and reg-GWN gradient-magnitude ridges. Subsequently, we develop an efficient and robust coarse-to-fine collision checking scheme for motion planning: UDF-based rejection first identifies potential collisions, while time-varying reg-GWN verification accurately resolves collision pairs for both solid and shell components. We validate WildFab on diverse inputs, demonstrating successful computation from non-manifold parametric surfaces, voxelized topology-optimization results, implicit models, raw scanned point clouds, and non-watertight meshes. The fabrication results highlight our method's ability to advance end-to-end design-to-3DP workflows.

### 论文解读
#### 研究问题与动机
多轴3D打印通常从水密、方向明确的实体网格开始，但真实设计与扫描数据往往并不“干净”：模型可能有非流形连接、自相交、开口表面、嵌套壳体，甚至同时包含实体和薄壳。现有流程先用布尔运算、偏置或封洞修复，既耗时，也可能改变设计意图，例如填掉应保留的空腔或增厚细小结构。基于有符号距离场的方法还依赖明确的内外部，而无符号距离场虽然能描述任意表面，却容易在表面附近出现方向歧义与不稳定响应。

WildFab的目标是绕过人工修复，直接把“野外模型”变成免支撑、无碰撞的多轴打印轨迹。它的核心假设是，UDF擅长提供连续的几何距离和方向，广义绕数擅长提供更稳定的整体结构线索；将二者结合，可以同时解决边界定位、路径生成和动态碰撞检查，并让同一套几何查询贯穿规划全过程。

#### 核心方法
系统用混合查询场表示空间点：一支是神经无符号距离场，学习点到最近表面的距离；另一支是带正则化核的广义绕数场。正则化消除了采样表面处的核奇异性。WildFab不把UDF的零值直接当作边界，而是沿负UDF梯度方向寻找正则绕数梯度模的脊线，即一阶方向导数为零且二阶导数为负的位置。这个设计把UDF的局部几何信息与绕数的全局结构信息结合起来；即使输入三角形法向方向翻转，梯度模脊线仍能稳定定位表面。

在路径生成阶段，作者优化一个标量引导场，联合考虑表面平滑、免支撑和边界保护。免支撑约束采用45°悬垂角，边界保护项使开口边缘不被当成需要封闭的破损。随后沿引导场追踪路径点，并在路径水平集与表面脊线之间迭代投影。运动规划则把碰撞、免支撑和姿态平滑纳入统一目标：先用UDF快速筛掉明显安全的候选，再用随打印对象增长的时变正则绕数场做精确验证，从而检查喷嘴与动态几何之间的真实交叉。

实现中，场网络采用5层、隐藏维度256的SIREN，使用Adam、学习率1×10^-4训练；UDF训练还加入边界、Eikonal和扩散约束以抑制ghost几何。推理时以最大深度16的八叉树加速UDF查询。实验使用Intel i9-14900K和24 GB显存的NVIDIA RTX 4090，实际打印平台为6自由度ABB机器人加2自由度旋转台。

#### 实验结果
作者从Thingi10K选取20个具有非流形或开口结构的挑战模型，并测试参数化B-rep、扫描点云和体素输入，比较S³-Slicer、INF-3DP、CAP-UDF与Cotangent等方法。以Lucy为例，混合场重建F-score达到98.98%，高于CAP-UDF的93.59%，说明绕数线索能显著改善UDF单独重建的边界质量。总体上，正则绕数脊线的表面定位误差比纯神经UDF低约50倍。

效率和制造结果也支持该设计：对多数超过100,000个点的模型，工具路径点生成耗时少于10秒；粗到细碰撞检查相较完全使用时变广义绕数检查获得约73倍加速。Kitten和Klein Bottle的CT扫描显示，打印件平均距离误差为0.7 mm，约占模型尺寸的0.5%。这些结果表明系统不仅能生成几何上可用的轨迹，也能在机器人平台上保持可接受的成形精度。

#### 局限性与意义
WildFab最有价值的地方在于把“输入必须先修复”改为“表示和查询本身容忍缺陷”：开口边缘可以被保护，非流形与混合实体/壳体可以直接参与路径规划，并且碰撞检测面向打印过程中的动态增长对象。该混合场思路还可能迁移到平面打印、焊接等需要沿复杂表面运动并规避碰撞的机器人任务，也为面向制造的几何表示提供了新方向。

不过，UDF训练仍约占总运行时间的70%，是部署前的主要成本；极尖锐或极小尺度的特征可能因神经场平滑而损失。碰撞规划还依赖正则宽度与采样分辨率，复杂模型上的精度—速度折中需要进一步研究。因此，论文展示的是对真实输入更鲁棒的完整管线，而不是消除了所有几何重建误差的通用解。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02413v1)
- [arXiv](https://arxiv.org/abs/2609.02413v1)

---

<a id='2609.02219v1'></a>
## [Hardware-Accelerated Instance Segmentation for Resource-Constrained Space Robotics with Criticality Analysis](https://arxiv.org/abs/2609.02219v1)

**Authors:** Siddhant Shete, Hilmi Dogu Kücüker, Udo Frese, Frank Kirchner

**Published:** 2026-09-02

**Categories:** cs.RO, cs.AR, cs.CV, cs.LG

**Abstract:**

Autonomous lunar missions require real-time per- ception under three coupled constraints: extreme low-light conditions, limited onboard compute, and radiation-induced hardware faults that can silently corrupt inference. We present a deployment-oriented instance segmentation framework for resource-constrained lunar robotics that jointly addresses quan- tization calibration and system-level fault exposure under strict compute constraints. First, we introduce Activation Variance Informative Sampling (AVIS), a label-free calibration strategy that deterministically selects calibration samples based on activation variance statistics. Second, we deploy a YOLO-based segmentation model on a Deep Learning Processor Unit (DPU) with architectural modifications that reduce CPU fallback paths and enable statically compiled execution with bounded latency in low-lighting conditions. We further introduce a software-level criticality analysis to estimate fault exposure and guide mitigation under radiation-constrained operation. On a lunar micro-rover platform, AVIS with bias correction recovers 69.8% of quantization-induced accuracy loss while achieving 309 ms inference latency and 5.7 W power consumption. Targeted mitigation reduces global criticality by 31.7%. The results demonstrate an integrated approach and a blueprint for a reliable and safe AI perception framework under space deployment constraints.

### 论文解读

#### 研究问题与动机

月球极区的视觉系统同时面对低光、强阴影、月壤反光、车载功耗限制和辐射造成的比特翻转。对岩石和陨石坑进行实例分割，可以提供比检测框更精确的几何边界，帮助探测车避障，但常规浮点模型难以在微型月球车上稳定运行。论文以 LuNiS 微型月球车为对象，尝试把精度、功耗、确定性和辐射风险放在同一个部署方案中解决。

#### 核心方法

框架采用 YOLOv8m 实例分割模型，训练数据为 25,000 张合成和自采的月球模拟图像。首先提出 AVIS（Activation Variance Informative Sampling）：让候选图像通过网络，统计多层中间激活图的空间方差，平均得到每张图的“信息量”分数，过滤分数为零的图像后确定性地选择 Top-K 样本作为 INT8 校准集。它不需要标签，也没有随机抽样，约可用一半校准数据获得更稳定的激活尺度。之后再使用偏差校正，补偿量化带来的层级激活偏移。

为了适配 Zynq UltraScale+ MPSoC 上的 DPU，作者将 SiLU/Swish 替换为 HardSwish，把会触发 CPU 回退的 scaled dot-product 改写为 DPU 支持的加法投影，并用固定张量维度替换动态 shape、meshgrid、arange。这样 backbone、neck 和分割头可以编译为单一 INT8 XMODEL 图；CPU 只负责固定内存中的置信度解码、NMS 和掩码重建，避免动态执行带来的延迟抖动。

#### 实验结果

GPU FP32 基线的 mAP/IoU 是 0.802/0.782。随机校准的 INT8 降至 0.749/0.731，mAP 相对下降 6.6%；只加偏差校正为 0.768/0.741；AVIS 加偏差校正达到 0.786/0.767，仅下降 2.0%，恢复了随机 INT8 相对 FP32 精度损失的 69.8%。低光图示也显示，随机校准会造成掩码碎裂和边界变形，而 AVIS 结果更接近 FP32。

GPU 虽然最快（121 ms），但功耗 11.8 W，超过微型车的预算；CPU 为 537 ms、8.3 W。CPU-DPU 混合方案为 309 ms、5.7 W、1.76 J/帧，相比 CPU-only 节能 60%。在 0.1 m/s、每 5 秒采集一帧的运行设定下，作者报告其具有超过 16 倍的避障时间余量。

这组对比也说明论文优化的是系统约束下的折中，而不是追求单一指标冠军：GPU 延迟最低却超出功耗预算，CPU 功耗尚可但响应慢，DPU 方案牺牲部分延迟换取更低能耗和更稳定的执行路径。所有延迟都包含设备端 DPU 推理和 CPU 后处理，没有把数据转移或后处理成本隐藏起来。

#### 局限性与意义

作者按内存占用和执行时间估计各功能块的发生度，再按故障可检测性和传播影响赋予 1–4 级严重度，以 (C=O\times S) 计算相对关键度。静态内存分配、选择性 TMR、EDAC 与内存清洗逐步将全局关键度从 0.3389 降到 0.2316，累计下降 31.7%。但这仍是软件层解析模型，不是绝对失效概率；论文没有进行真实质子/重离子辐照或系统性比特翻转注入。AVIS 也依赖校准数据能代表未来地形，面对分布变化的效果尚未验证。论文提及 Lunar rocks and craters dataset，但未给出完整代码和训练超参数。

#### 实际应用

这项工作适合功耗低于约 10 W、需要可预测时延且无法在轨重训的空间机器人。其可迁移价值不只在 YOLO，而在于一条工程路线：用激活统计做可复现校准，用算子改造消除加速器回退，再依据功能块风险把有限容错资源投向最危险的位置。换用新相机、新地形或更高速的车体时，仍需重新建立代表性校准集、重新编译并实测时延和辐射可靠性。这里的结果应理解为可部署性证据，而不是已经完成飞行认证。关键度模型忽略任务粒子通量、器件截面和闪存效应，只支持软件块之间的优先级比较；论文也没有真实质子或重离子辐照验证。AVIS 可能偏爱噪声或极端曝光样本，换传感器、光照、地形时必须重新检查数据覆盖。对工程团队，落地顺序应是先验证量化损失，再确认 DPU 图无 CPU 回退，最后进行运行时故障注入和功耗测试。

因此，本文最适合被看作空间机器人感知的工程蓝图：它把模型精度、编译器限制、能量预算和容错设计联系起来，指出每一项局部改动会如何影响整条推理链。后续若要用于真实任务，还需要物理辐照、更多地形和跨设备测试，并建立不确定性估计，让探测车能在分割置信度不足时主动降速或切换传感器。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02219v1)
- [arXiv](https://arxiv.org/abs/2609.02219v1)

---

<a id='2609.02077v1'></a>
## [KSG-Net: Key-Sparse and Global-Context Learning for Maritime 3D Ship Detection](https://arxiv.org/abs/2609.02077v1)

**Authors:** Zhouyuan Huai, Meiqi Wan, Yan Yang, Minshi Chen, Xin Yuan, Wei Wang, Xiao Wang

**Published:** 2026-09-02

**Categories:** cs.CV

**Abstract:**

Accurate 3D ship detection in maritime environments is critical for autonomous navigation, yet remains challenging due to large-scale vessel variations, sparse point clouds of small vessels, and severe sea-clutter interference. Existing methods, primarily based on 2D features or dense representations, struggle to balance detection accuracy and computational efficiency, while sparse 3D detectors designed for road scenes generalize poorly to maritime scenarios. This paper focuses on two key challenges in maritime LiDAR perception: weak feature representation for small and sparse vessels, and insufficient global structural modeling for large vessels due to the limited receptive field of local sparse convolutions. To address these issues, we propose KSG-Net, a Key-Sparse and Global-Context learning network for maritime 3D ship detection. The core idea is to jointly enhance local discriminative features and global structural awareness within a unified fully sparse detection framework. Specifically, a Key Sparse Multi-scale Aggregation (KSMA) module is designed to enhance the representation of small and sparse vessels by selecting informative key voxels and aggregating cross-scale neighborhood features. Furthermore, a Global Context Aggregation (GCA) module is introduced to capture long-range geometric dependencies through scene-level context modeling with gated residual interactions, thereby improving the representation of large vessels. Extensive experiments on the Thames River vessel dataset and simulated datasets demonstrate that KSG-Net consistently outperforms existing methods in multi-scale vessel detection and exhibits strong robustness in complex maritime environments.

### 论文解读
#### 研究问题与动机

海上自主航行需要可靠的三维船舶检测，但海面点云与城市道路点云很不一样：船只尺度差异大，小船往往只有少量、不连续的回波，海面杂波又会提供大量干扰。PointPillars、CenterPoint等密集BEV检测器需要在大范围空区域上计算，容易产生量化误差和冗余；VoxelNeXt等稀疏检测器虽然高效，却可能错过小船，并且局部稀疏卷积难以建立大船两端之间的长距离几何关系。

KSG-Net的出发点是把稀缺的计算和特征表达集中到真正关键的位置，同时补充场景级信息。作者将问题拆成两个互补目标：用局部多尺度聚合“看清”稀疏小目标，用全局上下文“稳住”大目标的整体结构。

#### 核心方法

KSG-Net采用全稀疏流水线。原始点云先经过Pillar VFE编码，再由稀疏3D骨干提取多尺度非空体素特征。关键稀疏多尺度聚合模块（KSMA）首先用轻量评分头为每个体素预测显著性，并按Top-K选出锚点；只有这些位置进入后续增强，从而避免对所有稀疏位置执行昂贵操作。对每个锚点，模块在不同空间尺度上搜索K近邻，聚合邻居特征，再用自适应权重融合多尺度结果，最后通过显著性门控的残差连接写回原特征。这样既能利用近邻补齐小船的断裂回波，也不会显著扩大计算量。

全局上下文聚合模块（GCA）处理另一类问题。它对每个样本的非空体素做全局池化，形成场景描述，再广播到各体素；一个由局部特征和全局描述共同决定的门控向量，为不同位置调节全局信息注入强度。局部几何因此得到保留，大船的远距离部分又能共享一致的场景线索。增强后的稀疏特征交给检测头，预测类别、中心位置、尺寸、朝向和IoU。

两个模块的协作逻辑并不是把特征无差别地变密集。KSMA先利用显著性分数控制“在哪里花算力”，并通过残差形式避免破坏骨干网络已有的几何表征；GCA再控制“哪些位置需要多少全局信息”。因此，前者主要针对回波少、局部不完整的小船，后者主要针对范围大、局部卷积难以覆盖的船体，最终仍保持稀疏坐标结构，适合大范围海面部署。

#### 实验结果

作者在真实港口Ship LiDAR数据集及泰晤士河真实、仿真数据集上验证方法，采用IoU=0.5的KITTI风格3D AP/mAP。训练配置为OpenPCDet/PyTorch、Adam优化器、学习率1×10^-3、OneCycle调度、80个epoch、batch size 4，硬件为NVIDIA RTX 4060。Ship LiDAR上，KSG-Net达到85.65% mAP，超过Fade3D的79.25%（提升6.40个百分点），同时达到33.11 Hz；作为对照，VoxelNeXt为74.50% mAP和23.55 Hz，DSVT为77.44%和14.97 Hz。

类别结果也体现了设计目标：游船AP为95.15%，快艇AP为82.79%。在泰晤士河真实数据上mAP为87.89%，仿真数据上为84.44%。消融实验中，基线mAP为78.74%，加入KSMA后升至83.72%，加入GCA后为80.92%，两者同时使用达到87.89%；小船AP则由57.86%提高到75.23%。这些结果说明局部稀疏增强和全局结构建模具有互补性，而非简单叠加同一种特征传播。

从效率角度看，KSG-Net并非所有指标都领先：Fade3D报告的速度为51.53 Hz，高于KSG-Net的33.11 Hz；但KSG-Net在精度上提高明显，说明作者选择了更偏向检测质量的折中。与DSVT的14.97 Hz相比，KSG-Net仍有较大实时性优势。实验还覆盖货船、游船、工程船和快艇等不同类型，结果并非只依赖单一船型。

#### 局限性与意义

KSG-Net的价值在于针对海事点云的非均匀稀疏性重新分配建模重点：KSMA以选择性计算改善小目标表达，GCA以轻量全局广播弥补局部感受野。优选设置包括[0.2, 0.2, 0.4]体素大小、15%的Top-K比例和8个近邻；这为在类似港口、河道或近岸机器人平台上复现提供了明确起点。

论文仍缺少极端天气、巨浪遮挡和严重点云缺失条件下的系统评估，也未在正文提供代码仓库链接。全局描述在目标拥挤或场景组成复杂时可能混合多个船只的统计信息，关键体素筛选也可能漏掉低显著性目标。未来可研究天气自适应、实例级全局建模与跨传感器迁移。总体而言，该方法展示了“关键位置做精细局部建模、所有位置共享适量全局约束”的海上3D检测思路，在保持33.11 Hz实时速度的同时，取得85.65% mAP，具有较好的工程启发性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02077v1)
- [arXiv](https://arxiv.org/abs/2609.02077v1)

---

<a id='2609.02002v1'></a>
## [InsightSeg: Reusing Correction Insights for Guideline-Consistent Segmentation](https://arxiv.org/abs/2609.02002v1)

**Authors:** Vanshika Vats, Ashwani Rathee, James Davis

**Published:** 2026-09-02

**Categories:** cs.CV, cs.AI

**Abstract:**

Guideline-consistent semantic segmentation requires more than category recognition, as real-world labeling policies demand fine-grained, task-specific decisions. Recent multi-agent refinement systems improve compliance with such textual guidelines by detecting and correcting errors. However, they are stateless: feedback from the critiquing agent is discarded, causing the same guideline-specific mistakes to be repeatedly rediscovered and corrected across the dataset at the cost of additional refinement. We introduce InsightSeg, an episodic memory mechanism that converts successful correction episodes into reusable, visually grounded insights. A meta-analyzer distills each qualifying episode into directive natural-language insights and anchors them to the local image regions that caused the error using patch-level visual concept vectors. On subsequent images, these concepts are matched against dense patch embeddings to retrieve relevant insights, which condition the segmenting agent before making its first prediction. This shifts the system from correcting recurring errors to preventing them, improving segmentation quality before any refinement occurs. Across Waymo and Cityscapes, InsightSeg improves both first-pass and final guideline-consistent segmentation performance while requiring fewer refinement steps, demonstrating that multi-agent refinement can become more accurate and efficient by drawing on past correction experience.

### 论文解读

#### 研究问题与动机
自动驾驶数据中的语义分割，难点并不止于判断“这是什么类别”，还包括是否符合一套具体标注准则。例如，推着自行车的人不能简单地当作行人，广告牌中的人影不应被当成真实行人，而背包有时又应纳入行人区域。现有多智能体分割系统通常由Worker生成掩码、Supervisor检查错误并反复细化，虽然能够纠正当前图片，却不会保留这次纠错经验。于是相同的准则错误会在后续图片中再次出现，既浪费模型调用，也使系统表现不稳定。

InsightSeg的出发点是：数据集中反复出现的准则冲突，往往也具有重复的局部视觉线索。如果能把一次成功的纠错概括成规则，并绑定到引发错误的局部区域，系统就可以在下一张相似图片首次预测之前主动提醒分割模型，把“发现错误再修复”变成“尽量第一次就做对”。

#### 核心方法
InsightSeg由Worker、Supervisor和Meta-Analyzer协作。Worker先生成分割结果，Supervisor评估漏检、误检和边界问题；当一次修正确实减少了问题且没有明显倒退时，Meta-Analyzer用视觉语言模型把完整纠错过程提炼成一条可复用的自然语言见解，例如“海报中的人物不是需要分割的真实行人”。论文用 (I_t=I_{miss}+I_{false}+0.1I_{ref}) 衡量问题，只有问题总数至少减少1、单步退化不超过0.5时才写入记忆，避免把偶然或低质量反馈保存下来。

每条见解还要有视觉依据。系统使用DINOv3 ViT-H+/16提取局部patch特征，在造成错误的support region内平均池化出concept vector，并把它作为文字规则的视觉锚点。文本相似的见解会合并，每条见解最多保留5个不同场景的锚点。处理新图像时，系统将密集patch embedding与记忆中的概念向量计算相似度：每个锚点取最相似的图像patch，再以其中最高分作为见解分数。分数超过0.6的前3条见解被写入Worker提示词，Worker据此完成首轮预测，再按需接受Supervisor细化，最多4轮。

这种设计的关键是“局部匹配”而非整图匹配。全局CLS特征容易因为街道、光照或相机视角相似而检索到泛化但无关的经验；patch级检索则会寻找伞、骑行姿态、背包或海报人物等真正导致规则冲突的区域。记忆保存的是可读的自然语言和视觉锚点，不改动基础模型参数，因此可审计，也能由人工检查或更新。

#### 实验结果
作者在102张Waymo guideline-consistent样本及Cityscapes验证集500张图像上测试，并与LISA、GroundedSAM、SegZero、READ和GuideSeg等系统比较。Waymo上InsightSeg最终gIoU达到83.51、mDice达到90.21；作为对照，先前最强的GuideSeg gIoU为80.57。更重要的是，平均细化次数从无记忆基线的2.66次降到1.01次，约减少62%，而不进行后续细化时，首轮gIoU也提升5.07，说明记忆确实改善了初始预测。

在Cityscapes上，细化后mDice由60.32升至62.46，gIoU达到52.46，mPr由69.83升至78.00，平均迭代次数由2.33降到1.28，减少约45%。消融实验显示，patch检索的Waymo gIoU为83.51，明显高于全局CLS检索的79.58。将样本顺序随机打乱后，gIoU仍有83.24（原顺序83.51），表明收益不是依赖某个特定排列。换用更强的Gemini-3 Flash Preview时，无记忆gIoU为75.8，加入记忆后升至84.1，说明机制不绑定某一个VLM。

系统也降低了实际调用成本：Waymo上无记忆基线每样本约需8.0次API调用、成本0.0088美元；InsightSeg约3.3次调用、成本0.0036美元，成本下降约2.4倍。记忆库本身很小，Waymo为869 KB、Cityscapes为1.7 MB。

#### 局限性与意义
InsightSeg并非没有边界。刚开始运行时记忆为空，存在冷启动阶段；远处或极小目标提供的视觉特征不足，可能无法匹配已有锚点；海报或广告上的人物又可能在局部外观上误触发“行人”见解。Cityscapes中的大量远距离小行人也使绝对召回率受限。即使加入最不相关的记忆，系统性能基本保持稳定，但这并不能完全消除语义混淆风险。

论文的价值在于展示了一种无需重新训练模型的在线学习路径：把多智能体纠错转化为可检索、可解释、可审计的规则资产。它适合标注政策复杂、错误模式会重复出现的自动驾驶和机器人视觉场景。后续若加入真实物体与二维图像的语义验证，并能自动合并、淘汰随政策变化而过时的见解，这种“视觉锚定记忆”有望成为通用分割代理持续改进的重要组件。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02002v1)
- [arXiv](https://arxiv.org/abs/2609.02002v1)

---

<a id='2609.02493v1'></a>
## [MS-MEM: Multi-Skill Manipulation-Enhanced Mapping via Uncertainty- and Disturbance-Aware Action Selection](https://arxiv.org/abs/2609.02493v1)

**Authors:** Yitian Shi, Jesper Mücke, Nils Dengler, Sicong Pan, Rania Rayyes, Maren Bennewitz

**Published:** 2026-09-02

**Categories:** cs.RO

**Abstract:**

Accurate scene understanding in confined, cluttered spaces such as shelves is essential for service robots, as many everyday tasks require them to locate and retrieve objects reliably. Yet, it remains challenging due to severe occlusions, restricted accessibility, and the need to avoid excessive scene changes. In this paper, we propose Multi-Skill Manipulation-Enhanced Mapping (MS-MEM), an evidential framework for uncertainty-aware mapping that integrates active viewpoint selection, object pushing, and grasping. MS-MEM combines scene-level metric-semantic evidential belief estimators with an uncertainty-aware grasp representation. This representation is learned using a novel full-evidential grasp estimator that models both grasp affordance and orientation uncertainty. In our framework, candidate perception and manipulation actions are evaluated within a unified action selection pipeline using a common information gain criterion. For manipulation actions, we further introduce a collateral disturbance constraint (CDC) that discourages excessive changes to confident regions of the scene belief. This enables MS-MEM to select actions that effectively reduce map uncertainty while limiting collateral scene changes. Experimental results show that, compared with single-skill and unconstrained baselines that ignore scene disturbance, MS-MEM achieves higher mapping accuracy while substantially reducing scene disturbance, highlighting the synergistic effects of active viewpoint selection, push, and grasp actions.

### 论文解读

#### 研究问题与动机
在货架、柜格等狭窄且拥挤的环境中，机器人即使拥有相机，也常常只能看到最外层物体，遮挡会同时损害占据建图、语义识别和后续操作。单纯移动视点无法保证隐藏区域暴露；已有的操作增强建图主要依靠推物，虽然能打开视线，却可能把许多物体推离原位，改变已经建立的地图。论文提出 MS-MEM，核心观点是把推物和抓取视为互补工具：推物负责分离相互遮挡的物体、创造可达空间，抓取则在合适位置选择性移除遮挡物，再配合主动视觉完成低扰动建图。

系统以共享的证据场景信念为中心，使每次操作都同时考虑“还能获得多少信息”和“会改变多少已经确认的场景”。占据体素用 Beta 分布表达，语义类别用 Dirichlet 分布表达，因此地图不仅有估计值，还有显式置信度。这一点很重要：机器人不必把所有地图变化都视为同等风险，而可以重点保护那些原本高置信的区域。

#### 核心方法
每轮决策首先生成三种候选动作。主动视觉模块产生下一最佳视点；UPS 产生推物候选；UGS 使用 FE-vMF-Contact 产生六自由度抓取假设。FE-vMF 将抓取可供性表示为 Beta 分布，并用两个 von Mises–Fisher 分布描述基准方向与接近方向，从概率上表达“能否抓住”和“应该从哪个方向接近”。其 Point Transformer v3 主干提取点云特征，再由 MLP 输出相关参数。抓取候选还通过 FE-UMGF 在多个视点间时序融合，使同一物体的方向和可供性估计不只依赖单帧证据。

对于每个视点、推物或抓取候选，CNABU 网络预测动作执行后地图信念如何变化，MSAS 再用统一的 DOIG 进行选择。DOIG 先计算遮挡感知信息增益：动作之后仍能通过下一视点获得的观测收益，加上操作动作带来的语义熵变化收益；然后定义扰动集合，统计那些原本高置信、动作后语义类别却发生变化的体素，并以 ζ_CDC 乘其数量进行扣分。抓取目标本身不计入该惩罚，因为它的移除是计划内结果。系统执行最高分动作、重新观测并更新地图；当语义置信度超过 τ_conf 后停止操作，剩余过程只用主动视觉收尾。

实验中的视点候选数为 300，最多执行 40 步。FE-vMF 在 4,000 个模拟场景上训练，每个物体生成 10^5 个反向抓取候选；抓取 CNABU 使用 7,000 个模拟抓取实例。优化器为 AdamW，学习率为 1e-5。这样的设计将抓取方向不确定性、遮挡收益和场景保护放到同一个闭环，而不是先固定技能顺序、再被动接受其副作用。

#### 实验结果
在 PyBullet 货架环境中，系统使用 UR5、Robotiq 2F-85 夹爪和腕载 L515 相机，在物体占据率 30%–45% 的 25 个困难场景上评估。40 步后，完整 MS-MEM 的占据 mIoU 为 0.899，语义 mIoU 为 0.767，平均位置变化为 0.707 m。Grasp Only 的对应结果是 0.880、0.681 和 0.228 m：它较少扰动，但受限于无法主动创造抓取空间。Push Only 的结果为 0.887、0.756 和 1.571 m，说明推物确实有助于暴露内容，却带来最大的场景改变。

去掉 CDC 后，占据和语义 mIoU 分别达到 0.905 和 0.791，但位置变化上升到 1.231 m；完整系统以少量地图指标差异换取约 42.6% 的位移降低。真实货架实验包含 5 个场景、共 69 个物体，完整系统正确找到 44 个物体，高于 Push Only 的 40 个和 Grasp Only 的 38 个。实验还显示，FE-vMF 输出的方向精度与实际角误差相符，表明不确定性并非只用于展示，而确实能帮助动作筛选。

#### 局限性与意义
MS-MEM 的主要意义是提供了一个可迁移的决策抽象：机器人可以把不同技能的候选动作都转换成预期信念更新，再用信息收益减去无意扰动的代价来选择。推物创造抓取机会、抓取移除关键遮挡物，二者因此形成比单技能策略更灵活的闭环。对于仓储整理、服务机器人货架盘点和受限空间感知，这种“主动改变环境但保护已知区域”的原则尤其有价值。

不过，系统效果依赖 ζ_CDC、置信度阈值 τ_χ 与 τ_conf 的校准。惩罚太弱会接近高扰动的推物策略，太强则可能因过度保守而留下遮挡。纯抓取仍无法解决不可达目标，方法也依赖 CNABU 对动作后信念的可靠预测。训练数据主要来自仿真，真实验证只有 5 个场景，并借助 Vicon 获取物体真值，因此对不同物体、传感器噪声和布局的泛化尚未充分证明。后续工作应扩大真实数据、研究在线阈值自适应，并检验更多操作技能接入统一 DOIG 框架后的稳定性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02493v1)
- [arXiv](https://arxiv.org/abs/2609.02493v1)

---

<a id='2609.02402v1'></a>
## [A Physics-Consistent Benchmark for Contact-Rich Human-Robot Interaction in Assistive Care](https://arxiv.org/abs/2609.02402v1)

**Authors:** Chengxiao He, Shanghai Yuan, Liuqun Fan, Shenzhen Zhu

**Published:** 2026-09-02

**Categories:** cs.RO

**Abstract:**

Conventional task-level evaluation asks whether a robot policy completes a specified action, but can miss failures that emerge only during physical human contact. This limitation is critical in contact-rich assistive tasks, where meaningful evaluation requires a physically responsive human, interaction-quality assessment beyond task success, and a leak-free observer-scorer protocol. We introduce a physics-consistent benchmark for contact-rich human-robot interaction, instantiated in robot-assisted bathing. The benchmark combines a deformable, passively responding human, physics-aware scores alongside task-level success, and a frozen vision-only / scorer-only evaluation protocol. To establish physical validity, region-wise simulated responses are calibrated against force-indentation measurements from Franka impedance pushes on a medical-care manikin. Under a frozen T1-T7 protocol with 140 runs per method, an LLM-augmented state machine (State Machine) achieves 72.9% task success but drops to 56.4% after correct-region and force-safety screening; VoxPoser produces lighter and more stable contact but completes only 27.9% of trials; and zero-shot pi0.5 achieves 0.7% task success with no correct-region or safety-gated successes. These results show that task completion alone does not imply physically valid contact and motivate physics-aware screening before deployment of contact-rich assistive robot policies.

### 论文解读

#### 研究问题与动机
辅助护理机器人并不是把末端执行器送到目标坐标就算成功。以洗澡、擦拭和推压为例，机器人必须面对会被压入的软组织、会被带动的被动关节、接触力上限以及舒适姿态约束。传统基准往往把人体简化为刚体，或用动捕轨迹驱动人体，因此无法回答“机器人是否以合适的力完成了接触”。此外，若策略能直接读取真关节角或仿真锚点状态，测试结果也难以迁移到现实。

本文提出一个物理一致性基准，核心是同时评估任务进度、接触安全和接触有效性。作者的基本判断是，名义上的任务完成率可能掩盖过大压力，也可能把“几乎没有接触”的安全运动误报为成功。

这种设计还把 sim-to-real 的观测边界写进了基准本身：现实机器人可以从相机和感知算法获得视觉线索，却不能直接读取仿真器中的精确接触锚点。因而，一个方法若依赖隐藏状态才能表现良好，就会在协议层面暴露出来，而不是等到真实护理时才发现问题。

#### 核心方法
基准接收自然语言指令和 RGB-D/点云观测，由视觉感知把目标定位到前臂、上臂或腹部等身体区域，再交给规划器生成触碰、擦拭或推压动作。环境中的人体由刚性骨架和可变形软组织外壳组成：接触残差通过刚度与阻尼模型转化为接触力，力再经接触雅可比传递到骨架，产生软组织形变和被动关节运动。

作者用 Franka Panda 对医疗护理模特进行阻抗推压，采集力—压入深度关系来校准仿真参数。评分包括峰值力 C1、关节极限余量 C2、任务成功率 C3、运动活跃度 C4 和力稳定性 C5，并设置安全门控。推理时只向策略开放 RGB-D、点云、零样本语义分割掩码和骨架关键点，不提供内部锚点或真状态。评测冻结随机种子、手臂舒适姿态 [45°,60°]，每种方法统一运行 140 次。

五项指标分别对应不同的失败模式：峰值力揭示瞬时危险，关节余量反映是否逼近不舒适或不可行姿态，成功率描述目标动作是否完成，运动活跃度衡量是否持续运动，力稳定性则区分平滑接触和剧烈波动。这样，基准不再把所有失败压缩成一个二元标签，也能解释为什么某个策略需要改进。

#### 实验结果
实验覆盖七类任务：前臂、上臂和腹部触碰与擦拭，以及前臂推压。LLM 增强状态机的名义成功率为 72.9%，但加入峰值力安全门控后只有 56.4%，其峰值力中位数达到 19.48 N，说明“做到了”不等于“做得安全”。

VoxPoser 的成功率为 27.9%，中位峰值力为 1.46 N，力稳定性为 0.21 N；它的接触非常轻且稳定，却常因避障而无法产生足够的压入或擦拭压力。其运动活跃度达到 0.883，但活跃动作没有带来相称的任务进展。预训练 VLA 模型 π_0.5 的成功率仅 0.7%；在 140 次测试中，106 次没有接触目标或接触到了错误区域。三组结果共同说明，必须把安全和有效接触与成功率一起报告。

安全门控敏感性实验进一步显示，状态机的结果会随安全系数变化而显著波动，而 VoxPoser 因为始终保持极轻接触，表现相对稳定。这不是简单地说明某一种规划器胜出，而是揭示了任务目标之间的张力：避碰策略可能安全却无效，动作原语可能有效却危险，端到端策略则可能连目标区域都无法可靠接触。

#### 局限性与意义
该基准仍以医疗护理模特替代真人组织，每项任务仅运行 20 次，样本量更适合刻画行为特征而非进行大规模统计推断。模型没有覆盖卸载后的组织恢复和能量耗散，上臂还复用了前臂的校准参数，因此材料差异可能造成偏差。仿真后端也不是本文的主要创新，价值在于实测校准、观测隔离和统一评分协议的组合。

尽管如此，这项工作为接触丰富的人机交互提供了可复现的安全筛选层：研究者可以定位策略是压力过大、接触不足、目标错误还是关节余量不足。未来将材料参数、真人安全标准和恢复动力学补齐后，协议有望迁移到穿衣、转移和清洁等护理任务，帮助在现实部署前淘汰物理上不可靠的策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.02402v1)
- [arXiv](https://arxiv.org/abs/2609.02402v1)

---

