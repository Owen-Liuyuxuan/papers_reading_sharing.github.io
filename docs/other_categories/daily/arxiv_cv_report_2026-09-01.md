time: 20260901

# Arxiv Computer Vision Papers - 2026-09-01

## Table of Contents

1. [$\mathcal{N}_0$-Foundation: Towards the Age of Tactile Intelligence](#2608.29601v1)
2. [MotionSync: Non-Causal Refinement of Causal Tracker for Label-Efficient 3D Perception](#2608.29567v1)
3. [Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception](#2608.29487v1)
4. [Generalizable Multi-Agent Planning from Signal Temporal Logic Specifications via Diffusion](#2608.29490v1)
5. [ARMOR: Manifold-Oriented Training for Adversarially Robust Aerial Object Detection under Data Scarcity](#2608.29510v1)
6. [AGM: Achievement-Grounded Memory for Closed-Loop Agents with Frozen VLA Policies](#2608.29537v1)
7. [Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models](#2608.29590v1)
8. [Module Number Adaptive Visual Shape Control for Serial Modular Soft Robots](#2608.29547v1)

---

## Papers

<a id='2608.29601v1'></a>
## [$\mathcal{N}_0$-Foundation: Towards the Age of Tactile Intelligence](https://arxiv.org/abs/2608.29601v1)

**Authors:**  NeoteAI Team,  Fudan TEAI Team

**Published:** 2026-08-30

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

We present $\mathcal{N}_0$-Foundation, a paradigm for tactile-enabled embodied manipulation, which integrates tactile sensing hardware, large-scale multimodal data, tactile representation learning, and standardized evaluation. First, we engineer the infrastructure for scalable data collection, including a vision-based tactile sensor, a tactile Universal Manipulation Interface (UMI), and a synchronized visuo-tactile data collection system supporting both robot embodiments and UMI-based demonstrations. Leveraging this infrastructure, we construct NeoData, which contains more than 30000 hours of synchronized visual and tactile demonstrations, spanning six embodiments, 450 tasks, and billions of paired RGB and tactile frames collected through a mixture of real-robot teleoperation and UMI-based demonstrations. To facilitate open research, we further release OpenNeoData, a 5000-hour open-source subset of NeoData. The dataset addresses a central limitation of existing manipulation corpora, critical for deformable-object manipulation, precise assembly, delicate force control, and sustained surface interaction. Capitalizing on the large-scale, heterogeneous tactile measurements, we propose NeoForce, a visuo-tactile representation model that learn transferable tactile representations across different sensor designs. To enable systematic evaluation of tactile embodied models built upon our infrastructure, datasets and tactile representations, we further propose a comprehensive benchmark, which combines the real-world NeoReal suite and the simulated NeoSim suite for standardized evaluation. Experiments across both suites show that policies benefit from the physical contact state rather than from the device-specific appearance of the tactile signal. We release the dataset, the representation, and the benchmark, aiming at supporting future work on tactile-enabled embodied manipulation.

### 论文解读

触觉是机器人处理柔性物体、精密装配和持续表面接触时缺失的重要信息：相机看得到物体，却难以直接判断压力、摩擦或即将发生的滑动。《N₀-Foundation》把触觉采集、数据、表示模型和评测整合成一套触觉具身操纵基础设施，目标是让策略学习“接触状态”，而不是记住某种传感器的外观。

#### 数据与核心方法

作者设计了视触觉传感器和手持式 N₀-TacUMI。后者集成双触觉传感器、160°腕部鱼眼相机、红外六自由度亚毫米级追踪器和磁编码夹爪测量器，因此人体可以直接演示操作，采集过程不必绑定某一台机械臂。由此构建的 NeoData 包含超过 30,000 小时同步视觉—触觉演示、140 万片段和 33 亿时间步，覆盖六种具身形态、450 余项任务、200 余种技能以及数十亿 RGB 与触觉帧；同时开放 5,000 小时的 OpenNeoData。数据经过完整性、动作稳定性、时长异常、画质对齐和逆运动学可转移性检查，并按任务、子任务、动作和原子片段进行层级标注。

NeoForce 不直接把设备相关的触觉图像当作通用特征，而是将信号统一为每个空间位置含切向力和法向压力的三维力场。模型采用 DINOv2 初始化的 ViT-B 共享骨干，同时输入连续四帧 RGB 与力场块；一项 Huber 损失重构力场，另一项掩码潜空间预测损失学习视觉与触觉的时空结构。消融结果显示，加入潜空间预测后，力场重构 MAE 从 0.070 降至 0.066，RMSE 从 0.095 降至 0.089，真实压力数据上的 R² 达到 0.950。

#### 实验结果与意义

真机 NeoReal 包含纸箱折叠、箱包打包、线缆缠绕、白板擦拭、杯子堆叠和插拔等十项接触密集任务。最佳基础策略 π₀.₅ 的平均成功率为 26.5%；采用 NeoForce 后提高到 32.5%，Progressive Score 从 38.1 提升至 47.5，说明触觉在盲操作、稳固抓握和寻找插接位置时尤其有用。仿真 NeoSim 覆盖十二项单臂和双臂任务，π₀.₅ 平均成功率为 45.8%；双臂交接等持续接触任务仍困难，最高成功率只有 25%。

#### 局限与适用场景

论文的优势在于把异构传感器信号转成物理量，并以大规模同步数据和真机/仿真基准支撑迁移研究；适合精密装配、柔性物体操纵、力控和表面交互。局限包括双臂协调仍弱、机器人关节极限和超时造成大量失败，以及手持演示到目标机器人之间的动力学差异。总体而言，这项工作提供了从采集到评测的可复用路线：先获得可靠的视觉—力场同步数据，再训练硬件无关的接触表示，最后在具体机器人策略中验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29601v1)
- [arXiv](https://arxiv.org/abs/2608.29601v1)

---

<a id='2608.29567v1'></a>
## [MotionSync: Non-Causal Refinement of Causal Tracker for Label-Efficient 3D Perception](https://arxiv.org/abs/2608.29567v1)

**Authors:** Rahul Ahuja, Bala Murali Manoghar Sai Sudhakar, Shashwata Gupta, Venkatraman Narayanan, Varun Ravi Kumar, Senthil Yogamani

**Published:** 2026-08-30

**Categories:** cs.CV

**Abstract:**

Three-dimensional box-and-track annotation is the cost bottleneck in autonomous-driving data engines, and the offline systems built to relieve it replace the online perception stack outright, so a team needing both regimes maintains and reconciles two. MotionSync makes the causal/non-causal boundary an explicit architectural seam instead. A strictly causal tracker, built on a strong published baseline and extended with innovation-driven uncertainty calibration, frame-rate-invariant kinematic association gates, and multi-hypothesis motion with learned mode selection, emits a valid online result. A non-causal pass then revises the buffered trajectories with Rauch--Tung--Striebel smoothing applied separately to pose, extent and yaw, physics-validated gap completion, and semantic pruning of ghost tracks against LiDAR point labels. The refiner never writes back, so one system serves both regimes and refinement's effect is a delta over an unaltered causal estimate. Used as an auto-labeller, a fixed 3D detector trained on 25% human labels plus MotionSync pseudo-labels reaches 96.9% of its full-supervision mean average precision (mAP) on Waymo, and at a 10% budget the non-causal pass accounts for +3.3 mAP/L2 over pseudo-labels from the same tracker's causal stage. Re-fitting the online tracker on its own refined output recovers 73% of the benefit of human supervision, while its causal output is worse supervision than no re-fitting at all. As a tracker MotionSync is at parity with the leading published offline entries on the headline metric and ahead of them on error composition, which is where a refinement pass can act at all: it reduces misses and fragmentations together, the signature of gap completion rather than of a tuned detector.

### 论文解读

#### 研究问题
自动驾驶 3D 感知需要大量人工框标注，而遮挡、漏检和朝向噪声又会让自动生成的轨迹断裂。在线跟踪只能利用过去信息，离线自动标注虽然能利用完整驾驶日志，却常常与车端在线系统完全不同。论文提出 MotionSync，目标是在不改变在线因果栈的前提下提升伪标签质量。

#### 核心方法
MotionSync 把系统分成两个阶段。在线阶段使用改进的 MCTrack，只根据当前和历史观测生成轨迹，并通过物理单位定义关联门，使不同采样频率下的阈值仍然具有一致含义；它还动态校准观测不确定性，利用匀速、匀加速、恒转率和静止四种运动假设进行交互多模型预测，并用运动方向修正目标朝向。轨迹随后写入缓冲区，但离线结果不会反馈到在线阶段。

完整日志可用后，非因果阶段使用 RTS 平滑整合未来证据。位置、尺寸和朝向分别处理：尺寸利用目标外形近似恒定性，朝向先消除角度绕回，再进行平滑。对遮挡造成的缺口，系统按缺口长度选择线性插值、样条或运动模型外推，并检查瞬时加速度；不符合物理约束的填充会被舍弃。最后，系统借助 3D 语义分割检查轨迹与点云语义是否一致，从而删除长期被植被等类别支持的错误目标轨迹。

#### 关键证据
在 Waymo 上，使用 25% 人工标注加 MotionSync 伪标签训练检测器，mAP/L2 达到 65.3，而全监督为 67.4，即达到其 96.9%。在 10% 标注预算下，纯人工、只有因果伪标签和加入非因果细化的结果分别为 53.6、59.1 和 62.4，说明完整日志带来的修正贡献了 3.3 mAP。跟踪实验中，KITTI 的 HOTA 达到 82.94，轨迹碎片数从 438 降至 61；Waymo 的漏检率下降 0.69、MOTA 提升 0.91。进一步闭环训练在线跟踪器可以恢复人工标注增益的 73%，而未经细化的因果轨迹可能把错误带入训练。

#### 局限与实际意义
该方案需要等待完整日志，因此适合自动标注、数据引擎和离线评估，不适合直接承担实时控制。语义过滤效果依赖分割模型质量，高动态转弯下的协方差近似也可能不够精确。总体而言，论文的价值在于提供了清晰的接口：任意可部署的因果跟踪器都可以先产出稳定轨迹，再接入利用未来信息的后处理，从而减少人工标注，同时避免维护两套互不一致的跟踪系统。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29567v1)
- [arXiv](https://arxiv.org/abs/2608.29567v1)

---

<a id='2608.29487v1'></a>
## [Blind Dexterity: Whole-Body Humanoid Manipulation via Pure Proprioception](https://arxiv.org/abs/2608.29487v1)

**Authors:** Aditya Bhatt, Oleg Kaidanov, Puze Liu, Jan Peters

**Published:** 2026-08-30

**Categories:** cs.RO

**Abstract:**

We present blind, whole-body manipulation skills on a Unitree G1 humanoid using only onboard proprioception, without cameras, markers, force-torque, or tactile sensors. Despite this minimal sensing, the trained policies exhibit surprising capability across qualitatively different tasks: push-resilient bipedal walking without IMU feedback, active soccer ball trapping with a foot, seeking and lifting a suitcase by its handle, and mounting a randomly positioned skateboard.   We argue that these capabilities arise from a key underappreciated signal: the way the joint encoder readouts evolve under purposeful compliant contact, effectively forming a whole-body tactile channel. By generating contact-rich motions, the trained policies actively probe the environment; as a result, task-relevant object state (e.g., pose) becomes increasingly decodable from short proprioceptive histories. We expose this information using compact task-specific state estimators trained alongside, but fully separately from, the policies; their prediction errors decrease rapidly after informative contact.   Our results indicate that joint encoder-based proprioception, combined with compliant actuation (now widely available on commercial robots and low-cost motors) is already a strong, practical substrate for whole-body dexterous manipulation and interactive perception, and therefore a natural foundation on which richer sensing can be layered.

### 论文解读

人形机器人做操作时通常依赖摄像头、力矩传感器或触觉皮肤，但这些传感器会受到遮挡、光照、安装位置和成本限制。本文研究一个更极端的问题：Unitree G1 能否只凭关节编码器、关节速度和 IMU 等机载本体感觉，主动感知物体并完成全身操作？作者的关键观察是，柔顺 PD 控制中期望关节角与实际关节角的偏差会因外部接触而改变，因此关节反馈可以充当一种低分辨率、覆盖全身的“隐式触觉”。

方法采用仿真训练、真实机器人执行的强化学习框架。策略输入当前本体感觉以及过去的动作和观测；交互任务使用约 0.1 秒的历史窗口，策略输出关节目标，再通过 \(q^{des}=q^{default}+0.25a\) 控制机器人。训练时 Critic 可以看到模拟器中的物体真值，而 Actor 始终只能看到本体感觉。一个独立的多层感知机状态估计器还从短时历史预测物体位置或姿态，并用均方误差监督学习。对于提箱任务，策略同时调节关节刚度，使机器人能够以较柔软的方式搜索接触。重要的是，机器人并非被动等待信号，而是学会扫动、摆动、轻敲等动作，让接触主动产生可辨识的反馈。

作者验证了无 IMU 行走、随机位置足球捕获、随机姿态滑板登乘和平衡，以及从随机高度桌面寻找箱柄并提起四类任务。行走实验中，同时移除 IMU 和动作历史会使生存率从 96.1% 降至 89.1%，线速度误差从 0.3370 增至 0.4408 m/s，说明历史动作有助于解释编码器残差。足球捕获的 Blind-SE 策略成功率达到 92.9±1.2%，平均定位误差 9.51±1.11 cm，显著超过蒸馏盲策略的 31.2±33.3% 成功率。滑板任务成功率约 88–90%；提箱中加入可变刚度后，成功率从 83.5% 提升到 85.1%，倾倒率从 22.5% 降至 17.9%。这些结果表明，主动接触策略比直接模仿拥有物体真值的老师更适合在信息不完整时工作。

局限也很明确：关节残差难以精确定位接触点，不同接触状态可能产生相似信号，短历史无法保存很久以前的证据；此外，仿真中的 PD 动力学和摩擦若不准确，迁移可靠性会下降。实验中的手部还被简化，复杂手指操作尚待验证。总体而言，这项工作为无法部署视觉或力传感器的移动操作平台提供了实用方向：通过动力学域随机化、短历史状态估计和可变刚度，让机器人用自身运动“制造”感知信息。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29487v1)
- [arXiv](https://arxiv.org/abs/2608.29487v1)

---

<a id='2608.29490v1'></a>
## [Generalizable Multi-Agent Planning from Signal Temporal Logic Specifications via Diffusion](https://arxiv.org/abs/2608.29490v1)

**Authors:** Joe Eappen, Zikang Xiong, Shreyash S. Iyengar, Suresh Jagannathan

**Published:** 2026-08-30

**Categories:** cs.MA, cs.AI, cs.RO

**Abstract:**

Multi-agent systems in the real-world (e.g., drone swarms, autonomous cars, warehouse robots) must satisfy rich, temporal tasks while avoiding collisions. Signal Temporal Logic (STL) elegantly encodes such objectives, but current STL planning methods face critical limitations. State-of-the-art optimization-based approaches can handle arbitrary STL specifications but struggle with scalability, becoming computationally impractical as the number of agents grows. Learning-based methods efficiently handle a large number of agents with rapid planning times but fare poorly when deployment-time objectives differ from those used during training, and do not support planning tasks that require different specifications to be ascribed to different agents (i.e., heterogeneity) or team-level specifications requiring coordination of multiple agents. This fundamental trade-off between generalizability and scalability presents a challenge for realizing multi-agent STL planning algorithms in practice. To overcome this challenge, we introduce a new diffusion method for multi-agent planning with STL specifications. Using a differentiable approximation of STL, we integrate the STL gradient in the denoising process, making our approach generalizable to novel formulas whose predicates are placed anywhere within the goal region covered during training, while achieving the same scalability as existing learning-based methods. Our method supports heterogeneous specifications, and by using diffusion models, naturally enhances plan diversity, thereby significantly reducing safety-related violations (e.g., collisions) among agents. A detailed evaluation study justifies the utility of STL-guided diffusion-based multi-agent planners for constructing generalizable, scalable, and diverse plans. Videos and code are available at https://www.jeappen.com/diff-ma-stl/ and https://github.com/jeappen/diff-ma-stl .

### 论文解读

#### 研究问题
复杂机器人任务常用信号时序逻辑（STL）描述，例如先后访问目标、持续覆盖区域或执行循环。传统混合整数优化能够表达这些要求，但多智能体数量增加后规划很慢；学习型规划器虽然快速，却往往只能处理训练时见过的规格，任务变化就需要重新训练，还可能让机器人走过于相似的路线而拥堵。论文关注的是：能否在不为每种新逻辑重新训练的前提下，快速生成安全、分散且可执行的多智能体计划。

#### 核心方法
DIFF-MA采用两阶段设计。离线阶段只用单智能体轨迹训练扩散模型，让模型学习环境中的运动分布和动力学可行性，不把具体任务逻辑写死在网络中。在线生成时，为每个智能体采样轨迹，并把它们作为一个联合系统进行去噪。作者依据可微STL语义计算任务健壮度，将其梯度加入每一步去噪，引导轨迹满足不同智能体各自的序列、覆盖、循环或分支要求。同时加入可达性损失，避免生成的目标轨迹超出底层控制器能够跟踪的范围。生成多组候选后选择健壮度最高的方案，执行阶段再用图控制屏障函数进行实时避障。

#### 实验证据与意义
作者在DubinsCar仿真和Robotarium差分驱动机器人上验证方法，并与STLPY-SA、GNN-ODE、梯度规划器及单智能体扩散版本比较。在拥挤任务中，32个智能体时DIFF-MA的平均成功率超过84%，相对优化基线提高约36%，规划时间约为1秒，速度超过优化方法55倍。扩大场景后，方法仍能处理128个智能体，规划时间增长较平缓；路径重叠率接近零，表明其能减少机器人聚集和潜在死锁。由于逻辑约束在推理时注入，模型可组合训练中未出现过的任务规格，实现零样本泛化。

#### 局限与适用场景
方法需要已知且较准确的动力学模型来计算可达性梯度；屏障函数在障碍密集或极度拥挤时可能过于保守，导致死锁。约1秒的生成速度适合任务级重规划，但仍不适合毫秒级控制回路。总体而言，它适合仓储机器人、无人机集群和协作移动机器人等需要频繁修改任务逻辑的场景：扩散模型提供自然、多样的运动先验，STL提供严格的任务语义，安全控制器负责执行层面的碰撞规避。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29490v1)
- [arXiv](https://arxiv.org/abs/2608.29490v1)

---

<a id='2608.29510v1'></a>
## [ARMOR: Manifold-Oriented Training for Adversarially Robust Aerial Object Detection under Data Scarcity](https://arxiv.org/abs/2608.29510v1)

**Authors:** Haoran Wang, Matthew Lau, Alec Helbling, Matthew Hull, ShengYun Peng, Mansi Phute, Martin Andreoni, Willian T. Lunardi, Duen Horng Chau, Wenke Lee

**Published:** 2026-08-30

**Categories:** cs.CV, cs.CR, cs.LG

**Abstract:**

Aerial object detection is increasingly deployed in real-world applications, but models remain vulnerable to physical, universal adversarial patches that cause them to miss objects. Furthermore, defenders face the practical constraint of training data scarcity: aerial imagery is costly to collect and label, so a deployment site typically yields hundreds of images rather than the tens of thousands that adversarial robustness benchmarks assume. To tackle model vulnerability and training data scarcity, we propose Adversarial Robustness with Manifold-Oriented Training (ARMOR), a novel defense that realizes the core insights of on-manifold adversarial training (OMAT) in low-data regimes. ARMOR builds on the insight of OMAT to model the data manifold - the compact structure capturing the data's relevant features - to learn and robustify these features during training. While OMAT relies on the data-intensive operations of training large generative models and adversarial training to achieve this, ARMOR adopts a data-efficient approach that reuses labels the detection task already supplies: ARMOR (i) masks image backgrounds to retain object-relevant features, and (ii) injects randomized patches on objects to improve feature robustness. Our low-data experiments with physically-realizable adversarial patches evaluate both query-free transfer attacks and defense-aware attacks. ARMOR maintains strong clean performance of over 0.90 model confidence, while improving adversarial robustness by up to 0.32 in model confidence over state-of-the-art defenses. Physical experiments with printed patches confirm that these gains survive deployment. Overall, ARMOR translates insights from manifold-based training to defend object detectors amidst training data scarcity.

### 论文解读

#### 研究问题
航空目标检测常部署在无人机或固定空中视角，既容易受到车辆车顶贴片造成的逃逸攻击，又常只有数百张站点标注图像。传统像素级对抗训练可能损害干净精度；流形对抗训练需要生成模型和大量数据。ARMOR提出一种面向低数据场景的轻量防御。

#### 核心方法
ARMOR先用站点数据进行域适配，再利用检测框构造训练信号：把所有真值框外的背景像素置黑，近似投影到“目标相关”流形，迫使模型减少对背景线索的依赖；同时在目标中心加入均匀随机噪声贴片，模拟贴片变化并增强特征稳定性。仅使用掩码会让模型把非黑区域误认为目标，因此最终将原图与掩码图混合训练，在保留鲁棒压力的同时恢复真实背景分布。实验涵盖YOLOv3和约3M参数的YOLOv11n，数据来自843张Sidestreet图像和541张Drone停车场图像，并以天气变化扩展训练样本。

#### 主要证据
在干净测试上，ARMOR取得约0.92检测置信度和1.00 AP，与普通微调的0.96和1.00接近。面对数字ON-DA攻击时，ARMOR仍保持0.62置信度；FT、PAD和SHIELD只有0.14–0.17，随机平滑约0.30。实体打印贴片测试中，ARMOR置信度为0.75，而普通微调为0.38，未防御模型仅0.01。消融表明，背景掩码是鲁棒性跃升的主要来源，但单独使用会使AP降到0.30；混合训练把AP恢复到0.97，并将鲁棒置信度推到0.62。表示分析报告内在维数约降低5.7%，与学习更结构化、局部线性的特征表示相符。

#### 局限与实际意义
ARMOR需要边界框标注，尚不适合完全无监督的现场适配；评估重点是车顶ON贴片，环境中的OFF贴片覆盖较少；高运动模糊下干净表现不如部分基线。尽管如此，它说明在标注和算力都有限的无人机边缘部署中，可以用简单的标注驱动图像变换替代昂贵的生成式流形建模，并将防御迁移到不同规模的YOLO检测器。实际应用时仍应根据目标大小、相机视角、背景变化和攻击位置重新校准贴片分布及原图/掩码图比例。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29510v1)
- [arXiv](https://arxiv.org/abs/2608.29510v1)

---

<a id='2608.29537v1'></a>
## [AGM: Achievement-Grounded Memory for Closed-Loop Agents with Frozen VLA Policies](https://arxiv.org/abs/2608.29537v1)

**Authors:** Hongbo Gao, Zeyu Ni, Xin Wen, Siyu Xu, Ruifeng Li

**Published:** 2026-08-30

**Categories:** cs.RO, cs.AI

**Abstract:**

Frozen vision-language-action (VLA) policies offer broad manipulation skills but execute open-loop action chunks without tracking task progress, so the agent cannot reliably decide whether to continue, retry, or terminate. External memory is a natural remedy, yet it can be harmful when attempted actions are treated as completed progress, turning local execution errors into persistent task-state errors. We propose Achievement-Grounded Memory (AGM), a lightweight closed-loop framework for frozen VLA policies that represents a task as a subgoal sequence with a progress pointer and advances this memory only after the current subgoal is verified by physical evidence. Proprioceptive interaction cues decide when to verify, while coherent point tracking and language-conditioned cross-view comparison, sourced from frozen foundation models through a single 2.43M-parameter verification head, decide what was achieved. AGM thereby converts open-loop execution into a closed loop of execution, verification, and progress, keeping the policy frozen without test-time large-model inference. On the RoboMME Counting benchmark, AGM reaches on PickXTimes and on BinFill, surpassing the strongest memory-augmented baseline by points on average, and the framework yields equally decisive gains on a physical robot. Reliable embodied memory thus depends more on disciplined state updates than on memory capacity.

### 论文解读

#### 研究问题

视觉语言动作（VLA）模型擅长根据语言完成操作，却常以动作块开环执行，不维护可靠的任务进度。在重复搬运、计数和容器填充中，每次观测可能几乎一样，机器人无法判断已经完成了几次；若记忆系统把“尝试动作”直接当成“完成子目标”，一次抓空就会造成后续状态持续错误。AGM（Achievement-Grounded Memory）研究的问题是：不重新训练昂贵的 VLA，如何让冻结策略获得可信、低延迟的闭环进度记忆？

#### 核心方法

AGM 把指令展开为抓取、可恢复放置和不可逆放置等子目标序列，并维护一个进度指针。冻结 VLA 负责产生动作，系统再用夹爪开合度的滞回状态机定位抓取和释放事件；只有发生真实交互，才启动成就验证。抓取验证追踪夹爪附近的物体点：如果点随夹爪产生足够一致的向上运动，才确认抓取成功。放置验证则比较动作前后的前视与腕部相机观测，将冻结 SigLIP 的视觉和文本表示、本体感知信息送入一个约 243 万参数的轻量验证头，判断目标是否处在期望语义关系中。VLA、点追踪器和视觉编码器均不更新。

验证成功后指针前进；抓取失败时停留在原子目标并重试；可恢复的放置失败会回退到相应抓取步骤。系统还根据错误是否可恢复采用不同接受阈值（0.5 和 0.05），使高风险状态更新更谨慎。这种“执行—验证—更新”机制把记忆变成控制闭环的一部分，而不只是动作日志。

#### 实验证据与意义

在 RoboMME Counting 的四项任务上，AGM 平均成功率为 55.96%，高于冻结 VLA 的 28.78% 和 MemER 的 48.83%。在 PickXTimes 重复取放任务上达到 100%，而两种对照方法分别为 42.89% 和 79.33%；在 BinFill 容器填充任务上达到 84%，对照为 30% 和 56.67%。当重复次数提高到 5 次时，冻结基线在 PickXTimes 上降为 0%，AGM 的模拟结果仍为 100%；真实 AgileX PiPER 机械臂在重复次数 1 到 5 都成功。消融还表明，按验证成就更新记忆时 PickXTimes 成功率为 100%，按动作尝试更新时仅为 32%；移除针对不同放置风险的阈值后，BinFill 从 84% 降至 74%。

#### 局限与适用场景

AGM 当前最适合重复抓取、分拣、计数和容器填充。它的事件检测与验证主要围绕抓取—释放设计，推、工具操作等任务需要新的交互检测器和成就判别器；单一线性指针也难表示分支计划或乱序完成。视觉位移阈值、机器人本体信号和相机标定换到新平台时需要校准，验证头的训练仅使用 788 个交互事件，跨任务泛化仍需更多数据。总体而言，AGM 展示了一个实用原则：在冻结大模型外增加小型、基于证据的状态接口，往往比扩大记忆容量更能抑制闭环执行中的错误累积。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29537v1)
- [arXiv](https://arxiv.org/abs/2608.29537v1)

---

<a id='2608.29590v1'></a>
## [Guardrail-Agnostic Societal Bias Evaluation in Large Vision-Language Models](https://arxiv.org/abs/2608.29590v1)

**Authors:** Yusuke Hirota, Michael Ross Boone, Arun George Zachariah, Jibin Rajan Varghese, Yu-Chiang Frank Wang, Boyi Li, Ryo Hachiuma

**Published:** 2026-08-30

**Categories:** cs.CV

**Abstract:**

We propose a societal bias evaluation method for large vision-language models (LVLMs) in the era of strong safety guardrails. Existing benchmarks rely on prompts that ask models to infer attributes of people in images (e.g., "Is this person a CEO or a secretary?"). However, we find that LVLMs with strong guardrails, such as GPT and Claude, often refuse these prompts, making evaluations unreliable. To address this, we change the prior evaluation paradigm by decoupling the task from the depicted person: instead of inferring person's attributes, we use prompts that do not ask about the person (e.g., "Write a fictional story about an imaginary person.") and attach the image as provisional user information to implicitly provide demographic cues, then compare outputs across user demographics. Instantiated across three tasks --- story generation, term explanation, and exam-style QA --- our method avoids refusals even in guardrailed LVLMs, enabling reliable bias measurement. Applying it to 20 recent LVLMs, both open-source and proprietary, we find that all models undesirably use user demographic information in person-irrelevant tasks; for instance, characters in stories are often portrayed as mechanic for male users and nurse for female users. Although still biased, proprietary models like GPT-5 show lower bias than open-source ones. We analyze potential factors behind this gap, discussing continuous model monitoring and improvement as a possible contributor for reducing bias.

### 论文解读

#### 研究问题
大型视觉语言模型在面对性别或种族问题时常会触发安全护栏并拒绝回答，因此传统的“根据照片判断人物属性”评测很难区分真正无偏与单纯拒答。本文提出一种更贴近实际使用的思路：不让模型直接评价照片中的人，而把照片视为用户上下文，观察模型在看似与人物无关的任务中是否仍因外观改变输出。

#### 核心方法
作者以FairFace为基础，覆盖Male/Female和七类种族，并在20个开源及专有LVLM上进行三类测试。第一类是开放式故事生成：模型被要求写一个关于虚构人物的故事，再由辅助评估器抽取职业、经济状况、教育和性格等属性。第二类是术语解释：对数学、计算机科学、物理、艺术、文学和音乐六个领域的120个大学程度术语进行解释，比较不同用户得到的技术性和术语密度。第三类是600道MMLU选择题，用正确率差异检验受约束的知识服务是否公平。

作者用归一化Total Variation Distance衡量不同人口组的输出分布差异，分数为0表示一致，100表示最大差异；同时通过更换人物背景检验模型是否被场景线索误导，并以人工标注检查辅助评估器。

#### 主要结果
新协议在所有测试模型上都没有出现拒答，而传统基准在专有模型上曾有80%–100%的拒答率。偏见对任务高度敏感：故事生成最明显，平均分约27；术语解释约4.6；考试问答约1.5。代表性地，GPT-5在故事任务的性别和种族分数为14.53和16.80，在考试任务约为0.50，但整体结果并不意味着某个模型在所有任务都公平。故事更常把男性用户联系到mechanic或software developer，把女性用户联系到nurse或pastry chef；Black用户更常得到社区健康工作者或经济困难叙事，White用户更常出现律师。技术解释中男性和White用户更容易获得技术化回答，计算机科学任务对男性的技术性选择率约为88%。辅助评估器与人工判断的一致率达到97%。

背景替换实验中，本文方法的JS距离为0.38，传统描述式方法为0.61，说明它较少依赖背景和物体等非人物线索。不同任务之间的偏见并不稳定，相关性最低约为-0.11；同一任务内性别与种族偏见的相关性则为0.49–0.93。因此，单一任务或单一平均分不足以代表模型整体公平性，模型规模和能力也不能保证去偏。

#### 局限与意义
研究依赖FairFace的离散人口标签，二元性别和七类种族无法覆盖真实身份，也没有评估年龄、残障等属性。辅助语言模型可能对“技术性”带有自身偏差。尽管如此，该框架为护栏环境下的多模态偏见审计提供了可操作路径：部署前可用开放式任务进行压力测试，部署后可持续监测职业、经济地位和知识服务深度等输出分布。实验还显示，图像上下文比文字Persona产生更高偏见，提醒开发者不能仅依靠拒答率或安全对齐来判断公平性。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29590v1)
- [arXiv](https://arxiv.org/abs/2608.29590v1)

---

<a id='2608.29547v1'></a>
## [Module Number Adaptive Visual Shape Control for Serial Modular Soft Robots](https://arxiv.org/abs/2608.29547v1)

**Authors:** Kyohei Akamine, Takato Horii, Yusuke Sakaue, Hiroki Ishizuka

**Published:** 2026-08-30

**Categories:** cs.RO

**Abstract:**

Image based shape control provides a simple means of controlling the whole body configuration of soft robots. However, existing data driven approaches are typically developed for fixed robot structures and require new control data when the number of modules changes. This paper presents a module number adaptive visual shape control method for serial modular soft pneumatic robots. A controller trained only on single module actuation shape data is reused for robots with one to five modules by decomposing whole body camera images into local module patches. A single common module segmenter localizes individual modules across all tested configurations, while the same local controller is applied to every extracted patch. Geometric data augmentation improves transferability to downstream modules, and a lightweight mask reconstruction network reconstructs a synthetically removed actuator mask channel. Experiments on physical robots demonstrate shape control across varying numbers of modules and under environmental changes and payload loading. The results show that single module control learning enables scalable whole body control without configuration specific control data collection.

### 论文解读

串联模块化软体机器人可以通过增减模块改变长度，但传统数据驱动控制器通常针对固定结构训练，模块数量一变就要重新采集大量数据。论文研究的问题是：能否只学习一个模块的驱动—形状关系，却控制1至5个模块组成的机器人全身形状？这对受限空间作业尤其重要，因为仅控制末端可能导致机器人身体碰撞环境。

核心做法是把全身视觉反馈拆成模块级反馈。模块分割器先在图像中检测所有模块，按垂直中心从上到下排序，并将每个模块裁成标准化局部补丁。局部控制器随后从补丁中分割左、中、右三个 McKibben 气动人工肌肉的掩码，将当前掩码、目标掩码和当前归一化电压图组成9通道输入，由 CNN 编码器和 MLP 回归头预测下一步期望电压。实际电压按 (v_{t+1}=clip(v_t+0.5(hat v_t-v_t),0,5.0)) 更新，限制每次变化以提升闭环稳定性。考虑到遮挡、姿态变化和光照会造成掩码缺失，轻量 U-Net 使用其余掩码及电压信息重建缺失通道，重建损失结合二元交叉熵和 Dice 损失。

训练只使用单模块的1331（11³）对电压—形状图像，并通过20倍随机旋转、位移增强模拟多模块链中不同位置的视觉变化；模块分割器仅用覆盖1—5模块随机姿态的50张标注图微调。实验表明，1至5模块配置都能约20秒收敛到目标形状。五模块最终归一化 MSE 约为 (0.33\pm0.01)，模块数增多虽会使误差和方差上升，但几何增强明显减缓这一趋势。改变光照（平均像素强度137.65降至125.25）、加入背景障碍物，或对反馈图像施加15—25像素平移和30°旋转，控制仍保持有效；末端增加500克负载后误差变大，但机器人仍能接近目标。

方法的意义在于把单模块知识组合为可变长度机器人的全身控制，减少因硬件配置变化产生的数据采集成本，适用于模块可重复、视觉上可分割的软体臂。局限是四至五模块时重力引起的负荷分布变化会造成稳态误差，单目相机也无法充分观测深度方向运动；跨到不同形态模块仍可能需要新数据。后续可加入重力矩补偿，采用轮廓或中轴线等更通用特征，并用 NeRF 或 3D Gaussian Splatting 扩充视角训练数据。

**Links:**

- [PDF](https://arxiv.org/pdf/2608.29547v1)
- [arXiv](https://arxiv.org/abs/2608.29547v1)

---

