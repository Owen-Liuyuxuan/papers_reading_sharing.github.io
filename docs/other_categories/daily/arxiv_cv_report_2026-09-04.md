time: 20260904

# Arxiv Computer Vision Papers - 2026-09-04

## Table of Contents

1. [Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving](#2609.03225v1)
2. [RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning](#2609.03199v1)
3. [BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI](#2609.03497v1)
4. [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](#2609.03677v1)
5. [RoughSense: Lightweight Terrain-Induced Rover Vibration Prediction Using Point Clouds and IMU Feedback](#2609.03720v1)
6. [Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations](#2609.03657v1)
7. [Unfold The World: Factorize 4D Properties in Reinforcing Spatial Reasoning](#2609.03729v1)

---

## Papers

<a id='2609.03225v1'></a>
## [Long-Horizon Consistent and Interaction-Aware World Models for Multi-Style End-to-End Driving](https://arxiv.org/abs/2609.03225v1)

**Authors:** Yuxuan Han, Kunyuan Wu, Liyunong Yang, Zilu Wang, Cansen Jiang, Yi Xiao, Liang Hu

**Published:** 2026-09-03

**Categories:** cs.RO

**Abstract:**

End-to-end autonomous driving has increasingly adopted world model-based reinforcement learning frameworks to improve learning efficiency through \textit{imagined rollouts}. However, existing world models suffer from three key limitations: temporal inconsistency in long-horizon imagined rollouts, inadequate modeling of ego-environment interactions, and limited adaptability to diverse driving styles. To address these challenges, we propose \textit{StyleDrive}, a world-model-based learning framework that jointly enforces long-horizon consistency, explicitly disentangles interactive traffic states, and supports multi-style policy optimization within a unified learning paradigm. First, we introduce a temporal consistency regularization that integrates historical latent states through gated cross-attention, stabilizing long-horizon imagined rollouts and mitigating error accumulation. Second, we design an explicit state disentanglement module that separates ego-relevant from ego-irrelevant interactive states, enabling more interpretable and efficient decision-making in complex traffic scenarios. Third, we enable multi-style driving behaviors through Group Relative Policy Optimization, which replaces per-step reward optimization with trajectory-wise relative advantages, reducing reward variance and supporting diverse driving styles without retraining. We evaluate StyleDrive on the Bench2Drive closed-loop driving benchmark, achieving a driving score of 88.44 (+17.08 over the previous best world model-based method) and a success rate of 66.82 (+16.58). Furthermore, we deploy StyleDrive on a real automated guided vehicle platform and demonstrate promising sim-to-real transfer capability in dynamic driving scenarios.

### 论文解读

#### 摘要翻译
论文提出 StyleDrive，一个面向多风格端到端驾驶的世界模型。它针对长时想象漂移、交通参与者交互建模不足和驾驶风格难切换，结合时间一致性正则化、状态解耦与组相对策略优化。在 Bench2Drive 闭环测试中，多模态版本取得 88.44 的驾驶分数和 66.82% 的成功率，并在真实 AGV 上部署展示。

#### 方法动机分析
模仿学习容易因分布偏移失效，直接强化学习又承担真实试错的安全代价，因此作者让世界模型在潜空间中想象动作后果。但单步先验递推会累积误差，造成语义退化和运动不连续；把车辆、行人与道路元素混成整体，也无法突出真正影响自车的对象。固定奖励权重还常只能得到一种驾驶风格。作者假设历史先验能校正长期预测，自车相关性可筛选关键交互，轨迹级相对回报可稳定学习风格偏好。

#### 方法设计详解
输入是环视相机图像与 LiDAR 点云，输出是连续转向、油门和制动。首先用 BEVFusion 对齐多模态几何信息，并用 ViT 提取视觉语义，投影融合为观测表示；随后进入双分支 RSSM。自车分支编码可控状态，环境分支编码交通参与者与场景变化；每支都维护确定性状态和随机潜变量，训练时由当前观测得到后验，推理或想象时用先验递推。解码器从潜状态重建观测，并预测奖励与延续信号。

为抑制长时漂移，模型缓存最近 5 步先验，经交叉注意力聚合历史信息，再以门控与当前预测融合：G=σ(φ([ẑ_t；a_t]))。门值高表示历史上下文更值得信任，门值低则保留当前变化。环境状态再形成局部区域表示，以自车和环境特征的余弦相似度计算依赖分数 ξ=1/2(cos(similarity)+1)；分数不低于 0.4 的区域作为自车相关状态，其余区域分开保留，使策略集中关注可能改变自车轨迹的目标。世界模型损失包含延续负对数似然、奖励均方误差、观测重建误差，以及两分支的 KL 正则。奖励由路点进度、到达目的地、时间、平顺、碰撞和偏离组成，改变安全与效率权重即可表达不同风格。策略先在冻结世界模型中用 PPO 学习闭环控制，再将风格权重向量作为条件输入，用 GRPO 比较 6 条并行轨迹的整段回报；组内标准化回报作为相对优势，配合 0.05 裁剪和 KL 约束，减少奖励尺度和价值估计误差对风格切换的影响。

#### 方法对比分析
相比只做单步自回归预测的世界模型，StyleDrive 用历史先验校正长期一致性；相比把环境整体编码的方法，它按语义依赖分离关键与无关交互；相比为每种偏好单独训练策略的方法，它用风格条件和 GRPO 联合优化。其创新在于把长期预测、交互选择和偏好控制联成一体，适合动态交通与可调行为的仿真驾驶或移动机器人；若缺少几何对齐或可比较的风格回报，收益会减弱。

#### 实验分析（精简版）
作者在 CARLA 的 Bench2Drive 上闭环评估，数据含 1000 个片段，覆盖 12 个城镇。多模态 StyleDrive 驾驶分数为 88.44，基线 Epona 为 71.36，提升 17.08；成功率为 66.82%，高于基线的 50.24%。去掉时间一致性后分数降至 70.11，说明长期潜状态稳定性影响闭环决策。局限是证据主要来自仿真，真实场景有限，FID/FVD 也不能替代极端交通下的安全统计。

#### 实用指南
复现可按“传感器对齐—BEVFusion 与 ViT—双分支 RSSM—历史门控—交互解耦—PPO/GRPO”实现。设置包括世界模型训练 30 个 epoch、batch size 64、8 张 A6000，历史窗口 5、相关阈值 0.4、想象时界 6、GRPO 组大小 6。论文展示了 Jetson AGX Orin 64GB 上的 Scout AGV，但未明确代码地址和推理延迟。迁移时需重做动作头、奖励、标定与安全限幅，并验证交互阈值。

#### 总结
核心思想：历史校正交互，组内学习风格
1. 融合相机和 LiDAR，得到自车与环境潜状态。
2. RSSM 用历史门控稳定未来想象，并筛出关键交互。
3. 冻结世界模型，先用 PPO 学控制，再用 GRPO 调风格。
4. 在闭环仿真和有限实车场景中验证效果与迁移潜力。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03225v1)
- [arXiv](https://arxiv.org/abs/2609.03225v1)

---

<a id='2609.03199v1'></a>
## [RoboTok: An Internet-Scale Data Engine for Human Demonstration Retrieval and Dexterous Manipulation Learning](https://arxiv.org/abs/2609.03199v1)

**Authors:** Howard Qian, Yiting Chen, Yunfei Xie, Kejia Ren, Podshara Chanrungmaneekul, Gaotian Wang, Bowen Wen, Chen Wei, Kaiyu Hang

**Published:** 2026-09-02

**Categories:** cs.CV, cs.RO

**Abstract:**

Robot learning increasingly depends on broad and diverse demonstrations, yet collecting robot data remains expensive and poorly suited to covering the long tail of real-world tasks. To address this bottleneck, we introduce RoboTok, an internet-scale data engine that, given a query human manipulation video, retrieves manipulation-relevant human demonstrations from web videos for training dexterous robot policies. Specifically, we learn a latent motion space from 3D hand trajectories expressed in estimated actor-centered reference frames. This representation enables manipulation behaviors to be compared across variations in camera viewpoint, scene appearance, and actor occlusions, while remaining compact enough for efficient search and continual indexing over internet-scale video collections. We evaluate RoboTok against existing robot-data retrieval approaches on retrieval benchmarks and downstream robot policy performance. Our results show that RoboTok retrieves more relevant manipulation demonstrations and improves downstream task success, establishing hand-pose trajectory-aware retrieval as a way to make web video a scalable and continuously growing source of supervision for robot learning.

### 论文解读

#### 摘要翻译

机器人学习需要大量多样的人类示教，但采集昂贵，难以覆盖长尾任务。RoboTok 提出互联网规模的数据引擎：给定人类操作查询视频，从网络视频中检索动作本质相近的示教片段，服务于灵巧操作学习。它把视频转成规范化三维手部运动，并学习可快速向量检索的运动嵌入。

#### 方法动机分析

网络视频数量丰富，却有视角、背景、遮挡和速度差异；视觉相似不代表手指协调相似。作者假设：以示教者为中心规范化的三维手轨迹，比原始像素更接近灵巧操作本质；动态时间规整（DTW）可消除快慢差异。系统因此依赖手部可见和可靠重建，不能单独表达物体形状、接触力或完整语义。

#### 方法设计详解

输入是人类操作短视频，输出是按运动相似度排序的示教片段。数据引擎先从 Action100M 筛选 4–8 秒、相机近静止且手部可见的片段，并限制左右手数量，减少干扰。随后以 5 fps 用 WiLoR 提取每只手 21 个三维关节，用 MoGe-2 估计度量深度，将关节放入相机坐标；失败或遮挡帧由 HaWoR 补全。

相机坐标仍随拍摄位置变化，因此系统用 torso-frame estimator 根据手腕帧推断示教者的静态躯干参考系，再把轨迹变换到 actor-centered 坐标。同一种拧、推或抓动作的相对方向和幅度因此更稳定。轻量 cross-attention 编码器接收规范化时空姿态，加入位置编码并聚合序列，输出单位化的 (d) 维向量 Γ(x)。

训练时以轨迹 DTW 产生监督，而非依赖人工动作标签：


(s(i,j)=-DTW(x_i,x_j)/((L_i+L_j)/2))。

其中 DTW 在允许时间轴拉伸的前提下寻找最小累计关节距离，长度归一化后取负值就得到相似度。若 (s(i,j)>s(i,k))，损失便要求 Γ(x_i) 与 Γ(x_j) 的内积高于与 Γ(x_k) 的内积。每组包含一个锚点、两个正样本和一个边界负样本，集合损失与排序损失联合优化：前者学相似集合边界，后者保留内部次序。训练后离线编码全库并建立索引，在线只编码查询并做余弦搜索，用廉价检索近似昂贵的 DTW 排序。

#### 方法对比分析

FlowRetrieval 依赖光流，容易混入相机运动；HAND 使用二维手路径和视觉过滤；STRAP 侧重视觉基础模型特征。RoboTok 的本质区别是用三维手关节和躯干中心坐标描述动作，并让嵌入保持 DTW 排序。它适合手部运动决定成败的长尾任务；若关键因素是物体几何或力，需加入额外表征。

#### 实验分析（精简版）

作者在网络视频验证集和跨域三维手标注数据上评测检索，并将检索距离用于灵巧策略探索。结果表明，嵌入邻居更接近动作轨迹而非外观；仿真困难任务的稀疏反馈探索也得到改善。代表性结果是 mAP@20 达到 0.3531，STRAP 为 0.0071。局限是依赖近静态相机与手部可见性，下游证据主要来自仿真。

#### 实用指南

复现应保持短片筛选、5 fps、21 关节、度量深度、姿态补全和躯干规范化；参考规模是约 10 万段训练片段与 1 万段验证片段，batch size 为 196。工程上将重建、规范化、编码和建索引放在离线阶段，在线仅做编码与近邻搜索。论文提供项目主页并列出所用组件，完整代码、权重和互联网索引的开放范围需以主页为准。迁移到新机械手要重建索引并设计人手关节到机器人自由度的映射；移动相机或接触主导任务需增加补偿或物体特征。

#### 总结

核心思想：规范化三维手运动检索示教

1. 从网络操作视频筛出短片并提取、补全三维手轨迹。
2. 用躯干参考系消除不同拍摄视角带来的坐标差异。
3. 用 DTW 关系训练能保持动作排序的向量编码器。
4. 离线建立索引，在线快速找出相似人类示教。
5. 将示教相似度转成探索信号，辅助灵巧策略学习。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03199v1)
- [arXiv](https://arxiv.org/abs/2609.03199v1)

---

<a id='2609.03497v1'></a>
## [BRIDGE: An Open-Source Humanoid Platform via Morphology-Control Co-Design for Physical AI](https://arxiv.org/abs/2609.03497v1)

**Authors:** Jianren Wang, Letian Qian, Zikai Wang, Weiwei Wu, Junjie Zong, Abhinav Gupta, Deepak Pathak

**Published:** 2026-09-03

**Categories:** cs.RO, cs.AI

**Abstract:**

Developing humanoid robots capable of leveraging human behavioral data is essential for general-purpose embodiment, yet conventional development remains bottlenecked by a decoupled paradigm that isolates hardware design from whole-body control. This approach leads to suboptimal systems that compromise human-like fluidity and agility. To bridge this gap, we introduce a data-driven morphology-control co-design framework that optimizes humanoid morphology for human-like movement. To quantify morphological fidelity, we also introduce a novel metric that jointly considers kinematic retargeting fidelity to human motion and dynamic tracking performance. Our framework achieves state-of-the-art (SOTA) performance across all metrics compared to baseline humanoids (Bumi, K1, and Toddlerbot). Finally, we realize this design in Bridge, an open-source, 88cm-tall humanoid platform released alongside its control policy. We demonstrate that Bridge captures human motion data with superior fidelity, exhibiting exceptional performance across foundational locomotion, robust balance, and highly dynamic maneuvers. Videos and open-source materials: https://sites.google.com/view/bridgerobot.

### 论文解读

#### 摘要翻译
BRIDGE 是一个面向 Physical AI 的开源小型人形平台。论文认为，人形机器人不能只在既定硬件上训练控制器：若关节比例、活动范围或执行器能力不合适，控制器只能被动补偿。作者提出形态—控制协同设计框架，用人体运动数据同时检验几何映射和动态可行性，最终制成高约 88 厘米、重约 13 千克、含 21 个主动自由度的机器人。

#### 方法动机分析
传统“先定机械、后做控制”的解耦流程依赖经验，执行器封装和真实扭矩往往在后期才暴露问题。论文的核心假设是：运动学误差小不代表机器人能稳定完成动作，必须把形态选择、闭环跟踪和硬件瓶颈放进同一迭代环。目标是在小于 90 厘米、空间和功率受限时，仍尽量利用丰富的人类行为数据。

#### 方法设计详解
流程输入是 SMPL 派生的人体动作、候选关节拓扑和执行器库，输出是可制造的机器人形态及其控制策略。第一步从 23 个自由度开始做压缩，比较腰部自由度组合的运动学重定向误差；去掉腰部 Pitch 后，再比较单自由度方案，最终保留腰部 Yaw，动态误差为 E_dyn=0.02115，形成 21-DoF 结构。

第二步做执行器感知的实例化：为每个关节选择体积较小的候选，同时写入质量、惯性、减速比、安装间隙和实测扭矩—速度曲线；复合关节的轴位置还要避免相邻电机碰撞。第三步在 MuJoCo 中把动作重定向到模型，先使用共享基础策略，再按形态和动作微调策略；LaFAN1 与 bones_seed 是主要动作来源。动作只有在稳定、完成度达标且动态误差低于阈值时才算成功。

失败后计算关节利用率 ρ_j=峰值扭矩/校准扭矩上限。某关节若在多次尝试中反复饱和，且放宽它便能成功，就只升级该执行器，随后重算布局和惯性、重建模型并重训。最终用 S_HL=exp[-(0.5E_kin+0.5E_dyn)/0.05] 合并几何与动态误差；其直觉是，静态“像人”和实体“跟得上”必须同时成立，而不是用更大电机掩盖结构问题。

#### 方法对比分析
BRIDGE 与固定形态上训练控制器的方法不同，把形态当作可优化变量，并以运动覆盖率和执行器饱和定位结构瓶颈。它的主要创新是让失败直接反向驱动局部机械升级。相比只关注运动学习的 SONIC、BeyondMimic 等相关路线，它补上了硬件形态回写控制的闭环；相比 Bumi、K1、ToddlerBot 等基线平台，重点是公开且经过运动—动力学联合筛选的设计。该思路适合受尺寸、重量和成本约束、需要复现人类动作的小型人形机器人，但候选拓扑和成功判据仍需人工设定。

#### 实验分析（精简版）
在 MuJoCo 的 Balance、Highly Dynamic 和 Daily Motion 评估中，BRIDGE 的总体跟踪成功率为 94.83%，高动态类别比最强基线高 4.70 个百分点；联合指标 S_HL=0.5252，高于 Bumi 的 0.4321。实验说明真实执行器限制会显著改变结果，失败引导的局部升级能恢复动态能力。证据主要来自仿真和少量实体动作演示，尚不足以证明重载操作、长期可靠性或复杂现实环境泛化。

#### 实用指南
论文将硬件、设计和全身控制策略开源，但未完整给出依赖版本、学习率、训练轮数、显卡或固定推理延迟。复现时应保留 SMPL 重定向、LaFAN1 与 bones_seed 动作，使用实测扭矩上限建立含惯性的模型，并统一报告 E_kin、E_dyn、运动覆盖率和关节利用率。迁移到其他机器人时须重换拓扑、连杆、执行器曲线并重新微调；不能直接复制 21-DoF 轴布局。小型机的工作空间和负载也限制了迁移范围。

#### 总结
核心思想：形态与控制共同塑造类人运动

1. 用人体动作筛选满足尺寸约束的关节形态。
2. 把真实执行器参数写入模型并训练跟踪策略。
3. 从饱和失败定位瓶颈，只升级必要关节。
4. 重建模型、重训策略，再以几何和动态指标验收。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03497v1)
- [arXiv](https://arxiv.org/abs/2609.03497v1)

---

<a id='2609.03677v1'></a>
## [Understanding Autonomous Driving Datasets by Describing Differences between Image Subsets in Natural Language](https://arxiv.org/abs/2609.03677v1)

**Authors:** Julian Truetsch, Felix Hauser, Christoph Stiller, Frank Bieder

**Published:** 2026-09-03

**Categories:** cs.CV, cs.CL, cs.LG, cs.NE, cs.RO

**Abstract:**

Understanding the composition of large-scale autonomous driving datasets is essential for safety, robustness, and reliable operation across domains. For example, domain shift between locations could lead to the operating environment being misaligned with the training data, resulting in potentially dangerous performance degradation. Yet, existing data analysis pipelines largely rely on metadata, predefined labels, or manual inspection, which provide limited semantic insight or do not scale. This paper studies set difference captioning: given two subsets of images, the goal is to produce a natural-language hypothesis describing differences between the target and reference set. Building on a two-stage formulation, we adapt the method to autonomous driving by focusing on object-centric patches derived from object detection, which simplifies aggregation and enables attribution of differences to specific object instances or categories. To evaluate this setting in-domain, we introduce a new benchmark, AD-Diff Bench. Low-concentration experiments assess the suitability of set-difference-captioning approaches to sparse, real-world differences. We restrict our experiments to open-weight models to support reproducibility and ease of deployment. The proposed benchmark and analysis provide a step towards practical, human-interpretable dataset introspection for autonomous driving datasets. Our implementation and benchmark dataset are available at https://github.com/KIT-MRT/AD-Diff

### 论文解读

#### 摘要翻译

论文研究如何用自然语言描述两个自动驾驶图像子集的分布差异。现有元数据、固定标签和人工检查难以覆盖服饰、车辆状态及长尾事件等语义。作者提出面向对象的集合差异描述方法，并建立 AD-Diff Bench，让系统输出“目标集相对参考集更常见什么”的可读假设，再用全量图像验证。方法完全采用开源模型，面向数据审计、域偏移诊断和安全相关的数据理解。

#### 方法动机分析

自动驾驶数据受地点、天气、传感器和交通参与者变化影响，训练集与部署集之间的隐性偏差可能造成泛化和安全风险。单张图像描述或固定类别统计只能看到局部、预设的属性；直接让视觉语言模型比较整组道路场景，又会受到场景拥挤、上下文长度和稀疏证据的限制。论文的核心假设是：先让模型提出足够开放的语言候选，再在完整集合上检验候选，能把分散的弱信号聚合为稳定差异；对象中心化则有助于把差异归因到车辆、行人等同类目标。

#### 方法设计详解

输入是目标集 A、参考集 B，以及每张图像中的对象框；输出是按区分能力排序的自然语言差异。首先用预训练 2D 检测器或人工标注取得对象框，将框扩大 50% 以保留环境，并绘制红框提示模型关注目标。随后进入 proposer：每轮从 A、B 各抽 20 个 patch，重复 3 轮，每轮生成 10 条候选。image-based 分支直接比较多幅 patch；caption-based 先生成单图描述，再汇总集合差异；feature-based 将图像缩放为 224×224，利用两集视觉嵌入均值差的方向辅助文字生成。proposer 负责覆盖可能的语义，不要求一次抽样就准确。

候选 h 再交给 ranker 在完整 A、B 上复核。SigLIP 2 Giant 分别编码图像与文本，并以余弦相似度 R(x,h) 作为“图像是否支持该描述”的分数；将 A、B 的分数视作二分类证据，以 AUROC 衡量 h 区分两集的能力，最高者成为结果。生成候选使用 Qwen3-VL-30B-A3B-Instruct。直观上，proposer 负责发散，ranker 负责让语言解释经受全量数据检验，从而兼顾可读性与可测性。

#### 方法对比分析

单阶段基线让模型一次处理较多图像并同时生成、排序，流程简短却容易被注意力分散和低频事件牵制。两阶段设计把开放式假设生成与集合级验证解耦；对象 patch 还减少无关车辆、道路背景造成的噪声。image-based 与 caption-based 更能捕捉细粒度语义，feature-based 虽有视觉统计线索，却难把嵌入差异翻译成准确语言。该方法适用于数据筛选、偏置审计和跨域诊断，不等同于因果解释、安全认证或精确统计检验。

#### 实验分析（精简版）

AD-Diff Bench 覆盖网页检索集合、来自 KITTI/nuImages/Waymo 标注筛选的车载集合，以及 CLIP 筛选的对象 patch，难度从明显类别差异延伸到细微状态和外观差异。两阶段 image-based 在三个 split 的 Acc@1 准确率分别为 73%、56%、64%；单阶段对应为 64%、53%、49%，说明全量排序在网页和 CLIP 场景收益更明显。稀释实验显示差异样本浓度低于约 0.5 后性能开始下降，极稀疏事件仍可能漏检；因此结果应作为审计线索，并由人工或统计方法复核。

#### 实用指南

复现时准备目标集、参考集和同类对象框，统一执行 50% 框扩展、红框标记及对象裁剪；按“各集 20 个 patch、3 轮、每轮 10 候选”运行 proposer，再用 SigLIP 2 Giant 对全量集合排序。生成模型可使用 Qwen3-VL-30B-A3B-Instruct，feature 分支采用 224×224 输入。项目已开源代码与数据（KIT-MRT/AD-Diff）；论文未报告训练 batch size、学习率、epoch、显存或端到端延迟，迁移到新领域时应重新校准对象定义、采样策略和模型提示，并记录随机种子与检测质量。

#### 总结

核心思想：少量提议，全量验证差异

1. 准备两组图像，检测并裁剪同类对象。
2. 从少量 patch 生成多条语言差异假设。
3. 用图文相似度在全量样本上检验并排序。
4. 输出可信描述，再由人和统计检查复核。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03677v1)
- [arXiv](https://arxiv.org/abs/2609.03677v1)

---

<a id='2609.03720v1'></a>
## [RoughSense: Lightweight Terrain-Induced Rover Vibration Prediction Using Point Clouds and IMU Feedback](https://arxiv.org/abs/2609.03720v1)

**Authors:** Gabriel Manuel Garcia, Stephanie Aravecchia, Miguel Angel Olivares-Mendez

**Published:** 2026-09-03

**Categories:** cs.RO

**Abstract:**

Autonomous navigation in space requires reliable terrain assessment for safe operations, especially in underground environments with limited communication, computing resources, and power budget. This paper presents a lightweight method for real-time vibration-aware traversability mapping using a Light Detecting And Ranging (LiDAR) point cloud and Inertial Measurement Unit (IMU) measurements. An initial vibration proxy is estimated from terrain geometry by applying Random sample consensus (RANSAC) to local point-cloud patches produced by a Simultaneous Localisation And Mapping (SLAM) algorithm. In parallel, the IMU provides local observations of the vibration experienced by the rover during traversal. The point-cloud-based prediction is then corrected online using Recursive Least Squares, allowing the system to adapt the geometric estimate to the measured rover response. The approach is evaluated in a lunar analogue environment, an outdoor field, and an underground mine.

### 论文解读

#### 摘要翻译

RoughSense 面向月面、地下矿井等通信受限、光照不足且算力紧张的场景，提出一种轻量级地形诱导振动预测方法。它先用激光雷达点云从几何上预估前方地形粗糙度，再用车载惯性测量单元（IMU）记录的真实响应在线修正预测，最终生成可供规划器使用的振动代价地图。目标是提前发现会造成车体、传感器或机械系统强烈颠簸的区域，而不只是判断障碍物是否存在。

#### 方法动机分析

纯点云方法有前视能力，却把几何粗糙度过度等同于振动，难以表达悬挂、轮胎、速度和地表材料的差异；纯 IMU 能测到真实动力学响应，却只能在车辆驶过后观测，存在时间滞后。论文的核心假设是：点云几何残差适合作为振动先验，而几何到车体响应的映射应由当前平台的 IMU 反馈持续校准。相比需要大量标注和算力的端到端学习，这种可解释的在线适应更适合资源受限机器人。

#### 方法设计详解

输入包括 Livox MID-360 点云、RTAB-Map 提供的位姿、IMU 加速度和车体速度；输出是更新后的局部或全局振动成本。系统没有离线训练，全部参数在推理过程中在线估计。流程为：点云结合位姿聚合并栅格化；每个栅格用 RANSAC 拟合局部地面平面，计算点到平面的残差均方根
\(V_{pc}=\sqrt{\sum_i r_i^2/N}\)，残差越大就越可能引起颠簸，并经手工最小—最大标定归一化。该外感知支路把风险投射到车辆尚未驶入的空间。

并行的内感知支路先从频谱中辨识 Leo Rover 约 27.5 Hz 的电机机械峰，用陷波器去除平台自振；再取加速度向量范数，并按 \(\max(v,\epsilon)\) 除法做速度归一化，避免速度差异主导分数。在 200 Hz IMU 上以 100 个样本（约 0.5 秒）滑窗，计算相对窗口均值的 RMSE，得到实际振动 \(V_{imu}\)。随后 RLS 用线性映射 \(V_{corrected}=\alpha V_{pc}+\beta\) 拟合两者，特征为 \([V_{pc},1]^T\)，每次新观测都递归更新比例、偏置和协方差，再把参数应用到代价地图。遗忘因子 \(\lambda=0.995\) 在稳定与适应之间折中；残差门控、参数裁剪和协方差膨胀用于抑制异常反馈。

#### 方法对比分析

它区别于纯几何粗糙度评分之处，在于不假设“形状相同就振动相同”；区别于纯 IMU 评估之处，在于保留前视规划能力；区别于深度网络之处，在于只需少量在线参数、无需离线反向传播，计算和解释成本都低。创新是把点云回答“风险在哪里”和 IMU 回答“当前车辆会怎样响应”组成闭环。适合轮式探测车及特种作业机器人，但换平台时应重新辨识机械频率和映射。

#### 实验分析（精简版）

作者在 Lunalab 月壤模拟场、Symphony Lake 碎石岸和 Walferdange 地下矿井测试 Leo Rover。相对未校正的点云预测，RLS 使三场景平均误差分别降低 46%、73% 和 25%，说明平台反馈在几何与实际冲击偏差较大的碎石地尤其有效。Jetson Orin 单核 CPU 上，点云预测约 42.39 ms，RLS 修正约 234 微秒；但矿井远距离点云稀疏时收益明显变弱，线性映射也难覆盖复杂悬挂和载荷变化。

#### 实用指南

论文提供 RoughSense 的 GitHub 实现。复现时应先校准 LiDAR、IMU 与 SLAM 坐标并检查时间对齐，再从目标车辆频谱重新设置陷波频率；不能直接照搬 27.5 Hz。可从 200 Hz、100 点窗口、\(\lambda=0.995\) 和非零速度下限开始，记录归一化边界，并在已知路面检验分数与冲击是否单调。栅格大小需在细节和有效点数间折中。迁移到其他轮式平台通常需重测频率、速度范围和标定边界；论文未给出完整数据划分或跨平台零样本保证。

#### 总结

核心思想：用 IMU 在线校准点云振动

1. 点云结合位姿铺成局部栅格。
2. RANSAC 残差预估前方粗糙度。
3. IMU 滤噪、归一化并计算真实振动。
4. RLS 学习映射并实时更新代价地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03720v1)
- [arXiv](https://arxiv.org/abs/2609.03720v1)

---

<a id='2609.03657v1'></a>
## [Rethinking 3D Noise: Learning 3D-Aware Video Priors via Optimization-Free Morphological Perturbations](https://arxiv.org/abs/2609.03657v1)

**Authors:** Onat Şahin, Mohammad Altillawi, George Eskandar, Carlos Carbone, Ziyuan Liu

**Published:** 2026-09-03

**Categories:** cs.CV

**Abstract:**

3D scene representations like NeRF and 3D Gaussian Splatting (3DGS) suffer severe artifacts in sparse-view settings. Recent generative 3D artifact fixers attempt to address this, but rely on paired corrupted and clean renders requiring costly, per-scene reconstructions across varying view configurations. While 2D image augmentations act as instant regularizers, no explicit equivalents exist for 3D representations to preserve spatial consistency across views, an essential property for 3D-aware training. We propose 3D Morphological Perturbations as an optimization-free regularizer that preserves spatial consistency. Leveraging explicit 3DGS, we treat each Gaussian as a fundamental building block - analogous to a 2D pixel - and apply perturbations across its morphological parameter space via scale, rotation, and pruning. Our method eliminates per-scene 3DGS optimization loops from dataset curation while enabling models to learn stronger geometric priors than sparse-view baselines in diagnostic ablations conducted on a lightweight video diffusion sandbox. Scaled to a 14B-parameter video model via ControlNet, our approach maintains visual fidelity while reducing mean depth error by 12.5% over state-of-the-art image-to-image 3D artifact refiners, ultimately boosting downstream robotics policy success rates by up to 8.0% across 3 of 4 manipulation tasks.

### 论文解读

#### 摘要翻译

论文针对稀疏视角三维重建中的空洞、漂浮物和跨视角不一致，提出三维形态学扰动（3DMP）。它直接修改三维高斯泼溅（3DGS）原语，再沿同一相机轨迹渲染退化视频，免去逐场景优化，生成具有共同三维原因的训练对。作者希望视频先验据此学习几何补全，而非逐帧凭纹理猜测。

#### 方法动机分析

2D模糊、抖动或逐帧噪声只在像素层制造退化，不能保持视角与时间的一致性；逐场景优化虽更真实，却昂贵且难扩展。论文的核心假设是：若先扰动共享的三维表示，所有帧会共同呈现同一结构错误，模型就能利用运动、遮挡边界和邻帧证据恢复缺失表面。

#### 方法设计详解

输入是清晰3DGS与相机轨迹，输出是退化视频和对应清晰视频。首先在高斯原语层施加3DMP：位置为μ′ᵢ=μᵢ+εμ，噪声按场景尺度归一化（γxyz=0.002）；颜色和不透明度加入标准差0.05的高斯噪声并裁剪合法范围。形态模块以pprune=0.5的Bernoulli掩码随机删点，模拟表面缺失；尺度加入σscale=0.01噪声，旋转四元数加入σrot=0.01后归一化，分别模拟体积扭曲和法线错位。然后沿同一轨迹渲染整段条件视频，并以清晰视频监督条件扩散模型。

轻量验证使用Stable Diffusion v1.5+AnimateDiff：将4通道噪声潜变量与4通道条件潜变量拼成8通道输入，在空间、时间注意力中加入rank=64、alpha=128的LoRA。主系统使用14B参数Wan2.2 DiT与膨胀式ControlNet，复制8个DiT block、膨胀步幅为3；新条件通道以原权重0.1倍初始化，避免条件信号过强。Wan训练视频为832×480、49帧，采用AdamW、学习率2×10^-4、余弦调度、5% warmup和bfloat16，在8张80GB GPU上训练30个epoch；推理用Flow-Match Euler去噪50步，再在CIELAB空间做全局颜色校正。

一阶近似V(η)≈V₀+J_Rη说明，三维扰动经渲染器雅可比J_R投影到图像；在这些样本上训练会惩罚模型对结构一致变化的敏感性，相当于几何流形正则。删点负责逼模型补洞，尺度、旋转和辐射扰动则覆盖局部形状与外观误差。

#### 方法对比分析

与2D增强相比，3DMP先改原语后渲染，退化天然跨帧、跨视角一致；与逐场景优化修复器相比，它按统一规则批量造数据，不需要为每个场景反复重建。其创新不只是位置抖动，而是把剪枝、尺度和旋转等形态变化纳入噪声设计。它适合新视图补全、深度修复和机器人感知；只追求单图审美修补时，三维建模成本未必值得。

#### 实验分析（精简版）

在DL3DV-10K的极端S80诊断设置中，单独3DMP达到PSNR 14.907、SSIM 0.517、F-score 0.762，高于3D朴素扰动的F-score 0.710和2D模糊的0.724，说明形态缺失是有效信号。六视角无重叠高难度测试中，Morph.的深度RMSE为29.53，Fixer为54.24；机器人Close Jar成功率也由51.2%升至59.2%。但Insert Peg约1.6%，表明视觉改善不能替代精密控制，收益有边界。

#### 实用指南

复现需准备可渲染3DGS、相机轨迹和清晰目标，按上述删点及参数噪声生成配对视频；主训练数据约1万对，来自DL3DV、ScanNet和ScanNet++，并保持场景隔离。实现时先确认同一掩码作用于整段序列，再检查删点后的可见性排序与渲染。论文未承诺可直接安装的完整实现；迁移到其他显式三维原语可沿用“参数扰动—轨迹渲染—视频条件生成”接口，但隐式NeRF需另设等价结构扰动。动态物体、严重遮挡或不可靠3DGS可能收益下降。

#### 总结

核心思想：在三维高斯上制造缺失

1. 输入3DGS和相机轨迹。
2. 随机删点并扰动形状、位置和外观。
3. 渲染退化视频，训练条件扩散模型补全。
4. 采样、校色，再检查深度与操作效果。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03657v1)
- [arXiv](https://arxiv.org/abs/2609.03657v1)

---

<a id='2609.03729v1'></a>
## [Unfold The World: Factorize 4D Properties in Reinforcing Spatial Reasoning](https://arxiv.org/abs/2609.03729v1)

**Authors:** Yijun Yang, Shenghe Zheng, Wenbo Li, Jianhui Liu, Haoze Sun, Yanbing Zhang, Jiaxiu Jiang, Lin Song, Haoyang Huang, Nan Duan, Lei Zhu

**Published:** 2026-09-03

**Categories:** cs.CV

**Abstract:**

Despite the remarkable prowess of Vision-Language Models (VLMs) in general multimodal tasks, they remain fundamentally ``flat'' when reasoning about the physical world. We argue that this spatial bottleneck stems from a profound dimensional mismatch: while VLMs are trained to interpret 2D projections, true spatial reasoning demands the recovery of latent 3D geometry and temporal continuity. To conquer this high-dimensional complexity, we advocate a shift from monolithic learning to a ``divide and conquer'' paradigm. We present FactoSR, a factorized reinforcement learning framework that explicitly interpret the dimensions collapsed by visual projection. At its core, FactoSR decomposes the monolithic problem of world-consistent reasoning into three orthogonal, geometric sub-objectives: planar correspondence ($XY$), depth consistency ($Z$), and temporal reversibility ($T$). By optimizing these verifiable constraints within a unified policy learning mechanism, we effectively transform an ill-posed projection recovery problem into a series of tangible reasoning steps. Extensive evaluations on multi-view and video benchmarks demonstrate that this elegant decomposition yields substantial gains in 3D and 4D reasoning, achieving a 5.9% boost on VSI-Bench and 4.5% on All-Angles-Bench. Our findings suggest that reinforcing explicit, factorized 4D consistency is a critical step toward evolving VLMs into robust, world-aware reasoners.

### 论文解读

#### 摘要翻译

FactoSR 面向视觉语言模型的四维空间推理：模型不仅要理解二维图像，还要恢复跨视角的三维几何与视频中的时间关系。作者指出，VLM 常凭物体外观和二维投影猜答案，因而在距离、遮挡、定位及动态路线任务上产生幻觉。方法把空间世界分解为 XY 平面对应、Z 深度顺序和 T 时间可逆性，并用可验证奖励引导模型遵守这些约束。

#### 方法动机分析

单图 SFT 难以教会模型多视图对齐；盲目增加空间数据或 token 又容易学习捷径，直接监督完整 4D 表征则难以定义稳定标签。FactoSR 的核心假设是：复杂空间能力可拆成若干相对独立、可程序检查的事实。只要同时约束“是否对应同一点”“谁在前谁在后”和“动作倒放是否自洽”，就能把语言上的合理猜测推向几何一致的推理。

#### 方法设计详解

输入是多视图图像或视频序列与空间问题，输出为带思考过程和答案的结构化文本。第一阶段 FactoSR-SFT 以 Qwen3-VL-8B-Instruct 为骨干，用约 8.2M 个通用及空间样本做空间感知微调，并以 Anchor–Transfer–Verify 组织跨帧推理：先在源视图找锚点，再转移到目标视图，最后核验对应关系。第二阶段 FactoSR-RL 用 GRPO 为每个问题采样 8 个回答，以组内相对优势更新策略，不需价值网络；KL 项抑制策略偏离参考模型。

奖励先检查输出格式，再组合语义准确率 (R_{acc})、对应奖励 (R_{XY})、深度奖励 (R_Z) 和时间奖励 (R_T)。XY 使用相机内参、位姿和深度把源像素反投影到三维，再投影到目标视图；预测点与重投影点距离为 (d) 时，奖励直觉上按 (exp(-d/sigma)) 衰减，并用可见性掩码忽略遮挡点。Z 不回归绝对米制深度，而以 Kendall-(\tau=(N_c-N_d)/[n(n-1)/2]) 衡量物体前后顺序。T 将正向轨迹与逆向轨迹同时验证，例如“向右转”的反向应为“向左转”。SFT 全局 batch 为 128、序列长度 8192；RL 学习率为 (10^{-6})、温度 1.0。

#### 方法对比分析

相较只优化答案准确率的 SFT，FactoSR 把生成结果放入相机几何和时序闭环中检查；相较绝对深度回归，Z 因子学习更稳健的相对秩序；相较 vanilla GRPO，它能区分对应、深度与时间方向的错误。创新在于把外部可计算约束变成因子化奖励，适合多视图理解、视频推理、机器人导航和路线规划。

#### 实验分析（精简版）

论文在 VSI-Bench、All-Angles-Bench 及 3D/4D 基准上与多种开源和闭源模型比较。FactoSR-8B-RL 在 VSI-Bench 达到 61.5%，较基础模型提升 5.9 个百分点；消融中 vanilla GRPO 仅提升 0.2%，加入 T 奖励后时间任务增益约 7.9，说明专门的物理约束是主要来源。代价是方法依赖可靠标定、深度和可见性信息，遮挡或非刚体运动会削弱奖励。

#### 实用指南

论文公开了 FactoSR 代码。复现需准备多视图内参、位姿、深度及可见性标注，严格实现结构化输出和先格式后几何的奖励门控，并按两阶段训练及每题 8 次 rollout 设置验证。迁移到导航时，可将 XY 换成路标重投影、Z 换成障碍物前后关系、T 换成往返路线一致性；没有标定时需改用学习式几何约束。

#### 总结

核心思想：用可验证因子强化四维推理

通俗 pipeline：
1. 多视图/视频输入并寻找锚点。
2. 转移锚点并检查 XY 对应。
3. 检查物体深度顺序。
4. 验证正逆时间轨迹。
5. 用 GRPO 联合优化答案与物理一致性。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.03729v1)
- [arXiv](https://arxiv.org/abs/2609.03729v1)

---

