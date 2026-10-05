time: 20261005

# Arxiv Computer Vision Papers - 2026-10-05

## Executive Summary

## 今日摘要

本次从 91 篇 CS.CV/CS.RO 记录中，经关键词政策筛出 37 篇可评分论文；逐篇评估摘要后选取 10 篇完整阅读。研究主题主要落在两条相互交织的主线上：一是让移动机器人更好地理解全景、鱼眼和跨平台视觉输入，二是提升视觉语言动作策略的动作表示、泛化能力与部署效率。

全景感知方向从不同任务补足同一类几何缺口。EmbPASS构建跨车辆、无人机、可穿戴和四足平台的开放词汇全景分割基准；OmniAct3D把透视基础模型适配到连续ERP全景三维检测；FUSEye则以重叠鱼眼视图和轻量适配器缓解径向畸变。值得关注的是，方法不只增加数据或模型容量，而是显式建模球面几何、跨视角关系及局部目标证据。

操作与策略方向出现了更丰富的中间表示和训练方式。Skill2Real研究代理式技能学习及零样本仿真到真实迁移；FastOPD面向轻量VLA部署进行在线策略蒸馏；Proprioceptive Sketches把粗粒度意图作为长时程动作条件；MixVLA、PointWAM和RRDF分别从非不变信息融合、场景与手部三维轨迹预测、视觉与状态token路由来增强泛化。RYOPO则关注类别级物体姿态估计的实时化。共同趋势是把原本隐式的动作条件、感知线索或跨模态路径变成可分析、可控的结构。

建议优先精读OmniAct3D与EmbPASS以了解全景感知的几何适配和跨具身评测；对机器人控制研究者，PointWAM、RRDF与MixVLA提供了互补的3D动作表示和融合设计。FastOPD、Skill2Real及Proprioceptive Sketches适合关注效率、迁移和长时程规划的读者。现阶段，多数证据仍来自特定基准或有限实机设置，跨平台、长时程和真实部署时的稳定性仍需独立验证。

---

## Table of Contents

1. [EmbPASS: Towards Cross-Embodiment Open Panoramic Segmentation](#2610.03248v1)
2. [OmniAct3D: Leveraging Foundation Geometry and Evidence-Grounded Reasoning for Panoramic 3D Detection](#2610.03015v1)
3. [FUSEye: Training-Light Fisheye Detection with Overlapping Views and Zero-Initialized Adapters](#2610.02799v1)
4. [Skill2Real: Agentic Skill Learning for Zero-Shot Sim-to-Real Robot Manipulation](#2610.02788v1)
5. [FastOPD: On-Policy Distillation for Lightweight VLA Deployment](#2610.02832v1)
6. [Proprioceptive Sketches as Long-Horizon Intent for Generative Action Policies](#2610.02759v1)
7. [RYOPO: Bringing End-to-End Category-Level Object Pose Estimation into Real Time](#2610.03013v1)
8. [MixVLA: Adaptive Mixing of Non-Invariant Information for Generalizable Vision-Language-Action Models](#2610.02898v1)
9. [PointWAM: 3D World Action Modeling for Dexterous Robotic Manipulation](#2610.02840v1)
10. [Register-Routed Delayed Fusion: Rewiring Shortcut-Prone Observation Fusion in Visuomotor Imitation](#2610.02813v1)

---

## Papers

<a id='2610.03248v1'></a>
## [EmbPASS: Towards Cross-Embodiment Open Panoramic Segmentation](https://arxiv.org/abs/2610.03248v1)

**Authors:** Pujun Guo, Yuanfan Zheng, Fei Teng, Mengfei Duan, Guoqiang Zhao, Yuheng Zhang, Kai Luo, Kailun Yang

**Published:** 2026-10-02

**Categories:** cs.CV, cs.RO

**Abstract:**

Panoramic images provide a complete 360-degree field of view, enabling comprehensive scene understanding for embodied perception. However, heterogeneous embodied platforms exhibit substantial differences in observation viewpoints and spatial layouts, giving rise to cross-embodiment observation shifts that pose additional challenges to consistent and reliable panoramic perception, while systematic studies of this problem remain limited. To bridge this gap, we introduce a new task, termed Cross-Embodiment Open Panoramic Segmentation. Meanwhile, we establish EmbPASS, a multi-platform panoramic semantic segmentation benchmark spanning Vehicle, Drone, Wearable, and Quadruped platforms under a unified semantic taxonomy, providing a testbed for systematically studying cross-embodiment panoramic perception. We further propose EPONet, an open-vocabulary panoramic semantic segmentation network that integrates Relation-Aware Metric Adapter (RAMA) and Content-Adaptive Semantic Transfer (CAST) to enhance spatial modeling and semantic transfer under heterogeneous embodied observations. Extensive experiments show that EPONet achieves the best platform-balanced performance on EmbPASS with 35.82% mIoU, outperforming the strongest baseline by 1.10%, while remaining competitive on existing panoramic segmentation benchmarks. The source code and EmbPASS benchmark will be made publicly available at https://github.com/guopj1/EmbPASS.

### 论文解读

#### 摘要翻译
全景图像能提供完整的360度视野，有利于具身系统理解周围环境。但车辆、无人机、可穿戴设备和四足机器人安装相机的位置不同，画面视角与空间布局也随之变化，使同一分割模型跨平台使用时容易失准。作者提出“跨具身平台开放全景分割”任务，并建立EmbPASS基准，包含四类平台的全景图像和统一语义标注。为解决这一问题，论文设计EPONet，在带标签的普通针孔图像上训练，再直接处理未见平台的全景图像。

#### 方法动机分析
开放词汇分割多针对透视图，全景方法也少覆盖多平台。等距柱状投影的畸变随纬度变化，安装位置不同会改变物体布局和形变。论文假设，按局部内容和方向关系调整采样，并选择性迁移CLIP语义，可缓解这种变化。

#### 方法设计详解
EPONet包含冻结的CLIP视觉编码器和可训练的Side Adapter。训练时使用普通针孔图像；推理时输入全景图。Side分支生成类别无关掩码候选和注意力信息，CLIP再分类候选区域，合成逐像素分割图。

RAMA负责适应全景图的局部几何。它先压缩Side特征，再比较每个位置与周围八个方向邻点的特征相似度。内容特征和方向关系共同预测一个局部Randers度量，用来约束3×3采样邻域的形状、尺度和方向偏置。随后把这个结构化采样位置转换为相对常规网格的偏移，执行特征聚合并残差回注。与直接独立预测采样点相比，这种做法让一组采样位置共享一个局部几何约束。

CAST负责迁移语义。它将当前Side特征和CLIP中间特征投影到较小空间，合成内容相关的重组权重，对CLIP特征作局部重排；另一个门控分支控制这些信息注入Side特征的程度。这样模型可以保留CLIP的类别知识，又根据当前全景内容调整特征对齐方式。作者发现，过早在最浅层使用CAST效果反而下降，因此只在较深的三个交互阶段使用。

#### 方法对比分析
以往全景分割方法常通过可变形卷积或几何变换处理投影畸变。RAMA与它们的区别是用一个共享的局部度量约束整组采样点，并把邻域方向关系纳入度量预测；对比MetricConv，它额外建模中心与不同方向邻点的关系。CAST也不同于直接把CLIP特征送入分支，而是同时参考CLIP和Side内容，动态重排后再门控传递。两者分别针对局部几何变化和预训练语义对齐，适合相机视角、安装高度或投影条件变化明显的全景感知任务。

#### 实验分析（精简版）
EmbPASS有1,000张图像，车载、无人机、可穿戴和四足平台各250张，统一标为19类。模型使用COCOStuff的118,000张训练图像，在单张RTX 3090上以batch 8训练60,000次。EPONet在EmbPASS上取得35.82%平台均衡mIoU，比最强对比方法OOOPS-RERP高1.10个百分点；在DensePASS上取得49.15% mIoU，为比较方法中最高。加入RAMA和CAST后，EmbPASS平均mIoU从基础模型的33.82%升到35.82%。不过无人机子集仍较难，EPONet为25.90%，略低于该子集最佳值。实验说明其改善了基准分割表现，尚不能证明下游导航收益或闭环实机效果。

#### 实用指南
作者说明代码与EmbPASS将公开。复现可从冻结CLIP ViT-B/16和Side Adapter入手，论文给出训练和推理设定：全景推理用640像素窗口、320步长。迁移前应核查新相机投影、纬度覆盖和类别标签。数据由既有平台数据与新采无人机图像组成，使用时还需确认发布版本和许可。

#### 总结
核心思想：按视角适配几何与语义
1. 用普通针孔图像训练冻结CLIP与轻量分割旁路。
2. 由邻域方向关系预测局部几何采样规则。
3. 根据当前内容重排并门控传递CLIP语义。
4. 组合候选掩码与开放词汇类别，输出全景分割。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.03248v1)
- [arXiv](https://arxiv.org/abs/2610.03248v1)

---

<a id='2610.03015v1'></a>
## [OmniAct3D: Leveraging Foundation Geometry and Evidence-Grounded Reasoning for Panoramic 3D Detection](https://arxiv.org/abs/2610.03015v1)

**Authors:** Runtong Wu, Fei Teng, Di Wen, Guoqiang Zhao, Kunyu Peng, Kailun Yang

**Published:** 2026-10-02

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Accurate 3D detection is essential for mobile embodied agents, while Vision Foundation Models (VFMs) offer transferable visual and geometric priors. Yet existing VFM-based 3D detectors rely on narrow-view monocular images or discrete perspective views, limiting coherent surround perception; equirectangular projection (ERP) instead encodes a continuous 360 scene in a single image. Direct transfer remains difficult because ERP organizes geometry and visual information differently, making object-relevant cues hard to model, localize, and preserve. We propose OmniAct3D, a framework that adapts perspective-trained VFM detectors to ERP while preserving transferable VFM priors. To resolve geometric mismatch, the ERP-Ray Geometry Adapter (ERGA-Ray) models spherical viewing rays and periodic spatial structure. To localize evidence in scene-wide context, the Visual-Action Reasoning Chain (VARC) grounds each hypothesis in relevant panoramic evidence and converts it into a structured geometric action. To recover local cues lost under fixed token budgets, the Appearance-Guided Heading Expert (AGHE) re-encodes object regions at higher resolution for heading estimation. Experiments show that OmniAct3D improves over the previous best 3D detector by 2.96 NDS points on Spheriverse and over the unadapted VFM baseline by 24.87 mAP points on PanoMMOcc. With target-specific geometry adaptation, VARC retains 95--98% of the same-configuration mAP, indicating reusable object-level 3D reasoning across sensing configurations. The source code will be made publicly available at https://github.com/FeiT-FeiTeng/OmniAct3D.

### 论文解读

#### 摘要翻译
单目三维检测视野有限，多摄像头方法依靠离散画面拼接。ERP全景图能在一张图中连续覆盖360度，但透视预训练的视觉基础模型不熟悉球面射线、纬度形变和水平接缝；全图压缩还会丢失小目标证据与局部朝向线索。OmniAct3D保留冻结的基础检测器，增加几何适配、证据推理与局部朝向专家，在两个全景三维检测基准上取得改进。

#### 方法动机分析
现有透视模型迁移到ERP的痛点，是视觉和几何先验都与球面投影错位。直接从整幅全景解码物体框，也难以定位每个目标真正相关的图像证据；固定token预算又可能压掉车头车尾等朝向特征。作者的核心假设是：保留预训练表示，再分别补上全景几何、目标证据定位和局部高分辨率外观，能改善全景三维框及朝向，而不必重训整个基础检测器。

#### 方法设计详解
输入ERP图像经冻结的SAM3与LingBot/DINOv2检测器，得到视觉特征与几何记忆。第一步，ERGA-Ray给几何token加入球面射线、经纬角编码，经FiLM调制和空间混合生成残差化全景记忆；水平方向用循环填充连接接缝，垂直方向不循环。第二步，检测器从适配特征产生初始3D框、对象查询和置信线索。VARC围绕每个框取13个支持点，从全景记忆读取目标证据，再结合框状态和对象查询预测受限的结构化几何动作，更新框假设。第三步，AGHE把目标框投影回原始全景图，在更高分辨率下重新编码裁剪区域，估计周期性朝向；它保留目标中心、尺寸、类别和置信度，只修正朝向。三个新增组件分别训练，基础检测器保持冻结。

#### 方法对比分析
常见BEV检测器面向透视多相机网格，简单把预训练特征用于ERP又忽略球面几何。ERGA-Ray同时表示实际观察射线和ERP横向周期；VARC区别于直接回归整图框，使用目标假设定位支持点证据并把推理写成几何动作；AGHE则通过目标裁剪恢复全图压缩丢失的细节。消融中单独几何适配贡献大，VARC更明显改善定位，AGHE主要改善朝向，模块职责与误差类型相对应。

#### 实验分析（精简版）
Spheriverse上mAP/NDS为0.1887/0.1825，优于最佳对照0.1689/0.1529；PanoMMOcc为0.5897/0.4032，对照0.5758/0.3813。AGHE不改变mAP，但将两数据集平均朝向误差从1.5585、2.0781降至0.8455、0.7902。VARC转移到另一传感配置时需使用该域专属几何适配器，冻结的VARC仍保留同配置mAP的95%和98%。这些基准使用静态全景图，不能据此推断闭环移动导航性能。

#### 实用指南
实现需提供完整ERP图、标定与三维框，将LiDAR坐标转换到全景相机坐标。论文输入缩放为2480×512，在四张RTX 3090上训练；冻结基础检测器，依次优化ERGA-Ray、VARC与AGHE。迁移时应独立检查水平接缝、不同垂直视场、纬度畸变和目标裁剪是否正确，并报告朝向误差及端到端延迟。新增几何模块和推理成本不高，但真实系统仍需测试小目标、遮挡、曝光差异和传感标定偏差。

#### 总结
核心思想：补足全景几何与目标细节
1. 用球面射线适配冻结的几何特征。
2. 围绕框假设读取全景证据并修正几何状态。
3. 高分辨率重编码目标区域，单独校正朝向。
实验显示模块分别提升检测定位与方向估计，跨配置推理仍依赖目标域几何适配；时序与闭环场景尚待验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.03015v1)
- [arXiv](https://arxiv.org/abs/2610.03015v1)

---

<a id='2610.02799v1'></a>
## [FUSEye: Training-Light Fisheye Detection with Overlapping Views and Zero-Initialized Adapters](https://arxiv.org/abs/2610.02799v1)

**Authors:** Wenya Su, Kai Luo, Di Wen, Ruiping Liu, Yufan Chen, Junwei Zheng, Kunyu Peng, Kailun Yang

**Published:** 2026-10-02

**Categories:** cs.CV, cs.RO, eess.IV

**Abstract:**

Fisheye cameras give mobile robots a single-sensor, low-cost view of their surroundings, yet the COCO-pretrained detectors that practitioners routinely reuse fail on them: strong radial distortion warps local image structure, while boundary compression shrinks objects to near-invisible sizes. Full fine-tuning closes much of the gap but requires abundant fisheye labels and compute. We present FUSEye, a training-light framework that turns a frozen-backbone COCO-pretrained extra-large YOLO26 detector (YOLO26-x) into a fisheye detector. FUSEye adds roughly 227k new parameters while updating the inserted modules and the pretrained detection head. It addresses the transfer gap at three causally linked levels. At the input level, overlapping grid view generation and box remapping (GridViews) enlarge compressed boundary regions. At the feature level, zero-initialized residual adapters (Z-Adapters) correct distortion-induced feature misalignment. At the decision level, learned cross-projection agreement fusion (AgreeFusion) promotes low-confidence detections only when they are supported by consistent evidence across multiple views. On the WoodScape surround-view fisheye benchmark, FUSEye raises YOLO26-x from 0.148 to 0.266 mAP50 and retains 84.3% fully fine-tuned accuracy. Moreover, randomly using only 25% of the labeled training images, FUSEye achieves 0.2597 mAP50, retaining 97.6% of its full-label performance. FUSEye also consistently improves YOLOv8-11 detectors, showing that the recipe is architecture-agnostic. Source code will be available at https://github.com/Su-wenya/FUSEye.

### 论文解读

#### 摘要翻译
鱼眼相机用单个传感器就能覆盖很大的视野，适合移动机器人和环视系统。但COCO预训练检测器主要见过普通透视图，鱼眼的径向畸变会拉伸图像局部结构，并把边缘物体压缩成很小的区域。全量微调虽然有效，却需要更多标注和计算。作者提出FUSEye，以冻结预训练主干为基础，组合重叠视图、零初始化残差适配器和跨视图一致性融合，适配鱼眼目标检测。

#### 方法动机分析
论文把迁移失败分为三个环节：全图输入下边缘小物体尺度不足；鱼眼畸变使预训练特征和目标外观错位；局部裁剪又带来重复框和相互冲突的检测。核心思路是分层处理：先放大局部观察，再在高层特征中学习轻量残差，最后只提升获得多视图支持的候选框。零初始化保证适配器刚插入时不改变原模型行为。

#### 方法设计详解
GridViews为一张鱼眼图生成全图与四个相互重叠的角落裁剪，共五个视图，均缩放至640×640。每个视图由共享的YOLO检测器处理，预测框按裁剪位置映回原图坐标。这样边缘物体在局部视图中占据更多像素，同时全图保留场景上下文。

Z-Adapter插入YOLO26主干的P4、P5和C2PSA阶段。支路先用1×1卷积降维，再以深度卷积提取局部特征，最后升维并与原特征相加。输出投影零初始化，使初始支路输出为零；训练时更新适配器和检测头，主干参数保持冻结。作者报告三个适配器共222,480个参数。

AgreeFusion将多视图框映回同一坐标系后按类别和IoU聚类，再利用覆盖视图数、置信度、框一致性和中心偏移等特征计算簇分数。得到多个视图支持的候选可以被重新评分并加权融合；位移较大的框保留最高置信预测。单视图检测沿用原置信度门限，最终再执行非极大值抑制。整个流程无需相机标定或图像去畸变，但每张图要运行五次检测器。

#### 方法对比分析
常规切片推理能放大小目标，却通常只做坐标映回和去重；通用参数高效微调减少训练参数，却没有针对鱼眼畸变设计。FUSEye把输入切片、特征适配和预测框融合结合起来，并让零初始化适配器尽量保留预训练能力。与全量微调相比，它训练参数更少、效果也更易迁移到多个YOLO型号，但推理仍需五次前向计算。

#### 实验分析（精简版）
实验在WoodScape镜面右视图MVR验证集上进行，训练使用前、后和镜面左视图，过滤后有6,082张训练图。YOLO26-x直接迁移的AP50为14.80%，FUSEye达到26.61%；AP50:95从10.96%升至17.26%。全量微调分别达到31.56%和22.82%，仍明显更高。仅用25%训练数据的三组结果平均AP50为25.97±0.34%，约为全数据26.61%的97.6%。不过公交车类别得分很低，FUSEye还低于直接迁移；这项评测仅含汽车、行人、公交车，也没有实车闭环结果。

#### 实用指南
作者称代码将公开。配置细节包括640×640输入、五视图裁剪，适配器和检测头训练8个epoch；融合评分器单独训练。迁移到机器人相机时，应先确认输入分辨率与鱼眼边缘物体比例，并评估五次前向带来的运行延迟和功耗。若目标类别与训练类别不同，需重做类别映射及融合器校准。

#### 总结
核心思想：分层修补鱼眼迁移
1. 重叠裁剪放大边缘物体。
2. 零初始化适配器校正特征。
3. 按多视图支持融合检测框。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02799v1)
- [arXiv](https://arxiv.org/abs/2610.02799v1)

---

<a id='2610.02788v1'></a>
## [Skill2Real: Agentic Skill Learning for Zero-Shot Sim-to-Real Robot Manipulation](https://arxiv.org/abs/2610.02788v1)

**Authors:** Xincheng He, Siyu Ma, Chang Yu, Yunuo Chen, Yanjia Huang, Ying Nian Wu, Yin Yang, Chenfanfu Jiang

**Published:** 2026-10-02

**Categories:** cs.RO

**Abstract:**

Transferring robotic skills from simulation to reality requires task knowledge that remains usable across differences in perception, dynamics, and embodiment. We introduce Skill2Real, an agentic policy framework that learns executable skills through a shared application programming interface (API). A Proposer-Verifier-Governor (PVG) loop uses privileged simulation evidence to diagnose outcomes and validate updates, while keeping learned skills grounded in public observations and API semantics. The Cerebellum first acquires local manipulation skills; the Brain then learns task-level composition with the Cerebellum frozen. Both memories transfer to the real robot without task-policy fine-tuning or skill-memory updates. As GPT-5.6 Sol learns skills on LIBERO-90, evaluating each frozen checkpoint with GPT-6 Astra raises LIBERO-Pro Long success from 2.0% to 56.3%, without training on Pro Long. Independent Robosuite training reaches 85.1% and 89.4% mean success with Sol and Opus 5 across seven tasks, respectively. Frozen Sol-trained LIBERO-90 skills achieve 78.75% mean completion across four real-world manipulation tasks with Astra. Removing the Verifier or Governor during LIBERO-90 training lowers final Pro Long success by 17.3 and 13.3 percentage points, respectively. These results support learning and transferring a hierarchy of executable skills through a common robot interface.

### 论文解读

#### 摘要翻译
仿真到真实的机器人迁移通常通过调整策略或表征来适应外观和动力学差异，但这可能把学习预算花在低层变化上，难以形成可复用的任务知识。Skill2Real以共享代码接口承载可执行程序，在仿真中学习局部操作技能及其长程组合。训练时，Proposer根据公开观察执行程序，Verifier可以查看仿真特权信息并给出公共可理解的反馈，Governor只接纳经后续验证支持的技能更新。部署时技能库被冻结，机器人通过同一语义接口执行，不做实机任务微调。

#### 方法动机分析
长程操作既要抓稳、放准，也要决定先后顺序并处理失败。现有方法的瓶颈是：单纯迁移低层策略难以复用任务层面的组织方式，预设代码原语又不会从交互反馈中持续学到技能。论文的关键设定是把“如何完成局部操作”和“如何组合任务”分开学习，并让仿真特权只用于训练反馈，不进入部署时的策略输入。跨域复用依赖仿真与实机后端提供语义一致的API。

#### 方法设计详解
整个流程使用三个训练角色。Proposer通过图像和API返回生成程序、操作机器人并提出候选技能修改。仿真评估器检查执行结果，Verifier可查看特权状态，但必须把诊断转写为公共观察和接口语义，不能把隐藏坐标直接交给Proposer。Proposer据此提出候选更新；Governor再根据后续验证rollout中的成功、失败和回归决定是否写入记忆。

技能分成两个层级。Stage I训练Cerebellum记忆，记录抓取、放置等局部交互方法及API代码模板。Stage II冻结Cerebellum，再训练Brain记忆，学习任务目标、步骤顺序、技能选择和恢复策略。实机执行时两个记忆均固定，Proposer根据当前图像重新绑定物体和位置；相机感知、标定和关节控制由实机后端负责。

#### 方法对比分析
传统域随机化通常调整策略或特征以适应仿真差异；Skill2Real传递的是由公共接口表达的程序与技能记忆。相较固定原语组合，它加入验证反馈和更新准入；相较端到端长程策略，它把局部操作和任务编排拆开。与只迁移策略参数或预设原语相比，其创新是通过验证准入积累两层可执行记忆；区别在于任务知识可组合，但依赖仿真与实机API语义一致。因此“零样本”指技能库不再训练，并不表示不需要机器人后端适配。

#### 实验分析（精简版）
作者在LIBERO-90、Robosuite、未见LIBERO-Pro Long和UR5e上评估。实机使用一台UR5e、Pika夹爪及场景和腕部RGB-D相机；四类任务每种方法各20次。完整层级在抓放、分类整理、算式拼装、抽屉操作上的完成率为95%、85%、85%、50%，均值78.75%；无学习技能的Astra基线为27.50%，仅Cerebellum为56.25%。未见LIBERO-Pro Long上，Sol训练的冻结层级由Astra评测为56.3%，由Opus 5评测为49.0%。这些是固定技能库的点估计，论文说明它们不反映独立训练运行之间的方差；位置扰动也仍明显困难。

#### 实用指南
方法需要可调用的图像、几何测量、动作和结果检查接口。论文附录特别强调，命令返回成功不等于物体已被抓住或任务关系已满足，执行后应重新观察。迁移时需要为目标机器人实现相机、标定和低层控制，并保持API语义一致；再分别验证局部技能和任务排序。训练时LIBERO每任务每轮使用5个种子rollout，候选更新再经过25次验证rollout；实机推理以冻结技能和新观察执行。完整复现还需对照模型和接口版本。

#### 总结
核心思想：分层学习并验证可执行技能
1. 仿真中从公共接口执行并诊断结果。
2. 先学习局部技能，再学习任务组合。
3. 验证后冻结记忆，通过实机后端执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02788v1)
- [arXiv](https://arxiv.org/abs/2610.02788v1)

---

<a id='2610.02832v1'></a>
## [FastOPD: On-Policy Distillation for Lightweight VLA Deployment](https://arxiv.org/abs/2610.02832v1)

**Authors:** Yoojin Oh, Jeongsol Kim, Yeonwoo Seo, Jangho Park, Seonghyun Jin, Sunwoo Park, Youngmin Kim, Youngjun Jun, Kyumin Choi, Jong Chul Ye

**Published:** 2026-10-02

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Vision-Language-Action (VLA) foundation models have scaled rapidly to enhance manipulation performance and generalizability, but this scaling incurs high computational costs that render real-world deployment increasingly challenging. Existing approaches typically mitigate this issue by designing smaller architectures or reducing the iterative denoising steps in flow-based policies. In this work, we propose FastOPD, a foundation-to-lightweight VLA framework that enables the practical deployment of large-scale VLAs through efficient on-policy distillation. Specifically, FastOPD adapts a flow map for single-state teacher supervision and combines it with a self-consistency objective to construct a compact student that learns the teacher dynamics. Furthermore, we theoretically demonstrate that minimizing this objective allows the distilled student to recover a distribution on par with that induced by an ideal few-step teacher model. We evaluate FastOPD across diverse foundation policies in simulation and real-world experiments. On LIBERO, FastOPD retains 84% of the performance of $π_{0.5}$ with only two inference steps, reducing inference latency by 78.1% while outperforming existing few-step distillation baselines in average success rate. With LingBot-VLA as the teacher, FastOPD improves the single-step success rate over the base student by 15.9 percentage points on RoboTwin 2.0. We further demonstrate its applicability to a World Action Model (WAM) and deploy a compact student distilled from MolmoAct2 on a real robot.

### 论文解读

#### 摘要翻译
视觉语言动作模型的规模快速增长，提升了操作表现与泛化能力，但也带来高计算成本，阻碍真实部署。已有方法常通过缩小架构或减少基于流策略的迭代去噪步数来降低成本。本文提出FastOPD，以高效的on-policy蒸馏将大规模策略转成轻量VLA。方法把流映射改造成单状态教师监督形式，并结合自一致性目标，使学生学习教师动力学。作者进一步证明，在一定条件下最小化该目标可使蒸馏学生恢复接近理想少步教师的动作分布。实验覆盖多种仿真基础策略和真实机器人：LIBERO上两步FastOPD保留π0.5教师84%的表现、推理延迟减少78.1%；以LingBot-VLA为教师时，RoboTwin单步成功率比基础学生提高15.9个百分点；另展示了WAM蒸馏及MolmoAct2真机部署。

#### 方法动机分析
动机来自基础VLA常以流或扩散过程逐步生成动作，每一步都需运行动作网络；计算瓶颈限制了实时控制。直接减少步数可能使学生偏离教师轨迹；逐步模仿又会反复调用昂贵教师。FastOPD试图同时减少教师查询与学生推理步数：在学生真实访问的状态上取得教师方向信息，再以区间一致性约束连接少量采样步。有效性仍取决于学生状态覆盖及任务分布。

#### 方法设计详解
整体流程分为学生初始化、on-policy速度匹配和区间一致性训练三个步骤。学生是451M参数SmolVLA。视觉语言主干冻结，动作专家和时间投影参与训练。流映射写作fθ(x,s,t)=x+(t−s)uθ(x,s,t)，uθ为速度场。OPFD在一个on-policy中间状态上仅查询一次教师速度，直接监督学生动力学；有限区间自一致性（LSC）使用中点关系约束学生在两个时间点的流映射，使区间预测保持一致。优化先预热流策略，再蒸馏。论文还给出Wasserstein终端分布界，推导使用速度场Lipschitz和rollout密度比受控等条件；该理论界有明确适用条件。

#### 方法对比分析
缩小VLA结构能降低每次网络运行成本，但可能牺牲能力；少步采样降低调用次数，却可能引入较大离散误差。FastOPD保留紧凑学生，并通过教师动力学监督和区间自一致性减少误差。与逐步教师监督相比，其目标是用一次状态查询支持多个少步动作；与只做轨迹回归相比，它直接约束流速度和时间区间关系。方法优势是无需在部署时运行大教师，代价是训练仍需要教师查询，且学生表现可能随推理步数改变。

#### 实验分析（精简版）
LIBERO含40项任务、每项50次试验。两步FastOPD平均成功率81.8%，初始SmolVLA为71.4%，π0.5教师97.5%；两步延迟66毫秒，基础学生十步为193毫秒。RoboTwin 2.0的50项双臂任务中，LingBot教师蒸馏学生单步成功率比基础学生高15.9个百分点；两步结果57.7%，延迟74毫秒，教师延迟197毫秒。不过四步和十步学生在该设置低于基础学生，收益并非单调。Fast-WAM实验中，451M单步策略成功率50.5%，基础学生35.3%。真机YAM双臂平台仅评测pnp-plate 50回合：四步FastOPD成功率50%，基线四步42%、十步44%；成功回合耗时17.38秒，对照19.32秒。真实训练为10,000步预热后蒸馏20,000步，批量64、学习率2×10⁻⁵，推理设备RTX4090；仿真延迟另在RTX3090测量。单任务、单平台数据不足以确立广泛真机泛化。

#### 实用指南
部署时固定动作块长度、采样步数和计时边界，分报网络延迟与完整控制周期，并按任务记录失败类型。论文所述真机结论来自一个任务，理论保证也有正则性前提；迁移到不同机械臂、相机或动作分布时，需要额外演示和闭环安全测试。若只看平均成功率，可能忽略某些推理步数下学生退化。

#### 总结
核心思想：单点监督加区间蒸馏
1. 初始化紧凑学生并预热流策略。
2. 用学生状态上的教师速度和区间自一致性训练。
3. 以少步推理部署，并逐任务评估延迟与成功率。
LIBERO和RoboTwin显示了延迟与成功率的潜力，真机也完成了初步单任务验证。但收益依赖推理步数，实际证据覆盖面有限；跨任务和设备的稳健性仍需实验确认。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02832v1)
- [arXiv](https://arxiv.org/abs/2610.02832v1)

---

<a id='2610.02759v1'></a>
## [Proprioceptive Sketches as Long-Horizon Intent for Generative Action Policies](https://arxiv.org/abs/2610.02759v1)

**Authors:** Fangyuan Wang, Songhao Huang, Haoxiang Sun, Shipeng Lyu, Chengyang He, Anqing Duan, Peng Zhou, David Navarro-Alarcon

**Published:** 2026-10-02

**Categories:** cs.RO

**Abstract:**

Generative robot policies predict short action chunks but lack explicit long-horizon intent. Recent methods expose longer-horizon structure through language plans, subgoal images, or video forecasts, which are costly to generate and still need to be translated into robot motion. Predicting future robot motions avoids this translation, but a dense, time-indexed trajectory requires numerous parameters to cover the full remaining task, and over a short horizon it largely repeats the action chunk and adds little guidance for action generation. We propose Proprioceptive Action Models (PAM), which jointly generate a compact, timing-free sketch of the robot's remaining joint-space path and a dense executable action chunk within a single transformer denoiser. The sketch parameterizes the path by arc length rather than time, capturing geometric intent invariant to execution timing. Block-causal attention and a staggered denoising schedule maintain directed sketch-to-action dependence, ensuring the action tokens condition on a progressively cleaner sketch throughout sampling. In simulation, PAM improves over its action-only counterparts on Push-T and LIBERO-Long; on four real-world bimanual tasks, it raises success from 47.5% to 75.0%. Project page: https://nicehiro.github.io/pam_dp/

### 论文解读

#### 摘要翻译
生成式机器人策略通常输出短动作块，却缺乏长程意图。语言计划、子目标图像或视频预测可以补充规划信息，但生成成本高，还需映射为具体机器人运动。本文提出本体感受动作模型PAM，在一个Transformer去噪器中联合生成两种结果：覆盖任务余下关节路径的紧凑草图，以及短期可执行动作块。草图用弧长而非时间参数化，因此同一路径快慢不同也能保持几何表示；分块因果注意力和错开去噪让动作以草图为条件。在Push-T、LIBERO-Long和双臂真机四任务中，PAM普遍超过相应动作基线，真机总体成功率从47.5%升至75.0%。

#### 方法动机分析
问题在于长程任务需要意图，而传统动作块只覆盖眼前几步。直接生成整段时间轨迹会随任务长度变长；只生成很短的未来轨迹又与动作块重复。PAM的动机是找到固定长度、能代表剩余任务的中间量，并让它与动作由同一模型共同生成。草图来自演示中已有的关节状态，不额外需要语义标注。设计挑战是让计划信息影响动作，但避免反向动作噪声污染计划流。

#### 方法设计详解
每个时刻以当前关节状态为起点，累积后续关节变化长度，转成0到1的弧长相位。把剩余路径在相位上均匀重采样，再用端点约束三次B样条表示；首控制点为零，其他控制点数量由任务决定。由于表示不绑定时间，速度变化和停顿不改变理想草图。训练目标与动作块一同加噪，草图token和动作token有各自的噪声水平。

整体流程是先从噪声联合生成路径草图和动作块，再通过分块因果注意力保持单向依赖，最后以错开去噪让草图先变清晰、动作随后读取它。草图token按路径相位因果排列，动作token按时间排列；动作读全部草图和过去动作，草图只读较早草图。推理输出仅执行动作块，草图供条件化和查看。论文默认lead为1/8，增加了网络调用次数。

#### 方法对比分析
未来图像和视频需要从场景预测再映射成机器人命令；语言计划需要解释并转换为连续控制。PAM直接预测关节配置路径，监督来自记录的本体感受数据。与B-spline Policy不同，后者把样条作为可执行动作轨迹，PAM将样条作为任务余下路径的意图表示，同时另生成密集短动作。相较动作单流基线，草图提供更长程上下文，但带来额外token和去噪计算。

#### 实验分析（精简版）
Push-T图像设置中，覆盖率均值由Diffusion Policy的0.66升至0.72，最佳值由0.78升至0.82；状态设置均值0.79升至0.88。LIBERO-Long十任务上，PAM-VLA为95.2±0.4%，FLOWER为94.9±1.2%，两者差距有限。双UR3真机四任务各20次，PAM-DP成功60/80（75.0%），Diffusion Policy成功38/80（47.5%），B-spline Policy为35/80（43.8%）。真机成功回合平均42.3秒，基线41.4秒。每任务演示48至89条，8Hz控制，预测16步、执行8步，批量64、AdamW学习率1e-4。消融显示草图、单向注意力、弧长参数和分流噪声均有关联；但lead增大也提高NFE，收益不能与计算成本分开理解。真机样本量与平台范围有限。

#### 实用指南
复现应按任务核对草图控制点数、训练数据划分、观测相机和两臂关节定义。真机测试需固定初始物体分布，并报告失败首因、完整运行时延和控制频率。部署时可先检查草图是否符合任务路径，再评估动作依赖是否稳定；弧长不变性不覆盖路径反向或改变两臂相对时序。方法不建模接触或物体动力学，不能直接当成碰撞安全规划器。

#### 总结
核心思想：长程意图条件化动作
1. 将剩余关节路径转为弧长B样条草图。
2. 单向注意力让动作读取草图条件。
3. 只执行短动作块，并评估额外去噪成本。
真机结果有提升，但每任务20次、单一平台，且总体动作略慢。后续仍需跨平台和闭环任务验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02759v1)
- [arXiv](https://arxiv.org/abs/2610.02759v1)

---

<a id='2610.03013v1'></a>
## [RYOPO: Bringing End-to-End Category-Level Object Pose Estimation into Real Time](https://arxiv.org/abs/2610.03013v1)

**Authors:** Hakjin Lee, Junghoon Seo, Jaehoon Sim

**Published:** 2026-10-02

**Categories:** cs.CV, cs.RO

**Abstract:**

Category-level object pose estimation predicts the rotation, translation, and metric size of unseen instances within known categories. Many accurate RGB-D methods rely on external instance segmentation and crop-based pose estimation, introducing separate stages and object-dependent processing costs that hinder real-time inference. To bring accurate pose estimation into real time, we present \ours{}, an end-to-end trainable query-based RGB-D set predictor. It jointly detects and segments objects and estimates their \mbox{9-DoF} poses without explicit CAD-derived shape priors or a separately trained instance segmentor. Shared image and scene encoding avoids repeated per-object crop encoding. A query-conditioned geometry pathway associates observed 3D points and RGB features with object queries and incorporates shared scene context. Object-centric refinement uses the resulting point descriptors to update an explicit pose state through pose-conditioned cross-attention and recurrent residual corrections. On NOCS, \ours{} substantially improves on published RGB-D joint detection and pose estimation results. It achieves competitive performance compared with two-stage methods under all-object evaluation on REAL275 and HouseCat6D, while enabling real-time full-frame pose estimation at $31.8$ FPS on an RTX~A6000. Project page: https://yopo-series.github.io/RYOPO-project-page/.

### 论文解读

#### 摘要翻译
类别级物体位姿估计要预测已知类别中未见实例的旋转、平移和尺寸。许多RGB-D方法依赖外部实例分割及逐对象裁剪估计，阶段多且对象数量会增加推理成本。RYOPO提出端到端、基于查询的RGB-D集合预测器，联合检测、分割和9自由度位姿估计，无需显式CAD形状先验或独立实例分割器。它共享图像与场景编码，以查询条件几何模块关联观测点和对象查询，再通过当前位姿条件化交叉注意力反复修正位姿。在NOCS上超过已发表的RGB-D联合检测位姿方法；在REAL275和HouseCat6D所有标注对象评估中接近两阶段方法，并在RTX A6000达到约31.8帧/秒的全帧位姿输出。

#### 方法动机分析
传统逐对象流程需要先分割，再逐个裁剪RGB和深度图做位姿网络，延迟随对象数上升。直接把RGB和深度早期拼接不能保证测得的三维几何真正参与位姿修正。RYOPO的设计挑战是同时获得共享全景计算和对象特定的三维观测，并让估计出的当前位姿继续约束后续修正。

#### 方法设计详解
输入是对齐RGB图、深度图和相机内参。DETR式查询先预测类别、框、掩码与粗位姿。每个框内构造8×8采样格，深度反投影得到点；掩码置信、深度有效标记和局部深度一致性用于抑制背景与不一致表面。聚合点带有相对几何、相机坐标、RGB外观及支持度。共享稀疏3D编码器每图只运行一次，提供查询之外的场景上下文。

完整流程分三阶段：先以查询条件几何提取每个对象的点描述，再由对象交叉注意力和场景交叉注意力反馈到下一层查询，最后在集合解码后冻结点描述并运行对象中心位姿精修。精修器把当前旋转、平移和尺寸编码到注意力查询中，结合点描述预测位姿残差；旋转复合更新，平移按当前尺寸缩放，尺寸在对数空间更新。它在前向传播中完成，不做推理时梯度优化。模型使用150查询、四层检测解码、三层位姿精修，每对象64点，5毫米场景体素。

#### 方法对比分析
与依赖Mask R-CNN和逐对象裁剪的方法相比，RYOPO在全帧共享图像和场景编码，避免重复运行裁剪编码器。相较RGB-only的YOPO，它引入真实深度点与位姿状态条件注意力；相较只做早期RGB-D融合的模型，它把对象点描述明确送入查询反馈和位姿精修。代价是需要深度与相机内参，并维护额外点云/稀疏场景计算；实时推理并不等同于端到端机器人闭环实时。

#### 实验分析（精简版）
NOCS CAMERA25上5°/5cm AP 85.7、10°/5cm 92.5；REAL275分别52.5和76.8，联合模型比较中表现突出。逐对象管线比较时采用所有标注对象评估：REAL275上RYOPO为51.1，CleanPose 55.1，AG-Pose 52.7；HouseCat6D共用检测掩码时为21.5，CleanPose 21.1，AG-Pose 17.7。不同指标和协议不能混排。RTX A6000、batch1实测31.4毫秒每帧，约31.8帧/秒，包含场景编码与对象精修；逐对象方案开销随候选数增加。消融中，去掉全部几何反馈后REAL275的5°/5cm AP由51.1降至37.6；去掉位姿精修为44.2。训练使用640×640输入，AdamW、有效batch72；NOCS 12 epoch、HouseCat6D 18 epoch，结果平均三次运行。

#### 实用指南
复现前检查RGB-D对齐、内参、深度有效掩码及位姿对称性协议；固定查询数、点采样网格和场景体素尺寸。计时要区分输入解码、相机对齐、网络推理和后处理，论文的A6000数字不包括采集和深度对齐。若用于抓取，应额外测闭环成功率和位姿误差传播。方法在紧致2厘米平移阈值下仍落后部分逐对象模型，也没有跨传感器和机器人操作验证。

#### 总结
核心思想：共享场景、查询几何、位姿反馈
1. 查询从对齐RGB-D中采集对象点描述。
2. 对象和场景几何反馈更新检测查询。
3. 当前位姿条件化残差精修，并一次处理全帧。
RYOPO把类别级位姿推理带到实时全帧计算，但速度测量尚未证明机器人闭环性能，且严格精度仍有差距。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.03013v1)
- [arXiv](https://arxiv.org/abs/2610.03013v1)

---

<a id='2610.02898v1'></a>
## [MixVLA: Adaptive Mixing of Non-Invariant Information for Generalizable Vision-Language-Action Models](https://arxiv.org/abs/2610.02898v1)

**Authors:** Pingrui Zhang, Yu Zhang, Pengyuan Wu, Bin Wang, Haoming Song, Xianqiang Gao,  ZhaxiZhuoma, Zhigang Wang, Dong Wang, Bin Zhao, Xuelong Li

**Published:** 2026-10-02

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models have achieved remarkable advances in robotic manipulation, yet their zero-shot generalization under out-of-distribution (OOD) conditions remains limited. These models often entangle task-relevant invariant structure with environment-specific non-invariant factors, causing policies to rely on spurious appearance cues during action prediction. In this work, we propose \textbf{MixVLA}, a model-agnostic training framework that improves the generalization of VLA models without requiring additional OOD data or architectural modifications. The key component of MixVLA is \textbf{Adaptive Mixing of Non-Invariant Information (AMI)}. AMI stochastically mixes non-invariant representations to regularize distribution-specific variability while preserving complementary predictive cues. The mixed non-invariant features are then fused with invariant representations for final action prediction, resulting in improved robustness without sacrificing policy expressiveness. Extensive experiments across challenging manipulation settings, including LIBERO, LIBERO-Plus, the RoboTwin perturbation suite, and real-world tasks, demonstrate that MixVLA improves overall zero-shot robustness while retaining strong in-domain performance.

### 论文解读

#### 摘要翻译
视觉语言动作模型在分布内操作任务表现良好，但遇到新光照、背景、纹理或视角时，零样本泛化仍不足。训练表示会同时编码任务相关的稳定信息，以及随环境改变的非不变因素。直接依赖后者可能产生伪相关；强制完全不变又会丢掉有用线索。MixVLA提出不改架构、不增加OOD数据的训练框架：先用信息瓶颈学习不变表示，再对非不变表示进行随机自适应混合，最后融合两类信息预测动作。论文在LIBERO、LIBERO-Plus、RoboTwin和Franka真实机器人上报告了稳健性提升。

#### 方法动机分析
驱动问题是VLA容易记住训练环境的背景和视角，遇到轻微视觉变化就失效。仅增加数据成本高，针对单一扰动定制模块也难覆盖复杂组合变化。作者提出保留表示中可预测动作的环境相关线索，但通过跨样本随机混合压低其环境特定波动。这里的“非不变分支”是候选互补信息，不能据名称理解为经过严格因果分离的纯干扰因素。

#### 方法设计详解
第一阶段分别训练Normal VLA与Invariant VLA。Normal使用原有动作损失；Invariant在动作损失之外加入互信息压缩惩罚，使用神经互信息估计器，期望保留动作预测力并压缩输入冗余。随后按权重相减构造Non-Invariant Net。作者把这一步视作经验性互补分支，而非精确分解。

第二阶段在小批量内随机为每个样本配对另一个样本的非不变embedding。软混合以自适应归一化处理：在标准化源特征上重建插值后的均值和方差；硬混合则直接线性插值两条embedding。混合结果与Invariant Net的表示拼接，送到原策略头，并用基础动作目标继续训练。完整流程是训练基准策略、训练瓶颈表示并相减得到互补分支、最后混合互补特征并融合预测。理论说明混合可收缩某些统计量的跨样本协方差；回归模型中的OOD优势只在给定噪声区间等假设下成立。

#### 方法对比分析
数据扩增通过更多演示覆盖环境，成本较高；专用模块可针对相机或几何问题，但限制跨架构使用。MixVLA在潜变量层做训练正则化，保留原模型动作头和调用接口。与仅使用IB不变表示相比，它保留互补信息；与直接ERM相比，它试图控制环境相关特征的波动。代价是需训练多个分支并调节信息瓶颈估计器、混合类型和概率，不能视作免成本适配。

#### 实验分析（精简版）
OpenVLA-OFT在LIBERO-Plus七类扰动总分76.2%，基线69.6%；相机扰动为49.6%，仍是弱项。LIBERO干净平均成功率96.7%，基线97.1%。π0.5在RoboTwin干净到随机化测试中从46.0%升至62.4%，干净测试从70.7%升至75.6%。真实Franka面包放置任务使用80条演示，两种灯光下每策略共10回合，MixVLA成功率60%，π0.5为20%；高频彩色闪光下分别1/5和0/5，样本有限。训练时Normal和Invariant各100,000步，AMI在LIBERO 1,000步、RoboTwin 10,000步；OpenVLA用8张A100，AMI批量48，π0.5全量微调、批量64。单次训练轨迹版本Self-MixVLA在RoboTwin随机化成功率60.8%，完整模型62.4%。

#### 实用指南
迁移时应保持干净基准测试，并按相机、光照、背景、语言等扰动分别测量。调节IB权重和混合概率时需留出验证分布；补充实验显示性能随混合概率非单调。OpenVLA和π0.5采用不同互信息估计器及微调策略，复现需核对这些实现细节。若部署场景存在显著相机视角变化，论文结果提示仅混合潜变量可能不足，应补充空间信息或多视角评测。

#### 总结
核心思想：混合环境相关潜变量
1. 用信息瓶颈提取任务稳定表示。
2. 权重相减构造候选互补分支，并在批内混合。
3. 融合两类特征预测动作并逐类评估OOD表现。
仿真稳健性提升明显，真机只有十回合单任务初步证据；高几何变化和混合参数敏感性仍需研究。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02898v1)
- [arXiv](https://arxiv.org/abs/2610.02898v1)

---

<a id='2610.02840v1'></a>
## [PointWAM: 3D World Action Modeling for Dexterous Robotic Manipulation](https://arxiv.org/abs/2610.02840v1)

**Authors:** Chunghyun Park, Beomjun Kim, Seungcheol Park, Heeseung Kwon, Yashu Shukla, Seunghoon Sim, Jinwoo Shin, Minsu Cho

**Published:** 2026-10-02

**Categories:** cs.RO, cs.CV

**Abstract:**

World action models jointly learn to forecast world dynamics and predict robot actions, such that the learned internal world dynamics guide accurate actions. Existing approaches typically represent the world as RGB frames or latent counterparts while predicting actions as end-effector poses or joint angles, but they often struggle to capture the 3D spatial structure and contact geometry central to dexterous manipulation. We introduce Point World Action Model (PointWAM), a 3D world action model that decomposes the world into a scene (i.e., environment) and hands (i.e., actor), and jointly forecasts both as 3D point trajectories within a shared space-time coordinate frame. This explicit, disentangled representation enables effective pre-training on large-scale human demonstration videos without requiring any task-specific object or keypoint selection. Given a colored point cloud and a language instruction, PointWAM predicts how the scene and hands co-evolve in 3D space over time, then retargets the forecast hand motion to robot actions. Pre-training on human videos improves average DexJoCo success by 56.9 percentage points, and scene-trajectory supervision adds 10.9 points over forecasting the hands alone. With both, PointWAM surpasses the prior state of the art on ten DexJoCo tasks by 11.7 points and outperforms strong VLAs on a real robot.

### 论文解读

#### 摘要翻译
世界动作模型尝试同时预测世界变化和引发变化的机器人动作。现有方法常用RGB帧或潜在视觉表示世界，而用末端位姿或关节角表达动作，不易刻画灵巧操作中的三维结构与接触几何。PointWAM将环境表示为场景点，将操作者表示为手部关键点，在同一时空坐标系共同预测两类3D点轨迹，并将预测手轨迹重定向为机器人动作。统一表示让模型能从人类演示视频中学习，不需要按任务选目标物体或场景点。论文报告DexJoCo十任务、RoboDojo精密操作和OpenArm双手实机结果。

#### 方法动机分析
指尖接触、物体表面及操作工具需要精确的3D空间结构。视频生成WAM预测像素或潜变量，机器人动作又是另一种输出；场景与动作的对应关系不直接。点策略可以预测手的3D轨迹，但通常不预测环境如何随手运动，也依赖外部筛选目标物体。因此作者的核心动机是让世界变化和动作中间量共享监督，缓解现有表示难以对齐接触几何的瓶颈，再把人类演示转为可迁移训练数据。

#### 方法设计详解
输入为彩色RGB-D点云、指令、机器人状态。场景点通过预训练Mosaic3D提取几何和语义特征；每只手以十个对应关键点表示。所有点映射到一厘米体素格，经稀疏卷积和五体素patch token化，再与冻结文本编码器的指令token输入Volt Transformer。解码后恢复逐点特征，两个输出头分别预测所有场景点和手关键点在未来时刻的3D位移。场景头监督世界模型，retargeter不直接使用场景预测。

动作重定向把预测的手轨迹与当前机器人状态编码为上下文，再由Transformer decoder预测动作块。整体流程先对齐人手与机器人手的关键点语义，再联合预测场景/手轨迹，最后把手轨迹映射成关节与末端动作。人类视频中的手姿来自EgoDex/VITRA，场景轨迹来自CoTracker3二维跟踪及VGGT-Ω深度；机器人侧用正向运动学和仿真回放构造监督。先以人类和机器人轨迹预训练，再以机器人示教联合微调轨迹预测器与动作重定向器。

#### 方法对比分析
相较图像/潜在视频WAM，PointWAM直接预测场景和手的3D点运动；相较只预测手轨迹的点策略，它为全部场景点增加未来运动监督；相较依赖目标选择的点云策略，它不要求按任务筛选场景点。Scene轨迹帮助共享骨干学习物体如何随手移动，但Retargeter仍是从机器人动作监督中学习的映射，没有显式逆运动学、碰撞检测或关节限位求解。

#### 实验分析（精简版）
DexJoCo十项多任务策略平均成功率69.0±2.0%，GR00T N1.6为57.3±1.8%，π0.5为50.0±1.9%；PointWAM在8/10项最好。人类视频预训练将从头模型12.1%提升至60.1%，使用1.15M片段为69.0%。RoboDojo-Precision上部分进度8.8、成功率4.8%，略高于GR00T的6.2和2.3%，但只训练和测试一个种子。真实OpenArm任务中，未见箱放置成功率58.3%，对照16.7%和25.0%；双手方孔插入50.0%，对照16.7%和25.0%，每项每方法24次。主训练使用1.15M人类片段预训练60,000步、DexJoCo再训练60,000步，4张B200、batch192、学习率1e-4。消融中移除场景轨迹监督使成功率64.4%降至53.5%。

#### 实用指南
复现需重建RGB-D对齐、深度尺度、手部十关键点对应、场景轨迹和机器人坐标系，并固定机器人演示切分。真实部署前评估相机遮挡、点云稀疏和透明物体，单独记录动作重定向延迟及安全约束。基准结果依赖人类轨迹估计与冻结3D编码器；RoboDojo为单种子评估。论文方法能提供预测轨迹，但不等于可直接用于碰撞安全的运动规划器。

#### 总结
核心思想：场景和双手共同预测
1. 将场景点与语义手点放入同一3D坐标。
2. 联合预测两类点的未来轨迹。
3. 将手轨迹重定向为动作，并用人类视频预训练。
结果显示3D场景监督和人类预训练有效，但数据重建、计算成本和安全动作约束仍是部署难点。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02840v1)
- [arXiv](https://arxiv.org/abs/2610.02840v1)

---

<a id='2610.02813v1'></a>
## [Register-Routed Delayed Fusion: Rewiring Shortcut-Prone Observation Fusion in Visuomotor Imitation](https://arxiv.org/abs/2610.02813v1)

**Authors:** Jieting Long, Weidong Cai, Weiming Zhi

**Published:** 2026-10-02

**Categories:** cs.RO

**Abstract:**

Visuomotor imitation policies combine high-dimensional visual observations with compact signals such as proprioception, and their fusion topology determines when and through which tokens these streams interact. In dense token fusion, visual tokens may attend directly to compact tokens from the first encoder layer, allowing action-predictive compact cues to influence spatial visual representations early in their formation. We ask whether controlling this route improves visual responsiveness and policy behavior. We introduce Register-Routed Delayed Fusion (RRDF), which masks direct compact-visual attention and stages cross-modal interaction through a learned register workspace. Its isolate-collect-route schedule protects an early stream-separated prefix and later permits only register-mediated exchange, while compact conditioning remains available to the native action generator. Across five simulation tasks and three real-robot tasks, RRDF matches or improves dense ACT under nominal conditions. Appearance-shift evaluations on four simulation tasks and held-out-position evaluations on three real-robot tasks also favor RRDF. Phase-matched input probes show lower measured state-to-image sensitivity, while ablations indicate that adding registers alone does not reproduce the full performance gain. These results support controlling cross-modal propagation while retaining compact action conditioning.

### 论文解读

#### 摘要翻译
视觉模仿策略需要融合图像与本体感觉等紧凑信号。若视觉token从编码器第一层便能直接读取状态，动作预测性强的状态信息可能提前写入视觉空间表示。作者提出寄存器路由延迟融合（RRDF）：屏蔽紧凑token与视觉token的直接注意力，通过学习型寄存器逐步交换信息，同时保留状态对动作生成器的条件作用。仿真和真机结果显示，RRDF在论文测试的名义条件与多种外观、位置变化下整体优于密集融合基线。

#### 方法动机分析
本体状态能帮助动作预测，但若模型过度依赖它，视觉变化就可能较少参与空间表征。常规融合把信息通路交给全连接自注意力，无法控制跨模态影响何时发生。现有方法的痛点是无法控制状态何时影响视觉；RRDF的核心假设是延后融合能缓解这一痛点。RRDF的重点不是删除状态，而是规定视觉和状态何时、通过什么中间表示交互。这样既保留动作控制需要的本体条件，也让视觉表征先形成一段不受状态直接改写的过程。

#### 方法设计详解
模型将输入划分为紧凑组C与视觉组V，并加入两组来源专属寄存器。Isolation阶段，紧凑侧寄存器只读取紧凑信号，视觉侧寄存器只读取图像，主token流彼此隔离；Collection阶段，寄存器可以汇集两侧信息；Routing阶段，视觉与紧凑token才可读取寄存器。C与V之间的直接注意力在整个编码器内保持屏蔽，因此跨模态信息必须经寄存器中转。四层ACT默认采用一层隔离、一层收集、两层路由，仿真使用两个寄存器，真机使用四个。推理时状态仍经原动作生成接口作为条件，编码器输出接口不变。

#### 方法对比分析
密集融合从开始就允许视觉和状态token直接交换信息，结构简单，但难以指定融合时机。单纯添加寄存器而维持密集通路，只带来较小收益；延迟直接融合的无寄存器设置也有明显提升，表明受控融合时机本身重要。RRDF在此基础上使用来源分组和阶段化路由。相较丢弃本体信息的方法，它保留了动作条件；相较无约束融合，它限制状态直接渗入视觉表征。不同路由变体仍有相近成绩，说明目前结果尚不能证明完整三阶段设计在所有场景都必要。

#### 实验分析（精简版）
仿真涵盖Can、Square、Stack-D1、Coffee-D2和Push-T。最终十检查点平均值中，Square成功率为86.13%，密集ACT为73.53%；Stack-D1为73.73%对66.33%。四个任务的八种渲染变化条件均报告更高观测成功率。真机采用Piper-X，每任务24条演示、每个任务条件20次测试。Soap Pick-up、Plate Stacking和Toolbox在名义位置分别达到100%、100%、85%，ACT为55%、65%、50%；未见位置分别为95%、100%、20%，ACT为30%、30%、0%。Toolbox在位置变化下仍是难点。状态/图像替换试验显示RRDF降低状态相对图像对视觉表征和动作的影响，但作者也指出这类探针不直接识别捷径因果机制。

#### 实用指南
复现时应固定演示数据、训练设置和评估检查点，匹配视觉与状态替换的轨迹相位，并报告成功率之外的完成时间及误差。部署迁移需确认新增寄存器与注意力掩码兼容既有位置编码、历史帧和动作头；再按任务调节隔离、收集、路由深度。建议同时检查视觉token响应是否降低、图像变化是否仍能影响动作，以及闭环成功率是否提高，避免单凭敏感性指标选模型。小样本真机结果尤其需要更多重复验证。

#### 总结
核心思想：视觉状态分阶段经寄存器融合
1. 将输入分成视觉token与紧凑状态token，并屏蔽二者直接注意力。
2. 先分别收集来源信息，再合并寄存器工作区，随后开放寄存器读回。
3. 保留原动作生成器的状态条件，并用任务成功与成对介入评估收益。
RRDF为融合结构增加了明确的信息路由约束，结果显示其在指定基准上有帮助；真机样本量、未见位置表现和因果解释仍需更多验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.02813v1)
- [arXiv](https://arxiv.org/abs/2610.02813v1)

---

