time: 20260910

# Arxiv Computer Vision Papers - 2026-09-10

## Executive Summary

## 执行摘要

本期论文集中体现了机器人学习从“单一任务、理想仿真”走向真实世界泛化与可验证执行的趋势。研究主题覆盖长时程可变形物体操作、颗粒地形上的人形机器人运动、连续体机器人规划、全身动态行为学习，以及面向自动驾驶和工业协作的多模态感知。共同关注点是：如何利用结构化表示、物理先验、数据复用和不确定性感知，降低真实机器人部署的脆弱性。

其中，FolDeX 建立了面向长时程可变形物体操作的真实机器人基准，强调跨任务、场景、物体类别和机器人本体的数据复用；GTA-2 则将多 VLM 任务分解、任务轴技能构造、控制器参数分配和视觉落地串成可解释流水线，在 14 个真实操作任务上展示了零样本技能生成与定向修正的价值。颗粒地形人形运动工作通过三维阻力理论和教师—学生强化学习把接触物理纳入训练，代表了仿真到现实迁移的重要方向。

感知与规划方面，CLFTv2 以分层特征金字塔和轻量残差解码器提升相机—LiDAR融合效率，PccDiffuser 用扩散式多解生成处理连续体机器人运动规划，分别体现了实时感知和多模态规划的结构化设计。JEPA Policy、SwingBot、执行器动力学课程学习等工作进一步说明，未来机器人策略学习将同时重视未来表征、全身动力学和训练课程设计，而不是只追求单一策略网络的规模。

最值得优先精读的是 GTA-2、FolDeX、颗粒地形人形运动和 CLFTv2：它们分别对应可解释技能合成、真实世界基准、物理一致的迁移训练和高效自动驾驶感知。整体来看，跨 embodiment 数据复用、结构化任务/动作表示、物理感知 sim-to-real，以及带置信度的多模态融合，是下一阶段机器人系统研究的主要增长点。

---

## Table of Contents

1. [GTA-2: A Multi-VLM Framework for Synthesizing Robot Manipulation Skills via Grounded Task Axes](#2609.09808v1)
2. [CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids](#2609.09881v1)
3. [PccDiffuser: Multi-solution Motion Planning for Continuum Robots](#2609.09745v1)
4. [JEPA Policy: Diffusion-Free Imitation Learning via Paired Action and Future Representation Prediction](#2609.09630v1)
5. [Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot Learning](#2609.09492v1)

---

## Papers

<a id='2609.09808v1'></a>
## [GTA-2: A Multi-VLM Framework for Synthesizing Robot Manipulation Skills via Grounded Task Axes](https://arxiv.org/abs/2609.09808v1)

**Authors:** M. Yunus Seker, Shobhit Aggarwal, Ruwan Wickramarachchi, Jonathan Francis, Oliver Kroemer

**Published:** 2026-09-09

**Categories:** cs.RO

**Abstract:**

Robotic manipulation tasks are often decomposed into behaviors or skills. However, one often needs to predefine these behaviors for specific tasks or try to cover a wide range of tasks using generic skills. As a result, these behaviors can remain too coarse to expose the geometric, control, and scene-dependent decisions required for execution. We introduce Grounded Task Axes v2 (GTA-2), a modular multi-VLM framework that constructs executable, task-bespoke manipulation skills from reusable object-centric task-axis components. Rather than predicting actions end-to-end or composing fixed task-level primitives, GTA-2 represents each skill as semantic subtasks comprising task-relevant keypoints and axes, controller compositions, and scene-dependent parameters. Four specialized VLM agents separately decompose the task, construct an abstract task-axis skill, assign controller parameters, and ground the required visual features from RGB-D observations. This abstraction-to-grounding factorization enables zero-shot skill generation without task-specific robot demonstrations, policy training, or fine-tuning. It also keeps intermediate decisions explicit, allowing targeted human feedback to refine an incorrect stage while preserving correct components. We evaluate GTA-2 on 14 real-robot manipulation tasks against a VLA policy pi_{0.5} and two Code-as-Policies baselines using task-axis controllers or conventional robot primitives. GTA-2 achieves an average zero-shot success rate of 73.9%, exceeding the strongest baseline by 31.4 percentage points, while targeted refinement raises GTA-2's average success rate to 90.7%. Project page: https://gta2-project.github.io/

### 论文解读

#### 摘要翻译
GTA-2 是一个多视觉语言模型框架：给定自然语言指令、RGB-D 观测和控制器库，系统合成可执行的机器人操纵程序。它在亚原子层面用对象中心的几何特征与任务轴表示技能，先生成可迁移的抽象配方，再将其落地到当前场景。方法无需任务演示或微调，并支持人类对显式中间结果进行局部修正。

#### 方法动机分析
端到端 VLA 将感知、规划和动作隐式耦合，失败难诊断且容易受布局、光照影响；传统 Code-as-Policies 依赖粗粒度 primitive，又难表达接触方向、工具几何和持续约束。GTA-2 的关键假设是，显式的对象中心中间表示能够兼顾可解释性、跨场景复用和低层控制精度。

#### 方法设计详解
系统输入语言指令、RGB-D 图像和控制器库，输出机器人程序。任务分解器先生成有序子任务；技能生成器为每项任务生成 lifted skill recipe；参数设置器填写偏移、力、轨迹尺度和时长；视觉模块把“杯口中心”“表面法向”等抽象特征解析为三维点或向量。抽象控制器包含类型、控制参考、对象参考、任务轴、参数占位符和优先级。库中有位置对齐、轴向对齐、恒力、螺旋 DMP 和夹爪控制器。确定性编译器再把 grounded skill 转成底层脚本，并以零空间投影组合不同优先级的约束。四个 VLM 代理均使用 Gemini 3.1 Pro，推理为零样本。

#### 方法对比分析
GTA-2 的创新在于把 VLA 的像素到动作黑盒改成显式子任务、配方、参数和三维参考，便于定位失败；相较传统 CaP-Primitive，它不只调用位移或力控制等粗基元，而是围绕对象几何和任务轴组合亚原子控制。它尤其适合擦除、熨烫、扫地等接触密集任务，但能力仍受预定义控制器和几何特征覆盖范围限制。

#### 实验分析（精简版）
在 14 类真实机器人任务上，GTA-2 零样本成功率为 73.9%，高于 CaP-Primitive 的 42.5%；一次人类反馈后达到 86.2%，两次后达到 90.7%。接触密集任务中任务轴更能表达持续接触与工具约束，而 π0.5 几乎全部失败。实验也显示早期反馈主要修正技能结构，后期主要修正视觉落地。主要边界是单一 RGB-D 视角、执行前静态感知，以及仅覆盖有限任务类别。

#### 实用指南
复现平台为 UR5e、Robotiq 2F-85 夹爪和 ZED 2i 深度相机，需要实现任务轴控制器、对象关键点及法向/切线等三维特征，并接入 Gemini 3.1 Pro。应分别评估零样本与反馈后的成功率，并测试布局变化下的 recipe 复用。项目主页提供演示与设置信息；论文未明确说明完整代码、权重或控制器库是否公开。迁移到其他机械臂时需重做坐标标定、TCP、控制器接口和力参数。

#### 总结
核心思想：显式任务轴合成可迁移技能

速记 pipeline：
1. 语言拆解为有序子任务。
2. 组合对象几何驱动的任务轴控制器。
3. 视觉落地三维参考并填写参数。
4. 按优先级编译执行，再针对具体模块反馈迭代。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.09808v1)
- [arXiv](https://arxiv.org/abs/2609.09808v1)

---

<a id='2609.09881v1'></a>
## [CLFTv2: Efficient Camera-LiDAR Fusion for Semantic Segmentation via Hierarchical Feature Pyramids](https://arxiv.org/abs/2609.09881v1)

**Authors:** Toomas Tahves, Mauro Bellone, Raivo Sell

**Published:** 2026-09-09

**Categories:** cs.CV, cs.RO

**Abstract:**

Semantic segmentation for autonomous driving requires reliable detection of vulnerable road users (VRUs) despite heavy class imbalance. We introduce CLFTv2, a hierarchical camera-LiDAR fusion framework replacing global ViT attention with a Swin-based multi-scale encoder and a lightweight FPN-style residual decoder. Operating in the 2D perspective domain, CLFTv2 integrates multi-scale geometric cues through shifted-window attention and per-scale residual fusion, avoiding the computational overhead of query-matching decoders. Across three driving datasets, CLFTv2 consistently improves VRU recall. On ZOD, CLFTv2-Large achieves 53.5\% mIoU, improving pedestrian IoU from 35.5\% to 44.9\% over the prior CLFT model. On Waymo, CLFTv2 reaches 61.7\% mIoU. Additionally, a modality-isolation study suggests ViT's global receptive field yields stronger fusion gains only under dense LiDAR returns. Compared to a Swin-based Mask2Former adaptation, CLFTv2 requires 1.4$\times$ fewer GFLOPs and delivers 2.2$\times$ higher throughput, while achieving comparable overall accuracy. These results demonstrate that hierarchical local-attention fusion offers an efficient, scalable alternative to global-attention and query-based decoders for real-time on-vehicle perception in intelligent transportation systems. Source code is publicly available.

### 论文解读

#### 摘要翻译

CLFTv2面向自动驾驶中的语义分割，尤其关注行人等易受伤道路使用者（VRU）。它以Swin Transformer多尺度编码器和轻量残差解码器替换原CLFT的全局注意力，在二维透视域融合RGB与LiDAR信息；相比Mask2Former，计算量更低、吞吐更高，并在多个数据集上改善VRU识别。

#### 方法动机分析

VRU只占很少像素，却是驾驶安全的关键。相机提供纹理，LiDAR提供深度，二者互补，但高分辨率融合容易变慢。原CLFT的ViT全局注意力复杂度随分辨率二次增长，且缺少天然多尺度特征；Mask2Former的查询匹配也给闭集语义分割带来额外训练负担。这些就是作者要解决的主要痛点和效率瓶颈。作者的核心假设是：局部窗口注意力配合层次金字塔，能够以更低成本保留小目标所需的细节，再用残差结构稳健融合两种模态。

#### 方法设计详解

系统输入RGB图像和投影LiDAR的[X,Y,Z]三通道坐标图，输入尺寸为Tiny的256×256或Base/Large的384×384。两路数据分别进入共享权重的Siamese SwinV2，移位窗口注意力只在局部窗口内计算，并通过patch merging产生H/4、H/8、H/16、H/32四级特征。各级用1×1卷积统一到256维。解码器从粗到细处理：RGB和LiDAR特征先分别经过模态专属ResConv，与上一层融合特征上采样2倍后的上下文相加，再经共享ResConv得到当前尺度表示。每个ResConv由两个3×3卷积、ReLU和跳连组成。最细特征经过3×3卷积、BN、ReLU、1×1卷积分割头，再双线性插值4倍输出像素类别。训练采用按类别频率加权的交叉熵；A100 80GB上batch size为8、基础学习率8×10^-5，Waymo训练100轮，ZOD和ISEAuto训练200轮。

#### 方法对比分析

CLFTv2与原CLFT的本质区别，是从全局单尺度ViT转向局部窗口的层次特征金字塔，并以闭集分割所需的残差解码器替代查询匹配。它的创新不只是更换backbone：两路特征并非简单相加，而是每种模态先独立提炼，再融合跨尺度上下文；这也区别于Add或Average等基线融合。因而适合重视延迟、吞吐和小目标细节的相机—LiDAR透视分割场景；但局部窗口对长程关系的建模能力较弱。

#### 实验分析（精简版）

作者在ZOD、Waymo和ISEAuto上，与CLFT、MaskFormer、Mask2Former、DeepLabV3+比较，并测试窗口、模态和融合方式。ZOD上CLFTv2-Large的mIoU为53.54%，CLFT-Large为46.82%，Human IoU从35.5%提升到44.9%，说明新融合对关键前景类确有帮助。效率上，A100上的CLFTv2-Tiny为30.9 GFLOPs、22.0 ms和187 MB，Mask2Former-Tiny则为42.3 GFLOPs、48.4 ms和314 MB，延迟和显存优势都较明确。Waymo上CLFT-Large达到68.26%，高于CLFTv2-Large的61.69%，说明局部感受野并非在所有数据分布上都占优；ZOD还受到伪标签质量影响，边缘设备实测也尚未覆盖。

#### 实用指南

论文提供代码仓库：https://github.com/taltech-av/paper-tvt2026-clftv2，训练结果页面：https://app.visin.eu/projects/clftv2。复现需完成LiDAR投影、RGB与坐标图配准、类别频率加权，并按数据集设置训练轮数；评估结果采用前十个验证checkpoint的平均值。迁移到新数据集时需重做投影和类别权重并重新训练，窗口大小要在小目标效果与计算成本之间调节。低功耗边缘设备上的表现和完整依赖版本，论文未说明。

#### 总结

核心思想：分层残差金字塔高效融合双模态。

1. RGB与LiDAR分别编码并生成四级特征。
2. 模态专属残差模块提炼局部信息。
3. 从粗到细注入上采样上下文并融合。
4. 分割头恢复分辨率，输出像素类别。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.09881v1)
- [arXiv](https://arxiv.org/abs/2609.09881v1)

---

<a id='2609.09745v1'></a>
## [PccDiffuser: Multi-solution Motion Planning for Continuum Robots](https://arxiv.org/abs/2609.09745v1)

**Authors:** Ke Qiu, Sifan Chen, Si Wang, Rong Xiong, Yue Wang, Haojian Lu

**Published:** 2026-09-09

**Categories:** cs.RO

**Abstract:**

We present the PccDiffuser, a conditional diffusion framework for continuum robots that learns a multimodal distribution over complete configuration-space paths and samples multiple candidate solutions in parallel, which are subsequently converted into an executable trajectory by time allocation considering actuator constraints. Under the piecewise constant-curvature model, we use exponential co-ordinates to describe the robot kinematics, and use graph neural network to encode a variable number of environment obstacles. Analytical differential kinematics is incorporated in the denoising process to improve terminal accuracy and whole-body clearance. On a mixed test set comprising configuration space with zero to four obstacles, PccDiffuser achieved a success rate of 91\%. Compared with existing sampling- and optimisation-based benchmarks, it delivered both a higher success rate and greater computational efficiency, with the latter advantage becoming more substantial when sampling more candidate solutions. Experiments on a three-section tendon-driven continuum robot further demonstrate consecutive planning, multi-solution planning, and whole-body obstacle avoidance.

### 论文解读

#### 摘要翻译

PccDiffuser 面向连续体机器人的多解运动规划。机器人自由度高、冗余强，同一目标可能对应多种姿态和避障路径。方法将起始构型、目标末端位置及障碍物环境作为条件，从高斯噪声逐步生成多条候选轨迹，并用解析微分运动学提高末端精度、减少碰撞。

#### 方法动机分析

RRT/RRT* 通常一次只找到一条路径，想获得备选方案就要重复搜索；工作空间规划可能陷入局部最优，构型空间规划又缺少任务引导。作者的假设是路径分布本身具有多峰性，扩散模型适合同时表达多个解，但神经网络输出仍需运动学纠偏。论文当前主要针对静态环境和球形障碍物。

#### 方法设计详解

输入是起始构型、目标位置、障碍图和带噪路径；首先用分段恒定曲率模型描述机器人，并采用指数坐标，避免曲率参数在直线状态附近的奇异性。起始构型与目标位置由 MLP 模块编码；每个球形障碍物作为图节点，由 GNN 模块聚合环境信息，因此障碍物数量可以变化。条件 temporal U-Net 接收带噪的 H×6 路径和扩散时间步，预测噪声；推理从高斯噪声开始，用 DDIM 逐步去噪，并行得到多条候选输出。为修正生成模型的几何误差，作者计算末端误差和障碍排斥项：末端梯度由解析雅可比的转置乘位置误差得到。Post Correction 直接修正预测的干净路径，Guided Prediction 则通过自动微分修改噪声预测。前者在精度和速度间更平衡。论文报告 AdamW 训练 100k 次更新。

#### 方法对比分析

PccDiffuser 与 RRT 的本质区别是把重复在线搜索改为条件生成，一次产生多模态方案；与 APF 的对比显示，它先学习全局路径分布，再用局部几何梯度校正，而非仅沿势场移动。其创新在于把 GNN、非奇异构型表征和推理期运动学引导结合起来，共同解决环境变长、参数奇异及末端偏差问题。它适合需要快速备选路径的静态狭窄空间。

#### 实验分析（精简版）

实验使用 353k 条无障碍路径和 243k 条含 0–4 个障碍物的路径，对比构型空间/工作空间 RRT、RRT* 与 APF。在混合测试集上成功率达到 91.11%，高于最高约 70% 的基线；单条成功路径平均推理时间为 34.0 ms。消融显示 Post Correction 使末端误差约降低 45%，且速度优于更重的引导预测。优势是多解、快速和精确，局限是动态障碍与复杂几何泛化尚未充分验证。

#### 实用指南

论文提供 GitHub 代码（qiuke-qiuke/pcc_diffuser）。复现时要保持 PCC 指数坐标、路径长度 H、障碍图构造、DDIM 设置和碰撞距离定义一致，并按 0–4 个球障碍物生成训练数据。迁移到另一台机器人，需要重写运动学、雅可比和碰撞模型并重训；复杂障碍不能简单继续使用球节点。推理速度是在 GeForce RTX 5090 上报告的 34.0 ms，换硬件后应重新测量。

#### 总结

核心思想：扩散生成多解并用运动学纠偏

1. 指数坐标稳定表示机器人构型。
2. MLP 与 GNN 编码任务和障碍物。
3. U-Net 从噪声并行生成多条路径。
4. 雅可比梯度纠正末端并避障。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.09745v1)
- [arXiv](https://arxiv.org/abs/2609.09745v1)

---

<a id='2609.09630v1'></a>
## [JEPA Policy: Diffusion-Free Imitation Learning via Paired Action and Future Representation Prediction](https://arxiv.org/abs/2609.09630v1)

**Authors:** Jie Xu, Kangjin Yu, Ziyi Jin, Junjie Gao, Liqing Chen, Yixian Li, Shuai Tian, Zhongpu Xia

**Published:** 2026-09-09

**Categories:** cs.RO

**Abstract:**

Standard behavior cloning supervises actions without explicitly constraining the future representation paired with each demonstrated action chunk. We introduce JEPA Policy, a diffusion-free framework that uses the action chunk and its observed future representation as paired training targets. Action and future-representation tokens interact in a shared Transformer and are refined through two forward passes. Future prediction can therefore shape the representation used to generate actions. Dual-branch and gradient-routing controls attribute the gain to this shared topology rather than to an auxiliary prediction head alone. Across nine simulated tasks, JEPA Policy improves mean success over the action-only MIP baseline and outperforms Diffusion Policy under the evaluated configurations, while adding 0.29 ms to MIP's model latency. A five-task, 630-episode physical-robot study produces the same pooled ranking. Further audits find no complete representation collapse under action supervision and identify a task-conditioned failure-ranking signal in future-prediction error. These results support paired future-representation supervision as a practical approach to low-latency visuomotor imitation without iterative generative sampling.

### 论文解读
#### 摘要翻译
JEPA Policy 提出一种无扩散的机器人模仿学习框架：根据当前图像和本体感知，同时预测专家动作序列与未来观测的潜在表示。未来预测塑造动作策略的表征，使模型在保持低推理延迟的同时提升控制成功率，并可用未来一致性误差辅助在线故障检测。
#### 方法动机分析
扩散策略效果好却要反复去噪，实时控制延迟高；普通行为克隆只拟合动作，不关心动作会把机器人带向什么未来状态。MIP 虽把过程简化为两步，但仍只预测动作。本文的假设是，专家动作应由其预期未来结果来约束，因此未来表征监督能改善长程、精细操作。
#### 方法设计详解
训练输入为当前观测、长度为 H 的专家 action chunk，以及 k 步后的观测。ResNet-18 编码当前观测得到上下文，未来观测编码成停止梯度目标；动作 token 和未来 token 进入同一个 Transformer，在每层互相注意，而非使用两个独立分支。模型沿用 MIP 的两步预测：第一步从零向量产生粗预测，第二步训练时接收带噪目标、推理时接收第一步输出并细化。损失由动作 MSE 和按目标 RMS 归一化的未来表示 MSE 组成，权重自适应平衡。训练设置为 300k 更新、batch size 256、学习率 1e-4，视觉编码器从零训练。
#### 方法对比分析
相较 100 步扩散去噪，JEPA 用两步确定性 refinement；相较 MIP，它增加未来结果监督，但仍接近 action-only 的速度。创新点在于让两类 token 在同一 Transformer 内逐层交互，而不是把未来预测作为孤立辅助头。消融显示，若动作和未来预测分开，性能退化，说明这种共享拓扑才是主要增益来源。该方法尤其适合低延迟、动作后果较长的视觉操作；确定性未来回归对多模态结果仍有限。
#### 实验分析（精简版）
在 9 个仿真任务上，JEPA 平均成功率 83.0%，高于 MIP 的 77.4% 和 Diffusion Policy 的 75.1%。推理延迟为 13.2 ms（p95 14.5 ms），而 100-step 扩散为 439.5 ms。5 个真机任务、630 回合的平均成功率为 66.9%，高于 MIP 的 54.7%。未来一致性误差在 MugMug 失败检测上达到 AUROC 0.754，说明它不只是训练辅助量，也有部署价值；但部分分析使用单种子，且多模态未来仍是局限。
#### 实用指南
论文提供 JEPA-POLICY 代码仓库与项目主页。复现需保持两步推理、动作块长度、未来步长、目标归一化和从零训练设置一致，并核对仓库中的依赖、真机部署和评估脚本。迁移机器人时需替换观测及动作维度、重算归一化统计量并重新训练；部署可记录未来一致性误差作为漂移告警。
#### 总结
核心思想：用未来表征塑造快速策略。
1. 编码当前观测与未来目标。
2. 在共享 Transformer 中交互动作和未来 token。
3. 两步确定性输出动作块与未来表征。
4. 用双重损失训练，并以一致性误差监测失败。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.09630v1)
- [arXiv](https://arxiv.org/abs/2609.09630v1)

---

<a id='2609.09492v1'></a>
## [Actuator Dynamics Curricula for Narrow-Viability Tasks in Legged Robot Learning](https://arxiv.org/abs/2609.09492v1)

**Authors:** Kousheek Chakraborty, Chandan K. Rajendra, Ayham Alharbat, Abeje Y. Mersha

**Published:** 2026-09-08

**Categories:** cs.RO

**Abstract:**

Reinforcement learning has produced capable controllers across a broad range of legged-robot tasks, but a subset of these tasks fail to converge under standard training: those for which most exploration trajectories terminate before producing useful gradient signal. To address such tasks we introduce the \emph{Actuator Dynamics Curriculum}, a procedure that initializes joint stiffness at a high value and anneals it toward the system-identified value as completed episode lengths grow. Using a cart-pole system as a representative example, we show that higher closed-loop joint natural frequency under critical damping enlarges the viability kernel of the underlying Markov Decision Process, increasing the fraction of initial states from which the task is feasible. We validate the kernel monotonicity on the cart-pole and apply the curriculum to a quadrupedal-to-handstand transition on the Boston Dynamics Spot, a narrow-viability task where training under fixed identified stiffness plateaus at a policy that never completes the transition. The trained policy executes the transition in simulation across 10 seeds and transfers to hardware. More broadly, our results suggest that simulated actuator dynamics is a useful axis along which to design curricula for tasks in which exploration is bottlenecked by termination conditions rather than by reward signal.

### 论文解读

#### 摘要翻译
足式机器人强化学习常遇到“窄可行性任务”：多数探索轨迹在得到有用梯度前就摔倒终止。论文提出驱动器动力学课程（ADC），先用高关节刚度训练，再随策略存活能力提升退火到真实硬件刚度，改善探索并实现实机迁移。

#### 方法动机分析
倒立起跳、四足到手倒立等动态动作要求机器人穿过很窄的状态区域；真实执行器响应不足时，随机探索几乎无法存活。改变奖励或地形不一定能解决动力学瓶颈，模仿学习又依赖专家示范。ADC的假设是，提高闭环带宽能扩大有限时间内的可行域，让策略先学会“活下来”，再适应真实动力学。

#### 方法设计详解
作者先用CMA-ES在Spot实机上辨识真实刚度K*，仿真从更高刚度K0开始。45维本体感知输入进入[512,256,128]的ELU-MLP，输出12维关节位置增量，由PPO训练。课程进度由平滑平均情节长度驱动：β=clip((L̄−Lmin)/(Lmax−Lmin),0,1)，刚度K=K0+β(K*−K0)；同步令阻尼B=2√(KM)，保持临界阻尼，避免把阻尼变化混入课程效果。也就是说，早期策略得到更快的姿态纠正，后期则必须在真实带宽下保留已经学到的动作。仿真采用4096个并行环境、物理步长0.002秒、控制频率50 Hz；训练完成后固定在K*进入推理，策略零样本部署到Spot。

#### 方法对比分析
与直接在真实刚度训练相比，ADC先扩大探索可行域；与固定高刚度相比，它最终回到K*，因此保留迁移能力；与刚度随机化或按时间退火相比，episode-length课程由策略能力触发，避免过早进入困难动力学。它的主要创新是把执行器刚度/带宽而非奖励或环境地形作为课程轴。

#### 实验分析（精简版）
实验使用4096个并行IsaacLab环境，物理步长0.002秒、控制频率50 Hz，单次训练约1.5小时。ADC平均情节长度为975±12，而直接在K*训练仅385±47，按均值计算约提升153%；时间退火为751±62。10个随机种子策略均在草地、地毯和木板上完成实机过渡。理论只覆盖简化倒立摆，统一刚度和人工课程超参数仍是局限。

#### 实用指南
论文未明确提供该方法专属代码仓库；复现需准备IsaacLab、RSL-RL/PPO、可调PD执行器和Spot系统辨识。关键是复现K*辨识、刚度与临界阻尼联动、平均情节长度进度、早停规则及45维观测/12维动作。迁移到其他机器人时要重新辨识执行器和质量参数，并通常重新训练；还要重新检查接触模型、跌倒终止条件与动作安全边界，不能直接照搬K0、Lmin或Lmax。

#### 总结
核心思想：用执行器刚度拓宽探索

1. 辨识真实刚度K*；
2. 从更高K0开始，让策略获得生存梯度；
3. 按情节长度把刚度退火到K*并保持临界阻尼；
4. 在真实刚度下收敛后零样本部署。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.09492v1)
- [arXiv](https://arxiv.org/abs/2609.09492v1)

---

