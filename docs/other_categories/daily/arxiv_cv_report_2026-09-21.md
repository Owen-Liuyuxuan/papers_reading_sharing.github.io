time: 20260921

# Arxiv Computer Vision Papers - 2026-09-21

## Executive Summary

## 执行摘要

本期 10 篇论文呈现出明显的“世界模型—动作生成—具身执行”主线，同时覆盖 3D 感知、操作规划、救援辅助与多机器人定位。多篇工作将预测式表示学习、持久记忆和跨实体泛化引入机器人基础模型，目标是让模型从被动视觉表征转向可用于动作决策的统一状态空间。

其中，Adaptive World Memory 3D Foundation Model 面向可扩展的 3D 建图、定位与渲染；MT-WAM、AtomEgo、PSR 与 SkelWAM 分别从动作生成、机器人—自我整合、接触丰富操作和跨实体迁移推进 world-action model。AcousticDiffusion 将语义条件音频引导用于搜索救援，展示了多模态感知对复杂环境辅助的价值。PointLAM、SAM3 引导的持久记忆策略、NeuRIO 和 TRACE 则分别聚焦高效点云检测、细粒度操作、零样本 sim-to-real 多机器人里程计和未知环境覆盖路径规划。

值得优先精读的是 Adaptive World Memory、MT-WAM、AtomEgo、PSR 与 SkelWAM：它们共同体现了从感知表征到可执行策略的研究趋势。NeuRIO 与 TRACE 对实际部署中的多机器人协同和未知环境探索也具有较强工程价值。后续值得关注的方向包括：统一世界模型与动作模型、跨 embodiment 的零样本迁移、长期记忆驱动的精细操作，以及将声音、触觉和视觉共同用于安全可靠的闭环控制。

---

## Table of Contents

1. [Adaptive World Memory 3D Foundation Model for Scalable 3D Mapping, Localization, and Rendering](#2609.21502v1)
2. [MT-WAM: Reorienting the One-Pass Predictive Representation Toward Action Generation](#2609.21474v1)
3. [AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining](#2609.21461v1)
4. [PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation](#2609.21753v1)
5. [AcousticDiffusion: Semantically Conditioned Audio-Guided Diffusion Policy for Search-and-Rescue Assistance](#2609.21792v1)
6. [SkelWAM: A Skeleton-Guided World-Action Model for Zero-Shot Cross-Embodiment Manipulation](#2609.21983v1)
7. [PointLAM: Local Attentive Mamba for Efficient Point-based 3D Object Detection](#2609.21780v1)
8. [Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning](#2609.21621v1)
9. [NeuRIO: A Streaming Neural Estimator for Zero-Shot Sim-to-Real Multi-Robot Relative Inertial Odometry](#2609.21707v1)
10. [TRACE: Coverage Path Planning for Unknown Environments Using Hierarchical Coverage Tree](#2609.21777v1)

---

## Papers

<a id='2609.21502v1'></a>
## [Adaptive World Memory 3D Foundation Model for Scalable 3D Mapping, Localization, and Rendering](https://arxiv.org/abs/2609.21502v1)

**Authors:** Tianchen Deng, Guole Shen, Yilin Shen, Wenhua Wu, Yilin Fang, Ziqi Ma, Tianjun Zhang, Shenghai Yuan, Wolfram Burgard, Hesheng Wang

**Published:** 2026-09-18

**Categories:** cs.CV, cs.RO

**Abstract:**

Recent 3D foundation models enable generalizable geometric reasoning from RGB images but remain limited in persistent memory, scalability, and renderable scene modeling. We present a memory-centric 3D foundation model for scalable robotic localization, reconstruction, and Gaussian rendering. Its core is an adaptive world memory mechanism that combines transformer-based gated updates with test-time temporal-spatial regulation. Learned gates control recurrent memory propagation, while temporal state evolution and spatial observation-state consistency regulate token-wise updates and forgetting over long image sequences. To support large-scale mapping, we organize memory into local submaps and integrate progressive mapping and tracking, loop closure, and SL(4)-based global refinement to maintain local accuracy and global consistency. A Gaussian reconstruction head decodes memory-enhanced features into renderable primitives, unifying camera pose estimation, dense point-cloud reconstruction, and photorealistic rendering within a single model. Experiments on public benchmarks and self-collected datasets from diverse robotic platforms demonstrate improved trajectory accuracy, reconstruction completeness, and rendering quality over existing 3D foundation reconstruction and SLAM baselines. These results support adaptive memory as a foundation for persistent robotic world modeling. The dataset and code will be made publicly available at \href{https://github.com/dtc111111/AWM-3DFM}{https://github.com/dtc111111/AWM-3DFM}.

### 论文解读
#### 摘要翻译
本文提出面向大规模三维建图、定位与渲染的自适应世界记忆三维基础模型。它用 Transformer 门控更新和测试时空调节维护长期记忆，将状态组织为局部子图，并结合回环检测与 SL(4) 全局优化；高斯重建头把增强特征转为可渲染三维高斯。

#### 方法动机分析
DUSt3R、VGGT 等模型偏向两帧或短窗口，长序列会带来计算/显存增长；简单循环更新还可能让遮挡、快速运动或低质量观测污染历史，造成遗忘和漂移。论文的核心假设是，用时空一致性筛选更新、用子图管理长期状态，可以同时获得可扩展性和全局一致性，并直接输出渲染地图。

#### 方法设计详解
连续单目 RGB 先经三维骨干提取几何 token。门控模块根据当前 token 与历史记忆生成候选状态，并以更新门混合新旧信息；测试时再将时间变化掩码与空间注意力掩码相乘，决定哪些 token 更新、哪些保留。几何/位姿头输出 metric pointmap、置信度和六自由度位姿，高斯头输出位置、旋转、尺度、不透明度与颜色。局部子图通过回环建立约束，在 SL(4) 流形上优化子图变换以校正尺度、剪切和投影误差。训练采用几何、位姿、RGB 渲染和高斯正则损失的组合，并用课程学习从短序列逐渐增加长度；论文报告使用 8 张 A100 80GB 训练，RTX 4090 推理时显存约 7GB。

#### 方法对比分析
相比短窗口三维基础模型，本文的本质差异是引入可持续、可筛选的世界记忆和子图级全局优化；相比传统 SLAM，又增加稠密度量点图和三维高斯渲染输出。门控、时空调节、回环/SL(4) 分别针对遗忘、污染和长程一致性，适合需要长期定位与新视角渲染的机器人场景。

#### 实验分析（精简版）
论文在 Sintel、Bonn、KITTI、Replica、7-Scenes、TUM RGB-D、ScanNet、BundleFusion 等数据集上与多种基础模型、SLAM 和高斯方法比较。Replica 中平均位姿误差为 3.05 cm，优于 Spann3R 的 38.94 cm 和 MASt3R-SLAM 的 5.10 cm。消融中，门控使漂移从2.98到1.42 m，时空调节使重建 Acc. 从35.19到9.14 cm，子图与回环使 ATE 从0.38到0.12 m。局限是动态物体、语义推理和极大规模长期一致性仍待增强；这些结果主要覆盖静态或受控场景。

#### 实用指南
论文给出 GitHub 项目 https://github.com/dtc111111/AWM-3DFM，并声明代码和数据将公开发布。复现需实现课程训练、四类损失、逐帧子图管理、回环和 SL(4) 优化；完整 batch size、学习率、epoch 与依赖版本未说明。迁移时需替换相机/场景数据并重新适配几何和位姿头，同时检查单目尺度、动态物体和回环质量。

#### 总结
核心思想：自适应记忆维护三维世界
1. RGB 序列提取几何 token。
2. 门控模块融合新观察与历史。
3. 时空掩码过滤不可靠更新。
4. 子图回环和 SL(4) 统一地图。
5. 多头输出定位结果与高斯渲染地图。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21502v1)
- [arXiv](https://arxiv.org/abs/2609.21502v1)

---

<a id='2609.21474v1'></a>
## [MT-WAM: Reorienting the One-Pass Predictive Representation Toward Action Generation](https://arxiv.org/abs/2609.21474v1)

**Authors:** Yiguang Yang, Jiankun Peng, Xiaoming Wang, Yiran Zhang, Zhibo Fang

**Published:** 2026-09-18

**Categories:** cs.CV, cs.RO

**Abstract:**

Fast-WAM shows that video-action co-training improves control without generating future video at inference, making the representation from a single video diffusion Transformer forward central to action generation. However, future-observation prediction does not explicitly prioritize the future dynamics and visual structure needed for control. We present MT-WAM, which retains the original training objectives and adds complementary supervision for future two-dimensional point trajectories and visual features. A lightweight dual-stream branch copied from the video backbone's final blocks provides target-specific processing, while a structured attention mask prevents cross-stream attention. Motion-stream tokens supply additional dynamics conditions to the action expert. Future visual-feature prediction provides supervision in a feature space that captures object and spatial structure. This supervision trains the video backbone to provide more informative visual context for action generation under changing visual conditions, without adding visual-feature-stream tokens to action conditioning. At inference, MT-WAM uses video and motion caches computed once per replan and skips future-video prediction. Without additional embodied policy pretraining, MT-WAM achieves 98.2% success on LIBERO and 73.7% on LIBERO-Plus, exceeding Fast-WAM by 23.8 percentage points on the latter. On RoboTwin 2.0 Clean2Rand, Random success increases from 6.30% to 19.40%; across four real-world tasks, average success increases from 67.0% to 77.8%.

### 论文解读
#### 摘要翻译
MT-WAM 将 FAST-WAM 的单次视频前向表示重新导向动作生成。它保留视频—动作协同训练，同时增加未来二维点轨迹和 DINOv2 视觉特征监督，让表示更关注运动与控制相关结构，而非背景、光照等像素细节。论文在 LIBERO-PLUS、ROBOTWIN 2.0 和真实机器人任务上验证了方法。
#### 方法动机分析
未来像素预测会编码大量不影响动作的外观信息，控制真正需要的是物体如何移动、关键区域如何变化。MT-WAM 的假设是，二维轨迹能提供紧凑的运动约束，DINOv2 特征能提供较稳健的视觉结构约束；二者联合可缩小预测表示与动作表示之间的差距。二维表示也有边界：它不能完整表达三维几何和接触状态。
#### 方法设计详解
输入为当前多视角 RGB、本体感受状态和语言指令，输出动作块。Video DiT 骨干基于 Wan 2.2-TI2V-5B。模型复制末端 M 个块形成双流动态分支：运动流预测 CoTracker3 提取的未来二维网格点相对位移，使用均方误差；视觉特征流预测 DINOv2 的未来特征图，使用余弦相似度损失。总损失同时包含视频、动作、运动和特征四项。结构化注意力掩码把当前帧副本分别提供给动作专家和动态分支；动作专家主要读取运动流，视觉流则通过反向传播改善骨干表示。推理时骨干与动态分支各运行一次，缓存视频和运动 KV，动作专家在多个去噪步复用缓存生成动作，不生成未来视频。论文报告 AdamW、学习率 1e-4、梯度裁剪 1.0。
#### 方法对比分析
FAST-WAM 主要依赖像素预测，MT-WAM 的本质创新是把“预测未来画面”改成同时约束未来运动与视觉结构，并用双流分支承载异构目标。它不是简单把更多 token 输入策略，而是让不同监督承担不同职责：运动流通过 KV 直接服务动作，视觉流通过反向传播改善共享骨干；结构化掩码还避免视觉流干扰控制。因而适合需要视觉泛化和低延迟动作生成的机器人任务，但对强三维接触任务仍需额外状态信息。
#### 实验分析（精简版）
LIBERO-PLUS 上 MT-WAM 成功率为 73.66%，FAST-WAM 为 49.86%，提升 23.8 个百分点；ROBOTWIN 2.0 Clean2Rand 从 6.30% 提升到 19.40%，真实四项任务平均成功率为 77.8%，FAST-WAM 为 67.0%。消融中去掉视觉特征流下降 4.9%，让动作专家同时读取两类流下降 2.8%，说明辅助监督和信息隔离都重要。局限是二维轨迹与有限视角对三维接触的覆盖不足。
#### 实用指南
论文提供代码仓库 https://github.com/Alexi1984/MT-WAM。复现需准备 CoTracker3 轨迹、DINOv2 特征和 Wan 2.2-TI2V-5B 权重，按论文联合训练四项目标；资源为 8 张 A100，LIBERO 约 16 小时、ROBOTWIN 2.0 约 4 天。迁移到新机器人时需重做相机与动作预处理、轨迹/特征目标和动作专家训练，并检查推理缓存与去噪步设置。
#### 总结
核心思想：用运动与特征监督塑造动作表示
1. 当前观察经视频骨干编码并建立任务隔离的参考流。
2. 动态分支分别学习二维运动和视觉特征。
3. 运动 KV 直接引导动作专家，视觉流反向改善共享表征。
4. 一次前向缓存后，多步去噪生成动作而不生成未来视频。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21474v1)
- [arXiv](https://arxiv.org/abs/2609.21474v1)

---

<a id='2609.21461v1'></a>
## [AtomEgo: Exploring Ego-Robot Integration for Embodied Foundation Model Pretraining](https://arxiv.org/abs/2609.21461v1)

**Authors:** Di Wu, Dongchen Zheng, Junhe Sheng, Zhongxing Wei, Songxin Zhang, Zejian Xie, Xiaoquan Sun, Junyang Zheng, Zhuoyang Song, Jiaxing Zhang, Jiayu Chen

**Published:** 2026-09-18

**Categories:** cs.RO, cs.AI

**Abstract:**

Embodied foundation models are constrained by the limited scale and diversity of robot demonstrations, motivating the use of large-scale egocentric human interaction data. However, how to effectively incorporate such data into embodied-model pre-training remains unclear because of substantial embodiment and action-space gaps between humans and robots. We present AtomEgo, a systematic study of ego--robot co-training supported by a curated corpus of approximately 2,659 hours and a scalable data processing pipeline. Across vision--language--action and world--action model architectures, we investigate three representative paradigms: joint co-training with domain-specific action heads, progressive ego-to-robot transfer through embodiment alignment, and joint video--action modeling. We evaluate these paradigms through multi-task real-robot experiments and language-conditioned cross-embodiment representation analysis. Our results reveal a simple principle: Data Scale * Alignment Quality --> Capability Gain; egocentric data can improve generalization, but their value depends on how effectively they are aligned and utilized. This principle can provide practical guidance for scalable ego--robot pre-training.

### 论文解读
#### 摘要翻译
AtomEgo 研究如何把大规模人类第一视角（Ego）视频用于具身基础模型预训练。作者整理约 2,659 小时语料，比较 Atom-DH、Atom-CL、Atom-WAM 三种整合范式，结论是数据规模必须与跨具身对齐质量共同发挥作用。

#### 方法动机分析
机器人演示采集昂贵且覆盖有限，而人类视频包含丰富交互先验；但人手和机械臂的形态、运动学及动作空间不同，Ego 视频也通常没有可执行动作标签。直接混合可能负迁移，因此需要区分共享视觉/物理知识与机器人特定控制。论文的关键假设是视觉交互规律可以跨具身迁移，但控制输出必须经过结构化对齐；目标是在扩大数据覆盖的同时保持可执行性、泛化性和训练稳定性。

#### 方法设计详解
多源数据先转为 RLDS，并映射到统一的 80 维状态—动作空间。策略输入一主视角、两个腕部视角、语言指令和机器人状态，输出未来 50 步动作块。VLA 版本采用 SigLIP、Gemma-2B VLM 与 Gemma-300M action expert。Atom-DH 共享主干、为 Ego 和机器人使用独立动作头，使同一表征能服务不同控制空间；Atom-CL 分三阶段：Ego 预训练、人机配对对齐、机器人微调，并在末端坐标系相对位移上计算对齐损失；Atom-WAM 则联合预测未来视频和动作，把视觉演化当作共同监督。训练使用 3×8 张 B200 约 4 天，微调使用 1×8 张 B200 约 12 小时。数据还经过突变检测、状态—动作趋势对齐和极值过滤，以减少噪声动作对预训练的影响。

#### 方法对比分析
相比仅使用机器人数据的基线，Atom-DH 的创新是以域特定输出头隔离动作空间，是轻量的共享—分离设计；Atom-CL 的主要创新是显式安排“先学交互、再对齐具身、最后控制微调”，把最关键的人机映射放在独立阶段，针对性最强；Atom-WAM 则假设共享视觉演化即可桥接具身差异，但视频预测不一定带来可控动作。论文还发现，单纯用最优传输拉近潜在表示可能破坏任务结构，因此“表征更相似”并不等价于“控制更好”。

#### 实验分析（精简版）
在 AgileX Piper 双臂平台的 7 个真实任务上，机器人基线 ID/OOD/综合成功率为 42.86%/22.86%/32.86%。Atom-DH 为 45.71%/41.43%/43.57%，主要改善 OOD；Atom-CL 为 64.29%/42.86%/53.57%，综合比基线高 20.71 个百分点。Atom-WAM 综合仅 20.71%，显示当前视频目标存在负迁移。OOD 测试改变物体重量、架子位置、光照或额外物体，因而能检验泛化而非只看训练分布拟合。实验仍集中于有限平台和任务，尚不足以证明对移动底盘或人形机器人的普适性。

#### 实用指南
论文给出 Atom-0 代码仓库，但预训练权重和完整数据的公开范围需查仓库，论文未明确保证。复现需实现 RLDS 统一格式、80 维映射、数据质量过滤、末端相对位姿对齐和三阶段训练。迁移到其他机器人时必须重建动作映射、准备人机对齐数据并进行目标平台微调，不能直接假设加入 Ego 视频有效。

#### 总结
核心思想：先学交互，再对齐具身
1. 统一 Ego 与机器人数据。
2. 从 Ego 视频学习交互先验。
3. 用人机配对对齐动作空间。
4. 在目标机器人上微调并预测动作块。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21461v1)
- [arXiv](https://arxiv.org/abs/2609.21461v1)

---

<a id='2609.21753v1'></a>
## [PSR: Predictive Sensorimotor Representation Learning for Contact-Rich Manipulation](https://arxiv.org/abs/2609.21753v1)

**Authors:** Shengbao Li, Peng Xu, Chao Tang, Hao Wei, Jiaheng Wang, Hong Yin, Jiangtao Chen, Jinxuan Zhu, Zhong Zhou, Mengfan Wang, Tingguang Li

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

Contact-rich manipulation requires policies to generate precise actions by reasoning over contact forces, robot configurations, and interaction histories beyond visual observations. Existing methods passively condition on force feedback rather than actively predicting future contact dynamics, limiting their ability to generate high-precision actions. To address this problem, we introduce Predictive Sensorimotor Representation (PSR) learning, a framework that learns a hierarchy of predictive representations from multimodal sensorimotor signals and integrates them into the action stream of a visuomotor policy. Specifically, during a pretraining stage, a multimodal Transformer is trained to learn a hierarchy of predictive representations by jointly forecasting future interaction dynamics. The learned hierarchy subsequently augments the action stream, enabling the resulting policy to exploit contact-relevant cues at multiple depths. We further instantiate PSR within a Vision-Language-Action (VLA) model, resulting in PSR-VLA, and evaluate it on six real-world contact-rich manipulation tasks. Experimental results show that PSR-VLA achieves 91.7% overall success, improving over $π_{0.5}$, ForceVLA-$π_{0.5}$, and ForceVLA2-$π_{0.5}$ by 30.0, 22.5, and 19.2 percentage points, respectively. These results demonstrate the effectiveness of the proposed PSR for force-aware, contact-rich manipulation. Videos of the tasks and stability tests are available at https://psr-vla.pages.dev/.

### 论文解读
#### 摘要翻译
接触密集型操作需要策略结合视觉、接触力、机器人构型和交互历史。论文提出 PSR，通过多模态 Transformer 主动预测未来感觉运动信号，并将分层表示接入视觉—语言—动作（VLA）策略。六项真实世界任务的综合成功率达到 91.7%。

#### 方法动机分析
插入、组装、擦拭时，视觉相似状态可能因接触状态不同而需要相反动作。传统 VLA 难以模拟接触动力学；已有力反馈方法多只是被动条件输入。PSR 的核心假设是，预测未来力/扭矩、关节出力和末端位移，能显式形成接触演化与运动意图的上下文，帮助策略提前调整动作。

#### 方法设计详解
模型输入视觉 token 与最近 h 帧的力/扭矩、关节状态、关节出力。PSR Encoder 采用浅层独立编码、中层模态融合、深层细化的层次结构，并在视觉和力/扭矩分支使用 MoE。固定容量瓶颈 token 通过跨注意力汇聚和广播信息，门控则保留模态特有细节。预测预训练同时约束未来力/扭矩、出力和末端位移，并加入 MoE 路由均衡损失。得到六个深度的表示后，分别以门控交叉注意力注入 VLA 的动作专家层，预测表示充当 key/value，输出连续动作块。训练使用 AdamW、余弦学习率调度和 8 张 H100；数据为 1,210 段演示，力/扭矩及出力采用逐通道分位数归一化。

#### 方法对比分析
π0.5 只依赖视觉语言，ForceVLA 把力反馈作为被动条件，ForceVLA2 使用混合力—位控制。PSR 的关键区别是先预测未来接触响应，再把不同抽象层级的预测表示对齐注入动作流，因此学习的是动力学时间结构，而非简单拼接传感器。它更适合接触状态无法由视觉唯一确定的任务。

#### 实验分析（精简版）
在六项真实任务上，PSR-VLA 成功 110/120 次（91.7%），比 π0.5 高 30 个百分点、比 ForceVLA2 高 19.2 个百分点；三向互锁组装和水管插入分别达到 18/20 与 19/20。去除预测预训练后，水管插入和双孔插头成功率降至 70% 与 55%，说明预测表示确实重要。未来信号预测 MSE 也低于持久性基线。主要局限是跨物体、跨接触条件和更长时域泛化尚未充分验证；RTX 5080 上相对 π0.5 增加约 10 ms 推理延迟。

#### 实用指南
论文提供视频与补充材料网站 psr-vla.pages.dev，但不能据此确认完整代码、权重和数据集均已公开。复现需同步视觉及感觉运动历史，执行分位数归一化，训练预测器，再按深度对齐实现门控交叉注意力。迁移到其他机器人需重新适配传感器坐标、状态和动作空间，并通常重新收集演示与训练。

#### 总结
核心思想：预测接触动力学

1. 编码视觉与感觉运动历史。
2. 预测未来接触信号和位移。
3. 将分层预测表示门控注入 VLA。
4. 生成动作块完成精细操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21753v1)
- [arXiv](https://arxiv.org/abs/2609.21753v1)

---

<a id='2609.21792v1'></a>
## [AcousticDiffusion: Semantically Conditioned Audio-Guided Diffusion Policy for Search-and-Rescue Assistance](https://arxiv.org/abs/2609.21792v1)

**Authors:** Iana Zhura, Didar Seyidov, Dmitrii Plotnikov, Hajira Amjad, Miguel Altamirano Cabrera, Dzmitry Tsetserukou

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

Navigating toward human callers is an important capability for rescue robots operating where visual contact is degraded or occluded. We present AcousticDiffusion, a semantically conditioned, audio-guided diffusion policy for human-directed navigation. A frozen pretrained audio recognizer processes 10.24 s windows, with speech gating and distress-aware prioritization converting recognition outputs into source-level navigation roles. Microphone-array direction-of-arrival measurements are recursively integrated into a robot-centric Bayesian bird's-eye-view belief field. Ego-motion compensation aligns successive observations, progressively constraining source position while preserving bearing-induced range uncertainty. The semantic belief, recent acoustic observations, audio features, and robot state condition a diffusion model that generates waypoint trajectories. On a synthetic-navigation validation set using recorded audio, AcousticDiffusion achieves a mean end-point bearing error of 11.20 degrees, with 91.78% of trajectories aligned within 30 degrees of the caller. Distractor rejection ranges from 89.20% to 98.99%, and the policy favors a HELP-designated caller over a competing speaker in 91.07% of windows. Deployed online on a ZSL-1 quadruped without additional retraining, it achieves a mean bearing error of 64.9 degrees, compared with 98.2 degrees for A* and 90.4 degrees for RRT, with a mean planner compute time of 6.07 ms. Despite imperfect acoustic localization, the reported mean final source distance is reduced from 3.96 m for the classical planners using ODAS-derived (Open embedded Audition System) guidance to 2.48 m, a 37.4% improvement. These results demonstrate the framework's ability to translate uncertain acoustic observations into closer approaches to human callers.

### 论文解读

#### 摘要翻译
AcousticDiffusion 面向视觉受遮挡的搜救机器人，通过识别呼救声并估计声源方向，生成接近呼叫者的导航轨迹。系统结合冻结音频识别器、递归贝叶斯鸟瞰信念图和条件扩散策略；在真实 ZSL-1 四足机器人上，平均最终声源距离从经典规划器的 3.96 m 降到 2.48 m。

#### 方法动机分析
烟尘、墙体或黑暗会使视觉定位失效，而单次 DoA 观测容易受混响和噪声影响。论文假设“HELP”等遇险语义能够筛选更重要的声音，连续 DoA 证据能够形成稳定空间记忆，扩散策略则适合在不确定声学信息下生成多种可行轨迹。它关注音频主导的搜救导航，并不等同于完整的灾害场景理解。

#### 方法设计详解
机器人以 16 kHz 采集音频，使用 10.24 s 窗口、0.5 s 步进输入冻结的 Audio Spectrogram Transformer，得到 128 维音频潜变量。声音被分为 ATTRACT、NEUTRAL 和 AVOID，遇险语音通过 β=0.6 的偏好权重增强。每个 DoA 在机器人中心网格上形成 von Mises 方向脊，并以 log-odds 累积；更新前按里程计增量变换旧信念图，再用时间衰减抑制移动声源的残影。BEV 信念、DoA token、机器人状态和音频潜变量融合后，条件化带 FiLM 与跳连的 1-D U-Net。模型对 80 个航点（10 Hz 下约 8 s）进行 DDPM 噪声训练，推理用 16 步 DDIM 从噪声生成轨迹，最后按最可能声源方向旋转对齐。论文报告训练 100 epochs。

#### 方法对比分析
A*（0.10 m 网格）和 RRT* 依赖 ODAS 声学引导后进行经典搜索，主要处理单次或显式规划约束。AcousticDiffusion 的区别在于把遇险语义、带自运动补偿的空间记忆和不确定轨迹生成联合起来，因此更能利用连续声音证据；代价是依赖训练分布、麦克风阵列和可靠 DoA。

#### 实验分析（精简版）
合成验证包含 414 个音频窗口（296 个有 HELP、118 个无呼叫），使用 LibriSpeech、RAVDESS 和多类干扰声。平均终点方位误差为 11.20°，91.78% 的轨迹落在呼叫者 ±30° 内，HELP 优先选择率为 91.07%。真实机器人上的平均最终距离为 2.48 m，相比经典规划器约 3.96 m 改善 37.4%；规划时间平均 6.07 ms、P95 为 12.67 ms。局限是实机平均方位误差升至 64.9°，且无呼叫时仅 1.69% 预测保持在 0.05 m 内，显示 sim-to-real 与停机能力仍不足。

#### 实用指南
复现时需保持音频采样率、窗口步长、三类语义标签、DoA 网格和 80 航点设置，并使用冻结的 AudioSet 预训练 AST、100 epochs 训练和 16 步 DDIM 推理。论文未说明优化器、学习率和 batch size，也未给出完整训练资源清单。作者表示 ROS 2 实现与训练检查点将在接收后发布，当前不应视为已公开。迁移到其他平台需重新标定阵列和 DoA、里程计坐标，并适配机器人运动学与控制接口，同时重新评估真实环境中的混响和误检。

#### 总结
核心思想：语义声学驱动扩散搜救

速记：
1. 音频识别遇险语义并形成目标偏好。
2. DoA 在自运动补偿的衰减 BEV 中递归定位。
3. 多模态条件控制扩散 U-Net，DDIM 生成并朝声源对齐的航点。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21792v1)
- [arXiv](https://arxiv.org/abs/2609.21792v1)

---

<a id='2609.21983v1'></a>
## [SkelWAM: A Skeleton-Guided World-Action Model for Zero-Shot Cross-Embodiment Manipulation](https://arxiv.org/abs/2609.21983v1)

**Authors:** Pengjun Niu, Yujia Xie, Rui Peng, Hang Zhao, Ke Liu

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

Reusing manipulation experience across robot embodiments is important for scaling robot learning and reducing repeated task-specific data collection. However, changes in embodiment alter visual appearance, action dimensionality and semantics, and the whole-body configurations that can realize the same tool pose. We present SkelWAM, a skeleton-guided world-action model that couples perception and control through one explicit geometric representation for single-source cross-embodiment manipulation. Arm centerline geometry, tool-center-point (TCP) pose, and parallel-jaw commands form a shared 25-D state. The same definition underlies canonical third-person and wrist observations and future whole-body action targets. Trained with predictive visual supervision, a video-action mixture of transformers predicts canonical skeleton action chunks, which embodiment-specific constrained decoders convert into joint or continuum-robot controls. This formulation requires no one-to-one joint correspondence and uses no target-task demonstrations or target policy updates. We introduce LIBERO-Cross10, a source-only cross-embodiment transfer benchmark covering ten tasks and ten target embodiments across four morphological groups. On this benchmark, Franka-trained SkelWAM achieves 43.3% success over 1,000 episodes, exceeding the best-performing evaluated baseline by 36.2 percentage points. We further deploy a JAKA mini2-trained policy on the Feagine A03 continuum robot for three tabletop manipulation tasks, illustrating the approach's potential for real-world cross-embodiment manipulation. Project page: http://www.liukepku.com/skelwam/index.html

### 论文解读
#### 摘要翻译
SkelWAM 面向零样本跨具身操纵：在 Franka 等源机器人上学习经验，再迁移到不同自由度、外形甚至连续体机器人。它用 25 维规范骨架统一表示臂形、工具中心点（TCP）位姿和夹爪状态，并让目标机器人通过自身约束解码器执行。LIBERO-Cross10 上平均成功率为 43.3%，比最佳基线高 36.2 个百分点。论文的关键贡献不是增加一个普通控制器，而是把可迁移意图和具身执行明确分层。

#### 方法动机分析
不同机器人会造成视觉外观、动作维度和运动学可行域的鸿沟。只重定向 TCP 无法确定冗余机械臂的全身姿态，端到端策略也难泛化到未见过的结构。论文的假设是，操纵意图可由与具体关节无关的 TCP、臂中心线和夹爪状态表达，再把可行性留给目标具身的运动学解码器。

#### 方法设计详解
输入是前视角、腕视角、语言指令和当前规范状态。状态包含五个相对中心线向量（15 维）、TCP 位置（3 维）、6D 旋转（6 维）及夹爪（1 维）。模型先遮罩机器人并填充背景，再投影骨架以削弱外观差异；随后由基于 DiT 的 30 层视频专家和 30 层动作专家预测 32 步骨架动作块。阶段 A 联合学习视频演化与 25D 几何动作，阶段 B 冻结视频专家，以流匹配优化动作生成；推理用 20 步采样。最后，IK 或连续体模型通过同时贴合中心线和 TCP、并惩罚配置突变，将规范动作转成本体控制，执行前缀后滚动重规划。

#### 方法对比分析
相较 OpenVLA 等隐式动作 token，SkelWAM 的中间变量有明确几何意义，并能被投影回感知输入；相较 TCP-only 重定向，它新增中心线约束来解决冗余姿态选择；相较通常需要目标数据的 Diffusion Policy，它把跨具身泛化放在规范状态和解码器接口上，不要求目标任务演示或策略微调。这一创新将“学什么”和“怎样由某台机器人执行”解耦。代价是必须有可靠的相机标定、机器人遮罩和目标具身解码器，因此适合能用骨架描述意图且可建立运动学约束的操纵任务。

#### 实验分析（精简版）
在含 10 个任务、10 种目标具身的 LIBERO-Cross10 上，模型使用 467 段 Franka 演示训练，平均成功率 43.3%；Diffusion Policy、OpenVLA-OFT、FastWAM 和 RoVi-Aug 分别为 1.3%、0.2%、0.4% 和 7.1%。去除骨架视觉或动作监督后性能接近 0%，去除解码器身体目标下降 2.4 个百分点。论文指出可达性、标定和遮挡背景填充是主要限制。

#### 实用指南
项目主页提供代码、权重和 LIBERO-Cross10。复现需保持双视角、骨架归一化与标定一致；训练阶段 A/B 分别为 6k/24k 次更新，推理采用 20 步流匹配和 32 步动作块。迁移新机器人时需替换运动学/连续体解码器和标定参数，并重新检查工作空间可达性；方法设定不需要目标任务演示。

#### 总结
核心思想：规范骨架传递操纵意向

1. 编码视觉、语言与当前骨架。
2. 生成规范骨架动作块。
3. 用具身约束同时跟踪臂形与 TCP。
4. 执行前缀并滚动重规划。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21983v1)
- [arXiv](https://arxiv.org/abs/2609.21983v1)

---

<a id='2609.21780v1'></a>
## [PointLAM: Local Attentive Mamba for Efficient Point-based 3D Object Detection](https://arxiv.org/abs/2609.21780v1)

**Authors:** Xuanming Shang, Weijia Zhang, Chao Ma

**Published:** 2026-09-18

**Categories:** cs.CV

**Abstract:**

3D object detection from LiDAR point clouds faces a fundamental dilemma: voxel-based methods achieve efficiency at the cost of geometric quantization, while point-based methods preserve fidelity but suffer from prohibitive computational bottlenecks. Specifically, point-based architectures are crippled by slow downsampling strategies (e.g., FPS) and expensive dynamic neighbor queries (e.g., k-NN) coupled with costly continuous interactions. To tackle these systemic inefficiencies, we propose PointLAM, a highly efficient and powerful point-based architecture driven by two synergistic innovations. First, to resolve the downsampling bottleneck, we develop the Laplacian Point Sampler (LPS). LPS employs an implicit discrete Laplacian high-pass filter and Doubly Sorted Sampling to achieve fast, structure-aware foreground preservation. Second, to overcome local modeling latency, we design the Local Hadamard Aggregator (LHA). LHA decouples spatial indexing from feature representation using transient grids, and replaces complex continuous interactions with a Hadamard Gating mechanism for topology-aware, attentive modulation. By coupling this local gating with Bi-Directional Mamba (BDM) layers for global sequence modeling, we formulate the Local Attentive Mamba (LAM) block. Powered by this architecture, PointLAM achieves competitive performance on nuScenes and Waymo for point-based detectors. It rivals highly optimized voxel competitors while requiring a fraction of the computational footprint, demonstrating marked superiority in detecting small instances and handling extreme sparsity. Project page: https://pointlam.github.io/.

### 论文解读
#### 摘要翻译
PointLAM 面向点云三维目标检测，试图同时保持点级几何细节与较高推理效率。它用 Laplacian Point Sampler（LPS）进行几何感知下采样，并以 Local Attentive Mamba（LAM）骨干联合建模局部与长程关系，在 nuScenes 和 Waymo 上验证。

#### 方法动机分析
体素化速度快，却会产生几何量化误差；点式方法保真度高，但最远点采样和动态邻域查询成本昂贵。论文的核心假设是，采样应优先保留边界、角点等几何显著点，而全局建模应在保持线性复杂度的同时补回三维空间关系。

#### 方法设计详解
原始 LiDAR 点云作为输入先进入 DevNet，计算点特征相对局部均值的离差，近似离散拉普拉斯响应：平滑区域响应小，几何突变处响应大。DSS 先按显著性排序，再按区域索引稳定排序；设定 k=1，在减少点数的同时保留区域覆盖和几何骨架。下采样点经过 4 个 LAM block。LHA 借助瞬态网格确定局部路由，用稀疏卷积聚合邻域，再以 Hadamard 逐元素门控调节通道并加入残差；这种门控本质上是按局部拓扑自适应改变通道幅值。BDM 沿 X/Y 轴双向扫描点序列，传播长程依赖，减轻一维序列化对三维结构的破坏。最终输出特征投影到 BEV，由检测头预测三维框。实现基于 OpenPCDet；nuScenes 训练 36 epochs、Waymo 训练 24 epochs，LHA 将 5×5×5 感受野拆成两个 3×3×3 稀疏卷积。nuScenes 网格为 0.3×0.3×0.25 m，Waymo 网格为 0.32×0.32×0.1875 m。

#### 方法对比分析
相较体素方法，PointLAM 保留点级几何；相较传统点式方法，LPS/DSS 替代昂贵 FPS，LHA 的网格路由减少动态邻域开销。相较单一 Mamba 扫描，BDM 与局部 LHA 协同，在近似线性复杂度下兼顾局部拓扑和远距离信息。论文的主要创新是把几何显著性采样、局部 Hadamard 注意力与双向状态空间扫描连成一条输入到输出的检测链路，而非只替换一个骨干层。消融表明两者不是可任意替换的工程组件：仅保留 BDM 或 LHA 都明显低于二者联合。轴向排序的延迟约 0.1 ms，而更复杂的 Hilbert 曲线约 5.5 ms，说明其设计强调效率与足够的空间结构，适合稀疏自动驾驶点云及小目标检测。

#### 实验分析（精简版）
nuScenes 验证集达到 72.2 NDS、67.8 mAP，高于 DSVT 的 71.1 NDS 和 LION 的 72.1 NDS；测试集 NDS 为 73.0。Waymo 验证/测试集 L2 mAPH 为 73.6/74.4。模型报告为 8.6M 参数、90.7G FLOPs、93.1 ms 延迟，FLOPs 比 LION 低 45%，而 LION 延迟为 195.3 ms。消融显示仅 BDM 或仅 LHA 为 69.58/69.40 NDS，结合后为 71.82；DevNet+DSS 为 71.82，高于 PFN+Pooling 的 71.42。结果支持精度、细节保留和效率的联合主张，但跨嵌入式硬件和超大范围场景的证据仍有限。

#### 实用指南
论文给出 OpenPCDet、A800、数据集网格尺寸、训练轮数、4 个 LAM block 和 DSS 的 k=1 等复现设定。项目主页为 pointlam.github.io；正文未明确给出稳定代码仓库、权重或完整依赖清单。迁移时需重新调整网格、采样显著性、轴向扫描范围和检测头，并重新训练。

#### 总结
核心思想：几何显著采样结合局部全局 Mamba
1. 拉普拉斯离差筛选关键点。
2. DSS 保持覆盖与几何骨架。
3. LHA 建模局部拓扑并门控。
4. BDM 传播全局信息后预测三维框。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21780v1)
- [arXiv](https://arxiv.org/abs/2609.21780v1)

---

<a id='2609.21621v1'></a>
## [Towards Fine-Grained Object Manipulation: SAM3-Guided Visuomotor Policy with Persistent Memory Learning and Focused Visual Conditioning](https://arxiv.org/abs/2609.21621v1)

**Authors:** Haolong Meng, Fangbo Qin, Mengchen Bai, Houwu Wang, Cirong Liu, Shan Yu

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

Fine-grained object (FO) manipulation requires robots to distinguish a specified FO from visually similar objects and execute actions reliably despite scene distractors. However, scene-level visual conditioning lacks explicit object selection, while category-level guidance cannot reliably distinguish FOs within the same category. We present a SAM3-guided visuomotor framework that addresses these challenges through persistent object memory and focused visual conditioning. First, we introduce FO Memory-driven SAM3 (FOM-SAM3), which learns reusable FO memory tokens from limited multi-view registration images while keeping SAM3 fully frozen. Through one-vs-rest learning, these tokens encode persistent memories for localizing target FOs and rejecting similar alternatives, which can be stored in a memory bank. Second, we propose Focused Spatial-Appearance Encoding (FSAE), which combines in-FO local appearance features with explicit bounding-box coordinates to condition action policies including Diffusion Policy (DP) and Action Chunking with Transformers (ACT). The effectiveness of the proposed FOM-SAM3 was validated on the FO-30 dataset comprising 30 physical objects across four coarse categories. Across three real-robot FO manipulation tasks, our FOM-SAM3-guided policies demonstrated robustness against distractors, discrimination ability among similar FOs, and extendibility to new FOs.

### 论文解读
#### 摘要翻译
论文面向细粒度物体操控：机器人需要在同类、外观相似的物体中准确选中指定实例。作者提出 SAM3 引导的视觉运动策略，由持久化 FO 记忆学习（FOM-SAM3）和聚焦空间-外观编码（FSAE）组成，并将结果用于 Diffusion Policy（DP）和 ACT。

#### 方法动机分析
场景级视觉特征容易混入背景和干扰物；类别级提示虽能找出“罐子”，却难以区分不同品牌的罐子。方法的关键假设是：通过少量多视图注册，可以把某个具体物体的辨识信息存成可复用记忆，并让策略只关注目标局部。若关键标志被遮挡或不在当前视角，识别仍会失败。

#### 方法设计详解
首先为目标 FO 采集多视图图像，在冻结的 SAM3 上学习记忆 token。线性变换作用于部分概念 token，形成 FO 专用提示；one-vs-rest 损失同时要求目标出现时分割、相似 hard negative 出现时不触发。推理时，从记忆库取出 token，SAM3 对侧视和腕视图输出 mask、边界框和特征图。FSAE 用 mask 筛选 FPN 特征，经 ROIAlign 得到 16×16×256 局部表示，并拼接归一化框坐标。给 DP 的表示通过空间 softmax 压成 16 个空间关键点，给 ACT 的表示投影成 64 个 512 维 token；再与机器人位姿、夹爪状态共同输入策略，输出动作块。感知训练使用 AdamW、学习率 10^-4、权重衰减 0.1、30 个 epoch；DP 预测/执行步长为 24/16，训练 150k 步，ACT 训练 100k 步。

#### 方法对比分析
FOM-SAM3 与普通文本提示的本质差别是从类别搜索转为注册后的实例记忆，并明确抑制相似物误报。FSAE 也不同于全场景条件化：它保留目标外观和位置，却减少无关背景干扰。相较 S²-Diffusion 的类别级引导，本方法更适合同类物体密集的分拣和服务机器人场景，但每个新物体都需要注册，且增加 SAM3 推理开销。

#### 实验分析（精简版）
FO-30 包含 30 个物体，标准测试集有 600 张图，挑战集有 400 张严重遮挡图。FOM-SAM3 的 mIoU 达 93.04%，高于原始 SAM3 的 70.18%；相似物误报率为 13.29%，原始方法为 35.46%。在 90 次综合操控尝试中，Ours-DP 成功 85 次（94.4%），DP 基线为 38 次（42.2%），S²-Diffusion 为 55 次（61.1%）。局限是遮挡视角、环境避障和额外延迟尚未完全解决；同时，现有结果主要覆盖 FO-30 及其设定，向未注册类别和更复杂动态环境泛化仍需额外验证。

#### 实用指南
复现时需准备目标物体的多视图注册样本，训练 FOM-SAM3，再按论文的负样本比例训练 DP 或 ACT；输入图像为 336×336，控制循环为 10 Hz。需实现 SAM3 分割、ROIAlign、空间编码和动作块训练。论文提到项目页会提供代码或视频，但正文未给出可核对的仓库地址。迁移到新物体需重新注册记忆，迁移到新机器人通常还需重训状态与动作策略。

#### 总结
核心思想：记住目标，再聚焦操控
1. 多视图注册具体物体的持久记忆。
2. 记忆提示 SAM3 找到目标实例。
3. mask 外观加边界框形成聚焦视觉 token。
4. DP/ACT 将 token 转为机器人动作块。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21621v1)
- [arXiv](https://arxiv.org/abs/2609.21621v1)

---

<a id='2609.21707v1'></a>
## [NeuRIO: A Streaming Neural Estimator for Zero-Shot Sim-to-Real Multi-Robot Relative Inertial Odometry](https://arxiv.org/abs/2609.21707v1)

**Authors:** Zhehan Li, Jiadong Lu, Shengwei Ren, Chao Xu, Yanjun Cao

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

We present NeuRIO, a streaming neural estimator for anchor-free 6-DoF relative inertial odometry using only identified inter-robot bearings, ranges, and IMU measurements. NeuRIO canonicalizes measurements into gravity-aligned coordinates, represents robots as nodes and mutual observations as factors, and uses attention for spatial reasoning and GRUs for temporal modeling. As a graph network, NeuRIO applies shared node-wise and factor-wise operators throughout the network, enabling it to handle different team sizes and time-varying observation graphs. NeuRIO is trained on a simulator that couples various motion patterns, device-level sensor characteristics, and diverse, realistic modeled, and temporally persistent sensor corruptions. In this way, NeuRIO achieves zero-shot sim-to-real transfer. Across $24$ real-world sequences, NeuRIO achieves $14.1\,\mathrm{cm}$ position RMSE and $3.9^\circ$ rotation RMSE. More importantly, NeuRIO demonstrates strong computational scalability, maintaining an update cost below $20\,\mathrm{ms}$ with up to $400$ robots in simulation, while optimization-based methods exceed $20\,\mathrm{ms}$ at only $24$ robots. Moreover, even trained on limited team sizes, NeuRIO transfers directly to unseen larger teams without architectural or parameter changes.

### 论文解读

#### 摘要翻译
NeuRIO 是一种流式神经估计器，只用机器人间的方位角、距离和 IMU 测量，在无锚点条件下估计六自由度相对惯性里程计。它把测量规范化到重力对齐坐标系，用图网络和注意力处理机器人关系，用 GRU 处理时间信息，并通过带持久性传感器故障的仿真训练实现零样本仿真到现实迁移。

#### 方法动机分析
传统滤波/优化方法精度高，却会随机器人数量扩大而变慢；已有学习方法又常依赖位姿先验或推理时几何精炼。论文希望得到一个端到端、连续运行、适应时变观测图和团队规模变化的估计器。关键假设是重力方向足以消除部分姿态变化，时空图网络则能从不完美测量流中恢复相对位姿。

#### 方法设计详解
输入包括各机器人 IMU、机器人对之间的 bearing/range、有效性掩码和角色属性。首先用加速度计估计重力并建立 canonical frame，降低 roll/pitch 变化造成的分布差异。机器人被编码为 128 维节点 token，观测被编码为 64 维因子 token。循环因子图层把机器人作为节点、测量作为有向因子，区分 observer 与 observed，通过交叉注意力交换空间信息，并用节点/因子 GRU 隐状态积累时间上下文。位姿头输出平移、连续 6D 旋转表示，以及位置和旋转的对数方差；异方差损失据此自动平衡位置与旋转误差。仿真随机化空中、地面、手持和静止运动，同时模拟遮挡、闪烁、黑屏、身份切换、NLOS、多径、丢包、IMU 漂移和振动，而且故障随时间持续演化。模型以 50 Hz 流式推理。

#### 方法对比分析
相比 CREPES-X、CT-RIO 等显式优化方法，NeuRIO 用共享权重的图网络换取更好的大规模实时性；相比 AnyAmber，它不需要位姿先验或真实数据微调。重力规范化、角色感知的循环因子图、不确定性输出和持久故障训练共同针对 sim-to-real 与观测失效问题，适合大规模相对定位，但不是全球绝对定位或长窗口平滑方案。

#### 实验分析（精简版）
真实 24 个序列上，NeuRIO 的位置/旋转误差为 14.1 cm/3.9°；AnyAmber（fine-tuned）为 22.3 cm/4.0°，CT-RIO 为 7.0 cm/2.2°，后者精度更高但在约 22 台机器人时超过 20 ms 更新预算。NeuRIO 在 400 台机器人时仍低于 20 ms；仅在 3–10 台上训练的模型在仿真 400 台时为 12.3 cm，限制邻居后为 4.6 cm。局限是长程时空约束和极端情况下的不确定性校准仍不足。

#### 实用指南
论文提供代码仓库 https://github.com/FAST-FIRE/NeuRIO。报告的复现设置为 batch size 25、学习率 1e-3、1000 epochs，单张 RTX 3090Ti 约训练 8.9 小时，模型约 1.67M 参数，采用 PyTorch/PyTorch Geometric。复现时要实现重力对齐、有向观测因子和持久性故障，而非只加入独立噪声；迁移到新机器人或传感器需重建测量/故障分布并重新训练。

#### 总结
核心思想：持久故障增强的流式图估计
1. 重力对齐输入。
2. 观测构成有向因子图。
3. 注意力传播空间、GRU记忆时间。
4. 不确定性头输出六自由度相对位姿。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21707v1)
- [arXiv](https://arxiv.org/abs/2609.21707v1)

---

<a id='2609.21777v1'></a>
## [TRACE: Coverage Path Planning for Unknown Environments Using Hierarchical Coverage Tree](https://arxiv.org/abs/2609.21777v1)

**Authors:** Zongyuan Shen, Haodong Liu, Gao Wang, Shancheng Zhao, Dehua Zhou, Yaming Ou, Zhongqiang Ren, Yikui Zhai, C. L. Philip Chen

**Published:** 2026-09-18

**Categories:** cs.RO

**Abstract:**

This paper presents a novel online coverage path planning (CPP) algorithm, called TRACE, for real-time coverage of unknown environments. TRACE is built upon a hierarchical coverage tree that provides a global representation of the evolving connectivity of the uncovered space. As the environment is incrementally revealed and covered, newly discovered obstacles and covered cells may fragment the remaining uncovered space into disconnected regions. TRACE recursively expands the corresponding tree nodes to explicitly represent these regions and organize them for subsequent coverage planning. Based on the updated tree, an incremental global tour is maintained to guide the coverage process. TRACE locally refines only the affected portions while preserving the visiting order of unchanged regions, thereby reducing the computational burden of global replanning and maintaining a consistent coverage progression. Guided by the global tour, a local planner generates back-and-forth coverage paths and switches to global-tour-aware planning to efficiently complete the target regions. Theoretical analysis establishes the computational complexity and complete coverage property of TRACE, and derives an approximation bound for the incremental global tour refinement. The performance of TRACE is evaluated through extensive high-fidelity simulations and real-robot experiments using a mobile robot. Comparative evaluations against six existing CPP methods demonstrate significant improvements in coverage time, path length, overlap ratio, and number of turns.

### 论文解读
#### 摘要翻译
TRACE 面向未知环境中的在线覆盖路径规划，提出层级覆盖树来表示未覆盖空间连通性的演化。环境逐步显现时，树递归扩展以表示区域分裂，并维护增量全局巡航路径；地图只发生局部变化时，仅优化受影响段落，保留其他区域的访问顺序。论文证明算法完备，并在仿真与实体机器人实验中显示其在覆盖时间、路径长度、重叠率和转弯次数方面优于现有方法。

#### 方法动机分析
未知环境中机器人只能边感知边规划。局部规则容易把剩余空间割裂成孤立区域，导致回溯和重复覆盖；每次全局重规划又开销大、会扰乱已经形成的访问顺序。TRACE 的核心假设是，新增地图信息主要影响覆盖结构中的局部区域，因此可以显式保存全局连通关系，同时只修复局部。论文主要面向二维静态环境。

#### 方法设计详解
输入是 LiDAR/IMU 等传感器数据与符号栅格地图。网格状态分为未知 U、障碍 O、已覆盖自由空间和未覆盖自由空间，并用形态学闭运算减弱噪声。算法为剩余空间建立层级覆盖树：发现某叶节点的残余区域后，用 flood-fill 做连通分析；若分成 J 个分量，就将该叶节点扩展为 J 个子节点。全局访问序列只在这个位置局部插入新子节点，并用固定起点和终点的 TSP 优化顺序。局部节点内按左—中—右扫描线生成往复覆盖；遇到死胡头则转向最近未覆盖单元，节点完成前用巡航路径连接下一目标。论文给出的单轮复杂度为 O(|T|²)，平均单次迭代计算时间为 10^-2 秒量级。

#### 方法对比分析
与 BINN、BA*、ε*、PPCPP、IBINN 等方法相比，TRACE 的区别不只是换一种局部启发式，而是用可递归更新的树显式维护剩余空间连通性，再用增量 TSP 修补局部巡航。这样既保留全局访问顺序，又避免频繁全局重规划。它适合传感器逐步揭示的二维静态场景；动态障碍和三维覆盖尚未验证。

#### 实验分析（精简版）
实验使用 Gazebo 六类 90m×90m 场景（Office、Warehouse、Forest、Mall 1–3），机器人配备量程 12m 的 360° LiDAR，并加入实体机器人测试。TRACE 在各场景报告最短覆盖时间和路径长度；部分场景重叠率接近 0%，而 PPCPP 可超过 60%。定位噪声标准差从 0.02m 增至 0.12m 时，覆盖率仍保持 98%以上。论文还报告单次迭代约 10^-2 秒，显示在线计算可行；但完整逐场景提升数值和动态环境证据有限。

#### 实用指南
复现需实现符号地图状态更新、形态学去噪、残余区域连通分解、层级树扩展、固定端点 TSP 与扫描线覆盖，并在 Gazebo 中配置差速机器人和 12m LiDAR。论文未明确提供代码仓库、依赖版本或训练超参数；方法本身是在线规划，不需要训练模型。迁移到其他机器人时需重设栅格分辨率、机体尺寸、扫描间距及运动学约束；动态或三维任务还需重构树更新与可达性处理。

#### 总结
核心思想：层级树增量维护覆盖连通性

1. 传感器更新并分类符号地图。
2. 对变化区域做连通分解，扩展覆盖树。
3. 仅局部用 TSP 修补全局巡航。
4. 节点内往复扫描，死胡同转向最近剩余单元。
5. 连接节点并持续输出覆盖轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2609.21777v1)
- [arXiv](https://arxiv.org/abs/2609.21777v1)

---

