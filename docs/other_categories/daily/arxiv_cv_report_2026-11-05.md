time: 20261105

# Arxiv Computer Vision Papers - 2026-11-05

## Executive Summary

## 今日研究脉络
今日十篇论文以机器人世界模型、视频到动作迁移和持续学习为主线。多篇工作尝试把大规模视觉动态压缩成更可控的中间表示：DREAMFREE蒸馏未来表征，DreamX学习潜动作，FloMo预测三维场景流，AIT对齐显式与隐式动作意图；PrimeSim及混合体素—粒子模型则侧重动作条件模拟与多模态世界状态。

实证较直接的工作包括DREAMFREE、FloMo、AIT和RoboTryOn，它们报告了仿真或真实机器人控制收益，也各自暴露任务数量、平台、硬件和伪标签等边界。PrimeSim与混合体素—粒子模拟器扩展了可控性、长时预测、几何和触觉评估，但仿真或开环指标不等同真机部署保证。DreamX展示视频扩散特征可形成紧凑动作潜变量，真实操作证据仍集中在有限设置。

系统层面的论文补足了另一组问题：Mamba3R以固定容量状态记忆处理长序列重建，但机器人评估为离线轨迹；CRONOS揭示稀疏重置下的分布漂移与恢复难点，证据来自仿真；GSBR为在线强化学习构造几何支持域约束，目前没有真机结果。阅读时应把表示质量、模拟预测、策略成功率和实时性分开判断，并重点核对各论文的测试协议与适用边界。

---

## Table of Contents

1. [Future Representation Distillation for Imagination-Free World Action Models](#2611.04527v1)
2. [PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations](#2611.04870v1)
3. [FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video](#2611.05062v1)
4. [Mamba3R: Fixed-Capacity Selective State Memory for Long-Sequence 3D Reconstruction](#2611.04221v1)
5. [Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation](#2611.04492v1)
6. [Action Intent as a Bridge: Transferring Manipulation Knowledge from Human-Centric Videos to Robots](#2611.04522v1)
7. [RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On](#2611.05163v1)
8. [DreamX: Video Diffusion Model is a Universal Latent Action Learner](#2611.05377v1)
9. [CRONOS: Benchmarking Non-Episodic Reinforcement Learning for Robotic Manipulation](#2611.05080v1)
10. [GSBR: Geometric Support Behavior Regularization for Online RL](#2611.05295v1)

---

## Papers

<a id='2611.04527v1'></a>
## [Future Representation Distillation for Imagination-Free World Action Models](https://arxiv.org/abs/2611.04527v1)

**Authors:** Jaehyeong Jo, Sangwon Jang, Sihyun Yu, Seonghyeon Ye, Yekyung Nah, Hojung Jung, Kimin Lee, Sung Ju Hwang

**Published:** 2026-11-03

**Categories:** cs.RO

**Abstract:**

World Action Models (WAMs) use the predictive knowledge of pretrained video diffusion models for robot control through future video prediction. However, iterative video denoising is costly, and inaccurate visual predictions can compromise action prediction. Our key idea is to make this predictive knowledge available through visual representations computed from the current observation, enabling action prediction without generating future videos. We introduce DreamFree, a WAM that repurposes a pretrained video diffusion model as a predictive visual encoder through representation distillation. The visual encoder learns to predict a frozen video diffusion model's representation using only the current observation. An action module is then trained to predict actions by extracting action-relevant information from the visual encoder. Across real-world and simulated tasks, DreamFree achieves higher success rates than state-of-the-art VLAs and WAMs, while performing real-time closed-loop control at 11Hz with a 14B video diffusion backbone. Experiments show that DreamFree can translate stronger video priors and additional video data into better robot control and enable learning new robot tasks from video demonstrations, including videos from a different embodiment, without action annotation for the new tasks.

### 论文解读

#### 摘要翻译

DREAMFREE提出未来表征蒸馏世界动作模型：用当前观测预测视频扩散教师对未来帧的内部表征，再从预测表征生成动作。部署时不必迭代生成未来视频。作者报告其在仿真与真实操控上有竞争力，并研究无动作标注视频的任务迁移。

#### 方法动机分析

世界动作模型从视频生成器获得物理动态先验，但推理时反复去噪昂贵，错误未来画面也会误导动作。直接使用当前帧特征又未必包含未来信息。作者假设教师未来特征可被蒸馏到只看当前帧的视觉编码器中，使动作模型保留预测能力而绕过视频生成。

#### 方法设计详解

第一阶段冻结视频扩散教师，给它当前与未来视频；学生编码器只看当前帧和指令，并通过结构化注意力遮断未来信息，再用投影器对齐教师未来令牌。生成流匹配损失保留学生视频先验，未来表征蒸馏让学生编码器预测动态。第二阶段冻结视觉编码器，使用64个查询读出动作相关特征，并由流匹配动作DiT生成32步动作块。部署仅单次视觉前向，动作头执行有限去噪。


实验使用Wan2.2-5B或14B视觉骨干，64个查询压缩动作相关视觉信息；动作头采用流匹配并预测32步动作块。

训练使用Wan2.2-5B或14B骨干，部署只做一次视觉前向；动作DiT生成32步控制块，不运行完整未来视频生成器。



训练步数和视频数据规模依数据集而变，DROID完整两阶段训练报告约一周、16张H200；推理不调用冻结教师。
#### 方法对比分析

相较部署时生成完整未来视频的WAM，DREAMFREE只生成动作块；相较直接训练动作策略，它用视频教师监督未来表示。适用于拥有视频预训练数据且对控制延迟敏感的机器人。训练仍然需要大量算力和动作示范，且视频特征质量不保证动作成功。

#### 实验分析（精简版）

RoboCasa 14B平均成功率71.7%，Fast-WAM 14B为44.7%；DROID十类实机任务成功率64%，DreamZero为56%、π0.5为16%，每任务5次评估。论文报告H200上11Hz和相对DreamZero 17.8倍推理加速。小样本任务评估和原子动作依赖是主要边界。


其视觉骨干先以Wan2.2视频数据适配，随后使用冻结教师和学生编码器做表征训练，再冻结视觉编码器学习动作头。DROID完整训练约一周、使用16张H200，说明高吞吐部署并不意味着训练成本低。
#### 实用指南

论文使用Wan2.2-5B与14B骨干、1B动作DiT；教师只在训练时存在，部署时视觉编码单次前向。复现需保存同一视频教师中间层特征，正确屏蔽未来帧，并对齐动作时间窗。作者提供代码链接。迁移到自动驾驶可试验预测中间特征，但要另行验证多主体危险事件和长尾场景。


单次视觉前向降低了未来视频生成开销，但动作DiT仍需多步采样。真实系统部署前应分别测视觉延迟、动作去噪延迟和闭环频率。
#### 总结

核心思想：把视频先验蒸馏成当前帧的未来表征。
1. 教师读取完整视频。
2. 学生只看当前观测并对齐未来特征。
3. 查询读出控制信息。
4. 动作头生成控制块。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.04527v1)
- [arXiv](https://arxiv.org/abs/2611.04527v1)

---

<a id='2611.04870v1'></a>
## [PrimeSim: A Generalist World Model Simulator from Large-Scale Cross-Embodiment Demonstrations](https://arxiv.org/abs/2611.04870v1)

**Authors:** Travis Davies, Wentao Ma, Lihao Guo, Tengfei Ma, Xiaowen Chang, Jiafeng Xu, Shilong Wang, Yiqi Deng, Shaowen He, Xiaoqi Li, Qian Zhu, Ran Cheng, Hao Dong

**Published:** 2026-11-03

**Categories:** cs.RO

**Abstract:**

A robot world model that serves as a simulator must do more than generate plausible videos: it must faithfully follow commanded actions, maintain consistency across camera views, and sustain coherent long-horizon rollouts. We present PrimeSim, an action-conditioned, multiview world model simulator that addresses these challenges through progressive midtraining and posttraining. Starting from a pretrained video generator, we midtrain on 30,110 hours of robot demonstrations spanning 22 platforms. To improve action following, we introduce a unified cross-embodiment action representation and frame-aligned dual-path conditioning to strengthen action-frame correspondence. To maintain multiview consistency, we develop view-aware interleaved attention with explicit camera identities, enabling cross-view information exchange while preserving view-specific representations. We further introduce a fast-video, slow-memory posttraining framework that transforms the bidirectional video generator into an efficient autoregressive simulator. By combining fast causal generation, self-rollout training, and hierarchical temporal memory, PrimeSim mitigates error accumulation from self-generated histories and supports coherent long-horizon simulation. PrimeSim runs at 10.1 FPS, remains stable over long-horizon rollouts on rigid and deformable manipulation tasks, and transfers to held-out platforms within 100 to 200 gradient steps where the pretrained backbone does not. Against five open-source action-conditioned world models it leads four of the five WorldSimProbe suites and posts the best overall EWMBench score, while fine-grained local calibration still trails the field; its rollouts are faithful enough to support GRPO-based policy optimization. We release the model weights, inference code, and training code.

### 论文解读

#### 摘要翻译

PrimeSim提出一种面向机器人控制的多视角世界模拟器。它在跨机器人平台数据上学习动作条件视频生成，并通过因果生成、自滚动训练和分层记忆改善长时预测。作者报告其在多个世界模型评测中具有竞争力，但细粒度局部校准仍有不足。

#### 方法动机分析

一个可用于策略训练的世界模型，不能只生成视觉上合理的视频，还要听从动作、保持不同相机视角一致，并在长时滚动时不迅速漂移。单一动作编码难以覆盖不同机器人，逐帧生成又容易让历史误差累积。作者假设：共享动作空间、显式相机身份和区分快速画面生成与慢速历史记忆，可以同时改善这些问题。

#### 方法设计详解

输入是语言、机器人观测、多视角视频和动作。首先将不同平台动作映射到31维共享表征，并用逐帧对齐的双路径条件把动作和对应画面关联起来。视角感知的交错注意力让不同相机的信息交换，同时保留各自身份。模型从预训练视频生成器起步，经机器人数据中训，再做因果化后训：快速路径逐帧生成，慢速路径维护分层时间记忆；自滚动训练让模型接触自己产生的历史，减轻训练与推理分布差异。


在迁移到新平台时，动作维度映射、相机内外参与时间戳对齐是必须先固定的接口；这些因素若变化，模型效果会与生成器本身混杂。
#### 方法对比分析

与只追求视频质量的生成模型相比，PrimeSim明确评测动作遵循和多视角一致性；与每帧独立预测相比，它用自生成历史和分层记忆处理长时误差。它的价值在于提供可控模拟接口，适合跨形态策略学习或虚拟评估。与显式三维物理模拟相比，它仍是数据驱动的视频模拟，不能自动提供碰撞安全或精确几何保证。

#### 实验分析（精简版）

作者报告30,110小时、22个平台的机器人数据训练；WorldSimProbe五项中领先四项，T1并非最佳。EWMBench整体得分高于所比较模型，但局部校准仍落后。模型支持100至200步梯度适配，并用于仿真内策略优化。上述结果证明的是基准和仿真用途，尚未证明真实机器人策略可依靠其安全训练。


读者应区分视频逼真度、动作条件遵循与策略可用性三种能力。论文的多项指标覆盖前两者，也有仿真策略实验，但没有把模拟策略放到真实机器人上验证。若用于离线数据扩增，应对同一动作的多视角轨迹做一致性审计，并比较模拟器上的策略排序与真实评测排序。

报告吞吐约10.1fps，EWMBench整体得分领先所比模型；这些数值来自视频与仿真基准，尚无真实机器人策略部署对照。


#### 实用指南

复现重点是动作单位与机器人坐标约定、相机标识、不同视角的帧同步，以及长时自滚动训练。论文报告10.1 FPS，画面为192×256；部署预算需结合目标平台重新测量。作者说明发布模型、推理及训练代码。迁移到移动操作时，可沿用共享动作与视角条件思路，但应补充遮挡、相机移动和真实轨迹上的误差评估。


报告10.1 FPS，但输入分辨率仅192×256；部署者应依据视角数、动作块长度和目标显卡重测吞吐。作者称提供训练代码与权重，迁移实现需处理不同机器人动作维度和相机标定。
#### 总结

核心思想：让动作条件模拟兼顾多视角与长时一致。
1. 对齐多平台动作。
2. 注入相机身份并联合建模视角。
3. 因果生成并用记忆稳定历史。
4. 在基准与策略仿真中评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.04870v1)
- [arXiv](https://arxiv.org/abs/2611.04870v1)

---

<a id='2611.05062v1'></a>
## [FloMo: 3D Scene Flow as a World Action Model Intermediate for Learning from Human Video](https://arxiv.org/abs/2611.05062v1)

**Authors:** Jeremy Collins, Namra Patel, Ayush Agarwal, Shitij Govil, Animesh Garg

**Published:** 2026-11-03

**Categories:** cs.CV, cs.AI

**Abstract:**

Pretrained video generation models distill internet-scale motion priors into robot policies, but full video is a redundant signal for action prediction. We argue that dense 3D motion is the medium through which video pretraining best transfers to robot actions, and that predicting motion rather than video affords better sample-efficiency for action prediction. FloMo is a 5B video-generation model fine-tuned to predict 3D scene-flow and action tokens. Rendering scene flow as RGB and encoding it into the same latent space as video enables a pretrained video generation backbone to efficiently adapt to predict 3D scene flow. Co-trained on action-free human video and action-labeled robot teleoperation data, our model benefits from human-video pretraining and generalizes to unseen objects, motion primitives, and tasks. Replacing motion prediction with a video prediction objective degrades out-of-distribution generalization, while adding a video prediction target alongside motion lowers policy success rather than improving it. When motion, video, and action are predicted jointly, action tokens attend far more to motion than to video, suggesting that predicting video wastes model capacity. Together, these results indicate that motion effectively mediates the transfer from video pretraining to action. More information can be found at flomo-wam.github.io.

### 论文解读

#### 摘要翻译

FloMo把三维场景流作为世界动作模型的预测目标。模型利用预训练视频生成骨干，联合预测连续的三维运动与机器人动作，并让无动作标签的人类视频参与预训练。作者报告实机操控和分布外任务上的成功率提升。

#### 方法动机分析

直接预测视频会消耗计算，也可能把动作无关的纹理、光照和背景误差带进控制；只预测动作又未充分利用视频模型学到的动态先验。作者假设几何运动是两者之间更合适的接口：它保留物体如何移动，却淡化外观变化。

#### 方法设计详解

输入为当前图像、语言和机器人状态。视频编码器从Wan 2.2特征中提取表示，模型以流匹配方式共同预测机器人动作与累计三维场景流。为使用预训练视频VAE，三维位移分量被映射为RGB图像，由解码器恢复运动。人类视频只提供视觉动态先验，机器人数据则提供动作监督。作者采用LoRA微调5B骨干，训练与推理的设置针对机器人操作数据。


作者以Wan 2.2的5B骨干和LoRA适配，动作块通过流匹配生成；规范坐标系与累计运动定义让相邻片段共享统一空间。

输入包括机器人状态、动作条件和语言目标；视频骨干编码场景后，运动头预测多帧累计位移，策略头输出对应动作块。


#### 方法对比分析

与像素级未来帧预测相比，场景流减少外观信息；与二维轨迹或潜在动作相比，它保留三维几何变化。实验显示在工具接触和跨场景任务中有优势。适用于几何关系决定成败的操作；若任务依赖材质、透明度或精确力反馈，场景流表示可能丢失必要线索。

#### 实验分析（精简版）

三个真实YAM任务平均成功率91%，高于VMA的51%和VA的40%。三项分布外任务平均成功率60%，而无人工视频的消融为7%；仿真RoboTwin平均63.6%，高于π0.5的44.8%。这些数字来自有限任务和特定平台；作者也报告场景流提取、追踪失败会影响性能。


作者采用Wan 2.2的5B视频骨干，并以LoRA适配；推理使用10步流匹配，报告H100上一个16动作块约0.7秒。此设定说明性能依赖较强GPU，不能只根据成功率估计实时能力。
#### 实用指南

复现要注意坐标系归一化、相机运动补偿、场景流可视化编码以及动作块长度。论文报告5B Wan骨干并用LoRA训练，流提取约需每16帧2.4秒的L40S计算。迁移到移动机器人时，可尝试以几何中间量预测接触和位移，但需额外处理自运动、遮挡与动态物体。


场景流的规范坐标系让外观变化不成为主要监督，但也让透明物体、软体形变和被遮挡区域的估计更困难。迁移时应分别检查运动估计误差与动作解码误差。
#### 总结

核心思想：用三维运动连接视频先验与动作。
1. 编码当前画面和指令。
2. 预测规范坐标系场景流。
3. 联合解码动作。
4. 用实机任务检验迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05062v1)
- [arXiv](https://arxiv.org/abs/2611.05062v1)

---

<a id='2611.04221v1'></a>
## [Mamba3R: Fixed-Capacity Selective State Memory for Long-Sequence 3D Reconstruction](https://arxiv.org/abs/2611.04221v1)

**Authors:** Zhenxiang Xiong, Tianchen Deng, Nailin Wang, Lei Zhao, Guole Shen, Weidong Che

**Published:** 2026-11-03

**Categories:** cs.CV

**Abstract:**

Streaming 3D reconstruction must maintain geometric quality over long inputs within a fixed GPU budget. A pretrained recurrent model provides bounded state, but freezing its backbone also fixes its learned state-observation interactions. We introduce Mamba3R, a jointly designed state-memory adapter that provides a trainable adaptation path without updating the backbone or task heads. A selective state-space mixer jointly transforms native state features and a fixed-capacity auxiliary memory, explicit read and write paths carry information across frames, and a decomposed feedback path regulates what the added memory feeds back. Only 1.28 million adapter parameters are trained. Under a shared periodic-reset protocol, it lowers trajectory error and improves accumulated surface and depth accuracy over recurrent baselines on public indoor and outdoor datasets and robot-recorded traversals, gaining up to 40 percent on one metric. Matched comparisons support both the choice of the mixing operator and the utility of carrying the auxiliary memory across frames. Allocated memory remains unchanged and throughput moves by at most 1.4 percent from one to two thousand frames. Replaying its recorded traversals offline, the platform's RTX 4060 averages 11.7 frames per second within 5.1 GiB with the same checkpoint, a mean throughput covering the 10 Hz sensor rate.

### 论文解读

#### 摘要翻译

Mamba3R为递归三维重建增加固定容量状态记忆，让模型在长视频中保留有用历史，而不需要存储和重复计算所有过去帧。作者在室内外重建和机器人记录序列上报告几何精度改善。

#### 方法动机分析

流式三维重建要在有限内存中利用历史。逐帧模型可能忘记过去，窗口注意力则需要缓存并受窗口边界影响。论文的假设是，固定大小的可读写记忆槽配合选择性状态空间更新，可积累长程信息，同时把额外参数控制在较小范围。

#### 方法设计详解

输入是一段连续RGB帧，冻结CUT3R提取图像特征并递归解码。新增适配器将当前内容和固定容量记忆拼接，通过Mamba选择性状态空间模块混合；显式读写路径决定哪些信息保留，并使用衰减反馈避免预测误差长期污染记忆。输出仍由原有三维点云、相机位姿和深度头生成。


额外适配器只有约1.28M可训练参数，容量固定；记忆读写在每帧递归执行，周期重置则用于限制长序列漂移。

记忆适配器约1.28M参数，固定容量读写并按协议周期重置；这一轻量设置是在冻结CUT3R骨干上训练的。



训练阶段只更新记忆适配器，推理阶段递归处理当前帧；其公开机器人测试按100帧周期重置。
#### 方法对比分析

不同于存储整段历史的注意力方法，Mamba3R用定长记忆；不同于测试时优化，它主要在冻结骨干的递归接口中学习记忆更新。适用于在线或长序列重建；若任务要求完整全局地图，仍需回环与后端优化，前端记忆不能替代SLAM全系统。

#### 实验分析（精简版）

500帧ScanNet评测中，Mamba3R相对重置协议匹配的TTT3R，CD、ATE和序列尺度AbsRel分别改善9.6%、7.3%和17.7%。机器人记录序列也显示误差下降，但数据是离线回放且每100帧重置；完整耦合系统约9.85 FPS。


记忆适配器的实验设计强调容量固定与重置协议匹配，避免把更长缓存带来的优势混入方法对比。其机器人结果是已录制轨迹的重放评估，不能说明系统在传感器延迟、运动模糊或实时回环下仍有同样改进。

实机记录回放中，室内平均ATE相对TTT3R改善27.9%，点云距离改善37.5%；它不是在线闭环机器人试验。


#### 实用指南

作者仅新增约1.28M可训练参数，骨干保持冻结。迁移时需统一重置频率、序列长度与传感器标定，否则性能比较会混入协议差异。对移动机器人可复用有限状态记忆思路，但真正在线应用还要测量延迟、漂移、回环和断点恢复。


复现可优先检查重置间隔、序列长度、点云参考构造和每帧算力预算；换传感器或相机频率时，记忆衰减参数也可能需要重估。
#### 总结

核心思想：用小型定长记忆支撑递归三维重建。
1. 冻结视觉重建骨干。
2. 读写有限记忆槽。
3. 衰减并筛选历史信息。
4. 输出连续几何结果。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.04221v1)
- [arXiv](https://arxiv.org/abs/2611.04221v1)

---

<a id='2611.04492v1'></a>
## [Hybrid Particle-Voxel Dynamics for Multimodal 3D World Simulation](https://arxiv.org/abs/2611.04492v1)

**Authors:** Jimmy Wang, Yifan Chou, Kaifeng Zhang, Yixuan Wang, Binghao Huang, Tao Yu, Saleh Nabi, Yunzhu Li

**Published:** 2026-11-03

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Predicting how the world evolves in response to robot actions is central to policy learning, evaluation, and planning. However, many action-conditioned world models predict future observations in 2D image or video space without explicitly modeling the underlying 3D scene, making spatial consistency across viewpoints and over long-horizon interactions challenging. We introduce 3D World Simulator, an action-conditioned multimodal world modeling framework that jointly predicts appearance, geometry, and tactile observations from a unified 3D representation. Our approach combines a hybrid particle-voxel autoencoder that compresses point clouds into compact 3D voxel latents with an action-conditioned latent dynamics model that predicts their evolution. The shared 3D representation supports geometrically consistent rendering from arbitrary viewpoints and flexible integration of spatially aligned tactile signals. Our experiments demonstrate interactive, long-horizon, multimodal prediction at 50 Hz. We then show how our model enables scalable data generation for policy training and evaluation using tactile and multi-view visual observations, including wrist-mounted and novel camera views. Furthermore, we scale our method to large datasets containing diverse objects and demonstrate model-based planning with unseen objects and robot configurations. Together, these results highlight the potential of hybrid particle-voxel dynamics as a foundation for multimodal 3D world simulation.

### 论文解读

#### 摘要翻译

这项工作提出混合体素与粒子的三维世界模拟器。模型把点云压缩到稀疏体素潜空间，再解码为可渲染的三维粒子，并在该空间预测动作条件下的视觉、几何和触觉变化。

#### 方法动机分析

二维视频容易丢失深度与遮挡关系，单纯粒子方法在长时预测中也可能积累几何误差。操控任务还需要预测触觉。作者的核心假设是体素适合建模规则空间动态，粒子适合高质量渲染和局部接触，两者组合可同时支撑几何和多模态预测。

#### 方法设计详解

输入是当前点云、机器人动作及过去潜状态。稀疏三维卷积编码器只处理占据体素，压缩为紧凑潜表示；解码器逐级上采样体素，并生成带颜色、尺度、旋转和透明度的粒子，支持新视角渲染。动力学模型以3D卷积和时空注意力预测未来潜状态。触觉信号被表示为三维粒子，沿用同一预测框架。


核心计算包含稀疏3D卷积与时空注意力；模型在当前潜状态、动作和历史状态条件下滚动预测，而不是逐像素外推整段RGB视频。

模型用稀疏三维卷积编码占据体素，并以动作、历史潜状态预测未来；约200步的开环评估检验误差累积。



训练时使用ALOHA示范学习动作条件动力学，推理按自回归方式滚动未来潜状态。
#### 方法对比分析

与2D视频世界模型相比，它显式预测几何并支持新视角；与纯点云预测相比，它保留可微渲染和触觉接口。对需要视角变化或接触反馈的桌面操控较有价值。代价是需要点云与触觉数据，体素计算昂贵，且长时开环滚动仍受训练任务分布限制。

#### 实验分析（精简版）

ALOHA六项任务的200步预测平均PSNR为23.24、LPIPS为0.240、Chamfer距离7.72、触觉L1为0.081，比较表中均优于基线。策略合成数据实验在部分任务接近真实数据表现。未见物体的Deform360图像指标较弱，显示外观泛化仍有限。


模型以动作条件和历史潜状态做自回归预测，论文将图像、深度、几何和触觉分别评测，并在多个视角考察渲染一致性。报告的200步开环指标比单帧指标更能体现累积误差，但具体政策任务仍仅覆盖有限工作台行为。
#### 实用指南

复现需要同步点云、动作与触觉；评估时应报告长时误差而非只看单步渲染。论文覆盖约200步开环与有限ALOHA任务。迁移到移动操作可借鉴体素状态和粒子渲染的分工，但需要重新验证地图范围、动态场景和真实触觉校准。


训练依赖高质量点云与同步触觉。换用腕部相机或移动底盘时，需重新验证坐标系对齐、可见区域变化与渲染新视角是否保持；不能仅凭桌面新视角结果推断全身机器人迁移。
#### 总结

核心思想：体素预测动态，粒子表达几何与触觉。
1. 稀疏编码点云。
2. 在潜空间预测动作后果。
3. 解码粒子和触觉。
4. 用长时任务评估模拟器。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.04492v1)
- [arXiv](https://arxiv.org/abs/2611.04492v1)

---

<a id='2611.04522v1'></a>
## [Action Intent as a Bridge: Transferring Manipulation Knowledge from Human-Centric Videos to Robots](https://arxiv.org/abs/2611.04522v1)

**Authors:** Zhiqian Xia, Lianyang Ma, Haifeng Xia, Siyu Xia, Wei Wang, Xiaochun Cao

**Published:** 2026-11-03

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models have achieved promising performance for generalized robotic manipulation, yet their reliance on expensive real-robot data remains a fundamental bottleneck. A scalable alternative is to transfer knowledge from off-the-shelf human-centric videos. However, existing methods extract latent action representations that are entangled with scene-specific noise, diluting the transferred knowledge. Motivated by this limitation, we propose Action-Intent Transfer (AIT), a two-stage method that transfers knowledge through a clean action representation shared across humans and robots. In the first stage, AIT extracts explicit action intents and disentangles the implicit action intent from manipulation videos. In the second stage, AIT aligns the human and robot domains within this shared intent space and decodes the intent into executable robot actions. Extensive experiments in both simulation and real-world environments demonstrate that AIT transfers human knowledge more effectively and consistently outperforms prior baselines.

### 论文解读

#### 摘要翻译

AIT提出Action-Intent Transfer，让人类视频中的操控知识经由显式和隐式动作意图迁移到机器人策略。显式意图提供动作语义锚点，隐式意图进一步保留连续运动细节；模型在共享意图空间预训练后再适配目标机器人。

#### 方法动机分析

人类视频数量丰富，却没有机器人动作标注；直接学习视频潜动作容易把背景、光照等场景变化混入控制表示。动作重定向又受人手与机器人形态差异影响。AIT假设人类与机器人共享“先规划意图、再执行动作”的层级结构，可以用它作为跨形态桥梁。

#### 方法设计详解

在人类视频中，视觉语言模型生成指令和原子动作标签；机器人数据则由末端位移与夹爪状态规则生成标签。第一阶段用DINOv2特征、动作标注和两个码本学习：先把场景变化编码到场景码，再冻结该码本，让隐式意图码解释剩余动作相关变化。第二阶段用Prismatic-7B联合预测显式、隐式意图，再在目标机器人数据上微调并用动作解码器生成低层控制。


实现中使用Prismatic-7B与DINOv2特征，序列长度分别为3个显式意图和4个隐式码；机器人适配阶段再学习动作解码器。

输入是视频、语言和机器人状态；DINOv2特征经过动作标注引导的场景/意图码本，再由Prismatic-7B预测意图并解码为动作。


#### 方法对比分析

与只用潜动作的方法相比，AIT增加可读的显式动作锚点，并以场景码分离背景变化；与动作重定向相比，它不要求把人手姿态逐关节映射到机器人。适合方向性指令和跨域操控预训练。原子动作词表较粗，对复杂接触和连续技能仍依赖隐式码及机器人数据。

#### 实验分析（精简版）

LIBERO四套件平均成功率91.3%，高于UniVLA的88.1%；12项Astribot实机任务平均82.08%，UniVLA为70.42%，每项20次试验。研究只覆盖一台机器人及定制任务集，控制频率8Hz是在RTX 4090上的报告值，不代表其他硬件。


显式意图词表包含六向移动、抓取和释放等原子动作，隐式码补足多轴运动细节；显式人类标注经人工核验，机器人标签按位移规则生成。论文报告训练包含多阶段长程优化，复现成本明显高于轻量监督微调。
#### 实用指南

论文报告显式意图序列长度3、隐式长度4，使用8张H100训练；潜变量标注质量和人类动作语义校验是复现重点。迁移到其他形态时，先定义共享原子意图，再设计从状态到意图的可靠标注规则。精密力控任务不能只靠方向标签。


迁移到轮式移动操作时，可把原子意图改为接近、绕行、接触等动作层次；这属于应用推断，需要新数据验证，不是本文实验结论。
#### 总结

核心思想：以共享意图桥接人类视频与机器人动作。
1. 生成显式意图标签。
2. 分离场景码和运动码。
3. 联合预训练共享策略。
4. 解码为机器人控制。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.04522v1)
- [arXiv](https://arxiv.org/abs/2611.04522v1)

---

<a id='2611.05163v1'></a>
## [RoboTryOn: Synthesizing Dexterous Robot Demonstrations from Human Videos via Virtual Try-On](https://arxiv.org/abs/2611.05163v1)

**Authors:** Seungyong Lee, Suhyeok Jang, Minyoung Kim, Jinwoo Shin

**Published:** 2026-11-03

**Categories:** cs.RO, cs.AI

**Abstract:**

Robot foundation models have shown remarkable performance in robot control, yet adapting them to a new task still relies heavily on teleoperated demonstrations. Egocentric human videos offer a scalable substitute, but overlaying rendered robots on human videos or retargeting human poses into robot actions leaves significant visual or action gaps for high-DoF robot embodiments. To address this limitation, we propose RoboTryOn, which treats human-to-robot translation as virtual try-on: try-on keeps the person and swaps the garment, RoboTryOn keeps the demonstrated motion and swaps the human for the robot. Specifically, RoboTryOn converts a human video into a video of the target robot specified by a reference image, while preserving the original task, objects, and background. To maintain visual consistency, we translate selected keyframes autoregressively and use them to guide video synthesis with motion conditioned on the human hand skeleton. Additionally, we use an inverse dynamics model (IDM) that incorporates depth features to infer action labels from synthesized videos for policy learning. Our experiments across multiple humanoid embodiments show that policies trained with our synthetic data improve the average success rate on novel tasks, including challenging bimanual manipulation, from 2% to 60%, without any real robot demonstration of those tasks. Moreover, training with our synthetic data matches the performance of policies trained solely on teleoperated demonstrations at approximately a third of the collection cost, demonstrating the cost efficiency of our approach.

### 论文解读

#### 摘要翻译

RoboTryOn将人类第一视角操作视频合成为目标机器人的演示视频，并用逆动力学模型补上机器人动作标签。它试图同时缩小人类与机器人之间的视觉差异和动作差异，让策略从无需机器人演示的新任务视频中学习。

#### 方法动机分析

人类视频容易获取，但人手形态、视觉外观和关节动作都不同于机器人；直接重定向手部姿态既不精确，也难表达抓取力。RoboTryOn的假设是先生成看起来像目标机器人的演示，再从生成视频估计动作，比把人手动作硬映射到机器人关节更可执行。

#### 方法设计详解

给定人类视频和机器人参考图像，先根据手腕轨迹选关键帧。图像模型用遮罩、骨架和前一机器人帧，自回归生成机器人关键帧；视频模型在关键帧间按手部骨架流合成连续机器人动作。最后逆动力学模型从生成视频估计关节动作，结合手腕重建和逆运动学修正末端位置，得到可用于策略训练的视觉、状态和动作序列。


实现由FLUX.2 9B关键帧模型、LTX-2.5视频模型和Depth Anything V2逆动力学模型组成；视频以20Hz处理，模型需要多卡GPU训练。

输入为人类视频与目标机器人参考图，图像生成模块自回归输出机器人关键帧；视频生成模块按骨架流补帧，逆动力学模块估计关节动作。模型以20Hz视频和20步动作片段训练。


#### 方法对比分析

EgoMimic把人手遮掉，Masquerade叠加渲染机器人，二者测试时也要编辑相机图像；RoboTryOn生成完整机器人视频，实机时可用原始相机视图。与直接重定向相比，动作由机器人视频逆推。适合桌面灵巧操作和双手任务，但依赖固定视角与可执行的人类动作。

#### 实验分析（精简版）

七项没有任务专属机器人演示的任务，加入合成数据后OpenArm平均成功率73.0%，遥操作单独训练3.1%；RB-Y1为46.9%对1.0%。每任务28次试验，包含新物体、目标、技能及双手任务。生成视频和伪动作仍可能失败，特别是力控和精细插入。


作者分别训练关键帧图像模型、视频模型和逆动力学模型，推理流水线需要先做人手重建，再逐帧生成与动作估计。一个15秒演示在单张H100上约需数分钟完成，因此其节省的是现场采集时间而非总计算成本。
#### 实用指南

模型由FLUX.2图像生成、LTX-2.5视频生成和Depth Anything V2逆动力学组成，训练需要多卡H100/H200。每增加一种机器人仍需数小时该形态视频；单个15秒演示在H100上的合成流水线约需数分钟。迁移前先核查相机视角、标定、动作可达性和生成失败率。


适合评估的下一步包括透明物体、移动相机、不同照明与接触力任务；论文已展示生成会丢物体、IDM会抓偏等失败类型，应用方应建立人工抽查与失败样本过滤。
#### 总结

核心思想：先把人类动作换成机器人视频，再推回动作。
1. 重建手部骨架并选关键帧。
2. 生成机器人关键画面。
3. 按骨架流补全视频。
4. 逆推动作并训练策略。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05163v1)
- [arXiv](https://arxiv.org/abs/2611.05163v1)

---

<a id='2611.05377v1'></a>
## [DreamX: Video Diffusion Model is a Universal Latent Action Learner](https://arxiv.org/abs/2611.05377v1)

**Authors:** Jisoo Kim, Chaehyun Kim, Junsu Kim, Jiho Park, Seongchan Kim, Eunbeen Hong, Soowon Son, Fu-En Yang, Seungryong Kim

**Published:** 2026-11-03

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Learning a universal action representation from video requires modeling dynamics that are consistent across different executions of the same action yet invariant to changes in embodiment and appearance. Existing latent action models infer actions from visual transitions, but their representations often entangle the underlying action with irrelevant visual variations, limiting transfer across embodiments. We introduce DreamX, which repurposes a pretrained video diffusion model as a universal latent action learner. Our approach extracts compact latent actions from the video model's rich spatiotemporal features, compressing each latent video frame by $\boldsymbol{300\times}$ ($\boldsymbol{1{,}200 \rightarrow 4}$ tokens). In addition, by leveraging the video model's generative prior, these latent actions capture action-relevant dynamics disentangled from visual appearance. The resulting latent space serves as a shared action interface across embodiments (human and robot), environments, and action parameterizations. On LARY-Bench, DreamX achieves state-of-the-art performance among latent action models. Moreover, actions encoded in one setting remain effective in another: the latent space supports any-to-any video transfer and zero-shot action transfer across diverse embodiments and environments. Furthermore, this shared interface also enables policy learning in a compact latent action space, reducing the computational cost of video-diffusion-based policies.

### 论文解读

#### 摘要翻译

DreamX从预训练视频扩散模型中提取紧凑动作潜变量，用这些变量表示观察到的时空运动，再由动作头预测机器人控制。这样可利用视频先验，而无需将未来视频生成作为唯一控制途径。论文还研究不同场景和形态间的视频与动作迁移。

#### 方法动机分析

视频扩散模型含有丰富动态知识，但直接用未来视频驱动动作会带来推理成本和视觉误差；普通潜动作又可能混入场景外观。DreamX假设，同一视频骨干既可编码观测动态，也可通过未来重建约束潜变量，让表示更集中于运动。

#### 方法设计详解

输入是当前视频帧、语言和可选机器人状态。模型在预训练视频骨干上添加可学习查询，每个潜帧分配4个查询，读取时空特征形成紧凑动作潜变量。解码器以初始帧和潜变量重建未来视频；有动作标签时，同时预测动作块。注意力掩码控制视频、查询和动作令牌间的信息流，避免动作与视频流直接串扰。下游策略可使用潜变量作为动作接口。


主设置使用Cosmos-Predict2.5 2B，潜变量每帧由4个查询构成；该表示比完整视频短得多，但仍通过未来重建保留时序信息。

主实验以Cosmos-Predict2.5 2B为视频骨干，查询潜变量每帧4个令牌；训练包括未来帧重建，部署时由动作头执行有限步采样。



训练阶段以大规模视频学习潜变量，动作模块再用机器人动作标签训练；推理时由查询潜变量预测动作。
#### 方法对比分析

与直接扩散未来视频的WAM相比，DreamX把视频动态压缩成较短的动作潜变量；与只做逆动力学的表示相比，它用生成式未来重建提供动态监督。适合跨场景、跨形态的运动迁移研究。其潜变量是否具有可组合的动作语义仍需检验，真实任务范围也较窄。

#### 实验分析（精简版）

H&R零样本人到机器人迁移中，DreamX的FVD为57.44、LPIPS为0.179，优于表中AdaWorld和VideoWorld2。真实抽屉任务子目标成功率96.7%，推理延迟1474毫秒，Mimic-Video为50%和28226毫秒。抽屉实验样本有限，这些结果不能代表普遍实机表现。


训练把潜变量未来帧重建和动作监督结合，视频特征每帧压缩为4个查询令牌；主实验使用Cosmos-Predict2.5 2B骨干。读者可将其理解为用生成器学习中间运动表征，再通过下游动作任务校准，而非把重建视频直接当控制指令。
#### 实用指南

论文使用预训练Cosmos-Predict2.5 2B，查询潜变量解码器与动作头共同训练；具体训练还需视频与机器人数据配对和动作接口定义。复现时应检验重建目标是否改善动作，而非只改善图像指标。迁移到自动驾驶可探索将交通参与者动态压缩成运动码，但需验证时序安全和多主体交互。


建议在复现中分开报告表征迁移质量、动作预测误差和实机成功率。三个指标之间有关联但并不等价，尤其视频相似度高不能证明动作执行可靠。
#### 总结

核心思想：把视频动态压成可供策略使用的潜动作。
1. 查询读取视频特征。
2. 以未来重建塑造潜变量。
3. 从潜变量解码动作。
4. 评测跨场景迁移。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05377v1)
- [arXiv](https://arxiv.org/abs/2611.05377v1)

---

<a id='2611.05080v1'></a>
## [CRONOS: Benchmarking Non-Episodic Reinforcement Learning for Robotic Manipulation](https://arxiv.org/abs/2611.05080v1)

**Authors:** Po-Yi Wu, Djengo Cyun-Jyun Fang, Dian Cheng, Tsung-Wei Ke

**Published:** 2026-11-03

**Categories:** cs.RO, cs.LG

**Abstract:**

Reinforcement learning (RL) is promising for adapting robot policies to unstructured real-world environments. However, standard RL pipelines rely on episodic training with frequent scene resets, which is impractical for real-world deployment due to the need for substantial human intervention. While prior work has studied non-episodic RL in relatively small policies trained from scratch, its behavior when fine-tuning modern large-scale pre-trained vision-language-action (VLA) models remains underexplored. In this work, we introduce CRONOS, a simulation benchmark for RL fine-tuning of VLAs under non-episodic scene resets. CRONOS leverages high-fidelity physics simulators and adopts shared-scene multi-task settings. Using CRONOS, we show that naively fine-tuning pre-trained policies fails in non-episodic settings due to shifts in object configurations and end-effector poses; however, these challenges can be mitigated through targeted reset allocation and by addressing biases in pre-trained models. Moreover, we find that non-episodic training enhances long-horizon manipulation and improves generalization to previously unseen object configurations and task sequences. Project page: https://embodiedai-ntu.github.io/cronos/

### 论文解读

#### 摘要翻译

CRONOS是用于研究稀疏场景重置下机器人强化学习的仿真基准。它发现预训练视觉语言动作策略在连续任务中会受到物体布局和机械臂初始位姿漂移影响，并研究末端重置、重置课程和不可恢复场景检测等策略。非情景式训练也改善了部分分布外泛化。

#### 方法动机分析

真实机器人训练频繁人工整理场景代价高，但仿真强化学习常默认每个任务都能免费重置。少重置时，前一任务留下的物体位置和机械臂状态会改变后续任务分布。CRONOS提出先量化这种漂移，再比较哪些重置能改善交互效率，而不是把“少重置”本身当成成功。

连续任务的主要瓶颈是场景和机械臂初始状态逐步偏离预训练分布；作者假设分层复位能减轻这种分布漂移。


#### 方法设计详解

基准包含4个共享场景、16个抓放任务，每个任务80步。情景训练每80步重置，非情景训练每320或2560步重置。研究加入末端执行器归位以消除机器人位姿偏差；用前期频繁、后期稀疏的课程降低场景分布突变；启发式重置在物体离桌或超出工作空间时恢复场景。比较OpenVLA和SpatialVLA，并评估重叠、杂乱布局及连续任务。


实现使用PPO微调OpenVLA-7B或SpatialVLA-4B，并以LoRA限制更新规模；重置策略改变的是任务之间的场景转移分布。

输入是共享场景中的语言任务与图像观测，策略输出动作；环境按间隔继续或重置，再依据状态触发额外归位和恢复。训练配置包括64个并行环境与PPO更新。


#### 方法对比分析

与传统每任务独立场景基准相比，CRONOS让任务共享场景，误差可跨任务累积；与学习式复位策略相比，启发式检测直接处理不可恢复状态，但使用模拟器特权信息。该基准适合分析训练重置预算和分布漂移，不能直接代表真实世界人工复位机制。

#### 实验分析（精简版）

SpatialVLA在重叠布局的成功率由情景训练55.2%升至非情景训练64.3%，杂乱布局由51.2%升至55.0%。长间隔下直接训练会明显退化，针对性重置可恢复表现。所有结果来自ManiSkill仿真，HSR依赖特权状态，仍需感知式检测。


作者采用OpenVLA-7B与SpatialVLA-4B、PPO及LoRA；训练使用64个并行仿真环境。结果揭示策略对训练初始姿态和布局分布的敏感性，而非提出一个可直接用于真机的通用复位器。
#### 实用指南

复现时保持任务顺序、重置预算和交互步数一致，单独报告末端重置与场景重置。论文设置64并行环境、PPO更新，并对两种VLA做LoRA微调。迁移到真实机器人可把“不可恢复场景”定义为视觉可检测事件，但检测漏报、人工成本与安全复位轨迹都需重新评估。


将HSR迁移到真实平台需要由相机或传感器判断物体是否可恢复，还要规划无碰撞的回收动作；仿真中特权状态检测不能替代这部分工程。
#### 总结

核心思想：用有针对性的复位应对连续训练漂移。
1. 共享场景连续做任务。
2. 记录状态与布局漂移。
3. 选择归位和场景重置。
4. 测试泛化与长程成功。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05080v1)
- [arXiv](https://arxiv.org/abs/2611.05080v1)

---

<a id='2611.05295v1'></a>
## [GSBR: Geometric Support Behavior Regularization for Online RL](https://arxiv.org/abs/2611.05295v1)

**Authors:** Charles Xu, Jacob Berg, Liyiming Ke, Sergey Levine

**Published:** 2026-11-03

**Categories:** cs.RO, cs.LG

**Abstract:**

Regularizing online reinforcement learning (RL) toward a pretrained base policy can improve sample efficiency and learning stability, but regularization that is too restrictive limits how much the policy can improve. Base policies -- typically diffusion or flow-matching models trained by imitation learning on large, diverse datasets -- capture many behaviors that are reasonable but suboptimal. Existing regularizers using such base policies are too restrictive in one of two ways: behavioral cloning regularizers pull the actor toward the full base-policy distribution, penalizing it for discarding suboptimal modes, while hard support constraints forbid improvements that lie just outside the base policy's support. We propose Geometric Support Behavior Regularization (GSBR), which penalizes actions by their distance from the base policy's support. Actions within the support incur no penalty, so the actor can concentrate on high-value modes; outside the support, where value estimates are less reliable, the penalty grows smoothly with distance while still permitting improvement when the value gained outweighs the penalty. GSBR estimates this distance from a finite set of base-policy samples, so it requires only sample access to the base policy. Across 19 tasks on four robotic benchmarks, GSBR raises the average success rate from 33% for the base policy to 95%, achieving the highest average success rate and throughput among all compared methods, with about 23% higher normalized throughput than the next-best RL method. Project website and open-source code at https://charlesxu0124.github.io/gsbr/ .

### 论文解读

#### 摘要翻译

GSBR是一种生成式机器人策略的行为正则化方法。它从基础策略动作样本构造自适应支持区域，只对区域外的动作施加距离惩罚，允许策略在区域内选择不同动作模式并做局部改进。作者在19项任务上报告了较高成功率与吞吐。

#### 方法动机分析

行为克隆策略能提供有用先验，但强化学习微调可能因过度偏离训练分布而失稳。把每个动作都拉向一个示例会限制多模态选择，硬性概率约束又不适合常见生成式策略。GSBR的核心假设是：基础策略样本附近的局部区域代表可信行为，区域外才需要持续惩罚。

#### 方法设计详解

给定状态，先从基础策略采样一组动作，计算样本间最近邻距离，并用其分位数估计局部尺度。以每个样本为球心、该尺度为半径，球的并集形成支持区域。策略动作落在并集中时不受支持损失；落在外面时，按其到并集最近点的欧氏距离惩罚。该损失可与任务奖励共同训练生成策略。理论分析给出近似误差界，并说明样本数量增加时误差趋零。


训练时将这一支持损失并入强化学习目标，推理时仍由策略产生动作；局部半径随基础策略样本的密度变化，而不是使用统一固定尺度。

训练时把正则项并入强化学习策略目标，推理时仍由策略直接采样动作；不需要把策略压回某一个参考动作。


#### 方法对比分析

相较将策略分布限制在单一参考动作附近的方法，GSBR保留样本邻域内的模式切换与局部优化；相较硬性截断，它对生成策略可微且更宽松。适合需要保留预训练策略能力、又希望强化学习做有限改进的场景。关键代价是需要维护参考样本并计算距离，半径分位数也需按任务调节。

#### 实验分析（精简版）

19项任务平均成功率为95%，RLT为92%；归一化吞吐为0.95，对照RLT为0.77。LIBERO上GSBR成功率85%，RLT为86%，但吞吐有竞争力。参考样本数8已表现良好，半径设为0时吞吐降到0.66。实验支持局部支持约束有用，但没有真实机器人验证。


本文的支持区域是基础策略样本的几何邻域，并不等于真实环境的安全集合；分布外碰撞或不可行动作仍可能落在该邻域。使用者应把GSBR与动作限幅、碰撞检查等安全机制分开评估。报告的吞吐是任务执行效率度量，不应误读为每秒控制周期。
#### 实用指南

实现时要在同一状态下采样基础动作，统一动作尺度后估计邻居距离，并检查球并集查询开销。论文在多种仿真任务和LIBERO评估；核心消融提示不要把半径压到零。迁移到新平台前，应确认基础策略样本覆盖可行动作，并测量控制频率、采样成本和动作安全边界。


当前证据适合判断算法在基准上的有效性，不足以确定部署端时延。复现建议先做不同样本数和半径分位数的敏感性检查，再测策略更新稳定度。
#### 总结

核心思想：只惩罚偏离可信行为区域的动作。
1. 从基础策略采样。
2. 估计局部支持半径。
3. 构造样本球并集。
4. 训练时惩罚区域外动作。

**Links:**

- [PDF](https://arxiv.org/pdf/2611.05295v1)
- [arXiv](https://arxiv.org/abs/2611.05295v1)

---

