time: 20260821

# Arxiv Computer Vision Papers - 2026-08-21

## Executive Summary

# 计算机视觉 Arxiv Daily 执行摘要  
**日期：2026 年 8 月 20 日**  
**论文数量：10 篇**

> 注：以下判断主要依据论文标题及研究方向归纳；具体方法、实验结果和结论应以论文摘要及正文为准。

## 1. 主要主题与趋势

### 1）视频生成与 4D 人体建模
- **4DAnyone** 聚焦从普通单目视频生成任意人物的 4D 表示，体现了从静态 3D 重建向**动态、可生成的人体数字资产**发展的趋势。
- **DreamHand** 将视频扩散模型重新用于自中心视角的 3D 手部运动恢复，说明大规模视频生成模型正在被迁移到**遮挡严重、观测不完整的姿态估计问题**。

### 2）视觉感知与机器人操作深度融合
- **Video2DoorTraversal**、**DECOWAM**、**RoMAN-Flow** 和 **Learning Highly Dynamic Skills Transition...** 共同体现了视觉研究与机器人控制、强化学习及世界模型的进一步结合。
- 研究重点从“识别物体”转向“理解环境并完成动作”，尤其关注：
  - 门等可交互物体的操作；
  - 全身移动操作；
  - 四足机器人动态跳跃；
  - 离线强化学习中的动作分布建模。

### 3）面向规划的自动驾驶与具身智能
- **Planning-Oriented End-to-End Autonomous Driving** 直接讨论自动驾驶系统从感知驱动向**规划目标驱动的端到端架构**转变。
- **DECOWAM** 和机器人运动规划论文也反映出具身智能研究正逐渐强调完整闭环：视觉理解、状态预测、动作生成和反馈控制，而非单一模块性能。

### 4）视频理解与时序定位
- **ID-VTG** 关注图像辅助的视频时序定位，可能针对仅凭视频内容难以消除的语义歧义，体现了**跨模态信息辅助视频理解**的发展方向。

### 5）局部特征与几何视觉基础能力
- **Unified and Efficient Point-Line Local Features** 重新审视点特征与线特征的统一表示，说明传统几何视觉仍是 SLAM、三维重建、匹配和机器人定位的重要基础，并正朝着**更高效、更统一的局部特征学习**发展。

### 6）人形机器人运动技能
- **Towards Professional Tennis Styles for Humanoid Robots...** 将自适应运动规划与轨迹跟踪用于网球风格动作，体现了人形机器人从基础行走、抓取向**高动态、具有风格和任务特异性的运动技能**拓展。

---

## 2. 特别值得关注的论文

### **4DAnyone**
该工作代表了“从视频创建可控动态人物”的重要方向。若其方法能够在普通单目视频下同时保持身份一致性、时空稳定性和可编辑性，将对数字人、虚拟现实、影视制作和人机交互产生较大影响。其关键价值在于降低高质量 4D 人体建模对多视角采集和专业设备的依赖。

### **DreamHand**
手部动作恢复长期受到遮挡、视角变化和手指细粒度运动的限制。将视频扩散模型用于自中心视角 3D 手部恢复，可能为“生成式先验 + 运动估计”提供有代表性的范式，尤其适用于第一视角操作视频、虚拟现实和人机交互。

### **DECOWAM**
“解耦的全身世界—动作模型”直接面向腿式移动操作这一复杂场景，涉及移动、全身协调、环境建模和任务执行的联合问题。如果其解耦设计能提升训练稳定性、泛化能力或长时域规划效果，对具身智能和通用机器人控制具有较高参考价值。

### **Video2DoorTraversal**
推门穿越看似简单，但同时要求机器人理解门的状态、估计交互动力学并完成连续动作。利用“模拟门双胞胎”进行训练或规划，代表了通过**仿真中可控的交互对象模型**解决真实世界数据稀缺问题的路线，值得关注其仿真到现实迁移效果。

### **Planning-Oriented End-to-End Autonomous Driving**
这篇论文更偏综述或观点性工作，可能有助于快速理解当前端到端自动驾驶在架构设计、评价指标和未来范式上的变化。对于从事自动驾驶系统、世界模型或规划学习的研究人员，具有较高的信息整合价值。

---

## 3. 正在形成的研究方向与技术路线

1. **生成模型作为视觉运动先验**  
   扩散模型不再局限于图像或视频生成，而是被用于 3D 姿态恢复、动作补全、遮挡处理和机器人策略学习。

2. **从 3D 表示走向 4D 动态世界建模**  
   人体、物体和机器人环境都开始以时空连续的 4D 形式建模，以支持预测、编辑和交互。

3. **世界模型驱动的机器人操作**  
   通过学习环境状态转移、物体交互动力学和动作后果，机器人系统正在从反应式控制转向预测式规划。

4. **仿真对象双胞胎与仿真到现实迁移**  
   对门、工具和其他可交互物体构建可控模拟副本，有望降低真实交互数据采集成本，并支持安全训练。

5. **全身协调与高动态机器人技能**  
   研究从单臂操作扩展至移动底盘、腿部、躯干和双臂的协同控制，同时覆盖跳跃、网球等高动态任务。

6. **规划导向的端到端学习**  
   自动驾驶和机器人学习都在重新强调规划、约束、可验证性和闭环评价，而不仅是感知精度或短期动作预测。

7. **统一的几何特征表示**  
   点、线等多种几何结构可能在统一特征空间中联合建模，以兼顾匹配鲁棒性、计算效率和几何约束。

---

## 4. 建议优先阅读全文的论文

### 第一优先级：适合大多数计算机视觉与具身智能研究者
1. **4DAnyone**  
   适合关注 4D 重建、数字人、视频生成和动态场景建模的读者。

2. **DreamHand**  
   适合研究人体姿态估计、手部跟踪、扩散模型和第一视角视觉的读者。

3. **DECOWAM**  
   适合研究世界模型、机器人学习、移动操作和多模态具身智能的读者。

4. **Planning-Oriented End-to-End Autonomous Driving**  
   适合希望快速掌握端到端自动驾驶研究趋势、评价方式和系统架构的读者。

### 第二优先级：适合机器人与控制方向研究者
5. **Video2DoorTraversal**  
   重点关注可交互物体建模、视觉运动规划和仿真到现实迁移。

6. **RoMAN-Flow**  
   适合关注离线强化学习、归一化流策略建模和机器人操作策略学习的读者。

7. **Learning Highly Dynamic Skills Transition...**  
   适合研究四足机器人、约束控制、跳跃技能和复杂地形运动的读者。

8. **Towards Professional Tennis Styles for Humanoid Robots...**  
   适合关注人形机器人、自适应运动规划、轨迹跟踪及高动态动作生成的读者。

### 第三优先级：适合视频理解与几何视觉研究者
9. **ID-VTG**  
   适合研究视频时序定位、图像—视频联合理解和跨模态消歧的读者。

10. **Unified and Efficient Point-Line Local Features**  
    适合研究局部特征、视觉匹配、SLAM、三维重建和几何计算效率的读者。

## 总结

本期论文的核心信号是：计算机视觉正快速从独立的感知模块，转向服务于**动态世界建模、机器人行动和闭环规划**。扩散模型、4D 表示、世界模型、仿真对象双胞胎以及规划导向的端到端学习，是最突出的技术关键词。若时间有限，建议优先阅读 **4DAnyone、DreamHand、DECOWAM、Video2DoorTraversal**，并以 **Planning-Oriented End-to-End Autonomous Driving** 作为领域趋势综述入口。

---

## Table of Contents

1. [4DAnyone: Create Anyone in 4D from a Casual Monocular Video](#2608.20335v1)
2. [DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery](#2608.20308v1)
3. [Video2DoorTraversal: Push Door Traversal via Simulated Door Twins](#2608.20251v1)
4. [RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation](#2608.20208v1)
5. [ID-VTG: Image-Disambiguated Video Temporal Grounding](#2608.20127v1)
6. [DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation](#2608.20114v1)
7. [Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms](#2608.20111v1)
8. [Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking](#2608.20087v1)
9. [Learning Highly Dynamic Skills Transition for Quadruped Jumping Through Constrained Space](#2608.19977v1)
10. [Unified and Efficient Point-Line Local Features](#2608.19894v1)

---

## Papers

<a id='2608.20335v1'></a>
## [4DAnyone: Create Anyone in 4D from a Casual Monocular Video](https://arxiv.org/abs/2608.20335v1)

**Authors:** Yudong Jin, Tao Xie, Qihang Zhang, Zehong Shen, Zhen Xu, Yujun Shen, Hujun Bao, Xiaowei Zhou, Yinghao Xu

**Published:** 2026-08-20

**Categories:** cs.CV

**Abstract:**

We present 4DAnyone, a framework for reconstructing 4D humans from an uncalibrated monocular video by generating reconstruction-grade multiview-consistent videos and lifting them into 4D Gaussian Splatting (4DGS). Existing camera-controlled video diffusion models synthesize plausible novel-view videos but fail to maintain consistency when scaled to the tens of target views required for 4DGS reconstruction. We identify this failure as a bounded-attention-context problem: when target views exceed the capacity of a single DiT forward pass, they must be split into groups, exposing two coupled bottlenecks. On the reference-context side, conditioning on all previously generated views grows as $O(N)$, weakening cross-view appearance guidance. On the target-context side, disjoint groups cannot directly exchange information, causing global structural drift. 4DAnyone addresses both bottlenecks with two complementary designs: Reference Context Packing (RCP) compresses growing reference views into a fixed-length mixed-resolution context with $O(1)$ reference-context complexity, while Target Context Routing (TCR) rotates target-view groupings during denoising to share context across groups at high-noise steps and stabilize details at low-noise steps. We further build the MVGameHuman dataset using our in-house game engine and combine it with light-stage and in-the-wild video datasets for training. Experiments on DNA-Rendering and DyMVHumans show that 4DAnyone outperforms prior methods in both novel-view video quality and downstream 4DGS reconstruction, with robust in-the-wild generalization. See our project page for video results and source code: https://4danyone.github.io.

**Analysis:**

# 1. 摘要翻译

本文提出 **4DAnyone**，旨在从一段未经标定的单目视频中重建可从任意视角观看的动态人体。方法首先生成具有重建质量的、多视角一致的人体视频，再利用4D Gaussian Splatting（4DGS）进行动态人体重建。现有相机控制视频扩散模型虽然能够生成合理的新视角视频，但当目标视角增加到4DGS所需的十几个甚至几十个时，容易出现外观不一致和结构漂移。作者将其归因于DiT单次前向传播的注意力上下文受限：参考视角数量增长会造成参考上下文复杂度为 \(O(N)\)，而将目标视角拆分为多个组后，各组之间又无法直接交换信息。为此，4DAnyone提出 **参考上下文打包（RCP）**，将不断增加的参考视频压缩为固定长度的多分辨率上下文，使参考开销降为 \(O(1)\)；同时提出 **目标上下文路由（TCR）**，在高噪声阶段动态轮换视角分组以传播全局结构，在低噪声阶段固定相邻分组以稳定局部细节。方法采用深度缓冲的3D人体骨架作为稀疏但可靠的几何条件，并构建MVGameHuman数据集。实验表明，该方法在新视角视频质量和下游4DGS重建方面均优于已有方法。

# 2. 方法动机分析

**驱动力**：单目视频缺少背面和侧面观测，直接重建会产生严重的遮挡区域缺失；但直接依赖相机控制扩散模型生成大量视角，又会破坏跨视角一致性。

**现有痛点**有两个：  
1. 参考视角越积越多，全部以高分辨率输入会造成上下文和计算量线性增长；  
2. 目标视角分组独立去噪，组间没有结构通信，容易出现人体形体、服装和纹理漂移。  

**核心假设**：人体的精确结构主要由可靠的3D骨架约束，而外观可以由视频生成模型补全；扩散早期决定全局结构，后期主要细化局部外观，因此两个阶段应采用不同的视角通信策略。

# 3. 方法设计详解

## 整体流程

1. **输入与人体运动估计**：输入 casual monocular video，使用GVHMR估计地面坐标系下的SMPL-X人体序列，并提取骨架关键点。  
2. **构造目标视角几何条件**：预先指定环绕人体的目标相机视角，将3D骨架投影到每个视角，并使用z-buffer进行深度遮挡处理，生成深度缓冲骨架视频。  
3. **3D骨架编码**：骨架编码器 \(g_\phi\) 将骨架视频映射为与DiT潜变量同分辨率的残差，并加入噪声目标潜变量：
\[
\tilde z_i^t=z_i^t+g_\phi(S_i).
\]
这不是从骨架直接生成图像，而是为扩散模型提供明确的空间和姿态提示。编码器由10层3D卷积组成，最终投影层零初始化，以避免破坏预训练Wan2.2模型的初始能力。  
4. **生成参考视角**：先用源视频生成若干覆盖不同方位的参考视频。视角通过最远点采样选择，优先补充与已有视角角距离最大的方向。  
5. **Reference Context Packing**：源视频使用普通patchify，参考视频使用2倍或4倍空间压缩的patchify层，将多个参考视角压入固定数量的token槽位：
\[
C_R=[P_1(V_{src}),P_2(V_a),P_4(V_b)].
\]
邻近视角通常存在冗余，因此牺牲部分空间分辨率即可保留整体布局和外观信息，避免上下文随视角数增长。  
6. **Target Context Routing**：剩余目标视角按每组4个进行联合去噪。高噪声阶段，每一步循环移动视角索引并重新分组，使同一视角在不同时间与不同视角共同注意，从而跨组传播结构；低噪声阶段固定相邻视角分组，减少细节闪烁和视角边界不连续。默认20步去噪中前16步动态路由、后4步固定分组。  
7. **4D重建**：生成全部多视角视频后，使用FreeTimeGS训练4DGS。其初始化依赖预测前景掩码的空间雕刻，最终获得可自由视点渲染的动态人体。

## 模型协同方式

DiT同时包含视频注意力、跨视角注意力和文本交叉注意力：视频注意力维持时间一致性，多视角注意力交换同一时刻不同视角的信息，RCP提供外观参考，骨架残差提供几何约束，TCR则在显存受限时实现跨组结构传递。

# 4. 方法对比与创新

与直接单目4D重建相比，4DAnyone通过生成不可见区域的多视角观测提升外观完整性；与深度/点云条件的视频扩散相比，它不依赖难以准确获得的度量深度和相机参数，而采用更稳健的3D骨架。其真正创新不只是“加入骨架”，而是针对**大规模多视角生成的上下文瓶颈**，同时从参考端和目标端进行设计：RCP解决“参考太多”，TCR解决“目标组不通信”。该方法最适合输入相机运动较小、人体主体清晰、需要自由视点视频或4D数字人的场景。

# 5. 实验分析

作者在DNA-Rendering和DyMVHumans上，以16个目标视角、4DGS重建质量、生成视频内部一致性及生成视频与真实视频的差异进行评估。主要结论是：4DAnyone在三类指标上整体领先，说明其优势不仅体现在单帧视觉质量，也体现在真正有利于4D重建的跨视角一致性。消融实验表明，去除RCP或TCR都会明显降低一致性，二者具有互补性；3D骨架的深度缓冲能够解决前后遮挡歧义。

局限是：大幅远离身体的宽松衣物缺少骨架约束，难以保持多视角一致；HMR姿态估计错误会被扩散模型忠实继承。

# 6. 实用指南

论文给出项目主页和源代码链接，但具体开源内容需以网页实际发布状态为准。复现关键点包括：以Wan2.2-TI2V-5B为基础，训练分三阶段；分辨率704×1280，学习率 \(10^{-5}\)，LPIPS权重0.25，20步采样，TCR切换比例 \(t_s/T=0.2\)。需使用GVHMR生成3D骨架，训练时采用多尺度patchify，并在单目数据阶段移除手指关键点。完整流程约需2分钟预处理、7分钟视频生成和30分钟4DGS训练。迁移到其他结构化对象时，可将人体骨架替换为可靠的3D关键点、关节或稀疏几何控制信号；RCP/TCR也可迁移到多视角物体、机器人或动态场景生成。

# 7. 总结

**核心思想：用骨架约束和上下文路由生成一致多视角。**

**速记版Pipeline：**
1. 从单目视频估计3D人体骨架。  
2. 把骨架投影到预设环绕视角，作为几何引导。  
3. 先生成少量参考视角，并压缩成固定长度上下文。  
4. 分组生成其余视角，高噪声轮换分组、低噪声固定邻组。  
5. 用生成的多视角视频训练4DGS，实现自由视点渲染。

**Key Findings:**

- We present 4DAnyone, a framework for reconstructing 4D humans from an uncalibrated monocular video by generating reconstruction-grade multiview-consistent videos and lifting them into 4D Gaussian Splatting (4DGS).
- Existing camera-controlled video diffusion models synthesize plausible novel-view videos but fail to maintain consistency when scaled to the tens of target views required for 4DGS reconstruction.
- Experiments on DNA-Rendering and DyMVHumans show that 4DAnyone outperforms prior methods in both novel-view video quality and downstream 4DGS reconstruction, with robust in-the-wild generalization.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20335v1)
- [arXiv](https://arxiv.org/abs/2608.20335v1)

---

<a id='2608.20308v1'></a>
## [DreamHand: Repurposing Video Diffusion Models for Occlusion-Robust Egocentric 3D Hand Motion Recovery](https://arxiv.org/abs/2608.20308v1)

**Authors:** Yufei Liu, Xixi Wang, Hao Li, Ganlong Zhao, Kaitong Cai, Chengkai Jin, Chunxiao Liu, Jianbo Liu, Siyuan Huang, Xingang Pan, Hongsheng Li

**Published:** 2026-08-20

**Categories:** cs.CV

**Abstract:**

Egocentric video offers scalable manipulation data for embodied AI, yet recovering metric 3D hand trajectories remains challenging due to severe object occlusion and frequent out-of-sight gaps. Existing single-frame and windowed temporal regressors fail when hand shortly leaves the frame, while recent video diffusion models (VDMs) rely on heavy, stochastic multi-step sampling as pixel-space renderers. We instead repurpose VDM into a deterministic geometry encoder. A single forward pass over the clean latent exposes scene content beyond current observations, including occluded and out-of-sight hands. We introduce DreamHand, an offline clip-level framework that extracts features via a Deterministic Clean-Latent Encoder and decodes them with a Bidirectional Spatiotemporal Decoder. DreamHand recovers continuous bimanual trajectories with metric placement and no external detector, while a Ray-Based Camera Solver supports a second configuration that needs no test-time camera intrinsics. Across five egocentric benchmarks, DreamHand sets a new state of the art, cutting MPJPE-p by 30% on occlusion-heavy ARCTIC and 40% on HOT3D. These gains reach 46%-61% once out-of-sight hands are included in the evaluation, offering a scalable path from everyday human video to robot manipulation data.

**Analysis:**

# 1. 摘要翻译

第一视角视频为具身智能提供了可规模化的操作数据，但由于严重的物体遮挡和频繁的视野外间隔，从视频中恢复具有度量尺度的三维手部轨迹仍十分困难。现有单帧方法和窗口式时序回归器在手部短暂离开画面后往往失效；近期视频扩散模型则主要作为像素渲染器，需要昂贵且随机的多步采样。本文将视频扩散模型重新设计为确定性的几何编码器：对干净潜变量进行一次前向传播，即可从特征中读取当前观测之外的场景内容，包括被遮挡和离开视野的手。作者提出DreamHand，由确定性干净潜变量编码器和双向时空解码器组成，在无需外部检测器的情况下恢复连续的双手度量轨迹；同时设计基于射线的相机求解器，使测试时可以不提供相机内参。实验表明，该方法在五个第一视角手部基准上取得领先结果，并显著改善遮挡和视野外场景下的恢复效果。

# 2. 方法动机分析

**驱动力与痛点：**第一视角手部运动具有两个特殊难点：手会被交互物体遮挡，或因头部快速转动完全离开视野。逐帧检测、裁剪再回归的方法没有缺失帧的恢复能力；因果视频模型只能利用过去信息，重新出现时容易发生身份切换和轨迹跳变；直接回归相机坐标中的三维位置，还会把手部运动与相机运动混在一起，并依赖准确内参。

**核心假设：**预训练视频扩散模型虽然为生成而训练，却已经学习了物体持续性、动作规律和遮挡关系。因此，与其让它耗时采样生成RGB，不如直接读取其中间特征，并用三维监督将其改造成几何表示。该任务本质上是利用完整视频的**离线双向时空推理**，而不是单帧估计或在线跟踪。

# 3. 方法设计详解

## 3.1 整体流程

1. 输入81帧RGB视频，先经冻结的Wan VAE编码为干净视频潜变量。  
2. 在噪声水平σ=0下，将潜变量送入Wan DiT，仅执行前30层中的前16个模块，在第15层读取特征，避免扩散采样。输出为21个潜在时刻、每个空间网格含3072维特征的时空特征。  
3. 通过LoRA和可训练Patch Embedding对扩散特征进行端到端适配，使其从“生成语义特征”转变为“几何感知特征”。  
4. 双向时空解码器对每个潜在时刻使用48个查询：2个手查询、42个关节查询和4个寄存器查询。关节查询通过空间交叉注意力从对应特征图中提取局部信息，再通过跨帧时序自注意力整合完整视频上下文。  
5. 解码器输出MANO全局旋转、关节姿态、关节三维位置、手部存在性和可见性；形状参数β在整个视频上池化，仅预测一次，从而避免形状和尺度抖动。  
6. 关节查询的注意力权重经过soft-argmax得到二维关节锚点。模型不直接回归完整平移，而是预测光学深度，再利用多关节几何约束求解平面内平移。  
7. Ray Head从扩散特征预测整张单位视线场。标准配置用真实内参计算射线；K-free配置则从预测射线场获得关节bearing，因此测试时无需输入内参。  
8. 通过混合PnP最小化多个关节的投影误差，求得相机坐标系下的三维平移；有效关节过少或重投影残差过大时，退化为沿手腕射线放置。

## 3.2 关键设计

- **空间接地查询：**每个关节拥有专属查询，而不是将整只手池化为一个向量，提升遮挡条件下的局部关节定位能力。  
- **双向注意力：**当前帧可同时利用过去和未来信息，因此手离开视野时能够根据离开前后的动作状态进行插值，而非单向外推。  
- **射线位置编码：**将预测视线方向编码后加入空间位置编码，使解码器显式感知相机几何。  
- **混合PnP：**只回归深度，把横向位置交给可解释的多关节最小二乘求解，降低直接平移回归的漂移。  
- **联合损失：**同时监督旋转、MANO姿态、根相对和相机坐标关节、二维投影、平移、存在性、时序加速度及射线方向。

# 4. 方法对比与创新

其本质区别不是“增加一个时序模块”，而是改变视频扩散模型的使用方式：从多步生成器变成一次前向的几何编码器；同时从“检测—裁剪—回归”转向全画面、全片段、双向恢复。主要创新包括：  
1. 干净潜变量上的确定性扩散特征读取；  
2. 空间接地查询与双向时空解码结合，用于视野外轨迹恢复；  
3. 射线场和混合PnP解耦姿态、深度与相机内参；  
4. 在训练中直接监督视野外手，而不是将其屏蔽。

最佳场景是离线数据采集、机器人模仿学习和长视频动作重建；不适合对低延迟闭环控制有严格要求的在线系统。

# 5. 实验分析

作者在ARCTIC、HOT3D、HOI4D、H2O和OakInk2上评估，并与十种单帧及视频基线比较，同时进行特征源、LoRA、查询形式、位置编码和相机求解消融。

代表性结论：  
- 相比ViDiHand，MPJPE-p在ARCTIC和HOT3D分别下降约30%和40%；视野外轨迹误差下降约46%–61%。  
- 单张A100上达到约63 fps，而ViDiHand约1.91 fps，速度约提升33倍。  

优势是遮挡/OOS鲁棒、轨迹连续、无需外部检测器，并能提供K-free配置。局限是离线依赖未来帧；K-free在广角图像边缘的射线估计不稳定，且OOS指标主要是腕部对齐误差，不能证明绝对空间位置准确。

# 6. 实用指南

代码已公开：`github.com/ggxxii/dreamhand`。复现需准备Wan2.2-Fun-5B-Control、Wan VAE、MANO模型及统一格式的数据。关键设置为81帧输入、21个潜在时刻、DiT第15层特征、LoRA rank=64、20k步AdamW训练；标准模型使用16张A100，K-free使用8张。视频通常按30 fps处理，并按数据集重新缩放分辨率和相机内参。迁移到人体、物体轨迹或机器人动作时，可保留扩散编码器和射线分支，将查询、几何头及监督目标替换为相应关键点、姿态或形状参数。

# 7. 总结

**核心思想：**把视频扩散模型变成几何编码器。

**速记版Pipeline：**
1. 用扩散模型一次读取整段视频特征；  
2. 用专属关节查询结合前后帧恢复动作；  
3. 预测二维锚点、姿态、深度和射线方向；  
4. 用多关节投影求解三维平移；  
5. 输出连续的双手MANO轨迹，并填补遮挡和视野外间隔。

**Key Findings:**

- We introduce DreamHand, an offline clip-level framework that extracts features via a Deterministic Clean-Latent Encoder and decodes them with a Bidirectional Spatiotemporal Decoder.
- Across five egocentric benchmarks, DreamHand sets a new state of the art, cutting MPJPE-p by 30% on occlusion-heavy ARCTIC and 40% on HOT3D.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20308v1)
- [arXiv](https://arxiv.org/abs/2608.20308v1)

---

<a id='2608.20251v1'></a>
## [Video2DoorTraversal: Push Door Traversal via Simulated Door Twins](https://arxiv.org/abs/2608.20251v1)

**Authors:** Xincheng Tang, Yiji Chen, Youhan Xie, Wanyu Li, Zhengjie Shu, Lai Jiang, Wenkang Hu, Yitong Li, Jinchuang Zhang, Xibin Song, Ruigang Yang

**Published:** 2026-08-20

**Categories:** cs.RO

**Abstract:**

Door opening and traversal is a long-horizon loco-manipulation task that requires precise handle interaction and coordinated base-arm control. We present Video2DoorTraversal, a single-video real-to-sim-to-real framework for wheel-legged mobile manipulators. Given one RGB video of a real door, DoorTwin reconstructs an instance-aligned, articulated, and simulation-ready door twin with realistic geometry and appearance. A simulation-in-the-loop agent converts the recovered articulation into a parameterized skill program and iteratively refines failed rollouts to generate physically executable demonstrations. These demonstrations are used to train ArticuACT, a dual-depth policy that predicts coordinated base, arm, and gripper commands using robot-centric camera conditioning and interaction-aware supervision. With all perception and policy inference running onboard, the system achieves a 96.57% average success rate across five real doors and an 80.95% zero-shot success rate on structurally similar unseen doors, while completing the full approach, opening, and traversal sequence in approximately 13s on average. Project Page: https://video2doortraversal.github.io/.

**Analysis:**

# 1. 摘要翻译

开门与穿越门洞是一项长时程移动操作任务，需要精确的把手交互以及底盘—机械臂协同控制。本文提出 **Video2DoorTraversal**：一种面向轮腿式移动操作机器人的单视频“真实—仿真—真实”框架。给定一段真实门的 RGB 视频，**DoorTwin** 重建与实例对齐、具有关节结构且可直接用于仿真的门数字孪生体。随后，仿真在环智能体将恢复出的关节运动转化为参数化技能程序，并通过分析失败轨迹和反复仿真验证，生成物理上可执行的示范数据。最后，利用这些示范训练 **ArticuACT**，其基于双视角深度、机器人中心相机几何和交互感知监督，预测底盘、机械臂和夹爪的协调控制指令。感知与推理均在机运行，在五扇真实门上平均成功率达到 96.57%，对结构相似的未知门零样本成功率达到 80.95%，完整接近、开门和穿越平均约需 13 秒。

# 2. 方法动机分析

**驱动力：** 开门不是单纯的抓取或推门，而是“接近—抓取—旋转把手—推门—穿越”的长时程接触任务，要求底盘和机械臂连续协调。作者希望仅凭一段门的视频，自动构造适配该实例的仿真环境，并减少真实机器人示范采集。

**现有痛点：**  
1. 多数方法依赖预制或程序生成的门模型，不能反映真实门的尺寸、把手位置和外观。  
2. 仅从 RGB 生成 3D 资产容易出现尺度、局部部件位置和关节定义错误。  
3. VLM 或规则方法生成的轨迹未必满足碰撞、摩擦和动力学约束。  
4. 现有开门策略常聚焦操作本身，忽略开门后的底盘穿越；第三视角点云也难以支持近距离把手交互。

**核心假设：** 如果门的实例级几何与关节结构足够准确，且示范轨迹经过仿真物理验证，那么结合机器人中心的视觉几何信息，策略可以从双视角深度中学习稳健的全身开门与穿越行为。

# 3. 方法设计详解

## 3.1 整体流程

**输入：** 手机拍摄的一段单门 RGB 视频。  
**输出：** 可部署的轮腿机器人控制策略。

### 阶段一：DoorTwin 门数字孪生

1. 用 DAGE 从非标定视频估计逐帧度量深度和相机位姿；玻璃、反光区域再用 LingBot-Depth 修正。  
2. 用 SAM3 分割每帧完整门体，将深度点通过相机位姿变换到统一门坐标系并聚合，得到多视角度量点云。  
3. 对点云做 PCA：最短方向近似门面法向，另外两方向对应门宽和门高；由此获得全局尺度。  
4. 额外分割门板和把手，计算把手相对门板坐标的位置  
   \[
   \Delta_{ph}=R_p^\top(c_h-c_p),
   \]
   即把手中心相对于门板中心的局部坐标，避免只依赖图像中的表观比例。  
5. 将门尺寸、把手相对位置和铰链类型写入 Articraft 的结构化生成约束：门框固定，门板通过转动关节连接，把手附着在门板上；采用“先整体、后把手”的粗到细生成方式。  
6. 渲染到参考视频视角，由视觉 critic 检查轮廓、铰链侧、把手类型、比例和位置；发现错误后修改生成程序，但保持视频测得的尺度约束不变。  
7. 几何通过结构验证后，再进行去光照和纹理/材质迁移，形成含视觉网格、碰撞网格和 URDF 关节的仿真门资产。

### 阶段二：仿真在环示范生成

智能体不是直接生成底层控制代码，而是组合参数化技能：
`BaseMoveTo、EEApproach、CloseGripper、RotateHandle、PushDoor、Pass、ReleaseAndRetract`。

每个技能含接近距离、抓取偏移、旋转角度、接触偏置、底盘速度和持续时间等参数。程序在 Isaac Gym 中执行；系统记录把手旋转、门角度、解锁状态、碰撞和身体是否通过门洞。失败时，智能体根据日志和关键帧判断原因，并对参数做有界修改，再在邻域内搜索。只有同时满足任务完成、无碰撞和运动学可行性的轨迹才被保留。

之后随机化初始位姿、摩擦、阻尼、开门阻力、相机外参和深度噪声，筛选仍能成功的轨迹，增强 sim-to-real 鲁棒性。

### 阶段三：ArticuACT 策略

输入为前视深度、腕部深度和 9 维机器人状态；ACT 输出长度为 100 的动作块，每步包括底盘前向速度、底盘偏航速度、6 个机械臂关节命令和夹爪命令。

创新一是 **机器人中心 Plücker 射线图**。对每个像素，根据相机内参和相对机器人底座的外参计算射线方向 \(d_b\) 与力矩 \(m_b=t_{bc}\times d_b\)，组成六维表示 \([d_b,m_b]\)。它把像素直接关联到机器人动作坐标系，减轻不同视角下的空间歧义。

创新二是 **交互状态辅助预测**。策略同时预测把手接触、把手旋转进度和开门进度，但这些状态只用于训练，不反馈控制。总损失由动作、KL、接触 BCE、把手进度和门进度损失组成，使隐变量更关注接触阶段和任务进程。

# 4. 方法对比与适用性

其本质区别是：用“实例级关节门模型”贯穿重建、专家生成和策略执行，而不是把仿真、规划和感知割裂开。主要创新包括单视频度量门孪生、仿真验证驱动的智能体示范生成，以及双深度机器人中心几何条件策略。

方法适合门、抽屉、柜门等具有明确关节结构的移动操作任务，尤其适合真实数据昂贵、需要快速适配具体物体的场景。对拉门、旋钮门、非刚性门及严重反光环境，仍需额外建模。

# 5. 实验分析

作者通过资产质量比较、仿真基线、策略消融、示范规模实验和五扇真实门测试验证方法。最关键结论是：真实门平均成功率 **96.57%**，相似未知门零样本成功率 **80.95%**；仿真中完整方法的穿越成功率达 **97.27%**，显著高于普通 ACT、扩散策略和点云策略。

优势是无需外部门位姿、可自动生成物理可行示范、底盘—机械臂协同紧密。局限是高度依赖深度/分割/生成模型质量，仍需为目标门拍视频并生成数据；实验主要覆盖推门和相似结构门，泛化范围有限。

# 6. 实用指南

论文提供项目主页，但文中未明确说明完整代码、模型权重或数据集已公开，因此目前不能视为完全开源。复现需准备 DAGE、SAM3、Articraft、Isaac Gym、ACT、双 RealSense 深度相机及对应轮腿机器人。关键设置包括：示范轨迹每门 200 条、仿真控制 50 Hz、训练数据 25 Hz、动作块长度 100、深度范围 0.2–1.5 m，并加入深度噪声、孔洞和遮挡增强。

迁移到抽屉或柜门时，应替换关节约束和局部交互坐标系，重新定义技能库、成功判据与交互辅助标签；Plücker 条件和“仿真验证—示范筛选—策略学习”的框架仍可保留。

# 7. 总结

**核心思想：** 单视频重建门孪生并仿真验证协同策略。

**速记版 pipeline：**
1. 从门视频恢复尺度、结构和把手位置。  
2. 生成并反复校正可运动的门仿真模型。  
3. 在仿真中执行、诊断并修正开门技能。  
4. 用随机化成功轨迹训练双视角深度策略。  
5. 在真实机器人上闭环完成接近、开门和穿越。

**Key Findings:**

- We present Video2DoorTraversal, a single-video real-to-sim-to-real framework for wheel-legged mobile manipulators.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20251v1)
- [arXiv](https://arxiv.org/abs/2608.20251v1)

---

<a id='2608.20208v1'></a>
## [RoMAN-Flow: Taming Autoregressive Normalizing Flows for Offline Reinforcement Learning in Robotic Manipulation](https://arxiv.org/abs/2608.20208v1)

**Authors:** Shaoxuan Wang, Guangting Zheng, Rui Huang, Zhipeng Tang, Sha Zhang, Jiajun Deng, Yanyong Zhang

**Published:** 2026-08-20

**Categories:** cs.CV

**Abstract:**

Offline reinforcement learning improves robotic policies using previously collected data without further environment interaction. Yet prevalent diffusion- and flow-matching robot policies lack tractable likelihoods, limiting their use in likelihood-based offline RL post-training. AR-NFs offer both expressive action modeling and exact likelihood evaluation, but their sequential sampling incurs substantial sampling overhead during policy optimization and deployment. We present RoMAN-Flow (Robotic Manipulation with Autoregressive Normalizing Flows), an offline reinforcement learning framework that makes AR-NF policies practical for robotic manipulation by addressing this sampling bottleneck in both stages. During policy optimization, RoMAN-Flow employs a sampling-free, advantage-weighted likelihood objective that assigns higher likelihood to high-advantage actions from the offline dataset without sampling from the autoregressive policy. For efficient deployment, it distills the optimized autoregressive policy into a one-step action generator, enabling low-latency action prediction. Experiments across multiple simulated manipulation benchmarks and real-world robotic platforms demonstrate that RoMAN-Flow achieves competitive policy performance while substantially reducing inference latency. Code is available at https://github.com/konnyaku28/RoMAN-Flow.

**Analysis:**

# 1. 摘要翻译

离线强化学习能够利用预先收集的数据改进机器人策略，而无需继续与环境交互。然而，当前主流的扩散策略和流匹配策略通常缺乏易于计算的精确似然，限制了其在基于似然的离线强化学习后训练中的应用。自回归归一化流（AR-NF）同时具备较强的动作建模能力和精确似然评估能力，但其顺序采样过程会在策略优化和实际部署中带来显著开销。

本文提出 RoMAN-Flow（Robotic Manipulation with Autoregressive Normalizing Flows），通过两个机制解决上述瓶颈：在策略优化阶段，采用无需策略采样的、基于优势加权的似然目标，仅提高离线数据中高优势动作的似然；在部署阶段，将优化后的自回归策略蒸馏为单步动作生成器，以实现低延迟预测。多个仿真基准和真实机器人实验表明，RoMAN-Flow在保持竞争力的同时，能够显著降低推理延迟。

# 2. 方法动机分析

**驱动力。** 扩散/VLA策略擅长表示复杂、多峰、连续动作，但难以直接计算动作似然；AR-NF可以精确计算似然，适合做类似AWR/IQL的离线策略优化，但逆变换按动作位置逐步生成，速度慢。

**核心假设。** 对离线RL而言，策略优化不一定需要从当前策略采样动作，只要能够评估数据动作的似然，并依据其优势重新加权即可；部署时则可以用蒸馏模型近似教师策略的逆变换。

# 3. 方法设计详解

## 3.1 总体Pipeline

1. 将轨迹切分为长度为 \(H\) 的重叠动作块 \((c_t,a_t)\)，其中 \(c_t\) 包含多视角图像、语言和本体感知状态，\(a_t=(a_t,\ldots,a_{t+H-1})\) 为连续动作序列。  
2. 通过预训练多模态编码器得到上下文 token \(C_t\)。  
3. 使用条件AR-NF将动作块正向映射为高斯潜变量：
\[
z_t=F_\theta(a_t;C_t).
\]
4. 先进行行为克隆，再进行NF-IQL后训练。  
5. 将最终AR-NF冻结为教师，训练一次前向即可输出完整动作块的学生策略。

## 3.2 AR-NF策略

模型由多个条件流模块组成。第 \(l\) 个模块在动作位置 \(j\) 根据上下文和前序动作预测平移量、尺度：
\[
(\mu^{(l)}_{t,j},s^{(l)}_{t,j})
=T_l(C_t,h^{(l-1)}_{t,<j}),
\]
并执行：
\[
h^{(l)}_{t,j}
=(h^{(l-1)}_{t,j}-\mu^{(l)}_{t,j})
\exp(-s^{(l)}_{t,j}).
\]

由于参数只依赖前序位置，Jacobian为三角矩阵，因此动作似然可精确计算：
\[
\log\pi_\theta(a_t|c_t)
=\log p_0(z_t)-\sum_{l,j}s^{(l)}_{t,j}.
\]

训练时完整动作块已知，因而所有位置可通过带因果掩码的Transformer并行计算；生成时从 \(z\sim\mathcal N(0,I)\) 出发执行逆变换，但必须逐位置恢复动作，这是训练快、推理慢的根源。

## 3.3 NF-IQL：无采样离线策略优化

作者不使用TD3+BC等需要当前策略生成动作的更新方式，而是只在数据动作上进行优化。

- **Prefix critic：** 每个 critic 在一次因果前向中，为动作块的各个前缀预测Q值。前缀目标为：
\[
y_{t,j}=\sum_{i=0}^{j}\gamma^ir_{t+i}
+\gamma^{j+1}V(c_{t+j+1}).
\]
这样既利用了动作块内部的中间奖励，也缓解了只给整个动作块一个Q值造成的信用分配问题。

- **IQL价值估计：** 对目标critic集成值做expectile回归。较大的expectile \(\tau>0.5\)使 \(V(c)\)偏向数据中较优行为的价值，而不是简单平均。

- **优势加权似然：**
\[
A_t=\bar Q(c_t,a_t)-V(c_t),\qquad
w_t=\exp(\beta A_t),
\]
然后优化：
\[
\mathcal L_\pi
=-\mathbb E[w_t\log\pi_\theta(a_t|c_t)].
\]

直观上，数据中“比该状态通常水平更好”的动作被赋予更高权重；低价值动作仍保留一定训练作用，从而避免完全脱离行为数据。关键是：整个更新只需AR-NF正向计算似然，无需顺序逆采样。

## 3.4 单步蒸馏

冻结NF-IQL策略作为教师。对离线动作加小噪声后，教师正向得到潜变量和中间流状态；学生采用双向Transformer，从同一潜变量和上下文并行预测完整动作及各层中间状态。损失由两部分组成：

1. 对齐教师逆过程中的中间状态；
2. 重构教师动作块。

此外，从高斯先验采样潜变量，用教师逆变换生成额外轨迹，再进行同样蒸馏，扩大学生覆盖的策略分布。最终学生一次前向生成全部动作，避免自回归逆变换。

# 4. 方法对比与创新

**本质区别：** 扩散策略依赖去噪，连续流通常需要ODE密度追踪；RoMAN-Flow利用AR-NF的精确似然直接做优势加权。相比传统离线RL，它把“策略采样”改为“数据动作重加权”。

**主要创新：**

- 将连续动作AR-NF用于机器人离线RL；
- 提出采样自由的NF-IQL；
- 用中间状态和先验采样联合蒸馏自回归逆过程。

适合已有高质量离线演示、动作连续且精细、需要奖励后训练，同时又重视部署延迟的机器人任务。

# 5. 实验分析

作者在MetaWorld、LIBERO、RoboMimic及Franka-XHand真实平台验证。代表性结论是：NF-IQL显著优于纯行为克隆，真实机器人平均成功率从57.3%提升至81.5%；单步蒸馏在LIBERO-Long上达到93.0%，并将动作块延迟从约697 ms降至81.5 ms，获得8.55倍加速。

**优势：** 连续动作、精确似然、离线优化无需采样、部署低延迟。  
**局限：** 教师逆变换本身较慢，蒸馏训练仍需大量教师计算；性能依赖critic准确性、优势温度和动作块设计；单步学生可能在真实分布外状态下丢失教师性能。

# 6. 实用指南

代码已开源于 GitHub。复现时需依次执行行为克隆、critic/value warm-up、NF-IQL后训练和学生蒸馏。重点调节 \(\tau,\beta\)、优势裁剪、折扣因子、critic更新速率及动作块长度；建议对奖励做轨迹级重标定以缓解稀疏奖励。迁移到其他连续控制任务时，只需替换多模态编码器、动作维度、数据切块方式和奖励定义；若任务对精确长程控制敏感，应优先增加AR-NF容量并谨慎选择蒸馏权重。

# 7. 总结

**核心思想：** 用精确似然离线提优，再蒸馏加速。

**速记版Pipeline：**

1. 把示范轨迹切成连续动作块。  
2. 用AR-NF把动作映射到高斯空间并计算精确似然。  
3. 用critic评估数据动作，只提高高优势动作的似然。  
4. 用教师中间状态和生成轨迹训练单步学生。  
5. 部署学生，一次并行输出完整动作块。

**Key Findings:**

- We present RoMAN-Flow (Robotic Manipulation with Autoregressive Normalizing Flows), an offline reinforcement learning framework that makes AR-NF policies practical for robotic manipulation by addressing this sampling bottleneck in both stages.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20208v1)
- [arXiv](https://arxiv.org/abs/2608.20208v1)

---

<a id='2608.20127v1'></a>
## [ID-VTG: Image-Disambiguated Video Temporal Grounding](https://arxiv.org/abs/2608.20127v1)

**Authors:** Minghang Zheng, Jingli Wei, Hongyi Yang, Yang Liu

**Published:** 2026-08-20

**Categories:** cs.CV

**Abstract:**

Video Temporal Grounding (VTG) faces significant challenges when natural language queries must distinguish between multiple events involving visually similar entities, particularly when relying on fine-grained visual attributes that are difficult to describe accurately in words alone. To address this, we introduce Image-Disambiguated Video Temporal Grounding (ID-VTG), a task that leverages multimodal queries combining a reference image and a text description to precisely localize segments where a specific instance performs a described action. To facilitate research, we construct two benchmarks: IDVTG-Gym, focusing on fine-grained, compositionally ordered gymnastics actions with athletes in similar uniforms; and IDVTG-InternVid, an open-world dataset featuring diverse entities (e.g., humans, animals, fictional characters) and significant temporal distractors. Methodologically, we propose the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework based on a dual-branch fast-slow architecture. The fast branch efficiently generates preliminary event proposals, while the slow branch performs fine-grained frame-level matching between video frames and the reference image. We enhance discriminability via two learnable tokens: a Compare Token, which represents hard negatives to probe for the presence of the target instance (as referred to by the query image), and a Depress Value, which represents text-irrelevant events. Proposals that the Compare Token identifies as lacking the target instance are pushed toward the Depress Value, thus easing disambiguation via the text query. Extensive experiments validate our approach, which achieves state-of-the-art results on the proposed benchmarks. Code is available at https://github.com/oceanflowlab/ID-VTG.

**Analysis:**

## 1. 摘要翻译
视频时序定位（VTG）在多个视觉相似实体执行相同动作时面临困难，尤其当区分目标依赖纹理、面部等难以用语言准确描述的细粒度属性时。为此，论文提出图像消歧视频时序定位（ID-VTG）：利用“参考图像+文本描述”的多模态查询，定位特定实例执行指定动作的片段。作者构建了IDVTG-Gym和IDVTG-InternVid两个数据集，并提出视觉引导消歧聚合框架（VGD-Agg）。该框架采用快慢双分支：快分支生成候选事件，慢分支进行视频帧与参考图像的细粒度匹配；同时引入Compare Token表示高相似视觉负样本，引入Depress Value表示与文本无关的事件，从而抑制干扰并提升定位精度。

## 2. 方法动机
**驱动力：**文本只能描述“做了什么”，难以可靠描述“是哪一个人/物”。  
**痛点：**传统VTG通常假设文本对应唯一事件；简单拼接图像或把图像转成文字会丢失身份细节，且无法专门处理强视觉干扰。  
**核心假设：**应先用图像判断候选片段是否包含目标实例，再用文本判断该实例是否执行了目标动作。

## 3. 方法设计详解
输入视频、文本查询和参考图像，输出目标时间区间。

1. **快分支：**用视觉编码器提取帧特征，经多尺度Transformer/ActionFormer式提议生成器建立不同时间范围的候选提议，获得具有长时序上下文的提议特征。该分支不进行图像匹配，负责高效覆盖可能事件。  
2. **慢分支：**将参考图像编码为视觉查询，与帧特征做跨模态注意力，得到增强后的帧表示及每帧图像相似度。为此生成两个视频特定表示：  
   - **Compare Token \(t_c\)：**由视频特征和图像查询共同生成，作为“难负样本基准”；  
   - **Depress Value \(v_d\)：**由视频特征和文本查询生成，表示文本无关内容。  
3. **相似度约束：**要求真实区间平均相似度、Compare Token、非真实区间平均相似度满足  
   \(\bar s_{gt}>\!s_c>\!\bar s_{non-gt}\)。  
   三个hinge损失分别约束正样本高于Token、Token高于负样本，以及正负样本整体间隔，从而使Token成为动态视觉判别边界。  
4. **视觉辅助消歧：**对每个时间提议收集其内部帧特征及相似度，并把Compare Token作为额外竞争项，与帧分数共同softmax聚合。若提议包含目标实例，目标帧权重更高；若只是视觉干扰，Compare Token占优，聚合结果转向Depress Value。最后将聚合特征与原提议残差融合。  
5. **文本定位：**将消歧后的提议特征与文本输入Transformer解码器，由分类头判断相关性，回归头预测边界；总损失为分类损失、DIoU回归损失和视觉匹配损失。

## 4. 方法对比分析
本质区别不是“加入图像”，而是把图像作为候选事件的**身份过滤器**，并显式建模视觉难负样本。创新主要包括ID-VTG任务、两类高歧义数据集、Compare Token/Depress Value及竞争式聚合。适合监控中特定人员追踪、体育运动员定位、视频中精确检索商品或角色等场景。

## 5. 实验分析
作者在Gym、InternVid及跨域Web集上，与多种VTG方法比较，并进行分支、Token、融合方式、歧义类型和图像退化实验。代表性结论是：VGD-Agg在Gym整体达到61.83% R@1@0.5，明显优于SnAG的53.64%；在InternVid及Web跨域测试中也保持领先。  
**优势：**能同时处理“多人同动作”和“同人多动作”，且对亮度、低分辨率图像较稳健。  
**局限：**慢分支带来额外计算；性能依赖参考图像质量和实例可见性；InternVid部分标注由MLLM自动生成，仍可能存在噪声；方法主要输出时间区间，不提供空间轨迹。

## 6. 实用指南
论文已开源代码：`github.com/oceanflowlab/ID-VTG`。复现时需使用CLIP ViT-L/14提取768维特征，AdamW学习率 \(2\times10^{-4}\)、权重衰减0.05，\(\alpha=2\)、margin \(m=1\)、回归权重2。关键是构造同文本多实例硬负样本，并保证参考图像确实对应目标实例。该思想可迁移至视频检索、事件检测、跟踪和多目标动作识别：将空间或身份查询替换为图像、框、轨迹或模板，再使用动态负样本门控。

## 7. 总结
**核心思想：**用图像筛身份，用文本定事件。  
**速记Pipeline：**  
1. 视频生成多尺度候选片段；  
2. 参考图像与视频帧逐帧匹配；  
3. 用动态难负样本判断目标是否出现；  
4. 抑制无关片段、突出目标片段；  
5. 用文本完成最终时间边界回归。

**Key Findings:**

- To address this, we introduce Image-Disambiguated Video Temporal Grounding (ID-VTG), a task that leverages multimodal queries combining a reference image and a text description to precisely localize segments where a specific instance performs a described action.
- Methodologically, we propose the Visually-Guided Disambiguation Aggregation (VGD-Agg) framework based on a dual-branch fast-slow architecture.
- Extensive experiments validate our approach, which achieves state-of-the-art results on the proposed benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20127v1)
- [arXiv](https://arxiv.org/abs/2608.20127v1)

---

<a id='2608.20114v1'></a>
## [DECOWAM: Decoupled Whole-Body World-Action Model for Legged Mobile Manipulation](https://arxiv.org/abs/2608.20114v1)

**Authors:** Siyuan Ma, Boshi Zhang, Yutian Zhang, Qinglian Wu, Jiaqi Zhai, Dong Wei, Qiaojun Yu

**Published:** 2026-08-20

**Categories:** cs.AI, cs.RO

**Abstract:**

Mobile manipulation requires a robot to predict how locomotion and arm motion jointly alter future observations and control. Existing world-action models, developed largely for fixed-base platforms, do not explicitly distinguish camera ego-motion from base and arm actions. Here we introduce DECOWAM, a whole-body world-action model that separates these factors through dedicated conditional interfaces. DECOWAM freezes an adapted FastWAM backbone and trains residual adapters, an action-equivalent future bottleneck distilled from privileged observations, adversarially separated base and arm latents, and base-velocity conditioning for video prediction. We further introduce ARMDOG, a real-robot dataset that synchronizes video, whole-body state and action, and language. On a fixed replay protocol, DECOWAM improved both future-video and action prediction over FastWAM, reducing action MSE by 21.7% with 25.95M trainable adaptation parameters. Across 79 closed-loop trials per method, it achieved the highest observed whole-body coordination and base-displacement robustness among the compared systems, while task completion remained comparable to the strongest baseline. These results show that embodiment-aware factorization can support parameter-efficient joint visual prediction and whole-body control under moving viewpoints.

**Analysis:**

## 1. 摘要翻译

移动操作要求机器人预测行走与机械臂运动如何共同改变未来观测和控制。现有世界—动作模型主要面向固定底座平台，通常没有明确区分相机自运动、底座动作和机械臂动作。本文提出 DECOWAM，一种面向全身控制的世界—动作模型，通过专门的条件接口分离这些因素。方法冻结经过适配的 FastWAM 主干，并训练残差适配器、由特权观测蒸馏得到的动作等价未来瓶颈、通过对抗学习分离的底座/机械臂潜变量，以及用于视频预测的底座速度条件。作者还构建了同步视频、全身状态动作和语言的真实机器人数据集 ARMDOG。在固定回放协议下，DECOWAM 相比 FastWAM 将动作 MSE 降低21.7%，且仅需训练2539万参数。79次闭环实验表明，该方法在全身协调和底座位移鲁棒性方面最佳，任务完成率与最强基线相当。

## 2. 方法动机

**驱动力与痛点：**移动机器人相机随底座运动，图像变化同时包含环境运动、机械臂运动和相机自运动；同时，机械臂是高频控制，底座速度是低频控制，二者语义和时间尺度不同。将14维动作直接送入统一模型，容易造成因素混淆，表现为错误的底座移动、视觉漂移和抓取过程中身体不稳定。

**核心假设：**如果显式提供“底座怎么动、机械臂怎么动、相机因底座速度产生怎样的视觉变化”三类接口，模型就能更容易学习全身协调；未来视觉虽然部署时不可用，但可在训练中作为特权信息帮助学习动作表示。

## 3. 方法设计详解

### Pipeline

输入当前RGB图像 \(x_0\)、14维机器人状态 \(s_0\) 和语言指令 \(\ell\)，输出8帧未来视频与48步、14维动作块。

1. **Stage 1领域适配：**先完整微调 FastWAM，使 Wan-2.2 视频分支和 ActionDiT 动作分支适应 ARMDOG 的移动相机、四足底座和机械臂动作空间。  
2. **冻结主干并加入残差适配器：**Stage 2冻结Stage-1模型，仅在各网络块后加入128维瓶颈残差分支，实现机器人特定修正，避免破坏原有视频先验。可训练参数从约60.2亿降至2595万。  
3. **动作等价未来瓶颈：**训练时教师编码器同时观察当前和未来VAE视觉摘要，学生编码器只观察当前摘要和状态。教师潜变量通过动作重建和几何约束学习“未来变化对应的动作信息”，再用蒸馏损失让学生复现该表示。部署时删除教师，仅保留因果学生。  
4. **底座—机械臂解耦：**将动作上下文映射为两个16维潜变量：\(z_{base}\) 表示底座速度，\(z_{arm}\) 表示机械臂/夹爪动作。直接预测头强化各自信息；GRL交叉预测头反向传播，抑制底座变量携带机械臂信息、机械臂变量携带底座信息。  
5. **自运动条件：**从状态中提取当前底座速度 \((v_x,v_y,\omega_z)\)，投影到视频主干维度，并加到每个视频token上，使视频分支不必仅凭像素猜测相机运动。  
6. **联合生成：**视频和动作分支均采用 conditional flow matching，通过噪声插值学习从噪声到真实未来视频/动作的速度场。总损失包括视频损失、动作损失、未来瓶颈损失和解耦损失。部署只使用当前图像、状态和语言，严格因果地产生动作。

### 关键设计理解

未来瓶颈不是直接把未来帧输入策略，而是把未来帧中与动作相关的结构压缩后蒸馏给当前信息，因此兼顾“训练时看得更远”和“部署时不能偷看未来”。GRL的作用也不是简单拆分动作维度，而是主动惩罚跨因素可预测性，减少底座与机械臂潜表示的语义泄漏。

## 4. 方法对比与创新

与固定底座VLA相比，DECOWAM同时预测未来视觉和全身动作；与普通WAM相比，它不把所有动作和图像变化放入同一未区分表示，而是引入未来特权蒸馏、底座/机械臂对抗解耦和显式速度条件。主要创新在于：**将身体因素分离从数据层面提升到模型条件接口层面**，并以参数高效方式完成移动机器人适配。

适合四足或轮腿移动操作、相机随底座变化、需要导航与抓取同步的任务；对固定相机、单纯机械臂或缺少同步底座状态的数据，收益可能有限。

## 5. 实验分析

作者使用ARMDOG固定回放、模块消融、VLA/WAM对比及79次真实机器人闭环实验验证方法。代表性结论是：相对50k步 FastWAM，视频MSE下降15.0%，动作MSE下降21.7%；真实实验中任务成功率58.2%，并在全身协调、位移鲁棒性和恢复能力上领先基线。

**优势：**未来预测与动作控制统一；显式处理移动视角；适配参数少。  
**局限：**部署网络总规模仍超过67亿参数，推理延迟略升；数据和实验任务主要集中在少数箱体操作场景，泛化性未知；部分数据统计存在不同版本/子集口径，复现时需严格核对划分。

## 6. 实用指南

文中未明确给出代码或数据下载链接，不能据此确认已开源。复现关键是：统一15Hz时间戳、构造14维动作、提取Wan VAE均值/方差摘要、先全量适配再冻结、最后训练四类增量模块。重要设置包括未来8帧、48步动作块、128维适配器瓶颈、64维未来瓶颈、16维底座/机械臂潜变量，损失权重约为视频/动作1、瓶颈0.2、解耦0.1。迁移到其他机器人时，应重新定义底座速度和动作分组，保留学生蒸馏、GRL解耦及自运动条件接口。

## 7. 总结

**核心思想：**显式分离底座、机械臂与相机自运动。

**速记版Pipeline：**
1. 用同步图像、状态、动作和语言适配基础模型。  
2. 冻结大模型，只学习机器人专属小型修正模块。  
3. 用训练期未来画面教会当前观测预测动作。  
4. 分开学习底座动作与机械臂动作，并输入底座速度解释画面变化。  
5. 部署时仅凭当前观测生成未来视频和全身动作。

**Key Findings:**

- Here we introduce DECOWAM, a whole-body world-action model that separates these factors through dedicated conditional interfaces.
- These results show that embodiment-aware factorization can support parameter-efficient joint visual prediction and whole-body control under moving viewpoints.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20114v1)
- [arXiv](https://arxiv.org/abs/2608.20114v1)

---

<a id='2608.20111v1'></a>
## [Planning-Oriented End-to-End Autonomous Driving: Architectures, Evaluation, and Emerging Paradigms](https://arxiv.org/abs/2608.20111v1)

**Authors:** Yanchen Guan, Xingcheng Liu, Bin Rao, Chengyue Wang, Guofa Li, Yunjian Li, Lishengsa Yue, Zhiyong Cui, Chengzhong Xu, Zhenning Li

**Published:** 2026-08-20

**Categories:** cs.RO, cs.ET

**Abstract:**

End-to-end autonomous driving has evolved from camera-to-control regression toward planning-oriented systems that use structured representations, trajectory-level outputs, and increasingly realistic evaluation protocols. This survey reviews this transition across behavior cloning, conditional imitation learning, privileged distillation, BEV and vectorized planning, unified perception-prediction-planning architectures, world-model-based planners, and vision-language-action systems. We argue that the key distinction in modern end-to-end driving is not whether intermediate representations are used, but whether they are learned, supervised, and evaluated to support safe, feasible, and route-compliant planning. To organize the literature, we synthesize existing methods along four axes: input representation, planning output, supervision signal, and evaluation protocol. We further examine the benchmark shift from open-loop trajectory matching to closed-loop simulation, non-reactive real-log evaluation, long-tail testing, and human-preference-aware metrics. Our analysis highlights that architectural progress is difficult to interpret without benchmark-consistent evaluation, and that displacement-based open-loop metrics alone provide limited evidence for safe and human-aligned driving. We conclude with open challenges in uncertainty-aware planning, learner-expert mismatch, runtime safety assurance, language-action grounding, world-model validation, and reproducible benchmarking.

**Analysis:**

# 1. 摘要翻译

端到端自动驾驶已从早期的“摄像头到控制量”回归，发展为面向规划的系统：这类系统采用结构化表示、轨迹级输出以及更加真实的评测协议。本文围绕行为克隆、条件模仿学习、特权信息蒸馏、BEV 与矢量化规划、感知—预测—规划统一架构、基于世界模型的规划器，以及视觉—语言—动作系统，综述了这一演进过程。

作者认为，现代端到端驾驶的关键区别并不在于是否使用中间表示，而在于这些表示是否经过学习、监督和评测，并且是否真正服务于安全、可行和符合路线的规划。为此，本文从四个维度组织相关工作：输入表示、规划输出、监督信号和评测协议。文章进一步分析了评测范式如何从开环轨迹匹配，转向闭环仿真、非反应式真实日志评测、长尾测试和考虑人类偏好的指标。

本文指出，如果缺乏与基准一致的评测，架构进步很难被正确解释；单纯依赖位移误差的开环指标，也不足以证明系统具备安全且符合人类驾驶习惯的能力。最后，文章总结了不确定性感知规划、学习者—专家差异、运行时安全保障、语言—动作对齐、世界模型验证和可复现实验等开放问题。

# 2. 方法动机分析

## 驱动力与痛点

这不是提出单一自动驾驶模型的论文，而是一篇**规划导向的系统性叙述综述**。其核心驱动力是：传统综述往往按传感器、骨干网络或模型年代分类，却没有同时回答“模型输出什么、如何监督、怎样评估、是否真的改善规划”。

作者重点针对三类问题：

1. **模块接口割裂**：感知、预测和规划分别优化，导致上游指标与最终驾驶行为脱节。
2. **端到端概念泛化**：现代系统大量使用 BEV、占据栅格、目标 token、世界模型等结构，不能再简单用“是否无中间表示”定义端到端。
3. **评测与能力错配**：开环 ADE/FDE 或 L2 只衡量是否接近日志轨迹，不能反映偏离后的恢复、交互、碰撞、路线进度和舒适性。

## 核心假设

**端到端的价值不在于去除结构，而在于让结构、监督和评测共同服务于最终规划目标。**

# 3. 方法设计详解

## 综述流程

本文采用结构化叙述综述流程：

1. **文献检索**：覆盖 IEEE Xplore、ACM、Springer、Elsevier、arXiv、OpenReview、CVF、基准仓库和项目主页；检索时间主要为 2015 至 2026 年 6 月。
2. **初筛**：依据标题、摘要和任务相关性去除纯感知、重复或信息不足的工作。
3. **资格筛选**：保留提出端到端驾驶范式、规划输出、驾驶基准、世界模型/VLA 机制，以及安全性、鲁棒性和可复现性分析的研究。
4. **统一编码**：为每项工作记录输入表示、规划输出、监督信号、评测协议和公开资源状态。
5. **四轴归类**：
   - 输入：前视图、多视角、LiDAR/radar、BEV、占据、对象/矢量 token、潜在世界状态、VLA token；
   - 输出：控制量、waypoint、轨迹、多模态分布、场景 rollout、动作 token；
   - 监督：行为克隆、扰动恢复、特权教师、辅助任务、世界模型预测、语言问答、偏好反馈；
   - 评测：开环、非反应式、反应式闭环、长尾、偏好感知和跨基准相关性。
6. **方法论比较**：不混合不同基准的分数，而是比较每种方法能够支持的具体论断。

## 关键设计视角

文章最重要的修正不是新网络，而是把比较单位从“模型名称”改为“**模型—监督—输出—评测协议所组成的完整主张**”。例如，同样使用 Transformer 的系统，可能分别属于 BEV 特权蒸馏规划器或语言—动作对齐规划器，二者不能仅按骨干网络比较。

文章还将世界模型区分为四种作用：表示预训练、未来场景生成、候选轨迹评估，以及交互式仿真。作者特别强调：生成逼真视频并不等于能够安全选择轨迹，世界模型的价值应通过其对轨迹排序和闭环结果的改善来验证。

# 4. 方法对比与创新

## 本质区别

相较于传统 E2E 综述，本文不以“端到端程度”或网络结构为核心，而以**规划输出空间、监督来源和评测有效性**为主线。

## 主要创新

1. 提出规划导向的四轴分类框架。
2. 系统梳理从直接控制到轨迹、BEV、世界模型和 VLA 的演进。
3. 明确区分开环、非反应式和闭环评测的能力边界。
4. 将学习者—专家不对称、语言—动作错配和世界模型校准提升为核心问题。
5. 给出包含传感器、控制器、安全包装、算力、随机种子和基准版本的复现清单。

## 适用场景

适用于撰写 E2E 自动驾驶综述、选择规划基线、设计跨基准实验，以及分析世界模型或 VLA 是否真正改善驾驶行为。

# 5. 实验分析（精简版）

本文没有提出统一模型，也没有传统意义上的训练实验。其“验证”主要来自文献对比、基准分析和跨协议讨论。

- 代表性结论一：开环轨迹误差不能单独证明闭环安全性，NAVSIM 等非反应式指标虽更接近规划，但仍不能替代交互式闭环。
- 代表性结论二：现代系统的性能往往同时取决于表示、监督、控制器、安全包装和评测协议，而非单一架构创新。

优势是视角统一、问题意识强；局限是大量 2025—2026 年工作尚缺乏独立复现，且综述结论依赖公开论文和基准质量。

# 6. 实用指南

## 开源与复现

本文本身不是一个可运行方法，未提供统一代码。复现其分析需建立文献数据库，并按四轴记录方法；比较时应固定基准版本、传感器配置、控制器、数据划分和指标实现。

## 实现注意点

若据此设计实验，应至少同时报告：

- 开环轨迹质量；
- 非反应式安全/进度/舒适度指标；
- 反应式闭环路线完成率与违规率；
- 长尾或偏好感知结果；
- 控制器、运行时安全模块、推理延迟和采样次数。

## 迁移可能

该四轴框架可迁移到机器人导航、具身智能和 VLA 任务：将“规划输出”替换为动作序列，将“路线合规”替换为任务完成度，并保留监督来源、运行时保障和闭环评测三个维度。

# 7. 总结

**核心思想：让结构服务于可验证的安全规划。**

## 速记版 pipeline

1. 搜集端到端驾驶及相关基准文献。  
2. 按输入、输出、监督、评测四个维度统一记录。  
3. 区分模型学会了什么与基准真正测量了什么。  
4. 用闭环、长尾和偏好评测检验规划质量。  
5. 通过复现清单和安全包装判断方法是否可部署。

**Key Findings:**

- Our analysis highlights that architectural progress is difficult to interpret without benchmark-consistent evaluation, and that displacement-based open-loop metrics alone provide limited evidence for safe and human-aligned driving.
- We conclude with open challenges in uncertainty-aware planning, learner-expert mismatch, runtime safety assurance, language-action grounding, world-model validation, and reproducible benchmarking.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20111v1)
- [arXiv](https://arxiv.org/abs/2608.20111v1)

---

<a id='2608.20087v1'></a>
## [Towards Professional Tennis Styles for Humanoid Robots with Adaptive Motion Planning and Tracking](https://arxiv.org/abs/2608.20087v1)

**Authors:** Tao Huang, Ruofei Liu, Xuchen Tang, Xinyin Zhang, Junli Ren, Huayi Wang, Feiyu Jia, Yukai Qi, Kangning Yin, Weishuai Zeng, Lipeng Chen, Xi Li, Ting Wu, Kailin Li, Ruoli Dai, Jingbo Wang, Lei Han, Jiangmiao Pang

**Published:** 2026-08-20

**Categories:** cs.RO, cs.AI

**Abstract:**

Humanoid robots have recently demonstrated promising capabilities in real-world ball sports. However, achieving professional motion styles while maintaining strong task performance remains challenging. In this work, we propose AdaPT, an Adaptive Motion Planning and Tracking framework that learns professional tennis serving and rally styles directly from broadcast videos. This hierarchical design is motivated by the key insight that the planner generates stylistic kinematic motions, while the tracker executes them with minimal interference with planning. Despite its effectiveness in simulation, a substantial sim-to-real gap emerges: tracking performance inevitably degrades on real robots, and this degradation is partially overlooked by autoregressive planning and further compounded by noisy perception. To address these issues, our adaptation mechanism improves tracking robustness by learning to track randomized execution speeds, while conditioning the planner on a learned motion-speed adapter to mitigate compounding errors. Real-world experiments on the Unitree G1 demonstrate the effectiveness of our adaptation mechanism in bridging the sim-to-real gap. We further deploy AdaPT policies on the full-size Dobot Atom humanoid robot (1.7m) and demonstrate in-the-wild serving without motion capture. Beyond these results, our real-world experiments reveal both algorithmic and engineering insights for future humanoid ball-sports systems. Videos and code are available on our \href{https://humanoidtennis.github.io/AdaPT/}{project website}.

**Analysis:**

# 1. 摘要翻译

人形机器人近年来已在真实世界球类运动中展现出潜力，但要同时实现职业化运动风格与较强任务性能仍十分困难。本文提出 AdaPT（Adaptive Motion Planning and Tracking），一种直接从广播视频学习职业网球发球与对拉风格的自适应运动规划与跟踪框架。其核心思想是：规划器生成具有风格的运动学动作，跟踪器则尽量少干扰规划地执行动作。针对真实机器人中跟踪退化、规划误差累积和感知噪声共同造成的仿真到现实差距，AdaPT在训练时随机化动作执行速度，使跟踪器具备速度适应性；同时让规划器显式输出运动速度表征，从而根据跟踪能力调整动作节奏，抑制长期误差累积。实验表明，AdaPT可使Unitree G1学习纳达尔、费德勒和德约科维奇的对拉与发球风格，并迁移至全尺寸Dobot Atom；此外，系统仅使用ZED X相机和VIVE Tracker实现了无动作捕捉的真实场地发球。

# 2. 方法动机分析

**驱动力与痛点：**以往网球机器人主要优化“击中”和“回球”，忽略了职业动作中的躯干旋转、脚步、引拍和随挥等整体协调。Vid2Player3D虽采用规划—跟踪解耦以保留风格，但真实部署时跟踪误差会使自回归规划逐步漂移，感知噪声又进一步放大误差。传统紧耦合运动先验方法响应快，却容易把任务控制与风格纠缠，造成动作变形。

**核心假设：**规划器负责“做什么以及以多快的节奏做”，跟踪器负责“如何稳定执行”；若跟踪器预先适应不同速度，规划器再根据当前球轨迹和执行能力调整速度，就能同时提高真实任务成功率与风格一致性。

# 3. 方法设计详解

## 3.1 数据构建

从纳达尔、费德勒、德约科维奇广播视频提取约2秒完整击球片段，并用GVHMR重建SMPL人体动作，再用GMR重定向至机器人。针对视频遮挡导致的手腕误差，按球员风格修正持拍腕部，并在对拉数据中随机扰动腕部方向。每段动作标注球员、正/反手、上旋/切削、击球时刻；发球额外标注抛球释放时刻。随后用通用运动跟踪器修正重定向动作，使其满足物理约束，得到可执行数据。

## 3.2 速度自适应跟踪器

跟踪器输入参考姿态、当前姿态、关节速度和基座角速度，输出PD控制目标。训练时不固定参考动作速度，而是用相邻参考状态插值：

\[
\hat q_t^\alpha=(1-\alpha)\hat q_{t-1}+\alpha\hat q_t,\quad
\alpha\sim U(\alpha_{\min},\alpha_{\max}).
\]

\(\alpha<1\)表示放慢动作，\(\alpha>1\)表示加速。这样跟踪器学习的不是单一节拍，而是“可变速执行能力”，从根源上降低真实电机、延迟和控制误差造成的动作偏离。

## 3.3 对拉规划

对拉动作多样，因此训练MVAE运动生成器。其输入潜变量和上一时刻运动状态，自回归生成下一动作；额外预测击球阶段、球种类和旋转类型，以提高可控性。高层规划器观察机器人状态、目标位置及未来球轨迹，同时输出运动潜变量 \(z_t\) 和速度系数 \(\alpha_t\)。生成的相邻动作按

\[
\hat q_t=(1-\alpha_t)\hat p_{t-1}+\alpha_t\hat p_t
\]

融合后交给跟踪器。也就是说，规划器不仅选择动作风格，还会根据来球提前量决定动作应快还是慢。

## 3.4 发球规划与残差控制

发球动作多样性较低，因此不再训练额外生成器，而是直接跟踪参考发球动作，并由规划器输出速度调整量。为应对真实抛球偏差，残差跟踪器输出修正动作：

\[
a_t^{serve}=a_t^{track}+\Delta a_t.
\]

其中基础动作维持职业风格，残差只负责根据球位置和落点目标修正球拍控制。作者还设置引拍最深处为关键帧，并施加高权重局部腕部/球拍跟踪奖励，避免策略为了击球成功而省略引拍。抛球轨迹用抛物线建模，并通过稠密奖励约束释放点、速度和击球点，从而提升落点控制。

## 3.5 真实部署

球的位置可由动作捕捉或YOLO+双目三角测量获得；未来轨迹由含重力、空气阻力和碰撞反弹的简化物理模型预测。实验室使用MoCap定位机器人，户外使用VIVE Tracker。对拉还采用站立策略与对拉策略切换，防止无球时MVAE自回归漂移。

# 4. 方法对比与创新

与RL-Scratch相比，AdaPT显式利用职业动作；与AMP相比，它不依赖单一对抗奖励，而是保留动作阶段和规划结构；与PULSE/NCP等紧耦合先验相比，AdaPT将风格生成与任务执行分离；与Vid2Player3D相比，新增了**速度自适应跟踪器和速度感知规划器**，专门处理真实部署中的时间漂移。其创新不是简单增加感知输入，而是把“执行速度”提升为规划与跟踪之间的显式通信变量。最适合需要模仿复杂动作风格、同时面对执行延迟和环境时序不确定性的机器人运动任务。

# 5. 实验分析

作者在Unitree G1上进行仿真、MoCap真实实验，并扩展至Dobot Atom及无MoCap发球。代表性结论是：对拉中，AdaPT在三名球员风格上均明显改善真实击球和过网表现，且自适应跟踪与自适应规划具有互补性；发球中，关键帧、抛球引导和速度调整共同提升了动作风格与落点控制。主要优势是风格保真、对执行速度变化鲁棒、可跨机器人迁移。局限是仍依赖较准确的球轨迹和机器人定位，远距离移动能力不足，真实回球质量显著低于仿真。

# 6. 实用指南

论文提供代码：`github.com/noitom-robotics/AdaPT`，并提供项目视频。复现重点包括：视频姿态重建与腕部修正、物理可行性动作校正、速度随机化、球轨迹噪声/延迟/丢包随机化，以及反弹参数系统辨识。训练使用PPO、4096并行环境、3层MLP和约200Hz物理仿真；服务需特别调节抛球扰动、初始姿态随机化、关键帧奖励和残差L2约束。该框架可迁移到羽毛球、乒乓球、投掷或舞蹈等任务：替换动作数据、交互对象动力学和任务奖励即可，但必须重新设计阶段标签及速度适应范围。

# 7. 总结

**核心思想：让规划器调节节奏，让跟踪器适应节奏。**

**速记版Pipeline：**
1. 从视频恢复并修正职业动作。  
2. 训练能快慢执行的动作跟踪器。  
3. 对拉用生成器选风格动作，发球直接跟踪参考动作。  
4. 规划器根据来球和抛球情况调整动作速度。  
5. 用视觉、物理预测和残差控制完成真实击球。

**Key Findings:**

- In this work, we propose AdaPT, an Adaptive Motion Planning and Tracking framework that learns professional tennis serving and rally styles directly from broadcast videos.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20087v1)
- [arXiv](https://arxiv.org/abs/2608.20087v1)

---

<a id='2608.19977v1'></a>
## [Learning Highly Dynamic Skills Transition for Quadruped Jumping Through Constrained Space](https://arxiv.org/abs/2608.19977v1)

**Authors:** Zeren Luo, Jiahui Zhang, Yimin Han, Ji Ma, Minghao Lu, Ioannis Havoutis, Peng Lu

**Published:** 2026-08-20

**Categories:** cs.RO

**Abstract:**

Although legged animals are capable of performing explosive motions while traversing confined spaces, replicating this behavior in quadrupedal robots has been a longstanding challenge. Here, we propose a hierarchical reinforcement learning pipeline that empowers the robots to perform aggressive locomotion through constrained obstacles--a narrow gate. The imitation learning technique is used to train the low-level policy, which mimics the behaviors of real animals and forms a set of diverse skills. The high-level controller, having an awareness of the capability of low-level skills and acquiring the gate information via vision-based detection, determines the suitable maneuvers with collision-free trajectories to traverse it dynamically. Notably, we also verify that this framework can be extended to other highly dynamic tasks. This is one of the first works that perform autonomous and agile aerial gate traversal tasks on ground-walking robots, extending the lifelike agility of legged robots to match that of their biological counterparts.

**Analysis:**

# 1. 摘要翻译

尽管腿足动物能够在狭窄空间中完成爆发性运动，但在四足机器人上复现这一能力仍极具挑战。本文提出一种分层强化学习框架，使机器人能够通过狭窄闸门执行激进运动。首先利用模仿学习训练低层策略，使其模仿真实动物动作并形成多样化技能；随后，高层控制器结合低层技能能力与视觉检测到的闸门信息，选择合适的机动方式，生成无碰撞轨迹并动态穿越闸门。实验表明，该框架还可扩展到其他高动态任务。这是较早在地面行走机器人上实现自主、敏捷空中穿越闸门的工作之一。

# 2. 方法动机分析

**驱动力**：四足机器人不仅要“会跳”，还要自主决定何时加速、何时起跳、如何在空中收腿并对准狭窄入口。作者希望同时解决高动态运动生成与任务级决策问题。

**现有痛点**：轨迹优化通常计算昂贵，或依赖简化模型而难以保证物理可行性；端到端强化学习直接搜索关节动作，动作空间巨大，跳跃行为稀疏且训练不稳定；离散技能切换又会产生不连续，容易错过最佳起跳状态。

**核心假设**：动物动作数据能够提供自然、可迁移的动态运动先验；若将低层技能抽象为连续速度命令，高层只需学习技能的连续过渡与时机选择，就能以较低训练难度完成复杂跳跃。

# 3. 方法设计详解

## 3.1 总体流程

1. **构建动作先验**：从动物运动数据中选取 pace、canter、jump、steering 等低速至高速片段，通过逆运动学重定向到机器人形态；跳跃片段采样权重设为普通动作的6倍，强化高难度技能学习。  
2. **训练低层策略**：在 Isaac Gym 中并行训练。输入为42维本体感知量——机体角速度、投影重力、12个关节角、12个关节速度及上一时刻动作——以及期望速度命令；输出12维关节角偏移，由PD控制器或真实执行器网络跟踪。  
3. **利用对抗模仿形成连续技能**：总奖励为  
   \[
   r=\alpha_T r^T+\alpha_S r^S+\alpha_R r^R.
   \]
   任务奖励约束线速度、偏航速度并鼓励腾空时足端高度；判别器比较参考动作与机器人状态动作分布，提供风格奖励；正则项抑制力矩、动作突变和关节加速度。速度命令因此成为一种“连续技能索引”：低速倾向 pacing，高速触发奔跑跳跃。  
4. **训练高层策略**：冻结低层，以10 Hz运行；低层以50 Hz执行，形成5∶1时间尺度。高层输入本体状态、上一高层动作、闸门中心相对方向、机器人速度与闸门内外尺寸，输出二维命令：前向线速度 \(v_x^{cmd}\) 与偏航角速度 \(\omega_z^{cmd}\)。  
5. **任务奖励与决策**：核心奖励是世界系机器人速度在闸门方向上的投影，并惩罚速度方向与闸门方向的偏差；同时限制命令范围、减少能量消耗。高层不直接规划关节轨迹，而是通过速度命令诱导低层在行走、加速、起跳、空中收腿和落地之间连续转换。  
6. **视觉感知闭环**：RGB-D相机检测黑色矩形框，依次进行二值化、形态学闭运算、Canny边缘、去畸变、边缘聚类，再用Douglas–Peucker与Quickhull提取四边形。结合深度和相机标定，将四个顶点恢复到世界坐标，求平均得到闸门中心和边长，并用三阶低通滤波后输入高层。

其中，方向向量
\[
\hat d_{BG}=\frac{p_{IG}-p_{IB}}{\|p_{IG}-p_{IB}\|}
\]
表示机器人指向闸门的方向；偏航误差是其水平投影与机器人水平速度方向的夹角。两者使策略具备“瞄准闸门中心”的明确几何目标。

# 4. 方法对比与创新

本质区别在于：作者不是预先设计固定跳跃轨迹，也不是让端到端策略直接输出关节动作，而是把**模仿得到的动态能力作为低层连续状态**，让高层学习“何时、以何种速度组合技能”。

主要创新包括：  
- 用对抗模仿将多种动物动态动作整合为统一低层策略；  
- 用速度命令而非离散技能编号实现平滑技能过渡，避免切换断裂；  
- 将视觉闸门几何信息纳入高层决策，实现感知驱动的起跳时机和轨迹调整；  
- 仅重构高层模块即可迁移到跨障碍、狭缝等任务，无需重新训练低层。

最适合高动态、动作连续、需要实时适应目标位置的腿足机器人任务；不适合需要完全非生物动作或精确单关节规划的场景。

# 5. 实验分析

作者在Unitree Aliengo上验证，并与端到端模仿、端到端任务奖励、Parkour及离散层次策略比较。代表性结论是：所提方法能以约2.5 m/s速度穿越与机体尺寸相近的闸门，腾空时间约0.44 s；面对不同闸门高度、横向位置、楼梯、坡面和间隙仍能成功，横向位置测试无失败。其速度跟踪误差约0.0258 m/s，角速度误差约0.0857 rad/s。

优势是训练搜索空间小、动作自然、技能过渡平滑且更利于实机部署。局限是速度抽象限制了高层对具体腿部构型的选择；视觉模块依赖黑色矩形闸门和外部VICON定位，真实开放环境中的遮挡与定位鲁棒性仍不足。

# 6. 实用指南

论文未明确声明完整代码开源，仅说明补充材料可从 Wiley 或作者处获得。复现需重点实现：5480并行仿真体、25000回合低层训练、10/50 Hz双层频率、跳跃数据6倍采样、执行器随机化（质量、强度、延迟、偏置）以及约束闸门位置、尺寸和厚度的域随机化。网络为4层MLP，隐藏层[512,256,128]、ELU；判别器隐藏层[1024,512]。迁移到其他任务时，保留低层动作库，仅替换高层输入中的障碍几何描述、任务奖励和目标方向。

# 7. 总结

**核心思想：用连续技能组合自主完成动态跳跃。**

**速记版Pipeline：**  
1. 用动物动作教机器人学会走、跑、跳。  
2. 把不同速度当作连续技能旋钮。  
3. 用相机测出闸门位置和大小。  
4. 高层调节速度与转向，决定加速、起跳和落地。  
5. 通过低层策略连续完成整段穿越。

**Key Findings:**

- Here, we propose a hierarchical reinforcement learning pipeline that empowers the robots to perform aggressive locomotion through constrained obstacles--a narrow gate.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.19977v1)
- [arXiv](https://arxiv.org/abs/2608.19977v1)

---

<a id='2608.19894v1'></a>
## [Unified and Efficient Point-Line Local Features](https://arxiv.org/abs/2608.19894v1)

**Authors:** François Costa, Raphael Kreft, Eckhard Goedeke, Felix Möller, Hardik Shah, Ramanathan Rajaraman, Shaohui Liu, Rémi Pautrat, Marc Pollefeys

**Published:** 2026-08-20

**Categories:** cs.CV

**Abstract:**

Multi-view computer vision pipelines typically rely on accurate sparse keypoints and robust descriptors. While incorporating line features has shown clear benefits for matching and pose estimation, existing point-line approaches remain inefficient: they detect points and lines separately, use increasingly heavy networks, and depend on CPU-bound heuristics that hinder real-time performance. We introduce a Unified Efficient Points and Lines (UPAL) feature extractor that jointly extracts keypoints, line segments, and feature descriptors within a single lightweight architecture. A shared backbone provides common representations that feed different branches for point and line features. Line segments are recovered through an accelerated post-processing stage, an enhanced and highly efficient variant of the LSD algorithm. UPAL matches or exceeds state-ofthe-art performance in both point and line applications while significantly reducing computational cost, achieving, for instance, a 4x speedup and 10x smaller memory footprint over the ALIKED + DeepLSD pipeline. Code is publicly available at https://github.com/francois141/upal.

**Analysis:**

### 1. 论文主要贡献概述

论文提出了统一且高效的点—线局部特征提取器 **UPAL（Unified Efficient Points and Lines）**，在一个轻量级网络中联合提取关键点、线段及其描述子，而不再分别运行点特征和线特征模型。该方法结合共享特征骨干网络与点、线分支，并采用加速版 LSD 后处理恢复线段；据摘要所述，相比 **ALIKED + DeepLSD**，其计算速度提升约 4 倍、内存占用降低约 10 倍，同时保持或超过现有方法的性能。

### 2. 关键创新与方法路线

- **点线联合建模**：  
  通过共享 backbone 提取通用视觉表示，再分别提供点特征分支和线特征分支，使两类几何特征能够共享计算资源。

- **统一的特征描述框架**：  
  UPAL 不仅检测关键点和线段，还在同一系统中生成对应的局部描述子，有利于构建统一的点—线匹配和几何估计流程。

- **高效线段提取**：  
  线段并非完全依赖复杂的深度网络预测，而是通过增强、加速的 LSD 类后处理方法恢复。这种设计可能在保证线段质量的同时，显著降低网络推理开销。

- **面向效率的系统设计**：  
  论文重点解决现有点线方法中的三个瓶颈：点和线分别检测、网络规模不断增大，以及 CPU 上启发式线段处理速度较慢。其核心思想是通过共享计算和优化后处理实现速度、内存与精度之间的平衡。

### 3. 对领域的潜在影响

UPAL 的重要性主要体现在它可能推动点—线特征从“高精度但复杂”的研究方案走向更实用的统一前端：

1. **降低点线视觉系统的部署成本**：  
   4 倍速度提升和 10 倍内存减少对于嵌入式设备、移动平台和实时机器人系统尤其有价值。

2. **促进点线联合几何估计**：  
   点特征通常具有较好的局部定位能力，而线特征在弱纹理、建筑结构和长距离几何约束中更稳定。统一提取点和线，有助于提升相机位姿估计、三维重建和视觉定位的鲁棒性。

3. **减少工程复杂度**：  
   现有系统往往需要拼接多个检测器、描述子和后处理模块。UPAL 若具有良好的接口和兼容性，可以简化多视图视觉前端的设计与维护。

4. **提供效率—性能折中范例**：  
   该工作表明，提升视觉前端效率不一定要依赖更大的网络，也可以通过共享表示、合理的任务分支以及更高效的经典算法后处理实现。

### 4. 可能受益的相关领域与应用

- **视觉 SLAM 与视觉里程计**：  
  点和线的联合约束可提高在室内、走廊、建筑物等结构化环境中的跟踪与定位稳定性。

- **多视图立体与三维重建**：  
  线段能够补充点匹配，尤其适合墙面边界、桌面边缘、建筑轮廓等几何结构明显的场景。

- **相机位姿估计与视觉定位**：  
  线特征可提供更长、更稳定的几何约束，适用于室内定位、城市环境定位和地图匹配。

- **增强现实与混合现实**：  
  实时点线特征提取有助于场景跟踪、平面和结构识别。

- **机器人导航与自动驾驶**：  
  在计算资源受限的平台上，低内存和高吞吐特征前端具有实际价值，特别是道路边界、建筑边缘和室内结构线明显的环境。

- **工业检测与三维测量**：  
  机械零件、装配结构和规则边缘通常包含大量线特征，点线联合匹配可能提升配准和姿态估计性能。

- **边缘计算和移动视觉应用**：  
  较小的内存占用有利于部署在 GPU 显存有限或需要低功耗运行的设备上。

### 5. 根据摘要可以推断的局限性

摘要没有提供完整实验细节，因此以下问题仍需通过论文正文和代码验证：

- **点线联合是否存在性能权衡**：  
  共享 backbone 能降低计算，但点特征和线特征可能需要不同的表示。某些场景下，联合架构是否会牺牲点定位精度、线段完整性或描述子区分能力，需要更细致的消融实验。

- **线段后处理的实际瓶颈尚不明确**：  
  论文称其采用了高效的 LSD 变体，但如果该步骤仍主要依赖 CPU，端到端速度可能受硬件、线程配置和图像分辨率影响。摘要中的 4 倍加速是否适用于完整 pipeline，而不仅是单次推理，仍需确认。

- **对场景类型的依赖**：  
  线特征在建筑、室内和人工环境中通常较有效，但在自然场景、纹理丰富但结构边缘较少的场景中，线特征的收益可能有限。

- **描述子和跨域泛化能力尚未知**：  
  摘要未说明训练数据、监督方式、跨数据集表现以及对光照变化、视角变化、运动模糊和低照度的鲁棒性。轻量化模型可能在极端条件下更容易出现性能下降。

- **与最新方法的比较范围需要核实**：  
  摘要主要提到与 ALIKED + DeepLSD 的比较，但“达到或超过 state of the art”需要结合具体基准、评价指标和运行硬件来判断，不能仅凭摘要确认其在所有点、线任务上都全面领先。

- **点线匹配与下游任务的兼容性**：  
  统一提取特征并不自动意味着点线联合匹配、几何模型估计或 SLAM 后端也已统一。其描述子是否适合现有匹配器，以及在实际位姿估计和重建中的收益，还需要端到端评估。

总体而言，UPAL 的趣味性在于它不是单纯提出更大的特征网络，而是从系统角度重新设计点线特征前端：用共享表示减少重复计算，用轻量后处理保留线段提取能力，并试图在精度、速度和内存之间取得更好的平衡。这种思路对实时多视图视觉和资源受限设备具有较强的应用潜力。

**Key Findings:**

- We introduce a Unified Efficient Points and Lines (UPAL) feature extractor that jointly extracts keypoints, line segments, and feature descriptors within a single lightweight architecture.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.19894v1)
- [arXiv](https://arxiv.org/abs/2608.19894v1)

---

