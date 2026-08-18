time: 20260818

# Arxiv Computer Vision Papers - 2026-08-18

## Executive Summary

# ArXiv 计算机视觉日报执行摘要  
**发布日期：2026-08-17**

> 注：以下判断主要依据论文标题及研究问题进行归纳；在未提供论文摘要、实验结果和代码的情况下，对具体方法细节与性能结论应保持审慎。

## 1. 总体趋势

本期 10 篇论文呈现出几个清晰的研究主线：

1. **视觉-语言-动作模型（VLA）向机器人真实部署演进**  
   - `τ₀-VLA` 将层次化机器人基础模型、世界模型和测试时计算结合起来。  
   - `HAF` 面向人形机器人全身移动操作，尝试将通用 VLA 适配到更复杂的运动控制场景。  
   - `MatchingPolicy` 则关注跨物体、跨实例的对应关系建模和上下文学习。  
   这些工作表明，研究重点正从“让模型理解指令”转向“让模型在新物体、新环境和长时序任务中可靠行动”。

2. **3D 几何先验与生成模型融合**  
   - `SplatGuide` 使用 3D Gaussian Splatting 提取几何先验，服务于无需相机位姿的新视角合成。  
   - `SQuad` 探索次二次复杂度的注意力蒸馏，以降低视频生成的计算开销。  
   共同趋势是：生成模型不再单纯依赖大规模数据和全局注意力，而是引入显式几何结构或更高效的计算机制。

3. **面向实际环境的鲁棒视觉感知**  
   - `DRAFE` 研究跨城市、细粒度交通目标检测。  
   - `Ultra` 针对恶劣天气下的图像恢复与分割协同。  
   - `DPNet` 关注视觉导航中的“死路”预测与规避。  
   - `Calibration-Free Vehicle Speed Estimation` 试图在无相机标定条件下进行车辆速度估计。  
   这些论文共同体现了从标准数据集性能向**跨域泛化、低先验依赖和安全性**转移的方向。

4. **评测、诊断和可靠性受到更多重视**  
   - `TRACE-Bench` 不仅评估多参考图像生成结果，还试图分解和诊断不同失败模式。  
   - `DPNet` 和 `Ultra` 也体现出对系统可靠性和恶劣条件下性能的关注。  
   这说明领域开始更加重视“模型为什么失败”以及“能否在真实场景中稳定工作”，而不仅是单一总体指标。

---

## 2. 可能最具创新性或影响力的论文

### **τ₀-VLA**
将**世界模型引导的测试时计算**引入层次化机器人基础模型，是本期最值得关注的方向之一。其潜在价值在于：机器人可以在执行前进行内部预测、规划或候选动作评估，从而提升复杂任务中的决策质量。若论文能够证明测试时额外计算在真实机器人任务中带来稳定收益，它可能推动 VLA 从一次性策略预测转向“推理—验证—执行”的工作范式。

### **HAF**
面向人形机器人的全身移动操作，结合**层次化动作流**与**频谱潜空间强化学习**，覆盖导航、平衡、抓取和操作等耦合问题。人形机器人控制的难点不只是视觉理解，而是多时间尺度、多自由度动作协调，因此该工作可能对通用 VLA 的具身化具有较高参考价值。

### **SplatGuide**
将 3D Gaussian Splatting 的几何信息用于**无位姿新视角合成**，代表了显式 3D 表示与生成式视觉模型结合的趋势。若其方法确实能在缺乏准确相机位姿的情况下保持几何一致性，可能对快速场景建模、机器人视觉和沉浸式内容生成具有实用意义。

### **TRACE-Bench**
多参考图像生成目前往往缺少细粒度、可诊断的评测体系。该基准若能区分身份保持、属性融合、布局一致性、关系建模和视觉质量等因素，将有助于推动生成模型从“总体评分竞争”转向更有解释性的能力评估。

### **SQuad**
视频生成的注意力计算成本是大规模应用的主要瓶颈之一。通过次二次复杂度的注意力蒸馏降低计算量，可能对长视频生成、实时生成和消费级硬件部署尤其重要。其关键价值取决于效率收益是否伴随较小的视频质量、时序一致性损失。

---

## 3. 新兴研究方向与技术

### 3.1 测试时推理与世界模型驱动的控制
`τ₀-VLA` 体现了将计算预算从训练阶段延伸到部署阶段的趋势。未来可能出现：
- 基于世界模型的动作候选模拟；
- 测试时规划与策略重排序；
- 面向不确定性的主动感知；
- 按任务难度动态分配推理计算。

### 3.2 VLA 的层次化和跨对象泛化
`HAF` 与 `MatchingPolicy` 分别从动作层次和视觉对应关系入手，解决通用策略迁移问题。潜在方向包括：
- 任务级、技能级和关节级控制的统一；
- 基于对应关系的跨物体 in-context learning；
- 新物体、新布局和新任务组合下的少样本适应；
- VLA 与强化学习、模仿学习的混合训练。

### 3.3 显式 3D 表示辅助生成
`SplatGuide` 说明 Gaussian Splatting 不仅可用于渲染，也可作为生成模型的几何约束或中间表示。后续可能进一步结合：
- 3D Gaussian、NeRF 与扩散模型；
- 无标定或弱标定的三维重建；
- 物理一致的新视角和视频生成；
- 面向机器人操作的可交互场景表示。

### 3.4 无标定、跨域和恶劣条件下的感知
`Calibration-Free Vehicle Speed Estimation`、`DRAFE` 和 `Ultra` 共同体现了降低部署门槛的方向：
- 从单一城市或天气条件迁移到新域；
- 减少对相机内外参和人工标注的依赖；
- 利用恢复、分割和检测任务之间的互补信息；
- 通过域不变特征或不对称融合提升跨域稳定性。

### 3.5 高效生成与可解释评测
`SQuad` 关注计算效率，`TRACE-Bench` 关注能力分解，二者分别对应生成模型发展的两个瓶颈：
- 如何以更低成本生成更长、更高质量的视频；
- 如何细粒度判断模型究竟在哪些能力上有效或失败。

---

## 4. 建议优先阅读全文的论文

### 第一优先级

1. **τ₀-VLA**  
   适合关注机器人基础模型、世界模型、测试时推理和具身智能的研究人员。重点查看：世界模型如何训练、测试时计算如何分配、是否有真实机器人验证，以及额外计算带来的收益是否具有泛化性。

2. **HAF**  
   适合关注人形机器人、强化学习和全身控制的研究人员。重点查看：层次化动作流与底层控制器的接口、频谱潜空间的具体作用、训练稳定性及 sim-to-real 结果。

3. **SplatGuide**  
   适合关注 3D 视觉、视图合成、Gaussian Splatting 和生成模型的研究人员。重点查看：无位姿设定是否真正成立、几何先验如何注入、对遮挡和大视角变化的处理效果。

4. **TRACE-Bench**  
   适合关注视觉生成评测、多模态生成和基准设计的研究人员。其价值不仅在于模型结果，也在于是否提供了可复用的错误分类、诊断协议和具有挑战性的测试集。

### 第二优先级

5. **SQuad**  
   如果研究重点是视频生成、长序列建模或推理效率，应优先阅读；否则可在上述论文之后关注。

6. **MatchingPolicy**  
   对机器人策略泛化、视觉对应关系和跨实例学习很有价值，尤其适合研究少样本任务迁移的读者。

7. **Ultra**  
   适合关注恶劣天气感知、无监督学习和恢复—分割联合建模的研究人员。

### 面向应用的优先阅读

8. **DRAFE**：交通检测、跨城市部署和域泛化。  
9. **DPNet**：无人机导航、安全避障和视觉预测。  
10. **Calibration-Free Vehicle Speed Estimation**：智能交通、视频分析和低成本摄像头部署。

## 一句话总结

本期论文的核心信号是：计算机视觉研究正在从单一任务和静态基准，进一步走向**具身智能中的推理与控制、显式 3D 约束、高效生成，以及跨域、恶劣环境和真实部署下的可靠性**；其中 `τ₀-VLA`、`HAF`、`SplatGuide`、`TRACE-Bench` 和 `SQuad` 最值得优先深入阅读。

---

## Table of Contents

1. [$τ_0$-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation](#2608.16885v1)
2. [SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis](#2608.16863v1)
3. [HAF: Adapting Generalist VLAs to Humanoid Whole-Body Loco-manipulation via Hierarchical Action Flow and Spectral Latent RL](#2608.16837v1)
4. [Calibration-Free Vehicle Speed Estimation: A Monocular Keypoint-Template Approach](#2608.16785v1)
5. [TRACE-Bench: Decomposing and Diagnosing Multi-Reference Image Generation](#2608.16765v1)
6. [MatchingPolicy: Correspondence-Aware Policy Enables Cross-Object In-Context Learning](#2608.16715v1)
7. [DPNet: Efficient Dead-End Prediction and Avoidance for Vision-Based UAV Navigation](#2608.16640v1)
8. [DRAFE: Domain-Robust Asymmetric Fusion of Heterogeneous Detection Transformers for Cross-City Fine-Grained Traffic Object Detection](#2608.16632v1)
9. [Ultra: Unsupervised Cross-Task Optimization for Reliable Restoration Segmentation Collaboration under Adverse Weather](#2608.16589v1)
10. [SQuad: Sub-Quadratic Attention Distillation for Efficient Video Generation](#2608.16585v1)

---

## Papers

<a id='2608.16885v1'></a>
## [$τ_0$-VLA: a Hierarchical Robot Foundation Model with World-Model-Guided Test-Time Computation](https://arxiv.org/abs/2608.16885v1)

**Authors:** Xiaowei Cai, Yunuo Cai, Bingao Chen, Jingxiao Chen, Zhi Chen, Siyuan Feng, Tengyu Hou, Jingshun Huang, Han Jiang, Runkun Ju, Dong Li, Mingxiang Li, Shaowei Li, Xinchen Li, Yifan Li, Yi Liu, Zhongyuan Liu, Jianlan Luo, Junwen Miao, Ruiqi Ni, Buqing Nie, Mingjie Pan, Xinlin Ren, Jianheng Song, Jiaxu Wang, Peiqi Wang, Sen Wang, Xiaoyan Wang, Dafeng Wei, Dongming Wu, Pengwei Xie, Pu Yang, Hangjian Ye, Xiangyu Yue, Jinyu Zhang, Qinglin Zhang, Xueyong Zhao, Pengfei Zhou, Yue Zhou

**Published:** 2026-08-17

**Categories:** cs.RO

**Abstract:**

Long-horizon robot manipulation requires a robot to both execute individual skills reliably and sequence them coherently over extended tasks. Most hierarchical vision-language-action (VLA) models make each such decision with a single forward pass, leaving no mechanism to allocate additional computation to difficult or consequential choices. We introduce $τ_0$-VLA, a hierarchical robot foundation model that formulates high-level subtask generation as a compute-scalable inference problem through world-model-guided test-time computation. At each inference step, the high-level policy uses execution memory to generate a subtask and, when needed, searches over alternatives before committing to its output. A low-level policy then executes the generated subtask across multiple robot embodiments. The policy is trained on 40,115 hours of heterogeneous real-world data with multimodal co-training. Across in-domain and distribution-shifted settings, allocating additional test-time computation substantially improves next-subtask prediction accuracy, and these gains translate into higher closed-loop success on long-horizon robot manipulation tasks.

**Analysis:**

## 1. 摘要翻译

长时程机器人操作要求机器人既能可靠执行单项技能，又能在较长时间范围内合理组织技能序列。现有多数分层视觉-语言-动作（VLA）模型通过一次前向传播完成每次高层决策，无法为困难或关键决策分配额外计算。本文提出 **τ0-VLA**：一种结合世界模型引导测试时计算的分层机器人基础模型。高层策略利用执行记忆生成下一子任务；当预测不确定时，通过搜索多个候选子任务，利用世界模型预测其视觉后果，并由价值模型评估，再由反思模型完成最终决策。低层策略负责跨多种机器人本体执行子任务。模型使用约40,115小时异构真实机器人数据和多模态数据训练。在分布内及分布偏移环境中，增加测试时计算均能提升下一子任务预测准确率，并进一步提高长时程真实机器人任务的闭环成功率。

## 2. 方法动机

**驱动力**：长时程任务的瓶颈往往不是动作控制，而是“下一步做什么”。机器人可能把错误子任务执行得很精准，但已经造成不可逆的环境变化，因此需要在执行前比较候选方案。

**现有痛点**：  
1. 单次高层推理固定计算预算，困难决策与普通决策待遇相同；  
2. 语言规划通常只比较文本逻辑，缺乏对真实视觉后果的验证；  
3. 直接执行全任务指令容易遗忘已完成阶段，难以处理不可见进度（如是否已加盐）；  
4. 一步采样或Best-of-N只能比较当前候选，不能评估多步后果。

**核心假设**：子任务是适合测试时搜索的决策粒度；若能预演候选子任务产生的视觉状态，并据此评分，就能减少错误承诺。

## 3. 方法设计详解

### 整体Pipeline

输入为任务指令、当前多视角图像、机器人状态、上一子任务和执行记忆。

1. **高层提议与记忆更新**：提议模型 \(P\)（Qwen3.5-9B）读取上下文，输出与当前视觉状态对齐的新记忆 \(M_t\) 和直接子任务 \(z_t^{dir}\)。  
2. **自适应路由**：根据生成token概率及记忆字段的logit间隔判断置信度。高置信度直接执行；低置信度进入测试时计算（TTC），避免所有决策都承担搜索成本。  
3. **候选生成**：对每个保留分支调用 \(P\) 多次，生成 \(N\) 个开放式语言子任务，而非从固定动作词表中选择。  
4. **后果想象**：世界模型 \(W\) 接收当前头部相机图像和候选子任务，预测执行结束时的头部图像：
\[
\hat{o}=W(\tilde{o},z)
\]
它只预测终止视觉状态，而非完整动作轨迹。  
5. **价值评估**：价值模型 \(V\) 根据全局任务、候选子任务和预测图像输出质量分数：
\[
v=V(\ell,z,\hat{o})
\]
分数表示该候选是否推动任务进展。  
6. **束搜索扩展**：每层保留累计价值最高的 \(B\) 条分支，并继续扩展至深度 \(D\)：
\[
S(b\oplus z)=S(b)+v(b\oplus z)
\]
分支内部维护独立记忆和预测图像，但不会污染真实执行记忆。  
7. **反思与提交**：反思模型 \(F\) 读取真实上下文和最终保留分支，生成最终子任务 \(z_t^\star\)。它可以选择候选，也可以综合后生成候选集之外的新子任务。  
8. **低层执行**：低层VLA（Qwen3.5-2B + Mixture-of-Transformers动作专家）根据图像、机器人状态、子任务和本体控制元数据生成动作块，并通过条件流匹配输出动作。真实执行后的观察在下一轮用于修正记忆，形成闭环。

### 低层策略

不同机器人被映射到统一的40维状态/动作空间，覆盖双臂末端位姿、夹爪、腰部、底盘速度和关节。通过掩码屏蔽不存在的控制维度，因此同一策略可支持固定底座、双臂和移动操作。训练分为知识隔离联合训练、端到端联合训练、任务特定适配三阶段。

### 记忆修正

作者不只保存历史，而是人为扰动记忆，构造“记忆落后、记忆超前、失败后未回退”等样本，训练模型依据当前视觉状态恢复正确进度。这是其区别于普通历史拼接的重要设计。

## 4. 方法对比与创新

本质区别在于：现有分层VLA通常“先生成子任务，再执行”；τ0-VLA则是“先生成候选、预测后果、比较分支，再提交”。创新主要包括：  
- 将高层子任务生成转化为可扩展的测试时推理问题；  
- 在执行前用视觉世界模型评估候选后果；  
- 结合束搜索、价值模型和开放式反思；  
- 引入可纠错执行记忆；  
- 以置信度路由计算，兼顾速度和决策质量。

适合任务步骤多、子任务有明显先后依赖、错误代价高或存在分布偏移的场景，如整理、烹饪、清洁和移动操作。

## 5. 实验分析

作者在清洁房间、备料、炒菜、奶茶制作、图书整理等真实机器人任务上评估，并比较Plan Once、Best-of-N和TTC。代表性结论是：TTC在OOD图书整理中的下一子任务准确率由Plan Once的50.0%提升至74.0%；闭环测试中，奶茶成功率由5/10提升至7/10，图书整理由6/10提升至9/10。优势是能按需增加推理、跨本体执行、显式维护进度。局限是世界模型预测可能不准确，价值模型存在偏差，搜索成本较高，且真实实验规模仍较小。

## 6. 实用指南

论文提供项目主页，但文中未明确说明完整代码、模型权重和数据是否公开。复现需准备：异构机器人轨迹、分段子任务标注、起止视觉帧，以及提议、世界、价值和反思四类模型。关键超参数为分支数 \(N\)、束宽 \(B\)、搜索深度 \(D\)；三者决定计算量。低层动作块长度为30，采用10次Euler更新。迁移到其他任务时，应重新定义子任务边界、价值标准和世界模型训练对；若视觉终态不足以判断进度，还需加入状态、触觉或结构化验证器。

## 7. 总结

**核心思想：先想象后果，再决定下一步。**

**速记版Pipeline**：  
1. 读取任务、画面和执行记录；  
2. 低置信度时提出多个下一步；  
3. 想象每一步完成后的画面并评分；  
4. 保留多步最有希望的方案并反思；  
5. 将最终子任务交给通用控制器执行。

**Key Findings:**

- We introduce $τ_0$-VLA, a hierarchical robot foundation model that formulates high-level subtask generation as a compute-scalable inference problem through world-model-guided test-time computation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16885v1)
- [arXiv](https://arxiv.org/abs/2608.16885v1)

---

<a id='2608.16863v1'></a>
## [SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis](https://arxiv.org/abs/2608.16863v1)

**Authors:** Yejun Zhang, Zihan Wang, Xu Ji, Yihao Wang, Yuxin Hou, Junyuan Fang, Juho-Matti Kilpeläinen, Arno Solin, Hamed Rezazadegan Tavakoli, Esa Rahtu, Juho Kannala

**Published:** 2026-08-17

**Categories:** cs.CV

**Abstract:**

Generating photorealistic novel views from unposed images requires both 3D geometric understanding and the ability to synthesize unseen content. A natural strategy combines feed-forward 3DGS reconstruction with multi-view diffusion. Yet prior pipelines extract at most one signal from the reconstruction, either pixel rendering or learned features, while none exploits per-Gaussian visibility for occlusion-aware reference selection. This *information disconnect* leaves renderable geometry, visibility cues, and learned features unused. SplatGuide closes this disconnect by reusing a single 3DGS scene across three complementary roles. Rendered images provide pixel-aligned geometric conditioning. Per-Gaussian source-view indices are rendered into a target-view voting map for occlusion-aware reference selection. Reconstruction tokens supply feature-level guidance via cross-attention. All three signals derive from the same reconstruction forward pass. Across RealEstate10K, DL3DV, Tanks-and-Temples, and Mip-NeRF 360, SplatGuide achieves state-of-the-art pose-free novel view synthesis. On RealEstate10K, with a moderate number of input views, it surpasses the ground-truth-pose baseline.

**Analysis:**

## 1. 摘要翻译

从无位姿图像生成逼真新视角，需要同时具备三维几何理解能力与未观测内容合成能力。一个自然方案是结合前馈式3D Gaussian Splatting（3DGS）重建与多视图扩散模型。然而，已有方法通常只利用重建结果中的一种信号：像素渲染或学习特征，且没有利用每个高斯的可见性进行遮挡感知的参考视图选择。SplatGuide通过同一个3DGS场景承担三种互补功能：渲染图像提供像素对齐的几何条件；将每个高斯对应的源视图索引渲染为目标视角投票图，实现遮挡感知的参考选择；重建token通过交叉注意力提供特征级引导。三类信号均来自一次重建前向推理。在RealEstate10K、DL3DV、Tanks-and-Temples和Mip-NeRF 360上，SplatGuide取得了先进的无位姿新视角合成效果，并在输入视图足够多时超过真实位姿基线。

## 2. 方法动机

**驱动力：**重建模型能恢复相机、几何和可渲染场景，但不会补全不可见区域；扩散模型能生成内容，却依赖准确位姿且缺少显式三维约束。作者认为瓶颈不是重建缺信息，而是现有方法没有充分使用重建信息。

**痛点：**位姿-only方法丢弃场景几何；像素条件方法只能提供局部渲染，且常绑定视频扩散架构；特征方法需要昂贵的稠密特征；主流参考选择依赖时间邻近或相机距离，无法处理遮挡和视图冗余。

**核心假设：**同一3DGS同时包含“几何、可见性、语义/场景上下文”三类互补先验，将其联合注入扩散模型，比单独传递位姿或单一条件更有效。

## 3. 方法设计详解

### Pipeline

1. **前馈重建：**输入N张无位姿RGB图像，经WorldMirror等模型预测参考相机位姿、像素对齐的3D高斯及相机/寄存器token。每个高斯保留其源图像编号，无需额外监督。
2. **几何条件：**在目标位姿和参考位姿渲染3DGS，得到粗糙但几何一致的目标/参考图。将渲染结果编码为VAE latent，并与扩散噪声latent、Plücker射线编码、视图类型mask拼接；U-Net首层扩展通道，新权重零初始化以保持预训练模型行为。
3. **可见性选择：**对高斯降采样，在目标视角执行首命中深度测试：每个像素只保留最近高斯，并用源视图对应的独特颜色编码。统计各颜色像素数  
   \(S(k)=\sum_p \mathbf1[\hat v(p)=k]\)，即候选视图k解释目标可见表面的程度，再贪心选Top-K。
4. **选择修正：**DeDup过滤与已选视图覆盖相同区域的候选；PoseAug寻找距离当前上下文最远的目标，并补入其最近候选，避免某些目标没有邻近参考。
5. **扩散生成：**将选中的参考RGB及其Plücker坐标、目标渲染图、目标射线和噪声输入SEVA扩散模型；相机token与4个寄存器token经线性映射后，通过专用交叉注意力注入，预测噪声并生成最终新视图。训练时冻结重建模型和VAE，仅优化扩散模型。

### 关键设计意义

首命中而非透明度混合保证索引解码准确，并显式处理遮挡；DeDup与PoseAug分别解决“视图重复”和“目标缺少近邻”问题。渲染负责局部空间对齐，token补充不可由可见像素表达的全局结构和纹理统计。

## 4. 方法对比与适用性

其本质区别不是新增复杂三维表示，而是把一个重建场景作为统一接口，同时服务于渲染、视图选择和特征条件。创新集中在：三信号联合利用、基于高斯源视图索引的遮挡感知选择，以及重建骨干与生成器的解耦。适合无标定照片、稀疏多视图、相机大幅外推及上下文预算有限的场景；动态场景、严重纹理缺失和大基线输入则较不适合。

## 5. 实验分析

作者在四个数据集上比较回归和扩散方法，并进行选择策略、条件信号、候选池规模及重建骨干替换实验。代表性结论是：RealEstate10K三视图时较无位姿SEVA提升约3.05 dB，九视图时超过真实位姿SEVA；上下文预算为6时，所提选择策略明显优于相机距离、时间和surfel可见性选择。优势是几何一致性强、外推能力好、选择开销极低且可零样本替换重建模型。局限是三种信号共享同一重建误差，动态内容或重建失败会同步损害全部条件。

## 6. 实用指南

论文正文未明确给出代码开源声明，复现需实现WorldMirror/AnySplat重建、3DGS渲染、索引颜色投票、DeDup/PoseAug及SEVA改造。关键设置包括重建分辨率448、生成与渲染576、上下文总长21、DDIM 50步、CFG 2.0、训练25,000步；选择阶段约使用10%的高斯。迁移到其他扩散模型时，可将渲染latent通过输入通道拼接，将重建token接入交叉注意力，并复用上游可见性选择器。

## 7. 总结

**核心思想：**一套高斯场景，统一提供几何、可见性和特征先验。

**速记版Pipeline：**
1. 从无位姿图片预测相机和3D高斯场景。  
2. 渲染目标布局，作为扩散生成的空间锚点。  
3. 用高斯来源和遮挡关系挑选互补参考图。  
4. 注入重建token，扩散模型补全不可见区域。

**Key Findings:**

- Generating photorealistic novel views from unposed images requires both 3D geometric understanding and the ability to synthesize unseen content.
- Across RealEstate10K, DL3DV, Tanks-and-Temples, and Mip-NeRF 360, SplatGuide achieves state-of-the-art pose-free novel view synthesis.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16863v1)
- [arXiv](https://arxiv.org/abs/2608.16863v1)

---

<a id='2608.16837v1'></a>
## [HAF: Adapting Generalist VLAs to Humanoid Whole-Body Loco-manipulation via Hierarchical Action Flow and Spectral Latent RL](https://arxiv.org/abs/2608.16837v1)

**Authors:** Langzhe Gu, Chengkai Hou, Meng Li, Xinhua Wang, Jiaming Liu, Xinyuan Lv, Bowei Zhang, Shuanghao Bai, Guangrun Li, Jingyang He, Gaole Dai, Ziluo Ding, Zhiyuan Xu, Kuan Cheng, Jian Tang, Zhengping Che, Shanghang Zhang

**Published:** 2026-08-17

**Categories:** cs.RO, cs.AI

**Abstract:**

Humanoid robots hold great promise as general-purpose agents in human-centered environments, yet generalist vision-language-action (VLA) foundation models are not readily applicable to humanoid whole-body loco-manipulation. The high dimensionality and interdependence of humanoid motions make it challenging for conventional single-stage VLA architectures to coordinate locomotion, waist posture, and dual-arm manipulation effectively. Moreover, policies trained through offline behavior cloning can remain suboptimal during real-world deployment. Although online reinforcement learning can refine policies through real-world interaction, directly tuning large VLA backbones demands excessive computation and may introduce safety risks during real-robot exploration. To address these bottlenecks, we introduce HAF (Humanoid Adaptation Framework), a two-part framework consisting of HAF-VLA and HAF-Steer that transfers off-the-shelf generalist VLA foundation models to humanoid whole-body loco-manipulation. HAF-VLA is a hierarchical action-flow generator built on a pretrained flow-matching VLA. It splits full-body action denoising into three sequential stages with stage embeddings and cross-stage KV caches that retain kinematic dependencies, avoiding incoherent whole-body actions from one-shot generation. On top of the frozen HAF-VLA, HAF-Steer is a latent offline-to-online RL pipeline that leverages flow-matching invertibility and DCT-based dimensionality reduction to restrict RL optimization to a compact noise subspace and train a regularized SAC policy. This avoids updating the large VLA backbone and enables efficient real-world policy refinement. Evaluated on seven real-world humanoid loco-manipulation tasks, HAF surpasses vanilla single-stage VLA baselines and improves whole-body coordination and task performance. Project website: https://grange007.github.io/HAF .

**Analysis:**

## 1. 摘要翻译

人形机器人有望成为服务于以人为中心环境的通用智能体，但通用视觉-语言-动作（VLA）基础模型尚不能直接适用于人形机器人全身移动操作。人形机器人的动作维度高、各关节运动相互依赖，使传统单阶段VLA难以协调行走、腰部姿态和双臂操作。此外，离线行为克隆策略在真实部署时可能因分布偏移而性能下降；直接通过在线强化学习调整大型VLA骨干网络又计算昂贵且存在安全风险。

为此，本文提出人形机器人适配框架HAF，由HAF-VLA和HAF-Steer组成。HAF-VLA建立在预训练流匹配VLA之上，将全身动作去噪划分为三个阶段：依次生成移动与头部动作、腰部姿态、双臂操作，并通过阶段嵌入和跨阶段KV缓存保留运动学依赖，避免一次性生成造成的不协调动作。HAF-Steer则冻结HAF-VLA，通过流匹配可逆性反推出示范动作对应的初始噪声，再利用DCT进行时序降维，仅在紧凑噪声子空间中训练带行为克隆正则的SAC策略，从而实现低成本、安全的真实机器人策略优化。实验表明，HAF在七项真实人形机器人移动操作任务上优于单阶段VLA基线。

## 2. 方法动机与核心假设

**驱动力：** 人形机器人首先需要稳定移动和调整躯干，之后才能进行精细操作；同时，离线策略需要适应真实环境，但不应破坏大型VLA已有能力。

**现有痛点：**
1. 单阶段VLA同时生成全身动作，忽略“移动—姿态—操作”的运动学依赖，易出现下肢漂移和上肢补偿抖动。
2. 人形动作空间与时序噪声维度巨大，直接在线RL探索成本高且不安全。
3. 仅重复一个噪声向量虽能降维，却失去时间变化；优化完整噪声又搜索空间过大。

**核心假设：** 以运动学依赖为顺序生成全身动作，比全维同步生成更稳定；流噪声中的低频时序成分足以表达有用的策略修正，因此可冻结VLA，仅优化少量谱系数。

## 3. 方法设计详解

### 3.1 HAF-VLA：分层动作流生成

输入为当前图像 \(I_t\)、机器人本体状态 \(q_t\) 和语言指令 \(l\)。模型输出长度为 \(H=100\) 的动作块，每次仅执行前40步，再重新推理。

每个动作被拆为四类：移动/技能模式、头部、腰部、双臂操作。三阶段采用累积动作集合：

- **阶段1：** 移动+头部；
- **阶段2：** 阶段1+腰部；
- **阶段3：** 阶段2+双臂。

视觉语言前缀只编码一次，形成共享KV缓存 \(P_t\)。每个阶段使用独立高斯噪声和阶段嵌入，并执行10步流匹配去噪。阶段1得到动作后重新编码为缓存 \(C_t^1\)，阶段2据此生成并形成 \(C_t^2\)，阶段3同时使用 \(P_t,C_t^1,C_t^2\) 生成最终完整动作。中间结果不执行，仅作为后续阶段的条件。

训练时使用真实动作进行teacher forcing构造跨阶段缓存，避免早期预测误差累积。对每个阶段，噪声与真实动作线性插值，模型学习目标速度场；未激活动作维度不被删除，而是用零速度目标约束。推理时则使用前一阶段的预测动作构造缓存，形成“先稳定底盘、再调整身体、最后精细操作”的粗到细过程。

### 3.2 HAF-Steer：谱域潜变量RL

对示范动作 \(A_i^*\)，利用冻结流模型进行数值反向积分，恢复其初始噪声 \(\epsilon_i^*\)。沿时间维做DCT，仅保留前 \(K=8\) 个低频系数：

\[
c_i^*=\mathrm{DCT}_K(\epsilon_i^*),\quad
z_i^*=(c_i^*-\mu_c)/(\sigma_c+\delta)
\]

因此优化维度由 \(H\times D\) 降为 \(K\times D\)，并天然抑制高频抖动。

训练分两步：
1. **行为克隆：** 让谱域策略预测示范系数；
2. **混合离线-在线SAC：** 同时从示范缓冲区和真实机器人缓冲区采样，使用成功终止奖励，并对离线样本加入BC正则，防止策略偏离专家分布；训练过程中逐步降低离线数据比例。

部署时，策略输出归一化谱动作，经反归一化和IDCT补零恢复完整时序噪声，再送入冻结VLA生成动作。对HAF-VLA，仅替换阶段3的初始噪声，阶段1、2仍正常生成并提供缓存。

## 4. 方法对比、创新与适用场景

HAF的本质区别不是更换VLA骨干，而是改变**动作生成结构**与**RL优化变量**：前者显式编码运动学顺序，后者在低维、平滑且可逆的噪声空间中适配策略。

主要创新包括：
1. 面向人形运动依赖的累积式三阶段动作流；
2. 用跨阶段KV缓存传递已生成动作，而非简单硬解耦；
3. 用DCT保留多种低频时间模式，兼顾降维与时序表达；
4. 通过流反演把示范动作转成可监督的谱域RL目标。

最适合长时程、需行走并结合双臂操作的任务，如搬运、取物、投掷和家务操作；不适合需要极高频接触反馈或基础VLA本身严重错误的场景。

## 5. 实验分析

作者在两种TienKung人形机器人、七项真实家务任务上评估。HAF-VLA平均归一化得分达到70.5%，明显高于最强基线π0.5的53.3%；消融实验显示，移动优先的层级设计优于全关节同步去噪和“先手臂后移动”。

HAF-Steer在玩具收纳、篮子搬运的ID/OOD测试中普遍提升成功率，说明谱域BC加在线SAC能适应目标位置变化。

**优势：** 不需更新大型VLA，探索维度低、动作更平滑，并能利用原有通用先验。  
**局限：** 三阶段推理增加计算与延迟；谱域RL受冻结VLA能力上限约束，极端OOD错误难以修复。

## 6. 实用指南

论文未明确提供代码仓库，给出了项目网站；复现需实现流匹配正向/反向数值积分、阶段mask与KV缓存、DCT/IDCT谱参数化及混合SAC。关键设置包括动作块长100、执行步长40、每阶段10次去噪、谱系数 \(K=8\)、离线BC预训练、成功终止奖励和BC正则。迁移到其他流匹配VLA时，只需替换其流映射与机器人动作定义；迁移到其他具身任务，则需重新设计动作分组、层级顺序和示范噪声统计。

## 7. 总结

**核心思想：** 按身体依赖生成，并在低维噪声中适配。

**速记版Pipeline：**
1. 将全身动作按“移动、腰部、手臂”分层。
2. 逐层生成，并把前层结果作为后层上下文。
3. 从示范动作反推出流噪声并压缩为低频时间成分。
4. 先模仿示范，再用真实交互强化优化。
5. 冻结原VLA，仅用优化后的噪声生成最终动作。

**Key Findings:**

- To address these bottlenecks, we introduce HAF (Humanoid Adaptation Framework), a two-part framework consisting of HAF-VLA and HAF-Steer that transfers off-the-shelf generalist VLA foundation models to humanoid whole-body loco-manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16837v1)
- [arXiv](https://arxiv.org/abs/2608.16837v1)

---

<a id='2608.16785v1'></a>
## [Calibration-Free Vehicle Speed Estimation: A Monocular Keypoint-Template Approach](https://arxiv.org/abs/2608.16785v1)

**Authors:** Gaofeng Su, Keya Li, Raja Sengupta, Kara M. Kockelman

**Published:** 2026-08-17

**Categories:** cs.CV

**Abstract:**

This paper proposes a calibration-free framework for reliably and effectively estimating vehicle speeds from monocular videos, without relying on roadway features, camera calibration, or roadway-feature-based reference objects. The proposed framework estimates vehicle speeds using a 36-keypoint vehicle template and a homography matrix updated at each frame. A YOLO-based keypoint detection module is trained on diverse datasets, and two estimation strategies are compared: keypoint-only tracking and warped optical flow with dense spatial aggregation. Speed is estimated by projecting displacements into metric space using the homography, with validation conducted on over 400 video clips from roadside and overhead datasets, covering speeds from 30 to 100 mph. The method achieves reliable speed estimation on the VS13 and BrnoCompSpeed datasets, with the warped optical flow method delivering MAEs of 15.0% and 9.7%, respectively, and 77.9% and 93.1% of estimates falling within +/-20% error. After applying a 10% trim to remove edge-of-frame outliers, performance improves to MAEs of 11.7% and 7.6%, with within-+/-20% accuracy increasing to 85.3% and 95.4%. This work addresses key limitations of existing vision-based approaches and enables low-cost and efficient speed enforcement using portable devices such as dashcams and smartphones, thereby supporting citizen-based enforcement programs for traffic safety.

**Analysis:**

# 1. 摘要翻译

本文提出一种**无需标定的单目车辆测速框架**，不依赖道路特征、相机标定参数或基于道路特征的参照物。方法使用包含36个关键点的车辆模板，并在每帧更新单应矩阵。作者训练了基于YOLO的车辆关键点检测模块，并比较了“仅关键点跟踪”和“经单应变换的稠密光流”两种策略。通过单应矩阵将图像位移投影到米制空间，方法在VS13和BrnoCompSpeed数据集上分别取得15.0%和9.7%的MAE，且77.9%和93.1%的结果误差在±20%以内。去除视频首尾各10%的边缘异常帧后，MAE分别降至11.7%和7.6%。

# 2. 方法动机分析

**驱动力**：传统视觉测速通常需要车道线、道路长度、消失点或相机内外参，部署成本高，不适合手机、行车记录仪等灵活设备。作者希望直接利用车辆自身的已知几何尺寸建立米制尺度。

**现有痛点**：  
1. 基于道路特征的方法依赖稳定、清晰的道路结构；  
2. 相机标定需要人工测量，难以规模化；  
3. 固定单应矩阵会因车辆靠近相机、姿态变化和透视变化产生漂移；  
4. 稀疏关键点容易受检测抖动影响，单个错误点会显著干扰速度。

**核心假设**：车辆可近似为若干局部平面；相邻帧内车辆主要进行纵向平移，横向位移和旋转较小；只要获得同一平面上至少4个非共线关键点及其真实米制坐标，就能建立图像到车辆平面的尺度映射。

# 3. 方法设计详解

## 3.1 Pipeline

输入连续帧 \(I_k,I_{k+1}\)，输出每辆车在时刻 \(k\) 的速度。

1. **车辆检测与跟踪**：MOT模块生成车辆框并维持车辆ID，将每辆车裁剪为ROI，减少背景干扰。  
2. **关键点检测与分类**：YOLO关键点网络检测车辆36个语义点，并判断其属于车顶、左侧或右侧等平面。模板为二维机械图，关键点坐标以米表示，轮距/轴距等尺度提供真实长度。  
3. **逐平面估计单应矩阵**：对某一平面的关键点，将图像坐标 \(p_k=[u,v,1]^T\) 与模板坐标对应，通过至少4个非共线匹配点估计 \(H_k\)。点较多时使用RANSAC剔除错误匹配。  
4. **计算运动对应关系**：  
   - KP策略：直接匹配相邻帧中的语义关键点；  
   - OF策略：计算稠密光流 \(F_k\)，令 \(p_{k+1}=p_k+F_k(p_k)\)。同时由平面关键点构造凸包，只在凸包内部采样光流点，避免把背景或其他平面的点送入同一个单应模型。  
5. **图像位移转米制位移**：利用当前帧的 \(H_k\) 将 \(p_k\) 和其下一帧对应点 \(p_{k+1}\) 映射到车辆平面：
\[
s_k=\sigma(H_kp_k),\quad s_{k+1}=\sigma(H_kp_{k+1})
\]
其中 \(\sigma\) 表示齐次坐标归一化。点速度为
\[
v_{s,k}=\frac{s_{k+1}-s_k}{\Delta t}.
\]
实际车辆速度需对平面内位移取纵向分量或向量范数。  
6. **聚合与更新**：对凸包内多个点的速度求平均或采用中位数、RANSAC等稳健聚合，得到车辆速度；随后用 \(I_{k+1}\) 中新检测的关键点重新估计 \(H_{k+1}\)，形成“先测速、后更新”的递推过程。

## 3.2 关键设计逻辑

作者真正的修正并非单纯使用单应矩阵，而是将其改为**车辆级、平面级、逐帧动态单应映射**。固定单应矩阵会随着车辆运动失配，误差逐帧累积；逐帧更新相当于不断重置当前几何状态。OF策略进一步把少量关键点扩展为平面内部的大量采样点，以空间聚合降低局部检测误差。

误差分析表明：
\[
\|\Delta v\|\leq \Delta t^{-1}\|J_\sigma\|\|\Delta H\|\|v_{OF}\|.
\]
因此误差主要受透视敏感性、单应矩阵估计误差和光流幅度共同影响；车辆过近、过远或快速尺度变化时，误差会被放大。

# 4. 方法对比与创新

与消失点、车道线或预先标定的主流方法相比，本方法把“道路尺度”替换为“车辆自身尺度”，不再需要固定道路参照物。主要创新包括：  
1. 36点车辆几何模板提供米制尺度；  
2. 对车辆不同局部平面分别建模；  
3. 每帧更新单应矩阵，缓解长期漂移；  
4. 用关键点凸包约束稠密光流，实现密集、稳健的速度估计。

适合道路结构不明确、相机位置灵活、需要低成本部署的场景，尤其是侧视和俯视交通视频。但它并非真正“无先验”：仍依赖车辆尺寸模板、相对静止相机以及局部平面和纵向运动假设。

# 5. 实验分析

作者在VS13和BrnoCompSpeed共443段视频、208辆车上验证，覆盖侧视和俯视视角。代表性结论是：OF方法在两数据集上的MAE为15.0%和9.7%，优于或显著稳定于KP方法；去除首尾10%不可靠帧后，MAE进一步达到11.7%和7.6%。

**优势**：无需相机标定；对道路特征依赖小；动态更新可抑制漂移；稠密光流比单纯关键点更抗异常点。  
**局限**：通用轿车模板与真实车辆尺寸不匹配；车辆靠近边界、尺度变化剧烈或光照变化时不稳定；相机运动、车辆旋转和横向运动会破坏平面假设。文中虽称使用RobustAgg，但实际流程主要采用算术平均，稳健性仍有改进空间。

# 6. 实用指南

论文未明确给出完整开源代码。复现需准备：车辆检测/跟踪器、36点关键点标注、米制车辆模板、YOLO关键点训练、光流模型、RANSAC单应估计和首尾异常帧过滤。关键条件是关键点必须正确对应、每个平面至少有4个非共线点，并保证模板尺寸可靠。该思路可迁移到人体、机器人或其他具有稳定几何尺寸的刚性目标，但必须重新设计多平面模板和运动约束。

# 7. 总结

**核心思想：用车辆自身几何尺度动态完成单目测速。**

**速记版Pipeline**：  
车辆跟踪 → 检测车身语义点 → 用已知车身尺寸建立平面尺度 → 在平面区域追踪稠密运动并换算距离 → 每帧更新几何映射后输出速度。

**Key Findings:**

- After applying a 10% trim to remove edge-of-frame outliers, performance improves to MAEs of 11.7% and 7.6%, with within-+/-20% accuracy increasing to 85.3% and 95.4%.
- This work addresses key limitations of existing vision-based approaches and enables low-cost and efficient speed enforcement using portable devices such as dashcams and smartphones, thereby supporting citizen-based enforcement programs for traffic safety.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16785v1)
- [arXiv](https://arxiv.org/abs/2608.16785v1)

---

<a id='2608.16765v1'></a>
## [TRACE-Bench: Decomposing and Diagnosing Multi-Reference Image Generation](https://arxiv.org/abs/2608.16765v1)

**Authors:** Haoran Wang, Chaofan Ma, Ran Yi, Lizhuang Ma

**Published:** 2026-08-17

**Categories:** cs.CV, cs.AI

**Abstract:**

Despite recent advances in unified multimodal models for multi-reference image generation, existing benchmarks remain organized around predefined task types (e.g., "subject composition"), which are ill-suited to this combinatorial setting and lead to fragmented coverage, uncontrolled complexity, and little diagnostic value. Recognizing that diverse multi-reference tasks share a common set of atomic operations, we adopt a capability-oriented perspective and formalize four operators: Anchor ($f$), Disentangle ($g$), Apply ($\oplus$), and Compose ($C$). Any multi-reference prompt can then be represented as a compositional formula over these operators, whose structural complexity is quantified by the number of operator slots. Building on this formulation, we construct TRACE-Bench, comprising approximately 1,600 evaluation cases across slot counts 1--8, built from 631 formula templates and around 4,000 reference images spanning diverse artistic styles and real-world subjects. The formula structure directly drives an operator-aligned evaluation protocol for per-capability scoring and a diagnostic tree analysis for recursive failure localization. Evaluating 9 leading models reveals insights invisible to holistic scoring: the primary bottleneck lies in disentanglement ($g$) and attribute binding ($\oplus$) rather than scene-level composition ($C$), with even the best model scoring only 0.74 on attribute fidelity. Project page: https://amuseum-whr.github.io/TraceBench

**Analysis:**

## 1. 摘要翻译

尽管统一多模态模型在多参考图像生成方面取得进展，现有基准仍按“主体组合”等预定义任务组织，难以覆盖组合空间，复杂度也难以控制，且无法诊断失败原因。TRACE-Bench 从能力角度出发，将多参考生成分解为四种原子操作：锚定（Anchor，\(f\)）、解耦（Disentangle，\(g\)）、应用/绑定（Apply，\(\oplus\)）和组合（Compose，\(C\)）。任意多参考提示均可表示为这些操作的组合公式，并以操作槽数量衡量结构复杂度。作者构建了约1,600个案例、631个公式模板和约4,000张参考图像，覆盖1—8个槽位；同时设计操作对齐的细粒度评测与诊断树，用于定位递归失败来源。对9个模型的实验表明，当前瓶颈主要在属性解耦与绑定，而非场景组合；最佳模型的属性保真度也仅约为0.74。

## 2. 方法动机分析

**驱动力与痛点：**传统基准按任务类型划分，无法覆盖多参考任务的组合爆炸；整体分数不能区分“找错主体、提取错属性、绑定错目标”还是场景布局失败；不同案例的结构复杂度缺乏统一度量。

**核心假设：**多参考生成看似任务多样，实质上共享少数原子能力；若用这些能力构造符号公式，便可同时实现案例生成、难度控制和失败归因。

## 3. 方法设计详解

### 核心表示

- **Anchor \(f(I,e)\)：**从图像 \(I\) 中定位实体 \(e\)，保留其身份特征。
- **Disentangle \(g(I,E,a)\)：**从实体或实体集合 \(E\) 中提取属性 \(a\)，并与原载体身份、形状等无关信息分离。
- **Apply \(\oplus\)：**把属性绑定到文本指定或锚定的目标实体，如“机器人 \(\oplus\) 人的跑步姿势”。
- **Compose \(C\)：**将多个实体组织为场景，并满足空间或交互关系。

公式按“实体—场景—完整提示”三层组织。例如：
\[
F=C(C(f_1,T_e\oplus g_1)\oplus g_{rel},f_2\oplus g_2)\oplus g_{global}.
\]
槽数定义为：
\[
slot(F)=|f|+|g|,
\]
即锚定项与解耦属性项总数；它控制结构复杂度，但不等同于最终难度。

### Benchmark pipeline

1. **图像池构建：**从Danbooru、LAION和真实人物数据集中筛选约5万张候选图，再进行质量检查，保留3,839张，并加入约200张合成图。
2. **结构化标注：**用Gemini-2.5-Pro提取实体、定位短语和属性。属性分为外观、形态、动态、全局四层，并支持服饰/组件转移 \(g_{attach}\) 与整体设计转移 \(g_{ip}\)。
3. **均衡采样：**根据标签特征计数迭代采样。公式 \(w_j=1/(1+c_j)\) 使已充分覆盖的类别权重下降，从而补充长尾属性。
4. **公式采样与提示生成：**在1—8槽位内采样631种模板，将公式与参考图绑定，再由VLM生成自然语言提示；同时保留明确的“来源—目标”映射，并人工过滤不合格案例。
5. **操作对齐评测：**针对每个操作生成多个二元问题。例如Anchor检查实体存在与身份一致性；Apply检查载体完整性、属性独占性和自然融合；Compose检查共存、关系、空间合理性、重复与内容泄漏。VLM接收参考图、生成图和问题，输出通过/失败。
6. **诊断树：**从完整公式开始，依次移除全局属性、展平组合、简化关系、删除附加属性；被移除的参考内容改写为文本描述以保持场景上下文。比较各节点结果，即可判断失败来自孤立能力不足，还是多参考交互干扰。

## 4. 方法对比、创新与适用性

其本质区别不是增加新的生成模型，而是把**基准构造逻辑、评价逻辑和诊断逻辑统一到同一公式空间**。创新主要包括：①四操作能力分解；②以槽位实现结构化难度控制；③公式自动派生细粒度检查项；④诊断树区分“固有失败”和“组合干扰”。适合评测多主体组合、虚拟试衣、群像布局、风格迁移和属性转移；不适合只关心审美质量或纯文本生成的任务。

## 5. 实验分析

作者在约1,600案例上评测9个模型，并用200个案例做人类一致性与诊断树审计。代表性结论是：Nano Banana 2平均表现最好，但解耦和绑定仍明显弱于组合；组合分数最高，说明模型更擅长生成“看起来合理”的场景，而不擅长精确转移属性。另一个结论是，Anchor更受参考图中实体数量影响，而非单纯受槽数影响。局限在于VLM评测存在主观偏差，案例生成依赖VLM标注与人工筛选，公式也主要覆盖有限嵌套结构。

## 6. 实用指南

论文提供项目页面，但文本未明确说明完整数据和评测代码是否公开。复现需实现：图像筛选与结构化标注、公式模板采样、自然语言提示生成、二元检查表和诊断树。关键注意事项是保持来源—目标映射、控制各槽位和全局/关系项比例，并对VLM judge进行人工校准。该框架可迁移到视频：将Anchor扩展为跨帧主体一致性，将Compose扩展为时序关系，并在节点上检查动作、接触和时间顺序。

## 7. 总结

**核心思想：**用原子能力公式化多参考生成并追踪失败。

**速记版Pipeline：**

1. 收集并标注实体、属性及其来源位置。  
2. 用锚定、解耦、绑定、组合四种操作生成公式。  
3. 按槽位采样公式并转成自然语言提示。  
4. 用分能力检查表评估生成结果。  
5. 逐步删减参考条件，定位失败究竟来自单项能力还是多参考干扰。

**Key Findings:**

- Evaluating 9 leading models reveals insights invisible to holistic scoring: the primary bottleneck lies in disentanglement ($g$) and attribute binding ($\oplus$) rather than scene-level composition ($C$), with even the best model scoring only 0.74 on attribute fidelity.
- Project page: https://amuseum-whr.github.io/TraceBench

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16765v1)
- [arXiv](https://arxiv.org/abs/2608.16765v1)

---

<a id='2608.16715v1'></a>
## [MatchingPolicy: Correspondence-Aware Policy Enables Cross-Object In-Context Learning](https://arxiv.org/abs/2608.16715v1)

**Authors:** Qijin She, Hanyang Yu, Zeming Li, Ping Tan

**Published:** 2026-08-17

**Categories:** cs.RO

**Abstract:**

In-context imitation learning enables few-shot policy generalization but struggles to maintain performance on unseen objects and novel scenarios. To address this, we introduce MatchingPolicy, a correspondence-driven framework that explicitly decouples demonstration-to-scene matching from policy learning. Central to our method is a correspondence-aware diffusion policy that conditions robotic actions directly on dense semantic correspondences. This architectural separation resolves the inherent conflict between correspondence identification and action adaptation, enabling robust out-of-distribution transfer. Our framework integrates vision foundation models with a novel two-stage matching algorithm to dynamically establish reliable correspondences. Extensive evaluations on RLBench and real-world manipulation tasks confirm that MatchingPolicy achieves superior few-shot performance, generalizing reliably across unseen object instances and semantic categories.

**Analysis:**

## 1. 摘要翻译

本文提出 **MatchingPolicy**：一种面向对应关系的跨物体上下文学习框架。方法将“示范—当前场景匹配”和“动作策略学习”显式解耦，并通过密集语义对应关系直接条件化扩散策略。结合视觉基础模型与两阶段匹配算法，系统能够在仅有少量示范的情况下，将操作意图迁移到未见过的物体、布局乃至语义类别上。RLBench 与真实机器人实验表明，该方法具有较强的跨实例、跨类别泛化能力。

## 2. 方法动机分析

**驱动力**：上下文模仿学习希望通过1–2条示范直接完成新场景任务，但策略必须同时解决“示范中哪个点对应当前场景哪个点”和“如何据此生成动作”两个问题。作者认为，这种耦合会使模型既难以稳定识别对应关系，也难以学习动作适配。

**现有痛点**：  
1. Transformer/GNN策略通常隐式推断跨场景对应关系，受机器人数据规模和视觉变化限制；  
2. 手工轨迹变换依赖几何启发式和人工设计，难处理遮挡、视角变化及新类别；  
3. 依赖物体几何的策略容易把“任务意图”误认为“物体形状或固定轨迹”。

**核心假设**：机器人主要需要知道“示范中哪些局部区域对应当前场景的哪些区域”，而不必完整重建物体。只要对应关系可靠，策略即可从示范动作中迁移交互意图。

## 3. 方法设计详解

### 3.1 总体流程

输入为1–2条示范轨迹和当前RGB-D观测。当前状态表示为点云 \(P_c\) 与夹爪状态 \(g_c\)，输出为未来 \(K\)步相对夹爪位姿及开合状态。

**步骤一：跨场景语义匹配。**  
先对每个视角的目标物体进行分割，并提取归一化密集特征。对于示范图像中的像素 \(u\)，在当前图像物体区域内寻找最大内积特征：
\[
v^*=\arg\max_v F_a(u)^TF_b(v).
\]
多相机情况下，先用整幅物体的平均特征枚举视角排列，选取整体相似度最高的视角配对，再仅在前两组视角中进行密集匹配。该设计避免简单按相机编号配对导致的大视角错误。

**步骤二：时序跟踪与3D提升。**  
初始匹配点通过CoTracker等点跟踪器传播到整条示范和当前执行序列，每个物体每视角跟踪512个点。只有同时满足“深度存在”和“跟踪可见”的点才保留，并利用相机标定将2D点反投影到世界坐标，再变换到对应夹爪坐标系。

**步骤三：对应感知关键点选择。**  
只保留在大多数示范帧中都能找到对应点的候选点，以过滤遮挡和错误匹配；随后用最远点采样保证空间覆盖，固定选取16个场景关键点，并额外加入6个夹爪关键点。相同编号的点在不同帧中构成同一“对应实体”。

**步骤四：图结构动作建模。**  
图节点包括示范帧、当前帧和未来 \(K\)个动作帧。每帧含场景节点和夹爪节点。图中有三类边：  
- 跨帧对应边：把当前点与示范中的匹配点直接连接；  
- 帧内空间边：建模场景点与夹爪点的相对位置；  
- 动作一致性边：约束未来动作与当前状态及前序动作连续。  

因此，示范信息不是经过全局视觉特征模糊传递，而是沿明确的“对应点—空间关系—夹爪动作”路径传播。

**步骤五：局部几何补偿与扩散解码。**  
未被选中的点被分配到最近关键点，形成局部点组，由预训练PointNet编码，以弥补稀疏对应点的几何信息不足。扩散模型对未来夹爪关键点位置加噪并预测去噪目标，输出平移、旋转诱导位移和夹爪状态残差。再用SVD从关键点位移恢复刚体变换，最后通过逆运动学转成关节命令。真实执行采用“重叠动作块”：只执行预测块前半段，并用剩余动作初始化下一次预测，提升平滑性和闭环修正能力。

### 3.2 训练数据与模型协同

作者构造MatchingDataset：约1.5万任务集合、15万以上轨迹和5万以上物体实例。每个任务由参考示范通过物体中心空间重定向生成不同布局，同时保持任务意图和交互区域。每个物体预采样2048个 canonical表面点，通过共享局部坐标获得跨轨迹真值对应关系与可见性标签。训练阶段使用真值对应关系，部署阶段替换为视觉基础模型和跟踪器。

## 4. 对比、创新与适用场景

**本质区别**：传统上下文策略让网络隐式学习“匹配+动作”；MatchingPolicy把匹配作为独立、可替换的前端，将策略输入从原始物体外观改为对应索引的局部点关系。

**主要创新**：  
1. 对应关系驱动的图扩散策略；  
2. “视角选择—密集语义匹配—时序跟踪—3D提升”的两阶段匹配；  
3. 基于对应可靠性和FPS的动态关键点选择；  
4. 大规模带3D对应标注的合成上下文数据集。

适合短时程、半静态、视觉变化大且需要跨物体迁移的抓取、开合、倾倒、插入等任务。

## 5. 实验分析

作者在36个RLBench任务和4个真实任务上比较InstantPolicy、KAT及RDT-1B，并进行匹配方式与动作规划消融。代表性结果是：真实任务中MatchingPolicy在Layout-Easy/Hard/Shape上的平均成功率分别为77.5%、65.0%、72.5%，显著优于InstantPolicy；两阶段匹配配合重叠动作块优于朴素匹配和整段执行。主要优势是跨布局、跨实例甚至跨类别泛化；局限是短时程、执行速度较慢，并依赖多相机视角配置、分割和跟踪质量。

## 6. 实用指南

论文给出可视化网站，但文本未明确提供代码、模型或数据集下载链接，开源情况需进一步核实。复现关键是：构造共享物体坐标的合成轨迹、保存可见性对应标签、接入DINO类VFM与点跟踪器、实现16个场景点+6个夹爪点的图网络。重要设置包括1–2条示范、512跟踪点/物体/视角、AdamW学习率 \(10^{-5}\)、四张A6000训练约7天。迁移到其他任务时，应重新定义任务相关物体和成功标准，并替换相机标定、分割及机器人运动学模块。

## 7. 总结

**核心思想：用显式对应关系迁移操作意图。**

**速记版Pipeline：**  
1. 找出示范物体与当前物体的语义对应点；  
2. 跟踪这些点并恢复三维位置；  
3. 选出稳定、分布均匀的关键点；  
4. 用图扩散模型结合示范关系预测夹爪运动；  
5. 滚动执行并持续重新规划。

**Key Findings:**

- In-context imitation learning enables few-shot policy generalization but struggles to maintain performance on unseen objects and novel scenarios.
- To address this, we introduce MatchingPolicy, a correspondence-driven framework that explicitly decouples demonstration-to-scene matching from policy learning.
- Central to our method is a correspondence-aware diffusion policy that conditions robotic actions directly on dense semantic correspondences.
- Our framework integrates vision foundation models with a novel two-stage matching algorithm to dynamically establish reliable correspondences.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16715v1)
- [arXiv](https://arxiv.org/abs/2608.16715v1)

---

<a id='2608.16640v1'></a>
## [DPNet: Efficient Dead-End Prediction and Avoidance for Vision-Based UAV Navigation](https://arxiv.org/abs/2608.16640v1)

**Authors:** Ruibin Zhang, Lun Pan, Zelong Xia, Jialiang Hou, Fei Gao

**Published:** 2026-08-17

**Categories:** cs.RO

**Abstract:**

Vision-based Unmanned Aerial Vehicles (UAVs) often suffer from navigation failures in dead ends due to limited sensing accuracy and range. To address this challenge, this paper proposes a systematic solution for efficient dead-end prediction and avoidance. The proposed method introduces a lightweight neural network to predict the relative distance and bearing of potential dead ends within the current field of view using RGB-D inputs. These predictions prune a predefined, compact trajectory library, enabling the planner to proactively avoid dead ends while maintaining navigational smoothness. Notably, our approach transfers across real-world scenarios without manual annotation or fine-tuning on real-world data. The system achieves high-frequency replanning at 50 Hz onboard. Extensive simulation benchmarks demonstrate superior performance in success rate, flight time, and trajectory length, and real-world experiments further validate its effectiveness in complex scenarios.

**Analysis:**

# 1. 摘要翻译

基于视觉的无人机常因感知精度和范围有限，在死胡同中导航失败。为此，本文提出一种高效的死胡同预测与规避系统 DPNet。该轻量网络以 RGB-D 图像为输入，预测当前视场内潜在死胡同的相对距离和方位，并据此裁剪预定义的紧凑轨迹库，使规划器能够主动避开死胡同，同时保持导航平滑性。该方法无需真实场景人工标注或微调，即可迁移到真实环境。系统在机载边缘计算平台上实现了 50 Hz 高频重规划。仿真和真实实验表明，DPNet 在成功率、飞行时间和轨迹长度方面均具有优势。

# 2. 方法动机分析

**驱动力：**传统局部规划器只依据当前视场内的障碍物做贪心决策，可能在尚未看见封闭端之前进入 U 形障碍、V 形结构或盲巷；一旦深入，局部可行轨迹消失，只能碰撞、振荡或回退。

**现有痛点：**全局地图和拓扑规划计算、维护代价高且易受里程计漂移影响；旋转、倒退等恢复策略属于事后补救，会破坏连续飞行；占据预测、场景重建等学习方法输出过于密集，推理延迟高，也未针对“死胡同”这一特殊风险设计。

**核心假设：**规避死胡同不需要重建完整未知环境，只需提前判断“是否存在死胡同、它在图像中的位置及距离”，再将其转化为规划器可理解的障碍约束。

# 3. 方法设计详解

## 3.1 整体流程

1. **输入感知：**前向 RGB-D 相机获得 RGB 图像 \(I\) 和深度图 \(D\)，输入统一缩放为 \(224\times224\)。
2. **多模态编码：**RGB 与深度分别经过视觉编码器。RGB 分支提取语义和外观线索，深度分支提取几何结构；两者特征在空间上对齐后进行通道拼接。
3. **特征融合：**采用轻量卷积块：先用 \(1\times1\) 卷积压缩通道，再经 BN、ReLU 和 \(3\times3\) 卷积提取局部上下文。
4. **双头预测：**Mask head 输出死胡同置信度图，Depth head 输出对应像素的死胡同深度图。经过
   \[
   \hat O=\hat M\odot\hat D
   \]
   仅保留被判定为死胡同的深度，避免无关区域给规划器增加约束。输出分辨率为 \(14\times14\)。
5. **自动标注训练：**仿真中生成 U/V 形死胡同及普通障碍。作者依据死胡同内侧顶点、外侧顶点的可见性自动生成标签：若内侧顶点全部可见，或至少一个内侧顶点可见且外侧顶点均未被遮挡，则认为结构具有足够可辨识性；否则 mask 置零。完整死胡同点云用于生成未截断的真实深度标签，迫使网络推断视场外结构。
6. **虚拟障碍生成：**对预测结果做离群点过滤，并把所有有效死胡同像素的深度统一设为最小预测深度 \(d_{\min}\)，通过相机内参反投影为
   \[
   p_i=d_{\min}K^{-1}[u_i,v_i,1]^T .
   \]
   这相当于在死胡同入口处生成一面平行于图像面的“虚拟墙”，而不是重建内部复杂几何；随后用 MLS 上采样 10 倍，形成致密障碍边界。
7. **轨迹筛选与执行：**将真实深度点云和虚拟障碍融合，并下采样到固定点数。对离线生成的时间最优运动原语逐条碰撞检测，删除进入死胡同的轨迹，再依据目标代价、边界代价和航向平滑代价选择轨迹。
8. **防止重新进入：**当死胡同暂时离开视场、虚拟障碍消失时，仅按目标距离会导致无人机突然转回陷阱。因此加入阈值化航向变化惩罚：只有当前后轨迹方向差超过阈值时才惩罚，兼顾必要绕行与连续性。

## 3.2 训练目标与实现逻辑

深度仅在真实死胡同区域计算 L1 损失，mask 使用 BCE 损失。RGB 编码器冻结以增强 sim-to-real 泛化；深度编码器微调以适配深度几何。数据量为 5 万对 RGB-D 样本，训练/验证/测试按 7:1:2 划分，Adam、学习率 \(5\times10^{-5}\)、batch size 128、训练 100 个 epoch。Jetson Orin NX 上 TensorRT FP16 推理约 9.2 ms，完整系统达到 50 Hz。

# 4. 方法对比与创新

**本质区别：**主流方法通常恢复地图、预测占据概率或在陷入后恢复；DPNet只预测对决策最有用的“死胡同入口约束”，并直接作用于轨迹裁剪。

**主要创新：**  
1. 将死胡同定义为需要提前规避的结构性风险，而非普通障碍；  
2. RGB 语义与深度几何联合预测视场外/遮挡结构；  
3. 用自动几何可见性规则生成训练标签，避免人工标注；  
4. 通过虚拟入口屏障实现稀疏、低成本的规划约束；  
5. 用航向平滑项解决预测障碍离开视场后的“重新进入”问题。

适用于前向视觉受限、局部规划、高机动且不能依赖全局地图的 UAV；对静态或结构规则的 U/V 型陷阱最有效。

# 5. 实验分析

作者在含随机障碍和死胡同的仿真环境中，与 EGO-Planner、Primitive-Planner 和 Faster 对比，并在真实室内场景中测试。代表性结论是：DPNet 在简单和困难设置下成功率分别达到 100% 和 99%，且相比 Faster 明显减少飞行时间；真实实验中能够绕开真实死胡同，同时通过可通行的开放结构，说明其不只是“保守拒绝所有凹形障碍”。

**优势：**轻量、无需真实标注、零样本迁移、对原规划器改动小、几乎无额外碰撞检测开销。  
**局限：**训练形状主要是 U/V 型，复杂迷宫、多死胡同和动态障碍尚未验证；14×14 输出较粗，预测错误可能产生过度保守或漏检；真实实验依赖动捕定位，尚未充分检验定位误差影响。

# 6. 实用指南与迁移可能

论文声明源码公开。复现需搭建仿真障碍生成器、实现可见性自动标注、训练双分支网络，并将预测 mask 反投影为虚拟屏障后接入运动原语规划器。重点注意深度无效值填补、相机内参一致性、传感器最大量程、离群点过滤、\(d_{\min}\)、MLS 上采样倍数及平滑阈值。该思想可迁移到地面机器人、自动驾驶或机械臂：将“死胡同”替换为相应任务中的局部结构风险，并把预测结果转化为代价区或不可行区域，而不必预测完整场景。

# 7. 总结

**核心思想：**预测陷阱入口，提前封堵轨迹。

**速记版 Pipeline：**  
1. 用彩色图和深度图判断前方是否存在死胡同；  
2. 估计其图像位置和距离，即使部分结构不可见；  
3. 把预测区域投影成入口处的虚拟墙；  
4. 删除会撞上虚拟墙的候选飞行路线；  
5. 用方向平滑避免绕开后又转回陷阱。

**Key Findings:**

- Notably, our approach transfers across real-world scenarios without manual annotation or fine-tuning on real-world data.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16640v1)
- [arXiv](https://arxiv.org/abs/2608.16640v1)

---

<a id='2608.16632v1'></a>
## [DRAFE: Domain-Robust Asymmetric Fusion of Heterogeneous Detection Transformers for Cross-City Fine-Grained Traffic Object Detection](https://arxiv.org/abs/2608.16632v1)

**Authors:** Divine Yao Agbobli, Geoffery Eyram Agorku, Israel Afriyie, Kwadwo Amankwah-Nkyi, Marvin Osei-Kuffour, Richmond Owusu Duah, Bright Seglah, Kelvin Asamoah Terkper, Kwabena Amoako Adjei

**Published:** 2026-08-17

**Categories:** cs.CV

**Abstract:**

Deep learning-based object detectors are fundamental to intelligent transportation systems, enabling traffic monitoring, vehicle analytics, and infrastructure management. However, achieving both fine-grained vehicle recognition and robust cross-city domain generalization remains challenging. We present the Domain-Robust Asymmetric Fusion Ensemble (DRAFE), which combines independently trained LW-DETR and RF-DETR detectors for cross-city fine-grained traffic object detection. DRAFE employs a two-stage training strategy that first pretrains complementary detectors on diverse public traffic datasets using pseudo-label expansion and human-in-the-loop annotation refinement, producing a curated corpus of 6,049 images and 203,619 annotations, before challenge-compliant fine-tuning on the Project Hafnia Track 6 dataset. At inference, DRAFE applies anchor-conditioned class-consistent matching, reliability-weighted coordinate fusion, agreement-aware confidence recalibration, and complementary hypothesis recovery. On AI City Challenge 2026 Track 6, DRAFE achieves 0.4022 mAP, ranks sixth among 25 participating teams, and improves by 0.0553 mAP over a preliminary ensemble evaluated under identical benchmark conditions.

**Analysis:**

## 1. 摘要翻译
基于深度学习的目标检测器是智能交通系统的重要组成部分，但细粒度车辆识别与跨城市域泛化仍然困难。本文提出领域鲁棒非对称融合集成方法（DRAFE），将独立训练的 LW-DETR 与 RF-DETR 用于跨城市细粒度交通目标检测。方法首先利用伪标签扩展和人工复核，在多个公共交通数据集上构建包含6049张图像、203619个标注的训练语料；随后在 Project Hafnia Track 6 上进行符合竞赛约束的微调。推理阶段采用基于锚点、类别一致的匹配，可靠性加权坐标融合，一致性感知置信度重校准，以及互补候选恢复。在 AI City Challenge 2026 Track 6 中，DRAFE 获得0.4022 mAP，在25支队伍中排名第六，较初始集成提升0.0553 mAP。

## 2. 方法动机分析
**驱动力：** Hafnia 不允许下载测试图像，且目标城市不可参与训练，因此问题本质上是单源域泛化；同时十类车辆外观相似、类别长尾明显。  
**现有痛点：** 普通 WBF 仅按空间重叠聚类，未显式保证类别一致，也没有“主模型—辅助模型”关系；简单平均可能损害高召回模型的候选空间，未匹配的正确目标还会被丢弃。  
**核心假设：** 选取高召回模型作为“锚点”保留主要假设空间，再利用不同模型的定位共识修正框，并保留辅助模型独有目标，可同时提升定位精度与跨域召回。

## 3. 方法设计详解
### （1）数据与训练
从五个公共交通数据集抽取图像，先人工完整标注1683张种子图像，统一为 Track 6 的十类标签。用种子集训练元模型，仅用于向6049张图像生成初始框；随后所有图像都在 CVAT 中人工检查和修正，因此伪标签不会未经审核直接进入训练。最终按4000/1049/1000划分训练、验证和测试集，并分别预训练三个检测器，再在 Hafnia 中独立微调。

### （2）非对称角色
- **LW-XA（LW-DETR XLarge）**：锚点，提供密集、高召回候选；
- **LW-XB**：同架构辅助模型，提供定位共识；
- **RF-B（RF-DETR Base）**：异构辅助模型，提供不同的误差与召回特征。

### （3）逐锚点融合
将 LW-XA 的检测按置信度降序处理。对每个锚点，只在辅助模型中寻找**同类别且 IoU≥0.55** 的最高 IoU 框，并采用一对一匹配；匹配后从候选池移除，避免一个辅助框支持多个锚点。

对匹配组中的每个框计算  
\(w_m=\alpha_m s_m\)，  
其中可靠性系数为 \((0.42,0.32,0.26)\)，再用置信度与模型可靠性共同加权平均四角坐标。这样不是简单平均，而是让更可靠、置信度更高的模型拥有更大定位影响力。

融合置信度取组内最大置信度，并按参与模型数增加小幅奖励：两模型一致增加3%，三模型一致增加6%。该分数主要用于排序，并不表示严格校准后的概率。

### （4）互补候选恢复与输出
所有未被匹配的 LW-XB/RF-B 检测仍被保留，但置信度乘以0.97，使锚点支持的结果略占优。融合结果与恢复结果合并后，不设前置置信度阈值、不再做 NMS，仅保留每张图排名最高的300个候选，从而避免过早抑制跨域场景中的不确定真目标。

## 4. 方法对比与创新
其本质区别不是提出新检测器，而是改变集成逻辑：从对称聚类变为锚点驱动；从无条件空间融合变为类别一致的一对一匹配；从只保留匹配结果变为恢复辅助模型独有假设。创新主要体现在推理策略和数据构建，而非网络结构本身。它适合隐私受限、目标长尾、跨域变化大且允许离线多模型推理的检测任务。

## 5. 实验分析
作者通过独立模型、初始集成、完整测试集及开发集消融进行验证。DRAFE 达到0.4022 mAP，较最强单模型提升0.0181；AP75提升更明显，说明融合主要改善了定位质量。  
**优势：** 不依赖目标域数据；兼顾定位共识与互补召回；人工复核保证扩展语料质量。  
**局限：** 三个模型推理成本高；固定权重和阈值缺乏跨域自适应；消融多基于单次实验，尚未证明各模块的独立贡献；小目标性能仍较弱。

## 6. 实用指南
论文声称代码发布于 VisionOps Trainer，但挑战数据无法公开下载，完整复现受 Hafnia访问限制。复现时需重点保持：统一十类标注规范、人工复核扩展数据、三个模型独立训练、IoU阈值0.55、权重0.42/0.32/0.26、奖励系数0.03、未匹配框缩放0.97及每图300框预算。该框架可迁移到行人、工业缺陷等任务：替换检测器和类别体系，重新估计模型可靠性，并在验证域上调节匹配阈值、权重及候选预算。

## 7. 总结
**核心思想：** 锚点保召回，协同提定位。  

**速记版 Pipeline：**
1. 汇集多源交通图像并统一标注，人工审核自动生成的初始框。  
2. 独立训练两个同类模型和一个异构模型。  
3. 让高召回模型逐个提出目标，其他模型只匹配同类别的对应目标。  
4. 按模型可靠性和置信度加权修正位置，同时保留辅助模型发现的新目标。  
5. 汇总排序后保留每图300个结果，避免过早删掉跨城市真目标。

**Key Findings:**

- We present the Domain-Robust Asymmetric Fusion Ensemble (DRAFE), which combines independently trained LW-DETR and RF-DETR detectors for cross-city fine-grained traffic object detection.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16632v1)
- [arXiv](https://arxiv.org/abs/2608.16632v1)

---

<a id='2608.16589v1'></a>
## [Ultra: Unsupervised Cross-Task Optimization for Reliable Restoration Segmentation Collaboration under Adverse Weather](https://arxiv.org/abs/2608.16589v1)

**Authors:** Shiqin Wang, Zhiqian Li, Haoyuan Du, Junming Chen, Jiayuan Li, Tianrun Xu, Haoyang Chen

**Published:** 2026-08-17

**Categories:** cs.CV

**Abstract:**

Unsupervised Domain Adaptation for Adverse Weather Semantic Segmentation (UDA-ASS) aims to transfer semantic knowledge from labeled normal-weather images to unlabeled adverse environments. Existing approaches implicitly assume that restoration and segmentation provide mutually beneficial guidance. However, under severe degradation and without target-domain supervision, the validity of cross-task optimization directions becomes fundamentally unidentifiable, leading to hallucination-driven error propagation. In this work, we propose a novel Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra), which reframes cross-task interaction as direction selection under uncertainty and causal effect estimation, enabling reliable collaboration through candidate direction generation and intervention-based filtering. In detail, we propose CTDN and CMIL. The former exploits complementary visual structures and semantic information to generate candidate optimization directions and performs cooperative direction selection between restoration and segmentation. The latter reformulates cross-task information transfer from correlation-based propagation into causal effect assessment, suppressing hallucination propagation. Extensive experiments on three widely used UDA-ASS benchmarks demonstrate state-of-the-art segmentation performance. Beyond segmentation, our framework achieves better unsupervised restoration results than existing UDA-ASS restoration methods and generalizes to unsupervised restoration and object detection collaboration tasks. Code and models will be available at https://github.com/Wang-Shiqin/Ultra.

**Analysis:**

## 1. 摘要翻译

恶劣天气语义分割的无监督域适应（UDA-ASS）旨在将有标签正常天气图像中的语义知识迁移到无标签恶劣环境。现有方法通常默认图像修复与语义分割能够相互促进，但在严重退化且缺乏目标域监督时，跨任务优化方向实际上无法可靠判定，容易产生由幻觉驱动的错误传播。为此，本文提出 Ultra，一种无监督修复—分割协同学习框架，将跨任务交互重新定义为不确定性下的方向选择与因果效应估计。具体而言，CTDN利用互补的视觉结构和语义信息生成候选优化方向，并在修复与分割之间选择协同方向；CMIL则将跨任务信息传递从相关性传播改写为因果效应评估，以抑制幻觉传播。在三个UDA-ASS基准上的实验表明，Ultra取得了先进的分割性能，同时获得优于现有方法的无监督修复效果，并可推广至修复—目标检测协同任务。

## 2. 方法动机

**驱动力：** 恶劣天气造成信息缺失，修复模型需要语义结构补全图像，分割模型也需要修复后的清晰证据，二者具有互补性。  
**现有痛点：** 一是没有目标域清晰图像和像素标签，无法判断当前修复或分割更新是否正确；二是错误伪语义和修复幻觉会循环注入另一任务，形成自强化错误。  
**核心假设：** 修复与分割不应盲目互相指导，而应先生成候选方向，再判断其对对方任务是否真正有益，只传播具有正向效果的信息。

## 3. 方法设计详解

### 整体流程

输入源域有标签图像与目标域无标签恶劣天气图像，经混合采样后同时进入分割分支和扩散修复分支。

1. **分割指导修复（SGRA）：**  
   分割特征 \(S_i\) 经1×1投影、普通深度卷积和空洞深度卷积提取语义先验 \(P_i^S\)，再生成缩放参数 \(\gamma_i\) 与偏移参数 \(\beta_i\)，对修复特征 \(R_i\)进行条件调制：
   \[
   R_i^+=R_i+\eta_i^{S\to R}[\text{Mod}(R_i,P_i^S)-R_i].
   \]
   因而 \(\Delta_i^{S\to R}=R_i^+-R_i\) 是“语义可能如何改善修复”的候选方向。残差系数零初始化，使训练初期不会破坏原模型。

2. **修复指导分割（RGSA）：**  
   将修复特征调整到分割分辨率，分解为低频 \(L_i\) 和高频 \(H_i\)，通过边界门控抑制颜色、亮度等外观噪声，形成修复先验 \(P_i^R\)。随后调制分割特征得到 \(S_i^+\)，候选方向为：
   \[
   \Delta_i^{R\to S}=S_i^+-S_i.
   \]
   该设计重点保留结构和边界，而非直接传递低层纹理。

3. **无监督纳什协商（UNB）：**  
   分别计算分割梯度 \(g_s\) 和修复梯度 \(g_r\)，用任务损失的指数移动平均倒数作为可靠性权重，并归一化。根据两梯度的Gram内积求解混合系数 \(\alpha^*\)，将两者组合为共享参数的最终更新方向。直观上，任务一致时强化共识，任务冲突时选择对双方伤害最小的方向。

4. **因果互介入学习（CMIL）：**  
   对每个跨任务候选方向建立“处理状态”和“控制状态”，分别模拟一次能量下降。  
   - 对 \(S\to R\)，比较加入语义介入与不加入介入后的修复能量下降速度；  
   - 对 \(R\to S\)，比较加入结构细化前后的无监督分割能量，分割能量由预测熵和可信伪标签交叉熵构成。  
   若介入使下一步能量下降更快，则生成较大的门控值 \(g\)；否则压低介入幅度，并利用安全损失使其趋近零。最终仅将有正向因果效果的跨任务信息送入后续网络。

## 4. 方法对比与创新

Ultra区别于“先修复、再分割”的串行范式，也区别于仅靠一致性约束或特征拼接的协同方法。其本质创新是：把跨任务交互从**无条件的信息传递**变成**候选方向协商+因果筛选**。CTDN解决“方向不确定”，CMIL解决“错误循环传播”；SGRA和RGSA则分别实现语义到修复、结构到分割的可靠接口。该框架适合目标域无标签、退化严重且两个任务互补的场景。

## 5. 实验分析

作者在 Cityscapes→ACDC、Cityscapes→Dark Zurich 及夜间驾驶数据上验证分割，在ACDC上验证修复，并迁移到BDD100K目标检测。代表性结果是：HRDA骨干在ACDC测试集和验证集均达到73.0% mIoU；在Dark Zurich和Nighttime Driving上达到52.8%和59.3% mIoU。消融实验显示，CTDN、UNB、CMIL逐步带来性能提升。  
**优势：** 任务协同可靠、能抑制幻觉、兼顾分割与修复、可迁移。  
**局限：** 因果效应只是基于一步模拟能量下降的近似；需要额外的双分支、介入和梯度计算，训练成本较高；无监督分割能量和可信伪标签仍可能产生偏差。

## 6. 实用指南

代码和模型计划发布于：`github.com/Wang-Shiqin/Ultra`。复现需实现SGRA、RGSA、UNB和CMIL，并处理共享参数梯度覆盖。论文使用PyTorch、单张80GB A100、60k迭代、1024×1024裁剪、AdamW、权重衰减 \(10^{-4}\)，并采用稀有类别采样。迁移到检测任务时，将分割能量替换为检测能量，将分割特征替换为分类/框回归特征，即可构造修复—检测协同。

## 7. 总结

**核心思想：** 只传播被验证有益的跨任务指导。

**速记版Pipeline：**

1. 恶劣图像同时进入修复和分割网络。  
2. 用语义生成修复候选，用修复结构生成分割候选。  
3. 根据两任务梯度协商共享更新方向。  
4. 模拟介入后的下一步效果，过滤有害信息。  
5. 用筛选后的特征联合优化两个任务。

**Key Findings:**

- In this work, we propose a novel Unsupervised Restoration-Segmentation Collaborative Learning Framework (Ultra), which reframes cross-task interaction as direction selection under uncertainty and causal effect estimation, enabling reliable collaboration through candidate direction generation and intervention-based filtering.
- In detail, we propose CTDN and CMIL.
- Extensive experiments on three widely used UDA-ASS benchmarks demonstrate state-of-the-art segmentation performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16589v1)
- [arXiv](https://arxiv.org/abs/2608.16589v1)

---

<a id='2608.16585v1'></a>
## [SQuad: Sub-Quadratic Attention Distillation for Efficient Video Generation](https://arxiv.org/abs/2608.16585v1)

**Authors:** Animesh Karnewar, Denis Korzhenkov, Amirhossein Habibian, Mohsen Ghafoorian

**Published:** 2026-08-17

**Categories:** cs.CV

**Abstract:**

Video Diffusion Transformers (DiTs) spend most of their compute inside the Self-Attention operation, whose cost grows quadratically, $\mathcal{O}(n^2)$, with the number of latent tokens $n$. For the task of video generation, the token count is large, so this term dominates runtime and memory, and thereby caps the resolution and duration we can generate. Linear $\mathcal{O}(n)$ and low-rank $\mathcal{O}(nk)$ surrogates of Self-Attention trade the full softmax $QK^T$ for cheaper kernels, but rarely recover the original's expressivity, leaving a stubborn quality gap. Motivated by this, we propose SQuad, a Sub-Quadratic Attention Distillation framework that achieves a complexity of $\mathcal{O}(n\sqrt{n})$ in the resulting distilled Attention, naturally balancing the efficiency v/s expressivity trade-off. Instead of training our own Video DiT from scratch, which is prohibitively expensive, we fit a pretrained full softmax Self-Attention DiT into our proposed SQuad-Attention one by distilling the former in two stages: Flow-Matching Supervised Fine-Tuning (SFT), followed by improved Distribution Matching Distillation (DMD2) which additionally makes the sampling more efficient. On the Wan~2.2 5B text-to-video model, SQuAD matches the quadratic teacher on VBench ($83.20$ v/s $83.08$) while cutting the per-step per-block attention FLOPs by $\sim$$67\times$ and attention latency by $\sim$$11\times$, and end-to-end DiT latency by 2$\times$, all while also generating a video in only $6$ Neural Functional Evaluations (NFEs) instead of the default $100$.

**Analysis:**

## 1. 摘要翻译

视频扩散 Transformer（DiT）的主要计算集中在自注意力，其时间和内存复杂度随视频潜变量 token 数 \(n\) 二次增长。线性注意力和低秩近似虽可降为线性或低秩复杂度，却通常牺牲 Softmax 注意力的表达能力。本文提出 SQuad（Sub-Quadratic Attention Distillation），将注意力复杂度降至 \(O(n\sqrt n)\)，在保留真正 Softmax 的同时平衡效率与表达能力。SQuad 通过两阶段蒸馏将预训练的全注意力 DiT 转换为 SQuad 注意力模型：先进行 Flow-Matching 监督微调（SFT），再进行改进的分布匹配蒸馏（DMD2）以减少采样步数。在 Wan 2.2 5B 上，SQuad 的 VBench 得分为 83.20，接近教师模型的 83.08；每个注意力模块的 FLOPs 降低约 67 倍、注意力延迟降低约 11 倍，端到端 DiT 延迟降低约 2 倍，并将采样 NFEs 从 100 降至 6。

## 2. 方法动机

**驱动力：** 视频 token 数量极大，标准全连接 Softmax 注意力的 \(O(n^2)\) 成为分辨率、时长和生成速度的瓶颈。  
**现有痛点：** 线性/低秩方法替换了 Softmax，难以保持其输入依赖的尖锐选择能力；稀疏注意力依赖动态 token 选择、专用 kernel 或复杂异构结构，部署成本较高。  
**核心假设：** 视频注意力虽形式上是稠密的，但注意力质量集中于少数关键 token；因此可以用结构化的两跳通信近似全局交互，同时仍保留 Softmax 的非线性表达能力。

## 3. 方法设计详解

### 3.1 SQuad 注意力

输入 token 特征 \(X\in\mathbb R^{n\times hd}\) 先经过线性投影得到 \(Q,K,V\)，再对 \(Q,K\) 使用 RoPE。与标准注意力直接计算 \(n\times n\) 矩阵不同，SQuad 将 token 网格划分为大小约为 \(w=\sqrt n\) 的窗口。

1. **局部视图：** 每个窗口内部进行 Softmax 注意力。共有 \(n/w\) 个窗口，每个序列长度为 \(w\)，成本为  
\[
C_L=O(nw).
\]
该步骤让窗口内 token 充分混合。

2. **全局视图：** 将所有窗口中“相同位置”的 token 重新排列到一起，在窗口之间进行 Softmax 注意力。序列长度为 \(n/w\)，共有 \(w\) 组，成本为  
\[
C_G=O(n^2/w).
\]

3. **顺序组合：** 论文最终采用“局部→全局”：  
\[
Y=\mathrm{Attn}_G(Q,K,\mathrm{Attn}_L(Q,K,V)).
\]
局部输出作为全局注意力的 value，而非与两次结果相加。任意源 token 可先在其窗口内移动到目标位置对应的 slot，再通过全局 pass 跨窗口传播，因此单层即可获得完整感受野。

总成本为  
\[
O(nw+n^2/w),
\]
在 \(w=\sqrt n\) 时达到 \(O(n\sqrt n)\)。这不是线性核近似，也不是固定局部稀疏，而是两次普通 Softmax 注意力的结构化重排。

### 3.2 两阶段蒸馏

替换注意力后，原模型参数不再适配新算子，因此：

- **阶段一 SFT：** 从原 DiT 权重初始化 SQuad 模型，使用原始 Flow-Matching 目标训练，令模型重新拟合噪声到干净 latent 的速度场，训练约 8k iterations。
- **阶段二 DMD2：** 以 SFT 模型为学生，原始 Wan 为真实分布教师，同时训练一个跟踪学生分布的 critic。通过教师 score 与学生 score 的差异更新生成器，将 100 步模型蒸馏为 6 步模型，并将 CFG 能力一并吸收，因此 NFE 等于实际前向次数。

DiT 中的 Cross-Attention、FFN、时间调制和 VAE 均保持不变，仅替换 Self-Attention。

## 4. 对比与创新

本质区别在于：SQuad **不替换 Softmax，也不依赖动态稀疏选择**，而是通过局部和跨窗口两次重排注意力实现全局通信。主要贡献包括：提出 \(O(n\sqrt n)\) 的可组合注意力结构；证明单层完整感受野；将结构蒸馏与步数蒸馏统一；且无需新增参数、定制 CUDA kernel 或专用硬件优化。

其最佳场景是高分辨率、长时长视频生成，尤其适合 token 数不断增大的 DiT。对图像、语言等长序列任务理论上也可迁移，但窗口几何和位置编码需重新设计。

## 5. 实验分析

作者在 Wan 2.2 5B 和 Wan 2.1 1.3B 上进行 VBench、用户偏好、FLOPs 和 GPU 延迟评估。代表性结论是：Wan 2.2 5B 上 VBench 基本保持教师水平，同时注意力 FLOPs 约降低 67 倍、端到端编译延迟从 667ms 降至 314ms。  
**优势：** 结构简单、零额外参数、保留 Softmax、可直接使用 PyTorch 编译。  
**局限：** 需要蒸馏训练；SFT 与 DMD2 的效果耦合；固定窗口未必适合所有数据分布；尚未证明从头预训练或跨模态迁移的效果。

## 6. 实用指南

论文文本未明确给出代码开源信息，不能假定已有官方实现。复现时需实现三项核心操作：按 \(T\times H\times W\) 重排 local/global 视图、对 padding token 做 mask、按 local→global 串联两次 Softmax 注意力。关键设置包括窗口 token 数接近 \(\sqrt n\)，论文中 Wan 2.2 使用约 \(21\times2\times4\) 的时空窗口；采用 bfloat16、Flow-Matching SFT 约 8k 步、DMD2 约 15k–30k 步、最终 6 NFE。迁移到其他 DiT 时，应保持 QKV/head 维度不变，并重新搜索窗口形状与蒸馏强度。

## 7. 总结

**核心思想：** 两次结构化Softmax替代全局二次注意力。

**速记版 pipeline：**
1. 将视频 latent 展平成 token，并生成 Q/K/V。  
2. 把 token 分组，在每个小窗口内做 Softmax。  
3. 按窗口内相同位置重新排列，跨窗口做 Softmax。  
4. 用 Flow-Matching 微调适配新注意力。  
5. 用 DMD2 将模型压缩到 6 步生成。

**Key Findings:**

- Motivated by this, we propose SQuad, a Sub-Quadratic Attention Distillation framework that achieves a complexity of $\mathcal{O}(n\sqrt{n})$ in the resulting distilled Attention, naturally balancing the efficiency v/s expressivity trade-off.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.16585v1)
- [arXiv](https://arxiv.org/abs/2608.16585v1)

---

