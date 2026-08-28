time: 20260828

# Arxiv Computer Vision Papers - 2026-08-28

## Executive Summary

# Arxiv 计算机视觉日报执行摘要（2026-08-27）

> **说明：**以下判断主要依据论文标题、作者信息及研究主题归纳；在未提供论文摘要、实验结果和代码的情况下，具体性能结论应以原文为准。

## 一、总体趋势

本期论文集中体现出计算机视觉与具身智能领域的四条主线：

1. **从视觉表征走向可交互的世界模型**  
   多篇工作尝试从视频、单张图像或跨具身数据中学习物理状态、动作后果和三维结构。代表论文包括 **CLAP、SpatialCrafter、Riemann-1.0**，显示研究重点正从“识别和生成内容”转向“预测环境并支持行动”。

2. **VLM/VLA 推理效率成为核心问题**  
   **FlashVLA、PACE、ClusterAttention** 分别从流式动作解码、视觉信息压缩和注意力计算简化入手，针对视觉语言模型及视觉-语言-动作模型的延迟、吞吐和部署成本进行优化。

3. **视频与动态场景的预训练、重建和鲁棒感知**  
   **LeVJEPA** 关注更高效、可扩展的视频预训练；**SSMB** 针对运动模糊下的局部特征检测；**Reconstructing Humans and Objects in Interaction** 则聚焦人与物体交互场景中的联合三维重建。

4. **生成模型的评价与反馈机制逐渐智能化**  
   **RubricRM** 将动态评价准则引入生成奖励建模，反映出图像生成研究正从单纯追求生成质量，转向更细粒度、可解释、面向任务的自动评估和优化。

---

## 二、值得重点关注的论文

### 1. **CLAP：跨具身视频世界模型**
题目提出“跨具身视频世界模型可实现零样本物理模拟”，其潜在意义较大：如果模型能够从不同机器人形态或不同主体的视频中学习共享的物理规律，就可能降低针对单一机器人重新采集数据和训练模型的成本。  
**关注重点：**跨具身迁移是否真正成立、零样本模拟的物理一致性、对未见动作和环境的泛化能力。

### 2. **LeVJEPA：高效且可扩展的视频预训练**
该工作强调“不依赖启发式”的视频预训练，可能代表视频表征学习从复杂训练技巧转向更统一、更可扩展的目标设计。若能同时改善训练效率、规模扩展性和下游性能，对视频理解、预测和世界模型都有基础性价值。  
**关注重点：**预训练目标、与现有 JEPA/视频生成方法的比较、计算效率和规模扩展曲线。

### 3. **FlashVLA：异步、流式动作解码**
VLA 模型的瓶颈不仅在模型大小，也在动作生成的等待时间和控制频率。FlashVLA 通过流式动作解码处理快速、异步的动作推理，直接面向真实机器人控制中的低延迟需求。  
**关注重点：**动作块缓存与更新策略、感知和控制的异步机制、延迟—成功率权衡，以及在真实机器人上的验证。

### 4. **SpatialCrafter：由单张图像构建世界模型**
从单幅图像生成三维代理并进一步实现空间或视角建模，是连接二维生成模型、三维重建和世界模型的重要方向。该工作若能从极少观测中构造可交互的空间表示，将对机器人导航、虚拟环境生成和视图合成具有吸引力。  
**关注重点：**生成的三维代理是否具有几何一致性、是否支持相机运动或环境交互，以及长期视角变化下的稳定性。

### 5. **Riemann-1.0：具身世界动作模型**
该论文将世界建模与动作建模结合，代表“物理 AI”从感知模型向统一行动模型演进的趋势。其价值取决于是否能在视觉、语言、动作和环境状态之间建立可泛化的联系。  
**关注重点：**模型是否支持规划与闭环控制、训练数据的具身多样性、跨任务迁移能力，以及仿真到真实的差距。

### 6. **RubricRM：基于动态准则的生成奖励模型**
传统图像生成评价通常依赖固定指标或单一偏好模型。动态 rubric 机制有望让奖励模型根据任务要求，从构图、主体一致性、编辑准确性等多个维度进行更细致的评价。  
**关注重点：**rubric 的生成和可靠性、奖励模型是否存在偏差、与人类偏好的一致性，以及对图像编辑任务的实际改进。

---

## 三、其他论文的研究价值

- **Reconstructing Humans and Objects in Interaction using Large Reconstruction Models**  
  针对人与物体接触、遮挡和相互作用场景的联合重建，属于比独立物体重建更具挑战性的三维视觉问题。对人机交互、动作理解和数字人的构建具有应用价值。

- **PACE：统一的压缩—提取范式用于快速 VLM 推理**  
  通过先压缩视觉信息、再提取任务相关内容，可能为视觉 token 削减和动态计算提供统一框架，适合关注 VLM 部署和推理成本的读者。

- **SSMB：运动模糊下的自监督局部特征检测**  
  面向真实动态场景中的低质量图像，强调无需人工标注的局部特征学习。对视觉里程计、匹配、三维重建和机器人定位较为实用。

- **ClusterAttention：双向注意力的免训练加速**  
  通过聚类或结构化近似减少双向注意力开销，特点是无需重新训练模型，可能更容易应用于现有 Transformer/VLM 系统。需要重点检验其近似误差和不同任务上的稳定性。

---

## 四、正在形成的研究方向

1. **跨具身、跨平台的世界模型**  
   未来模型可能不再针对单一机器人训练，而是学习与机器人形态相对解耦的环境动力学和动作表示。

2. **从视频预测到行动结果预测**  
   视频预训练正在由“预测下一帧”扩展到预测动作、状态变化和物理后果，世界模型与 VLA 的边界将进一步融合。

3. **低延迟、异步和持续推理的具身系统**  
   流式动作解码、增量更新、视觉 token 压缩和快速注意力将成为真实机器人部署的基础技术。

4. **单图像或少观测条件下的三维世界建模**  
   生成式三维代理可能成为连接图像生成、空间理解和交互式模拟的关键中间表示。

5. **面向任务的生成评价与奖励建模**  
   动态 rubric、可解释评价和多维奖励将推动图像生成与编辑系统从“看起来真实”转向“符合具体任务目标”。

6. **无标注、弱监督和训练后加速方法**  
   SSMB 和 ClusterAttention 分别代表利用自监督数据和无需重新训练的模型优化，能够降低数据与部署门槛。

---

## 五、建议优先阅读全文的论文

### 第一优先级：适合关注具身智能和世界模型的研究人员

1. **CLAP**  
   可能触及跨机器人泛化和零样本物理模拟这一核心问题，理论意义与应用潜力都较高。

2. **FlashVLA**  
   直接针对真实 VLA 系统的延迟和异步控制问题，工程价值突出。

3. **Riemann-1.0**  
   适合了解“世界模型 + 动作模型 + 物理 AI”统一趋势，但应重点审查其数据规模和实际控制验证。

4. **SpatialCrafter**  
   对三维生成、空间智能和单图像世界建模感兴趣的读者值得优先阅读。

### 第二优先级：适合关注基础模型效率和视频学习的研究人员

5. **LeVJEPA**  
   可能对视频预训练范式和计算效率带来更基础性的影响。

6. **PACE**  
   适合研究 VLM 推理优化、token 压缩和模型部署的读者。

7. **ClusterAttention**  
   若方法确实能够在不训练的情况下稳定降低注意力成本，其实际迁移价值较高。

### 第三优先级：适合关注具体视觉任务与生成评价的研究人员

8. **RubricRM**：图像生成、编辑、偏好学习和奖励建模方向优先。  
9. **Reconstructing Humans and Objects in Interaction**：三维重建、人体建模和交互理解方向优先。  
10. **SSMB**：局部特征、视觉定位、SLAM 和运动模糊方向优先。

## 总结

本期最突出的信号是：计算机视觉研究正从独立的感知任务快速迈向**可预测、可交互、可行动的视觉系统**。世界模型和物理 AI 构成上层研究主线，而流式推理、视觉压缩、快速注意力和高效预训练则解决其落地所需的计算瓶颈。若时间有限，建议优先阅读 **CLAP、FlashVLA、LeVJEPA、SpatialCrafter 和 Riemann-1.0**，它们分别覆盖了跨具身泛化、实时控制、视频基础模型、三维世界建模和统一行动模型这几条最重要的发展路线。

---

## Table of Contents

1. [CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators](#2608.27406v1)
2. [Reconstructing Humans and Objects in Interaction using Large Reconstruction Models](#2608.27407v1)
3. [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](#2608.27395v1)
4. [FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference](#2608.27384v1)
5. [PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference](#2608.27206v1)
6. [SSMB: Self-Supervised Local Feature Detection under Motion Blur](#2608.27181v1)
7. [SpatialCrafter: Single Image World Modeling with Generative 3D Proxies](#2608.27073v1)
8. [Riemann-1.0: An Embodied World Action Model for Physical AI](#2608.27033v1)
9. [ClusterAttention: A training-free speedup of bidirectional attention](#2608.26965v1)
10. [RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing](#2608.26956v1)

---

## Papers

<a id='2608.27406v1'></a>
## [CLAP: Cross-Embodiment Video World Models are Zero-Shot Physical Simulators](https://arxiv.org/abs/2608.27406v1)

**Authors:** Kechen Liu, Ola Shorinwa

**Published:** 2026-08-27

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

State-of-the-art action-conditioned video models are typically restricted to a single robot embodiment, preventing them from leveraging the vast corpus of heterogeneous video data that contains rich signals for learning generalizable physics. To bridge this gap, we introduce CLAP, a framework for cross-embodiment action-conditioned video generation capable of being trained on diverse, internet-scale videos across human and robotic agents. CLAP is grounded in the insight that universal physical laws govern spatiotemporal dynamics regardless of the actor. However, cross-embodiment learning is non-trivial because action representations vary sharply across robot platforms and are typically absent in human videos. CLAP addresses this fundamental challenge through the following core contributions. First, CLAP reconciles disparate action spaces using end-effector poses, language instructions, and latent actions. Second, to resolve their individual limitations, CLAP introduces a curriculum-based cross-embodiment learning recipe that first learns foundational physical priors across unlabeled video data using latent actions and subsequently grounds them in end-effector action spaces for zero-shot deployment to real-world tasks. Crucially, CLAP approaches or surpasses state-of-the-art single-embodiment video models in challenging environments like DROID. These performance advantages compound via few-shot adaptation to establish a novel paradigm for training single-embodiment video world models. Ultimately, CLAP delivers the most comprehensive suite of action-conditioned video world models to date - spanning diverse action-conditioning spaces (end-effector, language, and latent) and robot morphologies (including cross-embodiment, DROID, Bridge, bimanual YAM robots, and G1 humanoids). We open-source all code and models. Project Website at https://omni-clap.github.io .

**Analysis:**

# 1. 摘要翻译

现有的动作条件视频模型通常局限于单一机器人形态，无法利用包含丰富物理信息的大规模异构视频数据。为此，本文提出 CLAP：一种跨具身动作条件视频生成框架，可在人体与多种机器人数据上学习。其核心观察是：无论执行者形态如何，物体运动和交互都遵循统一的物理规律。CLAP通过末端执行器位姿、自然语言指令和学习得到的潜在动作，统一不同机器人及无动作标注的人类视频。进一步地，CLAP采用课程式训练：先利用潜在动作从无标注视频中学习基础物理先验，再使用末端执行器动作进行对齐，从而支持真实任务中的零样本部署。实验表明，CLAP在DROID等复杂环境中达到或超过单具身视频模型，并可通过少量数据适配新机器人。结合推理时跨策略规划和视频世界模型中的强化学习，CLAP提升了π0.5、MolmoAct-2等策略在真实操作任务中的表现。

# 2. 方法动机分析

**驱动力与痛点：**单机器人训练数据规模有限，且不同机器人的关节数、运动范围和观测形式差异显著；人类互联网视频数量巨大，却通常没有动作标签。末端动作虽精确，但不能用于无标注视频；潜在动作可利用无标注数据，却无法直接控制真实机器人；语言动作具有统一接口，但数值信息会受到文本分词精度限制。

**核心假设：**不同具身共享物理规律，因此应共享视频模型中的动力学先验，而不是为每种机器人分别训练模型。

# 3. 方法设计详解

## 3.1 统一动作表示

1. **末端执行器动作（CLAP-EE）**：通过正向运动学把关节状态转换为7维动作  
\((x,y,z,\text{roll},\text{pitch},\text{yaw},g)\)。不同机器人的绝对动作按各自运动范围归一化到[-1,1]，既保留精度，又消除尺度差异。作者发现，末端动作使用绝对坐标通常优于相对坐标，因为相对动作在长时预测中会累积误差。

2. **语言动作（CLAP-LANG）**：将数值动作转成模板文本，如“x=…, y=…, z=…, roll=…”。为减少分词器处理高精度数值的损失，语言动作采用简洁模板，并使用相对坐标缩小数值范围。其优点是接口统一，缺点是离散文本表示降低空间精度。

3. **潜在动作（CLAP-LAM）**：对视频帧对\((f_t,f_{t+\Delta t})\)编码，提取32维连续潜在向量，再由解码器根据当前帧和潜在动作重建未来帧。其本质是用“造成画面变化的隐变量”替代真实动作，因此无需动作标注，适合人体视频，但存在从潜在空间映射到机器人控制空间的部署鸿沟。

## 3.2 课程式潜在到末端动作训练

CLAP-CURR先在机器人和人类无标注视频上训练潜在动作模型，使视频UNet学习通用的时空变化和物理先验；随后替换动作输入头，将潜在动作头改为7维末端动作头，并在有动作标签的机器人视频上联合训练新动作头与视频模型。这样既获得无标注数据规模，又保留末端动作的真实可控性，避免额外适配器造成的预测误差。

## 3.3 视频世界模型结构

模型采用潜空间视频扩散架构：冻结视频VAE，将图像压缩为潜变量；使用SVD时空UNet预测未来视频噪声。输入包括历史帧、当前观测、任务语言和逐帧动作，输出未来帧。历史窗口为6帧，预测5帧；动作分别通过MLP或冻结CLIP编码，并在帧级别注入，使每个未来帧关注对应时刻的动作。多视角图像先垂直拼接，在统一潜空间中联合预测，以增强几何一致性。推理时采用50步EDM采样和自回归分块 rollout。

# 4. 对比与创新

其本质区别不是提出全新视频骨干，而是把**跨具身数据规模化、动作空间统一、潜在动作预训练与真实动作对齐**组合成一个训练范式。主要创新包括：  
- 用多种动作表示覆盖有标注与无标注数据；  
- 提出“潜在动作预训练→末端动作精调”的课程学习；  
- 将跨具身模型作为单机器人模型的少样本初始化器；  
- 用世界模型在推理时筛选多个策略的动作候选，并支持模型内RL精调。

适合机器人操作、跨平台策略评估、动作规划和新形态机器人适配；对移动机器人等浮动参考系任务，还需要额外坐标建模。

# 5. 实验分析

作者在OXE、EgoDex、DROID、Bridge等数据上验证。代表性结论是：在DROID上，CLAP-CURR的LPIPS为0.204，接近甚至优于单具身Ctrl-World的0.205；跨策略规划使测量胶带、鱼、龙虾任务成功率达到80%、75%、95%。少样本微调后，模型还能适配14维YAM和26维G1 humanoid动作空间。

**优势：**数据利用范围广、跨机器人泛化强、部署接口直接、适配成本低。  
**局限：**仍会产生视频幻觉；推理约1.49–3.24秒，不能实时运行；训练规模尚未达到真正互联网规模；Bridge等简单环境中仍可能逊于专用模型。

# 6. 实用指南

论文已开源代码和模型：`github.com/omni-CLAP/clap`。复现时需统一多视角图像尺寸，处理不同数据集的夹爪信号，将动作归一化，并按数据集比例采样。视频模型训练约100K步、8张H100/H200、有效batch size 64；LAM使用32维潜在动作，β-VAE的β为\(10^{-6}\)。迁移到新机器人时，保留UNet和VAE，仅替换为目标机器人的动作头，并用少量末端动作视频微调；若动作维度更高，则使用新的动作编码器。

# 7. 总结

**核心思想：**用跨具身视频学习共享物理先验。

**速记版Pipeline：**
1. 收集人类与多机器人视频，并统一画面和动作格式。  
2. 用帧变化学习无需人工标注的潜在动作。  
3. 在潜在动作上预训练视频世界模型。  
4. 用真实末端动作继续训练，使模型可直接控制机器人。  
5. 用模型预测并筛选策略动作，或少量微调适配新机器人。

**Key Findings:**

- State-of-the-art action-conditioned video models are typically restricted to a single robot embodiment, preventing them from leveraging the vast corpus of heterogeneous video data that contains rich signals for learning generalizable physics.
- To bridge this gap, we introduce CLAP, a framework for cross-embodiment action-conditioned video generation capable of being trained on diverse, internet-scale videos across human and robotic agents.
- CLAP addresses this fundamental challenge through the following core contributions.
- Crucially, CLAP approaches or surpasses state-of-the-art single-embodiment video models in challenging environments like DROID.
- These performance advantages compound via few-shot adaptation to establish a novel paradigm for training single-embodiment video world models.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27406v1)
- [arXiv](https://arxiv.org/abs/2608.27406v1)

---

<a id='2608.27407v1'></a>
## [Reconstructing Humans and Objects in Interaction using Large Reconstruction Models](https://arxiv.org/abs/2608.27407v1)

**Authors:** Agniv Chatterjee, Georgios Pavlakos

**Published:** 2026-08-27

**Categories:** cs.CV

**Abstract:**

Estimation of Human-Object Interactions in 3D (3D HOI) is a fundamental problem in 3D computer vision with applications in AR/VR, robotics, and embodied AI. However, reconstructing these interactions in 3D remains challenging due to depth ambiguities, occlusions, and object shape variability. Existing approaches are primarily concerned with reprojection and contact constraints, fitting parametric human models and object templates to 2D images. In this paper, we explore a different avenue. We present MILO, a framework that leverages the visual capabilities of Large Reconstruction Models (LRMs) to recover detailed 3D human-object interactions from a single image. Our key observation is that LRMs provide a powerful geometric scaffold that preserves relative human-object arrangement and proximity cues. This significantly simplifies the reconstruction procedure, reframing the problem as interpreting the LRM mesh: we segment it into human and object components, fit a parametric body model to the human part, and optionally align an object template to the object part (if such a template is available). MILO achieves strong reconstruction accuracy and outperforms existing baselines across multiple benchmarks and interaction scenarios. Our code is available at https://ac5113.github.io/MILO.

**Analysis:**

## 1. 摘要翻译

从单张图像估计三维人-物交互（3D HOI）是计算机视觉中的基础问题，可应用于 AR/VR、机器人和具身智能。然而，深度歧义、遮挡和物体形状多样性使其十分困难。现有方法主要依赖重投影与接触约束，将参数化人体模型和物体模板拟合到二维图像上。本文提出 **MILO**，利用大型重建模型（LRM）从单张图像恢复细致的三维人-物交互。核心观察是：LRM生成的网格虽不一定具有准确的度量尺度，却保留了人与物之间的相对布局和邻近关系，因此可作为几何支架。MILO将该网格分割为人体和物体部分，将参数化人体模型拟合到人体区域；若存在物体模板，则进一步将其对齐到物体区域。实验表明，MILO在多个基准和交互场景上优于现有方法。

## 2. 方法动机

**现有痛点：**传统方法从二维关键点、轮廓、重投影和接触标签反推三维，深度和物体位姿约束弱；模板检索受类别覆盖和形状相似度限制，且常需真实接触或深度等特权信息。独立重建人体和物体也会破坏二者的空间关系。

**核心假设：**LRM虽然可能幻觉、尺度不准或几何不完整，但其联合重建结果包含可靠的“人与物如何相对摆放”的信息。与其重新从二维拟合交互，不如解释这个联合网格。

## 3. 方法设计详解

### Pipeline

1. **联合网格生成**  
   输入RGBA图像，使用 Hunyuan3D-2.0 生成包含人体和交互物体的整体三维网格。该网格不被视为最终结果，而被视为记录相对位置与接近关系的非参数几何支架。

2. **多视角关键点估计**  
   从方位角每30°、仰角 \(0,\pm30,\pm60°\) 渲染60个虚拟视角。在每个视角运行 ViTPose 获取25个身体关键点，运行 HaMeR获取左右手各21个关键点。  
   对每个关键点，仅保留置信度大于0.6的观测；至少需要3个视角。利用所有视角对进行三角化，以重投影误差5像素寻找最大一致集，再对内点进行非线性优化，得到最多67个三维关键点及其平均置信度。这样可将二维检测器转化为位于LRM坐标系中的三维人体观测。

3. **SMPL-H人体拟合**  
   以 HMR2.0 初始化身体姿态与形状，以 HaMeR初始化手部姿态，固定LRM网格并分两阶段优化：  
   - **Root fitting：**只优化全局旋转、根平移和尺度，使SMPL-H关节对齐三维关键点；  
   - **Pose fitting：**进一步优化身体/手部姿态和形状。目标由关键点鲁棒损失、VPoser身体先验、MANO手部先验、形状L2先验组成。针对LRM遮挡区域可能产生错误姿态，引入相对HMR初值的软锚定损失；同时仅在可见关节对应的SMPL-H顶点上使用单向鲁棒Chamfer损失，避免用不可靠的幻觉区域约束人体。  
   总损失权重为：\(\lambda_{rf}=10,\lambda_{bp}=\lambda_{hp}=0.04,\lambda_\beta=0.05,\lambda_{ha}=10,\lambda_{3D}=50\)。

4. **人体—物体点云分割**  
   将每个LRM顶点投影到60个渲染视图，使用 Grounding DINO/SAM 产生“person、物体名称”掩码。对不同视角的标签进行聚合，并以视角质量 \(Q\) 加权：同时考虑可见顶点覆盖率和掩码质量。  
   人物边界附近最容易发生误分，因此构造多尺度膨胀边界，并使用距离变换得到边界权重。最终将非人体顶点作为物体点云，再进行离群点移除、局部邻域稀疏点剪除和DBSCAN最大簇筛选。

5. **可选模板对齐**  
   若有物体模板，先渲染模板和LRM网格，利用几何感知语义对应建立稀疏三维匹配。输入图像与LRM视图先选最相似的5个视角，每个模板视图保留3个最佳配对；通过像素描述子匹配、可见性和15像素空间门限将二维匹配提升到三维，并通过双向循环一致性过滤错误匹配。  
   随后用加权 Sim(3) Kabsch 求尺度、旋转和平移，再结合LRM点云最近邻约束进行ICP式迭代细化。无可靠语义对应时，退化为质心/尺度初始化加ICP。

## 4. 方法对比与创新

MILO的本质变化是：从“基于图像重新猜测人与物的三维关系”转向“解释LRM已经生成的联合三维关系”。创新主要包括：  
1. 首次将LRM作为HOI的联合几何支架，而非仅用于物体重建；  
2. 多视角三角化关键点 + 两阶段SMPL-H拟合，解决LRM人体网格不规则和尺度不确定问题；  
3. 面向接触边界设计的多视角点云分割；  
4. 模板仅作为可选精化工具，不依赖接触标注。  
适合单图像、物体类别开放、模板缺失和遮挡较强的场景。

## 5. 实验分析

作者在 InterCap、HODome、IMHD 上以PA-CD评估人体、物体及联合网格，并在InterCap上由几何邻近关系推断接触。代表性结论是：MILO在无模板、无接触信息时仍全面优于PICO等方法；联合LRM支架明显优于人体和物体独立重建。  
**优势：**不需要接触监督，交互布局自然，模板自由，能泛化到野外图像。  
**局限：**性能受LRM质量和点云分割影响；小物体、截断物体、多物体和对称物体容易失败，模板对应还可能产生方向翻转。核心流程约344秒/图，速度仍较慢。

## 6. 实用指南

论文提供代码：`https://ac5113.github.io/MILO`。复现时需准备LRM、HMR2.0、HaMeR、ViTPose、Grounding DINO/SAM及SMPL-H模型；严格实现60视角渲染、置信度阈值0.6、最少3视角、三角化误差5像素和上述优化权重。模板对齐中的语义对应耗时最大，若无可靠对应应关闭模板对齐。该框架可迁移到人体—场景、机器人—物体或视频HOI，只需替换对象分割器、结构化人体/主体模型及跨帧一致性模块。

## 7. 总结

**核心思想：用LRM联合网格支架重建交互。**

**速记版Pipeline：**  
1. 从图像生成包含人和物的三维网格；  
2. 多角度观察网格并恢复三维人体关节；  
3. 将标准人体模型贴合到这些关节和人体表面；  
4. 分离物体点云，必要时把已知模板对齐上去；  
5. 用人与物的三维距离直接获得交互和接触关系。

**Key Findings:**

- We present MILO, a framework that leverages the visual capabilities of Large Reconstruction Models (LRMs) to recover detailed 3D human-object interactions from a single image.
- MILO achieves strong reconstruction accuracy and outperforms existing baselines across multiple benchmarks and interaction scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27407v1)
- [arXiv](https://arxiv.org/abs/2608.27407v1)

---

<a id='2608.27395v1'></a>
## [LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics](https://arxiv.org/abs/2608.27395v1)

**Authors:** Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Florian Buettner

**Published:** 2026-08-27

**Categories:** cs.CV, cs.AI

**Abstract:**

Video carries the temporal structure of the physical world, yet learning representations from it has remained computationally expensive: prevailing self-supervised methods either prevent representation collapse through architectural asymmetries, coupling an exponential-moving-average target encoder, a stop-gradient, and a capacity-limited predictor, or circumvent it by reconstructing masked content in pixel space. We introduce LeVJEPA, the first video encoder trained under LeJEPA's collapse-free objective, which dispenses with both. A single encoder is trained with an invariance loss over global and local views of a clip, regularized by SIGReg, which excludes collapse with a provable guarantee. The architecture reduces to an encoder and a projector, and the objective to a single hyperparameter. This formulation admits two properties. First, the cost of pretraining is governed by the number of tokens the encoder observes; uniform random token dropping renders this number small while simultaneously improving downstream accuracy. At matched epochs on identical data, LeVJEPA matches or surpasses V-JEPA 2 across ViT-S/B/L at 5.6 to 20.8x less pretraining compute, and at matched total FLOPs it exceeds the strongest video baseline by 7.6 points on ImageNet-1K while remaining competitive on motion-centric benchmarks. Second, since no asymmetry between branches is required, the encoder can be trained with block-causal attention at no measurable accuracy cost: temporal ordering becomes a property of the encoder itself. Against a compute-matched DINOv2 trained on frames of the same videos, LeVJEPA approaches the image-pretrained encoder on appearance-centric evaluation while nearly doubling its motion-centric accuracy. These results indicate that, once its computational overhead is removed, video becomes a viable and in several respects preferable substrate for general-purpose visual pretraining.

**Analysis:**

## 1. 摘要翻译

视频包含物理世界的时间结构，但从中学习表征一直计算昂贵。现有自监督方法通常依赖结构不对称来防止表征坍塌，例如指数移动平均目标编码器、停止梯度和容量受限的预测器；或者通过专用解码器在像素空间重建被遮挡内容。本文提出 **LeVJEPA**：首个采用 LeJEPA 无坍塌目标的视频编码器。它仅使用一个编码器，通过全局视图与局部视图之间的一致性损失进行训练，并由 SIGReg 提供具有理论保证的防坍塌约束。因此，模型只需编码器和投影器，目标函数也只含一个超参数。

作者进一步发现，预训练成本主要由编码器实际处理的 token 数量决定。均匀随机丢弃 token 不仅大幅降低计算量，还能提升下游精度。在相同训练轮数和数据下，LeVJEPA 在 ViT-S/B/L 上以低 5.6–20.8 倍的预训练计算达到或超过 V-JEPA 2；在相同 FLOPs 下，ImageNet-1K 准确率领先最强视频基线 7.6 个百分点。同时，编码器可以采用块因果注意力，使每帧表征只依赖当前及过去帧，并且无需重新编码历史帧即可逐帧扩展。结果表明，视频可以成为高效且通用的视觉预训练载体。

## 2. 方法动机分析

**驱动力**：视频比图像提供额外的运动、因果和物体持续性信息，但传统视频预训练成本极高。作者希望同时解决“计算昂贵、训练结构复杂、缺乏因果性”三个问题。

**现有痛点**：V-JEPA 类方法需要在线编码器、EMA 目标编码器、停止梯度和预测器；VideoMAE 类方法需要像素重建和专门遮挡策略。它们的 mask 往往服务于预测任务，而非表征本身，且不能自然得到流式因果表征。

**核心假设**：只要显式约束 embedding 分布接近各向同性高斯，就能在不依赖教师网络等启发式结构的情况下避免坍塌；既然不需要重建缺失内容，token 采样就可以被视为数据增强，并可极度稀疏。

## 3. 方法设计详解

### Pipeline

1. 从视频采样 **16 帧 clip**，构造一个高分辨率全局视图和 \(V\) 个局部视图。局部视图共享相同时间窗口，但进行空间裁剪和光度增强。  
2. 每帧通过 \(16\times16\) 卷积 patch embedding；默认时间跨度 \(\tau=1\)，即每个 token 对应单帧空间块。  
3. 对每个视图的 patch token **均匀随机丢弃 95%**，仅保留 5%；[CLS] token 永不丢弃。与 tube mask 不同，各帧保留位置独立随机采样，从而获得跨空间、时间分布的稀疏观察。  
4. 所有视图输入同一个共享视频 ViT。默认采用块因果注意力：同一帧内双向注意，不同帧之间只看过去；[CLS] 汇聚全部 token，但不被 patch token 访问。  
5. 取各视图的 [CLS] 表征，经两层 MLP 投影器映射到 \(K=256\) 维 embedding \(z_v\)。投影器用于避开编码器末端 LayerNorm 对 SIGReg 造成的球面约束，预训练后丢弃。  
6. 优化  
\[
L=L_{\rm inv}+0.02L_{\rm SIGReg}.
\]
其中
\[
L_{\rm inv}=\frac1{V+1}\sum_v\|z_0-z_v\|_2^2
\]
使局部视图与全局视图表达一致，且两支路均反向传播；没有 stop-gradient 或目标编码器。  
7. SIGReg 将 embedding 投影到 1024 个随机方向，利用 Epps–Pulley 正态性检验，惩罚投影分布偏离标准高斯。依据 Cramér–Wold 定理，所有一维投影都服从标准正态，等价于整体接近各向同性高斯，因此常数坍塌会被排除。积分用 \([0,3]\) 上 17 个节点的梯形积分近似。

### 模型协同

编码器学习视频内容与时间结构；投影器只负责在适合分布正则化的空间计算损失；一致性项学习视图不变性，SIGReg 保持表达多样性。Polyak 平均模型仅作为评估检查点，不参与训练目标，区别于 EMA teacher。

## 4. 对比、创新与适用场景

本质区别是：LeVJEPA 不预测缺失 token，也不依赖教师—学生不对称，而是“稀疏观察 + 全局一致性 + 分布约束”。创新包括：将 LeJEPA 的理论防坍塌机制迁移到视频；把随机 token 丢弃从重建任务的 mask 改为表征增强；在预训练阶段直接内置块因果编码。适合资源受限的视频表征学习、流式感知、自动驾驶、机器人和世界模型；对高精度运动理解，过度稀疏可能损失跨帧对应信息。

## 5. 实验分析

作者在相同 K710 数据、训练轮数和 FLOPs 下，与 VideoMAEv2、V-JEPA 2、DINOv2 比较，并进行 token 丢弃、局部视图数量、时间 patch 聚合和注意力方向的消融。代表性结论：相同 FLOPs 下，LeVJEPA 的 ImageNet 准确率为 61.0%，领先最佳视频基线 7.6 个百分点；块因果注意力与双向注意力性能基本相当，同时 Something-Something-v2 的运动优势接近图像模型的两倍。

优势是结构简单、计算和显存高效、理论上防坍塌、天然支持因果推理。局限是高比例丢弃在短训练阶段会损害运动任务，且尚未充分验证大规模模型、密集预测和互联网级数据。

## 6. 实用指南

论文提供代码与模型：**levjepa.github.io**。复现重点是：16 帧 clip、\(\tau=1\)、95% 随机丢弃、4 个局部视图、\(\lambda=0.02\)、256 维投影、1024 个随机方向和块因果注意力。下游冻结编码器，用 attentive probing；Kinetics-400 使用 token 平均后的线性分类。迁移到分类、检索、跟踪或世界模型时，可丢弃投影器，使用编码器 token；流式任务直接缓存历史帧状态，避免重复编码。

## 7. 总结

**核心思想：用分布约束替代视频预训练启发式。**

**速记版 pipeline：**
1. 从同一视频片段生成全局和局部视图。  
2. 每个视图随机保留极少量 patch。  
3. 用同一因果编码器提取 [CLS] 表征。  
4. 让不同视图表征接近，同时强制整体分布接近高斯。  
5. 丢弃投影器，直接迁移编码器表征。

**Key Findings:**

- We introduce LeVJEPA, the first video encoder trained under LeJEPA's collapse-free objective, which dispenses with both.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27395v1)
- [arXiv](https://arxiv.org/abs/2608.27395v1)

---

<a id='2608.27384v1'></a>
## [FlashVLA: Streaming Action Decoding for Fast and Asynchronous VLA Inference](https://arxiv.org/abs/2608.27384v1)

**Authors:** Zekai Li, Jiaming Tang, Zhijian Liu

**Published:** 2026-08-27

**Categories:** cs.RO

**Abstract:**

Vision-Language-Action (VLA) models are increasingly promising for robotic manipulation, yet their real-world deployment remains bottlenecked by high inference latency and unstable asynchronous execution. This challenge is particularly pronounced in flow-matching-based VLA models, where action decoding requires multiple iterative steps conditioned on the VLM context. While efficient inference methods improve control frequency and asynchronous methods reduce execution idle time, existing approaches often fail to jointly achieve low-latency inference and accurate, temporally consistent asynchronous execution. We introduce \textbf{FlashVLA}, a streaming action decoding framework that addresses both challenges in a unified formulation. FlashVLA maintains a streaming action buffer with multiple chunks at different noise levels and decodes them using chunk-wise causal attention. This design allows FlashVLA to produce one executable action chunk per inference step. Moreover, its chunk-wise autoregressive formulation implicitly preserves action continuity, enabling smooth asynchronous execution without extra future-state conditioning. Across extensive simulated and real-world experiments, FlashVLA substantially improves inference speed while maintaining strong task performance. It can achieve $\geq$30\,Hz control frequency on a single GPU with smooth asynchronous inference in real-world deployment.

**Analysis:**

# 1. 摘要翻译

视觉-语言-动作（VLA）模型在机器人操作中很有潜力，但其实际部署受推理延迟高和异步执行不稳定的限制。对于基于流匹配的VLA，动作解码需要在视觉语言模型上下文条件下进行多次迭代，问题尤其突出。现有高效推理方法和异步执行方法通常只能分别降低延迟或缓解执行空闲，难以同时实现低延迟与准确、时间一致的异步控制。

本文提出 **FlashVLA**，一种流式动作解码框架。它维护一个包含多个不同噪声等级动作块的流式缓冲区，并通过块级因果注意力联合解码这些动作块，使每次推理都能产生一个可执行动作块。同时，块级自回归结构能够隐式保持动作连续性，无需额外的未来状态预测。实验表明，FlashVLA在仿真和真实机器人任务中显著提升推理速度，并保持较强任务性能；单GPU上可实现不低于30 Hz的平滑异步控制。

# 2. 方法动机

**驱动力**：π0.5等流匹配VLA的动作解码约占推理时间75%，一个动作块通常需要10次串行去噪，导致控制频率低、块切换时机器人停顿。

**现有痛点**：同步推理虽然观测和动作对齐，却在每个动作块边界等待解码；异步推理能够重叠执行和预测，但预测使用的是过时观测，动作真正执行时机器人已处于模型未见过的状态，且预测前视越长，误差越严重。VLASH等方法依赖未来状态预测，StreamingVLA等方法引入额外动作条件，增加了训练或结构复杂度。

**核心假设**：延迟和异步失配都源于“动作块彼此独立解码”。若让不同动作块联合去噪，并让未来块参考即将执行的近端块，就能同时摊薄去噪成本并保持轨迹连续。

# 3. 方法设计详解

## 3.1 流程

1. **构建动作缓冲区**：维护 \(N\) 个动作块  
   \[
   B_t=[x_{\tau_1}^{(1)},...,x_{\tau_N}^{(N)}],\quad \tau_1<...<\tau_N
   \]
   前端动作块噪声最低、即将执行；尾部动作块噪声最高、代表更远未来。噪声等级采用单调阶梯式安排，使“去噪进度”和“执行顺序”一致。

2. **联合前向计算**：一次模型前向同时更新缓冲区中的所有动作块，每个块沿自己的噪声等级前进一步，而非对单个块连续执行10次去噪。

3. **块级因果注意力**：噪声更高的未来块可以关注噪声更低的历史/近端块，但反向不可见。这样，未来动作会参考已经被细化的执行轨迹，获得隐式的未来状态条件；同时避免即将执行的块受到更远未来噪声的干扰。

4. **冷启动**：初始缓冲区由 \(N-1\) 个padding块和一个高斯噪声块组成，运行 \(N-1\) 次预热。预热期间不执行模型预测，而执行安全默认动作，因此不会将未成熟动作送给机器人。

5. **稳定流式阶段**：每次推理依次执行：  
   - 更新所有缓冲动作块；  
   - 弹出最前端、最干净的动作块；  
   - 执行该动作块；  
   - 缓冲区整体前移；  
   - 尾部加入新的纯噪声块。  
   经过预热后，每次前向都能输出一个动作块，原本集中在一次推理中的多步去噪被摊销到连续时间步中。

## 3.2 训练与结构

预训练VLA原本只见过“单块、从纯噪声开始”的去噪任务，因此作者设计**多缓冲区联合微调**：对同一观测构造从冷启动到稳态的全部 \(N\) 种缓冲配置，将它们打包为一个训练样本；不同配置之间用注意力掩码隔离，共享一次视觉语言编码。训练损失仍是流匹配速度场损失，但覆盖每个有效缓冲状态。

动作专家使用FiLM增强多层噪声时间步条件，使模型能够同时处理多个噪声等级。对于原本时间步条件较弱的模型，还需增加time-MLP和FiLM层。部分数据集会出现归一化层梯度爆炸，作者采用重新初始化动作专家归一化层的策略。

# 4. 方法对比与创新

FlashVLA的本质区别不是让每次前向更便宜，也不是简单减少去噪步数，而是改变“一次前向产生什么”：从“一次完成一个动作块”变为“一次并行推进多个动作块，并输出一个成熟块”。

主要创新包括：

- 将流式块级扩散/流匹配引入VLA动作解码；
- 用交错噪声缓冲区实现每步输出一个动作块；
- 用块级因果注意力把未来动作与当前执行轨迹结构性连接；
- 通过联合打包训练同时覆盖冷启动和稳态配置。

它适合流匹配或扩散式动作头、需要高控制频率和异步执行的机器人操作。对非迭代式动作头、极短任务或视觉/仿真开销远大于策略推理的场景，收益会较有限。

# 5. 实验分析

作者在LIBERO、RoboTwin 2.0、SmolVLA、LingBot-VLA及Franka真实机器人上验证。代表性结果是：LIBERO单步异步下，成功率由96.9%升至97.8%，每步时间由53.8 ms降至22.1 ms，达到2.43倍端到端加速；真实Franka上可实现30 Hz控制，平均完成时间约提升1.3倍。去除块级因果掩码后异步成功率平均下降约10个百分点，说明真正关键的是因果跨块条件，而不只是缓冲区。

主要优势：低延迟、异步连续性强、无需未来状态预测、可迁移到多种流匹配VLA。主要局限：需要专门微调；每回合存在 \(N-1\) 次冷启动；短任务中预热成本不可忽略，且仍受原始VLA视觉编码和系统开销限制。

# 6. 实用指南

论文提供代码仓库：`github.com/z-lab/flashvla`。复现时应先选择流匹配VLA，修改动作专家的时间步条件和块级注意力掩码，再进行多缓冲区联合微调。关键超参数是块大小 \(C\) 与缓冲长度 \(N\)，经验上 \(N\times C\) 接近原模型动作块长度；例如π0.5在LIBERO使用 \(C=10,N=4\)，RoboTwin使用 \(C=20,N=4\)。训练需保持动作归一化统计一致，并严格模拟冷启动padding、异步延迟和安全默认动作。

迁移到其他任务时，只要动作生成过程具有多步扩散/流匹配结构，就可复用“缓冲区+因果掩码+联合微调”框架；需重新匹配动作块跨度、控制频率和执行时域。

# 7. 总结

**核心思想：让动作块交错去噪并因果串联。**

**速记版Pipeline：**

1. 把未来多个动作块放入不同成熟度的缓冲区；  
2. 每次推理同时推进所有动作块；  
3. 让未来块只能参考已细化的近端块；  
4. 每步执行最成熟动作，并在末尾加入新噪声块；  
5. 用包含所有缓冲状态的联合样本微调模型。

**Key Findings:**

- We introduce \textbf{FlashVLA}, a streaming action decoding framework that addresses both challenges in a unified formulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27384v1)
- [arXiv](https://arxiv.org/abs/2608.27384v1)

---

<a id='2608.27206v1'></a>
## [PACE: A Unified Condense-and-Extract Paradigm for Fast VLM Inference](https://arxiv.org/abs/2608.27206v1)

**Authors:** Junjie Liu, Shengyuan Ye, Xu Chen

**Published:** 2026-08-27

**Categories:** cs.CV, cs.AI

**Abstract:**

Vision-Language Models (VLMs) demonstrate exceptional visual reasoning capabilities, yet their inference costs escalate rapidly with the proliferation of visual tokens. Existing visual token pruning methods exhibit two fundamental limitations. First, most approaches operate exclusively post-vision encoder, leaving the substantial latency of the visual encoding phase unoptimized. Second, under strict token budgets, these methods often fail to jointly preserve holistic visual contexts and fine-grained details, leading to performance degradation. To address these bottlenecks, we propose PACE (Pixel-Adaptive Condense and Extract), a training-free inference framework that accelerates both the vision encoder and the Large Language Model (LLM) via a unified Condense-and-Extract paradigm. During the Condense stage, an Adaptive Pixel Compressor (APC) evaluates visual information density prior to encoding, adaptively downsampling redundant inputs, curtailing encoder computation while preserving global context and essential visual cues. In the Extract stage, a Dynamic Dual-Attention Extractor (DDAE) selectively retains visual tokens via a fusion of internal visual signals from the encoder and semantic signals from the LLM, safeguarding task-critical details. By integrating PACE into Qwen2.5-VL-7B, the model retains 93.8% of its original performance while utilizing only 10% of the visual tokens, yielding a 3.1x speedup in time to first token (TTFT). Our code is available at https://github.com/jjL357/PACE.

**Analysis:**

## 1. 摘要翻译
视觉语言模型（VLM）具有出色的视觉推理能力，但视觉token数量增长会显著增加推理成本。现有视觉token裁剪方法主要在视觉编码器之后操作，无法降低视觉编码阶段的延迟；同时，在严格token预算下，也难以兼顾整体布局与细粒度细节。为此，作者提出无需训练的PACE（Pixel-Adaptive Condense and Extract）框架，通过统一的“压缩—提取”范式同时加速视觉编码器和大语言模型。压缩阶段的自适应像素压缩器（APC）在编码前评估图像信息密度，对冗余输入进行自适应下采样；提取阶段的动态双注意力提取器（DDAE）融合视觉编码器内部信号与LLM语义信号，选择任务关键token。集成到Qwen2.5-VL-7B后，仅保留10%视觉token仍保持原模型93.8%的性能，并实现3.1倍TTFT加速。

## 2. 方法动机
**驱动力：**高分辨率VLM存在“视觉编码器计算”和“LLM预填充”双重瓶颈。  
**现有痛点：**后编码裁剪只能减少LLM输入长度，无法降低ViT成本；单纯依赖LLM注意力容易删除未被问题直接提及但具有结构支撑作用的细节，如表格线、文字笔画。  
**核心假设：**应在编码前保留连续二维布局并压缩冗余，在编码后再结合“语义相关性”和“视觉显著性”进行token提取。

## 3. 方法设计详解
### 3.1 APC：编码前自适应压缩
1. **浅层预览：**先用ViT第1个block生成视觉特征，并对token做L2归一化。该预览比RGB、熵、边缘密度等低级统计更具语义性，但会引入额外预览开销。  
2. **全局信息密度：**计算所有token的平均两两余弦相似度φ，定义  
   \(\rho_g=1-\phi\)。相似度越高，说明图像越冗余，保留比例越低。  
3. **局部细节对比：**求所有token均值作为背景基准，计算每个token与基准的距离，取距离最大的前10%并求均值，再缩放得到\(\rho_d\)。该分数用于保护白底文档中的小字、细线等稀疏细节。  
4. **确定分辨率：**  
   \(\rho=\alpha\rho_g+(1-\alpha)\rho_d\)，令目标面积保留率 \(r=\rho\)，将图像宽高分别缩放为原来的\(\sqrt r\)。若硬预算更低，则优先服从预算约束。相比删除离散patch，整体缩放能保持二维拓扑结构。

### 3.2 DDAE：编码后动态提取
1. 从LLM早期层（默认第2层）获得跨模态注意力，形成语义图 \(S_{llm}\)。  
2. 从ViT末层获得视觉自注意力，形成视觉图 \(S_{vis}\)。两者均归一化到[0,1]。  
3. 用两张注意力图的标准差作为置信度：分布越尖锐，说明该信号越确信少数区域重要。经温度为τ的softmax获得动态权重。  
4. 融合为  
   \(S_{final}=\alpha_wS_{llm}+\beta_wS_{vis}\)，按分数选取top-K token送入后续LLM层。早期提取可减少更多层的全序列计算，但过早提取可能损失复杂推理信息。

## 4. 方法对比与适用性
PACE的本质区别是同时改变**输入像素预算**和**输出token预算**，而主流FastV、VisionZip、DivPrune等主要只做后编码裁剪。其主要创新是：①将预编码分辨率分配纳入token压缩；②用全局冗余与局部细节联合决定分辨率；③用置信度驱动的视觉—语言双注意力提取token。适合高分辨率图像、文档、OCR、图表和动态分辨率VLM；对固定网格视觉编码器，APC可能无法带来编码器加速。

## 5. 实验分析
作者在Qwen2.5-VL-3B/7B、InternVL3.5-4B及9个视觉任务上评估，并进行模块、分辨率、注意力来源和提取深度消融。代表性结果是：Qwen2.5-VL-7B保留10%视觉token时平均性能达93.8%，TTFT约加速3.1倍；APC对ChartQA、OCRBench、DocVQA等细节任务提升尤其明显。局限是APC查询无关且一次性缩放，可能丢失微小字符、低对比度目标；DDAE无法恢复已经在缩放阶段丢失的信息。

## 6. 实用指南
代码已开源：`github.com/jjL357/PACE`。复现时需实现ViT浅层预览、全局/局部密度计算、双注意力融合及top-K提取。默认参数为：预览深度K=1、\(\alpha=0.6\)、\(\gamma=1.5\)、LLM提取层2、温度τ=0.5，图像采用双三次插值并按\(\sqrt r\)缩放。迁移到其他VLM时需重新校准预览层、注意力接口、token映射和延迟收益；也可迁移到视频、文档问答，并结合查询条件或高分辨率局部回取机制。

## 7. 总结
**核心思想：**先压缩像素，再融合注意力提取。  

**速记版pipeline：**
1. 用浅层视觉特征判断图像冗余与细节密度；  
2. 按图像复杂度调整输入分辨率；  
3. 完成较低成本的视觉编码；  
4. 融合视觉关注区域与问题相关区域；  
5. 只将最重要的少量视觉token交给LLM。

**Key Findings:**

- To address these bottlenecks, we propose PACE (Pixel-Adaptive Condense and Extract), a training-free inference framework that accelerates both the vision encoder and the Large Language Model (LLM) via a unified Condense-and-Extract paradigm.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27206v1)
- [arXiv](https://arxiv.org/abs/2608.27206v1)

---

<a id='2608.27181v1'></a>
## [SSMB: Self-Supervised Local Feature Detection under Motion Blur](https://arxiv.org/abs/2608.27181v1)

**Authors:** Zhenjun Zhao, Fabio Bellavia, Wenting Wang, Fan Zhu, Jiajun Wu, Suryansh Kumar, Mingqiang Wei, Haoang Li, Javier Civera

**Published:** 2026-08-27

**Categories:** cs.CV

**Abstract:**

Keypoint detection under motion blur remains a significant challenge, as blur distorts local image structure and degrades the repeatability of feature localization. Existing approaches either rely on computationally expensive deblur-then-detect pipelines that may introduce restoration artifacts, or learn to regress the image positions of handcrafted keypoints extracted on sharp images, which reflects the assumptions of the handcrafted detector rather than what is truly repeatable under blur. We present SSMB, a deblur-free, self-supervised keypoint detector for motion-blurred images that requires neither handcrafted detectors nor external pseudo-labels. SSMB introduces the Local Discriminability Enhancement (LDE) module, which restores fine-grained local discriminability after global feature mixing. Training is performed in two stages. First, geometric pretraining on synthetic shapes bootstraps spatially discriminative keypoint detection without any external detector, just from the rendered geometry. Second, blur-aware training on real sharp-blur image pairs learns blur-invariant detection through a multi-component self-supervised objective that enforces cross-domain consistency, geometric alignment, and spatial coverage. Extensive evaluations on keypoint detection, image matching, relative pose estimation, and visual localization under motion blur demonstrate that SSMB establishes a new state-of-the-art among sparse keypoint detectors, consistently outperforming both supervised and self-supervised baselines across all tasks. Code, models, and datasets will be publicly available upon paper acceptance.

**Analysis:**

## 1. 主要贡献概述

SSMB 提出了一种无需去模糊、无需手工特征检测器和外部伪标签的自监督局部关键点检测方法，专门针对运动模糊导致的局部结构破坏和定位不稳定问题。该方法通过“合成几何预训练 + 真实清晰—模糊图像对上的模糊感知训练”学习在模糊条件下仍具有重复性的关键点，并据摘要称在关键点检测、图像匹配、相对位姿估计和视觉定位等任务上取得了领先性能。

## 2. 关键创新与方法

### （1）去模糊式检测，而非“先恢复、后检测”

传统方案通常先对运动模糊图像进行图像复原，再运行关键点检测器。这类流程计算成本较高，而且复原过程可能引入伪纹理或重建伪影。SSMB 直接在模糊图像上检测关键点，目标是学习“在模糊下仍然可重复”的位置，而不是试图恢复一张理想的清晰图像。

### （2）Local Discriminability Enhancement（LDE）模块

摘要指出，网络中的全局特征混合可能增强上下文建模能力，但也可能削弱局部纹理和空间细节。LDE 模块的作用是：

- 在全局特征交互之后恢复细粒度局部判别性；
- 帮助网络更准确地区分邻近位置；
- 缓解运动模糊下局部结构被平滑、混合或错位的问题。

这体现了该方法在“全局上下文”和“局部定位精度”之间进行针对性平衡。

### （3）两阶段自监督训练

**第一阶段：基于合成几何的预训练。**  
模型在合成形状或几何图案上训练，仅依赖已知的渲染几何信息，而不使用传统角点检测器生成的伪标签。这一步用于建立基本的空间判别能力，使网络学会哪些位置具有稳定、明确的局部结构。

**第二阶段：真实清晰—模糊图像对上的模糊感知训练。**  
通过真实图像及其对应的清晰/运动模糊版本，构造多成分自监督目标，主要包括：

- **跨域一致性**：同一场景在清晰域和模糊域中应产生相互一致的关键点；
- **几何对齐**：对应关键点的位置应满足图像变换或已知几何关系；
- **空间覆盖**：避免关键点集中在少数高响应区域，提升图像范围内的分布均衡性。

这种训练策略的核心并非复现清晰图像上的传统关键点，而是直接优化模糊环境中的可重复性和实用性。

## 3. 对领域的潜在影响

1. **重新定义模糊条件下的关键点学习目标**  
   该工作强调，模糊场景中的理想关键点不一定等同于清晰图像上由传统检测器选出的点。直接学习跨清晰—模糊域稳定的特征位置，有助于推动局部特征检测从“拟合手工检测器”转向“面向下游匹配和几何稳定性优化”。

2. **降低复杂视觉系统的推理成本**  
   如果直接检测模糊图像能够达到或超过“去模糊 + 检测”的效果，则可以减少额外的图像复原模块，降低延迟、显存和系统复杂度，尤其适合实时或资源受限平台。

3. **提升视觉几何任务在动态成像条件下的鲁棒性**  
   运动模糊会显著影响特征匹配、相机位姿估计、三维重建和定位。专门针对模糊设计的稀疏关键点检测器，可能成为现有视觉前端的重要补充。

4. **为自监督局部特征学习提供新的训练范式**  
   “合成几何初始化 + 真实退化一致性约束”的方式不依赖人工标注或外部检测器，可能推广到低照度、压缩伪影、噪声、雨雾和焦外模糊等其他成像退化问题。

## 4. 可能受益的相关领域与应用

- **机器人视觉与视觉里程计**：机器人快速运动或振动时容易产生运动模糊，鲁棒关键点有助于跟踪和定位。
- **无人机与自动驾驶**：高速运动、转向和复杂光照下的图像通常包含明显模糊，可改善匹配、定位和建图。
- **SLAM 与三维重建**：更稳定的局部特征能够减少跟踪丢失，提高地图构建和相机轨迹估计的可靠性。
- **视觉定位与地点识别**：移动平台采集的查询图像可能因运动而模糊，模糊不变特征有助于与清晰地图或数据库匹配。
- **增强现实与混合现实**：快速头部运动或相机移动会降低跟踪稳定性，鲁棒关键点可改善注册和姿态估计。
- **计算摄影和视频分析**：可用于模糊视频中的跨帧匹配、目标跟踪和几何分析，而不必先对每帧进行完整去模糊。
- **工业检测与医学/科学成像**：在采集过程中存在平台运动或曝光期间位移时，稳定的局部特征可能有助于图像配准。

## 5. 根据摘要可以推断的局限性

由于目前只有摘要，以下限制尚需通过正文和实验确认：

1. **对真实运动模糊分布的依赖**  
   第二阶段使用真实清晰—模糊图像对。如果训练数据中的相机运动、物体运动、曝光时间和场景类型不够丰富，模型可能对未见过的模糊轨迹、非均匀模糊或复杂动态场景泛化不足。

2. **清晰—模糊对应数据的获取成本**  
   虽然方法不需要人工关键点标签，但真实清晰—模糊图像对本身可能较难采集和严格对齐。若这些数据需要专门硬件、同步拍摄或额外标定，训练数据构建仍可能具有较高成本。

3. **对剧烈模糊和信息不可逆丢失的能力有限**  
   当运动模糊导致局部纹理完全消失时，任何检测器都无法恢复原本不存在的信息。SSMB 可能提升可重复性，但不一定能在极端模糊、严重遮挡或低信噪比条件下保持足够的关键点数量和定位精度。

4. **对非运动型退化的适用性未必成立**  
   摘要主要针对运动模糊。散焦模糊、滚动快门畸变、低照度噪声、压缩失真和雨雾等退化可能具有不同的统计特性，不能直接假设该方法同样有效。

5. **检测性能与描述子性能可能存在耦合问题**  
   论文重点是关键点检测，但实际匹配效果还取决于局部描述子、匹配器和几何验证模块。摘要中声称下游任务全面提升，但需要进一步确认收益究竟来自检测器本身、训练协议，还是与特定描述子和匹配管线的联合设计。

6. **空间覆盖约束可能带来精度—数量权衡**  
   强制关键点覆盖整个图像有助于几何估计，但在纹理稀疏或重复纹理区域，可能产生较低判别性的点。覆盖性约束如何影响误匹配率、关键点数量和定位精度，需要消融实验验证。

7. **计算开销和实时性尚不明确**  
   去掉去模糊模块通常有利于降低系统复杂度，但 LDE 和多阶段训练并不直接意味着推理速度更快。模型规模、硬件需求以及与现有稀疏检测器的实时性能仍需实验数据支持。

总体而言，SSMB 的有趣之处在于它没有把运动模糊简单视为需要消除的图像缺陷，而是将问题重新表述为：**哪些局部位置在清晰与模糊成像之间仍具有几何可重复性？** 这种面向跨退化域稳定性的自监督学习思路，对鲁棒局部特征和视觉几何系统具有较强的潜在价值。

**Key Findings:**

- We present SSMB, a deblur-free, self-supervised keypoint detector for motion-blurred images that requires neither handcrafted detectors nor external pseudo-labels.
- Extensive evaluations on keypoint detection, image matching, relative pose estimation, and visual localization under motion blur demonstrate that SSMB establishes a new state-of-the-art among sparse keypoint detectors, consistently outperforming both supervised and self-supervised baselines across all tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27181v1)
- [arXiv](https://arxiv.org/abs/2608.27181v1)

---

<a id='2608.27073v1'></a>
## [SpatialCrafter: Single Image World Modeling with Generative 3D Proxies](https://arxiv.org/abs/2608.27073v1)

**Authors:** Chuan Fang, Lingteng Qiu, Yixun Liang, Rui Chen, Kunming Luo, Zhaohua Zheng, Tongyuan Bai, Feipeng Tian, Zilong Dong, Zihan Zhou, Ping Tan

**Published:** 2026-08-27

**Categories:** cs.CV, cs.RO

**Abstract:**

Explorable image-to-scene generation is essential for applications in gaming, robotics, and virtual reality. Existing methods based on video diffusion model (VDM) commonly rely on incomplete conditioning signals such as sparse point clouds or 2D panoramas, leading to stochastic hallucinations, long-term drifts and suboptimal 3D consistency. We present SpatialCrafter, a novel two-stage framework that addresses these issues by introducing a global 3D proxy for high-fidelity image-to-scene generation. Specifically, we decompose the generation process into global proxy generation and appearance refinement. For proxy generation, we propose a Point-anchored Sparse Structure~(PaSS) Flow module that predicts a spatially aligned and geometrically consistent 3D proxy. For appearance refinement, we re-frame the VDM as a Generative Deferred Refiner which synthesizes high-frequency photorealistic details upon proxy-defined scene geometry. To better integrate the proxy with the pre-trained VDM, we introduce Parallel Geometry Injection and Proxy-Aware Corruption training strategies, which improve robustness to proxy artifacts without disrupting the pretrained generative manifold. Furthermore, as no suitable dataset exists for this explorable scene generation task, we construct a new large-scale dataset of 115K scenes. To the best of our knowledge, it is the first hybrid dataset for image-to-scene generation. Extensive experiments on both synthetic and real-world datasets show that SpatialCrafter outperforms state-of-the-art methods, mitigates long-term drift, and remains robust and consistent under rapid camera motion and extreme viewpoint changes. Code, models, and the newly constructed dataset will be publicly released. See more at https://fangchuan.github.io/SpatialCrafter/.

**Analysis:**

## 1. 摘要翻译
作者提出 **SpatialCrafter**，一种从单幅图像生成可探索三维场景的方法。框架分为两阶段：首先利用点锚定稀疏结构（PaSS）流模型生成与输入图像空间对齐、具有全局结构的三维代理；随后将视频扩散模型重构为“生成式延迟细化器”，在代理提供的几何约束上补充高频纹理并生成逼真的 RGB-D 视频。作者还提出并行几何注入和代理感知扰动，以提升模型对代理伪影的鲁棒性，并构建约11.5万个场景的数据集。实验表明，该方法在大幅相机运动和极端视角变化下具有更好的视觉质量与三维一致性。

## 2. 方法动机分析
**驱动力：**单图像探索式生成要求模型不仅“生成下一帧”，还要预先建立可支持任意视角查询的全局空间。  
**现有痛点：**2D历史帧记忆会累积误差；静态点云只覆盖输入视野；增量点云又依赖视频模型自身的空间感知，导致漂移、闭环失败和随机幻觉。全景图虽然扩大了覆盖范围，却缺乏真实视差。  
**核心假设：**若先从单图像生成一个全局且与输入严格对齐的3D代理，再让视频扩散模型只负责外观补全，就能把“空间建模”和“图像生成”解耦，从根本上减少长期漂移。

## 3. 方法设计详解
### Pipeline
1. 输入参考图像 \(I_0\)、相机轨迹和单目深度。将图像反投影为局部点云，并依据初始相机位姿变换、体素化，得到可见结构锚点 \(V_{cond}\)。  
2. **PaSS Flow：**在稀疏体素扩散/流匹配过程中，把 \(V_{cond}\) 与噪声体素在通道维拼接，条件生成剩余不可见结构。这样生成不是无约束猜测，而是“以输入可见几何为锚点的场景补全”。  
3. 对生成的稀疏结构使用 SLAT Flow 预测每个活跃体素的潜特征，再由3D Gaussian Splatting解码器生成全局3D代理。  
4. 沿用户指定轨迹渲染代理，得到连续的粗糙 RGB-D视频 \(c_{geo}\)。  
5. **并行几何注入：**分别编码粗RGB和深度为 \(z_{rgb},z_{depth}\)，沿宽度拼接成几何潜变量；再与噪声视频潜变量按通道拼接，并将参考图像潜变量按token拼接输入Video DiT。模型冻结预训练权重，仅训练LoRA，避免破坏原有视频生成能力。  
6. **代理感知扰动（PAC）：**训练时随机对粗条件施加高斯模糊、块状噪声或局部深度擦除，使细化器学会区分可靠几何与代理伪影，并主动修补孔洞、漂浮物和低频纹理。最终解码生成时空一致的RGB-D视频。

公式中的流匹配损失本质上是：在不同噪声时间点，训练网络预测“从噪声走向真实结构/视频”的方向；PaSS额外加入锚点，Video DiT额外接收几何潜变量。

## 4. 对比、创新与适用场景
本质区别在于：基线通常先重建**不完整代理**，再边生成边更新；SpatialCrafter先一次性生成**全局生成式代理**，再进行外观细化。创新包括PaSS的视图坐标对齐、几何与参考图像的双路注入、针对推理代理缺陷的PAC，以及混合场景数据引擎。适合室内导航、VR漫游、机器人仿真和可控新视角视频；不适合要求真实测量级几何或动态物体精确建模的任务。

## 5. 实验分析
作者在合成SpatialGen-Video、RealEstate10K和DL3DV上，与2D记忆及点云代理方法比较，并进行PaSS、RGB-D和PAC消融。代表性结论是：SpatialCrafter在三类数据上整体取得最佳或领先结果；去除PaSS会明显造成输入视图与代理错位，而加入PAC能有效修复粗代理中的黑洞和伪影。  
优势是长程一致性强、支持大位移相机、同时输出RGB与深度。局限是依赖单目深度、相机位姿和大规模高质量训练数据；“全局代理”仍包含生成性幻觉，且两阶段3D生成与视频扩散计算成本较高。

## 6. 实用指南
论文声称将开源代码、模型和数据，当前给出项目主页。复现需准备三类数据，完成点云/深度估计、SLAT与3D Gaussian构建，再分别训练两阶段模型。关键设置包括：代理阶段约2万步；细化阶段先训练256²、再训练512²，视频长度81帧；冻结Wan2.1类Video DiT，仅更新LoRA。迁移到其他任务时，可将代理替换为网格、NeRF或3DGS，并保留“几何条件+扩散细化”的解耦思想。

## 7. 总结
**核心思想：先生成全局代理，再细化视频。**

**速记版Pipeline：**
1. 单图像估计深度并提取可见三维锚点；  
2. 以锚点为约束补全全局3D场景；  
3. 沿相机轨迹渲染粗RGB-D序列；  
4. 视频模型结合几何和参考图像补全真实外观；  
5. 用扰动训练提升对代理缺陷的修复能力。

**Key Findings:**

- We present SpatialCrafter, a novel two-stage framework that addresses these issues by introducing a global 3D proxy for high-fidelity image-to-scene generation.
- For proxy generation, we propose a Point-anchored Sparse Structure~(PaSS) Flow module that predicts a spatially aligned and geometrically consistent 3D proxy.
- To better integrate the proxy with the pre-trained VDM, we introduce Parallel Geometry Injection and Proxy-Aware Corruption training strategies, which improve robustness to proxy artifacts without disrupting the pretrained generative manifold.
- Furthermore, as no suitable dataset exists for this explorable scene generation task, we construct a new large-scale dataset of 115K scenes.
- Extensive experiments on both synthetic and real-world datasets show that SpatialCrafter outperforms state-of-the-art methods, mitigates long-term drift, and remains robust and consistent under rapid camera motion and extreme viewpoint changes.
- Code, models, and the newly constructed dataset will be publicly released.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27073v1)
- [arXiv](https://arxiv.org/abs/2608.27073v1)

---

<a id='2608.27033v1'></a>
## [Riemann-1.0: An Embodied World Action Model for Physical AI](https://arxiv.org/abs/2608.27033v1)

**Authors:** Haofeng Sun, Jiangbo Pei, Fei Kang, Zexiang Liu, Yaokun Li, Boyi Jiang, Hua Xue, Cindy Zhou, Wei Li, Yichen Wei, Mengyin An, Fanliang Zhao, Biao Jiang, Zile Wang, Yang Liu, Yangguang Li

**Published:** 2026-08-27

**Categories:** cs.RO

**Abstract:**

We introduce Riemann-1.0, a fully causal autoregressive World Action Model for embodied intelligence. Riemann-1.0 jointly models multi-view visual observations, robot states, and embodiment-specific actions within a unified causal autoregressive sequence, representing robot actions and world evolution as causal state transitions. Unlike existing WAMs based on joint generation, video-first prediction, or decoupled modeling paradigms, Riemann-1.0 unifies online robot policy execution and action-conditioned world simulation within a single model, enabling it to function as both an executable robot policy and a multi-embodiment visual world simulator. To scale embodied experience across heterogeneous data sources, we further develop a progressive embodied pretraining framework that unifies learning from egocentric human videos, handheld-gripper demonstrations, and heterogeneous robot trajectories under a shared World Action Modeling objective. Built upon 200K+ hours of interaction data, Riemann-1.0 progressively transfers large-scale embodied experience into executable robot manipulation capabilities. Riemann-1.0 achieves state-of-the-art performance across both simulation benchmarks and real-world manipulation tasks. It achieves success rates of 94.3% on RoboTwin2.0, 99.0% on LIBERO, and 62.6% on the long-horizon compositional benchmark RoboCasa-365, outperforming the previous best method by 8.4% On long-horizon real-world manipulation tasks, Riemann-1.0 achieves a Success Rate (SR) of 85.0% and a Progress Success Rate (PSR) of 94.4%, exceeding the strongest open-source baseline by 15% in SR. These results demonstrate that unified World Action Modeling together with progressive embodied pretraining effectively transforms large-scale embodied experience into generalizable robot manipulation capabilities.

**Analysis:**

## 1. 摘要翻译

本文提出 **Riemann-1.0**，一种面向具身智能的全因果自回归世界动作模型（WAM）。模型在统一的因果序列中联合建模多视角视觉观测、机器人状态与特定于本体的动作，将动作及世界演化表示为因果状态转移。与联合生成、视频优先和动作—视频解耦方法不同，Riemann-1.0同时支持在线机器人策略执行与动作条件视觉世界模拟。作者进一步提出渐进式具身预训练，统一利用第一视角人类视频、手持夹爪演示和异构机器人轨迹，并基于20万小时以上交互数据逐步学习可执行机器人操作。在RoboTwin2.0、LIBERO、RoboCasa365及真实机器人任务上取得较强结果。

## 2. 方法动机

**驱动力：**机器人轨迹规模有限，而人类视频丰富但缺少可执行动作；同时，策略模型通常只预测动作，世界模型又常与策略分离。作者希望用统一模型连接“动作生成”和“动作后果预测”。

**痛点：**异构数据在视角、状态维度、坐标系、动作语义和监督强度上差异显著；已有WAM或联合生成带来模态耦合困难，视频优先方法推理延迟高，解耦方法破坏动作与视觉后果的直接因果关系。

**核心假设：**若按照真实交互顺序建模“历史观测→动作→后续观测”，并按监督强度逐步从人类视频过渡到机器人轨迹，则大规模弱监督经验能够迁移为可执行控制能力。

## 3. 方法设计详解

### （1）数据基础设施

数据分为：第一视角人类视频、UMI/外骨骼等手持设备演示、异构机器人轨迹。统一数据引擎依次进行视觉校正、VLM任务/动作分段与标注、质量筛选、3D手部重建、几何过滤和语义均衡采样。人类视频利用MANO重建手部轨迹，并结合相机位姿估计获得连续动作；机器人数据则进行时间对齐、动作校准、异常轨迹与静止片段剔除。最终统一为“任务文本—本体ID—多视角视频—状态—动作”的轨迹格式，并通过掩码处理不同本体的维度差异。

### （2）全因果Action-Video模型

其核心分解为：

\[
p(a_{1:T},z_{1:T})=\prod_t p(a_t|z_{<t},s_{<t},a_{<t},c)
p(z_t|z_{<t},s_{<t},a_{\le t},c)
\]

即先根据历史预测动作，再用当前动作预测视觉潜变量。真实部署时，预测动作执行后用环境返回的真实图像替代生成图像；模拟时则递归使用预测图像。

输入图像经Wan VAE和3D patch embedding转为视觉token；状态和动作经本体专属投影、归一化及padding后输入共享Transformer。本体ID选择专属动作/状态接口和动作输出头，从而共享视觉与时序推理，同时保留不同机器人的控制语义。模型采用结构化因果注意力，禁止目标token读取未来观测，避免教师强制训练中的信息泄漏。

视觉潜变量和动作均采用flow matching去噪，损失为：

\[
\mathcal L=(1-\lambda)\mathcal L_z+\lambda\mathcal L_a
\]

分别约束视觉动态和动作预测；不同本体使用独立有效性掩码。

### （3）渐进式预训练

1. **LAM-Action Bootstrap：**冻结Latent Action Model，从相邻人类视频帧提取32维潜在动作，作为伪动作，\(\lambda=0.1\)，主要学习视觉动态。  
2. **Trajectory-Grounded Alignment：**混合UMI、机器人轨迹和3D手部数据，使用真实动作进行跨本体对齐，\(\lambda=0.5\)。  
3. **Robot-Policy Enhancement：**仅用高质量机器人数据强化可执行策略，\(\lambda=0.9\)；下游微调进一步提高至0.95。

## 4. 对比与创新

本质区别在于：动作位于视觉后果之前，且策略和模拟器共享同一因果接口；创新主要是“全因果动作—视频建模”和“伪动作→真实轨迹→机器人专训”的监督迁移。适合多机器人操作、长时序任务、动作规划评估和数据驱动视觉模拟。

## 5. 实验分析

作者在真实双臂机器人、RoboCasa365、RoboTwin2.0和LIBERO上验证，报告真实任务平均SR/PSR为85.0%/94.4%，并在三项仿真基准取得领先。优势是统一策略与世界模型、跨本体共享、长时序闭环更自然。局限是核心数据、模型规模、训练成本及消融实验披露不足；伪动作质量和视频到机器人动作的真实可迁移性仍可能成为瓶颈。

## 6. 实用指南

论文未明确说明代码、完整数据或训练配置已开源，仅提供项目网站。复现需重点实现：多源数据标准化、本体专属接口、因果注意力掩码、LAM伪动作生成、分阶段调整\(\lambda\)，并严格处理padding噪声和有效性掩码。迁移到新机器人时，应新增动作/状态 canonical mapping、归一化统计、专属输出头，再用少量高质量轨迹进行第三阶段微调。

## 7. 总结

**核心思想：**用因果动作模型统一策略与世界模拟。

**速记版pipeline：**
1. 清洗并统一人类、手持设备和机器人数据；  
2. 从人类视频提取潜在伪动作；  
3. 按“视频预训练—真实轨迹对齐—机器人强化”训练；  
4. 先预测动作，再预测动作造成的视觉变化；  
5. 用真实观测闭环执行，或用生成观测进行模拟。

**Key Findings:**

- We introduce Riemann-1.0, a fully causal autoregressive World Action Model for embodied intelligence.
- Riemann-1.0 achieves state-of-the-art performance across both simulation benchmarks and real-world manipulation tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.27033v1)
- [arXiv](https://arxiv.org/abs/2608.27033v1)

---

<a id='2608.26965v1'></a>
## [ClusterAttention: A training-free speedup of bidirectional attention](https://arxiv.org/abs/2608.26965v1)

**Authors:** Kasper Nordenram, Amelie Dittmann

**Published:** 2026-08-27

**Categories:** cs.LG, cs.CV

**Abstract:**

This paper introduces ClusterAttention, a general training-free speedup of bidirectional attention layers. Existing sparse attention methods either rely on structure in the input, such as order in language or spatial proximity in images, or use slow clustering processes amortized over several forward passes. ClusterAttention instead uses a fast recursive clustering method that adapts to the geometry of the keys and queries in each attention head to produce useful clusters. This method allows setting the size of the clusters arbitrarily. We utilize this by setting all clusters to be a fixed size that is a power of two, allowing the block-sparse attention to run at the same latency per query-key interaction as dense attention on GPUs. We also derive an expression for the output error in sparse attention, that explains the counterintuitive experimental finding that tight clusters can lead to larger errors than random clusters. We then derive the error when excluded clusters are compensated through their centroids, and show that this error shrinks with tighter clusters. We integrate this compensation into the method.   On large-scale tabular data ClusterAttention speeds up TabPFN-3 arXiv:2605.13986 by two to six times, while retaining at least 99% of the dense accuracy. To our knowledge, it is the first training-free method that can be successfully applied in the setting of unstructured input and a single forward pass. For video generation with Wan 2.1-14B T2V arXiv:2503.20314 , ClusterAttention achieves output closer to dense attention and a larger speedup (1.8x versus 1.4x) compared to SVOO arXiv:2603.18636 , a leading method developed specifically for this domain, both run without offline calibration.

**Analysis:**

# 1. 摘要翻译

本文提出 **ClusterAttention**，一种适用于双向注意力层的、无需训练的加速方法。现有稀疏注意力通常依赖输入结构（如语言顺序、图像空间邻近性），或依赖可在多次前向传播中摊销的慢速聚类。ClusterAttention采用快速递归聚类，根据每个注意力头中键和查询的几何结构动态生成簇，并可任意设定簇大小。作者将簇大小设为固定的二次幂，使块稀疏注意力在GPU上具有接近稠密注意力的单次交互延迟。

论文进一步推导了稀疏注意力的输出误差，解释了为何紧密聚类有时反而比随机聚类产生更大误差；同时提出用未选中簇的键、值质心进行补偿，使误差随簇紧密度下降。该补偿称为条纹均值补偿（SMC）。在TabPFN-3上，方法获得约2–6倍加速，同时保留至少99%的稠密精度；在Wan 2.1视频生成模型上，相比SVOO获得更高加速和更接近稠密结果的输出。

# 2. 方法动机分析

**驱动力：** 双向注意力的计算和显存访问随token数平方增长，而许多query-key交互贡献很小。作者希望在不训练、不依赖输入顺序或空间结构、且只进行一次前向推理的情况下完成稀疏化。

**现有痛点：** 结构化方法难以迁移到表格、无序集合等数据；K-Means或迭代式聚类太慢；不规则簇会降低GPU块计算效率；仅保留高分簇虽然能提高注意力质量，却可能造成较大输出偏差。

**核心假设：** 注意力相关性不应在原始欧氏空间中衡量，而应在“下游交互真正感知的空间”中聚类；稀疏误差不仅取决于漏掉多少注意力质量，也取决于保留区域与排除区域的值均值差异。

# 3. 方法设计详解

## 3.1 总体流程

对每个注意力头分别执行：

1. **键、查询独立聚类。**  
   将键和查询划分为固定大小的簇，默认键簇128、查询簇64；不足部分补齐并加mask。

2. **构造任务感知空间。**  
   键不直接按自身欧氏距离聚类，而是利用当前query矩阵 \(Q\) 构造  
   \[
   M_q=Q^TQ/n
   \]
   并分解 \(M_q=R_q^TR_q\)，再用 \(R_qk\) 表示键。这样，若两个键对所有query产生相近logit，它们就会接近。  
   查询则使用中心化键矩阵 \(K_c\)，构造 \(M_k=K_c^TK_c/n\)，以消除softmax的平移不变性；在top-k路由中，还对 \(R_kq\) 归一化，使表示更关注键的相对排序。

3. **递归切分。**  
   对变换后的向量做对角化递归切分：先对全体数据做SVD/主成分分解，在每个节点中估计各主成分方向的方差，沿方差最大的方向切分，并尽量使两侧token数接近且为簇大小的整数倍。重复直到形成目标簇。其复杂度约为 \(O(nd^2+d^3)\)。

4. **簇级路由。**  
   计算query簇质心与所有key簇质心的点积，作为簇间平均logit；也可加入方差修正。随后选择top-k个key簇，或根据估计注意力质量自适应选择簇。作者实际更偏好top-k。

5. **块稀疏注意力。**  
   只对被选中的query簇-key簇块执行细粒度注意力，使用适配GPU tile的块稀疏kernel。

6. **SMC补偿。**  
   对未选中的key簇，不完全丢弃其贡献，而是用簇的键质心和值质心参与一次近似注意力。误差可分解为Jensen项和键值协方差项：簇越紧，质心近似越准确。该机制尤其改善top-k路由下的输出质量。

## 3.2 关键新视角

传统分析只关注“遗漏注意力质量” \(1-w_S\)。论文指出：
\[
o-o_S=(1-w_S)(o_{\bar S}-o_S)
\]
因此，即使保留了较多注意力质量，只要被选中和未选中区域的值均值差异很大，误差仍会显著。随机簇有时表现较好，正是因为它们使两部分值分布更接近。SMC则通过质心补偿，直接降低聚类带来的均值偏差。

# 4. 方法对比与适用性

与SpargeAttn依赖空间/顺序结构、SVOO依赖离线稀疏性分析不同，ClusterAttention在当前前向过程中动态聚类，适合无结构输入和单次推理。其创新主要包括：**交互感知的键/查询变换、固定大小的快速递归聚类、稀疏误差新分解、质心补偿机制**。

适合高token数、双向注意力、GPU块计算场景，如高分辨率视觉、视频扩散、表格基础模型和集合建模；低token数时，聚类、特征分解和补偿开销可能抵消收益。

# 5. 实验分析

作者在DINOv2-L、TabPFN-3和Wan 2.1-14B T2V上测试，并与稠密注意力、SpargeAttn、SVOO及随机聚类比较。代表性结论是：

- TabPFN-3上约2–6倍加速，仍保持超过99%的相对精度。
- 视频生成中相较SVOO，速度约1.8倍而非1.4倍，且PSNR、SSIM、LPIPS整体更接近稠密结果。

优势是无需训练、适应无结构输入、簇大小和稀疏率可控。局限是特征分解及SMC开销较高；当前实现依赖特定稀疏kernel，簇大小受限；实验规模较小，尚未充分验证训练场景和更多模型。

# 6. 实用指南

论文声明代码将发布于 `github.com/SpoketKasper/ClusterAttention`。复现时需重点确认：GPU块大小（键128、查询64）、top-k比例、是否启用SMC、是否使用键/查询变换，以及输入布局转置开销。固定长度任务可用CUDA Graph减少常数开销；视频场景可进一步缓存簇以摊销聚类成本。迁移到GQA/MQA时，需要决定共享键头之间是平均协方差矩阵，还是分别聚类。

# 7. 总结

**核心思想：在注意力感知空间聚类并用质心补偿稀疏误差。**

**速记版Pipeline：**

1. 按当前注意力关系重新表示键和查询；  
2. 快速递归切成固定大小的小组；  
3. 用小组中心挑选最相关的小组；  
4. 仅计算选中小组之间的细粒度注意力；  
5. 用未选小组的中心值补回被省略的信息。

**Key Findings:**

- To our knowledge, it is the first training-free method that can be successfully applied in the setting of unstructured input and a single forward pass.
- For video generation with Wan 2.1-14B T2V arXiv:2503.20314 , ClusterAttention achieves output closer to dense attention and a larger speedup (1.8x versus 1.4x) compared to SVOO arXiv:2603.18636 , a leading method developed specifically for this domain, both run without offline calibration.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.26965v1)
- [arXiv](https://arxiv.org/abs/2608.26965v1)

---

<a id='2608.26956v1'></a>
## [RubricRM: Generative Reward Modeling via Dynamic Rubrics for Image Generation and Editing](https://arxiv.org/abs/2608.26956v1)

**Authors:** Zijian Kan, Wei Wang, Long Luo, Bing Zhao, Xuan Ren, Weixu Qiao, Wenbo Li, Hu Wei, Lin Qu

**Published:** 2026-08-27

**Categories:** cs.CV

**Abstract:**

Reward models play an essential role in aligning visual generative models, yet most existing visual reward models use a single scalar score or rely on fixed criteria that cannot adapt to different instructions. This limits both interpretability and task sensitivity, especially for text-to-image generation and instruction-based image editing, where different inputs require different evaluation dimensions. We propose RubricRM, a pairwise generative reward modeling framework that first produces an input-specific rubric with evaluation dimensions, weights, and scoring criteria, and then applies the rubric to score candidate images. We train dedicated RubricRM models for text-to-image generation and image editing using a two-stage training pipeline: supervised fine-tuning teaches the model the rubric-based scoring paradigm, while GRPO further improves scoring through fine-grained dimension-level rewards. Experiments on multiple generation and editing benchmarks show that RubricRM outperforms existing specialized reward models and remains competitive with strong proprietary MLLM judges despite using smaller backbones. Our models, data, and code are available at https://github.com/zijiankan/RubricRM.

**Analysis:**

# 1. 摘要翻译

奖励模型在视觉生成模型对齐中十分重要，但现有视觉奖励模型大多输出单一标量，或依赖固定评价标准，难以适应不同指令。这使其在文生图和指令驱动图像编辑中缺乏可解释性与任务敏感性。本文提出 **RubricRM**：一种成对生成式奖励建模框架。模型首先针对输入生成动态评价量规，包括评价维度、权重和分级评分标准；随后依据该量规对候选图像逐维评分并作出偏好判断。训练采用两阶段流程：监督微调学习“生成量规—依据量规评分”的模式，GRPO则通过细粒度维度奖励进一步校准评分。实验表明，RubricRM在多个文生图和图像编辑基准上优于现有专用奖励模型，并以更小模型规模接近强大的闭源多模态模型评审器。

# 2. 方法动机分析

**驱动力与痛点：** CLIP类指标偏重整体图文相似度；ImageReward、HPS、PickScore等偏好模型通常只给标量，无法解释“为什么选A”。生成式评审模型虽能输出理由，但往往使用预定义维度，无法针对“现实感”“文字准确性”“主体保真”“编辑后内容保持”等不同任务动态调整评价重点。

**核心假设：** 偏好判断不应直接从指令映射到一个分数，而应先生成一个输入相关的评价协议；若模型明确“评什么、各项多重要、不同分数代表什么”，其判断将更准确、更可审查。

# 3. 方法设计详解

## 3.1 推理Pipeline

输入为指令 \(q\)、候选图像 \(I_A,I_B\)；编辑任务还包含源图像。模型在一次多模态生成过程中完成：

1. **意图解析：**提取主体、属性、布局、风格、编辑要求及需保留内容。
2. **动态量规生成：**选择3–5个原子维度 \(d_i\)，为每维分配权重 \(w_i\)，并保证权重和为1；同时生成该维度的0–4分级标准 \(L_i\)。例如编辑任务可能生成“指令遵循、源图保持、视觉融合、伪影控制”。
3. **逐维比较：**分别给A、B在每个维度打分 \(s_i^A,s_i^B\)，并给出对应证据。
4. **加权聚合：**计算
\[
S(I_k)=\sum_i w_i s_i^k
\]
总分较高者为最终偏好。

量规不是固定模板，而是连接“输入意图”和“偏好决策”的中间评价协议，因此同时提供了任务适应性与可解释轨迹。

## 3.2 数据与训练

作者整合文生图和编辑偏好数据，并对指令进行多标签分类和分层采样，补足文字渲染、逻辑推理、风格编辑等长尾类型。使用教师模型结合原始人工偏好标签合成完整轨迹：意图分析、维度与权重、评分标准、A/B逐维评分。随后过滤格式错误、分数越界、权重不归一或最终偏好与人工标签冲突的样本。

**阶段一：量规轨迹SFT。** 直接最大化教师轨迹的似然，使模型学会完整的“先拆解评价标准，再评分决策”范式。标签不仅教模型选谁，轨迹还教模型如何评价。

**阶段二：固定量规GRPO。** 训练时固定教师生成的量规，只让模型 rollout 后续逐维评分，避免不同rollout生成不同维度而无法比较。对每个维度定义参考分差与预测分差：
\[
\Delta_i^{gt}=s_i^{A,gt}-s_i^{B,gt},\quad
\Delta_i^{pred}=s_i^{A,pred}-s_i^{B,pred}.
\]
幅度奖励
\[
b_i=1-\frac{|\Delta_i^{pred}-\Delta_i^{gt}|}{2(s_{\max}-s_{\min})}
\]
衡量分差是否接近参考值；方向因子 \(\phi_i\) 在方向一致、平局错配、方向反转时分别取1、0.6、0.1。最终奖励为
\[
R=\sum_i w_i b_i\phi_i.
\]
这意味着模型不仅要“选对”，还要在每个维度上判断对错方向及差距大小。若一个GRPO组内奖励标准差低于0.05，则将优势置零，防止近乎相同的rollout因微小噪声产生剧烈更新。

# 4. 方法对比与创新

本质区别在于：传统方法是“固定标准/隐式特征→标量”，RubricRM是“输入→动态评价协议→逐维证据→加权偏好”。创新主要包括：动态生成维度、权重和分级标准；将偏好监督转化为结构化评价轨迹；用维度级分差奖励替代稀疏的最终标签奖励；采用固定量规和饱和组过滤提升GRPO稳定性。最适合文生图、图像编辑及需要解释性和任务定制评价的视觉对齐场景。

# 5. 实验分析

作者在MMRB2、GenAI-Bench、GenAI-Bench-Verified及多个编辑基准上比较专用奖励模型和MLLM评审器。代表性结论是：RubricRM-Gen-9B在三个文生图基准上均领先其他奖励模型；RubricRM-Edit-9B在编辑任务的总体指标上同样领先。消融显示，量规轨迹SFT贡献了主要性能提升，维度级GRPO带来稳定的额外增益。

优势是任务自适应、可解释、能细化监督；局限是依赖专有教师模型，可能继承其偏差，且当前仅支持静态图像。

# 6. 实用指南与迁移

论文声称开源代码、模型和数据。复现需准备偏好图像对，构建任务分类和均衡采样，调用教师生成并过滤量规轨迹；使用Qwen3.5-4B/9B，SFT学习率 \(5\times10^{-6}\)、2个epoch，GRPO学习率 \(5\times10^{-7}\)、组大小8、KL系数0.05。迁移到视频、3D或多模态回答评价时，只需重新定义任务量规、教师轨迹和可验证的维度级奖励，但需增加时间一致性、跨帧稳定性等维度。

# 7. 总结

**核心思想：** 先生成量规，再依据量规评图。

**速记版Pipeline：**
1. 解析指令，找出真正重要的评价点。  
2. 为评价点分配重要程度，并规定各分数含义。  
3. 分别检查两张图在每个评价点上的表现。  
4. 按重要程度汇总分数，选择更优图像。  
5. 用逐项分差训练模型，减少只看最终标签造成的误判。

**Key Findings:**

- We propose RubricRM, a pairwise generative reward modeling framework that first produces an input-specific rubric with evaluation dimensions, weights, and scoring criteria, and then applies the rubric to score candidate images.
- Experiments on multiple generation and editing benchmarks show that RubricRM outperforms existing specialized reward models and remains competitive with strong proprietary MLLM judges despite using smaller backbones.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.26956v1)
- [arXiv](https://arxiv.org/abs/2608.26956v1)

---

