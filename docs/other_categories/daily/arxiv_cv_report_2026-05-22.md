time: 20260522

# Arxiv Computer Vision Papers - 2026-05-22

## Executive Summary

## 每日报告执行摘要（2026-05-21）

本日10篇论文主要围绕**多模态视觉-语言-动作（VLA）模型、自动驾驶与机器人中的跨域/跨实体泛化、模型效率与可解释推理**三大主题展开。整体趋势显示出从单一视觉任务向**具身智能与交互理解**的明确迁移，同时**扩散模型、稀疏自编码器、反事实数据选择**等工具被引入以解决组合泛化与分辨率外推等核心瓶颈。

### 显著创新论文

- **GesVLA (第1篇)**：首次将**手势**显式嵌入VLA模型，为机器人或交互式AI提供更自然的多模态控制接口，是具身智能中“手势理解”方向的实质性推进。
- **Sensor2Sensor (第2篇)**：提出**跨实体传感器转换**框架，使不同传感器配置的自动驾驶车辆能共享感知特征，极大降低部署成本，对现实落地意义重大。
- **Slimmable ConvNeXt (第4篇)**：设计**宽度自适应推理**机制，模型可在推理时动态调整通道数，实现单模型多设备无缝部署，兼顾精度与效率，实用价值高。
- **SEGA (第6篇)**：提出**谱能量引导注意力**，解决扩散Transformer在推理时生成超高分辨率图像的外推问题，在不停机训练下大幅提升分辨率性能。
- **SegCompass (第7篇)**：将**稀疏自编码器**用于推理分割的可解释对齐，首次使多模态分割的中间推理路径可被人类理解，代表可解释性研究的新范式。

### 新兴研究方向与技术

1. **VLA模型的行为表示学习**（第1、5篇）：从“抽象意图”到“实例化动作”表示，结合手势或行为建模，正成为机器人学习的主流范式。
2. **反事实数据选择与组合泛化**（第8篇）：通过短语级反事实干预筛选预训练数据，提升视觉语言模型对未见组合的泛化能力，方法简洁有效。
3. **对称性在机器人中的跨空间组合**（第9篇）：系统研究任务对称性的组合规则，为利用物理对称性加速强化学习提供理论基础。
4. **视频主动合成驱动的时空推理基准**（第10篇）：通过可控视频生成创建对抗性测试集，评估模型对时序因果与空间关系的理解，推动评测从静态转向动态。

### 建议优先全文阅读的论文

- **GesVLA**（第1篇）：对于从事具身智能、人机交互或VLA研究的读者，手势维度的引入可能打开新方向。
- **Sensor2Sensor**（第2篇）：自动驾驶从业者需关注其跨实体泛化思路，对多模态传感器融合有启发。
- **SEGA**（第6篇）：从事扩散模型或高分辨率生成的读者应读，其谱能量引导思路简洁且有效。
- **VGenST-Bench**（第10篇）：基准建设者或时空推理研究者，该工作提供了可复现的动态评估框架。

总体而言，本日论文反映了计算机视觉正快速与机器人学、自动驾驶、交互式AI融合，**多模态表示学习、跨域泛化、可解释推理**成为最活跃的增长点。

---

## Table of Contents

1. [GesVLA: Gesture-Aware Vision-Language-Action Model Embedded Representations](#2605.22812v1)
2. [Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving](#2605.22809v1)
3. [Cross-Domain Human Action Recognition from Multiview Motion and Textual Descriptions](#2605.22697v1)
4. [Slimmable ConvNeXt: Width-Adaptive Inference for Efficient Multi-Device Deployment](#2605.22677v1)
5. [From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](#2605.22671v1)
6. [SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers](#2605.22668v1)
7. [SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation](#2605.22658v1)
8. [What Does the Caption Really Say? Counterfactual Phrase Intervention for Compositional Data Selection in Vision-Language Pretraining](#2605.22651v1)
9. [Symmetries Here and There, Combined Everywhere: Cross-space Symmetry Compositions in Robotics](#2605.22639v1)
10. [VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis](#2605.22570v1)

---

## Papers

<a id='2605.22812v1'></a>
## [GesVLA: Gesture-Aware Vision-Language-Action Model Embedded Representations](https://arxiv.org/abs/2605.22812v1)

**Authors:** Wenxuan Guo, Ziyuan Li, Meng Zhang, Yichen Liu, Yimeng Dong, Chuxi Xu, Yunfei Wei, Ze Chen, Erjin Zhou, Jianjiang Feng

**Published:** 2026-05-21

**Categories:** cs.RO, cs.CV

**Abstract:**

Vision-Language-Action (VLA) models have shown strong potential for general-purpose robot manipulation by unifying perception and action. However, existing VLA systems primarily rely on textual instructions and struggle to resolve spatial ambiguity in complex scenes with multiple similar objects. To address this limitation, we introduce gesture as a parallel instruction modality and propose a Gesture-aware Vision-Language-Action model (GesVLA). Our approach encodes gesture features directly into the latent space, enabling them to participate in both high-level reasoning and low-level action generation, and adopts a dual-VLM architecture to achieve tight coupling between gesture representations and action policies. At the data level, we construct a scalable gesture data generation pipeline by rendering hand models onto real-world scene images. This reduces the sim-to-real visual gap while producing rich data with diverse motion patterns and corresponding pointing annotations. In addition, we employ a two-stage training strategy to equip the model with both gesture perception and action prediction capabilities. We evaluate our approach on multiple real-world robotic tasks, including a controlled block manipulation task for validation and more practical scenarios such as product and produce selection. Experimental results show that incorporating gesture consistently improves target grounding accuracy and human-robot interaction efficiency, especially in complex and cluttered environments. Project page: https://gwxuan.github.io/GesVLA/.

**Analysis:**

### 1. 摘要翻译
视觉-语言-动作 (VLA) 模型通过统一感知与动作，在机器人操作领域展现了巨大潜力。然而，现有 VLA 系统主要依赖文本指令，在复杂且存在多个相似物体的场景中难以解决空间模糊性。为解决此限制，我们引入手势作为一种并行指令模态，并提出了一种手势感知视觉-语言-动作模型 (GesVLA)。该方法直接将手势特征编码到潜在空间，使其参与高层推理与底层动作生成，并采用双 VLM 架构实现手势表示与动作策略的紧密耦合。在数据层面，我们构建了一个通过将手模渲染到真实世界场景图像上的可扩展数据生成流水线，减少了虚实差距并产生了包含动作模式与指向标注的丰富数据。此外，我们采用两阶段训练策略赋予模型手势感知与动作预测能力。在多个真实世界机器人任务上的实验表明，引入手势能显著提高目标定位准确性和人机交互效率，特别是在复杂拥挤的环境中。项目主页：https://gwxuan.github.io/GesVLA/。

### 2. 方法动机分析
*   **驱动力**：在复杂场景下，仅靠语言指令（如“拿走那个”）难以精确锚定物体，引入人类自然的指向性手势作为“视觉辅助线索”是提升机器人交互鲁棒性的关键。
*   **痛点**：现有方法将手势视为后处理信号（如先预测坐标再传入VLM），导致空间信息损失及多模块带来的误差累积；且缺乏大规模标注好的手势指令数据集。
*   **核心直觉**：将手势视为与视觉、语言同等地位的“第一类模态”，并在潜在特征空间而非文本空间进行跨模态深度融合。

### 3. 方法设计详解
*   **模型架构（双VLM）**：
    *   **VLM_int (意图推理)**：接收语言指令和手势序列，输出文本 reasoning 和视觉 prompt（即在图像上标注出指向位置的掩码或点）。
    *   **VLM_per (在线感知)**：接收场景图像、语言以及 VLM_int 的输出，通过 Cross-Attention 机制直接获取意图特征，生成动作条件。
    *   **Action Expert (Fθ)**：基于 flow-based 策略，通过迭代去噪生成动作轨迹。
*   **关键处理逻辑**：
    *   **手势编码**：提取手指及手腕的关键点（MediaPipe），选取动作“停顿”时刻的帧作为关键帧，将其投影到连续特征空间作为输入。
    *   **不对称交互**：VLM_int 计算一次并缓存 Key-Value 状态，供 VLM_per 在后续步骤中反复调用，从而实现高效推理。
*   **数据工程 (Data Engine)**：
    *   通过 GroundingDINO 在真实场景图中检测候选目标，结合相机内参将 2D 坐标反投为 3D 点，并以此生成具有“自然指向抖动”和“拟人化移动轨迹”的合成手势数据（16k样本）。

### 4. 方法对比分析
*   **本质区别**：从传统的“手势-转文本”或“手势-转坐标”的解耦方案，转向了“手势-转潜空间向量”的端到端深度融合。
*   **创新点**：构建了“半合成数据流水线+双VLM异步架构”，解决了数据短缺和推理时延问题。
*   **适用场景**： cluttered (杂乱) 桌面级操作，如超市分拣、积木拾取等高精度指向任务。

### 5. 实验分析
*   **关键结果**：在88个测试样本上，GesVLA 意图理解准确率达到 94.3%，显著优于基于几何规则（59.1%）和传统LLM（38.6%）的方法；机器人操作任务在复杂场景下的成功率对比基线有明显提升（如 Select Fruit 任务从 40% 提升至 80%）。
*   **优势**：极强的抗干扰能力，即便手势不够标准，也能通过语言先验进行纠偏。
*   **局限**：目前仅支持常见指向手势，未覆盖复杂的人机协同动作；严重依赖初始的手势关键点定位精度。

### 6. 实用指南
*   **开源**：项目主页已提供。
*   **实现要点**：两阶段训练至关重要——第一阶段预训练 VLM_int，第二阶段冻结 VLM_int 训练动作专家，这防止了合成数据干扰真实的动作分布。
*   **迁移建议**：若要迁移至新任务，只需通过该流水线生成对应环境的合成数据，无需重新采集真实机器人示范数据。

### 7. 总结
*   **核心思想**：通过连续潜特征空间实现手势意图与机器人策略的深度耦合。
*   **速记版Pipeline**：
    1. 生成：在真实场景上合成带精准标注的手势轨迹数据。
    2. 推理：VLM_int 将手势和语言转化为意图 tokens。
    3. 感知：VLM_per 通过 Cross-Attention 对齐意图与当前视觉。
    4. 动作：Action Expert 通过流匹配预测精准轨迹。

**Key Findings:**

- To address this limitation, we introduce gesture as a parallel instruction modality and propose a Gesture-aware Vision-Language-Action model (GesVLA).
- Our approach encodes gesture features directly into the latent space, enabling them to participate in both high-level reasoning and low-level action generation, and adopts a dual-VLM architecture to achieve tight coupling between gesture representations and action policies.
- We evaluate our approach on multiple real-world robotic tasks, including a controlled block manipulation task for validation and more practical scenarios such as product and produce selection.
- Experimental results show that incorporating gesture consistently improves target grounding accuracy and human-robot interaction efficiency, especially in complex and cluttered environments.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22812v1)
- [arXiv](https://arxiv.org/abs/2605.22812v1)

---

<a id='2605.22809v1'></a>
## [Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving](https://arxiv.org/abs/2605.22809v1)

**Authors:** Jiahao Wang, Bo Sun, Yijing Bai, Vincent Casser, Songyou Peng, Zehao Zhu, Meng-Li Shih, Xander Masotto, Shih-Yang Su, Kanaad V Parvate, Tiancheng Ge, Linn Bieske, Dragomir Anguelov, Mingxing Tan, Chiyu Max Jiang

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

Robust training and validation of Autonomous Driving Systems (ADS) require massive, diverse datasets. Proprietary data collected by Autonomous Vehicle (AV) fleets, while high-fidelity, are limited in scale, diversity of sensor configurations, as well as geographic and long-tail-behavioral coverage. In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments. However, this unstructured, in-the-wild video data is incompatible with ADS expecting structured, multi-modal sensor inputs for validation and training. To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds. A core challenge is the lack of paired training data. We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering. Sensor2Sensor then utilizes a diffusion architecture to perform the generative conversion. We perform comprehensive quantitative evaluations on the fidelity and realism of the generated sensor data. We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇题为 **《Sensor2Sensor: Cross-Embodiment Sensor Conversion for Autonomous Driving》** 的论文分析如下：

### 1. 核心贡献摘要
该论文提出了 **Sensor2Sensor** 框架，旨在通过生成式建模将“野外（In-the-wild）”单目行车记录仪视频转换为自动驾驶系统（ADS）所需的结构化、多模态高保真数据（多视角相机图像及激光雷达点云）。这一研究解决了自动驾驶领域高质量、多样化训练数据匮乏的瓶颈，实现了从非结构化视觉数据到标准自动驾驶数据格式的“跨形态（Cross-Embodiment）”转换。

### 2. 关键创新与方法论
*   **双向转换范式（Cycle-like paradigm）**：由于缺乏配对的训练数据（即同一场景下的行车记录仪视频与专业AV传感器数据），论文采用了巧妙的“回译”策略：首先利用 **4D Gaussian Splatting (4DGS)** 对真实的AV日志进行高质量重建和新视角渲染，生成对应的“行车记录仪风格”视频，从而构建成对数据。
*   **扩散模型（Diffusion Architecture）**：在构建好配对数据集的基础上，利用扩散模型强大的生成能力，实现从简单的单目输入到复杂的空间化、多模态传感器数据的映射。
*   **跨形态转换技术**：这是该方法的核心，通过学习将低成本的“观察者”视图转换为高价值的“机器人（AV）”视角下的多传感器流，显著提升了数据利用率。

### 3. 对领域的潜在影响
*   **数据匮乏问题的范式转变**：传统方法依赖高昂的自动驾驶车队采集，该研究展示了利用互联网大规模视频数据（YouTube/Dashcam）来“合成为”自动驾驶训练数据的可能性，极大地拓宽了数据来源的边界。
*   **长尾场景的覆盖**：通过引入野外数据，自动驾驶系统能够以极低的成本学习极其稀有的长尾场景（如极罕见的交通冲突、恶劣天气、复杂的路口行为），这对于实现L4/L5级自动驾驶至关重要。
*   **仿真与验证的增强**：生成的真实传感器数据可直接注入现有的ADS感知系统进行离线测试，降低了对昂贵实车路测的依赖。

### 4. 受益的相关领域与应用
*   **具身智能（Embodied AI）**：机器人领域同样面临如何将人类视角视频转化为机器人感知数据的难题，此方法可直接迁移至通用机器人训练。
*   **数字孪生（Digital Twins）**：可用于构建城市规模的动态三维高保真模拟环境。
*   **数据隐私与合成数据生成**：无需保留原始敏感视频，通过合成的多模态传感器数据即可实现模型迭代，有助于解决数据隐私保护问题。

### 5. 可推断的局限性
*   **生成的一致性（Consistency）**：尽管扩散模型在静态图像生成上表现出色，但在处理长时间序列的多模态一致性（如多视角摄像头的时间同步及LiDAR点云的几何物理真实性）方面仍面临巨大挑战，可能会出现“伪影”或物理不合理现象。
*   **语义准确性**：生成的数据在几何上可能逼真，但生成的物体属性（如车辆速度、交通灯状态）是否与训练需求保持高度的“逻辑准确性”仍需考量。
*   **计算资源需求**：4DGS的重建过程与扩散模型的推理过程均计算密集，如何大规模落地应用需要平衡模型复杂度与生产效率。

---

**专家点评：**
这篇论文的有趣之处在于它成功地将 **4DGS（三维重建的最新突破）** 与 **扩散模型（生成式AI的核心）** 有机结合，解决了一个经典的感知工程难题。它不再试图“改进传感器本身”，而是通过“重塑数据源”来提升感知系统的鲁棒性，这种思路代表了当前学术界从“模型驱动”向“数据驱动”范式转型的趋势。

**Key Findings:**

- In contrast, in-the-wild data from sources like dashcams offers immense scale and diversity, capturing critical long-tail scenarios and novel environments.
- To bridge this data gap, we propose Sensor2Sensor, a novel generative modeling paradigm that translates in-the-wild monocular dashcam videos into a high-fidelity, multi-modal sensor suite (AV logs) comprising multi-view camera images and LiDAR point clouds.
- We address this by converting real AV logs into dashcam-style videos via 4D Gaussian Splatting (4DGS) reconstruction and novel-view rendering.
- We demonstrate Sensor2Sensor's practical utility by converting challenging in-the-wild internet and dashcam footage into realistic, multi-modal data formats, further unlocking vast external data sources for AV development.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22809v1)
- [arXiv](https://arxiv.org/abs/2605.22809v1)

---

<a id='2605.22697v1'></a>
## [Cross-Domain Human Action Recognition from Multiview Motion and Textual Descriptions](https://arxiv.org/abs/2605.22697v1)

**Authors:** Yannick Porto, Renato Martins, Thomas Chalumeau, Cedric Demonceaux

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

Robustness to domain changes is a key capability for effective deployment of human action recognition systems in real-world scenarios, where action categories at inference can present important domain shifts or even unseen actions from training. In this context, improving the recognition capabilities of Zero-Shot Action Recognition models (ZSAR), without requiring strong annotation efforts, remains a central challenge. Most ZSAR approaches assume that actions are observed under geometric conditions similar to those seen during training. In practice, variations in human body orientation and camera viewpoint add a significant domain gap in ZSAR, substantially limiting generalization to novel action-motion combinations. In this context, this paper presents a novel orientation-aware action recognition approach with improved cross-domain capabilities. Our approach combines motion cues of multiple camera viewpoints and text descriptions of human actions in the training phase. We present a new orientation-aware motion encoding network to learn different motion features, and adapt a specific orientation-aware text prompt to match the corresponding features at inference. Extensive experiments demonstrate that the proposed method consistently improves ZSAR performance across different recognition benchmarks, outperforming recent state-of-the-art zero-shot approaches on NTU-RGB+D, BABEL, NW-UCLA, and on two surveillance datasets. In addition, the learned representations exhibit strong transfer learning capabilities, yielding competitive performance on both cross-domain and same-domain recognition of seen actions. Code and trained models are available at: https://icb-vision-ai.github.io/OrientationAware-HAR

**Analysis:**

### 1. 摘要翻译
真实场景下，人类动作识别系统面临域偏移（Domain Shifts）和未知动作的挑战。现有的零样本动作识别（ZSAR）方法通常假设测试时的几何条件（如视角、人体方向）与训练时相似，这在实际应用中往往无法成立。本文提出了一种方向感知（Orientation-aware）的动作识别方法，旨在显著提升跨域性能。该方法在训练阶段结合多视角动作线索与人类动作的文本描述，设计了一种新的方向感知运动编码网络来学习运动特征，并通过特定的方向感知文本提示（Text Prompt）在推理时进行特征对齐。大量实验表明，该方法在多个数据集（NTU-RGB+D, BABEL, NW-UCLA及两个监控数据集）上优于当前最先进的零样本方法，并展现出极强的迁移学习能力。

### 2. 方法动机分析
- **驱动力**：解决动作识别在跨域部署中因视角和人体方向差异造成的性能下降问题，提升模型对几何域偏移的鲁棒性。
- **现有方法痛点**：传统ZSAR假设视角不变，忽略了现实中人体方向变化导致的特征畸变（如侧面与背面的动作外观差异）。现有方法多依赖“共识原则”（Consensus Principle），试图通过数据增强强行实现视角不变性，而忽略了多视角下各异的信息互补价值。
- **研究假设**：通过在训练中显式引入人体方向信息，并将其与动作语义文本对齐，模型可以学习到对视角变化具有本质鲁棒性的表示，从而实现更精准的跨域识别。

### 3. 方法设计详解
- **流程总结**：
  1. **虚拟投影（Projection）**：仅在训练阶段，利用SMPL参数模拟12种虚拟摄像头视角，生成多视角下的2D骨架序列，并引入遮挡模拟以逼近“野外”场景。
  2. **方向感知编码（Orientation-Aware Network）**：使用ProtoGCN作为主干，将人体方向角$\theta$编码为高频特征，通过双分支注意力机制（“方向作为查询”与“动作作为查询”）将方向信息动态注入到运动特征中。
  3. **文本增强（Action Description）**：利用GPT-3.5为动作类别生成包含人体方向描述（如“从背面视角观察到的踢腿动作”）的提示词，利用CLIP构建多模态语义空间。
  4. **联合优化**：利用对比损失（Contrastive Loss）拉近同类动作的运动特征与文本描述的距离，同时配合交叉熵分类损失进行约束。
- **模型结构**：投影组件处理几何差异，方向感知网络处理空间特征，多模态模块处理语义对齐。
- **关键公式**：$\gamma(\theta)$ 使用正弦/余弦位置编码将方向角映射到高维空间，通过双分支注意力实现特征对条件信息的深度融合。

### 4. 方法对比分析
- **本质区别**：从“寻求视角不变性”转向“利用视角差异性”。不再通过数据增强追求对齐，而是将视角作为一种条件信息（Conditioning）主动纳入特征提取过程。
- **创新贡献**：提出方向感知运动编码机制；将人体方向引入文本提示工程，通过LLM丰富动作语义，实现了运动与文本在视角条件下的细粒度对齐。
- **适用场景**：监控视频、机器人视觉、以及训练与推理视角存在显著几何偏差的各类HAR任务。

### 5. 实验分析
- **验证方法**：在多个数据集上进行ZSL（零样本学习）和ZSCD（跨域零样本）评测，对比传统ZSAR方法。
- **关键结果**：在NTU-RGB+D和NW-UCLA上分别取得显著精度提升；在监控场景（RHM-HAR/MCAD）中性能优于现有模型，证明了其应对高视角差异的能力。
- **优劣势**：
  - **优势**：极强的跨域迁移性，无需目标域微调；能处理因视角导致的遮挡问题。
  - **局限**：推理阶段需要显式估计人体方向，增加了系统计算开销。

### 6. 实用指南
- **开源情况**：代码和预训练模型已开源（https://icb-vision-ai.github.io/OrientationAware-HAR）。
- **实现细节**：$\mu=0.5, \lambda=0.5$（预训练）及 $\lambda=1$（微调）。注意需使用SMPL参数进行数据预处理。
- **迁移可能**：该框架的“文本提示增强+条件编码器”架构可直接迁移至涉及三维姿态的任何下游任务（如手势识别、舞蹈分析）。

### 7. 总结
- **核心思想**：通过显式建模视角信息并与文本语义对齐，打破跨域几何瓶颈。
- **速记版pipeline**：
  1. 虚拟多视角数据生成（训练特供）。
  2. 动作骨架特征编码（融入视角条件）。
  3. 动作语义文本增强（包含视角描述）。
  4. 运动与文本模态对齐。
  5. 多视角推理（取均值集成）。

**Key Findings:**

- In practice, variations in human body orientation and camera viewpoint add a significant domain gap in ZSAR, substantially limiting generalization to novel action-motion combinations.
- In this context, this paper presents a novel orientation-aware action recognition approach with improved cross-domain capabilities.
- Our approach combines motion cues of multiple camera viewpoints and text descriptions of human actions in the training phase.
- We present a new orientation-aware motion encoding network to learn different motion features, and adapt a specific orientation-aware text prompt to match the corresponding features at inference.
- Extensive experiments demonstrate that the proposed method consistently improves ZSAR performance across different recognition benchmarks, outperforming recent state-of-the-art zero-shot approaches on NTU-RGB+D, BABEL, NW-UCLA, and on two surveillance datasets.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22697v1)
- [arXiv](https://arxiv.org/abs/2605.22697v1)

---

<a id='2605.22677v1'></a>
## [Slimmable ConvNeXt: Width-Adaptive Inference for Efficient Multi-Device Deployment](https://arxiv.org/abs/2605.22677v1)

**Authors:** Janek Haberer, Jon Eike Wilhelm, Olaf Landsiedel

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

Deploying vision models across devices with varying resource constraints, or even on a single device where available compute fluctuates due to battery state, thermal throttling, or latency deadlines, typically requires training and maintaining separate models. Width-adaptive inference addresses this by training a single set of shared weights containing multiple nested subnetworks of increasing capacity, but prior CNN-based approaches required switchable batch normalization, while recent scalable methods have focused on Vision Transformers. We present Slimmable ConvNeXt, which shows that ConvNeXt's modern design, specifically LayerNorm and inverted bottlenecks, makes it particularly suited for channel-width slimming, eliminating the normalization overhead of classical slimmable networks and producing a simpler training pipeline than both prior CNN and ViT approaches. On ImageNet-1k, Slimmable ConvNeXt-T with 3 subnetworks achieves 80.8% top-1 accuracy at 4.5 GMACs and 77.4% at 1.2 GMACs, trained from scratch for 600 epochs. At comparable compute, this exceeds HydraViT's 6-head subnetwork (78.4% at 4.6 GMACs) by 2.4 percentage points and its 3-head configuration (73.0% at 1.3 GMACs) by 4.4 percentage points, while also outperforming MatFormer-S (78.6%) and SortedNet-S (78.2%) at the same GMACs. Scaling to Slimmable ConvNeXt-B further improves maximum accuracy to 82.8% at 15.35 GMACs.

**Analysis:**

### 1. 摘要翻译
跨设备部署视觉模型或应对单设备上的计算资源波动（如电池状态、热节流或延迟约束），通常需要训练和维护多个独立的模型。宽度自适应推理通过训练包含多个嵌套子网的单一权重集来解决此问题，但先前的CNN方案依赖于复杂的切换批量归一化（switchable batch normalization），而近期研究多集中在视觉Transformer（ViT）。本文提出“Slimmable ConvNeXt”，利用ConvNeXt的现代设计（特别是LayerNorm和反向瓶颈结构），使其天然适合通道宽度裁剪。这不仅消除了传统裁剪网络中归一化带来的额外开销，还简化了训练流程。在ImageNet-1k上，Slimmable ConvNeXt-T在4.5 GMACs下达到80.8%的Top-1准确率，且在低计算量下显著优于HydraViT、MatFormer等模型。

---

### 2. 方法动机分析
*   **驱动力**：在单一模型内实现多精度计算，以适应多变的硬件资源预算。
*   **现有方法痛点**：传统CNN裁剪方法（如US-Net）高度依赖Batch Normalization（BN）的统计量切换，增加了训练复杂度和模型参数；而ViT方法（如HydraViT）受限于自注意力机制，裁剪结构受限于注意力头数（head），难以实现细粒度的宽度调节。
*   **核心直觉**：ConvNeXt架构本身具备类似于现代ViT的组件（LayerNorm和反向瓶颈），这些组件天生对通道裁剪更友好，无需额外开销即可实现高效的弹性推理。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **定义裁剪比（Slimming Ratio）**：为每个ConvNeXt块分配比率 $p \in (0, 1]$，确定活跃通道数 $\lfloor p \cdot C \rfloor$。
    2.  **权重切片**：所有卷积层（深度/逐点）和LayerNorm参数均根据 $p$ 动态切片。
    3.  **残差处理**：当残差连接维度不匹配时（因裁剪），通过**零填充（zero-padding）**补全通道数，从而无需引入额外投影参数。
    4.  **联合训练**：采用随机采样策略，每次迭代随机选择一个子网配置进行前向/反向传播，实现多子网共享权重。
*   **关键公式**：
    $\min _{\theta _k} \sum _{i=1}^{B} \mathcal {L}\bigl (f_{\theta _k}(x_i), y_i\bigr )$
    通过随机采样子网索引 $k$，优化所有嵌套子网的权重 $\theta_k$。
*   **优势**：LayerNorm在通道维度上独立计算，使其无需像BN那样维护多套运行统计量，模型实现更为轻量。

---

### 4. 方法对比分析
*   **本质区别**：去除了CNN方法中繁琐的切换BN，且相比ViT方法避免了对注意力头数的结构性依赖，裁剪灵活性更高。
*   **创新贡献**：证明了现代CNN架构组件对宽度自适应的兼容性，为高效部署提供了一种比ViT更简单、更具硬件通用性的方案。
*   **适用场景**：适合需要频繁在不同计算预算间切换的移动端设备或异构集群。

---

### 5. 实验分析（精简版）
*   **验证方法**：在ImageNet-1k数据集上，通过对比HydraViT、MatFormer等先进架构，验证不同裁剪比下的Top-1准确率。
*   **关键结论**：Slimmable ConvNeXt-B在4.0 GMACs下实现81.6%的准确率，比HydraViT的同性能子网节省约13%计算量。
*   **主要局限**：零填充策略在极低裁剪比（如$p=0.25$）时会丢失大量特征，导致性能下降；且计算量（GMACs）不能完全代表实际推理延迟。

---

### 6. 实用指南
*   **实现细节**：
    *   **AutoSlim**：对于非均匀宽度，可采用贪婪搜索策略，根据验证集准确率调整各层裁剪比例。
    *   **训练策略**：采用AdamW优化器，配合余弦衰减及常用的数据增强（RandAugment, Mixup, CutMix），无需复杂的知识蒸馏即可收敛。
*   **迁移可能**：该架构设计理念可直接迁移至其他使用LayerNorm和点对点连接（pointwise layers）的现代化CNN变体中。

---

### 7. 总结
*   **核心思想**：利用ConvNeXt的现代特性，通过简单的通道切片实现高效宽度自适应。
*   **速记版pipeline**：
    1. 为各块分配通道裁剪比；
    2. 按需动态切片权重参数；
    3. 残差部分补零对齐维度；
    4. 随机抽样并行训练所有子网。

**Key Findings:**

- We present Slimmable ConvNeXt, which shows that ConvNeXt's modern design, specifically LayerNorm and inverted bottlenecks, makes it particularly suited for channel-width slimming, eliminating the normalization overhead of classical slimmable networks and producing a simpler training pipeline than both prior CNN and ViT approaches.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22677v1)
- [arXiv](https://arxiv.org/abs/2605.22677v1)

---

<a id='2605.22671v1'></a>
## [From Abstraction to Instantiation: Learning Behavioral Representation for Vision-Language-Action Model](https://arxiv.org/abs/2605.22671v1)

**Authors:** Bing Hu, Zaijing Li, Rui Shao, Junda Chen, April Hua Liu, Wei-Shi Zheng, Liqiang Nie

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

Vision-Language-Action (VLA) models often suffer from performance degradation under distribution shifts, as they struggle to learn generalized behavior representations across varying environments. While existing approaches attempt to construct behavior representations through action-centric latent variables, they are often limited by short-horizon temporal fragmentation and static execution-alignment, leading to inconsistent behaviors in complex scenarios. To address these limitations, we propose \textbf{BehaviorVLA}, a framework that facilitates robust manipulation through the learning of a temporally coherent behavioral representations. Our approach features two symmetric components: (1) the \textbf{Visuomotor Behavior Encoder (VBE)}, which utilizes a causal Mamba-based architecture to aggregate long-horizon trajectory information into a unified behavior representation; and (2) the \textbf{Phase-conditioned Behavior Decoder (PBD)}, which decodes this representation into precise actions by dynamically aligning task-level priors with real-time execution progress. Experiments on RoboTwin 2.0, LIBERO, and CALVIN demonstrate state-of-the-art success rates of 58\%, 98\%, and 4.36 (Avg.Len), respectively. Notably, in real-world sim-to-real transfer, BehaviorVLA matches the performance of OpenVLA-OFT using only 50\% of the demonstration data, showcasing its superior data efficiency and generalization.

**Analysis:**

# 论文方法分析：From Abstraction to Instantiation (BehaviorVLA)

### 1. 摘要翻译
视觉-语言-动作（VLA）模型在分布偏移下常表现出性能下降，因为它们难以跨环境学习通用的行为表示。现有方法虽尝试通过动作中心的潜在变量构建行为表示，但常受限于短时域碎片化和静态执行对齐，导致在复杂场景下行为不连贯。为此，我们提出了 **BehaviorVLA**，该框架通过学习时间上连贯的行为表示来促进鲁棒的操作。方法包含两个对称组件：（1）**视觉运动行为编码器（VBE）**，利用因果Mamba架构将长时域轨迹信息聚合为统一的行为表示；（2）**相位条件行为解码器（PBD）**，通过将任务级先验与实时执行进度动态对齐，将上述表示解码为精确动作。在RoboTwin 2.0、LIBERO和CALVIN上的实验表明其达到了最先进的成功率。特别是在零样本模拟到现实（sim-to-real）迁移中，BehaviorVLA仅用50%的演示数据就达到了微调后OpenVLA-OFT的性能，展示了卓越的数据效率和泛化能力。

### 2. 方法动机分析
- **驱动力**：解决VLA模型在面对分布偏移（如不同背景、光照、物体）时发生的灾难性性能退化问题。
- **现有方法痛点**：
    1. **短时域碎片化**：将轨迹切分为独立块或离散码，丢失了长程依赖，导致全局连贯性差。
    2. **静态执行对齐**：解码器脱离实时环境状态，导致生成的动作序列随时间产生漂移，无法动态响应。
- **研究假设**：鲁棒的VLA需要将“特定到一般”的抽象（提取任务拓扑）与“一般到特定”的实例化（投影为环境感知动作）相结合，通过显式的行为流形建模约束高维轨迹。

### 3. 方法设计详解
**核心 Pipeline：**
1. **VBE 抽象阶段**：输入观测和指令，通过因果三流（Vision, Action, Behavior）Mamba架构进行时域建模，利用交叉注意力机制进行多模态融合，提取全局原型（$z_{\text{proto}}$）和时间演变的相位状态（$z_{\text{phase}}$）。
2. **PBD 实例化阶段**：
    - **预测器 (Predictor)**：根据 $z_{\text{proto}}$ 生成动作骨架，利用 $z_{\text{phase}}$ 在流形上进行可微插值，得到局部几何上下文 $c_t$，进而初始化动作先验。
    - **校正器 (Corrector)**：使用条件流匹配（Conditional Flow Matching），将先验引导注入到去噪空间，约束流场生成精确轨迹，从而修正动作漂移。

**模型结构：**
- **VBE**：充当信息瓶颈，过滤环境噪声，保留拓扑信息。
- **PBD**：采用预测-校正范式，确保动作生成严格同步于物理执行进度。

### 4. 方法对比分析
- **本质区别**：与仅通过大模型隐式建模的方法不同，BehaviorVLA通过显示地解耦“全局任务拓扑”和“局部执行进度”，将行为表示显式化。
- **创新贡献**：引入了相位条件流匹配，将静态的潜在变量转化为动态对齐的动作流。
- **适用场景**：复杂、长时域、对环境变化敏感的机器人操作任务（如Sim-to-Real迁移）。

### 5. 实验分析（精简版）
- **关键结论**：在LIBERO上达到98%成功率；Real-World任务中，在仅使用50%数据的情况下性能持平OpenVLA-OFT。
- **主要优势**：极高的数据效率和长程任务下的行为一致性。
- **主要局限**：依赖离线原型库的拓扑多样性；流匹配的迭代求解增加了推理延迟。

### 6. 实用指南
- **开源情况**：详见项目主页 BehaviorVLA.github.io。
- **实现细节**：关键在于阶段性训练（先学流形，后调策略），并引入Stochastic Dropout防止后验坍塌。
- **迁移可能**：VBE架构可迁移至任何需要长时序动作生成的机器人任务，只需替换Vision Encoder底座。

### 7. 总结
- **核心思想**：通过解耦全局拓扑与动态相位，实现任务先验引导下的精确行为生成。
- **速记版pipeline**：
    1. **抽象**：三流Mamba编码器提炼动作与视觉中的长程任务逻辑。
    2. **原型检索**：从记忆库提取全局拓扑指导。
    3. **相位估计**：实时跟踪当前任务进度。
    4. **流匹配纠偏**：通过流模型将先验转化为高精度动作输出。

**Key Findings:**

- To address these limitations, we propose \textbf{BehaviorVLA}, a framework that facilitates robust manipulation through the learning of a temporally coherent behavioral representations.
- Our approach features two symmetric components: (1) the \textbf{Visuomotor Behavior Encoder (VBE)}, which utilizes a causal Mamba-based architecture to aggregate long-horizon trajectory information into a unified behavior representation; and (2) the \textbf{Phase-conditioned Behavior Decoder (PBD)}, which decodes this representation into precise actions by dynamically aligning task-level priors with real-time execution progress.
- Experiments on RoboTwin 2.0, LIBERO, and CALVIN demonstrate state-of-the-art success rates of 58\%, 98\%, and 4.36 (Avg.Len), respectively.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22671v1)
- [arXiv](https://arxiv.org/abs/2605.22671v1)

---

<a id='2605.22668v1'></a>
## [SEGA: Spectral-Energy Guided Attention for Resolution Extrapolation in Diffusion Transformers](https://arxiv.org/abs/2605.22668v1)

**Authors:** Javad Rajabi, Kimia Shaban, Koorosh Roohi, David B. Lindell, Babak Taati

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

Diffusion transformers (DiTs) have emerged as a dominant architecture for text-to-image generation, yet their performance drops when generating at resolutions beyond their training range. Existing training-free approaches mitigate this by modifying inference-time attention behavior, often through Rotary Position Embeddings (RoPE) extrapolation combined with attention scaling. However, these strategies apply a uniform and content-agnostic scaling across RoPE components with distinct frequency characteristics, inducing a trade-off between preserving global structure and recovering fine detail. We introduce SEGA, a training-free method that dynamically scales attention across RoPE components according to the latent's spatial-frequency structure at each denoising step. This adaptive scaling improves both structural coherence and fine-detail fidelity. Experiments show that SEGA consistently improves high-resolution synthesis across multiple target resolutions, outperforming state-of-the-art training-free baselines.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **SEGA (Spectral-Energy Guided Attention)** 的论文分析如下：

### 1. 核心贡献总结
SEGA 提出了一种无需训练（training-free）的注意力机制优化方法，专门解决扩散变换器（DiT）在超出训练分辨率范围时的生成退化问题。该方法通过引入频率感知的动态注意力缩放，取代了传统的统一缩放策略，从而在保持全局结构完整性的同时，显著增强了高分辨率图像的细节还原能力。

### 2. 关键创新与方法论
该论文的核心在于**频率驱动的自适应缩放（Frequency-Aware Adaptive Scaling）**：
*   **痛点发现**：现有的方法（如 RoPE 外推与统一缩放）忽略了不同频率组件对图像结构与细节的不同贡献，导致“顾此失彼”：要么牺牲全局结构，要么丢失高频细节。
*   **方法论**：SEGA 在去噪过程的每一步，实时分析潜在空间（Latent）的**空间频率结构（Spatial-Frequency Structure）**。它根据不同频率分量的能量分布，动态调整 RoPE 注意力中的缩放因子。这意味着低频分量（负责结构）和高频分量（负责细节）被区别对待，从而实现了更智能的推理策略。

### 3. 对领域的潜在影响
*   **突破生成分辨率限制**：这项工作使得现有的预训练扩散模型（DiT）能够以极低的额外计算成本，泛化到更高分辨率，极大地提升了预训练模型的使用价值。
*   **从“静态启发式”转向“动态计算”**：该方法挑战了现有推理时统一缩放的范式，证明了基于内容的自适应机制在解决外推问题上具有显著优势。
*   **推动 DiT 在高分辨率领域的落地**：DiT 目前是视频和图像生成的主流（如 Sora, SD3），SEGA 的出现可能成为解决长视频或超高分辨率图像生成中“连贯性丧失”问题的关键技术手段。

### 4. 相关领域与应用价值
*   **高清视频生成**：视频生成模型通常受限于训练帧长与分辨率，SEGA 的方法可直接迁移至时间维度或空间分辨率的提升。
*   **遥感图像处理**：卫星图像对细节和全局结构的双重高要求，非常契合 SEGA 的设计初衷。
*   **计算摄影与超分辨率**：该方法可为图像增强工具提供更强大的底层生成先验，使其在不通过昂贵微调的情况下，实现更高质量的细节外推。

### 5. 可推断的局限性
*   **实时计算成本**：虽然是“训练无需”，但需要在去噪每一步进行频率分析和计算，可能会略微增加单步推理的时间（虽然理论上可以通过高效的 FFT 实现优化，但仍需考量对延迟的影响）。
*   **频率感知的泛化性**：该方法高度依赖于对潜在空间频率特性的准确估计，在某些极端分布（如极简主义图像或具有高度非周期纹理的场景）中，其性能是否依然稳健有待验证。
*   **与采样器的耦合**：该方法是否能无缝兼容现有的主流采样器（如 ODE/SDE 求解器），或者是否需要特定的采样调整，尚需实证。

### 专家点评
这篇论文的有趣之处在于它将**信号处理领域的经典频域分析方法**引入了**深度生成的注意机制优化**中。在当前 DiT 模型普遍存在“分辨率灾难”的背景下，这种无需重训的轻量级方案具有极高的工业应用潜力。它避开了昂贵的计算资源消耗，通过对推理机制的“微手术”，实现了生成质量的质变，是典型的“以精巧架构设计解决大规模计算痛点”的优秀研究。

**Key Findings:**

- We introduce SEGA, a training-free method that dynamically scales attention across RoPE components according to the latent's spatial-frequency structure at each denoising step.
- Experiments show that SEGA consistently improves high-resolution synthesis across multiple target resolutions, outperforming state-of-the-art training-free baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22668v1)
- [arXiv](https://arxiv.org/abs/2605.22668v1)

---

<a id='2605.22658v1'></a>
## [SegCompass: Exploring Interpretable Alignment with Sparse Autoencoders for Enhanced Reasoning Segmentation](https://arxiv.org/abs/2605.22658v1)

**Authors:** Zhenyu Lu, Liupeng Li, Jinpeng Wang, Haoqian Kang, Yan Feng, Ke Chen, Yaowei Wang

**Published:** 2026-05-21

**Categories:** cs.CV, cs.LG, cs.MM, eess.IV

**Abstract:**

While large language models provide strong compositional reasoning, existing reasoning segmentation pipelines fail to transparently connect this reasoning to visual perception. Current methods, such as latent query alignment, are end-to-end yet opaque "black boxes". Conversely, textual localization readout is merely readable, not truly interpretable, often functioning as an unconstrained post-hoc step. To bridge this interpretability gap, we propose SegCompass, an end-to-end model that leverages a Sparse Autoencoder (SAE) to forge an explicit, interpretable, and differentiable alignment pathway. Given an image-instruction pair, SegCompass first generates a chain-of-thought (CoT) trace. The core of our method is an SAE that maps both the CoT and visual tokens into a shared, high-dimensional sparse concept space. A query codebook selects salient concepts from this space, which are then spatially grounded by a slot mapper into a multi-slot heatmap that guides the final mask decoder. The entire model is trained jointly, unifying reinforcement learning for the reasoning path with standard segmentation supervision. This SAE-driven interface provides a "white-box" connection that is significantly more traceable than latent queries and more coherent than textual readouts. Extensive experiments on five challenging benchmarks demonstrate that SegCompass matches or surpasses state-of-the-art performance. Crucially, our visual and quantitative analyses show a strong correlation between the quality of the learned sparse concepts and final mask accuracy, confirming that SegCompass achieves superior results through its enhanced and inspectable alignment. Code is available at https://github.com/ZhenyuLU-Heliodore/SegCompass.

**Analysis:**

以下是对 *SegCompass* 论文的深度分析：

### 1. 摘要翻译
尽管大型语言模型（LLMs）具备强大的组合推理能力，但现有的推理分割流水线无法透明地将这种推理与视觉感知连接起来。当前方法（如潜在查询对齐）通常是端到端的“黑盒”，而文本定位读取方法虽具可读性，但缺乏真正的解释性。为弥补这一差距，我们提出了 **SegCompass**，这是一种利用稀疏自编码器（SAE）构建显式、可解释且可微的对齐路径的端到端模型。给定图像-指令对，SegCompass首先生成思维链（CoT）追踪。模型核心在于一个SAE，它将CoT和视觉标记映射到一个共享的高维稀疏概念空间。查询码本从该空间选择显著概念，随后通过槽映射器将其空间接地，生成引导最终掩码解码器的多槽热力图。整个模型经过联合训练，统一了推理路径的强化学习和标准分割监督。这种SAE驱动的接口提供了比潜在查询更具可追踪性、比文本读取更连贯的“白盒”连接。在五个基准测试上的实验表明，SegCompass达到了或超越了当前最优水平。可视化与定量分析验证了所学稀疏概念质量与掩码精度之间的强相关性，证实了SegCompass通过增强和可检查的对齐实现了优越性能。

### 2. 方法动机分析
*   **驱动力**：解决推理分割中“推理过程”与“视觉分割”之间的鸿沟，提升模型的透明度与控制力。
*   **现有痛点**：
    *   **潜在查询对齐（Latent Query Alignment）**：属于“黑盒”，无法直观理解模型为何选中特定像素。
    *   **文本定位（Textual Readout）**：属于“后处理”，CoT与实际掩码生成缺乏强耦合，缺乏空间语义细节。
*   **核心假设**：高维稀疏概念空间能够解耦LLM内复杂的推理特征，通过将这些解耦的概念与视觉特征直接“对齐”，可构建透明的推理-感知映射通路。

### 3. 方法设计详解
*   **Pipeline流程**：
    1.  **推理生成**：通过MLLM生成思维链（CoT）及特定的“集中标记”（concentration tokens）。
    2.  **概念映射（SAE核心）**：将MLLM的中间层隐藏状态 $z$ 通过预训练好的稀疏自编码器（SAE）映射到高维稀疏激活空间 $h(z)$。每个维度对应一个可解释的“字典原子”。
    3.  **概念选择与整合**：利用查询码本（Query Codebook）从稀疏空间筛选出显著概念，并结合Transformer模块聚合为概念表示 $r_k$。
    4.  **空间接地（Slot Mapping）**：通过“槽映射器”将集中标记嵌入与概念表示融合，通过多头注意力机制与视觉Encoder输出的键（Key）交互，生成多槽热力图（Multi-slot Heatmap）。
    5.  **掩码解码**：利用热力图和视觉特征，通过双向交叉注意力机制（模仿SAM解码器）输出最终掩码。
*   **算法本质**：将复杂的LLM潜在表示转化为可审计的、稀疏的概念原子，从而实现推理逻辑与视觉空间定位的直接“翻译”。

### 4. 方法对比分析
*   **本质区别**：引入了SAE作为LLM与分割器之间的“中间翻译器”，实现了对推理逻辑的显示解构，而非简单地让模型“猜”对应关系。
*   **创新贡献**：首次将SAE用于推理分割的可解释性连接，并提出了一套将SAE激活特征转化为空间热力图的架构。
*   **适用场景**：机器人导航、复杂逻辑指令下的图像理解、需要人机协作反馈的视觉任务。

### 5. 实验分析
*   **关键结论**：在RefCOCO系列及ReasonSeg等基准测试上实现SOTA；定量分析显示SAE重建误差与掩码Dice损失存在强相关性，证明了中间推理过程对最终结果的决定性作用。
*   **优势**：真正的“白盒”可解释性；通过GRPO强化学习有效提升推理质量。
*   **局限**：对超大规模预训练SAE的依赖；端到端联合训练对显存开销较大。

### 6. 实用指南
*   **开源情况**：已开源，代码地址：`https://github.com/ZhenyuLU-Heliodore/SegCompass`。
*   **实现细节**：SAE需预训练并冻结，建议选定在LLM的中层（如13-16层）进行特征提取，以平衡语义与视觉能力。
*   **迁移建议**：该架构可直接迁移至任何基于MLLM的多模态任务，只需更换下游的检测或分割头，核心在于SAE部分的“概念空间”重用。

### 7. 总结
*   **核心思想**：利用稀疏编码解耦推理特征，将抽象逻辑转化为可解释的视觉空间热力图。
*   **速记版Pipeline**：
    1. 思维链推理与概念特征提取；
    2. 通过SAE将特征稀疏化解耦；
    3. 筛选显著概念并编码；
    4. 空间接地生成热力图；
    5. 解码输出最终分割掩码。

**Key Findings:**

- To bridge this interpretability gap, we propose SegCompass, an end-to-end model that leverages a Sparse Autoencoder (SAE) to forge an explicit, interpretable, and differentiable alignment pathway.
- The core of our method is an SAE that maps both the CoT and visual tokens into a shared, high-dimensional sparse concept space.
- Extensive experiments on five challenging benchmarks demonstrate that SegCompass matches or surpasses state-of-the-art performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22658v1)
- [arXiv](https://arxiv.org/abs/2605.22658v1)

---

<a id='2605.22651v1'></a>
## [What Does the Caption Really Say? Counterfactual Phrase Intervention for Compositional Data Selection in Vision-Language Pretraining](https://arxiv.org/abs/2605.22651v1)

**Authors:** Hyejin Go, Semi Lee, Hyesong Choi

**Published:** 2026-05-21

**Categories:** cs.CV

**Abstract:**

CLIP-style contrastive pretraining typically curates web-scale image-text pairs using sample-level filtering signals, often based on pair-level alignment. We show that this signal saturates: once coarse mismatches are removed, stricter global filtering no longer tracks the compositional supervision provided by the retained captions. The reason is structural - a global score conflates whether a pair is broadly plausible with whether the individual object, attribute, and relation phrases inside the caption materially support the image-text match. The latter is what compositional generalization demands, yet pair-level filters are blind to it. We address this with Counterfactual Phrase Intervention (CPI), a phrase-level curation framework that converts controlled nonce-token substitutions into image-conditioned phrase-sensitivity scores. CPI uses global alignment only for coarse mismatch removal, then ranks the surviving pool by whether caption phrases measurably affect the image-text score under controlled substitution. We frame CPI as a first-order phrase-sensitivity signal rather than a grounding or identification result, and evaluate it at CC3M scale. Ranking by this signal yields a 50%-data subset that improves VL-CheckList-VG Relation by +1.91 over the full-data baseline and +1.00 over alignment-only filtering at matched budget, while improving SugarCrepe overall and preserving general transfer. CPI is loss-orthogonal: applied unchanged to NegCLIP, it further improves VL-CheckList-VG Relation by +3.84, with additional CE-CLIP gains in the main text.

**Analysis:**

以下是对《What Does the Caption Really Say? Counterfactual Phrase Intervention for Compositional Data Selection in Vision-Language Pretraining》的深度分析。

### 1. 摘要翻译
CLIP风格的对比预训练通常通过样本级对齐过滤来筛选网络规模的图文对。研究发现该信号已趋于饱和：一旦去除了粗糙的不匹配对，更严格的全局过滤不再能捕捉到保留字幕提供的组合监督信息。其结构原因是全局分数将“整体合理性”与“字幕中各短语（对象、属性、关系）是否对图文匹配有实质贡献”混为一谈。组合泛化要求后者，但全局过滤器对此“视而不见”。为此，我们提出“反事实短语干预”（CPI），这是一种短语级数据筛选框架，通过可控的非词（nonce-token）替换，将控制下的代入结果转化为“图像条件短语敏感度分数”。CPI仅利用全局对齐去除粗糙样本，随后根据短语对匹配的影响程度对剩余数据池进行排序。我们将CPI定义为一阶短语敏感度信号，并在CC3M规模上进行评估。筛选出的50%数据子集使VL-CheckList-VG Relation指标比全量基线提升+1.91，比单纯对齐过滤提升+1.00，且在保持通用迁移能力的同时提升了SugarCrepe性能。CPI具有“损失正交性”（loss-orthogonal），可直接应用于NegCLIP等方法，进一步带来显著性能增益。

### 2. 方法动机分析
*   **驱动力**：解决VLM在组合泛化上的“结构性盲区”——模型往往通过单一显著对象或OCR片段即可维持高匹配分数，从而忽略了细粒度的关系描述。
*   **痛点**：现有的全局过滤（如CLIPScore、DFN）仅考察样本的整体一致性，无法区分哪些样本真正包含了有价值的组合监督信号（即各短语是否真的与图像对应）。
*   **研究假设**：如果通过可控的语义替换干预使得相似度下降，则说明原短语对匹配有贡献；通过聚合这种敏感度得分，能筛选出更具“组合感知”潜力的训练样本。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **阶段一（粗选）**：使用全局对齐（CLIP相似度）筛选掉大部分明显不匹配的样本（如论文中选用的前70%）。
    2.  **阶段二（CPI精选）**：
        *   **提取与替换**：对字幕中每个短语片段 $p_j$ 提取词头 $w_j$。
        *   **三不变协议（Three-Invariance Protocol）**：生成非词 $r_j$ 替换 $w_j$，需严格满足：①子词数不变；②表面句法形式不变（大小写、复数、形态）；③无词汇语义。
        *   **计算敏感度 $\Delta_j$**：$\Delta_j = s(I, T) - s(I, \tilde{T}_j)$，即替换前后的相似度降幅。
        *   **聚合**：计算均值 $\mu_i$ 并归一化得到最终样本分数，保留得分最高的部分。
*   **创新之处**：引入“三不变协议”避免了替换带来的噪声（如长度改变导致的Embedding偏移），确保 $\Delta_j$ 仅反映该短语的“贡献度”。

### 4. 方法对比分析
*   **本质区别**：不修改模型架构或训练目标（Loss），而是干预数据分布，属于“数据侧”的组合能力增强。
*   **适用场景**：适用于任何基于CLIP架构的预训练流程，作为一种插件式的预处理步骤。

### 5. 实验分析（精简版）
*   **关键结论**：在50%的数据预算下，CPI选出的样本在VL-CheckList-VG（关系理解）指标上表现远超同等规模的随机采样或单纯对齐筛选。
*   **优势**：插件式、损失正交（可直接提升NegCLIP等现有模型）、能有效缓解“组合空心化”问题。
*   **局限**：对计算资源有要求（需要对大量样本进行反事实评估）；目前的评估规模仅在CC3M，超大规模（如LAION-400M）的有效性及计算成本需进一步验证。

### 6. 实用指南
*   **实现细节**：关键在于“三不变协议”的实现，必须保证Nonce token的生成符合CLIP分词器逻辑。
*   **迁移迁移**：非常容易迁移。只需将数据预处理替换为CPI Pipeline，将得到的 subset 输入现有的 CLIP 训练流程即可。

### 7. 总结
*   **核心思想**：通过替换短语并观察相似度跌幅，筛选“组合知识丰富”的优质样本。
*   **速记版 Pipeline**：
    1. **去渣**：按全局相似度踢除低质量图文对。
    2. **干预**：按“三不变原则”用无意义符号替换短语。
    3. **计分**：测量相似度下降幅度，量化短语贡献。
    4. **精选**：只保留那些短语贡献度高的黄金样本。

**Key Findings:**

- We show that this signal saturates: once coarse mismatches are removed, stricter global filtering no longer tracks the compositional supervision provided by the retained captions.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22651v1)
- [arXiv](https://arxiv.org/abs/2605.22651v1)

---

<a id='2605.22639v1'></a>
## [Symmetries Here and There, Combined Everywhere: Cross-space Symmetry Compositions in Robotics](https://arxiv.org/abs/2605.22639v1)

**Authors:** Loizos Hadjiloizou, Rodrigo Pérez-Dattari, Noémie Jaquier

**Published:** 2026-05-21

**Categories:** cs.RO

**Abstract:**

Robots exhibit a rich variety of symmetries arising from their mechanical structure and the properties of their tasks. Although many robotics problems exhibit several symmetries simultaneously, existing approaches typically treat them in isolation, failing to exploit their combined potential. This paper introduces cross-space symmetry compositions, a framework for learning robot policies that are jointly equivariant to multiple symmetries across configuration and task spaces. Leveraging the differential-geometric structure of the forward kinematics map, we both descend symmetries from configuration to task space and lift symmetries from task to configuration space, enabling their composition within a unified representation space. We validate our framework on simulated and real-world experiments on a dual-arm robot, demonstrating that jointly leveraging multiple symmetries yields improved generalization.

**Analysis:**

### 1. 摘要翻译
机器人展现出源于其机械结构和任务特性的丰富对称性。尽管许多机器人问题同时存在多种对称性，但现有方法通常孤立地处理它们，未能利用其组合潜能。本文引入了“跨空间对称性组合（cross-space symmetry compositions）”，这是一个用于学习机器人策略的框架，使策略能够联合等变于配置空间和任务空间中的多种对称性。利用前向运动学映射的微分几何结构，我们既能将对称性从配置空间“降维（descend）”到任务空间，也能将其从任务空间“提升（lift）”到配置空间，从而在统一的表示空间内实现它们的组合。我们在双臂机器人的模拟和真实实验中验证了该框架，证明了联合利用多种对称性可以显著提高泛化能力。

### 2. 方法动机分析
*   **驱动力**：机器人任务通常涉及多种对称性（如形态对称、旋转对称、任务诱导对称），若能联合利用这些对称性作为归纳偏置，将极大提升策略的学习效率与泛化能力。
*   **现有痛点**：对称性往往分布在不同的空间（配置空间vs任务空间），直接组合会导致数学形式不兼容。现有ML研究多局限于单空间内的组合，缺乏跨空间转换机制。
*   **研究假设**：通过前向运动学（FK）的黎曼流形结构，可以将不同空间中的对称性映射到统一空间中进行组合。

### 3. 方法设计详解
*   **流程总结**：
    1.  **对称性映射**：利用FK映射的微分几何性质，将配置空间的对称性（如形态镜像）通过Jacobian矩阵“降维”到任务空间，或将任务空间的对称性（如旋转、缩放）通过逆运动学/伪逆“提升”到配置空间。
    2.  **兼容性检验**：利用李代数（Lie Algebra）的李括号（Lie Bracket）判断不同对称性向量场是否对易（Commutative）。
    3.  **组合策略**：根据对易性，分别采用**直接积（Direct Product）**（如果对易）或**半直接积（Semi-Direct Product）**（如果不对易，通过同态映射处理）进行组合。
    4.  **策略训练**：将组合后的对称性作为数据增强的归纳偏置，用于指导策略学习。
*   **关键算法**：核心在于利用FK的**黎曼淹没（Riemannian Submersion）**特性。通过计算水平提升（Horizontal Lift），将任务空间的速度场精确转化为配置空间的关节速度场，保证了变换在两个空间的一致性。

### 4. 方法对比分析
*   **本质区别**：传统方法要么假设对称性在同一空间，要么只做单一空间的数据增强；本文首次打通了配置空间与任务空间的对称性通道。
*   **创新贡献**：提出了一种通用的数学框架，通过黎曼几何将分布在不同空间的对称性进行严谨的数学组合（特别是半直接积的引入）。
*   **适用场景**：高自由度、具备复杂机械结构（如双臂、多足）且涉及多任务对称性要求的机器人系统。

### 5. 实验分析
*   **验证方法**：在双臂模拟器（字母书写任务）及真实双臂机器人（RB-Y1）上进行。
*   **关键结果**：全对称性组合策略（$\pi_{GMRT}$）在处理未见过的旋转、缩放和镜像任务时，RMSE误差远低于单一对称性或无对称性策略。
*   **局限**：目前的组合主要依赖数据增强，尚未完全实现架构层面的硬性等变（Equivariance），这是未来工作方向。

### 6. 实用指南
*   **实现关键**：必须确保机器人远离奇异点（Assumption A4），且前向运动学映射需满足微分几何中的黎曼淹没条件。
*   **迁移方法**：若要迁移至其他机器人，首先定义该机器人的形态对称群和任务对称群，然后通过Jacobian矩阵实现跨空间向量场转换，最后根据对称算子的对易性选择组合方式。

### 7. 总结
*   **核心思想**：利用黎曼几何实现跨空间对称性降维与提升，构建统一对称性表示。
*   **速记版Pipeline**：
    1. 将所有对称性统一转换到配置空间。
    2. 计算对称性的李括号以判断对易性。
    3. 构造直接或半直接积进行对称性组合。
    4. 利用组合后的对称群对轨迹数据进行增强训练。

**Key Findings:**

- Leveraging the differential-geometric structure of the forward kinematics map, we both descend symmetries from configuration to task space and lift symmetries from task to configuration space, enabling their composition within a unified representation space.
- We validate our framework on simulated and real-world experiments on a dual-arm robot, demonstrating that jointly leveraging multiple symmetries yields improved generalization.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22639v1)
- [arXiv](https://arxiv.org/abs/2605.22639v1)

---

<a id='2605.22570v1'></a>
## [VGenST-Bench: A Benchmark for Spatio-Temporal Reasoning via Active Video Synthesis](https://arxiv.org/abs/2605.22570v1)

**Authors:** Jinho Park, Youbin Kim, Hogun Park, Eunbyung Park

**Published:** 2026-05-21

**Categories:** cs.CV, cs.AI

**Abstract:**

Spatio-temporal reasoning is a core capability for Multimodal Large Language Models (MLLMs) operating in the real world. As such, evaluating it precisely has become an essential challenge. However, existing spatio-temporal reasoning benchmark datasets primarily rely on static image sets or passively curated video data, which limits the evaluation of fine-grained reasoning capabilities. In this paper, we introduce VGenST-Bench, a video benchmark that employs generative models to actively synthesize highly controlled and diverse evaluation scenarios. To construct VGenST-Bench, we propose a multi-agent pipeline incorporating a human quality control stage, ensuring the quality of all generated videos and QA pairs. We establish a comprehensive 3x2x2 video taxonomy, encompassing Spatial Scale, Perspective, and Scene Dynamics to span diverse scenarios. Furthermore, we design a hierarchical task suite that decouples low-level visual perception from high-level spatio-temporal reasoning. By shifting the paradigm from passive curation to active synthesis, VGenST-Bench enables fine-grained diagnosis of spatio-temporal understanding in MLLMs.

**Analysis:**

### 1. 摘要翻译
时空推理是多模态大语言模型（MLLMs）在真实世界中运作的核心能力，对其进行精确评估已成为一项关键挑战。然而，现有的时空推理基准测试主要依赖于静态图像集或被动采集的视频数据，这限制了对细粒度推理能力的评估。本文提出了 **VGenST-Bench**，这是一个利用生成模型主动合成高度受控且多样化评估场景的视频基准测试。为了构建该基准，我们提出了一个包含人类质量控制阶段的多智能体流水线，确保了所有生成的视频和问答对的质量。我们建立了一个 3 × 2 × 2 的视频分类体系，涵盖空间尺度、视角和场景动态，以跨越多种场景。此外，我们设计了一个分层任务套件，将底层视觉感知与高层时空推理进行了解耦。通过将范式从被动采集转向主动合成，VGenST-Bench 为 MLLMs 的时空理解能力提供了细粒度的诊断。

### 2. 方法动机分析
- **驱动力**：旨在克服现有基准测试由于“被动采集”导致的评估不精确和数据污染问题。
- **痛点**：
  1. **数据污染**：MLLMs 预训练数据量巨大，被动采集的基准测试极易出现 train-test 重叠，导致性能虚高。
  2. **捷径挖掘（Shortcut Exploitation）**：现有视频数据包含 distributional regularities（分布规律），模型可通过静态帧或伪相关性而非真正的时空逻辑来“投机取巧”。
  3. **可扩展性差**：从网络采集数据难以覆盖特定的时空推理场景，缺乏主动设计的灵活性。
- **核心假设**：主动合成的、高度可控的视频场景是评估 MLLMs 时空推理能力最可靠的测试床。

### 3. 方法设计详解
VGenST-Bench 的构建核心是一个多智能体生成流水线，旨在保证逻辑一致性和任务的解耦性：
1. **场景图生成（Scene Graph Agent）**：根据设定的任务类型和主题，生成描述实体、属性及空间关系的 JSON 格式结构化场景图。包含 Validator 循环，通过多轮迭代修正以满足任务约束。
2. **场景设计（Scenario Agent）**：基于场景图生成时间轴（timeline），明确“Setup -> Event -> Result”的逻辑演变。其 Validator 负责确保该时间轴能通过观察导出唯一的正确答案。
3. **视频渲染（Video Agent）**：
   - **分层提示（Prompting）**：使用“图像提示词转换器”生成Anchor Frame（首帧），再结合“视频提示词转换器”生成描述动作的 I2V（Image-to-Video）提示词。
   - **设计优势**：通过图像锚定减少视频生成中的视觉漂移，确保内容与场景图完全对应。
4. **问答生成（QA Agent）**：基于 task-QA 适用性矩阵，确保问答设计与视频内容严格相关。
5. **人类质控**：引入两阶段验证机制（视频质控 + 问答质控），只有通过双重验证的视频和问答才被纳入测试集。

### 4. 方法对比分析
- **本质区别**：VGenST-Bench 不再依赖外部网络视频采集，而是利用先进的生成模型“定制”评估样本。
- **创新贡献**：引入了基于生成式 AI 的“闭环验证管道”，实现了从逻辑场景描述到视觉呈现的完全可控。此外，通过构建不同改革变体（V1/V2/V3）对抗模型在 MCQ 中的投机心理。
- **适用场景**：不仅适用于通用的多模态模型评估，还特别适用于机器人、自动驾驶等需要精确时空推理的物理领域模型诊断。

### 5. 实验分析（精简版）
- **实验结论**：MLLMs 的性能随着推理难度（L1感知 -> L2理解 -> L3推理）的增加发生剧烈退化。
- **主要优势**：构建了一个不受数据污染影响、具有人类近乎完美基准（99%）的评估系统。
- **主要局限**：由于生成模型本身存在视觉先验，模型在这些合成数据上的表现未必能完全映射到自然环境中的真实推理性能。

### 6. 实用指南
- **开源/获取**：项目已开源（https://zinosii.github.io/VGenST-Bench/）。
- **注意点**：构建流程中的 Validator 对话逻辑至关重要，建议在迁移到其他任务时，务必定义清晰的逻辑约束，防止模型产生幻觉。
- **迁移建议**：该管道可直接迁移至机器人任务，只需替换 Scene Graph 生成中的 Task Definition，定义特定的物体互动规则即可。

### 7. 总结
- **核心思想**：通过主动生成高可控的视频场景，打造去捷径化的时空推理基准测试。
- **速记版pipeline**：
  1. 生成逻辑结构图；
  2. 设计演变时间轴；
  3. 锚定首帧并渲染视频；
  4. 依据逻辑生成问答；
  5. 严格人工质控筛选。

**Key Findings:**

- In this paper, we introduce VGenST-Bench, a video benchmark that employs generative models to actively synthesize highly controlled and diverse evaluation scenarios.
- To construct VGenST-Bench, we propose a multi-agent pipeline incorporating a human quality control stage, ensuring the quality of all generated videos and QA pairs.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.22570v1)
- [arXiv](https://arxiv.org/abs/2605.22570v1)

---

