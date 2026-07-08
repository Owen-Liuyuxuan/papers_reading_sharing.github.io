time: 20260708

# Arxiv Computer Vision Papers - 2026-07-08

## Executive Summary

## 执行摘要

本报告汇总了2026年7月7日arXiv上发表的10篇计算机视觉论文，涵盖多模态生成、具身智能、视频扩散、SLAM数据集等方向。以下提炼关键趋势、亮点及阅读建议。

### 1. 主要主题与趋势

- **多模态生成统一化**：多篇工作尝试将视觉任务纳入统一的生成框架（如第1篇、第8篇），利用大规模文本到图像模型解决密集预测问题，显示出“生成即理解”的范式潜力。
- **具身智能与世界模型**：机器人领域持续活跃，出现多个基于世界模型的操控与遥操作系统（第2、6篇），以及面向开放世界的导航与规划框架（第9、10篇）。模型开始融入4D时空信息与动力学约束。
- **视频生成向移动端迁移**：第4篇专门针对移动设备优化视频扩散模型，旨在缩小与桌面级质量的差距，体现边缘部署需求。
- **基准与数据集创新**：第3篇发布了一个包含360°视觉惯性数据和楼层平面图先验的SLAM新基准，有望推动定位算法的实用化进步。

### 2. 特别重要或创新的论文

- **第1篇《Vision as Unified Multimodal Generation》**：提出将视觉任务（检测、分割、生成等）全部统一为多模态生成问题，若方法有效，可能改变传统视觉架构设计。
- **第5篇《Lift3D-VLA》**：将视觉-语言-动作（VLA）模型提升至显式3D几何与动力学感知，解决了现有VLA模型缺乏物理世界理解的关键缺陷，对机器人操作有直接推动作用。
- **第8篇《From RGB Generation to Dense Field Readout》**：开创性地利用冻结的文本到图像扩散模型直接进行像素级密集预测（如深度、语义），为无需微调的零样本下游任务提供了新思路。
- **第3篇数据集**：结合高精度惯性测量、360°视觉与建筑平面图先验，填补了室内SLAM基准中结构与先验信息的空白，预计将成为定位领域的重要评估平台。

### 3. 新兴研究方向或技术

- **4D世界模型**：第2篇引入时间维度（4D）的具身世界模型，使机器人能预测动态场景演变，是传统3D世界模型的自然延伸。
- **动作条件世界模型**：第6篇将机器人动作显式编码为世界模型的条件，实现更可控的数字遥操作，为远程操控提供了新范式。
- **生成模型驱动密集预测**：第8篇所示，利用预训练文本到图像模型在像素空间做密集输出，可能孕育出一类“无需任务专用头”的通用视觉理解方法。
- **零样本导航与开放世界规划**：第9、10篇分别探索在未见环境中零样本导航以及不确定性下的假设驱动规划，反映了具身AI对泛化性和适应性的更高要求。

### 4. 建议全文阅读的论文

- **若关注多模态统一与生成范式**：必读第1篇和第8篇，两者互补地展示从生成到理解的路径。
- **若专注机器人操作与具身智能**：优先阅读第5篇（VLA三维化）和第2篇（4D世界模型），它们代表了当前端到端操控的前沿。
- **若从事SLAM或定位研究**：第3篇数据集论文是值得精读的基准资源，尤其关注其平面图先验的融合方式。
- **若对移动端视频生成感兴趣**：第4篇提供了实用的质量提升技术，可快速了解当前差距与解决方案。

这些论文共同指向一个趋势：**视觉正从“识别-预测”范式向“生成-推理”范式过渡，且与机器人具身系统的结合日益紧密**。建议根据研究方向选择性深入阅读。

---

## Table of Contents

1. [Vision as Unified Multimodal Generation](#2607.06560v1)
2. [RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation](#2607.06559v1)
3. [Hilti-Trimble-Oxford Dataset: 360 Visual-Inertial Benchmark with Floor Plan Priors for SLAM and Localization](#2607.06464v1)
4. [MobileWan: Closing the Quality Gap for Mobile Video Diffusion](#2607.06173v1)
5. [Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation](#2607.06564v1)
6. [RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation](#2607.06558v1)
7. [ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation](#2607.06555v1)
8. [From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models](#2607.06553v1)
9. [UniLM-Nav: A Unified Framework for Zero-Shot Last-Mile Navigation](#2607.06537v1)
10. [Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning](#2607.06501v1)

---

## Papers

<a id='2607.06560v1'></a>
## [Vision as Unified Multimodal Generation](https://arxiv.org/abs/2607.06560v1)

**Authors:** Xiaoyang Han, Jianhua Li, Kewang Deng, Zukai Chen, Xuanke Shi, Sihan Wang, Boxuan Li, Linyan Wang, Siyi Xie, Xin You, Jinsheng Quan, Zhongang Cai, Haiwen Diao, Ziwei Liu, Lei Yang, Dahua Lin, Quan Wang

**Published:** 2026-07-07

**Categories:** cs.CV

**Abstract:**

We formulate computer vision as unified multimodal generation, where heterogeneous visual tasks are expressed in the native text and image generation spaces of a unified multimodal model, without task-specific architectures. Under this formulation, SenseNova-Vision uses natural-language instructions and optional visual prompts to specify tasks, target regions or views, and decoding conventions, and generates responses as text for symbolic outputs, images for dense spatial predictions, or mixed text-and-image outputs for compositional tasks. To support large-scale training, we convert diverse computer vision annotations into instruction-response examples compatible with these generation spaces, resulting in the SenseNova-Vision Corpus, a computer-vision instruction-response corpus spanning text, image, and mixed targets. Starting from an off-the-shelf pretrained unified multimodal model, SenseNova-Vision is trained primarily on this corpus, with auxiliary multimodal data used as a capability-preserving mixture, and requires no task-specific prediction heads or architectural modifications. The resulting model covers a broad range of vision tasks, including detection, OCR, keypoint estimation, segmentation, depth estimation, surface normal prediction, point maps, and camera pose estimation, while supporting language-defined variants that combine category, color, region, and other visual cues. Experiments show that a single unified model can match leading task-specialized systems across structured visual understanding, dense geometric prediction, segmentation, and multi-view visual geometry. These results suggest unified multimodal generation as a scalable route for integrating computer vision capabilities into general-purpose foundation models. The model and corpus are publicly available.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Vision as Unified Multimodal Generation》（将计算机视觉统一为多模态生成）的论文分析如下：

### 1. 论文核心贡献总结
该论文提出了一种将各类计算机视觉任务统一建模为“多模态生成”任务的范式，通过自然语言指令和视觉提示，使单一模型能够处理从符号化输出（如检测）到密集空间预测（如深度估计、分割）乃至组合任务的所有视觉需求。这一框架彻底摒弃了针对特定任务的架构设计（Task-specific heads），通过构建大规模的“SenseNova-Vision”语料库，证明了仅靠统一的生成式路径即可实现与各类垂直领域专家系统相当的性能。

### 2. 关键创新与方法论
*   **统一生成范式（Unified Generation Space）：** 不同于传统的判别式模型（预测坐标或类别），该模型将所有视觉任务映射到“文本”或“图像”空间。例如，检测任务可生成文本描述，分割任务可生成掩码图像。
*   **指令微调方法（Instruction-based Interaction）：** 通过引入自然语言指令和视觉提示（Visual Prompts），模型能够灵活响应用户关于区域、视角、类别及颜色等细粒度约束的请求。
*   **架构无关性（Architecture Agnostic）：** 该模型无需修改预训练的多模态大模型底座，也无需增加特定的解码头（Prediction heads），这极大降低了系统的复杂性，提升了通用性。
*   **SenseNova-Vision Corpus：** 作者将传统的标注数据集（如COCO、ADE20K等）转化为了统一的指令-响应对格式，为领域内的多模态统一训练提供了重要的数据工程范式。

### 3. 对领域的潜在影响
*   **范式转移（Paradigm Shift）：** 这篇论文代表了从“专用模型（Specialized Model）”向“通用视觉模型（General-Purpose Foundation Model）”的彻底转变。它挑战了“计算机视觉需要深度定制化架构”的传统观念。
*   **可扩展性（Scalability）：** 证明了视觉理解任务可以像大语言模型一样通过扩充数据和统一格式来实现能力跃迁，这为构建“视觉-语言通用基础模型”提供了明确的可行路径。
*   **降低研发成本：** 将原本需要维护数十个不同检测、分割和深度模型的工作量，缩减为一个统一的模型，大幅简化了部署流程。

### 4. 相关领域与应用价值
*   **智能体（Agents）：** 这种基于指令的视觉生成模型是具身智能和视觉智能体的完美底座，能更好地理解并执行复杂的物理世界操作指令。
*   **多任务综合系统：** 如自动驾驶场景，可以在一个模型内同时完成车道线检测、路面深度预估和交通标志识别，无需多个分支模型协作。
*   **人机交互：** 使得用户能够通过对话式语言直接对图像内容进行“编辑、标注、分析”，极大降低了专业视觉分析的门槛。

### 5. 可推断的局限性
*   **实时性挑战：** 既然所有任务都转化为生成任务，其推理延迟可能远高于传统轻量级的判别式模型（如YOLO系列），在对延迟极度敏感的场景（如工业质检、高速自动驾驶）中可能存在瓶颈。
*   **幻觉与一致性：** 生成式模型固有的“幻觉”问题（在视觉输出中产生不符合事实的细节）在医疗或安防等高精度领域可能带来挑战。
*   **密集型任务的精度上限：** 相比于专门针对像素级分类优化过的架构，生成式路径在极精细的边界分割或极小物体检测上，是否能长期保持竞争力仍有待观测。
*   **数据合成成本：** 虽然构建了SenseNova-Vision Corpus，但将海量历史视觉数据转换为高质量指令对，涉及复杂的数据清洗和逻辑转换，这对后续研究者的复现工作提出了较高门槛。

**总结：** 该论文是视觉基础模型发展史上的一个标志性作品。它不仅展示了多模态大模型在视觉任务上的惊人适配性，还提供了一种将视觉世界“语言化/生成化”的可行蓝图，具有极高的学术意义和工业应用价值。

**Key Findings:**

- These results suggest unified multimodal generation as a scalable route for integrating computer vision capabilities into general-purpose foundation models.
- The model and corpus are publicly available.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06560v1)
- [arXiv](https://arxiv.org/abs/2607.06560v1)

---

<a id='2607.06559v1'></a>
## [RynnWorld-4D: 4D Embodied World Models for Robotic Manipulation](https://arxiv.org/abs/2607.06559v1)

**Authors:** Haoyu Zhao, Xingyue Zhao, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li

**Published:** 2026-07-07

**Categories:** cs.RO

**Abstract:**

Robotic manipulation in the open world requires not only recognizing what a scene looks like, but also anticipating how its 3D structure moves under interaction. We argue that synchronized RGB, depth, and optical flow, namely RGB-DF, provide a physically grounded representation that captures the underlying 4D dynamics of a scene. Compared to 2D pixel videos, this multi-modal synergy aligns visual appearance with geometric structure and temporal motion, creating a representation space significantly closer to the low-level end-effector actions demanded by robotic systems, thereby narrowing the gap between world prediction and policy learning. Building on this insight, we introduce RynnWorld-4D, a generative model that co-produces future RGB frames, depth maps, and optical flow from a single RGB-D image and a language instruction within one unified diffusion process. This 4D world model features a tri-branch architecture that integrates cross-modal attention with frame-wise 3D RoPE, ensuring that appearance, geometry, and motion evolve consistently. To supply training data at scale, we curate Rynn4DDataset 1.0, a massive dataset of over 254.4 million frames across egocentric human and robotic manipulation videos with high-quality pseudo-labels for depth and optical flow. We further propose RynnWorld-4D-Policy, an inverse dynamics head that consumes the internal 4D representations of RynnWorld-4D in a single forward pass, bypassing expensive multi-step denoising, to output robot actions in a closed-loop manner. Experiments show that RynnWorld-4D produces temporally and spatially coherent 4D predictions, and that RynnWorld-4D-Policy achieves state-of-the-art performance on real-world dexterous bimanual manipulation tasks, particularly excelling in tasks demanding spatial precision and temporal coordination.

**Analysis:**

### 1. 摘要翻译
开放世界中的机器人操作不仅需要识别场景，还需要预测物体在交互下的3D运动轨迹。我们认为同步的RGB、深度和光流（RGB-DF）提供了一种具备物理基础的表示方法，能捕捉场景的4D动态。与传统的2D像素视频相比，这种多模态协同将视觉外观与几何结构和时间运动对齐，使表示空间更接近机器人控制所需的动作空间，从而缩小了世界预测与策略学习之间的差距。为此，我们引入了 **RynnWorld-4D**，这是一个在一个统一扩散过程中，从单张RGB-D图像和语言指令共同生成未来RGB帧、深度图和光流的生成模型。该模型采用三分支架构，整合了跨模态注意力和帧间3D RoPE，确保外观、几何和运动的演变保持一致。为支持大规模训练，我们构建了 **Rynn4DDataset 1.0**，包含超过2.54亿帧数据。此外，我们提出 **RynnWorld-4D-Policy**，这是一种通过单次前向传递直接提取内部4D表示并输出机器人动作的逆动力学头，绕过了多步去噪过程。实验表明，RynnWorld-4D能生成时空连贯的4D预测，且RynnWorld-4D-Policy在真实世界的双臂操作任务中达到了最先进的性能，特别是在需要空间精度和时间协调的任务中表现优异。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决现有2D视频生成模型在机器人操作中由于缺乏几何感知和物理约束，导致无法实现高精度实时控制的痛点。
*   **现有方法痛点**：现有生成模型主要基于2D像素空间，导致深度歧义和几何结构缺失；而现有的4D建模（如NeRF/3DGS）虽然几何明确，但难以利用大规模视频生成模型的先验，且计算成本过高。
*   **研究假设**：通过引入RGB-DF（RGB+深度+光流）作为一种“轻量化投影4D表示”，既能保留大规模生成模型的缩放能力，又能显式定义物理运动和几何结构。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **输入**：单张RGB-D图像 + 任务文本。
    2.  **生成**：通过三分支Transformer扩散模型，同步预测未来序列的RGB、深度图和光流。
    3.  **表示**：利用深度图将像素提升至3D空间，通过光流建立帧间对应关系，计算出“3D场景流”。
    4.  **决策**：RynnWorld-4D-Policy直接利用冻结的生成模型中间层特征，通过“流形式（Flow Former）”进行特征压缩，并结合ODE求解器输出动作序列。
*   **模型结构**：
    *   **三分支架构**：RGB、深度、光流各占一个分支，通过“联合跨模态注意力（JA）”模块实现特征对齐。
    *   **JA模块**：每三层插入，使用3D RoPE注入位置信息，实现跨模态一致性。
    *   **逆动力学头**：绕过传统的反复去噪，通过单步前向传播提取时空动态特征，实现高频控制。
*   **关键公式意义**：$f_{3D} = P_{t+1} - P_t$。该公式将光流转化为度量场景流，将视觉感知直接锚定为物理位移，确保生成的轨迹符合物理规律。

---

### 4. 方法对比分析
*   **本质区别**：与仅基于外观预测的2D模型不同，它显式建模了运动场和深度，将“生成任务”转化为“物理动力学推断任务”。
*   **创新贡献**：提出了一种与大模型兼容的轻量化4D投影表示，且实现了通过单步推理进行闭环机器人控制。
*   **适用场景**：高精度操作、双臂协调、复杂物体推移等对空间坐标敏感的机器人任务。

---

### 5. 实验分析
*   **验证方法**：在6种真实世界机器人任务（双臂Picking、Lid Placement等）和大规模数据集上进行对比测试。
*   **关键结果**：在Lid Placement和Bowl Stacking等高精度任务中，成功率比最佳基线高出8.57%。
*   **优势**：显式几何先验提升了鲁棒性；动作切片（Action Chunking）技术使其能达到9Hz的有效控制频率。
*   **局限**：扩散模型的前向推理仍需计算资源支持，尚无法达到超高频（如500Hz）实时响应，且目前主要针对自中心视角。

---

### 6. 实用指南
*   **开源/实现**：项目已开源（GitHub链接可见），实现时需注意：使用预训练的Wan 2.2-TI2V-5B作为主干，阶段性训练（模态适应 -> 联合注意力 -> 全参数精调）对于收敛至关重要。
*   **迁移可能**：该框架的“三分支+JA模块”结构可直接迁移至任何多模态生成任务（如视频+深度+音频），只需替换对应领域的模态编码器即可。

---

### 7. 总结
*   **核心思想**：通过RGB-DF同步生成，将像素预测转化为可解释的4D物理场景演变。
*   **速记版Pipeline**：
    1. 输入单帧RGB-D并给定任务描述。
    2. 三分支Transformer生成一致的视觉与运动场。
    3. 将生成数据投影为几何与运动特征。
    4. 策略头通过单步推理计算动作序列。
    5. 执行动作并动态更新环境状态。

**Key Findings:**

- Building on this insight, we introduce RynnWorld-4D, a generative model that co-produces future RGB frames, depth maps, and optical flow from a single RGB-D image and a language instruction within one unified diffusion process.
- Experiments show that RynnWorld-4D produces temporally and spatially coherent 4D predictions, and that RynnWorld-4D-Policy achieves state-of-the-art performance on real-world dexterous bimanual manipulation tasks, particularly excelling in tasks demanding spatial precision and temporal coordination.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06559v1)
- [arXiv](https://arxiv.org/abs/2607.06559v1)

---

<a id='2607.06464v1'></a>
## [Hilti-Trimble-Oxford Dataset: 360 Visual-Inertial Benchmark with Floor Plan Priors for SLAM and Localization](https://arxiv.org/abs/2607.06464v1)

**Authors:** Samuele Centanni, Yuhao Zhang, Yifu Tao, Julien Kindle, Frank Neuhaus, Tilman Koß, Aryaman Patel, Michael Helmberger, Emilia Szymańska, Torben Gräber, Maurice Fallon

**Published:** 2026-07-07

**Categories:** cs.RO

**Abstract:**

Automated progress monitoring on construction sites is an active area of research and development. Robot and human-carried mapping systems have been developed to build 3D maps of building and infrastructure projects. While LiDAR-based mapping systems achieve high accuracy, the cost of LiDAR can be prohibitive. Consumer-grade cameras with wide field of view ("360 cameras") combined with embedded inertial measurement units (IMUs) provide a cost-effective alternative. To support change detection and progress monitoring, highly accurate visual Simultaneous Localization and Mapping (SLAM) and floor plan-referenced localization systems are required. In this paper we present a high-quality dataset collected at an active construction site, which captures realistic challenges such as variable lighting conditions, moving workers, fast motions, and repetitive structures. The dataset offers thirty visual-inertial sequences recorded across seven floors over an eight-month period of the construction project. Ground truth trajectories were collected using a high quality LiDAR-inertial SLAM system rigidly attached to the 360 camera. Additionally, we report the results of an open research challenge evaluating the best visual SLAM and localization systems from around the world. The Challenge attracted substantially higher participation in SLAM, with 62 teams compared to 22 in floor-plan-referenced localization, reflecting the broader maturity of SLAM methods. The higher errors in localization further highlight the difficulty of this task in construction and point to the need for continued research, which this dataset is intended to support. The dataset and the benchmark are publicly available at: https://hilti-trimble-challenge.com/dataset-2026.

**Analysis:**

### 1. 摘要翻译
建筑工地的自动进度监控是一项活跃的研究领域。机器人和人工携带的测绘系统已被开发用于构建建筑和基础设施项目的3D地图。虽然基于LiDAR的测绘系统实现了高精度，但其成本往往令人望而却步。具有宽广视野的消费级相机（“360度相机”）结合内置惯性测量单元（IMU）提供了一种低成本的替代方案。为支持变更检测和进度监控，需要高精度的视觉同步定位与建图（SLAM）及以平面图为参考的定位系统。本文介绍了一个在活跃建筑工地采集的高质量数据集，捕捉了可变照明条件、移动工人、快速运动和重复结构等现实挑战。该数据集涵盖了施工期间八个月内、跨越七个楼层的30个视觉惯性序列。地面真值轨迹是使用与360度相机刚性连接的高质量LiDAR惯性SLAM系统采集的。此外，我们报告了来自全球范围的视觉SLAM和定位系统的公开研究挑战赛结果。

### 2. 方法动机分析
- **核心驱动力**：解决建筑工地自动化测绘中，依赖昂贵LiDAR设备的局限性，探索利用低成本360度视觉传感器进行高效测绘的可行性。
- **现有痛点**：当前视觉SLAM方法难以应对建筑工地的特殊挑战（如重复纹理、极端光照、严重遮挡）；此外，缺乏将轨迹实时对齐到建筑平面图（BIM的简化版）的成熟低成本方案。
- **研究假设**：通过引入包含 floor plan（平面图）作为先验的复杂场景数据集，可以推动视觉SLAM与语义定位算法在极端工业环境下的鲁棒性提升。

### 3. 方法设计详解
本研究并非提出单一算法，而是构建了一个完整的评估框架与挑战赛体系：
- **数据采集与同步**：使用Insta360 ONE RS 1-Inch 360相机作为主要传感器，通过软件层面实现LiDAR与相机时钟的线性映射，以克服无硬件同步的难题。
- **地面真值构建（Ground Truth Pipeline）**：
  1. **两阶段优化**：先进行在线MC2SLAM估计，再进行离线大规模非线性联合优化。
  2. **LiDAR-相机外参标定**：采用6x6 AprilTag板，结合重投影误差和点面距离，通过非线性最小二乘法联合估计位姿。
  3. **平面图对齐**：先手动粗对齐，再利用ICP算法进行细化，确保LiDAR点云与建筑平面图的高精度贴合。
- **评价协议**：
  - **SLAM track**：利用Kabsch算法对齐轨迹，评估3D位姿误差。
  - **Localization track**：直接在平面图空间评估2D位姿误差，强制模型理解平面图约束。

### 4. 方法对比分析
- **本质区别**：从“仅SLAM”转变为“SLAM+语义定位”，重点在于将视觉输入与结构化的2D平面图先验挂钩。
- **创新点**：引入建筑生命周期中结构变化的动态环境数据集；强制模型利用平面图进行无初始化的全局定位，而非简单的相对里程计计算。
- **适用场景**：复杂、动态、弱纹理的室内建筑工地环境。

### 5. 实验分析
- **关键结论**：语义分割在Localization track中至关重要。排名前三的方案均使用了墙体语义提取，证实了通过提取稳定结构特征（墙体）而非依赖几何点云能够显著提升对齐精度。
- **优势**：数据集真实度极高，包含了长达8个月的施工演变，对算法的长期鲁棒性测试极佳。
- **局限**：存在严重的“地图-建筑不匹配”问题（建筑结构在测绘后可能发生改动），这要求算法具备处理陈旧先验图的能力。

### 6. 实用指南
- **开源情况**：数据集及挑战赛结果已开源（https://hilti-trimble-challenge.com/dataset-2026）。
- **迁移建议**：若要复现，重点在于对不同传感器的校准工具（如Kalibr）的熟练运用；若迁移到其他任务，建议重点借鉴论文中关于“将离散的视觉特征与结构化CAD平面图对齐”的后处理范式。

### 7. 总结
- **核心思想**：利用平面图先验提升视觉SLAM在复杂施工现场的定位精度与稳健性。
- **速记版pipeline**：
  1. 数据采集与传感器校准；
  2. 高精度LiDAR轨迹生成（Ground Truth）；
  3. 结构特征点与建筑平面图的多阶段对齐；
  4. 综合定位误差与轨迹一致性评价。

**Key Findings:**

- In this paper we present a high-quality dataset collected at an active construction site, which captures realistic challenges such as variable lighting conditions, moving workers, fast motions, and repetitive structures.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06464v1)
- [arXiv](https://arxiv.org/abs/2607.06464v1)

---

<a id='2607.06173v1'></a>
## [MobileWan: Closing the Quality Gap for Mobile Video Diffusion](https://arxiv.org/abs/2607.06173v1)

**Authors:** Mohsen Ghafoorian, Denis Korzhenkov, Adil Karjauv, Ioannis Lelekas, Noor Fathima, Spyridon Stasis, Hanno Ackermann, Boris van Breugel, Markus Nagel, Fatih Porikli, Animesh Karnewar, Amirhossein Habibian

**Published:** 2026-07-07

**Categories:** cs.CV

**Abstract:**

Recent advances in video diffusion have been driven by scaling transformer-based architectures to billions of parameters, substantially improving visual fidelity and motion coherence. In contrast, existing mobile video diffusion models remain limited to relatively small parameter budgets, typically 0.4-1.8B, restricting generation quality. In this work, we show that high-quality mobile video generation does not require small models. Instead, we demonstrate that a server-scale 5B-parameter video diffusion transformer can be deployed efficiently on memory-constrained mobile hardware through recurrent reformulation and structured compression. Starting from Wan2.2-5B, we rely on a recurrence distillation framework that converts video generation into a chunk-wise autoregressive process with constant-memory attention computation. Combined with causal linear attention, the model operates as an RNN at inference time while preserving temporal coherence across chunks. We further propose a learnable attention head pruning method based on binary per-head gates optimized end-to-end using a noise-biased sparsity objective and distillation-based finetuning. Together with sampling-step distillation and memory-optimized VAE decoding, MobileWan becomes the first 5B-scale video diffusion model deployable on a commercial mobile device. Our system generates 5-second 480x832 videos at 16 FPS in 20 seconds end-to-end latency, achieving a VBench score of 83.79 and establishing a new state of the art in mobile video generation. Project page: https://qualcomm-ai-research.github.io/mobilewan

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对 **MobileWan** 这篇论文的分析如下：

### 1. 论文核心贡献总结
MobileWan 提出了一种将 50 亿参数（5B）量级的视频扩散模型成功部署于受限移动设备上的解决方案，打破了移动端视频生成模型长期受限于参数规模（<2B）导致的质量瓶颈。通过递归重构（Recurrent Reformulation）和结构化压缩技术，该研究在保持高质量生成效果的同时，实现了移动端高效的实时推理，设定了移动端视频生成的全新技术标杆（SOTA）。

### 2. 关键创新与方法论
该论文的创新点在于平衡了“大模型质量”与“移动端资源约束”之间的矛盾，具体包括：
*   **递归重构与分块自回归（Recurrent Reformulation & Chunk-wise Autoregression）：** 将视频生成过程转化为 chunk-wise 的自回归过程，并结合因果线性注意力机制（Causal Linear Attention），使得模型在推理时呈现 RNN 的行为特征。这保证了在恒定内存占用下生成长视频，并维持了良好的时间连贯性。
*   **基于二进制门的结构化剪枝：** 提出了一种学习型的注意力头剪枝方法，通过噪声偏差稀疏性目标（Noise-biased Sparsity Objective）进行端到端优化，在保留核心特征提取能力的同时最大程度降低算力需求。
*   **全流程优化链：** 结合了采样步数蒸馏（Sampling-step Distillation）和显存优化版 VAE 解码器，构建了一个从训练到推理的完整高效管道。

### 3. 对领域的潜在影响
*   **范式转换：** 该研究挑战了“移动端必须使用轻量化小模型”的固有认知，证明了通过高效架构设计，即使是 5B 参数级别的大模型也能在移动端落地。
*   **移动端 AI 民主化：** 这一成果使得复杂、高保真的 AIGC 功能（如长视频生成）不再局限于云端或高性能服务器，推动了生成式 AI 在终端设备（On-device AI）的普及，显著增强了用户隐私保护和离线交互体验。

### 4. 受益的相关领域与应用
*   **移动端创作工具：** 短视频应用、社交媒体插件可以直接集成高质量的 AI 视频生成功能，实现“秒级”视频创作。
*   **边缘计算与端侧推理：** 对那些需要在存储和计算能力受限的情况下执行大规模神经网络推理的场景（如 AR/VR、车内娱乐系统）具有极高的参考价值。
*   **计算摄影：** 为移动端视频编辑、风格迁移以及动态视频补全提供了一种更强大的算力引擎。

### 5. 可推断的局限性
*   **生成延迟的权衡：** 尽管实现了 20 秒生成 5 秒视频，但对于极度敏感的实时应用（如直播特效），这一延迟可能仍有优化空间。
*   **算力依赖性：** 尽管在“商业移动设备”上可运行，但仍对移动端的 NPU/GPU 算力有特定要求，可能无法在旧款或低端芯片上保持该性能。
*   **长序列累积误差：** 尽管采用了递归和自回归设计，但在极长视频生成中，随着 chunk 的增加，误差累积或时间一致性偏移（Temporal Drift）仍可能是潜在挑战。

---

**专家点评：** 
MobileWan 的有趣之处在于其**“向下兼容大模型”**的思路，而非单纯的“向上优化小模型”。通过将视频扩散模型转化为类似 RNN 的线性复杂度结构，该团队巧妙地绕过了 Transformer 推理中的显存峰值问题，这对于目前所有试图在边缘设备上部署大语言模型（LLM）或视频大模型的团队都具有重要的启示作用。

**Key Findings:**

- In this work, we show that high-quality mobile video generation does not require small models.
- Instead, we demonstrate that a server-scale 5B-parameter video diffusion transformer can be deployed efficiently on memory-constrained mobile hardware through recurrent reformulation and structured compression.
- Our system generates 5-second 480x832 videos at 16 FPS in 20 seconds end-to-end latency, achieving a VBench score of 83.79 and establishing a new state of the art in mobile video generation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06173v1)
- [arXiv](https://arxiv.org/abs/2607.06173v1)

---

<a id='2607.06564v1'></a>
## [Lift3D-VLA: Lifting VLA Models to 3D Geometry and Dynamics-Aware Manipulation](https://arxiv.org/abs/2607.06564v1)

**Authors:** Jiaming Liu, Qingpo Wuwu, Nuowei Han, Hao Chen, Zhuoyang Liu, Fan Fei, Yueru Jia, Chenyang Gu, Yandong Guo, Boxin Shi, Shanghang Zhang

**Published:** 2026-07-07

**Categories:** cs.RO, cs.CV

**Abstract:**

Recently, Vision-Language-Action (VLA) models have demonstrated strong generalization across diverse tasks. However, effective robotic manipulation in physical environments fundamentally requires geometric understanding and spatial reasoning. While some VLA approaches attempt to incorporate 3D information, they are constrained by limited data availability and geometric information loss in current 3D encoding pipelines, and fail to jointly capture 3D geometry and temporally structured actions in dynamic environments. To address these limitations, we introduce Lift3D-VLA, a unified VLA framework that equips models with explicit 3D point cloud reasoning and enables temporally coherent action generation. First, building upon our previous work Lift3D, an enhanced 2D model-lifting strategy is proposed to geometrically align 3D points with pretrained 2D positional embeddings. This design enables direct point-cloud encoding within the VLA vision encoder while minimizing spatial information loss. Based on explicit 3D inputs, we propose Geometry-Centric Masked Autoencoding (GC-MAE), a dual-objective self-supervised framework that reconstructs the current point cloud while predicting its future geometric evolution. This formulation allows the 2D vision encoder to internalize both 3D structure and physical dynamics. To fully exploit 3D representations, we further design layer-wise temporal action modeling, which leverages multiple layers of the LLM to collaboratively predict action chunks, enabling temporally consistent predictions. Across 22 simulated tasks and 8 real-world manipulation tasks, Lift3D-VLA achieves 10.8% and 11.1% higher mean success rates on MetaWorld and RLBench than the best-performing prior VLA methods, and outperforms the strongest real-world baseline by 4 percentage points, while exhibiting stronger generalization to out-of-distribution perturbations.

**Analysis:**

### 1. 摘要翻译
近日，视觉-语言-动作（VLA）模型在各种任务中展现了强大的泛化能力。然而，物理环境下的机器人操作本质上需要几何理解和空间推理能力。现有的VLA方法在整合3D信息时受到数据稀缺和3D编码过程中几何信息丢失的约束，且难以在动态环境中同时捕捉3D几何与时间结构化的动作。为了解决这些局限性，我们提出了Lift3D-VLA，这是一个统一的VLA框架，能够进行显式的3D点云推理并实现时间一致的动作生成。首先，基于我们之前的Lift3D工作，我们提出了一种增强的2D模型提升策略，将3D点与预训练的2D位置嵌入进行几何对齐，从而在最大限度减少空间信息损失的同时，在VLA视觉编码器中实现直接的点云编码。基于显式3D输入，我们进一步提出了几何中心掩码自编码（GC-MAE），这是一个双目标自监督框架，在重构当前点云的同时预测其未来的几何演变，使2D视觉编码器能够内化3D结构与物理动态。此外，为了充分利用3D表示，我们设计了逐层时间动作建模，利用大语言模型（LLM）的多个层协作预测动作块，从而实现时间上一致的预测。在22个模拟任务和8个真实世界操作任务中，Lift3D-VLA在MetaWorld和RLBench上的平均成功率分别比现有的最优VLA方法高出10.8%和11.1%，且在真实世界基准测试中优于最强基线，同时展现出更强的分布外泛化能力。

---

### 2. 方法动机分析
- **驱动力**：解决VLA模型在复杂动态环境中因缺乏几何感知而导致的动作脆性问题。
- **痛点**：现有3D VLA方法要么依赖庞大且稀缺的3D编码器，要么通过损失精度的跨模态投影（如2D到3D或3D到2D）进行转换，导致几何保真度下降。此外，现有模型未能有效联合处理3D几何演变与动作的时间连续性。
- **研究假设**：通过显式对齐点云与预训练的2D位置嵌入，并结合动态演变的目标函数，可以弥合2D foundation model与3D物理世界之间的鸿沟，同时利用LLM的层级特征实现时间序列预测。

---

### 3. 方法设计详解
- **2D模型提升（2D Model-lifting）**：
  - 将3D点云投影到多个虚拟平面（cube-based projection，6个面），利用相机外参确保视角一致性，减少失真。
  - 将投影坐标关联到预训练的2D位置嵌入，通过平均化这些嵌入得到统一的$PE_{3D}$，直接送入视觉编码器。
- **几何中心掩码自编码（GC-MAE）**：
  - **静态分支**：对3D点云Token进行掩码，通过Transformer解码器重建，利用Chamfer Distance监督，强制模型理解静态结构。
  - **动态分支**：基于当前可见Token，直接预测下一时刻点云，学习物理动态演变规律。
- **逐层时间动作建模（Layer-wise Temporal Action Modeling）**：
  - 改变以往仅使用LLM最后一层输出动作的范式。
  - 将动作块中的不同步骤$t+k$映射到LLM的特定中间层（如第20, 24, 28, 32层）。
  - 各层协作预测，使得深层特征能够attend到浅层状态，从而增强时间序列的一致性。

---

### 4. 方法对比分析
- **本质区别**：不训练独立的3D编码器，而是通过“2D-pretraining reuse”策略复用现有的视觉编码器，实现显式3D对齐。
- **创新点**：引入了预测几何演变的动态自监督目标，以及利用LLM不同深度层级特征进行长序列动作建模。
- **适用场景**：需要精准空间接触的物体操作，如倒水、组装、精细抓取。

---

### 5. 实验分析
- **验证方法**：在MetaWorld（模拟）和RLBench（多任务模拟）、真实世界Franka机器人平台上进行测试。
- **关键结果**：在MetaWorld上S.R.达到88.6%，显著优于现有方法；真实世界长程任务（如连续舀水、堆叠）表现出显著的稳定性和连贯性。
- **优势**：极佳的几何保真度、强大的OOD（分布外）泛化能力、对长 horizon 任务的更好支持。
- **局限**：对透明或高反射物体的深度感测仍然受限于RealSense相机的底层缺陷，导致点云生成存在噪声。

---

### 6. 实用指南
- **开源情况**：项目主页：https://lift3dvla.github.io/
- **实现细节**：建议冻结Backbone大部分参数，仅通过LoRA微调；GC-MAE的掩码比例设定为0.6效果最佳；训练时使用多视角深度信息进行点云合成。
- **迁移建议**：本方法可迁移至任何基于VLM的具身智能任务，只需替换相应任务的动作输出头，并利用其点云tokenizer模块。

---

### 7. 总结
- **核心思想**：通过3D对齐与动态预测增强VLM的几何与时间推理能力。
- **速记Pipeline**：
  1. 将点云投影至虚拟平面并对齐2D位置嵌入。
  2. 通过GC-MAE进行静态重建与动态演变预测预训练。
  3. 利用LLM多层级输出同时预测动作块序列。
  4. 采用LoRA进行高效微调，整合多传感器信息。

**Key Findings:**

- To address these limitations, we introduce Lift3D-VLA, a unified VLA framework that equips models with explicit 3D point cloud reasoning and enables temporally coherent action generation.
- Based on explicit 3D inputs, we propose Geometry-Centric Masked Autoencoding (GC-MAE), a dual-objective self-supervised framework that reconstructs the current point cloud while predicting its future geometric evolution.
- Across 22 simulated tasks and 8 real-world manipulation tasks, Lift3D-VLA achieves 10.8% and 11.1% higher mean success rates on MetaWorld and RLBench than the best-performing prior VLA methods, and outperforms the strongest real-world baseline by 4 percentage points, while exhibiting stronger generalization to out-of-distribution perturbations.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06564v1)
- [arXiv](https://arxiv.org/abs/2607.06564v1)

---

<a id='2607.06558v1'></a>
## [RynnWorld-Teleop: An Action-Conditioned World Model for Digital Teleoperation](https://arxiv.org/abs/2607.06558v1)

**Authors:** Haoyu Zhao, Xingyue Zhao, Hangyu Li, Biao Gong, Kehan Li, Siteng Huang, Xin Li, Deli Zhao, Zhongyu Li

**Published:** 2026-07-07

**Categories:** cs.RO

**Abstract:**

Scaling robot learning requires massive, diverse trajectory data, yet collection is currently bottlenecked by physical teleoperation, where every demonstration binds operator time to specific hardware and workspaces. We introduce digital teleoperation, a paradigm that decouples data collection from physical constraints by replacing the real robot with a generative world model. In this framework, an operator's hand-pose stream drives a robot-centric generative world model to synthesize high-fidelity egocentric videos from a single reference image. The recorded pose stream serves as an embodiment-agnostic action label transferable to any target robot via standard retargeting, yielding complete state-action trajectories for imitation learning independent of physical hardware. We instantiate this paradigm in RynnWorld-Teleop, a system that integrates depth-aware skeletal conditioning, progressive human-to-robot training on a video Diffusion Transformer, and streaming autoregressive distillation. This pipeline compresses the generative process into a single-pass inference, enabling 40+ FPS, real-time interactive generation on a single H100 GPU. Policies trained exclusively on RynnWorld-Teleop-generated data achieve effective zero-shot Sim2Real transfer across dexterous and diverse bimanual tasks. Moreover, augmenting real-world datasets with our digitally teleoperated data consistently improves success rates, demonstrating that RynnWorld-Teleop serves as a high-fidelity, scalable data engine for the next generation of robotic agents.

**Analysis:**

### 1. 摘要翻译
机器人学习的扩展受限于物理远程操作，该过程将操作员时间束缚在特定硬件和工作空间上。我们引入了“数字远程操作”（digital teleoperation），该范式通过用生成式世界模型取代真实机器人，从而解耦了数据收集与物理约束。在此框架下，操作员的手部姿态流驱动一个以机器人为中心的世界模型，从单张参考图像合成高保真第一人称视频。记录的姿态流作为与载体无关的动作标签，可通过标准重定向技术转移至任何目标机器人，从而生成独立于物理硬件的完整状态-动作轨迹。我们在 RynnWorld-Teleop 中实现了这一范式，该系统整合了深度感知骨骼调节、人类到机器人的渐进式视频扩散Transformer（DiT）训练以及流式自回归蒸馏。该流水线将生成过程压缩为单次推理，在单张 H100 GPU 上实现了 40+ FPS 的实时交互生成。仅使用 RynnWorld-Teleop 生成数据训练的策略在灵巧操作和多样的双臂任务中实现了有效的零样本 Sim2Real 迁移。此外，利用我们的数字远程操作数据增强真实世界数据集可一致性地提高成功率，证明了 RynnWorld-Teleop 是下一代机器人代理的高保真、可扩展数据引擎。

### 2. 方法动机分析
*   **驱动力**：旨在克服物理远程操作中由于需要大量专家时间和固定硬件限制带来的数据稀缺瓶颈，实现规模化机器人数据采集。
*   **现有方法痛点**：传统 teleop 难以扩展；现有视频生成模型（如 Human-to-Robot translation）多为观测驱动（被动），缺乏动作控制能力，无法实现闭环控制；现有的动作条件视频模型多为以人为中心，难以直接应用于特定机器人硬件。
*   **研究假设**：通过深度感知的手部姿态流作为通用动作标签，可以驱动以机器人为中心的世界模型生成高质量、可控且物理一致的机器人执行视频，从而将数据生成与物理实体解耦。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据采集与准备**：记录操作员的手部姿态序列作为控制条件。
    2.  **深度感知骨骼渲染**：将 21 个手部关键点渲染为带有深度调节颜色和直径的骨骼视频，解决 2D 投影中的深度歧义。
    3.  **渐进式训练**：第一阶段在海量 egocentric 人类视频上预训练以学习交互动力学；第二阶段在少量真实机器人演示数据上微调，实现跨域知识迁移。
    4.  **架构与条件注入**：基于 Wan-I2V DiT，采用加法 patch 嵌入（Additive patch-embedding）结合分布对齐（Distribution alignment）将控制信号注入。
    5.  **自回归蒸馏**：通过两阶段优化将双向教师模型转化为因果学生模型，利用 KV Cache 支持 40+ FPS 实时推理。
*   **模型结构**：由基于 DiT 的世界模型构成，核心包括：深度感知骨骼编码器、分布对齐模块、视频生成主干以及用于实时生成的因果流匹配学生模型。
*   **关键算法**：使用 DMD（分布匹配蒸馏）在 4 步采样内实现高质量生成；chunked re-anchoring 技术通过周期性引入真实起始帧，有效解决了长序列生成的视觉漂移问题。

### 4. 方法对比分析
*   **本质区别**：从“被动视频合成”转向“主动、以机器人为中心的动作驱动交互生成”，实现了从人手 gesture 到机器人 joint-space action 的桥接。
*   **创新贡献**：提出深度感知骨骼 representation，解决了交互视频中的 3D 深度模糊问题；引入分布对齐的加法控制注入，保证了预训练 priors 的稳定性。
*   **适用场景**：复杂双臂协调、灵巧操作的离线数据增强及实时交互控制。

### 5. 实验分析（精简版）
*   **关键结果**：在 4 项复杂操作任务中，使用增强后的数据，平均成功率提升 20%；纯合成数据训练的策略（Zero-Real-Data）在 Block Pushing 任务中达到 82.86% 的成功率。
*   **优势**：极高的数据效率、无需繁琐的 3D 资产建模、不存在视觉域差异（Reality Gap）。
*   **局限**：对复杂流体动力学或高度形变物体的模拟能力仍显不足，跨不同构型机器人需针对性 fine-tuning。

### 6. 实用指南
*   **开源情况**：已开源（Github: alibaba-damo-academy/RynnWorld-Teleop）。
*   **实现细节**：patch embedding 层需零初始化，$\alpha$ 门控标量初始值设为 0.1；推理时需预分配 KV Cache。
*   **迁移可能**：通过更换骨骼重定向模块（IK Solver）和调整数据集即可适配不同类型的机械臂或手部配置。

### 7. 总结
*   **核心思想**：利用动作驱动的生成式世界模型作为高保真数据引擎，实现数字远程操作。
*   **速记版pipeline**：骨骼视频渲染 -> 动作条件注入 -> 视频实时生成 -> 数据轨迹重定向 -> 下游策略训练。

**Key Findings:**

- We introduce digital teleoperation, a paradigm that decouples data collection from physical constraints by replacing the real robot with a generative world model.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06558v1)
- [arXiv](https://arxiv.org/abs/2607.06558v1)

---

<a id='2607.06555v1'></a>
## [ProxyPose: 6-DoF Pose Tracking via Video-to-Video Translation](https://arxiv.org/abs/2607.06555v1)

**Authors:** Ruihang Zhang, Felix Taubner, Pooja Ravi, Kiriakos N. Kutulakos, David B. Lindell

**Published:** 2026-07-07

**Categories:** cs.CV

**Abstract:**

Tracking the six-degree-of-freedom (6-DoF) pose of objects and surfaces from monocular video is a long-standing problem in computer vision. To tackle this problem, existing methods require inputs beyond the video itself-such as 3D models, depth maps, object masks, or task-specific learned features-and they struggle with textureless, transparent, reflective, or deformable surfaces. Here, we introduce ProxyPose, which recasts 6-DoF pose tracking as video-to-video translation. Given only a video and a single marked pixel in the first frame, a fine-tuned video diffusion model translates the input into a proxy video-a synthetic video depicting a colored polyhedron undergoing the same local rigid-body motion as the surface region at the marked pixel. Because the proxy's geometry and appearance are known by construction, recovering its full 6-DoF trajectory reduces to classical pose estimation with off-the-shelf solvers. This formulation leverages large-scale video pre-training to absorb the hardest aspects of pose tracking-handling challenging materials, occlusions, and deformations-into the translation step, while operating at the pixel level with no assumptions about object identity, boundaries, or global rigidity. ProxyPose achieves state-of-the-art 6-DoF pose tracking accuracy without the additional inputs required by competing methods and after fine-tuning the video model only on synthetic data. We further demonstrate that ProxyPose extends to face tracking, camera pose estimation, and challenging in-the-wild scenes that are beyond the reach of existing approaches. Project page: https://ruihangzhang97.github.io/proxypose/.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文的分析如下：

### 1. 主要贡献总结
ProxyPose 提出了一种将 6-DoF 姿态跟踪转化为视频到视频（video-to-video）生成任务的新范式。通过利用微调后的视频扩散模型，该方法仅需单帧中的一个像素点提示，即可在无需 3D 模型、深度图或物体掩码的情况下，实现对复杂材质（如透明、反光、形变物体）的精确 6-DoF 姿态跟踪。

### 2. 核心创新与方法论
*   **范式转换（Paradigm Shift）**：将极其困难的动态场景跟踪问题简化为“视频翻译”问题。模型不直接预测姿态，而是生成一个几何属性已知的“代理视频”（Proxy Video，即带有颜色编码的多面体），从而将复杂的特征匹配过程转化为已知的几何对齐问题。
*   **利用预训练大模型的涌现能力**：该方法巧妙地利用了大规模视频扩散模型在理解运动、遮挡和光影方面的强悍先验，将“跟踪”中的高难度处理（如反光和形变带来的特征缺失）交由扩散模型在翻译阶段完成。
*   **通用性与极简输入**：打破了传统方法对特定领域知识（如 CAD 模型）或辅助输入（如深度图、分割掩码）的依赖，实现了“即点即跟踪”的极大便捷性。

### 3. 对该领域的潜在影响
*   **摆脱对几何先验的依赖**：传统姿态跟踪严重依赖物体预定义的 3D 模型，而 ProxyPose 证明了通过强大的视觉生成模型，可以在没有显式几何约束的情况下实现高精度跟踪。
*   **处理挑战性材质的新路径**：对于计算机视觉中最难处理的透明、高光、形变物体，该研究提供了一条通过生成式方法进行“归一化处理”的有效途径。
*   **任务泛化能力**：该论文展现了将姿态跟踪、人脸跟踪、相机位姿估计统一在一个框架下的潜力，这可能预示着未来基础视觉模型在运动分析方向的一个重要演进方向。

### 4. 受益的相关领域与应用
*   **增强现实（AR）与混合现实（MR）**：无需预先扫描场景或物体，即可直接在现实场景中进行高精度的对象增强。
*   **机器人感知**：在非结构化环境或面临未知、复杂材质物体时，机器人可以通过此方法快速建立对目标的空间感知。
*   **视频编辑与特效（VFX）**：为后期制作提供了一种无需昂贵设备、仅靠 AI 即可自动对场景物体进行运动提取和重构的方案。
*   **生物力学与人机交互**：在人脸捕捉、肢体动作分析等对精度要求高且干扰因素多的场景中具有广泛应用价值。

### 5. 可推断的局限性
*   **推理成本（Latency/Compute）**：视频扩散模型通常计算密集，该方法能否达到实时（Real-time）跟踪仍存疑，可能更偏向离线处理。
*   **对长序列的一致性挑战**：尽管扩散模型生成能力强，但在极长视频序列中，生成的“代理视频”是否存在随时间漂移（Drift）或几何一致性退化的问题，需要评估。
*   **对非刚性（Non-rigid）运动的定义范围**：虽然论文提到可以处理形变，但其核心是将运动转化为“局部刚体”来处理，对于极其剧烈且拓扑结构改变的形变，该方法的有效性可能受限。
*   **对训练数据的依赖（模拟到现实的鸿沟）**：虽然仅使用合成数据训练，但在面对极度极端或未曾见过的现实场景时，生成结果的鲁棒性可能依然面临挑战。

**专家点评：**
ProxyPose 的迷人之处在于它展示了**“生成式模型可以作为几何计算的稳健前端”**。它不仅仅是一个跟踪算法，更像是一个将复杂视觉信息“翻译”成几何语义的翻译器。这对于未来构建无需显式几何建模的通用运动感知系统具有极高的参考价值。

**Key Findings:**

- Here, we introduce ProxyPose, which recasts 6-DoF pose tracking as video-to-video translation.
- ProxyPose achieves state-of-the-art 6-DoF pose tracking accuracy without the additional inputs required by competing methods and after fine-tuning the video model only on synthetic data.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06555v1)
- [arXiv](https://arxiv.org/abs/2607.06555v1)

---

<a id='2607.06553v1'></a>
## [From RGB Generation to Dense Field Readout: Pixel-Space Dense Prediction with Text-to-Image Models](https://arxiv.org/abs/2607.06553v1)

**Authors:** Zanyi Wang, Xin Lin, Haodong Li, Dengyang Jiang, Yijiang Li, Pengtao Xie

**Published:** 2026-07-07

**Categories:** cs.CV

**Abstract:**

Large-scale text-to-image models are attractive backbones for dense prediction because RGB generation pretraining learns rich semantic, structural, and geometric priors. Existing generative and editing approaches reuse these priors by casting dense prediction as target generation: annotations such as depth, normals, alpha mattes, masks, and heatmaps are encoded into an RGB-trained VAE latent space and decoded back as image-like targets. We argue this inherits more of the generative output interface than dense prediction requires: unlike RGB synthesis, dense prediction asks for pixel-correct, task-native fields on the same image plane, not new RGB content to be rendered. Our key observation is that a pretrained DiT already organizes RGB inputs through a patch-to-token-to-patch lattice on the image plane, so each token indexes a fixed output patch whose channels can carry task-native quantities instead of RGB appearance. We instantiate this as ReChannel: we keep the VAE encoder for the DiT's input distribution but drop the target-side decoder, adapt the frozen DiT with task LoRA, and map each token to its p x p x K_t pixel-space patch through a shared token-local linear head--about 33K parameters, no spatial mixing. Using FLUX-Klein, we evaluate on six dense prediction tasks and over a dozen benchmarks. This minimal interface sets new state-of-the-art on trimap-free matting, KITTI depth, and referring segmentation, and stays competitive on normals, saliency, and pose. In a matched 4B setting it is more accurate and 2.48x faster than an edit-plus-latent-decode counterpart--dense perception can benefit from generative pretraining without inheriting its output interface.

**Analysis:**

## 1. 摘要翻译
大规模文本到图像（T2I）模型因其对RGB生成的预训练，习得了丰富的语义、结构和几何先验，成为密集预测任务的理想骨干网络。现有的生成与编辑方法通过将密集预测转化为目标生成来复用这些先验：即将深度、法线、遮罩等标注编码至RGB预训练的VAE潜在空间，并解码为图像类输出。我们认为这继承了不必要的生成式输出接口。与RGB合成不同，密集预测旨在图像平面上获取像素准确、任务原生的场，而非渲染新的RGB内容。我们观察到预训练的扩散Transformer（DiT）本身已通过patch→token→patch的格点结构组织了RGB输入。因此，我们提出**ReChannel**：保持VAE编码器以维持输入分布，移除目标端的VAE解码器，通过轻量级LoRA适配DiT，并利用共享的Token级线性头将每个Token直接映射为任务原生场。该方法在Trimap-free matting、KITTI深度估计和指代分割任务上达到了新的SOTA，且相比生成式基线，推理速度提升达2.48倍。

## 2. 方法动机分析
- **驱动力**：作者认为“密集预测”与“RGB图像生成”本质不同。生成任务关注RGB外观细节（渲染），而密集预测关注几何、语义等场的准确性（测量）。
- **现有方法痛点**：现有基于生成式的方法（如Marigold、GenPercept）强制将密集标注塞入RGB训练的VAE潜在空间进行重构，引入了不必要的渲染损耗和推理延迟。
- **研究假设**：预训练的DiT内部已形成了一套成熟的图像空间格点（Patch Lattice），每个Token已蕴含了该空间位置的丰富特征，无需VAE解码器，只需通过简单的线性头即可直接“读出”所需的任务场。

## 3. 方法设计详解
- **核心流程**：
  1. **输入处理**：图像通过预训练模型的VAE编码器进入潜空间，作为DiT的输入，保持分布一致。
  2. **特征处理**：DiT保持冻结，仅插入轻量级LoRA微调，负责将输入的RGB特征重塑为特定任务的语义空间。
  3. **读出机制（Readout）**：不使用反卷积解码器，而是通过一个共享的Token级线性映射层，将每个DiT输出的Token直接投影到 $p \times p \times K_t$ 的像素块（$p$为patch大小，$K_t$为任务输出通道数）。
  4. **输出**：直接拼接所有patch块，得到与原图空间结构对应的密集预测结果。
- **关键设计**：
  - **Token即载体**：Token不仅代表RGB外观，更是一个空间物理载体，通过LoRA适配，使其通道重映射为任务相关数据。
  - **解耦生成与读取**：完全舍弃解码器结构，将复杂映射转化为简单的Token局部线性投影，实现了从“生成任务”到“读出任务”的范式转换。

## 4. 方法对比分析
- **本质区别**：与现有方法将密集预测作为“生成式重建”不同，ReChannel将其定义为“预训练场中的通道读出（Rechanneling）”。
- **创新贡献**：提出了一种无需VAE解码器的密集预测接口，证明了生成预训练的威力并不在于VAE解码器，而在于其内部特征组织能力。
- **适用场景**：适用于任何需要像素级对齐的任务（深度、法线、分割、抠图、姿态估计）。

## 5. 实验分析
- **验证方法**：在6项密集预测任务、十余个基准数据集上，与各种生成式及判别式方法对比。
- **关键结论**：在保持4B量级的情况下，其精度不仅优于传统判别式方法，且推理速度比现有生成式重构方法快1.56x至2.48x。
- **局限**：方法依然依赖于预训练的DiT backbone，对于极致轻量化的端侧设备，其参数量可能仍较高。

## 6. 实用指南
- **开源情况**：已开源（见论文链接：https://github.com/xmz111/ReChannel）。
- **实现细节**：关键在于使用任务特定的LoRA进行适配，并确保训练过程中使用标准像素空间损失（Pixel Loss），而非潜空间重构损失。
- **迁移可能**：该方案极易迁移。只需将输出线性头的 $K_t$ 修改为新任务的通道数，并在相应数据集上微调LoRA即可。

## 7. 总结
- **核心思想**：将预训练的生成式特征直接线性读出为任务原生场，而非通过解码器重建。
- **速记版pipeline**：
  1. 冻结DiT主干，插入LoRA。
  2. 仅保留VAE编码器处理输入。
  3. 用线性头将Token直接映射为任务数值。
  4. 输出直接拼合即得预测场。

**Key Findings:**

- We argue this inherits more of the generative output interface than dense prediction requires: unlike RGB synthesis, dense prediction asks for pixel-correct, task-native fields on the same image plane, not new RGB content to be rendered.
- This minimal interface sets new state-of-the-art on trimap-free matting, KITTI depth, and referring segmentation, and stays competitive on normals, saliency, and pose.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06553v1)
- [arXiv](https://arxiv.org/abs/2607.06553v1)

---

<a id='2607.06537v1'></a>
## [UniLM-Nav: A Unified Framework for Zero-Shot Last-Mile Navigation](https://arxiv.org/abs/2607.06537v1)

**Authors:** Zhuofan Zhang, Tianxu Wang, Guoxi Zhang, Yixiong Lin, Xilin Wang, Hongming Xu, Qing Li, Song-Chun Zhu, Lifeng Fan

**Published:** 2026-07-07

**Categories:** cs.RO

**Abstract:**

Mobile manipulation requires a robot to navigate to a target object or receptacle and then perform intended manipulation. However, reaching the vicinity of the target does not guarantee a manipulation-ready base pose, a problem known as last-mile navigation. Prior methods for last-mile navigation either rely on manual pose annotation or task-specific training, limiting their scalability to open-vocabulary settings with fine-grained spatial constraints. We propose UniLM-Nav, a unified framework for zero-shot open-vocabulary last-mile navigation. UniLM-Nav decomposes last-mile navigation into view selection, task-conditioned affordance grounding, and geometry-aware base-pose reasoning, all resolved with a shared multimodal large language model (MLLM) backend. Specifically, UniLM-Nav first selects a reference view that best captures the target object or receptacle from recently collected observations. It then grounds task-relevant affordance point in the selected view and lifts the result into the robot-centric coordinate frame. Finally, conditioned on the grounded affordance, task context, and robot geometry, it infers a manipulation-ready base pose for the robot. We evaluate UniLM-Nav on the OVMM benchmark, where it outperforms the previous state-of-the-art method, MoTo, by 3.13 percentage points. Analyses show that the components of our method are crucial to final performance, and that the choice of MLLM also has a substantial effect. We further deploy UniLM-Nav on a Unitree B2 quadruped robot with a 6-DoF Unitree Z1 manipulator, validating its applicability to real-world mobile manipulation tasks.

**Analysis:**

作为计算机视觉与机器人领域的专家，我对 **UniLM-Nav** 这篇论文的分析如下：

### 1. 论文核心贡献总结
UniLM-Nav 提出了一个针对“最后一公里导航”（Last-Mile Navigation）的通用零样本（Zero-Shot）框架，旨在解决移动操作任务中如何确定“适合操作的机器人基座姿态”这一难题。该方法无需人工标注或特定任务训练，仅利用多模态大模型（MLLM）作为统一后端，实现了开放词汇场景下的目标定位与导航决策。

### 2. 关键创新与方法论
该研究的核心创新在于将复杂的导航任务进行**逻辑解耦与大模型重构**：
*   **任务解耦**：将导航拆解为“视图选择”（View Selection）、“任务条件下的可供性接地”（Task-conditioned Affordance Grounding）和“几何感知的基座姿态推理”（Geometry-aware Base-pose Reasoning）三个阶段。
*   **统一后端**：放弃了以往针对特定动作或物体的训练路径，转而使用单一的 MLLM 作为视觉推理与决策的核心，极大地提升了泛化能力。
*   **多模态融合**：通过将 2D 视觉信息“提升”（Lifting）到 3D 机器人坐标系中，结合任务上下文与机器人自身动力学几何约束，实现了端到端的零样本导航。

### 3. 对该领域的潜在影响
*   **范式转变**：证明了在大模型时代，传统的“导航（Navigation）+操作（Manipulation）”两阶段割裂问题可以通过 MLLM 的推理能力进行统一建模，降低了机器人落地应用对海量标注数据的依赖。
*   **工业实用价值**：UniLM-Nav 在 Unitree B2 四足机器人上的实机部署验证，标志着此类算法已跨越了从仿真到现实的鸿沟，为开放环境下移动操作机器人的广泛应用提供了可行的技术路径。
*   **性能标杆**：在 OVMM 基准测试中超越 SOTA（MoTo），为后续研究提供了新的性能基线。

### 4. 受益的相关领域与应用
*   **家庭服务机器人**：如整理桌面、抓取冰箱内的物品，需要极高的细粒度操作姿态对准。
*   **仓储物流自动化**：在动态、非结构化的仓库环境中，机器人需要对目标物体进行精准的近距离作业。
*   **人机协作（HRC）**：机器人需要根据任务需求灵活调整位置以辅助人类工作。
*   **具身智能（Embodied AI）**：为大型视觉-语言模型如何与机器人本体的物理约束交互提供了重要参考。

### 5. 可推断的局限性
*   **MLLM 推理延迟**：由于后端依赖复杂的 MLLM，在实时性要求极高的动态场景中，推理延迟可能成为瓶颈。
*   **对视图质量的敏感性**：虽然引入了视图选择模块，但如果初始采集的视觉信息不足（例如遮挡过重），模型可能无法提取有效的可供性点。
*   **几何与语义的对齐精度**：从 2D 图像到 3D 物理空间的“提升”过程，仍可能存在投影偏差，这在需要毫米级精度操作时可能引发失败。
*   **计算资源依赖**：作为一个基于 MLLM 的方案，其对机载边缘计算设备的算力要求较高，可能限制了轻量级机器人的应用。

**专家总结**：这篇论文最有趣之处在于它敏锐地捕捉到了**移动操作任务中“空间定位”与“语义理解”之间的缺口**，并通过多模态大模型的推理能力将这一缺口抹平。它不仅是在做一个导航算法，更是在探索如何让“大脑”（MLLM）更好地控制“身体”（机器人）。

**Key Findings:**

- We propose UniLM-Nav, a unified framework for zero-shot open-vocabulary last-mile navigation.
- We evaluate UniLM-Nav on the OVMM benchmark, where it outperforms the previous state-of-the-art method, MoTo, by 3.13 percentage points.
- Analyses show that the components of our method are crucial to final performance, and that the choice of MLLM also has a substantial effect.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06537v1)
- [arXiv](https://arxiv.org/abs/2607.06537v1)

---

<a id='2607.06501v1'></a>
## [Hypothesis-driven Model Expansion under Uncertainty for Open-World Robot Planning](https://arxiv.org/abs/2607.06501v1)

**Authors:** Anxing Xiao, Hanbo Zhang, Tianrun Hu, David Hsu

**Published:** 2026-07-07

**Categories:** cs.RO

**Abstract:**

We consider an open-world planning setting in which service robots must operate in unknown environments with incomplete knowledge of objects and actions. Traditional closed-world approaches with pre-programmed knowledge bases fail when robots encounter unexpected situations and tasks, posing a fundamental challenge for autonomous knowledge expansion in human environments. In this work, we propose an open-world planning framework that enables robots to automatically generate, verify, and update hypotheses about their abstract world models. Our key insight is to explicitly maintain uncertainty-aware knowledge expansion and integrate hypothesis verification into goal-reaching planning. The framework leverages foundation models to generate initial hypotheses over states and transitions, and applies automated planning to produce action sequences that jointly address hypothesis verification and task execution. Through iterative execution and refinement, the robot expands its knowledge by incorporating verification feedback from the foundation models when hypotheses prove incorrect. Extensive experiments in simulated and real-world environments demonstrate that our framework enables autonomous knowledge expansion and effective operation in open-world settings. These results indicate that integrating uncertainty-aware model expansion from robot foundation models with planning advances the practical deployment of household service robots.

**Analysis:**

### 1. 摘要翻译
我们研究了服务机器人如何在具有不完全知识的未知环境中运行的开放世界规划问题。传统的闭环世界假设在面对不可预见的情况时通常会失效。本文提出了一种开放世界规划框架，使机器人能够自动生成、验证和更新其抽象世界模型的假设。我们的核心洞察是将模型扩展视为一种不确定的、可主动验证的过程，并将其集成到目标导向的规划中。我们提出了“假设驱动的确定性感知模型扩展”（HUME）框架，利用基础模型生成关于状态和转换的初始假设，通过自动规划生成既能达成任务又能验证假设的动作序列。通过迭代执行与反馈更新，机器人能够持续扩展知识。在模拟和真实环境中的实验表明，该框架显著提升了机器人在开放世界中的自主扩展能力与作业效率。

### 2. 方法动机分析
- **驱动力**：旨在解决服务机器人在开放世界中因先验知识缺失或不精确而导致的任务执行失败问题。
- **现有方法痛点**：
    - 传统规划器（如PDDL）依赖完备的先验知识，泛化性差；
    - 基于LLM的规划器虽然泛化能力强，但容易产生“幻觉”，缺乏对自身不确定性的显式建模，导致执行盲目。
- **研究假设**：模型扩展本身应被视为一个“Bayes-adaptive”过程——将缺失知识建模为一组具有概率分布的假设，并通过主动采取“信息获取型动作”来降低模型不确定性。

### 3. 方法设计详解
HUME 的核心在于将开放世界的未知状态转化为可被经典规划器处理的假设：
- **假设生成（Hypothesis Generator）**：利用LLM根据当前任务目标和感知到的部分场景图，生成关于对象属性、存在性或动作效果的假设（如“杯子可能在冰箱里”）。
- **增强规划空间**：将假设视为待解决的变量，定义动作的先验成本。对于涉及未验证假设的动作，赋予高惩罚权重，迫使规划器优先调度“验证动作”（Verify Action）。
- **确定性分支（Determinization）**：将非确定性的验证动作拆分为“验证成功”和“验证失败”两个分支，通过规划图剪枝剔除导致任务失败的分支，从而实现“乐观规划”。
- **闭环更新（Verification & Update）**：在执行期间，机器人利用视觉-语言模型（VLM）获取观察结果，确认或反驳假设。反驳后的假设将被标注并记录在历史上下文中，驱动下一轮的重新规划。

### 4. 方法对比分析
- **本质区别**：从“被动接受LLM输出的计划”转变为“将模型未知项显式化为可规划的对象”。
- **创新贡献**：提出了一种将“逻辑推理（规划器）”与“常识推理（LLM）”结合的新方案，赋予机器人主动探究未知领域的能力。
- **适用场景**：长期、多步骤、环境具有部分观测性质的家用或工业服务任务。

### 5. 实验分析
- **关键结论**：在Block Processing World和AI2-THOR任务中，HUME在成功率和任务效率（SPL）上均大幅优于仅使用LLM规划或仅使用静态PDDL规划的方法。
- **优势**：显著减少了“静默失败”，提升了面对未知物体时的鲁棒性。
- **局限**：对验证动作的依赖性较高；如果LLM对验证条件的预测不准确，可能导致死锁或冗余探索。

### 6. 实用指南
- **开源情况**：项目主页：[open-world-planning.github.io](https://open-world-planning.github.io)。
- **实现关键**：
    - 需构建一个能够根据假设自动修改PDDL文件接口的LLM提示词工程；
    - 验证动作（Verify Action）的定义需匹配低级控制库。
- **迁移建议**：该方法非常适合迁移到任何具备结构化域定义（如PDDL/STRIPS）的机器人任务中。只需定义好“未知项”的表示方式，即可直接复用该扩展框架。

### 7. 总结
- **核心思想**：通过显式建模未知知识为可验证假设，将模型扩展融入主动规划过程。
- **速记版Pipeline**：
    1. **识别未知**：根据目标发现当前模型缺失的项。
    2. **生成假设**：LLM对缺失项提出猜想。
    3. **乐观规划**：规划包含验证任务的路径。
    4. **主动核验**：执行验证动作并根据反馈修正假设。
    5. **迭代闭环**：持续更新假设直至任务达成。

**Key Findings:**

- In this work, we propose an open-world planning framework that enables robots to automatically generate, verify, and update hypotheses about their abstract world models.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.06501v1)
- [arXiv](https://arxiv.org/abs/2607.06501v1)

---

