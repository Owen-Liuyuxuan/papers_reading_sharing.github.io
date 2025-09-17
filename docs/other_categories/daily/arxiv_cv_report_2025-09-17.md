time: 20250917

# Arxiv Computer Vision Papers - 2025-09-17

## Executive Summary

好的，这是一份针对2025年9月16日Arxiv计算机视觉论文的每日报告执行摘要，旨在帮助忙碌的研究人员快速了解最新进展。

---

**执行摘要：Arxiv 计算机视觉每日报告 (2025-09-16)**

**1. 主要主题与趋势概述：**

今天的论文集呈现出计算机视觉领域几个关键且相互关联的趋势：

*   **具身智能与三维感知：** 具身智能（Embodied AI）和自动驾驶是核心驱动力，对全向视觉、三维点云处理、三维人体姿态与形状估计以及高精度地图的需求日益增长。这表明研究正从纯粹的图像理解转向更复杂的、与物理世界交互的感知系统。
*   **数据驱动与基准建设：** 多个工作致力于构建大规模、多模态或特定场景的数据集（如街景树木、停车位），以推动特定任务的进展。这反映了高质量数据在深度学习时代的重要性，以及对更真实世界场景的关注。
*   **效率与鲁棒性：** 在实际应用中，对效率（如暴力检测）和鲁棒性（如低光照增强、长尾分布处理）的关注持续存在，研究人员正在探索新的模型架构和训练策略来解决这些挑战。
*   **生成模型与编辑：** 扩散模型（Rectified Flow）在图像生成和语义编辑方面的应用显示出其在内容创作和图像操作领域的潜力。

**2. 特别重要或创新的论文：**

*   **"PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era" (Xu Zheng et al.)：** 这篇综述性论文非常及时且具有前瞻性。它系统地梳理了全向视觉在具身智能中的重要性、挑战和未来方向，对于理解该领域宏观发展至关重要。其创新性在于对一个新兴且关键领域的全面展望。
*   **"Maps for Autonomous Driving: Full-process Survey and Frontiers" (Pengxin Chen et al.)：** 另一篇高质量的综述，深入探讨了自动驾驶地图的整个生命周期。考虑到自动驾驶的复杂性和对地图的依赖，这篇论文为研究人员提供了宝贵的知识体系和未来研究方向。
*   **"Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing" (Weiming Chen et al.)：** 这篇论文在生成模型领域展现了技术创新。通过引入Runge-Kutta近似和解耦注意力，它提升了Rectified Flow模型在图像反演和语义编辑方面的性能，为高质量图像生成和操作提供了新的工具。

**3. 新兴研究方向或技术：**

*   **全向视觉（Omnidirectional Vision）：** 随着具身智能和VR/AR的发展，全向视觉将成为一个越来越重要的研究方向，涵盖数据采集、模型设计和应用。
*   **LiDAR点云在人体感知中的应用：** "3D Human Pose and Shape Estimation from LiDAR Point Clouds" 指出LiDAR在隐私保护和全天候感知方面的优势，预示着LiDAR在人体感知领域的潜力。
*   **扩散模型（Rectified Flow）的精细化控制与应用：** "Runge-Kutta Approximation and Decoupled Attention..." 展示了如何通过算法优化来提升扩散模型在特定任务（如语义编辑）上的表现，预示着未来对生成模型更精细化控制的研究。
*   **多模态融合与特定场景数据集：** WHU-STree 和 Advancing Real-World Parking Slot Detection 等工作强调了为特定复杂场景构建多模态、大规模数据集的重要性，以及如何利用这些数据解决实际问题。

**4. 建议完整阅读的论文：**

为了全面了解当前趋势和潜在突破，建议完整阅读以下论文：

*   **"PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era" (Xu Zheng et al.)：** 提供宏观视角，理解具身智能背景下的全向视觉。
*   **"Maps for Autonomous Driving: Full-process Survey and Frontiers" (Pengxin Chen et al.)：** 深入了解自动驾驶核心技术之一的地图。
*   **"Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing" (Weiming Chen et al.)：** 了解生成模型前沿技术及其在图像编辑中的应用。
*   **"Deep learning for 3D point cloud processing - from approaches, tasks to its implications on urban and environmental applications" (Zhenxin Zhang et al.)：** 对3D点云处理的全面综述，对于理解三维感知基础至关重要。

---

这份摘要旨在为您的研究提供一个快速导航，帮助您优先关注与您研究方向最相关的最新进展。

---

## Table of Contents

1. [PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era](#2509.12989v1)
2. [Maps for Autonomous Driving: Full-process Survey and Frontiers](#2509.12632v1)
3. [Deep learning for 3D point cloud processing - from approaches, tasks to its implications on urban and environmental applications](#2509.12452v1)
4. [3D Human Pose and Shape Estimation from LiDAR Point Clouds: A Review](#2509.12197v1)
5. [Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance](#2509.13210v1)
6. [WHU-STree: A Multi-modal Benchmark Dataset for Street Tree Inventory](#2509.13172v1)
7. [Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline](#2509.13133v1)
8. [Using KL-Divergence to Focus Frequency Information in Low-Light Image Enhancement](#2509.13083v1)
9. [Dual-Stage Reweighted MoE for Long-Tailed Egocentric Mistake Detection](#2509.12990v1)
10. [Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing](#2509.12888v1)

---

## Papers

<a id='2509.12989v1'></a>
## [PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era](https://arxiv.org/abs/2509.12989v1)

**Authors:** Xu Zheng, Chenfei Liao, Ziqiao Weng, Kaiyu Lei, Zihao Dongfang, Haocong He, Yuanhuiyi Lyu, Lutao Jiang, Lu Qi, Li Chen, Danda Pani Paudel, Kailun Yang, Linfeng Zhang, Luc Van Gool, Xuming Hu

**Published:** 2025-09-16

**Categories:** cs.CV

**Abstract:**

Omnidirectional vision, using 360-degree vision to understand the
environment, has become increasingly critical across domains like robotics,
industrial inspection, and environmental monitoring. Compared to traditional
pinhole vision, omnidirectional vision provides holistic environmental
awareness, significantly enhancing the completeness of scene perception and the
reliability of decision-making. However, foundational research in this area has
historically lagged behind traditional pinhole vision. This talk presents an
emerging trend in the embodied AI era: the rapid development of omnidirectional
vision, driven by growing industrial demand and academic interest. We highlight
recent breakthroughs in omnidirectional generation, omnidirectional perception,
omnidirectional understanding, and related datasets. Drawing on insights from
both academia and industry, we propose an ideal panoramic system architecture
in the embodied AI era, PANORAMA, which consists of four key subsystems.
Moreover, we offer in-depth opinions related to emerging trends and
cross-community impacts at the intersection of panoramic vision and embodied
AI, along with the future roadmap and open challenges. This overview
synthesizes state-of-the-art advancements and outlines challenges and
opportunities for future research in building robust, general-purpose
omnidirectional AI systems in the embodied AI era.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供由Xu Zheng等人撰写的论文“PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era”的全面摘要。

---

**论文摘要：PANORAMA: The Rise of Omnidirectional Vision in the Embodied AI Era**

**1. 主要问题或研究问题：**
该论文旨在解决在具身AI时代，全向视觉（360度全景视觉）在理解环境方面日益增长的重要性与该领域基础研究长期落后于传统针孔视觉之间的差距。传统针孔视觉提供的是狭窄的、视锥受限的视角，而全向视觉能提供更全面的环境感知，这对于具身AI中更复杂的任务（如室内/室外导航）至关重要。论文探讨了如何克服数据瓶颈、模型能力限制和应用空白，以充分释放全向视觉在具身AI中的巨大潜力。

**2. 关键创新或方法论贡献：**
*   **PANORAMA系统架构：** 论文提出了一个理想的全景系统架构——PANORAMA，由四个关键子系统组成：
    *   **数据采集与预处理（Data Acquisition & Pre-processing）：** 负责捕获原始全向数据并转换为计算处理格式，包括数据捕获、格式转换、同步与校准。
    *   **感知（Perception）：** 对预处理后的全景数据进行基础场景感知，利用专门的深度学习模型（如球面CNN、Transformer）提取丰富、结构化的信息，执行特征提取和环境感知（语义分割、目标检测、深度估计）。
    *   **应用（Application）：** 将感知洞察转化为具身AI智能体的行动，服务于特定下游任务，如导航与SLAM、人机交互、数字孪生与3D重建。
    *   **加速与部署（Acceleration & Employment）：** 解决高分辨率全景数据处理的计算挑战，通过软件加速（模型量化、剪枝）和硬件部署（边缘计算平台）确保整个流程的计算可行性。
*   **全向视觉技术突破的综合概述：** 论文系统地总结了全向生成、全向感知、全向理解以及相关数据集的最新进展，突出了该领域在工业需求和学术兴趣驱动下的快速发展。
*   **具身AI时代全向视觉的路线图：** 论文提出了一个分阶段的未来路线图，包括数据集整合、多模态扩展、推理与具身数据、统一模型预训练、评估与基准测试、部署与泛化，旨在构建一个理想的、统一的全向任务模型。

**3. 主要结果及其意义：**
*   **全向视觉的潜力：** 论文强调全向视觉在机器人、工业检测和环境监测等领域的重要性，因为它提供了比针孔视觉更全面的环境感知，显著提升了场景感知的完整性和决策的可靠性。
*   **挑战的系统性分类：** 将全向视觉面临的问题归结为数据瓶颈、模型能力和应用空白三大类，为后续研究提供了清晰的框架。
*   **技术进展的梳理：** 详细介绍了全向生成（如Dream360、PanoDiffusion、OmniDrag）、全向感知（如GoodSAM、OmniSAM）和全向理解（如OSR-Bench、OmniVQA）的最新技术，展示了该领域在克服几何畸变、适应模型和构建数据集方面的努力。
*   **数据集的全面回顾：** 提供了室内、室外和无人机/飞行等领域23个代表性全向数据集的概述，涵盖了RGB全景图、深度、相机姿态和语义标签等模态，为研究人员提供了宝贵的资源。
*   **PANORAMA架构的愿景：** 提出的PANORAMA系统架构为具身AI中全向视觉的集成提供了一个全面的、端到端的解决方案，有望推动具身智能的发展。

**4. 论文中提及的局限性：**
*   **数据瓶颈：** 全景图像的标注成本高昂，且由于等距矩形投影（ERP）等几何畸变，传统自动化标注工具效率低下，阻碍了大规模高质量数据集的开发。
*   **模型能力限制：** 现有预训练模型（主要针对针孔图像设计）的归纳偏置（如平移不变性）不适用于全景图像的畸变特性，导致性能显著下降。
*   **应用空白：** 缺乏跨学科人才以及现有全景数据和模型的不足，导致全景生产安全检查、全景森林火灾检测等特定应用领域探索不足。
*   **泛化性和鲁棒性：** 当前模型多专注于特定场景或投影方法，难以泛化到多样化的全景传感器规格、应用场景和投影方法。
*   **动态畸变处理：** 现有方法将全景图像的畸变视为与帧无关的几何问题，未能充分考虑真实世界场景中畸变的动态性和在全向视频序列中的演变。
*   **缺乏大规模多模态预训练资源：** 现有方法在模型泛化方面受限，缺乏大规模多模态预训练资源，阻碍了具身AI的广泛发展。

**5. 潜在的未来研究方向：**
*   **数据集建设：** 规划和发布大规模、多任务的全向数据集，涵盖真实世界场景的复杂性，包括室内外、通用和具身智能场景。
*   **算法创新：** 超越基于针孔模型的简单适配，创建具有全向信息的新颖架构和动态学习范式，以应对全向视觉的独特挑战。
*   **应用探索：** 探索和展示全向感知在真实世界机器人和交互系统中的优势，弥合实验室研究与实际应用之间的鸿沟。
*   **泛化与鲁棒性：** 开发能够泛化到不同全景传感器规格、应用场景和投影方法的模型，并利用投影无关表示和自监督学习技术从无标签全向信息中学习不变特征。
*   **动态畸变处理：** 明确考虑全向视频序列中畸变的动态性和时间一致性。
*   **以行动为导向的表示学习：** 使模型能够学习全景图像中以行动为导向的表示，从而实现更有效和高效的机器人决策。
*   **可扩展的统一架构：** 创建专门为全向视觉设计的统一、多任务基础模型，通过在大量全景数据上进行预训练，捕捉全向几何和语义的基本理解。

---

**Key Findings:**

- Drawing on insights from
both academia and industry, we propose an ideal panoramic system architecture
in the embodied AI era, PANORAMA, which consists of four key subsystems.
- This overview
synthesizes state-of-the-art advancements and outlines challenges and
opportunities for future research in building robust, general-purpose
omnidirectional AI systems in the embodied AI era.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12989v1)
- [arXiv](https://arxiv.org/abs/2509.12989v1)

---

<a id='2509.12632v1'></a>
## [Maps for Autonomous Driving: Full-process Survey and Frontiers](https://arxiv.org/abs/2509.12632v1)

**Authors:** Pengxin Chen, Zhipeng Luo, Xiaoqi Jiang, Zhangcai Yin, Jonathan Li

**Published:** 2025-09-16

**Categories:** cs.CV

**Abstract:**

Maps have always been an essential component of autonomous driving. With the
advancement of autonomous driving technology, both the representation and
production process of maps have evolved substantially. The article categorizes
the evolution of maps into three stages: High-Definition (HD) maps, Lightweight
(Lite) maps, and Implicit maps. For each stage, we provide a comprehensive
review of the map production workflow, with highlighting technical challenges
involved and summarizing relevant solutions proposed by the academic community.
Furthermore, we discuss cutting-edge research advances in map representations
and explore how these innovations can be integrated into end-to-end autonomous
driving frameworks.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Pengxin Chen等人撰写的论文“Maps for Autonomous Driving: Full-process Survey and Frontiers”的全面摘要。

---

**论文摘要：自动驾驶地图：全流程综述与前沿**

**1. 论文解决的主要问题或研究问题：**
该论文旨在全面回顾自动驾驶地图（Maps for Autonomous Driving, MAD）的演变、生产流程、技术挑战以及未来发展方向。核心问题是，随着自动驾驶技术的发展，地图的表示和生产过程如何演变，以满足从传统导航到端到端自动驾驶系统不断增长的需求，并克服高成本、低更新频率和复杂性等限制。论文将地图演变分为三个阶段：高精地图（HD maps）、轻量化地图（Lite maps）和隐式地图（Implicit maps），并对每个阶段的生产流程、技术挑战和学术解决方案进行了深入分析。

**2. 关键创新或方法论贡献：**
该论文的主要贡献在于其全面的视角和对地图演变阶段的划分，这在现有综述中是独有的。具体方法论贡献包括：
*   **三阶段演进框架：** 首次系统地将自动驾驶地图的演进划分为HD Map、Lite Map和Implicit Map三个关键阶段，并详细阐述了每个阶段的特点、技术需求和挑战。
*   **HD Map阶段的全面回顾：** 详细介绍了HD Map的生产流程（测绘、感知、地图编译），并深入探讨了定位（GNSS、IMU、轮式里程计、视觉/激光雷达里程计、地面控制点）、多行程测绘（协同SLAM、闭环检测、SD地图匹配）、静态感知（路面、路标、车道线、交通标志、护栏、杆状物、行道树提取）和拓扑生成（图搜索、深度学习、分层推理、多模态融合）等关键技术。
*   **Lite Map阶段的重点关注：** 强调了Lite Map作为HD Map的轻量化替代方案，在解决城市道路自动驾驶挑战中的作用。重点介绍了在线矢量化（单帧检测、长序列建模）、众包地图维护（在线变化检测、奖励路线、交通流轨迹挖掘）等创新方法。
*   **Implicit Map阶段的前沿探索：** 深入探讨了隐式地图作为端到端自动驾驶系统发展趋势的一部分，包括查询表示方法（生成式与目标导向、上下文条件、统一跨任务）、潜在空间方法（结构化场景表示、动态感知潜在建模、鲁棒性增强）、神经辐射场（NeRF）方法（场景重建、语义理解、效率与鲁棒性增强）和世界模型（环境表示、动态演化、模型增强）等。
*   **工业视角的整合：** 结合了作者在工业界的经验，提供了对地图发展里程碑和关键公司技术的洞察，使综述更具实践指导意义。
*   **开源工作总结：** 附录中提供了隐式地图和在线矢量化相关开源工作的列表，方便研究人员快速查阅。

**3. 主要结果及其意义：**
*   **HD Map的奠基作用：** HD Map通过提供厘米级精度的车道级信息，首次使L2-L4级高速公路自动驾驶成为可能，奠定了地图在自动驾驶中的基础地位。
*   **Lite Map的实用性突破：** Lite Map通过众包数据和自动化生成技术，显著降低了生产成本并提高了更新频率，使其能够覆盖城市道路，推动了自动驾驶的商业化落地。
*   **Implicit Map的未来潜力：** 隐式地图作为端到端自动驾驶系统的一部分，通过将环境知识隐式编码到神经网络中，有望实现更像人类的、上下文感知的决策制定，并促进可微分处理和反向传播，为联合学习系统提供支持。
*   **地图在AD系统中不可或缺的地位：** 论文强调，无论地图形式如何演变，它始终是自动驾驶系统中不可或缺的元素，从传统的模块化架构到现代的端到端学习系统，地图都扮演着关键角色。

**4. 论文中提及的局限性：**
*   **HD Map的局限性：** 生产成本高昂、更新频率低、维护成本高，难以扩展到复杂的城市道路环境。
*   **Lite Map的局限性：** 尽管有所改进，但在线矢量化在地图精度和元素丰富度方面仍远未达到HD Map的水平，感知范围有限，且在复杂遮挡场景下性能受影响。众包维护面临数据传输量大、用户隐私和地图新鲜度等挑战。
*   **Implicit Map的局限性：**
    *   **查询表示方法：** 计算效率与表示能力之间存在权衡，依赖高质量输入，且缺乏正式的安全验证机制。跨模态接地存在模糊性，因果可解释性不足，时间鲁棒性有限。
    *   **潜在空间方法：** BEV投影受深度估计误差和遮挡伪影影响，图基方法依赖预定义拓扑模板，对象中心方法易受上游检测误差影响。动态感知潜在建模计算复杂，非线性运动建模不足，对数据稀疏性敏感。鲁棒性增强方法依赖高质量真实数据，对抗训练和域适应缺乏可量化鲁棒性边界，合成数据安全验证机制缺失。
    *   **NeRF方法：** 瞬态动态处理导致伪影，语义分辨率限制小物体识别，校准误差影响传感器融合。
    *   **世界模型：** 密集表示计算和内存开销大，对象中心方法易受上游感知模块误差传播影响，图基模型依赖定义关系的完整性，纯视觉模型在恶劣条件下鲁棒性有限。物理模型计算成本高，博弈论方法复杂且模型简化，生成模型推理速度慢且物理可控性差，贝叶斯方法计算密集且难以区分不确定性。模型增强方法验证范围有限，训练复杂且不稳定，元学习依赖元训练数据多样性，LLM模型推理延迟高、易“幻觉”，合成数据存在“现实鸿沟”。

**5. 潜在的未来研究方向：**
*   **全自动轻量化地图（Fully Automatic Lite Map）：** 持续改进Lite Map，实现数据采集、处理和更新的无缝集成，无需人工干预，以实现可扩展和经济高效的地图维护。
*   **零样本地图学习（Zero-shot Map Learning）：** 开发能够识别稀有或新型道路特征的模型，减少对标注数据的依赖，提高模型对未知环境的泛化能力，并结合不确定性估计以提高安全性和可靠性。
*   **VLA中的隐式地图（Implicit Map in VLA）：** 探索如何将地图经验存储在云端作为“记忆知识库”或“经验层”，以辅助车辆更好地理解驾驶场景，并支持VLA模型通过多模态训练数据隐式理解地图结构。
*   **基础模型（Foundation Model）的应用：** 利用基础模型学习潜在空间结构和语义上下文，实现多模态集成（视觉、语言、时间上下文），并通过Transformer的注意力机制动态检索空间和语义相关信息，从而支持隐式定位和地图构建。
*   **混合建模方法：** 结合结构化先验知识与神经网络，开发可解释的表示学习技术用于关键场景分析，并制定标准化评估指标以兼顾性能和安全要求。

---

**Key Findings:**

- For each stage, we provide a comprehensive
review of the map production workflow, with highlighting technical challenges
involved and summarizing relevant solutions proposed by the academic community.
- Furthermore, we discuss cutting-edge research advances in map representations
and explore how these innovations can be integrated into end-to-end autonomous
driving frameworks.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12632v1)
- [arXiv](https://arxiv.org/abs/2509.12632v1)

---

<a id='2509.12452v1'></a>
## [Deep learning for 3D point cloud processing - from approaches, tasks to its implications on urban and environmental applications](https://arxiv.org/abs/2509.12452v1)

**Authors:** Zhenxin Zhang, Zhihua Xu, Yuwei Cao, Ningli Xu, Shuye Wang, Shen'ao Cui, Zhen Li, Rongjun Qin

**Published:** 2025-09-15

**Categories:** cs.CV

**Abstract:**

Point cloud processing as a fundamental task in the field of geomatics and
computer vision, has been supporting tasks and applications at different scales
from air to ground, including mapping, environmental monitoring, urban/tree
structure modeling, automated driving, robotics, disaster responses etc. Due to
the rapid development of deep learning, point cloud processing algorithms have
nowadays been almost explicitly dominated by learning-based approaches, most of
which are yet transitioned into real-world practices. Existing surveys
primarily focus on the ever-updating network architecture to accommodate
unordered point clouds, largely ignoring their practical values in typical
point cloud processing applications, in which extra-large volume of data,
diverse scene contents, varying point density, data modality need to be
considered. In this paper, we provide a meta review on deep learning approaches
and datasets that cover a selection of critical tasks of point cloud processing
in use such as scene completion, registration, semantic segmentation, and
modeling. By reviewing a broad range of urban and environmental applications
these tasks can support, we identify gaps to be closed as these methods
transformed into applications and draw concluding remarks in both the
algorithmic and practical aspects of the surveyed methods.

**Analysis:**

这篇论文“Deep learning for 3D point cloud processing - from approaches, tasks to its implications on urban and environmental applications”由Zhenxin Zhang等人撰写，对深度学习在3D点云处理及其在城市和环境应用中的作用进行了全面的元综述。

以下是该论文的摘要：

1.  **主要问题或研究问题：**
    该论文旨在解决现有关于点云处理深度学习综述的不足。现有综述主要关注不断更新的网络架构以适应无序点云，但往往忽略了这些方法在实际点云处理应用中的实用价值，尤其是在处理超大数据量、多样场景内容、不同点云密度和数据模态时。因此，该研究的核心问题是：如何提供一个全面的元综述，连接深度学习算法与实际点云处理任务及其在城市和环境应用中的潜在价值，并识别这些方法在转化为实际应用时需要弥补的差距。

2.  **关键创新或方法论贡献：**
    *   **元综述方法：** 论文采用元综述的方法，而非仅仅关注最新的网络架构，而是将深度学习方法和数据集与点云处理的关键任务（如场景补全、配准、语义分割和建模）联系起来。
    *   **数据源的全面概述：** 详细介绍了各种点云数据采集方法，包括LiDAR、摄影测量、结构光系统、光度立体、SAR干涉测量和混合系统，并分析了它们的特点、成本、复杂性和精度。
    *   **任务与应用的连接：** 论文明确地将点云处理任务与广泛的城市和环境应用（如城市建模、林业、农业、生态学和公用事业测绘）联系起来，突出了深度学习在这些领域中的实际支持作用。
    *   **识别差距和挑战：** 论文不仅总结了现有方法的成就，还明确指出了这些方法在推广到实际应用中时面临的挑战，例如泛化能力、处理大规模数据、计算效率和模型可解释性。

3.  **主要结果及其意义：**
    *   **深度学习在点云处理中的主导地位：** 论文强调，深度学习方法已几乎完全主导了点云处理算法，显著提升了场景补全、配准、语义分割和几何建模等任务的性能。
    *   **广泛的应用潜力：** 深度学习在点云处理中的应用范围广泛，从智能城市基础设施管理、能源管理、文化遗产保护、道路网络测绘到灾害管理、林业、农业和生态监测等，都展现出巨大的潜力。
    *   **方法论的同质性：** 尽管点云任务多样，但特征提取过程大多遵循基于体素、基于视图和基于点的方法，这表明研究人员和开发者可以在不同任务之间利用相似的算法和概念。
    *   **性能提升：** 现有工作表明，不断更新的网络架构和共享训练数据的增加，使得深度学习在基准数据集上的性能得到了显著提升。

4.  **局限性：**
    *   **泛化能力不足：** 尽管在基准数据集上表现良好，但这些方法在应用于未见过的数据集时，其泛化能力和性能一致性尚未得到充分验证。
    *   **大规模数据处理挑战：** 现有方法在处理地理空间尺度上的超大数据量时，计算效率和内存消耗仍然是主要挑战，尤其是在精细配准任务中。
    *   **模型可解释性：** 深度学习模型的“黑箱”性质阻碍了其在智能城市控制系统等领域的解释性和信任度。
    *   **数据质量和可用性：** 模型的准确性和完整性高度依赖于高质量和可用的大规模训练数据集，这在实际应用中可能难以获得。
    *   **手动工作量：** 即使有深度学习的辅助，许多建模过程（如LoD3模型的生成）仍然需要大量手动工作。

5.  **潜在的未来研究方向：**
    *   **提升泛化能力：** 未来的研究应侧重于开发具有更强泛化能力的算法，使其在应用于不同领域和场景的未见过数据集时能保持一致的性能。
    *   **优化大规模数据处理：** 需要开发更高效的算法和架构，以处理地理空间尺度上的超大数据量，并优化计算效率和内存使用。
    *   **增强模型可解释性：** 探索提高深度学习模型可解释性的方法，以增强用户对模型的信任和理解，尤其是在关键决策领域。
    *   **多源数据融合：** 进一步研究如何有效融合多源点云数据（如LiDAR、图像、多光谱数据等），以获取更丰富、更全面的场景信息。
    *   **自动化与实时应用：** 推动算法向更高自动化水平发展，并优化推理时间，以支持实时交通管理和灾害响应等应用。
    *   **填补数据差距：** 解决数据稀缺和专有数据格式问题，以促进高质量训练数据集的创建和共享。

总而言之，这篇论文为深度学习在3D点云处理领域的应用提供了一个全面的视角，强调了其在城市和环境应用中的巨大潜力，同时也明确指出了当前方法在实际部署中面临的挑战和未来研究方向。

**Key Findings:**

- In this paper, we provide a meta review on deep learning approaches
and datasets that cover a selection of critical tasks of point cloud processing
in use such as scene completion, registration, semantic segmentation, and
modeling.
- By reviewing a broad range of urban and environmental applications
these tasks can support, we identify gaps to be closed as these methods
transformed into applications and draw concluding remarks in both the
algorithmic and practical aspects of the surveyed methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12452v1)
- [arXiv](https://arxiv.org/abs/2509.12452v1)

---

<a id='2509.12197v1'></a>
## [3D Human Pose and Shape Estimation from LiDAR Point Clouds: A Review](https://arxiv.org/abs/2509.12197v1)

**Authors:** Salma Galaaoui, Eduardo Valle, David Picard, Nermin Samet

**Published:** 2025-09-15

**Categories:** cs.CV

**Abstract:**

In this paper, we present a comprehensive review of 3D human pose estimation
and human mesh recovery from in-the-wild LiDAR point clouds. We compare
existing approaches across several key dimensions, and propose a structured
taxonomy to classify these methods. Following this taxonomy, we analyze each
method's strengths, limitations, and design choices. In addition, (i) we
perform a quantitative comparison of the three most widely used datasets,
detailing their characteristics; (ii) we compile unified definitions of all
evaluation metrics; and (iii) we establish benchmark tables for both tasks on
these datasets to enable fair comparisons and promote progress in the field. We
also outline open challenges and research directions critical for advancing
LiDAR-based 3D human understanding. Moreover, we maintain an accompanying
webpage that organizes papers according to our taxonomy and continuously update
it with new studies:
https://github.com/valeoai/3D-Human-Pose-Shape-Estimation-from-LiDAR

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Salma Galaaoui, Eduardo Valle, David Picard, Nermin Samet撰写的论文“3D Human Pose and Shape Estimation from LiDAR Point Clouds: A Review”的全面摘要。

---

**论文摘要：3D Human Pose and Shape Estimation from LiDAR Point Clouds: A Review**

这篇论文对从“野外”LiDAR点云中进行3D人体姿态估计（3D HPE）和人体网格恢复（HMR）的现有方法进行了全面回顾。随着LiDAR传感器在自动驾驶和机器人领域的普及，以及相关数据集的发布，利用LiDAR数据进行人体理解成为一个日益重要的研究方向。

**1. 主要问题或研究问题：**
论文主要关注如何从稀疏、不规则、可能存在遮挡和噪声的“野外”LiDAR点云中准确地估计3D人体姿态和恢复详细的人体网格。传统基于图像和视频的方法在深度信息、隐私保护和鲁棒性方面存在局限性，而LiDAR虽然提供了精确的3D几何信息，但也带来了自身的数据挑战。

**2. 关键创新或方法论贡献：**
*   **结构化分类法：** 论文提出了一个结构化的分类法，用于对现有的3D HPE和HMR方法进行分类，并分析了它们的优势、局限性和设计选择。该分类法基于学习范式（监督、弱监督、无监督）、输入模态（仅LiDAR或多模态融合）和网络架构（如Transformer、PointNet变体等）。
*   **方法分析：** 详细分析了32项2019年至2025年间发表的研究，涵盖了稀疏性处理、Transformer作为骨干网络、超越3D姿态的辅助监督任务、合成数据学习、弱监督下的标注鸿沟弥合以及多模态融合等关键方面。
*   **数据集定量比较：** 对Waymo Open Dataset、SLOPER4D和Human-M3这三个最广泛使用的LiDAR人体姿态数据集进行了定量比较，详细阐述了它们的特性、采集方式、数据格式和内在属性（如点云密度、人体-传感器距离、3D姿态多样性等）。
*   **统一评估指标：** 整理并统一了所有用于评估LiDAR点云3D HPE和HMR方法的评估指标定义，包括MPJPE、PA-MPJPE、PCK、PEM、MPVPE、MPERE、ADE、LAE、LLE、Accel Error和Chamfer Distance等。
*   **基准测试表：** 建立了在这三个数据集上针对3D HPE和HMR任务的基准测试表，旨在促进公平比较和领域进展。

**3. 主要结果及其意义：**
论文通过对现有方法的深入分析和基准测试，揭示了LiDAR-based 3D HPE和HMR领域的最新进展和挑战。研究表明，多模态融合（特别是LiDAR与RGB图像或IMU的融合）和利用合成数据进行预训练是解决数据稀疏性和标注不足的关键策略。Transformer架构因其全局感受野和处理不规则数据的能力，在LiDAR-based任务中表现出强大潜力。弱监督方法通过2D伪标签、投影一致性和辅助任务有效弥合了标注鸿沟。这些发现为研究人员提供了清晰的路线图，以理解当前技术水平并指导未来研究方向。

**4. 论文中提及的局限性：**
*   **数据稀缺性：** 缺乏大规模、高质量标注的LiDAR数据集是核心挑战。
*   **对辅助模态的依赖：** 多数弱监督方法仍严重依赖RGB图像或IMU信号等辅助数据模态。
*   **合成数据生成：** 当前合成数据生成管道存在领域不匹配（室内姿态与室外场景）和真实性差距（未能完全捕捉真实LiDAR传感器的噪声、稀疏性和视角特性）。
*   **相机参数依赖：** 多数多模态方法高度依赖精确的相机参数进行2D-3D对应，这在实际应用中带来了挑战。
*   **传感器域差异：** 不同LiDAR传感器特性（点云密度、范围、噪声模式）导致模型在不同数据集之间泛化能力差。
*   **扫描模式差异：** 不同LiDAR扫描模式（NRS和RMB）产生结构不同的点云分布，需要鲁棒的架构或适应策略。
*   **弱监督HMR的探索不足：** 相比3D HPE，弱监督HMR方法的研究相对较少。

**5. 潜在的未来研究方向：**
*   **减少对辅助模态的依赖：** 探索仅使用LiDAR数据实现弱监督HPE和HMR的方法，例如通过伪标签、自训练或对比学习。
*   **整合时间信息：** 利用LiDAR帧序列的时间连贯性，提取时间线索以增强姿态估计精度，而无需额外监督。
*   **更真实的合成数据生成：** 发展能够直接从真实世界分布中生成合成LiDAR数据的方法，特别是利用扩散模型等生成式模型。
*   **数据高效学习：** 预训练模型并以最少监督进行微调，以提高LiDAR-based HPE和HMR的数据效率。
*   **弱监督3D HMR：** 深入探索专门针对HMR的弱监督方法。
*   **消除相机参数依赖：** 开发可学习模块，实现端到端对齐，无需显式校准。
*   **域适应技术：** 针对不同LiDAR传感器特性和扫描模式之间的域差异，开发域适应技术，以提高模型的泛化能力。

---

这篇综述论文为LiDAR-based 3D人体理解领域提供了一个全面的概览，不仅系统地分类和分析了现有方法，还通过量化比较和基准测试为未来的研究奠定了基础，并指明了关键的开放挑战和有前景的研究方向。

**Key Findings:**

- In this paper, we present a comprehensive review of 3D human pose estimation
and human mesh recovery from in-the-wild LiDAR point clouds.
- Moreover, we maintain an accompanying
webpage that organizes papers according to our taxonomy and continuously update
it with new studies:
https://github.com/valeoai/3D-Human-Pose-Shape-Estimation-from-LiDAR

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12197v1)
- [arXiv](https://arxiv.org/abs/2509.12197v1)

---

<a id='2509.13210v1'></a>
## [Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance](https://arxiv.org/abs/2509.13210v1)

**Authors:** Ligang Chang, Shengkai Xu, Liangchang Shen, Binhan Xu, Junqiao Wang, Tianyu Shi, Yanhui Du

**Published:** 2025-09-16

**Categories:** cs.CV, I.2.10; I.4.8

**Abstract:**

Violence detection in public surveillance is critical for public safety. This
study addresses challenges such as small-scale targets, complex environments,
and real-time temporal analysis. We propose Vi-SAFE, a spatial-temporal
framework that integrates an enhanced YOLOv8 with a Temporal Segment Network
(TSN) for video surveillance. The YOLOv8 model is optimized with GhostNetV3 as
a lightweight backbone, an exponential moving average (EMA) attention
mechanism, and pruning to reduce computational cost while maintaining accuracy.
YOLOv8 and TSN are trained separately on pedestrian and violence datasets,
where YOLOv8 extracts human regions and TSN performs binary classification of
violent behavior. Experiments on the RWF-2000 dataset show that Vi-SAFE
achieves an accuracy of 0.88, surpassing TSN alone (0.77) and outperforming
existing methods in both accuracy and efficiency, demonstrating its
effectiveness for public safety surveillance. Code is available at
https://anonymous.4open.science/r/Vi-SAFE-3B42/README.md.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ligang Chang等人撰写的论文“Vi-SAFE: A Spatial-Temporal Framework for Efficient Violence Detection in Public Surveillance”的全面摘要。

---

### 论文《Vi-SAFE: 一种用于公共监控中高效暴力检测的时空框架》摘要

**1. 主要问题或研究问题：**
该论文旨在解决公共监控中高效暴力行为检测的挑战。具体来说，它关注于如何克服小目标、复杂环境以及实时时序分析的难题，以提高公共安全水平。现有的方法往往在准确性、计算效率或实时性方面存在不足，尤其是在资源受限的边缘设备上部署时。

**2. 关键创新或方法论贡献：**
Vi-SAFE框架的核心创新在于其独特的时空集成方法和对YOLOv8模型的优化：
*   **时空集成框架 (Vi-SAFE)：** 论文提出了一种新颖的端到端时空框架，将优化的YOLOv8s目标检测器与时序分割网络 (TSN) 相结合。YOLOv8s负责在视频帧中准确地定位潜在暴力活动中的个体（即提取人体区域），而TSN则对这些感兴趣区域 (ROIs) 的时序动态进行建模，以判断暴力行为。这种双2D CNN结构在实现高准确性的同时，保持了较低的计算复杂度。
*   **YOLOv8s模型优化 (GE-YOLOv8)：**
    *   **轻量级骨干网络：** 将YOLOv8s的骨干网络替换为GhostNetV3，通过廉价的线性操作生成额外的特征图，显著减少了FLOPs和参数，同时保持了准确的特征表示。
    *   **EMA注意力机制：** 在骨干网络的后期层中引入了指数移动平均 (EMA) 注意力机制，通过重新加权特征图来突出信息区域并抑制背景噪声，从而提高对小目标或遮挡目标的检测，并增强复杂场景中的时序一致性。
    *   **剪枝技术：** 应用基于GroupNorm重要性的通道剪枝技术，进一步压缩模型，在保持准确性的同时降低了参数和GFLOPs。

**3. 主要结果及其意义：**
*   **GE-YOLOv8的性能：** 经过GhostNetV3、EMA注意力机制和剪枝优化后的GE-YOLOv8模型，在保持高准确性（mAP 0.737）的同时，显著减少了参数（减少约一半）和GFLOPs（减少约一半），使其非常适合实时边缘部署。
*   **Vi-SAFE的整体性能：** 在RWF-2000数据集上的实验表明，Vi-SAFE实现了0.88的准确率，显著优于单独的TSN (0.77) 以及其他主流方法，如U-Net+LSTM (0.820) 和C3D (0.828)，甚至略微超越了Openpose+ST-GCN (0.878)。
*   **意义：** 这些结果证明了Vi-SAFE在公共安全监控中进行暴力检测的有效性、高效性和鲁棒性。其模块化架构也确保了可扩展性，便于集成额外的模块以适应更广泛的应用场景。

**4. 论文中提及的局限性：**
论文中并未明确指出当前Vi-SAFE框架的局限性，但从未来研究方向可以推断出一些潜在的改进空间：
*   **泛化能力：** 尽管Vi-SAFE在RWF-2000数据集上表现出色，但其在更复杂、更多样化的环境中的泛化能力仍有待进一步验证。
*   **优化策略：** 尽管已经进行了模型优化，但可能仍有进一步提升效率和准确性的空间。

**5. 潜在的未来研究方向：**
论文指出了以下未来研究方向：
*   **评估更多样化的数据集：** 在更复杂、更多样化的环境中评估Vi-SAFE的泛化能力。
*   **探索先进的优化策略：** 进一步研究和应用先进的优化策略，以持续提高模型的效率和性能。

---

这份摘要突出了Vi-SAFE框架在解决公共监控中暴力检测问题上的创新性，特别是在平衡准确性和计算效率方面的贡献，使其成为边缘设备部署的有力候选。

**Key Findings:**

- We propose Vi-SAFE, a spatial-temporal
framework that integrates an enhanced YOLOv8 with a Temporal Segment Network
(TSN) for video surveillance.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.13210v1)
- [arXiv](https://arxiv.org/abs/2509.13210v1)

---

<a id='2509.13172v1'></a>
## [WHU-STree: A Multi-modal Benchmark Dataset for Street Tree Inventory](https://arxiv.org/abs/2509.13172v1)

**Authors:** Ruifei Ding, Zhe Chen, Wen Fan, Chen Long, Huijuan Xiao, Yelu Zeng, Zhen Dong, Bisheng Yang

**Published:** 2025-09-16

**Categories:** cs.CV

**Abstract:**

Street trees are vital to urban livability, providing ecological and social
benefits. Establishing a detailed, accurate, and dynamically updated street
tree inventory has become essential for optimizing these multifunctional assets
within space-constrained urban environments. Given that traditional field
surveys are time-consuming and labor-intensive, automated surveys utilizing
Mobile Mapping Systems (MMS) offer a more efficient solution. However, existing
MMS-acquired tree datasets are limited by small-scale scene, limited
annotation, or single modality, restricting their utility for comprehensive
analysis. To address these limitations, we introduce WHU-STree, a cross-city,
richly annotated, and multi-modal urban street tree dataset. Collected across
two distinct cities, WHU-STree integrates synchronized point clouds and
high-resolution images, encompassing 21,007 annotated tree instances across 50
species and 2 morphological parameters. Leveraging the unique characteristics,
WHU-STree concurrently supports over 10 tasks related to street tree inventory.
We benchmark representative baselines for two key tasks--tree species
classification and individual tree segmentation. Extensive experiments and
in-depth analysis demonstrate the significant potential of multi-modal data
fusion and underscore cross-domain applicability as a critical prerequisite for
practical algorithm deployment. In particular, we identify key challenges and
outline potential future works for fully exploiting WHU-STree, encompassing
multi-modal fusion, multi-task collaboration, cross-domain generalization,
spatial pattern learning, and Multi-modal Large Language Model for street tree
asset management. The WHU-STree dataset is accessible at:
https://github.com/WHU-USI3DV/WHU-STree.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Ruifei Ding等人撰写的论文“WHU-STree: A Multi-modal Benchmark Dataset for Street Tree Inventory”的全面摘要。

---

**论文摘要：WHU-STree：用于行道树盘点任务的多模态基准数据集**

**1. 主要问题或研究问题：**
该论文旨在解决现有行道树数据集在规模、标注丰富性和模态单一性方面的局限性，这些局限性阻碍了对城市行道树进行全面分析和自动化盘点。传统的实地调查耗时费力，而现有的移动测绘系统（MMS）获取的树木数据集往往场景规模小、标注不完整或仅限于单一模态，无法支持复杂的分析任务。因此，研究者们需要一个大规模、多模态、标注丰富的行道树数据集，以推动基于深度学习的自动化行道树盘点算法的发展。

**2. 关键创新或方法学贡献：**
该论文的核心贡献是引入了WHU-STree数据集，其关键创新和方法学贡献包括：
*   **跨城市、多模态数据集成：** WHU-STree数据集首次整合了来自中国两个气候区和城市规划差异显著的城市（南京和沈阳）的同步点云和高分辨率图像数据。这种跨城市覆盖有助于评估算法的泛化能力和鲁棒性。
*   **丰富的标注：** 数据集包含21,007个标注的树木实例，涵盖50种树木物种和2个形态参数（树高和胸径）。这些详细标注支持多种任务，包括树种分类、单棵树分割和形态参数估计。
*   **多任务支持：** 凭借其独特的特性，WHU-STree能够同时支持超过10个与行道树盘点相关的任务，包括单模态或多模态输入、单任务或多任务学习、以及跨区域或区域内评估。这为开发综合性行道树盘点解决方案提供了基础。
*   **基准测试与分析：** 论文对树种分类和单棵树分割这两个关键任务的代表性基线算法进行了基准测试，并提供了深入的实验分析，强调了多模态数据融合和跨领域适用性的重要性。

**3. 主要结果及其意义：**
*   **多模态融合的优越性：** 实验结果表明，多模态方法（如TSCMDL）在树种分类任务中显著优于单一模态方法，尤其是在mIoU指标上有所提升（例如，TSCMDL比PointMLP提高了2.49%）。这强调了利用图像的纹理和光谱信息作为点云几何信息的补充，对于提高分类准确性的重要性。
*   **跨领域泛化能力：** 在跨城市评估中，所有算法都表现出稳健的性能，尤其是在WHU-STree-SY数据集上。多模态方法LCPS在F1分数上甚至超越了单模态方法，这表明多模态融合有助于缓解点云数据中的领域漂移，提高了算法在异构城市环境中的鲁棒性。
*   **多任务协作的潜力：** 初步实验表明，将树种分类整合到树木分割任务中，可以提高网络的细粒度特征捕获能力，从而间接改善整体分割效果（F1分数从69.2%提高到82.2%）。这为开发端到端的多任务学习框架提供了有力证据。
*   **挑战与机遇：** 尽管取得了进展，但算法在处理家具干扰、树冠重叠和块合并失败等复杂场景时仍面临挑战。最高的mIoU仅为64.09%，表明区分特定树种仍是一个难题。

**4. 论文中提及的局限性：**
*   **现有数据集的局限：** 论文指出，现有的MMS获取的树木数据集通常场景规模小、标注有限或仅限于单一模态，限制了其在全面分析中的实用性。
*   **算法性能的局限：** 尽管多模态融合有所改进，但现有算法的整体性能仍不足以满足实际部署的需求。例如，树种分类的最高mIoU仅为64.09%，表明在区分特定树种方面仍存在显著挑战。
*   **特定场景的挑战：** 算法在处理城市道路家具干扰、密集树冠重叠导致的欠分割以及块合并失败等复杂场景时仍存在问题。
*   **跨领域泛化的挑战：** 尽管WHU-STree-NJ数据集的丰富性有助于泛化，但不同城市间树种组成和形态的显著差异仍可能限制检测的泛化能力。

**5. 潜在的未来研究方向：**
*   **多模态融合策略：** 进一步研究任务特定的多模态融合策略，以更好地利用点云和图像数据之间的互补优势，解决投影误差和多视角不一致等问题。
*   **多任务协作框架：** 开发端到端的多任务学习模型，同时执行分类、分割和参数估计，促进跨任务知识迁移，并支持更全面的行道树盘点。
*   **跨领域泛化：** 探索能够处理领域漂移和提高算法在异构城市环境中鲁棒性的方法，例如开放词汇识别和新类别发现。未来将扩展WHU-STree到更多城市，并增加沈阳数据集的树种标注。
*   **空间模式学习：** 结合行道树的空间分布模式（如线性排列、种植间隔、拓扑关系）作为先验知识，以提高分割和分类性能，并支持城市绿化规划。
*   **多模态大语言模型（MLLM）应用：** 利用MLLM的推理和跨模态交互能力，整合专家知识和政府政策，构建一个无缝的“感知-分析-决策”闭环框架，实现智能化的行道树资产管理，包括健康状况评估、维护需求和风险评估。

---

这份摘要突出了WHU-STree数据集作为计算机视觉领域新颖且重要的基准，为解决城市行道树盘点中的实际挑战提供了坚实基础，并为未来的多模态、多任务和跨领域研究指明了方向。

**Key Findings:**

- To address these limitations, we introduce WHU-STree, a cross-city,
richly annotated, and multi-modal urban street tree dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.13172v1)
- [arXiv](https://arxiv.org/abs/2509.13172v1)

---

<a id='2509.13133v1'></a>
## [Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline](https://arxiv.org/abs/2509.13133v1)

**Authors:** Zhihao Zhang, Chunyu Lin, Lang Nie, Jiyuan Wang, Yao Zhao

**Published:** 2025-09-16

**Categories:** cs.CV

**Abstract:**

As automatic parking systems evolve, the accurate detection of parking slots
has become increasingly critical. This study focuses on parking slot detection
using surround-view cameras, which offer a comprehensive bird's-eye view of the
parking environment. However, the current datasets are limited in scale, and
the scenes they contain are seldom disrupted by real-world noise (e.g., light,
occlusion, etc.). Moreover, manual data annotation is prone to errors and
omissions due to the complexity of real-world conditions, significantly
increasing the cost of annotating large-scale datasets. To address these
issues, we first construct a large-scale parking slot detection dataset (named
CRPS-D), which includes various lighting distributions, diverse weather
conditions, and challenging parking slot variants. Compared with existing
datasets, the proposed dataset boasts the largest data scale and consists of a
higher density of parking slots, particularly featuring more slanted parking
slots. Additionally, we develop a semi-supervised baseline for parking slot
detection, termed SS-PSD, to further improve performance by exploiting
unlabeled data. To our knowledge, this is the first semi-supervised approach in
parking slot detection, which is built on the teacher-student model with
confidence-guided mask consistency and adaptive feature perturbation.
Experimental results demonstrate the superiority of SS-PSD over the existing
state-of-the-art (SoTA) solutions on both the proposed dataset and the existing
dataset. Particularly, the more unlabeled data there is, the more significant
the gains brought by our semi-supervised scheme. The relevant source codes and
the dataset have been made publicly available at
https://github.com/zzh362/CRPS-D.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zhihao Zhang等人的论文“Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline”的全面摘要。

---

### 论文摘要：Advancing Real-World Parking Slot Detection with Large-Scale Dataset and Semi-Supervised Baseline

**1. 主要问题或研究问题：**
该研究旨在解决自动泊车系统中停车位检测的准确性问题，尤其是在真实世界复杂场景下的挑战。现有数据集规模有限，且缺乏真实世界的噪声（如光照、遮挡等），导致模型泛化能力不足。此外，手动标注大规模数据集成本高昂且易出错。因此，核心问题是如何在复杂多变的真实世界环境中，实现高精度、鲁棒的停车位检测，并有效利用未标注数据以降低标注成本。

**2. 关键创新或方法论贡献：**
*   **CRPS-D大规模数据集：** 论文构建了一个名为CRPS-D的大规模真实世界停车位检测数据集。该数据集在规模上远超现有数据集，包含更多样化的光照分布、天气条件和更具挑战性的停车位变体（特别是更多倾斜停车位），场景密度更高，旨在更好地反映真实世界的复杂性。
*   **SS-PSD半监督基线模型：** 首次提出了用于停车位检测的半监督方法SS-PSD（Semi-Supervised Parking Slot Detection）。该模型基于教师-学生（Teacher-Student）框架，并引入了两项关键创新：
    *   **置信度引导掩码一致性（Confidence-Guided Mask Consistency, CGM）：** 针对未标注数据，通过可训练的置信度图为不同区域分配权重，并掩盖低置信度区域，以确保预测一致性，避免潜在错误预测的误导。
    *   **自适应特征扰动（Adaptive Feature Perturbation, Adaptive-VAT）：** 引入了一种自适应选择性的特征扰动机制，根据教师或学生模型对扰动的鲁棒性，生成更强但合理的对抗性噪声，以促进学生模型的有效训练。

**3. 主要结果及其意义：**
*   **数据集优势：** CRPS-D数据集在图像和停车位数量上远超现有数据集，且停车位密度更高，倾斜停车位比例显著增加（11.75%），为真实世界停车位检测提供了更具挑战性和代表性的基准。
*   **SS-PSD性能优越性：** 实验结果表明，SS-PSD在CRPS-D和现有数据集上均超越了现有最先进（SoTA）的全监督解决方案。尤其是在标注数据较少的情况下（例如1/24的标注比例），SS-PSD相较于DMPR-PS和GCN在APparking-slot上分别取得了19.02%和16.49%的显著提升。
*   **半监督学习的有效性：** 论文强调，SS-PSD能够有效利用未标注数据，随着未标注数据量的增加，性能提升越显著，证明了其在标签稀缺场景下的实用性和可扩展性。
*   **组件有效性：** 消融实验证实了CGM一致性和Adaptive-VAT对模型性能的积极贡献，CGM一致性使APparking-slot提升了3.60%，Adaptive-VAT进一步提升了0.82%。

**4. 论文中提及的局限性：**
尽管SS-PSD模型在大多数场景下表现出色，但在极端环境条件下仍存在局限性。论文中列举了以下失败案例：
*   **复杂光照条件：** 地下车库中光照不足、阴影和地面反射会导致误检。
*   **户外眩光：** 强烈的户外眩光会影响停车位检测。
*   **恶劣天气（大雨）：** 降低对比度、表面反射和部分遮挡的停车位标记会降低检测准确性。
*   **标记磨损：** 褪色或严重损坏的停车位标记难以辨别，导致漏检。
*   **严重遮挡：** 车辆或其他物体遮挡停车位视图时，模型性能下降。
*   **夜间场景：** 有限的光照和传感器噪声显著降低视觉清晰度，导致漏检。

**5. 潜在的未来研究方向：**
*   **扩展功能：** 将模型扩展到包含占用检测，以判断停车位是否被占用或空闲，从而构建更全面实用的自动泊车系统。
*   **多源数据融合：** 结合其他数据源，如视频序列中的时间线索或传感器融合，以提高模型在恶劣条件下的鲁棒性。
*   **V2X技术集成：** 探索将V2X技术（如RSU辅助的协同定位）集成到系统中，以增强在GNSS受限环境（如地下停车场和城市峡谷）中的系统鲁棒性，可能同时改善定位和检测性能。
*   **模型优化：** 进一步优化模型以处理更广泛的环境条件，并扩展其功能以支持其他交通相关应用。

---

这篇论文通过引入大规模、高挑战性的CRPS-D数据集和创新的SS-PSD半监督学习框架，为真实世界停车位检测领域做出了重要贡献。其半监督方法在标签稀缺场景下表现出显著优势，为未来自动泊车系统的发展提供了有力的技术支持和研究方向。

**Key Findings:**

- Additionally, we develop a semi-supervised baseline for parking slot
detection, termed SS-PSD, to further improve performance by exploiting
unlabeled data.
- Experimental results demonstrate the superiority of SS-PSD over the existing
state-of-the-art (SoTA) solutions on both the proposed dataset and the existing
dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.13133v1)
- [arXiv](https://arxiv.org/abs/2509.13133v1)

---

<a id='2509.13083v1'></a>
## [Using KL-Divergence to Focus Frequency Information in Low-Light Image Enhancement](https://arxiv.org/abs/2509.13083v1)

**Authors:** Yan Xingyang, Huang Xiaohong, Zhang Zhao, You Tian, Xu Ziheng

**Published:** 2025-09-16

**Categories:** cs.CV

**Abstract:**

In the Fourier domain, luminance information is primarily encoded in the
amplitude spectrum, while spatial structures are captured in the phase
components. The traditional Fourier Frequency information fitting employs
pixel-wise loss functions, which tend to focus excessively on local information
and may lead to global information loss. In this paper, we present LLFDisc, a
U-shaped deep enhancement network that integrates cross-attention and gating
mechanisms tailored for frequency-aware enhancement. We propose a novel
distribution-aware loss that directly fits the Fourier-domain information and
minimizes their divergence using a closed-form KL-Divergence objective. This
enables the model to align Fourier-domain information more robustly than with
conventional MSE-based losses. Furthermore, we enhance the perceptual loss
based on VGG by embedding KL-Divergence on extracted deep features, enabling
better structural fidelity. Extensive experiments across multiple benchmarks
demonstrate that LLFDisc achieves state-of-the-art performance in both
qualitative and quantitative evaluations. Our code will be released at:
https://github.com/YanXY000/LLFDisc

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供对Yan Xingyang等人撰写的论文“Using KL-Divergence to Focus Frequency Information in Low-Light Image Enhancement”的全面摘要。

---

### 论文《Using KL-Divergence to Focus Frequency Information in Low-Light Image Enhancement》全面摘要

**1. 论文主要问题或研究问题：**
低光照图像增强（LLIE）是一个核心的计算机视觉任务，但现有方法在处理低光照图像时存在局限性。具体来说，传统的基于傅里叶频率信息的增强方法通常采用像素级损失函数（如MSE），这导致模型过度关注局部信息，可能忽略全局频率分布，从而影响增强图像的整体质量和结构保真度。此外，现有的傅里IE增强方法未能充分利用振幅信息，并且在空间域特征提取时容易放大噪声，而基于Retinex的方法可能导致纹理失真和泛化能力差。

**2. 关键创新或方法论贡献：**
本文提出了一个名为LLFDisc的U形深度增强网络，并引入了多项关键创新：

*   **傅里叶KL散度损失（Fourier KL-Divergence Loss）：** 针对传统像素级损失的局限性，作者提出了一种新颖的、分布感知的傅里叶域损失函数。该损失函数将预测图像和真实图像的振幅和相位谱建模为高斯分布，并通过计算它们之间的KL散度来最小化差异。这种方法使得模型能够更鲁棒地对齐傅里叶域信息，捕获跨频率的联合分布结构，而非独立处理每个频率分量，从而避免了局部偏差和全局信息损失。
*   **增强型VGG感知损失（Enhanced VGG Perceptual Loss with KL-Divergence）：** 为了进一步提升结构保真度，作者将KL散度嵌入到VGG感知损失中。通过在VGG网络提取的深层特征上应用KL散度，模型能够衡量高层特征表示之间的分布相似性，从而更好地捕捉图像的感知质量和结构细节。
*   **LLFDisc网络架构：** 提出了一种流线型的U形编码器-解码器网络LLFDisc，它集成了交叉注意力（Cross-Attention）和门控机制，这些机制专为频率感知增强而设计。网络包含三个特征提取阶段，每个阶段都集成了增强型轻量级交叉注意力（EnhancedLCA）模块。
*   **EnhancedLCA模块：** 该模块是LCA的改进，包含DANCE（暗区和噪声校正增强）、IEL（信息增强层）、SE（Squeeze-and-Excitation）和CAB（交叉注意力块）模块。
    *   **DANCE模块：** 这是一个新提出的模块，用于暗区和噪声校正增强，通过噪声感知模块抑制噪声，暗区增强模块恢复低光照区域的细节，以及通道注意力模块重新加权特征。
    *   **IEL模块：** 通过门控机制增强特征选择性，动态调节特征流，捕获重要信息并抑制无关或噪声特征。
    *   **CAB模块：** 通过Query和Key-Value对机制选择性地融合特征，处理大尺度光照不均匀性，并动态调整注意力权重。

**3. 主要结果及其意义：**
*   **最先进的性能：** 在多个基准数据集（包括LOLv1、LOLv2-Real、LOLv2-Synthetic和LSRW-Huawei）上进行了广泛的实验。LLFDisc在定性和定量评估（PSNR、SSIM、LPIPS、NIQE）方面均取得了最先进的性能。
*   **傅里叶域拟合的有效性：** 实验证明，基于KL散度的损失函数在傅里叶域信息拟合方面显著优于基于MSE的损失，在定量指标和感知质量上均表现出色。
*   **结构保真度提升：** 结合KL散度的VGG感知损失显著提升了模型的表示能力和结构保真度。
*   **效率和鲁棒性：** LLFDisc网络设计流线型，参数量和计算复杂度较低（例如，在LOLv1上FLOPS为10.93G，参数量为0.923M），同时在不同光照条件下表现出良好的泛化能力和鲁棒性。
*   **CAM可视化：** CAM可视化结果显示，使用完整损失函数训练的模型（Full-CAM）生成的注意力热图与真实图像（GT-CAM）高度相似，表明模型有效捕捉了与真实激活对齐的感兴趣区域。
*   **消融研究：** 消融实验验证了所提出模块（CAB、IEL、SE、DANCE）和损失函数（傅里叶KL损失、VGG KL损失）的有效性，每个组件都对提升图像增强性能有显著贡献。

**4. 论文中提及的局限性：**
论文中并未明确提及当前方法的具体局限性。然而，从其创新点和改进方向来看，可以推断出：
*   虽然KL散度损失在傅里叶域和感知域表现出色，但其计算复杂性可能高于简单的像素级损失，尽管论文强调了其闭式解的优势。
*   将振幅和相位谱建模为高斯分布是一种简化，可能无法完全捕捉所有复杂的频率分布特征。
*   模型在处理极端复杂或特定类型的低光照场景（如极度黑暗、强噪声干扰等）时，可能仍有进一步优化的空间。

**5. 潜在的未来研究方向：**
*   **更复杂的频率分布建模：** 探索除了高斯分布之外，更复杂的概率分布模型来拟合傅里叶域的振幅和相位信息，以期捕捉更精细的频率特征。
*   **自适应损失权重：** 进一步研究自适应地调整复合损失函数中各项损失权重的策略，使其能够根据输入图像的特性或训练阶段动态优化。
*   **跨模态融合：** 结合其他模态信息（如深度、热成像等）来辅助低光照图像增强，尤其是在极端低光照或复杂场景下。
*   **实时应用优化：** 进一步优化网络架构和损失函数，以满足实时低光照图像增强应用的需求，例如在自动驾驶或监控系统中。
*   **无监督/自监督学习：** 探索在没有配对低光照-正常光照图像的情况下，利用KL散度进行无监督或自监督的低光照图像增强方法。

---

**Key Findings:**

- In this paper, we present LLFDisc, a
U-shaped deep enhancement network that integrates cross-attention and gating
mechanisms tailored for frequency-aware enhancement.
- We propose a novel
distribution-aware loss that directly fits the Fourier-domain information and
minimizes their divergence using a closed-form KL-Divergence objective.
- Extensive experiments across multiple benchmarks
demonstrate that LLFDisc achieves state-of-the-art performance in both
qualitative and quantitative evaluations.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.13083v1)
- [arXiv](https://arxiv.org/abs/2509.13083v1)

---

<a id='2509.12990v1'></a>
## [Dual-Stage Reweighted MoE for Long-Tailed Egocentric Mistake Detection](https://arxiv.org/abs/2509.12990v1)

**Authors:** Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Sicong Li, Qingming Huang

**Published:** 2025-09-16

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

In this report, we address the problem of determining whether a user performs
an action incorrectly from egocentric video data. To handle the challenges
posed by subtle and infrequent mistakes, we propose a Dual-Stage Reweighted
Mixture-of-Experts (DR-MoE) framework. In the first stage, features are
extracted using a frozen ViViT model and a LoRA-tuned ViViT model, which are
combined through a feature-level expert module. In the second stage, three
classifiers are trained with different objectives: reweighted cross-entropy to
mitigate class imbalance, AUC loss to improve ranking under skewed
distributions, and label-aware loss with sharpness-aware minimization to
enhance calibration and generalization. Their predictions are fused using a
classification-level expert module. The proposed method achieves strong
performance, particularly in identifying rare and ambiguous mistake instances.
The code is available at https://github.com/boyuh/DR-MoE.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Boyu Han等人撰写的论文“Dual-Stage Reweighted MoE for Long-Tailed Egocentric Mistake Detection”的全面摘要。

---

### 论文摘要：Dual-Stage Reweighted MoE for Long-Tailed Egocentric Mistake Detection

这篇论文提出了一种名为“双阶段重加权专家混合（Dual-Stage Reweighted Mixture-of-Experts, DR-MoE）”的框架，旨在解决从第一视角视频数据中检测用户操作错误的问题。该任务的挑战在于错误通常是细微、不频繁且数据分布严重不平衡的。

**1. 主要问题或研究问题：**
论文的核心研究问题是：如何有效地从第一视角视频数据中识别用户是否执行了不正确的操作（即“错误检测”），尤其是在错误事件稀有、细微且数据分布高度不平衡的情况下。传统的动作识别方法难以应对这种需要更精细时间分析和对用户行为细微偏差高度敏感的任务。

**2. 关键创新或方法论贡献：**
DR-MoE框架通过以下两个阶段的创新性结合来解决上述挑战：

*   **第一阶段：特征级专家混合（Feature Mixture-of-Experts, F-MoE）**
    *   该阶段利用两个基于ViViT的模型作为特征提取专家：一个**冻结的ViViT模型**用于捕捉通用的时空语义先验（粗粒度动作特征），另一个是经过**LoRA（低秩适应）微调的ViViT模型**，专注于对错误敏感的细粒度线索。
    *   这两个专家的输出通过一个可学习的F-MoE模块进行融合，该模块根据输入特性动态调整每个专家的贡献，生成统一的联合特征表示。

*   **第二阶段：分类级专家混合（Classification Mixture-of-Experts, C-MoE）**
    *   融合后的特征被送入三个独立优化的分类器，每个分类器都针对长尾识别问题采用不同的优化目标：
        *   **重加权交叉熵损失（Reweighted Cross-Entropy Loss）**：通过为稀有错误类别分配更高的权重来缓解类别不平衡问题，提高对欠表示错误实例的召回率。
        *   **AUC损失（AUC Loss）**：直接优化ROC曲线下面积，旨在提高正负实例之间的排名质量，这在错误类别欠表示时至关重要。
        *   **结合锐度感知最小化（SAM）的标签感知损失（Label-Aware Loss, LA Loss）**：通过调整logits并明确优化平坦最小值，促进更校准和鲁棒的决策边界，增强泛化能力。
    *   这些分类器的预测结果通过一个C-MoE模块进行自适应融合，该模块根据输入动态加权每个专家的贡献，从而在多样化的数据条件下实现灵活的决策。

**3. 主要结果及其意义：**
论文在HoloAssist 2025比赛的错误检测任务上进行了实验，并取得了显著的性能提升。
*   与Random和TimeSformer等基线模型相比，DR-MoE方法在F-score上实现了显著改进。
*   特别是在识别稀有和模糊的错误实例方面，该方法表现出强大的性能。
*   值得注意的是，DR-MoE仅使用RGB模态输入，就达到了甚至超越了依赖多模态输入的模型（如UNICT Solution）的竞争力，尤其是在错误召回率方面有显著提升（从UNICT Solution的0.09提升到DR-MoE的0.63）。
这些结果表明，DR-MoE框架通过结合互补的建模策略（包括特征提取和分类层面），能够有效应对长尾和细微错误检测的挑战，提高了模型的鲁棒性和准确性。

**4. 论文中提及的局限性：**
论文中并未明确提及当前方法的具体局限性。然而，从其方法论的复杂性来看，潜在的局限性可能包括：
*   **计算成本：** 结合多个ViViT模型、LoRA微调以及多专家分类器可能会增加模型的训练和推理成本。
*   **超参数调优：** 多个损失函数和专家混合模块引入了更多的超参数，可能需要精细的调优才能达到最佳性能。
*   **泛化性：** 尽管论文强调了泛化能力，但对于HoloAssist数据集之外的更广泛、更多样化的第一视角错误检测场景，其泛化能力仍需进一步验证。

**5. 潜在的未来研究方向：**
论文中没有明确提出未来的研究方向，但基于其贡献和潜在局限性，可以推断出以下方向：
*   **模型效率优化：** 探索更轻量级的专家模型或更高效的专家混合机制，以降低计算成本，使其更适用于实时或资源受限的应用。
*   **多模态融合的进一步探索：** 尽管当前方法仅使用RGB模态就表现出色，但结合其他模态（如深度、手部姿态、眼动追踪）的更高级融合策略，可能会进一步提升性能，尤其是在更复杂或更模糊的错误场景中。
*   **可解释性：** 深入研究F-MoE和C-MoE模块中专家权重分配的机制，以提高模型决策的可解释性，帮助理解模型在不同情况下如何权衡不同信息源。
*   **自监督或半监督学习：** 鉴于错误事件的稀有性，探索利用未标记数据进行自监督或半监督学习的方法，以进一步缓解数据不平衡问题。

---

**Key Findings:**

- To handle the challenges
posed by subtle and infrequent mistakes, we propose a Dual-Stage Reweighted
Mixture-of-Experts (DR-MoE) framework.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12990v1)
- [arXiv](https://arxiv.org/abs/2509.12990v1)

---

<a id='2509.12888v1'></a>
## [Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing](https://arxiv.org/abs/2509.12888v1)

**Authors:** Weiming Chen, Zhihan Zhu, Yijia Wang, Zhihai He

**Published:** 2025-09-16

**Categories:** cs.CV, cs.AI

**Abstract:**

Rectified flow (RF) models have recently demonstrated superior generative
performance compared to DDIM-based diffusion models. However, in real-world
applications, they suffer from two major challenges: (1) low inversion accuracy
that hinders the consistency with the source image, and (2) entangled
multimodal attention in diffusion transformers, which hinders precise attention
control. To address the first challenge, we propose an efficient high-order
inversion method for rectified flow models based on the Runge-Kutta solver of
differential equations. To tackle the second challenge, we introduce Decoupled
Diffusion Transformer Attention (DDTA), a novel mechanism that disentangles
text and image attention inside the multimodal diffusion transformers, enabling
more precise semantic control. Extensive experiments on image reconstruction
and text-guided editing tasks demonstrate that our method achieves
state-of-the-art performance in terms of fidelity and editability. Code is
available at https://github.com/wmchen/RKSovler_DDTA.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Weiming Chen, Zhihan Zhu, Yijia Wang, Zhihai He撰写的论文“Runge-Kutta Approximation and Decoupled Attention for Rectified Flow Inversion and Semantic Editing”的全面摘要。

---

**论文摘要：Runge-Kutta近似与解耦注意力在整流流反演和语义编辑中的应用**

这篇论文由Weiming Chen等人撰写，旨在解决整流流（Rectified Flow, RF）模型在实际应用中面临的两个关键挑战：低反演精度和多模态注意力纠缠。RF模型在生成性能上已超越基于DDIM的扩散模型，但在图像重建和语义编辑任务中仍存在不足。

**1. 主要问题或研究问题：**
论文主要关注RF模型在实际应用中的两个核心问题：
*   **低反演精度：** RF模型在将给定图像反演回其对应的噪声样本时，难以保持与源图像的一致性，这限制了其在图像重建和编辑中的应用。
*   **多模态注意力纠缠：** 在多模态扩散Transformer（MM-DiT）中，文本和图像注意力紧密耦合，导致难以进行精确的语义控制，从而影响了文本引导图像编辑的灵活性和准确性。

**2. 关键创新或方法论贡献：**
为了解决上述挑战，论文提出了两项主要创新：
*   **基于Runge-Kutta（RK）求解器的高阶反演方法：** 针对低反演精度问题，作者将数值分析中的Runge-Kutta方法引入RF采样过程，提出了一种高效的高阶求解器，用于RF模型的微分方程过程。这种方法能够更精确地近似微分轨迹，从而提高反演保真度。
*   **解耦扩散Transformer注意力（Decoupled Diffusion Transformer Attention, DDTA）：** 为了解决多模态注意力纠缠问题，论文引入了DDTA机制。该机制通过深入MM-DiT的内部结构，将文本和图像注意力解耦，从而实现更精确的语义控制。这使得在文本引导的图像编辑中，能够更灵活地利用源信息，平衡保真度和可编辑性之间的权衡。

**3. 主要结果及其意义：**
论文通过在图像重建和文本引导编辑任务上的广泛实验，验证了所提方法的有效性：
*   **图像重建任务：** RK求解器显著提高了反演保真度，在PSNR（峰值信噪比）上实现了高达2.39 dB的显著增益，优于现有RF反演方法。这表明高阶反演技术能有效克服RF潜在空间的稀疏性问题。
*   **文本引导编辑任务：** 结合RK求解器和DDTA的方法在保真度和可编辑性方面均达到了最先进的性能。DDTA机制通过解耦文本和图像注意力，实现了对语义编辑的精细控制，从而在保持源图像一致性的同时，提高了编辑的准确性。用户研究也进一步证实了该方法在编辑质量和忠实度方面的优越性。
*   **效率：** 论文的方法在显著减少采样步骤的情况下，实现了最佳的整体性能，表明了其卓越的效率。

**4. 论文中提及的局限性：**
论文也坦诚地指出了当前方法的局限性：
*   **计算开销：** 尽管RK求解器显著提高了保真度，但其高阶建模引入了额外的计算开销。
*   **内存消耗：** 保存解耦注意力图会带来显著的内存消耗。

**5. 潜在的未来研究方向：**
基于上述局限性，论文提出了未来的研究方向：
*   **低计算开销的高阶求解器：** 开发计算开销更低的高阶求解器，以提高RF模型的实用性。
*   **高效的注意力保持机制：** 设计更高效的注意力保持机制，以进一步优化内存使用和计算效率。

---

总而言之，这篇论文为RF模型在图像反演和语义编辑中的应用提供了重要的进展。通过引入Runge-Kutta近似和解耦注意力机制，作者有效地解决了RF模型在实际应用中面临的精度和控制性挑战，为未来基于扩散模型的图像生成和编辑技术开辟了新的道路。

**Key Findings:**

- To address the first challenge, we propose an efficient high-order
inversion method for rectified flow models based on the Runge-Kutta solver of
differential equations.
- To tackle the second challenge, we introduce Decoupled
Diffusion Transformer Attention (DDTA), a novel mechanism that disentangles
text and image attention inside the multimodal diffusion transformers, enabling
more precise semantic control.
- Extensive experiments on image reconstruction
and text-guided editing tasks demonstrate that our method achieves
state-of-the-art performance in terms of fidelity and editability.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.12888v1)
- [arXiv](https://arxiv.org/abs/2509.12888v1)

---

