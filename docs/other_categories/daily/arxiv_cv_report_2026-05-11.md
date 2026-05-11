time: 20260511

# Arxiv Computer Vision Papers - 2026-05-11

## Executive Summary

以下是为您准备的每日执行摘要（中文），涵盖2026年5月8日发表的10篇Arxiv计算机视觉论文，旨在帮助研究人员快速把握领域内的重要进展。

---

### **每日执行摘要：计算机视觉前沿 (2026-05-08)**

#### **1. 主要主题与趋势**

本日论文展现了几个清晰的交叉趋势：
- **多模态与统一框架**：多篇工作致力于融合不同数据源（文本、图像、视频、事件、轨迹）或任务（生成、理解、控制）于统一的模型或框架中，如自动驾驶数据统一（123D）、多模态生成（STARFlow2）、以及视觉-语言-动作模型（One Token Per Frame）。
- **流模型与扩散模型的深化应用**：流匹配（Flow Matching）和归一化流（Normalizing Flows）技术持续受到关注，不仅在图像生成（Flow-OPD, STARFlow2）中优化，还向轨迹建模（Normalizing Trajectory Models）等新领域拓展。
- **具身智能与主动视觉**：面向机器人学习，涌现出强调主动感知（TAVIS）和高效视觉表征（One Token Per Frame）的工作，旨在提升智能体在真实物理世界中的交互能力。
- **文档理解与视觉SLAM的务实推进**：文档解析领域提出了更全面的基准（PureDocBench），而SLAM则向异步事件相机（AERO-VIS）的实时应用迈进，显示出从实验室到真实场景的落地趋势。

#### **2. 重要与创新性论文**

- **最具系统性推进：** **“123D: Unifying Multi-Modal Autonomous Driving Data at Scale”** (Dauner et al.) 针对自动驾驶数据碎片化的痛点，提出了一个大规模、统一的多模态数据集和框架。这对于推动自动驾驶算法从实验室走向真实复杂场景具有显著的实用价值和影响力。
- **最具前瞻性设计：** **“One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy”** (Tang et al.) 挑战了视觉语言动作（VLA）模型中“高视觉带宽=更好性能”的常规假设。提出每帧仅用一个Token表示视觉信息，大幅降低了计算和存储需求。若其有效性得以验证，可能成为下一代轻量级具身智能模型的关键设计原则。

#### **3. 新兴研究方向与技术**

- **视觉-语言-动作（VLA）模型的极致轻量化**：One Token Per Frame 与 TAVIS 共同指向一个方向：如何以极低的计算成本，让机器人学习模型在实时、低延迟的交互环境中稳定运行。
- **异步事件驱动的实时SLAM**：AERO-VIS展示了异步事件相机在传统同步框架之外的潜力，对于高动态范围或快速运动场景下的机器人导航意义重大。这是一个从研究走向工程化的关键一步。
- **“测试时缩放”的反思**：Rethinking Dense Optical Flow without Test-Time Scaling 挑战了近年来依赖测试时计算量提升性能的主流范式。这项工作可能引导光流估计回归到更高效的模型设计，而非依赖昂贵的后处理。
- **基于流模型的轨迹规范化**：Normalizing Trajectory Models 将流模型引入轨迹生成领域（如路径规划、动力学建模），为解决非欧几里得空间下的概率建模与生成提供了新思路。

#### **4. 推荐全文阅读的论文**

1.  **“123D: Unifying Multi-Modal Autonomous Driving Data at Scale”** - **必读**。如果你是从事自动驾驶、多模态学习或大规模数据集研究的学者，这是本周最重要的论文。它解决了数据孤岛这一核心问题，其方法论和数据集将成为后续工作的基础。

2.  **“One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy”** - **推荐**。对具身智能、机器人学习、以及如何将视觉信息高效压缩并融入语言/动作模型的从业者极具启发。其简洁而反直觉的思路值得仔细推敲。

3.  **“Delta-Adapter: Scalable Exemplar-Based Image Editing with Single-Pair Supervision”** - **推荐**。在图像编辑领域，该工作以极弱监督（单对图像）实现了可扩展的基于范例的编辑，对于追求少样本学习和可控生成的AIGC研究者是重要参考。

4.  **“TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning”** - **推荐**。这是模仿学习领域一个高质量、聚焦于主动视觉和预期注视的基准。对研究智能体“看”与“做”之间协同关系的实验室，是不可多得的评估平台。

---

**总结**：今天的论文清单显示，计算机视觉领域正稳步迈向更统一、更轻量、更具物理交互能力的未来。**多模态统一**和**具身智能的高效推理**是两条最清晰的增长主线。建议优先阅读123D和One Token Per Frame，以把握本周最重要的两个突破方向。

---

## Table of Contents

1. [123D: Unifying Multi-Modal Autonomous Driving Data at Scale](#2605.08084v1)
2. [AERO-VIS: Asynchronous Event-based Real-time Onboard Visual-Inertial SLAM](#2605.07885v1)
3. [How Far Is Document Parsing from Solved? PureDocBench: A Source-TraceableBenchmark across Clean, Degraded, and Real-World Settings](#2605.07492v1)
4. [Normalizing Trajectory Models](#2605.08078v1)
5. [Flow-OPD: On-Policy Distillation for Flow Matching Models](#2605.08063v1)
6. [STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation](#2605.08029v1)
7. [Rethinking Dense Optical Flow without Test-Time Scaling](#2605.08000v1)
8. [TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning](#2605.07943v1)
9. [Delta-Adapter: Scalable Exemplar-Based Image Editing with Single-Pair Supervision](#2605.07940v1)
10. [One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy](#2605.07931v1)

---

## Papers

<a id='2605.08084v1'></a>
## [123D: Unifying Multi-Modal Autonomous Driving Data at Scale](https://arxiv.org/abs/2605.08084v1)

**Authors:** Daniel Dauner, Valentin Charraut, Bastian Berle, Tianyu Li, Long Nguyen, Jiabao Wang, Changhui Jing, Maximilian Igl, Holger Caesar, Boris Ivanovic, Yiyi Liao, Andreas Geiger, Kashyap Chitta

**Published:** 2026-05-08

**Categories:** cs.RO, cs.CV

**Abstract:**

The pursuit of autonomous driving has produced one of the richest sensor data collections in all of robotics. However, its scale and diversity remain largely untapped. Each dataset adopts different 2D and 3D modalities, such as cameras, lidar, ego states, annotations, traffic lights, and HD maps, with different rates and synchronization schemes. They come in fragmented formats requiring complex dependencies that cannot natively coexist in the same development environment. Further, major inconsistencies in annotation conventions prevent training or measuring generalization across multiple datasets. We present 123D, an open-source framework that unifies such multi-modal driving data through a single API. To handle synchronization, we store each modality as an independent timestamped event stream with no prescribed rate, enabling synchronous or asynchronous access across arbitrary datasets. Using 123D, we consolidate eight real-world driving datasets spanning 3,300 hours and 90,000 kilometers, together with a synthetic dataset with configurable collection scripts, and provide tools for data analysis and visualization. We conduct a systematic study comparing annotation statistics and assessing each dataset's pose and calibration accuracy. Further, we showcase two applications 123D enables: cross-dataset 3D object detection transfer and reinforcement learning for planning, and offer recommendations for future directions. Code and documentation are available at https://github.com/kesai-labs/py123d.

**Analysis:**

### 1. 摘要翻译
自动驾驶领域产生了极其丰富的传感器数据，但其规模和多样性仍未得到充分利用。现有数据集采用不同的2D和3D模态（相机、激光雷达、自车状态、标注、交通灯、高清地图等），且具有不同的采样率和同步机制，碎片化的格式导致开发环境难以兼容。此外，标注惯例的不一致阻碍了跨数据集的训练或泛化评估。我们提出了123D，一个通过单一API统一多模态自动驾驶数据的开源框架。为处理同步问题，我们将每个模态存储为独立的、无固定频率的时间戳事件流，从而实现跨任意数据集的同步或异步访问。利用123D，我们整合了涵盖3,300小时和90,000公里的八个真实世界数据集，以及一个具有可配置收集脚本的合成数据集，并提供了数据分析和可视化工具。我们进行了系统的研究，比较了标注统计信息，并评估了各数据集的位姿和标定精度。此外，我们展示了123D支持的两个应用：跨数据集3D目标检测迁移和规划强化学习，并为未来的研究方向提出了建议。

### 2. 方法动机分析
*   **驱动力**：为了解决自动驾驶数据集碎片化、异构化导致的“孤岛效应”，通过整合现有资源实现更大规模、更多样化的模型训练，从而提升泛化能力。
*   **痛点**：当前数据集在存储格式、模态同步机制（采样频率差异）、标注分类法（Label Taxonomies）以及物理定义（坐标系）上存在根本性的不一致，开发者难以在多个数据集上进行联合实验。
*   **研究假设**：通过将碎片化的异构数据集转换为基于时间戳事件流（Timestamped Event Stream）的统一接口，可以消除跨数据集训练的门槛，且通过数据混合训练能显著提升模型对未见域的泛化性能。

### 3. 方法设计详解
*   **流程总结**：
    1.  **解析与转换（Dataset Parser）**：为每个数据集编写专用的Parser，将原始数据转换为统一的Apache Arrow IPC格式。
    2.  **统一格式存储（Log/Map Writer）**：基于Apache Arrow建立统一格式。模态以事件流形式存储，并通过独立的`sync`表记录对齐关系，支持异步访问。
    3.  **API交互层（Scene/Map API）**：提供轻量级的Scene对象，通过LRU缓存实现按需加载。支持时间戳查询、频率重采样等操作。
    4.  **空间索引（STR Tree）**：在地图层面通过Sort-Tile-Recursive树结构，实现高效的空间查询。
*   **核心细节**：123D不强制要求对所有数据重采样，而是保留原始事件流，通过“同步表”预计算对齐索引，在读取时动态完成跨模态匹配。
*   **算法意义**：通过抽象出“事件流”而非传统的“固定帧”概念，解决了传感器异步触发的难题。

### 4. 方法对比分析
*   **本质区别**：与`trajdata`或`ScenarioNet`等仅针对特定任务（如预测/规划）的预处理框架不同，123D是面向传感器原始数据、地图及标注的全栈统一，不丢弃多模态传感信息。
*   **创新贡献**：提出了一种基于时间戳对齐的通用多模态数据存储格式，并实现了跨数据集的零散数据标准化（如统一坐标系到ISO 8855），显著降低了数据处理的工程复杂度。
*   **适用场景**：适用于需要联合多个数据集进行大模型预训练、跨数据集泛化研究及强化学习模拟的场景。

### 5. 实验分析（精简版）
*   **关键结果**：在3D目标检测中，联合训练（Mixed-5）在大部分数据集上达到了接近或超过单数据集训练的效果，验证了异构数据混合的有效性。
*   **优势**：极大地降低了数据预处理工程量，为通用自动驾驶大模型开发提供了基础设施。
*   **局限**：在未见过的强异构数据集上的跨域泛化能力依然存在瓶颈，且目前主要涵盖乘用车传感器布局。

### 6. 实用指南
*   **开源情况**：已开源，代码托管于 [kesai-labs/py123d](https://github.com/kesai-labs/py123d)。
*   **实现细节**：数据转换时，坐标系需严格遵循ISO 8855（自车）和OpenCV（相机）约定。
*   **迁移可能**：该框架通过定义统一的“Log”和“Map”接口，不仅能迁移到其他自动驾驶数据集，也可扩展到移动机器人等多平台任务。

### 7. 总结
*   **核心思想**：统一异构传感器数据流，构建通用自动驾驶大模型数据后端。
*   **速记版Pipeline**：
    1. 解析源数据并转为Arrow格式；
    2. 按模态存储为独立事件流；
    3. 预计算同步表以对齐时间戳；
    4. 通过API按需动态提取数据。

**Key Findings:**

- We present 123D, an open-source framework that unifies such multi-modal driving data through a single API.
- Further, we showcase two applications 123D enables: cross-dataset 3D object detection transfer and reinforcement learning for planning, and offer recommendations for future directions.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.08084v1)
- [arXiv](https://arxiv.org/abs/2605.08084v1)

---

<a id='2605.07885v1'></a>
## [AERO-VIS: Asynchronous Event-based Real-time Onboard Visual-Inertial SLAM](https://arxiv.org/abs/2605.07885v1)

**Authors:** Yannick Burkhardt, Sebastián Barbas Laina, Simon Boche, Leonard Freißmuth, Stefan Leutenegger

**Published:** 2026-05-08

**Categories:** cs.RO

**Abstract:**

The robustness of event cameras to high dynamic range and motion blur holds the potential to improve visual odometry systems in challenging environments. Although their high temporal resolution does not require synchronous processing, most event-based odometry methods still run at fixed rates, which simplifies system design but restricts latency and throughput. In this work, we present AERO-VIS, a stereo event-inertial SLAM system with an integrated, data-driven, robust, and performance-optimized keypoint detector. By processing the event stream asynchronously, the system dynamically adapts to downstream runtime demands, ensuring low-latency and real-time performance. When deploying AERO-VIS on a UAV, we achieve unprecedented accuracy in onboard event-based SLAM. These unique characteristics enable us to present the first purely event-based inertial SLAM system that demonstrates closed-loop UAV control and large-scale state estimation while relying solely on onboard compute. A video of the experiments and the source code are available at ethz-mrl.github.io/AERO-VIS.

**Analysis:**

以下是对AERO-VIS论文的方法分析与总结：

### 1. 摘要翻译
事件相机的鲁棒性（高动态范围和无运动模糊）为改善挑战性环境下的视觉里程计系统提供了潜力。尽管其高时间分辨率无需同步处理，但多数基于事件的里程计方法仍以固定频率运行，这简化了系统设计但限制了延迟和吞吐量。本文提出了AERO-VIS，一个具备集成式、数据驱动且性能优化关键点检测器的立体事件惯性SLAM系统。通过异步处理事件流，系统可动态适应计算需求，确保低延迟和实时性能。在无人机（UAV）上部署AERO-VIS时，我们实现了史无前例的机载事件SLAM精度，并演示了首个完全依赖机载计算的闭环无人机控制及大规模状态估计系统。

### 2. 方法动机分析
- **驱动力**：旨在解决无人机（UAV）在高速、强振动或HDR场景下，传统SLAM因计算资源限制或同步框架无法满足实时性与高精度平衡的问题。
- **痛点**：现有方法（如SuperEvent）依赖同步处理和固定速率，计算昂贵且无法适应边缘计算设备的算力波动；其他轻量级方法精度不足或在特定数据集上表现不稳定。
- **研究假设**：通过异步系统架构结合针对事件流优化的轻量化神经网络（SuperLitE），可以在边缘计算设备（如Jetson Orin NX）上实现实时闭环控制。

### 3. 方法设计详解
- **核心Pipeline**：
  1. **异步预处理**：事件流经过预处理，实时计算MCTS（多通道时间表面）并存入共享缓冲区。
  2. **异步前端**：前端异步运行，当空闲时从缓冲区“冻结”最新的MCTS数据，通过SuperLitE模型进行关键点检测与匹配。
  3. **后端优化**：在后端构建因子图，联合优化相机位姿与路标点，闭环检测在独立线程并行运行。
- **算法细节**：
  - **MCTS$N_e$ (Constant Event Count)**：论文改进了传统固定时间窗（$\Delta t$）的MCTS计算，转为固定事件计数（$N_e$）。公式：$\Delta t_k = \tau - t_{I-N_{e,k}}$。这使得在运动剧烈时仍能保持时间表面特征的可视相似度。
  - **SuperLitE模型**：通过设计空间搜索，采用四层轻量编码器，并将描述符维度从256压缩至64，显著加速推理速度（降至2.5ms）。
  - **量化与匹配**：使用8位整数（INT8）量化描述符，利用余弦距离代替欧氏距离，进一步降低计算开销。

### 4. 方法对比分析
- **本质区别**：从传统的同步、固定速率SLAM转变为完全的异步、动态计算分配系统。
- **创新贡献**：
  1. 引入基于固定事件计数的MCTS，提升了系统对运动幅度的鲁棒性。
  2. SuperLitE网络实现了高精度关键点提取，且推理速度比Baseline快约90%。
  3. 实现了首个在UAV机载环境下基于纯事件相机的闭环实时控制。
- **适用场景**：高动态、HDR、计算资源受限的边缘嵌入式平台（如无人机、移动机器人）。

### 5. 实验分析
- **关键结论**：在Event Camera Dataset (ECD)上，SuperLitE在保持极低推理延迟的同时，AUC性能优于基线；在真实环境UAV测试中，显著减少了快速飞行下的位姿漂移。
- **主要优势**：极高的实时响应速度、极强的HDR与运动模糊鲁棒性，且能耗比优异。
- **主要局限**：对无人机振动较敏感，精度在常规飞行条件下略逊于帧基相机SLAM。

### 6. 实用指南
- **开源情况**：代码及实验视频已开源至：`ethz-mrl.github.io/AERO-VIS`。
- **实现建议**：
  - 若需迁移，重点关注MCTS缓冲区的线程同步管理（如论文中的Freeze/Unfreeze机制）。
  - 在嵌入式平台上使用TensorRT进行模型加速是实现实时性的关键。
- **迁移性**：该方法中“固定事件计数”的MCTS输入设计可直接移植到其他事件驱动的感知任务（如目标检测、光流估计）。

### 7. 总结
- **核心思想**：异步驱动的轻量级事件SLAM架构。
- **速记版Pipeline**：
  1. 同步IMU与事件流。
  2. 动态窗口计算特征映射（MCTS）。
  3. 异步轻量推理提取特征点。
  4. 因子图更新位姿与闭环控制。

**Key Findings:**

- In this work, we present AERO-VIS, a stereo event-inertial SLAM system with an integrated, data-driven, robust, and performance-optimized keypoint detector.
- When deploying AERO-VIS on a UAV, we achieve unprecedented accuracy in onboard event-based SLAM.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.07885v1)
- [arXiv](https://arxiv.org/abs/2605.07885v1)

---

<a id='2605.07492v1'></a>
## [How Far Is Document Parsing from Solved? PureDocBench: A Source-TraceableBenchmark across Clean, Degraded, and Real-World Settings](https://arxiv.org/abs/2605.07492v1)

**Authors:** Zhiheng Li, Zongyang Ma, Jiaxian Chen, Jianing Zhang, Zhaolong Su, Yutong Zhang, Zhiyin Yu, Ruiqi Liu, Xiaolei Lv, Bo Li, Jun Gao, Ziqi Zhang, Chunfeng Yuan, Bing Li, Weiming Hu

**Published:** 2026-05-08

**Categories:** cs.CV

**Abstract:**

The past year has seen over 20 open-source document parsing models, yet thefield still benchmarks almost exclusively on OmniDocBench, a 1,355-pagemanually annotated dataset whose top scores have saturated above 90%. Athree-stage audit pipeline we run on OmniDocBench screens its 21,353evaluator-scored blocks and confirms 2,580 errors (12.08%); combined with overa year of public availability, both annotation quality and contamination riskcall its rankings into question. To address these issues, we presentPureDocBench, a programmatically generated, source-traceable benchmark thatrenders document images from HTML/CSS and produces verifiable annotations fromthe same source, covering 10 domains, 66 subcategories, and 1,475 pages, eachin three versions: clean, digitally degraded, and real-degraded (4,425 imagestotal). Evaluating 40 models spanning pipeline specialists, end-to-endspecialists, and general-purpose VLMs, we find: (i) document parsing is farfrom solved: the best model scores only ~74 out of 100, with a 44.6-point gapbetween the strongest and weakest models; (ii) specialist parsers with <=4Bparameters rival or surpass general VLMs that are 5-100x larger, yet formularecognition remains a shared bottleneck where no model exceeds 67% whenaveraging the formula metric across all three tracks; (iii) general VLMs loseonly 0.99/8.52 Overall points under digital/real degradation versus 4.90/14.21for pipeline specialists, producing ranking reversals that make clean-onlyevaluation misleading for deployment. All data, code, and artifacts arepublicly released.

**Analysis:**

### 1. 摘要翻译
过去一年中，尽管开源文档解析模型激增，但该领域几乎完全依赖 OmniDocBench 进行评估，而该数据集存在显著的饱和问题（顶尖分数超过 90%）。我们通过审计流程确认 OmniDocBench 存在 12.08% 的标注错误，且由于长期公开，面临严重的污染风险。为此，我们提出了 PureDocBench，这是一个可溯源的基准，通过 HTML/CSS 渲染文档图像并生成可验证的标注。它涵盖 10 个领域、66 个子类别和 1,475 个页面，并提供三种版本：原始、数字退化和真实退化。对 40 个模型的评估显示：(i) 文档解析远未解决，最优模型得分仅约 74；(ii) 参数量 $\le$ 4B 的专用解析器可媲美甚至超越远超其规模的通用 VLM；(iii) 通用 VLM 在退化场景下表现更鲁棒，而专用解析器在 clean-only 评估下的优异表现具有误导性。

### 2. 方法动机分析
- **驱动力**：解决现有文档解析基准（特别是 OmniDocBench）标注质量低、饱和度高以及严重数据污染的问题。
- **痛点**：现有基准严重依赖人工标注，无法验证正确性，且由于模型训练数据泄露，排行榜已失效。
- **研究假设**：通过程序化生成的“可溯源”基准，可以从源头控制标注正确性，并通过动态退化模拟提供更真实的评估环境，从而暴露现有模型在复杂、真实场景下的真实短板。

### 3. 方法设计详解
- **pipeline 流程**：
    1. **元提示设计 (Meta-prompting)**：定义文档类型、布局、语言、内容组成（表、公式密度等）。
    2. **LLM 自动化生成**：LLM 生成自包含的 HTML/CSS 源代码。
    3. **人机协同审计**：快速人工筛选无效布局，利用渲染引擎生成 PNG 图像。
    4. **标注提取与校验**：提取与渲染内容一一对应的 Ground Truth (GT)，并进行多轮自动化与人工交叉检查。
- **关键设计**：采用了“1:1:1  triple-version”设计，即每一页均有 clean、digitally degraded 和 real-degraded 三个版本，确保评估公平且具备鲁棒性。

### 4. 方法对比分析
- **本质区别**：从“人工标注的固态数据集”转变为“程序化生成的动态基准”，实现了全流程可溯源与污染抗性。
- **创新点**：
    1. **源头可溯源性**：HTML/CSS 作为真值，确保了标注与图像的绝对一致性。
    2. **退化链机制**：不仅模拟数字退化（如压缩、噪声），还包含了物理采集（拍照、屏幕截屏）的真实退化。
    3. **评估维度**：构建了包括文本、公式、表格、阅读顺序的“全谱”评估。

### 5. 实验分析
- **关键结果**：在 OmniDocBench 上饱和的模型在 PureDocBench 上分差拉大至 44.6 点，证明了该基准的区分度。
- **主要优势**：极强的鲁棒性评估能力；揭示了模型在公式识别上的共同瓶颈。
- **局限**：目前的生成主要是中英文，且尚未完全覆盖极端的长尾场景（如极度破损的古籍）。

### 6. 实用指南
- **开源情况**：已发布全部数据、生成 pipeline、评估代码及 corrected OmniDocBench 标注。
- **实现建议**：注意公式识别是当前最难的挑战，建议将公式渲染结果作为度量标准，而非仅仅匹配 LaTeX 字符串。
- **迁移性**：该基准的“程序化生成+自动校验”模式极易迁移到其他需要高质量布局数据的多模态任务中。

### 7. 总结
- **核心思想**：通过源头生成实现文档标注的可溯源与鲁棒评估。
- **速记版pipeline**：
    1. 使用元提示生成结构化网页代码。
    2. 自动渲染生成三种退化版本的图像。
    3. 提取 HTML 内容作为可验证的真值标注。
    4. 对所有模型进行跨轨道（Clean/Digital/Real）评估。

**Key Findings:**

- To address these issues, we presentPureDocBench, a programmatically generated, source-traceable benchmark thatrenders document images from HTML/CSS and produces verifiable annotations fromthe same source, covering 10 domains, 66 subcategories, and 1,475 pages, eachin three versions: clean, digitally degraded, and real-degraded (4,425 imagestotal).

**Links:**

- [PDF](https://arxiv.org/pdf/2605.07492v1)
- [arXiv](https://arxiv.org/abs/2605.07492v1)

---

<a id='2605.08078v1'></a>
## [Normalizing Trajectory Models](https://arxiv.org/abs/2605.08078v1)

**Authors:** Jiatao Gu, Tianrong Chen, Ying Shen, David Berthelot, Shuangfei Zhai, Josh Susskind

**Published:** 2026-05-08

**Categories:** cs.CV, cs.LG

**Abstract:**

Diffusion-based models decompose sampling into many small Gaussian denoising steps -- an assumption that breaks down when generation is compressed to a few coarse transitions. Existing few-step methods address this through distillation, consistency training, or adversarial objectives, but sacrifice the likelihood framework in the process. We introduce Normalizing Trajectory Models (NTM), which models each reverse step as an expressive conditional normalizing flow with exact likelihood training. Architecturally, NTM combines shallow invertible blocks within each step with a deep parallel predictor across the trajectory, forming an end-to-end network trainable from scratch or initializable from pretrained flow-matching models. Its exact trajectory likelihood further enables self-distillation: a lightweight denoiser trained on the model's own score produces high-quality samples in four steps. On text-to-image benchmarks, NTM matches or outperforms strong image generation baselines in just four sampling steps while uniquely retaining exact likelihood over the generative trajectory.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于 **Normalizing Trajectory Models (NTM)** 的论文分析如下：

### 1. 论文核心贡献总结
该论文提出了一种名为“归一化轨迹模型（NTM）”的新型生成框架，旨在解决扩散模型在极少步数采样时性能下降的问题。NTM 首次通过将反向去噪过程建模为一系列条件归一化流（Normalizing Flows），在保持**精确似然估计（Exact Likelihood）**的同时，实现了高质量的少步生成，填补了当前生成模型在采样速度与数学严谨性之间的空白。

### 2. 关键创新与方法论
*   **架构创新**：NTM 采用了双重结构设计：每个去噪步内集成“浅层可逆块”（Invertible Blocks）以捕获局部复杂性，跨整个采样轨迹使用“深层并行预测器”以捕获全局一致性。
*   **数学框架**：摒弃了传统的扩散模型近似假设，转而利用归一化流的严格数学框架，确保模型在训练阶段可以进行精确的似然优化。
*   **自蒸馏（Self-Distillation）**：利用模型自身的轨迹似然性进行自蒸馏，使得一个轻量级去噪器仅需 4 步即可生成高质量图像，且无需牺牲似然框架。
*   **兼容性**：该方法既支持从零开始训练，也能无缝接入预训练的流匹配（Flow Matching）模型进行微调。

### 3. 对该领域的潜在影响
*   **打破“精度与速度”的权衡**：过去，高质量少步生成（如蒸馏、GAN 辅助训练）通常会失去似然估计能力，导致无法直接评估模型密度或进行分布外检测。NTM 证明了无需这种权衡，这可能重新定义生成模型的设计范式。
*   **提升生成模型的数学可解释性**：能够提供精确似然估计意味着 NTM 在下游任务（如异常检测、数据压缩、密度估计）中将比现有的扩散模型更具竞争力。
*   **流模型（Flow-based Models）的复兴**：将轨迹建模与归一化流结合，可能引发对基于流的生成模型研究的新一轮兴趣。

### 4. 受益的领域或应用
*   **实时生成应用**：在对推理延迟极其敏感的场景（如交互式实时编辑、移动端 AR/VR 渲染）中具有极高价值。
*   **统计推断与数据压缩**：由于具备精确似然，该模型非常适合于需要评估概率密度的任务，例如高效图像压缩（作为熵模型）和异常检测。
*   **高性能生成式 AI**：为需要极高采样效率且对生成质量有严苛要求的领域（如工业设计中的即时预览）提供底层架构支持。

### 5. 可推断的潜在局限性
*   **训练复杂性**：相比于标准的扩散模型，引入归一化流架构（特别是可逆块）通常会增加内存占用，且对训练稳定性和架构设计要求更高。
*   **泛化能力**：由于 NTM 依赖于轨迹的精确建模，若采样轨迹过于稀疏（例如少于 4 步），其模型能力的上限是否会受到归一化流表达能力的限制，仍有待验证。
*   **算力成本**：尽管推理快，但“深层并行预测器”与“每步可逆块”的组合在训练初期可能需要比普通 U-Net 更高的计算资源。

---
**专家点评：**
这篇论文非常引人注目，因为它巧妙地解决了生成模型中的“逻辑困境”——即如何在极少的去噪步骤内，既获得极高的采样质量，又保持统计学上的严格性（精确似然）。如果 NTM 能在更大规模的数据集（如 ImageNet 或大型多模态预训练）中证明其扩展性，它很可能成为下一代高效生成模型的强有力竞争者。

**Key Findings:**

- We introduce Normalizing Trajectory Models (NTM), which models each reverse step as an expressive conditional normalizing flow with exact likelihood training.
- On text-to-image benchmarks, NTM matches or outperforms strong image generation baselines in just four sampling steps while uniquely retaining exact likelihood over the generative trajectory.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.08078v1)
- [arXiv](https://arxiv.org/abs/2605.08078v1)

---

<a id='2605.08063v1'></a>
## [Flow-OPD: On-Policy Distillation for Flow Matching Models](https://arxiv.org/abs/2605.08063v1)

**Authors:** Zhen Fang, Wenxuan Huang, Yu Zeng, Yiming Zhao, Shuang Chen, Kaituo Feng, Yunlong Lin, Lin Chen, Zehui Chen, Shaosheng Cao, Feng Zhao

**Published:** 2026-05-08

**Categories:** cs.CV, cs.AI

**Abstract:**

Existing Flow Matching (FM) text-to-image models suffer from two critical bottlenecks under multi-task alignment: the reward sparsity induced by scalar-valued rewards, and the gradient interference arising from jointly optimizing heterogeneous objectives, which together give rise to a 'seesaw effect' of competing metrics and pervasive reward hacking. Inspired by the success of On-Policy Distillation (OPD) in the large language model community, we propose Flow-OPD, the first unified post-training framework that integrates on-policy distillation into Flow Matching models. Flow-OPD adopts a two-stage alignment strategy: it first cultivates domain-specialized teacher models via single-reward GRPO fine-tuning, allowing each expert to reach its performance ceiling in isolation; it then establishes a robust initial policy through a Flow-based Cold-Start scheme and seamlessly consolidates heterogeneous expertise into a single student via a three-step orchestration of on-policy sampling, task-routing labeling, and dense trajectory-level supervision. We further introduce Manifold Anchor Regularization (MAR), which leverages a task-agnostic teacher to provide full-data supervision that anchors generation to a high-quality manifold, effectively mitigating the aesthetic degradation commonly observed in purely RL-driven alignment. Built upon Stable Diffusion 3.5 Medium, Flow-OPD raises the GenEval score from 63 to 92 and the OCR accuracy from 59 to 94, yielding an overall improvement of roughly 10 points over vanilla GRPO, while preserving image fidelity and human-preference alignment and exhibiting an emergent 'teacher-surpassing' effect. These results establish Flow-OPD as a scalable alignment paradigm for building generalist text-to-image models.

**Analysis:**

### 1. 摘要翻译
现有的流匹配（FM）文生图模型在多任务对齐中存在两个关键瓶颈：标量奖励导致的奖励稀疏性，以及联合优化异构目标带来的梯度干扰，这两者共同引发了竞争指标的“跷跷板效应”和奖励劫持。受大语言模型社区在线策略蒸馏（OPD）成功的启发，我们提出了 **Flow-OPD**，这是第一个将在线策略蒸馏集成到流匹配模型的统一后训练框架。Flow-OPD 采用两阶段对齐策略：首先通过单奖励 GRPO 微调培养领域专业教师模型，使各专家在独立状态下达到性能巅峰；随后通过基于流的冷启动方案建立稳健的初始策略，并利用在线策略采样、任务路由标注和密集轨迹级监督的三步编排，将异构专业知识无缝整合进单个学生模型中。我们还引入了流形锚点正则化（MAR），利用任务不可知的教师提供全数据监督，将生成锚定在高质量流形上，有效缓解了纯 RL 驱动对齐中常见的审美退化。基于 Stable Diffusion 3.5 Medium，Flow-OPD 将 GenEval 分数从 63 提高到 92，OCR 准确率从 59 提高到 94，较原始 GRPO 整体提升约 10 个点，同时保持了图像保真度和人类偏好对齐，并展现了涌现的“教师超越”效应。

---

### 2. 方法动机分析
*   **驱动力**：解决多任务对齐中奖励稀疏和梯度冲突导致的模型性能退化问题，构建一个既能兼顾多样化任务（如文字渲染、构图）又能保持高审美水平的通用模型。
*   **现有方法痛点**：
    *   **奖励稀疏/标量瓶颈**：标量奖励（如简单的分数）无法提供足够细粒度的反馈，导致多任务训练陷入“零和博弈”。
    *   **梯度干扰**：不同任务的优化方向冲突，导致模型在优化某项能力时，“灾难性遗忘”了其他已学知识。
*   **研究假设**：通过将多目标训练转化为密集、解耦的教师轨迹蒸馏，并引入流形锚定，能够消除任务间的梯度冲突，实现专家知识的有效融合。

---

### 3. 方法设计详解
*   **流程总结**：
    1.  **专家训练**：利用 GRPO 训练多个专注于特定领域（如 OCR、审美）的教师模型。
    2.  **冷启动**：通过 SFT 或模型合并将学生模型置于高质量的初始参数空间。
    3.  **在线蒸馏（OPD）**：
        *   **采样**：将确定性 ODE 转化为 SDE，进行随机采样以保持探索性。
        *   **任务路由**：基于输入提示词 $c$ 选择对应专家，提供密集轨迹级监督（即教师的向量场）。
        *   **正则化（MAR）**：引入固定 aesthetic 教师作为锚点，防止学生模型偏离高质量生成流形。
*   **算法解释**：将传统的离散策略梯度转换为连续时间内的**向量场拟合**。通过最小化学生与教师在动作空间（速度场）的 KL 散度，将原本稀疏的标量奖励变成了密集的监督信号，从而引导优化。

---

### 4. 方法对比分析
*   **本质区别**：从传统的“标量奖励引导的强化学习”转向“基于密集向量场拟合的在线策略蒸馏”，彻底解决了标量奖励带来的梯度竞争。
*   **创新贡献**：提出流形锚点正则化（MAR），解决了多任务对齐中常见的审美衰减问题；引入任务路由机制，实现专家能力的高效解耦融合。
*   **适用场景**：多任务文生图生成、复杂文字合成及对审美有严格要求的生成任务。

---

### 5. 实验分析
*   **关键结果**：在 GenEval 和 OCR 任务上取得了显著改进（GenEval 63→92, OCR 59→94），且在多个基准测试上实现整体 10% 的提升。
*   **主要优势**：展现了“教师超越”效应（学生性能优于单任务教师），具备出色的多任务平衡能力。
*   **主要局限**：模型性能依然受限于教师模型的上限；对教师-学生架构的同质性有一定要求。

---

### 6. 实用指南
*   **开源情况**：已开源，项目地址：[costaliya.github.io/Flow-OPD/](https://costaliya.github.io/Flow-OPD/)。
*   **实现细节**：训练需注意 MAR 正则化系数 $\beta$ 的调节（论文设为 0.02），且在多任务训练时需确保任务路由函数的准确性。
*   **迁移可能**：该框架可迁移至任何基于流匹配（Flow Matching）的生成模型（如 Flux 等），适用于需要多维对齐的生成领域。

---

### 7. 总结
*   **核心思想**：通过密集多专家向量场蒸馏与流形锚定，解决多任务生成的梯度冲突。
*   **速记版pipeline**：
    1. 分别训练各领域的单专家教师模型。
    2. 通过合并或监督微调初始化学生模型。
    3. 根据任务类型动态路由至特定教师获取密集指导。
    4. 加入审美锚点防止模型退化。

**Key Findings:**

- Inspired by the success of On-Policy Distillation (OPD) in the large language model community, we propose Flow-OPD, the first unified post-training framework that integrates on-policy distillation into Flow Matching models.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.08063v1)
- [arXiv](https://arxiv.org/abs/2605.08063v1)

---

<a id='2605.08029v1'></a>
## [STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation](https://arxiv.org/abs/2605.08029v1)

**Authors:** Ying Shen, Tianrong Chen, Yuan Gao, Yizhe Zhang, Yuyang Wang, Miguel Ángel Bautista, Shuangfei Zhai, Joshua M. Susskind, Jiatao Gu

**Published:** 2026-05-08

**Categories:** cs.CV, cs.LG

**Abstract:**

Deep generative models have advanced rapidly across text and vision, motivating unified multimodal systems that can understand, reason over, and generate interleaved text-image sequences. Most existing approaches combine autoregressive language modeling with diffusion-based image generators, inheriting a structural mismatch between causal text generation and iterative visual denoising. We observe that autoregressive normalizing flows are autoregressive Transformers--sharing the same causal mask, KV-cache mechanism, and left-to-right structure as LLMs--making them the most natural paradigm for true unified multimodal generation. We present STARFlow2, built on the Pretzel architecture that vertically interleaves a pretrained VLM stream with a TarFlow stream via residual skip connections, both operating under the same causal mask. Combined with a deep-shallow flow design and a unified FAE latent space, STARFlow2 enables cache-friendly interleaved generation where both text and visual outputs directly enter the KV-cache without re-encoding. Experiments demonstrate strong performance across image generation and multimodal understanding benchmarks, validating autoregressive flows as a viable foundation for unified multimodal modeling.

**Analysis:**

以下是对论文 **STARFlow2: Bridging Language Models and Normalizing Flows for Unified Multimodal Generation** 的深度分析：

### 1. 摘要翻译
统一的多模态模型在处理交错的文本-图像序列时，结构上往往支离破碎。现有方法要么通过离散标记化牺牲视觉保真度，要么通过结合因果文本生成与迭代扩散去噪引入结构不对称，亦或在进行生成训练时损害预训练的理解能力。我们观察到，自回归归一化流（Autoregressive Normalizing Flows）本质上就是自回归Transformer——它们共享与大语言模型（LLM）相同的因果掩码、KV-cache机制和从左至右的结构，这使其成为实现连续、单次通行、纯因果统一多模态生成的自然范式。我们提出了 **STARFlow2**，其核心是 **Pretzel架构**：通过残差跳跃连接将冻结的预训练VLM流与TARFlow流垂直交错，两者在同一因果掩码下运行。该设计在保持预训练多模态理解能力的同时，实现了高保真连续图像生成，并通过单一因果机制达成结构统一。结合深浅流设计和统一的FAE潜在空间，STARFlow2支持缓存友好的交错生成。

---

### 2. 方法动机分析
*   **驱动力**：解决现有“统一”模型在生成机制上的“伪统一”问题，即文本与图像生成在结构上的不匹配。
*   **现有痛点**：
    *   离散标记化（如VQ-VAE/VAE）导致严重的量化信息丢失，限制视觉质量。
    *   “语言模型+扩散模型”的混合范式需要不同的解码逻辑，导致生成图像无法像文本一样直接进入KV-cache，造成冗余的重编码开销。
    *   混合专家（MoT）架构虽然在参数层面共享，但属于横向切分，在“冻结VLM”与“微调VLM”之间存在不可调和的矛盾。
*   **研究假设**：基于因果Transformer的归一化流与LLM在结构上是同构的，无需适配即可天然地统一于单一模型中。

---

### 3. 方法设计详解
*   **Pretzel 架构**：这是核心创新，通过垂直交错将预训练VLM与TARFlow流融合。
    *   **垂直跳跃连接**：不同于横向路由，Pretzel通过零初始化的残差连接在每个位置实现信息互通。VLM注入语义，TARFlow进行空间纠偏。
*   **深浅流设计**：
    *   **浅流 (Shallow TARFlows)**：负责处理局部像素依赖，将原始 latents 转化为适合自回归预测的表示。
    *   **深流 (Deep TARFlow)**：模型主体，在全多模态语境下进行下一高斯预测 (Next Gaussian Prediction)。
*   **Pipeline**：
    1.  **阶段1（生成训练）**：冻结VLM，训练TARFlow进行文本到图像生成。
    2.  **阶段2（理解对齐）**：冻结深浅流，训练Visual Adapter，使视觉特征与VLM语义对齐。
    3.  **阶段3（端到端联合优化）**：激活垂直连接，在混合任务上进行微调。

---

### 4. 方法对比分析
*   **本质区别**：将图像生成建模为连续空间的“下一token预测”（预测高斯分布参数），而不是扩散过程的“迭代采样”，从而与LLM的解码逻辑完全一致。
*   **创新贡献**：提出Pretzel架构，首次实现了在同一个自回归Transformer中进行语义理解与连续高保真图像生成，无需重编码且支持Cache-friendly生成。
*   **适用场景**：实时多模态交互、复杂的图文交错任务、需要高保持一致性的长序列生成。

---

### 5. 实验分析
*   **关键结果**：在GenEval上达到0.82，在DPG-Bench上达到84.14，证明了在保留预训练理解能力的前提下，达到了与扩散模型相当的视觉质量。
*   **优势**：真正的单次通行（Single-pass）推理，极低延迟；保持了预训练LLM的强大推理与感知能力。
*   **局限**：目前的图像分辨率受限于FAE encoder，且对超大规模数据的端到端训练依赖较多阶段的复杂性。

---

### 6. 实用指南
*   **开源情况**：代码已开源（github.com/apple/ml-starflow）。
*   **实现细节**：关键在于**零初始化投影矩阵 (Wvlm, WD)**，这确保了训练初期模型能复用VLM的预训练能力，避免灾难性遗忘。
*   **迁移可能**：可直接替换骨干网（如更换更强的VLM）或扩展至视频生成（如引用中的STARFlow-v）。

---

### 7. 总结
*   **核心思想**：利用归一化流的自回归特性，将图像生成转化为与文本一致的序列建模。
*   **速记版Pipeline**：
    1. 使用FAE将图像转为连续Latent。
    2. 浅流进行局部视觉特征处理。
    3. 深流与VLM通过垂直跳跃连接融合，在同一因果掩码下统一预测。
    4. 对视觉位置输出高斯参数，对文本位置输出词汇分布。

**Key Findings:**

- We present STARFlow2, built on the Pretzel architecture that vertically interleaves a pretrained VLM stream with a TarFlow stream via residual skip connections, both operating under the same causal mask.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.08029v1)
- [arXiv](https://arxiv.org/abs/2605.08029v1)

---

<a id='2605.08000v1'></a>
## [Rethinking Dense Optical Flow without Test-Time Scaling](https://arxiv.org/abs/2605.08000v1)

**Authors:** Praroop Chanda, Suryansh Kumar

**Published:** 2026-05-08

**Categories:** cs.CV

**Abstract:**

Recent progress in dense optical flow has been driven by increasingly complex architectures and multi-step refinement for test-time scaling. While these approaches achieve strong benchmark performance, they also require substantial computation during inference. This raises a fundamental question: Is scaling test-time computation the only way to improve dense optical flow accuracy? We argue that it is not. Instead, powerful visual semantic and geometric priors encoded in modern foundation models can reduce, if not overcome, the need for computationally expensive iterative refinement at test-time. In this paper, we present a framework that estimates dense optical flow in a single forward pass, leveraging pretrained foundation representations, while avoiding iterative refinement and additional inference-time computation, thus offering an alternative to test-time scaling. Our method extracts visual semantic features from a frozen DINO-v2 backbone and combines them with geometric cues from a monocular depth foundation model. We fuse these complementary priors into a unified representation and apply a global matching formulation to estimate dense correspondences without recurrent updates or test-time optimization. Despite avoiding iterative refinement, our approach achieves strong cross-dataset generalization across challenging benchmarks. On Sintel Final, we obtain 2.81 EPE without refinement, significantly improving over state-of-the-art (SOTA) SEA-RAFT under comparable training conditions and outperforming RAFT, GMFlow (without refinement), and recent FlowSeek in the same setting. These results suggest that strong foundation priors can substitute for test-time scaling, offering a computationally efficient alternative to refinement-heavy pipelines.

**Analysis:**

### 1. 摘要翻译
当前的稠密光流估计研究主要依赖日益复杂的架构和多步迭代细化（Test-Time Scaling）来获取基准性能，但这伴随着极高的推理成本。本文提出了一个根本性问题：增加推理计算量是提升准确性的唯一途径吗？作者提出一个新框架，通过单一前向传播即可估计光流，无需迭代细化。该方法结合了冷冻的DINO-v2视觉语义特征和单目深度基础模型的几何先验，通过全局匹配完成对应关系估计。在Sintel Final基准上，该方法以2.81 EPE的成绩显著优于现有的SOTA方法（如SEA-RAFT），证明了强大的基础模型先验可以替代昂贵的测试时计算，为高效光流估计提供了新范式。

---

### 2. 方法动机分析
- **驱动力**：旨在摆脱对“高计算量迭代细化”的依赖，利用通用大模型（Foundation Models）已有的强表征能力，实现更高效的光流推理。
- **现有方法痛点**：当前主流方法（如RAFT、SEA-RAFT、FlowSeek）均高度依赖多步迭代更新或特定数据集上的监督微调，计算开销巨大且泛化能力受限于特定训练范式。
- **研究假设**：现代基础模型已编码了足够的空间一致性和几何边界信息，这些先验可以直接支持单次前向传播的高质量光流估计。

---

### 3. 方法设计详解
- **流程总结**：
    1. **特征提取**：输入两张RGB图，分别通过冷冻的**DINOv2-S**提取语义特征 $F^D$，以及冷冻的**Depth Anything V2**提取几何特征 $F^Z$。
    2. **特征对齐与融合**：利用投影网络 $\Psi_{\text{proj}}$ 对齐维度，通过连接层和轻量级融合网络 $\Psi_{\text{fusion}}$ 生成结合语义与几何的统一特征 $\hat{F}$。
    3. **全局匹配**：计算所有位置的特征相关性矩阵，经Softmax转化为概率分布，通过期望计算得出初始光流场 $\hat{V}_{\text{flow}}$。
    4. **流传播**：引入基于自相似性的注意力矩阵 $A$，将高可信区域的流传播至边缘或遮挡区域，生成最终光流。
- **模型结构**：仅优化轻量级的投影、融合模块和匹配逻辑，主干网络完全冻结。
- **算法解释**：核心在于“特征先验取代迭代”。通过语义（DINOv2）捕捉像素对应，通过几何（Depth Anything）约束运动边界，从而使得一次匹配就能达到以往多次迭代的效果。

---

### 4. 方法对比分析
- **本质区别**：从“依赖迭代优化”转向“依赖预训练表示先验”。
- **创新贡献**：首次提出在不需要任何测试时迭代的情况下，通过融合语义+几何基础模型，实现跨数据集的强大光流估计。
- **适用场景**：对实时性要求高、计算资源受限，但对精度仍有追求的视觉感知系统。

---

### 5. 实验分析
- **关键结果**：在Sintel Final上达到2.81 EPE，在不进行任何针对性精调的情况下，性能超越了需要多次迭代的SEA-RAFT等方法。
- **主要优势**：推理速度快（单次前向），对预训练表示的泛化性依赖强，部署简便。
- **主要局限**：在极端遮挡和超细微结构场景下，仍可能因缺乏局部修正而略逊于多次迭代的顶级模型。

---

### 6. 实用指南
- **迁移建议**：该思路具有极强的普适性。只需替换目标任务（如深度估计、特征匹配），并利用DINOv2等 frozen backbone 作为特征提取器，通过简单的融合层引入辅助任务先验即可迁移。
- **关键细节**：保持主干模型完全冻结是该架构成功的关键，否则容易过拟合。

---

### 7. 总结
- **核心思想**：利用冻结的大模型语义与几何先验，替代传统的迭代计算。
- **速记版pipeline**：
    1. 冻结模型提特征（语义+深度）。
    2. 融合两个模态特征。
    3. 全局匹配计算初始流。
    4. 注意力传播修正细节。

**Key Findings:**

- In this paper, we present a framework that estimates dense optical flow in a single forward pass, leveraging pretrained foundation representations, while avoiding iterative refinement and additional inference-time computation, thus offering an alternative to test-time scaling.
- Our method extracts visual semantic features from a frozen DINO-v2 backbone and combines them with geometric cues from a monocular depth foundation model.
- Despite avoiding iterative refinement, our approach achieves strong cross-dataset generalization across challenging benchmarks.
- On Sintel Final, we obtain 2.81 EPE without refinement, significantly improving over state-of-the-art (SOTA) SEA-RAFT under comparable training conditions and outperforming RAFT, GMFlow (without refinement), and recent FlowSeek in the same setting.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.08000v1)
- [arXiv](https://arxiv.org/abs/2605.08000v1)

---

<a id='2605.07943v1'></a>
## [TAVIS: A Benchmark for Egocentric Active Vision and Anticipatory Gaze in Imitation Learning](https://arxiv.org/abs/2605.07943v1)

**Authors:** Giacomo Spigler

**Published:** 2026-05-08

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

Active vision -- where a policy controls its own gaze during manipulation -- has emerged as a key capability for imitation learning, with multiple independent systems demonstrating its benefits in the past year. Yet there is no shared benchmark to compare approaches or quantify what active vision contributes, on which task types, and under what conditions. We introduce TAVIS, evaluation infrastructure for active-vision imitation learning, with two complementary task suites -- TAVIS-Head (5 tasks, global search via pan/tilt necks) and TAVIS-Hands (3 tasks, local occlusion via wrist cameras) -- on two humanoid torso embodiments (GR1T2, Reachy2), built on IsaacLab. TAVIS provides three evaluation primitives: a paired headcam-vs-fixedcam protocol on identical demonstrations; GALT (Gaze-Action Lead Time), a novel metric grounded in cognitive science and HRI that quantifies anticipatory gaze in learned policies; and procedural ID/OOD splits. Baseline experiments with Diffusion Policy and $π_0$ reveal that (i) active-vision generally helps, but benefits are task-conditional rather than uniform; (ii) multi-task policies degrade sharply under controlled distribution shifts on both suites; and (iii) imitation alone yields anticipatory gaze, with median lead times comparable to the human teleoperator reference. Code, evaluation scripts, demonstrations (LeRobot v3.0; ~2200 episodes) and trained baselines are released at https://github.com/spiglerg/tavis and https://huggingface.co/tavis-benchmark.

**Analysis:**

### 1. 摘要翻译
主动视觉——即策略在操纵过程中控制自身视线——已成为模仿学习的一项关键能力，过去一年中多个独立系统展示了其优势。然而，目前尚无共享基准来比较不同方法，或量化主动视觉在何种任务类型及条件下具有贡献。我们引入了 TAVIS，这是一个用于主动视觉模仿学习的评估基础设施，包含两个互补的任务套件——TAVIS-Head（5个任务，通过云台进行全局搜索）和 TAVIS-Hands（3个任务，通过腕部摄像头处理局部遮挡），并在两个基于 IsaacLab 的人形躯干实施例（GR1T2, Reachy2）上运行。TAVIS 提供了三个评估基元：在相同演示上的成对“头戴式摄像头 vs. 固定摄像头”协议；GALT（注视-动作前导时间），一个基于认知科学和人机交互（HRI）的新型指标，用于量化学习策略中的预期性注视；以及程序化的 ID（分布内）/OOD（分布外）划分。使用扩散策略（Diffusion Policy）和 $\pi_0$ 进行的基准实验表明：(i) 主动视觉通常有帮助，但其收益是任务依赖的，而非均匀的；(ii) 多任务策略在两个套件的受控分布偏移下性能急剧下降；(iii) 仅通过模仿即可获得预期性注视，且中位前导时间与人类远程操作参考相当。

---

### 2. 方法动机分析
- **驱动力**：作者旨在解决主动视觉模仿学习领域缺乏统一评估基准、难以定量比较不同系统（如不同构型的头部、不同的主动视觉实现方式）的问题。
- **痛点**：现有 benchmark（如 LIBERO、RLBench）均假设使用固定的第三人称摄像头，无法将主动视觉作为受控变量进行评估，且缺乏测量“注视与动作的时间耦合”这一关键指标的手段。
- **研究假设**：通过引入标准化任务套件和注视行为度量（GALT），可以揭示主动视觉在不同任务场景中的真实贡献，并促进对策略可解释性（legibility）的研究。

---

### 3. 方法设计详解
- **核心评估基元（GALT）**：这是本文的核心创新，用以衡量“注视是否先于动作”。
    - **逻辑**：定义 $GALT = t_{hand} - t_{head}$。
    - **实现细节**：通过 proprioception（本体感受）在轨迹中检测两个关键点：$t_{hand}$（ gripper 闭合时间点）和 $t_{head}$（颈部在任务相关注视点的稳定时刻）。
    - **算法逻辑**：算法在预定义的搜索窗口内查找满足速度阈值（$v_n < 0.1 \text{ rad/s}$）的固定点，并通过回溯修正确保捕捉到真正的注视 onset。若 $GALT > 0$，则表示注视领先于动作，具备认知科学定义的预期性。
- **任务套件**：
    - **TAVIS-Head**：利用云台，解决全局搜索、遮挡解除、条件判断（如根据颜色卡片决策）。
    - **TAVIS-Hands**：利用腕部视觉，解决局部遮挡（如窥视盒内、绕过屏幕抓取）。

---

### 4. 方法对比分析
- **本质区别**：与传统 benchmark 侧重于任务成功率不同，TAVIS 引入了**动作可解释性（Legibility）的度量**，将“感知策略”与“操纵策略”整合在一个闭环评估中。
- **创新贡献**：成功将“注视-动作的时间耦合”形式化为可量化的指标（GALT），并提供了一套涵盖跨本体实施例（GR1T2, Reachy2）的统一评估基准。

---

### 5. 实验分析
- **结论1**：主动视觉的收益是**任务条件性**的。在全局搜索任务（conditional-pick）中优势明显（+28pp to +45pp），但在已知场景中可能引入冗余变量。
- **结论2**：多任务训练在受控的 OOD 偏移（空间位移、初始位姿扰动）下性能下降显著。
- **结论3**：模仿学习策略能够自动习得类人预期性注视，GALT 中位数与人类远程操控者相当（约 2-3 秒）。

---

### 6. 实用指南
- **开源/获取**：代码与数据集已发布（`https://github.com/spiglerg/tavis`，Hugging Face）。
- **关键超参数**：GALT 检测算法依赖于速度阈值 $v_h=0.05 \text{ m/s}$, $v_n=0.1 \text{ rad/s}$。若在其他机器人上复现，需根据具体的电机噪声水平和执行器惯性调整这些门限。
- **迁移建议**：需构建满足规范动作空间（19-D 动作指令）的 wrapper；对于新机器人，只需提供 URDF 模型、关节索引和 hip-frame offset。

---

### 7. 总结
- **核心思想**：通过标准化主动视觉任务与时间耦合指标，量化机器人的注视智能。
- **速记版pipeline**：
    1. **统一化**：将不同机器人的指令封装为统一的 19 维动作空间。
    2. **成对评估**：在相同演示轨迹上分别回放头戴式和固定式视觉采集流。
    3. **指标计算**：通过本体感知识别 gripper 与 neck 的时间差计算 GALT。
    4. **鲁棒测试**：在 ID 和空间偏移的 OOD 分布下评估模型稳健性。

**Key Findings:**

- We introduce TAVIS, evaluation infrastructure for active-vision imitation learning, with two complementary task suites -- TAVIS-Head (5 tasks, global search via pan/tilt necks) and TAVIS-Hands (3 tasks, local occlusion via wrist cameras) -- on two humanoid torso embodiments (GR1T2, Reachy2), built on IsaacLab.
- TAVIS provides three evaluation primitives: a paired headcam-vs-fixedcam protocol on identical demonstrations; GALT (Gaze-Action Lead Time), a novel metric grounded in cognitive science and HRI that quantifies anticipatory gaze in learned policies; and procedural ID/OOD splits.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.07943v1)
- [arXiv](https://arxiv.org/abs/2605.07943v1)

---

<a id='2605.07940v1'></a>
## [Delta-Adapter: Scalable Exemplar-Based Image Editing with Single-Pair Supervision](https://arxiv.org/abs/2605.07940v1)

**Authors:** Jiacheng Chen, Songze Li, Han Fu, Baoquan Zhao, Wei Liu, Yanyan Liang, Li Qing, Xudong Mao

**Published:** 2026-05-08

**Categories:** cs.CV

**Abstract:**

Exemplar-based image editing applies a transformation defined by a source-target image pair to a new query image. Existing methods rely on a pair-of-pairs supervision paradigm, requiring two image pairs sharing the same edit semantics to learn the target transformation. This constraint makes training data difficult to curate at scale and limits generalization across diverse edit types. We propose Delta-Adapter, a method that learns transferable editing semantics under single-pair supervision, requiring no textual guidance. Rather than directly exposing the exemplar pair to the model, we leverage a pre-trained vision encoder to extract a semantic delta that encodes the visual transformation between the two images. This semantic delta is injected into a pre-trained image editing model via a Perceiver-based adapter. Since the target image is never directly visible to the model, it can serve as the prediction target, enabling single-pair supervision without requiring additional exemplar pairs. This formulation allows us to leverage existing large-scale editing datasets for training. To further promote faithful transformation transfer, we introduce a semantic delta consistency loss that aligns the semantic change of the generated output with the ground-truth semantic delta extracted from the exemplar pair. Extensive experiments demonstrate that Delta-Adapter consistently improves both editing accuracy and content consistency over four strong baselines on seen editing tasks, while also generalizing more effectively to unseen editing tasks. Code will be available at https://delta-adapter.github.io.

**Analysis:**

以下是对《Delta-Adapter: Scalable Exemplar-Based Image Editing with Single-Pair Supervision》这篇论文的深度分析：

### 1. 摘要翻译
基于示例的图像编辑通过源-目标图像对定义的转换应用于新的查询图像。现有方法依赖于“对-对”（pair-of-pairs）监督范式，要求两对共享相同编辑语义的图像来学习转换，这使得训练数据的规模化采集变得困难，且限制了对多样化编辑类型的泛化能力。我们提出了 Delta-Adapter，这是一种在单对监督下学习可迁移编辑语义的方法，且无需文本引导。我们利用预训练视觉编码器提取编码了视觉转换的“语义增量”（semantic delta），通过一个基于 Perceiver 的适配器将其注入预训练图像编辑模型。由于目标图像对模型不可见，它可作为预测目标，从而实现单对监督。此外，我们引入了语义增量一致性损失以增强转换迁移的保真度。实验表明，Delta-Adapter 在编辑准确性和内容一致性上均优于现有基线。

### 2. 方法动机分析
- **驱动力**：解决现有基于示例编辑模型对复杂成对数据（pair-of-pairs）的过度依赖，实现更高效、更通用的编辑模型。
- **现有方法痛点**：现有模型通常直接将“源-目标”全图作为条件输入，这导致模型直接“看见”了结果，监督信号因缺乏任务难度而退化，必须引入第二对图像进行对比学习，增加了数据获取成本。
- **核心假设**：将视觉转换提取为一种抽象的“语义增量”（Delta），而非直接输入结果图像，可以解耦编辑意图与图像内容，使模型能够在单对数据上进行自监督训练。

### 3. 方法设计详解
- **流程总结**：
  1. **语义增量提取**：输入源图 $a$ 和目标图 $a'$，使用 SigLIP 提取 patch 特征，计算归一化增量 $\Delta_{a \to a'} = \text{LN}(f_{a'}) - \text{LN}(f_{a})$。
  2. **门控残差细化**：通过门控机制 $(\text{tanh}(g))$ 和可学习投影对 $\Delta$ 进行去噪，过滤无关扰动。
  3. **Perceiver 重新采样**：将 patch 级增量转化为固定长度的 $N$ 个 Token，捕捉空间位置关系。
  4. **注入编辑特征**：通过解耦的 Cross-Attention 分支将 Edit Tokens 注入冻结的 DiT（Diffusion Transformer）主干。
- **模型结构**：采用冻结的 FLUX 作为主干，仅微调映射层（Perceiver）和 Cross-Attention 层。
- **算法精要**：语义增量一致性损失（$\mathcal{L}_{sdc}$）通过对预测出的编辑结果再次提取特征，计算其与 ground-truth 增量的加权余弦相似度，强制模型学习“编辑方向”而非“像素拷贝”。

### 4. 方法对比分析
- **本质区别**：从“直接映射目标图像”转变为“映射编辑语义增量”。
- **创新贡献**：提出了一种无需文本、无需成对示例，仅通过单对图像即可训练并具备测试时适应（TTA）能力的高效框架。
- **适用场景**：适用于各类风格迁移、物体替换、属性调整等需要精确控制的编辑任务。

### 5. 实验分析
- **验证方法**：在 Relation、Pico Banana 等百万级数据集上训练，评估 GPT 自动评测得分及定性对比。
- **关键结果**：在 unseen（未见过）任务上，GPT-A 得分从 2.884 显著提升至 4.008（Ours-1M）。
- **优势与局限**：优势在于泛化性极强且支持连续控制（通过调节 $\lambda_{ca}$）；局限在于对极细粒度文本渲染的捕捉能力受限于视觉编码器（SigLIP）的特征空间。

### 6. 实用指南
- **开源情况**：代码已开源（https://delta-adapter.github.io）。
- **实现细节**：建议在训练时使用较小的 $\lambda_{sdc}$ 以保持基模型的生成能力；测试时针对复杂样例执行约 20 步梯度下降（TTA）即可获得显著提升。
- **迁移可能**：该架构可直接迁移至任何基于 Transformer 的图像生成模型（如 Stable Diffusion 3 等），只需匹配对应的交叉注意力接口。

### 7. 总结
- **核心思想**：通过提取与注入“语义转换向量”替代“全图条件”，实现单对高效编辑。
- **速记版pipeline**：
  1. 算出两张图的特征差异。
  2. 用门控机制滤掉杂质。
  3. 通过重采样提取关键语义块。
  4. 将语义块注入生成模型进行定向修正。

**Key Findings:**

- Exemplar-based image editing applies a transformation defined by a source-target image pair to a new query image.
- We propose Delta-Adapter, a method that learns transferable editing semantics under single-pair supervision, requiring no textual guidance.
- To further promote faithful transformation transfer, we introduce a semantic delta consistency loss that aligns the semantic change of the generated output with the ground-truth semantic delta extracted from the exemplar pair.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.07940v1)
- [arXiv](https://arxiv.org/abs/2605.07940v1)

---

<a id='2605.07931v1'></a>
## [One Token Per Frame: Reconsidering Visual Bandwidth in World Models for VLA Policy](https://arxiv.org/abs/2605.07931v1)

**Authors:** Zuojin Tang, Shengchao Yuan, Xiaoxin Bai, Zhiyuan Jin, De Ma, Gang Pan, Bin Liu

**Published:** 2026-05-08

**Categories:** cs.CV, cs.AI

**Abstract:**

Vision-language-action (VLA) models increasingly rely on auxiliary world modules to plan over long horizons, yet how such modules should be parameterized on top of a pretrained VLA remains an open design question. Existing world-model-augmented VLAs typically pass the per-frame visual stream into the world module at high visual bandwidth and treat its rollout as a side product of action prediction; under a constrained adaptation budget on a frozen backbone, this leaves both the per-frame representation and the latent action coupling under-examined. We introduce OneWM-VLA, which compresses each view into a single semantic token per frame through an Adaptive Attention Pooling, and produces the resulting latent stream and the action trajectory under a single flow-matching objective rather than connecting them through a separate decoder. Empirically, we find that per-frame visual bandwidth can be reduced to a single token without compromising long-horizon performance under our setup. Trained with 14.71M LoRA parameters on a $π_0$ (2B) backbone, OneWM-VLA improves the average success rate from 47.9% to 61.3% on MetaWorld~MT50, reaches 95.6% on LIBERO-Long (vs.85.2% for $π_0$), and reaches 60.0% on the long-horizon deformable task Fold Cloth on a real Piper arm (vs.20.0% for $π_0$).

**Analysis:**

### 1. 摘要翻译
视觉语言动作（VLA）模型越来越依赖辅助世界模型来规划长程任务，但如何在预训练的VLA基础上参数化这些模块仍是一个开放的设计问题。现有的世界模型增强型VLA通常以高视觉带宽将逐帧视觉流传递给世界模型，并将其作为动作预测的副产品。在冻结骨干网络且受限于调整预算的情况下，这种做法使得逐帧表示和潜空间动作耦合都缺乏深入研究。我们提出了OneWM-VLA，它通过自适应注意力池化（Adaptive Attention Pooling）将每个视图压缩为每帧单个语义Token，并利用单一流匹配（flow-matching）目标同时生成潜空间流和动作轨迹，而非通过单独的解码器进行连接。实验表明，在我们的设定下，逐帧视觉带宽可缩减至单个Token而不影响长程性能。在π0（2B）骨干上使用14.71M LoRA参数训练，OneWM-VLA将MetaWorld MT50的平均成功率从47.9%提升至61.3%，在LIBERO-Long上达到95.6%，并在真实Piper机械臂的长程可变形任务Fold Cloth上达到60.0%的成功率（基线为20.0%）。

### 2. 方法动机分析
*   **驱动力**：解决VLA模型在长程任务中因逐帧预测像素级细节而导致的计算开销大、误差积累严重的问题。
*   **现有痛点**：传统方法将世界模型视为辅助，关注像素级重构导致计算冗余；长程规划导致计算量随步数激增，且缺乏有效的潜空间动作约束。
*   **研究假设**：在有限的适应预算下，长程任务的性能瓶颈不在于视觉像素细节，而在于能否提取出紧凑且对控制任务相关的语义表示。

### 3. 方法设计详解
*   **流程与模型结构**：
    1.  **视觉压缩（瓶颈层）**：利用自适应注意力池化，将预训练骨干（如PaliGemma）提取的N个特征Token压缩为1个语义Token。
        *   **多策略池化**：使用MAX（峰值响应）、SUM（全局响应）和LEARN（可学习MLP）三种策略提取 saliency 信息。
        *   **自适应融合**：通过可学习的凸组合（softmax归一化权重）动态加权上述三种池化结果。
    2.  **联合流匹配（规划层）**：将压缩后的“潜空间世界Token”与“动作轨迹”放入同一个流匹配生成器中。
        *   将潜空间状态作为动作生成的结构化先验，两者通过自注意力机制进行协同进化。
*   **关键公式解释**：
    *   联合损失函数 $\mathcal{L} = \lambda_a \mathbb{E}[\|v^a_\theta - u^a_t \|_1] + \sum \lambda_i \mathbb{E}[\|v^z_\theta - u^z_t \|_1]$：通过强制潜空间状态与动作轨迹共享流匹配时间步，实现状态转移预测与动作执行的紧密耦合，而非简单的辅助 loss。

### 4. 方法对比分析
*   **本质区别**：从传统的“预测像素”转向“紧凑潜空间动作联合流匹配”；从“辅助预测”转向“结构化先验约束”。
*   **创新贡献**：提出每帧单个语义Token的极端压缩方案，验证了其在长程任务中的有效性和隐式正则化作用。
*   **适用场景**：高维输入、长程复杂任务、计算资源受限或需要快速适应的机器人控制场景。

### 5. 实验分析
*   **关键结论**：随着每帧Token数从1增加到12，成功率反而单调下降，证明了单Token并非降级方案，而是最优运行点。
*   **优势**：在保持内存预算极低的情况下，显著提升长程任务成功率（MetaWorld +15.15%，真实Piper +40%）。
*   **局限**：目前的实验仅在特定的操控任务序列中验证，未探讨极高感知复杂度的环境。

### 6. 实用指南
*   **实现细节**：
    *   **超参数**：推荐设置 $\tau=0.1$ 进行融合权重计算；损失权重 $\lambda_r, \lambda_w \approx 0.1$。
    *   **架构**：在预训练VLA骨干上外接轻量化适配器（LoRA），冻结主干网络。
*   **迁移**：该方法模块化程度高，可轻松插入任何预训练VLA，只需替换视觉编码器输出的Token合并模块。

### 7. 总结
*   **核心思想**：通过极端视觉压缩与动作潜空间联合流匹配，实现高效长程规划。
*   **速记版pipeline**：
    1.  多策略池化压缩视觉特征。
    2.  可学习融合生成单语义Token。
    3.  联合流匹配预测世界状态与动作。
    4.  通过协同训练实现结构化长程约束。

**Key Findings:**

- We introduce OneWM-VLA, which compresses each view into a single semantic token per frame through an Adaptive Attention Pooling, and produces the resulting latent stream and the action trajectory under a single flow-matching objective rather than connecting them through a separate decoder.
- Trained with 14.71M LoRA parameters on a $π_0$ (2B) backbone, OneWM-VLA improves the average success rate from 47.9% to 61.3% on MetaWorld~MT50, reaches 95.6% on LIBERO-Long (vs.85.2% for $π_0$), and reaches 60.0% on the long-horizon deformable task Fold Cloth on a real Piper arm (vs.20.0% for $π_0$).

**Links:**

- [PDF](https://arxiv.org/pdf/2605.07931v1)
- [arXiv](https://arxiv.org/abs/2605.07931v1)

---

