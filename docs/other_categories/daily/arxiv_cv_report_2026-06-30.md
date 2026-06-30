time: 20260630

# Arxiv Computer Vision Papers - 2026-06-30

## Executive Summary

## 每日报告执行摘要：计算机视觉领域最新论文（2026-06-29）

### 一、主要主题与趋势

本批10篇论文主要围绕**机器人学习与操作**（4篇）、**3D场景理解与生成**（2篇）、**视频编辑与生成**（2篇）、**手物交互**（1篇）和**无人机路径规划**（1篇）展开。核心趋势包括：

- **合成数据与仿真到现实的迁移**愈发成熟（如VLK利用重建场景生成合成交互数据训练人形机器人）。
- **开放词汇与零样本能力**成为3D理解与生成的标准追求（如基于2D检测器的3D高斯分割、零样本铰接物体恢复）。
- **具身智能中视觉-语言-动作模型的深度集成**，尤其是引入链式思维（Chain-of-Thought）进行密集监督。
- **大规模基准与数据集**推动视频编辑、机器人规划等方向标准化（如百万级视频编辑数据集Goku）。
- **云-边协同计算**在视频生成中首次出现（EcoVideo），结合熵驱动策略平衡质量与效率。

### 二、特别重要或创新的论文

1. **UnfoldArt**（论文5）：提出零样本从文本或图像恢复完整铰接3D物体，无需任何3D标注，在3D生成与交互场景中极具潜力。
2. **Goku**（论文6）：首个百万级指令式视频编辑数据集与基准，将极大推动文本驱动视频编辑的标准化研究。
3. **VLK**（论文1）：结合重建场景与合成交互训练人形机器人全身移动操作，在仿真到真实迁移方面表现出色。
4. **GROW²**（论文3）：聚焦机器人工具使用中的“哪个工具”与“放在哪”的联合定位问题，是具身grounding任务的典型代表。
5. **Training VLA with Dense Embodied CoT**（论文10）：首次将密集链式思维监督引入视觉-语言-动作模型训练，显著提升长时程任务规划能力。

### 三、新兴研究方向或技术

- **基于2D检测器引导的3D高斯分割**（论文2）：利用成熟2D开放词汇检测器直接处理3D高斯场的指代分割，绕过昂贵的3D标注，有望成为3D场景理解的标配范式。
- **锚定机器人关键点的顺序规划**（论文4）：将物体或环境关键点作为规划锚点，简化复杂操作序列的表示与学习。
- **熵编排的云-边视频生成**（论文9）：根据内容复杂度动态分配云端与边缘计算资源，为实时视频生成提供新思路。
- **密集具身链式思维**（论文10）：将语言模型中的思维链思想推广到机器人动作生成，每个中间步骤都包含视觉、语言和动作的联合监督。

### 四、建议全文阅读的论文

- **VLK**（论文1）：适合人形机器人、仿真到现实迁移、操作学习方向的研究者。
- **UnfoldArt**（论文5）：对3D生成、铰接物体建模或零样本方法感兴趣者必读。
- **Goku**（论文6）：视频编辑、指令驱动生成、大规模数据集建设领域的核心参考。
- **Training VLA with Dense Embodied CoT**（论文10）：具身智能、视觉语言导航、任务规划方向的突破性工作。
- **3D高斯开放词汇分割**（论文2）：从事3D场景理解、高斯泼溅或开放词汇分割的研究者应重点关注。

---

## Table of Contents

1. [VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes](#2606.30645v1)
2. [Open-Vocabulary and Referring Segmentation for 3D Gaussians Using 2D Detectors](#2606.30638v1)
3. [GROW$^2$: Grounding Which and Where for Robot Tool Use](#2606.30632v1)
4. [Sequential Planning via Anchored Robotic Keypoints](#2606.30613v1)
5. [UnfoldArt: Zero-Shot Recovery of Full Articulated 3D Objects from Text or Image](#2606.30608v1)
6. [Goku: A Million-Scale Universal Dataset and Benchmark for Instruction-Based Video Editing](#2606.30599v1)
7. [Towards in-the-wild Egocentric 3D Hand-Object Pose Estimation](#2606.30598v1)
8. [MOAR Planner: Multi-Objective and Adaptive Risk-Aware Path Planning for Infrastructure Inspection with a UAV](#2606.30575v1)
9. [EcoVideo: Entropy-Orchestrated Video Generation Paradigm in Cloud-Edge Dynamics](#2606.30557v1)
10. [Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision](#2606.30552v1)

---

## Papers

<a id='2606.30645v1'></a>
## [VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes](https://arxiv.org/abs/2606.30645v1)

**Authors:** Yen-Jen Wang, Jiaman Li, Sirui Chen, Takara E. Truong, Pei Xu, Pieter Abbeel, Rocky Duan, Koushil Sreenath, Angjoo Kanazawa, Carmelo Sferrazza, Guanya Shi, Karen Liu

**Published:** 2026-06-29

**Categories:** cs.RO, cs.AI, cs.GR, eess.SY

**Abstract:**

Perception-based humanoid loco-manipulation requires connecting egocentric observations and task instructions to whole-body motion. Learning this mapping requires synchronized egocentric images, language commands, and robot-compatible kinematic trajectories, yet no existing data source provides this complete tuple at scale. We address this bottleneck by generating vision-language-kinematics (VLK) supervision synthetically in reconstructed scenes. Our pipeline leverages 3D Gaussian Splatting to reconstruct metric-scale indoor environments, synthesizes navigation and object-interaction trajectories using privileged scene information, and renders paired egocentric observations after the fact. We produce 48,000 paired trajectories with no human intervention and train a VLK policy that predicts short-horizon whole-body kinematic trajectories. A whole-body tracker converts these predictions into actions on the physical humanoid. We evaluate on the physical Unitree G1 performing navigation and single-object transport, demonstrating that synthesized interactions in reconstructed scenes provide effective supervision for sim-to-real perception-based humanoid loco-manipulation. Project Website: https://vision-language-kinematics.github.io/

**Analysis:**

这份报告对论文《VLK: Learning Humanoid Loco-Manipulation from Synthetic Interactions in Reconstructed Scenes》进行深度技术分析。

### 1. 摘要翻译
基于感知的类人机器人移动操作（Loco-manipulation）需要将视觉观察和任务指令映射为全身运动。这种映射学习依赖于同步的自中心视觉、语言指令和机器人兼容的运动轨迹，但目前缺乏大规模的此类数据。我们通过在重建场景中生成“视觉-语言-运动学”（VLK）监督数据解决了这一瓶颈。我们的流水线利用3D高斯溅射（3DGS）重建真实感室内环境，利用特权场景信息合成导航和物体交互轨迹，并渲染配对的自中心视觉观测。我们在无需人工干预的情况下生成了48,000条配对轨迹，并训练了一个预测短时程全身运动学轨迹的VLK策略。通过全身跟踪器，将预测转换为物理人形机器人的动作。实验验证了该合成数据可有效支持真实世界中基于感知的移动操作。

### 2. 方法动机分析
- **核心动机**：解决类人机器人移动操作领域“高维视觉-指令-运动”配对数据稀缺的问题，实现低成本、大规模的数据生成。
- **痛点分析**：
    - 真实世界遥操作（Teleoperation）采集成本高、难以扩展。
    - 现有动作捕捉数据集缺乏对应的机器人自中心视角。
    - 现有自中心视频数据集缺乏高质量、兼容的机器人运动学轨迹。
- **核心直觉**：利用3DGS实现“真实场景模拟（Real2Sim）”，结合预定义的机器人运动学任务实现“Sim2Real”策略，通过解耦场景渲染与行为合成，实现大规模高质量数据的自动化生成。

### 3. 方法设计详解
- **Pipeline**：
    1. **场景重构与标注**：利用iPhone扫描并使用3DGS重构场景，标注物体包围盒和可通行区域。
    2. **机器人运动合成**：基于场景约束，利用扩散模型（Diffusion model）合成包含导航和物体交互（抓取/放置）的轨迹，并显式包含腕部接触标签。
    3. **自中心渲染**：在Isaac Sim中放置G1机器人，结合相机畸变、灯光变化等域随机化技术渲染RGB图像。
    4. **VLK策略训练**：输入（图像+指令+当前状态），预测短时程（1秒）运动轨迹与接触标签，使用流匹配（Flow-matching）训练。
    5. **全身跟踪部署**：利用基于SceneBot的接触感知跟踪器，将VLK预测的运动轨迹转化为底层电机关节指令。
- **关键设计**：将接触标签作为运动学的“辅助状态”，通过显式接触状态约束，大幅提升了移动操作的物理稳定性。

### 4. 方法对比分析
- **本质区别**：从传统的“数据录制”转向“场景内生成”。不像VLA模型通过海量互联网数据学习，本方法通过结构化的合成数据，强制机器人学习特定任务的物理约束（如接触时机）。
- **创新点**：
    - 首次将3DGS场景重建与机器人全身动力学运动合成无缝结合。
    - 引入了基于显式接触标签的运动预测，解决了移动物体交互时常出现的抓取偏移问题。
- **适用场景**：复杂室内环境下的长时程导航与物体搬运任务。

### 5. 实验分析
- **验证方法**：MuJoCo仿真环境下的闭环评估 + 真实物理世界Unitree G1实机部署。
- **关键结果**：在移动与搬运任务中表现出高成功率，且证明了数据量增长与成功率的显著正相关性（尤其是抓取任务）。
- **局限性**：当前仅限于大中型物体的搬运，对小型、精细化操作（如拿取杯子、工具）的支持能力仍需进一步提升。

### 6. 实用指南
- **开源/复现**：项目主页：[vision-language-kinematics.github.io](https://vision-language-kinematics.github.io)。复现核心在于重构高质量的3DGS场景和交互模型的Retargeting过程。
- **实现细节**：建议严格配置域随机化（Domain Randomization）范围，尤其是相机扰动和光照变化，这在从模拟器迁移到实机时起决定性作用。
- **迁移性**：该框架高度解耦，只要更换运动合成的源数据集（Retargeting到对应机器人），即可迁移到其他形态的人形机器人上。

### 7. 总结
- **核心思想**：利用重建环境合成感知运动数据，解耦训练与数据瓶颈。
- **速记版pipeline**：
    1. 手机扫描场景并用3DGS建模；
    2. 合成包含抓取动作的机器人轨迹；
    3. 渲染带有域随机化的第一视角图；
    4. 训练策略预测未来动作与接触状态；
    5. 使用跟踪控制器驱动物理机器人。

**Key Findings:**

- We evaluate on the physical Unitree G1 performing navigation and single-object transport, demonstrating that synthesized interactions in reconstructed scenes provide effective supervision for sim-to-real perception-based humanoid loco-manipulation.
- Project Website: https://vision-language-kinematics.github.io/

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30645v1)
- [arXiv](https://arxiv.org/abs/2606.30645v1)

---

<a id='2606.30638v1'></a>
## [Open-Vocabulary and Referring Segmentation for 3D Gaussians Using 2D Detectors](https://arxiv.org/abs/2606.30638v1)

**Authors:** Jameel Hassan, Yasiru Ranasinghe, Vishal Patel

**Published:** 2026-06-29

**Categories:** cs.CV

**Abstract:**

3D Gaussian Splatting (3DGS) has emerged at the forefront of 3D scene reconstruction. Extending 3DGS with language-driven, open-vocabulary understanding has gained significant attention for real-world applications such as embodied AI. Recent methods achieve this by learning an instance feature attribute and assigning semantics by distilling high-dimensional Contrastive Language-Image Pretraining (CLIP) features directly into the scene representation. However, the instance grouping mechanisms of these methods either require a predefined number of instances or suffer from noise in their bottom-up grouping strategies. Furthermore, the reliance on CLIP restricts semantic understanding to simple noun phrases, preventing complex spatial reasoning and referential expression grounding. We present GaussDet, a method that circumvents the need for dense CLIP features by leveraging discrete, open-vocabulary 2D object detectors with referring expression capabilities. We learn instance features for individual Gaussians to decompose the scene into 3D instance groups. By rendering these groups and aggregating semantic votes from multi-view 2D detections, we generate a robust View-Aggregated Semantic Label Distribution (VASD) for each 3D instance. This view-aggregation strategy acts as a strong regularizer, attenuating spurious labels caused by low-quality instance grouping. Our approach enables a straightforward, zero-shot extension from simple language queries to complex referential grounding. Extensive evaluations across two key tasks -- open-vocabulary segmentation (LeRF-OVS, ScanNet) and referring expression grounding (Ref-LeRF) -- demonstrate that GaussDet achieves consistent improvements over existing methods. Most notably, we achieve a substantial 16.7% mIoU improvement in referential grounding within a strict zero-shot setting.

**Analysis:**

这是一份关于《GaussDet: Open-Vocabulary and Referring Segmentation for 3D Gaussians Using 2D Detectors》的专业技术分析：

### 1. 核心贡献摘要
该论文提出了 **GaussDet**，一种将 3D 高斯泼溅（3DGS）与开放词汇 2D 检测器相结合的新型语义分割框架。其核心贡献在于通过多视图投票机制（VASD）而非传统的 CLIP 特征蒸馏，实现了对 3D 场景的精准实例分割，显著提升了从简单语义查询到复杂指代关系（Referring Expression）的泛化能力。

### 2. 关键创新与方法论
*   **摆脱 CLIP 依赖：** 传统方法直接蒸馏 CLIP 特征，受限于其特征分辨率和对复杂指代理解的局限性。GaussDet 转向利用现有的高性能 2D 开放词汇检测器，直接处理对象级别的语义，从而更准确地识别和定位物体。
*   **视图聚合语义标签分布（VASD）：** 这是该方法的核心创新。通过将 3D 实例渲染并在多视图下与 2D 检测结果进行交叉验证，系统能够滤除低质量的聚类噪声。这种多视图聚合机制充当了强大的正则化器，保证了 3D 实例标签的一致性和鲁棒性。
*   **3D 实例特征学习：** 该方法通过学习每个高斯的实例特征，实现了 3D 空间内的显式实例分解，从而支持更精细的指代表达解析。

### 3. 对领域的潜在影响
*   **范式转换：** 标志着 3D 语义理解从“像素级特征蒸馏”向“对象级实例对齐”的范式转变。这为 3DGS 在下游复杂任务中的应用提供了更可靠的语义基础。
*   **指代理解突破：** 在零样本（Zero-shot）设置下实现 16.7% mIoU 的提升，证明了该方法在处理“位于桌子左侧的红色杯子”这类复杂空间指代问题上具有巨大潜力，这正是当前 Embodied AI（具身智能）亟需突破的技术难点。

### 4. 受益的相关领域与应用
*   **机器人操作与导航：** 在复杂室内环境中，机器人需要理解“把那本蓝色的书拿给我”这类自然语言指令，GaussDet 的高精度指代理解能力可直接应用于此类场景。
*   **AR/VR 交互：** 支持用户通过自然语言与虚拟或重建的真实环境进行交互，能够更准确地对特定对象进行操作。
*   **3D 场景理解：** 改进了大规模场景下的语义分割质量，对自动驾驶、智慧城市重建等依赖精确语义识别的领域具有直接的推动作用。

### 5. 可推断的局限性
*   **依赖 2D 检测器的性能上限：** 既然方法高度依赖 2D 检测器，如果 2D 检测器在某些极端视角或遮挡严重的情况下失效，该方法可能会产生错误的 3D 标签。
*   **渲染与聚合的计算成本：** 虽然该方法利用了 2D 检测器，但在训练或推理过程中，需要频繁进行多视图渲染和投票，这可能带来额外的计算开销和内存占用。
*   **细粒度几何限制：** 尽管语义理解有所增强，但如果 3DGS 本身的几何重建（Geometry）不够精确（如物体边界模糊），语义标签的边缘可能会出现“渗漏”或边界不够清晰的情况。

### 专家点评
这篇论文的趣味性在于它并没有盲目地去优化 CLIP 蒸馏这个“苦力活”，而是采取了一种“借力打力”的策略——利用 2D 检测器强大的归纳偏置（Inductive Bias）来校正 3D 空间中的噪声。这种**从 2D 到 3D 的验证机制**非常符合计算机视觉处理多视角一致性问题的逻辑，是解决当前 3DGS 语义模糊难题的一条极具前景的路径。

**Key Findings:**

- We present GaussDet, a method that circumvents the need for dense CLIP features by leveraging discrete, open-vocabulary 2D object detectors with referring expression capabilities.
- Our approach enables a straightforward, zero-shot extension from simple language queries to complex referential grounding.
- Most notably, we achieve a substantial 16.7% mIoU improvement in referential grounding within a strict zero-shot setting.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30638v1)
- [arXiv](https://arxiv.org/abs/2606.30638v1)

---

<a id='2606.30632v1'></a>
## [GROW$^2$: Grounding Which and Where for Robot Tool Use](https://arxiv.org/abs/2606.30632v1)

**Authors:** Yuhong Deng, Yuyao Liu, David Hsu

**Published:** 2026-06-29

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

Can the robot use a plate to cut a cake if no knife is available? Tool use greatly expands robot capabilities, but to use tools creatively beyond their intended functions, the robot faces the challenge of $\textit{open-world affordance grounding}$: select an open-category object to act as a tool and localize its specific region of action. To this end, we introduce GROW$^2$ (GROunding Which and Where), which leverages object parts as a natural abstraction to split the grounding process hierarchically into semantic and geometric levels, thus bypassing the need for data-heavy, end-to-end training. Semantically, GROW$^2$ harnesses the commonsense reasoning of Vision-Language Models (VLMs) to parse a natural-language task instruction, select a suitable object as the tool, and identify task-relevant parts on the tool and the target object. Geometrically, vision foundation models then ground the selected parts into precise 3D regions from a single RGB-D image. Experiments on established benchmarks show that GROW$^2$ outperforms state-of-the-art baselines on affordance prediction benchmarks. Further, it achieves zero-shot generalization over open-category objects and outperforms baselines in both simulated and real-world robot tool use experiments.

**Analysis:**

### 1. 摘要翻译
机器人能在没有刀的情况下用盘子切蛋糕吗？尽管工具使用极大地扩展了机器人的能力，但要创造性地使用超出其预期功能的工具，机器人面临“开放世界具身智能接地”（open-world affordance grounding）的挑战：即从开放类别对象中选择工具并定位特定的操作区域。为此，我们引入了 **GROW²** (Grounding Which and Where)。该方法利用物体部件作为自然抽象，将接地过程分层为语义和几何两个级别，从而绕过了对大规模端到端数据训练的依赖。在语义层，GROW² 利用视觉-语言模型（VLM）的常识推理来解析任务指令，选择合适的工具，并识别工具和目标对象上的任务相关部件。在几何层，视觉基础模型将选定的部件接地为单张RGB-D图像中的精确3D区域。实验证明，GROW² 在基准测试中优于现有的最先进方法，实现了对开放类别对象的零样本泛化，并在模拟和真实机器人工具使用实验中展现了卓越性能。

### 2. 方法动机分析
- **驱动力**：解决机器人在非结构化环境中处理未知工具（Open-world）的泛化难题，使机器人能像人类一样“即兴”使用工具。
- **现有痛点**：
    1. **数据依赖**：传统的端到端训练需要大规模标注数据，且难以泛化至未见过的任务-物体对。
    2. **空间推理能力不足**：直接让VLM进行细粒度的空间定位容易产生幻觉。
    3. **几何信息缺失**：纯2D接地方法无法提供物理操纵所需的3D深度和几何约束。
- **研究假设**：物体部件（Part）是连接“语义意图”与“几何操作”的理想中间表达。

### 3. 方法设计详解
**流程总结：**
1. **工具与部件选择（语义层）**：
   - 使用VLM枚举场景对象，并用SAM3辅助分割提取每个物体的语义候选部件。
   - 结合任务指令，利用VLM推理选出工具对象 $o_A$、目标对象 $o_B$ 及对应的抓取部件 $p_G$ 和交互部件 $p_A, p_B$。
2. **3D 具身接地（几何层）**：
   - **重建与配准**：利用单张RGB-D图像对目标物体进行3D重建，并使用ICP配准将网格映射到原始场景。
   - **多视角融合**：将重建的物体网格从8个视角渲染，利用SAM3在2D图上分割出选定的部件区域，通过反投影将掩码聚合到3D空间，最后使用DBSCAN聚类剔除噪声，形成精确的3D affordance区域。

### 4. 方法对比分析
- **本质区别**：GROW² 采用了“感知层级解耦”策略。它不是让一个网络学习所有映射，而是将语义决策留给VLM，将空间定位留给经过渲染的3D几何重建。
- **核心贡献**：提出了一种无需大规模训练数据的零样本工具使用范式，通过部件作为桥梁，解决了单视图下物体遮挡导致的定位精度问题。

### 5. 实验分析（精简版）
- **验证方法**：在AGD20K、PIAD数据集上测试泛化性能；在SAPIEN 3模拟器及Franka手臂上进行多任务工具使用测试。
- **结论**：GROW² 在绝大多数指标上优于基线；在模拟和真实世界实验中，GROW² 的任务成功率远超其他方法。
- **优缺点**：精度高、可解释性强；主要瓶颈在于VLM查询延迟和多阶段计算带来的约16秒处理时间。

### 6. 实用指南
- **开源情况**：论文中提及了相关技术栈（SAM3, SAM3D），未明确指出代码仓，但方案可复现性高。
- **关键细节**：
    - 多视角渲染（K=8）是消除遮挡的关键。
    - DBSCAN聚类是处理掩码噪声的必要步骤。
    - 使用VLM fallback机制（当部件描述模糊时）能显著提升鲁棒性。
- **迁移建议**：本方法非常适合应用于需要精细化操纵的任务（如装配、烹饪），其“部件抽象”的思想可直接复用于各类基于视觉的机器人操控。

### 7. 总结
- **核心思想**：以物体部件为桥梁，解耦语义决策与几何定位，实现工具使用的零样本泛化。
- **速记版pipeline**：
    1. VLM分析场景，选定工具、目标及对应关键部件。
    2. 基于单帧重建出物体精确3D网格。
    3. 多视角渲染物体，在2D图提取部件掩码。
    4. 反投影掩码到3D空间，聚合生成精确交互区。

**Key Findings:**

- To this end, we introduce GROW$^2$ (GROunding Which and Where), which leverages object parts as a natural abstraction to split the grounding process hierarchically into semantic and geometric levels, thus bypassing the need for data-heavy, end-to-end training.
- Experiments on established benchmarks show that GROW$^2$ outperforms state-of-the-art baselines on affordance prediction benchmarks.
- Further, it achieves zero-shot generalization over open-category objects and outperforms baselines in both simulated and real-world robot tool use experiments.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30632v1)
- [arXiv](https://arxiv.org/abs/2606.30632v1)

---

<a id='2606.30613v1'></a>
## [Sequential Planning via Anchored Robotic Keypoints](https://arxiv.org/abs/2606.30613v1)

**Authors:** Bryce Grant, Aryeh Rothenberg, Logan Senning, Zonghe Chua, Zach Patterson, Peng Wang

**Published:** 2026-06-29

**Categories:** cs.RO

**Abstract:**

We present Sequential Planning via Anchored Robotic Keypoints, SPARK, a training-free neurosymbolic manipulation system that reaches 43.7% on six LIBERO-PRO position \& task cells, more than doubling CaP-Agent0 and Vision-Language-Action (VLA) baselines. CaP-Agent0, a multi-turn code-generation agent, achieves 18.2% by re-querying an LLM at every turn, but its restart-from-scratch solution proves costly against minor policy failures. Perception is the layer that fails most under position and task changes so SPARK spends its computation there. A single Gemini call composes the plan as a typed behavior tree (BT) of composable primitives, each already containing the low-level control (motion, grasping, depth geometry) a code-generation agent would otherwise regenerate on every trial. The rest of the budget goes to perception: a second Gemini call proposes three alternative text prompts per object, SAM3 evaluates each, and we keep the prompt$\to$label pair with the most confident detection and a recovery loop then retries a failed primitive against freshly detected objects, with no new LLM call. The alternative prompts add +27.7 points on the spatial suite and +10.0 on the object suite, with the recovery loop adding +5.0 overall. SPARK runs the same primitives on three robot families (UR10e, Franka FR3, bimanual Franka) across nine unique tasks at twenty trials each, averaging 68%. Since the detector, planner, and controller modules sit behind the typed plan, they swap independently without training, and each primitive's checkable post-condition traces a failure to the corresponding module or a kinematic limit. Every trial logs a verified, labeled trajectory, so a training-free planner that already beats VLAs can supply the data those policies need without teleoperation. Project page: https://cwru-aism.github.io/spark-page/

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇关于 **SPARK (Sequential Planning via Anchored Robotic Keypoints)** 的论文进行了深入分析。以下是详细评估：

### 1. 论文核心贡献总结
SPARK 提出了一种无需训练的神经符号（Neurosymbolic）操纵系统，通过将任务分解为预定义的行为树（Behavior Trees, BT）原语，在 LIBERO-PRO 基准测试中实现了 43.7% 的成功率，显著优于现有的 VLA（视觉-语言-动作）模型。其核心贡献在于通过将计算重心从“重复调用大模型生成代码”转移到“高鲁棒性的感知与闭环恢复机制”，实现了在不同机器人平台上的零样本（zero-shot）泛化。

### 2. 关键创新与方法论
*   **训练自由的模块化架构**：摒弃了端到端 VLA 频繁重构控制代码的复杂性，将低层控制（运动、抓取、深度感知）封装在固定的“行为树原语”中。
*   **感知增强的提示策略 (Prompt Engineering for Perception)**：这是论文在计算机视觉方面的核心亮点。SPARK 通过 Gemini 自动生成多种文本提示，并利用 SAM (Segment Anything Model) 进行多方案评估，选取置信度最高的检测结果。这种策略将感知视为任务失败的主因并投入了主要的计算资源。
*   **无需重构的恢复机制**：当感知或执行出现故障时，系统触发内置的 recovery loop，针对新感知的对象进行重试，而无需重新调用 LLM，大大降低了延迟并提高了成功率。
*   **可解释性与数据生成**：由于每个原语都有可验证的后置条件（post-condition），系统能精确溯源故障点。此外，它能自动生成带标签的轨迹数据，为 VLA 的冷启动训练提供了高质量的数据源。

### 3. 对领域的潜在影响
*   **从“黑盒模型”转向“可控框架”**：该工作挑战了当前端到端 VLA 必须依赖海量标注数据才能泛化的范式，证明了基于物理原语的神经符号系统在复杂操作任务中具有更强的鲁棒性。
*   **感知能力的解耦**：论文通过 SAM 和多提示评估展示了如何通过强化视觉感知层来提升复杂任务执行成功率，为“感知-规划-动作”链条的优化提供了范例。
*   **弥合模拟与现实（Sim-to-Real）的成本差距**：通过自动生成可控轨迹，该研究为数据驱动的机器人学习提供了一种无需昂贵人类示教（Teleoperation）的数据获取新途径。

### 4. 潜在的应用领域
*   **工业自动化与柔性制造**：由于支持 UR10e、Franka 等多种机器人平台且无需训练，该方法非常适合多任务、小批量的生产场景。
*   **复杂环境下的辅助机器人**：在家庭或实验室等非结构化环境中，利用 SAM 进行自适应感知的策略可以应对物体遮挡和变动。
*   **机器人数据集构建**：作为一种“自动数据标注机”，SPARK 可用于大规模生成机器人操纵轨迹，加速通用机器人大模型的研发。

### 5. 推断的局限性
*   **任务定义的预设依赖**：尽管系统是无需训练的，但预定义行为树（BT）原语可能在遇到完全超出设计的复杂操作（如非刚性物体、高度复杂的精细组装）时显得僵化。
*   **感知与规划的串行开销**：虽然通过避免频繁调用 LLM 提升了速度，但多提示感知评估（Gemini + SAM 评估）仍可能带来较高的推理延迟，在大规模任务链中可能影响实时性。
*   **对预训练模型（Gemini, SAM）的依赖**：该系统的鲁棒性高度依赖于底层视觉模型（SAM）的分割精度和多模态模型（Gemini）的规划能力；如果遇到极端环境导致 SAM 分割失败，整个系统依然可能失效。

**总结建议：** 这篇论文对于计算机视觉研究者特别具有吸引力，因为它展示了**如何巧妙地利用现有的视觉基础模型（如 SAM）通过提示词工程来修正机器人感知的鲁棒性**，而不必追求庞大的端到端模型。这代表了当前机器人社区从“盲目堆砌模型参数”转向“优化模块化感知架构”的一种重要趋势。

**Key Findings:**

- We present Sequential Planning via Anchored Robotic Keypoints, SPARK, a training-free neurosymbolic manipulation system that reaches 43.7% on six LIBERO-PRO position \& task cells, more than doubling CaP-Agent0 and Vision-Language-Action (VLA) baselines.
- The rest of the budget goes to perception: a second Gemini call proposes three alternative text prompts per object, SAM3 evaluates each, and we keep the prompt$\to$label pair with the most confident detection and a recovery loop then retries a failed primitive against freshly detected objects, with no new LLM call.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30613v1)
- [arXiv](https://arxiv.org/abs/2606.30613v1)

---

<a id='2606.30608v1'></a>
## [UnfoldArt: Zero-Shot Recovery of Full Articulated 3D Objects from Text or Image](https://arxiv.org/abs/2606.30608v1)

**Authors:** Mohamed el amine boudjoghra, Ivan Laptev, Angela Dai

**Published:** 2026-06-29

**Categories:** cs.CV

**Abstract:**

Articulated 3D objects are essential for interactive environments in embodied AI, robotics, and virtual reality, but reconstructing their structure and motion from sparse observations remains challenging. Existing approaches remain largely constrained by lack of supervised data or lack the priors needed to reliably recover articulation, hidden geometry, and internal object structure. We present the first debate-driven agentic approach to articulated 3D object reconstruction from text or image inputs that both grounds articulation reasoning in concrete motion and exposes the occluded geometry revealed under articulation. High-level agents reason about object semantics and motion using knowledge from vision-language and video models, while low-level agents estimate articulation parameters and interaction points; together, they engage in a two-round structured debate that first exploits global--local disagreement and then grounds the agents in freely generated video. The same video prior, conditioned on the agreed articulation, then drives each part through its motion to expose occluded interiors and geometry that cannot be inferred from a single static view. By combining agentic reasoning with a video generative prior, our approach jointly infers articulation and reconstructs complete 3D articulated objects, producing high-fidelity geometry, internal structure, and motion-consistent states beyond directly observed surfaces.

**Analysis:**

以下是对《UnfoldArt: Zero-Shot Recovery of Full Articulated 3D Objects from Text or Image》的深入分析：

### 1. 摘要翻译
铰接式3D物体在具身智能、机器人和虚拟现实领域至关重要，但从稀疏观测中重建其结构和运动仍具挑战性。现有方法受限于监督数据缺失或缺乏可靠恢复铰接、隐藏几何结构及内部对象结构的先验。我们提出了首个**辩论驱动的智能体（Agentic）方法**，用于从文本或图像输入中进行铰接式3D重建。该方法将铰接推理建立在具体的运动之上，并揭示了铰接过程中才显露的被遮挡几何结构。高层智能体利用视觉语言和视频模型知识进行语义与运动推理，底层智能体估算铰接参数与交互点；通过两轮结构化辩论，利用全局-局部差异并结合生成的视频证据进行纠偏。该视频先验随后驱动各部件运动，暴露遮挡的内部结构，最终生成具有高保真几何、内部结构及运动一致性的完整交互式铰接3D物体。

### 2. 方法动机分析
*   **驱动力**：作者旨在解决“单图/单文本输入到完整铰接3D物体”这一零样本（Zero-shot）重建难题，特别关注如何恢复物体内部的“隐藏几何”。
*   **现有痛点**：
    1.  **监督依赖**：传统方法过度依赖PartNet-Mobility等小规模数据集，泛化性差。
    2.  **单一视图缺失信息**：静止视图无法推断运动后的被遮挡部分。
    3.  **模型不可靠**：直接提示（Prompting）大模型进行3D铰接推理往往产生幻觉，缺乏几何一致性。
*   **研究假设**：通过引入“辩论机制”（多视角/多智能体对质）和“视频先验”（以生成视频作为运动证据），可以消除铰接推理的歧义并实现几何补全。

### 3. 方法设计详解
*   **流程总结**：
    1.  **初始化**：使用TRELLIS生成初始3D Mesh，并通过Flux渲染得到视觉对齐的参考图。
    2.  **智能体层级推理**：将任务分解为Decomposer（全局语义）、Grounder（局部语义/分割）和Articulator（参数预估）。
    3.  **两轮辩论机制**：
        *   **Round 1（一致性检查）**：利用全局与局部智能体的视角差异，如果Decomposer和Grounder在铰链位置等参数上产生冲突，则触发出错标志。
        *   **Round 2（视频证据）**：利用WAN-VACE生成一段基于预测运动的视频，将此视频作为“真理”证据，让智能体重新审查并优化铰接参数。
    4.  **铰接导向的重建**：利用优化的铰接参数$J^*_i$，将部件运送到最大运动状态，通过3D latent-inpainting（利用RePaint）填补遮挡区域。
*   **算法解释**：关键在于将辩论视为一种纠偏机制。通过强制智能体在视频证据下“看图说话”，实现了从无监督到可靠铰接参数的跃迁。

### 4. 方法对比分析
*   **本质区别**：与传统“端到端生成”或“纯优化法”不同，该方法通过智能体间的“对抗/对质”产生纠偏信号，而非依赖单一模型的预测。
*   **创新贡献**：首次提出**辩论驱动的智能体框架**，并将**视频生成模型作为几何验证器**，巧妙地解决了遮挡物体的重建难题。
*   **适用场景**：适用于各类具有铰接结构的物体，尤其是零样本下的非标准化、复杂运动部件。

### 5. 实验分析（精简版）
*   **验证方法**：在Objaverse（Household及OOD split）和PartNet-Mobility上进行零样本实验。
*   **关键结论**：在超出训练分布的复杂物体（如机器人手臂）上，该方法展现出显著优于SINGAPO等监督方法的性能，大幅降低了轴向和枢轴的误差。
*   **局限**：重建过程较为耗时（13分钟至1小时），且对于平坦按钮等微小运动的捕捉存在挑战。

### 6. 实用指南
*   **实现细节**：建议关注Prompt Engineering（如本文提供的多种System Prompt）。核心逻辑在于将复杂任务拆解为多Agent协作，避免单个LLM处理过多维度的信息。
*   **迁移建议**：该辩论机制可轻易迁移至其他需要高精度推理的生成任务中，例如CAD建模或复杂场景的物理属性估计。

### 7. 总结
*   **核心思想**：通过智能体间的辩论与视频运动验证，实现可靠的零样本铰接3D重建。
*   **速记版Pipeline**：
    1. 生成基础Mesh并分割部件；
    2. 全局与局部智能体对参数进行对质辩论；
    3. 生成参考视频验证运动合理性并修正参数；
    4. 依据参数进行3D inpainting补全被遮挡的内部结构。

**Key Findings:**

- We present the first debate-driven agentic approach to articulated 3D object reconstruction from text or image inputs that both grounds articulation reasoning in concrete motion and exposes the occluded geometry revealed under articulation.
- By combining agentic reasoning with a video generative prior, our approach jointly infers articulation and reconstructs complete 3D articulated objects, producing high-fidelity geometry, internal structure, and motion-consistent states beyond directly observed surfaces.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30608v1)
- [arXiv](https://arxiv.org/abs/2606.30608v1)

---

<a id='2606.30599v1'></a>
## [Goku: A Million-Scale Universal Dataset and Benchmark for Instruction-Based Video Editing](https://arxiv.org/abs/2606.30599v1)

**Authors:** Sen Liang, Cong Wang, Zhentao Yu, Fengbin Guan, Zhengguang Zhou, Teng Hu, Youliang Zhang, Yuan Zhou, Xin Li, Qinglin Lu, Zhibo Chen

**Published:** 2026-06-29

**Categories:** cs.CV

**Abstract:**

Existing instruction-based video editing datasets commonly focus on single-task appearance editing, failing to meet the complex creative demands of real-world scenarios. To bridge this gap, we present Goku, a large-scale dataset featuring 2 million high-quality, instruction-aligned video editing pairs, which is the first to extend task boundaries from basic appearance editing to multi-task and structural manipulations(e.g., precise control of subject movement). To tackle the data synthesis challenges inherent in these complex tasks, we design an efficient data synthesis pipeline that decomposes complex edits into controllable sub-problems and introduce a progressive filtering system for data reliability throughout the whole process. Furthermore, we explore the optimal network structures on Goku, and propose Goku-Edit. To deeply comprehend complex editing instructions, Goku-Edit leverages an MLLM as its text encoder and adopts a decoupled dual-branch design: a dedicated mask branch handles structural control, freeing the main branch for appearance rendering. A comprehensive video editing benchmark, Goku-Bench, is also proposed with 1,000 human-verified test cases and 7 novel editing-specific metrics. Evaluated on Goku-Bench, Goku-Edit obtains up to +8% improvement on other open-source models in terms of instruction following.

**Analysis:**

以下是对论文《Goku: A Million-Scale Universal Dataset and Benchmark for Instruction-Based Video Editing》的方法分析。

---

### 1. 摘要翻译
现有的指令式视频编辑数据集主要局限于单任务外观编辑，难以满足真实场景中复杂的创作需求。为了弥补这一空白，我们提出了Goku，这是一个大规模数据集，包含200万个高质量的指令对齐视频编辑样本，首次将任务边界从基础外观编辑扩展到多任务和结构化操纵（如精确的主体运动控制）。为解决这些复杂任务中的数据合成挑战，我们设计了一个高效的数据合成流水线，将复杂编辑分解为可控子问题，并引入了贯穿全过程的渐进式过滤系统。此外，我们探索了Goku上的最优网络结构，并提出了Goku-Edit。Goku-Edit利用多模态大语言模型（MLLM）作为文本编码器，并采用解耦的双分支设计：专门的掩码分支处理结构控制，将主分支解放用于外观渲染。我们还提出了Goku-Bench，这是一个包含1000个经人工验证的测试用例及7项创新性编辑指标的全面基准。在Goku-Bench上的评估显示，Goku-Edit在指令遵循方面比现有开源模型提升高达8%。

---

### 2. 方法动机分析
*   **驱动力**：现有的指令式视频编辑（IVE）过度简化了任务，缺乏对“结构化变形”（如摄像机移动、主体位移）和“多任务联合编辑”的支持。
*   **现有痛点**：现有数据集（如Ditto、OpenVE-3M）质量参差不齐，且任务定义单一，无法处理复杂的组合编辑（如“给狗戴帽子”同时“转为迪士尼风格”）。
*   **研究假设**：通过将复杂任务分解为可独立执行的子问题，并引入具备空间引导能力的解耦双分支网络，可以显著提升模型对复杂指令的理解与执行力。

---

### 3. 方法设计详解
#### 流程总结
1.  **数据合成 pipeline**：
    *   **分解**：将指令分解为子问题（如：先移除对象，再填充背景，最后做风格迁移）。
    *   **过滤**：通过“渐进式过滤系统”（Tier 1: 源视频质量；Tier 2: 条件验证，如掩码IoU与语义一致性；Tier 3: 后合成验证，包含跨帧一致性与语义评估），淘汰88%的低质量样本。
2.  **Goku-Edit 模型架构**：
    *   **双分支设计**：主分支生成视频内容，掩码（Mask）分支负责捕捉空间/结构信息。
    *   **RoPE-Aligned 空间交叉注意（Cross-Attention）**：解决因掩码分支降采样（1/n）导致的空间对齐误差。通过将掩码坐标缩放至高分辨率网格，确保跨分支的特征一一对应。
    *   **SpatialCFG（空间增强无分类器引导）**：训练时解耦，推理时对比。显式放大交叉分支的空间约束，防止编辑溢出和边界抖动。

---

### 4. 方法对比分析
*   **本质区别**：传统模型常将编辑视作整体生成，易发生空间偏移；Goku-Edit通过“解耦结构与内容”的手段，实现了对视频特定区域的精确控制。
*   **创新贡献**：提出“渐进式过滤”以确保百万级数据集的精细度；提出“RoPE-aligned cross-attention”解决跨分辨率特征融合的根本问题。
*   **适用场景**：适用于需要精确空间位移、复杂指令理解及多逻辑同步的视频编辑任务。

---

### 5. 实验分析（精简版）
*   **验证方法**：在Goku-Bench上与多种主流开源模型及商业闭源模型进行定量与定性对比。
*   **关键结果**：指令遵循（IF）提升达8%；在物体运动和空间关系保持方面指标优势明显。
*   **主要局限**：对超大规模计算资源的依赖；虽然在任务理解上超群，但在极致审美（AES）上与顶级商业模型仍有微小差距。

---

### 6. 实用指南
*   **开源情况**：已发布数据集、基准及模型代码（详见论文首页链接）。
*   **实现细节**：关键超参数为下采样因子 $n=4$；推理阶段需使用 SpatialCFG 策略来平衡外观生成与空间对齐。
*   **迁移可能**：双分支架构具备通用性，可轻松迁移至图像修复（Inpainting）或更长时长的动态视频编辑任务。

---

### 7. 总结
*   **核心思想**：通过任务分解与空间解耦，实现精细化的复杂视频指令编辑。
*   **速记版pipeline**：
    1.  指令分解为可解子任务。
    2.  渐进式多轮过滤数据。
    3.  双分支（内容+空间）并行生成。
    4.  空间一致性增强推理。

**Key Findings:**

- To bridge this gap, we present Goku, a large-scale dataset featuring 2 million high-quality, instruction-aligned video editing pairs, which is the first to extend task boundaries from basic appearance editing to multi-task and structural manipulations(e.g., precise control of subject movement).
- A comprehensive video editing benchmark, Goku-Bench, is also proposed with 1,000 human-verified test cases and 7 novel editing-specific metrics.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30599v1)
- [arXiv](https://arxiv.org/abs/2606.30599v1)

---

<a id='2606.30598v1'></a>
## [Towards in-the-wild Egocentric 3D Hand-Object Pose Estimation](https://arxiv.org/abs/2606.30598v1)

**Authors:** Siddhant Bansal, Zhifan Zhu, Shashank Tripathi, Jiahe Zhao, Michael J. Black, Dima Damen

**Published:** 2026-06-29

**Categories:** cs.CV

**Abstract:**

Estimating accurate 3D hand-object pose from in-the-wild egocentric RGB remains challenging due to severe occlusions and ambiguous contact. Existing learning-based methods often struggle to generalise to in-the-wild scenes and are limited by the scarcity of supervision. We address these issues with two contributions. First, we introduce EPIC-Contact, an in-the-wild egocentric dataset of 2.3K clips (62.3K frames) with dense, bijective 3D hand-object contact correspondences and posed meshes. Second, we propose HOPformer, an end-to-end transformer that jointly predicts bi-manual hand and object pose in a single forward pass. A cross-attention decoder conditions object features on hand priors, producing robust pose estimation. We test HOPformer on the in-lab 3D dataset, ARCTIC, as well as our newly introduced EPIC-Contact dataset. HOPformer reaches 82.4% success rate on ARCTIC (+6.2 pts over current SOTA). On EPIC-Contact, it nearly doubles the success rate while reducing contact deviation by 75%. EPIC-Contact, HOPformer code and checkpoints are released: https://sid2697.github.io/epic-contact.

**Analysis:**

作为计算机视觉领域的专家，我对这篇论文《Towards in-the-wild Egocentric 3D Hand-Object Pose Estimation》的分析如下：

### 1. 主要贡献总结
该论文旨在解决第一视角（Egocentric）场景下复杂遮挡和接触模糊带来的手-物 3D 位姿估计难题。研究团队提出了一个大规模的在野外（in-the-wild）数据集 **EPIC-Contact**，以及一个能够联合预测双手与物体位姿的端到端 Transformer 模型 **HOPformer**。通过引入接触约束，该方法显著提升了在真实复杂场景下的位姿估计精度与鲁棒性。

### 2. 关键创新与方法论
*   **数据集建设 (EPIC-Contact)：** 该数据集不仅包含 6.2 万帧的野外场景数据，其核心贡献在于提供了**双向的（bijective）3D 手-物接触对应关系（correspondences）**及对应的网格（mesh）位姿。这种高质量的接触标注是解决“手与物体如何相互作用”这一深层语义问题的关键。
*   **模型架构 (HOPformer)：** 
    *   采用 **Transformer** 作为骨干，实现双手与物体的联合建模，避免了传统多阶段方法中的误差累积。
    *   **Cross-Attention 机制：** 这是一个精妙的设计，通过将手部先验（hand priors）注入到物体特征解码中，强制模型在预测物体位姿时考虑手部的空间位置和接触信息，从而有效解决了遮挡带来的歧义性。

### 3. 对领域的潜在影响
*   **突破“实验室环境”瓶颈：** 目前手-物交互研究大多局限于受控的实验室环境，该研究通过提供野外数据集，将该领域推向了真实世界应用，这是一个重要的技术范式转移。
*   **接触作为先验的范式：** 证明了将“物理接触”作为一种显式的几何约束引入模型，比单纯的位姿回归更能提升模型对遮挡的鲁棒性，为未来结合物理模拟（Physics-based simulation）的位姿估计提供了思路。

### 4. 相关应用领域
*   **增强现实（AR）/ 虚拟现实（VR）：** 对于提升虚拟交互的真实感至关重要，特别是需要精准操控数字孪生物体时。
*   **机器人操作学习：** 机器人学习人类的第一视角动作（Learning from Demonstration）时，对细粒度手-物接触的理解是实现复杂操作的前提。
*   **可穿戴设备辅助：** 如视障辅助系统或智能眼镜，这些设备需要实时精准理解用户正在处理的对象和意图。

### 5. 可推断的局限性
*   **计算开销与实时性：** 尽管使用了端到端的 Transformer，但联合建模双手与物体以及复杂的注意力机制可能会带来较大的计算开销，能否在轻量级移动设备上实现实时处理（Real-time inference）仍有待验证。
*   **物体多样性：** 尽管在 EPIC-Contact 上表现出色，但面对极度复杂、非刚性或极其罕见的物体形态时，模型的泛化能力是否依然强劲，仍取决于训练数据集对物体类别的覆盖广度。
*   **接触标注的隐性噪声：** 在野外环境采集 3D 接触标注极难，该数据集中的接触标签可能存在一定程度的自动标注噪声，模型对这些噪声的容忍度将影响其最终部署效果。

**专家点评：**
这篇论文的价值在于其**数据集与模型的深度绑定**。通过构建 EPIC-Contact，作者不仅提供了数据，更定义了解决该问题的评价范式。HOPformer 的成功印证了 Transformer 在处理多模态物体交互（手+物）时的优越性，是目前 3D 手-物交互理解领域极具竞争力的前沿工作。

**Key Findings:**

- We address these issues with two contributions.
- First, we introduce EPIC-Contact, an in-the-wild egocentric dataset of 2.3K clips (62.3K frames) with dense, bijective 3D hand-object contact correspondences and posed meshes.
- Second, we propose HOPformer, an end-to-end transformer that jointly predicts bi-manual hand and object pose in a single forward pass.
- We test HOPformer on the in-lab 3D dataset, ARCTIC, as well as our newly introduced EPIC-Contact dataset.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30598v1)
- [arXiv](https://arxiv.org/abs/2606.30598v1)

---

<a id='2606.30575v1'></a>
## [MOAR Planner: Multi-Objective and Adaptive Risk-Aware Path Planning for Infrastructure Inspection with a UAV](https://arxiv.org/abs/2606.30575v1)

**Authors:** Louis Petit, Alexis Lussier Desbiens

**Published:** 2026-06-29

**Categories:** cs.RO

**Abstract:**

The problem of autonomous navigation for UAV inspection remains challenging as it requires effectively navigating in close proximity to obstacles, while accounting for dynamic risk factors such as weather conditions, communication reliability, and battery autonomy. This paper introduces the MOAR path planner which addresses the complexities of evolving risks during missions. It offers real-time trajectory adaptation while concurrently optimizing safety, time, and energy. The planner employs a risk-aware cost function that integrates pre-computed cost maps, the new concepts of damage and insertion costs, and an adaptive speed planning framework. With that, the optimal path is searched in a graph using a discrete representation of the state and action spaces. The method is evaluated through simulations and real-world flight tests. The results show the capability to generate real-time trajectories spanning a broad range of evaluation metrics: around 90% of the range occupied by popular algorithms. The proposed framework contributes by enabling UAVs to navigate more autonomously and reliably in critical missions.

**Analysis:**

以下是针对《MOAR Planner: Multi-Objective and Adaptive Risk-Aware Path Planning for Infrastructure Inspection with a UAV》的深度分析：

### 1. 摘要翻译
无人机自主巡检需要在避障的同时处理动态风险（如气象、通信可靠性、电池续航）。本文提出了MOAR路径规划器，以应对任务中演变的复杂风险。它在优化安全性、时间和能源的同时，提供实时轨迹自适应。该规划器采用一种风险感知代价函数，集成了预计算代价图、新增的“损害”与“插入”成本，以及自适应速度规划框架。通过在离散状态空间图中搜索，该方法在仿真和真实飞行测试中表现出良好的实时适应性，其轨迹指标覆盖了主流算法约90%的范围。

### 2. 方法动机分析
*   **驱动力**：无人机在基础设施巡检中，面临的环境风险（如阵风、GPS丢失、电量损耗）是随时间动态变化的，现有的多目标规划（MOPP）缺乏实时的轨迹演变能力。
*   **现有痛点**：现有方法通常只能提供静态最优解，无法在任务过程中根据环境实时调整权重以改变导航偏好（如由“快速完成”转为“安全避障”）。此外，对“碰撞后果”评估不足，缺乏对结构体内部风险的量化。
*   **核心直觉**：通过实时感知任务风险指标（风、电池、通信等），动态调节代价函数中安全性、时间、能耗的权重系数，从而实现轨迹在“安全”与“高效”之间的平滑演变。

### 3. 方法设计详解
*   **流程总结**：
    1.  **环境表征**：将3D空间简化为垂直平面，利用LiDAR点云通过RANSAC进行分割，构建离散化状态图（x, y, θ）。
    2.  **代价建模**：定义 $J = k_S \cdot SC + k_T \cdot TC + k_E \cdot EC$。
        *   **安全代价(SC)**：引入“碰撞代价”（维持最小距离）、“凸包代价”（偏好远离资产）和“损害代价”（惩罚靠近结构体上方的危险区）。
        *   **时间/能量代价(TC, EC)**：利用变量速度映射函数 $v(d_{obs}, \theta)$，根据距离和航向决定巡航速度，能耗则与速度变化率挂钩。
    3.  **实时调整**：利用任务风险指标（WR, CR, LR, BR）实时重计算权重系数 $k_i$。
    4.  **路径搜索**：在离散图上使用A*（巡检）或Dijkstra（撤离）搜索最优轨迹。
*   **关键公式**：$C(d)$ 函数借鉴了人工势场法，通过缩放因子 $\alpha$ 在距离阈值内产生非线性的代价惩罚，确保轨迹平滑过渡。

### 4. 方法对比分析
*   **本质区别**：MOAR并非单纯寻找一条静态的最优路径，而是将环境风险映射为代价函数的“权重动态调整因子”，实现了路径搜索算法的“闭环响应”。
*   **创新贡献**：引入了损害代价（Damage Cost）和插入代价（Insertion Cost），针对工业巡检特点优化了结构接近度评估；提出了基于任务风险的自动化系数调参框架。
*   **适用场景**：高风险、高价值资产的基础设施巡检（如输电线、桥梁），以及需要根据电池状态和天气改变飞行策略的场景。

### 5. 实验分析
*   **验证方法**：在Gazebo仿真环境中使用Hydro-Quebec提供的电力线模型进行对比实验，并在真实LiDAR装载的LineDrone上进行实机验证。
*   **关键结果**：在30x20m的环境中，计算耗时仅为0.01s，满足实时性要求。轨迹表现不仅覆盖了从“最安全”到“最快”的性能区间，且在真实飞行测试中，UAV在复杂障碍物下的 clearance 表现稳定。
*   **局限**：目前的简化模型假设垂直平面导航，在极度复杂的非结构化3D场景下可能需要进一步扩展维度。

### 6. 实用指南
*   **开源情况**：作者提供了论文相关的演示网站 (`edu.louispetit.be`)，核心框架基于ROS实现。
*   **实现细节**：权重调整公式中的风险因子（WR, CR等）需根据具体平台传感器数据流（如GPS DOP值、电流计读数）进行 empiric（经验性）归一化。
*   **迁移可能**：该框架的“风险驱动权重调整”思想极易迁移到其他任何多目标优化任务中，只需替换特定任务的代价图（Cost Map）即可。

### 7. 总结
*   **核心思想**：基于实时风险感知的动态代价函数多目标路径规划。
*   **速记版pipeline**：
    1. 计算环境风险参数。
    2. 根据风险动态调整优化权重。
    3. 查询代价地图与速度映射。
    4. 在离散化图上执行A*搜索。
    5. 实时输出最优轨迹。

**Key Findings:**

- The planner employs a risk-aware cost function that integrates pre-computed cost maps, the new concepts of damage and insertion costs, and an adaptive speed planning framework.
- The results show the capability to generate real-time trajectories spanning a broad range of evaluation metrics: around 90% of the range occupied by popular algorithms.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30575v1)
- [arXiv](https://arxiv.org/abs/2606.30575v1)

---

<a id='2606.30557v1'></a>
## [EcoVideo: Entropy-Orchestrated Video Generation Paradigm in Cloud-Edge Dynamics](https://arxiv.org/abs/2606.30557v1)

**Authors:** Jiayu Chen, Hengyi Zhang, Maoliang Li, Minyu Li, Zihao Zheng, Xuanzhe Liu, Guojie Luo, Xiang Chen

**Published:** 2026-06-29

**Categories:** cs.CV

**Abstract:**

DiT video generation is latency-intensive due to iterative full-frame denoising, while prior cloud-edge methods largely rely on static inter-step decoupling and cannot leverage inter-frame similarity or adapt to system dynamics. We propose EcoVideo, an entropy-orchestrated framework for dynamic inter-frame decoupling: early-stage self-attention entropy provides a training-free estimate of frame-wise information density for frame selection; a cloud large model denoises sparse high-entropy keyframes; and an edge lightweight model reconstructs the remaining frames via motion-aware interpolation with refinement for temporal stability. EcoVideo further adapts the keyframe budget and edge refinement depth to real-time bandwidth and compute availability, optimizing end-to-end latency under constraints. Experiments on representative DiT video generators show improved quality--efficiency trade-offs and up to 2.9x end-to-end speedup in low-bandwidth, compute-limited edge settings. Code is available at https://github.com/IF-LAB-PKU/EcoVideo.

**Analysis:**

以下是对 EcoVideo 论文的深入分析：

### 1. 摘要翻译
DiT（扩散Transformer）视频生成因需要迭代式全帧去噪而导致延迟极高。现有的云边协作方法大多依赖于静态的“步间解耦”（inter-step decoupling），无法利用帧间相似性或适应系统动态。我们提出 **EcoVideo**，这是一个用于动态“帧间解耦”（inter-frame decoupling）的熵编排框架。该框架利用早期的自注意力熵作为训练无关的信号，估算帧级信息密度并进行帧选择；云端大模型仅对稀疏的高熵关键帧进行去噪；边缘侧轻量级模型通过具备细化能力的运动感知插值来重构其余帧，以确保时间稳定性。此外，EcoVideo 还能根据实时带宽和算力资源动态调整关键帧预算和边缘细化深度，在约束条件下优化端到端延迟。在代表性 DiT 视频生成器上的实验表明，该方法在低带宽、受限算力的边缘环境下实现了高达 2.9 倍的端到端加速。

### 2. 方法动机分析
- **驱动力**：旨在解决分布式云边计算中视频生成的高延迟与算力限制瓶颈。
- **现有方法痛点**：现有“步间解耦”（如 HybridSD）假设所有帧同等重要，导致云端对每一帧都进行重复的高昂去噪，且边缘小模型能力不足，易产生纹理闪烁和细节崩溃等时间不一致问题。
- **研究假设**：视频序列在时序上存在信息密度差异，可以通过注意力熵识别高密度关键帧，并通过云边协同与帧间插值实现更高效的生成。

### 3. 方法设计详解
- **核心流程**：
    1. **熵分析（Warm-up）**：在前 10% 的去噪步骤计算 token 级注意力熵，经指数移动平均（EMA）聚合得到稳定的帧级熵。
    2. **关键帧选择**：根据熵序列，保留首尾帧并按熵值选取 Top-K 关键帧，由云端大模型进行去噪，其余帧视为“非关键帧”。
    3. **上下文注入**：为了保持全局一致性，在云端 denoising 时，将非关键帧的潜在表示（Latents）作为 frozen context 注入，引导生成。
    4. **动态插值（EcoVFI）**：边缘侧利用贪婪策略进行插值。通过计算运动、结构（DINO特征）、纹理（像素差异）等四个维度指标，为最“困难”的区间分配更多的插值步数，缓解 ghosting 和闪烁。
- **模型结构**：云端（大模型）+ 边缘（轻量插值模型 EcoVFI）。
- **关键算法**：通过 sigmoid 映射后的加权难度分数 $g(I_a, I_b)$ 实现自适应插值，让模型更关注“难处理”的帧间变化。

### 4. 方法对比分析
- **本质区别**：从“步间解耦”（切分去噪步骤）转向“帧间解耦”（切分生成帧内容），利用时序冗余大幅减少云端计算压力。
- **创新贡献**：提出训练无关的注意力熵分析作为调度信号，设计了动态平衡算力与带宽的自适应配置搜索机制。
- **适用场景**：实时视频生成服务、带宽受限的移动端或边缘计算设备。

### 5. 实验分析
- **验证方法**：在 Wan2.1/2.2 和 CogVideoX 上进行实测，对比 HybridSD 和 EC-Diff。
- **关键结果**：在 Wan2.1-14B 上将通信量从 17.23MB 降至 1.10MB，边缘延迟降至原有的 1/6，整体端到端提速明显。
- **主要优势**：极好地平衡了生成质量（VBench指标）与实时性，显著改善了时间一致性。
- **主要局限**：在极端剧烈运动或快速场景切换时，边缘插值模型仍面临挑战。

### 6. 实用指南
- **开源情况**：已开源（见论文链接）。
- **实现细节**：关键参数为 warm-up 比例（建议10%）和候选集 $\{S_K, S_D\}$ 的搜索空间大小。
- **迁移可能**：该框架核心是“基于重要性权重的 workload 划分”，可直接迁移至其他基于 Diffusion 的生成任务（如音频、长序列时间序列预测）。

### 7. 总结
- **核心思想**：通过熵感知实现关键帧云端生成与边缘动态插值的最优调度。
- **速记版pipeline**：
    1. 预热期测量每一帧的重要程度（熵）。
    2. 云端只画重点帧，并提供全局概览。
    3. 边缘自动识别画面差异，重点填补变化剧烈的帧。
    4. 根据网络速度，实时调整云端干活的量。

**Key Findings:**

- We propose EcoVideo, an entropy-orchestrated framework for dynamic inter-frame decoupling: early-stage self-attention entropy provides a training-free estimate of frame-wise information density for frame selection; a cloud large model denoises sparse high-entropy keyframes; and an edge lightweight model reconstructs the remaining frames via motion-aware interpolation with refinement for temporal stability.

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30557v1)
- [arXiv](https://arxiv.org/abs/2606.30557v1)

---

<a id='2606.30552v1'></a>
## [Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision](https://arxiv.org/abs/2606.30552v1)

**Authors:** Haoyang Li, Guanlin Li, Youhe Feng, Chen Zhao, Zhuoran Wang, Yang Li, Qizhe Wei, Shifeng Bao, Haitao Shen, Yihan Zhao, Tong Yang, Jing Zhang

**Published:** 2026-06-29

**Categories:** cs.RO, cs.CV

**Abstract:**

Cross-embodiment transfer in vision-language-action (VLA) models remains challenging because low-level state and action spaces differ fundamentally across robot platforms. We observe that the high-level cognitive process underlying manipulation, including scene perception, object identification, task planning, and sub-task decomposition, is largely shared across embodiments. Based on this observation, we present ZR-0, a 2.6 billion parameter end-to-end VLA model that uses dense Embodied Chain-of-Thought (ECoT) supervision to align cross-embodiment representations within the vision-language model (VLM). ZR-0 adopts a dual-stream architecture: a pre-trained VLM (System 2) generates structured ECoT reasoning during training, while a Diffusion Transformer-based action expert (System 1) produces continuous action chunks via flow matching. The two components are coupled through cross-attention, with an attention mask that restricts the action expert to input prompt features only, enabling ECoT generation to be entirely skipped at inference without any performance loss. ZR-0 is pre-trained on ProcCorpus-60M, a large-scale dataset comprising approximately 60 million frames (approximately 1,000 hours) from over 400K trajectories, with dense ECoT annotations covering 96.8% of all frames. We evaluate ZR-0 on three simulation benchmarks spanning single-arm (LIBERO), bimanual (RoboTwin 2.0), and humanoid (RoboCasa GR-1 Tabletop) embodiments, as well as real-world experiments on the xArm platform, demonstrating strong performance across all settings. Code and model checkpoints are available at https://github.com/RUCKBReasoning/ZR-0.

**Analysis:**

以下是对论文《Training Vision-Language-Action Models with Dense Embodied Chain-of-Thought Supervision》的深度分析：

### 1. 摘要翻译
跨具身迁移在视觉-语言-动作（VLA）模型中仍具挑战，因为不同机器人平台的低级状态和动作空间存在根本差异。我们观察到，操作背后的高级认知过程（场景感知、物体识别、任务规划、子任务分解）在不同具身间是高度共享的。基于此，我们提出了 **ZR-0**，一个26亿参数的端到端VLA模型，它利用密集的具身思维链（ECoT）监督，在视觉-语言模型（VLM）内部对齐跨具身表示。ZR-0采用双流架构：预训练的VLM（系统2）在训练期间生成结构化的ECoT推理，而基于扩散Transformer的动作专家（系统1）通过流匹配生成连续动作块。两者通过交叉注意力耦合，并施加掩码限制动作专家仅关注输入提示特征，使ECoT生成在推理阶段可完全跳过且无性能损失。ZR-0在ProcCorpus-60M数据集上预训练，在三个模拟基准（LIBERO, RoboTwin 2.0, RoboCasa）和真实xArm平台上展现了强劲的跨具身迁移性能。代码和模型已开源。

### 2. 方法动机分析
*   **驱动力**：旨在解决VLA模型在面对异构机器人（如不同DoF、不同运动学结构）时，难以学习通用“物理常识”和“语义对齐”的难题。
*   **现有痛点**：以往方法多依赖格式对齐（补零、标准化），未能解决**语义对齐**问题。即不同机器人的同一动作维度在物理含义上不同，单纯的动作拟合无法泛化。
*   **研究假设**：虽然低级动作具身特定，但任务的高级认知流程（感知→规划→分解）是“具身无关”的，通过ECoT监督可以强制VLM学习这些通用的表示。

### 3. 方法设计详解
*   **双流架构**：
    *   **System 2 (VLM)**：基于Qwen3-VL-2B。输入指令和多视角图像，输出ECoT序列（场景描述、进度、未来计划、子任务分解、目标物体坐标、动作token）。该部分提供高层语义对齐。
    *   **System 1 (Action Expert)**：基于Diffusion Transformer (DiT)。接收VLM特征和机器人状态，通过流匹配预测$H$步连续动作块。
*   **关键机制**：**推理优化**。训练时使用ECoT监督，但通过**交叉注意力掩码（Attention Mask）**，限制动作专家仅 attending 到VLM的输入提示（图像+指令），从而实现推理时彻底丢弃ECoT文本生成，仅需VLM一次前向传递，极大降低了延迟。
*   **ECoT构成**：包含“To-Do Actions”，将复杂动作分解为“动词+物体+方位”的原子子任务，通过这种通用的自然语言表示实现了跨机器人的语义对齐。

### 4. 方法对比分析
*   **本质区别**：与传统将ECoT作为推理辅助不同，ZR-0将ECoT视为一种**特征提取的隐式监督手段**。通过掩码设计，实现了“训练有逻辑、推理零负担”。
*   **创新点**：提出了基于DiT的动作专家与VLM的高效耦合方式，并证明了在推理时完全移除ECoT文本生成并不会导致语义对齐信息的丢失。
*   **适用场景**：适用于需要处理多种形态机器人、具有长周期任务序列的具身智能应用。

### 5. 实验分析（精简版）
*   **关键结论**：在LIBERO-10（长周期任务）上达到96.4%，大幅优于现有方法；在RoboCasa上以69.3%的SR领先第二名6.1个点。
*   **主要优势**：极强的跨具身迁移能力；推理效率高（90ms/chunk）；通过ECoT有效缓解了动作预测中的语义模糊。
*   **主要局限**：在涉及多阶段复杂接触交互的任务（如开关柜子）上，由于预训练数据中此类行为较少，表现略弱。

### 6. 实用指南
*   **开源信息**：[GitHub: RUCKBReasoning/ZR-0](https://github.com/RUCKBReasoning/ZR-0)
*   **迁移建议**：若需迁移到新机器人，重点在于构建高质量的ECoT标注。利用现有的VLM（如GPT-4o等）针对目标机器人数据进行预标注，是复现该方法的关键途径。
*   **训练细节**：学习率采用了余弦调度，且在预训练时混合了通用视觉-语言数据（如CapsFusion），这对于防止模型在机器人数据微调中出现灾难性遗忘至关重要。

### 7. 总结
*   **核心思想**：利用具身思维链（ECoT）作为语义桥梁，实现跨具身表示对齐，且推理时无额外开销。
*   **速记版pipeline**：
    1. 输入指令与视觉观测；
    2. VLM编码并生成高层推理逻辑（训练时）；
    3. 动作专家通过交叉注意力获取VLM语义特征；
    4. 动作专家通过流匹配输出连续动作轨迹。

**Key Findings:**

- Based on this observation, we present ZR-0, a 2.6 billion parameter end-to-end VLA model that uses dense Embodied Chain-of-Thought (ECoT) supervision to align cross-embodiment representations within the vision-language model (VLM).

**Links:**

- [PDF](https://arxiv.org/pdf/2606.30552v1)
- [arXiv](https://arxiv.org/abs/2606.30552v1)

---

