time: 20260707

# Arxiv Computer Vision Papers - 2026-07-07

## Executive Summary

## 每日报告执行摘要：2026年7月6日 Arxiv 计算机视觉论文

### 1. 主要主题与趋势

本期10篇论文呈现三大核心趋势：
- **机器人操作与具身智能**（6篇）：围绕灵巧抓取、视触觉感知、长时操控和多智能体协作展开，强调从仿真到真实的零样本迁移（sim-to-real）以及无监督自学习。
- **相机与三维感知**（4篇）：关注动态相机内参估计、流式三维重建的可靠性校准，以及从固定视角到自由相机的泛化能力。
- **生成模型控制**（2篇）：聚焦扩散模型中的概念精确移除和智能体视觉生成中知识边界的动态演化。

### 2. 特别重要或创新的论文

- **《Closing the Reality Gap》**（论文1）：提出零样本从仿真到真实部署的灵巧力控抓取方法，在无需真实数据微调的情况下实现了跨域鲁棒操作，为机器人行业落地提供了关键突破。
- **《From Fixed to Free Cameras》**（论文2）：首个免标定的视角鲁棒视觉-语言-动作（VLA）模型，摆脱了对固定相机标定的依赖，显著提升了具身智能体的环境适应性。
- **《Deform360》**（论文3）：发布大规模多视角视触觉数据集，覆盖可变形物体的360度触觉与视觉融合信息，为变形世界模型的训练奠定了数据基础。
- **《Search Beyond What Can Be Taught》**（论文5）：引入进化算法动态拓展智能体视觉生成中的知识边界，突破了传统监督学习的预设局限，开辟了“可教范围之外”的生成能力探索新方向。
- **《Erasing Without Collateral Damage》**（论文9）：提出精确概念移除方法，在不影响其他生成内容的前提下从扩散模型中擦除特定概念（如侵权风格），解决了生成模型安全性与可用性之间的核心矛盾。

### 3. 新兴研究方向与技术

- **视触觉驱动的变形世界模型**：结合多视角视觉和触觉信号（论文3）用于可变形物体建模，将推动机器人精细操作与柔性物体交互的研究。
- **流式三维重建的可靠性感知学习率**（论文8）：针对在线重建中数据非平稳特性设计的校准机制，有望成为实时SLAM和动态场景理解的标准组件。
- **多智能体自学习与自动化**（论文7）：将图策略与自学习结合用于变分自动化任务，展示了群体智能在工业场景中的潜力。
- **频率-空间域协同的小目标检测**（论文10）：融合频域与空间域信息的DETR变体，为小目标检测提供了一种新的多尺度表征范式。

### 4. 建议全文阅读的论文

- **论文1**（零样本灵巧抓取）：对于机器人领域研究者，这是实现低成本大规模部署的关键技术。
- **论文2**（免标定VLA模型）：具身智能与视觉语言模型交叉方向的必读之作，打破相机设置限制。
- **论文5**（智能体生成知识边界演化）：生成式AI前沿探索，对理解模型创造力边界有启发意义。
- **论文9**（精确概念移除）：关乎扩散模型的安全可控应用，适合模型微调与伦理研究的读者。
- **论文3**（大规模视触觉数据集）：从事多模态感知与变形体建模的研究者应深入阅读，数据价值极高。

---

**总结**：本期论文体现了计算机视觉向“泛化性、鲁棒性、安全性”协同发展的趋势，机器人具身智能与生成模型的精确控制是当前最活跃的两大方向。

---

## Table of Contents

1. [Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation](#2607.04940v1)
2. [From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model](#2607.05396v1)
3. [Deform360: A Massive Multi-view Visuotactile Dataset for Deformable World Models](#2607.05390v1)
4. [InFlux++: Real and Synthetic Data for Estimating Dynamic Camera Intrinsics](#2607.05389v1)
5. [Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation](#2607.05382v1)
6. [Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation](#2607.05377v1)
7. [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](#2607.05369v1)
8. [ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction](#2607.05356v1)
9. [Erasing Without Collateral Damage: Precise Concept Removal in Diffusion Models](#2607.05274v1)
10. [FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection](#2607.05176v1)

---

## Papers

<a id='2607.04940v1'></a>
## [Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation](https://arxiv.org/abs/2607.04940v1)

**Authors:** Zhe Zhao, Zhibin Li, Yilin Ou, Mengshi Qi

**Published:** 2026-07-06

**Categories:** cs.RO

**Abstract:**

Human-like dexterous hands with multiple fingers offer human-level manipulation capabilities but remain difficult to train the control policies that can deploy on real hardware due to contact-rich physics and imperfect actuation. We present a sim-to-real reinforcement learning method that leverages dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions. To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling with randomization to account for non-ideal torque-speed effects and bridge the actuation gaps. Using an asymmetric actor-critic PPO pipeline, we train policies entirely in simulation and deploy them directly to a five-finger hand. The resulting policies demonstrate two essential human-hand skills: (1) command-based controllable grasp force tracking and (2) reorientation of objects in the hand, both of which are robustly executed without fine-tuning on the robot. By combining tactile and torque in the observation space with scalable sensing and actuation modeling, our system provides a practical solution to achieve reliable dexterous manipulation. To our knowledge, this is the first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot on real hardware.

**Analysis:**

### 1. 摘要翻译
人类灵巧手具备类人操纵能力，但因接触丰富的物理特性和不完美的驱动，难以训练出可部署在真实硬件上的控制策略。我们提出了一种强化学习（RL）框架，通过利用密集触觉反馈与关节力矩感知来显式调节物理交互。为实现有效的虚实迁移，我们引入了：(i) 一种计算快速的触觉模拟技术，通过并行正向运动学计算密集虚拟触觉单元与物体间的距离，提供高频、高分辨率的触觉信号；(ii) 一种电流-力矩标定方法，通过映射电机电流到关节力矩，消除了对昂贵力矩传感器的需求；(iii) 带有随机化的执行器动力学建模，以应对非理想的力矩-速度效应并弥补驱动差距。使用非对称Actor-Critic PPO流水线，我们在模拟中训练策略并直接部署到五指灵巧手。实验结果证明该策略具备两种核心技能：(1) 指令驱动的可控抓取力跟踪；(2) 手内物体旋转，且无需在真机上进行微调。该系统为实现可靠的灵巧操纵提供了实用方案，是首次在多指灵巧手上实现模拟训练并零样本迁移至真实硬件的可控抓取演示。

### 2. 方法动机分析
*   **驱动力**：解决灵巧手在复杂接触环境下的“虚实鸿沟”（Sim-to-Real Gap），实现高精度的力控制任务。
*   **现有痛点**：
    1.  **触觉仿真计算压力**：传统高分辨率触觉仿真过于缓慢，限制了RL大规模探索。
    2.  **驱动不一致**：大多数灵巧手缺乏力矩传感器，仅有电流反馈，导致模拟器与真实动力学不匹配。
*   **研究假设**：通过融合触觉与电流感知的全状态反馈，结合计算高效的几何近似仿真与执行器参数随机化，可以有效弥合虚实差距。

### 3. 方法设计详解
*   **触觉仿真 (Parallel Forward Kinematics)**：摒弃了计算昂贵的物理碰撞引擎，利用并行正向运动学，计算灵巧手上数千个虚拟触觉单元（Tacxiel）到目标物体表面的欧氏距离，通过点积检测（$v_{oij} \cdot n_o < 0$）判定接触。
*   **电流-力矩标定**：拟合电机电流与输出力矩的非线性映射（拟合曲线），将电流作为力矩的观测输入，从而在无力矩传感器的硬件上获取力反馈。
*   **执行器模型随机化**：在模拟中加入反向间隙（Backlash）、力矩-速度饱和效应（基于DC电机特性建模）以及随机化的摩擦与负载补偿因子（$\eta$），提高策略的鲁棒性。
*   **动作空间与策略**：采用非对称Actor-Critic结构，策略输入包含触觉中心、接触力、关节状态和指令力/目标位姿。

### 4. 方法对比分析
*   **本质区别**：与视觉主导的方法不同，本方法强调触觉的高分辨率与力矩的隐式建模。
*   **创新贡献**：提出了一种基于距离场的触觉模拟器，使其在保持高分辨率的前提下，具备极高的计算采样效率。
*   **适用场景**：适用于具备电流反馈的商业灵巧手（如XHand），尤其适合需要接触力精确控制的任务。

### 5. 实验分析
*   **验证方法**：在真实XHand硬件上进行抓取力跟踪测试及手内旋转测试，并进行消融实验。
*   **关键结论**：触觉反馈、接触力加权（Force-weighted）和6D位姿表示是成功的关键；缺乏触觉的基线模型在真机上几乎无法完成任务。
*   **优势**：零样本迁移，无需真机fine-tune；抗干扰能力强。
*   **局限**：目前的实验主要针对特定几何形态的物体，对高度不规则或变性物体的泛化仍需进一步验证。

### 6. 实用指南
*   **开源情况**：基于IsaacLab环境。
*   **实现细节**：关键在于电流-力矩映射的离线校准和PD控制器的随机化参数（$\eta$, $\tau_0$, $q_{max}$）配置。
*   **迁移**：该方法论（高效几何触觉仿真+执行器参数随机化）可直接迁移至其他具有高自由度机器人末端的力控制任务中。

### 7. 总结
*   **核心思想**：高效几何感知与物理随机化驱动的虚实零样本迁移。
*   **速记版Pipeline**：
    1.  **触觉建模**：用距离计算替代碰撞引擎。
    2.  **驱动校准**：通过电流估算电机力矩。
    3.  **动力学随机化**：模拟真实电机反向间隙与饱和。
    4.  **端到端训练**：使用强化学习学习全状态策略。

**Key Findings:**

- We present a sim-to-real reinforcement learning method that leverages dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions.
- To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling with randomization to account for non-ideal torque-speed effects and bridge the actuation gaps.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.04940v1)
- [arXiv](https://arxiv.org/abs/2607.04940v1)

---

<a id='2607.05396v1'></a>
## [From Fixed to Free Cameras: Calibration-Free View-Robust Vision-Language-Action Model](https://arxiv.org/abs/2607.05396v1)

**Authors:** Wenhao Li, Xueying Jiang, Quanhao Qian, Deli Zhao, Shijian Lu, Gongjie Zhang, Ran Xu

**Published:** 2026-07-06

**Categories:** cs.CV, cs.AI, cs.LG, cs.RO

**Abstract:**

Real-world robot deployment rarely maintains the training-stage camera setup, where cameras often experience repositioning or remounting depending on actual scenarios. Existing view-robust Vision-Language-Action (VLA) policies tolerate such camera variations only when the camera extrinsics are explicitly provided, making them fragile and hard to use especially when view robustness is critical. We argue that the policy should not be told where the camera is, but rather figure it out by itself. To this end, we introduce Camera-Centric VLA (CamVLA), a new VLA model that decouples manipulation controls from camera geometry by predicting (i) a camera-centric end-effector action expressed in the local camera frame, and (ii) a 6-DoF hand-eye matrix relating cameras to the robot base. A deterministic geometric transformation composes the two predictions into a robot base-frame action. This disentangles how I should move in pose-independent camera-centric action generation from where I am looking from in camera-perspective geometric grounding. The resulting policy is calibration-free, depth-free, and single-view, requiring only a single monocular RGB image as the visual observation and task instruction at deployment. Evaluations in both simulation and real-world robot data show that CamVLA consistently improves success rates across diverse unseen viewpoints. Project page: https://alibaba-damo-academy.github.io/CamVLA/.

**Analysis:**

# 论文方法分析与总结：CamVLA

### 1. 摘要翻译
真实世界的机器人部署很少能维持训练时的摄像机设置，摄像机常因场景需求被移动或重装。现有的视点鲁棒性视觉-语言-动作（VLA）策略仅在显式提供相机外参时才能容忍此类变化，这在视点鲁棒性至关重要时显得脆弱且难以使用。我们认为，机器人策略不应被告知摄像机的位置，而应自行推断。为此，我们引入了**相机中心化VLA（CamVLA）**，这是一种通过预测以下两点来解耦操作控制与相机几何的模型：（i）在局部相机坐标系中表达的相机中心动作，以及（ii）关联相机与机器人基座的6自由度手眼矩阵。通过确定的几何变换将两者合成为基座坐标系动作。这种方法将“我该如何移动”的姿态无关动作生成与“我从哪里观察”的相机视角几何定位解耦，实现了校准自由、无需深度、单目视觉下的鲁棒操作。

### 2. 方法动机分析
- **核心动机**：旨在解决传统VLA策略对于相机视点偏移的极端脆弱性，摆脱对预先校准外参的依赖。
- **现有方法痛点**：传统方法隐式地将手眼变换耦合在权重中，强制模型记忆视点依赖的坐标映射。现有“视点鲁棒”方法大多依赖外部获取精准相机参数（内参/外参），在相机位置漂移或动态更换时极易失效。
- **研究假设**：通过将“动作生成”与“几何定位”解耦，使动作在局部相机框架下产生，机器人能够像人类一样，通过感知环境同时自适应地推断观察视角，从而实现跨视点的零样本泛化。

### 3. 方法设计详解
- **流程总结**：
  1. **输入阶段**：接收单目RGB图像、本体状态和自然语言指令。
  2. **双头并行预测**：VLM骨干网络提取特征后，分为两个头：
     - **Action Head**：预测在相机坐标系下的增量动作 $\Delta A_{c,t}$（包含平移、旋转及抓手状态）。
     - **Geometric Head**：回归出当前的6自由度手眼矩阵 $T_t$（相机到机器人基座的变换）。
  3. **几何融合**：通过确定性公式 $\Delta p_{b,t} = R_t \Delta p_{c,t}$ 和 $\Delta r_{b,t} = R_t \Delta r_{c,t}$，利用预测出的旋转矩阵 $R_t$ 将动作转换到基座坐标系。
- **模型结构**：Auxiliary Geometric Head 采用了轻量级的三层MLP，作用于视觉特征。该设计仅增加了约0.19%的参数量和1ms的推理延迟。
- **算法精髓**：**平移不变性**。数学推导证明了相对平移量的转换仅依赖旋转矩阵 $R_t$ 而与平移向量 $\tau_t$ 无关，这使得即使几何头预测出的绝对位置有一定漂移，只要旋转预测准确，动作输出依然稳健。

### 4. 方法对比分析
- **本质区别**：从“端到端隐式记忆”转变为“显式动作生成与几何变换解耦”。
- **创新点**：提出了一种无需外部校准的自定位机制，将视点鲁棒性问题转化为简单的几何回归任务。
- **适用场景**：适用于工业、家庭等相机位置可能因振动、更换或移动而频繁变化的非结构化场景。

### 5. 实验分析
- **关键结果**：在RLBench和实机实验中，CamVLA在未见视角下成功率显著提升（例如：$\pi_0$ 基础模型在15度偏移下从6.3%提升至29.3%）。
- **主要优势**：极强的零样本泛化能力；无需任何额外的校准流程（Calibration-free）；对自我定位误差具有良好的容忍度。
- **主要局限**：对极端的视点变化（超出训练分布范围）仍会因为视觉特征偏移而导致性能下降；对高精度精细操作任务仍有提升空间。

### 6. 实用指南
- **开源情况**：官方提供了项目主页（https://alibaba-damo-academy.github.io/CamVLA/）。
- **实现细节**：
  - 训练目标为 $L = L_{act} + \lambda L_{ext}$，其中 $\lambda$ 建议设为0.1。
  - 使用了Image Encoder特征而非深度层语义特征作为几何头输入，以保证空间定位的稳定性。
- **迁移建议**：可将该解耦结构轻松嵌入到现有的RT-1、Octo或$\pi_0$等各类VLA架构中，作为增强视点鲁棒性的“插件”。

### 7. 总结
- **核心思想**：动作与观察视角解耦，通过自推断几何实现视点鲁棒。
- **速记版pipeline**：
  1. 提取单张RGB图像的特征。
  2. 分别预测局部动作和当前相机视角。
  3. 利用几何变换将局部动作转换至机器人基座空间。
  4. 丢弃显式校准需求，实现动态鲁棒执行。

**Key Findings:**

- To this end, we introduce Camera-Centric VLA (CamVLA), a new VLA model that decouples manipulation controls from camera geometry by predicting (i) a camera-centric end-effector action expressed in the local camera frame, and (ii) a 6-DoF hand-eye matrix relating cameras to the robot base.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05396v1)
- [arXiv](https://arxiv.org/abs/2607.05396v1)

---

<a id='2607.05390v1'></a>
## [Deform360: A Massive Multi-view Visuotactile Dataset for Deformable World Models](https://arxiv.org/abs/2607.05390v1)

**Authors:** Hongyu Li, Wanjia Fu, Xiaoyan Cong, Zekun Li, Binghao Huang, Hanxiao Jiang, Xintong He, Yiqing Liang, Rao Fu, Tao Lu, Srinath Sridhar, Kevin A. Smith, George Konidaris, Yunzhu Li

**Published:** 2026-07-06

**Categories:** cs.RO, cs.CV

**Abstract:**

Predicting object dynamics (i.e., world modeling) is a fundamental challenge for robotic manipulation, and modeling deformable objects presents a particularly difficult case due to their high-dimensional state spaces and complex material properties. While current world models approach this through two distinct paradigms: learning the dynamics over the 2D pixel space or more explicit 3D geometric space. A systematic understanding of their relative strengths and limitations remains elusive due to the lack of diverse, large-scale real-world data. To address this, we present Deform360, a large-scale visuotactile dataset featuring 198 daily-life objects, 1,980 interaction sequences, and over 215 hours of observations from 41 surround-view cameras and bimanual tactile grippers to capture both global motion and contact-induced local deformations. Leveraging a novel markerless visuotactile 3D tracking pipeline to extract dense geometry and motion, we systematically evaluate current state-of-the-art world models, comparing 2D video models against 3D particle models. Finally, we provide a preliminary demonstration indicating the real-world applicability of our dataset by performing robot planning tasks on deformable objects. Our analysis reveals key insights into the trade-offs between structural priors and scalability, providing a solid benchmark for future research in generalizable deformable object-centric world modeling. Project website: https://deform360.lhy.xyz

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **Deform360** 的论文摘要分析如下：

### 1. 主要贡献总结
Deform360 填补了机器人操作领域在可变形物体交互数据上的巨大空白，提供了一个大规模、高质量、多模态（视觉+触觉）的基准数据集。该研究通过对比 2D 视频生成模型与 3D 粒子动力学模型，系统性地揭示了不同世界模型范式在处理复杂形变物体时的核心权衡与局限，为未来通用机器人世界模型的发展奠定了基石。

### 2. 关键创新与方法论
*   **多模态感知布局**：该数据集不仅包含 41 个环绕视角的高清视觉数据，还结合了双臂触觉感知，这是理解物体全局动态与局部接触物理属性的关键。
*   **无标记视觉-触觉 3D 跟踪管道（Markerless Visuotactile 3D Tracking Pipeline）**：这是本文的技术核心。作者开发了一套自动提取密集几何与运动信息的流程，解决了可变形物体在长序列交互中难以进行精确几何标注的瓶颈。
*   **跨范式基准评估**：不仅限于数据的发布，论文还提出了一个统一的评估框架，将“基于像素的 2D 视频预测”与“基于几何的 3D 动力学建模”两种主流范式置于同一量纲下进行比较。

### 3. 对领域的潜在影响
*   **推动“具身智能”的世界模型演进**：通过提供大规模真实世界数据，它将推动从简单的刚体操作向复杂的形变操作（如布料折叠、揉捏、软体工具使用）转型。
*   **桥接视觉与触觉感知**：以往模型多依赖视觉，该研究证明了触觉数据对于捕捉精细化形变不可或缺，将推动多模态传感器融合算法的进步。
*   **量化研究范式**：它明确了“结构先验（3D模型）”与“可扩展性（2D生成式模型）”之间的性能权衡，为未来架构设计提供了数据驱动的理论支撑。

### 4. 受益的相关领域与应用
*   **机器人操作（Robotic Manipulation）**：直接受益于家居整理、软性材料处理、手术机器人等需要精细接触控制的任务。
*   **物理模拟与渲染（Simulation & Graphics）**：数据集中的真实轨迹和形变模式可用于校准和验证物理仿真引擎的准确性。
*   **生成式世界模型（Generative World Models）**：为视频生成模型（如 Sora 类架构）引入物理几何约束提供训练和测试集，提升模型在复杂物理环境下的时序连贯性。

### 5. 可推断的局限性
*   **计算与硬件复杂度**：依赖 41 个摄像头和双臂触觉传感器，该系统的搭建和数据采集成本极高，普通实验室难以复现。
*   **数据泛化性边界**：虽然涵盖 198 种物体，但相比于无限的现实世界，其多样性仍有限，模型在面对未见过的复杂材料（如粘弹性材料、各向异性材料）时表现仍存疑。
*   **实时性能瓶颈**：所提出的 3D 跟踪管道和世界模型可能涉及巨大的计算量，在实际机器人执行任务时的推理延迟（Latency）可能是一个主要挑战。

---

**专家点评：**
这篇论文的有趣之处在于它**跳出了纯算法改进的圈子，通过“系统工程”解决了 AI 训练中最棘手的“数据缺乏”问题**。在计算机视觉领域，我们常说“数据决定性能的上限”，Deform360 的出现，意味着对于“可变形物体建模”这一硬骨头，学界终于从“小样本玩具实验”转向了“大规模、系统化、多模态”的工业级研究路径。这是一个标志性的学术贡献。

**Key Findings:**

- To address this, we present Deform360, a large-scale visuotactile dataset featuring 198 daily-life objects, 1,980 interaction sequences, and over 215 hours of observations from 41 surround-view cameras and bimanual tactile grippers to capture both global motion and contact-induced local deformations.
- Leveraging a novel markerless visuotactile 3D tracking pipeline to extract dense geometry and motion, we systematically evaluate current state-of-the-art world models, comparing 2D video models against 3D particle models.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05390v1)
- [arXiv](https://arxiv.org/abs/2607.05390v1)

---

<a id='2607.05389v1'></a>
## [InFlux++: Real and Synthetic Data for Estimating Dynamic Camera Intrinsics](https://arxiv.org/abs/2607.05389v1)

**Authors:** Erich Liang, Caleb Kha-Uong, Chinmaya Saran, Sreemanti Dey, David W. Liu, Junhan Ouyang, Benjamin Zhou, Jia Deng

**Published:** 2026-07-06

**Categories:** cs.CV

**Abstract:**

Camera intrinsics are vital for recovering 3D structure from 2D video. However, most 3D algorithms assume fixed intrinsics throughout a video, an assumption that often fails for real-world in-the-wild videos. Consequently, estimating per-frame intrinsics from RGB images is critical for making 3D methods robust to videos with dynamic intrinsics. InFlux previously advanced this research direction by establishing the first real-world benchmark with per-frame ground truth intrinsics for dynamic intrinsics videos. Nevertheless, existing methods remain inaccurate due to two obstacles: (i) training data is scarce and lacks intrinsics diversity; and (ii) benchmarks, including InFlux, have limited scene and camera motion diversity, making it difficult to properly evaluate methods. To address both gaps, we present InFlux++, consisting of two components. InFlux++ Synth is a large-scale procedurally generated synthetic video dataset with 441K+ annotated frames from 1841 high-resolution videos, providing accurate per-frame ground truth intrinsics for training dynamic intrinsics prediction models; a subset also includes per-frame pose, depth, and normals. The videos feature rich intrinsics diversity through changes in camera zoom and focus, as well as dynamic objects and realistic rendering effects such as lens distortion and defocus blur. InFlux++ Real is a large-scale real-world benchmark that extends InFlux with 514K+ newly captured frames across 334 high-resolution videos, spanning a wider range of scenes and camera motions. Finetuning existing intrinsics prediction methods on InFlux++ Synth consistently improves focal length estimation across both InFlux++ Real and InFlux, suggesting that synthetic supervision is promising for RGB-based intrinsics prediction. For the dataset, benchmark, code, videos, submission instructions, and live leaderboard, please visit https://influx.cs.princeton.edu/ .

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对《InFlux++》这篇论文的分析如下：

### 1. 论文核心贡献总结
该论文针对视频中“动态相机内参（Dynamic Camera Intrinsics）”难以估计的问题，推出了大规模数据集 **InFlux++**。该数据集由 **InFlux++ Synth**（大规模合成数据）和 **InFlux++ Real**（大规模真实场景基准）两部分组成，通过提供高质量的逐帧内参标注，有效填补了训练数据匮乏与评估场景多样性不足的空白，推动了从单目RGB视频中精确恢复动态内参的进展。

### 2. 关键创新与方法论
*   **合成数据赋能（Synthetic Supervision）：** 利用过程化生成（Procedural Generation）技术构建了包含 441K+ 帧的合成数据集，涵盖了变焦、对焦、镜头畸变及动态物体等复杂因素。这种“合成监督”策略证明了在缺乏大规模标注真实数据的情况下，通过合成数据预训练或微调模型是提升内参预测精度的有效路径。
*   **真实基准扩展：** 在原有的 InFlux 基础上，通过增加 514K+ 帧的高清真实视频，显著提升了场景与相机运动的多样性。这为解决“in-the-wild”（野外环境）下内参估计的泛化能力问题提供了更严苛的测试平台。
*   **动态内参先验学习：** 将静态相机假设转化为动态处理流程，通过模型学习焦距随时间变化的特征，从而克服了传统 3D 重建中因内参变化导致的误差累积问题。

### 3. 对领域的潜在影响
*   **提升 3D 重建的鲁棒性：** 传统的 SfM（运动结构恢复）和 SLAM（同步定位与建图）算法常因假设内参固定而在变焦镜头视频中失效。InFlux++ 的出现将推动这些经典视觉任务在动态视频中的应用落地。
*   **强化生成式 AI 的几何一致性：** 对于视频生成模型（如 Sora 类模型）而言，保持摄像机参数的几何一致性至关重要，该研究成果可作为几何一致性约束的训练先验。
*   **确立学术标准：** 通过建立统一的 Leaderboard 和评估基准，该工作将有效规范该细分领域的评价体系，促进不同算法间的公平竞争与迭代。

### 4. 相关领域与应用收益
*   **计算摄影（Computational Photography）：** 动态变焦视频的超分辨率、去模糊及防抖算法将直接受益于更准确的动态内参。
*   **视频编辑与特效（VFX）：** 自动化的摄像机追踪（Camera Tracking）和 3D 对象插入，在面对变焦镜头视频时无需繁琐的人工标定。
*   **机器人视觉：** 移动机器人或无人机在变焦拍摄过程中的实时定位与建图精度将得到提升。
*   **虚拟现实（VR/AR）：** 改善视频转三维场景过程中的几何畸变校准。

### 5. 可推断的局限性
*   **领域鸿沟（Domain Gap）：** 尽管合成数据极大提升了模型性能，但真实世界中极其复杂的物理光学效应（如特殊的滤镜、光晕、传感器的非线性响应）仍可能在合成数据中被简化，导致模型在极端场景下表现受限。
*   **计算成本：** 虽然预训练模型提升了性能，但对于大规模合成数据的生成和处理，需要相当规模的计算资源。
*   **算法本质依赖：** 该研究侧重于数据与评估，方法论上仍依赖于现有的深度学习架构，若模型本身对动态内参的回归逻辑存在瓶颈，数据的扩充可能仅能带来边际效应递减的提升。

**专家点评：**
InFlux++ 的独特价值在于它精准地切中了 3D 视觉研究中的一个“痛点”——即**数据驱动的几何估计**。它将问题从单纯的几何推导转变为可以通过大数据训练的学习任务，这是迈向通用视觉模型的重要一步。该论文不仅提供了工具，更是在呼吁领域重新审视“相机内参固定”这一长期被视为理所当然的简化假设。

**Key Findings:**

- To address both gaps, we present InFlux++, consisting of two components.
- InFlux++ Real is a large-scale real-world benchmark that extends InFlux with 514K+ newly captured frames across 334 high-resolution videos, spanning a wider range of scenes and camera motions.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05389v1)
- [arXiv](https://arxiv.org/abs/2607.05389v1)

---

<a id='2607.05382v1'></a>
## [Search Beyond What Can Be Taught: Evolving the Knowledge Boundary in Agentic Visual Generation](https://arxiv.org/abs/2607.05382v1)

**Authors:** Haozhe Wang, Weijia Feng, Jinpeng Yu, Che Liu, Ping Nie, Fangzhen Lin, Jiaming Liu, Ruihua Huang, Jimmy Lin, Wenhu Chen, Cong Wei

**Published:** 2026-07-06

**Categories:** cs.CV, cs.AI

**Abstract:**

Visual generators excel at rendering, but they confidently fabricate what they do not know. User requests are unbounded, evolving, and deeply long-tailed: new characters, trending entities, post-cutoff events, and more. This world-knowledge bottleneck is structural: generators are trained on fixed corpora, but the visual world is open-ended. We construct SearchGen-20K and SearchGen-Bench, with 20,839 prompts spanning twelve failure categories and twenty-two domains, paired with a pre-executed multimodal SearchGen-Corpus-1M to support offline, reproducible research. On SearchGen-Bench, frontier open generators score only 21 to 28 out of 100, a 40-point collapse invisible to existing benchmarks. The natural remedy is to employ search tools, enabling agentic visual generation. However, we find that naive search fails: it retrieves indiscriminately, injecting noise into prompts the generator already handles. We trace the root cause to a generator-specific, evolving knowledge boundary: the divide between what a generator can internalize through training and what must remain in external context. Although this boundary is hard to specify in advance, we show that it is discoverable through a teach-then-search co-training framework. Even a minimal version of this co-training recipe produces monotonic improvement, laying the foundation for recursive self-improvement in visual generation that can meet world-knowledge-grounded requests. We release the full dataset, co-training corpus, and search corpus as a replayable harness for tool-augmented, world-knowledge-grounded visual generation.

**Analysis:**

### 1. 摘要翻译
视觉生成器擅长渲染，但却会自信地捏造其并不掌握的知识。用户请求通常是无界的、演进的且呈现长尾分布（如新角色、趋势实体、 cutoff 后的事件等）。这种“世界知识瓶颈”是结构性的：生成器在固定语料上训练，而视觉世界是开放的。我们构建了 **SEARCHGEN-20K** 和 **SEARCHGEN-BENCH**（包含 20,839 个提示词，覆盖 12 个故障类别和 22 个领域），并配以预执行的多模态语料库以支持离线可复现研究。测试表明，前沿生成器在此基准上的性能会大幅下滑。解决此瓶颈的自然方法是引入搜索工具实现“智能体化生成”，但 naive search 会引入噪声，导致生成器原有性能下降。我们追溯其根源在于“生成器特定且不断演进的知识边界”——即生成器通过训练能内化的知识与必须依赖外部语境的知识之间的鸿沟。我们提出了一个“先教后搜”的协同训练框架，证明该边界可以通过该框架被发现和优化。我们开源了完整数据集和协同训练语料，为工具增强的知识 grounded 视觉生成构建了可重演的基座。

### 2. 方法动机分析
- **驱动力**：解决视觉生成器因“知识 cutoff”导致的“自信幻觉”问题，即生成器在处理长尾、动态世界知识时的能力缺失。
- **痛点**：当前盲目引入搜索（naive search）会产生**搜索噪声（search noise）**：无关视觉细节、风格污染、不合理的结构复制等。现有生成器无法区分哪些信息是“已内化知识”，哪些是“必须外部检索的知识”。
- **研究假设**：存在一个**生成器特定的知识边界（Knowledge Boundary）**，该边界随生成器训练的演进而动态移动；通过“教学（内化）+ 搜索（补充）”的协同训练，可以使生成器最大限度内化知识，同时令搜索策略精准对齐其剩余的认知空白。

### 3. 方法设计详解
#### 流程 pipeline：
1. **Gate（门控阶段）**：给定提示词 $p$，判定是否需要外部知识。若涉及关键视觉身份或历史事实，触发搜索；否则跳过以避免噪声。
2. **Filter（过滤阶段）**：从多模态搜索结果中挑选最精准的内容，仅保留关键部分，摒弃背景、光照等无关干扰，防止“复制粘贴”式的伪造。
3. **Integrate（整合阶段）**：将检索到的知识转化为自然语言描述和 grounded 引用（例如：“跟随图片I，渲染xxx”），防止直接输入原始像素造成的风格污染。

#### 关键技术：
- **协同训练架构**：
    - **Phase 1 (Teach)**：使用 **在线迭代 DPO**。通过让生成器在搜索增强后的输入下进行采样，并根据质量反馈进行强化，推动“知识边界”向外扩展（即把原本需要搜的变成自己能画的）。
    - **Phase 2 (Learn to Search)**：**拒绝采样微调 (RFT)**。训练 reasoner 学习“何时 abstains”，即只有当 Generator 在 DPO 后依然无法解决的难题才触发搜索，从而避免在简单问题上引入噪声。

### 4. 方法对比分析
- **本质区别**：从传统的“固定搜索策略”转变为“生成器-搜索者协同进化”模式。
- **创新贡献**：首次定义并量化了“生成器特定的知识边界”，并提出了通过“先内化、后精简策略”来消除搜索噪声的闭环机制。
- **适用场景**：适用于需要精确知识支撑的复杂生成任务（如历史细节、特定角色、近期事件）。

### 5. 实验分析（精简版）
- **验证方法**：在 SEARCHGEN-BENCH 上进行对比，设置 NoSearch, BlindSearch, ReasonedSearch 三个维度。
- **关键结果**：在协同训练后，模型不仅在搜索密集型任务中性能显著提升，在无需搜索的任务中也通过“选择性 abstains”规避了性能倒退，在 4B 模型下达到了接近前沿 API 的效果。
- **局限**：目前的评价指标仍依赖 VLM 自动评测，存在 reward noise 风险；且目前只针对了“搜索”这一种工具，未覆盖其他模态。

### 6. 实用指南
- **开源情况**：已发布数据集、搜索语料库和评估基准（Project Page: https://haozheh3.github.io/SearchGen）。
- **实现建议**：
    - 关键参数：DPO 的 $\beta=100$，梯度使用 flow-matching 损失。
    - 注意：务必引入 SSIM 惩罚项以防止 DPO 过程中的 degenerate pairs。
- **迁移可能**：该“先教后搜”框架可直接迁移至 RAG 驱动的视频生成、3D 资产生成等需要复杂知识 ground 的领域。

### 7. 总结
- **核心思想**：通过协同训练将知识边界内化，让搜索成为仅针对未知领域的“补丁”。
- **速记版 pipeline**：
    1. **评估空缺**：识别知识缺口。
    2. **精准过滤**：剔除检索噪声。
    3. **语言化嵌入**：将搜索结果转为文本指令。
    4. **迭代内化**：DPO 强化 generator 的参数化记忆。
    5. **策略校准**：RFT 训练 reasoner 仅在必要时出手。

**Key Findings:**

- User requests are unbounded, evolving, and deeply long-tailed: new characters, trending entities, post-cutoff events, and more.
- Although this boundary is hard to specify in advance, we show that it is discoverable through a teach-then-search co-training framework.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05382v1)
- [arXiv](https://arxiv.org/abs/2607.05382v1)

---

<a id='2607.05377v1'></a>
## [Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation](https://arxiv.org/abs/2607.05377v1)

**Authors:** Jiaqi Peng, Xiqian Yu, Delin Feng, Yuqiang Yang, Wenzhe Cai, Jing Xiong, Ganlin Yang, Jinliang Zheng, Jiafei Cao, Xueyuan Wei, Jiangmiao Pang, Yuan Shen, Tai Wang

**Published:** 2026-07-06

**Categories:** cs.RO, cs.AI, cs.CV

**Abstract:**

While recent Vision-Language-Action (VLA) models show promise toward generalist manipulation policies, they struggle with long-horizon tasks due to their Markovian nature-relying solely on current observations. Hierarchical dual-system methods address this but suffer from a gap between high-level planning semantics and low-level execution kinematics. We introduce Cortex, a bidirectionally aligned embodied agent framework with a customized planning interface that conveys executable and tractable subtask plans from high-level VLM to low-level VLA. Specifically, we standardize manipulation subtasks into 32 canonical skill primitives and inject tractability principles, such as representative object attributes and improved trajectory reachability, into the data generation pipeline. This enables automatic annotation of over 4k hours of open-source video data and generation of 30 hours of simulation data. We further devise an event-balanced sampling strategy to construct training data for fine-tuning the framework to better handle planning ambiguity during subtask transitions, enhanced by carefully designed harness engineering from task contexts to skill constraints during inference. Both open-loop VLM and closed-loop system evaluations demonstrate Cortex's efficacy, e.g., it outperforms monolithic baselines by 3.1% on Libero-long and 4.1% on RoboTwin. Notably, Cortex's generalist VLM enables zero-shot completion of unseen real-world long-horizon tasks, such as multi-stage chemistry experiments, by simply combining with a fine-tuned VLA-a capability infeasible through VLA fine-tuning alone.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇论文《Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation》的分析如下：

### 1. 主要贡献总结
该论文提出了一种名为“Cortex”的具身智能体框架，旨在解决现有视觉-语言-动作（VLA）模型在处理长时程任务时的规划与执行脱节问题。通过引入双向对齐机制和标准化的技能原语（Skill Primitives），该框架成功打通了高层视觉语言模型（VLM）的语义规划与底层VLA的运动执行，显著提升了模型在复杂、多阶段操作任务中的泛化能力与成功率。

### 2. 关键创新与方法论
*   **双向对齐的层级架构**：不同于传统的单体式（Monolithic）模型，Cortex 构建了一个双层系统，通过专门的“规划接口”确保高层语义指令能够转化为可执行、具备可控性的底层动作。
*   **标准化技能原语（Canonical Skill Primitives）**：将复杂的操纵任务拆解为32种标准化的技能原语，并将对象属性和轨迹可达性（Reachability）注入数据生成管线，从而实现大规模高质量数据的自动化标注。
*   **事件均衡采样与上下文约束（Event-balanced Sampling & Harness Engineering）**：针对任务转换阶段的规划模糊性，引入了事件均衡采样策略，并在推理阶段通过上下文到技能约束的工程化设计，有效增强了长时程任务执行的稳定性。

### 3. 对领域的潜在影响
*   **范式转移**：该论文挑战了当前工业界过度依赖单一庞大VLA模型的趋势，展示了通过“语义规划+技能执行”的解耦方案，在复杂长序列任务中能取得更好的效果，这为未来具身智能的设计提供了重要参考。
*   **数据效率与规模化**：通过自动标注4000小时开源视频数据，展示了如何利用成熟的视频数据资源来训练具身智能体，缓解了机器人领域长期以来高昂的数据获取成本问题。
*   **零样本迁移能力**：研究证明了通过组合通用的VLM与微调后的VLA，能够实现对未见过的现实任务（如多阶段化学实验）的零样本执行，这在实际生产和科研实验自动化中具有极高的应用潜力。

### 4. 受益的相关领域与应用
*   **自动化实验室与科研自动化**：论文明确提到的多阶段化学实验，直接指向了需要高度逻辑推理与精确物理操作的科研场景。
*   **家庭服务机器人**：处理如“整理房间”或“制作简易餐点”等需要长时序、多物体交互的任务。
*   **精密制造与仓储物流**：涉及长链路操作、避障以及复杂装配的工业场景。

### 5. 可推断的局限性
*   **技能定义的完备性限制**：尽管标准化了32个原语，但在未定义的未知操作空间或环境高度动态变化时，该框架对原语库的依赖可能成为瓶颈。
*   **层级延迟与误差传播**：双层系统虽然解耦，但高层规划若出现错误（即语义理解偏差），可能会迅速导致底层执行的连锁失效，如何实时反馈纠偏是一个潜在隐患。
*   **硬件依赖性**：尽管实验显示了泛化能力，但在不同机器人构型（如不同手爪、不同自由度机械臂）之间的通用性，以及在Sim-to-Real（仿真到现实）过程中对物理参数差异的鲁棒性，仍有待进一步验证。

**专家点评：**
这篇论文的趣味性在于它平衡了**“大模型的语义泛化能力”**与**“运动控制的确定性”**。在当前的计算机视觉研究中，视觉模型往往偏向于识别，而动作模型偏向于控制，Cortex 通过将语义空间压缩为标准化的技能序列，巧妙地解决了长时程任务中“想得好却做不出”的尴尬，是具身智能从“玩具任务”迈向“现实任务”的重要实践。

**Key Findings:**

- We introduce Cortex, a bidirectionally aligned embodied agent framework with a customized planning interface that conveys executable and tractable subtask plans from high-level VLM to low-level VLA.
- Both open-loop VLM and closed-loop system evaluations demonstrate Cortex's efficacy, e.g., it outperforms monolithic baselines by 3.1% on Libero-long and 4.1% on RoboTwin.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05377v1)
- [arXiv](https://arxiv.org/abs/2607.05377v1)

---

<a id='2607.05369v1'></a>
## [GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness For Variational Automation Tasks](https://arxiv.org/abs/2607.05369v1)

**Authors:** Kaiyuan Chen, Shuangyu Xie, Letian Fu, Justin Yu, William Pacini, Sandeep Bajamahal, Hudson Kim, Jaimyn Drake, Daehwa Kim, Haoru Xue, Jonathan Francis, Christian Juette, Peter Schaldenbrand, Muhammet Yunus Seker, Ruwan Wickramarachchi, Uksang Yoo, Guanzhi Wang, Adithyavairavan Murali, Balakumar Sundaralingam, S. Shankar Sastry, Spencer Huang, Yuke Zhu, Linxi "Jim" Fan, Ken Goldberg

**Published:** 2026-07-06

**Categories:** cs.RO, cs.AI, cs.CL, cs.LG

**Abstract:**

For robots to work reliably in commercial and industrial applications, can recent advances in agentic coding systems combine interpretable robot programming with the open-world adaptability of model-free policies? We focus on "Variational Automation" (VA), a class of tasks that have larger variations in object geometry and pose than fixed automation. Model-free policies often struggle to close the reliability gap for VA tasks, which must be executed persistently and reliably in commercial and industrial applications. Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL). GaP then generates an internal simulation environment to rehearse task instances with different graphs in parallel to iteratively refine the graph structure and parameters to improve success rates and throughput. Evaluation with 8 new open VA task benchmarks, 4 in-simulation and 4 in real-world, suggests that GaP can achieve success rates that significantly outperform baselines. Details, code, and data can be found online: https://graph-robots.github.io/gap

**Analysis:**

以下是对 GaP (Graph-as-Policy) 论文的方法分析与总结：

### 1. 摘要翻译
为了使机器人能在商业和工业应用中可靠工作，最近的智能体编码系统能否将可解释的机器人编程与开放世界的模型无关策略适应性结合起来？我们专注于“变分自动化（VA）”——一类对象几何形状和位姿变化比固定自动化更大的任务。模型无关策略在处理此类任务的可靠性差距时常表现不佳。受任务与运动规划（TAMP）和机器人操作系统（ROS）的启发，我们引入了“策略即图（GaP）”，这是一个多智能体编码框架，利用模块化机器人技能库（MORSL）生成包含感知、规划和控制节点的有向计算图。GaP 通过内部仿真环境并行重演任务实例，迭代优化图结构和参数，从而提高成功率和吞吐量。在8个新的开放式VA任务基准测试中的评估显示，GaP 的成功率显著优于现有基线。

### 2. 方法动机分析
- **驱动力**：解决工业环境中“固定自动化”（灵活性差）与“通用机器人”（可靠性低）之间的矛盾，实现具备高鲁棒性和适应性的变分自动化（VA）。
- **痛点**：端到端模型（VLA）在处理变分对象时常因几何变化而失效，且缺乏可解释性；单一智能体编码（CaP）易产生幻觉、难以管理长程逻辑。
- **假设**：通过将复杂的策略分解为结构化的、可验证的“有向计算图”，并利用多智能体进行局部优化和模拟重演，可以显著降低幻觉并提升系统可靠性。

### 3. 方法设计详解
GaP 的核心在于将策略表示为有向计算图 $G=(V, E)$，其中节点 $V$ 为原子技能，边 $E$ 为依赖关系。
- **工作流（Pipeline）**：
  1. **任务分解**：由“编排智能体”将任务需求拆解为功能性语义段。
  2. **计算图合成**：利用“技能智能体”从 MORSL 库中映射并配置原子节点，构建初始图 $G$。
  3. **重演优化（Self-Learning）**：
     - **采样**：在仿真中从置信空间 $B$ 采样任务实例。
     - **并行评估**：在 Isaac 模拟器中执行图，并记录每一步的感知状态、接触点与误差。
     - **迭代更新**：当性能未达标时，LLM 智能体根据反馈分析失败的几何根源，修改图结构（调整逻辑、更换技能节点或参数）。
- **模型结构**：MORSL 提供了预定义的技能接口（如 `perception_single`, `grasp_moe`），每个技能定义了明确的输入输出类型合同，确保了图的可组合性。

### 4. 方法对比分析
- **区别**：不同于传统的“代码即策略”，GaP 强制执行图结构约束，通过“节点”替代“原始代码”，使得每个环节（感知、抓取、路径规划）都可被隔离测试和替换。
- **创新点**：引入了“仿真重演反馈循环”，这是机器人编码领域首个将物理模拟器的定量反馈用于智能体策略图优化的系统。
- **适用场景**：已知工作站环境但物体位姿和几何具有中等程度变分的应用场景（如自动分拣、组装）。

### 5. 实验分析
- **关键结果**：在8个 VA 基准测试中，GaP 在几乎所有变分条件下（如 $20 \times 20 cm$ 区域内物体位姿变化）表现出极高的鲁棒性，成功率远超基线。
- **优势**：不仅大幅提升成功率，还能实现闭环重演的持续优化。
- **局限**：目前的实验仍偏向准静态任务，对于高动态、强力控反馈要求或可变形物体任务的处理能力尚需进一步扩展。

### 6. 实用指南
- **开源情况**：项目主页：https://graph-robots.github.io/gap
- **实现建议**：系统架构深度依赖于“结构化协议（Protobuf）”，在定义自定义技能时，务必严格遵守 `MORSL` 的类型签名（如 `Se3Pose`, `OrientedBoundingBox`），否则无法实现图的自动组装。
- **迁移性**：该方法通过“节点+边”的抽象，非常易于迁移至任何支持 ROS 的机器人平台。

### 7. 总结
- **核心思想**：通过分层计算图抽象，将机器人策略构建转化为可自动优化、可验证的模块化节点链接问题。
- **速记版pipeline**：
  1. 拆解：自然语言任务拆分为原子功能段。
  2. 建图：从预置库（MORSL）中抓取技能节点组装计算图。
  3. 执行：在仿真中运行并记录执行轨迹数据。
  4. 诊断：LLM 分析执行数据中的失败根本原因。
  5. 优化：智能体调整图结构以迭代改进成功率。

**Key Findings:**

- Motivated by prior work on Task and Motion Planning (TAMP) and the Robot Operating System (ROS), we introduce Graph-as-Policy (GaP), a multi-agent coding harness that generates directed computation graphs with perception, planning, and control nodes from a Modular Open Robot Skill Library (MORSL).
- Evaluation with 8 new open VA task benchmarks, 4 in-simulation and 4 in real-world, suggests that GaP can achieve success rates that significantly outperform baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05369v1)
- [arXiv](https://arxiv.org/abs/2607.05369v1)

---

<a id='2607.05356v1'></a>
## [ReCal3R: Reliability-Calibrated Learning Rates for Streaming 3D Reconstruction](https://arxiv.org/abs/2607.05356v1)

**Authors:** Xinze Li, Yiyuan Wang, Pengxu Chen, Wentao Fan, Weifeng Su, Weisi Lin, Wentao Cheng

**Published:** 2026-07-06

**Categories:** cs.CV

**Abstract:**

Streaming 3D reconstruction relies on a compact recurrent scene state to process long image streams in linear time and bounded memory. However, repeated updates can gradually corrupt this state, causing reliable historical information to be overwritten by noisy or ambiguous observations. We introduce ReCal3R, a reliability-calibrated learning rate method for recurrent 3D reconstruction. Instead of directly applying a candidate learning rate, our method estimates state token reliability from the maintained scene state and uses it to calibrate a candidate learning rate derived from token alignment, state reconstruction residual, and recent update pressure. The resulting token-wise learning rate interpolates between a conservative base rate and the candidate rate, suppressing aggressive updates on unreliable tokens while preserving adaptation to informative frames. Applied to CUT3R as a training-free calibration rule, ReCal3R reaches strong performance on long sequences in pose, depth, and reconstruction quality, including a 3.7$\times$ reduction in ATE, with comparable runtime and memory. Code is available at: https://github.com/Powertony102/ReCal3R.

**Analysis:**

### 1. 摘要翻译
流式3D重建依赖于紧凑的递归场景状态来处理线性时间且内存有限的长图像流。然而，重复的更新会逐渐破坏这种状态，导致可靠的历史信息被噪声或模糊的观测结果所覆盖。我们引入了ReCal3R，这是一种用于递归3D重建的可靠性校准学习率方法。我们的方法不是直接应用候选学习率，而是根据维持的场景状态估计状态Token的可靠性，并利用它来校准源自Token对齐、状态重建残差和近期更新压力的候选学习率。所得的Token级学习率在保守的基础速率和候选速率之间进行插值，从而在抑制不可靠Token的激进更新的同时，保留了对信息量大的帧的适应性。

### 2. 方法动机分析
*   **驱动力**：解决长序列流式重建中，因“盲目”进行状态更新导致的灾难性遗忘与信息累积错误问题。
*   **现有方法痛点**：现有的自适应更新策略（如TTT3R）主要依据当前观测帧与状态的对齐程度，忽略了被更新的状态Token本身是否仍是“可靠的历史几何载体”。
*   **研究假设**：如果状态Token因多次更新已偏离初始几何意义，或者当前观测与其对应关系模糊，则该Token不可靠，应降低其学习率，避免被噪声覆盖。

### 3. 方法设计详解
ReCal3R在CUT3R等递归框架之上引入了校准机制，其Pipeline如下：

1.  **候选学习率构建 ($\tilde{\beta}^{(t)}$)**：综合三个维度生成基础更新强度：
    *   **Token对齐**：沿用TTT3R的对齐门控。
    *   **重建残差 ($r^{(t)}$)**：若当前帧能被现有状态解释（残差小），则该帧信息冗余，应降低学习率。
    *   **更新压力 ($h^{(t)}$)**：利用指数移动平均监测Token的更新频率，频率越高则压力越大，通过指数衰减函数抑制饱和状态下的过度修改。
2.  **状态可靠性估计 ($R^{(t)}$)**：核心创新点，评估Token是否值得被更新：
    *   **状态稳定性 ($1-d_m^{(t)}$)**：衡量Token与其初识状态的偏离度，偏离越大越不可靠。
    *   **注意力集中度 ($1-e_m^{(t)}$)**：利用Shannon熵评估Token对图像Token的注意力，分布越分散（熵越大），关系越模糊。
    *   **信度校准**：通过一致性池化（Agreement-based pooling）结合上述两指标，并使用距离模糊中心（0.5）的置信度因子进行加权，输出最终可靠性分数。
3.  **最终动态更新**：$\beta^{(t)} = (1 - R^{(t)})\beta_{base} + R^{(t)} \odot \tilde{\beta}^{(t)}$。利用可靠性分数在保守基准速率与候选速率间进行软插值。

### 4. 方法对比分析
*   **本质区别**：从单纯的“观测驱动更新”转变为“状态可靠性主导的按需更新”。
*   **创新贡献**：提出了一种无需训练的闭式解校准方案，首次明确定义并量化了递归状态的“可靠性”。
*   **适用场景**：任何使用紧凑递归状态（Latent/Hidden State）进行流式在线学习的3D任务。

### 5. 实验分析
*   **验证方法**：在ScanNet、7-Scenes、TUM-Dynamics等数据集上进行长序列评估。
*   **关键结果**：在1000帧长序列中，ATE（绝对轨迹误差）较基线CUT3R降低了3.7倍，表现出极强的长期稳定性。
*   **主要优势**：即插即用、无需训练、显著提升几何一致性。
*   **主要局限**：对局部细节更新存在一定的保守性，可能微弱影响极短序列上的局部精度。

### 6. 实用指南
*   **开源情况**：已开源，地址：`https://github.com/Powertony102/ReCal3R`。
*   **实现细节**：关键超参数包括衰减因子 $\lambda=0.95$ 和保守基准速率 $\beta_{base}=0.1$。实现时应确保状态偏差计算使用第一帧的初始状态作为固定锚点。
*   **迁移可能**：可直接集成到任何基于RNN或类似Transformer Fast-Weight机制的场景表示模型中。

### 7. 总结
*   **核心思想**：基于状态Token的几何可靠性，对学习率进行动态门控。
*   **速记版Pipeline**：
    1. 计算当前帧的信息增益（对齐度+残差+压力）。
    2. 检查待更新Token的“历史信用度”（初始偏离+关注度熵）。
    3. 综合判断：可靠则大胆更新，不可靠则按预设底线微调。

**Key Findings:**

- We introduce ReCal3R, a reliability-calibrated learning rate method for recurrent 3D reconstruction.
- Instead of directly applying a candidate learning rate, our method estimates state token reliability from the maintained scene state and uses it to calibrate a candidate learning rate derived from token alignment, state reconstruction residual, and recent update pressure.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05356v1)
- [arXiv](https://arxiv.org/abs/2607.05356v1)

---

<a id='2607.05274v1'></a>
## [Erasing Without Collateral Damage: Precise Concept Removal in Diffusion Models](https://arxiv.org/abs/2607.05274v1)

**Authors:** Parth Upman, Nishita Jain, Shreyank N Gowda

**Published:** 2026-07-06

**Categories:** cs.CV

**Abstract:**

Training-free concept erasure is an attractive mechanism for controlling text-to-image diffusion models, but precise erasure often comes at the cost of damaging semantically related non-target concepts. Existing value-space methods remove the component of each cross-attention value along the target concept direction, implicitly treating target identity and shared visual structure as the same signal. We argue that this is the source of much of the collateral damage in prior preservation. We introduce CARE, a closed-form concept erasure operator that replaces the raw target direction with a kept-subspace-aware direction computed from a small bank of retained concept anchors. The resulting edit is applied directly in cross-attention value space, requires no model fine-tuning, and adds only a negligible offline computation. A single shrinkage parameter controls the erase-preserve trade-off. We further show that the operator admits a minimum-disturbance interpretation and, in its projection form, leaves the kept subspace invariant. Experiments under the standard concept-erasure protocol show that our method preserves non-target concepts more faithfully while maintaining competitive erasure across instance, style, and celebrity concepts. Code: https://github.com/parthupman/care

**Analysis:**

这份报告深入分析了论文《Erasing Without Collateral Damage: Precise Concept Removal in Diffusion Models》所提出的 **CARE** 方法。

### 1. 摘要翻译
训练阶段无需微调的概念擦除（Training-free concept erasure）是控制文本到图像扩散模型的诱人机制，但精确的擦除往往以破坏语义相关且非目标的概念为代价。现有的值空间（value-space）方法通过移除跨注意力（cross-attention）值在目标概念方向上的分量来实现擦除，这隐含地将目标身份与共享的视觉结构视为同一信号，我们认为这是导致先前保存性（preservation）中出现“附带损害”的主要原因。我们引入了 **CARE**，一种闭式（closed-form）概念擦除算子，它用从少量保留概念锚点（retained concept anchors）计算出的“保持子空间感知”（kept-subspace-aware）方向替换了原始目标方向。该编辑直接应用于跨注意力值空间，无需微调，且仅增加极少的离线计算量。一个单一的收缩参数控制了擦除与保留的权衡。实验表明，我们的方法在维持实例、风格和名人概念竞争性擦除效果的同时，更忠实地保留了非目标概念。

### 2. 方法动机分析
- **驱动力**：解决概念擦除时的“附带损害”问题，即在擦除特定目标（如特定人物）时，模型不应丧失对相关类别（如其他人物、艺术风格）的生成能力。
- **痛点**：现有方法将目标概念的向量完全移除，由于目标概念（如特定卡通角色）与保留概念（如其他卡通角色）共享视觉特征，完全移除目标向量必然会导致保留概念的视觉质量下降。
- **核心直觉**：擦除不应盲目移除整个目标方向，而应识别并保留目标向量中属于“共享视觉特征空间”的分量，仅消除目标特有的差异化分量。

### 3. 方法设计详解
- **核心流程**：
  1. **构建锚点库**：为目标概念选择少量不需要擦除的“保留锚点”概念（如对于Bruce Lee，选取其他名人的Anchor）。
  2. **记录锚点值**：在冻结的文本编码器中，获取这些锚点在跨注意力层的表现向量 $B_j$。
  3. **计算协方差矩阵**：计算保留锚点的协方差 $\Sigma_{R,j} = \frac{1}{M} B_j^\top B_j + \gamma I_D$，用于识别哪些方向是“保留概念所重要的”。
  4. **推导CARE方向**：通过 $d_j = \Sigma_{R,j}^{-1} t_j$ 计算新的擦除方向，该方向本质上是对原始方向 $t_j$ 的“白化”或降权处理，使得目标方向中与保留锚点重合度高的成分被抑制。
  5. **应用更新**：利用公式 $v_j^{\text{CARE}} = v_j - \delta(\cdot) \frac{\langle d_j, v_j \rangle}{\langle d_j, d_j \rangle} d_j$ 进行值空间修正。
- **算法精髓**：利用Woodbury恒等式，使得在 $M \ll D$ 的情况下，通过低秩矩阵求逆高效计算方向，确保了训练即时的计算开销极低。

### 4. 方法对比分析
- **本质区别**：从单纯的“投影移除”（投影到目标向量的法平面）转变为“协方差加权判别移除”。它不仅考虑目标本身，还通过锚点库主动学习“不可触碰”的共享子空间。
- **贡献**：引入了协方差结构来调节擦除方向，使算子具备了“最小扰动”解释，且仅需一个超参数 $\gamma$ 即可灵活调节擦除与保留的平衡点。

### 5. 实验分析（精简版）
- **主要结果**：在实例擦除、艺术风格擦除和名人擦除任务中，CARE 在保留目标概念的 FID 分数上显著优于现有的 AdaVD 方法。
- **优势**：极高的灵活性，在不牺牲擦除效果（CS值）的前提下，大幅提升了对非目标概念的保留质量（FID降低）。
- **局限**：如果目标概念与保留概念高度纠缠（如两人长相极其相似），擦除效果会有所妥协，这符合“鱼与熊掌不可兼得”的权衡。

### 6. 实用指南
- **开源情况**：代码已开源（github.com/parthupman/care）。
- **关键细节**：
  - **超参数 $\gamma$**：这是核心调节器。$\gamma$ 越大，接近原始擦除（擦得干净，保留差）；$\gamma$ 越小，越注重保护。
  - **锚点库**：建议选取与目标具有一定共性的样本，但不应选取与目标几乎重合的样本。
- **迁移性**：此方法本质上是值空间内的线性代数运算，可直接迁移至任何具有跨注意力架构的Latent Diffusion模型中（如 SDXL）。

### 7. 总结
- **核心思想**：利用保留样本的协方差重塑擦除方向，精准剔除特有分量而非共享特征。
- **速记版pipeline**：
  1. 获取目标向量。
  2. 获取保留概念集合并计算其协方差。
  3. 用协方差矩阵加权调整目标方向。
  4. 进行门控的值空间投影减法。

**Key Findings:**

- We introduce CARE, a closed-form concept erasure operator that replaces the raw target direction with a kept-subspace-aware direction computed from a small bank of retained concept anchors.
- Experiments under the standard concept-erasure protocol show that our method preserves non-target concepts more faithfully while maintaining competitive erasure across instance, style, and celebrity concepts.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05274v1)
- [arXiv](https://arxiv.org/abs/2607.05274v1)

---

<a id='2607.05176v1'></a>
## [FSDC-DETR: A Frequency-Spatial Domain Collaborative DETR for Small Object Detection](https://arxiv.org/abs/2607.05176v1)

**Authors:** Aiwen Liu, Chengguang Zhu, Gang Wang, Dandan Zhu, Haodong Lin, Yan Wang, Huiyu Zhou, Zhiyi Pan

**Published:** 2026-07-06

**Categories:** cs.CV

**Abstract:**

Small object detection (SOD) remains a challenging task in real-world applications. Despite recent advances, existing detectors remain limited by rigid processing that entangle spatial aggregation with implicit frequency aliasing and truncation, leading to inadequate preservation of high-frequency components for SOD. To tackle these limitations, we propose a Frequency-Spatial Domain Collaborative Detection Transformer (FSDC-DETR), a novel collaborative framework that explicitly models complementary spatial and frequency representations. Specifically, we first introduce Dual-Branch Frequency-Spatial Adaptive Fusion (DBFSAF) to enhance frequency diversity and adaptively capture frequency-spatial domain discriminative representations. Building on these representations, a frequency-spatial interaction scheme is further explored within the hybrid encoder to enable progressive feature propagation to the decoder. In particular, structure-aware frequency-spatial aggregation is achieved through Shunt Frequency-Spatial Feature Fusion (SFS-FF), establishing bidirectional interaction and progressive cross-scale propagation between frequency and spatial representations for coherent discriminative modeling. Meanwhile, informative high-frequency responses are preserved during scale transitions through Frequency-Spatial Dynamic Downsampling (FSD-Down), thereby minimizing frequency degradation throughout multi-scale fusion for the precise SOD. Experimental results demonstrate that FSDC-DETR achieves state-of-the-art performance, improving AP by 6.4 on VisDrone-DET2019 and 6.6 on AITODv2, with gains of 6.8 and 6.9 AP for small objects. The code is available at github.com/nevereverinsomnia/FSDC-DETR.

**Analysis:**

### 1. 摘要翻译
小目标检测（SOD）在现实应用中仍然是一项具有挑战性的任务。尽管近期已有进展，但现有检测器仍受限于僵化的处理过程，即在空间聚合中混入了隐式的频率混叠和截断，导致SOD任务中高频信息无法得到充分保留。为解决这些限制，我们提出了频率-空间域协同检测Transformer（FSDC-DETR），这是一种通过显式建模互补的空间和频率表示的创新协同框架。具体而言，我们首先引入了双分支频率-空间自适应融合（DBFSAF）模块，以增强频率多样性并自适应地捕获频率-空间域的判别性表示。在此基础上，进一步在混合编码器中探索了频率-空间交互方案，以实现特征向解码器的渐进式传播。特别地，通过分流频率-空间特征融合（SFS-FF）实现了结构感知的频率-空间聚合，在频率和空间表示之间建立了双向交互和渐进式的跨尺度传播。同时，通过频率-空间动态下采样（FSD-Down）在尺度转换过程中保留了信息丰富的高频响应，从而最大限度地减少了多尺度融合过程中的频率退化，以实现精确的SOD。实验结果表明，FSDC-DETR达到了最先进的性能，在VisDrone-DET2019和AITODv2上AP分别提升了6.4和6.6，小目标AP分别提升了6.8和6.9。代码已开源。

### 2. 方法动机分析
- **驱动力**：解决小目标在深度特征提取和多尺度融合过程中，由于过度平滑或空间压缩导致的“高频细节丢失”这一根本问题。
- **痛点**：现有CNN和ViT架构要么因卷积受限于局部，要么因ViT全局注意力导致低通滤波效应（平滑细节）。直接融合两者且缺乏显式频率感知，会导致特征混叠与关键判别信息丢失。
- **研究假设**：通过显式建模频率和空间信息的互补性，并在下采样及多尺度融合中维持频率完整性，能显著提升小目标的定位精度。

### 3. 方法设计详解
- **流程总结**：
  1. **DBFSAF**：对双分支（ViT+CNN）特征进行频率动态卷积，将特征分为两部分，分别进行空间/频率精炼，最后融合。
  2. **FSC-Hybrid Encoder**：在特征金字塔的各层中，通过SFS-FF模块利用傅里叶变换（FFT/IFFT）和空间精炼模块（SRM）进行跨尺度交互。
  3. **FSD-Down**：在尺度转换（下采样）时，使用基于可学习小波分解（DWT）和组卷积的模块，替代传统卷积，从而保留高频成分。
- **算法核心**：
  - **DBFSAF**：利用FDConv进行频率多样性增强，通过Partial操作平衡冗余。
  - **SFS-FF**：通过傅里叶变换将特征转入频域进行处理，耦合频域的全局结构信息与空间域的精细纹理。
  - **FSD-Down**：通过DWT将特征分解为四种频率分量，结合可学习的缩放因子$\rho$动态重加权，确保下采样后目标边缘依然“锋利”。

### 4. 方法对比分析
- **本质区别**：从传统的“单纯追求空间结构优化”转向“空间+频率的双域协同优化”，即不仅要看目标在哪里，还要关注如何不让目标特征在降维中被“抹平”。
- **创新贡献**：提出了一种显式的频率感知下采样机制（FSD-Down）和基于双域融合的特征精炼框架，解决了DETR类模型处理小目标时的“盲区”。

### 5. 实验分析（精简版）
- **关键结论**：在VisDrone-DET2019和AITODv2上，FSDC-DETR在小目标AP上分别有显著增益（+6.8/+6.9），验证了显式保留高频细节对SOD的必要性。
- **主要优势**：极佳的边界精细化能力，在复杂背景下对极小目标的漏检和误检更少。
- **主要局限**：由于引入了傅里叶变换及多分支处理，计算图较复杂，训练时对显存要求较高。

### 6. 实用指南
- **开源情况**：https://github.com/nevereverinsomnia/FSDC-DETR
- **实现细节**：建议关注`partial ratio \gamma`（论文中最佳为0.5），这是平衡频率精炼冗余与效果的关键参数。
- **迁移可能**：该机制可直接迁移至任何基于CNN或Transformer的密集预测任务（如分割、去噪），特别是那些对纹理敏感的任务。

### 7. 总结
- **核心思想**：利用频域建模恢复下采样中丢失的高频细节。
- **速记版pipeline**：
  1. 双分支特征提取（CNN+ViT）；
  2. 频空自适应融合（DBFSAF）；
  3. 频域协同跨尺度特征传递（SFS-FF）；
  4. 小波驱动的无损下采样（FSD-Down）；
  5. 最终检测头输出。

**Key Findings:**

- To tackle these limitations, we propose a Frequency-Spatial Domain Collaborative Detection Transformer (FSDC-DETR), a novel collaborative framework that explicitly models complementary spatial and frequency representations.
- Experimental results demonstrate that FSDC-DETR achieves state-of-the-art performance, improving AP by 6.4 on VisDrone-DET2019 and 6.6 on AITODv2, with gains of 6.8 and 6.9 AP for small objects.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.05176v1)
- [arXiv](https://arxiv.org/abs/2607.05176v1)

---

