time: 20260316

# Arxiv Computer Vision Papers - 2026-03-16

## Executive Summary

## Arxiv计算机视觉领域论文日报执行摘要（2026年3月13日）

### 1. 核心主题与趋势
今日论文呈现三大交叉趋势：
- **具身智能与机器人应用**（论文1,2,3,8）：超过三分之一的研究聚焦于机器人或数字人运动控制，强调从人类数据学习（论文3）、跨形态泛化（论文2）及物理合理性（论文1），显示CV正从感知向物理世界交互深化。
- **多模态与时空理解**（论文4,6,7,9）：视频场景图生成（论文4）、视觉语言导航（论文6）、VideoLLM中的几何引导（论文7）等研究，共同推进对动态场景的结构化、可推理表征。
- **生成模型的安全与效率**（论文5,10）：在扩散模型主流化背景下，研究向**高效压缩**（DiT-IC的Transformer编解码）和**安全抑制**（防记忆、防抄袭）两个实用化方向延伸。

### 2. 突出创新论文
- **《PhysMoDPO》**：将直接偏好优化（DPO）引入人形运动生成，通过物理约束奖励机制生成更稳定、自然的动作，为机器人运动提供新范式。
- **《DiT-IC》**：首次将扩散Transformer（DiT）与图像压缩对齐，在保持生成质量的同时提升压缩效率，可能推动编解码架构革新。
- **《Mitigating Memorization...》**：提出“区域感知提示增强+多模态抄袭检测”双机制，直击扩散模型抄袭训练数据的关键痛点，具有重要伦理与工程价值。

### 3. 新兴研究方向
- **静态到动态学习**（FLUX论文）：将静态场景先验快速适配到动态导航，可能成为机器人快速部署的新路径。
- **解耦式视觉语言导航**（DecoVLN）：将观察、推理、纠正分离，提升导航系统的可解释性与抗干扰能力，符合AI安全导向。
- **全景语义占据预测**（论文8）：结合全景视觉与多模态输入，为足式机器人构建3D动态场景理解，是自动驾驶“占据网络”向机器人域的延伸。

### 4. 推荐精读论文
根据研究价值与影响力，建议优先阅读：
1. **《PhysMoDPO》**（机器人/动画领域必读）：方法新颖，物理合理性是关键瓶颈。
2. **《DiT-IC》**（压缩/生成模型方向）：技术融合具有架构启发性。
3. **《Mitigating Memorization...》**（生成模型安全方向）：问题紧迫，方案系统。
4. **《DecoVLN》**（具身AI方向）：解耦设计思想可能影响多智能体协作架构。

### 总结
今日论文整体呈现 **“感知→交互→安全”** 的演进脉络：研究重心从传统视觉任务转向机器人具身应用（尤其运动控制），同时强化生成模型的实用化与安全性。多模态时空推理成为高阶智能的共性基础。建议关注**物理约束的强化学习**、**扩散模型的安全压缩**及**解耦式系统设计**三个技术动向。

--- 
**注**：本摘要基于论文标题与作者信息推断核心内容，完整贡献请以原文为准。

---

## Table of Contents

1. [PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](#2603.13228v1)
2. [FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning](#2603.12806v1)
3. [Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data](#2603.12686v1)
4. [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](#2603.13185v1)
5. [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](#2603.13162v1)
6. [DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](#2603.13133v1)
7. [Geometry-Guided Camera Motion Understanding in VideoLLMs](#2603.13119v1)
8. [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](#2603.13108v1)
9. [InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](#2603.13082v1)
10. [Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](#2603.13070v1)

---

## Papers

<a id='2603.13228v1'></a>
## [PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization](https://arxiv.org/abs/2603.13228v1)

**Authors:** Yangsong Zhang, Anujith Muraleedharan, Rikhat Akizhanov, Abdul Ahad Butt, Gül Varol, Pascal Fua, Fabio Pizzati, Ivan Laptev

**Published:** 2026-03-13

**Categories:** cs.LG, cs.AI, cs.CV, cs.RO

**Abstract:**

Recent progress in text-conditioned human motion generation has been largely driven by diffusion models trained on large-scale human motion data. Building on this progress, recent methods attempt to transfer such models for character animation and real robot control by applying a Whole-Body Controller (WBC) that converts diffusion-generated motions into executable trajectories. While WBC trajectories become compliant with physics, they may expose substantial deviations from original motion. To address this issue, we here propose PhysMoDPO, a Direct Preference Optimization framework. Unlike prior work that relies on hand-crafted physics-aware heuristics such as foot-sliding penalties, we integrate WBC into our training pipeline and optimize diffusion model such that the output of WBC becomes compliant both with physics and original text instructions. To train PhysMoDPO we deploy physics-based and task-specific rewards and use them to assign preference to synthesized trajectories. Our extensive experiments on text-to-motion and spatial control tasks demonstrate consistent improvements of PhysMoDPO in both physical realism and task-related metrics on simulated robots. Moreover, we demonstrate that PhysMoDPO results in significant improvements when applied to zero-shot motion transfer in simulation and for real-world deployment on a G1 humanoid robot.

**Analysis:**

以下是对论文《PhysMoDPO: Physically-Plausible Humanoid Motion with Preference Optimization》的深入分析：

### 1. 摘要翻译
近期文本驱动的人体运动生成主要依赖于在大规模运动数据上训练的扩散模型。在此基础上，现有方法尝试通过全身体控制器（WBC）将生成的运动转换为可执行轨迹以用于角色动画和机器人控制。然而，WBC轨迹虽符合物理规律，却常导致运动大幅偏离原始设计。为此，我们提出了PhysMoDPO，一个基于直接偏好优化（DPO）的框架。与依赖手工编写的物理启发式规则（如足部滑动惩罚）的方法不同，我们将WBC集成到训练流水线中，优化扩散模型，使得WBC的输出既符合物理规律又忠实于原始文本指令。我们部署了基于物理和任务特定的奖励函数，通过它们为生成的轨迹分配偏好。在模拟机器人上的广泛实验表明，PhysMoDPO在物理真实性和任务相关指标上均有提升。此外，我们证明了PhysMoDPO在模拟到真实环境的零样本运动迁移以及G1人形机器人实机部署中具有显著优势。

### 2. 方法动机分析
*   **驱动力**：解决生成式模型在“运动学空间（Kinematic Space）”与“物理执行空间（Deployed Space）”之间的鸿沟。
*   **痛点**：扩散模型生成的动作往往因物理约束（如动力学、支撑力）而无法直接执行。强行使用WBC修正会导致生成的运动发生剧烈形变，偏离输入语义；而现有基于手工惩罚项（如滑动、浮空）的方法难以覆盖复杂的物理动态。
*   **核心假设**：通过将WBC作为反馈环的一部分，利用物理与任务奖励对运动样本进行偏好排序，并利用DPO进行微调，可以引导模型学习生成“天生”符合物理执行约束的运动。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **数据采样**：给定条件 $C$，利用预训练扩散模型 $G_\theta$ 生成 $K$ 个候选运动 $X$。
    2.  **物理执行**：将候选运动 $X$ 输入到固定的WBC（如DeepMimic）中，转化为模拟轨迹 $X' = \mathcal{T}(X)$。
    3.  **奖励计算**：在执行轨迹 $X'$ 上计算：
        *   **物理奖励**：轨迹追踪误差 $\mathcal{R}_{\text{track}}$ 和防滑动奖励 $\mathcal{R}_{\text{slide}}$。
        *   **任务奖励**：文本对齐 $\mathcal{R}_{\text{M2T}}$ 和空间控制对齐 $\mathcal{R}_{\text{control}}$。
    4.  **偏好排序**：基于多目标支配原则（即胜出者需在所有维度上均优于落败者），构建偏好对 $(X_{\text{win}}, X_{\text{lose}})$。
    5.  **迭代优化**：使用DPO损失结合SFT损失微调 $G_\theta$，并递归更新以应对模型变化带来的转移误差。
*   **算法本质**：将WBC视为黑盒，通过强化学习思想中的“偏好学习”间接优化不可导的物理模拟过程。

### 4. 方法对比与创新
*   **本质区别**：不试图直接建模复杂的物理方程，而是将物理模拟器作为评估器，通过奖励函数引导生成分布平移。
*   **创新贡献**：提出了一种基于支配逻辑的多目标偏好构建方法，无需手工精细调整各种奖励的权重，且实现了生成器与物理模拟器的端到端（间接）对齐。
*   **最佳应用场景**：人形机器人运动控制、复杂物理约束下的骨架动画生成。

### 5. 实验分析
*   **验证方法**：在HumanML3D和OMOMO数据集上，对比MaskedMimic、OmniControl及SFT基线。
*   **关键结论**：在保持文本对齐的前提下，显著降低了轨迹追踪误差（Err）和动作加速度跳变（Jerk），证明了生成的运动不仅符合逻辑，而且更适合机器人电机驱动。
*   **局限性**：依赖于现有的追踪控制器；在非常复杂的非平地场景中表现仍需进一步泛化。

### 6. 实用指南
*   **开源地址**：[https://mael-zys.github.io/PhysMoDPO/](https://mael-zys.github.io/PhysMoDPO/)。
*   **关键超参数**：SFT loss权重 $\lambda_{\text{SFT}}=2$，DPO温度参数 $\beta=20$。
*   **迁移技巧**：该框架是模型无关的，可插入任意以动作作为输出的扩散模型中，只需确保有一个可用的追踪控制器。

### 7. 总结
*   **核心思想**：利用物理模拟反馈构建偏好，通过DPO迭代实现生成器物理感知微调。
*   **速记版pipeline**：
    1. 采样多个运动；
    2. WBC模拟器执行；
    3. 物理与任务打分；
    4. 筛选优胜轨迹；
    5. DPO偏好对齐。

**Key Findings:**

- Moreover, we demonstrate that PhysMoDPO results in significant improvements when applied to zero-shot motion transfer in simulation and for real-world deployment on a G1 humanoid robot.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13228v1)
- [arXiv](https://arxiv.org/abs/2603.13228v1)

---

<a id='2603.12806v1'></a>
## [FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning](https://arxiv.org/abs/2603.12806v1)

**Authors:** Zeying Gong, Yangyi Zhong, Yiyi Ding, Tianshuai Hu, Guoyang Zhao, Lingdong Kong, Rong Li, Jiadi You, Junwei Liang

**Published:** 2026-03-13

**Categories:** cs.RO

**Abstract:**

Autonomous navigation requires a broad spectrum of skills, from static goal-reaching to dynamic social traversal, yet evaluation remains fragmented across disparate protocols. We introduce DynBench, a dynamic navigation benchmark featuring physically valid crowd simulation. Combined with existing static protocols, it supports comprehensive evaluation across six fundamental navigation tasks. Within this framework, we propose FLUX, the first flow-based unified navigation policy. By linearizing probability flow, FLUX replaces iterative denoising with straight-line trajectories, improving per-step inference efficiency by 47% over prior flow-based methods and 29% over diffusion-based ones. Following a static-to-dynamic curriculum, FLUX initially establishes geometric priors and is subsequently refined through reinforcement learning in dynamic social environments. This regime not only strengthens socially-aware navigation but also enhances static task robustness by capturing recovery behaviors through stochastic action distributions. FLUX achieves state-of-the-art performance across all tasks and demonstrates zero-shot sim-to-real transfer on wheeled, quadrupedal, and humanoid platforms without any fine-tuning.

**Analysis:**

作为计算机视觉与具身智能（Embodied AI）领域的专家，我对这篇论文《FLUX: Accelerating Cross-Embodiment Generative Navigation Policies via Rectified Flow and Static-to-Dynamic Learning》的分析如下：

### 1. 主要贡献总结
该论文提出了一个统一的导航策略框架 **FLUX**，通过引入基于“修正流”（Rectified Flow）的生成式模型，成功将静态导航与动态社交导航任务整合在同一架构下。研究团队同时发布了 **DynBench** 基准，通过“静态到动态”的课程学习策略，实现了在多种具身形态（轮式、四足、人形）上的高效零样本（zero-shot）部署。

### 2. 核心创新与方法论
*   **基于“修正流”的导航决策：** FLUX 将导航任务视为概率流的轨迹生成问题，通过线性化概率流路径（Linearizing probability flow），将原本耗时的多步迭代去噪过程简化为直线轨迹。这直接显著降低了推理延迟（比扩散模型快 29%，比同类流模型快 47%），解决了生成式导航在大规模实时任务中的性能瓶颈。
*   **静态到动态的课程学习（Curriculum Learning）：** 该方法不仅学习了几何先验（静态任务），还通过强化学习在动态环境中进行微调。这种设计使得模型能够通过学习随机动作分布来捕获“恢复行为”（recovery behaviors），从而显著提升了复杂社交环境下的鲁棒性。
*   **跨形态泛化（Cross-Embodiment）：** 框架的设计不依赖于特定载体，实现了在异构硬件（轮式、四足、人形）上的零样本迁移，展现了通用具身智能策略的潜力。

### 3. 对该领域的潜在影响
*   **推动生成式导航的工业化落地：** 扩散模型虽强但推理慢，FLUX 通过修正流技术证明了“高性能”与“低延迟”在具身导航中是可以兼得的。这将推动决策式模型从感知任务向规划与执行任务的深度迁移。
*   **基准统一化：** DynBench 有望成为领域内衡量导航算法能力的“统一标尺”，解决当前导航评价体系碎片化的问题，促进算法横向对比的公平性。
*   **强化学习与生成建模的融合：** 论文证明了生成模型在处理复杂环境动态演化时，能够通过 RL 微调获取具备社交意识的决策能力，这为具身 Agent 的学习范式提供了一个新颖的思路。

### 4. 潜在应用领域
*   **室内配送机器人与服务机器人：** 在办公楼、商场等充满行人的人流密集场所，FLUX 的实时性与社交避障能力将极大提升用户体验。
*   **仿人与四足机器人的户外移动：** 该技术可直接应用于野外或复杂地形下的自主探索任务，利用其零样本迁移特性，减少不同硬件平台的适配成本。
*   **人机协作系统：** 机器人需要预测人的行为并与其进行社会互动，该研究中对“社交感知”的强调使其非常适用于辅助机器人或陪护机器人领域。

### 5. 可推测的局限性
*   **训练稳定性与数据依赖：** 虽然采用了课程学习，但将几何先验与社交动态 RL 结合可能存在训练不稳定性，对于大规模分布外场景（Out-of-Distribution）的泛化能力仍需进一步验证。
*   **对模拟器的高度依赖：** 该方法依赖 DynBench 这一物理有效的模拟环境，如果真实世界的物理交互（如摩擦、软接触、复杂地形）与模拟器存在巨大偏差，零样本转移的效果可能会打折扣。
*   **计算资源限制：** 尽管通过修正流优化了推理效率，但其模型本身（基于生成式框架）可能仍需要较高的内存占用，在嵌入式端（如轻量级无人机或小型机器人）的实际资源开销仍是挑战。

**总结：** FLUX 的核心亮点在于它并没有止步于“生成导航路径”，而是通过物理上更优的流场建模，解决了具身智能中最棘手的**实时性与通用性**冲突，是迈向通用机器人导航系统的一项重要工作。

**Key Findings:**

- We introduce DynBench, a dynamic navigation benchmark featuring physically valid crowd simulation.
- Within this framework, we propose FLUX, the first flow-based unified navigation policy.
- FLUX achieves state-of-the-art performance across all tasks and demonstrates zero-shot sim-to-real transfer on wheeled, quadrupedal, and humanoid platforms without any fine-tuning.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12806v1)
- [arXiv](https://arxiv.org/abs/2603.12806v1)

---

<a id='2603.12686v1'></a>
## [Learning Athletic Humanoid Tennis Skills from Imperfect Human Motion Data](https://arxiv.org/abs/2603.12686v1)

**Authors:** Zhikai Zhang, Haofei Lu, Yunrui Lian, Ziqing Chen, Yun Liu, Chenghuai Lin, Han Xue, Zicheng Zeng, Zekun Qi, Shaolin Zheng, Qing Luan, Jingbo Wang, Junliang Xing, He Wang, Li Yi

**Published:** 2026-03-13

**Categories:** cs.RO

**Abstract:**

Human athletes demonstrate versatile and highly-dynamic tennis skills to successfully conduct competitive rallies with a high-speed tennis ball. However, reproducing such behaviors on humanoid robots is difficult, partially due to the lack of perfect humanoid action data or human kinematic motion data in tennis scenarios as reference. In this work, we propose LATENT, a system that Learns Athletic humanoid TEnnis skills from imperfect human motioN daTa. The imperfect human motion data consist only of motion fragments that capture the primitive skills used when playing tennis rather than precise and complete human-tennis motion sequences from real-world tennis matches, thereby significantly reducing the difficulty of data collection. Our key insight is that, despite being imperfect, such quasi-realistic data still provide priors about human primitive skills in tennis scenarios. With further correction and composition, we learn a humanoid policy that can consistently strike incoming balls under a wide range of conditions and return them to target locations, while preserving natural motion styles. We also propose a series of designs for robust sim-to-real transfer and deploy our policy on the Unitree G1 humanoid robot. Our method achieves surprising results in the real world and can stably sustain multi-shot rallies with human players. Project page: https://zzk273.github.io/LATENT/

**Analysis:**

### 1. 摘要翻译
人类运动员通过高速挥拍与快速移动来完成激烈的网球对攻。然而，在人形机器人上复现此类行为极其困难，部分原因是缺乏高质量的人形网球动作数据或参考用的运动学数据。本文提出了LATENT，一个从不完美人类动作数据中学习运动型人形网球技能的系统。该不完美数据仅包含原始的动作片段，而非完整的比赛过程，这极大降低了数据采集难度。我们的核心直觉是，这些准现实数据提供了人类基础网球动作的先验知识。通过进一步的校正与组合，我们训练出一种人形策略，能够在各种条件下稳定回球，同时保持自然的运动风格。我们还设计了一系列鲁棒的虚实迁移（sim-to-real）方案，并在Unitree G1人形机器人上进行了部署，实现了与人类选手稳定对打的效果。

### 2. 方法动机分析
*   **驱动力**：旨在解决人形机器人进行高动态、高精度体育竞技（网球）时，高质量动作数据获取成本过高、动作复现困难的问题。
*   **痛点**：现有的动作捕捉系统难以获取完整的网球实战轨迹；直接从原始动作学习存在“embodiment gap”（具身差距），导致机器人运动僵硬且难以处理精细的球拍控制。
*   **研究假设**：尽管动作片段是不完整的，但只要经过合理的 latent space（潜空间）建模、修正与约束，就能将这些零散的先验知识转化为高效的网球运动控制策略。

### 3. 方法设计详解
LATENT系统的pipeline分为三个核心步骤：
1.  **不完美动作数据的获取与预训练**：邀请业余选手采集基础动作片段（正手、反手、垫步），利用 LocoMuJoCo 进行动作重定向，并训练一个运动追踪器（Motion Tracker）。关键点在于，训练中故意移除了对右腕的控制信号并添加随机干扰，强迫Tracker专注于身体其他部位的协调。
2.  **在线蒸馏与潜空间构建**：利用变分信息瓶颈（Variational Information Bottleneck）构建潜空间。通过编码器 $E$ 将动作片段映射至潜变量 $z$，解码器 $D$ 还原动作。引入**可学习的条件先验（Conditional Prior）**替代传统的固定高斯分布，以捕捉不同运动状态（如垫步 vs 挥拍）下的动作分布特征。
3.  **高层策略修正与组合**：高层策略（Policy）同时输出潜空间动作 $a^{\text{latent}}$ 与腕部修正动作 $a^{\text{correct}}$。设计了**潜在动作屏障（Latent Action Barrier）**，基于马氏距离（Mahalanobis Distance）限制高层策略的探索范围，防止为了完成任务而采样出 jittery（抖动）或非自然的组合动作。

### 4. 方法对比分析
*   **本质区别**：与现有方法（如MotionVAE）不同，LATENT采用“修正+组合”的混合控制策略，并不要求基座动作数据必须完美，而是通过高层策略动态修正手腕轨迹。
*   **创新贡献**：提出“潜在动作屏障”（LAB）机制，有效解决了强化学习在潜空间探索时可能导致的运动不自然问题。
*   **适用场景**：适用于需要高动态性、精细操作且难以获取完整高质量演示数据的全身运动类任务（如足球、跑酷）。

### 5. 实验分析
*   **验证方法**：在MuJoCo中进行大规模（10,000次）仿真训练，对比PPO、AMP、ASE、PULSE等算法；在真实物理环境中进行20场人机对抗测试。
*   **结论**：LATENT在成功率（SR）和回球距离误差（DE）上均显著优于现有算法。
*   **优势**：极强的抗干扰能力和自然的回球风格，证明了对“不完美数据”进行校正的有效性。
*   **局限**：系统目前依赖室内的光学动作捕捉系统，未来需结合主动视觉以提升通用性。

### 6. 实用指南
*   **开源情况**：已开源，代码见：`https://github.com/GalaxyGeneralRobotics/LATENT`。
*   **实现建议**：在构建潜空间时，必须根据下游任务特征（如网球的挥拍）对特定关节（如手腕）进行独立建模与干扰训练，这对于虚实迁移至关重要。
*   **迁移方向**：该架构可直接迁移至其他需要复杂全身协调的运动场景，只需更换动作片段采集集即可。

### 7. 总结
*   **核心思想**：通过潜空间动作组合与手腕轨迹实时修正，从不完美数据中提取高动态技能。
*   **速记版pipeline**：
    1. 收集片段数据并训练抗干扰追踪器。
    2. 将数据蒸馏至带有状态分布先验的潜空间。
    3. 利用策略网络组合潜动作并修正手腕。
    4. 添加动态随机化实现机器人实机部署。

**Key Findings:**

- In this work, we propose LATENT, a system that Learns Athletic humanoid TEnnis skills from imperfect human motioN daTa. The imperfect human motion data consist only of motion fragments that capture the primitive skills used when playing tennis rather than precise and complete human-tennis motion sequences from real-world tennis matches, thereby significantly reducing the difficulty of data collection.
- Our method achieves surprising results in the real world and can stably sustain multi-shot rallies with human players.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.12686v1)
- [arXiv](https://arxiv.org/abs/2603.12686v1)

---

<a id='2603.13185v1'></a>
## [Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](https://arxiv.org/abs/2603.13185v1)

**Authors:** Rohith Peddi,  Saurabh, Shravan Shanmugam, Likhitha Pallapothula, Yu Xiang, Parag Singla, Vibhav Gogate

**Published:** 2026-03-13

**Categories:** cs.CV

**Abstract:**

Spatio-temporal scene graphs provide a principled representation for modeling evolving object interactions, yet existing methods remain fundamentally frame-centric: they reason only about currently visible objects, discard entities upon occlusion, and operate in 2D. To address this, we first introduce ActionGenome4D, a dataset that upgrades Action Genome videos into 4D scenes via feed-forward 3D reconstruction, world-frame oriented bounding boxes for every object involved in actions, and dense relationship annotations including for objects that are temporarily unobserved due to occlusion or camera motion. Building on this data, we formalize World Scene Graph Generation (WSGG), the task of constructing a world scene graph at each timestamp that encompasses all interacting objects in the scene, both observed and unobserved. We then propose three complementary methods, each exploring a different inductive bias for reasoning about unobserved objects: PWG (Persistent World Graph), which implements object permanence via a zero-order feature buffer; MWAE (Masked World Auto-Encoder), which reframes unobserved-object reasoning as masked completion with cross-view associative retrieval; and 4DST (4D Scene Transformer), which replaces the static buffer with differentiable per-object temporal attention enriched by 3D motion and camera-pose features. We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction. WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.

**Analysis:**

### 1. 摘要翻译
世界场景图生成（WSGG）提供了一种模拟不断演变的物体交互的原则性表示方法，但现有方法本质上仍以帧为中心：它们仅对当前可见的物体进行推理，在遮挡时丢弃实体，且仅在2D空间内操作。为解决此问题，我们首先引入了**ActionGenome4D**数据集，通过前馈3D重建、为每个参与动作的物体提供世界坐标系下的定向边界框，以及包括因遮挡或相机运动而暂时不可见的物体的密集关系标注，将Action Genome视频升级为4D场景。基于此数据，我们将**世界场景图生成（WSGG）**形式化为一项任务，即在每个时间戳构建包含场景中所有交互物体（包括可见和不可见）的世界场景图。为此，我们提出了三种互补的方法，每种方法探索了关于不可见物体推理的不同归纳偏置：PWG（持久世界图），通过零阶特征缓冲区实现物体永久性；MWAE（掩码世界自编码器），将不可见物体推理重构为具有跨视图关联检索的掩码补全任务；以及4DST（4D场景Transformer），用基于3D运动和相机姿态特征的可微分逐物体时间注意力机制取代了静态缓冲区。我们进一步通过基于Graph RAG的方法评估了强开源视觉-语言模型（VLMs）在WSGG任务上的表现，建立了不可定位关系预测的基准。因此，WSGG推动了视频场景理解向以世界为中心、时间持久且具有可解释性的场景推理发展。

---

### 2. 方法动机分析
- **驱动力**：现有的视频场景图生成（VidSGG）局限于摄像机瞬时视角，缺乏物理常识中的“物体永久性”（Object Permanence），即当物体离开视场或被遮挡时，便会从场景图中消失，这严重限制了下游机器人任务（如导航、操作）的长期推理能力。
- **痛点**：缺乏3D空间定位和跨时间的一致性，无法维护一个持久的“世界记忆”。
- **核心直觉**：物体在离开视线后仍然存在，模型应当通过持久的几何支架（3D边界框）和关联检索机制，在物理空间中追踪并推理这些“不可见”物体的交互关系。

---

### 3. 方法设计详解
WSGG的核心是将场景图从2D帧图像迁移到4D世界空间。
- **Pipeline关键步骤**：
  1. **3D场景重建与几何对齐**：使用$\pi^3$进行单目视频的3D重建，并结合束调整（Bundle Adjustment）和基于SMPL的底面对齐，生成统一的世界坐标系。
  2. **世界状态初始化**：为每个物体分配全局唯一的持久ID，并计算其3D定向边界框（OBB）。
  3. **特征融合与推理（针对三种方法）**：
     - **PWG**：利用“最近邻查找”逻辑，将物体特征冻结在最后可见帧，通过缓冲区填补“雾区”的物体信息。
     - **MWAE**：将“不可见”视为自然掩码，通过关联检索器查询已观测到的视觉片段，重建缺失物体的特征表示。
     - **4DST**：引入时间Transformer，通过对全视频序列的物体的多模态特征（视觉、运动、相机姿态）进行双向注意力计算，实现端到端的长程推理。
  4. **关系预测**：在包含所有物体的世界图中，通过Temporal Edge Attention强化时间一致性，输出注意力、空间和交互三种关系。

---

### 4. 方法对比与创新
- **创新点**：首次将场景图生成从“以视线为中心”转向“以世界为中心”，并将“物体永久性”引入到场景图推理中。
- **本质区别**：传统VidSGG是独立的静态图序列，WSGG是一个动态演变的图，即使物体不可见，其节点依然存在于图结构中，并与环境保持语义连接。

---

### 5. 实验与总结
- **验证方法**：在ActionGenome4D数据集上进行PredCls（给定标签/定位）和SGDet（全自动检测）任务评估。
- **结论**：4DST通过可微分的时间Transformer在大多数指标上表现最优，确认了端到端时间建模的有效性；MWAE在多标签关系（空间、接触）上展现出更强的鲁棒性。
- **优势**：实现了对被遮挡或离开视场物体的持续推理，支持更复杂的机器人交互规划。
- **局限**：对长尾谓词（罕见关系）预测准确率较低；依赖于预处理得到的3D几何结构，pipeline复杂度高。

---

### 6. 实用指南
- **开源情况**：代码已开源（详见论文GitHub链接）。
- **注意点**：需要处理大规模3D重建数据的存储和对齐；在训练时注意VLM伪标签的噪声处理（使用了label smoothing和加权损失）。
- **迁移性**：该框架可以轻松迁移至其他涉及“长程物体交互”的任务，如室内机器人导航任务或视频行为总结。

---

### 7. 总结
- **核心思想**：利用3D几何支架和深度时间模型，实现物体交互的持续推理。
- **速记版pipeline**：
  1. 3D重建并对齐场景。
  2. 统一物体ID与空间坐标。
  3. 记忆或重构不可见物体。
  4. Transformer建模时空关联。
  5. 预测全场景交互关系。

**Key Findings:**

- We further design and evaluate the performance of strong open-source Vision-Language Models on the WSGG task via a suite of Graph RAG-based approaches, establishing baselines for unlocalized relationship prediction.
- WSGG thus advances video scene understanding toward world-centric, temporally persistent, and interpretable scene reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13185v1)
- [arXiv](https://arxiv.org/abs/2603.13185v1)

---

<a id='2603.13162v1'></a>
## [DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression](https://arxiv.org/abs/2603.13162v1)

**Authors:** Junqi Shi, Ming Lu, Xingchen Li, Anle Ke, Ruiqi Zhang, Zhan Ma

**Published:** 2026-03-13

**Categories:** eess.IV, cs.CV

**Abstract:**

Diffusion-based image compression has recently shown outstanding perceptual fidelity, yet its practicality is hindered by prohibitive sampling overhead and high memory usage. Most existing diffusion codecs employ U-Net architectures, where hierarchical downsampling forces diffusion to operate in shallow latent spaces (typically with only 8x spatial downscaling), resulting in excessive computation. In contrast, conventional VAE-based codecs work in much deeper latent domains (16x - 64x downscaled), motivating a key question: Can diffusion operate effectively in such compact latent spaces without compromising reconstruction quality? To address this, we introduce DiT-IC, an Aligned Diffusion Transformer for Image Compression, which replaces the U-Net with a Diffusion Transformer capable of performing diffusion in latent space entirely at 32x downscaled resolution. DiT-IC adapts a pretrained text-to-image multi-step DiT into a single-step reconstruction model through three key alignment mechanisms: (1) a variance-guided reconstruction flow that adapts denoising strength to latent uncertainty for efficient reconstruction; (2) a self-distillation alignment that enforces consistency with encoder-defined latent geometry to enable one-step diffusion; and (3) a latent-conditioned guidance that replaces text prompts with semantically aligned latent conditions, enabling text-free inference. With these designs, DiT-IC achieves state-of-the-art perceptual quality while offering up to 30x faster decoding and drastically lower memory usage than existing diffusion-based codecs. Remarkably, it can reconstruct 2048x2048 images on a 16 GB laptop GPU.

**Analysis:**

这是一份关于论文 **DiT-IC: Aligned Diffusion Transformer for Efficient Image Compression** 的深度分析：

### 1. 摘要翻译
基于扩散模型的图像压缩展现出优异的感知保真度，但高昂的采样开销和内存占用限制了其实际应用。现有扩散编码器多采用 U-Net 架构，其分层下采样强制扩散过程在浅层潜在空间（通常仅 8 倍下采样）运行，导致计算冗余。相比之下，传统的 VAE 编码器在更深层的潜在空间（16-64 倍下采样）工作。为此，我们提出了 DiT-IC，一种对齐扩散 Transformer 图像压缩框架，通过 DiT 取代 U-Net，在 32 倍下采样空间内执行扩散。DiT-IC 通过三种对齐机制将预训练的文本到图像 DiT 转换为单步重构模型：(1) 方差引导重构流；(2) 自蒸馏对齐；(3) 潜在条件引导。DiT-IC 在保持 SOTA 感知质量的同时，解码速度提升 30 倍，且显著降低内存开销。

### 2. 方法动机分析
*   **驱动力**：解决扩散模型在图像压缩任务中“计算效率低、内存占用高”的矛盾，实现感知质量与实用效率的平衡。
*   **现有方法痛点**：U-Net 分层结构导致扩散过程在浅层潜在空间运行，计算冗余且难以适应深度压缩。直接微调预训练模型会导致生成式先验与压缩重构目标不匹配，引发畸变。
*   **研究假设**：在深度压缩潜在空间（32×）内，通过特定的对齐机制，可以将复杂的多步迭代去噪过程简化为单步确定性重构，从而兼顾效率与质量。

### 3. 方法设计详解
*   **核心逻辑**：将扩散模型的生成过程视为一种自适应重构流，利用潜在变量的方差刻画不确定性，并强制模型向编码器定义的潜在几何结构对齐。
*   **流程总结**：
    1.  **方差引导重构流**：利用潜在分布的方差 $\sigma(y_t)$ 作为“时间戳”$t$，不同区域根据不确定性采取不同的去噪强度。
    2.  **自蒸馏对齐**：冻结编码器，将编码器的输出 $y_0$ 作为 DiT 的监督目标，强制单步输出 $\hat{y}_0$ 与编码器特征对齐，蒸馏掉多步迭代过程。
    3.  **潜在条件引导**：将复杂的文本 prompt 替换为轻量化的 latent 投影 ($c_{lat} = \text{Proj}_\psi(\hat{y})$)，通过对比学习将 latent 嵌入与文本空间对齐，实现无文本推理。

### 4. 方法对比分析
*   **本质区别**：与 U-Net 类扩散模型（如 StableCodec）相比，DiT-IC 采用 Transformer 架构，能在保持空间分辨率不变的同时处理深层压缩潜在特征，避免了分层结构的计算开销。
*   **创新贡献**：提出了一种将“生成式先验”高效转化为“压缩重构先验”的通用框架，特别是在极低位率下能保持语义的一致性。

### 5. 实验分析（精简版）
*   **关键结论**：在 2048×2048 分辨率下，DiT-IC 仅需 16GB 显存即可运行；解码延迟相比现有 diffusion 类编码器缩短达 30 倍；BD-rate 显著优于对比基线。
*   **主要优势**：极高的硬件适应性和高分辨率下的稳定计算效率；重构细节丰富且感知质量极佳。

### 6. 实用指南
*   **开源情况**：代码已开源至 [https://njuvision.github.io/DiT-IC/](https://njuvision.github.io/DiT-IC/)。
*   **实现细节**：关键参数在于 LoRA rank 的选择（建议 32/64），以及两阶段优化策略（先放松约束后微调）。对比学习 loss 在初始 30% 迭代后需退火处理。
*   **迁移可能**：该对齐机制可直接迁移至其他基于生成式扩散的底层视觉恢复任务（如超分、去噪），特别是针对需要“生成先验”与“确定性重构”融合的场景。

### 7. 总结
*   **核心思想**：通过语义对齐将扩散模型重构为高效的单步压缩解码器。
*   **速记版pipeline**：
    1. 获取压缩 latent；
    2. 根据 latent 方差确定去噪强度；
    3. 通过单步 DiT 投影进行特征恢复；
    4. 结合语义 latent 进行条件引导输出。

**Key Findings:**

- To address this, we introduce DiT-IC, an Aligned Diffusion Transformer for Image Compression, which replaces the U-Net with a Diffusion Transformer capable of performing diffusion in latent space entirely at 32x downscaled resolution.
- With these designs, DiT-IC achieves state-of-the-art perceptual quality while offering up to 30x faster decoding and drastically lower memory usage than existing diffusion-based codecs.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13162v1)
- [arXiv](https://arxiv.org/abs/2603.13162v1)

---

<a id='2603.13133v1'></a>
## [DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](https://arxiv.org/abs/2603.13133v1)

**Authors:** Zihao Xin, Wentong Li, Yixuan Jiang, Bin Wang, Runming Cong, Jie Qin, Shengjun Huang

**Published:** 2026-03-13

**Categories:** cs.RO

**Abstract:**

Vision-and-Language Navigation (VLN) requires agents to follow long-horizon instructions and navigate complex 3D environments. However, existing approaches face two major challenges: constructing an effective long-term memory bank and overcoming the compounding errors problem. To address these issues, we propose DecoVLN, an effective framework designed for robust streaming perception and closed-loop control in long-horizon navigation. First, we formulate long-term memory construction as an optimization problem and introduce adaptive refinement mechanism that selects frames from a historical candidate pool by iteratively optimizing a unified scoring function. This function jointly balances three key criteria: semantic relevance to the instruction, visual diversity from the selected memory, and temporal coverage of the historical trajectory. Second, to alleviate compounding errors, we introduce a state-action pair-level corrective finetuning strategy. By leveraging geodesic distance between states to precisely quantify deviation from the expert trajectory, the agent collects high-quality state-action pairs in the trusted region while filtering out the polluted data with low relevance. This improves both the efficiency and stability of error correction. Extensive experiments demonstrate the effectiveness of DecoVLN, and we have deployed it in real-world environments.

**Analysis:**

### 1. 摘要翻译
Vision-and-Language Navigation (VLN) 要求智能体遵循长视距指令并在复杂的3D环境中导航。然而，现有方法面临两大挑战：构建有效的长期记忆库和克服累积误差问题。为此，我们提出了 **DecoVLN**，一个为鲁棒流式感知和闭环控制设计的有效框架。首先，我们将长期记忆构建建模为一个优化问题，并引入了自适应细化机制（AMR），通过迭代优化评分函数，从历史候选中选择帧，以平衡语义相关性、视觉多样性和时间覆盖率。其次，为减轻累积误差，我们引入了状态-动作对级别的纠正性微调策略，通过测地距离精确量化偏离程度，在受信任区域内收集高质量数据，剔除污染数据，从而提高纠正效率和稳定性。大量实验证明了 DecoVLN 的有效性，且已在真实物理环境完成部署。

---

### 2. 方法动机分析
- **驱动力**：旨在解决现有 VLN 方法在处理长视距任务时，由于无效信息堆积导致的感知模糊（Perceptual Blindness）和闭环控制失效问题。
- **现有方法痛点**：
    1. **Context Pollution**：均匀采样策略无法根据任务相关性筛选信息，导致大量无关场景（如墙壁、角落）干扰推理。
    2. **Storage/IO Bottleneck**：传统流式感知频繁在内存和显存间切换，导致推理延迟高。
    3. **Compounding Errors**：现有模型缺乏在偏离轨迹时“实时自我纠偏”的机制，一旦 off-course 难以挽回。
- **核心假设**：导航任务的有效记忆应具备动态可更新性，且纠正应发生在动作执行之前的“状态-动作”对级别，而非事后的粗粒度修正。

---

### 3. 方法设计详解
- **流程总结**：
    1. **自适应记忆精简 (AMR)**：实时感知流中的每一帧被送入 AMR 模块。该模块计算当前帧与指令的相关性、与现有记忆库的视觉多样性、以及时间覆盖率，通过优化公式 $f^* = \arg \max [ \lambda_R \cdot \text{Sim}_{\text{Sem}} - (1 - \lambda_R)(w_V \cdot \text{Sim}_{\text{Vis}} + w_T \cdot \text{Sim}_{\text{Temp}} ) ]$ 动态保留最关键的 $K$ 帧。
    2. **动作块预测**：LLM 接收经过压缩的精简记忆，输出包含 4 个动作的“动作块”。
    3. **状态-动作纠正性微调**：通过计算 agent 当前位置与专家轨迹的测地距离（Geodesic Distance）判断是否处于“信任区域”。若在范围内但偏离，则强制查询专家动作，构建纠正数据集，进行微调。
- **关键公式解析**：公式 (1) 是一个多目标优化器，平衡了“该帧对当前指令是否有用（相关性）”、“是否提供新视角（多样性）”以及“是否覆盖时间跨度（覆盖率）”，从而保证了长视距上下文的高密度。

---

### 4. 方法对比分析
- **本质区别**：从“全局存储+统一采样”转变为“任务导向的实时预过滤”；从“被动序列学习”转变为“主动介入式的状态空间纠偏”。
- **创新贡献**：提出了自适应记忆精简机制，极大提升了模型长时上下文的处理效率；实现了基于测地距离的状态-动作级纠正，显著降低了长链路中的累积误差。
- **适用场景**：复杂、长路径、需要精细避障和语义感知的室内外机器人导航。

---

### 5. 实验分析
- **验证方法**：在 R2R 和 RxR 基准集上进行零样本及微调实验，并引入自建长视距导航验证集。
- **关键结果**：相比 StreamVLN，SR 指标在 R2R 和 RxR 上分别提升 3.5% 和 5.6%。
- **核心优势**：在仅依赖 RGB 输入的情况下，超越了许多需要深度感知或多传感器融合的模型，展示了极强的数据效率和 Sim-to-Real 泛化能力。

---

### 6. 实用指南
- **开源情况**：论文明确提及已部署在真实 Unitree GO2 机器人，框架设计具有较好的参考性。
- **实现细节**：
    - 关键参数：记忆容量 $K=8$；信任阈值 $\tau=3$。
    - 训练：需包含 SFT 和 ECF 两阶段，ECF 阶段需使用 Habitat 模拟器。
- **迁移可能**：AMR 模块是一个通用的“上下文压缩组件”，可轻松迁移到视频问答、长篇文档理解等需要处理冗余流数据的任务中。

---

### 7. 总结
- **核心思想**：通过任务导向的记忆压缩与几何约束下的在线纠偏，实现鲁棒的长视距导航。
- **速记版pipeline**：
    1. 实时感知环境信息。
    2. 根据任务相关性与多样性，动态过滤并存储关键帧。
    3. 基于受信任的状态偏差，主动进行专家数据纠偏。
    4. 预测动作块并平滑控制机器人。

**Key Findings:**

- To address these issues, we propose DecoVLN, an effective framework designed for robust streaming perception and closed-loop control in long-horizon navigation.
- Second, to alleviate compounding errors, we introduce a state-action pair-level corrective finetuning strategy.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13133v1)
- [arXiv](https://arxiv.org/abs/2603.13133v1)

---

<a id='2603.13119v1'></a>
## [Geometry-Guided Camera Motion Understanding in VideoLLMs](https://arxiv.org/abs/2603.13119v1)

**Authors:** Haoan Feng, Sri Harsha Musunuri, Guan-Ming Su

**Published:** 2026-03-13

**Categories:** cs.CV, cs.AI

**Abstract:**

Camera motion is a fundamental geometric signal that shapes visual perception and cinematic style, yet current video-capable vision-language models (VideoLLMs) rarely represent it explicitly and often fail on fine-grained motion primitives. We address this gap with a framework of $\textbf{benchmarking}$, $\textbf{diagnosis}$, and $\textbf{injection}$. We curate $\textbf{CameraMotionDataset}$, a large-scale synthetic dataset with explicit camera control, formulate camera motion as constraint-aware multi-label recognition, and construct a VQA benchmark--$\textbf{CameraMotionVQA}$. Across diverse off-the-shelf VideoLLMs, we observe substantial errors in recognizing camera motion primitives. Probing experiments on a Qwen2.5-VL vision encoder suggest that camera motion cues are weakly represented, especially in deeper ViT blocks, helping explain the observed failure modes. To bridge this gap without costly training or fine-tuning, we propose a lightweight, model-agnostic pipeline that extracts geometric camera cues from 3D foundation models (3DFMs), predicts constrained motion primitives with a temporal classifier, and injects them into downstream VideoLLM inference via structured prompting. Experiments demonstrate improved motion recognition and more camera-aware model responses, highlighting geometry-driven cue extraction and structured prompting as practical steps toward a camera-aware VideoLLM and VLA system. The dataset and benchmark is publicly available at https://hf.co/datasets/fengyee/camera-motion-dataset-and-benchmark.

**Analysis:**

这是一份针对论文《Geometry-Guided Camera Motion Understanding in VideoLLMs》的深度技术分析报告。

---

### 1. 摘要翻译
摄像机运动是塑造视觉感知和电影风格的核心几何信号，然而现有的视频视觉语言模型（VideoLLMs）极少显式表征它，且在细粒度运动基元识别上表现不佳。我们提出了一个包含基准测试、诊断和注入的框架来解决这一差距。我们策划了 *CameraMotionDataset*（具有显式摄像机控制的合成数据集），将摄像机运动表述为约束感知的多标签识别问题，并构建了 *CameraMotionVQA* 基准。通过在 Qwen2.5-VL 上的探测实验，我们发现摄像机运动线索在深度 ViT 层中会减弱，这解释了模型的失效模式。为此，我们提出了一种无需额外训练的轻量级 pipeline：从 3D 基础模型（3DFMs）提取几何摄像机线索，利用时间分类器预测运动基元，并通过结构化提示（structured prompting）将其注入到下游 VideoLLM 的推理中。实验证明，该方法显著提升了运动识别准确率及描述的摄像机感知能力。

### 2. 方法动机分析
*   **驱动力**：摄像机运动是电影语法的关键，直接关联视觉叙事和空间推理，目前的 VideoLLM 侧重于语义内容（“是什么”），忽视了镜头运动方式（“如何拍摄”）。
*   **现有方法痛点**：1) 数据集缺乏针对摄像机运动的显式监督；2) 视觉特征压缩导致时序运动线索在编码过程中丢失；3) 通用视频模型在处理“平移、推拉”等细粒度几何运动时存在识别盲点。
*   **研究假设**：通过引入冻结的 3D 基础模型（3DFMs）提取几何特征，并以结构化提示的形式注入到 VideoLLM，能够在不微调主干网络的情况下弥补语义与几何感知之间的差距。

### 3. 方法设计详解
*   **Pipeline流程**：
    1.  **预处理与分段**：将视频切分为 1 秒的非重叠片段。
    2.  **线索提取 (Teacher)**：利用冻结的 3D 基础模型（如 VGGT）在每一帧提取几何摄像机 tokens。
    3.  **约束运动识别**：训练一个轻量级的 Transformer 编码器（Temporal Classifier），输入提取的 tokens，输出受限的多标签分类结果（15个基元，强制满足互斥性约束）。
    4.  **结构化注入**：将识别出的运动标签序列化（例如：“Per-second camera motion: [pan-left, tilt-up, ...]”），作为提示词前缀直接输入到冻结的 VideoLLM。
    5.  **蒸馏优化 (Student)**：为了降低推理成本，通过蒸馏将 VGGT 的感知能力转移至轻量级模型（VGGT–Q-Former），保留跨帧感知能力。
*   **核心模块**：约束损失函数 `Linc` (Incompatibility) 和 `Lcard` (Cardinality) 是关键，确保模型输出符合物理规律（例如：不能同时“左平移”和“右平移”）。

### 4. 方法对比分析
*   **本质区别**：不试图通过昂贵的端到端微调让 LLM“自动学会”物理，而是采用“外挂感知”的思想，通过显式的几何先验引导模型的推理。
*   **创新贡献**：
    1.  首个基于几何监督的相机运动基准；
    2.  提出轻量级的 Plug-and-Play 几何线索提取范式；
    3.  利用结构化提示有效规避了 LLM 权重修改带来的灾难性遗忘风险。
*   **适用场景**：任何需要精准电影制作级理解的视频分析任务，如自动剪辑辅助、视觉描述生成、影视检索。

### 5. 实验分析（精简）
*   **验证方法**：在 *CameraMotionVQA* 协议下，测试多种主流开源 VideoLLM（Qwen2.5-VL, InternVL 等）。
*   **关键结论**：大多数模型接近随机猜测（25%），验证了“摄像机运动盲点”的存在；注入几何线索后，识别准确率显著提升。
*   **优势/局限**：优势是训练高效、无需修改 LLM 主干；局限是合成数据与真实世界存在 Gap，且目前仅支持外参控制，暂不支持内参（如变焦）的细粒度理解。

### 6. 实用指南
*   **开源**：模型和数据集将随论文发表。
*   **实现细节**：关键参数为 λinc 和 λcard，用于控制运动的物理约束。推理时阈值 $\tau=0.5$ 配合 canonicalization（正则化处理）是保证输出准确的关键。
*   **迁移方向**：该方法逻辑可迁移至任何需要多模态模型具备物理/3D 属性的任务中（如导航、机器人视频预测）。

### 7. 总结
*   **核心思想**：引入 3D 几何先验通过提示注入，增强 VideoLLM 的相机运动感知。
*   **速记版 Pipeline**：
    1. 视频片段化；
    2. 3D 模型提取运动特征；
    3. 约束分类器预测标签；
    4. 序列化注入 Prompt。

**Key Findings:**

- To bridge this gap without costly training or fine-tuning, we propose a lightweight, model-agnostic pipeline that extracts geometric camera cues from 3D foundation models (3DFMs), predicts constrained motion primitives with a temporal classifier, and injects them into downstream VideoLLM inference via structured prompting.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13119v1)
- [arXiv](https://arxiv.org/abs/2603.13119v1)

---

<a id='2603.13108v1'></a>
## [Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots](https://arxiv.org/abs/2603.13108v1)

**Authors:** Guoqiang Zhao, Zhe Yang, Sheng Wu, Fei Teng, Mengfei Duan, Yuanfan Zheng, Kai Luo, Kailun Yang

**Published:** 2026-03-13

**Categories:** cs.RO, cs.CV, eess.IV

**Abstract:**

Panoramic imagery provides holistic 360° visual coverage for perception in quadruped robots. However, existing occupancy prediction methods are mainly designed for wheeled autonomous driving and rely heavily on RGB cues, limiting their robustness in complex environments. To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes. (2) We propose a panoramic multimodal occupancy perception framework, VoxelHound, tailored for legged mobility and spherical imaging. Specifically, we design (i) a Vertical Jitter Compensation (VJC) module to mitigate severe viewpoint perturbations caused by body pitch and roll during mobility, enabling more consistent spatial reasoning, and (ii) an effective Multimodal Information Prompt Fusion (MIPF) module that jointly leverages panoramic visual cues and auxiliary modalities to enhance volumetric occupancy prediction. (3) We establish a benchmark based on PanoMMOcc and provide detailed data analysis to enable systematic evaluation of perception methods under challenging embodied scenarios. Extensive experiments demonstrate that VoxelHound achieves state-of-the-art performance on PanoMMOcc (+4.16%} in mIoU). The dataset and code will be publicly released to facilitate future research on panoramic multimodal 3D perception for embodied robotic systems at https://github.com/SXDR/PanoMMOcc, along with the calibration tools released at https://github.com/losehu/CameraLiDAR-Calib.

**Analysis:**

这是一份关于论文《Panoramic Multimodal Semantic Occupancy Prediction for Quadruped Robots》的深入技术分析。

### 1. 摘要翻译
全景影像为四足机器人感知提供了360°全方位覆盖。然而，现有的语义占用预测方法主要针对轮式自动驾驶平台，过分依赖RGB数据，在复杂环境中鲁棒性不足。为弥补这一差距：(1) 我们提出了PanoMMOcc，这是首个面向四足机器人的全景多模态占用预测数据集，包含四种传感模态和多样化场景。(2) 我们提出了VoxelHound感知框架，专为足式运动和球形成像设计。具体包括：(i) 垂直抖动补偿（VJC）模块，用于减轻机器人运动中俯仰和翻滚引起的严重视点扰动，从而实现更一致的空间推理；(ii) 多模态信息提示融合（MIPF）模块，通过将全景视觉特征与辅助模态联合，增强占用预测能力。(3) 我们建立了基准测试并进行了详尽的数据分析。实验表明，VoxelHound在PanoMMOcc数据集上实现了SOTA性能（mIoU提升4.16%）。

### 2. 方法动机分析
*   **驱动力**：解决四足机器人在复杂地形（如森林、绿地）中因传感器姿态剧烈波动（俯仰/翻滚）和极端光照变化导致的感知失效问题。
*   **现有方法痛点**：现有方法多基于轮式平台，假设传感器视点稳定；依赖单一RGB模态在暗光或纹理缺失环境下易失效；融合策略简单（常采用简单拼接），未能有效利用不同模态的特性。
*   **研究假设**：通过显式建模机器人足式运动的抖动规律（VJC）及采用不对称的“几何引导语义”融合策略（MIPF），可以显著提升动态机器人平台的3D占用感知鲁棒性。

### 3. 方法设计详解
*   **流程 Pipeline**：
    1.  **多分支特征编码**：对全景RGB、热成像、偏振图及LiDAR点云进行独立编码。
    2.  **VJC 补偿**：提取特征图的垂直分布信息，通过1D卷积预测垂直位移，并在2D BEV投影前利用双线性采样对特征进行校正。
    3.  **多模态融合 (MIPF)**：LiDAR BEV作为几何查询（Query），将图像模态压缩为语义提示（Prompt），进行交叉注意力计算。
    4.  **占用解码**：利用FPN和占用头（Occ Head）将多模态特征转换为3D语义占用栅格。
*   **关键模块**：
    *   **VJC (Vertical Jitter Compensation)**：本质上是一个视点稳定器。它将特征图列求和得到垂直结构信息，预测偏移量 $\Delta h$ 并调整采样网格，从而在特征空间“抵消”了机器人的步态振动。
    *   **MIPF (Multimodal Information Prompt Fusion)**：采用“几何主导+语义补充”原则。LiDAR BEV特征保持结构完整性，图像模态通过GAP（全局平均池化）转化为低维Prompt，作为Key和Value进行注意力增强。避免了全空间注意力带来的计算开销和噪声干扰。

### 4. 方法对比与创新
*   **本质区别**：与传统BEV融合不同，VoxelHound显式地将机器人步态动力学因素（抖动）纳入网络结构，并采取了不对称的提示融合策略。
*   **创新贡献**：提出了首个针对足式机器人步态特性的抖动补偿机制；实现了多模态（RGB+热+偏振+LiDAR）在全景感知任务中的有效整合。

### 5. 实验分析
*   **有效性验证**：在PanoMMOcc数据集上进行了广泛对比实验。
*   **关键结果**：在复杂场景下的mIoU达到23.34%，较基线方法有显著提升，特别是在暗光场景下，引入热成像和偏振信息后性能优势突出。
*   **主要局限**：在极端远距离区域，受限于LiDAR点云稀疏性，预测仍存在 sparsity 问题。

### 6. 实用指南
*   **开源/实现**：数据集与代码开源。实现时，需注意VJC模块的超参数 $C_{hd}$ 调优，以及MIPF模块中Prompt维度的平衡。
*   **迁移可能**：VJC模块可以无缝迁移至任何运动不稳定的机器人平台；MIPF融合策略可用于其他需要结合强几何线索与弱语义线索的多模态感知任务。

### 7. 总结
*   **核心思想**：显式建模运动抖动，并通过不对称多模态提示实现几何与语义的稳健融合。
*   **速记版Pipeline**：
    1. 独立编码多源传感器特征；
    2. 预测垂直偏移以动态补偿步态抖动；
    3. 以LiDAR为空间基准注入语义提示；
    4. 解码生成3D语义占用图。

**Key Findings:**

- To bridge this gap, (1) we present PanoMMOcc, the first real-world panoramic multimodal occupancy dataset for quadruped robots, featuring four sensing modalities across diverse scenes.
- (2) We propose a panoramic multimodal occupancy perception framework, VoxelHound, tailored for legged mobility and spherical imaging.
- Extensive experiments demonstrate that VoxelHound achieves state-of-the-art performance on PanoMMOcc (+4.16%} in mIoU).

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13108v1)
- [arXiv](https://arxiv.org/abs/2603.13108v1)

---

<a id='2603.13082v1'></a>
## [InterEdit: Navigating Text-Guided Multi-Human 3D Motion Editing](https://arxiv.org/abs/2603.13082v1)

**Authors:** Yebin Yang, Di Wen, Lei Qi, Weitong Kong, Junwei Zheng, Ruiping Liu, Yufan Chen, Chengzhi Wu, Kailun Yang, Yuqian Fu, Danda Pani Paudel, Luc Van Gool, Kunyu Peng

**Published:** 2026-03-13

**Categories:** cs.CV, cs.RO, eess.IV

**Abstract:**

Text-guided 3D motion editing has seen success in single-person scenarios, but its extension to multi-person settings is less explored due to limited paired data and the complexity of inter-person interactions. We introduce the task of multi-person 3D motion editing, where a target motion is generated from a source and a text instruction. To support this, we propose InterEdit3D, a new dataset with manual two-person motion change annotations, and a Text-guided Multi-human Motion Editing (TMME) benchmark. We present InterEdit, a synchronized classifier-free conditional diffusion model for TMME. It introduces Semantic-Aware Plan Token Alignment with learnable tokens to capture high-level interaction cues and an Interaction-Aware Frequency Token Alignment strategy using DCT and energy pooling to model periodic motion dynamics. Experiments show that InterEdit improves text-to-motion consistency and edit fidelity, achieving state-of-the-art TMME performance. The dataset and code will be released at https://github.com/YNG916/InterEdit.

**Analysis:**

### 1. 摘要翻译
本文介绍了多人3D动作编辑任务，旨在基于源动作和文本指令生成目标动作。为支持此任务，我们提出了InterEdit3D——首个包含人工双人动作变更标注的数据集及相应的文本引导多人动作编辑（TMME）基准。我们提出了InterEdit，一种用于TMME的同步无分类器条件扩散模型。该模型引入了语义感知计划令牌对齐（Semantic-Aware Plan Token Alignment）以利用可学习令牌捕捉高层交互意图，并提出了交互感知频率令牌对齐（Interaction-Aware Frequency Token Alignment）策略，利用离散余弦变换（DCT）和能量池化技术来建模周期性动作动态。实验表明，InterEdit显著提升了文本-动作一致性和编辑保真度，达到了TMME任务的领先水平。

### 2. 方法动机分析
- **驱动力**：现有的文本驱动动作编辑多局限于单人场景，而多人交互场景涉及复杂的时空耦合（如同步、相位对齐、角色切换等）。如何在修改交互语义的同时，严谨地保持未受影响的动作成分是核心挑战。
- **痛点**：缺乏针对多人交互的基准数据集，且现有生成模型难以显式解耦“变更部分”与“保持部分”，导致编辑时出现全局漂移或交互一致性破坏。
- **研究假设**：通过在扩散模型中引入专门的“语义计划令牌”来引导高层意图，并利用基于频域的“交互特征令牌”来约束时空同步动力学，可以实现对多人动作的精确、协调编辑。

### 3. 方法设计详解
InterEdit采用基于Transformer的条件扩散框架，关键设计如下：
- **对称交替令牌聚合**：构建时空对称的交替序列表示（将两人动作以因果关系交替排列，并引入角色交换的对称副本），以捕捉两人之间的时序影响和角色动态。
- **语义感知计划令牌对齐**：引入$N_M$个可学习计划令牌，通过自注意力机制与运动令牌交互，并在训练中通过InfoNCE损失使其与预训练的“动作教师”编码器（TMR）输出的目标嵌入对齐，提供高层语义约束。
- **交互感知频率令牌对齐**：
    - **交互信号构造**：将双人动作拆解为“平均信号”（捕捉同步/协同成分）和“差异信号”（捕捉相对/对抗成分）。
    - **频率建模**：对交互信号进行DCT变换，并计算低/中/高频带的能量描述符。
    - **频率引导**：将这些能量特征映射为频率控制令牌，并要求模型回归目标的频域分布，从而显式约束交互节奏、步调和同步性。

### 4. 方法对比分析
- **本质区别**：与传统编辑方法不同，InterEdit通过频域约束直接作用于“交互动力学”，而非仅仅关注空间上的位置变化。
- **创新贡献**：
    1. 提出了首个大规模多人动作编辑基准InterEdit3D。
    2. 将时空交互分解为频域描述符进行显式监督，有效解决了多人协同的同步难题。
    3. 引入基于对比学习（InfoNCE）的计划令牌对齐，确保了编辑指令的高保真度。

### 5. 实验分析（精简版）
- **关键结论**：在定量指标（g2t, g2s, FID）上全面超越了现有的单人编辑器及多人生成基线。
- **优势**：在复杂的交互（如“交替踢腿”转为“同步踢腿”）中表现出极强的同步控制能力，同时极小化了对未编辑动作成分的破坏。
- **局限**：在极端复杂的长序列舞动交互中，偶尔会出现细微的角色漂移，这反映了长时序关系建模的潜在瓶颈。

### 6. 实用指南
- **开源情况**：已开源，代码与数据集发布在 https://github.com/YNG916/InterEdit。
- **实现关键**：
    - 预训练一个高质量的动作教师编码器对计划令牌的对齐至关重要。
    - 频域能量 Pooling 窗口（$r_{low}, r_{mid}, r_{high}$）设置需根据动作数据的采样率进行微调。
- **迁移可能**：频率令牌对齐策略可直接迁移至机器人多智能体协同控制，或多人的动作序列预测任务中。

### 7. 总结
- **核心思想**：通过语义计划令牌与时空频域约束，实现对多人动作交互的高精细度、可控编辑。
- **速记版pipeline**：
    1. 构建双人动作的时空对称交替表示；
    2. 引入计划令牌引导语义层面的编辑目标；
    3. 使用频域能量映射约束两人间的同步动力学；
    4. 执行条件扩散预测最终的目标动作。

**Key Findings:**

- We introduce the task of multi-person 3D motion editing, where a target motion is generated from a source and a text instruction.
- To support this, we propose InterEdit3D, a new dataset with manual two-person motion change annotations, and a Text-guided Multi-human Motion Editing (TMME) benchmark.
- We present InterEdit, a synchronized classifier-free conditional diffusion model for TMME.
- Experiments show that InterEdit improves text-to-motion consistency and edit fidelity, achieving state-of-the-art TMME performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13082v1)
- [arXiv](https://arxiv.org/abs/2603.13082v1)

---

<a id='2603.13070v1'></a>
## [Mitigating Memorization in Text-to-Image Diffusion via Region-Aware Prompt Augmentation and Multimodal Copy Detection](https://arxiv.org/abs/2603.13070v1)

**Authors:** Yunzhuo Chen, Jordan Vice, Naveed Akhtar, Nur Al Hasan Haldar, Ajmal Mian

**Published:** 2026-03-13

**Categories:** cs.CV

**Abstract:**

State-of-the-art text-to-image diffusion models can produce impressive visuals but may memorize and reproduce training images, creating copyright and privacy risks. Existing prompt perturbations applied at inference time, such as random token insertion or embedding noise, may lower copying but often harm image-prompt alignment and overall fidelity. To address this, we introduce two complementary methods. First, Region-Aware Prompt Augmentation (RAPTA) uses an object detector to find salient regions and turn them into semantically grounded prompt variants, which are randomly sampled during training to increase diversity, while maintaining semantic alignment. Second, Attention-Driven Multimodal Copy Detection (ADMCD) aggregates local patch, global semantic, and texture cues with a lightweight transformer to produce a fused representation, and applies simple thresholded decision rules to detect copying without training with large annotated datasets. Experiments show that RAPTA reduces overfitting while maintaining high synthesis quality, and that ADMCD reliably detects copying, outperforming single-modal metrics.

**Analysis:**

### 1. 摘要翻译
现代文本转图像扩散模型虽然表现出色，但可能记忆并复现训练图像，导致版权和隐私风险。现有的推理时提示词扰动方法虽然能降低复制概率，却往往损害了提示词与图像的对齐度及整体质量。为此，我们提出了两种互补的方法：首先是**区域感知提示词增强 (RAPTA)**，利用目标检测器定位显著区域，并将其转化为语义扎实的提示词变体，在训练过程中随机采样以增加多样性，同时保持语义对齐。其次是**注意力驱动的多模态复制检测 (ADMCD)**，它通过轻量级Transformer聚合局部补丁、全局语义和纹理特征，生成融合表示，并应用阈值决策规则进行检测，无需大规模标注数据集训练。实验表明，RAPTA在保持高质量合成的同时减少了过拟合，而ADMCD在复制检测上表现稳健，优于单一模态的度量标准。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决生成式模型在训练阶段过度拟合导致的“复制粘贴”式记忆问题，并建立一套无需昂贵标注即可检测这种记忆行为的可靠标准。
*   **现有方法痛点**：
    *   **推理时扰动**：如随机插入噪声或词汇，虽然干扰了复制路径，但破坏了提示词的语义一致性。
    *   **检测指标单一**：传统SSIM、LPIPS或CLIP相似度在面对几何变形或风格化差异时鲁棒性差，难以区分精确复制与风格相似。
*   **研究假设**：复制行为源于模型对“图片-描述”强关联的过拟合；通过训练时引入空间结构化的提示词多样性，可以打破模型对单一Caption的锚定；通过多模态特征融合，可以构建更具抗干扰能力的复制检测器。

---

### 3. 方法设计详解
#### **流程总结**：
1.  **RAPTA (训练阶段)**：
    *   **目标检测**：对每张训练图执行Faster R-CNN，获取显著对象的框、类别和置信度。
    *   **位置数字化**：将框中心归一化至3x3网格（共9个空间位置）。
    *   **模板增强**：基于原提示词 `p` 和检测结果 `(c, pos)`，利用预设模板生成变体池 `V`。
    *   **动态采样**：使用CLIP计算变体与原图的相似度，作为采样概率权重，每轮训练随机选择一个变体参与条件扩散。
2.  **ADMCD (检测阶段)**：
    *   **多流提取**：同时提取图像的ViT补丁特征（局部几何）、CLIP向量（全局语义）和ResNet纹理描述符。
    *   **注意力融合**：将三者送入轻量级Transformer，输出ℓ2归一化的融合向量 `f_fus`。
    *   **分级决策**：先通过 `Sfus` 阈值 `τ1` 判断是否为复制，若标记为复制，再基于加权分数 `S_bar` 的阈值 `τ2` 区分“精确/提取型复制”与“风格化 mimicry”。

---

### 4. 方法对比分析
*   **本质区别**：与仅在推理端打补丁的方法不同，RAPTA从**数据增强视角**在训练阶段注入结构化多样性，ADMCD则从**多传感器融合视角**解决检测鲁棒性问题。
*   **创新贡献**：提出了“位置-对象”感知提示词增强策略；构建了首个利用Transformer融合多尺度特征的轻量级复制检测框架。
*   **适用场景**：适用于对生成内容版权合规性要求极高的企业级或商业化扩散模型部署环境。

---

### 5. 实验分析（精简版）
*   **关键结果**：RAPTA在三个基准模型上显著降低了复制率（平均降幅高达50%以上），同时FID和KID指标保持稳健。
*   **主要优势**：ADMCD在应对缩放、旋转、遮挡等10种常见攻击时，相似度得分波动极小，具备极强的零样本泛化能力。
*   **主要局限**：依赖预训练的物体检测器，检测器的泛化能力限制了RAPTA在稀有对象类别上的表现。

---

### 6. 实用指南
*   **实现细节**：`τ1` 设为 0.938，`τ2` 设为 0.970。加权和权重为：`wvis=0.24, wclip=0.38, wtex=0.38`。
*   **迁移可能**：ADMCD的融合框架极易迁移到图像检索、版权追溯等其他任务中。只需更换基础Encoder即可。

---

### 7. 总结
*   **核心思想**：通过空间化提示词增强注入多样性，并利用多模态注意力融合实现稳健的复制检测。
*   **速记版pipeline**：
    1.  用检测器提取图中的物体与位置。
    2.  根据位置生成多种描述提示词并随机训练。
    3.  融合几何、语义、纹理特征进行相似度比对。
    4.  通过阈值分级区分是直接复制还是风格模仿。

**Key Findings:**

- State-of-the-art text-to-image diffusion models can produce impressive visuals but may memorize and reproduce training images, creating copyright and privacy risks.
- To address this, we introduce two complementary methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.13070v1)
- [arXiv](https://arxiv.org/abs/2603.13070v1)

---

