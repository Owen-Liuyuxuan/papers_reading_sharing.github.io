time: 20260504

# Arxiv Computer Vision Papers - 2026-05-04

## Executive Summary

以下是针对2026年5月1日Arxiv计算机视觉领域10篇论文的每日报告执行摘要：

---

### 一、主要主题与趋势

本日论文集中体现了以下核心趋势：

1. **从感知到行动的闭环学习**：多篇论文（如#1、#7、#9）关注让视觉模型不仅“看懂”世界，还能通过强化学习、自我蒸馏或条件执行来实现具身智能和GUI交互，反映出计算机视觉向自主决策系统演进的趋势。
2. **生成模型的精度与效率平衡**：流匹配、扩散先验、KV缓存轻量化等方向（#2、#6、#8）持续推动生成式视觉模型在保真度与计算成本间的优化。
3. **多模态融合与统一框架**：CLIP、LVLM等模型被用于增强视觉任务（#3、#4、#5），同时出现统一视频生成框架（#8），体现多模态一体化的研究热情。
4. **跨域与跨源对齐**：从GUI元素到LiDAR点云，跨模态、跨传感器的特征对齐与位姿精化（#9、#10）仍是实际部署中的重要挑战。

---

### 二、特别重要或有创新性的论文

- **#1 “Learning while Deploying”**：首次提出在机器人部署过程中以车队规模进行在线强化学习，打破“先训后部署”的范式，对通用机器人策略的实用性意义重大。
- **#2 “Posterior Augmented Flow Matching”**：提出一种新的生成建模方法，通过后验增强流匹配，在图像生成质量和采样速度上均表现出明显优势，有望成为扩散模型的有力替代。
- **#4 “Let ViT Speak”**：将视觉Transformer直接与语言生成预训练结合，放弃传统的对比学习范式，开创了“视觉说语言”的新路径，对多模态预训练有潜在颠覆性。
- **#3 “Persistent Visual Memory”**：为大视觉语言模型引入持久视觉记忆机制，使生成过程能够持续感知上下文，显著提升长序列生成的一致性。

---

### 三、新兴研究方向与技术

- **在线策略自我蒸馏（#9）**：通过模型自身的动作预测进行自我纠偏，降低对人工标注的依赖，是GUI智能体训练的低成本新范式。
- **MoE + CLIP 的上下文感知（#5）**：混合专家模型与CLIP结合用于目光估计，展现了大规模预训练模型在细粒度物理交互任务中的适应能力。
- **跨源LiDAR点云配准（#10）**：针对空中与地面LiDAR点云的高度分层配准方法，解决了不同传感器密度与视角下的位姿精化难题，对自动驾驶与测绘有直接应用价值。
- **KV缓存轻量化（#6）**：面向大规模视觉语言模型的推理加速技术，为移动端或实时系统部署大型模型提供实际解决方案。

---

### 四、推荐精读的论文

| 优先级 | 论文 | 理由 |
|--------|------|------|
| ⭐⭐⭐ | #1 “Learning while Deploying” | 范式转变，实用性极高，适合机器人、具身智能研究者 |
| ⭐⭐⭐ | #2 “Posterior Augmented Flow Matching” | 生成模型新方向，适合图像生成、概率建模研究者 |
| ⭐⭐⭐ | #4 “Let ViT Speak” | 多模态预训练新思路，对NLP与视觉交叉领域启发大 |
| ⭐⭐ | #3 “Persistent Visual Memory” | 长序列生成一致性关键突破，适合LVLM与视频理解 |
| ⭐⭐ | #8 “UniVidX” | 统一视频生成框架，适合多媒体与生成任务 |
| ⭐ | #9 “Learn where to Click from Yourself” | 轻量级GUI智能体训练方法，适合交互式AI |

---

## Table of Contents

1. [Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies](#2605.00416v1)
2. [Posterior Augmented Flow Matching](#2605.00825v1)
3. [Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs](#2605.00814v1)
4. [Let ViT Speak: Generative Language-Image Pre-training](#2605.00809v1)
5. [GMGaze: MoE-Based Context-Aware Gaze Estimation with CLIP and Multiscale Transformer](#2605.00799v1)
6. [Make Your LVLM KV Cache More Lightweight](#2605.00789v1)
7. [Affordance Agent Harness: Verification-Gated Skill Orchestration](#2605.00663v1)
8. [UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors](#2605.00658v1)
9. [Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding](#2605.00642v1)
10. [Paired-CSLiDAR: Height-Stratified Registration for Cross-Source Aerial-Ground LiDAR Pose Refinement](#2605.00634v1)

---

## Papers

<a id='2605.00416v1'></a>
## [Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies](https://arxiv.org/abs/2605.00416v1)

**Authors:** Yi Wang, Xinchen Li, Pengwei Xie, Pu Yang, Buqing Nie, Yunuo Cai, Qinglin Zhang, Chendi Qu, Jeffrey Wu, Jianheng Song, Xinlin Ren, Jingshun Huang, Mingjie Pan, Siyuan Feng, Zhi Chen, Jianlan Luo

**Published:** 2026-05-01

**Categories:** cs.RO

**Abstract:**

Generalist robot policies increasingly benefit from large-scale pretraining, but offline data alone is insufficient for robust real-world deployment. Deployed robots encounter distribution shifts, long-tail failures, task variations, and human correction opportunities that fixed demonstration datasets cannot fully capture. We present Learning While Deploying (LWD), a fleet-scale offline-to-online reinforcement learning framework for continual post-training of generalist Vision-Language-Action (VLA) policies. Starting from a pretrained VLA policy, LWD closes the loop between deployment, shared physical experience, policy improvement, and redeployment by using autonomous rollouts and human interventions collected across a robot fleet. To stabilize learning from heterogeneous, sparse-reward fleet data, LWD combines Distributional Implicit Value Learning (DIVL) for robust value estimation with Q-learning via Adjoint Matching (QAM) for policy extraction in flow-based VLA action generators. We validate LWD on a fleet of 16 dual-arm robots across eight real-world manipulation tasks, including semantic grocery restocking and 3--5 minute long-horizon tasks. A single generalist policy improves as fleet experience accumulates, reaching an average success rate of 95%, with the largest gains on long-horizon tasks.

**Analysis:**

以下是对论文《Learning while Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies》的深入分析：

### 1. 摘要翻译
通用机器人策略日益受益于大规模预训练，但仅靠离线数据不足以实现鲁棒的现实部署。部署后的机器人会遇到分布偏移、长尾失效和人类修正机会，这些是固定的演示数据集无法完全捕获的。我们提出了“部署中学习”（Learning While Deploying, LWD），这是一个用于通用视觉-语言-动作（VLA）策略持续后训练的舰队规模离线到在线强化学习框架。LWD利用机器人舰队收集的自主回放和人类干预，闭环连接了部署、物理经验共享、策略改进和再部署。为了从异构、稀疏奖励的舰队数据中稳定学习，LWD结合了用于鲁棒价值估计的分布隐式价值学习（DIVL），以及用于流式VLA动作生成器中策略提取的伴随匹配Q学习（QAM）。我们在16台双臂机器人的舰队上验证了LWD，涵盖了包括语义杂货补货和3-5分钟长周期任务在内的八项现实操纵任务。单一通用策略随着舰队经验的积累而提升，平均成功率达到95%，在长周期任务上增益尤为显著。

### 2. 方法动机分析
- **驱动力**：旨在将机器人部署从单纯的“测试”转变为“数据生成”，通过持续的物理交互实现策略的自我进化。
- **痛点**：现有的离线预训练对真实世界分布偏移适应性差；在线强化学习通常只针对单一任务，且无法高效利用大规模异构离线数据集；在线微调往往会导致策略崩溃。
- **研究假设**：通过分布式的舰队部署收集异构经验，利用一致的离线到在线强化学习目标，可以实现通用策略的稳健、持续后训练。

### 3. 方法设计详解
- **流程总结**：
    1. **离线预训练**：在静态离线Buffer（包含演示、历史回放、人类指导的故障数据）上预训练VLA策略和DIVL价值函数。
    2. **在线部署与交互**：部署策略至机器人舰队，自动收集交互数据，并在有人类干预时记录修正片段。
    3. **混合重放学习**：在线数据不断加入Buffer，利用包含离线和在线数据的混合Buffer进行同步更新。
    4. **再部署**：定期将更新后的策略权重推送到舰队。
- **关键模块**：
    - **DIVL (Distributional Implicit Value Learning)**：用分布估计替代标量估计，解决异构数据下回报分布的多模态特性，利用τ分位数作为目标，实现隐式、鲁棒的价值学习。
    - **QAM (Q-learning with Adjoint Matching)**：将 critic 的价值梯度转化为流匹配策略的局部监督信号，解决了直接通过多步去噪流程反向传播价值梯度的不稳定性。
- **算法精髓**：公式 (19) 的n步Chunk-level TD目标是冷启动价值函数的关键，通过 n=10 传播稀疏奖励，解决长周期任务学习慢的问题。

### 4. 方法对比分析
- **本质区别**：从传统的“离线-预训练-再在线微调”脱节流程，转向了“统一优化目标下的端到端数据飞轮”模式。
- **创新点**：首次将分布强化学习（DIVL）与基于流的策略优化（QAM）耦合于大规模分布式舰队系统中。
- **适用场景**：需要部署通用机器人策略，且能够容忍长周期、多任务环境下稀疏奖励的复杂操纵场景。

### 5. 实验分析
- **验证方法**：在16台G1机器人上，对4个长周期（如做茶、鸡尾酒）和4个杂货补货任务进行对比实验。
- **关键结论**：LWD(Online) 达到 0.95 的平均成功率，对比预训练基线有巨大提升，且在长周期任务上显著优于 RECAP 等基线。
- **主要优势**：极强的样本效率，仅需数小时在线交互即可获得显著增益；分布价值学习显著缓解了多模态数据下的过拟合。
- **主要局限**：在线更新调度较为直接，缺乏针对超大规模部署的资源调度优化；未显式建模执行安全性。

### 6. 实用指南
- **复现关键**：重点在于分布式基础设施的搭建（图10），必须确保同步机制（Snapshot-bound view）的鲁棒性。
- **实现细节**：建议严格参考作者的归一化熵 schedule (Eq. 17-18) 进行动态τ调整，这是实现稳定学习的关键超参数。
- **迁移性**：DIVL 和 QAM 模块可独立迁移到任何基于流匹配（Flow Matching）的VLA架构中，作为其强化学习后训练的核心组件。

### 7. 总结
- **核心思想**：通过分布式数据飞轮，实现通用机器人策略的持续自主进化。
- **速记版pipeline**：
    1. 预训练基准策略；
    2. 舰队部署并搜集实时交互与人类修正；
    3. 利用分布价值模型评估稀疏奖励；
    4. 通过梯度匹配更新策略；
    5. 循环迭代模型。

**Key Findings:**

- We present Learning While Deploying (LWD), a fleet-scale offline-to-online reinforcement learning framework for continual post-training of generalist Vision-Language-Action (VLA) policies.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00416v1)
- [arXiv](https://arxiv.org/abs/2605.00416v1)

---

<a id='2605.00825v1'></a>
## [Posterior Augmented Flow Matching](https://arxiv.org/abs/2605.00825v1)

**Authors:** George Stoica, Sayak Paul, Matthew Wallingford, Vivek Ramanujan, Abhay Nori, Winson Han, Ali Farhadi, Ranjay Krishna, Judy Hoffman

**Published:** 2026-05-01

**Categories:** cs.CV

**Abstract:**

Flow matching (FM) trains a time-dependent vector field that transports samples from a simple prior to a complex data distribution. However, for high-dimensional images, each training sample supervises only a single trajectory and intermediate point, yielding an extremely sparse and high-variance training signal. This under-constrained supervision can cause flow collapse, where the learned dynamics memorize specific source-target pairings, mapping diverse inputs to overly similar outputs, failing to generalize. We introduce Posterior-Augmented Flow Matching (PAFM), a theoretically grounded generalization of FM that replaces single-target supervision with an expectation over an approximate posterior of valid target completions for a given intermediate state and condition. PAFM factorizes this intractable posterior into (i) the likelihood of the intermediate under a hypothesized endpoint and (ii) the prior probability of that endpoint under the condition, and uses an importance sampling scheme to construct a mixture over multiple candidate targets. We prove that PAFM yields an unbiased estimator of the original FM objective while substantially reducing gradient variance during training by aggregating information from many plausible continuation trajectories per intermediate. Finally, we show that PAFM improves over FM by up to 3.4 FID50K across different model scales (SiT-B/2 and SiT-XL/2), different architectures (SiT and MMDiT), and in both class and text conditioned benchmarks (ImageNet and CC12M), with a negligible increase in the compute overhead. Code: https://github.com/gstoica27/PAFM.git.

**Analysis:**

# 论文分析：Posterior Augmented Flow Matching (PAFM)

### 1. 摘要翻译
流匹配（FM）通过训练随时间变化的向量场，将样本从简单的先验分布映射到复杂的真实数据分布。然而在处理高维图像时，每个训练样本仅监督单一轨迹和中间点，这导致了极端稀疏且高方差的训练信号。这种约束不足的监督可能引发“流崩溃”（flow collapse），使模型过度拟合特定的源-目标对，缺乏泛化能力。我们提出了“后验增强流匹配”（PAFM），这是对FM的一种理论支撑更强的推广。PAFM用给定中间状态和条件下的有效目标补全的近似后验期望，取代了单目标监督。PAFM将这一难以处理的后验分解为：(i) 给定假设端点下中间状态的似然，以及 (ii) 给定条件下端点的先验概率，并利用重要性采样方案构建多个候选目标的混合。我们证明，PAFM是原始FM目标的无偏估计，同时在训练中通过聚合多个合理的延续轨迹信息，显著降低了梯度方差。最终实验表明，PAFM在不同模型规模（SiT-B/2, SiT-XL/2）和架构（SiT, MMDiT）下，以及在ImageNet和CC12M基准测试中，FID50K指标提升高达3.4，且计算开销极小。

### 2. 方法动机分析
*   **驱动力**：旨在解决流匹配在高维数据训练中的稀疏监督问题，提升生成模型对连续分布的学习能力，防止模型塌陷。
*   **现有方法痛点**：FM 在训练时采用“一对一”配对策略，即一个噪声点仅对应一个目标图像。在高维空间中，这只是无数个合理轨迹中的一条，导致训练信号非常稀疏且方差巨大。
*   **研究假设**：与其被迫学习唯一的、随机采样的轨迹，不如将该中间点视为通往多个合理目标分布的交汇点，通过学习所有合理目标的“加权平均”方向，可以得到更平滑、更鲁棒的流场。

### 3. 方法设计详解
*   **流程总结**：
    1.  **目标集合构建**：对于当前的中间点 $z_t^i$，获取一组候选目标 $\{z^j\}_{j=1}^K$（可采用K近邻检索、数据增强或VAE采样）。
    2.  **权重计算**：计算每个目标的权重 $w_j = p_t(z_t^i|z^j) \cdot p(y^i|z^j)$。前者基于高斯分布度量与中间点的几何邻近度，后者度量与语义条件的对齐程度（如CLIP分数或分类标签匹配）。
    3.  **目标聚合**：将训练目标从单一的 $v(z_t^i|z^i)$ 修改为多目标的加权和 $\sum_{j=1}^K w_j v(z_t^i|z^j)$。
    4.  **梯度更新**：模型根据上述加权后的速度场进行监督学习。
*   **核心算法**：利用 **Self-Normalized Importance Sampling (SNIS)** 实现了对不可直接采样的后验分布的近似，确保了目标函数的无偏性。
*   **数学意义**：将原有的点监督扩展为分布监督，本质上是在进行一种平滑操作，通过K个目标显著降低了梯度估计的波动。

### 4. 方法对比分析
*   **本质区别**：从传统的“点对点”监督变成了“点对分布”的期望监督。
*   **创新贡献**：在不改变网络架构的情况下，通过巧妙的损失函数重构，赋予模型处理高方差、稀疏监督信号的能力，理论证明了其能通过增加有效样本量（ESS）来降低梯度方差。
*   **适用场景**：适用于所有流匹配（FM/Rectified Flow）框架，特别是在数据多样性丰富但训练容易陷入局部最优的生成任务中。

### 5. 实验分析
*   **关键结论**：在ImageNet-1K（类条件）和CC12M（文本条件）上均取得显著性能提升，证明了该方法具备良好的通用性。
*   **主要优势**：显著降低了梯度方差，大幅减少了流崩溃现象，训练过程更稳定，且引入的计算开销极低（仅增加少量预处理或缓存读取）。
*   **主要局限**：性能高度依赖于候选目标集合 $\{z^j\}$ 的构建策略，若候选目标质量不高，反而可能引入噪声。

### 6. 实用指南
*   **开源情况**：代码已开源（https://github.com/gstoica27/PAFM.git）。
*   **实现细节**：
    *   推荐在数据预处理阶段通过FAISS预计算K近邻，避免在线搜索带来的耗时。
    *   在小数据集或高稀疏场景下，增大 $K$ 值可带来更明显的训练稳定性。
*   **迁移可能**：该方法极易迁移到任何基于ODE的流匹配模型中，作为通用的训练增强插件。

### 7. 总结
*   **核心思想**：通过聚合多个合理目标，将稀疏监督转化为平滑的分布期望。
*   **速记版pipeline**：
    1. 选出多个相似目标；
    2. 计算每个目标的合理度权重；
    3. 加权求和得到目标流向；
    4. 监督模型向该期望方向优化。

**Key Findings:**

- We introduce Posterior-Augmented Flow Matching (PAFM), a theoretically grounded generalization of FM that replaces single-target supervision with an expectation over an approximate posterior of valid target completions for a given intermediate state and condition.
- Finally, we show that PAFM improves over FM by up to 3.4 FID50K across different model scales (SiT-B/2 and SiT-XL/2), different architectures (SiT and MMDiT), and in both class and text conditioned benchmarks (ImageNet and CC12M), with a negligible increase in the compute overhead.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00825v1)
- [arXiv](https://arxiv.org/abs/2605.00825v1)

---

<a id='2605.00814v1'></a>
## [Persistent Visual Memory: Sustaining Perception for Deep Generation in LVLMs](https://arxiv.org/abs/2605.00814v1)

**Authors:** Siyuan Huang, Xiaoye Qu, Yafu Li, Tong Zhu, Zefeng He, Muxin Fu, Daizong Liu, Wei-Long Zheng, Yu Cheng

**Published:** 2026-05-01

**Categories:** cs.CV, cs.AI

**Abstract:**

While autoregressive Large Vision-Language Models (LVLMs) demonstrate remarkable proficiency in multimodal tasks, they face a "Visual Signal Dilution" phenomenon, where the accumulation of textual history expands the attention partition function, causing visual attention to decay inversely with generated sequence length. To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception. Integrated as a parallel branch alongside the Feed-Forward Network (FFN) in LVLMs, PVM establishes a distance-agnostic retrieval pathway that directly provides visual embeddings for precise visual perception, thereby structurally mitigating the signal suppression inherent to deep generation. Extensive experiments on Qwen3-VL models demonstrate that PVM brings notable improvements with negligible parameter overhead, delivering consistent average accuracy gains across both 4B and 8B scales, particularly in complex reasoning tasks that demand persistent visual perception. Furthermore, in-depth analysis reveals that PVM can resist length-induced signal decay and accelerate internal prediction convergence.

**Analysis:**

### 1. 摘要翻译
尽管自回归大型视觉语言模型（LVLM）在多模态任务中表现出色，但它们面临“视觉信号稀释”现象，即文本历史的积累扩大了注意力分配函数，导致视觉注意力随生成序列长度增加而衰减。为解决这一问题，我们提出了持久视觉记忆（PVM），这是一个轻量级可学习模块，旨在确保持久、按需的视觉感知。PVM作为LVLM中前馈网络（FFN）的并行分支集成，建立了一条与距离无关的检索路径，直接提供视觉嵌入以实现精确的视觉感知，从而从结构上缓解了深度生成中固有的信号抑制问题。在Qwen3-VL模型上的广泛实验表明，PVM在参数开销可忽略的情况下带来了显著的改进，在4B和8B规模上均实现了持续的平均准确率提升，特别是在需要持久视觉感知的复杂推理任务中。此外，深入分析表明，PVM能够抵御长度诱导的信号衰减并加速内部预测收敛。

### 2. 方法动机分析
- **驱动力**：作者发现随着文本序列增长，注意力机制的归一化过程使得有限的视觉信息被海量文本信息淹没（Visual Signal Dilution），导致模型在长文本生成中无法保持视觉忠实度。
- **现有方法痛点**：现有的“视觉注入”（Visual Injection）方法（如插入原始token或融合特征）通常会打断原有自回归路径的语义一致性，或在强行强化视觉存在感的同时，扰动了复杂推理所需的逻辑状态。
- **研究假设**：通过在Transformer模块中构建一个与FFN并行的、独立的视觉检索分支，可以将视觉信息的读取与文本的生成逻辑解耦，从而在不干扰主干逻辑的情况下实现持久的视觉感知。

### 3. 方法设计详解
- **架构设计**：PVM被设计为Transformer块中FFN的并行分支。
  - **推理路径（原分支）**：保持原有的FFN处理逻辑。
  - **查看路径（PVM分支）**：利用隐藏状态 $x$ 作为Query，通过交叉注意力（Cross-Attention）检索视觉嵌入。
- **计算流程**：
  1. **投影（Projection）**：将输入隐藏状态和视觉特征通过线性层映射到低维潜在空间 $d'$，以提升效率并过滤冗余。
  2. **潜空间检索（Latent Retrieval）**：利用投影后的Query检索投影后的视觉Keys/Values，并通过轻量级FFN进一步精炼。
  3. **恢复与门控（Restoration & Gating）**：将精炼后的特征映射回高维空间，通过一个可学习的标量门 $\lambda$（初始为0）进行缩放，并结合“视觉屏蔽掩码（Silencing Mask）”仅在文本生成步激活，最终通过残差连接融合到输出中。

### 4. 方法对比分析
- **本质区别**：与现有注入方法不同，PVM实现了**独立的注意力归一化**。其分区函数仅在视觉域内进行，不受文本历史长度 $t$ 的干扰，从而从数学上消除了视觉信号随 $t$ 增大而衰减的结构性原因。
- **创新贡献**：提出“持久视觉记忆”架构，通过数学推导证明了该设计满足局部不变性 $\frac{\partial\|h_{pvm}\|}{\partial t} = 0$。
- **适用场景**：极度依赖视觉输入、长文本输出的复杂多模态推理场景。

### 5. 实验分析
- **验证方法**：在8个主流多模态基准（如MMMU, MathVerse, AI2D等）上对比了Qwen3-VL-8B/4B基线。
- **关键结果**：在8B backbone上，PVM（SFT+GRPO）平均准确率提升4.8%；针对“长文本”生成任务，相对基线提升高达27.3%。
- **主要优势**：不仅提升性能，还通过LogitLens分析证实其能加速模型预测的内部收敛。
- **主要局限**：目前的实验仅在Qwen3-VL上进行，且对于极长序列的动态交互（如长视频理解）尚处于初步探索阶段。

### 6. 实用指南
- **开源情况**：代码已开源（https://github.com/huaixuheqing/PVM）。
- **关键细节**：作者推荐采用“分层（Strided）”插入策略（例如8B模型注入在第8, 16, 24层），这比单纯的“峰值注入”效果更好，因为其提供了全局覆盖。Bottleneck维度建议设为512。
- **迁移可能**：该架构与骨干网络解耦，可轻松迁移到其他基于Transformer的视觉语言模型中，只需匹配插入层的维度。

### 7. 总结
- **核心思想**：构建并行视觉记忆路径，通过独立归一化抵消文本历史对视觉信号的干扰。
- **速记版pipeline**：
  1. **并行分支**：在Transformer FF层旁建立专用视觉检索路。
  2. **低维投影**：将视觉和文本特征压缩以高效提取语义。
  3. **静默激活**：仅在生成文本时通过门控机制注入视觉线索。
  4. **残差融合**：将视觉增强后的信息叠加回主推理流。

**Key Findings:**

- To counteract this, we propose Persistent Visual Memory (PVM), a lightweight learnable module designed to ensure sustained, on-demand visual perception.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00814v1)
- [arXiv](https://arxiv.org/abs/2605.00814v1)

---

<a id='2605.00809v1'></a>
## [Let ViT Speak: Generative Language-Image Pre-training](https://arxiv.org/abs/2605.00809v1)

**Authors:** Yan Fang, Mengcheng Lan, Zilong Huang, Weixian Lei, Yunqing Zhao, Yujie Zhong, Yingchen Yu, Qi She, Yao Zhao, Yunchao Wei

**Published:** 2026-05-01

**Categories:** cs.CV

**Abstract:**

In this paper, we present \textbf{Gen}erative \textbf{L}anguage-\textbf{I}mage \textbf{P}re-training (GenLIP), a minimalist generative pretraining framework for Vision Transformers (ViTs) designed for multimodal large language models (MLLMs). To better align vision encoders with the autoregressive nature of LLMs, GenLIP trains a ViT to predict language tokens directly from visual tokens using a standard language modeling objective, without contrastive batch construction or an additional text decoder. This design offers three key advantages: (1) \textbf{Simplicity}: a single transformer jointly models visual and textual tokens; (2) \textbf{Scalability}: it scales effectively with both data and model size; and (3) \textbf{Performance}: it achieves competitive or superior results across diverse multimodal benchmarks. Trained on 8B samples from Recap-DataComp-1B, GenLIP matches or surpasses strong baselines despite using substantially less pretraining data. After continued pretraining on multi-resolution images at native aspect ratios, GenLIP further improves on detail-sensitive tasks such as OCR and chart understanding, making it a strong foundation for vision encoders in MLLMs.

**Analysis:**

### 1. 摘要翻译
本文提出了一种极简的生成式视觉-语言预训练框架——GenLIP，旨在为多模态大语言模型（MLLM）打造视觉编码器。为了更好地将视觉编码器与大语言模型的自回归特性对齐，GenLIP采用标准的语言建模目标，训练视觉Transformer（ViT）直接从视觉标记预测语言标记，无需对比学习批次构建或额外的文本解码器。该设计具有三大优势：（1）简洁：单一Transformer架构联合建模视觉和文本标记；（2）可扩展性：在数据量和模型规模上表现出良好的扩展性；（3）高性能：在多项多模态基准测试中表现卓越。仅使用8B样本训练，GenLIP即达到或超越了更强大的基线模型。此外，通过在原生分辨率图像上的持续预训练，GenLIP在OCR和图表理解等细节敏感任务上显著提升，成为MLLM强有力的视觉底座。

### 2. 方法动机分析
*   **驱动力**：作者追求一种极简、直接的生成式预训练范式，以缓解对比学习（如CLIP）与自回归LLM之间存在的“目标不匹配”问题。
*   **现有方法痛点**：
    *   **对比学习局限**：偏向判别性对齐，与MLLM的生成式目标不兼容。
    *   **架构复杂**：现有生成式方法常使用“编码器-解码器”双塔结构或混合架构，导致计算开销大、优化间接、训练效率低。
    *   **注意力汇聚（Attention Sink）**：在简单的统一架构中，模型倾向于将视觉信息压缩进少数几个“汇聚标记”，导致空间表征能力下降。
*   **研究假设**：通过简化架构并强制视觉标记直接服务于下一标记预测（Let ViT Speak），辅以门控注意力机制缓解信息崩溃，可以以更小的数据量获得更强的视觉编码器。

### 3. 方法设计详解
*   **流程Pipeline**：
    1.  **数据格式**：将图像划分为patch序列 $\{v_0, \dots, v_M\}$，文本标记为 $\{t_0, \dots, t_L\}$，拼接为输入序列 $S = [v_0, \dots, v_M, t_0, \dots, t_L]$。
    2.  **位置编码**：弃用绝对位置编码，改用多模态旋转位置编码（MRoPE）。
    3.  **注意力机制**：使用Prefix-LM注意力（图像双向感知，文本因果掩码），并在Transformer块中引入**门控注意力（Gated Attention）**，通过 $G = \sigma(XW_g + b_g)$ 对注意力输出进行自适应调制，防止注意力过分集中于首个标记。
    4.  **训练目标**：应用标准的自回归语言建模（NLL损失），强制ViT直接预测后续文本标记。
*   **核心模块——门控注意力**：这不仅是简单的正则化，其核心在于通过输入依赖的门控机制（Element-wise multiplication），动态调节信息流，迫使模型利用空间分布的视觉特征，而非仅仅依赖局部汇聚点，从而稳定了训练并提升了判别能力。

### 4. 方法对比分析
*   **本质区别**：与需要对比损失或额外解码器的方案不同，GenLIP实现了“单塔、单目标、无对比损失”的纯生成式预训练。
*   **创新贡献**：提出门控注意力机制，解决了混合序列建模中常见的Attention Sink导致的判别能力下降问题，使单模型结构在生成和判别任务上达到平衡。
*   **适用场景**：极高的数据效率场景、对推理延迟敏感的MLLM视觉底座构建。

### 5. 实验分析（精简版）
*   **验证方法**：在14个多模态基准（Doc&OCR, VQA, Caption）上进行frozen feature评估。
*   **关键结论**：在仅使用8B训练样本的情况下，其Doc&OCR任务得分显著领先于使用40B样本的SigLIP2；且模型规模扩展性极佳。
*   **优势**：数据效率极高，OCR能力突出，架构极其轻量。
*   **局限**：目前验证基于LLaVA-NeXT框架，在更大规模上的Scaling Law表现仍有待进一步探索。

### 6. 实用指南
*   **开源/实现**：项目主页为`vitspeak`。实现时，关键在于将`flex-attention`应用于Prefix-LM掩码，并在Transformer block中加入由Sigmoid输出的门控矩阵。
*   **实现细节**：建议使用两阶段训练，第一阶段固定低分辨率以快速收敛，第二阶段适配原生宽高比。
*   **迁移建议**：该方法非常适合需要轻量化视觉编码器的任务，只需将ViT输出连接至LLM的投影层（Projector）即可直接迁移。

### 7. 总结
*   **核心思想**：统一视觉与语言序列，以极简自回归目标驱动ViT学习通用视觉表征。
*   **速记版pipeline**：1.构建图文拼接序列；2.引入MRoPE位置编码；3.采用Prefix-LM注意力；4.通过门控机制抑制注意力汇聚现象；5.执行文本生成任务以微调视觉编码。

**Key Findings:**

- In this paper, we present \textbf{Gen}erative \textbf{L}anguage-\textbf{I}mage \textbf{P}re-training (GenLIP), a minimalist generative pretraining framework for Vision Transformers (ViTs) designed for multimodal large language models (MLLMs).

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00809v1)
- [arXiv](https://arxiv.org/abs/2605.00809v1)

---

<a id='2605.00799v1'></a>
## [GMGaze: MoE-Based Context-Aware Gaze Estimation with CLIP and Multiscale Transformer](https://arxiv.org/abs/2605.00799v1)

**Authors:** Xinyuan Zhao, Yihang Wu, Ahmad Chaddad, Sarah A. Alkhodair, Reem Kateb

**Published:** 2026-05-01

**Categories:** cs.CV

**Abstract:**

Gaze estimation methods commonly use facial appearances to predict the direction of a person gaze. However, previous studies show three major challenges with convolutional neural network (CNN)-based, transformer-based, and contrastive language-image pre-training (CLIP)-based methods, including late fusion of image features, lack of factor-aware conditioning, and impractical capacity scaling. To address these challenges, we propose Globally-conditioned Multi-scale Gaze estimation (GMGaze), which leverages a multi-scale transformer architecture. Specifically, the model first introduces semantic prototype conditioning, which modulates the CLIP global image embedding using four learned prototype banks (i.e., illumination, background, head pose and appearance) to generate two complementary context-biased global tokens. These tokens, along with the CLIP patch and CNN tokens, are fused at the first layer. This early unified fusion prevents information loss common in late-stage merging. Finally, each token passes through sparse Mixture-of-Experts modules, providing conditional computational capacity without uniformly increasing dense parameters. For cross-domain adaptation, we incorporate an adversarial domain adaptation technique with a feature separation loss that encourages the two global tokens to remain de-correlated. Experiments using four public benchmarks (MPIIFaceGaze, EYEDIAP, Gaze360, and ETH-XGaze) show that GMGaze achieves mean angular errors of 2.49$^\circ$, 3.22$^\circ$, 10.16$^\circ$, and 1.44$^\circ$, respectively, outperforming previous baselines in all within-domain settings. In cross-domain evaluations, it provides state-of-the-art (SOTA) results on two standard transfer routes.

**Analysis:**

# 论文分析：GMGaze

## 1. 摘要翻译
现有的注视点估计方法通常利用人脸外观来预测注视方向。然而，已有研究表明，基于CNN、Transformer以及对比语言-图像预训练（CLIP）的方法存在三大主要挑战：图像特征的滞后融合、缺乏因素感知调节以及计算能力扩展不便。为解决这些问题，我们提出了全局条件化多尺度注视估计（GMGaze），该架构利用多尺度Transformer，引入了语义原型调节机制，利用四个学习到的原型库（光照、背景、头部姿态和外观）对CLIP全局图像嵌入进行调制，从而生成两个互补的上下文偏置全局令牌。这些令牌与CLIP补丁（patch）和CNN令牌在第一层进行融合，这种早期统一融合避免了后期合并中的信息丢失。最后，每个令牌通过稀疏专家混合（MoE）模块处理，在不均匀增加稠密参数的情况下提供了条件计算能力。为实现跨域适应，我们引入了一种结合特征分离损失的对抗域适应技术，鼓励两个全局令牌保持去相关。在四个公共基准测试上的实验表明，GMGaze在所有域内设置下均优于先前的基线。

## 2. 方法动机分析
- **驱动力**：旨在克服现有模型在不同现实环境（如光照、背景变化）下泛化能力差的问题，同时提升模型在处理复杂场景时的计算效率。
- **现有痛点**：
    1. **特征滞后融合**：多尺度信息在深层才合并，导致细粒度特征丢失。
    2. **信息耦合**：全局表征纠缠了光照、身份等干扰因素，导致注视特征提取不纯。
    3. **计算冗余**：标准Transformer对所有区域分配相同计算资源，缺乏针对性。
- **研究假设**：通过显式的语义条件化（原型库）将全局表征解耦，并利用MoE实现令牌级的动态计算分配，能显著提升模型的鲁棒性与效率。

## 3. 方法设计详解
- **pipeline**：
    1. **语义原型调节**：利用预定义的文本提示（Prompt）生成语义原型库，计算全局CLIP特征与原型的相似度，通过Argmax（配合Straight-Through Estimator, STE）选择特定的语义偏置令牌 $f_1, f_2$。
    2. **多尺度融合**：将上述语义令牌、CLIP的Patch令牌、以及CNN提取的高分辨率局部特征，在第一层进行统一维度的拼接。
    3. **MoE Transformer层**：在每一层Transformer中，用路由-共享的MoE模块取代标准FFN，根据令牌特征动态路由至Top-2专家。
    4. **特征分离**：引入特征分离损失（$L_{sep}$），强制 $f_1$ 和 $f_2$ 向量在空间上正交，保证信息互补。
- **算法解释**：$L_{sep}$ 惩罚两个归一化全局特征向量的余弦相似度，几何上迫使它们向不同方向优化，从而实现语义解耦。

## 4. 方法对比分析
- **本质区别**：从“通用建模”转变为“语义指导下的动态条件建模”。
- **创新贡献**：提出语义原型条件化机制实现特征解耦，并结合Token级MoE实现了计算与表征的双重优化。
- **适用场景**：高动态、多干扰场景（如自动驾驶、户外移动设备）的注视追踪。

## 5. 实验分析
- **验证方法**：在MPIIFaceGaze、EYEDIAP、Gaze360、ETH-XGaze四个基准上进行域内与跨域评估。
- **关键结果**：在大多数数据集上取得了SOTA结果（如ETH-XGaze误差降至1.44°）。
- **优势**：极强的抗干扰能力，计算效率与精度平衡优异。
- **局限**：Argmax离散选择可能限制了对连续性变化（如平滑光照渐变）的建模；对低多样性数据集的泛化效果有待提升。

## 6. 实用指南
- **开源情况**：代码已公开（https://github.com/AIPMLab/GazeFormer-MoE）。
- **迁移可能**：该框架（尤其是原型库+MoE路由）可直接迁移至头部姿态估计或视频行为分析任务。
- **实现细节**：注意STE的应用确保反向传播可行；专家Dropout设置（0.2）对防止过拟合至关重要。

## 7. 总结
- **核心思想**：通过语义原型引导和令牌级专家路由实现解耦表征与动态计算。
- **速记版pipeline**：
    1. 提示词生成原型库；
    2. 相似度匹配选取条件化令牌；
    3. 拼接语义、Patch和CNN三类特征；
    4. 令牌经由稀疏MoE网络深度处理并回归方向。

**Key Findings:**

- To address these challenges, we propose Globally-conditioned Multi-scale Gaze estimation (GMGaze), which leverages a multi-scale transformer architecture.
- In cross-domain evaluations, it provides state-of-the-art (SOTA) results on two standard transfer routes.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00799v1)
- [arXiv](https://arxiv.org/abs/2605.00799v1)

---

<a id='2605.00789v1'></a>
## [Make Your LVLM KV Cache More Lightweight](https://arxiv.org/abs/2605.00789v1)

**Authors:** Xihao Chen, Yangyang Guo, Roger Zimmermann

**Published:** 2026-05-01

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Key-Value (KV) cache has become a de facto component of modern Large Vision-Language Models (LVLMs) for inference. While it enhances decoding efficiency in Large Language Models (LLMs), its direct adoption in LVLMs introduces substantial GPU memory overhead due to the large number of vision tokens processed during the prefill stage. To tackle this problem, we propose LightKV, a novel approach that reduces KV cache size by exploiting the redundancy among vision-token embeddings. Guided by text prompts, LightKV employs cross-modality message passing to aggregate informative messages across vision tokens and progressively compress them during prefill. This prompt-aware guidance distinguishes our method from prior vision-only compression strategies. We evaluate LightKV on eight open-source LVLMs across eight public benchmark datasets, e.g., MME and SeedBench. Experimental results demonstrate that with only 55% of the original vision tokens, LightKV (a) halves the vision-token KV cache size, (b) reduces computation by up to 40%, and (c) preserves general-purpose performance while significantly outperforming existing baselines.

**Analysis:**

### 1. 摘要翻译
键值（KV）缓存已成为现代大型视觉语言模型（LVLMs）推理中不可或缺的组件。虽然它增强了大型语言模型（LLMs）的解码效率，但在LVLMs中直接使用会因预填充阶段处理大量视觉标记（tokens）而产生显著的GPU内存开销。为了解决这一问题，我们提出了LightKV，这是一种通过利用视觉标记嵌入之间的冗余来减小KV缓存大小的新颖方法。在文本提示的引导下，LightKV采用跨模态消息传递在不同视觉标记间聚合信息，并在预填充阶段逐步压缩它们。这种提示感知的引导方式使我们的方法区别于以往仅针对视觉信息的压缩策略。我们在八个数据集上的八个开源LVLM模型上评估了LightKV。实验结果表明，在仅保留55%原始视觉标记的情况下，LightKV (a) 将视觉标记的KV缓存大小减半，(b) 将计算量减少高达40%，且 (c) 在保持通用性能的同时显著优于现有基线。代码已开源。

### 2. 方法动机分析
*   **驱动力**：LVLMs在预填充阶段面临极高的内存压力，主要源于图像/视频被分解为数以千计的视觉Token，导致KV缓存体积臃肿，限制了上下文长度和推理效率。
*   **现有方法痛点**：基于训练的方法（如GQA）需要昂贵的重训练成本；而现有的推理时剪枝方法（如FastV）多采用视觉单模态视角，忽视了文本提示对视觉特征重要性的调控作用，导致跨模态推理信息丢失。
*   **研究假设**：视觉Token之间存在高度冗余，且并非所有Token对模型生成结果同等重要。利用跨模态注意力机制将提示信息融入压缩过程，能更精准地保留关键视觉特征，实现更优的压缩比与性能平衡。

### 3. 方法设计详解
LightKV的流程分为三个阶段：
1.  **分窗口图构造（Intra-window Graph Construction）**：将图像视觉Token划分为 $w \times w$ 的不重叠窗口，在每个窗口内将Token视为图节点，按奇偶索引构建二分图，并计算特征发散度（FD）以衡量节点间的相似性。
2.  **提示感知的消息传递（Cross-modal Message Passing）**：利用冻结模型预填充阶段已有的注意力分数（$\xi$），量化每个视觉Token与文本提示的相关性。随后，将A组节点的信息聚合到B组，并结合提示权重对信息进行加权更新。
3.  **分层压缩（Hierarchical Compression）**：在不同Decoder层级中重复上述过程。随着网络深度增加，动态调整窗口大小（$w$），实现从局部语义聚合向全局语义整合的过渡，逐步降低Token序列长度。

### 4. 方法对比分析
*   **本质区别**：与仅基于图像相似度合并（如ToMe）不同，LightKV引入了跨模态信号，确保“留下的Token”是与当前任务提示最相关的视觉核心信息。
*   **创新贡献**：提出了一种提示引导的动态图压缩方案，无需训练，即可在推理过程中实现高效的KV缓存缩减。
*   **适用场景**：适用于资源受限环境下、处理高分辨率或长序列输入的Decoder-only LVLM模型。

### 5. 实验分析
*   **关键结果**：在LLaVA系列及InternVL等模型上，保留55%的视觉Token，模型平均性能损失极小（甚至在某些任务上略有提升），同时FLOPs减少40%。
*   **优势**：训练免费，插件式部署，兼容各种Backbone。
*   **局限**：由于使用了二分图匹配算法，每轮迭代最大压缩率为50%，需多轮迭代才能实现更高压缩比；显式计算注意力矩阵与FlashAttention存在一定的兼容性冲突（需在压缩层退回至eager计算）。

### 6. 实用指南
*   **开源地址**：[https://github.com/howtoosee/LightKV](https://github.com/howtoosee/LightKV)
*   **迁移建议**：该方法逻辑通用，可直接应用于任何支持交叉注意力（Cross-Attention）机制的LVLM。只需配置压缩层索引 $\Lambda$ 和窗口大小 $W$，即可实现即插即用。
*   **注意事项**：对超参数 $W$（窗口大小）敏感，对于高分辨率输入（Token数多），建议采用较大的窗口设置；压缩层不宜过深，否则无法有效减少预填充计算压力。

### 7. 总结
*   **核心思想**：利用文本提示引导视觉Token间的图信息聚合与压缩。
*   **速记版pipeline**：
    1.  图像切块并转化为节点二分图；
    2.  利用模型内置注意力计算Token与文本的相关性；
    3.  通过图消息传递合并冗余Token；
    4.  在不同层级循环上述操作以逐步缩小序列。

**Key Findings:**

- To tackle this problem, we propose LightKV, a novel approach that reduces KV cache size by exploiting the redundancy among vision-token embeddings.
- This prompt-aware guidance distinguishes our method from prior vision-only compression strategies.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00789v1)
- [arXiv](https://arxiv.org/abs/2605.00789v1)

---

<a id='2605.00663v1'></a>
## [Affordance Agent Harness: Verification-Gated Skill Orchestration](https://arxiv.org/abs/2605.00663v1)

**Authors:** Haojian Huang, Jiahao Shi, Yinchuan Li, Yingcong Chen

**Published:** 2026-05-01

**Categories:** cs.RO, cs.CV

**Abstract:**

Affordance grounding requires identifying where and how an agent should interact in open-world scenes, where actionable regions are often small, occluded, reflective, and visually ambiguous. Recent systems therefore combine multiple skills (e.g., detection, segmentation, interaction-imagination), yet most orchestrate them with fixed pipelines that are poorly matched to per-instance difficulty, offer limited targeted recovery from intermediate errors, and fail to reuse experience from recurring objects. These failures expose a systems problem: test-time grounding must acquire the right evidence, decide whether that evidence is reliable enough to commit, and do so under bounded inference cost without access to labels. We propose Affordance Agent Harness, a closed-loop runtime that unifies heterogeneous skills with an evidence store and cost control, retrieves episodic memories to provide priors for recurring categories, and employs a Router to adaptively select and parameterize skills. An affordance-specific Verifier then gates commitments using self-consistency, cross-scale stability, and evidence sufficiency, triggering targeted retries before a final judge fuses accumulated evidence and trajectories into the prediction. Experiments on multiple affordance benchmarks and difficulty-controlled subsets show a stronger accuracy-cost Pareto frontier than fixed-pipeline baselines, improving grounding quality while reducing average skill calls and latency. Project page: https://tenplusgood.github.io/a-harness-page/.

**Analysis:**

### 1. 摘要翻译
Affordance（可供性）感知旨在识别智能体在开放世界场景中“在哪”以及“如何”进行交互，而该任务常受限于小尺寸、遮挡、反射及视觉歧义等挑战。现有系统多依赖固定流水线来编排感知技能，难以应对实例差异，且缺乏针对性的错误修复能力或经验复用机制。为此，作者提出了**Affordance Agent Harness (A-Harness)**，一个闭环运行时框架。该系统整合了异构感知技能，通过证据存储（Evidence Store）和成本控制机制，检索情境记忆（Episodic Memory）以提供针对性先验，并利用路由器（Router）进行自适应技能调度。此外，文中设计了一个特定的验证器（Verifier），通过自洽性、跨尺度稳定性和证据充足性来门控（gate）提交决策，并触发针对性重试（targeted retries）。实验表明，该方法在多个基准测试中实现了更优的准确率-成本帕累托前沿，在提升感知质量的同时降低了平均技能调用次数与延迟。

---

### 2. 方法动机分析
- **驱动力**：将复杂的“可供性感知”从“单次预测任务”重构为“带预算的证据寻求过程”。
- **现有方法痛点**：
    1. **固定脚本僵化**：无法根据实例难度调整感知策略，易对简单案例过处理，对复杂案例缺乏补充证据。
    2. **缺乏闭环校准**：感知结果冲突时仅依赖后期融合，未能追根溯源并修正错误。
    3. **缺乏经验积累**：对象感知需从零开始，浪费了重复出现物体的历史交互信息。
- **核心假设**：引入验证机制和闭环反馈，能在受限预算内通过动态调整证据采集策略，显著提升鲁棒性。

---

### 3. 方法设计详解
**A-Harness 的闭环流程：**
1. **记忆检索**：根据当前场景输入，从公共常识库（Common-sense bank）和情境测试库（Test-time episodic bank）中检索相似交互序列与参数。
2. **路由器调度**：基于当前证据存储（Evidence Store）的状态与剩余预算，计算各技能的预期收益，选择下一步动作。
3. **闭环验证与重试**：利用 Verifier 评估当前感知证据：
    - **跨工具一致性**：检测器与分割器结果是否收敛？
    - **跨尺度稳定性**：缩放/重裁后预测是否漂移？
    - **证据充足性**：当前信息是否足以支撑结论？
    - 若验证不通过，Verifier 指出缺陷源（如 $\omega, \zeta, \mu$），Router 自动触发针对性修正（如缩放、重检测或搜索知识）。
4. **决策融合**：达到验证阈值后，通过 $A = \Phi(\dots)$ 融合所有证据输出最终 mask。

---

### 4. 方法对比分析
- **本质区别**：从传统的“感知模型堆叠”转变为“基于验证反馈的智能体控制”。
- **创新贡献**：引入**显式验证器**实现基于诊断的闭环重试，结合**双层记忆**实现知识的离线先验与在线迭代。
- **适用场景**：开放世界中具备高视觉歧义、复杂交互意图的零样本感知任务。

---

### 5. 实验分析
- **验证方法**：在 ReasonAff、UMD Part Affordance、RAGNet 三个基准上对比固定脚本流水线及多种单体大模型。
- **关键结论**：在各数据集上均取得了最佳的准确率-延迟平衡，Claude-Opus-4.6 作为决策脑效果最优。
- **主要优势**：显著减少了不必要的冗余计算，在高难度、少见样本上具有极强的迁移鲁棒性。
- **主要局限**：对长尾物体与极端设计仍存在数据覆盖不足问题。

---

### 6. 实用指南
- **开源情况**：论文明确提到了项目主页。
- **迁移建议**：本框架核心在于“Verifier-Router”的循环模式，其代码模块解耦，非常适合迁移至其他涉及多工具协作（如 VQA、机器人导航）的任务中。
- **超参数注意**：$\delta$（置信阈值）和 $\omega$（一致性底线）是控制性能与成本的关键，建议在特定部署场景下根据算力预算进行微调。

---

### 7. 总结
- **核心思想**：通过验证驱动的闭环路由，将感知转变为受预算约束的证据自洽过程。
- **速记版pipeline**：
    1. **检索**（获取历史经验）；
    2. **路由**（选择最佳感知工具）；
    3. **诊断**（验证证据一致性与稳定性）；
    4. **修正**（对失败点针对性补救）；
    5. **融合**（输出最终结果）。

**Key Findings:**

- We propose Affordance Agent Harness, a closed-loop runtime that unifies heterogeneous skills with an evidence store and cost control, retrieves episodic memories to provide priors for recurring categories, and employs a Router to adaptively select and parameterize skills.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00663v1)
- [arXiv](https://arxiv.org/abs/2605.00663v1)

---

<a id='2605.00658v1'></a>
## [UniVidX: A Unified Multimodal Framework for Versatile Video Generation via Diffusion Priors](https://arxiv.org/abs/2605.00658v1)

**Authors:** Houyuan Chen, Hong Li, Xianghao Kong, Tianrui Zhu, Shaocong Xu, Weiqing Xiao, Yuwei Guo, Chongjie Ye, Lvmin Zhang, Hao Zhao, Anyi Rao

**Published:** 2026-05-01

**Categories:** cs.CV

**Abstract:**

Recent progress has shown that video diffusion models (VDMs) can be repurposed for diverse multimodal graphics tasks. However, existing methods often train separate models for each problem setting, which fixes the input-output mapping and limits the modeling of correlations across modalities. We present UniVidX, a unified multimodal framework that leverages VDM priors for versatile video generation. UniVidX formulates pixel-aligned tasks as conditional generation in a shared multimodal space, adapts to modality-specific distributions while preserving the backbone's native priors, and promotes cross-modal consistency during synthesis. It is built on three key designs. Stochastic Condition Masking (SCM) randomly partitions modalities into clean conditions and noisy targets during training, enabling omni-directional conditional generation instead of fixed mappings. Decoupled Gated LoRA (DGL) introduces per-modality LoRAs that are activated when a modality serves as the generation target, preserving the strong priors of the VDM. Cross-Modal Self-Attention (CMSA) shares keys and values across modalities while keeping modality-specific queries, facilitating information exchange and inter-modal alignment. We instantiate UniVidX in two domains: UniVid-Intrinsic, for RGB videos and intrinsic maps including albedo, irradiance, and normal; and UniVid-Alpha, for blended RGB videos and their constituent RGBA layers. Experiments show that both models achieve performance competitive with state-of-the-art methods across distinct tasks and generalize robustly to in-the-wild scenarios, even when trained on fewer than 1,000 videos. Project page: https://houyuanchen111.github.io/UniVidX.github.io/

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对 **UniVidX** 这篇论文的分析如下：

### 1. 论文核心贡献总结
UniVidX 提出了一种统一的多模态视频生成框架，旨在打破传统视频扩散模型（VDM）中针对特定任务（如 RGB 转法线图或 RGBA 合成）开发独立模型的局限性。该框架通过共享的多模态空间和创新的条件机制，使模型能够在一个统一的架构下处理多种像素对齐的生成任务，实现了跨模态的深度协作与一致性。

### 2. 核心创新点与方法论
该论文的创新主要体现在三个精心设计的模块，旨在实现“通用性”与“先验保持”的平衡：
*   **随机条件掩码 (SCM)：** 通过在训练期间随机切分模态作为“条件”或“目标”，打破了固定的输入输出映射，使模型具备了任意模态间相互生成的灵活性。
*   **解耦门控 LoRA (DGL)：** 利用针对特定模态的 LoRA 适配器，在保持 VDM 强大的预训练先验的同时，针对不同生成目标进行轻量级参数微调。
*   **跨模态自注意力 (CMSA)：** 通过共享 Key 和 Value 矩阵但解耦 Query 矩阵的策略，在保留模态特性的前提下，促进了不同模态间的深层特征交换与对齐。

### 3. 对该领域的潜在影响
*   **范式转变：** UniVidX 推动了从“一任务一模型”向“通用基础模型”的范式演进，显著降低了多模态视频处理的部署和维护成本。
*   **低资源高效能：** 证明了在大规模预训练先验的加持下，即使在不足 1,000 条视频的小样本数据上，也能实现极具竞争力的生成效果，这对数据稀缺领域的应用具有重要意义。
*   **空间一致性：** 该框架在处理内禀图（Intrinsic maps）和 RGBA 分层生成上的成功，展示了其在 3D 场景理解和计算机图形学底层任务中的巨大潜力。

### 4. 相关领域与应用前景
*   **计算机图形学（CG）：** 自动化生成材质贴图（Albedo）、光照图（Irradiance）和几何法线，可极大提升 3D 内容创作的效率。
*   **影视后期制作：** 自动分离视频图层（RGBA），简化复杂的抠图和合成工作流。
*   **机器人视觉：** 通过学习模态间的相关性（如 RGB 到深度、RGB 到语义分割），辅助机器人在复杂环境中进行实时感知与决策。
*   **数字孪生与 AR/VR：** 在视频流中实时构建场景的内禀属性，提升虚拟现实的交互真实感。

### 5. 可推断的局限性
*   **计算开销与推理延迟：** 虽然使用了 LoRA 进行轻量化，但由于采用了跨模态自注意力（CMSA）机制，在处理长序列视频或高分辨率图像时，其推理阶段的显存占用和计算延迟可能高于专用小模型。
*   **语义与几何一致性的长程维持：** 扩散模型在生成超长视频时通常面临时间一致性挑战，摘要未明确提及如何处理长视频跨帧的几何平滑度。
*   **对数据的依赖性：** 虽然号称“小样本训练”，但本质上高度依赖于 VDM 基座模型原始的分布先验。如果任务超出了基座模型见过的领域（Out-of-distribution），模型的泛化表现可能大幅下降。
*   **模态扩展性：** 论文目前仅在 RGB、内禀图和 RGBA 层验证，引入更多模态（如音频、点云）是否会引起模态间冲突（干扰），仍需进一步验证。

**总结：** UniVidX 的优雅之处在于它没有试图重构扩散模型，而是通过“可插拔”的组件（SCM, DGL, CMSA）将扩散模型“激活”为多模态处理引擎。这对于追求模型通用性和工程落地效率的视觉研究者来说，具有极高的参考价值。

**Key Findings:**

- We present UniVidX, a unified multimodal framework that leverages VDM priors for versatile video generation.
- Experiments show that both models achieve performance competitive with state-of-the-art methods across distinct tasks and generalize robustly to in-the-wild scenarios, even when trained on fewer than 1,000 videos.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00658v1)
- [arXiv](https://arxiv.org/abs/2605.00658v1)

---

<a id='2605.00642v1'></a>
## [Learn where to Click from Yourself: On-Policy Self-Distillation for GUI Grounding](https://arxiv.org/abs/2605.00642v1)

**Authors:** Yan Zhang, Daiqing Wu, Huawen Shen, Yu Zhou, Can Ma

**Published:** 2026-05-01

**Categories:** cs.AI, cs.CV

**Abstract:**

Graphical User Interface (GUI) grounding maps natural language instructions to the visual coordinates of target elements and serves as a core capability for autonomous GUI agents. Recent reinforcement learning methods (e.g., GRPO) have achieved strong performance, but they rely on expensive multiple rollouts and suffer from sparse signals on hard samples. These limitations make on-policy self-distillation (OPSD), which provides dense token-level supervision from a single rollout, a promising alternative. However, its applicability to GUI grounding remains unexplored. In this paper, we present GUI-SD, the first OPSD framework tailored for GUI grounding. First, it constructs a visually enriched privileged context for the teacher using a target bounding box and a Gaussian soft mask, providing informative guidance without leaking exact coordinates. Second, it employs entropy-guided distillation, which adaptively weights tokens based on digit significance and teacher confidence, concentrating optimization on the most impactful and reliable positions. Extensive experiments on six representative GUI grounding benchmarks show that GUI-SD consistently outperforms GRPO-based methods and naive OPSD in both accuracy and training efficiency. Code and training data are available at https://zhangyan-ucas.github.io/GUI-SD/.

**Analysis:**

### 1. 摘要翻译
图形用户界面（GUI）接地（grounding）将自然语言指令映射到目标元素的视觉坐标，是自主GUI代理的核心能力。现有的强化学习方法（如GRPO）性能虽强，但依赖昂贵的多重采样（rollouts），且在困难样本上反馈信号稀疏。这使得在线策略自蒸馏（OPSD）——一种通过单次采样提供稠密token级监督的方法——成为极具吸引力的替代方案，但其在GUI接地领域的应用尚未被探索。本文提出了GUI-SD，这是首个专为GUI接地定制的OPSD框架。首先，它构建了包含目标边界框和高斯软掩码的视觉增强特权上下文，为教师模型提供信息丰富且不泄露精确坐标的指导。其次，它采用了熵引导蒸馏机制，根据数字的重要性和教师的置信度自适应加权，将优化集中在最具影响力和可靠的位置。在六个代表性基准测试上的实验表明，GUI-SD在准确率和训练效率上均显著优于GRPO基准和朴素的OPSD。

### 2. 方法动机分析
*   **驱动力**：解决GUI接地中强化学习训练成本高（多重采样）和反馈信号稀疏的问题。
*   **痛点**：
    1.  **蒸馏到SFT坍塌**：直接将坐标作为文本输入给教师，导致教师分布变为近乎One-hot的硬标签，损失了软标签的“暗知识”价值。
    2.  **不加区分的优化**：均匀地对所有token进行反向KL散度优化，忽略了高位数字（如百位）对坐标精度的支配作用，且无法过滤不确定的噪声梯度。
*   **研究假设**：通过视觉特权上下文引导和基于位置/熵值的自适应加权蒸馏，能实现更高效、更精准的token级监督。

### 3. 方法设计详解
*   **流程总结**：
    1.  **特权上下文构建**：学生模型输入原始GUI图像，教师模型输入“原始图像 + 红色边界框 + 高斯软掩码 + 指令提示”。高斯掩码通过降低目标区域外像素的权重，实现对目标的视觉“聚焦”。
    2.  **双分支蒸馏**：利用同一模型，学生输出分布 $P_S$，教师在特权上下文下输出分布 $P_T$。
    3.  **熵引导损失计算**：在计算 $D_{KL}(P_S || P_T)$ 时引入权重 $w(t) = w_{pos}(t) \cdot w_{ent}(t)$。
*   **算法核心**：
    -   **位置重要性加权 ($w_{pos}$)**：对高位数字分配指数级权重（千位 > 百位 > 十位 > 个位），因为高位误差对空间坐标偏移影响极大。
    -   **熵门控监督 ($w_{ent}$)**：根据教师分布的熵值进行加权。教师预测越确定（熵越低），权重越大；反之则过滤掉不确定性带来的干扰信号。

### 4. 方法对比分析
*   **本质区别**：从传统的“基于奖励的强化学习（RLVR）”转向“基于视觉特权的在线自蒸馏（OPSD）”，从序列级评价转为token级指导。
*   **创新贡献**：提出“视觉特权上下文”解决蒸馏坍塌，提出“熵引导加权”解决优化不均匀问题。
*   **适用场景**：高分辨率、小目标、复杂布局的GUI交互任务。

### 5. 实验分析
*   **验证方法**：在6个主流GUI接地数据集（如ScreenSpot-Pro, OSWorld-G）上与GRPO-Binary/Distance/Gaussian对比。
*   **关键结果**：在ScreenSpot-Pro上准确率达60.7%，较最优GRPO基线提升3.3%以上；训练速度提升约4倍。
*   **优势**：显著降低计算资源需求；通过软标签提升泛化性能；精细化的梯度分配提升了对困难样本的修复能力。
*   **局限**：目前主要针对Qwen3-VL-8B验证，未探索更大规模模型的效果；主要针对单步接地，尚未扩展至长周期规划任务。

### 6. 实用指南
*   **开源情况**：代码及数据已公开（详见论文github链接）。
*   **实现细节**：需注意教师模型并非固定，而是采用EMA（指数滑动平均）策略更新，衰减系数0.95是关键超参数。
*   **迁移可能**：该框架可直接迁移至任何具备多模态输入、输出数字序列的视觉接地任务（如OCR、文档理解、机器人导航）。

### 7. 总结
*   **核心思想**：通过视觉特权与自适应加权实现精准的token级自蒸馏。
*   **速记版pipeline**：
    1. 教师看带“标记”的图，学生看裸图。
    2. 算两者的预测概率差异。
    3. 根据位数给重要数字“加码”。
    4. 根据自信程度给不稳的预测“减压”。
    5. 用上述带权差异引导模型训练。

**Key Findings:**

- In this paper, we present GUI-SD, the first OPSD framework tailored for GUI grounding.
- Extensive experiments on six representative GUI grounding benchmarks show that GUI-SD consistently outperforms GRPO-based methods and naive OPSD in both accuracy and training efficiency.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00642v1)
- [arXiv](https://arxiv.org/abs/2605.00642v1)

---

<a id='2605.00634v1'></a>
## [Paired-CSLiDAR: Height-Stratified Registration for Cross-Source Aerial-Ground LiDAR Pose Refinement](https://arxiv.org/abs/2605.00634v1)

**Authors:** Montana Hoover, Jing Liang, Tianrui Guan, Dinesh Manocha

**Published:** 2026-05-01

**Categories:** cs.RO, cs.CV

**Abstract:**

We introduce Paired-CSLiDAR (CSLiDAR), a cross-source aerial-ground LiDAR benchmark for single-scan pose refinement: refining a ground-scan pose within a 50 m-radius aerial crop. The benchmark contains 12,683 ground-aerial pairs across 6 evaluation sites and per-scan reference 6-DoF alignments for sub-meter root-mean-square error (RMSE) evaluation. Because aerial scans capture rooftops and canopy while ground scans capture facades and under-canopy, the two modalities share only a fraction of their geometry, primarily the terrain surface, causing standard registration methods and learned correspondence models to converge to metrically incorrect local minima. We propose Residual-Guided Stratified Registration (RGSR), a training-free, geometry-only refinement pipeline that exploits the shared ground plane through height-stratified ICP, reversed registration directions, and confidence-gated accept-if-better selection. RGSR achieves 86.0% S@0.75 m and 99.8% S@1.0 m on the primary benchmark of 9,012 scans, outperforming both the confidence-gated cascade at 83.7% and GeoTransformer at 76.3%. We validate RMSE-based pose selection with independent survey control and trajectory consistency, and show that added Fourier-Mellin BEV proposals can reduce RMSE while increasing actual pose error under extreme partial overlap. The dataset and code are being prepared for public release.

**Analysis:**

以下是对该论文的方法深度分析与总结：

### 1. 摘要翻译
我们引入了 Paired-CSLiDAR，这是一个用于单帧姿态精校准的跨源空中-地面 LiDAR 基准数据集。针对空中 LiDAR（屋顶/树冠）与地面 LiDAR（立面/低处植被）之间极端的方向性重叠差异，传统方法极易陷入局部最优。为此，我们提出了 **RGSR（残差引导分层配准）**，这是一种无需训练、基于几何的精校准流水线。它利用共享的地面结构作为最可靠的对齐约束，通过高度分层采样和双向 ICP 生成多样化候选姿态，并利用“接收-如果-更好”（accept-if-better）的 RMSE 策略确保不发生回归。RGSR 在 9,012 个扫描对上达到 86.0% 的 S@0.75m，显著优于现有学习基线。

### 2. 方法动机分析
*   **驱动力**：解决“跨源 LiDAR 配准中严重的覆盖不对称性”导致算法失效的问题。
*   **痛点**：空中视角和地面视角共享的几何信息极少（仅地面），传统的 ICP 和 learned 描述符在处理这种“非对称重叠”时，倾向于将地面的点与空中的非重叠区域匹配，导致收敛到错误的局部极小值。
*   **研究假设**：地面平面是跨视角下最稳定、最可靠的共享几何特征；通过显式利用这种不对称的重叠结构（通过高度分层和反向配准），可以跳出错误的局部最优。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **分层初始化**：将源点云按高度百分位数（p=15, 30, 45, 60）切分。
    2.  **双向假设生成**：分别以“地面作为源、空中作为目标”以及“空中作为源、地面作为目标”进行配准，利用重叠度的方向性差异创造不同的搜索路径。
    3.  **置信度门控（Accept-if-better）**：所有候选姿态必须通过 RMSE 回归检测，仅当误差显著降低时才更新当前最优姿态。
    4.  **残差引导细化**：对于处于模糊阈值间的残差，自动将点云按高度分箱，选择几何一致性最好的箱体进行局部紧半径 ICP。
*   **核心逻辑**：算法并非试图拟合所有点，而是通过“排除法”剔除由于视角遮挡产生的非重叠区干扰，利用地面平面这一“置信域”来锚定垂直度与姿态。

### 4. 方法对比分析
*   **本质区别**：传统算法（如 ICP/GICP）假设重叠是均匀或对称的；RGSR 显式承认并利用了“空中→地面”与“地面→空中”覆盖率的巨大不对称性。
*   **贡献**：无需训练的几何分层流水线，为高遮挡的跨源定位提供了稳健的基准。
*   **适用场景**：适用于城市环境中地面机器人（UGV）与卫星/无人机高程地图（3DEP）的实时精校准。

### 5. 实验分析（精简版）
*   **结论**：在极端低覆盖站点（如 UMD-IdeaF, 20%覆盖），RGSR 相比学习模型（GeoTransformer）性能提升 16-34 pp。
*   **局限**：若初始重叠过低或缺乏可辨识的地平面，RMSE 可能会“欺骗”算法（即误差下降但姿态错误），因此作者提出了“RMSE-TRE 偏离”现象，警示仅靠几何一致性评价在极端情况下存在风险。

### 6. 实用指南
*   **开源**：数据集 CC BY 4.0，代码 MIT。
*   **关键点**：实现时需注意 `reval`（评价阈值）的设定和高度百分位数的选择。无需 GPU 训练，主要依赖 CPU 算力，非常适合边缘部署。
*   **迁移**：该“分层+门控”策略可直接迁移至任何涉及异构传感器（如热成像与 LiDAR）的配准任务中。

### 7. 总结
*   **核心思想**：通过分层地面结构和双向搜索，破解异构传感器覆盖不对称难题。
*   **速记版 Pipeline**：
    1. 按高度切分点云以提取地面支撑。
    2. 同时进行正向与反向的配准搜索。
    3. 若 RMSE 降低则采纳结果。
    4. 对残差较大的情况进行局部细化修正。

**Key Findings:**

- We introduce Paired-CSLiDAR (CSLiDAR), a cross-source aerial-ground LiDAR benchmark for single-scan pose refinement: refining a ground-scan pose within a 50 m-radius aerial crop.
- We propose Residual-Guided Stratified Registration (RGSR), a training-free, geometry-only refinement pipeline that exploits the shared ground plane through height-stratified ICP, reversed registration directions, and confidence-gated accept-if-better selection.

**Links:**

- [PDF](https://arxiv.org/pdf/2605.00634v1)
- [arXiv](https://arxiv.org/abs/2605.00634v1)

---

