time: 20260731

# Arxiv Computer Vision Papers - 2026-07-31

## Executive Summary

## 执行摘要：2026-07-30 Arxiv 计算机视觉论文

本期10篇论文主要围绕**具身智能、机器人操作策略、视觉-语言模型（VLM）智能体、4D空间感知与模型公平性**展开，整体趋势是基础模型与机器人/智能体决策深度结合，同时开始关注安全可控性和社会偏见。

### 1. 主要主题与趋势
- **具身智能与机器人学习**成为绝对主线：涉及人形机器人安全控制（PAC-MAN）、接触丰富操作（FA-RDP）、导航策略泛化（X-NavDP）、零样本操作技能迁移（SemAnCorr）以及手物交互综述。
- **扩散策略（Diffusion Policy）**被广泛应用于机器人行为生成，并开始探索跨行为、跨具身泛化。
- **VLM 正从感知模型走向交互智能体**：如视觉检索token改进（ReToken）、第一人称具身世界建模（EgoGenesis）和GUI智能体（Qwen-UI-Agent）。
- **4D/时空重建与理解**持续发展，特别是无人机单目4D重建和第一人称动态场景建模。
- **可信与公平性问题**进入社区视野：有工作直接质疑“扩大VLM规模即可缓解偏见”的假设。

### 2. 重要或创新性论文
- **PAC-MAN**：将感知感知控制屏障函数（CBF）与强化学习结合，解决人形机器人全身安全问题，兼具理论保障与机器人实用性，值得重点关注。
- **FA-RDP**：提出频率自适应反应式扩散策略，针对接触丰富操作中的高频反馈需求，是对标准扩散策略的重要改进。
- **SemAnCorr**：通过语义锚定对应实现零样本操作技能迁移，可能显著降低跨机器人、跨任务的数据收集成本。
- **EgoGenesis**：提出在线锚定投影记忆与Action-3D RoPE，为第一人称具身世界-动作联合建模提供了新思路。
- **Scaling VLM Is Not Enough to Mitigate Bias**：该论文具有警示意义，提示仅靠“暴力缩放”无法解决视觉语言模型中的偏见问题，对基础模型研发策略有直接影响。

### 3. 新兴研究方向与技术
- 扩散策略的**鲁棒性与泛化**：新行为、新具身、接触力自适应。
- **安全控制与学习相结合**：CBF + RL/CBF + 扩散策略，用于安全关键型机器人系统。
- **语义锚定与跨域对应**：从语义对应出发实现零样本技能迁移。
- **第一人称/自我中心世界模型**：记忆机制、空间旋转位置编码（RoPE）在具身智能中的应用。
- **面向真实世界的GUI智能体**：结合VLM与交互能力的Agent落地方向。
- **基础模型公平性评估**：从“性能提升”转向“风险与偏见度量”。

### 4. 推荐精读论文
- 若关注**机器人安全与人形控制**：**PAC-MAN**。
- 若关注**接触丰富的操作策略**：**FA-RDP**，兼顾反应速度与精细操控。
- 若关注**零样本/跨具身迁移**：**SemAnCorr**和**X-NavDP**。
- 若关注**VLM与具身智能体**：**EgoGenesis**、**Qwen-UI-Agent**、**ReToken**。
- 若关注**可信AI与偏见治理**：**Scaling Vision-Language Models Is Not Enough to Mitigate Bias**。
- 若希望建立全景认知：手物交互综述 **Hand-Object Interaction in the Age of Large Foundation Models** 提供了跨重建、生成和具身迁移的系统梳理。

---

## Table of Contents

1. [ReToken: One Token to Improve Vision-Language Models for Visual Retrieval](#2607.28627v1)
2. [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball](#2607.28623v1)
3. [FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation](#2607.28596v1)
4. [X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching](#2607.28560v1)
5. [Hand-Object Interaction in the Age of Large Foundation Models:Reconstruction, Generation, and Embodied Transfer](#2607.28394v1)
6. [SemAnCorr: Semantic Anchored Correspondence for Zero-Shot Manipulation Skill Transfer](#2607.28382v1)
7. [AdaAnchor4D: Anchor-Conditioned Spatiotemporal Feature Aggregation for Monocular UAV 4D Reconstruction](#2607.28320v1)
8. [EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE](#2607.28243v1)
9. [Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents](#2607.28227v1)
10. [Scaling Vision-Language Models Is Not Enough to Mitigate Bias](#2607.28211v1)

---

## Papers

<a id='2607.28627v1'></a>
## [ReToken: One Token to Improve Vision-Language Models for Visual Retrieval](https://arxiv.org/abs/2607.28627v1)

**Authors:** Yao Xiao, Reuben Tan, Zhen Zhu, Yuqun Wu, Jianfeng Gao, Derek Hoiem

**Published:** 2026-07-30

**Categories:** cs.CV, cs.AI, cs.LG

**Abstract:**

Long visual context poses a challenge for vision-language models: performance degrades as the number of distractors grows, and processing all tokens at once is computationally infeasible under GPU memory constraints. We present ReToken, a single learnable embedding trained as an explicit retrieval target that selects a sparse set of query-relevant visual tokens from a pre-filled visual KV cache. Trained on only a small image-QA dataset, ReToken yields consistent gains across image and video benchmarks: on Visual Haystacks it improves Qwen3VL-8B by 13.4 points and InternVL3.5 by 12.4 points (>20% relative), and on LVBench it transfers zero-shot to long video for an 8.0-point gain with Qwen3VL-8B. Thanks to its lightweight design, both training and long-video inference fit on a single H100. Code is available at: https://github.com/avaxiao/ReToken

**Analysis:**

这是一份关于《ReToken: One Token to Improve Vision–Language Models for Visual Retrieval》的深度技术分析：

### 1. 摘要翻译
长视觉上下文为视觉语言模型（VLM）带来了挑战：随着干扰项增加，性能随之下降，且在GPU内存限制下一次性处理所有Token在计算上不可行。我们提出了RETOKEN，一种可学习的嵌入（embedding），被训练为显式检索目标，用于从预填充的视觉KV缓存中选择稀疏的查询相关视觉Token。仅在小规模图像QA数据集上训练，RETOKEN在图像和视频基准测试中均表现出显著增益：在Visual Haystacks上，它将Qwen3VL-8B的性能提升了13.4个点，InternVL3.5提升了12.4个点；在LVBench上，它以零样本方式在长视频任务上取得了8.0个点的提升。得益于其轻量级设计，训练和长视频推理均可在单张H100上完成。

### 2. 方法动机分析
- **驱动力**：解决VLM在长视觉上下文中检索准确度低以及计算成本过高的问题。
- **现有方法痛点**：当前基于注意力机制（Attention-based）的检索（如ReKV）不仅在VLM中相关性较弱，且注意力计算是为下一Token预测设计的，而非为检索优化。此外，简单的查询聚合（对Question Token取平均）会引入大量噪声，导致检索偏离。
- **研究假设**：Transformer中**值空间（Value Space）**承载了实际传播给后续层的内容信息，比查询-键空间（Query-Key Space）具有更强的语义相关性和更好的检索鉴别力。

### 3. 方法设计详解
- **核心思想**：引入一个名为RETOKEN的可学习Token，将其作为显式检索算子，通过余弦相似度在Value空间进行检索。
- **流程Pipeline**：
  1. **视频/图像预编码**：先进行一次完整视频编码，将所有视觉Token的KV状态存入持久化缓存。
  2. **检索触发**：将RETOKEN拼接到问题文本后，将其通过模型最后一层，得到映射后的查询向量 $Z_r = W_r X_r^{(N)}$。
  3. **Value空间计算**：对每个帧 $f$，计算其Token在最后一层的平均Value向量 $\bar{v}_f^{(N)}$。
  4. **相似度排序**：计算 $Z_r$ 与各帧 $\bar{v}_f^{(N)}$ 的余弦相似度，提取Top-K帧。
  5. **生成阶段**：模型仅读取Top-K帧的缓存进行后续预测。
- **算法细节**：使用类平衡二分类交叉熵损失（Class-balanced BCE Loss）对RETOKEN进行监督，不仅惩罚相关性低的检索结果，也通过显式训练强化了检索能力。

### 4. 方法对比分析
- **本质区别**：传统检索依赖Attention Map（Query-Key），RETOKEN转向利用Value信息，将检索任务从“隐式关注”转变为“显式匹配”。
- **创新点**：
    1. 识别并证明了Value空间在视觉检索中的信息优势。
    2. 设计了轻量级的“单Token”检索机制，避免了复杂模型架构的变动。
    3. 展示了仅在多图像QA上训练即可泛化至极长视频零样本理解的迁移能力。

### 5. 实验分析
- **关键结果**：在Visual Haystacks (C=50) 上，RETOKEN 相比强基线实现了超过20%的相对性能提升；在长达1小时的视频基准测试（LVBench）中实现8.0点零样本增益。
- **局限性**：增加了推理延迟（尽管幅度极小），且对于依赖跨帧时间关联而非单帧内容的总结类问题，性能有下降。

### 6. 实用指南
- **开源情况**：代码已开源至 [github.com/avaxiao/ReToken](https://github.com/avaxiao/ReToken)。
- **实现建议**：
    - 关键参数：设置一个轻量级映射矩阵 $W_r$，保持VLM冻结，仅训练RETOKEN及其投影层。
    - 迁移：该方法极易迁移，只需在多图QA数据上进行短训练，即可嵌入到任意预训练VLM中作为检索插件。
    - 注意：如果模型预训练数据已过拟合，需过滤掉“不经检索也能回答”的简单样本。

### 7. 总结
- **核心思想**：利用Transformer值空间的高鉴别力，通过可学习Token实现高效视觉检索。
- **速记版Pipeline**：
    1. 预存视觉KV缓存。
    2. 插入RETOKEN并运行前向传播。
    3. 在最后一层通过Value相似度锁定关键帧。
    4. 仅载入关键帧KV进行生成。

**Key Findings:**

- We present ReToken, a single learnable embedding trained as an explicit retrieval target that selects a sparse set of query-relevant visual tokens from a pre-filled visual KV cache.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28627v1)
- [arXiv](https://arxiv.org/abs/2607.28627v1)

---

<a id='2607.28623v1'></a>
## [PAC-MAN: Perception-Aware CBF-RL for Whole-Body Safety in Humanoid Dodgeball](https://arxiv.org/abs/2607.28623v1)

**Authors:** Lizhi Yang, Junheng Li, Aaron D. Ames

**Published:** 2026-07-30

**Categories:** cs.RO, cs.AI

**Abstract:**

We present PAC-MAN, a perception-aware CBF-RL framework that couples control-barrier safety with deployment-realistic onboard sensing for whole-body humanoid dodgeball. The deployed policy sees the ball only as segmentation-masked depth from a head-mounted camera, while training-time CBF guidance represents clearance to every body link, and an adversarial motion prior regularizes the resulting evasive reflexes. We evaluate on a controlled any-link contact benchmark with seeded throws in two regimes: single throws and a deployment loop in which the robot walks back to its station and recovers between throws. On this benchmark, the policy comes within a few points of a privileged state oracle: a fixed onboard camera alone is adequate for evasion. We find that usable barrier structure depends on perceptual observability: Joint-CBF gives the best performance with accurate ball states, degrades under fixed-camera observations when used only as training guidance, and recovers with a ball-tracking gimbal or privileged runtime filter. We therefore deploy a lightweight Link-CBF policy zero-shot on the Unitree G1 in the real world, where it tolerates imperfect perception, succeeds on 95% of throws, and uses semantic segmentation to dodge different balls.

**Analysis:**

## 1. 摘要翻译
我们提出了PAC-MAN，这是一个感知感知的CBF-RL（控制障碍函数-强化学习）框架，旨在将控制障碍函数的安全性与部署场景下真实的机载传感能力相结合，用于全身人形躲避球任务。在部署阶段，策略仅通过头戴式摄像头的分割掩码深度图来观测球体。训练阶段则利用基于全身各肢体 clearance（间隙）的CBF指导，并结合对抗性运动先验（AMP）来规范化躲避动作。我们在两种受控的“全肢体接触”基准上进行了评估：单次投掷和包含行走复位的部署循环。在该基准上，该策略的性能仅略低于拥有特权状态信息的 oracle；仅依靠固定的机载摄像头即可实现有效的规避。我们发现，可用的屏障结构依赖于感知的可观测性：Joint-CBF在拥有精确球体状态时表现最佳，但在仅作为训练指导且仅有固定摄像头观测时性能下降；而利用球体跟踪云台或特权运行时过滤器可以恢复其性能。因此，我们将轻量级的Link-CBF策略零样本部署到Unitree G1实机上，该策略容忍了感知的不完美，在95%的投掷中获得成功，并利用语义分割实现了对不同球体的规避。

---

## 2. 方法动机分析
- **驱动力**：旨在解决人形机器人在高动态、短时间内进行全身避障的挑战，特别是在部分可观测（Partial Observability）环境下，如何将“安全性”有效注入到策略中。
- **痛点**：传统的CBF方法通常假设全状态已知（如精确的目标位置和速度），但在现实部署中，传感器输入往往存在噪声、被遮挡或丢失，导致难以在运行时强制执行标准的CBF约束。
- **研究假设**：通过感知感知的训练框架（Perception-Aware），让策略在训练阶段内化（Internalize）复杂的屏障结构，从而在部署时无需复杂的运行时过滤器即可实现鲁棒的安全避障。

---

## 3. 方法设计详解
- **核心流程**：
    1. **感知前端**：通过RGB-D流进行实时语义分割，将球体mask后生成压缩的深度观测（16x9），并进行多帧时序堆叠以编码运动信息。
    2. **训练时引导（Safety-Guided Training）**：引入双层屏障机制，利用特权状态（Privileged States）计算安全性指标：
        - **Link-CBF**：基于全身所有肢体的距离惩罚，作为轻量级Reward。
        - **Joint-CBF**：基于关节空间的投影，提供更强的安全约束。
    3. **策略优化**：使用PPO算法结合AMP运动先验，使得生成的动作在满足安全的同时，具备类似人类的躲避姿态（如蹲下、侧移）。
- **关键算法**：
    - **Link-CBF**：将碰撞风险量化为每个连杆的间隙函数 $h_i$。当 $h_i$ 接近零时触发Reward惩罚。
    - **Joint-CBF投影**：将避障要求转化为关节空间的速度约束（线性不等式），通过二次规划或闭式投影公式，在满足安全前提下修正策略输出。

---

## 4. 方法对比分析
- **本质区别**：与仅在线强制执行CBF的方法不同，PAC-MAN侧重于“内化”。如果感知受限，则退而求其次选择Link-CBF引导；如果感知充足，则使用Joint-CBF。
- **创新点**：提出了“感知-屏障协同设计”（Co-design），证明了屏障结构的强度必须与感知能力相匹配。
- **适用场景**：高动态、需全身协作的短时避障任务。

---

## 5. 实验分析
- **关键结论**：在固定摄像头下，Link-CBF策略优于Joint-CBF（因后者要求过高的感知精度导致训练崩溃），证明了“轻量化结构更适配不完美感知”。
- **性能指标**：实机测试中实现了95%的成功率。
- **局限**：若球体彻底消失或感知误差过大，仍可能发生碰撞；目前仅限于固定站姿下的躲避，而非运动中的动态避障。

---

## 6. 实用指南
- **开源情况**：已发布benchmark及训练流水线（参考文中提到的 mjlab 和 RSL-RL 库）。
- **实现细节**：
    - **感知堆叠**：使用非连续的稀疏帧（如[0, 3, 8, 18]）比连续帧更有效。
    - **Reward设计**：必须包含动作平滑项（Action Rate）和基于AMP的风格reward，否则避障动作会剧烈抖动。
- **迁移可能**：可迁移至无人机动态避障或四足机器人复杂地形敏捷行走。

---

## 7. 总结
- **核心思想**：屏障安全性需根据感知精度“量体裁衣”进行训练内化。
- **速记版Pipeline**：
    1. **视觉处理**：球体分割并压缩为低维深度图。
    2. **安全训练**：利用特权真值计算肢体碰撞风险。
    3. **动作内化**：通过强化学习学习避障 reflexes。
    4. **零样本部署**：仅依靠 onboard 视觉和本体感知执行动作。

**Key Findings:**

- We present PAC-MAN, a perception-aware CBF-RL framework that couples control-barrier safety with deployment-realistic onboard sensing for whole-body humanoid dodgeball.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28623v1)
- [arXiv](https://arxiv.org/abs/2607.28623v1)

---

<a id='2607.28596v1'></a>
## [FA-RDP: A Frequency-Adaptive Reactive Diffusion Policy for Contact-Rich Manipulation](https://arxiv.org/abs/2607.28596v1)

**Authors:** Lifeng Zhuo, Wendi Chen, Han Xue, Shirun Tang, Jun Lv, Cewu Lu, Chuan Wen

**Published:** 2026-07-30

**Categories:** cs.RO

**Abstract:**

In contact-rich manipulation, action multimodality and reactivity dominate different stages of a single episode. Before contact, multiple trajectories might be equally valid, making it important to preserve diverse action modes. After contact, geometric constraints and force limits narrow the solution space, while successful execution demands rapid responses to force feedback. However, standard diffusion policies use a fixed inference frequency and sampling steps throughout the episode, forcing a fundamental compromise: low-frequency, multi-step sampling better preserves pre-contact multimodality but responds slowly to force feedback, whereas high-frequency sampling improves reactivity but tends to collapse distinct pre-contact modes. To resolve this tradeoff, we present FA-RDP, a frequency-adaptive reactive diffusion policy. A shared multi-frequency visual-force Transformer predicts action chunks at both low and high frequencies, while a learned multimodality indicator dynamically selects multi-step low-frequency sampling before contact and one-step high-frequency sampling as action ambiguity decreases. We further introduce Manifold Consistency Distillation (MCD), which reparameterizes the diffusion network to predict actions on the robot action manifold while retaining DDPM-based residual supervision. Experiments on three contact-rich manipulation tasks show that FA-RDP achieves the highest success rate while preserving diverse pre-contact trajectory modes. Code and videos are available at https://fa-rdp.github.io.

**Analysis:**

### 1. 摘要翻译
在接触丰富的操作任务中，动作的多模态性与响应能力在单个任务的不同阶段占据主导地位。接触前，多种轨迹可能同样有效，保持多样的动作模式至关重要；接触后，几何约束和力限制缩小了求解空间，成功执行则要求对力反馈做出快速响应。然而，标准扩散策略在整个过程中使用固定的推理频率和采样步数，迫使我们在保持预接触多模态性的低频多步采样（响应慢）和改善接触后反应的高频采样（易导致模式崩塌）之间做出权衡。为此，我们提出了FA-RDP，一种频率自适应的反应式扩散策略。该方法利用共享的多频率视觉-力Transformer在低频和高频下预测动作块，并通过一个学习到的多模态指标，动态选择接触前的多步低频采样和接触后的单步高频采样。此外，我们引入了流形一致性蒸馏（MCD），通过在机器人动作流形上重参数化扩散网络来预测动作，同时保持基于DDPM的残差监督。在三项接触密集型操作任务上的实验表明，FA-RDP在保持预接触轨迹模式多样性的同时，实现了最高的成功率。

---

### 2. 方法动机分析
- **驱动力**：解决接触丰富操作中“预接触阶段多样性需求”与“接触后阶段实时反馈需求”之间的根本矛盾。
- **痛点**：现有扩散策略（如Diffusion Policy/ImplicitRDP）要么固定低频导致接触后反馈延迟，要么强制高频导致接触前多模态分布崩塌。
- **研究假设**：通过引入可学习的指标实时判断操作阶段，并结合多频率Transformer及针对性的蒸馏技术，可以在同一模型框架下实现阶段自适应的性能优化。

---

### 3. 方法设计详解
- **流程pipeline**：
    1. **多频率Transformer**：利用频率自适应位置编码（共享底层的视觉-力Transformer），在同一时间轴上通过不同的采样密度（稀疏对应低频，密集对应高频）处理不同频率的动作块。
    2. **多模态指标学习**：在阶段2训练一个评估指标，计算当前采样分布与演示分布的残差。指标低表示处于预接触自由运动阶段，指标高表示处于约束接触阶段。
    3. **流形一致性蒸馏（MCD）**：将高频策略蒸馏为单步推理模型。不同于常规预测噪声（epsilon），MCD直接在机器人动作流形上进行预测，通过强制输出接近演示动作，实现了高效、稳定的单步高频响应。
- **关键技术**：
    - **共享主干**：通过位置编码差异实现频率切换，避免了维护两个独立模型的计算负担。
    - **DDPM残差监督**：在蒸馏过程中保留了对原始DDPM扩散目标的追踪，确保模型学习到的动作空间符合物理可行性。

---

### 4. 方法对比分析
- **本质区别**：与RDP等层级策略不同，FA-RDP是单一模型下的频率自适应，而非显式切换两个策略。
- **创新点**：
    - **频率自适应机制**：基于指标而非硬编码阶段，更具鲁棒性。
    - **流形一致性蒸馏**：解决了高频采样下动作平滑度和力反馈准确性的冲突。

---

### 5. 实验分析
- **关键结果**：在三项任务中平均成功率达81.7%，显著优于基线（如ImplicitRDP的51.7%）。
- **优势**：成功保持了预接触的多模态分布（如Fig. 8所示），并有效消除了接触后的力响应延迟。
- **局限**：三阶段训练流程（预训练、指标训练、蒸馏）较为繁琐，且目前仅限于单任务设置。

---

### 6. 实用指南
- **开源情况**：代码及演示已公开（fa-rdp.github.io）。
- **实现建议**：
    - **力补偿**：必须加入文中提到的100Hz力补偿公式（$p_{cmd} = p_{policy} - \lambda f_{ext}$），这是消除阻抗控制导致跟踪误差的关键。
    - **超参数**：$\lambda = 10^{-4}$，蒸馏步数选取 $\{99, 79, 59, 39, 19, 0\}$ 等关键点。
- **迁移性**：该框架易于迁移至任何需要结合预规划（路径多样性）与快速反馈（力控/触觉控制）的任务中。

---

### 7. 总结
- **核心思想**：基于阶段感知的双模态频率自适应扩散策略。
- **速记版pipeline**：
    1. 共享主干预测不同频率动作。
    2. 实时监测多模态指标。
    3. 根据接触状态切换采样频率。
    4. 蒸馏高频策略以提升实时响应。

**Key Findings:**

- To resolve this tradeoff, we present FA-RDP, a frequency-adaptive reactive diffusion policy.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28596v1)
- [arXiv](https://arxiv.org/abs/2607.28596v1)

---

<a id='2607.28560v1'></a>
## [X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching](https://arxiv.org/abs/2607.28560v1)

**Authors:** Tianyu Yang, Yiming Zeng, Wenzhe Cai, Yuqiang Yang, Jiaqi Peng, Hui Cheng, Jiangmiao Pang, Tai Wang

**Published:** 2026-07-30

**Categories:** cs.RO

**Abstract:**

Pretraining navigation diffusion policies rely on large-scale expert demonstrations. These data are typically generated by a fully-informed oracle planner suited to a single nominal robot. This limits the policy's generalization to diverse embodiments and challenging scenarios (e.g., escaping dead ends or detouring long obstacles) that demand diverse local reactive behaviors with only onboard local observations. Post-training the policy with reinforcement learning (RL) offers a principled remedy. However, previous RL for diffusion approaches lead to only marginal improvements. This is because the intractable likelihood of diffusion policies renders policy gradients unstable in addition to inefficient policy exploration. To address these challenges, we propose a data-efficient diffusion RL post-training framework - GQRM (Group Q-score Reweighted Matching). Our framework introduces two complementary designs: (i) a self-bootstrapped exploration strategy with behavior perturbation that preserves the pretrained policy prior, and (ii) a group Q-score normalization mechanism that computes per-trajectory values on each state for efficient reweighted score matching. By conducting distributed online RL training across heterogeneous embodiments, the resulting fine-tuned policy, X-NavDP, achieves state-of-the-art cross-embodiment visual navigation performance, improving the overall success rate from 61.20% to 84.28% in simulation and 10% to 65% in real-world hard cases. The code and model are publicly available at https://yty-sky.github.io/x-navdp-project-page.

**Analysis:**

# 论文方法深度分析：X-NavDP

### 1. 摘要翻译
预训练的导航扩散策略依赖于大规模专家演示，这些数据通常由单机型专用的全知预言机规划器生成，限制了策略在多样化实体和复杂场景（如死胡同逃生、绕过长障碍物）中的泛化能力，这些场景仅依靠机载局部观测即可完成。通过强化学习（RL）对策略进行后训练是一个原则性的补救措施，但现有的扩散模型RL方法效果有限，主要原因是扩散策略难以处理的似然函数导致策略梯度不稳定，且探索效率低下。为解决这些挑战，本文提出了高效的扩散RL后训练框架——**GQRM (Group Q-score Reweighted Matching)**。该框架包含两个互补设计：(i) 结合行为扰动的自举探索策略，以保留预训练策略先验；(ii) 一种组内Q值归一化机制，通过计算每种状态下的轨迹价值，实现高效的加权得分匹配。通过跨异构实体的分布式在线RL训练，所得的微调策略X-NavDP在模拟环境下将成功率从61.20%提升至84.28%，在现实世界难题中从10%提升至65%。

---

### 2. 方法动机分析
*   **驱动力**：解决预训练导航策略在未见实体、复杂障碍物及需要自主失败恢复场景下的泛化瓶颈。
*   **现有方法痛点**：
    *   **Imitation Learning限制**：预言机生成的全局最优轨迹在实时局部观测中存在“决策歧义”，且难以在死胡同等场景实现自主恢复。
    *   **RL不稳定**：扩散策略的链式去噪过程导致似然计算复杂，直接进行策略梯度优化极不稳定。
    *   **探索失效**：现有RL方案中的全局价值归一化在困难状态下（所有候选样本价值均较低）无法提供有效的梯度信号。
*   **研究假设**：通过保留扩散策略的先验分布，并引入针对“同状态下候选集”的局部Q值归一化，可以更有效地从次优探索中引导策略学习高质量行为。

---

### 3. 方法设计详解
*   **核心模块**：
    1.  **自举轨迹扰动 (Self-Bootstrapped Perturbation)**：利用同一个预训练模型，当条件输入从“目标导向”切换为“目标无关”时，能生成场景一致但更具探索性的轨迹。通过混合这两种轨迹并进行坐标翻转，在保持动力学可行性的同时增加了探索的多样性。
    2.  **组内Q值归一化 (Group Q-score Reweighted Matching, GQRM)**：摒弃了跨minibatch的全局归一化，改为在同一个状态$s$下对采样的一组候选动作$\{a_i\}$计算均值和方差，通过局部加权来反向传播奖励信号。这使得即便在整体奖励较低的困难状态下，策略仍能聚焦于该状态下“相对较优”的轨迹。
    3.  **机身调制 (Embodiment Modulation)**：通过FiLM层将机器人嵌入（Robot Embedding）注入解码器，使一个共享的扩散网络能够适配轮式、四足、人形等不同动力学约束的机器人。
*   **流程总结**：
    1.  **输入**：当前局部观测+目标。
    2.  **生成**：利用扰动策略生成多样化轨迹候选集。
    3.  **评价**：通过Critic网络获取Q值。
    4.  **归一化**：在状态内部进行Q值归一化，计算加权系数。
    5.  **更新**：执行加权得分匹配损失函数，微调扩散去噪网络。

---

### 4. 方法对比与创新
*   **本质区别**：与传统RL方法不同，GQRM不直接对策略密度建模（避免了难算的似然），而是通过加权得分匹配（Reweighted Score Matching）将RL目标转化为监督学习目标。
*   **核心贡献**：首次提出了“同状态组内归一化”策略，有效解决了RL在稀疏奖励环境下的梯度消失和对齐难题。
*   **适用场景**：适用于具有预训练视觉导航先验、但在长周期任务和复杂避障方面表现不足的各类机器人平台。

---

### 5. 实验分析
*   **关键结果**：在IsaacLab仿真中成功率显著提升；在物理世界的复杂布局中展现出极强的零样本迁移能力（从10%提至65%）。
*   **优势**：极强的探索效率，只需12小时即可完成跨平台的RL微调，且通过FiLM模块实现了极佳的跨机型通用性。
*   **局限**：目前的长期记忆能力依赖于短时窗口，且对于透明物体（如玻璃）的感知受限于RGB-D传感器的物理局限。

---

### 6. 实用指南
*   **开源建议**：该工作基于IsaacLab，利用其并行仿真能力是实现的关键，建议在复现时优先部署IsaacLab环境。
*   **实现细节**：λ（温度参数）的设置对平衡策略改进幅度至关重要；在微调时建议只对去噪过程的后6个低噪声步进行训练，以保持预训练的鲁棒性。
*   **迁移建议**：GQRM的“同状态归一化”思想可直接迁移到任何基于扩散策略的RL任务中（如机器人操控、端到端自动驾驶）。

---

### 7. 总结
*   **核心思想**：通过组内局部奖励归一化，稳定扩散策略的在线RL微调。
*   **速记版pipeline**：
    1. 从预训练模型采样多条轨迹；
    2. 对轨迹进行多样化扰动；
    3. 在同状态下对比轨迹价值并归一化；
    4. 将得分作为权重指导去噪网络训练。

**Key Findings:**

- To address these challenges, we propose a data-efficient diffusion RL post-training framework - GQRM (Group Q-score Reweighted Matching).
- By conducting distributed online RL training across heterogeneous embodiments, the resulting fine-tuned policy, X-NavDP, achieves state-of-the-art cross-embodiment visual navigation performance, improving the overall success rate from 61.20% to 84.28% in simulation and 10% to 65% in real-world hard cases.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28560v1)
- [arXiv](https://arxiv.org/abs/2607.28560v1)

---

<a id='2607.28394v1'></a>
## [Hand-Object Interaction in the Age of Large Foundation Models:Reconstruction, Generation, and Embodied Transfer](https://arxiv.org/abs/2607.28394v1)

**Authors:** Weiquan Lin, Yu Deng, Shiyang Liu, Luping Xiao, Xu Tang, Junzhi Yu, Jiaolong Yang, Lei Zhang, Xingyu Chen

**Published:** 2026-07-30

**Categories:** cs.CV

**Abstract:**

Hand-object interaction (HOI) modeling remains challenging because it requires joint reasoning about hand articulation, object geometry, contact, semantics, and dynamics under severe visual uncertainty. Foundation models introduce transferable prior knowledge learned from large-scale cross-domain data, offering new ways to address these challenges beyond task-specific data and models. However, the rapidly growing literature remains fragmented, and existing studies typically describe these methods simply as ``using large models'' without systematically characterizing what knowledge is introduced, where it enters the HOI pipeline, or which HOI uncertainty it helps reduce. This survey presents the first systematic review of foundation-model priors for HOI. We organize the literature into six HOI tasks spanning reconstruction and generation. More importantly, we establish a taxonomy of eight foundation-model sub-priors grouped into geometric, semantic, and visual families. Geometric priors encompass shape retrieval, shape reconstruction, and spatial reconstruction; semantic priors include semantic grounding and language reasoning; and visual priors cover visual representation, image generation, and video generation. Based on this taxonomy, we systematically analyze how different priors are represented, injected, and adapted across HOI pipelines and tasks. Beyond how foundation models empower HOI, we further examine how HOI-derived knowledge is used in robot learning, including human-data pretraining, human-to-robot skill transfer, and HOI-to-robot data generation. Finally, we summarize datasets and evaluation protocols, and discuss limitations and future directions toward more generalizable HOI systems. To support long-term progress, we curate a live repository that continuously aggregates emerging methods and benchmarks.

**Analysis:**

这份综述论文不仅是对现有HOI（手-物体交互）工作的归纳，更重要的是它**建立了一个基于“基础模型先验（Foundation-Model Priors）”的系统性分析框架**，将原本碎片化的研究范式统一到了“先验注入”的视角下。

---

### 1. 摘要翻译
手-物体交互（HOI）建模因需要同时处理手部关节、物体几何、接触关系、语义和动态，且面临严重视觉不确定性，至今仍极具挑战。基础模型通过大规模跨域数据学习，引入了可迁移的先验知识，为解决这些挑战提供了新途径。然而，当前文献碎片化严重。本研究首次对HOI领域的基础模型先验进行了系统综述，将研究划分为六大HOI任务，并提出了一个包含几何、语义和视觉三大类八种子先验的分类学。基于该分类学，我们系统分析了不同先验在HOI流水线中的表现、注入方式及适应性。此外，我们探讨了HOI知识如何通过人类数据预训练、人机技能迁移及HOI转机器人数据生成，赋能机器人学习。最后，我们总结了相关数据集与评估协议，并展望了通向更通用化HOI系统的未来方向。

### 2. 方法动机分析
*   **驱动力**：旨在解决传统HOI方法受限于任务特定标注数据和领域知识的瓶颈，利用Foundation Model（FM）的通用表征能力减少HOI建模中的五大不确定性（形状、空间、物理、语义、动态）。
*   **痛点**：现有研究通常简单概括为“使用大模型”，缺乏对知识引入机制（先验注入）的深度解构，导致模型的设计逻辑不清晰。
*   **研究假设**：HOI问题的核心在于缓解由于遮挡和动态复杂性导致的信息缺失，通过引入具备通用先验的FM，可以显著增强模型在遮挡环境下的推理和泛化能力。

### 3. 方法设计详解
论文的核心贡献是**将“先验知识注入”显式化为一种设计模式**，其处理流程可归纳为：
1.  **确定不确定性与先验族**：根据任务（如R1-R3，G1-G3）中的痛点（如遮挡导致形状不确定），从几何（Shape Retrieval, Rec, Spatial）、语义（Grounding, Reasoning）、视觉（Rep, Image Gen, Video Gen）中匹配先验。
2.  **注入机制（Injection Operators）**：
    *   **几何先验**：通过“检索/对齐”或“Token融合”将外部先验知识（如MoGe-2, InstantMesh）植入模型。
    *   **语义先验**：通过“区域调节（Region Conditioning）”将 Grounding 结果作为Mask或Box约束；通过“条件/融合（Condition/Fuse）”将LLM的语言指令转化为交互约束。
    *   **视觉先验**：通过“Token融合”使用通用特征，或通过“去噪分数引导（Score-Guided Regularization）”在优化过程中对生成结果进行正则化。
3.  **结果输出**：通过这些先验的共同作用，在保留HOI后端（如MANO, DDF, Transformer）的基础上，增强输出的空间一致性与物理合规性。

### 4. 方法对比分析
*   **本质区别**：本文将目光从模型架构本身转向了**外部知识的来源与注入接口**。传统的模型依赖于特定数据集的监督，而先验驱动的方法依赖于FM的通用知识。
*   **创新贡献**：提出了系统化的先验分类法和注入算子库（如Token Fusion, Score-Guided Regularization），为后继者构建系统提供了一套模块化设计图谱。
*   **适用场景**：特别适用于数据稀缺的交互场景，以及需要在大规模遮挡或开放词汇环境下进行交互重建的任务。

### 5. 实验与总结
*   **验证方法**：通过梳理涵盖300余篇文献的分类矩阵，验证了该taxonomy的解释力。
*   **主要优势**：将“先验”这一模糊概念转化为“可操作的算子”，大幅提升了研究的可复现性与组件互换性。
*   **主要局限**：单纯依赖FM先验无法完全解决“物理合规性（Physical Validity）”问题（如生成的图像可能看起来逼真但手部穿模）。

### 6. 实用指南
*   **开源情况**：综述作者维护了一个[Live Repository](https://github.com/SeanChenxy/Hand3DResearch/tree/hoi-survey)，持续更新相关基准。
*   **迁移建议**：若想在自己的任务中引入FM，首先应定位目标不确定性（如图3），随后参考表3找到对应的先验注入方式（如使用DINOv2特征作为token融合）。

### 7. 总结
*   **核心思想**：利用基础模型提供的通用先验，模块化地解决手-物体交互中的五大不确定性。
*   **速记版Pipeline**：
    1. 识别交互任务中的不确定性（如遮挡、语义缺失）。
    2. 选择对应的几何/语义/视觉基础模型作为先验源。
    3. 通过融合、注入、条件约束等算子将知识植入HOI骨干。
    4. 对输出进行物理与交互一致性验证。

**Key Findings:**

- Foundation models introduce transferable prior knowledge learned from large-scale cross-domain data, offering new ways to address these challenges beyond task-specific data and models.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28394v1)
- [arXiv](https://arxiv.org/abs/2607.28394v1)

---

<a id='2607.28382v1'></a>
## [SemAnCorr: Semantic Anchored Correspondence for Zero-Shot Manipulation Skill Transfer](https://arxiv.org/abs/2607.28382v1)

**Authors:** Xiaoxiang Dong, William Baron, Hongyi Chen, Uksang Yoo, Jeffrey Ichnowski, Weiming Zhi

**Published:** 2026-07-30

**Categories:** cs.RO

**Abstract:**

Transferring manipulation skills across object instances that share functionality but differ in geometry remains a fundamental challenge in robot learning. While recent correspondence methods leverage dense visual descriptors and 3D feature fields, nearest-neighbor feature matching often produces spatially incoherent correspondences that fail to recover the local geometric frames required for reliable skill transfer. We introduce SemAnCorr, a training-free framework that establishes dense correspondence by selecting semantically consistent anchor regions through joint pose-correspondence optimization and propagating these constraints over the object surface using functional maps. The resulting correspondences preserve both semantic consistency and geometric coherence, enabling object-centric manipulation skills to transfer across geometrically diverse instances. We evaluate SemAnCorr on a dense correspondence benchmark built on PartNet-Mobility, achieving 90.8% semantic accuracy in our benchmark evaluation while improving geometric coherence over recent state-of-the-art baselines. Finally, we show that these improvements translate directly into real-world manipulation performance: using a single demonstration, SemAnCorr enables substantially more reliable zero-shot manipulation skill transfer to previously unseen objects than existing correspondence methods. Videos and additional visualizations are available at [https://semancorr.github.io](https://semancorr.github.io) .

**Analysis:**

### 1. 摘要翻译
将操作技能跨越具有相同功能但几何形状不同的物体实例进行迁移是机器人学习中的基础性挑战。尽管现有的对应关系（Correspondence）方法利用了稠密视觉描述子和3D特征场，但最近邻特征匹配往往产生空间不一致的对应关系，无法恢复可靠技能迁移所需的局部几何坐标系。我们提出了 SemAnCorr，一种无需训练的框架。它通过联合姿态与对应关系优化来选择语义一致的锚点区域，并利用函数映射（Functional Maps）将这些约束传播到整个物体表面。所产生的对应关系既保持了语义一致性又兼顾了几何连贯性，使得以物体为中心的操纵技能能够跨越几何形状各异的实例进行迁移。在 PartNet-Mobility 构建的稠密对应基准测试中，SemAnCorr 达到了 90.8% 的语义准确率，并在几何连贯性上优于当前最先进的基线方法。最后，我们展示了这些改进直接转化为现实世界中的操纵性能：利用单次演示，SemAnCorr 实现了比现有方法更可靠的零样本操作技能迁移。

---

### 2. 方法动机分析
*   **驱动力**：解决机器人跨几何差异物体的“零样本”技能迁移问题，实现对物体功能性（如：如何抓握把手）而非仅仅几何位置的理解。
*   **现有方法痛点**：现有方法（如D3Fields）依赖近邻匹配，虽能定位大体区域，但缺乏几何全局平滑性，导致局部几何坐标系扭曲，无法执行精细的物体交互（如旋转、推拉等）。
*   **研究假设**：通过引入“语义锚点”并在函数映射框架下进行全局表面约束，可以同时强制执行语义一致性和几何平滑性。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **3D语义获取**：使用预训练的SigLip2Vision模型提取多视角图像特征，通过升维操作（Lifting）投影至3D点云，形成语义点云。
    2.  **语义聚类**：将特征降维后与3D位置拼接，通过K-means聚类划分出具有语义意义的部件（如把手、杯身），并计算部件描述符。
    3.  **锚点优化（核心）**：通过“相对余弦相似度”去除类别信号干扰，利用“双边边缘置信度”筛选高可信度的部件匹配对，并联合优化刚性对齐（R, t）与簇间对应关系。
    4.  **函数映射传播**：利用选定的锚点约束Laplace-Beltrami谱基，通过ZoomOut算法进行迭代优化，恢复稠密顶点级映射，确保几何平滑。
*   **公式解读**：`Score`函数将几何兼容性与语义相似度耦合，在优化过程中通过余弦调度从偏向语义切换到偏向几何，动态引导对应关系对齐。

---

### 4. 方法对比分析
*   **本质区别**：传统方法往往侧重于局部特征匹配（易产生空间破碎），而SemAnCorr引入了**基于语义先验的全局函数映射约束**。
*   **创新贡献**：提出了一种无需训练的框架，通过锚点引导的联合优化有效解决了语义对齐与几何平滑之间的矛盾。
*   **适用场景**：适用于具有功能相似部件但外形存在显著几何差异的刚性或铰接物体操作任务。

---

### 5. 实验分析（精简版）
*   **关键结果**：在PartNet-Mobility基准上达到90.8%的语义准确率；在5项真实世界机械臂操纵任务中，成功率显著高于D3Fields（如Task 3为7/10 vs 3/10）。
*   **主要优势**：同时实现了语义对齐（知“何处”操作）与几何连贯（知“如何”操作），在复杂交互任务中表现稳健。
*   **主要局限**：对初始Mesh的质量及视角依赖较高，且每对物体需要进行 per-object-pair 优化（约6秒），在实时性要求极高的场景下存在瓶颈。

---

### 6. 实用指南
*   **开源情况**：作者提供了主页 `semancorr.github.io`。
*   **实现细节**：关键参数：聚类簇数 $K=6$，锚点数 $\alpha=3$，谱基维度 $k=30$。
*   **迁移建议**：该方法逻辑通用，可直接迁移至任何具有零件级语义标签的3D形状对应任务。若需提速，可考虑预计算特征或使用轻量化的近似函数映射求解器。

---

### 7. 总结
*   **核心思想**：以语义锚点约束函数映射，实现语义几何双一致。
*   **速记版pipeline**：
    1. 提取物体语义特征并聚类。
    2. 筛选语义和几何最优匹配锚点。
    3. 联合优化姿态与对应关系。
    4. 通过函数映射实现平滑稠密对齐。

**Key Findings:**

- We introduce SemAnCorr, a training-free framework that establishes dense correspondence by selecting semantically consistent anchor regions through joint pose-correspondence optimization and propagating these constraints over the object surface using functional maps.
- We evaluate SemAnCorr on a dense correspondence benchmark built on PartNet-Mobility, achieving 90.8% semantic accuracy in our benchmark evaluation while improving geometric coherence over recent state-of-the-art baselines.
- Finally, we show that these improvements translate directly into real-world manipulation performance: using a single demonstration, SemAnCorr enables substantially more reliable zero-shot manipulation skill transfer to previously unseen objects than existing correspondence methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28382v1)
- [arXiv](https://arxiv.org/abs/2607.28382v1)

---

<a id='2607.28320v1'></a>
## [AdaAnchor4D: Anchor-Conditioned Spatiotemporal Feature Aggregation for Monocular UAV 4D Reconstruction](https://arxiv.org/abs/2607.28320v1)

**Authors:** Peiyi Xu, Junpeng Zhang, Guanbin Li, Ronghua Shang, Mingtao Feng, Le Dong, Weisheng Dong, Guangming Shi, Jie Feng

**Published:** 2026-07-30

**Categories:** cs.CV

**Abstract:**

Monocular UAV videos provide valuable observations for dynamic reconstruction of complex urban scenes. However, such scenes exhibit pronounced spatiotemporal heterogeneity: different regions follow distinct temporal activity patterns, while the motion states of some dynamic regions may further evolve over time. Although dynamic Gaussian methods based on decomposed shared spatiotemporal feature fields have achieved efficient and accurate reconstruction in object-centric or relatively compact scenes, their commonly adopted fixed plane-wise feature combination mechanisms are less suited to the heterogeneous local dynamics of UAV scenes, often leading to ghosting artifacts and blurred dynamic details. To address this challenge, we propose AdaAnchor4D, an adaptive anchor deformation framework for monocular UAV dynamic scene reconstruction. At its core, Anchor-Conditioned Feature Aggregation (ACFA) adaptively aggregates shared spatiotemporal features using anchor-specific aggregation embeddings and temporal information, allowing different local units to obtain dynamic representations tailored to their local and temporal states. Decoupled Local Geometry Deformation (DLGD) separates anchor-state deformation from local Gaussian geometry deformation, while Density-Adaptive Coordinate Warping (DACW) reparameterizes feature-query coordinates according to the axis-wise anchor distributions, alleviating the mismatch between non-uniform geometric sampling and uniform grid parameterization. Experiments on UAV-Arc4D, VisDrone, and UAVDT show that AdaAnchor4D achieves higher rendering quality than representative dynamic Gaussian methods while maintaining real-time rendering performance. The code will be made publicly available.

**Analysis:**

这是一份针对论文《AdaAnchor4D: Anchor-Conditioned Spatiotemporal Feature Aggregation for Monocular UAV 4D Reconstruction》的深度分析报告。

---

### 1. 摘要翻译
单目无人机视频为复杂城市场景的动态重建提供了宝贵的观测视角。然而，这些场景表现出显著的时空异质性：不同区域遵循不同的时间活动模式，且动态区域的运动状态会随时间演化。虽然基于分解共享时空特征场的动态高斯方法在物体中心场景中取得了高效重建，但其常用的固定平面特征组合机制难以适应无人机场景中异质的局部动态，常导致重影和动态细节模糊。为此，我们提出了 AdaAnchor4D，一个用于单目无人机动态重建的自适应锚点形变框架。其核心是锚点条件特征聚合（ACFA），通过锚点特定的聚合嵌入和时间信息自适应聚合共享时空特征，使局部单元能获得针对其时空状态定制的动态表示。此外，解耦局部几何形变（DLGD）将锚点状态形变与局部高斯几何形变分离，密度自适应坐标扭曲（DACW）根据坐标轴的锚点分布重参数化特征查询坐标，缓解了采样分布不均与均匀网格参数化之间的不匹配。实验表明，AdaAnchor4D 在保持实时性能的同时提升了渲染质量。

### 2. 方法动机分析
*   **驱动力**：解决单目无人机视频中复杂的动态场景，克服传统动态高斯方法中“固定特征融合规则”无法适应时空异质性的问题。
*   **现有痛点**：现有方法将空间和时间维度强制映射到共享的特征场，并采用固定的权重组合（如简单的乘法），导致无法捕捉不同位置、不同时间点的独特动态需求。
*   **研究假设**：如果能够根据每个锚点及其所处的时间点，动态预测特征平面的聚合权重，并解耦锚点级与局部高斯级的形变，就能显著提升对复杂异质动态的建模能力。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **坐标映射（DACW）**：利用场景锚点的空间分布构建非线性扭曲映射，将非均匀的原始空间映射到更合理的网格空间。
    2.  **动态特征提取（ACFA）**：每个锚点维护一个可学习的 `Aggregation Embedding`。将该嵌入与当前时间 $t$ 输入 MLP，预测每个特征平面的通道级权重，从而实现对共享特征的加权聚合。
    3.  **形变解耦（DLGD）**：将动态预测分为两路。第一路处理“锚点级”的位置与特征形变；第二路利用专门的几何嵌入，处理“局部高斯”的偏移与尺度变化。
    4.  **渲染**：通过形变后的锚点实例生成 3D 高斯，进行可微分渲染。

*   **关键点**：ACFA 的核心在于用“动态分配”替代“固定叠加”。通过引入时间依赖，模型能够识别出“静态”区域倾向于空间特征，“动态”区域倾向于时空平面特征，从而实现精准建模。

### 4. 方法对比分析
*   **本质区别**：从传统的固定规则驱动转向“条件自适应”驱动。不再是所有空间位置共享同一组聚合权重，而是每个 anchor 拥有自己的时空“过滤器”。
*   **创新贡献**：
    1.  **ACFA**：提出了基于嵌入的动态权重预测机制。
    2.  **DLGD**：通过解耦策略，避免了锚点与局部几何信息的过度耦合。
    3.  **DACW**：针对无人机视角下非均匀采样分布设计的坐标warp技术，提升了有限网格的利用率。

### 5. 实验分析
*   **关键结论**：在 UAV-Arc4D、VisDrone 和 UAVDT 数据集上，AdaAnchor4D 在 PSNR/SSIM 指标上均优于目前主流的 4D-SFGS 和 MoRel 等方法。
*   **优势**：渲染质量高、动态细节捕捉准确、训练过程稳定（得益于零初始化策略）。
*   **局限**：对相机姿态预估计有一定依赖，在极度稀疏或遮挡严重场景下的表现仍有提升空间。

### 6. 实用指南
*   **开源与实现**：代码计划开源。关键在于实现高效的 `Aggregation MLP`，并在优化阶段通过“零初始化”输出层确保训练平稳启动。
*   **迁移与应用**：该方法非常适合需要处理“局部动态差异大”的任务（如无人机航拍、城市交通流分析）。只需替换对应的锚点生成模块即可。

### 7. 总结
*   **核心思想**：基于 anchor 状态及时间信息，动态调节特征聚合权重，实现高精度建模。
*   **速记版 Pipeline**：
    1.  统计锚点分布并扭曲查询坐标；
    2.  根据锚点特征和时间预测聚合权重；
    3.  加权求和得到动态特征；
    4.  解耦预测位置偏移和局部形变；
    5.  最终渲染。

**Key Findings:**

- To address this challenge, we propose AdaAnchor4D, an adaptive anchor deformation framework for monocular UAV dynamic scene reconstruction.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28320v1)
- [arXiv](https://arxiv.org/abs/2607.28320v1)

---

<a id='2607.28243v1'></a>
## [EgoGenesis: Egocentric World-Action Modeling with Online Anchored Projective Memory and Action-3D RoPE](https://arxiv.org/abs/2607.28243v1)

**Authors:** Zexuan Yan, Yuzhou Wu, Yue Ma, Zonghang He, Kaibo Yin, Xiaobing Tu, Yinggui Wang, Jinkui Ren, Xiantao Zhang, Shijian Wang, Jinghong Liu, Linfeng Zhang

**Published:** 2026-07-30

**Categories:** cs.CV, cs.AI

**Abstract:**

Egocentric video offers rich manipulation experience for embodied AI, yet collecting diverse egocentric data across scenes, objects, motions, and embodiments remains costly. We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data. \method{} builds on a pretrained video generation prior and introduces two geometry-aware conditioning mechanisms. Online Anchored Projective Memory (OAPM) preserves a first-frame 3D scene anchor while periodically refreshing a recent state during autoregressive generation. Action-3D Rotary Position Embedding (A3D-RoPE) encodes end-effector motion with camera-aware 3D rotary coordinates, injecting action geometry into skeleton-to-video cross-attention for precise control. Together, these components improve visual fidelity, geometric stability, and action alignment in long egocentric rollouts. Moreover, augmenting 400 real trajectories with 400 \method-generated trajectories improves out-of-distribution real-robot success from 77\% to 84\% on single-arm tasks and from 53\% to 70\% on dual-arm tasks, demonstrating that the synthesized data substantially improve downstream WAM generalization.

**Analysis:**

### 1. 摘要翻译
具身智能领域中， egocentric（第一人称）视频能提供丰富的操作经验，但跨场景、物体、动作和形态的多样化数据采集成本高昂。我们提出了 **EGOGENESIS**，一个egocentric世界-动作模拟器，通过合成可控、高质量的操作视频来扩展稀缺的真实世界训练数据。EGOGENESIS建立在预训练视频生成先验之上，并引入了两种几何感知调节机制：**在线锚定投影记忆 (OAPM)** 在自回归生成过程中保留首帧3D场景锚点并周期性刷新近期状态；**动作-3D旋转位置编码 (A3D-RoPE)** 利用相机感知的3D旋转坐标编码末端执行器运动，将动作几何特征注入骨架到视频的交叉注意力机制中，以实现精确控制。这些组件共同改善了长序列生成中的视觉保真度、几何稳定性和动作对齐。此外，使用EGOGENESIS生成的400条轨迹增强400条真实轨迹，在单臂任务上的OOD成功率从77%提高到84%，在双臂任务上从53%提高到70%，证明了合成数据能显著改善下游WAM（世界动作模型）的泛化能力。

---

### 2. 方法动机分析
*   **驱动力**：旨在解决具身智能中“几何感知的egocentric视频数据稀缺”问题，通过合成数据增强真实机器人任务的泛化性能。
*   **现有痛点**：
    1.  缺乏几何约束：导致生成的视频动作轨迹与场景不符（如相机漂移、物体形变）。
    2.  缺乏长期一致性：随着生成时长增加，场景背景和物体识别会逐渐漂移，难以保持第一人称视角的稳定性。
*   **研究假设**：通过显式引入3D几何锚定和将动作空间投影到旋转位置编码中，可以在生成过程中约束动作与场景的相对几何关系，从而实现长序列的一致性生成。

---

### 3. 方法设计详解
*   **模型结构**：EGOGENESIS基于DiT（Diffusion Transformer）架构，主要包含两个核心模块：
    1.  **OAPM (Online Anchored Projective Memory)**：将场景信息解耦为两部分：一个不可变的“初始场景锚点 (Ma)”和动态刷新的“近期状态片段 (Mr)”。这种设计保证了全局布局不漂移，同时允许局部交互动态更新。
    2.  **A3D-RoPE (Action-3D Rotary Position Embedding)**：这是本论文的核心创新。它不再简单地将动作作为条件注入，而是将骨架/末端执行器的3D坐标转换为旋转角度，直接作用于cross-attention的Q/K矩阵。这意味着模型“理解”了物体在3D空间中的运动逻辑，而非仅仅拟合像素像素块。
*   **流程总结**：
    1.  初始化：利用首帧图像构建Ma，设置空Mr。
    2.  推理/生成：在每一步block生成中，将Ma和Mr的特征作为条件注入。
    3.  几何注入：通过A3D-RoPE将3D动作坐标映射到特征空间的旋转相位，增强模型对运动路径的几何感知。
    4.  在线刷新：每隔一定stride（sr），解码已生成的视频块，更新Mr，确保长期一致性。

---

### 4. 方法对比分析
*   **本质区别**：区别于以往仅依赖2D关键点或 masks 的方法，EGOGENESIS通过“投影几何特征”显式地将3D metric信息引入生成过程。
*   **创新点**：A3D-RoPE将空间坐标作为位置编码参数，实现了动作与视频特征在几何层面的“硬”对齐。
*   **适用场景**：高精度要求、长序列交互、多模态（人手/机械臂）的机器人操作仿真任务。

---

### 5. 实验分析
*   **验证方法**：在EgoDex、AgiBot等数据集上进行生成质量评估，并在Tianji M6机器人上进行下游任务的OOD泛化验证。
*   **关键结论**：在所有指标中（包括物理忠实度Phys.Faith和背景一致性Bg. Cons.），模型均优于现有的EgoSim等主流方法。
*   **优势/局限**：优势在于显著提升了下游任务的泛化成功率（如单臂7%提升，双臂17%提升）；局限在于对预训练视频 prior 的依赖较大，如果底层 prior 对特定物体理解较差，生成的细节仍有瑕疵。

---

### 6. 实用指南
*   **开源情况**：已开源，项目主页：https://egogenesis.github.io/
*   **关键点**：注意A3D-RoPE在推理时需严格对齐anchor frame的相机参数。训练时需要平衡动作与视频生成分支的权重，文中提到使用SNR-shift技术来优化生成质量。
*   **迁移建议**：该框架易于迁移至任何具备DiT架构的视频生成模型，只需将其交叉注意力层替换为本文定义的GatedCrossAttn结构，并配置相应的A3D-RoPE适配器。

---

### 7. 总结
*   **核心思想**：通过3D几何锚定与动作旋转编码，实现一致性与可控性并存的生成。
*   **速记版pipeline**：
    1. **锚定场景**：固定首帧几何信息防漂移。
    2. **实时刷新**：定期更新近期状态维持一致性。
    3. **几何编码**：将空间动作转化为旋转矩阵。
    4. **交叉注入**：在注意力层实现空间动作约束。

**Key Findings:**

- We present \method, an egocentric world-action simulator that synthesizes controllable, high-quality manipulation videos to expand scarce real-world training data.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28243v1)
- [arXiv](https://arxiv.org/abs/2607.28243v1)

---

<a id='2607.28227v1'></a>
## [Qwen-UI-Agent Technical Report: Toward Next-Generation Real-World Centric Foundation GUI Agents](https://arxiv.org/abs/2607.28227v1)

**Authors:** Hanzhang Zhou, Panrong Tong, Xu Zhang, Quyu Kong, Chenglin Cai, Tianyu Xia, Gongjie Zhang, Jianan Zhang, Long Li, Long Chen, Lei Wang, Gaole Dai, Pengxiang Li, Liangyu Chen, Yue Wang, Steven Hoi

**Published:** 2026-07-30

**Categories:** cs.AI, cs.CV

**Abstract:**

GUI agents have the potential to become a general purpose executor over existing digital devices. To advance them toward real-world use, we envision agents that operate reliably on real devices, execute workflows across platforms, combine GUI interaction with CLI execution, complete long-horizon tasks, proactively initiate useful services, and autonomously improve their capabilities with minimal human effort. Guided by this vision, we present Qwen-UI-Agent, a real-world centric foundation GUI agent spanning mobile, computer-use, web, and DeepSearch environments. Qwen-UI-Agent combines diverse sandbox environments with a large-scale real-device mobile runtime. Its unified action space interleaves GUI operations with CLI execution and generates batched actions in a single model turn. An AutoResearch-style data flywheel uses agents to construct tasks and environments, diagnose failures, and plan subsequent iterations. Online RL supports training on trajectories exceeding 100 turns, with over 10,000 concurrent environments accelerating rollout. A lightweight harness layer supports proactive service initiation and stateful workflows across mobile and computer.   Across a broad suite of evaluations, Qwen-UI-Agent sets state-of-the-art performance on mobile-use benchmarks while delivering competitive performance on computer- and browser-use tasks against frontier models, including Opus 4.8, Gemini 3.1 Pro, and GPT-5.6 Sol. On mobile use, it achieves 82.1% on MobileWorld, 92.2% on MobileWorld-Real, and 97.5% on AndroidDaily. On computer use, it achieves 79.5% on OSWorld-Verified and a 40.0% partial-progress score on OSWorld-v2. On browser use and GUI grounding, it achieves 73.6% on WebArena and 81.5% on ScreenSpot-Pro, respectively.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对这篇关于 **Qwen-UI-Agent** 的技术报告进行了深入分析。以下是针对该论文的专业评估：

### 1. 核心贡献总结
Qwen-UI-Agent 提出了一个跨平台、以真实世界为中心的通用 GUI 智能体框架，旨在通过整合移动端、桌面端及 Web 端环境，实现复杂的跨设备长序列任务自动化。该工作通过创新的“数据飞轮”自进化机制和大规模在线强化学习（RL），显著提升了智能体在真实设备环境下的操作成功率与长程决策能力，树立了 GUI 智能体领域的新性能基准。

### 2. 关键创新与方法论
*   **统一动作空间与批处理执行**：打破了 GUI 操作与 CLI（命令行）的界限，允许模型在单次推理轮次中输出批处理动作，有效降低了交互延迟，提高了处理复杂系统级任务的效率。
*   **AutoResearch 风格的数据飞轮（Data Flywheel）**：这不仅是一个自动化数据生成过程，更是一个“智能体驱动的智能体优化”闭环，实现了从环境构建、故障诊断到迭代规划的全流程自动化，极大地降低了对人类标注数据的依赖。
*   **大规模分布式训练与在线 RL**：支持超过 10,000 个环境的并发并行训练，并能优化超过 100 步的长序列任务，解决了传统智能体在处理长程决策时容易出现的“偏移”与“停滞”问题。
*   **轻量化部署架构**：通过轻量级 Harness 层实现状态保持，使得智能体具备了跨移动端和计算机的“主动服务发起”能力，使其不再仅仅是被动响应，而能具备一定的预见性。

### 3. 对计算机视觉领域的潜在影响
该论文对 CV 领域极具启发意义，原因在于：
*   **从静态感知向动态交互演进**：GUI 智能体将 CV 的任务重心从“视觉问答（VQA）”或“目标检测”转变为“视觉导向的动作空间决策”，推动了多模态大模型在像素级感知与动作输出之间的实时协同（Visual-to-Action Mapping）。
*   **多模态融合的深度实践**：在移动端和桌面端环境中，UI 布局的解析、文字的 OCR 以及图标的语义匹配均依赖于高性能的视觉模型。该工作验证了视觉特征在长链路决策中的核心驱动作用。
*   **真实世界数据的视觉挑战**：在真实设备（Real-device）上的高分表现表明，该模型能够克服现实世界中由于屏幕尺寸变异、视觉遮挡和 UI 动态渲染带来的强干扰，这对视觉抗干扰算法的鲁棒性提出了新标准。

### 4. 关联领域与应用场景
*   **数字孪生与自动化运维（AIOps）**：智能体可直接替代人类完成服务器配置、软件部署等复杂任务。
*   **无障碍技术（Accessibility）**：为视障或行动不便人士提供自动操控数字设备的辅助接口。
*   **个人数字助理**：具备跨终端（手机+电脑）的代理能力，能主动完成跨 App 的复杂服务流程（如：根据日历自动预订机票并同步到行程表）。
*   **移动端测试与自动化开发**：在 AndroidDaily 等真实环境下的高表现，可直接赋能软件开发流程中的自动化测试与回归验证。

### 5. 可推断的潜在局限性
*   **算力成本高昂**：尽管论文提到支持 10,000 个环境并发，但维持如此大规模的在线强化学习训练对计算资源（GPU 集群）的需求极高，普通开发者难以复现。
*   **安全性与隐私风险**：作为一个能够自主执行 CLI 命令并控制真实设备的智能体，在实际部署中如何防范恶意操作（如误删系统文件、执行高风险脚本）仍存在安全隐患。
*   **极端 UI 的泛化能力**：虽然在现有基准测试上表现出色，但面对未见过的、高度定制化或极端畸变的非标准 UI 界面时，视觉定位模型的潜在失效风险依然存在。
*   **任务中断后的恢复能力**：尽管强调了“状态保持”，但在复杂异步中断（如网络突然中断、系统强制弹窗）后的鲁棒性恢复机制在摘要中尚未详述。

**总结：** Qwen-UI-Agent 不仅仅是一个模型，它代表了 AI 从“生成式内容”向“操作式智能”转型的关键一步，标志着我们正进入“设备即环境”的具身智能新阶段。

**Key Findings:**

- Guided by this vision, we present Qwen-UI-Agent, a real-world centric foundation GUI agent spanning mobile, computer-use, web, and DeepSearch environments.
- Across a broad suite of evaluations, Qwen-UI-Agent sets state-of-the-art performance on mobile-use benchmarks while delivering competitive performance on computer- and browser-use tasks against frontier models, including Opus 4.8, Gemini 3.1 Pro, and GPT-5.6 Sol.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28227v1)
- [arXiv](https://arxiv.org/abs/2607.28227v1)

---

<a id='2607.28211v1'></a>
## [Scaling Vision-Language Models Is Not Enough to Mitigate Bias](https://arxiv.org/abs/2607.28211v1)

**Authors:** Ioannis Sarridis, Ioannis Kompatsiaris, Symeon Papadopoulos

**Published:** 2026-07-30

**Categories:** cs.CV

**Abstract:**

Vision-Language Models (VLMs) such as CLIP are now foundational to multimodal systems, yet their robustness to spurious correlations remains poorly understood at scale. We present the first large-scale empirical study of 194 publicly available VLMs, including 16 model families, covering a wide range of model sizes, 24 training datasets, and three evaluation benchmarks, namely ImageNet (overall performance), CelebA (typical single-attribute bias), and UrbanCars (complex multi-attribute biases). Across these settings, the Spearman correlation between model scale and performance weakens as evaluation shifts from ImageNet ($ρ{=}0.68$) to single-attribute ($ρ{=}0.48$) and further to multi-attribute ($ρ{=}0.05$) bias benchmarks. In contrast, properties of the training data (size and quality) show more consistent relationships with worst-group accuracy across both bias benchmarks. Notably, curated datasets yield improvements of up to 25% over uncurated alternatives at a comparable scale. Finally, the effect of architectural choices (e.g., patch size, image resolution) is highly context-dependent, varying with the nature of the benchmark, including the type of bias and its spatial distribution within images.

**Analysis:**

这是一份针对论文《Scaling Vision-Language Models Is Not Enough to Mitigate Bias》的深度分析报告：

### 1. 摘要翻译
视觉-语言模型（VLMs，如CLIP）已成为多模态系统的基石，但其在规模化部署下对虚假相关性（spurious correlations）的鲁棒性尚不清楚。我们对194个公开的VLM模型进行了首次大规模实证研究，涵盖了16个模型家族、24个训练数据集以及三个评价基准（ImageNet总体表现、CelebA单属性偏差和UrbanCars复杂多属性偏差）。结果显示，随着偏见复杂度增加，模型规模与性能的相关性显著减弱（从ImageNet的$\rho=0.68$降至多属性偏差下的$\rho=0.05$）。相比之下，训练数据的大小和质量是更为稳健的预测指标，通过精心策划的数据集，最差组准确率（WGA）可提升高达25%。此外，架构选择的影响高度依赖于具体的偏差类型和任务。

### 2. 方法动机分析
- **驱动力**：打破“规模即正义”的迷思。当前领域过度依赖ImageNet等总体表现指标，假设模型越大越好，但忽略了模型在现实复杂环境下的鲁棒性表现。
- **现有方法痛点**：目前尚无研究系统地衡量模型设计选择（架构、数据、规模）如何影响对虚假相关性的抵抗能力。
- **研究假设**：VLM在面临不同复杂度的虚假相关性时，存在“偏见复杂度敏感性”（bias complexity sensitivity），即传统的缩放定律（Scaling Laws）在偏见鲁棒性任务中失效。

### 3. 方法设计详解
本研究采用了大规模实证评估框架：
- **评估逻辑**：将模型从总体认知能力（ImageNet）到简单偏差（CelebA：性别相关偏见）再到复杂偏差（UrbanCars：背景+共现物双重偏差）进行全维度测试。
- **WGA作为核心指标**：作者通过计算最差组准确率（WGA）来定量评估模型抵抗偏见的上限，公式为$WGA(f) = \min_{g \in G} \frac{1}{|D_g|} \sum_{(x,y') \in D_g} 1[f(x)=y']$。
- **解耦分析**：为了排除模型家族和数据的混淆影响，作者识别了11组“匹配对照组”（Matched Groups），固定其余超参数，仅改变单一变量（如模型大小、训练数据源、分辨率或补丁大小），从而精确量化单一因素的因果影响。

### 4. 方法对比分析
- **本质区别**：与现有研究仅关注模型性能不同，本研究通过“复杂度分级”评估，揭示了模型性能与偏见对抗能力的内在冲突。
- **核心创新**：提出了“偏见复杂度敏感性”概念，证实了盲目追求模型参数规模在复杂场景下反而可能降低对偏差的防御力（如UrbanCars WGA随规模增加平均下降4.2%）。

### 5. 实验分析
- **关键结论**：
    1. **数据决定鲁棒性**：训练数据的大小和 curation 质量是鲁棒性的最强预测指标。
    2. **规模陷阱**：模型规模是提升ImageNet准确率的利器，但对处理虚假相关性贡献微乎其微，甚至有害。
    3. **Token颗粒度**：较小的patch size（更高的Token密度）在复杂任务中表现更好。
- **局限性**：受限于现有公开开源模型，未包含极大规模的闭源模型（如GPT-4o等），且实验目前主要针对零样本设置。

### 6. 实用指南
- **开源情况**：代码已开源（[项目链接](https://github.com/gsarridis/vlm-spurious-robustness)）。
- **策略建议**：在追求鲁棒性的实际应用中，开发者应优先选择“数据质量”优异的模型（如使用DFN过滤的数据集），而非盲目扩充参数量；对于复杂任务，优先考虑高分辨率和细粒度token化架构。

### 7. 总结
- **核心思想**：规模并非万能，数据 curation 与细粒度架构是解决偏见问题的关键。
- **速记版Pipeline**：
    1. 选定不同复杂度的基准数据集；
    2. 使用控制变量法识别架构与数据影响；
    3. 评估不同规模模型在最差子群下的表现；
    4. 筛选出Pareto最优模型组合。

**Key Findings:**

- We present the first large-scale empirical study of 194 publicly available VLMs, including 16 model families, covering a wide range of model sizes, 24 training datasets, and three evaluation benchmarks, namely ImageNet (overall performance), CelebA (typical single-attribute bias), and UrbanCars (complex multi-attribute biases).

**Links:**

- [PDF](https://arxiv.org/pdf/2607.28211v1)
- [arXiv](https://arxiv.org/abs/2607.28211v1)

---

