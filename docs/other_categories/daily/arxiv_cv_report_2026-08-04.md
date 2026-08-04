time: 20260804

# Arxiv Computer Vision Papers - 2026-08-04

## Executive Summary

## 执行摘要（2026-08-03）

本期 arXiv 计算机视觉论文呈现出 **具身智能/机器人学习、高效生成模型、基础视觉模型与经典任务精细化** 四条主线。其中，与机器人相关的论文占比最高（4/10），显示“视觉 + 机器人”的交叉研究正成为当下热点；同时，视频生成、3D 场景表示、开放词汇分割等方向也在追求“效率”与“可扩展性”。以下是主要观察：

---

### 一、主要主题与趋势

1. **机器人学习走向数据驱动与真实部署**
   - 从人类第一视角视频合成机器人训练数据（Ego2Robot）；
   - 全身体感遥操作 + 全景感知驱动的移动操作 VLA（Panorama-Aware VLA）；
   - 人形机器人运动跟踪中的摔倒恢复与平滑类人行为（StableMimic）；
   - 对行为克隆中“动作分块”机制的深入分析（Action Chunking）。

2. **生成式视觉模型追求高效与高质量**
   - 视频生成提出 Token Radius Attention，降低注意力计算成本；
   - 大基线单目新视角合成借助隐式高斯解码（InfiniSplat），提升复杂场景重建能力。

3. **基础视觉模型进入“单次前向 + 开放词汇”阶段**
   - EOVSAM 仅用一次前向即可完成开放词汇分割，直接建在 SAM 3 之上；
   - Douyin 多模态嵌入模型展示了工业级多模态表征的落地路径。

4. **小目标检测与传感器标定持续精细化**
   - DyFrDet 结合动态频率抑制与标签消歧，解决小目标检测中的判别难题；
   - CalibBEV 通过 BEV 对齐实现 LiDAR-相机标定，面向实际自动驾驶系统。

---

### 二、特别值得关注的重要/创新论文

- **Ego2Robot**：直接将大规模人类第一视角数据转化为机器人训练数据，有望缓解机器人数据匮乏瓶颈，具备很强的可扩展性。
- **InfiniSplat**：将隐式解码与 3D Gaussian Splatting 结合，支持大基线单目新视角合成，是 3D 视觉与生成模型融合的重要进展。
- **EOVSAM**：将开放词汇分割与 SAM 3 的强分割先验统一在一次前向中，显著降低推理成本，实用价值高。
- **DyFrDet**：在频率域动态抑制干扰并结合标签消歧，为小目标检测提供了一种新的思路，值得小目标/遥感/检测方向研究者关注。
- **Why Does Action Chunking Improve Behavioral Cloning?**：对机器人学习常用技巧进行原理性剖析，有助于社区从“经验调参”走向“可解释设计”。

---

### 三、新兴研究方向与技术

- **跨域数据转化**：人类视频 → 机器人动作/轨迹数据，成为具身智能数据扩展的重要手段。
- **机器人与基础模型的结合**：全景视觉-语言-动作模型、遥操作学习、人形机器人安全恢复行为等方向快速涌现。
- **高效注意力机制**：视频生成中通过局部 token 半径注意力大幅降低计算量，是长视频/高分辨率生成的重要趋势。
- **隐式表示 + 显式 3D 基元**：InfiniSplat 表明隐式解码与 3DGS 的结合能更好处理大基线、复杂遮挡场景。
- **开放词汇分割轻量化**：一次前向完成多任务/多类别分割，正在成为 SAM 类模型落地的关键方向。
- **频率域/几何对齐在视觉任务中的应用**：小目标检测和传感器标定都开始利用频率特性或 BEV 几何一致性。

---

### 四、建议精读论文

按优先级推荐：

1. **Ego2Robot** —— 对机器人数据合成和具身智能研究具有全局意义。
2. **EOVSAM** —— 若关注开放词汇分割或 SAM 系列应用的效率问题。
3. **InfiniSplat** —— 若从事 3D 重建、新视角合成或 Gaussian Splatting 相关工作。
4. **DyFrDet** —— 若研究小目标检测、遥感或细粒度识别。
5. **Why Does Action Chunking Improve Behavioral Cloning?** —— 若从事机器人模仿学习、动作生成研究，值得精读以获得理论洞察。

对于更偏系统的研究者，CalibBEV 和 Panorama-Aware VLA 也具有很强的工程参考价值。

---

## Table of Contents

1. [Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data](#2608.02580v1)
2. [Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](#2608.02547v1)
3. [Douyin Multimodal Embedding Model Technical Report](#2608.02148v1)
4. [Token Radius Attention for Efficient Video Generation](#2608.02504v1)
5. [DyFrDet: Towards Accurate Small Object Detection via Dynamic Frequency Suppression with Label Disambiguation](#2608.02495v1)
6. [InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis](#2608.02437v1)
7. [StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking - Learning Beyond the Tracking Distribution for Structured Post-Fall Behavior](#2608.02385v1)
8. [CalibBEV: LiDAR-Camera Calibration via BEV Alignment](#2608.02309v1)
9. [EOVSAM: Efficient Open-Vocabulary Segmentation with SAM 3 in One Pass](#2608.02284v1)
10. [Learning Panorama-Aware VLA for Mobile Manipulation with Whole-Body Teleoperation](#2608.02257v1)

---

## Papers

<a id='2608.02580v1'></a>
## [Ego2Robot: Scalable Robot Data Synthesis from Egocentric Human Data](https://arxiv.org/abs/2608.02580v1)

**Authors:** Ye Wang, Pei Lin, Xiong-Hui Chen, Haoqi Yuan, Zhixuan Liang, Yiyang Huang, Anzhe Chen, Zixing Lei, Jie Zhang, Tao Zhang, Haoyang Li, Tong Zhang, Chenxi Xiao, Ziyuan Jiao, Qin Jin

**Published:** 2026-08-03

**Categories:** cs.RO

**Abstract:**

Learning generalizable robot manipulation policies requires large-scale and diverse demonstration data. Egocentric human manipulation videos offer rich scene and task diversity, and prior work has shown that retargeting and rendering such videos into robot-format data can yield effective per-task policies at small scale. However, whether this approach can provide pretraining benefits for vision-language-action models at scale remains unexplored. We present \textbf{Ego2Robot}, a scalable pipeline that converts egocentric human manipulation videos into robot training data through action retargeting, robot-arm visual synthesis, and multi-level quality curation. Ego2Robot supports both curated datasets and in-the-wild videos, producing 18,561 hours of robot training data spanning 15 robot morphologies, making it the largest ego-to-robot dataset to date. To evaluate generalization, we extend RoboTwin2.0 with disentangled perturbation axes covering visual appearance, scene layout, embodiment morphology, and task semantics. Experiments show that joint pretraining on Ego2Robot-synthesized and robot data consistently improves out-of-distribution generalization across multiple perturbation types, with benefits validated on real-robot deployment. Project page: https://www-ye.github.io/ego2robot_blog/

**Analysis:**

### 1. 摘要翻译
学习可泛化的机器人操作策略需要大规模、多样化的示范数据。以自我为中心的（Egocentric）人类操作视频提供了丰富的场景和任务多样性，既往工作已证明将此类视频重定向并渲染为机器人格式数据可以产生有效的任务特定策略，但这种方法在大规模视觉-语言-动作（VLA）模型预训练中的潜力尚待探索。我们提出了 **Ego2Robot**，一个可扩展的流水线，通过动作重定向、机器人手臂视觉合成和多级质量筛选，将以人为中心的视频转换为机器人训练数据。Ego2Robot 支持已标注和无标注的“野外”视频，产生了 18,561 小时的机器人训练数据，涵盖 15 种机器人形态，是迄今为止最大的“人转机”（ego-to-robot）数据集。为了评估泛化能力，我们扩展了 RoboTwin2.0，引入了涵盖视觉外观、场景布局、本体形态和任务语义的解耦扰动轴。实验表明，联合预训练能持续提升模型在多种分布外（OOD）环境下的泛化性能，并在真实机器人部署中得到了验证。

### 2. 方法动机分析
- **驱动力**：机器人数据采集成本高、规模受限，而人类第一视角视频蕴含丰富的操作先验，具备海量获取潜力。
- **现有方法痛点**：既往研究（Retarget-and-render）多局限于小规模任务；缺乏针对不同机器人形态的标准化跨平台数据转换；且缺乏针对视觉、语义、形态等单一维度的解耦泛化评估。
- **研究假设**：尽管存在巨大的形态差异，但 egocentric 视频中蕴含的交互规律在经过适当的动作与视觉对齐后，能为机器人策略提供可迁移的鲁棒性增益。

### 3. 方法设计详解
Ego2Robot 流水线包含三个核心阶段：
- **动作对齐（Action Alignment）**：将手部关键点映射为末端执行器（EEF）轨迹。通过 Savitzky-Golay 滤波平滑噪声，并根据动作速度分布对数据进行抽样，以匹配机器人 teleoperation 的物理动力学。
- **视觉对齐（Visual Alignment）**：核心是将“人手”替换为“机器人手臂”。
  - **分割与修复**：利用 SAM 3 提取手臂掩码，结合 ProPainter 进行视频修复以重构背景。
  - **底座优化**：这是关键步骤，通过求解 Inverse Kinematics (IK) 在 15 种机器人模型中寻找最佳底座位姿，确保轨迹的运动学可行性（最大覆盖范围）。
  - **深度合成**：通过深度感知合成，将渲染的机器人手臂正确嵌入场景（遮挡关系处理）。
- **质量筛选（Quality Curation）**：
  - L1 (Pipeline-internal)：自动剔除 IK 失败、碰撞等无效帧。
  - L2 (Statistical)：剔除动作突变和噪声。
  - L3 (VLM Consistency)：利用 Qwen3.5 作为审核员，判定合成视频语义是否符合任务描述。

### 4. 方法对比分析
- **本质区别**：从传统的“单任务/单形态”适配进化为“多形态/大规模”通用预训练范式，强调数据流的自动化和语义一致性。
- **创新贡献**：实现了 15 种不同形态机器人的并行渲染；提出了基于深度感知与 VLM 审核的高保真数据过滤流水线；构建了完全解耦的泛化评估基准。
- **适用场景**：适用于任何具备第一视角视频源（如 YouTube、EgoDex 等）的机器人策略预训练，特别适合解决机器人本体或环境布局的分布外问题。

### 5. 实验分析
- **验证方法**：在 RoboTwin2.0 上进行多维度（视觉、场景、形态、语义）的 OOD 测试，并在真实 ARX ACone 机器人上进行 Long-horizon 任务验证。
- **关键结果**：Ego2R+Robot (1:1) 在分布外测试中性能提升显著（+5.9%），尤其在视觉外观和任务语义上优势明显。
- **优势与局限**：优势在于泛化 invariance（不变性）；局限在于对极度复杂光照下的遮挡处理可能产生伪影，且对末端细粒度手指动作的处理尚显粗糙。

### 6. 实用指南
- **开源情况**：项目主页已开放，关键实现依赖 MuJoCo IK 求解器。
- **实现细节**：建议在进行动作重定向时，务必对齐手部左右手标志位（Handedness Sign），并在多形态渲染时严格控制底座空间优化，以保证轨迹的可执行性。
- **迁移可能**：该流水线具有极强的通用性，仅需修改 URDF 文件和对应的 IK 约束，即可接入任何新型机器人。

### 7. 总结
- **核心思想**：利用 VLM 辅助的解耦流水线，将海量人手视频转化为多形态机器人的通用先验数据。
- **速记版pipeline**：
  1. **手部动作提取**：从人手关键点计算机器人末端轨迹。
  2. **机器人渲染**：自动选择机器人底座并合成手臂图像。
  3. **VLM质量过滤**：自动剔除逻辑不通或不匹配的合成数据。
  4. **联合训练**：将合成数据与真实机器人数据混合训练。

**Key Findings:**

- We present \textbf{Ego2Robot}, a scalable pipeline that converts egocentric human manipulation videos into robot training data through action retargeting, robot-arm visual synthesis, and multi-level quality curation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02580v1)
- [arXiv](https://arxiv.org/abs/2608.02580v1)

---

<a id='2608.02547v1'></a>
## [Why Does Action Chunking Improve Behavioral Cloning Performance in Robotic Control?](https://arxiv.org/abs/2608.02547v1)

**Authors:** Filippo Lazzati, Kyle Stachowicz, William Chen, Alberto Maria Metelli, Andrew Wagenmaker, Sergey Levine

**Published:** 2026-08-03

**Categories:** cs.RO

**Abstract:**

Action chunking---predicting and executing multiple actions instead of a single action---has proven to be a critical component for learning effective robotic control policies. However, our precise understanding of why action chunking improves performance has remained limited. In this work we seek to close this gap. Through rigorous experimental evaluations in both simulated and real-world settings, we show that existing hypotheses for the success of action chunking---temporal consistency, horizon reduction, and representation learning---fail to explain the success of action chunking. Instead, we find that action chunking benefits from greater non-Markovian expressivity and reduced compounding error compared to Markovian policies, but, in many settings of interest, these effects can be fully captured by delayed policies, which at each step predict a single action based on the observation $k$ steps in the past. We then show that there exists an additional benefit of action chunking that we refer to as implicit ensembling. In particular, by learning a diversity of temporal relationships (that is, $a_t | o_t, a_t | o_{t-1}, \ldots$), action-chunked policies exhibit behavior matching that of a model ensemble, increasing their robustness and generalization ability over policies that only learn a single temporal relationship. Building on these insights, we show that in simulated and real-world robotic control settings, we can match the performance of action chunking without action chunking---by deploying an action chunking policy as an ensemble of policies with randomized delays. Furthermore, we propose a policy class that amplifies the benefits of action chunking by explicitly instantiating an ensemble, and which we show significantly improves over the performance of action chunking in many domains.

**Analysis:**

以下是对该论文的深度分析：

### 1. 摘要翻译
动作分块（Action Chunking）——即预测并执行一连串动作而非单个动作——已被证明是训练高效机器人控制策略的关键组件。然而，关于其为何能提升性能的理解尚不深入。本文旨在填补这一空白。通过在模拟和真实世界环境中的严谨实验评估，我们证明，现有的关于动作分块成功的假设（如时间一致性、地平线缩减和表征学习）均无法完全解释其表现。相反，我们发现动作分块的收益主要源于：更强的非马尔可夫表达能力、降低了复合误差（相比马尔可夫策略），且在许多场景下，这些效果可完全通过“延迟策略”（在每一步基于过去 $k$ 步的观测预测单个动作）来复现。此外，我们揭示了动作分块的另一个额外收益，即“隐式集成”（Implicit Ensembling）。通过学习多种时间关系，分块策略展现出类似模型集成的行为，增强了鲁棒性和泛化能力。基于这些洞察，我们展示了通过部署带有随机延迟的集成策略，无需显式动作分块即可匹配甚至超越其性能。

### 2. 方法动机分析
*   **驱动力**：解决动作分块技术长期存在的“黑盒”质疑，明确其性能提升的根本原因。
*   **痛点**：学界广泛应用动作分块，但对其作用机制的理解仅停留在经验层面的猜想（如 temporal consistency 等），缺乏理论和实验验证。
*   **核心假设**：动作分块的成功并非源于其预设的“预测长序列”本身，而是因为分块操作**隐式地实现了对历史信息的利用（即处理非马尔可夫性）以及对多时间尺度关系的集成（即隐式集成）**。

### 3. 方法设计详解
*   **核心 Pipeline**：
    1.  **延迟策略（Delayed Policy）**：将传统动作分块策略转化为 $\pi(a_t | o_{t-k})$，即利用过去时刻的观测来预测当前动作，从而模拟分块策略在处理观测延迟时的鲁棒性。
    2.  **隐式集成（Implicit Ensembling）**：动作分块策略 $\pi(a_{t:t+k} | o_t)$ 本质上是一个多头网络，输出的序列包含了 $\{a_t|o_t, a_t|o_{t-1}, \dots, a_t|o_{t-k+1}\}$ 等多个时间映射关系。
    3.  **随机延迟集成（Randomized Delay Ensemble, RDE）**：为了显式利用上述集成效应，论文提出在推理时随机采样延迟 $i \in \{0, \dots, n-1\}$，并执行 $\pi(a_t | o_{t-i})$，从而以更低开销实现比单次动作分块更好的性能。
*   **算法本质**：通过数学证明（Theorem 2），证明了在确定性平滑动力学下，利用过去状态进行预测能有效减少复合误差，而动作分块和延迟策略在数学上具有等价的误差界限。

### 4. 方法对比分析
*   **本质区别**：与现有 ACT 或 Diffusion Policy 不同，本文并不追求长序列的“连贯性”，而是将动作分块解构为“对过去观测的利用”和“对多延迟的集成”。
*   **创新贡献**：
    1.  揭穿了“时间一致性”作为动作分块核心收益的假象。
    2.  提出“隐式集成”视角，解释了分块策略泛化能力强的深层原因。
    3.  提供了一种无需显式动作分块的替代方案（RDE），在复杂任务中效果更好。

### 5. 实验分析
*   **验证方法**：在 Libero-90 和 Robomimic 两个主流机器人控制基准上进行对比，包含模拟和真实物理机器人实验。
*   **关键结论**：在绝大多数任务中，延迟策略足以捕获非马尔可夫性；当任务复杂度增加（如 Robomimic），RDE 策略通过显式集成，成功匹配或超越了传统动作分块。

### 6. 实用指南
*   **实现细节**：
    *   **策略部署**：如果不想使用复杂的分块预测，可以直接训练一个单步预测网络，但在输入端增加观测延迟。
    *   **集成部署**：若性能瓶颈源于泛化，采用 RDE 策略（随机采样延迟）是非常低成本的提升手段。
*   **迁移建议**：该方法适用于任何基于 BC 的机器人策略，特别是当机器人执行多阶段任务或环境存在观测噪声时，迁移只需调整采样延迟区间。

### 7. 总结
*   **核心思想**：动作分块是“延迟观测预测”与“隐式集成”的结合体。
*   **速记版 Pipeline**：
    1.  放弃对长序列预测的执念。
    2.  训练策略以学习多时间跨度的动作映射。
    3.  推理时随机选择不同的延迟步数进行预测。
    4.  通过集成这些预测结果，获得鲁棒的控制动作。

**Key Findings:**

- Through rigorous experimental evaluations in both simulated and real-world settings, we show that existing hypotheses for the success of action chunking---temporal consistency, horizon reduction, and representation learning---fail to explain the success of action chunking.
- Building on these insights, we show that in simulated and real-world robotic control settings, we can match the performance of action chunking without action chunking---by deploying an action chunking policy as an ensemble of policies with randomized delays.
- Furthermore, we propose a policy class that amplifies the benefits of action chunking by explicitly instantiating an ensemble, and which we show significantly improves over the performance of action chunking in many domains.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02547v1)
- [arXiv](https://arxiv.org/abs/2608.02547v1)

---

<a id='2608.02148v1'></a>
## [Douyin Multimodal Embedding Model Technical Report](https://arxiv.org/abs/2608.02148v1)

**Authors:** Haonan Chen, Chu Li, Zhicheng Wang, Yuanwei Liu, Yuanjiang Wang, Shaohua Jiang, Zhicheng Dou

**Published:** 2026-08-03

**Categories:** cs.IR, cs.CL, cs.CV

**Abstract:**

Multimodal representation learning is a cornerstone of modern AI. By encoding multimodal queries and targets into vectors, it powers industrial search and recommendation and underpins modern agents. Real-world platforms with complex modalities and massive-scale content, such as Douyin, Xiaohongshu, and YouTube, demand both efficiency under billion-scale indexing and fine-grained discrimination for hard matching. Existing MLLM embedding models rarely satisfy both. Contrastive models are efficient but rely on pair-level supervision too coarse for fine-grained distinctions, while CoT-based models improve discrimination through explicit generation impractical to serve online. We present Douyin Multimodal Embedding (DME), a model trained in two stages to combine both strengths. Stage 1 performs large-scale contrastive pre-training that establishes a unified multimodal embedding space with broad modality and task coverage. Stage 2 supplements semantic sufficiency, the property that an embedding is grounded in retrieval-relevant evidence and preserves fine-grained counterpart-side semantics, via two mechanisms. Evidence-Grounded Typed Latent Reasoning organizes retrieval evidence through hidden-space latent reasoning, and Cross-Conditional Reconstruction enforces counterpart-side semantics through cross-directional autoregressive reconstruction. Both act only during training and add only marginal query-side overhead, so DME serves as efficiently as a standard contrastive encoder. On MMEB-v2, DME reaches state-of-the-art results at comparable scales for its 2B and 9B variants (74.8 and 78.4), with especially strong video and visual-document tasks. In production, DME delivers a 2.92% relative gain on Douyin's in-house offline evaluation set, is deployed across Douyin scenarios such as generative, image, and AI search, and yields a 0.1% Lifetime (LT) gain in online A/B testing on Douyin search.

**Analysis:**

以下是对Douyin Multimodal Embedding (DME) 技术报告的深度分析：

### 1. 摘要翻译
多模态表示学习是现代AI的基石。Douyin、Xiaohongshu和YouTube等拥有海量内容和复杂模态的工业平台，既要求十亿级规模下的索引效率，又要求针对困难样本的细粒度语义区分能力。现有的多模态大语言模型（MLLM）嵌入方案通常难以兼顾二者：对比学习模型高效但监督信号粗糙；基于思维链（CoT）的模型判别力强但推理成本高。为此，我们提出了Douyin多模态嵌入（DME）。DME通过两阶段训练：第一阶段进行大规模对比预训练；第二阶段引入“基于证据的类型化潜在推理”和“交叉条件重构”机制，在保持Bi-Encoder推理效率的同时，增强了表示的语义完备性和细粒度判别力。实验表明，DME在MMEB-v2上达到SOTA，并已在抖音生产环境大规模部署，实现了显著的业务指标提升。

### 2. 方法动机分析
*   **驱动力**：解决工业界在超大规模搜索场景下，如何在推理延迟极低（Bi-Encoder架构）的前提下，实现比传统对比学习更精准的语义对齐。
*   **现有痛点**：纯对比学习（CLIP类）虽然对齐了全局向量，但丢失了“为什么相关”的细粒度证据；CoT类方法虽然精准，但生成式推理会导致计算延迟不可控，无法用于十亿级索引。
*   **研究假设**：通过在隐空间引入辅助的“潜在推理”路径和“内容重构”瓶颈，强迫模型在编码过程中“思考”并保留关键证据，从而在无需显式推理的情况下获得高表达力表示。

### 3. 方法设计详解
DME采用两阶段训练：
*   **阶段一（大规模对比预训练）**：在2500万样本上建立基础对齐，奠定稳定的多模态检索空间。
*   **阶段二（语义完备性学习）**：
    *   **证据引导的类型化潜在推理**：通过引入锚点（Anchor）标记定位多模态输入中的关键信息（如图像区域、文本片段），并结合类型化的潜在状态（如Localization, Align_pos等）进行隐式推理。这些 latent tokens 不生成显式文本，仅在Encoder前向过程中辅助生成 readout 向量。
    *   **交叉条件重构（NTP/MTP）**：在训练时，强迫查询（Query）侧嵌入重建文档（Document）侧内容，反之亦然。这构建了一个“语义瓶颈”，确保嵌入向量中包含了能够复原对方语义的关键信息。
    *   **读取机制**：将潜在推理的末端状态与证据池融合，最终提取一个紧凑的检索表示。

### 4. 方法对比分析
*   **本质区别**：VLM2Vec是“端到端编码”，CoT-VLM2Vec是“显式生成推理”，而DME是“隐空间推理+瓶颈重构”，它是效率与精度的完美平衡点。
*   **创新贡献**：首次将结构化的CoT推理过程“无损地”压缩进Bi-Encoder的隐状态中，且利用重构能力作为语义度量指标。
*   **适用场景**：对检索准确度要求极高、内容模态极其复杂（视频、OCR文档等）的工业级推荐/搜索系统。

### 5. 实验分析
*   **关键结果**：在MMEB-v2上，DME-2B/9B分别达到74.8/78.4的SOTA得分；且在抖音实际业务中，离线指标提升2.92%，线上A/B测试LT gain提升0.1%。
*   **优势**：极低的推理 overhead（相比不加latent token仅微增 <1ms），能够完美兼容现有的向量检索库。
*   **局限**：对教师模型（Seed-2.0-Pro）生成的结构化CoT数据依赖性较强，数据构建成本较高。

### 6. 实用指南
*   **实现细节**：NTP/MTP只在训练时使用，推理时必须丢弃。在构建训练数据时，建议使用强力的教师模型（如Qwen-VL系列）预先标注关键区域和推理路径。
*   **迁移建议**：若在自己的任务上实现，关键在于设计能够反映任务特点的“类型化潜在状态”（如对分类任务定义标签类token，对检索任务定义对比类token）。

### 7. 总结
*   **核心思想**：通过隐空间推理与生成式重构，将显式推理能力转化为紧凑的高质量向量表示。
*   **速记版pipeline**：
    1. 大规模对比预训练，打好底座。
    2. 加入锚点tokens，在Encoder内部执行隐式逻辑推理。
    3. 训练阶段强迫Query/Doc向量相互重构，压缩语义。
    4. 推理时仅使用最终的紧凑向量，保持毫秒级检索效率。

**Key Findings:**

- We present Douyin Multimodal Embedding (DME), a model trained in two stages to combine both strengths.
- On MMEB-v2, DME reaches state-of-the-art results at comparable scales for its 2B and 9B variants (74.8 and 78.4), with especially strong video and visual-document tasks.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02148v1)
- [arXiv](https://arxiv.org/abs/2608.02148v1)

---

<a id='2608.02504v1'></a>
## [Token Radius Attention for Efficient Video Generation](https://arxiv.org/abs/2608.02504v1)

**Authors:** Jiayu Chen, Zhikun Jiang, Maoliang Li, Jiayi Luo, Jiawei Yang, Zihao Zheng, Hengyi Zhang, Guojie Luo, Xiang Chen

**Published:** 2026-08-03

**Categories:** cs.CV

**Abstract:**

Video Diffusion Transformers (VDiTs) enable high-fidelity generation but incur quadratic cost from dense 3D self-attention. Existing head- and block-level sparse methods share computation budgets across queries, overlooking token-specific attention demand. We observe that retained density varies across queries yet correlates log-linearly with attention entropy, while dominant interactions form query-centered neighborhoods with token-dependent radii. Based on these findings, we propose Token Radius Attention (TRA), a training-free framework that maps query entropy to an analytic token budget and converts it into a temporally decayed radius without explicit key ranking. Fused entropy extraction, warm-up reuse, and block-sparse mask construction further reduce overhead. Across seven Wan2.1, Wan2.2, and HunyuanVideo T2V/I2V configurations, TRA retains only 9-19% of attention interactions and achieves 1.56x-2.05x speedup with competitive generation quality. Code is available at https://github.com/IF-LAB-PKU/Token-Radius-Attention.

**Analysis:**

### 1. 摘要翻译
视频扩散Transformer（VDiTs）在实现高保真视频生成的同时，因密集的3D自注意力计算面临二次方计算成本。现有的头（head）级和块（block）级稀疏方法在所有查询（query）间共享计算预算，忽略了不同查询对注意力需求的不同。我们观察到，保留的密度在不同查询间差异巨大，且与注意力熵呈对数线性相关，而主导的交互则表现为具有令牌依赖半径的查询中心型邻域。基于此，我们提出了Token Radius Attention (TRA)，这是一个无需训练的框架，它将查询熵映射为解析的令牌预算，并将其转换为随时间衰减的半径，无需显式的键排序。融合熵提取、预热重用和块稀疏掩码构建进一步降低了开销。在七种Wan2.1、Wan2.2和HunyuanVideo的文生视频/图生视频配置中，TRA仅保留了9%–19%的注意力交互，实现了1.56倍–2.05倍的加速，并保持了极具竞争力的生成质量。

### 2. 方法动机分析
*   **驱动力**：旨在解决VDiTs中自注意力机制计算负担重、且现有的稀疏化策略（头/块级）因假设查询计算需求统一而导致效率与保真度冲突的问题。
*   **现有方法痛点**：头/块级稀疏化方案往往“一刀切”，对简单的查询造成浪费，对复杂的查询则可能丢失关键信息，未实现真正意义上的查询级别自适应。
*   **核心直觉**：注意力熵能有效度量计算需求。低熵意味着注意力集中，高熵意味着分布广泛。主导的视觉交互通常围绕查询位置呈圆形分布，且随空间距离呈指数衰减。

### 3. 方法设计详解
TRA的Pipeline如下：
1.  **熵引导的令牌预算估算**：利用注意力熵 $H_i$（由在线softmax状态实时累积得出）直接估算每个查询的令牌预算 $B_i$。避免了昂贵的全局Top-k排序。
2.  **半径构建与转换**：将标量预算 $B_i$ 映射为查询特定的基准空间半径 $r_i$。引入时间距离衰减因子 $\phi(\delta)$，构建覆盖全视频帧的3D“圆柱形”空间支持，确保计算分配的结构化。
3.  **块稀疏掩码转换**：设计了一个融合CUDA内核，将逻辑上的令牌半径掩码转换为符合FlashInfer执行的块稀疏掩码。通过tile-major布局重新排列令牌，使得相邻视觉令牌在内存中连续，极大地提升了块稀疏存取效率。
4.  **计算重用**：在早期密集预热阶段计算熵，后续阶段直接重用该熵预算与半径图，通过牺牲极小的计算开销消除了每步计算掩码的开销。

### 4. 方法对比分析
*   **本质区别**：从传统的“固定规则稀疏”转向“查询自适应半径稀疏”。
*   **创新点**：建立了“熵 $\rightarrow$ 预算 $\rightarrow$ 半径”的解析映射模型，将复杂的动态稀疏选择转化为高效的几何半径计算，且不需要任何训练。
*   **适用场景**：所有基于自注意力机制的视频扩散模型（如Wan, HunyuanVideo），特别适合长视频及高分辨率任务。

### 5. 实验分析
*   **关键结论**：在保持PSNR与SSIM极其接近密集注意力（Dense）的情况下，实现了平均1.5-2倍的推理加速，显著优于SVG及Radial Attention基线。
*   **主要优势**：不仅大幅降低了FLOPs和内存占用，且在主观质量（如主体一致性、减少闪烁）上表现优异。
*   **主要局限**：对长程非局部运动的捕捉受限于预定义的半径模式；当前基于块稀疏后端实现，尚未达到极限的硬件底层优化。

### 6. 实用指南
*   **开源情况**：已开源，GitHub地址见文中。
*   **实现细节**：关键参数为时间衰减率（Wan系列取0.6，Hunyuan取0.95）；必须配置密集预热步长（建议25%）。
*   **迁移建议**：可直接将TRA模块接入标准的Transformer架构中，通过替换原始的softmax或在其后加Mask乘法实现。

### 7. 总结
*   **核心思想**：利用查询熵作为度量，动态配置半径以实现令牌级别的精细化注意力稀疏。
*   **速记版Pipeline**：
    1.  **预热**：密集计算前几步，记录各查询的注意力熵。
    2.  **映射**：根据熵计算所需的注意力半径。
    3.  **采样**：基于半径构建空间遮罩，过滤无关的键值块。
    4.  **加速**：使用融合内核执行高效的块稀疏注意力运算。

**Key Findings:**

- Based on these findings, we propose Token Radius Attention (TRA), a training-free framework that maps query entropy to an analytic token budget and converts it into a temporally decayed radius without explicit key ranking.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02504v1)
- [arXiv](https://arxiv.org/abs/2608.02504v1)

---

<a id='2608.02495v1'></a>
## [DyFrDet: Towards Accurate Small Object Detection via Dynamic Frequency Suppression with Label Disambiguation](https://arxiv.org/abs/2608.02495v1)

**Authors:** Zihan Yang, Yang Guo, Hongxing Zhang, Dan Lu, Siyuan Yao

**Published:** 2026-08-03

**Categories:** cs.CV, cs.AI

**Abstract:**

Despite the remarkable progress over the past decades, accurately identifying small objects remains challenging because of their insufficient visual cues. Previous works typically attempt to construct discriminative representation of the small objects. However, the wide range frequency domain noises and label ambiguities have been greatly overlooked, which significantly hinders the accurate localization. To address these issues, we propose a novel small object detection (SOD) detector termed DyFrDet, which is able to precisely localize the small object by dynamically suppressing the background distractions in frequency domain. Specifically, we propose a Dynamic Frequency-aware Feature Pyramid Network (DyFrFPN) to adaptively suppress low-frequency redundancy and excessive high-frequency noises. The DyFrFPN transforms the hierarchical features into frequency domain representation, and introduces a Dynamic Band Predictor (DBP) to preserve the discriminative components for small object identification. Afterwards, we present a novel Label Disambiguation Module (LDM), which leverages probabilistic distributions to explicitly model and alleviate the inherent ambiguity of target labels, yielding efficient improvement in localization precision of the small objects with low-resolution. Extensive experiments demonstrate that DyFrDet achieves state-of-the-art performance across multiple benchmarks, indicating its effectiveness and robustness in various challenging scenarios. Our code is available at https://github.com/ManOfStory/DyFrDet.

**Analysis:**

## 1. 摘要翻译
尽管过去几十年来在目标检测领域取得了显著进展，但由于视觉线索不足，准确识别小目标仍然具有挑战性。以往的研究通常试图构建小目标的判别性特征表示，但却大大忽视了宽频域噪声和标签歧义，这严重阻碍了精确的定位。为了解决这些问题，我们提出了一种名为DyFrDet的新型小目标检测（SOD）器，它能够通过动态抑制频域中的背景干扰来精确定位小目标。具体而言，我们提出了动态频率感知特征金字塔网络（DyFrFPN），以自适应地抑制低频冗余和过度的过高频噪声。DyFrFPN将分层特征转换为频域表示，并引入了动态频带预测器（DBP）来保留用于小目标识别的判别性成分。此外，我们提出了一种新颖的标签消歧模块（LDM），它利用概率分布来明确建模并缓解目标标签的固有歧义，从而有效提高了低分辨率小目标的定位精度。大量的实验表明，DyFrDet在多个基准测试中达到了最先进的性能，证明了其在各种复杂场景下的有效性和鲁棒性。我们的代码可在 https://github.com/ManOfStory/DyFrDet 获取。

---

## 2. 方法动机分析
*   **驱动力**：旨在解决复杂背景下小目标视觉信息稀疏导致的“定位模糊”和“背景干扰”问题。
*   **现有方法痛点**：传统方法过度依赖空间域特征，忽视了频域中存在的低频背景冗余和高频噪声干扰；同时，小目标低分辨率导致的标签标注歧义严重影响了回归质量。
*   **研究假设**：小目标的判别信息与背景信息在频域上是可分且可动态调整的，通过概率分布建模坐标回归可以有效对抗标注歧义。

---

## 3. 方法设计详解
*   **流程总结**：
    1.  **特征变换**：通过FFT（快速傅里叶变换）将FPN输出的特征图转入频域。
    2.  **动态频带预测（DBP）**：利用频域表示和空间特征，通过卷积块预测通道级的频率抑制阈值 $[\alpha_1, \alpha_2]$。
    3.  **动态滤波**：根据预测的阈值生成空间mask，直接滤除高频/低频成分，通过IFFT还原空间特征。
    4.  **标签消歧（LDM）**：将坐标回归建模为高斯分布，引入加权函数 $\omega(\sigma_m)$，在训练过程中动态降低高歧义样本的损失权重。
*   **模型结构**：DyFrFPN负责特征层面的“提纯”，LDM负责监督层面的“修正”。
*   **算法解释**：公式(7)定义的掩码 $M$ 实现频域门控；公式(12)的权重函数 $\omega(\sigma_m)$ 实现了对高不确定性标注的自动降权，类似于一种动态的样本筛选机制。

---

## 4. 方法对比分析
*   **本质区别**：与HS-FPN等使用静态高通滤波的方法不同，DyFrDet实现了“动态”且“全频段”的频域抑制，能够自适应地根据特征图内容决定保留哪些频段。
*   **创新贡献**：将频域抑制机制与标注不确定性建模结合，实现了从“特征输入”到“标签监督”的全流程优化。
*   **适用场景**：极小目标、高密集场景（如遥感图像、无人机监控）。

---

## 5. 实验分析（精简版）
*   **验证方法**：在AI-TOD和SODA（SODA-D/A）数据集上与SOTA方法对比。
*   **关键结果**：在AI-TOD测试集上，相比HS-FPN，AP值显著提升（+3.6%）。
*   **主要优势**：极强的抗干扰能力，特别是在高噪声环境下对小目标的召回率和定位精度提升明显。
*   **主要局限**：频域变换（FFT/IFFT）增加了计算开销，可能影响推理实时性。

---

## 6. 实用指南
*   **开源情况**：已开源，GitHub地址：https://github.com/ManOfStory/DyFrDet。
*   **实现细节**：
    *   **抑制策略激活**：建议在训练后期（如第24 epoch）才激活频域抑制，防止训练初期特征未收敛时过早滤除有用信息。
    *   **超参数**：$\beta=0.5$（抑制率）是权衡性能的关键。
*   **迁移可能**：该频域滤波模块可直接迁移至任何基于FPN的检测网络（如RetinaNet, FCOS），作为即插即用的模块增强特征提取能力。

---

## 7. 总结
*   **核心思想**：利用动态频域滤波与不确定性损失，对小目标实施“去噪”与“降歧”。
*   **速记版pipeline**：
    1. 特征图转为频域；
    2. 智能预测滤波范围并切除干扰噪声；
    3. 特征回归变为概率分布预测；
    4. 自动识别并弱化模糊标注数据的学习权重。

**Key Findings:**

- To address these issues, we propose a novel small object detection (SOD) detector termed DyFrDet, which is able to precisely localize the small object by dynamically suppressing the background distractions in frequency domain.
- Specifically, we propose a Dynamic Frequency-aware Feature Pyramid Network (DyFrFPN) to adaptively suppress low-frequency redundancy and excessive high-frequency noises.
- Afterwards, we present a novel Label Disambiguation Module (LDM), which leverages probabilistic distributions to explicitly model and alleviate the inherent ambiguity of target labels, yielding efficient improvement in localization precision of the small objects with low-resolution.
- Extensive experiments demonstrate that DyFrDet achieves state-of-the-art performance across multiple benchmarks, indicating its effectiveness and robustness in various challenging scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02495v1)
- [arXiv](https://arxiv.org/abs/2608.02495v1)

---

<a id='2608.02437v1'></a>
## [InfiniSplat: Implicit Gaussian Decoding for Large-Baseline Monocular View Synthesis](https://arxiv.org/abs/2608.02437v1)

**Authors:** Jiawei Wang, Hao Yu, Yongzhen Hu, Xinyi Yang, Tao Ni, Xin Zhan, Junbo Chen, Xiaowei Zhou, Ruizhen Hu, Sida Peng

**Published:** 2026-08-03

**Categories:** cs.CV

**Abstract:**

Single-image feed-forward 3D Gaussian Splatting (3DGS) aims to directly generate a renderable 3D scene representation from one input image, avoiding the cost of multi-view capture and per-scene optimization. However, existing methods are often constrained by a pixel-aligned representation, where Gaussians are predicted from fixed image-grid locations. Such pixel-aligned primitives can produce promising nearby-view renderings, but they remain weakly coupled to underlying scene surfaces and struggle to preserve coherent structures under large viewpoint shifts. We present InfiniSplat, a feed-forward single-image 3DGS framework that moves from a pixel-aligned representation toward a surface-aligned representation. InfiniSplat constructs this representation by first using geometry-guided sampling to place 2D supports according to depth-induced local surface structure, and then applying a query-conditioned implicit decoder to predict Gaussian attributes from the image features queried at these supports.By grounding support locations in geometry while decoupling Gaussian prediction from fixed pixel centers, InfiniSplat produces Gaussian layouts that better follow scene surfaces and reduce scattered primitives caused by grid discretization.Across multiple cross-dataset NVS evaluations, InfiniSplat achieves state-of-the-art performance compared with single-image feed-forward baselines, and demonstrates zero-shot generalization from Hypersim indoor synthetic training to complex open-world scenes.Project page: https://zju3dv.github.io/InfiniSplat.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我对 **InfiniSplat** 这篇论文的分析如下：

### 1. 论文核心贡献总结
InfiniSplat 提出了一种全新的前馈式单图 3D 高斯泼溅（3DGS）框架，成功打破了传统方法依赖“像素对齐（pixel-aligned）”的范式，转向了“表面对齐（surface-aligned）”的新路径。通过引入几何引导的采样策略与查询条件下的隐式解码器，该方法能够根据场景深度结构自适应地放置高斯基元，从而在大幅度视角变换下仍能保持几何结构的一致性与渲染质量。

### 2. 关键创新点与方法论
*   **摆脱固定网格束缚**：传统方法（如 PixelSplat 等）通常从固定图像网格预测高斯，导致在处理复杂几何时容易出现伪影。InfiniSplat 通过**几何引导采样（geometry-guided sampling）**，根据深度信息在表面结构上动态放置“2D 支撑点（2D supports）”。
*   **解耦预测机制**：将高斯属性的预测与像素中心解耦，通过**查询条件下的隐式解码器（query-conditioned implicit decoder）**从特征图中提取信息。这种设计使得高斯基元的分布不再受限于输入图像的像素分辨率，而是更符合物理空间的表面几何。
*   **结构的一致性与鲁棒性**：通过将 3DGS 锚定在物体表面而非固定视角的像素点上，InfiniSplat 有效减少了视点平移时的杂乱基元（scattered primitives），实现了更稳健的零样本（Zero-shot）泛化能力。

### 3. 对领域的潜在影响
*   **从“图像预测”到“几何重构”的范式转移**：该研究标志着单图 3D 生成从简单的“像素重投影”向“基于理解的场景构建”演进。它证明了引入几何约束（而非单纯依赖 CNN 对特征的预测）是提升 3DGS 泛化能力的关键。
*   **推动高斯泼溅的实用化**：通过前馈（feed-forward）方式省去了昂贵的逐场景优化过程，使其能够实时或近实时地生成可交互的 3D 场景，为单图 3D 重建的工业级应用扫清了障碍。

### 4. 受益的相关领域与应用
*   **自动驾驶与机器人**：需要从稀疏相机输入中实时推断出周围环境的 3D 几何，InfiniSplat 的大视角合成能力非常适合这类动态环境建模。
*   **AR/VR 内容生成**：能够从一张手机拍摄的照片中快速生成高质量的 3D 资产，极大降低了 3D 内容创作的门槛。
*   **沉浸式在线购物与遗产数字化**：为单图文物扫描、商品展示提供了一种高效、高保真的数字化解决方案。

### 5. 可推断的局限性
*   **对深度估计的依赖**：由于采用了“几何引导采样”，该方法的性能高度依赖于单图深度估计（Monocular Depth Estimation）的准确性。若场景中存在深度预测失真，可能会导致高斯基元分布错位。
*   **遮挡区域的补全难题**：尽管文章强调了结构一致性，但对于单张图片中完全不可见的背面（Hidden side），其生成质量仍取决于模型在训练数据集（如 Hypersim）上学到的先验知识，面对极端未见过的物体可能仍会产生不确定性。
*   **计算开销与推理效率**：虽然相比优化过程更高效，但查询条件隐式解码器（implicit decoder）的加入可能会增加推理时的计算复杂度和显存占用，与超轻量化边缘设备部署可能仍存在差距。

**总结：** InfiniSplat 在单图 3DGS 领域迈出了重要一步，通过将几何先验显式地引入到高斯基元的布局中，解决了长久以来“视角移动即破碎”的核心痛点，是当前前馈式 3D 生成方向上极具竞争力的前沿工作。

**Key Findings:**

- We present InfiniSplat, a feed-forward single-image 3DGS framework that moves from a pixel-aligned representation toward a surface-aligned representation.
- InfiniSplat constructs this representation by first using geometry-guided sampling to place 2D supports according to depth-induced local surface structure, and then applying a query-conditioned implicit decoder to predict Gaussian attributes from the image features queried at these supports.By grounding support locations in geometry while decoupling Gaussian prediction from fixed pixel centers, InfiniSplat produces Gaussian layouts that better follow scene surfaces and reduce scattered primitives caused by grid discretization.Across multiple cross-dataset NVS evaluations, InfiniSplat achieves state-of-the-art performance compared with single-image feed-forward baselines, and demonstrates zero-shot generalization from Hypersim indoor synthetic training to complex open-world scenes.Project page: https://zju3dv.github.io/InfiniSplat.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02437v1)
- [arXiv](https://arxiv.org/abs/2608.02437v1)

---

<a id='2608.02385v1'></a>
## [StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking - Learning Beyond the Tracking Distribution for Structured Post-Fall Behavior](https://arxiv.org/abs/2608.02385v1)

**Authors:** Weihao Wu, Ming Huang, Ruofei Liu, Jinglei Nie, Shuxiang Guo, Chunying Li

**Published:** 2026-08-03

**Categories:** cs.RO

**Abstract:**

Humanoid motion trackers perform reliably within learned tracking distributions, but falls can move the robot into low-height, contact-rich states from which an advancing command is temporarily unreachable. Tracking-only policies may chase infeasible references, producing rapid, large-amplitude limb corrections that increase risk to the robot and its surroundings. We present StableMimic, a unified tracker trained beyond the nominal tracking distribution. Perturbed resets around multiple human get-up references expose prone, supine, off-balance, and intermediate ground-contact states, shaping structured recovery that returns the robot to the trackable region. Because tracking and recovery occupy markedly different state--action distributions, StableMimic uses dedicated experts for each regime and a proprioceptive gate that continuously blends their actions. A hidden successor-state objective teaches human-reference-shaped recovery without exposing reference identity or phase to the deployed Actor; deployment requires no get-up reference, recovery command, trajectory retrieval, or external policy switch. On the complete retargeted LAFAN1 dance subset, StableMimic achieves the lowest errors on all four tracking metrics among five methods. Across 100 matched push-to-fall trials per method, it recovers in 100/100 and attains the lowest values on six of seven post-fall motion and load measures, supporting improved interaction safety under this protocol. Real Unitree G1 dance and standing-reference deployments qualitatively demonstrate bounded limb motion, autonomous recovery, and command resumption.

**Analysis:**

以下是对《StableMimic: Smooth Human-Like Recovery for Humanoid Motion Tracking》这篇论文的深入分析：

### 1. 摘要翻译
人形运动追踪器在预设分布内运行可靠，但在发生跌倒后往往进入低高度、接触丰富的状态，导致难以再次执行推进指令。此时，仅具备追踪功能的策略往往会盲目追逐不可行的参考，产生剧烈肢体动作，增加对机器人及环境的风险。本文提出了StableMimic，这是一个经过预设分布外训练的统一追踪器。通过针对多种人体起立参考轨迹进行扰动重置，该方法暴露了俯卧、仰卧、失衡等状态，塑造了结构化的恢复行为，使机器人能自动回归可追踪区域。由于追踪和恢复任务的状态-动作分布差异巨大，StableMimic采用双专家架构（MoE），并通过本体感觉门控机制平滑融合动作。此外，一种隐式后继状态目标（Hidden successor-state objective）能在不向Actor暴露参考信息的情况下，引导机器人执行拟人化恢复。在完整LAFAN1舞蹈数据集测试中，该方法实现了最优追踪指标，并在100次摔倒测试中全部实现成功恢复，且无缝衔接后续指令。

### 2. 方法动机分析
*   **驱动力**：解决人形机器人在复杂动态任务中因摔倒导致的任务中断问题，实现“追踪-恢复-重连”的全自动化闭环。
*   **现有痛点**：传统追踪策略（如BeyondMimic）对分布外状态缺乏应对机制；现有恢复方案（如独立起立控制器）与追踪任务分离，会导致部署复杂、切换不平滑以及动作不符合预期。
*   **研究假设**：通过在训练中引入“perturbed get-up resets”（扰动起立重置）并结合专门的专家混合架构，能够在一个策略内统一学习追踪与恢复，且无需外部干预。

### 3. 方法设计详解
*   **流程总结**：
    1.  **统一输入**：Actor接收实时指令及本体感觉历史（Proprioception）。
    2.  **双专家混合（MoE）**：设计“运动追踪专家”与“恢复专家”。
    3.  **门控机制（Gate）**：仅基于本体感觉，通过Softmax动态输出权重，对两专家动作进行线性加权融合。
    4.  **隐式学习**：训练时利用特权信息（Privileged Critic）和辅助监督路径学习恢复行为，部署时删除特权模块，保持接口简洁。
*   **核心模块**：
    *   **Proprioception-Gated MoE**：通过门控动态调配任务权重。训练时，根据ROLLOUT标签（追踪/恢复/过渡）进行交叉熵监督。
    *   **Hidden Successor-State Objective**：利用Simulator生成的特权状态构造奖励，引导策略逼近恢复轨迹，但不在Actor层暴露这些轨迹的ID或相位。
*   **关键公式意义**：$K_{ct}(s, \bar{s})$ 通过高斯核计算参考追踪误差，配合权重$\lambda_{get}/\lambda_{cmd}$在不同 regime 间平滑切换，避免了硬切换带来的抖动。

### 4. 方法对比分析
*   **本质区别**：StableMimic将恢复任务内嵌于追踪策略，实现了“Always-on”的恢复逻辑，而非“事件触发型”切换策略。
*   **创新贡献**：提出了一种无需特定起立命令或外部状态标记即可实现拟人化恢复的鲁棒策略，极大地简化了真实机器人的部署难度。
*   **适用场景**：适用于需要连续执行任务且环境存在突发接触或摔倒风险的人形机器人，尤其是舞蹈、长时间巡逻等任务。

### 5. 实验分析
*   **验证方法**：在MuJoCo环境下，利用LAFAN1舞蹈动作集，对比了BeyondMimic、KungFuAthlete等方法，并进行了Unitree G1真实机器人部署。
*   **关键结果**：在100次摔倒测试中达到100%恢复成功率，同时在所有运动误差指标（MPBPE, MJAE等）上均优于基线。
*   **优势**：恢复过程平滑、符合人体工学、无需额外的任务触发器。
*   **局限**：对超出“扰动重置”训练分布的复杂地形适应性仍受限于离线模拟器训练的数据质量。

### 6. 实用指南
*   **实现细节**：
    *   **超参数**：重点在于 gate 的损失函数系数（如transition=4.0, consistency=0.01），这直接影响任务切换的平滑度。
    *   **数据预处理**：训练集需要包含大量摔倒后的扰动，确保模型见过足够多的“落地姿态”。
*   **迁移可能**：该MoE架构可迁移至任何需要多任务切换的控制任务中（如抓取-搬运-放置），只需替换相应的专家专家目标函数。

### 7. 总结
*   **核心思想**：通过门控双专家策略，实现任务间的隐式平滑切换与鲁棒恢复。
*   **速记版pipeline**：
    1. **观测处理**：输入当前命令与自身本体感觉。
    2. **权重决策**：门控网络实时评估是否发生跌倒。
    3. **专家融合**：按权重合并追踪与恢复动作。
    4. **平滑输出**：执行低级PD控制以实现平顺动作。

**Key Findings:**

- We present StableMimic, a unified tracker trained beyond the nominal tracking distribution.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02385v1)
- [arXiv](https://arxiv.org/abs/2608.02385v1)

---

<a id='2608.02309v1'></a>
## [CalibBEV: LiDAR-Camera Calibration via BEV Alignment](https://arxiv.org/abs/2608.02309v1)

**Authors:** Filippo D'Addeo, Lorenzo Cipelli, Adriano Cardace, Emanuele Ghelfi, Andrea Zinelli, Massimo Bertozzi

**Published:** 2026-08-03

**Categories:** cs.CV

**Abstract:**

We present CalibBEV, a novel Bird's Eye View (BEV) alignment approach for LiDAR-camera calibration. Our method unifies LiDAR and camera data into a shared 3D spatial representation, enabling accurate and robust cross-modal calibration. CalibBEV extracts sensor-wise BEV features from each modality using domain-specific architectures and estimates the calibration matrix through a two-step alignment process. First, we perform an implicit alignment by regressing a coarse calibration matrix directly from the BEV features. To ease this alignment, we enforce semantic consistency between BEV representations across modalities using a contrastive loss inspired by CLIP, guiding both networks toward a unified feature space. In the second step, we leverage our BEV formulation to explicitly align the features of one modality with the other, refining the initial coarse estimate into a final, more accurate calibration matrix. CalibBEV significantly outperforms prior point-to-pixel matching methods, achieving state-of-the-art calibration accuracy. On the KITTI and nuScenes benchmarks, our method reduces the Relative Rotation Error (RRE) by 51% and 68%, and the Relative Translation Error (RTE) by 80% and 91%, respectively, compared to previous methods.

**Analysis:**

## 1. 摘要翻译
本文提出了 **CalibBEV**，一种用于激光雷达-相机校准的全新鸟瞰图（BEV）对齐方法。该方法将激光雷达和相机数据统一到共享的3D空间表示中，从而实现了准确且稳健的跨模态校准。CalibBEV利用特定领域的架构提取传感器维度的BEV特征，并通过“两步对齐”流程估计校准矩阵。首先，通过从BEV特征回归粗略的校准矩阵实现隐式对齐，并引入受CLIP启发的对比损失强制BEV表示在语义上保持一致；随后，利用BEV公式显式地对齐模态特征，从而将粗校准细化为最终的高精度结果。CalibBEV在KITTI和nuScenes基准测试上大幅优于现有的点对像素匹配方法，相对旋转误差（RRE）分别降低了51%和68%，相对平移误差（RTE）分别降低了80%和91%。

## 2. 方法动机分析
*   **驱动力**：旨在解决激光雷达与相机之间因模态差异大而导致的难以建立共享特征空间的问题，从而实现稳健的传感器外参标定。
*   **现有方法痛点**：传统方法依赖点到像素的匹配，由于不同模态（CNN图像特征 vs. MLP点云特征）缺乏固有的一致性，导致特征对齐困难。此外，许多方法过度依赖RANSAC等迭代算法，计算效率低且难以处理大幅度偏差。
*   **研究假设**：通过将不同模态投影到统一的3D鸟瞰图（BEV）空间，模态差异将被空间几何约束所缓解；CNN强大的全局感受野足以从 mis-aligned 的BEV特征中推断出空间位移参数。

## 3. 方法设计详解
*   **流程总结**：
    1.  **特征提取**：分别通过ResNet-50（图像）和Point Transformer（点云）提取特征。
    2.  **BEV提升**：通过相机内参矩阵将2D图像特征投影至3D BEV空间；将点云直接散射至BEV空间。
    3.  **隐式对齐**：将两种BEV特征沿通道堆叠，利用CLIP loss强化语义一致性，并通过CNN解码器直接回归粗校准矩阵 $T_{coarse}$。
    4.  **显式对齐**：利用 $T_{coarse}$ 对点云BEV特征进行仿射扭曲（Warping），使其在空间上更接近图像BEV特征。
    5.  **精细调整**：将对齐后的特征送入第二个解码器预测最终修正矩阵 $T_{fine}$，通过 $T = T_{fine} \cdot T_{coarse}$ 获得最终结果。
*   **关键公式解释**：Eq. 5中的对比损失 $S$ 作用于BEV特征图，其目的是通过对齐点与像素特征的相似性矩阵，诱导两个独立的骨干网络映射到一个统一的特征空间，降低后续解码器的空间对齐难度。

## 4. 方法对比分析
*   **本质区别**：从传统的“点-像素对应搜索”转向“3D空间特征图对齐”。
*   **创新贡献**：引入了“隐式+显式”的两阶段BEV对齐范式，通过特征扭曲（Feature Warping）而非仅仅是点云坐标变换，提高了对复杂场景的适应能力。
*   **适用场景**：自动驾驶中需要在线、快速进行高精度外参标定的多传感器融合系统。

## 5. 实验分析
*   **关键结果**：在KITTI odometry数据集上，RTE达到0.04m，RRE达到0.61°，性能全面领先ICLM和GraphI2P等SOTA模型。
*   **主要优势**：极高的精度和鲁棒性；不仅适用于单相机，且天然支持多相机全景拼接的跨模态标定。
*   **主要局限**：两阶段训练流程（冻结部分网络）虽然收敛快，但在未来仍需探索端到端联合优化的可能性。

## 6. 实用指南
*   **开源情况**：目前参考了各基线代码库，未提供显式官方GitHub链接，但方法实现逻辑清晰。
*   **实现细节**：
    *   BEV维度建议设置为 $200 \times 200 \times 8$。
    *   CLIP loss系数 $\alpha$ 设置为0.5。
    *   训练时建议先训练隐式模块，再冻结其骨干网络训练显式模块。
*   **迁移可能**：该框架可迁移至任何需要跨模态几何对齐的任务，如多模态SLAM回环检测、或异构机器人位姿对齐。

## 7. 总结
*   **核心思想**：将标定建模为空间BEV特征的对齐问题。
*   **速记版Pipeline**：
    1. 将图像和点云转换成统一的3D空间特征图。
    2. 用相似性损失让两个特征图先“对齐语义”。
    3. 神经网络直接猜出一个粗糙的偏移矩阵。
    4. 把点云特征“扭曲”修正并再次精细化，得到最终矩阵。

**Key Findings:**

- We present CalibBEV, a novel Bird's Eye View (BEV) alignment approach for LiDAR-camera calibration.
- Our method unifies LiDAR and camera data into a shared 3D spatial representation, enabling accurate and robust cross-modal calibration.
- CalibBEV significantly outperforms prior point-to-pixel matching methods, achieving state-of-the-art calibration accuracy.
- On the KITTI and nuScenes benchmarks, our method reduces the Relative Rotation Error (RRE) by 51% and 68%, and the Relative Translation Error (RTE) by 80% and 91%, respectively, compared to previous methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02309v1)
- [arXiv](https://arxiv.org/abs/2608.02309v1)

---

<a id='2608.02284v1'></a>
## [EOVSAM: Efficient Open-Vocabulary Segmentation with SAM 3 in One Pass](https://arxiv.org/abs/2608.02284v1)

**Authors:** Haomin Peng, Yongkang Li, Zhaoxiang Liu, Xiaojie Jin, Shiguo Lian, Yunchao Wei, Xinggang Wang

**Published:** 2026-08-03

**Categories:** cs.CV

**Abstract:**

Open-vocabulary segmentation identifies and segments objects from arbitrary textual descriptions. SAM 3 supports noun-phrase-guided segmentation and achieves competitive open-vocabulary performance through exhaustive vocabulary traversal, yet suffers from prohibitive computational overhead as target categories scale. In this paper, we propose an Efficient Open-Vocabulary segmentation framework with SAM 3 (EOVSAM), which adapts SAM 3 for single-pass prediction. EOVSAM removes prompt conditioning to turn SAM 3 into an efficient mask generator and introduces a new Attentional Aggregation strategy to optimize open-vocabulary classification end-to-end. This formulation avoids the multi-stage pipelines and post-processing heuristics commonly used by existing methods, while mitigating the closed-set collapse that can arise when classification is optimized directly. EOVSAM consistently improves segmentation accuracy over vanilla SAM 3 on all evaluated datasets and accelerates inference by up to 338$\times$. Furthermore, EOVSAM maintains high accuracy at lower resolutions while achieving even more remarkable inference speeds. Experiments on standard semantic and panoptic segmentation benchmarks show that EOVSAM combines competitive or state-of-the-art accuracy with a substantial speed advantage over existing open-vocabulary segmentation models. Code and models are available at https://github.com/hustvl/EOVSAM.

**Analysis:**

### 1. 摘要翻译
开放词汇分割旨在通过任意文本描述来识别和分割物体。SAM 3 虽然支持名词短语引导的分割，并通过遍历词汇表实现了极具竞争力的性能，但随着目标类别规模的扩大，其计算开销极其高昂。在本文中，我们提出了 EOVSAM，一种基于 SAM 3 的高效开放词汇分割框架，它将 SAM 3 适配为单次推理（single-pass）预测模型。EOVSAM 移除了提示词依赖，将 SAM 3 转化为一个高效的掩码生成器，并引入了“注意力聚合”（Attentional Aggregation）策略，实现了端到端的开放词汇分类。该架构避免了现有方法中常见的多阶段流水线和后处理启发式规则，同时缓解了直接优化分类时可能出现的闭集塌陷问题。EOVSAM 在所有评估数据集上均显著提升了分割精度，并将推理速度加快了 338 倍。

### 2. 方法动机分析
*   **驱动力**：旨在解决 SAM 3 在开放词汇任务中因“每个类别需要独立推理一次”而导致的推理速度过慢问题，实现实时、高效的分割。
*   **现有痛点**：现有的 SAM 适配方法要么需要多遍循环推理，要么依赖昂贵的后处理，且难以在不损失精度的情况下保持高效。
*   **研究假设**：通过架构重构，利用 SAM 3 强大的定位先验，结合特征融合与注意力聚合，可以在单次前向传播中同时完成掩码定位与多类别的开放词汇识别。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **特征提取**：利用 C-RADIOv4 主干网络并行提取 SAM 3 特征（$F_{sam}$）和 SigLIP 2 特征（$F_{siglip}$）。
    2.  **增强与生成**：将 $F_{sam}$ 经融合编码器处理得到 $F_{enh}$。检测解码器基于学习到的查询（Queries）输出掩码嵌入 $E_{mask}$ 和对应的注意力图 $A$。
    3.  **语义分割**：通过 $E_{mask}$ 与高分辨率特征图 $F_{seg}$ 的点积生成最终掩码 $M$。
    4.  **注意力聚合**：利用注意力图 $A$ 动态聚合 $F_{siglip}$，生成与物体区域对齐的嵌入 $O$，最后与 SigLIP 文本分类器进行余弦相似度计算。
*   **关键公式**：$o_i = \sum_{u,v} A(i, u, v) F_{siglip}(:, u, v)$。该公式通过 $A$ 动态池化视觉特征，使得物体嵌入 $o_i$ 能够直接与文本嵌入 $t_k$ 进行分类对比，消除了非微分的掩码二值化操作。

### 4. 方法对比分析
*   **本质区别**：从传统的“先分割后识别”范式转变为“基于注意力的单次联合优化”范式。
*   **创新贡献**：引入注意力聚合机制，在保持掩码质量的前提下，完全去除了对提示词（Prompt）的依赖，实现了计算效率的几何级增长。
*   **适用场景**：极度依赖实时性与计算效率的场景（如边缘设备），同时在通用开放词汇任务中表现出优异的泛化能力。

### 5. 实验分析
*   **关键结果**：在 ADE20K 等数据集上实现了 state-of-the-art 的性能，且推理速度比 vanilla SAM 3 加快 338 倍。
*   **主要优势**：极高的推理效率；支持单模型处理不同输入分辨率；泛化能力强。
*   **主要局限**：分类精度高度依赖于预训练主干（C-RADIOv4）和 SigLIP 2 的对齐质量。

### 6. 实用指南
*   **开源情况**：代码已开源至 https://github.com/hustvl/EOVSAM。
*   **实现细节**：训练时冻结了视觉主干和文本编码器。设置了辅助边界框监督以增强定位。超参数 $\lambda$ 的平衡（如 $\lambda_{ce}=3, \lambda_{bce}=5$）对于性能至关重要。
*   **迁移可能**：该框架的“注意力聚合”模块极易迁移到其他基于 Transformer 的掩码生成模型（如 Mask2Former 系列）。

### 7. 总结
*   **核心思想**：利用注意力聚合实现单次推理，将分割与识别深度解耦并端到端统一。
*   **速记版 Pipeline**：
    1. 主干网络提取双流特征。
    2. 解码器生成掩码与注意力图。
    3. 利用注意力图从特征中提取物体信息。
    4. 对齐文本空间进行零样本分类。

**Key Findings:**

- In this paper, we propose an Efficient Open-Vocabulary segmentation framework with SAM 3 (EOVSAM), which adapts SAM 3 for single-pass prediction.
- EOVSAM removes prompt conditioning to turn SAM 3 into an efficient mask generator and introduces a new Attentional Aggregation strategy to optimize open-vocabulary classification end-to-end.
- Experiments on standard semantic and panoptic segmentation benchmarks show that EOVSAM combines competitive or state-of-the-art accuracy with a substantial speed advantage over existing open-vocabulary segmentation models.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02284v1)
- [arXiv](https://arxiv.org/abs/2608.02284v1)

---

<a id='2608.02257v1'></a>
## [Learning Panorama-Aware VLA for Mobile Manipulation with Whole-Body Teleoperation](https://arxiv.org/abs/2608.02257v1)

**Authors:** Donglin Yang, Haoran Chen, Xingyu Chen, Lixing Liu, Manyi Li, Changhe Tu, Ke Xu, Xiaojian Ma, Si Liu

**Published:** 2026-08-03

**Categories:** cs.RO

**Abstract:**

Mobile manipulation is a key capability for embodied intelligence, enabling robots to accomplish complex multi-stage tasks in open-world environments. However, mobile manipulation poses two key challenges for vision-language-action (VLA) policies: At the data level, the efficient collection of high-quality whole-body demonstrations demands the coordinated control of both the mobile base and the robotic arms; at the model level, existing VLA models predominantly rely on local camera observations, whose limited field of view hinders global spatial understanding. To address these challenges, we develop a whole-body teleoperation system and a panoramic-aware VLA policy. The system enables coordinated control of a wheeled bimanual robot through a single VR interface and supports the acquisition of a real-world mobile manipulation dataset comprising 5.5 hours of multimodal demonstrations. Building upon this dataset, we propose PanoVLA, a panorama-aware vision-language-action policy for mobile bimanual manipulation. Built upon a Mixture-of-Transformers architecture, PanoVLA introduces global spatial context through dedicated panorama encoding and fusion modules, enabling effective integration of panoramic observations with language instructions and robot states for action generation. Evaluation on four real-world mobile manipulation tasks demonstrates that PanoVLA achieves an average stage completion rate of 91.3\% and an end-to-end success rate of 73.4\%, substantially outperforming local-view baselines. These results demonstrate that incorporating panoramic spatial context improves spatial understanding and closed-loop manipulation performance in mobile robots.

**Analysis:**

### 1. 摘要翻译
移动操作（Mobile manipulation）是具身智能的关键能力，使机器人能够在开放环境中执行复杂的多阶段任务。然而，移动操作为视觉-语言-动作（VLA）策略带来了两个挑战：在数据层面，采集高质量的全身体验需要移动底盘与机械臂的协同控制；在模型层面，现有的VLA模型主要依赖局部摄像机观测，视野受限阻碍了全局空间理解。为解决这些问题，我们开发了一种全身遥操作系统和一种全景感知VLA策略。该系统通过单一VR界面实现轮式双臂机器人的协同控制，并支持采集包含5.5小时多模态演示的真实世界移动操作数据集。基于此数据集，我们提出了PanoVLA，一种用于移动双臂操作的全景感知视觉-语言-动作策略。PanoVLA基于混合Transformer架构，通过专用的全景编码和融合模块引入全局空间上下文，实现了全景观测与语言指令及机器人状态的有效集成以生成动作。在四个真实移动操作任务上的评估表明，PanoVLA的平均阶段完成率为91.3%，端到端成功率为73.4%，大幅优于局部视角基线。这些结果证明，引入全景空间上下文能显著提升移动机器人的空间理解能力和闭环操作性能。

### 2. 方法动机分析
*   **驱动力**：在长跨度、大空间移动操作任务中，仅依靠局部相机（如腕部相机）容易造成目标丢失，且难以建立全局空间一致性。
*   **现有方法痛点**：现有局部感知VLA缺乏全局感知，导致在需要跨视场、多阶段状态追踪的任务中频繁失败。简单堆叠图像或拼接视角（如Stacked Pano）无法解决畸变带来的语义损耗和空间连贯性问题。
*   **研究假设**：通过全景感知与专用编码专家，将机器人周围的全局空间上下文显式注入动作生成器，能显著提升复杂场景下的任务成功率。

### 3. 方法设计详解
*   **流程总结**：
    1.  **数据采集**：通过VR设备实时捕捉人体运动，通过GMR（广义运动重定向）算法将运动解耦为底盘与双臂的协同动作。
    2.  **全景处理**：将双鱼眼图像重投影为等距柱状投影（ERP）图像，使用MTPano编码器提取包含几何与语义信息的密集特征。
    3.  **多专家融合**：PanoVLA采用混合Transformer（MoT）架构，包含：
        *   **VLM专家**：处理局部语义与指令。
        *   **全景专家**：将全景视觉特征转译为任务相关的空间上下文Token。
        *   **动作专家**：结合前两者的Cache（KV Cache），利用条件流匹配技术生成动作序列。
*   **模型结构**：通过注意力机制，全景专家的Cache与VLM Cache在拼接后送入动作专家。这种设计允许动作生成器同时获取微观（局部操作）和宏观（全局环境）信息。
*   **关键公式**：$C_t = [C_t^{vlm}; C_t^{pano}]$，通过将全景空间特征显式整合进Key-Value Cache，使得后续动作Token在解码时即时具备全局感知。

### 4. 方法对比分析
*   **本质区别**：与简单多视角拼接方法不同，PanoVLA引入了针对全景畸变的专用编码器（MTPano）及跨专家融合的Transformer架构，而非简单的特征串联。
*   **创新贡献**：提出了一种全景增强的MoT架构，实现了感知侧的几何感知表示学习，弥合了全景畸变带来的语义分布偏移。
*   **适用场景**：涉及大 workspace、多阶段交互、需要物体重新定位的长序列任务。

### 5. 实验分析（精简版）
*   **验证方法**：在四个移动操作任务（搬运笔、搬运块、开关窗帘、擦桌子）上进行15次闭环真实机器人实验。
*   **关键结论**：PanoVLA在成功率（73.4%）和阶段完成率（91.3%）上全面领先，特别是在擦桌子这类需要跨视觉阶段追踪的任务中表现最为稳健。
*   **局限**：模型参数规模较大（百亿级量级），且重度依赖特定编码器（MTPano）的预训练表现。

### 6. 实用指南
*   **开源建议**：目前论文提及使用MTPano和Gemma-2B，开发建议关注这些预训练模型的对齐适配。
*   **实现细节**：训练时冻结VLM和全景编码器，仅微调全景专家和动作专家，这在资源受限场景下尤为关键。
*   **迁移可能**：该全景编码范式可直接迁移至任何具有360度传感器的机器人平台，通过更换投影映射层，可适配多种鱼眼布局。

### 7. 总结
*   **核心思想**：通过全景感知专家的空间注入，增强具身模型对全局场景的语义与几何理解。
*   **速记版pipeline**：
    1. 使用VR设备录制协同操作数据。
    2. 将机器人周围环境投影为全景图。
    3. 利用全景专家提取空间上下文特征。
    4. 将全局上下文与局部观察拼接融合。
    5. 通过动作专家生成协同动作序列。

**Key Findings:**

- To address these challenges, we develop a whole-body teleoperation system and a panoramic-aware VLA policy.
- Building upon this dataset, we propose PanoVLA, a panorama-aware vision-language-action policy for mobile bimanual manipulation.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.02257v1)
- [arXiv](https://arxiv.org/abs/2608.02257v1)

---

