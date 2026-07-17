time: 20260717

# Arxiv Computer Vision Papers - 2026-07-17

## Executive Summary

以下是根据2026年7月16日Arxiv计算机视觉领域10篇论文整理的执行摘要，旨在帮助研究人员快速把握最新动态。

---

### 一、主要主题与趋势概述

本期论文覆盖了机器人学习、多模态理解、生成模型、定位与文档分析等方向，整体呈现出以下趋势：

- **机器人策略与基础模型**：多篇工作将大规模预训练或上下文缩放思想引入机器人领域（如`RoboTTT`、`Scaling Behavior Foundation Model for Humanoid Robots`），试图提升策略泛化性与行为基础能力。
- **生成模型向推理与结构化任务渗透**：扩散/流匹配模型被用于多步视觉推理（`Hierarchical Denoising`）、显著目标检测（`Weakly-Supervised RGB-D`）以及世界建模（`DriftWorld`），去噪框架逐渐从图像生成走向决策与理解。
- **多模态融合迈向精细空间绑定**：`SceneBind`和`Beyond Single Expert`分别探索了视觉-音频-语言的“什么与哪里”绑定，以及多视觉先验在MLLM中的动态融合，凸显出对**空间理解**与**跨模态对齐**的持续关注。
- **弱监督与伪标注方法成熟**：利用SAM等基础模型生成伪标签，结合扩散或状态空间模型完成下游任务（如显著性检测），表明大规模视觉基础模型正在成为弱监督学习的新支柱。
- **面向特定场景的推理增强**：地理定位中的地标偏差（`HoloGeo`）和报纸图像层次结构理解（`Towards Hierarchical Structure Understanding`），体现了将领域知识与推理机制结合的趋势。

### 二、特别值得关注的创新论文

- **`RoboTTT: Context Scaling for Robot Policies`**  
  将测试时训练（Test-Time Training）的思想系统性地引入机器人策略，通过动态调整上下文长度来扩展策略对未知场景的适应能力，为可泛化机器人学习提供了新路径。

- **`SceneBind: Binding What and Where Across Vision, Audio and Language`**  
  提出了一个统一框架，同时绑定物体身份（what）、空间位置（where）以及多模态信息（视觉、音频、语言），在场景图构建和多模态问答中表现出显著优势，是多模态认知的里程碑式工作。

- **`DriftWorld: Fast World Modeling through Drifting`**  
  通过“漂移”机制的轻量世界模型更新，在不依赖完整重渲染的条件下快速建模动态环境，有望推动机器人主动感知和在线规划。

- **`Beyond Single Expert: Harmonizing Diverse Visual Priors in MLLMs for Spatial Understanding`**  
  提出在MLLM中动态整合多个视觉专家模型（如深度、分割、边缘检测等）的先验知识，显著提升空间推理能力，代表了MLLM从“单一输入”向“专家协作”演进的重要方向。

### 三、新兴研究方向与技术

- **前向过程强化学习与流生成结合**（`MeanFlowNFT`）：将强化学习引入平均速度生成器的前向流程，可能开辟生成模型与决策优化交叉的新范式。
- **状态空间模型（SSM）用于扩散过程**（`Weakly-Supervised RGB-D`中的“State Space Interaction-based Diffusion”）：利用SSM高效建模长程依赖，替代传统U-Net中的自注意力，有望降低扩散模型计算成本。
- **层次化去噪推理**（`Hierarchical Denoising`）：将推理过程分解为多步去噪，在视觉问答等任务中实现了更好的可解释性和逐步准确性，与思维链（CoT）形成互补。
- **证据驱动的地标偏差缓解**（`HoloGeo`）：通过引入可解释的证据推理机制，而非单纯的数据增强，来对抗地理定位中地标分布不均带来的偏差，为鲁棒定位提供了新思路。

### 四、最值得全文阅读的建议

以下四篇推荐优先阅读，因其对各自领域具有较强启发性或潜在影响力：

1. **`RoboTTT`** – 机器人领域研究者必读，测试时训练在策略缩放上的成功应用可能改变机器人数据收集与部署范式。
2. **`SceneBind`** – 多模态和场景理解方向的核心工作，其“什么-哪里-模态”绑定框架极具通用性。
3. **`DriftWorld`** – 关注在线世界模型与动态环境建模，对机器人、自动驾驶等实时系统有直接参考价值。
4. **`Beyond Single Expert`** – 对MLLM空间理解有深度改进，适合做大模型视觉推理的研究人员，其中专家先验融合机制可迁移至其他多任务场景。

---

希望这份摘要能帮助您快速聚焦关键进展。如需对某篇论文进行深入解析或对比分析，请随时告知。

---

## Table of Contents

1. [RoboTTT: Context Scaling for Robot Policies](#2607.15275v1)
2. [Hierarchical Denoising For Multi-Step Visual Reasoning](#2607.15278v1)
3. [MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators](#2607.15273v1)
4. [SceneBind: Binding What and Where Across Vision, Audio and Language](#2607.15265v1)
5. [HoloGeo: Mitigating Landmark Bias in Geo-localization via Evidence-Driven Reasoning](#2607.15255v1)
6. [Scaling Behavior Foundation Model for Humanoid Robots](#2607.15163v1)
7. [Towards Hierarchical Structure Understanding of Newspaper Images](#2607.15082v1)
8. [DriftWorld: Fast World Modeling through Drifting](#2607.15065v1)
9. [Beyond Single Expert: Harmonizing Diverse Visual Priors in MLLMs for Spatial Understanding](#2607.15054v1)
10. [Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion](#2607.15041v1)

---

## Papers

<a id='2607.15275v1'></a>
## [RoboTTT: Context Scaling for Robot Policies](https://arxiv.org/abs/2607.15275v1)

**Authors:** Yunfan Jiang, Yevgen Chebotar, Ruijie Zheng, Fengyuan Hu, Yunhao Ge, Jimmy Wu, Tianyuan Dai, Scott Reed, Li Fei-Fei, Yuke Zhu, Linxi "Jim" Fan

**Published:** 2026-07-16

**Categories:** cs.RO, cs.AI, cs.LG

**Abstract:**

Recent robot foundation models operate with single-step or short-history visuomotor context. We introduce Test-Time-Training Robot Policies (RoboTTT), a robot model and training recipe that scale visuomotor context to 8K timesteps, three orders of magnitude beyond state-of-the-art policies, without growing inference latency. At this context length, we unlock new robot capabilities: one-shot in-context imitation from human video demonstrations, on-the-fly policy improvement, robustness to perturbations, and stronger performance on multi-stage, long-horizon tasks. We also observe, for the first time, steady gains in closed-loop performance as pretraining context length scales. At its core, RoboTTT integrates Test-Time Training into robot foundation models such as Vision-Language-Action policies, yielding a sequence model whose recurrent state consists of fast weights, parameters updated by gradient descent during both training and inference, compressing histories into weight space and retrieving contextual information for long-context conditioning. To scale training context length, the recipe combines sequence action forcing with truncated backpropagation through time. On challenging real-robot manipulation tasks, RoboTTT improves overall performance by 87% over the single-step context baseline and fully completes a five-minute, ten-stage assembly task, which no baseline ever does. RoboTTT trained with 8K-timestep context outperforms the same model pretrained with 1K timesteps by 62%, suggesting context length as a new scaling axis for robot foundation models. Videos are available at https://research.nvidia.com/labs/gear/robottt/

**Analysis:**

# RoboTTT: 面向长程机器人策略的上下文缩放方法分析

### 1. 摘要翻译
现有的机器人基础模型通常运行在单步或短历史的视觉运动上下文下。我们引入了**测试时间训练机器人策略（RoboTTT）**，这是一种将视觉运动上下文扩展到 8K 时间步长（比现有最优策略多三个数量级）的机器人模型和训练方法，且无需增加推理延迟。在这一上下文长度下，RoboTTT 展现了新能力：从人类视频演示中进行一次性上下文模仿、即时策略改进、鲁棒性提升以及在复杂多阶段长程任务中的优异表现。核心机制在于将测试时间训练（TTT）整合进视觉-语言-动作（VLA）策略中，通过“快速权重”（Fast Weights）将长历史压缩至权重空间。我们还提出了序列动作强制（Sequence Action Forcing）和截断时间反向传播（TBPTT）来解决长序列训练问题。实验表明，上下文长度可作为机器人基础模型的新型缩放轴，RoboTTT 在复杂长程任务中性能提升显著。

### 2. 方法动机分析
*   **驱动力**：旨在克服机器人策略对长历史上下文处理的局限，提升长程、多阶段任务的执行能力。
*   **痛点**：现有方法（如 Transformer）随上下文长度增加，推理成本（KV Cache）线性增长；而基于循环神经网络（RNN）的方法在长程依赖捕捉上能力有限，难以实现复杂的在线适应。
*   **核心直觉**：通过 TTT 机制，将长历史信息动态地压缩进模型的“快速权重”中，实现推理成本与上下文长度的解耦（固定开销），同时利用梯度下降进行在线权重更新，从而具备学习和适应能力。

### 3. 方法设计详解
*   **模型结构**：基于预训练的 VLA 模型（GR00T N1.7）架构。关键创新是在 Diffusion Transformer (DiT) 的注意力层之后插入 **TTT 层**。
*   **核心机制**：
    *   **快速权重更新（Update then Apply）**：利用输入序列（查询-键-值）通过梯度下降实时更新模型内的一层小型 MLP（快速模型）。
    *   **Tanh Gating**：引入学习门控机制 `tanh(α) * TTT_output + Attention_output`，在保留原有预训练能力的同时，平滑地激活 TTT 的长上下文能力。
*   **长序列训练方法**：
    *   **序列动作强制（Sequence Action Forcing）**：对动作块序列中的不同部分独立采样噪声水平，防止训练不稳定性。
    *   **截断时间反向传播（TBPTT）**：将超长序列划分为多个段，段内传播梯度，段间仅传递更新后的快速权重，实现固定显存预算下的长上下文训练。
*   **DAgger Distillation**：将“失败-纠正”的映射过程蒸馏进快速权重，使模型在推理时能根据自身历史表现实现即时策略优化。

### 4. 方法对比分析
*   **本质区别**：区别于通过显式存储（如 KV Cache）或线性注意力进行推理的方法，RoboTTT 将上下文隐式建模为权重矩阵的更新。
*   **创新贡献**：首次证明上下文长度是机器人策略的有效缩放轴，并提出了将 TTT 集成到 VLA 模型中的工程实践，成功突破了 8K 时间步长。
*   **适用场景**：复杂、多阶段、长时间跨度的机器人操控任务（如装配、电路组装等）。

### 5. 实验结论
*   **关键结果**：在 8K 上下文下，RoboTTT 在长程装配任务上较单步 baseline 提升 87%，且能在 10 次尝试中实现 6 次成功的一次性模仿，优于所有 baseline。
*   **主要优势**：推理延迟固定，且随上下文长度增加，性能稳步提升，无饱和迹象。
*   **主要局限**：训练成本较高；目前未能解决所有部署中的失败模式，未来可引入强化学习优化。

### 6. 实用指南
*   **开源/实现**：项目已开源（research.nvidia.com/labs/gear/robottt）。
*   **关键点**：注意 `tanh` 门控初始值需接近零；实现时务必确保快速权重的梯度更新流程在训练和推理的一致性；TBPTT 中必须进行梯度切断（Detach）。
*   **迁移建议**：该架构可作为通用模块替换现有的 DiT/Transformer 动作头，特别适用于需要长时序记忆的控制任务。

### 7. 总结
*   **核心思想**：利用快速权重动态压缩长时序信息，实现高效的长程机器人策略适配。
*   **速记版 Pipeline**：
    1.  预处理机器人轨迹流；
    2.  利用快速模型计算历史数据的梯度更新；
    3.  通过门控机制融合快速权重提取的上下文特征；
    4.  执行去噪动作预测；
    5.  更新状态并滚动推演。

**Key Findings:**

- We introduce Test-Time-Training Robot Policies (RoboTTT), a robot model and training recipe that scale visuomotor context to 8K timesteps, three orders of magnitude beyond state-of-the-art policies, without growing inference latency.
- At this context length, we unlock new robot capabilities: one-shot in-context imitation from human video demonstrations, on-the-fly policy improvement, robustness to perturbations, and stronger performance on multi-stage, long-horizon tasks.
- RoboTTT trained with 8K-timestep context outperforms the same model pretrained with 1K timesteps by 62%, suggesting context length as a new scaling axis for robot foundation models.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15275v1)
- [arXiv](https://arxiv.org/abs/2607.15275v1)

---

<a id='2607.15278v1'></a>
## [Hierarchical Denoising For Multi-Step Visual Reasoning](https://arxiv.org/abs/2607.15278v1)

**Authors:** Zezhong Qian, Xiaowei Chi, Chak-Wing Mak, Tianze Zhou, Ruibin Yuan, Yuhan Rui, Hengzhe Sun, Zhuoqun Wu, Yuming Li, Siyuan Qian, Sirui Han, Shanghang Zhang

**Published:** 2026-07-16

**Categories:** cs.CV

**Abstract:**

Video models are evolving into vision foundation models, yet they still lack human-like multi-step reasoning. Streaming autoregressive diffusion models are efficient but limited in reasoning, while bidirectional diffusion enables global revision with high inference costs due to dense frame-level denoising. Both paradigms struggle to achieve logical consistency and low-latency streaming for complex reasoning tasks. We propose HDR (Hierarchical Denoising for Visual Reasoning), a unified framework that integrates hierarchical latents into causal video generation for multi-step reasoning. HDR organizes video latents into a tree-structured hierarchy, enabling coarse-to-fine reasoning before streaming output. Coarse denoising layers preserve uncertain hypotheses for global planning, while finer layers progressively refine them into concrete visual states. A sparse hierarchical attention pattern (SHAP) further reduces temporal attention costs. We introduce a level-stratified multi-step video reasoning benchmark with out-of-distribution cases, covering six tasks: maze navigation, Tower of Hanoi, one-line drawing, sliding puzzle, Sokoban, and water pouring. Compared with streaming autoregressive diffusion baselines, HDR improves success from 34.22 to 60.29 (76.2% relative gain) and increases average progress from 76.00 to 89.56, demonstrating more consistent reasoning trajectories. HDR maintains low-latency streaming at 0.70 seconds per latent, achieving 54.2 times faster inference than bidirectional diffusion. It also retains 82.9% of full-data performance with only 2% training data, compared with 52.0% for bidirectional diffusion. Real-world robot experiments further demonstrate HDR's potential for physical interaction and world modeling. Project demo: https://hierarchical-diffusion-reasoning.github.io/.

**Analysis:**

## 1. 摘要翻译
视频模型正在演变为视觉基础模型，但仍缺乏类似人类的多步推理能力。现有的流式自回归扩散模型效率高但推理能力不足，而双向扩散模型虽然支持全局修订，但由于在固定序列去噪中需要密集帧计算，导致推理成本高昂。因此，两种范式都难以在复杂推理任务中兼顾逻辑一致性和低延迟流式生成。为弥补这一差距，我们提出了 **HDR（层次化视觉推理去噪）**，这是一个通过将层次化潜空间嵌入因果视频生成过程，实现多步推理的统一框架。HDR 将视频潜空间组织为树状结构，在流式输出前执行从粗到细的推理。粗去噪层维持全局规划的不确定假设，而精细去噪层则将其逐步细化为具体的视觉状态。此外，我们引入了稀疏层次化注意力模式（SHAP）来降低时间注意力成本。我们在包含 maze navigation、Tower of Hanoi 等六项任务的基准测试中证明，HDR 显著提升了多步推理的准确性、数据效率，并维持了低延迟的流式生成特性。

## 2. 方法动机分析
- **核心动机**：在保持“因果性/流式生成”（适合实时交互与机器人控制）的前提下，赋予视频模型“全局规划与修订”的能力。
- **痛点分析**：
    - **AR（自回归）模型**：虽效率高，但受限于从左到右的单向生成，一旦早期决策错误，后续无法回溯或修正，导致逻辑断层。
    - **双向扩散模型**：虽支持全局一致性修正，但由于需要对整个序列进行反复密集更新，推理延迟极高，难以部署于实时流式场景。
- **研究假设**：视频推理的核心在于“中间去噪状态的维持与迭代”。通过将潜空间结构化为树状，并让模型在不同层次保持不同程度的不确定性（粗层保持多方案，精细层细化方案），可以在不牺牲流式效率的前提下实现逻辑一致的推理。

## 3. 方法设计详解
- **层次化潜空间（Tree-Structured Hierarchy）**：将视频表征组织为 $L$ 层树状结构，粗层（Top levels）总结全局时间结构（High-level plans），细层（Bottom levels）编码视觉细节（Fine-grained states）。
- **层级匹配去噪调度（Hierarchy-Matched Schedule）**：摒弃对所有层分配等量去噪步数的做法。HDR根据不同层级的熵，分配不同的去噪预算 $K_\ell$。粗层保留更高的噪声水平（保持不确定性/多种全局假设），精细层分配较少噪声（强化细节还原）。
- **SHAP（稀疏层次化注意力模式）**：这是性能优化的核心。不再执行全序列注意力，而是定义一个固定掩码：每个 Token 只与自身的父级节点、兄弟节点以及首帧条件通信。这种“局部+父级”的注意力机制使得计算复杂度随序列长度线性增长，而非双向模型的平方增长。
- **生成流程**：训练时通过流匹配（Flow Matching）目标优化各层；推理时将树状 Token 扁平化，先生成粗层 Token，再逐步细化下层，所有中间计算结果通过 KV 缓存复用。

## 4. 方法对比分析
- **本质区别**：将“固定序列生成”转化为“树状结构生成”，并引入“层级去噪步数差异化”策略。
- **创新点**：
    - **SHAP 机制**：将全局注意力降维为稀疏树状注意力，极大降低了推理成本。
    - **熵匹配调度**：不仅优化了模型推理性能，还从原理上解决了 AR 模型不可回溯的问题（粗层实质上提供了“规划空间”）。
- **适用场景**：实时机器人操纵、长视频叙事、复杂逻辑推理（如路径规划、任务排序）。

## 5. 实验分析
- **关键结论**：在多步推理基准测试中，HDR 将成功率从 34.22 提升至 60.29（76.2% 相对增益），平均进度显著提升；推理延迟保持在 0.7s/latent，比双向扩散快 54 倍。
- **优势**：数据效率极高（仅 2% 数据即保持 82.9% 成功率），具备强大的迁移能力和对不同难度场景的鲁棒性。
- **局限**：在极端复杂的长序列结尾处，偶有细微状态一致性漂移（如墙壁丢失），需进一步引入约束检查。

## 6. 实用指南
- **开源/复现**：项目主页 `https://hierarchical-diffusion-reasoning.github.io/`。
- **关键细节**：
    - 必须使用熵匹配调度（$\beta=0.66$）而非简单的全 50 步去噪。
    - 训练时需采用分类器自由引导（CFG）以增强条件控制。
- **迁移建议**：该架构可直接用于任何需要长程逻辑规划的序列生成任务（如机器人动作序列建模、长文案生成）。

## 7. 总结
- **核心思想**：利用层次化树状潜空间与稀疏注意力，实现低延迟下的全局推理与局部细化。
- **速记版pipeline**：
    1. 构建视频数据的树状层次表征。
    2. 根据层级分配不同的去噪预算（粗层多假设，细层多细节）。
    3. 利用 SHAP 机制限制注意力计算开销。
    4. 逐级进行扁平化 autoregressive 推理，实现从规划到生成的平滑过渡。

**Key Findings:**

- We propose HDR (Hierarchical Denoising for Visual Reasoning), a unified framework that integrates hierarchical latents into causal video generation for multi-step reasoning.
- We introduce a level-stratified multi-step video reasoning benchmark with out-of-distribution cases, covering six tasks: maze navigation, Tower of Hanoi, one-line drawing, sliding puzzle, Sokoban, and water pouring.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15278v1)
- [arXiv](https://arxiv.org/abs/2607.15278v1)

---

<a id='2607.15273v1'></a>
## [MeanFlowNFT: Bringing Forward-Process RL to Average-Velocity Generators](https://arxiv.org/abs/2607.15273v1)

**Authors:** Yushi Huang, Xiangxin Zhou, Jun Zhang, Liefeng Bo, Tianyu Pang

**Published:** 2026-07-16

**Categories:** cs.CV, cs.LG

**Abstract:**

MeanFlow generators achieve fast few-step sampling by predicting average velocities over time intervals, making them attractive for efficient generation. Reinforcement learning (RL) has become a powerful way to align diffusion and flow models with human preferences and task-specific objectives. In particular, DiffusionNFT offers an efficient forward-process RL framework that does not require reverse-process trajectories or likelihood estimation. However, applying such RL methods to MeanFlow remains underexplored. DiffusionNFT optimizes instantaneous velocities, whereas MeanFlow samples with average velocities. To bridge this gap, we introduce MeanFlowNFT. Inspired by the MeanFlow identity, which bridges average and instantaneous velocities, we construct an induced instantaneous-velocity predictor. We apply the DiffusionNFT objective to this predictor, making reward optimization well-defined for MeanFlow. Sampling remains based on the average velocity, preserving MeanFlow's fast few-step generation. We further prove that MeanFlowNFT inherits DiffusionNFT's strict policy-improvement guarantee. Experiments on image and video generation show that MeanFlowNFT consistently improves baselines. Moreover, it outperforms prior state-of-the-art RL-tuned few-step generators on most metrics ($6$ of $8$ on SD3.5-M), and can even surpass multi-step RL-tuned diffusion while using only a few sampling steps. For instance, on Wan 2.1, $4$-step MeanFlowNFT reaches a VBench score of $84.33$, surpassing $50$-step LongCat-Video RL ($82.57$).

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇论文《MeanFlowNFT》的分析如下：

### 1. 核心贡献总结
该论文提出了 **MeanFlowNFT**，这是首个将“前向过程强化学习”（Forward-Process RL）引入平均速度生成器（MeanFlow）的框架。通过构建一个诱导的瞬时速度预测器，该方法在保持 MeanFlow 高效少步采样特性的同时，实现了对生成模型的有效对齐优化，显著提升了图像与视频生成的质量与人类偏好一致性。

### 2. 关键创新与方法论
*   **弥合理论鸿沟（Bridging the Gap）：** 传统 DiffusionNFT 依赖于瞬时速度优化，而 MeanFlow 依赖于跨时间区间的平均速度。该研究的核心创新在于利用 MeanFlow 的数学恒等式，推导出一个**诱导的瞬时速度预测器（Induced Instantaneous-Velocity Predictor）**。
*   **策略改进保障：** 该方法理论上证明了其继承了 DiffusionNFT 的严格策略改进保障，确保了强化学习优化过程的稳定性。
*   **解耦优化与采样：** 该方法的巧妙之处在于：**利用瞬时速度进行奖励优化（Reward Optimization），但采样时依然沿用原始的平均速度（Average Velocity）**。这种设计既满足了 RL 优化的数学严谨性，又保留了 MeanFlow 本身在推理速度上的绝对优势。

### 3. 对领域的潜在影响
*   **打破“采样步数 vs. 生成质量”的权衡（Trade-off）：** 长期以来，RL 对齐（如 DPO、PPO）通常需要较长的采样链来保证梯度稳定性，这限制了其在少步模型上的应用。MeanFlowNFT 证明了仅需 4 步采样即可超越传统 50 步的 RL 对齐效果，这直接推动了高质量生成式 AI 的“实时化”和“高效化”。
*   **重新定义高效对齐标准：** 该研究在 SD3.5-M 和 Wan 2.1 等前沿模型上的表现，预示着基于平均速度的生成框架（MeanFlow）将成为大模型对齐领域的一个强力竞争者，甚至可能改变未来扩散模型训练和优化的范式。

### 4. 受益的相关领域与应用
*   **视频生成（Video Generation）：** 如论文所述，Wan 2.1 等视频模型的性能提升非常显著，这对于高分辨率、长时序视频合成至关重要。
*   **资源受限场景的实时推理：** 在边缘设备、移动端或者对推理延迟极其敏感的应用中，该技术能让模型在极少步数下达到甚至超越“重型”模型的生成水平。
*   **复杂交互生成：** 需要进行大规模人类偏好对齐（如复杂指令遵循、美学评分优化）的任务，均能通过此框架在不牺牲推理速度的前提下获益。

### 5. 可推断的局限性
*   **模型架构适配性：** 虽然在 SD3.5 和 Wan 2.1 上表现优异，但该方法高度依赖于 MeanFlow 的数学特性，对于非流场（Flow-based）的其他扩散模型（如传统的 ODE 或 SDE 扩散）是否具备同样的高效迁移性尚不明确。
*   **训练稳定性挑战：** 尽管理论上证明了策略改进保障，但基于诱导瞬时速度的优化过程在实际操作中可能对奖励函数（Reward Model）的质量和超参数极其敏感，尤其是在处理高维视频数据时。
*   **奖励偏移（Reward Hacking）：** 在极少步数（如 4 步）下进行强化学习优化，模型是否更容易产生特定的捷径（Shortcut）或奖励欺骗现象，仍需更深入的实证观察。

**专家总结：** 这篇论文的趣味性在于它巧妙地通过数学推导将两个看似互斥的领域（高效的平均速度采样与复杂的强化学习对齐）结合在了一起。它不仅是在指标上的刷分，更是对生成模型训练范式的一次重要修正，极具工业界落地潜力。

**Key Findings:**

- To bridge this gap, we introduce MeanFlowNFT.
- Moreover, it outperforms prior state-of-the-art RL-tuned few-step generators on most metrics ($6$ of $8$ on SD3.5-M), and can even surpass multi-step RL-tuned diffusion while using only a few sampling steps.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15273v1)
- [arXiv](https://arxiv.org/abs/2607.15273v1)

---

<a id='2607.15265v1'></a>
## [SceneBind: Binding What and Where Across Vision, Audio and Language](https://arxiv.org/abs/2607.15265v1)

**Authors:** Mingfei Chen, Zijun Cui, Ruoke Zhang, Hyeonggon Ryu, Eli Shlizerman

**Published:** 2026-07-16

**Categories:** cs.CV, cs.AI, cs.MM, cs.SD

**Abstract:**

We present SceneBind, an omni-modal representation of realistic scenes with joint semantic and 3D spatial understanding across vision, audio and language. Existing omni-modal encoders excel at instance-level semantics (i.e., what is present), but often lack explicit spatial structure (i.e., where it is). SceneBind addresses this gap by representing each scene as a semantic-spatial entity, combining a global semantic embedding with object-centric semantic-spatial slots. This representation explicitly captures object-level semantics, spatial attributes, and uncertainty. We further propose SceneBind Matching, a semantic-spatial matching scheme that integrates global scene similarity with object alignment, supporting cross-modal scene retrieval and object grounding. To train and evaluate SceneBind, we curate a novel real-world binaural audio-visual dataset with structured semantic and spatial annotations, and propose a training protocol for aligning semantic and spatial signals across modalities. SceneBind is compatible with large-scale pretrained semantic encoders, adds lightweight spatial modeling with only a few additional tokens. It achieves state-of-the-art scene and spatial retrieval while enabling strong zero-shot transfer to downstream tasks such as audio-visual localization.

**Analysis:**

作为计算机视觉与机器学习领域的专家，我为您分析这篇《SceneBind》论文。以下是针对该工作的详细评估：

### 1. 核心贡献总结
SceneBind 提出了一种跨视觉、音频和语言的“全模态”（Omni-modal）表示框架，旨在解决现有模型在处理场景时语义丰富但空间结构缺失的痛点。该论文通过引入“语义-空间实体”表示法，成功将全局场景语义与对象级的空间定位（Where）进行了统一建模，实现了跨模态的精准场景检索与对象接地（Grounding）。

### 2. 关键创新与方法论
*   **语义-空间槽位（Semantic-Spatial Slots）机制：** 这是该模型的核心创新。它不再仅仅将场景视为一个扁平的特征向量，而是结合了“全局语义嵌入”与“对象中心化的槽位”。这种设计显式地捕捉了物体类别、空间坐标及对应的不确定性，有效弥补了现有大模型在空间感知上的空白。
*   **SceneBind Matching 匹配方案：** 该方案突破了传统的全局相似度计算，通过分层匹配——先进行全局场景对齐，再深入到物体级的细粒度空间对齐，从而实现了更符合人类认知模式的跨模态交互。
*   **数据集构建与训练范式：** 针对性地构建了带结构化标注的双耳音频-视觉数据集，解决了多模态空间对齐缺乏基准数据的难题。

### 3. 对领域的潜在影响
*   **跨模态语义与空间的深度融合：** 该研究标志着多模态大模型从“看/听见什么”向“理解场景布局”迈进。这对于通向具身智能（Embodied AI）至关重要，因为机器人不仅需要识别物体，更需要精确的 3D 空间定位以进行交互。
*   **轻量化架构的可迁移性：** 由于其架构兼容现有的预训练编码器（如 CLIP 等）且仅增加少量 tokens，这种设计范式具有极高的实用价值，能够低成本地赋予现有视觉模型强大的空间感知能力。

### 4. 相关领域与潜在应用
*   **具身智能/机器人导航：** 机器人能够更好地理解复杂环境中声音（如报警声、人声）的 3D 空间源头，从而进行准确的听觉定位与视觉跟踪。
*   **增强现实（AR）与虚拟现实（VR）：** 在构建交互式场景时，能够更智能地将虚拟对象与现实世界的语义和空间布局锚定。
*   **多模态检索系统：** 允许用户输入诸如“左侧有狗叫声的公园场景”这类复杂的时空查询，并获得精确检索结果。
*   **自动驾驶：** 提升系统对周围物体空间分布及多源传感器（雷达/音视频）的语义融合感知能力。

### 5. 可推断的局限性
*   **对数据标注的高度依赖：** 尽管提出了结构化标注数据集，但如何在不依赖大规模人工标注的情况下扩展到“开放世界（Open-world）”场景仍是一个挑战。
*   **计算复杂度的权衡：** 引入槽位机制（Slots）虽然比全空间建模轻量，但在实时处理超大规模场景或高密度对象时，其推理延迟和内存消耗是否满足工业级需求，仍有待评估。
*   **空间精度的边界：** 对于遮挡严重或极度复杂的动态场景，基于 Slot 的离散化表示可能在空间解析度上仍存在上限，难以达到激光雷达（LiDAR）级别的几何精度。

**专家点评：**
《SceneBind》的趣味性在于它触及了多模态大模型最“尴尬”的盲区——**空间理解的匮乏**。它通过引入槽位（Slots）这种优雅的方式赋予了模型“空间常识”，这比单纯堆叠参数更具学术启发性。对于希望在多模态理解基础上实现更深层次场景推理的研究者来说，这是一篇值得高度关注的前沿工作。

**Key Findings:**

- We present SceneBind, an omni-modal representation of realistic scenes with joint semantic and 3D spatial understanding across vision, audio and language.
- To train and evaluate SceneBind, we curate a novel real-world binaural audio-visual dataset with structured semantic and spatial annotations, and propose a training protocol for aligning semantic and spatial signals across modalities.
- It achieves state-of-the-art scene and spatial retrieval while enabling strong zero-shot transfer to downstream tasks such as audio-visual localization.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15265v1)
- [arXiv](https://arxiv.org/abs/2607.15265v1)

---

<a id='2607.15255v1'></a>
## [HoloGeo: Mitigating Landmark Bias in Geo-localization via Evidence-Driven Reasoning](https://arxiv.org/abs/2607.15255v1)

**Authors:** Pengcheng Zhou, Xuanyu Liu, Yanchen Yin, Bobo Li, Shengqiong Wu, Mong-Li Lee, Wynne Hsu

**Published:** 2026-07-16

**Categories:** cs.CV

**Abstract:**

Recent advances in Vision-Language Models (VLMs) have significantly improved image geo-localization, yet existing models remain susceptible to landmark bias, causing them to overlook geographical cues or form spurious correlations, ultimately resulting in inaccurate localization. To systematically investigate this issue, we first design two quantitative metrics, Bias Intensity (BI) and Bias Harmfulness (BH), to characterize the impact of landmarks exerted on model reasoning, and establish a comprehensive benchmark, LandmarkBias-3K. To mitigate landmark bias, we further propose an evidence-driven reasoning framework, HoloGeo, to improve the reliability of geo-localization. HoloGeo is supported by a high-quality dataset, BF-30k, annotated with structured multi-evidence bias-free reasoning chains. By incorporating multi-dimensional rewards, HoloGeo explicitly encourages balanced attention over diverse visual cues and achieves evidence-driven joint reasoning. Extensive experiments demonstrate that HoloGeo not only maintains excellent performance on IM2GPS3K and YFCC4k but also significantly outperforms existing open-source VLMs on LandmarkBias-3K, validating its effectiveness for robust geospatial reasoning.

**Analysis:**

这是一篇针对视觉地理定位（Image Geo-localization）领域中核心痛点——“地标偏见（Landmark Bias）”的深度研究。以下是对该论文的专业分析：

### 1. 核心贡献摘要
《HoloGeo》通过量化和缓解视觉语言模型（VLM）在地理定位任务中对地标（Landmarks）的过度依赖，解决了模型因产生虚假相关性（Spurious Correlations）而导致的泛化能力差的问题。该研究通过引入系统性的偏见量化指标、构建包含推理链的数据集以及提出证据驱动的推理框架，显著提升了地理定位任务的鲁棒性与可解释性。

### 2. 关键创新与方法论
该论文的创新之处在于将“消除偏见”从单纯的数据层面提升到了“推理过程”层面：
*   **量化指标体系**：提出了 **Bias Intensity (BI)** 和 **Bias Harmfulness (BH)**，首次为地标偏见提供了可度量的数学定义，使得研究者能够精确衡量模型对“地标”的依赖程度。
*   **证据驱动的推理框架 (HoloGeo)**：改变了传统模型“输入图像 -> 输出坐标”的黑盒模式，引入了结构化的多维度推理链，强制模型关注地理环境中的非地标性线索。
*   **多维奖励机制**：通过在训练中引入针对“平衡视觉注意力”的奖励函数，强迫模型不再仅仅依赖视觉显著性（如埃菲尔铁塔等）进行预测，而是将环境纹理、植被、建筑风格等多元地理线索纳入推理。
*   **基准数据集构建**：**LandmarkBias-3K**（专门用于测试偏见）和 **BF-30k**（含有偏见无关的推理链），为该领域的进一步研究提供了坚实的标准化资产。

### 3. 对计算机视觉领域的潜在影响
*   **提升模型可信度**：在自动驾驶、无人机导航等对地理位置精度要求极高的场景中，减少对特定地标的依赖意味着模型在陌生区域的鲁棒性将大幅提升。
*   **推动VLM的“推理”化转型**：该研究展示了通过“证据驱动（Evidence-driven）”引导模型推理的方法论，这为解决多模态模型中常见的“伪推理”问题提供了新的范式，不仅限于地理定位，对医疗影像诊断、安防监控等依赖上下文的任务也具有启发意义。

### 4. 相关的应用领域
*   **智能交通与自动驾驶**：在高精地图缺失或地标被遮挡的情况下，依赖辅助地理环境特征（如地形、地表覆盖）实现精确定位。
*   **文化遗产保护与旅游科技**：能够精准识别未被广泛报道或缺乏显著地标的区域，支持更精细的全球地理信息标注。
*   **地理空间情报（GEOINT）**：在军事或救援场景中，面对被干扰或缺乏已知地标的区域时，基于环境特征的定位能力至关重要。
*   **机器人导航**：帮助无人机等自主设备在环境多变、地标频繁更换或损毁的场景下保持航向可靠性。

### 5. 可推断的局限性
*   **数据标注成本极高**：**BF-30k** 要求高质量的“结构化多证据推理链”标注，这种高昂的构建成本可能限制了该方法在更广阔地理范围内的快速推广。
*   **推理计算开销**：由于引入了显式的推理链和多维度奖励机制，推理阶段的计算复杂度可能高于单纯的端到端（End-to-End）模型，这对资源受限的边缘设备（如无人机内置芯片）可能构成挑战。
*   **偏见定义的通用性**：虽然 BI 和 BH 是有效的度量，但不同地理区域的“偏见”来源不同（如某些地区地标稀少但植被特征明显，另一些地区则反之），模型在极端多变环境下的表现是否依然稳定，仍需进一步验证。

**总结建议**：这篇论文非常符合当前 CV 领域从“追求高准确率”向“追求可解释性和鲁棒性”转型的趋势。对于从事多模态大模型研究的团队来说，其“证据链引导推理”的思路极具参考价值。

**Key Findings:**

- Extensive experiments demonstrate that HoloGeo not only maintains excellent performance on IM2GPS3K and YFCC4k but also significantly outperforms existing open-source VLMs on LandmarkBias-3K, validating its effectiveness for robust geospatial reasoning.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15255v1)
- [arXiv](https://arxiv.org/abs/2607.15255v1)

---

<a id='2607.15163v1'></a>
## [Scaling Behavior Foundation Model for Humanoid Robots](https://arxiv.org/abs/2607.15163v1)

**Authors:** Weishuai Zeng, Kangning Yin, Xiaojie Niu, Shunlin Lu, Weixiang Zhong, Jiahe Chen, Feiyu Jia, Xiao Chen, Zirui Wang, Furui Xu, Ming Zhou, Kailin Li, Weinan Zhang, He Wang, Li Yi, Dahua Lin, Jiangmiao Pang, Jingbo Wang

**Published:** 2026-07-16

**Categories:** cs.RO, cs.AI

**Abstract:**

Humanoid control requires natural whole-body coordination, precise real-time responses to control signals, and robust generalization across diverse environmental contexts, making it a cornerstone for generalist embodied agents. Behavior Foundation Models (BFMs) have recently emerged as a promising solution to address these challenges by leveraging large-scale behavioral data to achieve superior expressiveness, versatility and generalization. However, despite growing interest in scaling BFMs to further improve their capabilities, it remains unclear how key factors, including the learning paradigm, behavioral data and model architecture should be coordinated to enable effective scaling. In this work, we revisit the scaling recipe for BFMs and demonstrate that substantial performance gains can be achieved through the coordination of three core components: 1) the learning paradigm of motion tracking that reformulates diverse humanoid control problems as the reproduction of integrated whole-body behaviors in the global frame; 2) the strategic synergy between on-policy rollout quantity and reference motion diversity; and 3) the expressive and scalable model architecture termed Humanoid Transformer that facilitates the natural emergence of structured behavioral representations. Through extensive experiments in both simulation and real-world deployment, we demonstrate that our approach yields significant improvements in control fidelity and task generalization, reducing Mean Per-Keypoint Position Error (MPKPE) on the test set by over 10% in local mode and 82% in global mode compared with existing humanoid controllers. These results establish BFM as a principled and effective foundation for scalable and general-purpose humanoid control.

**Analysis:**

以下是对这篇论文《Scaling Behavior Foundation Model for Humanoid Robots》的深入分析：

### 1. 摘要翻译
人形机器人控制需要自然的全身协调、实时响应能力及在多样环境中的泛化能力。行为基础模型（BFMs）通过利用大规模行为数据展现出了优越的表现力、通用性和泛化性。然而，如何协同学习范式、行为数据和模型架构来实现有效扩展仍不清晰。本文重新审视了BFM的扩展方案，证明了通过协调三大核心组件可实现显著性能提升：1）将多样化的人形控制问题转化为全局框架下全身行为复现的运动追踪学习范式；2）策略性协同策略（on-policy）数据量与参考动作多样性；3）名为“Humanoid Transformer”的具表达力且可扩展的模型架构。实验表明，该方法显著提升了控制精度与任务泛化能力，在本地模式下Mean Per-Keypoint Position Error (MPKPE) 降低超10%，全局模式下降低超82%，确立了BFM作为可扩展通用人形控制的基础。

### 2. 方法动机分析
- **驱动力**：旨在克服现有专用人形控制器任务单一、奖励工程繁琐的瓶颈，实现更具通用性的“行为基础模型”。
- **痛点**：现有研究虽尝试扩展BFM，但缺乏系统性的“Scaling Recipe”。数据量的增加往往等同于简单的动作数量堆砌，忽视了策略（PPO）对环境交互数据量的依赖；且模型架构多依赖简单的MLP，缺乏对时间序列信息的有效建模。
- **假设**：通过统一的“运动追踪”代理任务，并协调环境交互数据规模（宽度与深度）、动作数据多样性及更高效的Transformer架构，可以系统性地提升BFM的泛化性能与行为多样性。

### 3. 方法设计详解
- **核心范式**：运动追踪（Motion Tracking）。将所有控制任务（如行走、搬运）统一建模为对参考轨迹的 imitation。
- **数据扩展协同**：
    - **宽度/深度（数量）**：通过增加并行环境数量（width）和延长rollout horizon（depth）增加on-policy数据量，提供更稳定的梯度估计。
    - **多样性（数据分布）**：通过采集1.02亿帧大规模、多样化数据集，并通过“自适应采样”机制，闭环聚焦于模型表现较差的困难样本。
- **模型结构（Humanoid Transformer）**：
    - ** temporal window**：输入由历史状态、动作组成的序列，并通过tokenization处理。
    - **Learnable Query Token**：利用该token与上下文序列交互，聚合历史信息进行动作预测。
    - **Cross-Attention**：将目标（Goal）嵌入注入到Transformer骨干网络，实现对不同控制模式的灵活调节。
    - **RMSNorm**：在单位超球面上进行归一化，诱导出一套自然结构化的 latent space（行为空间），无需额外监督。

### 4. 方法对比分析
- **根本区别**：不仅是将运动追踪作为目标，而是将其作为“代理任务”来学习通用的底层表征。通过Masked Goal的设计，使同一模型可直接切换Root模式、全身模式等多种控制接口。
- **创新点**：提出了基于Transformer的端到端扩展架构；揭示了Scaling数据时，策略交互规模与参考数据分布之间的协同效应。

### 5. 实验分析
- **验证方法**：在IsaacLab仿真与Unitree G1实机上进行大规模基准测试。
- **关键结果**：在复杂动作上，相较于原有基线（如SONIC、GMT），模型显著提升了成功率，并实现了极大幅度的跟踪误差降低。
- **主要局限**：对分布式训练的资源要求较高；部分任务存在性能饱和现象，表明Scaling并非在所有控制模式下线性有效。

### 6. 实用指南
- **开源情况**：论文明确表示将开源相关资源（官网：scalebfm.github.io）。
- **关键细节**：
    - **Masking机制**：在目标设置中加入随机链接掩码，是实现单一模型多模式控制的关键。
    - **Reward设计**：需保持全局框架下轨迹的完整性，不要随意剥离根节点位置控制。
- **迁移建议**：若要迁移至新机器人，需重新进行Skeleton Retargeting，确保参考动作与自身 morphology 一致。

### 7. 总结
- **核心思想**：通过全局轨迹追踪的统一代理任务与Transformer架构，实现行为模型的大规模扩展。
- **速记版pipeline**：
    1. 整合大规模多样化动作数据集。
    2. 采用基于Transformer的高维时序行为编码架构。
    3. 协同优化PPO环境交互策略与动作分布。
    4. 通过掩码目标接口实现多任务控制。

**Key Findings:**

- Through extensive experiments in both simulation and real-world deployment, we demonstrate that our approach yields significant improvements in control fidelity and task generalization, reducing Mean Per-Keypoint Position Error (MPKPE) on the test set by over 10% in local mode and 82% in global mode compared with existing humanoid controllers.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15163v1)
- [arXiv](https://arxiv.org/abs/2607.15163v1)

---

<a id='2607.15082v1'></a>
## [Towards Hierarchical Structure Understanding of Newspaper Images](https://arxiv.org/abs/2607.15082v1)

**Authors:** William Mocaër, Solène Tarride, Thomas Constum, Merveilles Agbeti-Messan, Tom Simon, Clément Chatelain, Stéphane Nicolas, Pierrick Tranouez, Sébastien Cretin, Thierry Paquet

**Published:** 2026-07-16

**Categories:** cs.CV, cs.AI, cs.IR

**Abstract:**

Understanding newspaper images remains a challenging task due to their complex, nested hierarchical structures and dense, heterogeneous layouts. In this paper, we explore two complementary approaches for newspaper structure understanding. First, we present a modular bottom-up pipeline that combines state-of-the-art open-source models: YOLO for layout detection, LayoutReader for reading order prediction, and a custom algorithm for article segmentation. This approach leverages existing robust components while maintaining flexibility and interpretability. Second, we introduce Tiramisu (Tiered Transformers for Hierarchical Structure Understanding), a novel end-to-end transformer-based architecture that explicitly models document hierarchy through an iterative tiered process. Tiramisu performs section and article separation, block localization, semantic categorization, and reading order prediction using highly parallelized attention mechanisms. Finally, we release Finlam La Liberté, a new dataset designed specifically for evaluating hierarchical information retrieval in historical newspapers. Experimental results demonstrate the effectiveness of both approaches in reconstructing complex newspaper hierarchies, with comparative analysis highlighting their respective strengths for scalable document digitization. The Tiramisu training code, including the synthetic newspaper generator, is available at https://git.litislab.fr/tiramisu/tiramisu-newspaper-articles-extractor.

**Analysis:**

### 1. 摘要翻译
由于报纸图像具有复杂、嵌套的层级结构和密集、异构的排版，对其进行理解仍是一项具有挑战性的任务。在本文中，我们探索了两种互补的报纸结构理解方法。
首先，我们提出了一个模块化的自下而上（bottom-up）流水线，结合了YOLO（用于布局检测）、LayoutReader（用于阅读顺序预测）以及自定义的算法（用于文章分割），在保持灵活性和可解释性的同时利用了现有鲁棒组件。
其次，我们引入了Tiramisu（一种用于层级结构理解的级联Transformer），这是一种新颖的端到端Transformer架构，通过迭代的分层过程显式建模文档层级。Tiramisu利用高度并行的注意力机制执行版块和文章分割、区域定位、语义分类以及阅读顺序预测。
最后，我们发布了Finlam La Liberté，这是一个专门为评估历史报纸中层级信息检索而设计的新数据集。实验结果证明了两种方法在重建复杂报纸层级方面的有效性，比较分析突出了它们各自在可扩展文档数字化方面的优势。

### 2. 方法动机分析
- **驱动力**：旨在解决历史报纸数字化中因页面版面布局极其复杂、异构（嵌套结构、密集排版）而导致的自动结构分析难题。
- **现有方法痛点**：现有技术要么依赖针对特定任务的孤立模型（缺乏全局视角），要么依赖复杂的、基于OCR的传统流水线（严重依赖手工规则，泛化能力差）。
- **研究假设**：报纸结构具有自然的层级逻辑，可以通过自下而上的组装或自上而下的端到端建模来显式学习和提取。

### 3. 方法设计详解
#### 3.1 自下而上流水线 (Bottom-up Pipeline)
1.  **块检测与分类**：使用YOLO26模型，在调整大小的图像上检测页面单元块并进行分类。
2.  **阅读顺序预测**：将无序的检测块输入到LayoutReader中，结合LSD（Line Segment Detection）获取的物理分隔符信息，预测块的阅读序列。
3.  **文章与版块分割**：采用自定义的后处理算法，遍历预测出的有序区域，当检测到“ARTICLE-TITLE”或“SECTION-TITLE”时执行分割。

#### 3.2 Tiramisu模型 (Top-down Transformer)
- **架构**：采用Swin Transformer作为编码器，提取图像特征；采用Transformer解码器，通过分层级联策略生成结构。
- **多轮解码流程**：
    - **Pass 1**：提取版块（Section），包含第一块和第一个Token。
    - **Pass 2**：在版块内识别文章（Article），关注其首块和首Token。
    - **Pass 3**：提取每篇文章内的所有块。
    - **Pass 4**：提取每个块的具体Token内容。
- **算法解释**：引入`<lvl1>`至`<lvl4>`特殊提示Token引导解码器。通过将前一轮的输出（例如版块）作为下一轮的上下文输入，实现了从高层到细粒度的迭代式结构化提取。

### 4. 方法对比分析
- **本质区别**：流水线法是离散化的，通过组合多个已验证模型来降低复杂性；Tiramisu法是统一的，试图通过一个模型联合学习所有任务。
- **创新贡献**：提出了一种基于层级Prompt（提示）和多轮迭代解码的Transformer架构，能显式建模报纸的树状层级。
- **适用场景**：流水线法适用于高性能、低延迟的部署场景；Tiramisu法适用于追求高集成度、端到端学习能力和更强的层级推理场景。

### 5. 实验分析
- **验证方法**：在Finlam La Liberté数据集上进行实验，并与Arcanum（商业服务）进行对比。
- **关键结果**：流水线法在mAP和推理速度上优于Tiramisu；Tiramisu在层级实体的计数指标（Jaccard Index）上表现更佳。
- **主要优势**：流水线法具有高效率和高灵活性；Tiramisu法具备统一的端到端建模能力，部署更新简单。
- **主要局限**：流水线法依赖多组件，系统复杂度高；Tiramisu法推理较慢，且如果早期层级检测失败会产生级联错误。

### 6. 实用指南
- **开源情况**：数据集（HuggingFace）、评估框架（GitLab）、Tiramisu训练代码及生成器（GitLab）均已开源。
- **实现细节**：Tiramisu采用课程学习（Curriculum Learning），训练时混合真实数据和合成数据，且真实数据占比逐步增加。
- **迁移可能**：该架构可直接迁移至科学文档、表格文档等具有类似层级逻辑的文档分析任务中。

### 7. 总结
- **核心思想**：通过分层Prompt技术，利用迭代式解码结构显式实现文档层级特征的精准重构。
- **速记版pipeline**：1.图像进编码器；2.分四个层级迭代解码；3.通过提示Token引导路径；4.并行处理同级内容；5.输出最终结构。

**Key Findings:**

- Understanding newspaper images remains a challenging task due to their complex, nested hierarchical structures and dense, heterogeneous layouts.
- In this paper, we explore two complementary approaches for newspaper structure understanding.
- First, we present a modular bottom-up pipeline that combines state-of-the-art open-source models: YOLO for layout detection, LayoutReader for reading order prediction, and a custom algorithm for article segmentation.
- Second, we introduce Tiramisu (Tiered Transformers for Hierarchical Structure Understanding), a novel end-to-end transformer-based architecture that explicitly models document hierarchy through an iterative tiered process.
- Finally, we release Finlam La Liberté, a new dataset designed specifically for evaluating hierarchical information retrieval in historical newspapers.
- Experimental results demonstrate the effectiveness of both approaches in reconstructing complex newspaper hierarchies, with comparative analysis highlighting their respective strengths for scalable document digitization.
- The Tiramisu training code, including the synthetic newspaper generator, is available at https://git.litislab.fr/tiramisu/tiramisu-newspaper-articles-extractor.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15082v1)
- [arXiv](https://arxiv.org/abs/2607.15082v1)

---

<a id='2607.15065v1'></a>
## [DriftWorld: Fast World Modeling through Drifting](https://arxiv.org/abs/2607.15065v1)

**Authors:** Susie Lu, Haonan Chen, Weirui Ye, Yilun Du

**Published:** 2026-07-16

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

Predictive world models enable robots to plan by imagining the outcomes of their actions, but their value for control hinges on generating many rollouts quickly. This creates a bottleneck for diffusion-based world models: multistep sampling makes each rollout expensive, limiting large-scale action search at inference time. We introduce DriftWorld, an action-conditioned world model based on drifting generative models. Rather than denoising iteratively at inference, DriftWorld learns an action-conditioned drift during training, allowing it to generate future frames from the current observation and a candidate action sequence in a single forward pass at 30+ fps, which is 17x faster on average than diffusion based baselines. We evaluate DriftWorld on standard vision-based robotic manipulation benchmarks, including Bridge-V2, RT-1, Language Table, Push-T, and Robomimic. By producing rollouts that are both accurate and fast, DriftWorld achieves state-of-the-art decision-making performance with far less inference time than diffusion-based world model baselines. Beyond online control, DriftWorld can also serve as an offline simulator for ranking real-world robot policies, with rollout-based scores correlating with ground truth at up to 0.99. These results show that drifting models are a strong fit for robot world modeling, where fast, high-quality imagination directly supports planning and policy evaluation.

**Analysis:**

### 1. 摘要翻译
预测性世界模型使机器人能够通过想象行为结果进行规划，但其控制价值依赖于快速生成大量滚动（rollout）。基于扩散模型（diffusion-based）的世界模型由于需要多步采样，计算开销巨大，限制了推理时的规模化动作搜索。我们引入了 DriftWorld，一种基于漂移生成模型（drifting generative models）的动作条件世界模型。DriftWorld 不再在推理时进行迭代去噪，而是在训练期间学习一种动作条件漂移场，从而能够在单次前向传递中以 30+ fps 的速度生成未来帧，比基于扩散的基线平均快 17 倍。我们在标准视觉机器人操作基准上评估了 DriftWorld，证明其在保持高生成质量的同时，以极低的推理时间实现了最先进的决策性能。此外，DriftWorld 还可以作为离线模拟器用于对真实机器人策略进行排名。

### 2. 方法动机分析
- **驱动力**：旨在解决实时机器人规划中，现有扩散模型生成视频速度过慢、无法进行大规模动作搜索的痛点。
- **现有方法痛点**：扩散模型依赖迭代去噪，单步生成需要多次前向计算，导致生成一次 rollouts 耗时过长，阻碍了大规模策略探索。
- **研究假设**：通过引入“漂移场（drifting field）”，将生成过程从迭代去噪简化为单次前向移动，可以在保证精度的前提下实现从噪声到数据分布的高效映射。

### 3. 方法设计详解
- **核心 Pipeline**：
  1. **输入处理**：输入历史观察 $o_{t-F:t}$ 和未来动作序列 $a_{t:t+T}$。
  2. **漂移生成**：利用训练好的 U-Net 结构 $f_\theta(\epsilon, c)$，直接从噪声 $\epsilon$ 推理出未来视频帧 $x$。
  3. **动作条件化**：通过 FiLM 对动作进行帧级条件注入，确保特定帧由特定动作驱动。
  4. **漂移场学习**：在训练中定义漂移场 $V_{p,q}(x)$，将生成的负样本推向正样本（ground truth），在达到平衡状态时（即 $q=p$），模型即收敛。
- **关键技术**：
  - **动作加权漂移**：通过混合无动作样本，强制模型学习动作与结果之间的因果关系。
  - **DINOv2/v3 特征空间**：为解决复杂场景背景下的生成模糊问题，引入预训练特征空间计算漂移 loss，增强语义感知。
  - **运动权重（Motion Weighting）**：针对机器人背景静止、目标运动的特点，根据运动量加权损失函数，避免模型倾向于复制静止背景的简单解。

### 4. 方法对比分析
- **本质区别**：从传统的“反向扩散/去噪”转向“单步漂移”，将生成过程由多步迭代坍缩为一次映射。
- **创新贡献**：首次将漂移生成模型应用于动作条件下的视频生成，并针对性地提出了动作加权漂移场和运动敏感损失。
- **适用场景**：高频机器人规划、离线策略评估及需要大规模动作搜索的复杂操作任务。

### 5. 实验分析
- **关键结论**：在 5 个主流机器人基准测试中，DriftWorld 的推理速度比扩散模型快 17 倍，同时保持了相当甚至更优的图像质量（SSIM, PSNR, FVD）。
- **主要优势**：极高的推理效率使得实时动作搜索（如 GPC-RANK）成为可能。
- **主要局限**：模型依赖预训练特征提取器（DINOv3）来保持清晰度；漂移框架在训练时需要生成多个负样本，消耗较多显存。

### 6. 实用指南
- **开源情况**：已开源，代码位于 [GitHub](https://github.com/Susie-Lu/driftworld)。
- **实现细节**：训练需分两阶段（即“自强制” self-forcing），第一阶段基于真值历史，第二阶段引入模型自身预测作为历史，以提高长期 autoregressive 稳定性。
- **迁移建议**：对于新任务，重点在于特征空间的选择（pixel vs. DINO），如果环境包含大量微小物体运动，运动加权项至关重要。

### 7. 总结
- **核心思想**：通过学习单步漂移场，将迭代式视频生成简化为高效的一次性推理。
- **速记版 pipeline**：
  1. 编码历史状态与动作；
  2. 通过 U-Net 一步生成未来帧；
  3. 计算特征空间的运动加权漂移损失；
  4. 反向更新模型以驱动生成向真值靠拢。

**Key Findings:**

- We introduce DriftWorld, an action-conditioned world model based on drifting generative models.
- By producing rollouts that are both accurate and fast, DriftWorld achieves state-of-the-art decision-making performance with far less inference time than diffusion-based world model baselines.
- These results show that drifting models are a strong fit for robot world modeling, where fast, high-quality imagination directly supports planning and policy evaluation.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15065v1)
- [arXiv](https://arxiv.org/abs/2607.15065v1)

---

<a id='2607.15054v1'></a>
## [Beyond Single Expert: Harmonizing Diverse Visual Priors in MLLMs for Spatial Understanding](https://arxiv.org/abs/2607.15054v1)

**Authors:** Xiao Lin, Xiaohu Huang, Kai Han

**Published:** 2026-07-16

**Categories:** cs.CV

**Abstract:**

Multimodal Large Language Models (MLLMs) have demonstrated substantial promise in spatial understanding. Existing works typically incorporate prior knowledge extracted from a pre-trained foundation model to further enhance the spatial awareness of MLLMs. In this paper, we first reveal that when integrating diverse foundation models into MLLMs, different models provide complementary spatial priors that benefit different tasks. Motivated by this, we propose $\textbf{ViPS}$, a novel multi-model prior framework designed to fully unleash the potential of incorporating multiple $\textbf{Vi}$sual $\textbf{P}$riors from diverse models into MLLMs for $\textbf{S}$patial understanding. Specifically, ViPS introduces an Efficient Prior Proxy to generate multiple foundational priors with minimal inference overhead, and a Dynamic Prior Fusion mechanism to achieve harmonious and context-aware prior fusion and injection from the prior proxies. Extensive experiments demonstrate that ViPS successfully harmonizes diverse visual priors, establishing new state-of-the-art performance across multiple complex spatial reasoning and 3D spatial understanding benchmarks. Project page: https://visual-ai.github.io/vips

**Analysis:**

### 1. 摘要翻译
多模态大语言模型（MLLMs）在空间理解方面展现出巨大潜力。现有的研究通常结合预训练基础模型提取的先验知识来增强MLLMs的空间感知能力。本文揭示了在MLLMs中集成多个不同基础模型时，它们能提供互补的空间先验，从而有利于不同的任务。为此，我们提出了ViPS，这是一个多模型先验框架，旨在充分释放将来自不同模型的多个视觉先验融入MLLMs的空间理解潜力。具体而言，ViPS引入了“高效先验代理”（Efficient Prior Proxy）来以极小的推理开销生成多种基础先验，并利用“动态先验融合”（Dynamic Prior Fusion）机制实现来自先验代理的和谐且上下文感知的融合与注入。大量实验表明，ViPS成功协调了多样化的视觉先验，在多个复杂空间推理和3D空间理解基准测试中达到了新的先进水平。

### 2. 方法动机分析
*   **驱动力**：单一视觉先验模型具有局限性，不同基础模型在不同任务上表现出显著的互补性，因此需要一种能同时利用多种专家模型知识的框架。
*   **现有痛点**：
    *   **计算成本高**：直接并行运行多个大型专家模型会造成严重的推理延迟和内存开销。
    *   **模型混淆**：不同模型带来的先验分布差异较大，直接堆叠或简单融合会导致MLLM难以优化且易产生认知偏差。
*   **研究假设**：通过蒸馏方式用轻量级代理模拟多种大型基础模型的先验，并利用任务感知的动态权重进行融合，可以低成本地实现空间能力的全面提升。

### 3. 方法设计详解
*   **流程总结**：
    1.  **统一特征提取**：利用单一基模型（Base Model）处理视觉输入，生成通用基础表示 $F_{base}$。
    2.  **高效先验代理（Efficient Prior Proxy）**：实例化多个轻量级MLP，将 $F_{base}$ 映射到特定空间先验维度，并通过Alignment Loss（$L_{alignment}$）监督其对齐 ground-truth 先验。
    3.  **动态先验融合（Dynamic Prior Fusion）**：基于MLLM最后的问题token生成动态权重 $w$；先将各先验分支通过“零初始化卷积（Zero-init conv）”以平滑引入，再进行加权求和，最后注入MLLM内部层。
*   **核心模块**：
    *   **Zero-init conv**：确保训练初期各先验对MLLM原有空间没有破坏性扰动，实现“渐进式注入”。
    *   **MLP权重生成器**：根据输入查询的语义（即Instruction），动态决定哪些先验模型在该任务中更重要。

### 4. 方法对比分析
*   **本质区别**：传统方法是“单专家”或“并行计算”，ViPS实现了“轻量化知识蒸馏+上下文动态调度”。
*   **创新贡献**：
    1.  通过代理机制（Proxy）彻底解耦了模型数量与推理成本。
    2.  零初始化机制保证了多源异构先验在训练初期的高稳定性。
*   **适用场景**：适用于资源受限但需要综合多维度空间知识（如深度、轨迹、几何）的3D场景理解任务。

### 5. 实验分析
*   **关键结论**：在VSI-Bench和ScanNet-series benchmarks上全面超越现有最强模型（如VLM-3R）。
*   **主要优势**：实现了性能提升的同时，保持了与单专家模型相近的推理延迟。
*   **主要局限**：目前的实验受限于7B参数量级的MLLM和中等规模数据集，尚未在大规模通用多模态任务上进行广泛验证。

### 6. 实用指南
*   **开源情况**：项目主页为 https://visual-ai.github.io/vips。
*   **实现细节**：
    *   **数据对齐**：Alignment Loss 是Proxy准确性的关键，需保证 ground-truth 先验特征维度的统一。
    *   **超参数**：训练过程中优先训练Proxy部分，随后冻结Proxy并进行LLM部分的LoRA微调，这是性能保障。
*   **迁移建议**：该Proxy-Fusion架构具有极强通用性，可直接迁移至需要引入外部视觉/模态先验的任意MLLM框架中。

### 7. 总结
*   **核心思想**：通过轻量化代理蒸馏与动态任务加权，实现多模型先验的低成本协同。
*   **速记版pipeline**：
    1. 提取基础视觉表示。
    2. 轻量代理映射多源先验。
    3. 生成查询相关权重。
    4. 零初始化平滑注入融合。

**Key Findings:**

- Motivated by this, we propose $\textbf{ViPS}$, a novel multi-model prior framework designed to fully unleash the potential of incorporating multiple $\textbf{Vi}$sual $\textbf{P}$riors from diverse models into MLLMs for $\textbf{S}$patial understanding.
- Extensive experiments demonstrate that ViPS successfully harmonizes diverse visual priors, establishing new state-of-the-art performance across multiple complex spatial reasoning and 3D spatial understanding benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15054v1)
- [arXiv](https://arxiv.org/abs/2607.15054v1)

---

<a id='2607.15041v1'></a>
## [Weakly-Supervised RGB-D Salient Object Detection via SAM-driven Pseudo Annotation and State Space Interaction-based Diffusion](https://arxiv.org/abs/2607.15041v1)

**Authors:** Wenqi Si, Gongyang Li, Shixiang Shi, Weisi Lin

**Published:** 2026-07-16

**Categories:** cs.CV

**Abstract:**

Weakly-supervised RGB-D Salient Object Detection (SOD) is explored to reduce the heavy burden of pixel-level annotations. But scribble annotations lack the structure and details of objects, resulting in inaccurate saliency maps. In this paper, we propose a novel scribble-supervised RGB-D SOD method, consisting of a Segment Anything Model (SAM)-driven pseudo annotation generation method (\emph{SAM-PAG}) and a state space interaction-based conditional diffusion model (\emph{$S^2$Diff}). Specifically, SAM-PAG is tailored to address the issue of sparse supervision information. In SAM-PAG, we adopt the advanced SAM to expand sparse scribbles to dense pixel-level pseudo annotations through the dual-branch structure and the consistency of segmentation masks. In $S^2$Diff, we adopt the diffusion model to iteratively refine the noisy saliency maps with the guidance of conditional information, generating accurate saliency maps. Naturally, the core of our $S^2$Diff lies in the acquisition of conditional features and the denoising of saliency maps. For the former, we employ a cross-modal conditional generation module to interweave cross-modal features through frequency integration and implicit-explicit state space interaction, effectively achieving global conditional features. For the latter, we employ a context injection module to mitigate noise interference and to enhance object information with the conditional context. With the close cooperation of SAM-PAG and $S^2$Diff, our method outperforms relevant scribble-supervised methods and achieves competitive performance compared to fully-supervised methods on seven datasets. The code and results of our method are available at https://github.com/Switch457/WeakS2Diff_SOD.

**Analysis:**

### 1. 摘要翻译
本文旨在降低弱监督RGB-D显著目标检测（SOD）中对密集像素级标注的依赖。针对现有涂鸦（scribble）标注缺乏物体结构细节，导致显著图不准确的问题，本文提出了一种基于SAM驱动的伪标注生成方法（SAM-PAG）和一种基于状态空间交互的条件扩散模型（$S^2Diff$）。SAM-PAG利用强大的Segment Anything Model（SAM），通过双分支结构和一致性掩码融合，将稀疏涂鸦扩展为高质量像素级伪标注。$S^2Diff$通过跨模态条件生成模块（CCGM）进行频率集成与状态空间交互，并利用上下文注入模块（CIM）在扩散过程中进行条件去噪，从而生成精确的显著图。在七个数据集上的实验表明，该方法不仅优于现有的弱监督方法，还达到了与全监督方法竞争的水平。

### 2. 方法动机分析
- **驱动力**：旨在以极低的标注成本（涂鸦）实现接近全监督模型的高性能显著性分割。
- **痛点**：现有的弱监督方法（基于涂鸦或模型预测优化）生成的伪标签往往粗糙，且缺乏对跨模态特征（RGB与深度）的有效融合，难以在复杂场景下生成高质量结果。
- **研究假设**：SAM具有强大的通用分割先验，通过巧妙的Prompt工程（多视角变换与提示扩展）可以弥补涂鸦信息的稀疏性；同时，基于状态空间模型（SSM）的扩散去噪能实现更高效的全局上下文特征交互。

### 3. 方法设计详解
- **SAM-PAG（伪标注生成）**：
  - **双分支结构**：一是“图像变换分支”，通过旋转、缩放、翻转增强SAM对输入变换的鲁棒性；二是“提示扩展分支”，利用超像素聚类将稀疏涂鸦扩展至物体潜在区域，通过冲突剔除和一致性优化生成高质量初始掩码。
  - **一致性融合**：计算各分支掩码的置信度与跨模态一致性得分（IoU），执行加权融合与DenseCRF后处理，得到精炼的密集伪标注。
- **$S^2Diff$（条件扩散显著性检测）**：
  - **条件特征生成网络**：使用CCGM融合RGB和深度信息。首先在频率域交换振幅与相位以对齐跨模态特征，接着在空间域通过Intra-SSM（实现模态内交互）和Inter-SSM（实现模态间互补）提取全局条件特征。
  - **去噪网络**：采用CIM模块，通过通道注意力（CA）机制，使条件信息指导噪声特征的恢复，从而实现更精准的显著图重构。

### 4. 方法对比分析
- **本质区别**：不同于以往将SAM仅作为单一辅助工具，本文通过设计多变换、提示扩展及加权一致性策略，将SAM的泛化能力彻底转化为高质量的离线监督标签。此外，首次将Mamba等状态空间模型引入条件扩散模型中，解决了传统Transformer计算复杂度和CNN感受野有限的问题。
- **创新点**：SAM-PAG框架的提示策略与CCGM在频域与空间域的状态空间交互机制是本文的核心创新。

### 5. 实验分析（精简版）
- **关键结论**：在七个数据集上，该方法全面优于现有的弱监督算法，在多个指标上媲美全监督方法。
- **优势**：伪标签质量高；SSM架构提供了优异的全局依赖建模能力。
- **局限**：推理需要迭代10步去噪，计算开销较大；SAM对复杂遮挡或极低对比度下的物体边缘仍存在分割困难。

### 6. 实用指南
- **开源情况**：已开源，代码见 https://github.com/Switch457/WeakS2Diff_SOD。
- **实现细节**：SAM选择ViT-H版本；超像素聚类数（SLIC）设为70；建议将点采样数固定为10；使用SNR-based噪声调度。
- **迁移性**：$S^2Diff$中CCGM和CIM模块可直接迁移至其他RGB-D、RGB-T多模态融合任务。

### 7. 总结
- **核心思想**：利用SAM先验构建高质量伪标签，结合SSM驱动的扩散模型实现精准目标分割。
- **速记版pipeline**：
  1. 多视角变换增强SAM分割能力。
  2. 超像素提示扩展涂鸦覆盖范围。
  3. 一致性加权融合生成精炼伪标签。
  4. 频域与SSM交互生成条件特征。
  5. 引导扩散去噪得到显著图。

**Key Findings:**

- In this paper, we propose a novel scribble-supervised RGB-D SOD method, consisting of a Segment Anything Model (SAM)-driven pseudo annotation generation method (\emph{SAM-PAG}) and a state space interaction-based conditional diffusion model (\emph{$S^2$Diff}).
- With the close cooperation of SAM-PAG and $S^2$Diff, our method outperforms relevant scribble-supervised methods and achieves competitive performance compared to fully-supervised methods on seven datasets.
- The code and results of our method are available at https://github.com/Switch457/WeakS2Diff_SOD.

**Links:**

- [PDF](https://arxiv.org/pdf/2607.15041v1)
- [arXiv](https://arxiv.org/abs/2607.15041v1)

---

