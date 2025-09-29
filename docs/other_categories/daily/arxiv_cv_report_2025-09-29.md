time: 20250929

# Arxiv Computer Vision Papers - 2025-09-29

## Executive Summary

## Arxiv 计算机视觉每日报告执行摘要 (2025-09-26)

**概述与趋势：**

今日 Arxiv 计算机视觉论文呈现出多模态学习、具身智能与机器人控制、以及数据效率与模型优化等多个交叉领域的热点。具身智能和机器人导航是显著的趋势，多篇论文探索了如何利用视觉语言模型 (VLM) 或像素级运动扩散模型实现更智能的机器人行为。此外，数据标注与策展的自动化、以及模型量化与纹理合成等基础研究也持续受到关注。

**重要与创新论文亮点：**

*   **"Pixel Motion Diffusion is What We Need for Robot Control" (E-Ro Nguyen et al.)**：这篇论文提出了一种新颖的像素运动扩散模型，可能为机器人控制提供了一种更直观、更有效的范式，有望简化复杂的机器人任务规划。其创新性在于直接在像素层面进行运动扩散，可能比传统的状态空间方法更具泛化性。
*   **"See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation" (Chih Yao Hu et al.)**：该工作展示了一个无需学习的 VLM 框架，用于通用无人机导航。其“学习-自由”的特性非常引人注目，表明 VLM 在特定任务中可能具备强大的零样本或少样本泛化能力，极大地降低了部署成本和数据依赖。
*   **"LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision" (Debargha Ganguly et al.)**：这篇论文提出了一种用于自动化数据策展的深度研究代理。在数据驱动的计算机视觉领域，自动化数据标注和策展是提高效率和质量的关键瓶颈，该工作有望显著加速研究和开发周期。

**新兴研究方向与技术：**

1.  **具身智能与机器人控制的融合：** 多篇论文（如 "Pixel Motion Diffusion" 和 "See, Point, Fly"）强调了将计算机视觉与机器人控制深度结合，实现更自主、更智能的具身智能体。
2.  **视觉语言模型 (VLM) 的泛化与解释性：** VLM 不仅被用于导航（"See, Point, Fly"），其内部机制和注意力模式也受到了关注（"Where MLLMs Attend and What They Rely On"），这对于理解和改进 VLM 至关重要。
3.  **数据效率与自动化：** "LABELING COPILOT" 强调了自动化数据策展的重要性，而 "CapRL" 则通过强化学习刺激密集图像字幕能力，都旨在提高数据利用效率和模型性能。
4.  **世界模型 (World Model) 的构建：** "WoW: Towards a World omniscient World model Through Embodied Interaction" 旨在通过具身交互构建一个“全知”的世界模型，这是迈向通用人工智能的重要一步。

**建议阅读全文的论文：**

对于忙碌的研究人员，以下论文可能最值得深入阅读：

*   **"Pixel Motion Diffusion is What We Need for Robot Control" (E-Ro Nguyen et al.)**：对于从事机器人控制或具身智能的研究人员，这篇论文可能提供了一种全新的视角和方法。
*   **"See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation" (Chih Yao Hu et al.)**：对于关注 VLM 泛化能力、零样本学习或无人机应用的研究人员，这篇论文具有很高的参考价值。
*   **"LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision" (Debargha Ganguly et al.)**：对于任何处理大规模数据集、寻求提高数据标注和策展效率的研究人员，这篇论文提供了实用的解决方案。
*   **"WoW: Towards a World omniscient World model Through Embodied Interaction" (Xiaowei Chi et al.)**：对于对通用人工智能、世界模型构建和具身学习感兴趣的研究人员，这篇论文描绘了一个宏大的研究方向。

今天的报告揭示了计算机视觉领域在多模态、具身智能和数据效率方面的持续进步，预示着未来智能系统将更加自主、高效和通用。

---

## Table of Contents

1. [Category Discovery: An Open-World Perspective](#2509.22542v1)
2. [Pixel Motion Diffusion is What We Need for Robot Control](#2509.22652v1)
3. [See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation](#2509.22653v1)
4. [CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning](#2509.22647v1)
5. [WoW: Towards a World omniscient World model Through Embodied Interaction](#2509.22642v1)
6. [LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision](#2509.22631v1)
7. [CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach](#2509.22627v1)
8. [Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](#2509.22496v1)
9. [$γ$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition](#2509.22448v1)
10. [NIFTY: a Non-Local Image Flow Matching for Texture Synthesis](#2509.22318v1)

---

## Papers

<a id='2509.22542v1'></a>
## [Category Discovery: An Open-World Perspective](https://arxiv.org/abs/2509.22542v1)

**Authors:** Zhenqi He, Yuanpei Liu, Kai Han

**Published:** 2025-09-26

**Categories:** cs.CV

**Abstract:**

Category discovery (CD) is an emerging open-world learning task, which aims
at automatically categorizing unlabelled data containing instances from unseen
classes, given some labelled data from seen classes. This task has attracted
significant attention over the years and leads to a rich body of literature
trying to address the problem from different perspectives. In this survey, we
provide a comprehensive review of the literature, and offer detailed analysis
and in-depth discussion on different methods. Firstly, we introduce a taxonomy
for the literature by considering two base settings, namely novel category
discovery (NCD) and generalized category discovery (GCD), and several derived
settings that are designed to address the extra challenges in different
real-world application scenarios, including continual category discovery,
skewed data distribution, federated category discovery, etc. Secondly, for each
setting, we offer a detailed analysis of the methods encompassing three
fundamental components, representation learning, label assignment, and
estimation of class number. Thirdly, we benchmark all the methods and distill
key insights showing that large-scale pretrained backbones, hierarchical and
auxiliary cues, and curriculum-style training are all beneficial for category
discovery, while challenges remain in the design of label assignment, the
estimation of class numbers, and scaling to complex multi-object
scenarios.Finally, we discuss the key insights from the literature so far and
point out promising future research directions. We compile a living survey of
the category discovery literature at
\href{https://github.com/Visual-AI/Category-Discovery}{https://github.com/Visual-AI/Category-Discovery}.

**Analysis:**

以下是Zhenqi He, Yuanpei Liu, Kai Han撰写的论文“Category Discovery: An Open-World Perspective”的全面摘要：

**1. 主要问题或研究问题：**
该论文主要关注“类别发现”（Category Discovery, CD）这一新兴的开放世界学习任务。CD旨在自动对包含来自未见类别的未标记数据进行分类，同时利用来自已知类别的少量标记数据。这解决了传统深度学习模型在封闭世界假设下（即所有类别在训练时都已知）的局限性，并使其能够适应现实世界中不断出现新类别的场景。

**2. 关键创新或方法论贡献：**
该综述论文本身的主要贡献在于对CD领域进行了系统性的回顾和分析，而非提出新的CD方法。其关键创新和贡献体现在：

*   **全面的分类法：** 论文首先提出了一个全面的CD分类法，将其分为两种基本设置——新颖类别发现（Novel Category Discovery, NCD）和广义类别发现（Generalized Category Discovery, GCD），以及七种派生设置，包括持续类别发现（Continual Category Discovery, CCD）、领域漂移下的类别发现（CD with Domain Shift）、联邦类别发现（Federated Category Discovery, FCD）等，以应对更复杂的现实世界挑战。
*   **详细的方法分析：** 对每种设置下的方法进行了深入分析，重点关注其三个核心组成部分：表示学习（representation learning）、标签分配（label assignment）和类别数量估计（estimation of class number）。
*   **基准测试和关键洞察：** 论文对现有方法进行了全面的基准测试，并提炼出关键洞察，指出大规模预训练骨干网络、分层和辅助线索以及课程式训练对类别发现的益处。

**3. 主要结果及其意义：**
该论文通过对现有文献的综合分析和基准测试，得出了以下主要结果和重要意义：

*   **预训练骨干网络的重要性：** 大规模自监督预训练视觉骨干网络（如DINOv2）在类别发现任务中表现出显著优势，显著提升了特征质量、鲁棒性和可扩展性。
*   **分层信息和辅助线索的价值：** 分层信息被证明是类别发现的重要归纳偏置，能够有效提升性能。此外，除了原始图像特征之外的辅助信息（如对象级或文本信息）也极大地促进了语义理解和类别发现。
*   **挑战与局限：** 尽管取得了进展，但在标签分配策略设计、类别数量的准确估计以及扩展到复杂多对象场景方面仍存在挑战。例如，NCD方法通常假设未标记数据只包含新颖类别，这在现实世界中并不总是成立。

**4. 论文中提及的局限性：**
该综述指出了CD领域现有研究的几个局限性：

*   **标签分配策略的普适性：** 缺乏普遍最优的标签分配策略，其有效性高度依赖于表示学习范式和数据集特性（如类别粒度、分布偏斜）。
*   **类别数量估计的效率和稳定性：** 现有方法通常将类别数量估计视为后处理步骤，在复杂场景下仍面临效率、可扩展性和稳定性问题。
*   **跨模态信息的利用不足：** 大多数CD研究主要集中在视觉管道，对跨模态线索（特别是文本）的利用相对不足，尽管VLMs（如CLIP）显示出巨大潜力，但其开放世界性能可能源于记忆而非泛化，存在数据泄露风险。
*   **复杂多对象场景的适用性：** 大多数CD研究在单对象数据集上开发和评估，这与现实世界中包含多个交互对象、杂乱背景和遮挡的复杂场景不符。
*   **现实世界设置的整合：** 现有方法很少能同时处理持续学习、跨领域泛化、类别不平衡和领域漂移等现实世界挑战。

**5. 潜在的未来研究方向：**
基于上述分析，论文提出了以下有前景的未来研究方向：

*   **自适应或混合标签分配策略：** 开发能够更好地与学习到的表示和数据集特性对齐的自适应或混合标签分配策略。
*   **训练中类别数量的自适应推断：** 研究能够在训练过程中有效、高效地自适应推断类别数量的框架。
*   **语义先验的显式整合：** 在不过度依赖VLM骨干网络的情况下，显式整合从语言中提取的语义先验（如文本描述或属性名称），以避免记忆或领域偏置的潜在陷阱。
*   **多对象场景的类别发现：** 将CD扩展到复杂多对象场景，解决实例间特征解耦、对象发现（可能根据用户兴趣）以及将实例级证据与新兴类别关联的问题。
*   **统一的现实世界框架：** 开发能够整合持续学习、跨领域泛化、类别不平衡和领域漂移等多种现实世界挑战的统一框架。
*   **扩展到其他模态和任务：** 将类别发现扩展到实例级或像素级发现，以及3D深度估计、视频分析和动作识别等相关任务。

总而言之，这篇综述为类别发现领域提供了一个全面的路线图，不仅总结了现有技术，还明确指出了未来的研究方向，对于推动开放世界学习和人工智能在现实世界中的应用具有重要意义。

**Key Findings:**

- Firstly, we introduce a taxonomy
for the literature by considering two base settings, namely novel category
discovery (NCD) and generalized category discovery (GCD), and several derived
settings that are designed to address the extra challenges in different
real-world application scenarios, including continual category discovery,
skewed data distribution, federated category discovery, etc.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22542v1)
- [arXiv](https://arxiv.org/abs/2509.22542v1)

---

<a id='2509.22652v1'></a>
## [Pixel Motion Diffusion is What We Need for Robot Control](https://arxiv.org/abs/2509.22652v1)

**Authors:** E-Ro Nguyen, Yichi Zhang, Kanchana Ranasinghe, Xiang Li, Michael S. Ryoo

**Published:** 2025-09-26

**Categories:** cs.RO, cs.CV

**Abstract:**

We present DAWN (Diffusion is All We Need for robot control), a unified
diffusion-based framework for language-conditioned robotic manipulation that
bridges high-level motion intent and low-level robot action via structured
pixel motion representation. In DAWN, both the high-level and low-level
controllers are modeled as diffusion processes, yielding a fully trainable,
end-to-end system with interpretable intermediate motion abstractions. DAWN
achieves state-of-the-art results on the challenging CALVIN benchmark,
demonstrating strong multi-task performance, and further validates its
effectiveness on MetaWorld. Despite the substantial domain gap between
simulation and reality and limited real-world data, we demonstrate reliable
real-world transfer with only minimal finetuning, illustrating the practical
viability of diffusion-based motion abstractions for robotic control. Our
results show the effectiveness of combining diffusion modeling with
motion-centric representations as a strong baseline for scalable and robust
robot learning. Project page: https://nero1342.github.io/DAWN/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供E-Ro Nguyen等人撰写的论文“Pixel Motion Diffusion is What We Need for Robot Control”的全面摘要。

---

**论文题目：** Pixel Motion Diffusion is What We Need for Robot Control

**作者：** E-Ro Nguyen, Yichi Zhang, Kanchana Ranasinghe, Xiang Li, Michael S. Ryoo

### 论文摘要

这篇论文提出了一种名为DAWN（Diffusion is All We Need for robot control）的统一扩散模型框架，用于语言条件下的机器人操作。DAWN通过结构化的像素运动表示，将高层运动意图与低层机器人动作连接起来。

**1. 主要问题或研究问题：**
现有的基于像素运动的两阶段机器人控制框架，在处理高层运动生成和低层控制器方面未能充分利用视觉生成模型和扩散策略的最新进展，导致性能存在差距。论文旨在解决如何构建一个可解释、模块化且数据高效的机器人控制系统，该系统能够有效桥接高层语言指令和低层机器人动作，尤其是在复杂的多任务和真实世界场景中。

**2. 关键创新或方法论贡献：**
*   **两阶段扩散模型框架 (DAWN)：** 论文核心创新在于提出了一个两阶段的扩散模型框架。高层控制器（Motion Director）是一个潜在扩散模块，负责从当前视觉观测和语言指令生成所需的密集像素运动表示。低层控制器（Action Expert）是一个扩散Transformer，将生成的像素运动与额外输入（视觉、机器人状态、语言指令）结合，生成最终的机器人动作序列。
*   **结构化像素运动表示：** DAWN使用显式密集像素运动作为高层运动意图和低层机器人动作之间的结构化中间表示。这种表示方式不仅可解释，而且能够有效桥接不同模态的信息。
*   **利用预训练模型：** Motion Director利用预训练的潜在扩散模型进行RGB图像生成，并将其适应于像素运动生成。Action Expert则利用预训练的视觉和语言模型（如ConvNeXt-S DINOv3和T5-small）来增强其表示能力。
*   **端到端可训练系统：** 整个DAWN系统是完全可训练的，同时保持了中间运动抽象的可解释性。

**3. 主要结果及其意义：**
*   **最先进的性能：** DAWN在CALVIN基准测试上取得了最先进的结果，展示了强大的多任务性能。在MetaWorld基准测试上也验证了其有效性，并显著优于现有方法，尤其是在语义相似但任务不同的场景（如“开门”与“关门”）。
*   **数据高效和真实世界迁移：** 尽管在模拟和现实之间存在显著的领域差距，且真实世界数据有限，DAWN通过最少的微调实现了可靠的真实世界迁移。这表明扩散模型结合运动中心表示在可扩展和鲁棒的机器人学习中具有实际可行性。
*   **可解释性和模块化：** 像素运动作为中间表示，使得系统具有良好的可解释性。模块化设计允许两个扩散模型并行训练，并可独立升级，便于未来视觉或控制领域的新进展集成。

**4. 论文中提到的局限性：**
*   尽管DAWN在数据有限的情况下表现出色，但与某些需要大量预训练数据的SOTA方法（如VPP）相比，其预训练规模仍有提升空间。论文提到，在与VPP进行公平比较时，他们采用了VPP的预训练检查点，并对DAWN进行了微调，这暗示了在完全从头开始训练时，数据量可能仍是一个考虑因素。
*   在某些真实世界任务中，尽管DAWN表现优异，但仍有改进空间，例如在某些特定任务（如MetaWorld的“hammer assembly”和“faucet-close”）上，成功率并非100%。

**5. 潜在的未来研究方向：**
*   **更广泛的预训练：** 进一步探索在更大规模的机器人演示数据集上进行预训练，以进一步提升DAWN的泛化能力和性能。
*   **更复杂的运动抽象：** 探索除了像素运动之外，更丰富、更复杂的中间运动抽象，以处理更精细或更抽象的机器人控制任务。
*   **多模态融合：** 进一步优化不同模态（视觉、语言、机器人状态、像素运动）的融合策略，以实现更鲁棒和智能的机器人行为。
*   **自适应扩散步数：** 论文提到增加扩散步数可以提高性能，但达到一定阈值后收益递减。未来可以研究自适应地确定扩散步数，以平衡性能和计算效率。
*   **多臂协调：** 论文在双臂操作中展示了DAWN的有效性，未来可以进一步探索其在更复杂的多臂协调任务中的应用。

---

总而言之，这篇论文通过引入DAWN框架，成功地将扩散模型与结构化像素运动表示相结合，为语言条件下的机器人操作提供了一个强大、可解释且数据高效的解决方案。其在多个基准测试上的优异表现，以及在真实世界中的可靠迁移能力，为机器人学习领域开辟了新的研究方向。

**Key Findings:**

- We present DAWN (Diffusion is All We Need for robot control), a unified
diffusion-based framework for language-conditioned robotic manipulation that
bridges high-level motion intent and low-level robot action via structured
pixel motion representation.
- DAWN
achieves state-of-the-art results on the challenging CALVIN benchmark,
demonstrating strong multi-task performance, and further validates its
effectiveness on MetaWorld.
- Despite the substantial domain gap between
simulation and reality and limited real-world data, we demonstrate reliable
real-world transfer with only minimal finetuning, illustrating the practical
viability of diffusion-based motion abstractions for robotic control.
- Our
results show the effectiveness of combining diffusion modeling with
motion-centric representations as a strong baseline for scalable and robust
robot learning.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22652v1)
- [arXiv](https://arxiv.org/abs/2509.22652v1)

---

<a id='2509.22653v1'></a>
## [See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation](https://arxiv.org/abs/2509.22653v1)

**Authors:** Chih Yao Hu, Yang-Sen Lin, Yuna Lee, Chih-Hai Su, Jie-Ying Lee, Shr-Ruei Tsai, Chin-Yang Lin, Kuan-Wen Chen, Tsung-Wei Ke, Yu-Lun Liu

**Published:** 2025-09-26

**Categories:** cs.RO, cs.AI, cs.CL, cs.CV, cs.LG

**Abstract:**

We present See, Point, Fly (SPF), a training-free aerial vision-and-language
navigation (AVLN) framework built atop vision-language models (VLMs). SPF is
capable of navigating to any goal based on any type of free-form instructions
in any kind of environment. In contrast to existing VLM-based approaches that
treat action prediction as a text generation task, our key insight is to
consider action prediction for AVLN as a 2D spatial grounding task. SPF
harnesses VLMs to decompose vague language instructions into iterative
annotation of 2D waypoints on the input image. Along with the predicted
traveling distance, SPF transforms predicted 2D waypoints into 3D displacement
vectors as action commands for UAVs. Moreover, SPF also adaptively adjusts the
traveling distance to facilitate more efficient navigation. Notably, SPF
performs navigation in a closed-loop control manner, enabling UAVs to follow
dynamic targets in dynamic environments. SPF sets a new state of the art in DRL
simulation benchmark, outperforming the previous best method by an absolute
margin of 63%. In extensive real-world evaluations, SPF outperforms strong
baselines by a large margin. We also conduct comprehensive ablation studies to
highlight the effectiveness of our design choice. Lastly, SPF shows remarkable
generalization to different VLMs. Project page: https://spf-web.pages.dev

**Analysis:**

好的，这是一篇关于Chih Yao Hu等人撰写的论文“See, Point, Fly: A Learning-Free VLM Framework for Universal Unmanned Aerial Navigation”的全面摘要：

**论文摘要：See, Point, Fly: 一种用于通用无人机导航的免训练VLM框架**

**1. 解决的主要问题或研究问题：**
该论文旨在解决无人机（UAV）自主导航中的一个核心挑战：如何在各种复杂和动态环境中，使无人机能够根据任意形式的自然语言指令，无需特定任务训练地导航到任何目标。现有的基于视觉-语言模型（VLM）的方法通常将动作预测视为文本生成任务，这限制了其精度和泛化能力，尤其是在需要精细动作控制和处理未见环境时。

**2. 关键创新或方法论贡献：**
*   **将动作预测重新定义为2D空间定位任务：** 与将动作预测视为文本生成不同，SPF将无人机导航的动作预测视为2D空间定位任务。VLM被用于将模糊的语言指令分解为输入图像上的2D航点迭代标注。
*   **2D航点到3D位移向量的转换：** SPF将预测的2D航点与预测的行进距离结合，将其转换为3D位移向量作为无人机的动作指令，从而实现精确的3D动作控制。
*   **自适应行进距离调整：** SPF能够自适应地调整行进距离，以实现更高效的导航，在开放区域采取更大步长，在障碍物附近则更谨慎。
*   **闭环控制与动态目标跟踪：** SPF以闭环控制方式执行导航，使无人机能够在动态环境中跟踪移动目标。
*   **免训练框架：** SPF是一个完全免训练的框架，直接利用预训练的VLM进行高层空间推理，无需额外的神经网络训练、技能库、外部深度传感器或策略优化。

**3. 主要结果及其重要性：**
*   **DRL模拟基准的新SOTA：** 在DRL模拟基准测试中，SPF的性能超越了之前最佳方法，成功率绝对提高了63%。
*   **真实世界评估的卓越表现：** 在广泛的真实世界评估中，SPF也以显著优势超越了强大的基线方法。
*   **出色的泛化能力：** SPF对不同的VLM（如Gemini 2.5 Pro, GPT-4.1, Claude 3.7 Sonnet, Llama 4 Maverick等）表现出显著的泛化能力，证明了其设计的有效性。
*   **效率提升：** 自适应步长控制器显著缩短了任务完成时间，同时保持或提高了成功率。

**4. 论文中提到的局限性：**
*   **VLM不准确性：** VLM可能存在幻觉和误解，对小型或远距离目标的定位精度可能下降。
*   **自适应步长启发式方法的局限性：** 自适应步长启发式方法提供了隐式深度，但可能不精确。
*   **对提示词措辞的敏感性：** 性能可能对提示词的措辞敏感。
*   **对动态障碍物的反应性：** 由于VLM推理延迟（约1-3秒），对高度动态障碍物的反应性有限。
*   **搜索模式的非最优性：** VLM生成的搜索模式不保证是最优的。

**5. 潜在的未来研究方向：**
*   提高感知鲁棒性。
*   改进定位机制。
*   降低系统延迟以提高反应性。
*   探索VLM的微调。
*   开发更复杂的探索策略。

总而言之，SPF提出了一种新颖的、免训练的VLM框架，通过将动作预测重新定义为2D空间定位任务，并结合自适应距离调整和闭环控制，显著提升了无人机在复杂、动态环境中的导航能力。其在模拟和真实世界中的卓越性能以及对不同VLM的泛化能力，使其成为通用无人机导航领域的一个重要进展。

**Key Findings:**

- We present See, Point, Fly (SPF), a training-free aerial vision-and-language
navigation (AVLN) framework built atop vision-language models (VLMs).
- SPF sets a new state of the art in DRL
simulation benchmark, outperforming the previous best method by an absolute
margin of 63%.
- In extensive real-world evaluations, SPF outperforms strong
baselines by a large margin.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22653v1)
- [arXiv](https://arxiv.org/abs/2509.22653v1)

---

<a id='2509.22647v1'></a>
## [CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning](https://arxiv.org/abs/2509.22647v1)

**Authors:** Long Xing, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jianze Liang, Qidong Huang, Jiaqi Wang, Feng Wu, Dahua Lin

**Published:** 2025-09-26

**Categories:** cs.CV, cs.AI, cs.CL

**Abstract:**

Image captioning is a fundamental task that bridges the visual and linguistic
domains, playing a critical role in pre-training Large Vision-Language Models
(LVLMs). Current state-of-the-art captioning models are typically trained with
Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable
data annotated by humans or proprietary models. This approach often leads to
models that memorize specific ground-truth answers, limiting their generality
and ability to generate diverse, creative descriptions. To overcome the
limitation of SFT, we propose applying the Reinforcement Learning with
Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning.
A primary challenge, however, is designing an objective reward function for the
inherently subjective nature of what constitutes a "good" caption. We introduce
Captioning Reinforcement Learning (CapRL), a novel training framework that
redefines caption quality through its utility: a high-quality caption should
enable a non-visual language model to accurately answer questions about the
corresponding image. CapRL employs a decoupled two-stage pipeline where an LVLM
generates a caption, and the objective reward is derived from the accuracy of a
separate, vision-free LLM answering Multiple-Choice Questions based solely on
that caption. As the first study to apply RLVR to the subjective image
captioning task, we demonstrate that CapRL significantly enhances multiple
settings. Pretraining on the CapRL-5M caption dataset annotated by CapRL-3B
results in substantial gains across 12 benchmarks. Moreover, within the Prism
Framework for caption quality evaluation, CapRL achieves performance comparable
to Qwen2.5-VL-72B, while exceeding the baseline by an average margin of 8.4%.
Code is available here: https://github.com/InternLM/CapRL.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Long Xing等人撰写的论文“CapRL: Stimulating Dense Image Caption Capabilities via Reinforcement Learning”的全面摘要。

---

**论文摘要：CapRL：通过强化学习激发密集图像描述能力**

**1. 主要问题或研究问题：**
图像描述是连接视觉和语言领域的关键任务，在大型视觉-语言模型（LVLMs）的预训练中扮演重要角色。然而，当前最先进的图像描述模型通常采用监督微调（SFT）进行训练，这依赖于昂贵且难以扩展的人工标注数据。SFT方法往往导致模型记忆特定的真实答案，限制了其泛化能力和生成多样化、创造性描述的能力。核心挑战在于，如何为图像描述这一开放式、本质上主观的任务设计一个客观的奖励函数。

**2. 关键创新或方法论贡献：**
为了克服SFT的局限性，论文提出了**CapRL（Captioning Reinforcement Learning）**，一个新颖的训练框架，将“好的”描述质量重新定义为其“效用”：一个高质量的描述应该能够让一个非视觉语言模型（LLM）准确回答关于对应图像的问题。CapRL引入了一个解耦的两阶段流水线：
*   **第一阶段：** LVLM生成图像描述。
*   **第二阶段：** 一个独立的、无视觉的LLM仅基于生成的描述来回答多项选择题（MCQs），其回答的准确性作为LVLM的客观奖励信号，用于强化学习训练。
*   **奖励函数设计：** 通过将描述质量与LLM回答MCQs的准确性挂钩，CapRL将主观的描述评估转化为客观、可验证的奖励信号，有效避免了传统奖励模型中常见的冗余或简洁偏好等奖励欺骗问题。
*   **QA数据策展：** 论文还开发了一个特定的QA策展流水线，以确保MCQs数据的质量，并保证问题只能通过分析图像内容本身来回答，避免信息泄露。

**3. 主要结果及其意义：**
*   **性能显著提升：** 作为首次将可验证奖励强化学习（RLVR）应用于主观图像描述任务的研究，CapRL在多个设置下显著提升了性能。
*   **大规模数据集预训练：** 在由CapRL-3B标注的CapRL-5M描述数据集上进行预训练，模型在12个基准测试中取得了显著提升。
*   **与先进模型媲美：** 在Prism框架下进行描述质量评估时，CapRL实现了与Qwen2.5-VL-72B模型相当的性能，并平均超越基线8.4%。
*   **泛化能力和准确性：** 结果验证了CapRL能够有效训练模型生成更通用、更准确的图像描述，超越了传统SFT模型在多样性和创造性方面的局限性。
*   **数据效率：** CapRL通过RLVR实现了卓越的数据效率，即使是稀疏的QA监督也足以带来显著的描述能力提升。

**4. 论文中提及的局限性：**
论文主要关注了SFT的局限性，即模型倾向于记忆特定答案，导致泛化能力和生成多样化描述的能力受限。此外，早期基于LVLM作为评判者/奖励模型的尝试存在奖励欺骗（如偏好冗余或简洁描述）和训练曲线不稳定的问题。虽然CapRL通过其解耦的VQA奖励机制解决了这些问题，但论文并未明确指出CapRL自身在特定场景下的局限性，例如在极端复杂或高度抽象的图像描述任务中可能面临的挑战。

**5. 潜在的未来研究方向：**
论文强调了CapRL框架在多模态预训练中的实用价值，因为它能够以非常低的标注成本构建高质量、可扩展的数据集。这暗示了未来研究可以进一步探索：
*   **更大规模的应用：** 将CapRL应用于更大规模的图像-文本对数据集，以进一步提升LVLM的性能。
*   **更复杂的奖励设计：** 探索更精细的奖励函数设计，以应对图像描述中更细微的主观性挑战。
*   **多模态智能的融合：** 结合CapRL的优势，推动从静态感知到交互式、端到端多模态智能的发展，包括长上下文多模态、智能体行为以及统一的预训练目标。
*   **高效适应性：** 研究轻量级微调和检索等高效适应性方法，以安全地将CapRL系统部署到更广泛的领域和设备中。

---

**Key Findings:**

- Current state-of-the-art captioning models are typically trained with
Supervised Fine-Tuning (SFT), a paradigm that relies on expensive, non-scalable
data annotated by humans or proprietary models.
- To overcome the
limitation of SFT, we propose applying the Reinforcement Learning with
Verifiable Rewards (RLVR) paradigm to the open-ended task of image captioning.
- We introduce
Captioning Reinforcement Learning (CapRL), a novel training framework that
redefines caption quality through its utility: a high-quality caption should
enable a non-visual language model to accurately answer questions about the
corresponding image.
- As the first study to apply RLVR to the subjective image
captioning task, we demonstrate that CapRL significantly enhances multiple
settings.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22647v1)
- [arXiv](https://arxiv.org/abs/2509.22647v1)

---

<a id='2509.22642v1'></a>
## [WoW: Towards a World omniscient World model Through Embodied Interaction](https://arxiv.org/abs/2509.22642v1)

**Authors:** Xiaowei Chi, Peidong Jia, Chun-Kai Fan, Xiaozhu Ju, Weishi Mi, Kevin Zhang, Zhiyuan Qin, Wanxin Tian, Kuangzhi Ge, Hao Li, Zezhong Qian, Anthony Chen, Qiang Zhou, Yueru Jia, Jiaming Liu, Yong Dai, Qingpo Wuwu, Chengyu Bai, Yu-Kai Wang, Ying Li, Lizhang Chen, Yong Bao, Zhiyuan Jiang, Jiacheng Zhu, Kai Tang, Ruichuan An, Yulin Luo, Qiuxuan Feng, Siyuan Zhou, Chi-min Chan, Chengkai Hou, Wei Xue, Sirui Han, Yike Guo, Shanghang Zhang, Jian Tang

**Published:** 2025-09-26

**Categories:** cs.RO, cs.CV, cs.MM

**Abstract:**

Humans develop an understanding of intuitive physics through active
interaction with the world. This approach is in stark contrast to current video
models, such as Sora, which rely on passive observation and therefore struggle
with grasping physical causality. This observation leads to our central
hypothesis: authentic physical intuition of the world model must be grounded in
extensive, causally rich interactions with the real world. To test this
hypothesis, we present WoW, a 14-billion-parameter generative world model
trained on 2 million robot interaction trajectories. Our findings reveal that
the model's understanding of physics is a probabilistic distribution of
plausible outcomes, leading to stochastic instabilities and physical
hallucinations. Furthermore, we demonstrate that this emergent capability can
be actively constrained toward physical realism by SOPHIA, where
vision-language model agents evaluate the DiT-generated output and guide its
refinement by iteratively evolving the language instructions. In addition, a
co-trained Inverse Dynamics Model translates these refined plans into
executable robotic actions, thus closing the imagination-to-action loop. We
establish WoWBench, a new benchmark focused on physical consistency and causal
reasoning in video, where WoW achieves state-of-the-art performance in both
human and autonomous evaluation, demonstrating strong ability in physical
causality, collision dynamics, and object permanence. Our work provides
systematic evidence that large-scale, real-world interaction is a cornerstone
for developing physical intuition in AI. Models, data, and benchmarks will be
open-sourced.

**Analysis:**

好的，这是一篇关于Xiaowei Chi等人撰写的论文“WoW: Towards a World-Omniscient World-Model Through Embodied Interaction”的摘要，重点突出其在计算机视觉和机器学习领域的贡献：

**论文摘要：WoW：通过具身交互迈向世界全知世界模型**

**1. 主要问题或研究问题：**
当前视频生成模型（如Sora）主要依赖被动观察，难以真正理解物理因果关系。这导致它们在处理需要真实物理推理的场景时表现出脆弱性和物理幻觉。论文的核心研究问题是：如何通过大规模、因果丰富的真实世界交互，使具身世界模型发展出真实的物理直觉，从而克服现有模型的局限性，实现对物理世界的深刻理解和预测？

**2. 关键创新或方法论贡献：**
*   **具身世界模型WoW：** 提出并实例化了一个140亿参数的生成式世界模型WoW，该模型通过大规模（200万条）机器人交互轨迹进行训练，旨在直接合成像素级的未来预测，并通过生成本身进行想象和推理。
*   **SOPHIA框架：** 引入了一种新颖的架构范式SOPHIA，将视觉语言模型（VLM）的推理能力与扩散Transformer（DiT）的生成能力相结合。SOPHIA通过“预测-批评-精炼”的迭代循环，将想象和推理统一为具身智能的基本组成部分，主动将模型的生成能力约束到物理现实。
*   **Flow-Mask逆动力学模型（FM-IDM）：** 设计了一个视频到动作的模型，将预测的视频帧转化为可执行的机器人动作，从而闭合了“想象到行动”的循环。FM-IDM通过分析当前状态和想象的下一状态之间的光流和场景上下文，推断出执行转换所需的7自由度末端执行器动作。
*   **WoWBench基准：** 建立了一个新的基准，专注于视频中的物理一致性和因果推理，包含4个核心能力和20个子任务，用于全面评估具身世界模型。

**3. 主要结果及其意义：**
*   **物理直觉的提升：** WoW模型在物理理解方面表现出显著能力，尤其是在物理因果关系、碰撞动力学和物体永恒性方面，在WoWBench基准测试中取得了最先进的性能（人类和自主评估均表现优异），指令理解准确率达到96.53%，物理定律遵循准确率达到80.16%。这有力地支持了论文的核心假设，即大规模真实世界交互是AI发展物理直觉的基石。
*   **克服物理幻觉：** SOPHIA框架通过VLM代理对DiT生成的输出进行评估，并通过迭代演化语言指令来指导其精炼，有效解决了模型理解物理的概率性分布导致的随机不稳定性和物理幻觉问题。
*   **想象到行动的闭环：** 逆动力学模型将精炼后的计划转化为可执行的机器人动作，成功实现了从想象到物理行动的闭环，使模型能够指导物理机器人执行任务。
*   **泛化能力：** WoW展示了对新颖机器人具身、操作任务和视觉领域的强大泛化能力，证明其学习的是交互的底层物理原理，而非仅仅是训练数据的上下文。

**4. 论文中提及的局限性：**
*   尽管WoW在物理推理方面取得了显著进展，但模型对复杂、困难的物理推理任务的掌握仍然具有挑战性，需要进一步的扩展（scaling）。
*   模型性能的提升随着参数数量的增加而显著减速，在性能和效率之间存在关键的权衡。
*   当前视频世界模型在3D一致性、物理连贯性和时间推理方面仍存在局限性，这些是实现忠实环境模拟所必需的。

**5. 潜在的未来研究方向：**
*   继续扩展模型规模和训练数据量，以进一步提升模型在复杂物理推理任务上的性能。
*   探索更有效的架构设计和训练策略，以优化性能与推理效率之间的平衡。
*   进一步研究如何增强世界模型的3D一致性、物理连贯性和时间推理能力，以实现更忠实的环境模拟。
*   将WoW作为认知沙盒，进一步探索其在VLM任务规划中的应用，通过模拟反馈帮助VLM调试逻辑谬误，提升长程规划能力。
*   将模型、数据和基准开源，为具身世界模型领域的未来研究提供基础。

**Key Findings:**

- To test this
hypothesis, we present WoW, a 14-billion-parameter generative world model
trained on 2 million robot interaction trajectories.
- Furthermore, we demonstrate that this emergent capability can
be actively constrained toward physical realism by SOPHIA, where
vision-language model agents evaluate the DiT-generated output and guide its
refinement by iteratively evolving the language instructions.
- We
establish WoWBench, a new benchmark focused on physical consistency and causal
reasoning in video, where WoW achieves state-of-the-art performance in both
human and autonomous evaluation, demonstrating strong ability in physical
causality, collision dynamics, and object permanence.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22642v1)
- [arXiv](https://arxiv.org/abs/2509.22642v1)

---

<a id='2509.22631v1'></a>
## [LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision](https://arxiv.org/abs/2509.22631v1)

**Authors:** Debargha Ganguly, Sumit Kumar, Ishwar Balappanawar, Weicong Chen, Shashank Kambhatla, Srinivasan Iyengar, Shivkumar Kalyanaraman, Ponnurangam Kumaraguru, Vipin Chaudhary

**Published:** 2025-09-26

**Categories:** cs.CV, cs.CL

**Abstract:**

Curating high-quality, domain-specific datasets is a major bottleneck for
deploying robust vision systems, requiring complex trade-offs between data
quality, diversity, and cost when researching vast, unlabeled data lakes. We
introduce Labeling Copilot, the first data curation deep research agent for
computer vision. A central orchestrator agent, powered by a large multimodal
language model, uses multi-step reasoning to execute specialized tools across
three core capabilities: (1) Calibrated Discovery sources relevant,
in-distribution data from large repositories; (2) Controllable Synthesis
generates novel data for rare scenarios with robust filtering; and (3)
Consensus Annotation produces accurate labels by orchestrating multiple
foundation models via a novel consensus mechanism incorporating non-maximum
suppression and voting. Our large-scale validation proves the effectiveness of
Labeling Copilot's components. The Consensus Annotation module excels at object
discovery: on the dense COCO dataset, it averages 14.2 candidate proposals per
image-nearly double the 7.4 ground-truth objects-achieving a final annotation
mAP of 37.1%. On the web-scale Open Images dataset, it navigated extreme class
imbalance to discover 903 new bounding box categories, expanding its capability
to over 1500 total. Concurrently, our Calibrated Discovery tool, tested at a
10-million sample scale, features an active learning strategy that is up to 40x
more computationally efficient than alternatives with equivalent sample
efficiency. These experiments validate that an agentic workflow with optimized,
scalable tools provides a robust foundation for curating industrial-scale
datasets.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Debargha Ganguly等人撰写的论文“LABELING COPILOT: A Deep Research Agent for Automated Data Curation in Computer Vision”的全面摘要。

---

### 论文摘要：LABELING COPILOT: 计算机视觉自动化数据管理深度研究代理

**1. 主要问题或研究问题：**
该论文旨在解决计算机视觉领域中一个关键的瓶颈：高质量、领域特定数据集的策展。在处理庞大、未标记的数据湖时，数据质量、多样性和成本之间需要进行复杂的权衡，这严重阻碍了鲁棒视觉系统的部署和可扩展性。传统的管道式方法无法有效应对这种动态且需要智能决策的挑战。

**2. 关键创新或方法论贡献：**
Labeling Copilot引入了首个用于计算机视觉数据策展的深度研究代理，其核心创新在于：
*   **Agentic框架：** 论文提出并实现了一个新颖的Agentic系统，将数据检索、合成和标注统一到一个连贯的、目标驱动的工作流中。一个由大型多模态语言模型驱动的中央协调代理，通过多步推理来执行专业工具。
*   **校准发现工具（Calibrated Discovery）：** 该工具结合了主动学习策略和分布外（OOD）检测，能够从大型数据存储库中高效地发现相关、分布内的数据。它通过将经典主动学习算法重构为基于FAISS的近似最近邻框架，实现了在千万级样本规模下的可扩展性。
*   **可控合成工具（Controllable Synthesis）：** 代理利用指令遵循扩散模型和多模态大型语言模型，生成针对稀有场景的新颖数据，并进行鲁棒过滤，以解决数据多样性不足的问题。
*   **共识标注策略（Consensus Annotation）：** 代理通过协调多个基础模型（如DETIC、GroundingDINO）作为独立“专家”工具，并采用新颖的共识机制（结合非极大值抑制和投票），从嘈杂的弱标签中生成准确、高质量的伪标签。

**3. 主要结果及其意义：**
Labeling Copilot的组件经过大规模验证，证明了其有效性：
*   **共识标注模块在目标发现方面表现出色：** 在密集的COCO数据集上，它平均每图像生成14.2个候选提案（几乎是7.4个真实对象的两倍），最终标注mAP达到37.1%。
*   **处理极端类别不平衡：** 在网络规模的Open Images数据集上，该模块成功发现了903个新的边界框类别，使其总能力扩展到1500多个。
*   **校准发现工具的计算效率：** 在千万级样本规模下测试时，该工具的主动学习策略比具有同等样本效率的替代方案计算效率高出40倍。
*   **Agentic工作流的鲁棒性：** 这些实验验证了Agentic工作流与优化、可扩展工具相结合，为策展工业级数据集提供了坚实的基础。

**4. 论文中提到的局限性：**
*   **数据策展的复杂性：** 论文指出，数据策展是一个动态挑战，需要智能决策，而非线性管道。这本身就是Agentic框架诞生的原因，但同时也意味着该过程固有的复杂性。
*   **精度-召回率权衡：** 在大规模多类别环境中，管理精度-召回率的权衡是一个根本性挑战，尤其是在Open Images数据集上，尽管召回率高，但精度较低，预测对象数量远高于正确检测数量。
*   **评估挑战：** 评估一个长期运行、自主的代理（如Labeling Copilot）提出了非平凡的基准挑战，因为其目标是生成数据集，这是一个具有广阔和开放式行动空间的任务。传统的端到端评估可能无法捕捉代理在不同领域中的通用效用。

**5. 潜在的未来研究方向：**
*   **工具扩展性：** 论文提到可以轻松添加新的“原始工具”，例如“数据修复”工具用于查找和修复标注错误，或“数据隐私”工具用于自动模糊敏感信息。
*   **解耦智能：** 代理的推理（何时合成更多数据）与工具的执行是分离的，这使得代理的战略智能可以独立于工具的能力进行改进。未来的研究可以进一步探索如何优化这种解耦。
*   **迭代适应：** 代理通过结合工具输出和人工反馈，学习和适应，在迭代周期中逐步改进数据集。未来的工作可以进一步优化这种人机协作和迭代学习机制。
*   **更广泛的CV任务应用：** 代理及其核心工具包可广泛应用于从目标检测到全景分割等各种CV任务，未来的研究可以探索其在更多特定领域（如医学影像、工业检测）中的定制化应用。

---

总而言之，Labeling Copilot通过引入一个创新的Agentic框架，将数据发现、合成和标注整合到一个统一的、智能驱动的工作流中，显著提升了计算机视觉数据集策展的效率、可扩展性和质量。其模块化设计和对大规模数据的处理能力，使其成为解决工业级数据策展挑战的强大工具。

**Key Findings:**

- A central orchestrator agent, powered by a large multimodal
language model, uses multi-step reasoning to execute specialized tools across
three core capabilities: (1) Calibrated Discovery sources relevant,
in-distribution data from large repositories; (2) Controllable Synthesis
generates novel data for rare scenarios with robust filtering; and (3)
Consensus Annotation produces accurate labels by orchestrating multiple
foundation models via a novel consensus mechanism incorporating non-maximum
suppression and voting.
- On the web-scale Open Images dataset, it navigated extreme class
imbalance to discover 903 new bounding box categories, expanding its capability
to over 1500 total.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22631v1)
- [arXiv](https://arxiv.org/abs/2509.22631v1)

---

<a id='2509.22627v1'></a>
## [CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach](https://arxiv.org/abs/2509.22627v1)

**Authors:** Alexandre Lopes, Roberto Souza, Helio Pedrini

**Published:** 2025-09-26

**Categories:** cs.CV

**Abstract:**

Depth Estimation plays a crucial role in recent applications in robotics,
autonomous vehicles, and augmented reality. These scenarios commonly operate
under constraints imposed by computational power. Stereo image pairs offer an
effective solution for depth estimation since it only needs to estimate the
disparity of pixels in image pairs to determine the depth in a known rectified
system. Due to the difficulty in acquiring reliable ground-truth depth data
across diverse scenarios, self-supervised techniques emerge as a solution,
particularly when large unlabeled datasets are available. We propose a novel
self-supervised convolutional approach that outperforms existing
state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers
(ViTs) while balancing computational cost. The proposed CCNeXt architecture
employs a modern CNN feature extractor with a novel windowed epipolar
cross-attention module in the encoder, complemented by a comprehensive redesign
of the depth estimation decoder. Our experiments demonstrate that CCNeXt
achieves competitive metrics on the KITTI Eigen Split test data while being
10.18$\times$ faster than the current best model and achieves state-of-the-art
results in all metrics in the KITTI Eigen Split Improved Ground Truth and
Driving Stereo datasets when compared to recently proposed techniques. To
ensure complete reproducibility, our project is accessible at
\href{https://github.com/alelopes/CCNext}{\texttt{https://github.com/alelopes/CCNext}}.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Alexandre Lopes, Roberto Souza, Helio Pedrini撰写的论文“CCNeXt: An Effective Self-Supervised Stereo Depth Estimation Approach”的全面摘要。

---

### 论文摘要：CCNeXt: 一种有效的自监督立体深度估计方法

**1. 主要问题或研究问题：**
该论文旨在解决自监督立体深度估计领域中存在的关键挑战：如何在实现最先进的深度估计准确性的同时，显著降低计算成本和运行时，以满足机器人、自动驾驶和增强现实等应用对实时性和计算效率的需求。现有的方法往往在Transformer-based模型的高精度与CNN-based模型的低计算成本之间存在权衡。

**2. 关键创新或方法论贡献：**
CCNeXt引入了以下关键创新来解决上述问题：
*   **新型编码器-解码器架构：** 提出了一种基于ConvNeXt的自监督立体深度估计编码器-解码器架构。ConvNeXt作为骨干网络，相比ResNet和Transformer-based模型，在性能和效率之间取得了更好的平衡。
*   **窗口化极线交叉注意力模块（Windowed Epipolar Cross-Attention）：** 在编码器中引入了一种新颖的窗口化极线交叉注意力机制。该机制利用立体几何特性，将注意力限制在有效极线候选范围内，从而显著减少计算复杂性，并防止几何上不合理的错误匹配。
*   **深度估计解码器的全面重新设计（ICEP）：** 提出了一种名为“Individual Contextual Expansive Path (ICEP)”的轻量级解码器，通过改进的卷积层位置和跳跃连接，增强了高分辨率输出的质量和度量，同时降低了执行时间。它通过延迟最高层跳跃连接的融合，确保精细尺度特征更好地用于最终预测，并使用内部跳跃块保留梯度流。
*   **计算效率与准确性平衡：** CCNeXt明确设计用于平衡度量性能和计算效率，旨在超越现有CNN和ViT架构，同时显著提高运行速度。

**3. 主要结果及其意义：**
*   **KITTI Eigen Split数据集：** CCNeXt在KITTI Eigen Split测试数据上取得了具有竞争力的指标，并且比当前最佳模型快10.18倍。
*   **KITTI Eigen Split Improved Ground Truth和Driving Stereo数据集：** 在KITTI Eigen Split Improved Ground Truth和Driving Stereo数据集的所有指标上，CCNeXt均达到了最先进的性能。
*   **计算效率：** 通过FLOPs分析，CCNeXt的完整架构FLOPs远低于ChiTransformer（65.942 GFLOPs vs. 665 GFLOPs），其窗口化交叉注意力模块相比标准全行交叉注意力，计算成本降低了70.6%。
*   **统计学意义：** 对KITTI Eigen Split的统计分析表明，CCNeXt在AbsRel、SqRel、RMSE和δ < 1.25等指标上表现出统计学上的显著优越性。
*   **天气条件下的鲁棒性：** 在DrivingStereo数据集的天气子集（多云、多雾、晴天）上，CCNeXt表现出一致的优异性能，仅在雨天条件下性能略有下降，突显了其在不同环境下的鲁棒性。

**4. 论文中提及的局限性：**
*   **推理对未见数据集的依赖：** 模型在推理时依赖于相机系统的内部参数和外参，这使得在未见数据集上进行推理时可能出现问题，尤其是在使用预测的重新缩放视差时。
*   **多数据集训练的挑战：** 类似地，在多个数据集上进行训练时，重新缩放视差的限制也存在。
*   **对校正图像的依赖：** 模型需要一对校正过的图像来进行深度和视差匹配，不适用于未校正的图像对。

**5. 潜在的未来研究方向：**
*   **整合对比学习或自蒸馏目标：** 将几何约束注意力与对比学习或自蒸馏目标相结合，以进一步提高表征质量，而无需增加监督要求。
*   **更轻量级的架构：** 探索更轻量级的架构，以实现在嵌入式平台上部署，满足严格的延迟和功耗预算。
*   **恶劣环境下的鲁棒性：** 解决在极端天气、夜间或传感器噪声等不利环境因素下的鲁棒性挑战，这需要更大规模的数据集和自适应学习策略。
*   **在线校正方法：** 进一步探索在线校正方法，以处理未校正的立体图像对。

---

这篇论文通过引入创新的ConvNeXt骨干网络、窗口化极线交叉注意力机制和重新设计的解码器，成功地在自监督立体深度估计领域实现了性能和效率的显著提升，为实时应用提供了更可行的解决方案。

**Key Findings:**

- We propose a novel
self-supervised convolutional approach that outperforms existing
state-of-the-art Convolutional Neural Networks (CNNs) and Vision Transformers
(ViTs) while balancing computational cost.
- The proposed CCNeXt architecture
employs a modern CNN feature extractor with a novel windowed epipolar
cross-attention module in the encoder, complemented by a comprehensive redesign
of the depth estimation decoder.
- Our experiments demonstrate that CCNeXt
achieves competitive metrics on the KITTI Eigen Split test data while being
10.18$\times$ faster than the current best model and achieves state-of-the-art
results in all metrics in the KITTI Eigen Split Improved Ground Truth and
Driving Stereo datasets when compared to recently proposed techniques.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22627v1)
- [arXiv](https://arxiv.org/abs/2509.22627v1)

---

<a id='2509.22496v1'></a>
## [Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation](https://arxiv.org/abs/2509.22496v1)

**Authors:** Ruoyu Chen, Xiaoqing Guo, Kangwei Liu, Siyuan Liang, Shiming Liu, Qunli Zhang, Hua Zhang, Xiaochun Cao

**Published:** 2025-09-26

**Categories:** cs.CV

**Abstract:**

Multimodal large language models (MLLMs) have demonstrated remarkable
capabilities in aligning visual inputs with natural language outputs. Yet, the
extent to which generated tokens depend on visual modalities remains poorly
understood, limiting interpretability and reliability. In this work, we present
EAGLE, a lightweight black-box framework for explaining autoregressive token
generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual
regions while quantifying the relative influence of language priors and
perceptual evidence. The framework introduces an objective function that
unifies sufficiency (insight score) and indispensability (necessity score),
optimized via greedy search over sparsified image regions for faithful and
efficient attribution. Beyond spatial attribution, EAGLE performs
modality-aware analysis that disentangles what tokens rely on, providing
fine-grained interpretability of model decisions. Extensive experiments across
open-source MLLMs show that EAGLE consistently outperforms existing methods in
faithfulness, localization, and hallucination diagnosis, while requiring
substantially less GPU memory. These results highlight its effectiveness and
practicality for advancing the interpretability of MLLMs. The code is available
at https://github.com/RuoyuChen10/EAGLE.

**Analysis:**

以下是Ruoyu Chen等人撰写的论文“Where MLLMs Attend and What They Rely On: Explaining Autoregressive Token Generation”的全面摘要：

**1. 主要问题或研究问题：**
多模态大型语言模型（MLLMs）在将视觉输入与自然语言输出对齐方面表现出色，但其生成的文本在多大程度上依赖于视觉模态仍不清楚，这限制了模型的可解释性和可靠性。此外，MLLMs容易产生幻觉，这在安全关键领域（如医疗保健和自动驾驶）中会损害信任。因此，该研究旨在开发一种高效且忠实的归因方法，以提高决策透明度，诊断错误，并增强MLLMs的安全性与可信度。

**2. 关键创新或方法论贡献：**
该论文提出了一个名为EAGLE（Explaining Autoregressive Generation by Language priors or Evidence）的轻量级黑盒归因框架，用于解释MLLMs中的自回归令牌生成。其主要创新包括：
*   **统一的客观函数：** EAGLE引入了一个结合了“洞察力得分”（sufficiency）和“必要性得分”（indispensability）的客观函数。洞察力得分衡量最小输入区域集在最大化目标标签生成概率方面的充分性，而必要性得分则识别移除后会显著降低生成概率的关键区域。
*   **贪婪搜索优化：** 通过对稀疏图像区域进行贪婪搜索来优化客观函数，以实现忠实高效的归因，从而构建一个有序排名，解释哪些感知区域促进了MLLMs的生成。
*   **模态感知分析：** 除了空间归因，EAGLE还进行模态感知分析，量化每个生成的令牌是更多地依赖语言先验还是感知证据，从而提供更细粒度的模型决策可解释性。
*   **幻觉诊断与缓解：** 该方法能够准确识别导致幻觉的视觉元素，并通过移除最少量的干扰区域来缓解幻觉。

**3. 主要结果及其重要性：**
在LLaVA-1.5、Qwen2.5-VL和InternVL3.5等开源MLLMs上进行的广泛实验表明，EAGLE在以下方面持续优于现有方法：
*   **忠实性（Faithfulness）：** 在图像字幕和VQA任务中，EAGLE在插入和删除指标上显著优于现有方法，例如在图像字幕任务中，插入和删除指标平均分别提高了20.0%和13.4%。
*   **定位（Localization）：** 在Pointing Game任务中，EAGLE在箱级和掩码级标注下均取得了最佳结果，证明其能够将预测结果准确地定位到特定对象。
*   **幻觉诊断（Hallucination Diagnosis）：** 在RePOPE基准测试中，EAGLE在平均最小校正区域（AMCR）和预算下校正成功率（CSR@10%）方面显著优于现有方法，表明它能有效识别导致幻觉的输入区域，并通过移除少量区域来纠正幻觉。
*   **GPU内存效率：** EAGLE所需的GPU内存显著少于现有方法，例如在Qwen2.5-VL 7B上仅需17.68 GB，而IGOS++需要96.90 GB，这凸显了其在现代MLLMs中的实用性。

这些结果强调了EAGLE在提高MLLMs可解释性方面的有效性和实用性，能够提供更忠实、资源效率更高且人类可理解的解释。

**4. 论文中提及的局限性：**
该研究提到了两个主要局限性：
*   **可扩展性限制：** 迭代子集选择和贪婪搜索策略限制了EAGLE与轻量级可视化方法相比的可扩展性。
*   **侧重于解释和部分缓解：** 该框架主要关注幻觉的解释和部分缓解，尚未探索主动预防幻觉的方法。

**5. 潜在的未来研究方向：**
未来的研究将探索以下方向：
*   **更快的搜索策略：** 开发更快的搜索策略以提高可扩展性。
*   **解释引导的去偏：** 探索基于解释的去偏方法，用于训练MLLMs，以主动预防幻觉。

**Key Findings:**

- In this work, we present
EAGLE, a lightweight black-box framework for explaining autoregressive token
generation in MLLMs. EAGLE attributes any selected tokens to compact perceptual
regions while quantifying the relative influence of language priors and
perceptual evidence.
- Extensive experiments across
open-source MLLMs show that EAGLE consistently outperforms existing methods in
faithfulness, localization, and hallucination diagnosis, while requiring
substantially less GPU memory.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22496v1)
- [arXiv](https://arxiv.org/abs/2509.22496v1)

---

<a id='2509.22448v1'></a>
## [$γ$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition](https://arxiv.org/abs/2509.22448v1)

**Authors:** Mishal Fatima, Shashank Agnihotri, Marius Bock, Kanchana Vaishnavi Gandikota, Kristof Van Laerhoven, Michael Moeller, Margret Keuper

**Published:** 2025-09-26

**Categories:** cs.CV

**Abstract:**

Most pattern recognition models are developed on pre-proce\-ssed data. In
computer vision, for instance, RGB images processed through image signal
processing (ISP) pipelines designed to cater to human perception are the most
frequent input to image analysis networks. However, many modern vision tasks
operate without a human in the loop, raising the question of whether such
pre-processing is optimal for automated analysis. Similarly, human activity
recognition (HAR) on body-worn sensor data commonly takes normalized
floating-point data arising from a high-bit analog-to-digital converter (ADC)
as an input, despite such an approach being highly inefficient in terms of data
transmission, significantly affecting the battery life of wearable devices. In
this work, we target low-bandwidth and energy-constrained settings where
sensors are limited to low-bit-depth capture. We propose $\gamma$-Quant,
i.e.~the task-specific learning of a non-linear quantization for pattern
recognition. We exemplify our approach on raw-image object detection as well as
HAR of wearable data, and demonstrate that raw data with a learnable
quantization using as few as 4-bits can perform on par with the use of raw
12-bit data. All code to reproduce our experiments is publicly available via
https://github.com/Mishalfatima/Gamma-Quant

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Mishal Fatima等人撰写的论文“$\gamma$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition”的全面摘要。

---

### 论文摘要：$\gamma$-Quant: Towards Learnable Quantization for Low-bit Pattern Recognition

**1. 主要问题或研究问题：**
当前大多数模式识别模型依赖于预处理数据，例如计算机视觉中的RGB图像经过图像信号处理（ISP）管道处理以适应人类感知，以及人体活动识别（HAR）中来自高位模数转换器（ADC）的归一化浮点数据。然而，这种预处理对于自动化分析是否最优，以及其在低带宽、低能耗场景（如可穿戴设备）中的效率低下（导致数据传输量大、电池寿命受影响）是一个关键问题。论文旨在解决传感器受限于低位深度捕获的场景，探究如何为模式识别任务学习一种最优的非线性量化方法。

**2. 关键创新或方法论贡献：**
论文提出了**$\gamma$-Quant**，一种针对特定任务学习非线性量化的方法。其核心思想是，不是使用固定的线性或对数量化，而是通过一个可学习的参数$\gamma$（类似于伽马校正）和一个偏移量$\mu$来参数化量化函数：
$Q(Χ, γ, μ) = Q_N(\text{sign}(X – μ) \cdot |X – μ|^γ)$
这个可学习的非线性量化器与神经网络一起进行优化，以适应特定任务。通过直通估计器（straight-through estimator）处理量化操作的不可微性，从而实现基于梯度的优化。这种方法允许在模拟信号被数字化之前，在ADC层面进行任务特定的量化学习。

**3. 主要结果及其意义：**
*   **性能提升：** $\gamma$-Quant在原始图像目标检测和可穿戴数据HAR任务中均表现出系统性改进。
*   **低位深度下的高表现：** 论文展示了使用低至4比特的原始数据（通过可学习量化）在目标检测任务中可以达到与使用12比特原始数据相当的性能。在HAR任务中，甚至2比特数据也能取得与高比特数据相近的性能。
*   **能耗和带宽优化：** 这种方法显著减少了传感器的数据传输量和能耗，对可穿戴设备等资源受限的应用具有重要意义。
*   **优于线性量化：** $\gamma$-Quant始终优于基于线性量化的模型，尤其是在低亮度区域的细节处理上，这对于目标检测至关重要。
*   **对数与$\gamma$-Quant：** 论文指出，在某些情况下，对数量化（通过精心选择的$\epsilon$值）的性能与$\gamma$-Quant相当，但$\gamma$-Quant的优势在于其参数是自动学习的，无需手动调整超参数，使其更具通用性和适应性。

**4. 论文中提到的局限性：**
*   **模拟信号的代理：** 论文承认，由于无法直接获取真正的模拟信号，其工作中使用高位深度RAW图像作为模拟信号的代理。尽管这在信号处理理论中是有效的假设，但真正的模拟信号与高位深度数字输入之间仍存在信息损失。
*   **理想测试环境：** 理想情况下，作者希望能在传感器上直接测试$\gamma$-Quant，处理模拟信号，以期获得进一步的精度提升。

**5. 潜在的未来研究方向：**
*   **更灵活的量化参数化：** 探索更灵活的量化函数参数化，例如在对数量化中学习$\epsilon$值。
*   **结合网络架构优化：** 将$\gamma$-Quant框架与高效的网络架构和网络量化相结合，以进一步提高计算效率和推理性能。
*   **直接在模拟信号上测试：** 在未来，如果技术允许，直接在传感器上对模拟信号应用$\gamma$-Quant，以验证其在真实世界场景中的潜力。

---

总而言之，这篇论文为在资源受限环境下进行模式识别提供了一种新颖且高效的解决方案。通过引入任务特定的可学习非线性量化，$\gamma$-Quant在保持甚至超越高位深度数据性能的同时，显著降低了数据传输和能耗，为构建更高效、可持续的机器学习工作流奠定了基础。

**Key Findings:**

- We propose $\gamma$-Quant,
i.e.~the task-specific learning of a non-linear quantization for pattern
recognition.
- We exemplify our approach on raw-image object detection as well as
HAR of wearable data, and demonstrate that raw data with a learnable
quantization using as few as 4-bits can perform on par with the use of raw
12-bit data.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22448v1)
- [arXiv](https://arxiv.org/abs/2509.22448v1)

---

<a id='2509.22318v1'></a>
## [NIFTY: a Non-Local Image Flow Matching for Texture Synthesis](https://arxiv.org/abs/2509.22318v1)

**Authors:** Pierrick Chatillon, Julien Rabin, David Tschumperlé

**Published:** 2025-09-26

**Categories:** cs.CV, cs.LG

**Abstract:**

This paper addresses the problem of exemplar-based texture synthesis. We
introduce NIFTY, a hybrid framework that combines recent insights on diffusion
models trained with convolutional neural networks, and classical patch-based
texture optimization techniques. NIFTY is a non-parametric flow-matching model
built on non-local patch matching, which avoids the need for neural network
training while alleviating common shortcomings of patch-based methods, such as
poor initialization or visual artifacts. Experimental results demonstrate the
effectiveness of the proposed approach compared to representative methods from
the literature. Code is available at https://github.com/PierrickCh/Nifty.git

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我对Pierrick Chatillon, Julien Rabin, David Tschumperlé撰写的论文“NIFTY: a Non-Local Image Flow Matching for Texture Synthesis”进行了全面分析，并提供以下摘要：

**论文摘要：NIFTY: 一种用于纹理合成的非局部图像流匹配方法**

**1. 解决的主要问题或研究问题：**
该论文旨在解决基于样本的纹理合成问题。传统的基于补丁的纹理优化方法（如Kwatra等人的工作）虽然有效，但常面临初始化敏感、视觉伪影以及难以捕捉长距离相关性等问题。同时，虽然扩散模型在图像生成方面表现出色，但它们通常需要大量的神经网络训练，且推理过程可能较慢。NIFTY的目标是开发一种无需神经网络训练，同时克服传统基于补丁方法缺陷的纹理合成方法。

**2. 关键创新或方法论贡献：**
NIFTY（Non-local Image Flow-matching Texture sYnthesis）是一个混合框架，其核心创新在于：
*   **非参数流匹配模型：** NIFTY将纹理优化问题重新构建为流匹配（Flow Matching, FM）框架下的一个时间积分问题，但避免了对神经网络的训练。它通过显式计算补丁分布上的速度场来实现流匹配，而不是通过CNN近似。
*   **非局部补丁匹配：** 该方法利用非局部补丁匹配来计算速度场，这与非局部均值（Non-Local Means）的思想相似，通过聚合相似补丁来生成新的纹理。
*   **效率优化策略：** 为了降低计算成本，NIFTY引入了多尺度合成、top-k最近邻（NN）采样以及记忆机制。top-k NN采样限制了速度场计算中考虑的补丁数量，而记忆机制则保留了迭代过程中表现更好的k-NN索引，以减少后续计算量并提高稳定性。
*   **与传统TO的联系：** 论文指出，NIFTY在特定条件下（例如k=1，T=1）与传统的纹理优化（TO）算法有相似之处，但通过流的融合而非直接最近邻替换，NIFTY表现出更稳定的行为。

**3. 主要结果及其意义：**
*   **鲁棒性和图像质量：** 实验结果表明，NIFTY在初始化鲁棒性、图像质量和速度方面优于现有方法。与传统的纹理优化算法相比，NIFTY能够避免复制区域之间的强伪影，并生成更逼真的纹理。
*   **创造性合成：** NIFTY能够通过拼接原始纹理的局部副本，生成未曾见过但合理的新结构，展示了其在合成中的“创造性”。
*   **效率提升：** 通过top-k采样和记忆机制，NIFTY在保持高质量合成的同时，显著减少了计算成本和所需的步骤数量。定量分析（如W2距离、SIFID、自相关性）证实了NIFTY在质量和速度上优于TO和小型U-Net模型。
*   **插值能力：** 论文还展示了NIFTY在纹理混合方面的应用，包括分布级混合、像素级混合和空间插值，证明了其在生成不同纹理之间平滑过渡的能力。
*   **潜在空间应用：** NIFTY可以与预训练的自编码器结合，在潜在空间中进行纹理合成，这表明了其方法的通用性和扩展潜力。

**4. 论文中提及的局限性：**
论文中并未明确列出NIFTY方法的具体局限性，但可以从其设计和比较中推断出一些潜在的方面：
*   **计算成本：** 尽管NIFTY通过top-k NN和记忆机制降低了计算成本，但对于非常大的图像或需要极高精度的场景，非局部补丁匹配的计算量仍然可能是一个挑战。
*   **超参数敏感性：** 尽管NIFTY比传统TO对初始化不那么敏感，但其性能可能仍然依赖于补丁大小、步长、k值、时间步数T等超参数的选择。
*   **长距离相关性：** 尽管NIFTY通过非局部匹配试图捕捉长距离相关性，但对于具有非常复杂或高度结构化全局模式的纹理，其捕捉能力可能仍有提升空间。

**5. 潜在的未来研究方向：**
*   **潜在空间中的补丁流匹配：** 进一步探索在合适的潜在空间中进行补丁流匹配，以利用更抽象的特征表示。
*   **集成更复杂的归纳偏置：** 未来的工作可以考虑集成更复杂的归纳偏置，例如通过非局部补丁聚合建模注意力模块，以进一步提升模型的性能和泛化能力。
*   **实时或更快速的合成：** 探索进一步优化算法，以实现更快的合成速度，甚至实时纹理生成。
*   **处理非平稳纹理：** 尽管论文主要关注平稳纹理，但NIFTY的非局部特性可能使其有潜力扩展到非平稳纹理的合成。

总而言之，NIFTY提出了一种新颖的非参数流匹配方法，通过将扩散模型和传统补丁优化技术相结合，在无需神经网络训练的情况下，有效地解决了基于样本的纹理合成问题，并取得了令人信服的实验结果。

**Key Findings:**

- Experimental results demonstrate the
effectiveness of the proposed approach compared to representative methods from
the literature.
- Code is available at https://github.com/PierrickCh/Nifty.git

**Links:**

- [PDF](https://arxiv.org/pdf/2509.22318v1)
- [arXiv](https://arxiv.org/abs/2509.22318v1)

---

