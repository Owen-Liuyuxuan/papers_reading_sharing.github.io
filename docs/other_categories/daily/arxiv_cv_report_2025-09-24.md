time: 20250924

# Arxiv Computer Vision Papers - 2025-09-24

## Executive Summary

好的，这是一份针对您提供的 Arxiv 计算机视觉论文列表的每日执行摘要，旨在帮助忙碌的研究人员快速了解最新进展。

---

**每日 Arxiv 计算机视觉论文执行摘要 (2025-09-23)**

**概述与主要趋势：**

今天的论文涵盖了计算机视觉领域的多个前沿方向，呈现出以下几个主要趋势：

1.  **多模态与大模型持续演进：** 以 Qwen3-Omni 为代表的大型多模态模型在通用能力上持续突破，预示着更强大的感知与理解能力。
2.  **可解释性与鲁棒性日益受重视：** 随着 AI 应用的深入，对模型决策的理解（XAI-CV）和在复杂环境（如恶劣天气）下的鲁棒性（RoSe）成为关键研究点。
3.  **3D 视觉与新颖表示方法：** 3D Gaussian Splatting 及其变体（VolSplat）继续是热门方向，探索更高效、高质量的 3D 重建与渲染。
4.  **特定应用场景的深度探索：** 论文涵盖了从工业自动化（Intermodal Loading Unit Identification）、创意内容生成（Layout-to-Image）、视频分析（Video Similarity）到环境监测（Arctic Seals Survey）等多个垂直领域，显示出计算机视觉技术在解决实际问题中的广泛应用。
5.  **基准测试与评估：** 多个新基准（OverLayBench, ConViS-Bench）的提出，反映了社区对更全面、更具挑战性的评估标准的需求。

**特别重要或创新的论文：**

*   **Qwen3-Omni Technical Report (Jin Xu et al.):** 这篇技术报告无疑是今日最受关注的论文之一。作为大型多模态模型的最新进展，它可能在通用视觉理解、跨模态交互和多任务处理方面带来显著提升，对整个领域具有深远影响。
*   **VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction (Weijie Wang et al.):** 在 3D Gaussian Splatting 领域，这篇论文通过引入 Voxel-Aligned Prediction 重新思考前向传播，有望在渲染质量和效率上取得新的突破，是 3D 视觉研究的重要进展。
*   **xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision (Nguyen Van Tu et al.):** 这篇综述论文对于理解当前计算机视觉领域可解释 AI 的现状、挑战和未来方向至关重要。在 AI 伦理和信任日益重要的背景下，其价值不言而喻。

**新兴研究方向或技术：**

*   **多模态大模型与通用智能：** Qwen3-Omni 的出现进一步巩固了多模态大模型作为未来 AI 核心的发展方向，其通用性将是未来研究的重点。
*   **高效且高质量的 3D 表示与渲染：** VolSplat 等工作表明，在 3D Gaussian Splatting 的基础上，如何进一步优化其结构、提高效率和渲染质量仍是活跃的研究领域。
*   **文本驱动的视觉检索与生成：** "Vision-Free Retrieval" 和 "Layout-to-Image Generation" 强调了文本作为核心媒介在视觉任务中的强大潜力，预示着更智能、更灵活的跨模态交互。
*   **恶劣环境下的鲁棒性：** RoSe 专注于恶劣天气下的立体匹配，凸显了在非理想条件下保持模型性能的重要性，这对于自动驾驶、机器人等实际应用至关重要。

**建议阅读的论文：**

对于希望全面了解最新进展的研究人员，建议优先阅读以下论文：

1.  **Qwen3-Omni Technical Report (Jin Xu et al.):** 了解大型多模态模型的最新能力和技术细节。
2.  **xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision (Nguyen Van Tu et al.):** 掌握可解释 AI 在计算机视觉中的全貌和未来趋势。
3.  **VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction (Weijie Wang et al.):** 如果您关注 3D 视觉和新颖的场景表示方法。
4.  **RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions (Yun Wang et al.):** 如果您对恶劣环境下的鲁棒性视觉感知感兴趣，特别是自动驾驶或机器人领域。
5.  **OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps (Bingnan Li et al.):** 如果您从事生成式 AI 或图像合成领域，这个新基准可能提供新的研究方向。

---

这份摘要旨在提供一个高层次的概览，帮助您快速筛选出最相关的论文。详细内容仍需查阅原文。

---

## Table of Contents

1. [xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision](#2509.18913v1)
2. [Qwen3-Omni Technical Report](#2509.17765v1)
3. [Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review](#2509.17707v1)
4. [VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction](#2509.19297v1)
5. [OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](#2509.19282v1)
6. [Adversarially-Refined VQ-GAN with Dense Motion Tokenization for Spatio-Temporal Heatmaps](#2509.19252v1)
7. [ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](#2509.19245v1)
8. [Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions](#2509.19203v1)
9. [RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions](#2509.19165v1)
10. [KAMERA: Enhancing Aerial Surveys of Ice-associated Seals in Arctic Environments](#2509.19129v1)

---

## Papers

<a id='2509.18913v1'></a>
## [xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision](https://arxiv.org/abs/2509.18913v1)

**Authors:** Nguyen Van Tu, Pham Nguyen Hai Long, Vo Hoai Viet

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

Deep learning has become the de facto standard and dominant paradigm in image
analysis tasks, achieving state-of-the-art performance. However, this approach
often results in "black-box" models, whose decision-making processes are
difficult to interpret, raising concerns about reliability in critical
applications. To address this challenge and provide human a method to
understand how AI model process and make decision, the field of xAI has
emerged. This paper surveys four representative approaches in xAI for visual
perception tasks: (i) Saliency Maps, (ii) Concept Bottleneck Models (CBM),
(iii) Prototype-based methods, and (iv) Hybrid approaches. We analyze their
underlying mechanisms, strengths and limitations, as well as evaluation
metrics, thereby providing a comprehensive overview to guide future research
and applications.

**Analysis:**

好的，这是一篇关于“xAI-CV: An Overview of Explainable Artificial Intelligence in Computer Vision”的全面摘要，用中文写成：

**论文摘要：xAI-CV：计算机视觉中可解释人工智能的概述**

**1. 主要问题或研究问题：**
深度学习模型在计算机视觉任务中取得了最先进的性能，但其“黑箱”特性使得决策过程难以解释，这在关键应用中引发了对可靠性的担忧。本文旨在解决这一挑战，通过对可解释人工智能（xAI）在视觉感知任务中的四种代表性方法进行系统性概述，为人类理解AI模型如何处理和决策提供方法。

**2. 关键创新或方法论贡献：**
本文的主要贡献在于对xAI领域四种主要方法的结构化分析和综合，阐明了它们的底层机制、优势、局限性以及评估指标。这四种方法是：
*   **显著性图（Saliency Maps）：** 回答“模型看哪里？”的问题，通过热力图突出图像中重要的像素级区域。
*   **概念瓶颈模型（Concept Bottleneck Models, CBM）：** 回答“模型在思考什么？”的问题，通过强制模型通过人类可理解的高级概念进行推理。
*   **基于原型的方法（Prototype-based methods）：** 回答“这个数据像哪个案例？”的问题，通过与从数据集中学习到的原型进行比较来解释预测。
*   **混合方法（Hybrid approaches）：** 结合上述方法，以利用各自的优势，提供更全面、可靠和灵活的解释。

论文详细分析了每种方法的演进阶段、实际应用以及评估指标，包括AOPC、熵、删除/插入、指向游戏、任务准确性、概念准确性、可干预性测试、显著性图对齐、定量检查、CGIM、CEM、CLM以及TCAV分数等。

**3. 主要结果及其意义：**
*   **显著性图**作为基础工具，能够识别对预测影响最大的图像区域，是调试和检测虚假特征的诊断步骤。
*   **概念瓶颈模型**引入了干预能力，允许人类与模型交互、纠正错误并测试因果假设，这对于在专业领域建立信任至关重要。
*   **基于原型的方法**通过原型示例进行解释，揭示了模型学习到的数据分布并识别边界案例，其推理方式更接近人类思维。
*   **混合方法**通过结合不同技术的优势，提供了更全面的解释，例如将显著性图的空间定位能力与CBM的语义推理相结合，同时回答“哪里”和“为什么”的问题，从而提高了整体可靠性。

这些方法共同为理解模型行为提供了多方面的工具包，标志着xAI领域从回答“模型看哪里？”的基本问题，发展到解决“模型用什么概念进行推理？”等更复杂查询的演进。

**4. 论文中提及的局限性：**
尽管xAI取得了显著进展，但仍面临系统性挑战：
*   **解释的忠实性和鲁棒性：** 解释是否真正反映了模型的内部推理过程仍是一个主要问题。显著性图可能不稳定，CBM也可能学习“捷径”而非忠实地代表人类定义的概念。
*   **评估和标准化问题：** 缺乏标准化指标和基准数据集，使得量化“好”的解释变得困难，难以公平比较不同方法。
*   **对人类的依赖和可扩展性：** CBM和基于原型的方法通常需要大量的人工参与来定义概念或解释原型，这限制了它们的可扩展性。
*   **显著性图的局限性：** 结果不稳定、量化困难、存在误解风险、缺乏概念层面解释。
*   **CBM的局限性：** 概念定义负担重、存在捷径学习风险、可扩展性挑战。
*   **基于原型方法的局限性：** 需要修改模型架构、难以学习有意义的原型、计算成本高。
*   **混合方法的局限性：** 实现、计算和解释结果的复杂性高、标准化评估困难、需要用户投入大量精力。

**5. 潜在的未来研究方向：**
为了解决上述系统性挑战，未来的研究方向包括：
*   **从基于相关性的解释转向因果推理：** 未来的方法需要能够生成反事实解释，以查明驱动模型决策的真正因果因素。
*   **建立全面和标准化的评估框架：** 需要包含忠实性和鲁棒性的计算度量，并结合人机交互研究来评估解释对最终用户的有用性和有效性。
*   **开发能够自动发现有意义概念和原型的方法：** 旨在提高可扩展性并减少对领域专家的依赖。
*   **构建原则性的混合模型：** 能够协调空间准确性与语义丰富性，将xAI从实验室诊断工具转变为真实世界关键AI系统中不可或缺的伙伴。

总之，本文提供了一个框架，帮助研究人员和实践者驾驭当前的xAI格局，并做出明智的决策，推动xAI向构建真正值得信赖的系统发展，以实现AI在关键领域的安全和负责任部署。

**Key Findings:**

- Deep learning has become the de facto standard and dominant paradigm in image
analysis tasks, achieving state-of-the-art performance.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.18913v1)
- [arXiv](https://arxiv.org/abs/2509.18913v1)

---

<a id='2509.17765v1'></a>
## [Qwen3-Omni Technical Report](https://arxiv.org/abs/2509.17765v1)

**Authors:** Jin Xu, Zhifang Guo, Hangrui Hu, Yunfei Chu, Xiong Wang, Jinzheng He, Yuxuan Wang, Xian Shi, Ting He, Xinfa Zhu, Yuanjun Lv, Yongqi Wang, Dake Guo, He Wang, Linhan Ma, Pei Zhang, Xinyu Zhang, Hongkun Hao, Zishan Guo, Baosong Yang, Bin Zhang, Ziyang Ma, Xipin Wei, Shuai Bai, Keqin Chen, Xuejing Liu, Peng Wang, Mingkun Yang, Dayiheng Liu, Xingzhang Ren, Bo Zheng, Rui Men, Fan Zhou, Bowen Yu, Jianxin Yang, Le Yu, Jingren Zhou, Junyang Lin

**Published:** 2025-09-22

**Categories:** cs.CL, cs.AI, cs.CV, eess.AS

**Abstract:**

We present Qwen3-Omni, a single multimodal model that, for the first time,
maintains state-of-the-art performance across text, image, audio, and video
without any degradation relative to single-modal counterparts. Qwen3-Omni
matches the performance of same-sized single-modal models within the Qwen
series and excels particularly on audio tasks. Across 36 audio and audio-visual
benchmarks, Qwen3-Omni achieves open-source SOTA on 32 benchmarks and overall
SOTA on 22, outperforming strong closed-source models such as Gemini-2.5-Pro,
Seed-ASR, and GPT-4o-Transcribe. Qwen3-Omni adopts a Thinker-Talker MoE
architecture that unifies perception and generation across text, images, audio,
and video, yielding fluent text and natural real-time speech. It supports text
interaction in 119 languages, speech understanding in 19 languages, and speech
generation in 10 languages. To reduce first-packet latency in streaming
synthesis, Talker autoregressively predicts discrete speech codecs using a
multi-codebook scheme. Leveraging the representational capacity of these
codebooks, we replace computationally intensive block-wise diffusion with a
lightweight causal ConvNet, enabling streaming from the first codec frame. In
cold-start settings, Qwen3-Omni achieves a theoretical end-to-end first-packet
latency of 234 ms. To further strengthen multimodal reasoning, we introduce a
Thinking model that explicitly reasons over inputs from any modality. Since the
research community currently lacks a general-purpose audio captioning model, we
fine-tuned Qwen3-Omni-30B-A3B to obtain Qwen3-Omni-30B-A3B-Captioner, which
produces detailed, low-hallucination captions for arbitrary audio inputs.
Qwen3-Omni-30B-A3B, Qwen3-Omni-30B-A3B-Thinking, and
Qwen3-Omni-30B-A3B-Captioner are publicly released under the Apache 2.0
license.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供《Qwen3-Omni Technical Report》的全面摘要。

---

### 《Qwen3-Omni Technical Report》摘要

**1. 解决的主要问题或研究问题：**
该论文旨在解决当前多模态大模型普遍存在的模态间性能权衡问题，即在提升某一模态性能时，往往伴随着其他模态的性能下降。Qwen3-Omni致力于构建一个单一的多模态模型，能够在文本、图像、音频和视频等多种模态上同时保持最先进的性能，且不发生任何性能退化，并显著增强跨模态推理和交互能力。

**2. 关键创新或方法论贡献：**
Qwen3-Omni在Qwen2.5-Omni的Thinker-Talker架构基础上进行了五项关键升级，实现了其卓越性能：
*   **MoE架构升级：** Thinker和Talker均升级为MoE（Mixture-of-Experts）架构，以支持高并发和快速推理。
*   **AuT音频编码器：** 引入了从头训练的AuT（Audio Transformer）编码器，取代了Whisper音频编码器，在2000万小时的监督音频数据上训练，生成更强的通用音频表示，并采用块级窗口注意力机制实现实时预填充缓存。
*   **多码本语音生成：** 采用多码本表示和多轨编解码建模，Talker通过MTP模块自回归预测多个码本层，以忠实地建模多样化的声音、副语言线索和声学现象。
*   **轻量级ConvNet波形合成：** 波形生成阶段用轻量级因果ConvNet取代了计算密集型的块级扩散模型（DiT），实现了从第一个编解码帧开始的流式合成，显著降低了推理延迟和计算成本。
*   **低延迟流式交互：** 输入和输出音频码率降低至12.5 Hz，输出编解码器支持单帧即时语音合成，在冷启动设置下实现了234毫秒的理论端到端首包延迟。
*   **显式多模态推理：** 引入了一个“Thinking”模型，能够显式地对来自任何模态的输入进行推理，进一步增强了多模态推理能力。
*   **音频字幕模型：** 通过微调Qwen3-Omni-30B-A3B，得到了Qwen3-Omni-30B-A3B-Captioner，能够为任意音频输入生成详细、低幻觉的字幕。

**3. 主要结果及其意义：**
*   **SOTA性能：** Qwen3-Omni在36个音频和视听基准测试中，在32个基准测试上实现了开源SOTA，并在22个基准测试上取得了总体SOTA，超越了Gemini-2.5-Pro、Seed-ASR和GPT-4o-Transcribe等强大的闭源模型。
*   **模态间无退化：** 首次证明了通过集成多模态训练，可以在所有模态上实现性能均等，即没有模态特定的性能下降，同时显著增强了视频理解等跨模态能力。
*   **广泛的语言支持：** 支持119种文本交互语言，19种语音理解语言和10种语音生成语言。
*   **实时交互能力：** 能够处理长达40分钟的音频录音，实现ASR和口语理解，并支持通过用户定义的系统提示进行细粒度的对话语气和角色定制。
*   **开放性：** Qwen3-Omni-30B-A3B、Qwen3-Omni-30B-A3B-Thinking和Qwen3-Omni-30B-A3B-Captioner均在Apache 2.0许可下公开发布。

**4. 论文中提及的局限性：**
*   **长视频基准测试性能次优：** 当前模型在长视频基准测试上的性能仍有提升空间，这主要源于两个架构限制：位置外推能力有限和受限的上下文长度。
*   **计算成本高昂：** 由于实验成本高昂，未能对所有模型规模进行全面的性能评估。
*   **语言能力提升不明显：** 经验表明，添加视觉或音频信号并未在语言能力上带来可衡量的提升。

**5. 潜在的未来研究方向：**
*   **多扬声器ASR：** 进一步支持多扬声器自动语音识别。
*   **视频OCR：** 增强视频中的光学字符识别能力。
*   **视听主动学习：** 探索视听领域的主动学习机制。
*   **基于代理的工作流和函数调用：** 增强对基于代理的工作流和函数调用的支持。
*   **解决长视频理解的局限性：** 改进架构以提升模型在长视频基准测试上的性能，特别是解决位置外推和上下文长度的限制。

---

总而言之，Qwen3-Omni代表了多模态AI领域的一个重要里程碑，它首次在不牺牲任何单模态性能的前提下，实现了文本、图像、音频和视频的统一SOTA表现。其创新的Thinker-Talker MoE架构、AuT编码器、多码本语音生成和轻量级ConvNet，共同构建了一个高效、低延迟且功能强大的多模态系统，为未来的多模态研究和应用奠定了坚实基础。

**Key Findings:**

- We present Qwen3-Omni, a single multimodal model that, for the first time,
maintains state-of-the-art performance across text, image, audio, and video
without any degradation relative to single-modal counterparts.
- To further strengthen multimodal reasoning, we introduce a
Thinking model that explicitly reasons over inputs from any modality.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17765v1)
- [arXiv](https://arxiv.org/abs/2509.17765v1)

---

<a id='2509.17707v1'></a>
## [Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review](https://arxiv.org/abs/2509.17707v1)

**Authors:** Emre Gülsoylu, Alhassan Abdelhalim, Derya Kara Boztas, Ole Grasse, Carlos Jahn, Simone Frintrop, Janick Edinger

**Published:** 2025-09-22

**Categories:** cs.CV

**Abstract:**

The standardisation of Intermodal Loading Units (ILUs), such as containers,
semi-trailers and swap bodies, has revolutionised global trade yet their
efficient and robust identification remains a critical bottleneck in
high-throughput ports and terminals. This paper reviews 63 empirical studies
that propose computer vision (CV) based solutions. It covers the last 35 years
(1990-2025), tracing the field's evolution from early digital image processing
(DIP) and traditional machine learning (ML) to the current dominance of deep
learning (DL) techniques. While CV offers cost-effective alternatives for other
types of identification techniques, its development is hindered by the lack of
publicly available benchmarking datasets. This results in high variance for the
reported results such as end-to-end accuracy ranging from 5 % to 96 %. Beyond
dataset limitations, this review highlights the emerging challenges especially
introduced by the shift from character-based text recognition to scene-text
spotting and the integration of mobile cameras (e.g. drones, sensor equipped
ground vehicles) for dynamic terminal monitoring. To advance the field, the
paper calls for standardised terminology, open-access datasets, shared source
code, while outlining future research directions such as contextless text
recognition optimised for ISO6346 codes.

**Analysis:**

好的，这是一篇关于Emre Gülsoylu等人撰写的论文“Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review”的全面摘要：

**论文题目：** 使用计算机视觉的自动多式联运装载单元识别：一项范围界定综述

**1. 解决的主要问题或研究问题：**
该论文旨在解决多式联运装载单元（ILU，如集装箱、半挂车和可换箱体）在港口和码头进行高效、鲁棒识别的关键瓶颈。尽管ILU的标准化彻底改变了全球贸易，但其识别过程仍面临挑战。该综述通过回顾过去35年（1990-2025）基于计算机视觉（CV）的解决方案，探讨了该领域的发展、当前技术状况、挑战以及未来的研究方向。

**2. 关键创新或方法论贡献：**
*   **全面综述：** 本文对63项实证研究进行了系统性综述，涵盖了从早期数字图像处理（DIP）和传统机器学习（ML）到当前深度学习（DL）技术的演变。
*   **术语标准化：** 论文重新评估并提供了ILU识别领域中使用的术语的清晰定义，包括DIP、ML、DL、对象检测、实例分割、字符分割、字符检测、场景文本检测、场景文本识别和文本识别（Text Spotting）。
*   **方法论演变分析：** 详细分析了ILU识别方法随时间推移的演变，指出从字符级识别向场景文本识别的转变，以及固定摄像头向移动摄像头（如无人机、地面车辆）的转变。
*   **数据集特性总结：** 总结了现有数据集的采集设置（固定/移动摄像头）、多样性（真实世界/受控环境、天气、光照、损坏等）和可用性，强调了公共基准数据集的缺乏。
*   **评估指标分析：** 讨论了ILU识别中常用的评估指标，并强调了端到端准确率作为最严格和最相关的指标。

**3. 主要结果及其意义：**
*   **技术演变：** ILU识别领域已从DIP和传统ML方法发展到DL技术的显著主导，尤其是在2016-2020年之后，DL方法在处理复杂条件下的鲁棒性表现出优越性。
*   **地域集中：** 亚洲地区在ILU识别研究中占据主导地位（79.71%的出版物），这反映了该地区作为全球贸易枢纽对高效物流的迫切需求和研发投入。
*   **资金来源：** 公共资金是主要支持来源（38.10%），表明政府对基础设施和经济增长的重视。
*   **数据集限制：** 绝大多数数据集（85.71%）不公开可用，导致研究结果的可重复性和比较性差，端到端准确率报告范围从5%到96%不等，差异巨大。
*   **挑战转变：** 识别任务从字符级文本识别转向场景文本识别，并整合移动摄像头（如无人机、地面车辆），引入了新的复杂性，如动态场景监控和姿态估计。
*   **出版物分布：** 超过一半的论文发表在未排名的期刊或会议上，这表明该领域需要更强调新颖贡献、可比性和结果可重复性。

**4. 论文中提到的局限性：**
*   **缺乏公开基准数据集：** 这是该领域研究和开发的最大障碍，导致模型训练数据不足，结果可比性差，且难以进行公平的方法比较。
*   **术语不标准化：** 现有文献中术语使用不一致，影响了研究的清晰度和检索效率。
*   **数据多样性不足：** 尽管一些数据集试图包含多样化的条件，但来自多个地点和场景的图像数据集仍然稀缺，限制了模型的泛化能力。
*   **DL模型训练数据量小：** 现有数据集的规模对于DL模型的鲁棒训练来说通常太小，迫使研究人员适应通用文本检测和识别模型。
*   **性能评估差异大：** 由于数据集不同，报告的端到端准确率差异巨大，难以对方法进行有效比较。

**5. 潜在的未来研究方向：**
*   **标准化术语：** 建立统一的术语体系，以提高研究的清晰度和可比性。
*   **开放获取数据集和共享代码：** 呼吁创建公开可用的基准数据集和共享源代码，以促进研究的可重复性、公平比较和领域进步。
*   **无上下文文本识别：** 针对ISO6346代码的无上下文文本识别进行优化，因为这些代码缺乏自然语言上下文，现有依赖语言模型的场景文本识别模型可能效率低下。
*   **移动摄像头集成：** 进一步研究和开发适用于移动摄像头（如无人机、地面车辆）的ILU识别系统，以实现动态终端监控，并解决姿态估计等新任务。
*   **实时处理优化：** 关注算法优化，使其能在移动设备上实时运行，并通过视频流分析提高系统效率。
*   **合成数据生成：** 探索使用游戏引擎等方法生成合成数据，以弥补真实世界数据集的不足。
*   **图像质量提升：** 采用图像增强技术（如降噪、GANs）和后处理步骤来处理低光照、遮挡、损坏等挑战性条件。
*   **网络架构优化：** 改进DL网络架构，增加感受野，并引入注意力机制以提高识别准确率。
*   **多视角识别：** 利用从不同角度捕获的多个图像进行多视角识别，以应对退化ID代码的挑战。
*   **公私合作：** 鼓励公共部门和私营企业之间的合作，共同资助和开发ILU识别技术。

总而言之，这篇综述为计算机视觉在ILU识别领域的应用提供了全面的视角，强调了DL技术的崛起、亚洲在该领域的领导地位，并明确指出了当前面临的数据集和标准化挑战，为未来的研究指明了方向。

**Key Findings:**

- Beyond
dataset limitations, this review highlights the emerging challenges especially
introduced by the shift from character-based text recognition to scene-text
spotting and the integration of mobile cameras (e.g. drones, sensor equipped
ground vehicles) for dynamic terminal monitoring.
- To advance the field, the
paper calls for standardised terminology, open-access datasets, shared source
code, while outlining future research directions such as contextless text
recognition optimised for ISO6346 codes.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17707v1)
- [arXiv](https://arxiv.org/abs/2509.17707v1)

---

<a id='2509.19297v1'></a>
## [VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction](https://arxiv.org/abs/2509.19297v1)

**Authors:** Weijie Wang, Yeqing Chen, Zeyu Zhang, Hengyu Liu, Haoxiao Wang, Zhiyuan Feng, Wenkang Qin, Zheng Zhu, Donny Y. Chen, Bohan Zhuang

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective
solution for novel view synthesis. Existing methods predominantly rely on a
pixel-aligned Gaussian prediction paradigm, where each 2D pixel is mapped to a
3D Gaussian. We rethink this widely adopted formulation and identify several
inherent limitations: it renders the reconstructed 3D models heavily dependent
on the number of input views, leads to view-biased density distributions, and
introduces alignment errors, particularly when source views contain occlusions
or low texture. To address these challenges, we introduce VolSplat, a new
multi-view feed-forward paradigm that replaces pixel alignment with
voxel-aligned Gaussians. By directly predicting Gaussians from a predicted 3D
voxel grid, it overcomes pixel alignment's reliance on error-prone 2D feature
matching, ensuring robust multi-view consistency. Furthermore, it enables
adaptive control over Gaussian density based on 3D scene complexity, yielding
more faithful Gaussian point clouds, improved geometric consistency, and
enhanced novel-view rendering quality. Experiments on widely used benchmarks
including RealEstate10K and ScanNet demonstrate that VolSplat achieves
state-of-the-art performance while producing more plausible and view-consistent
Gaussian reconstructions. In addition to superior results, our approach
establishes a more scalable framework for feed-forward 3D reconstruction with
denser and more robust representations, paving the way for further research in
wider communities. The video results, code and trained models are available on
our project page: https://lhmd.top/volsplat.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Weijie Wang等人撰写的论文“VolSplat: Rethinking Feed-Forward 3D Gaussian Splatting with Voxel-Aligned Prediction”的全面摘要。

---

**论文摘要：VolSplat：重新思考前向3D高斯泼溅与体素对齐预测**

**1. 主要问题或研究问题：**
该论文旨在解决前向3D高斯泼溅（3DGS）方法中像素对齐范式的固有局限性。现有方法将每个2D像素映射到一个3D高斯，导致重建的3D模型严重依赖输入视图数量、视图偏置的密度分布，并在源视图存在遮挡或低纹理时引入对齐误差。这些限制导致几何精度不足、浮点问题以及无法根据场景复杂性自适应控制高斯密度，从而影响新颖视图渲染的质量和表示的鲁棒性。

**2. 关键创新或方法贡献：**
VolSplat引入了一种新颖的多视图前向范式，用**体素对齐高斯**取代了传统的像素对齐。其核心创新和贡献包括：
*   **体素对齐预测：** VolSplat直接从预测的3D体素网格中预测高斯，从而克服了像素对齐对易出错的2D特征匹配的依赖，确保了鲁棒的多视图一致性。
*   **自适应高斯密度控制：** 该方法能够根据3D场景的复杂性自适应控制高斯密度，从而生成更忠实的高斯点云、改进的几何一致性并增强新颖视图的渲染质量。
*   **3D特征构建与精炼：** 首先，通过Transformer网络和平面扫描构建每视图成本体，并使用深度预测模块估计深度图。然后，将2D特征反投影到3D空间形成体素特征网格。接着，使用稀疏3D U-Net解码器对这些特征进行精炼，以预测每个占据体素的3D高斯参数。这种残差精炼架构有助于学习校正项，同时保持粗粒度体素信息。
*   **解耦3D表示与2D像素网格：** 通过体素对齐，该方法将3D表示从2D像素网格的约束中解耦出来，解决了高斯密度与输入图像分辨率刚性耦合的问题。

**3. 主要结果及其重要性：**
*   **最先进的性能：** 在RealEstate10K和ScanNet等广泛使用的基准测试中，VolSplat实现了最先进的性能，生成了更合理、视图一致的高斯重建。
*   **更少的浮点和伪影：** 渲染图像在物体边界处基本没有竞争方法中常见的浮点和伪影，这直接归因于模型解决3D特征表示中多视图对齐问题的能力。
*   **卓越的泛化能力：** 在跨数据集泛化实验（例如在ACID数据集上进行零样本迁移）中，VolSplat表现出显著更高的性能，这表明其体素对齐框架具有固有的鲁棒性。
*   **高效的高斯密度管理：** 与像素对齐方法相比，VolSplat能够根据场景复杂性自适应地分配高斯，避免了简单区域的过度密集化和复杂几何区域的不足。它通常以更高效、更紧凑的高斯集实现卓越的渲染质量。

**4. 论文中提及的局限性：**
论文中没有明确指出VolSplat的局限性，但从其方法描述和消融研究中可以推断出一些潜在的考虑因素：
*   **体素大小选择：** 体素大小是一个关键的超参数，它在几何表示的保真度和计算资源消耗之间存在权衡。虽然论文通过实验找到了一个有效的平衡点，但对于特定应用或极端场景，体素大小的选择可能仍需仔细调整。
*   **计算资源：** 尽管稀疏3D U-Net提高了效率，但处理3D体素网格（尤其是对于非常精细的体素大小）仍然可能需要显著的内存和处理时间，这可能限制了其在资源受限设备上的应用。

**5. 潜在的未来研究方向：**
*   **更高效的体素表示：** 进一步探索更高效的稀疏体素数据结构或自适应体素网格，以在保持几何细节的同时减少内存和计算开销。
*   **动态场景和实时应用：** 将VolSplat扩展到动态场景，并进一步优化其推理速度，以支持更严格的实时应用，例如机器人导航和AR/VR。
*   **多模态融合：** 探索将VolSplat与更多模态输入（如LiDAR点云、语义信息等）融合，以增强3D重建的鲁棒性和细节。
*   **更广泛的泛化：** 进一步研究如何提高模型在更多样化、更具挑战性的场景（例如极端光照、反射表面等）中的泛化能力。
*   **与下游任务的集成：** 将VolSplat生成的鲁棒3D表示与各种下游任务（如物体检测、语义分割、路径规划等）更紧密地集成。

---

总而言之，VolSplat通过引入体素对齐预测范式，对前向3DGS领域做出了重大贡献。它有效地解决了现有像素对齐方法的关键局限性，提供了更鲁棒、更一致且自适应的3D场景重建。这项工作为未来在更广泛社区中进行前向3D重建的研究奠定了基础。

**Key Findings:**

- Feed-forward 3D Gaussian Splatting (3DGS) has emerged as a highly effective
solution for novel view synthesis.
- To address these challenges, we introduce VolSplat, a new
multi-view feed-forward paradigm that replaces pixel alignment with
voxel-aligned Gaussians.
- Furthermore, it enables
adaptive control over Gaussian density based on 3D scene complexity, yielding
more faithful Gaussian point clouds, improved geometric consistency, and
enhanced novel-view rendering quality.
- Experiments on widely used benchmarks
including RealEstate10K and ScanNet demonstrate that VolSplat achieves
state-of-the-art performance while producing more plausible and view-consistent
Gaussian reconstructions.
- In addition to superior results, our approach
establishes a more scalable framework for feed-forward 3D reconstruction with
denser and more robust representations, paving the way for further research in
wider communities.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19297v1)
- [arXiv](https://arxiv.org/abs/2509.19297v1)

---

<a id='2509.19282v1'></a>
## [OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps](https://arxiv.org/abs/2509.19282v1)

**Authors:** Bingnan Li, Chen-Yu Wang, Haiyang Xu, Xiang Zhang, Ethan Armand, Divyansh Srivastava, Xiaojun Shan, Zeyuan Chen, Jianwen Xie, Zhuowen Tu

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

Despite steady progress in layout-to-image generation, current methods still
struggle with layouts containing significant overlap between bounding boxes. We
identify two primary challenges: (1) large overlapping regions and (2)
overlapping instances with minimal semantic distinction. Through both
qualitative examples and quantitative analysis, we demonstrate how these
factors degrade generation quality. To systematically assess this issue, we
introduce OverLayScore, a novel metric that quantifies the complexity of
overlapping bounding boxes. Our analysis reveals that existing benchmarks are
biased toward simpler cases with low OverLayScore values, limiting their
effectiveness in evaluating model performance under more challenging
conditions. To bridge this gap, we present OverLayBench, a new benchmark
featuring high-quality annotations and a balanced distribution across different
levels of OverLayScore. As an initial step toward improving performance on
complex overlaps, we also propose CreatiLayout-AM, a model fine-tuned on a
curated amodal mask dataset. Together, our contributions lay the groundwork for
more robust layout-to-image generation under realistic and challenging
scenarios. Project link: https://mlpc-ucsd.github.io/OverLayBench.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供对Bingnan Li等人撰写的论文“OverLayBench: A Benchmark for Layout-to-Image Generation with Dense Overlaps”的全面摘要。

---

### 论文摘要：OverLayBench: 用于密集重叠布局到图像生成的基准

**1. 主要问题或研究问题：**
尽管布局到图像（L2I）生成取得了显著进展，但现有方法在处理包含显著重叠边界框的布局时仍面临挑战。具体来说，论文指出两个主要问题：(1) 大面积重叠区域，以及 (2) 语义区分度极低的重叠实例。这些因素导致生成图像的质量下降，表现为物体融合、空间模糊和视觉失真等伪影。因此，该研究旨在系统地评估和改进L2I模型在复杂重叠场景下的性能。

**2. 关键创新或方法论贡献：**
*   **OverLayScore：** 论文引入了一个新颖的度量标准OverLayScore，用于量化重叠边界框的复杂性。它通过计算所有实例对的IoU（交并比）之和，并根据CLIP嵌入的语义相似性进行加权，从而捕捉布局中空间和语义重叠的难度。高OverLayScore值表示生成难度更大。
*   **OverLayBench基准：** 为了解决现有基准偏向简单布局的问题，论文提出了OverLayBench，这是一个新的L2I生成基准。它具有高质量的标注，包括详细的图像和密集实例描述，以及跨OverLayScore不同难度级别（简单、常规、复杂）的平衡分布，旨在更严格地评估模型在复杂和重叠布局下的鲁棒性。
*   **CreatiLayout-AM模型：** 作为改进复杂重叠性能的初步尝试，论文提出了CreatiLayout-AM。这是一个在精心策划的非模态（amodal）掩码数据集上进行微调的模型，通过引入两个额外的损失项（Ltoken和Lpixel）来鼓励模型注意力图与非模态掩码之间的对齐，从而缓解实例遮挡引起的生成伪影。

**3. 主要结果及其意义：**
*   **OverLayScore的有效性：** 实验结果表明，随着OverLayScore的增加，现有L2I模型的生成质量（mIoU）持续下降，验证了OverLayScore能有效反映重叠布局的生成难度。
*   **现有基准的局限性：** 分析发现，COCO、LayoutSAM和HiCo等现有L2I基准在OverLayScore分布上存在严重偏差，大多数样本属于低难度范围，限制了它们在复杂场景下评估模型性能的有效性。
*   **OverLayBench的优势：** OverLayBench通过提供更平衡的难度分布，能够对L2I模型在空间和语义复杂布局下的鲁棒性进行更严格的评估。
*   **CreatiLayout-AM的性能提升：** CreatiLayout-AM在OverLayBench的简单和常规难度级别上优于原始CreatiLayout，特别是在O-mIoU（重叠区域mIoU）上取得了显著提升（+15.90%和+5.42%），验证了非模态掩码监督在改善重叠边界框下的L2I生成质量方面的有效性。在复杂难度级别上，其性能也保持了竞争力。
*   **错误模式分析：** 论文详细分析了现有L2I模型在重叠场景中常见的五种失败模式：不正确的物体数量、物体融合、物体失真、不正确的类别和边界框错位，为未来的研究指明了方向。

**4. 论文中提及的局限性：**
论文主要关注现有L2I模型在处理重叠布局时的不足，并提出了新的基准和初步解决方案。虽然CreatiLayout-AM在简单和常规重叠场景中表现良好，但在“复杂”重叠场景中，其性能提升相对较小，甚至在某些指标上略有下降。这表明，当训练集与测试集的分布差异较大时，模型仍面临挑战。此外，CreatiLayout-AM的非模态掩码监督方法是初步探索，可能还有进一步优化的空间。

**5. 潜在的未来研究方向：**
*   **更强大的非模态掩码监督：** 进一步探索更先进的非模态掩码监督方法，以在更复杂的重叠场景中实现更大的性能提升。
*   **提升模型对分布偏移的鲁棒性：** 研究如何使L2I模型在训练数据与实际复杂场景存在分布差异时，仍能保持稳定的生成质量。
*   **更精细的空间推理和组合理解：** 鼓励开发具有更强空间推理和组合理解能力的新方法，以更好地处理密集重叠场景中的物体交互。
*   **多模态融合：** 结合其他辅助模态（如深度图、边缘图）可能有助于模型更好地理解三维场景和物体遮挡关系。

---

这篇论文为布局到图像生成领域，特别是处理密集重叠场景，提供了一个重要的基准和新的研究方向。通过引入OverLayScore和OverLayBench，作者们为未来更鲁棒、更真实的图像生成奠定了基础。

**Key Findings:**

- Through both
qualitative examples and quantitative analysis, we demonstrate how these
factors degrade generation quality.
- To systematically assess this issue, we
introduce OverLayScore, a novel metric that quantifies the complexity of
overlapping bounding boxes.
- To bridge this gap, we present OverLayBench, a new benchmark
featuring high-quality annotations and a balanced distribution across different
levels of OverLayScore.
- Together, our contributions lay the groundwork for
more robust layout-to-image generation under realistic and challenging
scenarios.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19282v1)
- [arXiv](https://arxiv.org/abs/2509.19282v1)

---

<a id='2509.19252v1'></a>
## [Adversarially-Refined VQ-GAN with Dense Motion Tokenization for Spatio-Temporal Heatmaps](https://arxiv.org/abs/2509.19252v1)

**Authors:** Gabriel Maldonado, Narges Rashvand, Armin Danesh Pazho, Ghazal Alinezhad Noghre, Vinit Katariya, Hamed Tabkhi

**Published:** 2025-09-23

**Categories:** cs.CV, cs.AI

**Abstract:**

Continuous human motion understanding remains a core challenge in computer
vision due to its high dimensionality and inherent redundancy. Efficient
compression and representation are crucial for analyzing complex motion
dynamics. In this work, we introduce an adversarially-refined VQ-GAN framework
with dense motion tokenization for compressing spatio-temporal heatmaps while
preserving the fine-grained traces of human motion. Our approach combines dense
motion tokenization with adversarial refinement, which eliminates
reconstruction artifacts like motion smearing and temporal misalignment
observed in non-adversarial baselines. Our experiments on the CMU Panoptic
dataset provide conclusive evidence of our method's superiority, outperforming
the dVAE baseline by 9.31% SSIM and reducing temporal instability by 37.1%.
Furthermore, our dense tokenization strategy enables a novel analysis of motion
complexity, revealing that 2D motion can be optimally represented with a
compact 128-token vocabulary, while 3D motion's complexity demands a much
larger 1024-token codebook for faithful reconstruction. These results establish
practical deployment feasibility across diverse motion analysis applications.
The code base for this work is available at
https://github.com/TeCSAR-UNCC/Pose-Quantization.

**Analysis:**

好的，这是一篇关于“Adversarially-Refined VQ-GAN with Dense Motion Tokenization for Spatio-Temporal Heatmaps”论文的全面摘要，由Gabriel Maldonado等人撰写：

**论文摘要：对抗性精炼VQ-GAN与密集运动标记化用于时空热图**

**1. 主要问题或研究问题：**
该论文旨在解决计算机视觉领域中理解连续人体运动的核心挑战。人体运动数据具有高维度和固有的冗余性，这使得高效的压缩和表示对于分析复杂的运动动态至关重要。具体来说，现有方法在压缩时空热图时，往往难以在保持运动保真度（例如避免运动模糊和时间错位等重建伪影）的同时实现高效压缩。

**2. 关键创新或方法论贡献：**
作者引入了一个新颖的**对抗性精炼VQ-GAN框架**，并结合**密集运动标记化**来压缩时空热图。其主要创新点包括：
*   **对抗性精炼VQ-GAN框架：** 这是首个用于人体运动编码的VQ-GAN框架，它将2D和3D时空热图序列离散化为紧凑的潜在标记。通过引入对抗性训练目标，该框架能够确保时间连贯性，有效消除非对抗性基线中常见的运动模糊和错位伪影。
*   **密集运动标记化和分析：** 提出了一种新颖的密集标记化策略，能够捕获稀疏表示中常常丢失的细微运动模式。通过系统分析压缩因子和词汇量大小的作用，论文深入探讨了运动的内在复杂性。
*   **优越的压缩和保真度性能：** 该框架在运动压缩方面建立了新的技术水平，其对抗性增强的离散嵌入在重建质量和时间稳定性方面均优于dVAE模型。

**3. 主要结果及其意义：**
*   **性能优越性：** 在CMU Panoptic数据集上的实验表明，该方法在SSIM（结构相似性指数）方面比dVAE基线高出9.31%，并将时间不稳定性降低了37.1%。这证实了对抗性目标在消除时间伪影和保持运动精细轨迹方面的关键作用。
*   **运动复杂性分析：** 密集标记化策略揭示了2D运动可以通过紧凑的128个标记词汇表进行最佳表示，而3D运动的复杂性需要更大的1024个标记码本才能实现忠实重建。这一发现为设计高效、维度感知的压缩模型提供了原则性指导。
*   **实际部署可行性：** 论文结果表明，该框架在实际压缩率下（例如F16压缩时2D运动SSIM达到95.4%，3D运动SSIM达到91.2%）仍能保持高保真度，证明了使用离散标记化处理高要求运动分析任务的可行性。

**4. 论文中提及的局限性：**
论文承认，尽管其架构建立了新的技术水平，但更具表现力的编码器（如基于Transformer的设计）代表了未来增强的有前景的途径。这暗示了当前模型可能尚未充分利用所有先进的编码技术。

**5. 潜在的未来研究方向：**
*   **下游应用：** 模型生成的紧凑且语义丰富的运动标记可以作为未来下游任务（如动作分类、运动预测和异常检测）的基础骨干。通过用运动标记序列替换原始高维视频数据，分类器可以在更小但信息丰富的输入上运行，从而可能实现更快的推理和更好的泛化。
*   **异常检测：** 通过在大量正常人体行为数据集上进行预训练，学习到的运动词汇表可用于识别异常或不寻常的运动序列。
*   **更先进的编码器：** 探索将Transformer等更具表现力的编码器集成到VQ-GAN框架中，以进一步提升性能。

**Key Findings:**

- In this work, we introduce an adversarially-refined VQ-GAN framework
with dense motion tokenization for compressing spatio-temporal heatmaps while
preserving the fine-grained traces of human motion.
- Our approach combines dense
motion tokenization with adversarial refinement, which eliminates
reconstruction artifacts like motion smearing and temporal misalignment
observed in non-adversarial baselines.
- Our experiments on the CMU Panoptic
dataset provide conclusive evidence of our method's superiority, outperforming
the dVAE baseline by 9.31% SSIM and reducing temporal instability by 37.1%.
- Furthermore, our dense tokenization strategy enables a novel analysis of motion
complexity, revealing that 2D motion can be optimally represented with a
compact 128-token vocabulary, while 3D motion's complexity demands a much
larger 1024-token codebook for faithful reconstruction.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19252v1)
- [arXiv](https://arxiv.org/abs/2509.19252v1)

---

<a id='2509.19245v1'></a>
## [ConViS-Bench: Estimating Video Similarity Through Semantic Concepts](https://arxiv.org/abs/2509.19245v1)

**Authors:** Benedetta Liberatori, Alessandro Conti, Lorenzo Vaquero, Yiming Wang, Elisa Ricci, Paolo Rota

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

What does it mean for two videos to be similar? Videos may appear similar
when judged by the actions they depict, yet entirely different if evaluated
based on the locations where they were filmed. While humans naturally compare
videos by taking different aspects into account, this ability has not been
thoroughly studied and presents a challenge for models that often depend on
broad global similarity scores. Large Multimodal Models (LMMs) with video
understanding capabilities open new opportunities for leveraging natural
language in comparative video tasks. We introduce Concept-based Video
Similarity estimation (ConViS), a novel task that compares pairs of videos by
computing interpretable similarity scores across a predefined set of key
semantic concepts. ConViS allows for human-like reasoning about video
similarity and enables new applications such as concept-conditioned video
retrieval. To support this task, we also introduce ConViS-Bench, a new
benchmark comprising carefully annotated video pairs spanning multiple domains.
Each pair comes with concept-level similarity scores and textual descriptions
of both differences and similarities. Additionally, we benchmark several
state-of-the-art models on ConViS, providing insights into their alignment with
human judgments. Our results reveal significant performance differences on
ConViS, indicating that some concepts present greater challenges for estimating
video similarity. We believe that ConViS-Bench will serve as a valuable
resource for advancing research in language-driven video understanding.

**Analysis:**

以下是Benedetta Liberatori等人撰写的论文“ConViS-Bench: Estimating Video Similarity Through Semantic Concepts”的全面摘要：

**1. 论文主要解决的问题或研究问题**
该论文旨在解决视频相似性评估中长期存在的问题。传统方法通常依赖于宽泛的全局相似度分数，这使得模型难以像人类一样根据视频的不同语义概念（如动作、地点、主体等）进行细致的比较。具体来说，论文提出了一个核心问题：如何量化视频在特定语义概念上的相似性，并为视频理解模型提供一个能够像人类一样进行概念化推理的基准。

**2. 主要创新点或方法学贡献**
*   **引入ConViS任务：** 论文提出了“基于概念的视频相似性估计”（ConViS）这一新任务。它超越了传统的全局评分，通过计算跨预定义语义概念（如主要动作、主体、物体、地点和动作顺序）的可解释相似度分数来比较视频对。这种方法允许用户根据自然语言定义的语义维度进行灵活的比较，并可聚合为整体相似度分数。
*   **发布ConViS-Bench基准数据集：** 为了支持ConViS任务，论文构建并发布了ConViS-Bench，这是一个包含610对视频对的新基准数据集。这些视频对经过人工标注，包含概念级别的相似度分数（1到5分）以及对相似点和差异点的自由文本描述。该数据集涵盖了16个不同的领域，比现有基准更广泛，并且视频平均时长更长，提供了丰富的语义和视觉多样性。
*   **对LMMs进行广泛基准测试：** 论文对多种最先进的大型多模态模型（LMMs）在ConViS任务上的性能进行了评估，包括mPLUG-Owl3、LLaVA系列、Qwen-VL系列、InternVL系列和Gemini 2.0-Flash。这提供了关于这些模型与人类判断一致性的深入见解，并揭示了它们在概念理解方面的优势和局限性。

**3. 主要结果及其意义**
*   **LMMs性能差异显著：** 评估结果显示，LMMs在ConViS任务上表现出显著的性能差异。较大的模型通常优于较小的模型，其中LLaVA-OV-7B在整体相关性方面表现最佳。
*   **概念挑战性：** 某些概念（如“动作顺序”）对所有模型来说都更具挑战性，而其他概念（如“主要物体”或“地点”）则相对容易。这表明现有模型在建模空间上下文和时间结构方面存在不足。
*   **时间上下文的重要性：** 未在FineVideo上预训练的模型（如LLaVA-OV/LLaVA-Video/Qwen2.5-VL）在输入帧数减少时，性能显著下降，表明它们对丰富的时间上下文的依赖性。而InternVL系列模型（在FineVideo上预训练过）表现出相对平稳的性能趋势，可能存在记忆效应。
*   **全局表示的局限性：** 对计算全局视频相似度分数的模型进行分析发现，视频到视频的方法更侧重于“地点”概念，而文本到文本的方法则擅长捕获与“动作”相关的概念。跨模态方法（如VQAScore）在所有概念上取得了最高的平均相关性，但没有单一模型在所有概念上都表现最佳。
*   **概念条件视频检索：** 在概念条件视频检索任务中，LMMs的R@1分数普遍高于P@1，表明它们在检索相关视频方面表现良好，但也产生较多的假阳性。虽然LMMs表现出一定的能力，但概念条件视频到视频检索仍是其局限性。

**4. 论文中提及的局限性**
*   **概念集的范围：** 当前的ConViS概念集是通用性的，可能无法捕获所有特定领域或细粒度的视频相似性方面。
*   **数据集规模：** ConViS-Bench目前包含相对适度的视频对数量。虽然论文优先考虑了标注质量而非数量，但规模的限制可能会影响更广泛的评估。

**5. 潜在的未来研究方向**
*   **扩展概念分类：** 将ConViS框架扩展到包含更多特定领域或细粒度的概念，以实现更丰富的专业化分析。
*   **扩充数据集：** 在保持质量标准的前提下，扩充ConViS-Bench数据集的规模，以支持更广泛的评估和研究。
*   **改进模型的时间理解：** 进一步研究和开发能够更好地理解视频时间结构和动作顺序的模型。
*   **开发更具解释性和可控性的视频理解系统：** ConViS为构建更具解释性、可控性和用户对齐的视频理解系统提供了途径。

**Key Findings:**

- Large Multimodal Models (LMMs) with video
understanding capabilities open new opportunities for leveraging natural
language in comparative video tasks.
- We introduce Concept-based Video
Similarity estimation (ConViS), a novel task that compares pairs of videos by
computing interpretable similarity scores across a predefined set of key
semantic concepts.
- ConViS allows for human-like reasoning about video
similarity and enables new applications such as concept-conditioned video
retrieval.
- To support this task, we also introduce ConViS-Bench, a new
benchmark comprising carefully annotated video pairs spanning multiple domains.
- Additionally, we benchmark several
state-of-the-art models on ConViS, providing insights into their alignment with
human judgments.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19245v1)
- [arXiv](https://arxiv.org/abs/2509.19245v1)

---

<a id='2509.19203v1'></a>
## [Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions](https://arxiv.org/abs/2509.19203v1)

**Authors:** Ioanna Ntinou, Alexandros Xenos, Yassine Ouali, Adrian Bulat, Georgios Tzimiropoulos

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

Contrastively-trained Vision-Language Models (VLMs), such as CLIP, have
become the standard approach for learning discriminative vision-language
representations. However, these models often exhibit shallow language
understanding, manifesting bag-of-words behaviour. These limitations are
reinforced by their dual-encoder design, which induces a modality gap.
Additionally, the reliance on vast web-collected data corpora for training
makes the process computationally expensive and introduces significant privacy
concerns. To address these limitations, in this work, we challenge the
necessity of vision encoders for retrieval tasks by introducing a vision-free,
single-encoder retrieval pipeline. Departing from the traditional text-to-image
retrieval paradigm, we migrate to a text-to-text paradigm with the assistance
of VLLM-generated structured image descriptions. We demonstrate that this
paradigm shift has significant advantages, including a substantial reduction of
the modality gap, improved compositionality, and better performance on short
and long caption queries, all attainable with only a few hours of calibration
on two GPUs. Additionally, substituting raw images with textual descriptions
introduces a more privacy-friendly alternative for retrieval. To further assess
generalisation and address some of the shortcomings of prior compositionality
benchmarks, we release two benchmarks derived from Flickr30k and COCO,
containing diverse compositional queries made of short captions, which we coin
subFlickr and subCOCO. Our vision-free retriever matches and often surpasses
traditional multimodal models. Importantly, our approach achieves
state-of-the-art zero-shot performance on multiple retrieval and
compositionality benchmarks, with models as small as 0.3B parameters. Code is
available at: https://github.com/IoannaNti/LexiCLIP

**Analysis:**

好的，这是一篇关于“Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions”论文的全面摘要，由Ioanna Ntinou, Alexandros Xenos, Yassine Ouali, Adrian Bulat和Georgios Tzimiropoulos撰写。

**论文摘要：Vision-Free Retrieval: Rethinking Multimodal Search with Textual Scene Descriptions**

**1. 主要问题或研究问题：**
该论文旨在解决当前对比训练的视觉-语言模型（VLMs，如CLIP）在检索任务中存在的几个关键局限性。这些问题包括：
*   **浅层语言理解和“词袋”行为：** VLMs往往缺乏对语言结构的深入理解，导致其在处理复杂或组合性查询时表现不佳。
*   **模态鸿沟（Modality Gap）：** 双编码器设计（图像和文本分别编码）在不同模态的嵌入空间之间造成了不一致性，影响了模型的公平性和组合能力。
*   **计算成本和隐私问题：** 训练VLMs需要大量网络收集的数据，这不仅计算昂贵，还引发了显著的隐私担忧。
*   **现有基准的不足：** 现有的文本-图像检索基准（如Flickr30k和COCO）主要依赖长句描述，未能充分评估模型在短的、组合性查询上的性能。

为了解决这些问题，论文挑战了在检索任务中视觉编码器的必要性，并提出了一种新的“无视觉”检索范式。

**2. 关键创新或方法论贡献：**
该论文的核心创新在于引入了一个名为**LexiCLIP**的“无视觉、单编码器检索管道”，其主要贡献包括：
*   **范式转变：从文本-图像到文本-文本检索：** 论文放弃了传统的文本-图像检索，转而采用文本-文本范式。通过大型视觉-语言模型（VLLM）生成结构化的图像描述，将图像内容完全转换为文本，从而使语言模型能够纯粹通过文本推理视觉内容。
*   **图像到文本转换管道：** 提出了一种鲁棒、有原则且精心设计的管道，用于将丰富的视觉信息准确地转换为文本描述。这包括生成详细的场景描述和结构化的对象注释（包含对象类别、属性、动作、位置等），并采用JSON格式以确保输出的连贯性和组织性。
*   **单编码器架构：** 通过将图像转换为文本，LexiCLIP能够利用共享的单编码器架构，显著减少了模态鸿沟，并利用预训练语言模型的丰富语言知识。
*   **轻量级校准：** 该方法仅需在少量GPU上进行数小时的校准，即可实现显著性能提升，避免了昂贵的从头开始训练。
*   **隐私友好：** 用文本描述替代原始图像，提供了一种更注重隐私的检索替代方案，因为大多数身份相关信息（如面部、私人房间）在转换过程中被移除。
*   **新基准数据集：** 为了更全面地评估模型在短的、组合性查询上的泛化能力，论文发布了两个新的基准数据集：**subFlickr**和**subCOCO**，它们从Flickr30k和COCO中提取，包含多样化的短组合性查询。

**3. 主要结果及其意义：**
*   **模态鸿沟显著减少：** LexiCLIP通过其单编码器设计，显著缩小了文本和图像嵌入之间的模态鸿沟，提高了模型在跨模态检索中的对齐能力。
*   **改进的组合性和性能：** 在短和长标题查询上，LexiCLIP表现出更好的组合性和性能。在SugarCrepe和SugarCrepe++等组合性基准测试中，即使是0.3B参数的小模型，也达到了最先进的零样本性能，甚至超越了传统的、参数量更大的多模态模型。
*   **零样本和微调性能：** 在Flickr30k和COCO等标准基准上，LexiCLIP在零样本设置下表现出色，经过少量文本数据微调后，性能进一步提升，甚至超越了OpenCLIP等大型模型。
*   **长文本检索能力：** 利用预训练语言模型处理通用文本的能力，LexiCLIP在长文本检索任务（如Urbanlk数据集）上表现优异，超越了所有其他CLIP变体和专门为长标题微调的模型。
*   **对VLLM选择的鲁棒性：** 论文分析了不同VLLM架构和大小对性能的影响，发现InternVL-2.5-8B-MPO表现最佳，且LexiCLIP对图像描述生成器的规模具有鲁棒性，即使是较小的1B模型也能实现高效性能。

**4. 论文中提到的局限性：**
*   **对VLLM的强依赖：** 该方法严重依赖VLLM生成图像描述，可能导致某些视觉细节（尤其是拥挤场景或小物体）的丢失。
*   **VLLM偏差和幻觉的继承：** VLLM可能存在偏差或幻觉，生成的描述可能会继承这些错误，并传播到检索过程中。
*   **额外的计算开销：** 使用VLLM作为图像描述器引入了额外的计算开销（尽管这一步骤是离线执行的，且检索本身仍然高效）。

**5. 潜在的未来研究方向：**
*   **缓解VLLM局限性：** 未来的工作将尝试采用过滤方法、集成多个描述器（ensemble captioners），甚至引入人工验证，以减轻VLLM的偏差和幻觉问题。
*   **优化计算效率：** 通过优化实现（如v1lm项目）和量化技术，进一步降低VLLM生成图像描述的预处理成本。
*   **探索更小的生成器：** 论文指出，使用较小的生成器（如2B InternVL）也能获得相似的检索性能，这为降低预处理成本提供了方向。

总而言之，这篇论文通过引入无视觉、单编码器的LexiCLIP框架，对多模态检索领域提出了根本性的挑战和创新。它不仅解决了现有VLMs的模态鸿沟、浅层语言理解和隐私问题，还在多个检索和组合性基准上实现了最先进的性能，为未来更高效、更隐私、更具组合性的多模态搜索奠定了基础。

**Key Findings:**

- We demonstrate that this
paradigm shift has significant advantages, including a substantial reduction of
the modality gap, improved compositionality, and better performance on short
and long caption queries, all attainable with only a few hours of calibration
on two GPUs. Additionally, substituting raw images with textual descriptions
introduces a more privacy-friendly alternative for retrieval.
- Importantly, our approach achieves
state-of-the-art zero-shot performance on multiple retrieval and
compositionality benchmarks, with models as small as 0.3B parameters.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19203v1)
- [arXiv](https://arxiv.org/abs/2509.19203v1)

---

<a id='2509.19165v1'></a>
## [RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions](https://arxiv.org/abs/2509.19165v1)

**Authors:** Yun Wang, Junjie Hu, Junhui Hou, Chenghao Zhang, Renwei Yang, Dapeng Oliver Wu

**Published:** 2025-09-23

**Categories:** cs.CV, cs.AI

**Abstract:**

Recent self-supervised stereo matching methods have made significant
progress, but their performance significantly degrades under adverse weather
conditions such as night, rain, and fog. We identify two primary weaknesses
contributing to this performance degradation. First, adverse weather introduces
noise and reduces visibility, making CNN-based feature extractors struggle with
degraded regions like reflective and textureless areas. Second, these degraded
regions can disrupt accurate pixel correspondences, leading to ineffective
supervision based on the photometric consistency assumption. To address these
challenges, we propose injecting robust priors derived from the visual
foundation model into the CNN-based feature extractor to improve feature
representation under adverse weather conditions. We then introduce scene
correspondence priors to construct robust supervisory signals rather than
relying solely on the photometric consistency assumption. Specifically, we
create synthetic stereo datasets with realistic weather degradations. These
datasets feature clear and adverse image pairs that maintain the same semantic
context and disparity, preserving the scene correspondence property. With this
knowledge, we propose a robust self-supervised training paradigm, consisting of
two key steps: robust self-supervised scene correspondence learning and adverse
weather distillation. Both steps aim to align underlying scene results from
clean and adverse image pairs, thus improving model disparity estimation under
adverse weather effects. Extensive experiments demonstrate the effectiveness
and versatility of our proposed solution, which outperforms existing
state-of-the-art self-supervised methods. Codes are available at
\textcolor{blue}{https://github.com/cocowy1/RoSe-Robust-Self-supervised-Stereo-Matching-under-Adverse-Weather-Conditions}.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Yun Wang等人撰写的论文“RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions”的全面摘要。

---

**论文题目：** RoSe: Robust Self-supervised Stereo Matching under Adverse Weather Conditions

**作者：** Yun Wang, Junjie Hu, Junhui Hou, Chenghao Zhang, Renwei Yang, Dapeng Oliver Wu

**摘要：**

1.  **主要问题或研究问题：**
    该论文旨在解决自监督立体匹配方法在恶劣天气条件（如夜晚、雨天和雾天）下性能显著下降的问题。作者识别出导致性能下降的两个主要弱点：
    *   恶劣天气引入的噪声和能见度降低，使得基于CNN的特征提取器难以处理反射和无纹理区域等退化区域。
    *   这些退化区域会破坏准确的像素对应关系，导致基于光度一致性假设的监督信号失效。

2.  **关键创新或方法论贡献：**
    为了解决上述挑战，作者提出了RoSe（Robust Self-supervised Stereo Matching），一个两阶段的自监督训练范式，其核心创新包括：
    *   **鲁棒特征提取器：** 提出将视觉基础模型（如SAM和DAMv2）的鲁棒先验注入到CNN-based特征提取器中，以改善恶劣天气下的特征表示。此外，引入了**抗恶劣天气特征增强模块（AFEM）**，通过在空间、通道和频率域处理，有效分离与退化相关的噪声和有意义的场景特征，生成降级不变的特征。
    *   **场景对应先验构建鲁棒监督信号：** 不再仅仅依赖光度一致性假设，而是引入场景对应先验来构建鲁棒的监督信号。具体做法是：
        *   创建合成立体数据集，包含具有真实天气退化的清晰和恶劣图像对，这些图像对保持相同的语义上下文和视差，从而保留场景对应属性。
        *   **两阶段自监督训练范式：**
            *   **第一步：鲁棒自监督场景对应学习：** 引入两个分支分别处理清晰和恶劣天气条件下的图像对。利用场景对应先验，提出了**特征一致性损失（Feature Consistency Loss）**和**视差一致性损失（Disparity Consistency Loss）**，确保清晰和恶劣天气条件下学习到的特征和视差值保持一致，从而学习到受天气退化影响较小的潜在空间。
            *   **第二步：恶劣天气蒸馏：** 利用第一步中冻结模型生成的清晰图像对的高质量伪标签作为监督信号，有效缓解光度一致性假设在遮挡区域等病态区域的失效问题。

3.  **主要结果及其意义：**
    *   广泛的实验证明了RoSe解决方案的有效性和通用性。在DrivingStereo和MS2等包含恶劣天气条件的数据集上，RoSe的性能优于现有的最先进自监督方法。
    *   在KITTI 2012和KITTI 2015基准测试上，RoSe也取得了极具竞争力的性能，并且在标准和挑战性条件下，其框架显著优于之前的自监督工作。
    *   定性和定量结果表明，RoSe在各种恶劣天气条件下（如夜间、雨天、雾天）都能进行鲁棒的视差估计，尤其在挑战性区域表现出色。
    *   模型效率方面，RoSe在内存消耗和推理时间上表现合理。

4.  **论文中提及的局限性：**
    *   RoSe高度依赖于图像到图像翻译模型生成的合成恶劣天气对的真实性和一致性。翻译中的不准确或不一致（如几何失真或语义不匹配）可能向监督信号引入噪声，从而可能降低整体性能。
    *   在严重的图像退化条件下，RoSe可能会产生不可靠的视差估计，这在自动驾驶等安全关键应用中构成风险，因为准确的深度感知对于导航和避障至关重要。

5.  **潜在的未来研究方向：**
    论文中没有明确提出未来的研究方向，但从其局限性可以推断出：
    *   进一步提升图像到图像翻译模型的真实性和一致性，以生成更高质量的恶劣天气合成数据。
    *   研究在极端恶劣图像退化条件下提高视差估计可靠性的方法，以满足安全关键应用的需求。
    *   探索如何将RoSe的鲁棒性扩展到更多样化的恶劣天气类型或更复杂的场景。

---

总而言之，RoSe论文通过引入视觉基础模型的鲁棒先验和场景对应先验，并设计了一个两阶段的自监督训练范式，成功解决了自监督立体匹配在恶劣天气下性能下降的问题。其核心贡献在于通过降级不变特征提取和鲁棒监督信号构建，显著提升了模型在挑战性环境中的鲁棒性和泛化能力，为自动驾驶等实际应用提供了更可靠的深度感知解决方案。

**Key Findings:**

- To address these
challenges, we propose injecting robust priors derived from the visual
foundation model into the CNN-based feature extractor to improve feature
representation under adverse weather conditions.
- With this
knowledge, we propose a robust self-supervised training paradigm, consisting of
two key steps: robust self-supervised scene correspondence learning and adverse
weather distillation.
- Extensive experiments demonstrate the effectiveness
and versatility of our proposed solution, which outperforms existing
state-of-the-art self-supervised methods.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19165v1)
- [arXiv](https://arxiv.org/abs/2509.19165v1)

---

<a id='2509.19129v1'></a>
## [KAMERA: Enhancing Aerial Surveys of Ice-associated Seals in Arctic Environments](https://arxiv.org/abs/2509.19129v1)

**Authors:** Adam Romlein, Benjamin X. Hou, Yuval Boss, Cynthia L. Christman, Stacie Koslovsky, Erin E. Moreland, Jason Parham, Anthony Hoogs

**Published:** 2025-09-23

**Categories:** cs.CV

**Abstract:**

We introduce KAMERA: a comprehensive system for multi-camera, multi-spectral
synchronization and real-time detection of seals and polar bears. Utilized in
aerial surveys for ice-associated seals in the Bering, Chukchi, and Beaufort
seas around Alaska, KAMERA provides up to an 80% reduction in dataset
processing time over previous methods. Our rigorous calibration and hardware
synchronization enable using multiple spectra for object detection. All
collected data are annotated with metadata so they can be easily referenced
later. All imagery and animal detections from a survey are mapped onto a world
plane for accurate surveyed area estimates and quick assessment of survey
results. We hope KAMERA will inspire other mapping and detection efforts in the
scientific community, with all software, models, and schematics fully
open-sourced.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Adam Romlein等人撰写的论文“KAMERA: Enhancing Aerial Surveys of Ice-associated Seals in Arctic Environments”的全面摘要。

---

**论文摘要：KAMERA: 提升北极环境中冰相关海豹航空测量的效率**

这篇论文介绍了KAMERA（Knowledge-guided Image Acquisition ManagER and Archiver），一个旨在显著提升北极地区冰相关海豹和北极熊航空测量效率和准确性的综合系统。

**1. 主要问题或研究问题：**
传统的冰相关海豹航空测量方法面临多重挑战：数据处理时间长（通常需要数月甚至数年），系统组件（硬件和软件）之间耦合松散导致数据同步和对齐困难，以及专有软件限制了科学社区的广泛应用。这些问题阻碍了对海豹种群的及时监测和管理，尤其是在海冰栖息地不断变化的背景下。

**2. 关键创新或方法论贡献：**
KAMERA系统通过以下创新解决了上述问题：
*   **多相机、多光谱同步：** KAMERA实现了多相机、多光谱图像采集的精确同步，所有数据都通过单一外部时间脉冲收集，并聚合到带有详细元数据的一个存储位置。这使得能够利用不同光谱（RGB、IR、UV）的数据进行更准确的物体检测。
*   **实时检测：** 系统利用机载GPU对同步图像进行实时分析，能够实时识别海豹和北极熊，并根据检测结果决定是否存档图像，从而大幅减少了需要存储和处理的空白图像数据量。
*   **精确测绘：** 所有图像和动物检测结果都被映射到世界平面上，以实现精确的测量区域估算和快速评估测量结果。这通过严格的相机校准（包括内参和外参）以及与惯性导航系统（INS）的对齐来实现。
*   **开放源代码：** KAMERA的所有软件、模型和原理图都是完全开源的（Apache License和CC BY 4.0），促进了科学社区的协作和复用。
*   **两阶段检测管线（针对2021年测量）：** 结合红外（IR）热点检测和彩色图像中的物种分类模型。IR模型首先识别热点，然后裁剪出相应的高分辨率彩色图像区域，再由物种特异性彩色检测器进行分类。

**3. 主要结果及其意义：**
*   **数据处理时间大幅减少：** KAMERA系统将数据集处理时间比以前的方法减少了高达80%，例如，与2016年楚科奇海的测量相比，2021年南博福特海的测量时间从6个月缩短到5周。
*   **高精度检测模型：** 论文展示了IR热点检测模型、两种海豹物种分类模型和北极熊检测模型的验证结果。IR热点检测和海豹模型在召回率上表现出色（均达到0.93或更高），这对于数据收集决策至关重要。
*   **实际应用：** KAMERA已成功应用于阿拉斯加白令海、楚科奇海和博福特海的航空测量，收集了数百万图像样本。
*   **用户友好界面：** 开发了图形用户界面（GUI），支持实时相机控制、系统监控和飞行后数据评估，即使非技术用户也能有效操作。

**4. 论文中提及的局限性：**
*   **校准漂移：** 相机校准会随时间漂移，需要定期检查和调整。
*   **SIFT特征的局限性：** 在多光谱匹配中，SIFT特征在稀疏特征区域和不同光谱之间匹配效果不佳。
*   **模型泛化能力：** 2025年测量的初步评估显示，IR热点模型在新热像仪和更新的NUC校准下性能有所下降，表明模型在不同硬件和图像域之间的泛化能力有待提高。
*   **数据不平衡：** 训练数据中海豹类别之间以及与北极熊标签之间存在严重不平衡，需要采用Focal Loss等技术来解决。
*   **UV模型开发：** 目前尚未训练UV模型，仍在开发中，主要用于北极熊和白色幼海豹的检测。

**5. 潜在的未来研究方向：**
*   **改进校准方法：** 探索基于深度学习的特征和关键点（如SuperPoint、SuperGlue和ALIKED）来改进多光谱校准。
*   **增强模型鲁棒性：** 进一步分析和改进模型，以提高其在不同硬件和图像域下的泛化能力，例如利用2025年收集的数据训练新的IR-RGB模型，并探索YOLOv9等更现代的架构。
*   **UV模型开发：** 完成并部署用于北极熊和白色幼海豹检测的UV模型。
*   **更广泛的应用：** 将KAMERA系统推广到其他野生动物调查，例如驼鹿、大象或海龟。

---

总而言之，KAMERA系统代表了航空测量领域的一个重大进步，通过其多相机、多光谱同步、实时检测和精确测绘能力，显著提升了数据收集和分析的效率与准确性。其开放源代码的特性有望推动科学社区在野生动物监测和保护方面的进一步创新。

**Key Findings:**

- We introduce KAMERA: a comprehensive system for multi-camera, multi-spectral
synchronization and real-time detection of seals and polar bears.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.19129v1)
- [arXiv](https://arxiv.org/abs/2509.19129v1)

---

