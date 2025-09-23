time: 20250923

# Arxiv Computer Vision Papers - 2025-09-23

## Executive Summary

## Arxiv 计算机视觉与机器学习日报执行摘要 (2025-09-22)

**概述：**

今日 Arxiv 计算机视觉与机器学习领域的论文呈现出多模态大模型（尤其是视频-语言模型）、3D 几何重建、可控生成以及机器人感知与导航等多个活跃方向。特别值得关注的是，多模态模型在处理视频、音频和文本信息方面的能力持续增强，并开始探索更精细的控制和应用。

**主要主题与趋势：**

1.  **多模态大模型与视频理解 (V-LLMs)：** 多篇论文聚焦于视频-语言模型 (V-LLMs) 的发展，包括其架构、训练策略（如时间采样）以及对音频模态的考量。这表明 V-LLMs 仍是当前研究热点，旨在实现更全面、更鲁棒的视频内容理解。
2.  **可控生成与编辑：** 在图像和视频生成领域，研究人员正致力于提升生成内容的精细控制能力，例如通过属性特定提示进行人物图像生成，以及为图像到视频生成提供鲁棒的水印方案。
3.  **3D 几何重建：** 稀疏体素在几何精确表面重建中的应用是一个重要方向，旨在克服传统方法的局限性，实现更高质量的3D模型。
4.  **机器人感知与导航：** 机器人领域的研究侧重于在动态环境中实现指令遵循的导航，并结合感知能力优化机器人检查任务的效率。
5.  **模型效率与鲁棒性：** 论文也关注模型合并的效率和准确性，以及生成模型的水印技术，体现了对模型实用性和安全性的考量。

**特别显著或创新论文：**

*   **"Qwen3-Omni Technical Report" (Jin Xu et al.):** 作为一份技术报告，它通常会介绍一个大型、多功能的模型，预示着未来多模态大模型的发展方向和能力边界。如果 Qwen3-Omni 像其前身一样具有广泛影响力，这份报告将是理解其核心技术和潜力的关键。
*   **"TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs" (Yunheng Li et al.):** 该论文通过强化学习微调来优化视频 LLMs 的时间采样策略，解决了视频理解中的一个核心挑战，即如何高效地从长视频中提取关键信息。这种结合强化学习的优化方法具有创新性。
*   **"ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion" (Zichao Hu et al.):** 将可组合扩散模型应用于动态环境中的指令遵循导航，为机器人导航提供了一种新颖且可能更鲁棒的解决方案，尤其是在复杂、不可预测的场景中。

**新兴研究方向或技术：**

*   **强化学习在多模态模型优化中的应用：** "TempSamp-R1" 展示了强化学习在优化 V-LLMs 内部机制（如时间采样）方面的潜力。
*   **可组合扩散模型在机器人控制中的应用：** "ComposableNav" 探索了扩散模型在生成复杂、多步骤行为序列方面的能力，为机器人任务规划和执行开辟了新途径。
*   **多模态模型中的音频模态重要性再评估：** "Does Audio Matter for Modern Video-LLMs and Their Benchmarks?" 提出了对现有 V-LLMs 和基准中音频作用的深入思考，这可能引导未来模型设计更加全面地整合音频信息。

**建议阅读全文的论文：**

1.  **"Qwen3-Omni Technical Report" (Jin Xu et al.):** 对于关注多模态大模型最新进展和未来趋势的研究人员，这份报告是必读的，它将提供一个全面且前沿的视角。
2.  **"TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs" (Yunheng Li et al.):** 对于从事视频理解和 V-LLMs 优化的研究人员，该论文提供了一种创新的方法来解决时间采样效率问题，具有很高的参考价值。
3.  **"ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion" (Zichao Hu et al.):** 对于机器人学和具身智能领域的研究人员，该论文展示了将生成模型应用于复杂导航任务的潜力，值得深入研究。
4.  **"ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation" (Guocheng Gordon Qian et al.):** 对于关注可控图像生成和人像合成的研究人员，该论文提供了一种精细控制生成内容的方法，具有实际应用价值。

这份摘要旨在帮助您快速把握今日 Arxiv 计算机视觉与机器学习领域的关键发展，并为进一步深入阅读提供指导。

---

## Table of Contents

1. [Qwen3-Omni Technical Report](#2509.17765v1)
2. [Automatic Intermodal Loading Unit Identification using Computer Vision: A Scoping Review](#2509.17707v1)
3. [ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation](#2509.18092v1)
4. [GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction](#2509.18090v1)
5. [TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](#2509.18056v1)
6. [ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion](#2509.17941v1)
7. [Does Audio Matter for Modern Video-LLMs and Their Benchmarks?](#2509.17901v1)
8. [Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection](#2509.17877v1)
9. [Accurate and Efficient Low-Rank Model Merging in Core Space](#2509.17786v1)
10. [I2VWM: Robust Watermarking for Image to Video Generation](#2509.17773v1)

---

## Papers

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

**Qwen3-Omni 技术报告摘要**

**1. 解决的主要问题或研究问题：**
该论文旨在解决当前多模态模型普遍存在的模态间性能退化问题，即在提升某一模态性能时，其他模态的性能会随之下降。研究目标是开发一个统一的多模态模型（Qwen3-Omni），使其在文本、图像、音频和视频等多种模态上都能保持最先进的性能，且不发生任何性能退化，同时显著增强跨模态能力，如视频理解。

**2. 关键创新或方法论贡献：**
*   **Thinker-Talker MoE 架构：** Qwen3-Omni 采用 Thinker-Talker 混合专家（MoE）架构，统一了文本、图像、音频和视频的感知与生成，实现了流畅的文本和自然的实时语音。Thinker 和 Talker 都升级为 MoE 设计，以支持高并发和快速推理。
*   **AuT 音频编码器：** 替换了 Whisper 音频编码器，采用从零开始训练的 AuT（Audio Transformer）编码器，在 2000 万小时的监督音频数据上训练，生成更强的通用音频表示，并采用块式窗口注意力实现实时预填充缓存。
*   **多码本语音生成：** 在语音生成方面，采用多码本表示，通过 MTP 模块自回归预测多个码本层，并用轻量级因果 ConvNet 替换计算密集型的块式扩散模型，实现从第一个码本帧开始的流式合成，显著降低了首包延迟。
*   **Thinking 模型：** 引入了一个 Thinking 模型，明确地对来自任何模态的输入进行推理，以增强多模态推理能力。
*   **非退化多模态训练：** 论文提出并验证了在文本预训练早期阶段混合单模态和跨模态数据，可以实现所有模态的性能均等，同时显著增强跨模态能力。
*   **音频字幕模型：** 针对通用音频字幕模型的缺失，微调了 Qwen3-Omni-30B-A3B，得到了 Qwen3-Omni-30B-A3B-Captioner，能为任意音频输入生成详细、低幻觉的字幕。

**3. 主要结果及其意义：**
*   **SOTA 性能：** Qwen3-Omni 在 36 个音频和音视频基准测试中，在 32 个基准测试上实现了开源 SOTA，并在 22 个基准测试上达到了总体 SOTA，超越了 Gemini-2.5-Pro、Seed-ASR 和 GPT-4o-Transcribe 等强大的闭源模型。
*   **无模态退化：** 首次证明了该模型在文本、图像、音频和视频上均保持了与同等规模单模态模型相当的先进性能，没有出现模态退化。
*   **音频任务优势：** 在音频任务上表现尤为出色，支持 119 种语言的文本交互、19 种语言的语音理解和 10 种语言的语音生成。
*   **低延迟流式合成：** 在冷启动设置下，实现了 234 毫秒的理论端到端首包延迟，确保了高并发工业部署中的低延迟语音交互。
*   **跨模态推理增强：** Thinking 模型和多模态训练显著提升了模型在复杂推理任务上的表现，尤其是在需要整合音频和视觉信息的场景中。

**4. 论文中提及的局限性：**
*   **长视频基准测试性能次优：** 当前模型在长视频基准测试上的性能次优，这源于两个架构限制：位置外推能力有限和上下文长度受限。
*   **语言能力提升不明显：** 经验表明，添加视觉或音频信号并未在语言能力方面带来可衡量的提升。

**5. 潜在的未来研究方向：**
*   **多说话人 ASR：** 进一步支持多说话人自动语音识别。
*   **视频 OCR：** 增强视频中的光学字符识别能力。
*   **音视频主动学习：** 探索音视频领域的主动学习。
*   **基于代理的工作流和函数调用：** 增强对基于代理的工作流和函数调用的支持。
*   **解决长视频性能限制：** 解决当前模型在长视频理解方面的架构限制，提升其在该领域的性能。

总而言之，Qwen3-Omni 代表了多模态模型发展的一个里程碑，首次提供了全面集成、端到端的多模态训练可以在不降低核心语言能力和其他模态性能的情况下实现，同时在音频任务上表现卓越，并显著增强了跨模态推理能力。

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

**论文题目：** 使用计算机视觉的联运装载单元自动识别：一项范围审查

**1. 主要问题或研究问题：**
该论文旨在解决联运装载单元（ILUs，如集装箱、半挂车和可换箱体）在港口和码头进行高效、鲁棒识别的关键瓶颈。尽管ILUs的标准化彻底改变了全球贸易，但其识别过程仍面临挑战。该研究通过对计算机视觉（CV）领域现有解决方案的全面审查，探讨了如何利用CV技术克服这些挑战，并识别该领域的发展趋势、局限性及未来方向。

**2. 关键创新或方法论贡献：**
这篇论文本身是一项范围审查，其主要贡献在于对现有文献的系统性分析和综合，而非提出新的算法或模型。其关键创新和方法论贡献包括：
*   **首次全面审查：** 这是对基于CV的ILU识别研究的首次综合性审查，涵盖了1990年至2025年间的63项实证研究。
*   **领域演变追踪：** 详细追溯了该领域从早期数字图像处理（DIP）和传统机器学习（ML）到当前深度学习（DL）技术主导的演变过程。
*   **标准化术语的呼吁：** 强调了该领域术语不统一的问题，并呼吁采用标准化术语，例如“联运装载单元（ILU）识别”，以提高概念清晰度并促进跨学科研究。
*   **数据集和评估指标的综合分析：** 详细分析了现有研究中使用的图像采集设置、数据集特性（可用性、多样性）和评估指标，揭示了公共基准数据集的严重缺乏。
*   **识别新兴挑战：** 突出了从基于字符的文本识别转向场景文本识别（scene-text spotting）的趋势，以及移动摄像头（如无人机、车载传感器）在动态码头监控中的应用所带来的新挑战。

**3. 主要结果及其意义：**
*   **CV作为经济高效的替代方案：** 计算机视觉技术为ILU识别提供了比RFID等其他技术更具成本效益的替代方案，且准确性更高。
*   **DL技术的主导地位：** 研究趋势显示，深度学习方法已成为ILU识别的主导范式，尤其是在2016-2020年之后，其在处理复杂条件下的鲁棒性优于传统方法。
*   **亚洲研究的突出贡献：** 亚洲地区在ILU识别研究中占据主导地位（79.71%的论文来自亚洲），这反映了该地区作为全球贸易枢纽对高效物流的迫切需求和大量投资。
*   **公共资金支持：** 超过三分之一的研究得到公共资金支持，表明政府对交通基础设施和经济增长的重视。
*   **性能差异巨大：** 由于缺乏公开可用的基准数据集，报告的端到端准确率差异巨大，从5%到96%不等，这使得方法间的公平比较变得困难。
*   **从字符识别到场景文本识别的转变：** 随着移动摄像头的普及，任务已从简单的字符识别演变为更复杂的场景文本识别，需要更强大的检测和识别方法。

**4. 论文中提到的局限性：**
*   **缺乏公开可用的基准数据集：** 这是该领域研究和开发的最大障碍。绝大多数数据集（85.71%）是私有的，且许多研究依赖于受控的图像采集设置，导致模型泛化能力受限，并阻碍了结果的重现性和公平比较。
*   **术语不统一：** 现有文献中术语使用不一致，影响了研究的清晰度和可比性。
*   **DL训练数据不足：** 现有数据集的规模对于鲁棒的深度学习训练来说往往过小，迫使研究人员适应通用文本检测和识别模型，可能无法充分优化ISO6346代码的特定识别。
*   **对非主流期刊的依赖：** 超过一半的文章发表在未排名的期刊和会议上，这可能影响了该领域研究的可见度和影响力。
*   **上下文无关文本识别的挑战：** ISO6346代码的上下文无关性质使得场景文本识别任务更具挑战性，因为缺乏自然语言线索，现有依赖语义上下文的模型效果不佳。

**5. 潜在的未来研究方向：**
*   **标准化和开放获取数据集：** 呼吁建立公开可用的基准数据集，并采用标准化术语和评估方法，以促进研究成果的可比性和可重现性。
*   **上下文无关场景文本识别：** 针对ISO6346代码的上下文无关特性，开发专门优化的场景文本识别架构，避免依赖自然语言模型。
*   **实时处理和边缘设备优化：** 关注算法优化，以实现移动设备上的实时处理，并通过视频流分析增强系统性能。
*   **移动摄像头设置的重点研究：** 更多地关注车载摄像头（如无人机、传感器配备的地面车辆）的设置，以应对动态场景带来的新挑战，并探索ILU姿态估计等新任务，实现码头的精确监控。
*   **数据增强和合成数据生成：** 探索使用生成对抗网络（GANs）或游戏引擎等技术生成合成数据，以弥补真实数据集的不足，并模拟各种挑战性条件（如阴影、锈迹、污垢、运动模糊）。
*   **图像质量提升和后处理：** 继续研究图像增强技术（如降噪、超分辨率）和后处理步骤（如利用校验位和所有者代码验证识别结果），以提高识别准确性。
*   **统一的端到端模型：** 开发将文本检测和识别整合到单个前向传播中的统一端到端模型，以简化处理流程并提高整体效率。

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

<a id='2509.18092v1'></a>
## [ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation](https://arxiv.org/abs/2509.18092v1)

**Authors:** Guocheng Gordon Qian, Daniil Ostashev, Egor Nemchinov, Avihay Assouline, Sergey Tulyakov, Kuan-Chieh Jackson Wang, Kfir Aberman

**Published:** 2025-09-22

**Categories:** cs.CV

**Abstract:**

Generating high-fidelity images of humans with fine-grained control over
attributes such as hairstyle and clothing remains a core challenge in
personalized text-to-image synthesis. While prior methods emphasize identity
preservation from a reference image, they lack modularity and fail to provide
disentangled control over specific visual attributes. We introduce a new
paradigm for attribute-specific image prompting, in which distinct sets of
reference images are used to guide the generation of individual aspects of
human appearance, such as hair, clothing, and identity. Our method encodes
these inputs into attribute-specific tokens, which are injected into a
pre-trained text-to-image diffusion model. This enables compositional and
disentangled control over multiple visual factors, even across multiple people
within a single image. To promote natural composition and robust
disentanglement, we curate a cross-reference training dataset featuring
subjects in diverse poses and expressions, and propose a multi-attribute
cross-reference training strategy that encourages the model to generate
faithful outputs from misaligned attribute inputs while adhering to both
identity and textual conditioning. Extensive experiments show that our method
achieves state-of-the-art performance in accurately following both visual and
textual prompts. Our framework paves the way for more configurable human image
synthesis by combining visual prompting with text-driven generation. Webpage is
available at: https://snap-research.github.io/composeme/.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Guocheng Gordon Qian等人撰写的论文“ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation”的全面摘要。

---

**论文摘要：ComposeMe: Attribute-Specific Image Prompts for Controllable Human Image Generation**

**1. 主要问题或研究问题：**
该论文旨在解决个性化文本到图像合成领域的一个核心挑战：如何生成高保真的人类图像，并对发型、服装和身份等属性进行细粒度控制。现有方法虽然强调从参考图像中保留身份，但缺乏模块化，无法对特定视觉属性提供解耦控制，尤其是在需要组合来自不同来源的多个属性时。

**2. 关键创新或方法论贡献：**
ComposeMe引入了一种新颖的“属性特定图像提示”范式，其核心创新包括：

*   **属性特定图像提示（Attribute-Specific Image Prompts）：** 该方法将人类主体分解为可配置的属性（面部身份、发型和服装），并为每个属性使用不同的参考图像集进行引导生成。这使得能够组合来自不同来源的视觉属性，实现更灵活的控制。
*   **ComposeMe管线：** 采用基于适配器的解决方案，通过三个阶段实现：
    *   **属性特定标记化（Attribute-Specific Tokenization）：** 为每个视觉组件（面部、发型、服装）使用专用的特征标记器处理参考图像，捕获属性特定特征。
    *   **多属性合并（Multi-Attribute Merging）：** 将来自不同属性的标记合并，形成多属性主体表示。
    *   **注入预训练扩散模型：** 将合并后的标记注入冻结的预训练文本到图像扩散模型中，通过解耦的交叉注意力机制实现图像生成。
*   **多属性交叉引用训练（Multi-Attribute Cross-Reference Training）：** 提出了一种新颖的两阶段训练策略。在第一阶段进行复制粘贴预训练以预热适配器。第二阶段是多属性交叉引用微调，通过使用来自不同个体和姿态的输入和目标进行监督，明确解耦了属性之间的纠缠（例如，服装、面部和头发与身体姿态、表情或头部方向的纠缠）。这种训练鼓励模型从错位的属性输入生成忠实且自然对齐的输出，同时保持身份和文本条件。

**3. 主要结果及其意义：**
广泛的实验表明，ComposeMe在准确遵循视觉和文本提示方面达到了最先进的性能。

*   **高保真和解耦控制：** 该方法能够生成高保真的人类图像，对多个视觉因素（包括面部表情、头部姿态、身体姿态和风格）进行细粒度、解耦的控制，即使在单个图像中包含多个人物也能实现。
*   **优于现有方法：** 在多属性、单ID个性化任务中，ComposeMe在身份、发型和服装的保留方面显著优于OmniGen和GPT-40等现有方法。在单属性、多ID个性化任务中，ComposeMe也表现出最高的身份保留、组合质量和整体图像质量。
*   **自然组合和鲁棒解耦：** 交叉引用训练对于实现高保真图像生成至关重要，它能有效利用属性特定的视觉提示，即使这些提示存在错位，也能实现精确、解耦的面部、头发和服装控制。

**4. 论文中提到的局限性：**
*   **封闭集个性化：** ComposeMe是一个封闭集个性化方法，目前训练用于最多2个身份和3个属性。虽然对于大多数以人为中心的用例（面部、头发、服装和姿态）来说，这种覆盖范围已足够，但扩展到新的视觉提示（如场景、配饰）仍需进一步研究。
*   **小脸样本的失真：** 当文本提示指定远距离拍摄时，生成的面部会变得很小（小于图像的1%），导致面部出现明显的失真。这归因于训练数据过滤器移除了小脸样本，未来工作可以通过包含小脸样本来解决。

**5. 潜在的未来研究方向：**
*   **扩展到新的视觉提示：** 可以通过连接新的视觉提示来微调ComposeMe，或者为额外类别训练新的适配器并在推理时进行组合。
*   **提高图像质量：** 在高质量数据上进行微调和RL（强化学习）微调是提高图像质量的有前景的方向。
*   **更灵活和可重用的个性化管线：** 该框架为更可配置的人类图像合成铺平了道路，允许每个属性独立地进行策划、更新或替换，而无需重新训练整个身份表示。

---

**Key Findings:**

- We introduce a new
paradigm for attribute-specific image prompting, in which distinct sets of
reference images are used to guide the generation of individual aspects of
human appearance, such as hair, clothing, and identity.
- Our method encodes
these inputs into attribute-specific tokens, which are injected into a
pre-trained text-to-image diffusion model.
- Extensive experiments show that our method
achieves state-of-the-art performance in accurately following both visual and
textual prompts.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.18092v1)
- [arXiv](https://arxiv.org/abs/2509.18092v1)

---

<a id='2509.18090v1'></a>
## [GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction](https://arxiv.org/abs/2509.18090v1)

**Authors:** Jiahe Li, Jiawei Zhang, Youmin Zhang, Xiao Bai, Jin Zheng, Xiaohan Yu, Lin Gu

**Published:** 2025-09-22

**Categories:** cs.CV

**Abstract:**

Reconstructing accurate surfaces with radiance fields has achieved remarkable
progress in recent years. However, prevailing approaches, primarily based on
Gaussian Splatting, are increasingly constrained by representational
bottlenecks. In this paper, we introduce GeoSVR, an explicit voxel-based
framework that explores and extends the under-investigated potential of sparse
voxels for achieving accurate, detailed, and complete surface reconstruction.
As strengths, sparse voxels support preserving the coverage completeness and
geometric clarity, while corresponding challenges also arise from absent scene
constraints and locality in surface refinement. To ensure correct scene
convergence, we first propose a Voxel-Uncertainty Depth Constraint that
maximizes the effect of monocular depth cues while presenting a voxel-oriented
uncertainty to avoid quality degradation, enabling effective and robust scene
constraints yet preserving highly accurate geometries. Subsequently, Sparse
Voxel Surface Regularization is designed to enhance geometric consistency for
tiny voxels and facilitate the voxel-based formation of sharp and accurate
surfaces. Extensive experiments demonstrate our superior performance compared
to existing methods across diverse challenging scenarios, excelling in
geometric accuracy, detail preservation, and reconstruction completeness while
maintaining high efficiency. Code is available at
https://github.com/Fictionarry/GeoSVR.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Jiahe Li等人撰写的论文“GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction”的全面摘要。

---

**论文摘要：GeoSVR: Taming Sparse Voxels for Geometrically Accurate Surface Reconstruction**

**1. 解决的主要问题或研究问题：**
该论文旨在解决当前基于辐射场的表面重建方法（特别是基于高斯溅射（Gaussian Splatting）的方法）在几何精度、细节保留和重建完整性方面存在的局限性。这些现有方法常受限于表示瓶颈，例如依赖结构化点云初始化导致不完整或不准确的区域，以及高斯基元缺乏清晰边缘导致几何模糊。GeoSVR探索了稀疏体素在实现精确、详细和完整表面重建方面的潜力，并解决了稀疏体素固有的场景约束缺失和表面细化局部性问题。

**2. 关键创新或方法论贡献：**
GeoSVR提出了一个显式的、基于体素的框架，其主要创新包括：

*   **体素不确定性深度约束（Voxel-Uncertainty Depth Constraint）：** 针对稀疏体素缺乏强结构先验的问题，该方法利用单目深度线索作为场景约束。它通过评估每个体素的几何不确定性，自适应地确定对外部深度线索的依赖程度，从而在避免质量下降的同时，实现有效且鲁棒的场景约束，并保留高精度的几何结构。
*   **稀疏体素表面正则化（Sparse Voxel Surface Regularization）：** 为了解决稀疏体素局部性过强导致表面形成不准确的问题，该方法设计了两种体素级正则化：
    *   **几何正则化与体素丢弃（Voxel Dropout）：** 通过随机丢弃一部分体素，强制每个微小体素在更大的区域内保持全局几何一致性，从而扩大正则化范围，纠正错误的几何结构。
    *   **表面校正（Surface Rectification）：** 限制表面形成与唯一体素对齐，以减少深度偏差。
    *   **缩放惩罚（Scaling Penalty）：** 消除几何不准确的大体素参与表面形成，进一步促进尖锐和准确的表面重建。

**3. 主要结果及其重要性：**
GeoSVR在DTU、Tanks and Temples以及Mip-NeRF 360等多个具有挑战性的数据集上进行了广泛实验，结果表明其性能优于现有方法，在几何精度、细节保留和重建完整性方面表现出色，同时保持了高效率。具体来说：

*   在DTU数据集上，GeoSVR在Chamfer距离上实现了最高的重建质量。
*   在Tanks and Temples数据集上，GeoSVR在F1分数上表现最佳，尤其在复杂建筑和弱纹理区域展现出卓越的细节捕捉能力。
*   在Mip-NeRF 360数据集上，GeoSVR在渲染质量方面也表现出竞争力。
*   该方法继承了SVRaster的高效率，在推理速度上表现快速，且GPU内存占用较低。

这些结果证明了GeoSVR在处理反射区域、覆盖不足区域以及需要精细细节的场景时，能够提供更准确、更完整的表面重建，克服了基于高斯溅射方法在初始化点云不足时的局限性。

**4. 论文中提及的局限性：**
论文指出了GeoSVR当前存在的局限性，主要包括：

*   **反射区域：** 在具有严重反射的区域，由于光度不一致的强烈误导，模型性能可能受限。
*   **无纹理区域：** 在缺乏纹理信息的区域，几何重建可能面临挑战。
*   **透明表面：** 当前的辐射场方法在处理复杂光线追踪方面存在不足，导致透明物体的重建效果不理想。

**5. 潜在的未来研究方向：**
为了解决上述局限性，论文提出了以下未来研究方向：

*   引入更高效的光线追踪技术，以更好地处理反射和透明表面。
*   改进体素的全局性，使其能够更好地应对光照变化和无纹理区域等挑战。
*   开发针对透明物体的解决方案，以提高其重建质量。

---

这份摘要突出了GeoSVR在利用稀疏体素进行高精度表面重建方面的创新性，以及其在解决现有方法局限性方面的贡献。

**Key Findings:**

- In this paper, we introduce GeoSVR, an explicit voxel-based
framework that explores and extends the under-investigated potential of sparse
voxels for achieving accurate, detailed, and complete surface reconstruction.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.18090v1)
- [arXiv](https://arxiv.org/abs/2509.18090v1)

---

<a id='2509.18056v1'></a>
## [TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs](https://arxiv.org/abs/2509.18056v1)

**Authors:** Yunheng Li, Jing Cheng, Shaoyong Jia, Hangyi Kuang, Shaohui Jiao, Qibin Hou, Ming-Ming Cheng

**Published:** 2025-09-22

**Categories:** cs.CV

**Abstract:**

This paper introduces TempSamp-R1, a new reinforcement fine-tuning framework
designed to improve the effectiveness of adapting multimodal large language
models (MLLMs) to video temporal grounding tasks. We reveal that existing
reinforcement learning methods, such as Group Relative Policy Optimization
(GRPO), rely on on-policy sampling for policy updates. However, in tasks with
large temporal search spaces, this strategy becomes both inefficient and
limited in performance, as it often fails to identify temporally accurate
solutions. To address this limitation, TempSamp-R1 leverages ground-truth
annotations as off-policy supervision to provide temporally precise guidance,
effectively compensating for the sparsity and misalignment in on-policy
solutions. To further stabilize training and reduce variance in reward-based
updates, TempSamp-R1 provides a non-linear soft advantage computation method
that dynamically reshapes the reward feedback via an asymmetric transformation.
By employing a hybrid Chain-of-Thought (CoT) training paradigm, TempSamp-R1
optimizes a single unified model to support both CoT and non-CoT inference
modes, enabling efficient handling of queries with varying reasoning
complexity. Experimental results demonstrate that TempSamp-R1 outperforms
GRPO-based baselines, establishing new state-of-the-art performance on
benchmark datasets: Charades-STA (R1@0.7: 52.9%, +2.7%), ActivityNet Captions
(R1@0.5: 56.0%, +5.3%), and QVHighlights (mAP: 30.0%, +3.0%). Moreover,
TempSamp-R1 shows robust few-shot generalization capabilities under limited
data. Code: https://github.com/HVision-NKU/TempSamp-R1

**Analysis:**

好的，这是一篇关于“TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs”的全面摘要：

**论文摘要：TempSamp-R1：基于强化学习微调的视频LLM有效时间采样**

**1. 主要问题或研究问题：**
该论文旨在解决多模态大型语言模型（MLLMs）在视频时间定位任务中面临的挑战。现有的强化学习（RL）方法，如Group Relative Policy Optimization (GRPO)，主要依赖于“on-policy”采样进行策略更新。然而，在时间搜索空间巨大的任务中，这种策略效率低下且性能有限，因为它往往难以识别时间上精确的解决方案。这导致了在稀疏监督下学习不稳定、收敛到次优解以及在复杂视频理解任务中泛化的困难。

**2. 关键创新或方法论贡献：**
TempSamp-R1引入了一个新的强化学习微调框架，通过以下创新点解决了上述问题：

*   **混合策略采样与off-policy指导：** TempSamp-R1将高质量的外部解决方案（例如，ground-truth标注）作为off-policy监督引入策略优化过程。这提供了时间上精确的指导，有效弥补了on-policy解决方案中常见的稀疏性和错位问题，从而提高了策略优化的稳定性和效率。
*   **非线性软优势计算：** 为了进一步稳定训练并减少基于奖励更新的方差，TempSamp-R1提出了一种非线性软优势计算方法。该方法通过不对称变换动态重塑奖励反馈，区分高奖励和低奖励解决方案的学习动态，压缩接近最优解决方案的优势值，并放大次优解决方案之间的相对奖励差距，从而生成更具信息量的梯度并促进稳定的策略优化。
*   **混合思维链（CoT）训练范式：** TempSamp-R1采用混合CoT训练范式，优化一个统一模型以支持CoT和非CoT推理模式。这使得模型能够高效处理不同推理复杂度的查询，同时在两种模式下都表现出鲁棒性能。

**3. 主要结果及其意义：**
实验结果表明，TempSamp-R1在多个基准数据集上超越了基于GRPO的基线，并建立了新的最先进性能：

*   **Charades-STA：** R1@0.7达到52.9%，相对GRPO提升2.7%。
*   **ActivityNet Captions：** R1@0.5达到56.0%，相对GRPO提升5.3%。
*   **QVHighlights：** mAP达到30.0%，相对GRPO提升3.0%。

此外，TempSamp-R1在有限数据下表现出强大的少样本泛化能力，这凸显了其在数据稀缺场景下的实用性。消融研究进一步证实了off-policy监督和软优势整形策略对提高性能和训练稳定性的重要性。

**4. 论文中提到的局限性：**
论文提到了TempSamp-R1的几个局限性：

*   **依赖高质量off-policy监督：** 该框架目前依赖于高质量的off-policy监督（例如，ground-truth时间戳），这在弱标注场景中可能无法获得。
*   **未探索其他视频推理任务：** 尽管TempSamp-R1在时间定位和高光检测任务上进行了评估，但其在其他视频推理任务（例如，多事件跟踪）上的有效性仍有待探索。

**5. 潜在的未来研究方向：**
虽然论文没有明确列出未来研究方向，但从其局限性可以推断出以下潜在方向：

*   **在弱监督或无监督场景下的应用：** 探索TempSamp-R1在没有高质量ground-truth标注的情况下如何工作，例如通过自监督或弱监督学习来获取off-policy指导。
*   **扩展到更广泛的视频推理任务：** 将TempSamp-R1应用于其他复杂的视频理解任务，如多事件跟踪、视频问答等，以验证其通用性和鲁棒性。
*   **进一步优化off-policy指导的获取：** 研究更智能、更自适应的方法来生成或选择off-policy解决方案，以减少对外部标注的依赖。
*   **探索更复杂的奖励整形机制：** 进一步研究非线性奖励整形，以适应更广泛的任务和数据分布，从而实现更精细的策略优化。

总而言之，TempSamp-R1通过创新的混合策略采样、非线性软优势计算和混合CoT训练范式，显著提升了MLLMs在视频时间定位任务中的性能和稳定性，为长视频理解领域的强化学习微调开辟了新的方向。

**Key Findings:**

- This paper introduces TempSamp-R1, a new reinforcement fine-tuning framework
designed to improve the effectiveness of adapting multimodal large language
models (MLLMs) to video temporal grounding tasks.
- Experimental results demonstrate that TempSamp-R1 outperforms
GRPO-based baselines, establishing new state-of-the-art performance on
benchmark datasets: Charades-STA (R1@0.7: 52.9%, +2.7%), ActivityNet Captions
(R1@0.5: 56.0%, +5.3%), and QVHighlights (mAP: 30.0%, +3.0%).

**Links:**

- [PDF](https://arxiv.org/pdf/2509.18056v1)
- [arXiv](https://arxiv.org/abs/2509.18056v1)

---

<a id='2509.17941v1'></a>
## [ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion](https://arxiv.org/abs/2509.17941v1)

**Authors:** Zichao Hu, Chen Tang, Michael J. Munje, Yifeng Zhu, Alex Liu, Shuijing Liu, Garrett Warnell, Peter Stone, Joydeep Biswas

**Published:** 2025-09-22

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG

**Abstract:**

This paper considers the problem of enabling robots to navigate dynamic
environments while following instructions. The challenge lies in the
combinatorial nature of instruction specifications: each instruction can
include multiple specifications, and the number of possible specification
combinations grows exponentially as the robot's skill set expands. For example,
"overtake the pedestrian while staying on the right side of the road" consists
of two specifications: "overtake the pedestrian" and "walk on the right side of
the road." To tackle this challenge, we propose ComposableNav, based on the
intuition that following an instruction involves independently satisfying its
constituent specifications, each corresponding to a distinct motion primitive.
Using diffusion models, ComposableNav learns each primitive separately, then
composes them in parallel at deployment time to satisfy novel combinations of
specifications unseen in training. Additionally, to avoid the onerous need for
demonstrations of individual motion primitives, we propose a two-stage training
procedure: (1) supervised pre-training to learn a base diffusion model for
dynamic navigation, and (2) reinforcement learning fine-tuning that molds the
base model into different motion primitives. Through simulation and real-world
experiments, we show that ComposableNav enables robots to follow instructions
by generating trajectories that satisfy diverse and unseen combinations of
specifications, significantly outperforming both non-compositional VLM-based
policies and costmap composing baselines. Videos and additional materials can
be found on the project page: https://amrl.cs.utexas.edu/ComposableNav/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Zichao Hu等人撰写的论文“ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion”的全面摘要。

---

### 论文摘要：ComposableNav: Instruction-Following Navigation in Dynamic Environments via Composable Diffusion

**1. 主要问题或研究问题：**
该论文旨在解决机器人在动态环境中遵循复杂指令进行导航的问题。核心挑战在于指令规范的组合性质：一个指令可能包含多个子规范（例如，“超车行人并靠右行驶”包含“超车行人”和“靠右行驶”两个规范），随着机器人技能集的扩展，可能的规范组合数量呈指数级增长，这使得传统的基于学习的方法（如模仿学习或强化学习）因数据和计算资源需求巨大而变得不切实际。

**2. 关键创新或方法论贡献：**
为了应对上述挑战，ComposableNav提出了以下关键创新：
*   **组合式扩散模型（Composable Diffusion Models）：** 论文的核心思想是，遵循指令涉及独立满足其组成规范，每个规范对应一个独特的运动原语。ComposableNav利用扩散模型的可组合性，单独学习每个运动原语，然后在部署时并行组合这些原语，以满足训练中未见过的规范组合。
*   **两阶段训练程序：** 为了避免对每个单独运动原语进行演示的繁重需求，论文提出了一种两阶段训练方法：
    1.  **监督式预训练（Supervised Pre-training）：** 学习一个用于动态导航的基础扩散模型，生成多样化、无碰撞、目标导向的轨迹。
    2.  **强化学习微调（Reinforcement Learning Fine-tuning）：** 将基础模型塑造成不同的运动原语，通过设计特定于原语的奖励函数来确保指令依从性，从而避免了对特定演示数据集的需求。
*   **实时部署机制：** 结合模型预测控制器（MPC）和在线重规划策略，确保了ComposableNav在真实世界机器人上的实时性能。

**3. 主要结果及其意义：**
通过仿真和真实世界实验，ComposableNav展示了显著的性能优势：
*   **卓越的泛化能力：** ComposableNav能够生成满足多样化且训练中未见过的规范组合的轨迹，显著优于非组合式VLM（视觉语言模型）策略和基于成本图组合的基线方法。
*   **处理复杂指令：** 随着指令复杂性（规范数量）的增加，ComposableNav的成功率保持较高，而所有基线方法的性能则迅速下降，这证明了其在处理复杂、未见过的指令组合方面的鲁棒性。
*   **实时操作：** 在真实世界机器人上部署时，ComposableNav实现了实时重规划，即使在涉及四个原语的最复杂情况下，初始规划平均仅需0.4秒，重规划仅需0.06秒。
*   **无需特定演示数据：** 两阶段训练方法成功地使扩散模型学习了有效的运动原语，而无需为每个原语提供专门的演示数据。

**4. 论文中提及的局限性：**
*   **运动原语数量有限：** 目前只考虑了六种常用的导航运动原语，这些原语相对简单，可以用基于规则的奖励函数描述。手动设计奖励函数的可扩展性有限。
*   **依赖上游模块：** 论文假设指令解析为规范和相关环境观测（例如，通过LLM和VLM）可以由现有方法处理，这部分工作在实验中被抽象化。
*   **组合策略的局限性：** 尽管ComposableNav表现出色，但随着指令规范数量的增加，成功率仍有下降。这可能源于当前组合策略（通过对单个去噪网络的预测噪声求和）可能导致次优结果。

**5. 潜在的未来研究方向：**
*   **利用VLM作为验证器：** 探索使用VLM作为验证器来自动学习多样化和复杂的行为，以提高奖励函数设计的可扩展性。
*   **集成高层VLM模块和任务规划器：** 将ComposableNav与高层VLM模块和任务规划器结合，以实现更长期的指令遵循导航。
*   **改进组合采样技术：** 探索更先进的扩散模型采样技术（如哈密顿蒙特卡洛），以提高在更高指令复杂性下的组合性能。

---

总而言之，ComposableNav通过引入可组合扩散模型和创新的两阶段训练流程，为机器人在动态环境中遵循复杂指令导航提供了一个强大且可扩展的解决方案，显著推动了该领域的研究进展。

**Key Findings:**

- For example,
"overtake the pedestrian while staying on the right side of the road" consists
of two specifications: "overtake the pedestrian" and "walk on the right side of
the road." To tackle this challenge, we propose ComposableNav, based on the
intuition that following an instruction involves independently satisfying its
constituent specifications, each corresponding to a distinct motion primitive.
- Using diffusion models, ComposableNav learns each primitive separately, then
composes them in parallel at deployment time to satisfy novel combinations of
specifications unseen in training.
- Additionally, to avoid the onerous need for
demonstrations of individual motion primitives, we propose a two-stage training
procedure: (1) supervised pre-training to learn a base diffusion model for
dynamic navigation, and (2) reinforcement learning fine-tuning that molds the
base model into different motion primitives.
- Through simulation and real-world
experiments, we show that ComposableNav enables robots to follow instructions
by generating trajectories that satisfy diverse and unseen combinations of
specifications, significantly outperforming both non-compositional VLM-based
policies and costmap composing baselines.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17941v1)
- [arXiv](https://arxiv.org/abs/2509.17941v1)

---

<a id='2509.17901v1'></a>
## [Does Audio Matter for Modern Video-LLMs and Their Benchmarks?](https://arxiv.org/abs/2509.17901v1)

**Authors:** Geewook Kim, Minjoon Seo

**Published:** 2025-09-22

**Categories:** cs.CV, cs.MM, cs.SD

**Abstract:**

Modern multimodal large language models often claim "video understanding,"
yet most evaluations use muted videos or simply discard audio. We ask a direct
question: how much does audio actually matter for contemporary Video-LLMs and
the benchmarks that certify them? We audit widely used suites and observe that
many items are even solvable from a single frame, rendering audio largely
redundant. Building on LLaVA-OneVision architecture, we attach a speech/audio
encoder (e.g., Whisper) and analyze when audio helps, while addressing audio
token explosion with a lightweight Mamba-based state-space token compressor. We
find that audio yields minimal gains on recent video benchmarks but is decisive
on curated, audio-sensitive subsets. To enable faithful evaluation, we release
AVQA-Hard and Music-AVQA-Hard, our model, and code. Our findings surface a
growing gap between current academic practice and real-world expectations, and
provide practical tools for scalable audio-visual Video-LLMs. We will fully
open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Geewook Kim和Minjoon Seo撰写的论文“Does Audio Matter for Modern Video-LLMs and Their Benchmarks?”的全面摘要。

---

### 论文摘要：Does Audio Matter for Modern Video-LLMs and Their Benchmarks?

**1. 主要问题或研究问题：**
该论文的核心研究问题是：在当前的视频大型语言模型（Video-LLMs）及其评估基准中，音频究竟有多重要？作者指出，尽管现代多模态大型语言模型声称具备“视频理解”能力，但大多数评估却使用静音视频或直接忽略音频。这引发了一个关键疑问：这种做法是否掩盖了音频在真实视频理解中的潜在价值？

**2. 关键创新或方法论贡献：**
为了解决上述问题，论文提出了以下关键创新和方法论贡献：

*   **音频敏感基准的审计与策展：** 作者审计了广泛使用的视频理解基准套件，发现许多任务仅凭单帧视觉信息即可解决，使得音频信息在很大程度上是冗余的。为了解决这一问题，论文通过过滤掉可以仅凭单帧解决的项目，策展并发布了两个新的、更具挑战性的、音频敏感的基准数据集：**AVQA-Hard** 和 **Music-AVQA-Hard**。
*   **LLaVA-AV-SSM 模型架构：** 论文基于强大的 LLaVA-OneVision 架构，通过附加一个语音/音频编码器（例如 Whisper）来注入音频 token，从而构建了 **LLaVA-AV-SSM** 模型。这种双编码器架构允许对音频的边际价值进行受控研究。
*   **基于 Mamba 的音频 token 压缩器：** 针对音频 token 数量爆炸的问题（例如，一小时视频可能产生约 90k 个音频 token），论文引入了一种轻量级的、基于 Mamba 的状态空间模型（SSM）token 压缩器。该压缩器能够将长音频流聚合为紧凑的 token 集合，同时最大限度地减少信息损失，从而实现了可扩展的音视频 Video-LLMs。它通过在每 R 步插入一个可训练查询并仅保留查询位置的输出来实现 25 倍的压缩（从 25Hz 降至 1Hz）。

**3. 主要结果及其意义：**
论文的主要发现及其意义如下：

*   **音频在现有基准中的作用有限：** 在大多数现有视频基准上，添加音频带来的性能提升微乎其微，甚至在方差范围内，这与作者的审计结果一致，即这些基准很少需要音轨信息。
*   **音频在策展基准中的决定性作用：** 在新策展的 AVQA-Hard 和 Music-AVQA-Hard 等音频敏感子集上，音频带来了显著的性能提升。例如，在 AVQA-Hard 上，准确率从 67.13% 提高到 71.58%。这验证了对音频敏感评估的需求，并强调了在这些任务中“倾听”的重要性。
*   **Mamba 压缩器的有效性：** 基于 Mamba 的音频 token 压缩器不仅有效解决了音频 token 爆炸问题，实现了长视频的线性时间推理，而且在音频敏感基准上提升了准确率。这表明 Mamba 压缩器是实现可扩展音视频 Video-LLMs 的有效工具。
*   **学术实践与现实世界期望的差距：** 论文的发现揭示了当前学术实践（忽略音频）与现实世界期望（用户自然地假设系统既“看”又“听”）之间日益扩大的差距。

**4. 论文中提及的局限性：**
论文中提及的局限性包括：

*   **音频编码器和数据的探索范围：** 目前的研究主要使用了 Qwen2-Audio 的 Whisper-based 编码器，未来可以探索更广泛的音频编码器和数据集。
*   **统一的跨模态 SSM：** 目前的音频压缩器是独立的，未来可以扩展到超越纯音频压缩，实现一个统一的跨模态 SSM，能够联合分配视觉和音频的预算。

**5. 潜在的未来研究方向：**
基于上述局限性，论文指出了以下潜在的未来研究方向：

*   探索更广泛的音频编码器和训练数据，以进一步提升音视频 Video-LLMs 的性能。
*   开发统一的跨模态状态空间模型，能够智能地在视觉和音频模态之间分配计算资源，实现更高效、更全面的视频理解。
*   继续策展和开发更具挑战性的、真正需要音视频联合推理的基准，以推动该领域的发展。

---

总而言之，这篇论文对现代 Video-LLMs 中音频的作用进行了深入的批判性审视。通过引入音频敏感的基准和基于 Mamba 的高效音频压缩方法，作者不仅揭示了当前评估实践的不足，还为构建更符合现实世界需求的、能够真正“看”和“听”的音视频大型语言模型提供了实用的工具和方向。

**Key Findings:**

- To enable faithful evaluation, we release
AVQA-Hard and Music-AVQA-Hard, our model, and code.
- Our findings surface a
growing gap between current academic practice and real-world expectations, and
provide practical tools for scalable audio-visual Video-LLMs. We will fully
open-source our work at https://github.com/naver-ai/LLaVA-AV-SSM.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17901v1)
- [arXiv](https://arxiv.org/abs/2509.17901v1)

---

<a id='2509.17877v1'></a>
## [Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection](https://arxiv.org/abs/2509.17877v1)

**Authors:** Richard Kuhlmann, Jakob Wolfram, Boyang Sun, Jiaxu Xing, Davide Scaramuzza, Marc Pollefeys, Cesar Cadena

**Published:** 2025-09-22

**Categories:** cs.RO, cs.CV

**Abstract:**

Autonomous inspection is a central problem in robotics, with applications
ranging from industrial monitoring to search-and-rescue. Traditionally,
inspection has often been reduced to navigation tasks, where the objective is
to reach a predefined location while avoiding obstacles. However, this
formulation captures only part of the real inspection problem. In real-world
environments, the inspection targets may become visible well before their exact
coordinates are reached, making further movement both redundant and
inefficient. What matters more for inspection is not simply arriving at the
target's position, but positioning the robot at a viewpoint from which the
target becomes observable. In this work, we revisit inspection from a
perception-aware perspective. We propose an end-to-end reinforcement learning
framework that explicitly incorporates target visibility as the primary
objective, enabling the robot to find the shortest trajectory that guarantees
visual contact with the target without relying on a map. The learned policy
leverages both perceptual and proprioceptive sensing and is trained entirely in
simulation, before being deployed to a real-world robot. We further develop an
algorithm to compute ground-truth shortest inspection paths, which provides a
reference for evaluation. Through extensive experiments, we show that our
method outperforms existing classical and learning-based navigation approaches,
yielding more efficient inspection trajectories in both simulated and
real-world settings. The project is avialable at
https://sight-over-site.github.io/

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Richard Kuhlmann等人撰写的论文“Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection”的全面摘要。

---

### 论文摘要：Sight Over Site: Perception-Aware Reinforcement Learning for Efficient Robotic Inspection

**1. 主要问题或研究问题：**
传统的机器人自主巡检任务通常被简化为导航问题，即机器人需要到达预定义的物理位置并避开障碍物。然而，这种方法未能充分捕捉真实巡检任务的本质。在实际环境中，巡检目标可能在机器人到达其精确坐标之前就已经可见，导致后续移动冗余且低效。因此，论文旨在解决的核心问题是：如何使机器人能够从一个“感知感知”的角度进行巡检，即不仅仅是到达目标位置，而是将机器人定位在一个能够清晰观察到目标的视点，并以最短路径实现这一目标，同时不依赖于预先构建的地图。

**2. 关键创新或方法论贡献：**
*   **感知感知巡检任务的重新定义与框架：** 论文将巡检任务重新定义为以目标可见性为核心，而非简单的点目标导航。为此，作者提出了一个端到端的强化学习（RL）框架，将目标可见性明确地作为主要优化目标。
*   **无地图的RL策略：** 提出的策略不依赖于全局地图，而是利用自我中心深度图像输入、机器人与目标之间的相对姿态以及过去的动作来学习高效的巡检行为。这种方法避免了显式地图构建的开销，使其能够扩展到大型环境。
*   **模拟到真实世界的泛化能力：** 策略完全在模拟环境中进行训练，但能够很好地泛化到未见的模拟环境以及真实的四足机器人（Boston Dynamics Spot），实现了零样本迁移。
*   **地面真实最短巡检路径算法：** 论文开发了一种算法来计算地面真实的最短巡检路径，为评估提供了可靠的参考基准。该算法考虑了可遍历性、传感器范围和可见性约束，通过A*搜索找到从起始点到目标可见视点的最短路径。
*   **奖励函数设计：** 奖励函数被精心设计，包含稀疏的成功/失败奖励和密集的奖励项（如目标方向对齐、向最优视点移动、惩罚原地旋转和鼓励探索），以平衡导航和可见性，并避免局部最优。

**3. 主要结果及其意义：**
*   **优于现有导航方法：** 实验结果表明，该方法在模拟和真实世界环境中均优于现有的经典和基于学习的导航方法，能够生成更高效的巡检轨迹。
*   **更高的成功加权路径长度（SPL）：** 尽管在某些碰撞和滑行允许的设置下，DD-PPO的成功率可能略高，但本方法在SPL指标上表现更优，这意味着它能以更短的路径实现目标可见性。
*   **强大的碰撞避免能力和鲁棒性：** 在严格的无碰撞设置下，本方法仍能保持高成功率（81.49%），并展现出强大的碰撞避免能力。与DD-PPO相比，本策略在不同环境设置下表现出更好的鲁棒性，不易受模拟器特定动态的影响。
*   **平衡导航与可见性：** 策略能够有效地平衡导航和可见性，在目标较远时优先导航，在目标附近时则转向寻找最佳视点，从而避免了冗余移动。
*   **成功的Sim-to-Real迁移：** 在Boston Dynamics Spot机器人上的真实世界实验验证了策略的零样本迁移能力，证明了其在实际应用中的有效性。

**4. 论文中提到的局限性：**
*   **地面真实路径计算的效率问题：** 在RL训练过程中，重复计算地面真实最短巡检路径（包括可见性检查）效率低下，因此在训练时采用了基于奖励函数的设计而非直接使用地面真实路径作为奖励信号。
*   **传感器噪声：** 真实世界实验中，深度传感器测量存在噪声，需要通过平均深度值等方法来缓解其对可见性判断的影响。
*   **训练环境的复杂性：** 尽管在Habitat模拟器上进行了训练，但真实世界的复杂性和不可预测性仍可能带来挑战。

**5. 潜在的未来研究方向：**
*   **更复杂的巡检场景：** 探索在更复杂、动态或大规模环境中的巡检任务，例如多目标巡检或需要与环境交互的巡检。
*   **结合语义信息：** 将语义信息（例如目标类型、功能区域）融入感知感知巡检框架，以实现更智能、更具上下文意识的巡检行为。
*   **在线适应与持续学习：** 研究如何在部署后使策略能够在线适应新的环境或目标，实现持续学习和性能提升。
*   **多模态感知融合：** 探索除了深度图像之外，结合RGB图像、激光雷达等多种传感器模态，以获取更丰富、更鲁棒的环境感知。
*   **可解释性与安全性：** 进一步提高RL策略的可解释性，并增强其在安全关键巡检任务中的可靠性和安全性保障。

---

**Key Findings:**

- We propose an end-to-end reinforcement learning
framework that explicitly incorporates target visibility as the primary
objective, enabling the robot to find the shortest trajectory that guarantees
visual contact with the target without relying on a map.
- Through extensive experiments, we show that our
method outperforms existing classical and learning-based navigation approaches,
yielding more efficient inspection trajectories in both simulated and
real-world settings.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17877v1)
- [arXiv](https://arxiv.org/abs/2509.17877v1)

---

<a id='2509.17786v1'></a>
## [Accurate and Efficient Low-Rank Model Merging in Core Space](https://arxiv.org/abs/2509.17786v1)

**Authors:** Aniello Panariello, Daniel Marczak, Simone Magistri, Angelo Porrello, Bartłomiej Twardowski, Andrew D. Bagdanov, Simone Calderara, Joost van de Weijer

**Published:** 2025-09-22

**Categories:** cs.CV, cs.AI

**Abstract:**

In this paper, we address the challenges associated with merging low-rank
adaptations of large neural networks. With the rise of parameter-efficient
adaptation techniques, such as Low-Rank Adaptation (LoRA), model fine-tuning
has become more accessible. While fine-tuning models with LoRA is highly
efficient, existing merging methods often sacrifice this efficiency by merging
fully-sized weight matrices. We propose the Core Space merging framework, which
enables the merging of LoRA-adapted models within a common alignment basis,
thereby preserving the efficiency of low-rank adaptation while substantially
improving accuracy across tasks. We further provide a formal proof that
projection into Core Space ensures no loss of information and provide a
complexity analysis showing the efficiency gains. Extensive empirical results
demonstrate that Core Space significantly improves existing merging techniques
and achieves state-of-the-art results on both vision and language tasks while
utilizing a fraction of the computational resources. Codebase is available at
https://github.com/apanariello4/core-space-merging.

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供Aniello Panariello等人撰写的论文“Accurate and Efficient Low-Rank Model Merging in Core Space”的全面摘要。

---

### 论文摘要：Accurate and Efficient Low-Rank Model Merging in Core Space

**1. 主要问题或研究问题：**
该论文旨在解决大型神经网络的低秩适应（如LoRA）模型合并所面临的挑战。尽管LoRA微调本身效率很高，但现有的模型合并方法通常通过合并全尺寸权重矩阵来牺牲这种效率，导致计算成本高昂且无法扩展到大型模型。核心问题是如何在保持低秩适应效率的同时，有效且准确地合并这些模型，以处理多任务学习场景。

**2. 关键创新或方法论贡献：**
该论文提出了一个名为“Core Space”的创新合并框架，其核心贡献包括：

*   **Core Space 合并框架：** 引入了一个参数高效的子空间——Core Space，作为所有任务特定低秩组件的通用对齐基础。它允许在保持低秩适应效率的同时，在更小的维度空间（r x r）中执行模型合并，而无需重建全尺寸权重矩阵。
*   **信息无损投影：** 论文提供了正式的证明，表明投影到Core Space并返回原始空间是无损的，确保在合并过程中不会丢失信息。Core Space被设计为可逆的，并且其维度仅取决于任务数量和LoRA秩，与基础模型大小无关。
*   **效率提升：** 通过在Core Space中进行合并，论文显著降低了计算成本。与现有方法（如KnOTS）相比，Core Space合并在计算资源使用上实现了显著的加速（例如，对于Llama 3 8B模型，速度提升超过600倍）。
*   **子空间对齐改进：** Core Space通过强制所有任务共享一个公共基础来提高子空间对齐比率（SAR），过滤掉任务特定噪声并促进对齐，从而提高合并模型的性能。

**3. 主要结果及其意义：**
论文通过广泛的实证结果证明了Core Space合并的有效性：

*   **最先进的性能：** 在视觉和语言任务（包括ViT-B/32、ViT-L/14和Llama 3 8B骨干网络）上，Core Space显著优于现有合并技术，并实现了最先进的性能。例如，在Llama 3 8B上，它将TSV的平均归一化准确率提升至94.16%。
*   **计算效率：** Core Space合并在保持高准确率的同时，仅使用了竞争方法一小部分的计算资源。其时间复杂度与Task Arithmetic在全空间中的复杂度相当，但避免了高维SVD操作。
*   **对异构秩和额外PEFT方法的通用性：** Core Space能够无缝处理具有异构LoRA秩的模型合并，并且可以扩展到其他PEFT方法，如VeRA，同样表现出优越的性能。
*   **信息密度：** 分析表明，Core Space是一个信息密集的空间，截断任何组件都会导致性能下降，这与全空间中存在大量冗余或未使用组件形成对比。

**4. 论文中提及的局限性：**
论文中没有明确提及显著的局限性。然而，可以推断出一些潜在的方面：

*   **线性合并函数的假设：** 尽管论文证明了Core Space在非线性合并函数下也能提高性能，但其信息无损特性在理论上主要依赖于合并函数是线性的情况（如Task Arithmetic）。对于更复杂的非线性合并策略，其理论保证可能需要进一步的细化。
*   **参考基础的选择：** 论文虽然展示了其选择的参考基础（通过垂直堆叠A(t)和水平堆叠B(t)的SVD获得）是最佳的，但对于其他可能的参考基础选择，其性能可能会有所不同。
*   **特定任务设置：** 论文主要关注多任务合并，即合并多个在不同任务上微调的模型。对于其他合并场景（例如，模型压缩或知识蒸馏），Core Space的适用性可能需要进一步探索。

**5. 潜在的未来研究方向：**
论文为未来的研究提供了几个潜在方向：

*   **更复杂的合并策略：** 探索在Core Space中应用更先进、更复杂的非线性合并策略，以进一步提高性能或解决特定挑战。
*   **理论泛化：** 进一步研究Core Space框架在更广泛的PEFT方法和模型架构上的理论泛化能力。
*   **动态Core Space：** 探索动态调整Core Space维度或结构的方法，以适应不断变化的任务数量或模型复杂性。
*   **与其他模型压缩技术的结合：** 将Core Space合并与量化、剪枝等其他模型压缩技术相结合，以实现更极致的模型效率。
*   **在更广泛应用中的评估：** 在更多样化的应用领域（例如，机器人、强化学习等）中评估Core Space合并的有效性。

---

总而言之，这篇论文通过引入Core Space框架，为低秩适应模型的合并提供了一个高效且准确的解决方案。它不仅解决了现有方法在效率上的不足，还在多任务视觉和语言任务上取得了显著的性能提升，为大型模型的参数高效适应和多任务学习开辟了新的途径。

**Key Findings:**

- We propose the Core Space merging framework, which
enables the merging of LoRA-adapted models within a common alignment basis,
thereby preserving the efficiency of low-rank adaptation while substantially
improving accuracy across tasks.
- Extensive empirical results
demonstrate that Core Space significantly improves existing merging techniques
and achieves state-of-the-art results on both vision and language tasks while
utilizing a fraction of the computational resources.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17786v1)
- [arXiv](https://arxiv.org/abs/2509.17786v1)

---

<a id='2509.17773v1'></a>
## [I2VWM: Robust Watermarking for Image to Video Generation](https://arxiv.org/abs/2509.17773v1)

**Authors:** Guanjie Wang, Zehua Ma, Han Fang, Weiming Zhang

**Published:** 2025-09-22

**Categories:** cs.CV

**Abstract:**

The rapid progress of image-guided video generation (I2V) has raised concerns
about its potential misuse in misinformation and fraud, underscoring the urgent
need for effective digital watermarking. While existing watermarking methods
demonstrate robustness within a single modality, they fail to trace source
images in I2V settings. To address this gap, we introduce the concept of Robust
Diffusion Distance, which measures the temporal persistence of watermark
signals in generated videos. Building on this, we propose I2VWM, a cross-modal
watermarking framework designed to enhance watermark robustness across time.
I2VWM leverages a video-simulation noise layer during training and employs an
optical-flow-based alignment module during inference. Experiments on both
open-source and commercial I2V models demonstrate that I2VWM significantly
improves robustness while maintaining imperceptibility, establishing a new
paradigm for cross-modal watermarking in the era of generative video.
\href{https://github.com/MrCrims/I2VWM-Robust-Watermarking-for-Image-to-Video-Generation}{Code
Released.}

**Analysis:**

好的，作为计算机视觉和机器学习领域的专家，我将为您提供对Guanjie Wang, Zehua Ma, Han Fang, Weiming Zhang撰写的论文“I2VWM: Robust Watermarking for Image to Video Generation”的全面摘要。

---

### 论文摘要：I2VWM：图像到视频生成中的鲁棒水印技术

**1. 主要问题或研究问题：**
随着图像引导视频生成（I2V）技术的快速发展，其在虚假信息传播和欺诈方面的潜在滥用引起了广泛关注。现有的数字水印方法虽然在单一模态内表现出鲁棒性，但在I2V场景中，它们无法有效追踪源图像，即水印信号在视频生成过程中难以保持其时间上的鲁棒性。这导致了一个核心问题：如何在跨模态（图像到视频）生成过程中，确保水印信号的持久性和可验证性。

**2. 关键创新或方法论贡献：**
为了解决上述问题，本论文提出了以下关键创新和方法论贡献：
*   **鲁棒扩散距离（Robust Diffusion Distance, RDD）概念的引入：** 首次定义了RDD，用于量化水印信号在生成视频中随时间推移的持久性，即水印在视频中仍能可靠验证的最大帧索引。这为评估I2V场景下的水印鲁棒性提供了一个新的量化指标。
*   **I2VWM框架：** 提出了一种跨模态水印框架I2VWM，旨在增强水印在时间维度上的鲁棒性。该框架基于编码器-解码器架构，并包含以下核心组件：
    *   **视频模拟噪声层（Video-simulation Noise Layer）：** 在训练阶段引入，通过模拟视频生成固有的压缩、噪声添加/移除和像素偏移等失真，使水印模型能够显式地学习并抵抗这些生成引起的修改，从而提高水印信号的鲁棒性。
    *   **基于光流的对齐模块（Optical-flow-based Alignment Module）：** 在推理阶段使用，通过估计视频帧与参考帧之间的光流，将远离初始帧的视频帧对齐，从而补偿时间上的退化，显著延长水印的鲁棒扩散距离。
    *   **投票机制（Voting Scheme）：** 在视频水印提取模式下，对所有对齐帧提取的水印进行聚合，以获得最终的水印。

**3. 主要结果及其意义：**
*   **显著提高鲁棒性：** 在多个开源（如CogVideoX, Wan, Hunyuan, Stable Video Diffusion）和商业I2V模型上的实验表明，I2VWM在I2V场景下显著提高了水印的鲁棒性，尤其是在鲁棒扩散距离（RDD）和帧准确率（FACC）方面优于现有基线方法。
*   **保持不可感知性：** I2VWM在提高鲁棒性的同时，仍能保持水印的不可感知性，通过PSNR、SSIM和LPIPS等视觉质量指标验证了其性能。
*   **泛化能力强：** I2VWM在不同生成模型上的表现一致，表明其具有强大的泛化能力，不依赖于特定的生成器。
*   **对经典噪声的鲁棒性：** I2VWM对经典图像失真（如裁剪、高斯模糊、JPEG压缩等）也表现出具有竞争力的鲁棒性，这得益于其噪声层设计中对数值和几何失真的综合考虑。
*   **光流对齐模块的有效性：** 消融实验证实，基于光流的对齐模块能够显著提升水印提取的准确性，尤其是在时间上远离初始帧的视频帧中，进一步验证了其对水印信号持久性的贡献。

**4. 论文中提及的局限性：**
*   **对时间顺序的依赖：** I2VWM的有效性依赖于视频中一致的时间顺序假设，无法处理时间裁剪或帧打乱等失真。
*   **未完全解决经典失真：** 论文指出，I2VWM尚未完全解决所有经典失真，例如随机旋转，这在视频帧处理中也常见。
*   **缺乏视频生成质量指标：** 论文的评估缺乏对视频生成质量的指标。作者认为，高质量的视频往往能更好地保留源图像中的水印信号，因此平衡水印鲁棒性与视频质量是一个重要的考量。

**5. 潜在的未来研究方向：**
*   **处理时间无序失真：** 进一步研究如何使I2VWM能够应对时间裁剪或帧打乱等破坏时间顺序的失真，以提高数据保护能力。
*   **增强对所有经典失真的鲁棒性：** 探索更全面的方法来解决包括随机旋转在内的所有经典失真。
*   **整合视频生成质量评估：** 在未来的工作中，将视频生成质量指标纳入评估体系，以更全面地衡量水印方法在保持水印鲁棒性和视频质量之间的平衡。
*   **深入研究噪声层设计：** 进一步探索和优化噪声层设计，以应对更复杂的数值和几何失真组合。

---

总而言之，这篇论文首次提出了图像到视频生成场景下的跨模态鲁棒水印挑战，并引入了“鲁棒扩散距离”这一新颖指标。通过结合视频模拟噪声层和基于光流的对齐模块，I2VWM框架在保持水印不可感知性的同时，显著提升了水印信号在生成视频中的时间鲁棒性。这项工作为生成视频时代下的跨模态水印技术开辟了新范式，具有重要的理论和实际意义。

**Key Findings:**

- To address this gap, we introduce the concept of Robust
Diffusion Distance, which measures the temporal persistence of watermark
signals in generated videos.
- Building on this, we propose I2VWM, a cross-modal
watermarking framework designed to enhance watermark robustness across time.
- Experiments on both
open-source and commercial I2V models demonstrate that I2VWM significantly
improves robustness while maintaining imperceptibility, establishing a new
paradigm for cross-modal watermarking in the era of generative video.

**Links:**

- [PDF](https://arxiv.org/pdf/2509.17773v1)
- [arXiv](https://arxiv.org/abs/2509.17773v1)

---

