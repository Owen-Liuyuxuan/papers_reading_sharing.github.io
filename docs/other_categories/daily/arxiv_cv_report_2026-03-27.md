time: 20260327

# Arxiv Computer Vision Papers - 2026-03-27

## Executive Summary

### **Arxiv 计算机视觉领域论文日报执行摘要**
**报告日期：** 2026年3月26日  
**分析论文数：** 10篇

---

#### **1. 核心主题与趋势观察**

今日论文集中反映了计算机视觉领域的三个核心演进方向：

*   **1.1 基础模型的规模化与专业化：** 多篇论文致力于扩展基础模型的边界。**《Intern-S1-Pro》** 和 **《MuRF》** 分别从“科学多模态”和“多尺度潜力”角度，推动通用视觉模型向更专业、更精细的感知和理解能力发展。
*   **1.2 具身智能与自动驾驶的“个性化”和“自然交互”：** 以 **《Vega》** 和 **《Drive My Way》** 为代表，研究焦点从传统的感知与控制，转向如何让智能体（如自动驾驶系统）理解并适应人类的**自然语言指令**和**个性化偏好**，标志着人机交互范式的重要转变。
*   **1.3 高效三维与动态场景建模：** 一系列工作（如 **《WAFT-Stereo》、《Less Gaussians, Texture More》、《MegaFlow》、《Out of Sight...》**）共同关注如何更高效、更精确地重建、表示和理解三维动态世界，核心挑战在于处理**大位移、长序列、高保真纹理**，同时优化计算与存储效率。

#### **2. 重点与创新性论文亮点**

*   **《Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale》**：**极具野心的工作**。提出构建面向科学领域的万亿参数多模态基础模型，旨在统一处理科学图像、图表、文本等数据。若其愿景实现，将极大加速科学发现，是“AI for Science”在CV领域的里程碑式尝试。
*   **《Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving》**：**关键应用方向的突破**。将大语言模型与人类偏好对齐的技术（源自RLHF）成功引入自动驾驶的视觉-语言-动作模型中，使驾驶策略能个性化适配不同用户的风格（如激进或保守），是迈向“人车共驾”的关键一步。
*   **《PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference》**：**方法论上的巧妙创新**。提出一种高效的训练策略，仅用短视频训练即可让模型具备生成长视频和进行长上下文推理的能力，**显著降低了视频生成与理解模型的训练成本与数据需求**，具有很高的实用价值。

#### **3. 新兴研究方向与技术**

*   **“训练-推理解耦”的高效学习范式：** 《PackForcing》和《No Hard Negatives Required...》都体现了这一思想——通过改进训练目标或策略，使模型在推理时能泛化到远超训练数据范围（如更长视频、更复杂组合）的场景，这将成为扩展模型能力的关键技术路径。
*   **混合记忆架构用于动态世界模型：** 《Out of Sight but Not Out of Mind》提出的混合记忆系统，为解决长序列视频理解中的长期依赖和遗忘问题提供了新思路，对构建持续学习的自主智能体至关重要。
*   **无需复杂负样本的对比学习：** 《No Hard Negatives Required...》挑战了对比学习对困难负样本的依赖，提出以概念为中心的学习方法，在保持零样本能力的同时提升组合性。这可能简化训练流程并提高可解释性。

#### **4. 全文精读建议**

根据研究者的不同兴趣，优先推荐如下论文：

*   **所有研究者必读（领域风向标）：**
    *   **《Intern-S1-Pro》**：了解超大规模科学基础模型的最新蓝图与挑战。
    *   **《Drive My Way》**：学习如何将“对齐”技术应用于具身智能，把握人机交互前沿。

*   **3D视觉与图形学研究者：**
    *   **《Less Gaussians, Texture More》**：关注高保真、高效率的3D表示新进展（4K纹理的Feed-Forward生成）。
    *   **《WAFT-Stereo》** 或 **《MegaFlow》**：前者关注立体匹配的新范式（仅靠扭曲），后者关注大位移光流的零样本泛化，均属底层视觉的重要创新。

*   **视频生成与理解研究者：**
    *   **《PackForcing》**：其高效训练思想可能具有广泛的迁移应用价值。
    *   **《Out of Sight but Not Out of Mind》**：关注动态世界模型的长时序记忆机制。

*   **表征学习与基础模型研究者：**
    *   **《No Hard Negatives Required...》**：深入理解对比学习理论的可能演进。
    *   **《MuRF》**：探索释放视觉基础模型多尺度潜力的具体方法。

---

**总结：** 本期论文显示，计算机视觉研究正沿着 **“更大更专的基础模型”、“更人性化的具身智能”** 和 **“更高效动态的3D感知”** 三条主线快速推进。研究范式从追求通用性能，逐步深入到个性化对齐、训练效率和解耦、以及长程记忆等核心机制创新。

---

## Table of Contents

1. [Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale](#2603.25040v1)
2. [WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching](#2603.24836v1)
3. [Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting](#2603.25745v1)
4. [MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models](#2603.25744v1)
5. [Vega: Learning to Drive with Natural Language Instructions](#2603.25741v1)
6. [Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](#2603.25740v1)
7. [MegaFlow: Zero-Shot Large Displacement Optical Flow](#2603.25739v1)
8. [PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference](#2603.25730v1)
9. [No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](#2603.25722v1)
10. [Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models](#2603.25716v1)

---

## Papers

<a id='2603.25040v1'></a>
## [Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale](https://arxiv.org/abs/2603.25040v1)

**Authors:** Yicheng Zou, Dongsheng Zhu, Lin Zhu, Tong Zhu, Yunhua Zhou, Peiheng Zhou, Xinyu Zhou, Dongzhan Zhou, Zhiwang Zhou, Yuhao Zhou, Bowen Zhou, Zhanping Zhong, Zhijie Zhong, Haiteng Zhao, Penghao Zhao, Xiaomeng Zhao, Zhiyuan Zhao, Yechen Zhang, Jin Zhang, Wenwei Zhang, Hongjie Zhang, Zhuo Zhang, Wenlong Zhang, Bo Zhang, Chao Zhang, Chen Zhang, Yuhang Zang, Fei Yuan, Jiakang Yuan, Jiashuo Yu, Jinhui Yin, Haochen Ye, Qian Yao, Bowen Yang, Danni Yang, Kaichen Yang, Ziang Yan, Jun Xu, Yicheng Xu, Wanghan Xu, Xuenan Xu, Chao Xu, Ruiliang Xu, Shuhao Xing, Long Xing, Xinchen Xie, Ling-I Wu, Zijian Wu, Zhenyu Wu, Lijun Wu, Yue Wu, Jianyu Wu, Wen Wu, Fan Wu, Xilin Wei, Qi Wei, Bingli Wang, Rui Wang, Ziyi Wang, Zun Wang, Yi Wang, Haomin Wang, Yizhou Wang, Lintao Wang, Yiheng Wang, Longjiang Wang, Bin Wang, Jian Tong, Zhongbo Tian, Huanze Tang, Chen Tang, Shixiang Tang, Yu Sun, Qiushi Sun, Xuerui Su, Qisheng Su, Chenlin Su, Demin Song, Jin Shi, Fukai Shang, Yuchen Ren, Pengli Ren, Xiaoye Qu, Yuan Qu, Jiantao Qiu, Yu Qiao, Runyu Peng, Tianshuo Peng, Jiahui Peng, Qizhi Pei, Zhuoshi Pan, Linke Ouyang, Wenchang Ning, Yichuan Ma, Zerun Ma, Ningsheng Ma, Runyuan Ma, Chengqi Lyu, Haijun Lv, Han Lv, Lindong Lu, Kuikun Liu, Jiangning Liu, Yuhong Liu, Kai Liu, Hongwei Liu, Zhoumianze Liu, Mengjie Liu, Ziyu Liu, Wenran Liu, Yang Liu, Liwei Liu, Kaiwen Liu, Junyao Lin, Junming Lin, Tianyang Lin, Dahua Lin, Jianze Liang, Linyang Li, Peiji Li, Zonglin Li, Zehao Li, Pengze Li, Guoyan Li, Lingkai Kong, Linglin Jing, Zhenjiang Jin, Feifei Jiang, Qian Jiang, Junhao Huang, Zixian Huang, Haian Huang, Zhouqi Hua, Han Hu, Linfeng Hou, Yinan He, Conghui He, Tianyao He, Xu Guo, Qipeng Guo, Aijia Guo, Yuzhe Gu, Lixin Gu, Jingyang Gong, Qiming Ge, Jiaye Ge, Songyang Gao, Jianfei Gao, Xinyu Fang, Caihua fan, Yue Fan, Yanhui Duan, Zichen Ding, Shengyuan Ding, Xuanlang Dai, Erfei Cui, Ganqu Cui, Pei Chu, Tao Chu, Guangran Cheng, Yu Cheng, Kai Chen, Yongkang Chen, Chiyu Chen, Guanzhou Chen, Qiaosheng Chen, Sitao Chen, Xin Chen, Haojiong Chen, Yicheng Chen, Weihan Cao, Yuhang Cao, Qinglong Cao, Lei Bai

**Published:** 2026-03-26

**Categories:** cs.LG, cs.CL, cs.CV

**Abstract:**

We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model. Scaling to this unprecedented size, the model delivers a comprehensive enhancement across both general and scientific domains. Beyond stronger reasoning and image-text understanding capabilities, its intelligence is augmented with advanced agent capabilities. Simultaneously, its scientific expertise has been vastly expanded to master over 100 specialized tasks across critical science fields, including chemistry, materials, life sciences, and earth sciences. Achieving this massive scale is made possible by the robust infrastructure support of XTuner and LMDeploy, which facilitates highly efficient Reinforcement Learning (RL) training at the 1-trillion parameter level while ensuring strict precision consistency between training and inference. By seamlessly integrating these advancements, Intern-S1-Pro further fortifies the fusion of general and specialized intelligence, working as a Specializable Generalist, demonstrating its position in the top tier of open-source models for general capabilities, while outperforming proprietary models in the depth of specialized scientific tasks.

**Analysis:**

这是一份针对《Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale》论文的方法分析报告。

---

### 1. 摘要翻译
我们推出了 Intern-S1-Pro，这是首个万亿参数规模的科学多模态基础模型。通过扩展至这一前所未有的规模，该模型在通用和科学领域均实现了全面提升。除了更强的推理和图文理解能力，其智能水平通过高级智能体能力得到增强，能够自主规划并执行复杂的科学工作流。同时，其科学专业知识库已极大扩展，掌握了跨越化学、材料、生命科学和地球科学等关键领域的100多种专业任务。这一巨大规模的实现得益于 XTuner 和 LMDeploy 的强大基础设施支持，它们在万亿参数级别上实现了高效的强化学习（RL）训练，并确保了训练与推理之间的严格精度一致性。通过无缝整合这些进步，Intern-S1-Pro 进一步巩固了通用与专用智能的融合，展示了其作为“可专业化的通才”（Specializable Generalist）在开源模型中的顶尖地位。

### 2. 方法动机分析
*   **驱动力**：科学领域的高度专业化、多学科交叉性以及对长尾知识的需求，要求模型在保持通用能力的同时，必须具备极高的任务适应深度。
*   **痛点**：现有模型在处理科学领域复杂的结构化数据（如化学式、蛋白质序列）时，往往因为数据分布偏移或缺乏领域内逻辑一致性，导致性能退化或灾难性遗忘。
*   **研究假设**：一个足够大的通用模型，若通过联合训练（Joint Training）和特定的架构设计，能够超越针对单一任务微调的专用模型，实现智能的协同进化。

### 3. 方法设计详解
*   **流程总结**：
    1.  **专家扩展（Architecture Expansion）**：基于 Intern-S1 扩展至 1T MoE 模型。
    2.  **分组路由（Grouped Routing）**：将专家划分为与设备映射一致的组，在组内进行 Top-K 选择，强制实现绝对的负载均衡。
    3.  **STE 梯度估计**：在 MoE 路由中引入 Straight-Through Estimator，通过在后向传播中利用稠密 Softmax 分布，确保每个专家都能接收到梯度反馈，解决稀疏路由带来的训练不稳定问题。
    4.  **科学数据转换与隔离**：通过模板化处理将科学结构化数据转化为描述性文本，并通过 System Prompt Isolation 实现领域内外的语境隔离。
*   **模型结构**：包含 Vision Transformer (ViT)、专用时间序列编码器、以及引入 FoPE（Fourier Position Encoding）的 Transformer 骨干网络。
*   **关键公式意义**：$p_i^{\text{STE}}$ 结合了前向稀疏选择（保证计算效率）和后向梯度稠密流动（保证参数充分训练），实现了性能与训练稳定性的双赢。

### 4. 方法对比分析
*   **本质区别**：与传统模型相比，Intern-S1-Pro 并非简单的“堆参数”，而是通过 Grouped Routing 和 STE 解决了超大规模 MoE 在训练中的 load imbalance 和梯度缺失问题。
*   **创新贡献**：提出了“可专业化的通才”范式，证明了在科学任务上，联合训练优于单一任务微调，且通过 FoPE 更好地捕获了物理信号的连续波形特征。
*   **适用场景**：极高复杂度、多模态科学数据处理，如蛋白质预测、材料科学设计及高分辨远程遥感。

### 5. 实验分析（精简版）
*   **关键结论**：在 100+ 科学任务中，Intern-S1-Pro 显著优于 GPT-5.2 等专用/闭源模型（SciReasoner 55.5 vs 13.6）。在相同数据下，其生物任务性能全面超过 Biology-Instruction 专用模型。
*   **优势**：极强的多领域泛化能力与科学推理深度，训练稳定性高。
*   **局限**：万亿参数规模对算力基础设施依赖极高，训练成本巨大。

### 6. 实用指南
*   **开源情况**：模型已开源（https://huggingface.co/internlm/Intern-S1-Pro）。
*   **实现细节**：建议关注其 FP8 混合精度 RL 训练策略，这是支撑万亿规模模型训练不崩溃的核心工程技巧。
*   **迁移可能**：Grouped Routing 架构可直接迁移至任何大规模分布式 MoE 训练任务中，尤其是当面临特定硬件映射约束时。

### 7. 总结
*   **核心思想**：万亿级架构下的通用智能与科学专业知识的深度融合。
*   **速记版pipeline**：
    1. 引入分组路由实现算力负载均衡；
    2. 采用梯度估计器确保专家高效训练；
    3. 科学数据模板化与系统提示隔离；
    4. 联合训练实现科学专业知识涌现。

**Key Findings:**

- We introduce Intern-S1-Pro, the first one-trillion-parameter scientific multimodal foundation model.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25040v1)
- [arXiv](https://arxiv.org/abs/2603.25040v1)

---

<a id='2603.24836v1'></a>
## [WAFT-Stereo: Warping-Alone Field Transforms for Stereo Matching](https://arxiv.org/abs/2603.24836v1)

**Authors:** Yihan Wang, Jia Deng

**Published:** 2026-03-25

**Categories:** cs.CV

**Abstract:**

We introduce WAFT-Stereo, a simple and effective warping-based method for stereo matching. WAFT-Stereo demonstrates that cost volumes, a common design used in many leading methods, are not necessary for strong performance and can be replaced by warping with improved efficiency. WAFT-Stereo ranks first on ETH3D, KITTI and Middlebury public benchmarks, reducing the zero-shot error by 81% on ETH3D benchmark, while being 1.8-6.7x faster than competitive methods. Code and model weights are available at https://github.com/princeton-vl/WAFT-Stereo.

**Analysis:**

作为计算机视觉领域的专家，我对这篇关于 **WAFT-Stereo** 的论文分析如下：

### 1. 论文核心贡献总结
WAFT-Stereo 提出了一种革新性的立体匹配范式，证明了在高性能立体匹配任务中，传统的“代价卷”（Cost Volume）结构并非不可或缺。通过引入基于“独立变换（Warping-Alone）”的架构，该方法在保持顶尖精度的同时，显著提升了推理效率，并在多个主流基准测试（ETH3D, KITTI, Middlebury）中实现了 SOTA 性能。

### 2. 关键创新与方法论
*   **范式转换（Paradigm Shift）：** 过去十余年，立体匹配领域（如 PSMNet, RAFT）主要依赖构建代价卷来计算视差匹配度。WAFT-Stereo 的核心创新在于**抛弃了代价卷**，转而通过特征变换（Warping）直接进行匹配预测。
*   **效率优先的设计：** 由于省去了构建和处理高维代价卷的昂贵计算开销，该模型在推理速度上比同类方法快 1.8 到 6.7 倍。
*   **零样本迁移能力（Zero-shot Generalization）：** 摘要中提到的“零样本误差降低 81%”说明该模型具有极强的泛化能力，能够很好地处理未见过的数据分布，这通常意味着其学习到的特征表示更具鲁棒性。

### 3. 对领域的潜在影响
*   **打破“代价卷”依赖：** 这篇论文挑战了该领域的主流范式，可能会引发学术界对于“特征提取 vs. 匹配机制”重要性的重新思考，推动研究者寻找更轻量、更直接的匹配算法。
*   **部署价值：** 对于实时性要求极高的边缘计算场景（如自动驾驶、AR/VR 设备），WAFT-Stereo 提供了极具吸引力的解决方案，因为其兼顾了高精度与低延迟。

### 4. 潜在的应用领域与受益方向
*   **机器人与自动驾驶：** 实时深度估计是避障和导航的基础，WAFT-Stereo 的高效性使其非常适合嵌入式系统。
*   **增强现实（AR/VR）：** 头戴式设备对功耗和延迟非常敏感，该模型可以实现更高精度的实时环境感知。
*   **多视图立体视觉（MVS）：** 类似的 Warping 思想可以推广到 MVS 中，用于处理多视角图像，进一步提升 3D 重建的效率。

### 5. 推断出的局限性（专家视角）
虽然摘要展示了极其亮眼的结果，但从算法逻辑上，我们可以推测其可能存在以下局限：
*   **遮挡区域处理：** 没有代价卷的显式建模，模型在处理严重的遮挡（Occlusion）或无纹理区域时，可能依赖于极强的特征提取器（如 Transformer-based Backbone）进行推断，其对复杂纹理缺失场景的鲁棒性仍需验证。
*   **大视差范围：** 对于大视差范围（Large Disparity），Warping 操作可能需要多尺度金字塔的支持，这可能会增加训练的复杂度或对显存的非线性需求。
*   **对图像对齐的依赖：** 基于 Warping 的方法通常对相机校准（Rectification）的精度比基于代价卷的方法更敏感，如果图像预处理存在偏差，可能会直接影响匹配质量。

---
**总结点评：**
WAFT-Stereo 的价值在于其**“减法哲学”**。在深度学习领域普遍追求“更深、更重”的今天，这篇论文通过重新审视底层逻辑，证明了更简单的架构往往能带来更强的性能，这是典型的“大道至简”型研究，极具工业应用前景。

**Key Findings:**

- We introduce WAFT-Stereo, a simple and effective warping-based method for stereo matching.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.24836v1)
- [arXiv](https://arxiv.org/abs/2603.24836v1)

---

<a id='2603.25745v1'></a>
## [Less Gaussians, Texture More: 4K Feed-Forward Textured Splatting](https://arxiv.org/abs/2603.25745v1)

**Authors:** Yixing Lao, Xuyang Bai, Xiaoyang Wu, Nuoyuan Yan, Zixin Luo, Tian Fang, Jean-Daniel Nahmias, Yanghai Tsin, Shiwei Li, Hengshuang Zhao

**Published:** 2026-03-26

**Categories:** cs.CV

**Abstract:**

Existing feed-forward 3D Gaussian Splatting methods predict pixel-aligned primitives, leading to a quadratic growth in primitive count as resolution increases. This fundamentally limits their scalability, making high-resolution synthesis such as 4K intractable. We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier. By predicting compact Gaussian primitives coupled with per-primitive textures, LGTM decouples geometric complexity from rendering resolution. This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives. Project page: https://yxlao.github.io/lgtm/

**Analysis:**

作为计算机视觉领域的专家，我对这篇题为《LGTM: Less Gaussians, Texture More》的论文分析如下：

### 1. 论文核心贡献总结
该论文提出了一种名为 **LGTM** 的前馈（Feed-Forward）3D高斯泼溅（3DGS）框架，旨在解决现有方法在处理高分辨率（如4K）渲染时因图元数量呈二次方增长而导致的不可扩展性问题。通过将几何结构与视觉纹理进行解耦，该方法在无需逐场景优化的情况下，实现了高效、高质量的4K新视角合成，并在大幅减少高斯图元数量的同时提升了渲染性能。

### 2. 关键创新与方法论
*   **解耦几何与分辨率**：传统方法（如Pixel-aligned primitives）将每个像素或区域映射为高斯图元，导致分辨率越高，所需图元越多。LGTM的核心创新在于**“紧凑几何+逐图元纹理”（Compact Gaussians + Per-primitive Textures）**的表征范式。
*   **非依赖像素的建模**：通过预测带有纹理属性的紧凑型高斯图元，几何复杂性不再直接受制于输出分辨率。这种设计使得模型能够像传统图形学中的纹理映射一样，以较少的几何代理（Primitives）支撑高精度的细节表现。
*   **前馈推断**：完全舍弃了传统3DGS所需的繁琐的逐场景优化（Per-scene Optimization），使其具备了真正的实时新视角生成能力。

### 3. 对领域的潜在影响
*   **突破前馈式3DGS的“分辨率瓶颈”**：该研究解决了当前大规模场景合成中“分辨率越高、内存和计算开销越大”的矛盾，是通往大规模、高清晰度虚拟现实和数字孪生技术的关键一步。
*   **范式转换**：它推动了从“海量高斯图元拟合”向“结构化与纹理增强表示”的转型。这种思路与传统计算机图形学中的渲染管线（如网格+纹理贴图）有一定的内在联系，可能引领一种结合神经渲染与传统渲染优势的新方向。

### 4. 相关受益领域与应用
*   **虚拟现实（VR/AR）与元宇宙**：极高分辨率（4K+）的实时渲染是实现沉浸式体验的前提，LGTM直接解决了这一痛点。
*   **自动驾驶仿真**：自动驾驶系统需要实时生成高分辨率的道路环境场景，该技术可显著提升合成效率和细节保真度。
*   **大规模场景重建与数字化**：在城市级或建筑级的三维重建中，该方法可以极大地降低数据存储量和渲染负担。
*   **游戏开发**：为游戏引擎中的实时资产生成提供了一种基于前馈神经渲染的高效替代方案。

### 5. 可推断的局限性
*   **纹理映射的复杂性**：引入“逐图元纹理”后，如何在高斯球体的空间畸变下保持纹理的采样质量（如纹理过滤、抗锯齿），可能是一个技术挑战。
*   **动态场景泛化**：虽然解决了分辨率问题，但前馈方法在处理剧烈运动或复杂遮挡时的鲁棒性仍需进一步验证。
*   **模型训练成本**：虽然推理阶段不需要优化，但要实现这一效果，模型在前馈训练阶段可能需要极大的数据集支撑及复杂的损失函数设计（如为了平衡几何准确性和纹理细节）。
*   **几何覆盖的局限性**：在极高频细节（如头发、复杂的镂空结构）上，若图元数量过少，能否仅靠纹理完全补偿几何信息的缺失，仍需观察其在极端复杂几何下的表现。

**总结：** LGTM 是一篇极具实用价值的论文，它通过巧妙的解耦设计，将 3DGS 从“高昂的优化开销”带入了“高效的前馈推理”时代，是当前神经渲染领域向工业级应用落地的重要推动力。

**Key Findings:**

- We introduce LGTM (Less Gaussians, Texture More), a feed-forward framework that overcomes this resolution scaling barrier.
- This approach enables high-fidelity 4K novel view synthesis without per-scene optimization, a capability previously out of reach for feed-forward methods, all while using significantly fewer Gaussian primitives.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25745v1)
- [arXiv](https://arxiv.org/abs/2603.25745v1)

---

<a id='2603.25744v1'></a>
## [MuRF: Unlocking the Multi-Scale Potential of Vision Foundation Models](https://arxiv.org/abs/2603.25744v1)

**Authors:** Bocheng Zou, Mu Cai, Mark Stanley, Dingfu Lu, Yong Jae Lee

**Published:** 2026-03-26

**Categories:** cs.CV

**Abstract:**

Vision Foundation Models (VFMs) have become the cornerstone of modern computer vision, offering robust representations across a wide array of tasks. While recent advances allow these models to handle varying input sizes during training, inference typically remains restricted to a single, fixed scale. This prevalent single-scale paradigm overlooks a fundamental property of visual perception: varying resolutions offer complementary inductive biases, where low-resolution views excel at global semantic recognition and high-resolution views are essential for fine-grained refinement. In this work, we propose Multi-Resolution Fusion (MuRF), a simple yet universally effective strategy to harness this synergy at inference time. Instead of relying on a single view, MuRF constructs a unified representation by processing an image at multiple resolutions through a frozen VFM and fusing the resulting features. The universality of MuRF is its most compelling attribute. It is not tied to a specific architecture, serving instead as a fundamental, training-free enhancement to visual representation. We empirically validate this by applying MuRF to a broad spectrum of critical computer vision tasks across multiple distinct VFM families - primarily DINOv2, but also demonstrating successful generalization to contrastive models like SigLIP2.

**Analysis:**

### 1. 摘要翻译
视觉基础模型（VFMs）已成为计算机视觉的基石，但在推理时通常局限于单一的固定分辨率。这种单尺度范式忽略了视觉感知的一个基本属性：不同分辨率提供互补的归纳偏置，其中低分辨率视图擅长全局语义识别，而高分辨率视图对细粒度精修至关重要。本文提出了多分辨率融合（MuRF），这是一种简单且通用的推理时策略，用于挖掘这种协同效应。MuRF不依赖单一视图，而是通过冻结的VFM处理多个分辨率的图像，并将得到的特征进行融合，构建统一表示。MuRF的通用性是其最显著的优势，它不绑定特定架构，而是作为一种通用的、无需训练的视觉表示增强手段。我们在DINOv2和SigLIP2等多种VFM系列上，针对多项关键计算机视觉任务验证了MuRF的有效性。

### 2. 方法动机分析
*   **驱动力**：旨在打破当前VFM推理中“单一尺度”的僵化限制，通过利用视觉信息在不同尺度下的互补性，提升模型对全局语义和局部细节的综合捕捉能力。
*   **现有方法痛点**：单尺度推理往往面临“两难”：低分辨率虽然全局一致性好但丢失局部细节；高分辨率虽有细节但容易引入噪声并破坏语义的一致性。
*   **研究假设**：通过在特征空间（而非输入空间）显式聚合多分辨率特征，可以实现各尺度优势的有机结合，且无需对冻结的VFM主干进行代价高昂的微调。

### 3. 方法设计详解
*   **流程总结**：
    1.  **输入金字塔构建**：将原图像 $x$ 调整为多个缩放因子 $S_{res} = \{s_1, s_2, \dots, s_k\}$ 对应的图像集。
    2.  **特征提取**：使用冻结的VFM编码器 $\Phi$ 对各缩放图像进行特征提取，得到 patch 级特征图 $\mathcal{F}_s$。
    3.  **上采样与对齐**：将各特征图 $\mathcal{F}_s$ 通过双线性插值上采样至统一的目标分辨率（通常为原图大小）。
    4.  **通道级串联（Concatenation）**：将对齐后的特征在通道维度进行串联，得到最终的 MuRF 表示 $\mathcal{F}_{MuRF}$。
*   **算法核心**：采用**通道级串联**而非加权平均或注意力机制。这是为了保留不同尺度特征的“严格独立性”，使下游轻量级头能够自适应选择不同尺度的响应，避免破坏语义特征的局部性。

### 4. 方法对比分析
*   **本质区别**：区别于传统FPN（需要特定训练）或简单的 tiling 拼接（破坏空间连续性），MuRF 是纯推理时的**即插即用**策略。
*   **创新贡献**：提出了一种无需训练（training-free）且与主干解耦的通用增强手段，证明了“多尺度融合”不仅能提升 dense prediction，还能在 MLLM 和工业异常检测中产生显著增益。

### 5. 实验分析（精简版）
*   **验证方法**：在 ADE20K、PASCAL VOC（分割）、NYU Depth V2、SUN RGB-D（深度估计）、MLLM（VQA）以及 MVTec AD（异常检测）上进行广泛测试。
*   **关键结果**：MuRF 在所有任务中均显著超越了最优单尺度基线，且在异常检测中达到了 state-of-the-art 水准。
*   **主要优势**：不仅性能提升显著，且因保持 Backbone 冻结，推理成本可控，下游微调效率极高。
*   **主要局限**：推理时需要多次前向传播，显存占用随尺度增加而增长。

### 6. 实用指南
*   **开源情况**：项目主页为 [MuRF-VFM.github.io](https://MuRF-VFM.github.io)，代码库 [MuRF-VFM](https://github.com/orgs/MuRF-VFM)。
*   **实现细节**：对于语义分割和深度估计，推荐至少使用 3 个尺度；对于 Anomaly Detection，建议 5 个尺度以捕捉从微观到宏观的异常。需确保下游 Head 能够适应通道维度 $D = |S_{res}| \times d$ 的变化。
*   **迁移可能**：该方法极易迁移到任何基于 ViT 的 VFM 架构，只需替换主干模型并调整 Resizing 因子即可。

### 7. 总结
*   **核心思想**：通过多尺度特征通道串联，实现冻结模型下的多尺度信息协同。
*   **速记版pipeline**：
    1. 生成不同尺寸输入图；
    2. 输入 VFM 提取特征；
    3. 统一分辨率上采样；
    4. 特征通道拼接输出。

**Key Findings:**

- In this work, we propose Multi-Resolution Fusion (MuRF), a simple yet universally effective strategy to harness this synergy at inference time.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25744v1)
- [arXiv](https://arxiv.org/abs/2603.25744v1)

---

<a id='2603.25741v1'></a>
## [Vega: Learning to Drive with Natural Language Instructions](https://arxiv.org/abs/2603.25741v1)

**Authors:** Sicheng Zuo, Yuxuan Li, Wenzhao Zheng, Zheng Zhu, Jie Zhou, Jiwen Lu

**Published:** 2026-03-26

**Categories:** cs.CV, cs.AI, cs.RO

**Abstract:**

Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.

**Analysis:**

这是一份关于论文 **"Vega: Learning to Drive with Natural Language Instructions"** 的深度技术分析。

---

### 1. 摘要翻译
视觉-语言-动作模型已将语言整合进自动驾驶决策，但现有流程多局限于场景描述或推理，缺乏响应多样化指令进行个性化驾驶的灵活性。为此，我们首先构建了大规模驾驶数据集 **InstructScene**，包含约10万条带有多样化驾驶指令与对应轨迹的标注场景。随后，我们提出了统一的视觉-语言-世界-动作模型 **Vega**，用于指令引导的生成与规划。我们采用自回归范式处理视觉输入与语言指令，并利用扩散范式生成未来预测（世界模型）与轨迹（动作）。通过联合注意力机制实现模态间交互，并使用各模态独立的投影层以提升能力。实验表明，Vega不仅在规划性能上达到顶尖，更展现出强大的指令遵循能力，为智能化、个性化的驾驶系统奠定了基础。

### 2. 方法动机分析
*   **驱动力**：现有的自动驾驶模型要么模仿平均驾驶策略，要么只能理解简单的导航指令（如“直行”、“左转”）。作者旨在打造一个能像人类一样，根据用户开放式、灵活的指令（例如“超车以赶上绿灯”）进行决策的智能体。
*   **痛点**：高维视觉-指令输入与低维动作预测之间存在严重的信息鸿沟，导致模型难以学习到高层指令与低层动作之间的泛化映射。
*   **研究假设**：通过显式引入“未来视觉生成”作为辅助任务，可以提供密集的、像素级的监督信号，强迫模型学习指令、动作与环境动态之间的因果关系，从而弥补仅有稀疏动作标注带来的性能短板。

### 3. 方法设计详解
*   **Pipeline**：
    1.  **输入处理**：历史观察 $I_{t-T \dots t}$、动作 $A_{t-T \dots t-1}$ 及用户指令 $L_t$ 被拼接为序列。
    2.  **理解与规划（Autoregressive）**：使用基于 Qwen2.5-72B 的理解 Transformer 处理视觉与指令 tokens。
    3.  **动作生成（Diffusion）**：将动作视为扩散过程的目标，预测序列 $A_t \dots A_{t+N-1}$。
    4.  **未来预测（World Modeling）**：根据预测的动作，利用扩散模型生成未来的视觉帧 $I_{t+K}$，提供监督信号。
*   **关键架构**：采用了 **Mixture-of-Transformers (MoT)** 设计。与 MoE 不同，MoT 将每个模块（理解、动作、生成）的完整参数（Attention+FFN）独立，有效解耦了不同任务，提升了模型容量与收敛速度。
*   **训练策略**：为了避免 Inference 时的“训练-测试不一致”问题，作者在训练时采用了“复制延迟（Latent Duplication）”策略：将输入的 Noisy Latent 副本用于预测，而 Clean Latent 用于下游注意力计算，确保了因果遮蔽的正确性。

### 4. 方法对比分析
*   **本质区别**：现有的 VLA 模型多将语言作为辅助，而 Vega 将“未来视觉生成”视为核心，构建了一个闭环的 World-Action 联合学习架构。
*   **创新贡献**：提出了 InstructScene 数据集；设计了结合 autoregressive 和 diffusion 的 MoT 架构，成功实现指令到动作及视觉的一体化生成。
*   **适用场景**：复杂、高动态的城市驾驶环境，特别需要根据突发指令进行灵活变道的场景。

### 5. 实验分析
*   **关键结果**：在 NAVSIM v2 基准上，Vega 的 EPDMS 达到 89.4，显著优于 DriveVLA-W0 和其他 SOTA 方法。
*   **主要优势**：极强的指令遵循泛化能力；未来预测任务显著提升了规划轨迹的合理性。
*   **主要局限**：单模态（单目）输入在某些复杂的多视角场景下仍受限于视野，依赖于高质量、大规模的指令标注数据。

### 6. 实用指南
*   **开源情况**：代码已开源（https://github.com/zuosc19/Vega）。
*   **关键细节**：
    *   **数据构建**：利用大模型（Qwen2.5-VL）+ 规则方法自动生成指令是实现 InstructScene 的核心。
    *   **训练技巧**：将图像与动作序列进行“交替式”输入，比单纯的拼接更容易收敛。
*   **迁移可能**：该架构（统一理解+生成+规划）可直接迁移至机器人操作、具身智能控制等领域，通过引入环境生成的奖励机制进行强化学习。

### 7. 总结
*   **核心思想**：利用未来视觉生成作为密集监督信号，强化动作规划的指令一致性。
*   **速记版pipeline**：
    1. 视觉输入与指令编码为序列；
    2. 用理解Transformer处理多模态信息；
    3. 扩散模型联合预测轨迹动作；
    4. 生成未来图像验证预测合理性。

**Key Findings:**

- Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25741v1)
- [arXiv](https://arxiv.org/abs/2603.25741v1)

---

<a id='2603.25740v1'></a>
## [Drive My Way: Preference Alignment of Vision-Language-Action Model for Personalized Driving](https://arxiv.org/abs/2603.25740v1)

**Authors:** Zehao Wang, Huaide Jiang, Shuaiwu Dong, Yuping Wang, Hang Qiu, Jiachen Li

**Published:** 2026-03-26

**Categories:** cs.RO, cs.AI, cs.CV, cs.LG, cs.MA

**Abstract:**

Human driving behavior is inherently personal, which is shaped by long-term habits and influenced by short-term intentions. Individuals differ in how they accelerate, brake, merge, yield, and overtake across diverse situations. However, existing end-to-end autonomous driving systems either optimize for generic objectives or rely on fixed driving modes, lacking the ability to adapt to individual preferences or interpret natural language intent. To address this gap, we propose Drive My Way (DMW), a personalized Vision-Language-Action (VLA) driving framework that aligns with users' long-term driving habits and adapts to real-time user instructions. DMW learns a user embedding from our personalized driving dataset collected across multiple real drivers and conditions the policy on this embedding during planning, while natural language instructions provide additional short-term guidance. Closed-loop evaluation on the Bench2Drive benchmark demonstrates that DMW improves style instruction adaptation, and user studies show that its generated behaviors are recognizable as each driver's own style, highlighting personalization as a key capability for human-centered autonomous driving. Our data and code are available at https://dmw-cvpr.github.io/.

**Analysis:**

# 论文深度分析：Drive My Way (DMW)

### 1. 摘要翻译
人类驾驶行为是个性化的，受长期习惯和短期意图影响。然而，现有的端到端自动驾驶系统通常优化通用目标或依赖固定驾驶模式，缺乏适应个人偏好或解读自然语言意图的能力。为此，我们提出了“Drive My Way (DMW)”，一个个性化的视觉-语言-动作 (VLA) 驾驶框架，旨在对齐用户的长期驾驶习惯并适应实时指令。DMW 通过从个性化驾驶数据集学习用户嵌入来规划策略，并结合自然语言提供短期引导。Bench2Drive 上的闭环评估表明，DMW 提升了风格指令的适应性；用户研究验证了其驾驶行为能被识别为特定驾驶员的风格，突显了个性化对于以人为本的自动驾驶的关键作用。

### 2. 方法动机分析
*   **驱动力**：实现真正“千人千面”的自动驾驶，让车辆不仅能安全行驶，还能根据乘客的长短期需求（如“赶时间”或“身体不适”）动态调整驾驶策略。
*   **现有痛点**：现有方法要么过于通用（缺乏个性），要么局限于简单的预设模式（如“运动/舒适”），且无法处理复杂的实时自然语言指令或理解隐含的长期驾驶习惯。
*   **研究假设**：驾驶行为可以分解为：基于用户历史数据的长期偏好（静态嵌入）和基于实时自然语言的短期意图（动态修正），两者通过一个统一的VLA框架进行融合与对齐。

### 3. 方法设计详解
*   **流程总结**：
    1.  **用户偏好编码**：利用长短期偏好编码器，处理用户历史轨迹和个人档案，通过对比学习（InfoNCE）生成用户嵌入 $z^m_p$。
    2.  **VLA策略规划**：以SimLingo为基础，输入图像、导航信息、指令及 $z^m_p$，生成基础动作。
    3.  **动态奖励微调**：引入GRPO（组相对策略优化），通过预训练的大模型（LLM）根据场景和指令动态生成权重，对策略进行微调。
    4.  **残差动作修正**：利用残差解码器预测离散的动作调整（速度/转向），结合PID控制器输出最终控制量。
*   **模型结构**：核心是SimLingo VLA主干，通过LoRA适配器进行轻量化个性化定制，配合残差解码器实现细粒度动作偏移。
*   **关键算法解释**：
    *   **InfoNCE对比学习**：强制同一驾驶员在不同轨迹下的嵌入靠近，不同驾驶员的嵌入远离，从而精准捕捉个人特征。
    *   **Style-Aware Reward**：将LLM的语义推理引入奖励函数，动态调整安全性、舒适性、效率的权重，实现了对驾驶风格的量化控制。

### 4. 方法对比分析
*   **本质区别**：DMW将长期风格学习与短期语言指令适应解耦，通过动态调整奖励权重来改变驾驶逻辑，而非简单的模版切换。
*   **创新贡献**：构建了首个多模态个性化驾驶数据集(PDD)；提出了基于LLM辅助的动态奖励机制，实现了更符合人类语义理解的驾驶风格调整。
*   **适用场景**：复杂动态交互场景（路口、汇入、避障等），特别适合需要高度自定义舒适度或特定驾驶节奏的用户体验场景。

### 5. 实验分析
*   **关键结果**：在Bench2Drive基准下，DMW在保持高成功率的同时，大幅提升了对不同风格指令的适应能力，其生成的动作轨迹与人类驾驶员的AS（对齐分数）显著高于基线模型。
*   **优势**：极强的风格泛化能力，即使在未见过的驾驶员数据上也能表现出一致的风格偏好。
*   **局限**：对计算资源有一定要求（依赖LLM推理辅助），在极度拥堵场景下可能为了安全牺牲一定的灵活性。

### 6. 实用指南
*   **开源情况**：已开源（详见官网 https://dmw-cvpr.github.io/）。
*   **实现建议**：微调时需关注残差解码器的训练，确保基础策略（VLA）能够稳健地与残差结合；数据预处理中的“对齐驾驶轨迹”对于AS分数至关重要。
*   **迁移可能**：该框架的“偏好嵌入+动态奖励对齐”模式，可直接迁移至机器人操控、智能家居等需要个性化偏好对齐的决策任务中。

### 7. 总结
*   **核心思想**：通过长短期偏好融合与动态语义奖励机制，实现个性化的自动驾驶行为。
*   **速记版pipeline**：
    1. 挖掘驾驶员历史行为特征；
    2. 输入自然语言实时指令；
    3. 动态调整安全与效率权重；
    4. 对模型基础动作进行残差微调。

**Key Findings:**

- To address this gap, we propose Drive My Way (DMW), a personalized Vision-Language-Action (VLA) driving framework that aligns with users' long-term driving habits and adapts to real-time user instructions.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25740v1)
- [arXiv](https://arxiv.org/abs/2603.25740v1)

---

<a id='2603.25739v1'></a>
## [MegaFlow: Zero-Shot Large Displacement Optical Flow](https://arxiv.org/abs/2603.25739v1)

**Authors:** Dingxi Zhang, Fangjinhua Wang, Marc Pollefeys, Haofei Xu

**Published:** 2026-03-26

**Categories:** cs.CV

**Abstract:**

Accurate estimation of large displacement optical flow remains a critical challenge. Existing methods typically rely on iterative local search or/and domain-specific fine-tuning, which severely limits their performance in large displacement and zero-shot generalization scenarios. To overcome this, we introduce MegaFlow, a simple yet powerful model for zero-shot large displacement optical flow. Rather than relying on highly complex, task-specific architectural designs, MegaFlow adapts powerful pre-trained vision priors to produce temporally consistent motion fields. In particular, we formulate flow estimation as a global matching problem by leveraging pre-trained global Vision Transformer features, which naturally capture large displacements. This is followed by a few lightweight iterative refinements to further improve the sub-pixel accuracy. Extensive experiments demonstrate that MegaFlow achieves state-of-the-art zero-shot performance across multiple optical flow benchmarks. Moreover, our model also delivers highly competitive zero-shot performance on long-range point tracking benchmarks, demonstrating its robust transferability and suggesting a unified paradigm for generalizable motion estimation. Our project page is at: https://kristen-z.github.io/projects/megaflow.

**Analysis:**

### 1. 摘要翻译
大规模位移光流估计仍是关键挑战。现有方法通常依赖局部迭代搜索或特定领域微调，严重限制了其在大位移和零样本泛化场景下的表现。为此，我们引入了MegaFlow，这是一个简单且强大的零样本大规模位移光流模型。MegaFlow不依赖复杂的任务特定架构，而是通过适配强大的预训练视觉先验来生成时间一致的运动场。具体而言，我们将光流估计公式化为全局匹配问题，利用预训练的全局视觉Transformer特征自然地捕获大位移，随后通过轻量级迭代细化进一步提升亚像素精度。实验表明，MegaFlow在多个光流基准测试上实现了最先进的零样本性能，并在长距离点追踪任务上表现优异，为通用运动估计提供了一个统一的范式。

---

### 2. 方法动机分析
*   **驱动力**：旨在克服传统光流模型对特定领域微调的依赖，构建一个具备强大零样本泛化能力的大位移运动估计基础模型。
*   **现有痛点**：基于局部迭代搜索（如RAFT系列）的方法在处理跨度巨大的空间位移时，容易陷入局部最优或产生匹配歧义，难以跨域推广。
*   **研究假设**：通过引入预训练的全局视觉Transformer特征作为强大的几何先验，可以将光流估计从“局部局部搜索”转换为“全局匹配+局部细化”的范式，从而根本上解决大位移问题。

---

### 3. 方法设计详解
*   **Pipeline**：
    1.  **特征提取与融合**：利用冻结的DINOv2提取多尺度特征，并通过轻量级CNN编码器补充空间细节，经由DPT-style融合头生成包含全局语义与局部结构的融合特征图。
    2.  **全局匹配**：计算帧间特征的全局全对相关性（All-pairs correlation），通过Softmax归一化转换为概率分布，利用坐标网格的期望来计算初始的大位移流场。
    3.  **局部递归细化**：利用ConvNeXt模块和时间注意力机制（Temporal Attention），在初始流场基础上，对局部特征进行多步迭代细化。
*   **模型核心逻辑**：将大位移处理交给全局匹配（Global Matching），将亚像素精度的保障交给轻量级递归模块，实现了全局与局部的解耦优化。

---

### 4. 方法对比分析
*   **本质区别**：不同于RAFT类方法的“局部相关性迭代更新”，MegaFlow强调“全局特征匹配作为先验，局部细化作为修正”，通过Transformer的全局注意机制处理长程依赖。
*   **创新贡献**：成功将预训练视觉基础模型（DINOv2）适配于动态光流估计，且通过一个简单的统一架构同时解决了光流与点追踪（TAP）问题。
*   **适用场景**：极端大位移、复杂光影变化及长程视频序列的任务。

---

### 5. 实验分析（精简版）
*   **验证方法**：在Sintel和KITTI数据集上进行严格的零样本（Zero-shot）评测。
*   **关键结论**：在大位移区间（$s_{40+}$）性能提升显著，在Sintel Clean上EPE达到0.91，大幅优于同类方法。
*   **优势**：极强的泛化能力，无需针对特定数据集微调。
*   **局限**：对长序列处理时的计算开销随帧数增加而提升。

---

### 6. 实用指南
*   **开源情况**：项目主页：https://kristen-z.github.io/projects/megaflow/
*   **实现细节**：推理阶段默认使用T=4帧。训练需注意Stage阶段的课程学习，先训练基础特征，再引入时序注意力。
*   **迁移建议**：该架构中全局匹配的设计可直接迁移至任何视频语义匹配任务。若要处理超长视频，建议采用滑动窗口推理策略。

---

### 7. 总结
*   **核心思想**：全局匹配捕捉大位移，递归修正保障亚像素级细节。
*   **速记版pipeline**：
    1. 多帧特征提取融合
    2. 全局匹配确定初始大位移
    3. 时序注意力驱动的循环细化
    4. 产生鲁棒一致的流场/轨迹

**Key Findings:**

- To overcome this, we introduce MegaFlow, a simple yet powerful model for zero-shot large displacement optical flow.
- Extensive experiments demonstrate that MegaFlow achieves state-of-the-art zero-shot performance across multiple optical flow benchmarks.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25739v1)
- [arXiv](https://arxiv.org/abs/2603.25739v1)

---

<a id='2603.25730v1'></a>
## [PackForcing: Short Video Training Suffices for Long Video Sampling and Long Context Inference](https://arxiv.org/abs/2603.25730v1)

**Authors:** Xiaofeng Mao, Shaohao Rui, Kaining Ying, Bo Zheng, Chuanhao Li, Mingmin Chi, Kaipeng Zhang

**Published:** 2026-03-26

**Categories:** cs.CV, cs.AI

**Abstract:**

Autoregressive video diffusion models have demonstrated remarkable progress, yet they remain bottlenecked by intractable linear KV-cache growth, temporal repetition, and compounding errors during long-video generation. To address these challenges, we present PackForcing, a unified framework that efficiently manages the generation history through a novel three-partition KV-cache strategy. Specifically, we categorize the historical context into three distinct types: (1) Sink tokens, which preserve early anchor frames at full resolution to maintain global semantics; (2) Mid tokens, which achieve a massive spatiotemporal compression (32x token reduction) via a dual-branch network fusing progressive 3D convolutions with low-resolution VAE re-encoding; and (3) Recent tokens, kept at full resolution to ensure local temporal coherence. To strictly bound the memory footprint without sacrificing quality, we introduce a dynamic top-$k$ context selection mechanism for the mid tokens, coupled with a continuous Temporal RoPE Adjustment that seamlessly re-aligns position gaps caused by dropped tokens with negligible overhead. Empowered by this principled hierarchical context compression, PackForcing can generate coherent 2-minute, 832x480 videos at 16 FPS on a single H200 GPU. It achieves a bounded KV cache of just 4 GB and enables a remarkable 24x temporal extrapolation (5s to 120s), operating effectively either zero-shot or trained on merely 5-second clips. Extensive results on VBench demonstrate state-of-the-art temporal consistency (26.07) and dynamic degree (56.25), proving that short-video supervision is sufficient for high-quality, long-video synthesis. https://github.com/ShandaAI/PackForcing

**Analysis:**

这份对 **PackForcing** 的深度分析报告如下：

### 1. 摘要翻译
自回归视频扩散模型在长视频生成中面临KV缓存线性增长、时间重复和累积误差等瓶颈。为此，我们提出了 PackForcing，一种通过新型三分区KV缓存策略高效管理生成历史的统一框架。该框架将历史上下文分为：(1) 保持全分辨率以维持全局语义的“Sink tokens”；(2) 通过双分支网络进行约32倍时空压缩的“Mid tokens”；(3) 保持全分辨率以确保局部一致性的“Recent tokens”。通过引入动态top-k选择机制和持续的Temporal RoPE调整，PackForcing 能在单块H200 GPU上以受控内存（~4GB）生成长达2分钟的视频，实现24倍的时间外推，并在VBench上达到业界领先的性能。

### 2. 方法动机分析
- **驱动力**：在长视频生成的自回归范式中，如何在有限的显存限制下，既能存储足够长的上下文以避免语义漂移，又能保持高效的推理速度。
- **痛点**：现有方法要么通过滑动窗口/丢弃机制（FIFO）导致长程记忆丢失，要么因KV缓存无限增长导致显存耗尽。
- **研究假设**：视频生成的注意力分布具有稀疏性，且早期锚点（Sink）对语义稳定至关重要，因此可以通过三分区结构进行差异化管理。

### 3. 方法设计详解
- **流程pipeline**：
    1. **Sink管理**：固定保留生成初期的前N个块作为全局语义锚点，永不剔除。
    2. **Mid压缩与路由**：将中间段历史通过“HR（3D CNN）+ LR（VAE重编码）”双分支网络压缩至原体积的1/128（约32倍Token减少）。通过计算查询-键（Q-K）亲和度，利用动态Top-K机制仅路由最相关的块进入注意力计算。
    3. **Recent缓存**：全分辨率保留最近生成的窗口，确保局部平滑过渡。
    4. **RoPE位置修正**：当Mid区旧块被剔除时，利用RoPE相乘性质，仅对Sink区的旋转位置进行增量平移，消除了因丢弃导致的位置断层。
- **模型结构**：HR分支捕捉细粒度细节，LR分支捕捉全局结构，二者相加融合。该压缩层与主网络进行端到端联合训练。

### 4. 方法对比分析
- **本质区别**：从传统的“全存”或“全删”转变为“分级存储与动态压缩”。
- **创新贡献**：
    1. 三分区KV缓存策略。
    2. 融合空间结构与全局语义的双分支压缩模块。
    3. 低开销的增量RoPE位置调整，解决动态缓存带来的不连续性。
- **适用场景**：适用于资源受限下的超长视频（分钟级） autoregressive 生成任务。

### 5. 实验分析
- **关键结论**：在120秒生成任务中，PackForcing 在动态程度（54.12）和时间一致性上表现最佳，解决了传统方法长程生成中出现的颜色漂移和语义崩坏问题。
- **优势**：显著提升长视频生成的稳定性，显存占用仅4GB左右。
- **局限**：目前的压缩比是固定设计的，尚未实现根据场景复杂度的完全自适应调整。

### 6. 实用指南
- **开源情况**：已发布项目主页：https://github.com/ShandaAI/PackForcing
- **实现细节**：建议关注 `Nsink=8` 和 `Ntop=16` 的参数设置；压缩层训练必须与主模型进行联合优化。
- **迁移可能**：该三分区缓存结构可直接迁移至其他基于 Transformer 的超长序列生成任务（如长文本摘要、长音频生成）。

### 7. 总结
- **核心思想**：分级缓存与动态压缩实现长视频记忆管理。
- **速记版pipeline**：
    1. 冻结开头作为语义锚点。
    2. 中间历史压缩并智能路由。
    3. 最近历史精细保留。
    4. 位置索引自动平滑修正。

**Key Findings:**

- To address these challenges, we present PackForcing, a unified framework that efficiently manages the generation history through a novel three-partition KV-cache strategy.
- To strictly bound the memory footprint without sacrificing quality, we introduce a dynamic top-$k$ context selection mechanism for the mid tokens, coupled with a continuous Temporal RoPE Adjustment that seamlessly re-aligns position gaps caused by dropped tokens with negligible overhead.
- Extensive results on VBench demonstrate state-of-the-art temporal consistency (26.07) and dynamic degree (56.25), proving that short-video supervision is sufficient for high-quality, long-video synthesis.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25730v1)
- [arXiv](https://arxiv.org/abs/2603.25730v1)

---

<a id='2603.25722v1'></a>
## [No Hard Negatives Required: Concept Centric Learning Leads to Compositionality without Degrading Zero-shot Capabilities of Contrastive Models](https://arxiv.org/abs/2603.25722v1)

**Authors:** Hai X. Pham, David T. Hoffmann, Ricardo Guerrero, Brais Martinez

**Published:** 2026-03-26

**Categories:** cs.CV, cs.LG

**Abstract:**

Contrastive vision-language (V&L) models remain a popular choice for various applications. However, several limitations have emerged, most notably the limited ability of V&L models to learn compositional representations. Prior methods often addressed this limitation by generating custom training data to obtain hard negative samples. Hard negatives have been shown to improve performance on compositionality tasks, but are often specific to a single benchmark, do not generalize, and can cause substantial degradation of basic V&L capabilities such as zero-shot or retrieval performance, rendering them impractical. In this work we follow a different approach. We identify two root causes that limit compositionality performance of V&Ls: 1) Long training captions do not require a compositional representation; and 2) The final global pooling in the text and image encoders lead to a complete loss of the necessary information to learn binding in the first place. As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder. With these two changes and simple auxiliary contrastive losses, we obtain SOTA performance on standard compositionality benchmarks, while maintaining or improving strong zero-shot and retrieval capabilities. This is achieved without increasing inference cost. We release the code for this work at https://github.com/SamsungLabs/concept_centric_clip.

**Analysis:**

### 1. 摘要翻译
对比视觉-语言（V&L）模型在学习组合性表示方面存在局限。现有方法通常通过生成困难负样本来缓解该问题，但这往往是针对特定基准的，且会导致零样本或检索等基础性能显著下降。本文提出一种新方法：识别到长训练标题和全局池化机制是限制组合性学习的根本原因。为解决此问题，我们提出：1）利用NLP工具提取短概念中心标题并与图像对齐；2）引入参数自由的跨模态注意力池化，从图像编码器中获取概念中心视觉嵌入。这两项改进及辅助对比损失，使模型在标准组合性基准上达到SOTA，同时保持强劲的零样本和检索性能，且无需增加推理成本。

### 2. 方法动机分析
*   **驱动力**：打破“硬负样本”依赖，解决对比学习中因依赖全局池化（丢失属性-对象绑定）和冗长标题（Bag-of-Words捷径）导致组合性能力差的问题。
*   **现有痛点**：硬负样本生成不仅计算昂贵、容易引入噪声，且模型倾向于拟合特定规则而非习得通用的组合逻辑，导致在分布外数据上泛化性差。
*   **研究假设**：通过在全局池化前显式引入“概念级”的注意力引导，迫使模型将属性与名词绑定，即可在无需硬负样本的情况下习得组合表示。

### 3. 方法设计详解
*   **Pipeline总结**：
    1.  **文本侧**：使用spaCy从标题中提取名词短语（noun-phrases），作为独立的“概念”进行学习。
    2.  **概念对比损失（$L_{npc}$）**：将提取出的名词短语（作为正样本）与全局图像表示对齐，通过SigLIP的sigmoid loss进行多正样本优化。
    3.  **跨模态池化损失（$L_{xac}$）**：利用文本侧的概念嵌入作为Query，对图像的Patch特征（Key/Value）进行交叉注意力池化，提取与概念相关的视觉表示，并与对应的名词短语对齐。
*   **模型结构**：该方法基于SigLIP架构，在原有流程基础上，仅在训练阶段额外增加了一条分支进行概念级对齐。推理阶段不使用该额外模块，因此无额外推理开销。
*   **关键公式意义**：
    *   $L_{npc}$强制模型将全局图像理解为多个独立概念的集合。
    *   $L_{xac}$通过注意力机制直接定位图像中对应概念的Patch，将绑定操作前置到池化层之前，从根本上纠正了“后期池化导致绑定信息丢失”的缺陷。

### 4. 方法对比分析
*   **本质区别**：不生成任何合成负样本，而是通过“概念解耦与重绑定”直接优化编码器的特征对齐逻辑。
*   **创新贡献**：提出了一种无需训练参数、无需推理额外成本的组合性改进方案，证明了利用原始数据中的名词短语即可诱导组合性。
*   **适用场景**：适用于所有基于ViT的CLIP类模型微调，特别是在需要兼顾零样本性能与细粒度对象属性理解的场景。

### 5. 实验分析
*   **验证方法**：在SugarCrepe和SugarCrepe++数据集上进行组合性评估，并在ImageNet进行零样本分类。
*   **关键结果**：在组合性基准上达到SOTA，且显著优于依赖硬负样本的方法。
*   **主要优势**：通用性强，检索性能提升，且无推理负担。
*   **主要局限**：在ImageNet等极度关注中心目标的任务上，因模型转向“场景中心”表示而有微小性能下降。

### 6. 实用指南
*   **开源地址**：`https://github.com/SamsungLabs/concept_centric_clip`
*   **实现要点**：使用spaCy进行词法分析提取名词；损失函数参数$\lambda_{npc}=1, \lambda_{xac}=0.01$；训练过程仅需在预训练好的SigLIP上 fine-tune 几个epoch。
*   **迁移建议**：本方法逻辑与编码器解耦，可直接迁移至其他基于注意力池化的多模态架构（如LLaVA的视觉编码器部分）。

### 7. 总结
*   **核心思想**：通过概念级对齐与注意力引导，实现特征空间内的属性-对象绑定。
*   **速记版Pipeline**：
    1. 提取标题中的名词短语。
    2. 将名词短语作为对齐的正样本。
    3. 交叉注意力提取局部视觉特征。
    4. 执行辅助损失联合训练。

**Key Findings:**

- As a remedy, we propose two simple solutions: 1) We obtain short concept centric caption parts using standard NLP software and align those with the image; and 2) We introduce a parameter-free cross-modal attention-pooling to obtain concept centric visual embeddings from the image encoder.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25722v1)
- [arXiv](https://arxiv.org/abs/2603.25722v1)

---

<a id='2603.25716v1'></a>
## [Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models](https://arxiv.org/abs/2603.25716v1)

**Authors:** Kaijin Chen, Dingkang Liang, Xin Zhou, Yikang Ding, Xiaoqiang Liu, Pengfei Wan, Xiang Bai

**Published:** 2026-03-26

**Categories:** cs.CV, cs.AI

**Abstract:**

Video world models have shown immense potential in simulating the physical world, yet existing memory mechanisms primarily treat environments as static canvases. When dynamic subjects hide out of sight and later re-emerge, current methods often struggle, leading to frozen, distorted, or vanishing subjects. To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals. To facilitate research in this direction, we construct HM-World, the first large-scale video dataset dedicated to hybrid memory. It features 59K high-fidelity clips with decoupled camera and subject trajectories, encompassing 17 diverse scenes, 49 distinct subjects, and meticulously designed exit-entry events to rigorously evaluate hybrid coherence. Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism. By selectively attending to relevant motion cues, HyDRA effectively preserves the identity and motion of hidden subjects. Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality.

**Analysis:**

作为计算机视觉和机器学习领域的专家，我对这篇题为《Out of Sight but Not Out of Mind: Hybrid Memory for Dynamic Video World Models》的论文分析如下：

### 1. 论文核心贡献总结
该论文针对视频世界模型在处理“动态主体消失与重现”时的不一致性问题，提出了**混合记忆（Hybrid Memory）**范式，明确将静态背景记忆与动态主体跟踪解耦。研究团队构建了首个专门针对该挑战的大规模数据集 **HM-World**，并设计了 **HyDRA** 架构，实现了对隐藏主体运动轨迹的精确建模与重现，显著提升了视频生成的时空一致性。

### 2. 关键创新与方法论
*   **混合记忆范式（Hybrid Memory Paradigm）：** 核心创新在于将世界记忆拆分为“静态存档（Static Archivist）”与“动态追踪（Vigilant Tracker）”。这种分治策略解决了以往模型因过度关注全局场景而丢失局部动态主体细节的问题。
*   **HyDRA 架构：**
    *   **记忆标记化（Memory Tokenization）：** 将记忆压缩为高效的 Token，降低了长视频记忆检索的算力负担。
    *   **时空相关性驱动检索（Spatiotemporal Relevance-driven Retrieval）：** 这是一种基于注意力的动态检索机制，通过选择性关注与当前帧相关的运动线索，确保即使主体在屏幕外，其身份特征和运动惯性也能被持续编码，从而避免重现时的“形变”或“闪烁”。
*   **HM-World 数据集：** 首次提供了解耦相机与主体运动轨迹的数据集，通过精心设计的“进出帧”事件，为评估世界模型的长期记忆能力提供了标准基准。

### 3. 对领域的潜在影响
*   **推动视频生成向“长时一致性”跨越：** 当前的视频生成模型（如 Sora 类模型）往往在长时间序列下表现出对象丢失或幻觉，该研究提出的记忆机制为解决这一底层瓶颈提供了范式。
*   **提升物理模拟的严谨性：** 通过引入“动态主体追踪”，视频世界模型将从单纯的“视觉模拟”转向具有“物理对象恒常性（Object Permanence）”理解能力的系统，这是通向通用人工智能（AGI）物理感知的重要一步。

### 4. 相关领域与受益应用
*   **自动驾驶仿真：** 模拟遮挡情况下的行人或车辆运动是自动驾驶测试的核心，该研究能显著提高仿真环境的真实感。
*   **机器人感知与规划：** 能够“记住”离开视野的物体对于机器人的空间导航和任务规划至关重要。
*   **影视与游戏工业：** 对于长时间跨度、复杂交互的数字资产生成，该技术能有效降低后期的手工修复工作量。
*   **AR/VR 交互：** 在增强现实中，确保虚拟对象在移出画面后再进入时保持物理特性的一致性，是提升沉浸感的关键。

### 5. 可推断的局限性
*   **计算复杂度的权衡：** 尽管使用了 Token 压缩，但在极长序列下，检索机制如何在大规模词汇库中保持高效且不牺牲检索精度，仍是一个潜在挑战。
*   **动态交互复杂性：** 论文主要关注“进出视野”，对于多个动态主体之间发生的复杂遮挡、变形或物理交互，HyDRA 的鲁棒性仍需进一步验证。
*   **泛化能力：** 该方法高度依赖于 HM-World 数据集中的特定场景，模型在处理未见过的高度复杂、非刚体变形的动态主体时，性能是否会下降仍需观察。

---
**专家点评：**
这篇论文的有趣之处在于它触及了视频生成领域的一个核心矛盾——**“即时性渲染”与“长期记忆维护”的冲突**。通过明确提出“混合记忆”这一架构设计，它将视频生成的研究重点从单纯的“像素生成”转向了“对象时空驻留”的逻辑建模，这是迈向更智能、更具物理一致性的世界模型的必经之路。

**Key Findings:**

- To address this, we introduce Hybrid Memory, a novel paradigm requiring models to simultaneously act as precise archivists for static backgrounds and vigilant trackers for dynamic subjects, ensuring motion continuity during out-of-view intervals.
- Furthermore, we propose HyDRA, a specialized memory architecture that compresses memory into tokens and utilizes a spatiotemporal relevance-driven retrieval mechanism.
- Extensive experiments on HM-World demonstrate that our method significantly outperforms state-of-the-art approaches in both dynamic subject consistency and overall generation quality.

**Links:**

- [PDF](https://arxiv.org/pdf/2603.25716v1)
- [arXiv](https://arxiv.org/abs/2603.25716v1)

---

