time: 20261102

# Arxiv Computer Vision Papers - 2026-11-02

## Executive Summary

## 本期概览

本期入选的十篇论文横跨机器人感知与控制、视觉生成和评测、长尾学习及自动驾驶可解释性。几项工作共同强调把“总体分数”拆成可核验的中间证据：CompoSET固定场景并只改变一个因素，检查视觉语言模型是否真正辨认组合关系；WeGenEditBench把复杂图像编辑请求拆为逐条约束，再将失败诊断连到定向数据改进；Beyond Scarcity则分离尾类独立样本数与表示阶段的曝光配额，直接衡量何时继续更新编码器有收益。

机器人方向的两篇图控制方法分别把当前可达能力和备份恢复能力纳入安全决策。CC-HGCBF用状态相关能力特征支持未见底盘的零样本导航；CA-BRF检查多个交互机器人执行当前动作后是否仍保有联合安全停车方案。二者都报告仿真及有限实体实验，结论仍依赖所建模的动力学、初始安全条件与测试范围。TinyAstral展示了先改进教师周围的工具与恢复流程、再把轨迹经验蒸馏给小模型的系统路径；其4B学生接近教师的若干基准结果，部署收益则以论文指定的加速器计时为准。

其余工作补充了互补的表示路线：MotorSense以同步肌电为视觉动作学习提供训练期监督；FlowMoDE以两阶段流匹配预测单目深度，在多套真实数据集上评估；Coarse3D针对多视图图像到3D的粗结构阶段，同时处理朝向与几何忠实度；自动驾驶机制分析用稀疏自编码器特征检验规划模型内部表示，并在开环设置评估轨迹误差。它们分别提示传感器监督、生成式感知和可解释特征可能带来的价值，也都需要在不同硬件、数据域或闭环条件下继续检验。

若优先完整阅读，建议从 **TinyAstral**（Harness演化、统计筛选与小模型迁移的完整链路）、**CC-HGCBF**（异构运动能力表示及留出底盘实验）、**CompoSET**（控制变量如何改变组合能力结论）和 **Beyond Scarcity**（能区分数据稀缺与表示曝光的匹配实验）开始。关注安全控制的读者可并读 **CA-BRF**；关注图像生成评估的读者可并读 **WeGenEditBench**。本期证据主要来自论文报告的基准、仿真或小规模实体实验，阅读时应同时关注各自协议、基线和适用边界。

---

## Table of Contents

1. [CompoSET: A Controlled Single-Edit Benchmark for Vision-Language Compositionality](#2610.41118v1)
2. [From Evaluation to Evolution: Benchmarking Image Editing and Evolving Image Models](#2610.41210v1)
3. [Beyond Scarcity: When Encoder Adaptation Is Worthwhile](#2610.41213v1)
4. [Mechanistic Interpretability for End-to-End Self-Driving](#2610.41185v1)
5. [A Capability-Aware Backup-Recoverability Filter for Heterogeneous Multi-Robot Navigation](#2610.41188v1)
6. [TinyAstral: Transferring Self-Evolving Harnesses to Tiny Generalist Manipulation Agents](#2610.41216v1)
7. [Coarse3D: Controllable and Faithful Sparse Structure Generation for Image-to-3D](#2610.41231v1)
8. [MotorSense: A Video-EMG Dataset of Motoric Representations for Action Understanding](#2610.41127v1)
9. [Capability-Conditioned Graph Control Barrier Functions for Zero-Shot Safe Navigation of Heterogeneous Mobile Robot Teams](#2610.41214v1)
10. [FlowMoDE: Coarse-to-fine Flow Matching for Detail-aware Sim-to-Real Monocular Depth Estimation](#2610.41190v1)

---

## Papers

<a id='2610.41118v1'></a>
## [CompoSET: A Controlled Single-Edit Benchmark for Vision-Language Compositionality](https://arxiv.org/abs/2610.41118v1)

**Authors:** Laure Delisle, Pietro Perona

**Published:** 2026-10-30

**Categories:** cs.CV

**Abstract:**

Vision-language compositionality is the ability to identify which properties, relations, quantities, and states bind to which entities in both an image and a caption, and to match those bindings across modalities. Existing benchmarks evaluate this ability through image-text matching: a model ranks candidate image-caption match similarities and succeeds when the correct match comes on top. In such datasets, paired images differ by more than a single binding: scene, layout, and content all differ, with 97% of pixels changing on average. Such pairs do not isolate compositionality: results are confounded by global image differences. We introduce Compositional Single-EdiT (CompoSET), a benchmark dataset in which each base image-caption pair has an edited counterpart, or foil, that preserves the scene but differs by a single controlled edit, such as color, count, spatial relation, or object state. Base and foil images preserve naturalistic scene complexity yet are visually near-identical: only 8% of pixels change on average. Our dataset has many foils per base scene (~22) in order to separate scene difficulty from edit difficulty, a distinction one-foil-per-base benchmarks cannot make. CompoSET contains 1,776 instances across 80 naturalistic scenes, spanning 16 edit types in 4 semantic categories, with captions at 3 verbosity levels. We test 22 dual-encoder VLMs and find that all models perform at or below chance level on CompoSET, despite scoring well on prior benchmarks. This suggests that apparent progress under prior image-text matching protocols may be an artifact arising from confounds in benchmark design, and that compositional understanding in dual-encoder VLMs is far from solved. CompoSET addresses the confounds and offers a better controlled benchmark to measure future progress.

### 论文解读

#### 摘要翻译

本文提出CompoSET，用于测量视觉语言模型对图像组合关系的理解。作者认为，已有评测常把不同场景的图片放在一起比较，模型可能依靠整体场景差异答题，而非辨认目标属性或关系。CompoSET让基础图像与反事实图像保持同一场景，只改变一个明确因素，并提供多种编辑与不同长度的描述。数据集由80个基础场景构成，共有1,776组图像文本实例，覆盖16类编辑。22个双塔模型在双向匹配的严格分数上都没有超过随机水平。论文据此指出，既有基准上的高分不能保证模型能解决细粒度的同场景组合判断。

#### 方法动机分析

组合性评估的核心痛点是变量混淆：当两张图片来自不同场景时，背景、物体、构图等大量线索可能帮助模型区分，掩盖它是否真的绑定了“谁在做什么”“物体位于何处”或“数量有几件”。作者的设计假设是，把场景固定、只编辑一个语义因素，就能让测量更接近目标能力。每个基础场景还配有多种反事实编辑，因此可以分析究竟是某些场景天生难，还是某类编辑普遍难。这种设计强调可解释诊断，而不是把一个综合分数当作所有视觉推理能力的替代物。

#### 方法设计详解

流程分为场景生成、单项编辑、图像质检和评测四步：先生成多物体基础场景及文字，再提出属性或关系编辑并生成反事实图像，之后删除对象数量不一致或出现额外变化的样本。80个基础场景共形成1,776个实例，覆盖外观、数量、空间、状态、动作和物体身份等16类编辑，每个实例有短、中、长描述。评测输入为图文组合，输出是双向2×2匹配结果：给定图像选描述、给定描述选图像；两边都对才计入严格Group分数。实验再按编辑类别、描述长度、模型和场景难度拆分。

#### 方法对比分析

与跨场景评测相比，CompoSET控制同一场景中的单一编辑，并为每个场景构造多个反事实。它要求模型区分图像与文字两侧，而非只替换文本。22个双塔模型在既有基准上表现较高，却未在CompoSET超过随机水平，说明评测设置会影响能力判断。模型生成图像便于控制变量，但不等同真实照片；有限编辑类型也不代表开放式推理。

#### 实验分析（精简版）

CompoSET共有1,776个实例、80个场景和16类编辑。双塔模型的严格Group随机水平为16.7%，最佳模型SigLIP-2 SO400M/14达到13.9%；按场景聚类的置信区间上界为15.55%，仍未越过随机线。颜色等部分外观编辑相对容易，而空间关系和目标存在性对模型尤为困难。长描述会使部分动作或姿态判断变差。另一个生成式模型切片中，Claude Sonnet 4.6报告42.5% Group，但它使用200例子集和不同提示评分方案，不能与双塔模型作直接同口径比较。数据由模型生成，人工检查由单一作者完成，故结果支持的是这套受控基准内的结论。

#### 实用指南

使用前核对版本、许可、代码和图像预处理。双塔模型应报告双向结果、严格Group分数及编辑类型拆分，避免平均数隐藏空间关系弱项。生成式模型提示结果需单独呈现。迁移前应加入真实照片、更多场景和其他语言，并安排多人独立复核。作者公开了代码与数据；复现时保存模型、预处理设置和数据版本，保证比较可追溯。

#### 总结

核心思想：用同场景单编辑检验组合理解。

1. 固定场景，只改变一个被标注的属性、关系或动作。
2. 用双向2×2匹配同时测图像到文字与文字到图像。
3. 按编辑类型和场景分析失败，谨慎外推到真实图像与开放式推理。

CompoSET揭示了双塔模型在严格同场景比较上的明显困难，也为设计更可解释的组合性评测提供了方法。但合成数据、有限场景和单人质控意味着，真实环境中的能力仍需独立验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41118v1)
- [arXiv](https://arxiv.org/abs/2610.41118v1)

---

<a id='2610.41210v1'></a>
## [From Evaluation to Evolution: Benchmarking Image Editing and Evolving Image Models](https://arxiv.org/abs/2610.41210v1)

**Authors:** Jun He, Canmiao Fu, Ying Zhang, Zhipeng Huang, Chenyu Mu, Yazhen Xie, Chen Li, Jing Lyu, Weijia Li, Xiang Zhang

**Published:** 2026-10-30

**Categories:** cs.CV, cs.AI

**Abstract:**

Real-world image editing requires coordinating multiple visual transformations with content preservation and reference constraints, yet aggregate scores can readily obscure unmet local requirements. Effective evaluation must therefore identify local failures and translate them into priorities for subsequent model improvement. We introduce WeGenEditBench, comprising 2,740 evaluation instances and 148 task types across single-image, text-intensive, and multi-reference editing. Pre-generated checklists shared across models decompose each request into atomic requirements; item-level deductions and visual evidence are established first and then summarized into scores for edit execution, preservation, visual quality, and task-specific correctness. To support the experience-acquisition component of recursive self-improvement (RSI), we develop WeGenEvoStudio, a unified multi-agent harness for RSI-oriented learning that integrates model-conditioned failure diagnosis, retrieval-grounded data construction, supervised adaptation, and re-evaluation. Among 15 evaluated configurations, GPT-Image-2 leads single-image editing with an Overall score of 4.076, while Seedream-5.0-Pro leads text-intensive editing at 4.291. Evolution experiments across four generation and editing benchmarks demonstrate substantial targeted gains: on their report-selected Top-5 single-image targets, FLUX.2 [dev] improves Overall from 1.083 to 3.517, while Qwen-Image-Edit-2511 gains 13.2%. Together, our benchmark and harness provide a foundation for developing more reliable image models through interpretable evaluation and feedback-driven learning.

### 论文解读

#### 摘要翻译

本文提出WeGenEditBench和WeGenEvoStudio，分别用于细粒度图像编辑评测和面向失败诊断的模型改进。基准包含2,740个实例和148种任务类型，覆盖单图编辑、文本密集编辑和多参考编辑。每个样本预先配有模型共享的检查清单，把复杂请求拆成具体要求，先记录逐项证据再汇总为编辑执行、内容保留、视觉质量和任务特定能力等分数。适配框架根据错误模式制定任务级数据计划，检索素材、生成训练样本、监督微调并重新评测。实验显示模型优势因场景而异，定向训练也可能伴随退步。

#### 方法动机分析

复杂编辑请求常要求同时改变多个目标、保留非目标对象、精确处理文字，并把属性绑定到正确参考图。单一总分可能被醒目的局部变化或画面质量抬高，却无法指出遗漏了哪条约束。作者据此提出两个假设：逐条核验可以暴露被平均分掩盖的局部错误；将错误映射为任务级数据需求，可以比盲目扩充样本更有效地支持定向适配。第一项是评测设计，第二项是训练工作流，两者通过样本级理由和任务处方连接。

#### 方法设计详解

WeGenEditBench包含单图、文本密集和多参考三个子集，共2,740例。评测流程先为输入图像和指令生成共享清单，再让固定多模态裁判逐条检查目标、位置、属性、保留条件和参考来源，记录理由与扣分，最后汇总为编辑目标、保留、视觉质量和领域能力等维度。文本任务分别检查字词与图像布局，多参考任务核验各属性是否来自指定图片。WeGenEvoStudio的流程包括评测、失败诊断、数据处方、素材检索与新指令合成、人工或模型复核、监督微调和重新评测。诊断会混合类别相对差距与绝对弱项，并用失败案例确定需要加强的操作和保留边界；训练后重新评测可更新下一轮优先级。

#### 方法对比分析

相比只给出整体美学或任务分数的基准，共享清单保留了每条具体要求是否完成的证据。它还利用逐项理由和案例对照，为新样本指定目标与保留条件。实验中没有任何模型在所有设置都领先：GPT-Image-2在单图任务表现突出，Seedream-5.0-Pro在文本密集任务领先；对弱底座FLUX.2进行定向适配后提升很大。结果也说明定向收益依赖模型、任务和维度，不能理解为普遍自我进化能力。

#### 实验分析（精简版）

15种模型配置参与主要基准比较。单图编辑中GPT-Image-2 Overall为4.076；文本密集编辑中Seedream-5.0-Pro为4.291。GPT-Image-2单图高分输出里，48.9%仍至少有一项条目得分不高于2，体现局部失败可被综合分遮蔽。专家随机检查每个子集10%的清单，修订后重评。定向适配每实验使用25,000个合成样例、五个目标类别和rank-16 LoRA；FLUX.2在所选单图任务Overall从1.083升至3.517，Qwen提升0.373，但Qwen完整GEdit Overall下降0.190。训练数据检索的类别效应也不一致，说明局部改善不等同于整体提升。

#### 实用指南

复用前确认数据、代码、许可和评测版本。定义可观察要求，保留逐项错误，并报告子集和有效样本数。VLM裁判适合扩展评测，但需定期人工抽检；本文的10%专家审计并未给出裁判准确率或标注者间一致性。定向微调应从少量弱项开始，同时检查非目标任务是否退步；文本和多参考场景还需验证OCR与来源绑定。

#### 总结

核心思想：逐条查错，再把错误转成定向训练数据。

1. 为复杂编辑请求建立模型共享的可观察要求清单。
2. 以条目级证据定位遗漏、保留失败和来源绑定错误。
3. 构造针对性数据并重新评测，检查收益与跨任务退步。

论文将精细评测和数据适配连接起来，给出了有价值的设计与实验。但单一VLM裁判、有限适配底座和合成训练数据限制了结论范围；定向增益应与全基准表现一起解读。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41210v1)
- [arXiv](https://arxiv.org/abs/2610.41210v1)

---

<a id='2610.41213v1'></a>
## [Beyond Scarcity: When Encoder Adaptation Is Worthwhile](https://arxiv.org/abs/2610.41213v1)

**Authors:** Zexing Zhang, Jichao Li, Tianyang Lei, Yiyin Zhang, Kewei Yang

**Published:** 2026-10-30

**Categories:** cs.LG, cs.CV

**Abstract:**

Long-tailed recognition often learns a shared representation before correcting the classifier for class imbalance. Once the classifier has been rebalanced, however, final accuracy does not reveal whether further encoder adaptation remains worthwhile. Standard imbalance benchmarks obscure this question because they jointly vary the number of distinct examples available to rare classes and the frequency with which those examples shape the representation. We introduce a controlled evaluation framework that separates unique support from representation allocation and estimates the incremental value of encoder updates through matched classifier-only and full-network continuations from a common corrected model. The framework combines stage-specific controls, multiple representation and adaptation objectives, whole-distribution metrics, and hierarchical inference over class assignments, support reconstructions, and optimization runs. Across the controlled study, the same low-support regime moves from one in which classifier-only training is adequate to one in which encoder adaptation yields a practically meaningful benefit solely as representation allocation becomes more imbalanced. The transition persists across classifier corrections, representation objectives, normalization and augmentation controls, and ImageNet-scale evaluation, while preserving frequent-class accuracy. Under the prespecified one-percentage-point operating margin, the two regimes are separated by simultaneous confidence bounds. These results show that benchmark construction can alter a post-correction training decision and that support size alone does not characterize the remaining value of representation adaptation.

### 论文解读

#### 摘要翻译

论文研究长尾识别中，分类器校正后继续更新编码器是否值得。常见不平衡基准会同时改变稀有类独立样本数量及其表示学习频率，难以判断收益来源。本文将唯一支持与表示分配拆开，并从同一个校正模型出发，对比只训练分类器和更新整个网络的两条匹配分支。实验显示，在低支持量固定时，平衡的表示分配令编码器增益低于预设实用边际；长尾分配则使增益明显超过边际。该现象在多种校正、表示目标和数据设置中保持一致。

#### 方法动机分析

长尾类别表现差，可能是因为可用图像少，也可能因为这些图像在共享表示学习中曝光不足。常见基准同时改变两者，使最终准确率无法告诉研究者应该补充数据、改变采样，还是继续微调编码器。作者提出分别控制独立图像支持量和表示学习抽样预算，再在相同分类器校正起点上做匹配对照。这样的设计把问题聚焦为后处理的增量收益，并能检验基准构造是否改变训练选择。

#### 方法设计详解

实验在表示学习阶段固定每类可用图像集合，再单独调整该类图像被抽样的总次数；尾类支持数为50至5，分配设为平衡、中间或长尾。之后用类别平衡分类器校正得到共同检查点。比较流程包含两条支路：分类器支路冻结编码器，完整网络支路允许编码器更新，两者使用相同样本配额、数据增强、训练步数和分类器学习率计划。主要指标是尾类宏平均准确率之差。作者使用FT-CE、FT-BCL、FT-BCE3S三种适配目标，并用层级bootstrap纳入类别角色、支持集合重建和优化运行的不确定性；另测试不同校正方法、BN统计、增强复用、ImageNet-100和ImageNet-LT。

#### 方法对比分析

与比较不同长尾算法最终分数相比，匹配延续更直接估计分类器校正后开放编码器的操作性收益。唯一支持和表示分配拆开后，样本稀缺造成的难度与表示历史造成的剩余收益可以分别观察。作者也对多种校正方式和表示目标做稳健性检查。方法的结论依赖所选优化预算、模型和决策边际，因此结果应报告为收益及置信界，而不是无条件的“冻结”或“微调”建议。

#### 实验分析（精简版）

CIFAR-100最低支持设置中，平衡分配的增益为0.17–0.28 pp，上界不超过0.45；长尾分配为1.61–1.94 pp，下界至少1.40，FT-BCL增益从0.26到1.86。以预设1个百分点为边际时，前者只训练分类器，后者更新编码器。支持量从50降至5使尾类准确率下降21.30 pp，但平衡分配下编码器收益仍低。多种校正与归一化控制保持这一切换。ImageNet-100也出现相同趋势；ImageNet-LT上总体增益0.80、few-shot类别增益1.60个百分点，不过自然基准不能区分支持量和曝光分配。

#### 实用指南

复现长尾实验时，除不平衡因子外，应报告每类独立图像数及表示阶段抽样预算。评估是否值得微调，可从统一校正检查点比较冻结编码器和更新编码器的匹配支路，保持后续输入、增强与训练步数一致，并提供置信界及GPU成本。低支持实验应重采样支持集合，而非只重跑优化。应用一个百分点阈值前，要按实际延迟、算力和尾类错误成本设定自己的容忍范围。当前证据限于监督ResNet及研究中的数据集，迁移到预训练模型需重新验证。

#### 总结

核心思想：把样本数量与表示阶段曝光分开测量。

1. 固定独立样本集合，改变表示学习期间的类别配额。
2. 从共同校正模型匹配比较冻结与更新编码器。
3. 依据收益区间和计算成本决定是否微调。

这项研究说明，长尾基准的构造可能影响后续训练决策。它为表示微调提供了更清晰的比较方式，但效用边际、架构范围和机制解释仍需结合应用重新验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41213v1)
- [arXiv](https://arxiv.org/abs/2610.41213v1)

---

<a id='2610.41185v1'></a>
## [Mechanistic Interpretability for End-to-End Self-Driving](https://arxiv.org/abs/2610.41185v1)

**Authors:** Manav Gagvani, Benjamin Namikas, Sivamurugan Velmurugan, Nysa Kumar, Shrey Sharma, Thomas Peterson, Pratyush Mathur, Hyunseo Chang, Rohan Paleja

**Published:** 2026-10-30

**Categories:** cs.RO, cs.CV, cs.LG

**Abstract:**

End-to-end (E2E) autonomous driving models achieve strong planning performance but remain largely opaque, limiting observability and trust. We apply mechanistic interpretability to E2E driving by training Sparse Autoencoders (SAEs) on the internal activations of a simple, single-camera end-to-end path planning model. Unsupervised clustering of SAE latents reveals interpretable structure aligned with route intent, vehicle speed, and braking behavior, recovering route intent with 98.3% accuracy without supervision. Through paired counterfactual image edits, we further show that specific SAE features are visually grounded and systematically respond to scene changes such as traffic light state and stop sign presence. Steering individual SAE latents produces predictable, monotonic changes in planner outputs including lateral displacement, curvature, and braking. Finally, we demonstrate a practical application of these representations: we build a lightweight adapter trained on SAE latents that reduces trajectory selection error by 3.44%, outperforming an equivalent adapter that omits SAE latents, confirming that the sparse representation captures useful information for interpretability. Code and visualizations are available at https://mgagvani.github.io/mechanistic-interpretability-for-e2e-driving/

### 论文解读

#### 摘要翻译

本文将机制可解释性方法用于端到端自动驾驶规划器。作者在单摄像头路径规划模型的内部激活上训练稀疏自编码器，并发现潜变量聚类与路线意图、速度和制动相关。路线意图聚类达到98.3%准确率。通过成对反事实图像编辑，部分潜变量会随交通灯或停车标志变化而系统响应；逐个操纵潜变量会带来可预测的横向位置、曲率和制动变化。用SAE特征训练的轻量适配器还将5秒时域轨迹选择误差降低3.44%。

#### 方法动机分析

端到端驾驶模型从传感器输入直接产生规划结果，内部决策难观察，限制了调试与信任。中间占用或运动预测虽提供额外输出，却不一定揭示模型内部具体依赖哪些因素。作者针对内部不透明的挑战，尝试把激活分解为稀疏特征，再通过视觉干预和潜变量操纵验证这些特征是否具有语义含义及行为关联。研究意图是提供可分析的内部表示，而不是单纯提高驾驶成绩。

#### 方法设计详解

作者训练集为单摄像头规划模型的隐藏激活；在其上训练稀疏自编码器，将密集激活映射成较稀疏的潜变量。对潜变量进行无监督聚类，并与路线意图、速度和制动标签进行比较。为检验视觉依据，使用成对反事实图像编辑改变交通灯状态、停车标志等因素，观察潜变量是否随之改变。随后逐个调整特征，测量规划输出的横向偏移、曲率与制动变化。最后训练一个轻量轨迹选择适配器，并与不含SAE特征的对应模型对照，评估特征表示是否有下游效用。

#### 方法对比分析

相较于只用注意力热图或中间预测解释模型，SAE尝试将网络激活分解为可独立分析的稀疏维度；反事实编辑和特征干预为“特征有意义”提供了比相关性可视化更直接的验证。轻量适配器则检验表示是否能帮助下游误差。区别于正式因果识别或闭环安全证明，这些测试仍主要针对特定模型、编辑协议和开环轨迹输出，不能说明所有特征在真实驾驶中具有稳定语义。

#### 实验分析（精简版）

路线意图潜变量聚类准确率为98.3%。论文展示交通灯状态和停车标志变化引起相应特征响应，并报告逐特征操纵会单调改变部分路径输出。使用SAE潜变量的适配器将5秒时域ADE降低3.44%，优于不含SAE表示的同类适配器。实验基于单摄像头开环规划模型，未通过闭环车辆运行证明安全性；图像编辑对照和特征干预也需区分相关结构与因果机制。

#### 实用指南

机制分析可用于发现驾驶模型候选决策特征，但应跨场景、路线、天气和模型版本检验稳定性。反事实图像需核查除目标因素外没有改变其他线索；潜变量干预后应观察多种规划指标，而非单一ADE。任何适配器应先在闭环仿真和失效场景评估，再考虑真实系统。复现时需固定激活层、SAE稀疏度、聚类流程和图像编辑协议，并披露基础模型限制。

#### 总结

核心思想：以SAE发现并检验驾驶特征。

1. 对隐藏激活训练稀疏自编码器并聚类潜变量。
2. 编辑交通场景并操纵特征，观察规划输出变化。
3. 用SAE特征训练轻量适配器，测量下游轨迹误差。

研究为驾驶模型内部分析提供了可操作流程，结果仍限于开环单摄像头模型，尚不能作为安全保证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41185v1)
- [arXiv](https://arxiv.org/abs/2610.41185v1)

---

<a id='2610.41188v1'></a>
## [A Capability-Aware Backup-Recoverability Filter for Heterogeneous Multi-Robot Navigation](https://arxiv.org/abs/2610.41188v1)

**Authors:** Joonkyung Kim, Yanze Zhang, Wenhao Luo, Yiwei Lyu

**Published:** 2026-10-30

**Categories:** cs.RO

**Abstract:**

Safe navigation in heterogeneous multi-robot teams faces two coupled challenges: differences in robots' braking and turning capabilities limit how each robot can respond to an approaching neighbor, while simultaneous interactions require a single control action to satisfy multiple collision-avoidance constraints. Together, these challenges can make local safety constraints incompatible with a robot's admissible controls, leaving a decentralized safety controller without a feasible action. We propose a Capability-Aware Backup-Recoverability Filter (CA-BRF), based on the insight that navigation actions should preserve access to a certified recovery maneuver. First, we construct a backup-recoverable set in a shared Cartesian operational space by evaluating whether prescribed backup maneuvers maintain pairwise safety until the robots come to rest. Each maneuver respects the corresponding robot's kinematic and actuation constraints, making the recovery certificate capability-dependent. Second, CA-BRF filters candidate controls by selecting admissible interpolations between candidate and backup actions while coordinating the selection across simultaneous interactions. This preserves backup recoverability while retaining as much of the candidate action as possible and ensures that all pairwise recovery conditions are satisfied by a single executable control for each robot. Together, we establish recursive feasibility and safety for the discrete-time closed-loop system even when the nominal safety-constrained optimization becomes infeasible. The validity and effectiveness of the proposed CA-BRF are validated in simulations with up to 30 heterogeneous robots and experiments of robots with five different kinematic classes.

### 论文解读

#### 摘要翻译

CA-BRF是面向异构多机器人导航的能力感知备份恢复过滤器。方法先为每种机器人定义遵循其运动学和控制限制的备份停车策略，再判断机器人对能否沿备份轨迹保持安全。期望控制与备份控制之间进行插值，并在存在多个同时交互时对整个交互连通分量采用一致比例，避免每个机器人单独决策后组合成不安全动作。理论分析在特定初始可恢复条件下给出离散控制时刻的递归可行与安全性；仿真和实体机器人实验展示其有效性。

#### 方法动机分析

研究动机是解决多交互下候选动作不能共同满足安全约束的挑战。HOCBF控制器可以生成满足局部安全约束的候选动作，但多对机器人同时交互时，各约束的交集可能为空，或者单独检查后执行的组合并不等于被检查的组合。不同底盘刹车和转向能力又使共享的恢复模型不可靠。作者提出更前瞻的问题：执行当前动作后，是否仍保留一个符合各机器人实际约束的安全停车方案？该恢复能力可作为动作过滤的边界。

#### 方法设计详解

算法先通过高阶控制障碍函数产生名义候选控制，同时为双积分器、独轮车、差速车、类车和前向-only车辆指定不同备份策略。对机器人对，计算沿联合备份轨迹的几何安全裕度及包含接近速度的第一层屏障量；整个停车过程两者均非负，才认为当前状态属于备份可恢复集合。接着，在候选动作和备份动作之间按有限比例网格插值，并验证新状态仍能恢复且动作对底盘可执行。交互图中的连通分量共享一个比例，因此同一机器人涉及的多组安全条件针对实际执行的共同动作进行验证。若候选控制不可行，完整备份始终作为已验证端点。

#### 方法对比分析

与仅执行瞬时HOCBF候选相比，CA-BRF检查未来停车过程；与共享刹车模型相比，它使用每个底盘的实际备份；与各机器人独立选择插值比例相比，连通分量共同决策能避免局部认可动作在组合后破坏恢复证书。相应代价是全备份动作会降低部分到达率，而且保证依赖可验证的备份策略、初始可恢复集合和离散更新模型。

#### 实验分析（精简版）

五种运动学类别、N=10/20/30的随机导航各进行50次试验。CA-BRF安全成功率分别为98.0%、92.0%、85.6%，150次试验无安全违规。N=30时直接用HOCBF候选的违规率为86%；共享恢复模型的违规率为28%。独立插值决策在N=30的50次试验中有26次组合失败，而连通分量一致决策未观察到此类失败。实体实验使用五台LIMO Pro机器人配置五种运动学类别进行中心区域交叉。仿真无障碍，实体实验规模有限。

#### 实用指南

部署前需为每种机器人建立可执行的备份控制，并确认有限时间停车和控制周期假设；新底盘不能只增加型号标签，必须定义其约束与备份策略。验证初始团队状态位于备份可恢复集合，并测试模型误差、通信延迟、定位噪声和动态障碍。记录过滤频率、插值步长及全备份触发比例。该保证只覆盖离散更新时刻，不能替代连续时间碰撞检查、急停机制或死锁处理。

#### 总结

核心思想：执行动作须保留安全恢复能力。

1. 为每类底盘定义约束兼容的停车备份。
2. 沿备份轨迹检查间距与闭合速度证书。
3. 对共享交互的机器人统一选择候选/备份插值比例。

CA-BRF为异构团队提供了明确的恢复安全机制；保证适用范围受初始可恢复条件、离散控制时刻和备份模型精度约束。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41188v1)
- [arXiv](https://arxiv.org/abs/2610.41188v1)

---

<a id='2610.41216v1'></a>
## [TinyAstral: Transferring Self-Evolving Harnesses to Tiny Generalist Manipulation Agents](https://arxiv.org/abs/2610.41216v1)

**Authors:** Kehong Gong, Xiaoxu Zheng, Guanli Hou, Seunghun Lee, Mengqi Guo, Runjia Tan, Zhengyu Wen, Qi Wang, Xiaoyu He, Mingyuan Zhang, Michael Bi Mi

**Published:** 2026-10-30

**Categories:** cs.RO, cs.CV

**Abstract:**

Robot agents built on vision-language models depend on both the planner and its harness: the tools, feedback, memory, and recovery logic around it. Can improvements to a large teacher's harness be carried into a compact agent with lower inference compute cost? We present TinyAstral, a two-stage approach. We first freeze an open-weight teacher and improve its harness through recursive self-improvement (RSI): frontier LLM agents read the teacher's trajectories and propose harness edits, and each edit is accepted only if it passes a paired statistical test. We then freeze the final harness, collect teacher trajectories with reasoning and failure recovery, and distill a small student that replaces the teacher in the same harness. On RoboTwin, harness evolution raises the teacher's success from 33.4% to 63.6%; the final 4B student reaches 62.8%. Across four harness versions, distilled 4B students score close to their corresponding teachers, and later versions yield stronger students, showing that harness evolution gains carry into compact models. The harness gains also transfer to other frozen planners without retraining. On identical episodes, the final 4B student outperforms open 4B agents, the open 27B harness-RSI model SenseNova-RoboRSI, and our own teacher model placed in three competitors' harnesses. It stays close to its teacher on all three training benchmarks, and on three benchmarks never used for RSI or training it scores close to a 552B planner running the same harness. On the measured deployment episodes, it uses about one fifth of its teacher's accelerator time.

### 论文解读

#### 摘要翻译

TinyAstral提出两阶段的机器人操作策略迁移方法。第一阶段冻结大模型教师，通过主控智能体读取失败轨迹、提出Harness改动，并用预先设定的配对统计检验决定是否接受。第二阶段冻结演化后的Harness，采集含推理、工具调用和失败恢复的教师轨迹，训练集包含教师轨迹与真实机器人数据，用于微调0.8B至4B学生，并在同一Harness中部署。教师在RoboTwin成功率从33.4%提升到63.6%，4B学生达到62.8%；论文还在三个训练基准和三个未参与演化的基准上评估迁移。

#### 方法动机分析

大型教师模型能力强，但每次操作推理成本高；小型学生便于部署，却可能缺少任务规划、反馈理解和失败恢复经验。作者区分模型权重与外部Harness：操作工具、记忆、执行反馈和恢复逻辑本身可能显著影响表现。若先演化教师周围的系统，再把改进后的推理与恢复轨迹蒸馏给学生，小模型或许能继承一部分系统级收益。研究假设是Harness进步可以跨规划器迁移，并且能通过监督数据继续迁移到小模型。

#### 方法设计详解

流程分两阶段。首先冻结Qwen3.8-Flash-Next教师，主控智能体阅读任务执行图像、工具调用和失败类型，提出通用Harness改动。候选版本在相同任务和种子上与当前版本配对测试，按预注册统计门槛接受或拒绝；训练、选择和最终评测种子分离，教师权重始终不变。然后固定最终Harness，收集成功轨迹、推理过程、工具调用和注入抓取/放置扰动后的恢复分支，混合公开真实机器人监督数据训练0.8B、2B和4B学生。部署时学生替代教师，但继续使用同一冻结Harness。

#### 方法对比分析

与直接微调或压缩大模型相比，TinyAstral先改进模型外部工具、反馈、记忆和恢复流程，随后才蒸馏策略；与只用成功主轨迹模仿相比，还加入失败恢复分支和推理文本。不同Harness版本的学生表现都接近对应教师，支持收益可迁移的判断。该结果依赖Harness评估器、任务接口和教师示范质量，不能自动外推到不同机器人平台或更广泛任务。

#### 实验分析（精简版）

在相同500个RoboTwin测试回合上，冻结教师从H1的33.4%升至H*的63.6%，4B学生从36.2%升至62.8%。最终学生在Guava为93.6%、LIBERO-PRO为78.3%，分别接近教师的94.7%和80.0%。三个零样本基准上，4B学生距教师在5.0个百分点内，距552B规划器在2.4个百分点内。移除推理监督使LIBERO-PRO降低19.3个百分点、SimplerEnv Google VM降低11.2个百分点、VLABench降低10.0个百分点。部署计时中学生每回合25.3秒、教师123.8秒，约4.9倍差异；这是论文的加速器计时口径。

#### 实用指南

已有机器人工具系统的团队可分别评估模型、Harness和学生，而不是只比较总成功率。Harness改动应采用配对任务/种子和事前接受标准，保留独立评测种子；学生训练数据应包含反馈与恢复轨迹，并用严格最终状态成功度量。部署成本需分开记录教师调用、轨迹生成、微调和每回合推理。实机测试应覆盖错误抓取、任务终止误判和恢复失败，并配置独立安全停止机制。

#### 总结

核心思想：先改进教师周围的Harness，再蒸馏推理和恢复经验。

1. 冻结教师权重，以配对统计门槛演化Harness。
2. 收集成功、失败恢复和真实机器人监督轨迹。
3. 训练小型学生并在同一Harness中部署验证。

TinyAstral展示了系统级能力向小模型迁移的潜力，但收益依赖工具接口、评估规则和示范数据，实机可靠性仍需更广验证。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41216v1)
- [arXiv](https://arxiv.org/abs/2610.41216v1)

---

<a id='2610.41231v1'></a>
## [Coarse3D: Controllable and Faithful Sparse Structure Generation for Image-to-3D](https://arxiv.org/abs/2610.41231v1)

**Authors:** Gaige Wang, Bowen Pan, Hao Ren, Jiangjing Lyu, Chengfei Lyu

**Published:** 2026-10-30

**Categories:** cs.CV, cs.AI

**Abstract:**

Coarse-to-fine 3D-native generation relies heavily on the first-stage sparse structure, which determines the global orientation and overall geometry for subsequent refinement. However, existing coarse generators suffer from two key limitations: their orientation is difficult to control due to ambiguous semantic canonicalization, and their structures are often inconsistent with input images due to weak 2D-3D grounding. We propose Coarse3D, a sparse-structure generation model that produces orientation-controllable and structurally faithful voxel geometry from arbitrary unposed images. Coarse3D introduces three designs: a first-frame anchored canonical space that aligns generation with the first input view while preserving object-centric normalization; a dual-tower DiT that couples generation with auxiliary feed-forward reconstruction to learn spatially grounded features; and spatial-prior guided attention that regularizes cross-attention with pixel-to-voxel correspondence. Coarse3D can directly replace the coarse stage of existing pipelines. Experiments show that it substantially improves sparse structure quality and consistently enhances final 3D generation, achieving state-of-the-art performance.

### 论文解读

#### 摘要翻译

Coarse3D面向从多张未标定图像生成3D资产的级联流程，针对粗稀疏结构阶段的朝向控制和几何忠实度问题。作者提出三项设计：以第一张输入图像锚定物体方位、使用生成与前馈重建双塔共同学习空间特征，以及通过像素到体素空间先验约束跨注意力。方法可以替换多个既有流水线的粗阶段，后续细化网络保持不变。GSO和Toys4K实验显示粗结构指标与最终几何质量都有提升。训练时使用相机几何先验，推理阶段不需要相机位姿。

#### 方法动机分析

粗到细3D生成中，第一阶段给出整体朝向和粗形状，错误会传递到细化阶段。标准规范坐标要求模型猜测语义正面，但风扇等物体可能没有唯一正面，不同来源3D资产的朝向也不一致。另一方面，常规跨注意力让体素token自行从图像token中学2D—3D对应，几何监督较弱。论文将问题拆为朝向歧义与局部结构证据不足，分别用首帧方位锚定、重建监督和显式空间注意力先验处理。

#### 方法设计详解

流程包含坐标锚定、联合生成重建和注意力正则三部分。坐标模块只将物体方位对齐第一视角，仍保持中心化单位立方体与原有竖直方向。双塔DiT的图像塔处理多视角并预测相机与点图，几何塔逐层接收同一层图像特征，生成稀疏体素结构；联合损失同时约束流匹配生成和重建结果。最后，训练时把真值体素投影到图像平面，形成像素—体素局部对应，并与基础模型注意力结合为伪目标，以KL损失微调。推理时不需要相机参数。该粗体素输出可接入接受相同物体中心占据网格的细化网络。

#### 方法对比分析

与标准规范坐标相比，首帧锚定不要求推断语义正面；与完全相机对齐的像素空间相比，它仍保留中心化尺度和重力方向。与先重建再生成的分阶段方案相比，双塔将重建和生成置于共享坐标并逐层传递特征。与无显式对应的跨注意力相比，空间先验提供训练期局部几何引导。代价是模型规模和训练计算较大，且坐标接口限制了对已有管线的兼容范围。

#### 实验分析（精简版）

Coarse3D在GSO和Toys4K的粗结构PA-IoU分别为0.5364和0.6368，Chamfer距离为0.0124和0.0091。替换Trellis、Trellis2或ReconViaGen的粗阶段后，最终形状CD下降，F-Score提升约0.24–0.44；GSO上Trellis2的CD由0.0613降至0.0127。消融显示双塔主要增强结构，首帧锚定缩小方向误差，空间先验继续改善局部对应。视角数由1增加到6时重建持续提升，但收益递减。Pixal3D多视角对照使用真值相机位姿，其他方法使用无位姿输入，需谨慎比较。

#### 实用指南

使用前确认后端细化器接受同分辨率的物体中心占据体素，并明确输入图像首帧的方向语义。评测应报告输入视角数、是否提供相机位姿，并同时给出对朝向敏感的IoU、方向对齐后的PA-IoU和几何距离。训练约需20亿参数、数十张H20级资源及大规模3D资产；部署或复现前需核对计算预算、内部数据依赖、权重和数据许可。该方法适用于粗结构可替换的级联系统，不能直接迁移到坐标空间不同的流水线。

#### 总结

核心思想：以首视角定向，用重建与空间对应约束粗结构。

1. 把物体方位锚定首视角，保留规范尺度和竖直轴。
2. 双塔联合学习多视角几何特征与稀疏结构生成。
3. 训练期加入像素—体素注意力先验，再替换兼容管线的粗阶段。

Coarse3D在两类数据集和多个下游细化流程中取得明显几何收益。数据和算力要求、基准范围及不同位姿协议仍限制了结论的外推。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41231v1)
- [arXiv](https://arxiv.org/abs/2610.41231v1)

---

<a id='2610.41127v1'></a>
## [MotorSense: A Video-EMG Dataset of Motoric Representations for Action Understanding](https://arxiv.org/abs/2610.41127v1)

**Authors:** Eadom Dessalene, Michael Maynord, Amir-Hossein Shahidzadeh, Botao He, Yasamin Tabatabaee, Ruoshi Liu, Ruohan Gao, Paul Pu Liang, Yiannis Aloimonos

**Published:** 2026-10-30

**Categories:** cs.CV, cs.RO

**Abstract:**

Computer vision models learn hand action only from its visual consequences, without access to the underlying embodied motor signals. We address this absence with MotorSense, the first large-scale paired egocentric video and surface electromyography (EMG) dataset. MotorSense pairs bimanual EMG bands with egocentric video and yields synchronized video, bimanual EMG and IMU, and audio across 100 hours of unscripted manipulation. EMG offers a powerful scalable alternative to manual annotation: it captures finger movement, hand configuration, and the forces exerted between hand and environment as a dense signal free of occlusion and labeler bias, whereas manual annotation is costly, subjective, and unreliable under occlusion. We use MotorSense to learn sensorimotor alignment, producing an organization of action that is shared across the visual and motoric spaces, through cross-modal pre-training between video and EMG. The resultant vision models are then transferred to downstream tasks that benefit from sensorimotor alignment: in this case, action recognition, 3D hand reconstruction, and 3D contact reconstruction. We further demonstrate a generative application in which EMG motor signals are used to condition a video diffusion model, yielding videos of fine-grained hand manipulation. Across four action-centric computer vision tasks and eight popular video datasets, we demonstrate that sensorimotor alignment provides a stronger foundation for visual representation learning than conventional self-supervised learning methods over video, outperforming every baseline across all tasks and datasets evaluated.

### 论文解读

#### 摘要翻译

MotorSense是一个同步采集第一人称视频与表面肌电的手部操作数据集，包含约100小时自然手物交互，并记录双手肌电、IMU和音频。作者使用视频与肌电对比预训练，让视觉表示捕捉肌肉动作中的细粒度结构；训练完成后可移除肌电编码器，将视觉骨干迁移到动作识别、三维手部与接触重建等任务。论文还将肌电压缩为离散动作码，用于条件视频生成。实验在多个动作相关任务与视频数据集上报告了相对视觉自监督方法的优势。

#### 方法动机分析

手部操作视频容易出现遮挡，且画面只能呈现动作结果，难以直接反映手指发力、抓握构型或接触变化。人工标注既昂贵，也可能受动作词表和可见性限制。作者以肌电作为身体内部动作信号，假设它能为视频提供更密集的动作监督。核心好处是训练时引入额外模态，部署时仍能只使用视觉；对于生成任务，肌电也有望提供比自然语言更精细的动作控制。

#### 方法设计详解

数据采集在厨房环境进行，使用双侧腕带，每侧8个表面肌电通道并记录IMU，和眼镜式第一人称视频同步。训练集由配对视频和肌电时间窗构成；视频编码器和肌电/IMU编码器处理对应窗口，映射到共享嵌入空间，通过对称对比损失使同一时间窗的跨模态向量靠近。为避免模型仅识别记录会话，训练批次从同一录制中采样。预训练结束后去掉肌电塔，把视觉骨干迁移到动作识别、手部三维重建与接触重建。生成分支则用VQ-VAE把EMG/IMU时间序列量化为离散motor codes，再作为图生视频扩散模型的条件。

#### 方法对比分析

与只用RGB自监督相比，肌电提供手部动作的内部信号；与手套或手部传感器相比，前臂腕带对操作的物理遮挡较少。它与动作文本、关键点和音频条件也不同：文本表达高层语义，关键点难体现力量，音频反映环境声而非肌肉状态。优势在于表征动作与接触的密集信号；代价是需要额外同步硬件和数据采集，训练域也受到传感器和任务场景影响。

#### 实验分析（精简版）

论文在四类动作相关视觉任务及八个视频数据集上比较；清洗时剔除约4%的异常候选窗口，报告跨模态预训练优于所测视觉自监督基线。实验包含动作识别、手部重建、接触重建和肌电条件视频生成。跨场景检索例子中，模型可依据肌电检索到物体和录制环境不同、但动作相似的视频片段。上述结果支持肌电对动作表征的帮助；数据以厨房操作为主，不能直接代表任意日常行为或机器人控制任务。

#### 实用指南

研究手物交互或肌电控制时，可考虑在训练阶段加入EMG监督，同时让部署模型只依赖视觉。复现应关注视频和肌电时间对齐、电极位置、用户划分、记录会话隔离和不同设备间迁移，并使用相同视频架构做公平基线。若将motor codes用于生成，需另外检查动作细节、跨用户稳定性和物理合理性。腕带虽较轻便，仍会带来传感器依赖及佩戴差异。

#### 总结

核心思想：用肌电监督视觉动作表示。

1. 同步采集视频、双侧肌电和IMU。
2. 以跨模态对比学习塑造视觉表示，再移除肌电塔迁移。
3. 将离散肌电动作码作为视频生成的控制信号。

MotorSense提供了丰富的传感器数据和多任务实验，展示了训练时身体信号的潜在价值；应用范围仍受采集环境、用户和硬件条件限制。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41127v1)
- [arXiv](https://arxiv.org/abs/2610.41127v1)

---

<a id='2610.41214v1'></a>
## [Capability-Conditioned Graph Control Barrier Functions for Zero-Shot Safe Navigation of Heterogeneous Mobile Robot Teams](https://arxiv.org/abs/2610.41214v1)

**Authors:** Joonkyung Kim, Janghyun Cho, Jeehyun Lee, Changjoo Nam, Yiwei Lyu

**Published:** 2026-10-30

**Categories:** cs.RO

**Abstract:**

Multi-robot systems are increasingly deployed as heterogeneous fleets with diverse kinematic and actuation capabilities. While graph control barrier function (GCBF) methods provide a scalable framework for decentralized safe control, existing approaches typically assume homogeneous robot dynamics. Methods that directly encode robot classes can distinguish known types, but do not represent state-dependent motion capabilities or naturally transfer to unseen robot models. To address this challenge, we propose the Capability-Conditioned Heterogeneous Graph Control Barrier Function (CC-HGCBF), defined in a shared second-order reference-point operational space. This formulation provides a common geometric representation of safety across heterogeneous robots while retaining their model-specific dynamics and admissible control constraints. To enable zero-shot transfer, we introduce the Radial Capability Field (RCF), a compact, fixed-dimensional representation of each robot's state-dependent finite-horizon reachable velocity set. Conditioning the shared graph barrier and decentralized policy on the RCF enables them to adapt to changes in robot state, kinematic constraints, and control limits without relying on predefined robot identities. We jointly train the barrier and policy using proactive safety labeling and neural CBF-QP supervision, and evaluate them on mixed teams containing both seen and unseen robot models. Experiments demonstrate scalable safe coordination and zero-shot transfer to robot models and capability variations withheld during training. Physical-robot experiments further demonstrate zero-shot deployment to unseen kinematic models without retraining.

### 论文解读

#### 摘要翻译

CC-HGCBF研究异构移动机器人团队的分布式安全导航。作者把不同机器人的运动统一映射到共同的二阶参考点空间，同时保留各自动力学和控制约束。提出Radial Capability Field（RCF），以固定维度描述机器人从当前状态出发、在有限时间内能实现的速度变化。图屏障和策略根据能力特征协同学习，并通过前瞻安全标签与神经控制障碍函数二次规划监督训练。仿真和实体机器人实验展示对未见机器人型号及运动能力变化的零样本泛化。

#### 方法动机分析

传统图控制障碍方法多假设同构动力学；已有异构方法使用机器人类型ID或补零编码，能辨别已知类别，却无法反映具体状态下可转向、加速或制动的能力。安全决策需要知道机器人当前能怎样响应，而不只是它属于哪一类。作者的关键假设是，若把状态相关可达运动集合压缩成共同空间中的能力描述，统一策略就能适配训练中没见过的型号。

#### 方法设计详解

方法先把不同底盘通过参考点映射转为共同的平面位置、速度和二阶操作动力学。随后估算机器人在有限预测时域内的可达速度集合，并投影至共同操作空间。RCF以当前速度为中心，按多个方向测量速度变化的径向范围，再用傅里叶特征编码为固定长度向量。节点能力特征与位置、速度、目标和障碍观测组合成局部交互图。训练集由8台机器人构成，覆盖四类已知运动学；共享图屏障与分布策略使用前瞻安全标签和神经CBF-QP控制示例联合训练。执行时每台机器人使用本地邻居信息与二维激光雷达，输出经型号控制接口转换的动作。

#### 方法对比分析

对比不输入能力的GCBF+，RCF提供运动约束信息；对比one-hot型号编码，它能描述训练未见型号和状态变化；对比静态可行速度集合，它还反映当前速度下的有限时域响应能力。实验显示在训练见过的型号上各能力版本差距较小，而未见类车、前向-only底盘和转向参数变化时，状态相关能力特征更有价值。方法的代价是需要模型相关的操作空间映射、可达集合计算和可信控制接口。

#### 实验分析（精简版）

仿真包含全向双积分器、独轮车、差速车和类车机器人，训练集每种两台机器人、总计8台；测试队伍扩到128台。无障碍N=128时，RCF成功率95.2%、安全率98.9%；障碍密集设置仍保持较高安全表现。前向-only机器人在训练时完全未出现；未见类车和转向参数变化也无需重训。作者报告RCF在留出类车上相较静态能力及离散类型基线有明显安全、成功率提升。实体测试用五台RoboMaster S1，在未见运动能力下展示零样本运行，但规模和场景覆盖有限。

#### 实用指南

异构车队可考虑用可达能力特征替代离散型号标签，尤其是底盘或控制限制会动态变化的场景。复现时需准确实现参考点映射、控制约束和可达速度近似，并核对激光雷达范围、控制周期与执行接口。应在不同密度、留出型号和参数边界上评估安全率，同时报告成功率。仿真和小规模实体演示不能替代部署安全论证；安全关键系统应保留独立监控与急停策略。

#### 总结

核心思想：以RCF描述当前运动能力。

1. 将异构运动学映射到统一操作空间。
2. 用RCF编码有限时域、状态相关速度变化。
3. 将能力注入图屏障与分布式策略并做留出模型验证。

论文展示了异构车队零样本控制的可行方向，但理论条件、感知模型和实机规模仍限定了外推范围。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41214v1)
- [arXiv](https://arxiv.org/abs/2610.41214v1)

---

<a id='2610.41190v1'></a>
## [FlowMoDE: Coarse-to-fine Flow Matching for Detail-aware Sim-to-Real Monocular Depth Estimation](https://arxiv.org/abs/2610.41190v1)

**Authors:** Liangjing Shao, Wanhao Liu, Jinsong Lin, Zhiwei Fang, Hongliang Ren

**Published:** 2026-10-30

**Categories:** cs.CV

**Abstract:**

Monocular depth estimation is a fundamental capability for 3D spatial perception in robotic and intelligent systems. While data-driven discriminative methods have achieved compelling performance for monocular depth estimation, generative approaches can reconstruct fine structural details with stronger generalization. However, most existing generative diffusion models suffer from low inference efficiency caused by multi-step sampling, whereas single-step diffusion models struggle to recover high-fidelity depth maps. To mitigate these drawbacks, this paper presents FlowMoDE, a coarse-to-fine flow-matching-based framework for efficient, structure-aware monocular depth estimation supporting sim-to-real and cross-scene generalization. The core architecture first conducts coarse-level decoding conditioned on diffused features extracted from a pre-trained semantic encoder, and subsequently restores fine-grained structural patterns via expanded decoding layers. Trained exclusively on a single synthetic indoor dataset, our proposed method outperforms prior methods across five real-world indoor and outdoor datasets, yielding an average 14.1% reduction in relative error against state-of-the-art works as well as 9.9\% reduction in chamfer distance for point cloud reconstruction. We further validate the generalization capability of FlowMoDE on realistic robotic scenes, with qualitative comparisons against state-of-the-art methods. The project page is https://flowmode-shao.netlify.app/

### 论文解读

#### 摘要翻译

FlowMoDE是一种面向单目深度估计的粗到细流匹配模型。它先利用预训练语义特征预测粗深度，再通过细粒度解码恢复局部结构，推理时使用两次流更新。模型只在单一合成室内数据集上训练，却在五个真实室内和室外基准取得较强结果，并在机器人图像中展示泛化。作者报告平均相对误差降低14.1%，点云重建Chamfer距离下降9.9%。

#### 方法动机分析

判别式模型往往能给出稳定整体深度，但大块编码会损失边界和细结构；生成式扩散能利用强先验恢复细节，却通常需要许多采样步，单步生成又可能精度不足。FlowMoDE的假设是，语义先验可先确定全局结构，较轻的细解码器再恢复高频深度信息；少量流匹配步便能在细节和效率之间折中。

#### 方法设计详解

模型由语义编码器、语义解码器、粗解码器和细粒度解码器组成。冻结DINOv2-B提取图像语义，深度图在训练时加噪，网络以图像、噪声深度和时间条件预测速度场。粗解码器利用全局语义构造深度轮廓，细解码器经上采样后修复像素级结构。条件通过调制机制注入Transformer块。推理从随机噪声开始，对同一个粗到细预测网络执行两个半步更新，输出干净深度图。训练使用有效深度掩码下的速度均方误差。

#### 方法对比分析

与基于潜空间VAE的扩散方案相比，直接对深度图流匹配可减少压缩带来的边界损失；与多步像素扩散相比，两步推理更省计算；与单步生成模型相比，粗到细语义条件试图保留精度。实验显示模型速度和计算量低于若干生成基线，但比UniDepth v2更慢，适用性取决于是否优先考虑局部结构。它只训练于Hypersim，减少了多数据源需求，却也可能限制数据域覆盖。

#### 实验分析（精简版）

FlowMoDE在NYUv2、KITTI、ETH3D、ScanNet和DIODE评测，报告五集平均排名2.2，平均相对误差较所比方法降低14.1%。五个数据集相对误差分别为4.0、6.2、3.9、4.3、18.8。NYUv2和ETH3D点云重建平均Acc为179.5、Chamfer距离为199.7，优于表中对手。计算量为3,140.1 GFLOPs，推理约53.4 ms/图；相较UniDepth v2的23.3 ms/图更慢，但低于多步生成式方法。作者指出patch处理会产生飞点，远处边缘可能模糊。

#### 实用指南

适用于希望兼顾结构细节与低于多步扩散成本的单目深度应用。部署前应在目标相机和真实场景测量延迟、内存与深度误差；不能只看生成模型之间的效率排序。复现时保持输入缩放、深度归一化、无效像素掩码和两步积分一致。远景、稀疏纹理和机器人本体等区域需要单独检查边缘与飞点，必要时进行目标域微调或后处理。

#### 总结

核心思想：语义引导的两步深度流匹配。

1. 从图像提取全局语义并预测粗深度结构。
2. 细解码器恢复边界和高频几何。
3. 两次速度更新生成最终深度图。

五个真实数据集上的结果显示合成到真实迁移潜力，但训练域有限，且推理仍慢于轻量判别方案，需结合目标部署约束评估。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.41190v1)
- [arXiv](https://arxiv.org/abs/2610.41190v1)

---

