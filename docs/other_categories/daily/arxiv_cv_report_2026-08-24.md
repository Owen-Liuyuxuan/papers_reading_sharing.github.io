time: 20260824

# Arxiv Computer Vision Papers - 2026-08-24

## Executive Summary

# ArXiv 计算机视觉论文 Daily 执行摘要  
**发布日期：2026-08-21｜论文数：10篇**

> 注：以下判断主要依据论文标题及研究主题归纳；若需精确比较方法、数据集和实验结果，建议进一步查阅论文摘要与正文。

## 一、总体趋势

本期论文呈现出几个较为集中的研究方向：

1. **视觉—触觉融合与机器人操作**
   - *ViTacPhys*、*VT-MUSE* 聚焦视觉与触觉信息的联合表征，以及物理属性感知的抓取与操作。
   - 研究重点正从“看见物体”转向理解物体的**接触状态、材质、形变和可操作性**。

2. **机器人策略的自我改进与物理推理**
   - *Beyond Imitation* 和 *PhysCaP* 探索如何突破单纯模仿学习，通过离策略 Q 规划、物理约束探索和代码策略提升机器人自主性。
   - 这反映出机器人学习正从“复现示范”发展为“利用示范并主动优化”。

3. **面向真实环境的三维感知与协同感知**
   - *Stream3Dv2* 关注流式、零样本三维场景理解；
   - *CoAnchor* 处理多智能体感知中的时空不同步问题。
   - 共同趋势是提升模型在**在线、开放环境和传感器不完美条件下**的鲁棒性。

4. **几何与物理先验重新成为核心设计要素**
   - *The Coastline as a Structural Constraint* 将海岸线几何用于自主水面船定位；
   - *Neural-Primitive* 使用运动原语进行无人机局部规划；
   - *Fast Coordinated Bimanual Motion Planning* 强调双臂运动规划中的硬约束。
   - 这些工作体现了从纯数据驱动模型转向**学习方法与几何、动力学、约束优化结合**的趋势。

5. **扩散模型和多模态模型的推理效率**
   - *Anchoring Instruction Outside Mask* 关注扩散 Transformer 中上下文指令的高效缓存。
   - 研究重点从单纯提高生成质量扩展到降低重复计算、提升长上下文和迭代生成效率。

---

## 二、值得特别关注的论文

### 1. **ViTacPhys：将物理属性纳入视觉—触觉抓取**
该工作从“视觉—触觉示范”出发，并明确强调物理属性感知，可能涉及材质、刚度、摩擦或形变等因素。其潜在价值在于：

- 提高对透明、柔软、易滑或易损物体的抓取能力；
- 将触觉从事后反馈提升为策略学习中的核心信息；
- 为“以物体物理属性为条件”的通用操作策略提供方向。

**推荐理由：** 视觉—触觉融合是机器人操作的重要增长点，且物理属性建模可能比单纯多模态特征拼接更具长期影响。

### 2. **VT-MUSE：统一的序列视觉—触觉表征学习**
该论文强调 *Multimodal Unified Sequential Representation Learning*，重点可能在于对视觉和触觉时序信号进行统一建模。相比单次感知融合，序列建模更适合：

- 接触前、接触中和接触后的状态预测；
- 从动作结果反推物体属性；
- 支持长时程操作任务中的状态估计与决策。

**推荐理由：** 如果其统一表征能够迁移到多个操作任务或机器人平台，将对多模态机器人基础模型具有参考价值。

### 3. **Beyond Imitation：基于离策略 Q 规划的自我改进策略**
该工作直接针对模仿学习的局限，尝试利用离策略数据和 Q 函数规划提升策略性能。其重要性在于：

- 减少对高质量人工示范的依赖；
- 允许机器人从失败经验和历史数据中持续改进；
- 连接行为克隆、离线强化学习与模型规划。

**推荐理由：** “模仿学习 + 离策略改进”是当前机器人学习走向闭环自主学习的关键路线，建议重点关注其稳定性、数据效率和真实机器人实验。

### 4. **PhysCaP：物理启发的代码策略探索**
该论文将代码作为策略表达形式，并通过物理信息引导探索，可能将高层任务规划、可执行程序和低层物理反馈结合起来。潜在创新包括：

- 让视觉语言或代码代理生成更符合物理规律的动作程序；
- 通过物理约束降低试错成本；
- 增强代理在未见环境中的泛化能力和可解释性。

**推荐理由：** 该方向位于视觉语言代理、机器人规划和世界模型交叉处，可能代表“语言/代码规划器 + 物理执行验证”的新范式。

### 5. **CoAnchor：利用对象级锚点解决协同感知错位**
多车、多机器人协同感知往往受时间延迟、位姿误差、通信不同步影响。该工作以对象级锚点提升时空对齐鲁棒性，具有较强的实际意义：

- 比依赖精确全局坐标变换更适应真实部署；
- 对动态目标和异步传感器更友好；
- 可服务于自动驾驶、无人机编队和多机器人系统。

**推荐理由：** 该问题是协同感知从实验室走向真实环境时的关键瓶颈，建议重点检查其对延迟、定位误差和通信受限条件的评估。

---

## 三、新兴研究方向与技术信号

### 1. 视觉—触觉基础表征逐渐走向时序化
本期两篇视觉—触觉论文同时出现，说明该领域正在从简单的多模态融合转向：

- 统一视觉、触觉和动作序列；
- 学习接触事件和状态转移；
- 通过触觉补足视觉在遮挡、材质和接触状态上的不足。

### 2. 机器人策略从模仿走向“规划—探索—自我改进”
*Beyond Imitation* 与 *PhysCaP* 共同指向一个趋势：机器人不再只是复制示范，而是结合：

- Q 值估计；
- 离策略数据；
- 物理仿真或物理约束；
- 代码化动作规划；
- 在线反馈和失败恢复。

### 3. 几何先验与硬约束重新融入学习系统
*Coastline*、*Neural-Primitive* 和双臂规划工作表明，在安全性、可解释性和数据稀缺场景下，结构化先验仍然十分重要。可能持续升温的技术包括：

- 对象级几何锚点；
- 运动原语；
- 可行域和碰撞约束；
- 物理一致性损失；
- 学习模型与优化器的混合架构。

### 4. 在线三维理解与零样本泛化
*Stream3Dv2* 结合流式处理、几何—语义融合和零样本三维理解，体现了三维视觉系统的两个实际要求：

- 不能只处理离线静态场景；
- 需要在类别变化和开放世界条件下保持泛化能力。

### 5. 面向扩散模型的上下文计算复用
*Anchoring Instruction Outside Mask* 关注扩散 Transformer 中的指令缓存，说明扩散模型优化正在从网络结构改进进一步转向：

- KV 或上下文缓存；
- 重复条件计算消除；
- 长序列推理加速；
- 训练和推理阶段的内存效率优化。

---

## 四、建议优先阅读顺序

### 第一优先级：机器人学习与物理推理

1. **Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning**  
   适合关注强化学习、模仿学习和机器人策略优化的研究人员。

2. **PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration**  
   适合关注视觉语言代理、代码生成、机器人规划和物理世界交互的读者。

3. **ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations**  
   适合研究机器人抓取、触觉感知和多模态示范学习的读者。

4. **VT-MUSE: Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation**  
   建议与 ViTacPhys 配套阅读，比较两者在表征学习、时序建模和任务迁移方面的差异。

### 第二优先级：真实环境感知与协同系统

5. **CoAnchor: Robust Collaborative Perception under Spatio-Temporal Misalignment via Object-Level Anchors**  
   对自动驾驶、多机器人感知和分布式传感系统具有较强应用价值。

6. **Stream3Dv2: Geometric-Semantic Fusion Enhanced Streaming Zero-Shot 3D Scene Understanding**  
   适合关注在线三维感知、开放词汇识别和场景理解的研究人员。

7. **The Coastline as a Structural Constraint: Harnessing Scene Geometry for Autonomous Surface Vessel Localization**  
   对自主航行、海事机器人和基于环境结构的定位方法尤其相关。

### 第三优先级：规划与模型效率

8. **Fast Coordinated Bimanual Motion Planning With Hard Constraints**  
   如果研究重点是双臂操作、实时规划或安全约束，应提升阅读优先级。

9. **Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight**  
   对无人机导航、局部规划和运动原语学习有直接参考价值。

10. **Anchoring Instruction Outside Mask: Exact Reference Caching for Efficient In-Context Diffusion Transformers**  
   对扩散模型推理优化和多模态生成系统工程实现感兴趣的读者值得阅读；其影响范围可能更多集中于模型效率而非机器人或三维视觉本身。

## 五、结论

本期论文的核心信号是：计算机视觉研究正在进一步向**具身智能、物理交互和真实环境部署**延伸。视觉系统不再仅承担识别任务，而是与触觉、动作、几何约束、物理推理和在线规划紧密结合。最值得关注的主线包括：

- 视觉—触觉统一时序表征；
- 模仿学习向离策略自我改进发展；
- 代码代理与物理约束结合；
- 面向异步和不确定环境的对象级协同感知；
- 几何先验与硬约束驱动的高效规划。

若时间有限，建议优先阅读 **Beyond Imitation、PhysCaP、ViTacPhys、VT-MUSE 和 CoAnchor**；它们最集中地体现了当前机器人视觉从感知走向自主决策与物理交互的方向。

---

## Table of Contents

1. [ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations](#2608.21355v1)
2. [VT-MUSE: Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation](#2608.21290v1)
3. [The Coastline as a Structural Constraint: Harnessing Scene Geometry for Autonomous Surface Vessel Localization](#2608.21276v1)
4. [Anchoring Instruction Outside Mask: Exact Reference Caching for Efficient In-Context Diffusion Transformers](#2608.21229v1)
5. [Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning](#2608.21204v1)
6. [Stream3Dv2: Geometric-Semantic Fusion Enhanced Streaming Zero-Shot 3D Scene Understanding](#2608.21136v1)
7. [CoAnchor: Robust Collaborative Perception under Spatio-Temporal Misalignment via Object-Level Anchors](#2608.21055v1)
8. [PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration](#2608.21031v1)
9. [Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight](#2608.20948v1)
10. [Fast Coordinated Bimanual Motion Planning With Hard Constraints](#2608.20946v1)

---

## Papers

<a id='2608.21355v1'></a>
## [ViTacPhys: Physical Property-Aware Grasping from Human Visual-Tactile Demonstrations](https://arxiv.org/abs/2608.21355v1)

**Authors:** Yiwen Liu, Yujun Zhu, Kui Jia, Zhao Liao, Yangwei You, Shuaijun Wang

**Published:** 2026-08-21

**Categories:** cs.RO

**Abstract:**

Recent vision-based action models have demonstrated strong capabilities in complex manipulation, but they rarely leverage explicit object physical properties to adapt their policies. We introduce ViTacPhys, a visual-tactile framework and data acquisition system that estimates object mass and friction-coefficient classes, together with continuous stiffness, from human manipulation demonstrations. Trained on data from 60 rigid and deformable objects, ViTacPhys combines temporal visual-tactile modeling, cross-attention multimodal fusion, and a semantic prior derived from a vision-language model. On seen objects, it achieves 97.2% mass classification accuracy, 98.8% friction-coefficient classification accuracy, and a stiffness mean absolute percentage error (MAPE) of 5.51%. On held-out objects from known categories, it achieves 87.5% mass accuracy, 97.5% friction-coefficient accuracy, and a stiffness MAPE of 9.08%. We transfer ViTacPhys from the human domain to the robot domain using limited robot teleoperation data, robot-style video augmentation, and human demonstrations with matched actions, and deploy it as an online module for adaptive grasping. The resulting physical-property-conditioned policy achieves total grasping success rates of 95.0% on in-distribution objects and 83.4% on out-of-distribution objects. For out-of-distribution objects successfully grasped by both methods, its force profiles are more consistent with human teleoperation than those produced by ACT. These results demonstrate the feasibility of explicitly estimating and conditioning on object physical properties for real-world adaptive grasping.

**Analysis:**

## 1. 摘要翻译
近年来，基于视觉的动作模型在复杂操作中表现突出，但很少显式利用物体物理属性来调整策略。本文提出 **ViTacPhys**，一种结合视觉—触觉的框架及数据采集系统，用于从人类操作示范中预测物体的质量、刚度和摩擦系数类别。模型在60个刚性与可变形物体上训练，融合时序视觉—触觉建模、基于交叉注意力的多模态融合及VLM生成的语义先验。在已见物体上，质量和摩擦系数准确率分别为97.2%和98.8%，刚度MAPE为5.51%；在已知类别中的留出物体上，三者分别为87.5%、97.5%和9.08%。作者进一步利用少量遥操作数据、机器人风格视频增强和动作匹配的人类示范，将模型迁移至机器人，并在线服务于自适应抓取。物理属性条件策略在ID和OOD物体上的总抓取成功率分别达到95.0%和83.4%。

## 2. 方法动机
**驱动力**：相似外观物体可能具有不同重量、软硬度和表面摩擦，固定抓力容易导致滑落、挤压或过度用力。  
**现有痛点**：纯视觉难以识别伪装材料和填充状态；纯触觉只能感知局部且必须接触；许多交互式估计依赖专门的推、拉、滑实验，与最终抓取脱节，并且通常只是离线预测，未进入控制策略。  
**核心假设**：视觉提供外观、形状和填充状态先验，触觉及其随时间变化反映真实接触力学；二者结合可在短暂接触后估计属性，并指导抓力。

## 3. 方法设计详解
### 数据与标签
可穿戴系统同步腕部RGB、拇指/食指/中指压力阵列和动作捕捉。每段示范裁为1秒、30 Hz。质量由五次称量取均值；刚度由稳定加载阶段的力—位移线性回归斜率计算，但它是包含手、传感器和接触顺应性的“操作刚度”；摩擦系数通过硅胶斜面临界角计算，\(\mu_s=\tan\theta\)，因此是接触对属性而非物体固有属性。质量和摩擦用全数据预先拟合的三档有序类别，刚度连续回归。

### 预测器Pipeline
1. 对视觉帧和触觉图计算相邻帧Farneback光流，分别形成内容流与运动流。  
2. 两种模态各自使用ResNet-18提取特征，再投影、拼接并送入GRU，保留14步时序信息。  
3. 视觉查询触觉、触觉查询视觉，进行双向交叉注意力并时序池化，得到视觉—触觉表示。  
4. 仅使用接触前5帧调用VLM，生成类别、材质、纹理、填充状态及属性置信度描述；冻结Sentence-BERT提取句级特征、BERT提取词级特征。  
5. 用视觉—触觉特征查询文本token，并通过可学习门控调节语义先验强度；门控较小时仍保留传感器分支，避免VLM错误主导。  
6. 三个任务头输出质量/摩擦有序logit和连续刚度。质量、摩擦采用带两个有序阈值的ordinal loss，刚度采用MSE，并以GradNorm动态平衡三任务。

部署时，接触检测触发30帧滚动队列；不足帧重复最早接触观测，立即产生预测。对连续窗口进行投票，将质量、刚度十等分bin、摩擦类别和接触状态嵌入为物理属性token，输入ACT式策略。

### 人到机器人迁移
使用同类视觉和触觉传感器降低硬件差异；结合机器人遥操作、动作匹配的人类数据及机器人手风格视频增强，对预测器微调。策略训练时使用带高斯扰动的真实属性标签，部署时换成ViTacPhys预测，提升对预测波动的容忍度。

## 4. 对比与创新
本质区别不是简单把触觉接入ACT，而是把“物理属性估计”作为连接感知与控制的结构化中间变量。创新包括：面向自然人类抓取的显式质量—刚度—摩擦数据；融合内容、光流和VLM先验的时序跨模态预测；通过有限机器人数据将属性感知迁移到自适应抓取。适合接触丰富、物体外观相似但力学差异明显的抓取、搬运和柔性物体操作。

## 5. 实验结论
留出物体上达到87.5%质量准确率、97.5%摩擦准确率和9.08%刚度MAPE；物理属性条件策略相对ACT在ID/OOD上的干净成功率分别提升12.5和38.9个百分点。主要优势是能减少过度用力，并使OOD抓力排序更接近人类。局限是仅60个物体、单人采集，触觉主要测法向压力，OOD试验规模小，VLM首次调用约10秒。

## 6. 实用指南
论文提供项目主页，但给定材料未显示代码、数据或完整训练脚本已公开。复现需重点保持30 Hz同步、15帧窗口/14帧光流、ResNet-18+GRU、学习率 \(5\times10^{-5}\)、30 epochs、batch size 32、AdamW及GradNorm；必须在划分前固定类别边界，避免测试泄漏。迁移到推力估计、插拔或柔性装配时，可将质量等属性替换为任务相关参数，并重新设计测量协议、属性token和下游策略。

## 7. 总结
**核心思想：从人类视触觉示范估计物性并调节抓力。**

**速记Pipeline：**
1. 同步记录人类看到的画面和手指压力。  
2. 从外观、接触变化和动作过程估计物体属性。  
3. 将属性压缩成结构化提示。  
4. 用少量机器人示范迁移并在线调整抓力。

**Key Findings:**

- We introduce ViTacPhys, a visual-tactile framework and data acquisition system that estimates object mass and friction-coefficient classes, together with continuous stiffness, from human manipulation demonstrations.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21355v1)
- [arXiv](https://arxiv.org/abs/2608.21355v1)

---

<a id='2608.21290v1'></a>
## [VT-MUSE: Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation](https://arxiv.org/abs/2608.21290v1)

**Authors:** Congsheng Xu, Qiaochu Yang, Fangyuan Shi, Yifan Han, Baijun Chen, Yiming Wang, Haonan Zhao, Daolin Ma, Xiaokang Yang, Hesheng Wang

**Published:** 2026-08-21

**Categories:** cs.RO, cs.CV

**Abstract:**

We propose VT-MUSE, a Multimodal Unified SEquential representation learning framework for visuotactilemanipulation. Existing approaches often encode visual and tactile observations independently before fusion, limiting their ability to capture fine-grained cross-modal dependencies. Moreover, most methods focus on observations at the current time step and overlook the temporal evolution of contact. VT-MUSE addresses both limitations through a two-stage representation learning framework. In Stage I, modality specific encoders are jointly adapted via cross-modal temporal alignment and masked-view consistency. In Stage II, a conditional variational latent model processes masked visual sequences together with full tactile histories. Auxiliary decoders reconstruct the masked recent visual observations and predict tactile depth changes, encouraging the latent representation to retain both global visual context and local contact dynamics. The learned representation is subsequently integrated into a lightweight Transformer policy through gated cross-attention. On the simulation benchmark, VT-MUSE outperforms the strongest baseline evaluated on all tasks by 11 percentage points and also achieves substantial improvements in real-world experiments.

**Analysis:**

## 1. 摘要翻译
本文提出 VT-MUSE（多模态统一序列视觉触觉表征学习）框架，用于视觉—触觉机器人操作。现有方法通常先独立编码视觉与触觉，再进行融合，难以捕捉细粒度跨模态依赖；同时，多数方法只利用当前时刻，忽略接触状态的时间演化。VT-MUSE采用两阶段学习：第一阶段通过跨模态时间对齐和视觉掩码一致性联合适配模态编码器；第二阶段在被掩码的视觉序列和完整触觉历史上训练条件变分潜变量模型，通过重建近期视觉、预测触觉深度变化，使潜变量同时保留全局视觉环境和局部接触动态。最终，序列表征通过门控交叉注意力接入轻量级Transformer策略。实验表明，该方法在仿真和真实机器人任务上均显著优于基线。

## 2. 方法动机
**痛点：**独立预训练的视觉、触觉特征只在策略阶段融合，无法显式学习“视觉变化—接触变化”的对应关系；短历史、单步预测或与特定策略绑定的动力学模型，也难形成可复用的交互状态记忆。  
**核心假设：**可靠的接触操作状态，应由“场景级视觉上下文+局部触觉几何变化+其时间演化”共同决定；即使近期视觉缺失，触觉历史仍可帮助恢复任务相关状态。

## 3. 方法设计详解
### Pipeline
1. **构造历史窗口：**在时间 \(t\) 取长度为 \(L\)、步长为 \(s\) 的视觉和双侧触觉序列，并加入任务ID。默认最优窗口为 \(L=5\)。  
2. **Stage I模态适配：**视觉与触觉分别输入不共享参数的ViT；双侧触觉图像先空间拼接为三通道图。每个token加入模态、时间槽和相对时间嵌入。使用三种损失：  
   - 对齐损失：同一时刻视觉—触觉token为正样本，批内其他样本为负样本，用双向InfoNCE建立跨模态对应；  
   - 时间一致性损失：相邻时刻的多模态状态靠近，约束接触演化连续；  
   - 掩码一致性损失：同一窗口使用两种随机视觉掩码，使表示保持稳定。仅更新预训练ViT最后三层，减少灾难性遗忘。  
3. **Stage II序列建模：**冻结两种ViT，在视觉序列尾部遮蔽最近 \(K\) 个视觉token，而触觉始终完整可见。按时间交错视觉、触觉token，送入Temporal Transformer。最后 \(K\) 个触觉token作为查询，从全历史中检索与当前接触最相关的信息。  
4. **条件变分表示：**聚合检索结果、触觉查询和任务上下文，得到紧凑潜变量。部署时使用只依赖可见输入的条件先验；训练时额外使用真实视觉尾部构造后验，并通过KL损失令先验逼近后验。  
5. **辅助预测：**潜变量一方面重建被遮蔽的近期RGB图像，另一方面预测双侧触觉深度差分（depth flow）。前者保留物体、位姿和场景几何，后者强化局部接触变化。  
6. **策略接入：**冻结先验编码器，经两层适配器映射为策略记忆；策略Transformer通过门控交叉注意力读取该记忆，而非简单拼接输入。可学习门控初始影响较小，避免破坏原有动作生成路径。

公式上，Stage I为三项损失加权；Stage II为视觉重建、触觉深度预测和KL正则；策略阶段使用动作块L1损失。其本质是把“历史感知”训练成可查询的独立记忆，再交给策略使用。

## 4. 对比与创新
与“当前视觉/触觉直接融合”相比，VT-MUSE显式建模历史和缺失视觉；与普通跨模态对比学习相比，它增加了时间连续性、视觉恢复和触觉几何预测；与世界模型相比，它不预测动作或完整未来轨迹，而是学习可迁移的感知状态。主要创新是**掩码视觉+完整触觉的非对称序列建模**、**视觉重建与触觉深度流的互补监督**以及**门控交叉注意力式策略读取**。适合插入、探索、柔顺接触、遮挡操作等接触丰富任务。

## 5. 实验分析
作者在4个UniVTAC仿真任务和4个真实任务验证。仿真平均成功率为55.25%，高于完整结果最强基线39.00%；真实机器人达到95%。去除视觉重建、触觉预测或时间对比后性能均明显下降，说明全局几何、局部接触和时间结构具有互补性。  
优势是小规模任务示范下仍能利用多任务预训练；不足是窗口步长固定、仅扩展到最多7个任务，且真实实验次数较少，泛化结论有限。

## 6. 实用指南
文中未明确提供代码仓库，仅提到项目页面和视频。复现需准备同步RGB/双侧触觉序列、任务ID及深度流标签；先训练Stage I，再冻结ViT训练Stage II，最后冻结先验编码器训练任务策略。关键设置包括 \(L=5\)、最近视觉尾部掩码、最后三层ViT微调、双向InfoNCE、KL约束和门控交叉注意力。迁移到其他任务时，只需替换传感器预处理、深度变化定义和动作头；若无双侧触觉，可改为单传感器或力觉序列，但需重新设计几何监督。

## 7. 总结
**核心思想：用视觉触觉历史学习可查询的接触记忆。**

**速记版Pipeline：**
1. 收集连续视觉与触觉片段；  
2. 对齐同刻信息并约束相邻状态连续；  
3. 遮住近期视觉，用触觉历史补全交互状态；  
4. 同时恢复视觉和预测触觉变化；  
5. 让动作策略按需读取这段历史记忆。

**Key Findings:**

- We propose VT-MUSE, a Multimodal Unified SEquential representation learning framework for visuotactilemanipulation.
- On the simulation benchmark, VT-MUSE outperforms the strongest baseline evaluated on all tasks by 11 percentage points and also achieves substantial improvements in real-world experiments.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21290v1)
- [arXiv](https://arxiv.org/abs/2608.21290v1)

---

<a id='2608.21276v1'></a>
## [The Coastline as a Structural Constraint: Harnessing Scene Geometry for Autonomous Surface Vessel Localization](https://arxiv.org/abs/2608.21276v1)

**Authors:** Derek R. Benham, Joshua G. Mangelson

**Published:** 2026-08-21

**Categories:** cs.RO, cs.CV

**Abstract:**

Coastal environments contain rich, largely unexploited geometric structure capable of providing globally referenced localization cues. In this work, we present two complementary localization frameworks that exploit shoreline and water-surface geometry for GPS-denied autonomous surface vessel localization. The first framework leverages LiDAR observations of the water surface to estimate roll, pitch, and heave (vertical motion), while recovering global position and heading through direct registration of shoreline observations against a satellite-derived coastline map. The second framework relies solely on passive imagery to detect the shoreline and horizon through semantic segmentation. Using the proposed coastal scene geometry, shoreline distance is inferred from monocular imagery. Shoreline observations are accumulated into short-duration local submaps, registered against the same satellite-derived coastline map, and fused within a hierarchical factor graph. Evaluated across three real-world coastal datasets, the LiDAR pipeline consistently improves trajectory accuracy over standard baselines, while the monocular architecture maintains bounded long-term drift. In addition, we establish that modern zero-shot foundation models can reliably extract shoreline observations across diverse coastal environments. Together, these results demonstrate that coastal geometry provides a powerful and dependable source of globally referenced information for GPS-denied maritime localization.

**Analysis:**

# 1. 摘要翻译

沿海环境包含丰富但尚未充分利用的几何结构，可提供全球参考的定位线索。本文提出两种互补的定位框架，用于无 GPS 条件下的自主水面艇定位。第一种方法利用 LiDAR 对水面的观测估计横滚、俯仰和升沉，并将岸线观测直接与卫星提取的海岸线地图配准，从而恢复全球位置与航向。第二种方法仅依赖被动视觉，通过语义分割检测岸线与地平线，并依据单目几何关系推断岸线距离；随后将岸线观测累积为短时局部子图，与同一卫星海岸线地图配准，并在分层因子图中融合。三组真实沿海数据表明，LiDAR 流程持续优于标准基线，而单目框架能够抑制长期无界漂移。实验还证明，现代零样本基础模型可以在多种沿海环境中可靠提取岸线。总体而言，沿海几何结构是无 GPS 海上定位中有效且稳定的信息源。

# 2. 方法动机与核心假设

**驱动力：**海上缺少陆地机器人常用的密集近距离特征，传统 LiDAR/视觉里程计容易退化；但海岸线、水面和地平线具有天然的几何约束，并且岸线可同时在船载传感器和卫星图像中观察到。

**现有痛点：**传统 SLAM 只能建立局部坐标系；LiDAR 在水面上回波稀疏、受波浪和镜面反射干扰；视觉方法无法直接测距，且岸线远时像素误差会放大为米级位置误差；单一固定窗优化器又无法回溯修正较早轨迹。

**基本假设：**任务时间较短，平均水面可近似为稳定平面；岸线是水面与陆地的边界；卫星地图中的平均岸线与实时可见岸线具有足够结构一致性。

# 3. 方法设计详解

## 3.1 统一场景模型

将平均水面设为世界坐标系中的水平参考面，岸线定义为陆地与水面的交界。若传感器高度为 \(h\)，观测射线相对水平面的俯角为 \(\alpha\)，则岸线距离近似为：

\[
d_{\text{shore}}=h\tan(\alpha)
\]

该关系将视觉中的二维边界转化为水面上的二维空间点。

## 3.2 LiDAR：Coastal-KISS

1. **水面平面估计：**  
   对 LiDAR 原始距离图进行形态学闭运算，抑制孤立回波；每列从低俯角方向向上搜索第一个有效非零回波；对边界进行一维中值滤波，并反投影为三维点。由于这些点可能来自岸线、尾流或波峰，使用 RANSAC 拟合局部水面平面。

2. **恢复姿态与升沉：**  
   平面法向量 \((n_x,n_y,n_z)\) 给出世界竖直方向在传感器坐标系中的表达，进而计算：
   \[
   \phi=\operatorname{atan2}(n_y,n_z),\quad
   \theta=\arcsin(-n_x)
   \]
   平面距离 \(d\) 经姿态补偿后得到高度：
   \[
   z=d\cos\phi\cos\theta
   \]
   因此水面提供横滚、俯仰和高度约束。

3. **构造“模拟卫星视图”：**  
   用估计的横滚、俯仰校正原始点云，将点云投影到水平二维栅格；去除船尾尾流，再用形态学操作填补稀疏植被回波。提取外轮廓，并沿传感器向外发射射线，仅保留每条射线遇到的第一个轮廓点，以近似俯视卫星图像中的可见岸线，避免树冠和深层陆地结构造成匹配偏差。

4. **岸线全球配准：**  
   对卫星图像预先人工提取岸线地图；以当前位姿为初值，在局部窗口内使用 ICP，仅优化平面 \(x,y\) 和航向 \(\psi\)。

5. **因子图融合：**  
   KISS-ICP提供相邻 LiDAR 帧之间的相对位姿；水面因子以约 10 Hz 约束 \((\phi,\theta,z)\)；每十帧将水面估计与岸线配准结果组成完整 SE(3) 位姿因子。固定时延平滑器通过 iSAM2 增量优化，在保持实时性的同时限制漂移。

## 3.3 单目视觉与分层因子图

1. **零样本语义分割：**  
   Grounding DINO 初始化“水、天空”等语义提示，SAM 2 利用时序记忆持续跟踪分割区域。图像每四帧处理一次，约 3.75 Hz；每 100 帧重新初始化以抑制掩膜漂移和显存累积。

2. **边界分类与姿态估计：**  
   逐列自下而上寻找水域上边界；若另一侧是天空，则判为地平线，用 RANSAC 拟合直线并估计横滚、俯仰；若另一侧是陆地，则判为岸线，保留其像素位置。

3. **岸线三维重建：**  
   将每个岸线像素转为相机射线，根据估计姿态变换到世界坐标，并与平均水面求交，获得相对岸线点。连续约 10 秒的观测在局部里程计坐标系中累积，投影到 1 m 栅格，删除低于第 90 百分位密度的单元，再用 PCA 提取局部主岸线。

4. **可观测性感知配准：**  
   在当前全球位置附近截取 \(200m\times200m\) 卫星岸线子图，使用相关扫描匹配搜索 \(x,y\) 平移。由于直线岸线无法可靠估计航向，航向由磁力计提供。匹配热图的高分区域再经 PCA 分析，形成沿岸方向大、不确定度大，垂岸方向小、不确定度小的各向异性因子。

5. **分层优化：**  
   局部固定窗估计器融合 IMU 预积分、地平线姿态、磁力计航向、弱速度先验和弱高度先验，用于实时轨迹与子图构建。全局稀疏位姿图每秒加入关键帧，并连接局部平滑器输出的边缘化相对因子；岸线匹配作为间歇性的全球 \(x,y\) 约束，使新获得的几何信息能够回溯修正历史轨迹。

# 4. 对比、创新与适用性

**本质区别：**方法不是把海上环境当作“特征稀疏的陆地”，而是主动建模水面—岸线—地平线的场景几何，并用卫星岸线作为全球参照。

**主要创新：**  
1. 用水面平面同时恢复 LiDAR 的姿态和升沉；  
2. 将岸线转换为近似卫星视角后进行跨视角配准；  
3. 针对视觉岸线的“沿岸不可观”显式构造各向异性不确定度；  
4. 采用局部高频估计与全局稀疏图分离的层次化架构；  
5. 使用 Grounding DINO+SAM 2 实现无需任务专属训练的岸线提取。

适合近岸、存在可见稳定岸线且可获得卫星地图的 ASV；不适合远离海岸、岸线被遮挡、潮汐快速变化或动态船只密集的场景。

# 5. 实验分析

作者在夏威夷三组、总计近 5 km 的真实数据上评估。代表性结论是：LiDAR Coastal-KISS 在开阔水域将水平误差显著降低至约 7–8 m，而 KISS-ICP约为 22–40 m；视觉分层框架在三组数据上将平移误差控制在约 11–14 m，传统 VINS-Mono 在部分数据上达到数百米或失败。  
优势是无需 GPS、无需专门训练的语义模型，并能抑制长期漂移；局限是依赖先验岸线地图、水面平面和潮位稳定性，视觉航向仍依赖磁力计，LiDAR/视觉均会受到镜面反射影响。

# 6. 实用指南

论文未给出明确代码仓库，不能视为已开源。复现需准备：同步的 LiDAR、相机、IMU、磁力计和高精度真值；标定内外参；制作卫星岸线地图；实现 KISS-ICP、RANSAC、ICP/相关匹配、GTSAM/iSAM2。关键设置包括 1 m 栅格、视觉约 10 s 子图、90%密度筛选、\(200m\times200m\) 搜索窗口、每十帧 LiDAR 全位姿更新和每 100 帧视觉掩膜重初始化。迁移到湖泊、河流或港口时，应重新估计水面高度、潮位变化、传感器高度和岸线地图，并针对反射、动态目标加入鲁棒核或时序一致性筛选。

# 7. 总结

**核心思想：**用海岸几何替代GPS定位。

**速记版 Pipeline：**

1. 从水面或地平线估计船体姿态与高度；  
2. 从 LiDAR 或图像提取岸线；  
3. 将岸线转换到局部地图并累积降噪；  
4. 与卫星岸线地图匹配获得全球位置；  
5. 在局部实时估计器和全局图中融合，持续回溯纠偏。

**Key Findings:**

- In this work, we present two complementary localization frameworks that exploit shoreline and water-surface geometry for GPS-denied autonomous surface vessel localization.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21276v1)
- [arXiv](https://arxiv.org/abs/2608.21276v1)

---

<a id='2608.21229v1'></a>
## [Anchoring Instruction Outside Mask: Exact Reference Caching for Efficient In-Context Diffusion Transformers](https://arxiv.org/abs/2608.21229v1)

**Authors:** Yangshuai Liu, Zheming Li, Jiaao Li, Kang He, Ziliang Lai, Zhitai Liu, Chengru Song

**Published:** 2026-08-21

**Categories:** cs.CV

**Abstract:**

Omnimodal generation is central to a wide range of content creation and editing applications. In-context conditioning is essential to this paradigm. It allows diffusion transformers to process text instructions and visual references in a shared attention sequence. However, each reference image introduces thousands of tokens. Computation therefore grows rapidly with the number of references. Existing methods reduce computation through structured sparse attention, which limits interactions between reference and target tokens. This structure also makes the reference K and V independent of the denoising target, allowing them to be computed once and reused across steps. However, it blocks visual references from attending to the text instruction. This substantially degrades instruction following and reference fidelity in multi-reference editing. To resolve this conflict, we jointly redesign the token sequence and attention mask. Our beyond-mask design uses static text anchors to connect the instruction to the reference branch. It preserves exact K and V reuse without adding parameters. However, this direct architectural conversion degrades generation quality. We recover the lost performance through teacher-forced velocity distillation, followed by a short on-policy stage in which the teacher supervises student-visited states. To our knowledge, this is the first use of on-policy distillation for architectural recovery in diffusion models. Across three image-editing benchmarks, our method matches full-attention generation quality. With five reference images, it accelerates the complete 40-step denoising process by 3.92x, while static text anchors introduce negligible runtime overhead; the speedup reaches 5.47x at ten references in our scaling study.

**Analysis:**

## 1. 摘要翻译

上下文生成是内容创作与编辑的重要能力。扩散Transformer通常将文本指令、目标图像和参考图像放入同一注意力序列，但每张参考图像会引入数千个token，使计算量随参考图像数量快速增长。现有结构化稀疏注意力通过隔离参考分支，使其Key和Value与去噪目标无关，从而可以跨步复用；但这也阻断了参考图像访问文本指令的路径，导致多参考编辑中的指令遵循和主体保真度下降。

本文联合重新设计token序列与注意力掩码，提出一种“掩码之外”的静态文本锚点机制：利用固定文本锚点向参考分支传递指令信息，同时保持参考K/V完全可缓存，且不引入新参数。由于结构改变会造成质量下降，作者进一步采用“教师强制速度蒸馏+短程在线策略蒸馏”恢复性能。实验表明，该方法在三个图像编辑基准上接近全注意力模型；多参考场景下可实现约3.92倍加速，十张参考图像时达到5.47倍。

## 2. 方法动机

**核心驱动力**：参考图像token数量大，且全注意力会在每个去噪步重复计算参考分支；作者希望同时获得“参考特征精确复用”和“参考内容理解文本指令”。

**现有痛点**：  
1. 全注意力计算和显存开销随参考数量迅速增长。  
2. 隔离式缓存虽然能复用参考K/V，却使参考token只能看到原始、未被目标上下文更新的文本，容易选错主体或参考属性。  
3. 仅修改原序列上的mask无法兼得两者：文本若先接收目标信息，再传给参考分支，就会形成目标到参考的依赖，缓存失效。

**基本假设**：只要建立一条“包含指令、但不包含动态目标信息”的静态路径，就能让参考表示指令感知，同时保持跨步不变。

## 3. 方法设计详解

### 3.1 整体流程

输入为文本指令、噪声目标图像和多张参考图像。序列扩展为：

\[
[LT,\ X_t,\ ST,\ R]
\]

其中，\(LT\)是正常文本分支，\(X_t\)是随去噪变化的目标token，\(R\)是参考token，\(ST\)是复制的静态文本锚点。

**步骤1：构建缓存。**  
将\(ST\)的时间步固定为0，并让\(ST\)与\(R\)双向注意力交互。与此同时，\(ST\)和\(R\)都禁止接收\(LT\)或\(X_t\)的信息。这样，静态子图\([ST,R]\)只依赖指令和参考图像，可在去噪前执行一次。随后仅保存参考分支的K/V，丢弃ST隐藏状态。

**步骤2：逐步去噪。**  
每个去噪步只计算\(LT\)和\(X_t\)的动态分支，并让它们访问已缓存的参考K/V。参考侧不再重复经过Transformer，因此其计算成本被从每一步移除。

**步骤3：结构恢复训练。**  
由于学生模型的注意力拓扑已改变，直接继承教师权重会产生分布和信息流不匹配。作者冻结全注意力教师，训练缓存式学生预测教师速度场。

### 3.2 两阶段蒸馏

**阶段一：教师强制速度蒸馏。**  
由干净样本\(x_0\)和高斯噪声\(x_1\)构造流匹配状态：

\[
x_t=(1-t)x_0+t x_1
\]

在数据分布产生的状态上，最小化学生与教师速度预测的MSE：

\[
\|v_\theta(x_t,c,t)-v_{\text{full}}(x_t,c,t)\|_2^2
\]

它主要修复结构修改带来的初始性能损失。

**阶段二：在线策略蒸馏。**  
让学生从噪声出发完整采样，收集其真实去噪轨迹上的状态；对这些状态停止梯度，再查询教师速度并进行同样的MSE回归。其关键不在于改变教师目标，而在于改变“教师被查询的位置”：从数据插值状态转为学生实际会访问的状态，因此可直接修正学生推理路径上的误差。实验中使用40步rollout，每条轨迹采样4个状态。

## 4. 方法对比与创新

与普通隔离缓存相比，本文不是简单放宽mask，而是**增加静态token载体，改变依赖图**；因此在不打开\(X_t\rightarrow R\)路径的情况下，实现\(ST\rightarrow R\)的指令传递。与近邻时间缓存相比，它是结构保证下的精确缓存，而非近似复用。

主要创新包括：  
1. 无新增参数的静态文本锚点；  
2. 对“原序列mask为何无法解决问题”给出依赖图层面的不可能性分析；  
3. 首次将学生自采样状态上的在线策略蒸馏用于扩散架构恢复。

适合多参考图像编辑、视觉上下文生成以及参考token占主导的扩散Transformer。若参考数量较少，缓存收益有限；若模型结构不保留文本—目标交互，锚点的必要性也会降低。

## 5. 实验分析

作者在OmniContext、GEdit-Bench和ImgEdit-Bench上比较全注意力、隔离缓存和本文方法，并进行在线蒸馏及参考数量扩展实验。代表性结论是：本文模型质量接近甚至略高于全注意力教师，同时五张参考图像时约3.92倍加速、十张时5.47倍加速。继续教师强制训练基本饱和，而在线策略蒸馏仍能提升性能。

**优势**：精确缓存、无额外参数、兼容常规高性能注意力内核、参考越多收益越明显。  
**局限**：需要针对新注意力拓扑进行蒸馏；阶段二仍需教师rollout，训练成本较高；静态锚点虽固定，却可能与某些模型的时间嵌入或位置编码设计不兼容。论文中摘要称“五张参考图像”，正文效率表述出现“五张/八张”不一致，复现时应核对实验配置。

## 6. 实用指南

论文未明确提供代码或正式模型权重，不能视为已开源实现。复现需：复制文本嵌入形成ST；固定其时间步为0；实现\([ST,R]\)一次性前向并缓存R的逐层K/V；动态阶段只运行LT与目标分支；先进行30k步教师强制蒸馏，再进行500步在线蒸馏。关键设置包括学习率\(5\times10^{-6}\)、bf16、全参数训练、梯度裁剪0.05、40步rollout、每条轨迹4个查询状态；训练阶段不使用CFG，推理时使用CFG=4。

该思想可迁移到视频扩散、3D生成或多模态Transformer：只要条件分支可被隔离，并能构造一个固定的条件载体，就可将“动态上下文访问”和“可缓存条件表示”解耦。

## 7. 总结

**核心思想：用静态文本锚点实现指令感知缓存。**

**速记版Pipeline：**
1. 复制一份固定文本，作为参考图像的临时指令入口。  
2. 只让固定文本与参考图像交互，提前算好参考特征。  
3. 丢弃固定文本，去噪时反复复用参考特征。  
4. 先用教师在普通状态上校准学生。  
5. 再让学生自己采样，并在其实际轨迹上继续纠错。

**Key Findings:**

- Across three image-editing benchmarks, our method matches full-attention generation quality.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21229v1)
- [arXiv](https://arxiv.org/abs/2608.21229v1)

---

<a id='2608.21204v1'></a>
## [Beyond Imitation: Self-Improving Robot Policies via Off-Policy Q-Planning](https://arxiv.org/abs/2608.21204v1)

**Authors:** Varun Giridhar, Anant Khandelwal, Jeremy A. Collins, Ignat Georgiev, Animesh Garg

**Published:** 2026-08-21

**Categories:** cs.RO, cs.LG

**Abstract:**

Behaviour Cloning (BC) has driven remarkable progress in robot manipulation, yet it is fundamentally limited by its inability to self-improve: a policy that fails cannot learn from that failure without additional human demonstrations. Reinforcement Learning fine-tuning offers a path to self-improvement but has proven difficult to scale to the multi-billion-parameter models underpinning modern robot policies. We propose Q-Planning, which equips a large visuomotor BC policy with a small off-policy Q-function. Because a Q-function estimates value rather than imitates actions, it can be trained on the same successful demonstrations as the BC policy and later absorb both successful and failed deployment rollouts, an asymmetry BC does not have. We exploit this asymmetry to enable value-guided action selection at inference (a single-step Q-weighted average over BC draws) and online self-improvement that fine-tunes only the Q-function, leaving the BC weights untouched. On LIBERO and bimanual RoboTwin, ten iterations of self-improvement lift every benchmark score we tested (LIBERO-10 93% to 99%, RoboTwin 83.8% to 91.4%) and shorten successful episodes on the near-ceiling suites (LIBERO-Object, LIBERO-Goal). On two contact-rich bimanual real-robot tasks, the same loop (BC frozen, no human intervention) improves purely from its own deployment rollouts: stack-cups 40% to 90% and insert-wallet 25% to 80% in five iterations, whereas SFT on successful rollouts alone stalls at 55% and 30%. Under an identical online budget Q-Planning is the only method, among Best-of-N, filtered SFT, IBRL, DSRL, and DAWR, that improves stably from failures without training an auxiliary actor.

**Analysis:**

# 1. 摘要翻译

行为克隆（BC）推动了机器人操作的快速发展，但其根本局限在于无法自我改进：策略失败后，若没有额外的人类示范，就无法从失败中学习。强化学习微调提供了自我提升的可能，但难以扩展到支撑现代机器人策略的数十亿参数模型。本文提出 **Q-Planning**：为大型视觉运动行为克隆策略配备一个较小的离策略 Q 函数。由于 Q 函数估计的是价值而非模仿动作，它既可使用成功示范训练，也能进一步吸收部署过程中的成功与失败轨迹，这是 BC 不具备的能力。作者利用这一不对称性，在推理时从 BC 采样多个动作块，并通过 Q 值加权平均选择动作；在线自我改进时仅微调 Q 函数，保持 BC 权重不变。在 LIBERO 和 RoboTwin 上，十轮自我改进将性能分别提升至 99% 和 91.4%；真实双臂任务中，stack-cups 从 40% 提升至 90%，insert-wallet 从 25% 提升至 80%。在相同在线数据预算下，Q-Planning 是唯一能够稳定利用失败轨迹改进、且无需额外 actor 的方法。

# 2. 方法动机分析

**驱动力：** 大型 BC/VLA 模型能力强但受“示范数据上限”限制；直接微调整个策略成本高，且容易破坏原有模仿能力。

**现有痛点：**
- BC 只能模仿成功行为，失败轨迹无法直接用于训练；
- 在线 RL 需要更新巨大策略网络，计算昂贵、训练不稳定；
- 随机搜索或 MPPI 在高维动作块空间中容易产生脱离 BC 分布的异常动作；
- 许多价值引导方法只能做一次推理时重排，不能持续吸收部署经验。

**核心假设：** BC 已经能够以一定概率生成成功动作；因此无需重训 actor，只需学习“哪些 BC 候选更可能成功”，并用失败数据校准 Q 函数，就能逐步放大正确行为。

# 3. 方法设计详解

## 3.1 整体流程

1. **初始化数据与模型**  
   使用成功示范训练大型 BC 策略 \(\pi_{BC}\)，并冻结其参数。另训练一个独立的 Q 网络，初始数据同样来自示范集。

2. **动作候选生成**  
   给定图像观测 \(o_t\) 和语言指令 \(\ell\)，BC 通过 3 步 flow-matching 采样 \(N\) 个长度为 \(H=32\) 的动作块，而不是只输出单一均值。候选保留了 BC 的行为流形与多模态特征。

3. **Q 评分与动作聚合**  
   Q 网络分别计算候选动作块的价值：
   \[
   w_n=\frac{\exp(Q_\phi(o,\ell,a_n)/\lambda)}
   {\sum_m\exp(Q_\phi(o,\ell,a_m)/\lambda)}
   \]
   最终执行加权平均动作块：
   \[
   \bar a=\sum_n w_n a_n
   \]
   这不是简单选择最高 Q 值的候选，而是进行软组合，可降低单个候选估计误差造成的突变。

4. **部署并收集数据**  
   执行规划器，记录完整成功或失败轨迹。失败轨迹保留在 replay buffer 中，而不是被丢弃。

5. **仅更新 Q 函数**  
   使用 chunk-level Bellman 目标训练 Q：
   \[
   y=\sum_{i=0}^{H-1}\gamma^i r_{t+i}
   +\gamma^H Q_{\bar\phi}(o_{t+H},\ell,a_{t+H:t+2H})
   \]
   下一动作块直接取自 buffer，而非每次训练都重新调用规划器。目标网络通过 EMA 更新。BC 始终冻结，随后进入下一轮部署。

## 3.2 模型结构与关键设计

Q 网络拥有独立的 DinoV2 图像编码器、T5 文本编码器和约 5 亿参数的 Transformer decoder。视觉与语言特征作为上下文，动作块转化为 query token，通过 cross-attention 得到价值预测。

作者没有直接回归标量 Q 值，而是采用 **HL-Gauss 分类式价值学习**：将目标价值投影到 \([0,1]\) 的 101 个离散 bin 上，用交叉熵训练，再对类别概率求期望得到 Q 值。该设计更适合稀疏、近似二值的成功回报，可减轻 MSE 对异常目标和尺度的敏感性。

**Q-chunking** 将连续 \(H\) 步动作作为一个“超动作”，使 bootstrap 频率下降、有效时序长度缩短，从而缓解长时域误差累积。

# 4. 方法对比分析

**本质区别：** 主流 RL 更新 actor，Q-Planning 将“行为生成”和“行为评价”彻底解耦：BC 负责提出可信候选，Q 负责从中选择和组合；在线学习只发生在较小的 critic 上。

**主要创新：**
1. 将大型冻结 BC 视为可采样的 proposal distribution，而非待优化 actor；
2. 通过 Q 加权平均利用多模态 BC 候选，避免高成本迭代搜索；
3. 首次将成功、失败部署数据统一用于 Q-only 自我改进；
4. 无需辅助 actor，降低大模型在线训练成本。

**适用场景：** 有较强初始示范策略、任务具有成功检测信号、动作空间连续且可由 BC 产生多样候选的机器人操作任务。若 BC 完全不会生成成功行为，Q 函数无法创造新行为。

# 5. 实验分析

作者在四个 LIBERO 套件、47 个 RoboTwin 任务及两个真实双臂接触任务上比较冻结 BC、离线 Q 引导、Q-Planning 自改进和多种在线基线。

代表性结论：
- LIBERO-10 从 93% 提升至 99%，RoboTwin 从 83.8% 提升至 91.4%；
- 真实任务中 stack-cups 从 40% 到 90%，insert-wallet 从 25% 到 80%，而仅用成功轨迹 SFT 明显停滞。

**优势：** BC 不被破坏；能利用失败；无需额外 actor；推理延迟可控。  
**局限：** 依赖 BC 的探索边界、Q 值可能存在分布外误估计，并依赖可靠的终止成功标签；候选数增大时 Q decoder 成为瓶颈。

# 6. 实用指南

论文提供项目主页，但正文未明确说明完整训练代码、模型权重和数据是否全部开源，因此不能视为完全可复现开源项目。复现关键是：冻结已有 BC；训练独立视觉语言 Q 网络；采用 \(H=32\)、\(N=64\)（LIBERO）或 \(N=32\)（RoboTwin）、\(\lambda=1\)、\(\gamma=0.99\)、101 个 HL-Gauss bins；每轮收集在线轨迹后进行 200 次 Q-only 更新，batch 一半来自示范、一半来自在线数据。迁移到其他任务时，只需替换候选生成器、观测编码器和成功判定器；前提是候选动作足够多样且价值标签可靠。

# 7. 总结

**核心思想：** 冻结策略，用价值函数从失败中学习。

**速记版 pipeline：**
1. 用示范训练一个固定的动作生成器。  
2. 为每个状态生成多个可能动作。  
3. 用价值模型给动作打分并加权执行。  
4. 收集成功与失败轨迹。  
5. 只更新价值模型，循环提升选择能力。

**Key Findings:**

- We propose Q-Planning, which equips a large visuomotor BC policy with a small off-policy Q-function.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21204v1)
- [arXiv](https://arxiv.org/abs/2608.21204v1)

---

<a id='2608.21136v1'></a>
## [Stream3Dv2: Geometric-Semantic Fusion Enhanced Streaming Zero-Shot 3D Scene Understanding](https://arxiv.org/abs/2608.21136v1)

**Authors:** Jie Xu, Na Zhao

**Published:** 2026-08-21

**Categories:** cs.CV

**Abstract:**

Recently, open-vocabulary zero-shot 3D scene understanding using vision foundation models has emerged as a promising alternative to data-intensive supervised methods. However, deploying these models in real-world scenarios is severely hindered by their inability to efficiently handle streaming RGB-D inputs and their inherent vulnerability to noise 2D segmentation masks. To address these critical limitations, we propose Stream3Dv2, a novel training-free framework designed for robust streaming 3D perception. Stream3Dv2 processes sequential data through an original nested local-to-historical architecture, capturing multi-view consistency while circumventing the high computational overhead so as to support timely responses. At its core, we introduce a comprehensive geometric-semantic fusion mechanism that resolves geometric noise and semantic ambiguity by explicitly utilizing semantic guidance and formulating 3D segmentation as solving point-and-set merging and partitioning problems. Furthermore, we present an innovative manifold-distance-based point cloud refinement strategy. This approach leverages local manifold graphs for point-to-manifold optimization that mitigates the boundary delineation failures caused by Euclidean-distance metrics, and employs geometric bounding boxes to dynamically activate and update historical instances for achieving rapid manifold-to-manifold refinement. Extensive experiments on public datasets demonstrate that Stream3Dv2 consistently outperforms existing baselines in foundational open-vocabulary streaming 3D segmentation and detection. Finally, we show that integrating our framework with an LLM-based agent enables advanced language-driven 3D scene understanding, underscoring its potential for open-world embodied intelligence. Code will be updated at https://github.com/SubmissionsIn/Stream3D.

**Analysis:**

# 1. 摘要翻译

近年来，利用视觉基础模型进行开放词汇零样本三维场景理解受到关注，但其难以高效处理流式 RGB-D 输入，并且容易受到二维分割噪声的影响。为此，本文提出无需训练的 Stream3Dv2，用于鲁棒的流式三维感知。方法采用嵌套的“局部到历史”架构，在降低计算开销的同时利用多视角一致性。其核心是几何—语义融合机制：通过语义引导，将三维分割转化为点集覆盖、掩码合并与集合划分问题，以抑制几何噪声和语义歧义。此外，作者提出基于流形距离的点云优化策略：利用局部流形图进行点到流形分配，避免欧氏距离跨越曲面或薄结构造成的边界错误；再借助几何包围盒快速激活和更新历史实例，实现高效的流形到流形优化。实验表明，该方法在开放词汇流式三维分割与检测上优于现有方法，并可结合大语言模型实现语言驱动的三维场景理解。

# 2. 方法动机

**驱动力：**在没有三维标注、且 RGB-D 帧持续到达的情况下，实时获得稳定的三维实例及其语义标签。  
**现有痛点：**离线方法需要完整场景和全部历史帧，计算随序列增长；二维 VFM 掩码投影到三维后会产生断裂、过分割、错分类和跨实例污染；传统基于欧氏距离的补点还可能跨越曲面边界。  
**核心假设：**网格提示具有高覆盖率但缺少语义，语义提示边界准确但覆盖不足；局部多视角的一致性可去噪，而点云实例更适合用流形上的距离传播进行补全。

# 3. 方法设计详解

整体采用滑动窗口，将流式任务拆为两个嵌套阶段：

**(1) 二维分割与三维投影。**对最近 \(k\) 帧分别使用网格提示生成 class-agnostic 掩码，用语义词提示生成带类别的掩码；结合深度和相机位姿反投影到三维，得到局部点云掩码。

**(2) 局部粗粒度分割。**  
① 单帧几何去噪：要求掩码内部点通过 DBSCAN 保持连通，并删除距离其他掩码过近的边界点。  
② 多视角噪声过滤：对局部点云进行 FPS 采样得到关键点，把所有候选掩码视为集合，使用贪心 SCP 选择覆盖全部关键点且数量尽可能少的“关键掩码”，从而去除冗余和孤立噪声。  
③ 关键掩码合并：计算掩码 IoU，IoU 超过阈值的掩码建边，通过连通分量合并，获得覆盖较完整的粗粒度实例。

**(3) 语义驱动细粒度分割。**先对语义掩码进行逐点投票并去重。对每个语义掩码，计算其被各粗掩码包含的比例，分配给包含度最高且超过阈值的粗掩码。随后通过 SPP 决定：保留原粗掩码，还是用多个语义子区域拆分它；与任何粗掩码不重叠的语义掩码则作为新实例。该步骤不是简单求交，而是利用语义信息主动纠正过合并和过分割。

**(4) 点到流形优化。**将体素内点聚合为 super-point，计算其协方差并取逆作为局部黎曼度量；相邻 super-point 间采用对称 Mahalanobis 距离建图。该距离使沿表面方向移动代价较低、垂直表面移动代价较高。以已有掩码点为多个源，通过 Bellman 迭代求各实例的图最短路距离，将未分配点归入流形距离最小的实例。

**(5) 局部到历史更新。**为局部掩码建立 AABB，仅检索与其空间相交的历史掩码。若存在点级重叠，则视为动态实例，并将局部掩码合并到重叠最多的历史实例，同时从其他实例中删除重复点；无重叠者视为静态实例，完全没有候选者的局部掩码作为新实例。

# 4. 对比与适用性

其本质区别是：从“全局离线匹配”转向“局部处理、历史更新”，从单一几何或语义线索转向显式的几何—语义协同，并以流形图而非欧氏距离完成补点。创新主要集中在 SCP 去噪、SPP 语义拆分、黎曼图优化和包围盒加速更新。适合室内 RGB-D、机器人在线建图、开放词汇查询和具身导航；对大规模室外、严重动态遮挡和低质量位姿仍不够稳健。

# 5. 实验分析

作者在 ScanNet200、ScanNet++ 和 MatterPort3D 上评估分割，在 ScanNet200 上评估检测，并进行模块、提示方式、参数及效率消融。代表性结论是：Stream3Dv2 在三数据集上整体优于流式零样本基线；ScanNet200 上 class-agnostic AP 为 27.1%、semantic AP 为 20.6%，明显超过 Stream3D。其优势是无需训练、语义与几何互补、可流式运行；局限是依赖 SAM 类模型和预设语义词，历史内存仍增长，参数及几何阈值具有经验性。

# 6. 实用指南

论文提供代码链接：<https://github.com/SubmissionsIn/Stream3D>。复现时需准备 RGB-D、相机位姿、点云投影和二维 VFM；默认 \(k=20\)、关键点率 \(\gamma=0.05\)、IoU 阈值 \(\alpha=0.2\)、距离阈值 \(\delta=0.05\)、Bellman 迭代 \(R=5\)。可迁移到三维检测、目标检索、语言 grounding、导航和机器人操作；迁移重点是替换二维提示词、开放词汇分类器及下游决策模块。

# 7. 总结

**核心思想：**局部几何语义融合，持续优化历史实例。

**速记版 pipeline：**

1. 用网格提示找全对象，用语义提示补类别。  
2. 将多帧掩码投影到三维并清除噪声。  
3. 依据空间覆盖合并对象，再用语义拆分错误合并。  
4. 沿点云表面传播距离，补回未标记点。  
5. 用包围盒匹配新旧实例，持续更新三维场景。

**Key Findings:**

- To address these critical limitations, we propose Stream3Dv2, a novel training-free framework designed for robust streaming 3D perception.
- At its core, we introduce a comprehensive geometric-semantic fusion mechanism that resolves geometric noise and semantic ambiguity by explicitly utilizing semantic guidance and formulating 3D segmentation as solving point-and-set merging and partitioning problems.
- Furthermore, we present an innovative manifold-distance-based point cloud refinement strategy.
- Extensive experiments on public datasets demonstrate that Stream3Dv2 consistently outperforms existing baselines in foundational open-vocabulary streaming 3D segmentation and detection.
- Finally, we show that integrating our framework with an LLM-based agent enables advanced language-driven 3D scene understanding, underscoring its potential for open-world embodied intelligence.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21136v1)
- [arXiv](https://arxiv.org/abs/2608.21136v1)

---

<a id='2608.21055v1'></a>
## [CoAnchor: Robust Collaborative Perception under Spatio-Temporal Misalignment via Object-Level Anchors](https://arxiv.org/abs/2608.21055v1)

**Authors:** Chi Li, Rui Lin, Aobo Ji, Dongzhu Xu

**Published:** 2026-08-21

**Categories:** cs.CV, cs.AI

**Abstract:**

Collaborative perception extends the sensing range of a single vehicle by fusing observations from nearby agents, which improves the robustness of autonomous driving. In realistic deployments, however, the received collaborator messages are often affected by both communication delay and relative-pose noise, which jointly cause stale observations, spatial misalignment, and unstable feature fusion. Existing methods usually address these issues from either the spatial or temporal side, but handling them jointly in a unified and efficient manner remains challenging. In this paper, we propose CoAnchor, an anchor-centric spatio-temporal alignment framework for asynchronous collaborative perception. Instead of directly reasoning on dense BEV features, CoAnchor builds sparse object-level spatio-temporal anchors as a shared interface for pose correction and tightly connects spatial refinement, temporal propagation, and current-time verification within one unified loop, while keeping the overall correction process lightweight. Extensive experiments on both simulated and real-world datasets illustrate that CoAnchor remains competitive under clean settings and improves the robustness under joint delay and pose perturbations with a favorable practical accuracy-efficiency trade-off.

**Analysis:**

## 1. 摘要翻译

协同感知通过融合邻近车辆的观测，扩展单车感知范围并提升自动驾驶的鲁棒性。然而在真实部署中，接收到的协作者消息往往同时受到通信延迟和相对位姿噪声影响，导致观测过时、空间错位以及特征融合不稳定。现有方法通常分别从空间或时间角度处理这些问题，但如何以统一且高效的方式联合解决二者仍较困难。本文提出 **CoAnchor**，一种面向异步协同感知的、以对象为中心的时空对齐框架。CoAnchor不直接在稠密BEV特征上推理，而是构建稀疏的对象级时空锚点，将位姿修正、时间传播和当前时刻验证连接到统一闭环中，同时保持较低的计算开销。在模拟和真实数据集上的实验表明，CoAnchor在干净条件下具有竞争力，并在联合延迟与位姿扰动下获得更强鲁棒性和较好的精度—效率权衡。

## 2. 方法动机分析

**驱动力**：协作者特征同时存在“位置不准”和“时间过时”两类误差。若先粗略修正位姿，再直接进行时间补偿，错误的空间起点会被继续传播，最终产生重复框、错位框和运动鬼影。

**现有痛点**：CoAlign等空间方法依赖高质量检测框和正确匹配，容易被错误对应关系带偏；TraF-Align等时间方法假设输入已处于可靠空间坐标，无法识别传播后仍存在的空间偏差；简单串联两者并不能形成可靠的当前时刻证据。

**核心假设**：对象的中心、速度、朝向及短期轨迹，比稠密BEV特征更适合作为空间校正、运动预测和可靠性判断的共同接口。只有同时满足历史一致性和当前时刻一致性的匹配对象，才应参与后续位姿优化。

## 3. 方法设计详解

### Pipeline

1. **对象检测与历史构建**  
   每个车辆运行单车检测器，得到对象框及短期历史。对象状态表示为位置、平面速度和朝向  
   \([p_x,p_y,v_x,v_y,\psi]\)。

2. **延迟时刻匹配**  
   将邻车在延迟时刻的检测框用原始位姿粗略变换到ego坐标系。先用位置距离门限剔除明显不可能的匹配，再依据速度差和局部邻域结构建立代价矩阵，利用Hungarian算法完成一对一匹配。

3. **轨迹感知位姿修正**  
   对匹配对象在多个历史帧上的中心进行加权刚体对齐。权重初始由匹配代价转换而来，采用Huber损失和IRLS求解SE(2)变换，从而降低异常匹配对全局位姿的影响。

4. **构建时空锚点并传播**  
   将可靠匹配转为带协方差的锚点。使用常速度模型将延迟状态传播到当前时刻：位置按速度更新，协方差随延迟增大。随后利用当前ego检测框的位置和朝向进行Kalman式校正。

5. **当前时刻验证与闭环反馈**  
   计算预测与当前观测之间的归一化创新平方（NIS）。若超过卡方阈值，或找不到可靠当前匹配，则该对象降级为普通传播对象，不再参与位姿反馈。对保留锚点进一步计算短历史不一致度，并以  
   \(q=q^{(0)}\exp[-\lambda(d^2+r^{hist})]\)  
   更新匹配权重，再反馈给下一轮IRLS位姿优化。默认仅使用一轮反馈。

6. **锚点引导特征融合**  
   用最终修正位姿将邻车BEV特征变换到当前ego坐标系，并将邻车框区域通过非学习式box-wise feature mover移动到传播后的当前位置，之后再与ego特征进行多尺度融合和检测解码。

### 设计本质

CoAnchor并非简单串联“空间模块+时间模块”，而是将**匹配、位姿、运动预测、当前验证和可靠性反馈**组织为闭环。它只在稀疏对象锚点上进行复杂修正，避免反复处理稠密BEV特征。

## 4. 方法对比与适用场景

其根本区别是：主流方法多在特征层隐式学习对齐，或将空间校正和时间补偿作为独立阶段；CoAnchor则以对象级状态作为显式中间表示，并允许当前时刻观测反向修正早期匹配和位姿。

主要创新包括：  
1. 短历史辅助的对象匹配与轨迹位姿优化；  
2. 基于当前时刻NIS检验的选择性锚点校正；  
3. 将验证结果反馈至后续位姿优化的闭环机制；  
4. 面向前景对象的轻量特征移动。

适合车车协同、传感器异步、定位不稳定且对象运动相对连续的场景，尤其适用于真实道路中的中等延迟和中等位姿噪声。

## 5. 实验分析

作者在OPV2V和V2V4Real上，与CoAlign、TraF-Align、V2X-ViT及级联方法比较，并进行模块消融和运行时间测试。代表性结论是：在V2V4Real的200 ms延迟、(0.6 m, 0.6°)噪声下，CoAnchor达到67.04 AP@0.5，较级联基线高6.00；去除历史初始化、ego校正或反馈均会下降。单轮反馈约51.62 ms/frame，明显快于V2X-ViT和TraF-Align。

优势是对联合时空扰动鲁棒、能选择性抑制错误协作者证据、额外计算开销小。局限是常速度模型难以处理急转弯和长延迟，且性能仍依赖检测质量与跨车匹配准确性。

## 6. 实用指南

文中未明确给出公开代码链接，但提供了较完整的实现配置。关键设置包括：历史长度3帧、位置门限4 m、至少3条轨迹才优化位姿、Huber参数0.5、NIS阈值7.815、闭环轮数1。数据以10 Hz处理，协同检测器基于PointPillars，多尺度BEV融合；训练时对协同分支注入位姿噪声，推理时运行规则式对象模块。

迁移到多目标跟踪、跨摄像头融合或机器人协作时，可将“对象框”替换为目标状态，将SE(2)替换为相应空间变换，并保留“预测—当前验证—可靠性反馈”的闭环结构。

## 7. 总结

**核心思想：用可验证对象锚点闭环对齐时空信息。**

**速记版Pipeline：**  
1. 用历史轨迹匹配两车对象；  
2. 通过可靠对象修正相对位姿；  
3. 将延迟对象预测到当前时刻；  
4. 用当前观测筛掉错误对象并反馈权重；  
5. 移动可信对象特征后完成融合。

**Key Findings:**

- In this paper, we propose CoAnchor, an anchor-centric spatio-temporal alignment framework for asynchronous collaborative perception.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21055v1)
- [arXiv](https://arxiv.org/abs/2608.21055v1)

---

<a id='2608.21031v1'></a>
## [PhysCaP: Grounding Code-as-Policy Agent with Physics-Informed Exploration](https://arxiv.org/abs/2608.21031v1)

**Authors:** Chen-Yu Lin, Jing-Wen Chen, Hsueh-En Chang, Hung-An Chen, Sheng-Hsun Chang, Chi-Pin Huang, Fu-En Yang, Min-Hung Chen, Yi-Ting Chen, Yu-Chiang Frank Wang, Shao-Hua Sun

**Published:** 2026-08-21

**Categories:** cs.RO

**Abstract:**

We present PhysCaP, a Physics-Informed Code-as-Policy agent for active perception in robotic manipulation. While vision-language-action policies excel at imitating demonstrations, they rely on passive observation and fail to infer latent physical properties critical for manipulation. PhysCaP augments code-as-policy frameworks with a physics-informed exploration layer that enables explicit information-seeking through interaction. It introduces training-free physical property extraction modules that estimate object mass and stiffness from robot proprioception without additional sensors. To balance exploration costs and the efficiency of information obtained, PhysCaP employs a dual-agent design: a Planner that decides when to explore and when to stop, and a Prioritizer that filters implausible interactions and ranks the remainder using a heuristic priority score, enabling efficient, targeted exploration. We evaluate PhysCaP on real-world tabletop manipulation tasks (searching for hidden objects, detecting empty cans, and finding ripe avocados) and a simulated task in LIBERO. The results show that existing passive and naive interactive baselines either fail when physical properties are hidden or over-explore, whereas PhysCaP achieves comparable performance with fewer interactions and reduced execution time. Ablation studies further validate the effectiveness of the proposed physical property extraction modules. Project page: https://physcap.github.io

**Analysis:**

## 1. 论文主要贡献概述

PhysCaP 提出一种面向机器人主动感知的物理信息增强型 Code-as-Policy 智能体，使机器人能够通过有目的的交互主动推断物体的质量、刚度等潜在物理属性，而不仅依赖被动视觉观察或示范模仿。该方法通过“规划器（Planner）+ 优先级排序器（Prioritizer）”的双智能体结构，在决定是否探索、何时停止以及选择何种交互之间进行权衡，从而减少不必要的探索次数和执行时间。

## 2. 关键创新与方法路线

### 2.1 将物理属性推断引入 Code-as-Policy

传统视觉语言动作策略通常根据图像和语言指令直接生成动作，擅长模仿已见过的行为，但难以处理“视觉外观相似、物理属性不同”的物体。例如：

- 空罐与装满液体的罐子外观可能相近，但重量不同；
- 未成熟和成熟的牛油果外观差异不明显，但硬度不同；
- 被遮挡的物体无法仅通过视觉确定其存在状态。

PhysCaP 通过机器人交互过程中的本体感知信息，例如运动响应、关节状态、力矩或执行误差等，训练无关地提取物体的质量和刚度等属性。其核心思想是把“交互反馈”作为一种物理感知信号，从而弥补纯视觉观察的不足。

### 2.2 物理信息驱动的主动探索

该方法不是让机器人盲目尝试所有可能动作，而是将探索视为一个信息获取问题：

1. 判断当前视觉和任务信息是否足够；
2. 如果不足，选择可能最有效的交互；
3. 根据交互结果更新对物体物理属性或状态的判断；
4. 在信息足够时主动停止探索。

这体现了从“被动感知”到“主动感知”的转变，也是其与普通 VLA 策略或简单交互式策略的主要区别。

### 2.3 Planner 与 Prioritizer 的双代理设计

- **Planner**：决定是否需要探索、探索何时结束，以及当前任务是否已经获得足够信息。
- **Prioritizer**：过滤明显不合理或不可行的交互，并根据启发式优先级对候选动作进行排序。

这种设计试图在两个目标之间取得平衡：

- 获取足够多的物理信息；
- 降低交互成本、动作数量和执行时间。

从方法论上看，它将高层任务规划、物理属性推断和低层动作筛选进行了模块化组合，有利于在代码生成式策略框架中实现可解释的主动探索。

## 3. 对领域的潜在影响

### 3.1 推动机器人从“看起来像”走向“物理上理解”

计算机视觉和机器人学习系统通常根据外观判断物体类别、位置和状态，但实际操作还需要理解：

- 物体是否为空；
- 物体是否可压缩或易变形；
- 物体是否足够重；
- 物体是否适合抓取、推动或搬运。

PhysCaP 的价值在于说明视觉感知不必局限于图像输入，机器人本体感知和交互反馈也可以被用于构建更接近物理理解的感知系统。

### 3.2 降低对大规模示范数据的依赖

摘要强调物理属性提取模块是 training-free 的。如果这一设计在更广泛任务上成立，则机器人可以通过少量通用规则或模型，在部署时进行在线探索，而无需为每种新物体、新物理属性收集大量示范数据。这对于长尾物体和开放世界机器人操作尤其有意义。

### 3.3 提高主动探索的效率和可解释性

许多交互式机器人策略存在两类问题：

- 探索不足，无法区分关键的隐藏状态；
- 探索过多，导致时间、能耗和物体损伤增加。

PhysCaP 的显式规划和优先级排序机制，为“为什么要探索”“探索哪个动作”“何时停止”提供了相对清晰的决策结构。这可能有助于构建更可解释、更易调试的机器人智能体。

### 3.4 对视觉语言动作模型的补充

该工作并非完全替代视觉语言动作模型，而是为其增加一个物理信息获取层。其潜在方向包括：

- 将视觉语言模型用于生成候选操作；
- 使用物理感知模块筛选和重排序候选动作；
- 将本体感知反馈转化为高层策略可理解的状态描述；
- 让代码策略根据在线交互结果动态修改行为。

因此，它对“视觉—语言—动作—物理反馈”一体化智能体具有一定启发意义。

## 4. 可能受益的相关领域与应用

### 4.1 家庭服务机器人

家庭环境中经常存在视觉上难以区分、但物理属性不同的物体，例如：

- 判断容器是否装有液体；
- 区分空包装和有内容物的包装；
- 判断物体是否易碎、柔软或过重；
- 在杂乱环境中搜索被遮挡物品。

### 4.2 食品处理与农业机器人

摘要中的牛油果成熟度检测体现了其在农业和食品操作中的潜力。类似应用包括：

- 通过柔性或轻微按压估计水果成熟度；
- 检测果蔬是否腐败或内部空心；
- 根据硬度选择抓取力度；
- 对不同农产品进行分级和分类。

### 4.3 仓储、物流与包装检测

机器人可以通过抓取、抬升或轻推来推断：

- 包装箱是否为空；
- 容器内是否存在物品；
- 物品是否超重；
- 包装是否发生结构损坏或变形。

这类信息往往无法仅凭 RGB 图像可靠获得。

### 4.4 人机协作与安全操作

在与人共享空间的环境中，机器人需要判断物体的重量、刚度和可变形性，以决定：

- 是否适合直接抓取；
- 应采用多大力度；
- 是否需要双手搬运；
- 是否可能损坏物体或伤害人。

### 4.5 柔性物体与接触丰富的操作

衣物、海绵、软包装、食物和橡胶制品等物体具有明显的形变和接触动力学特征。视觉外观通常不足以确定其可操作性，基于本体感知的主动探索可能对此类任务更有帮助。

### 4.6 具身智能和世界模型

PhysCaP 还可以被视为具身智能中的一种轻量级物理推理机制：机器人不必先学习完整的世界模型，而是通过少量有针对性的交互获得任务相关的物理信息。这对开放环境中的在线适应和快速决策具有潜在价值。

## 5. 从摘要可以推断出的局限性

由于目前只有摘要，以下问题尚不能得到确认，但值得关注。

### 5.1 物理属性估计可能具有较强的任务和硬件依赖

仅依靠本体感知估计质量和刚度，在理论上通常是欠定问题。相同的关节响应可能由多种因素造成，例如：

- 摩擦和传动误差；
- 抓取位置不同；
- 接触几何形状不同；
- 物体表面滑移；
- 机器人控制器和末端执行器差异。

因此，训练无关的提取模块可能依赖特定机器人平台、动作模板或校准条件，在不同硬件和物体类别上的泛化能力仍需验证。

### 5.2 启发式优先级可能限制通用性

Prioritizer 使用启发式优先级对交互进行排序，优点是简单高效，但也可能存在：

- 对复杂任务或新物体缺乏适应性；
- 依赖人工设计的候选动作集合；
- 难以处理多个物理属性相互耦合的情况；
- 无法保证选择的信息增益接近最优。

未来可能需要将其扩展为基于不确定性、信息增益或贝叶斯决策的探索策略。

### 5.3 探索成本和风险可能被低估

摘要主要关注交互次数和执行时间，但真实机器人探索还可能带来：

- 物体损伤；
- 物品污染或倾倒；
- 夹具磨损；
- 对周围环境的碰撞风险；
- 对人类或其他机器人造成安全隐患。

如果 Planner 只判断“信息是否足够”，而没有显式建模风险和失败代价，那么在高价值或易损物体上仍可能不够安全。

### 5.4 任务覆盖范围有限

摘要中的真实任务包括隐藏物体搜索、空罐检测和牛油果成熟度判断，均属于较明确的桌面操作场景。尚不清楚该方法能否推广到：

- 多物体密集堆叠；
- 动态环境；
- 复杂遮挡和反射；
- 非桌面操作；
- 高自由度柔性物体；
- 需要长期多步规划的任务。

LIBERO 仿真结果能够支持一定的可重复性，但不一定能充分反映复杂真实环境中的物理不确定性。

### 5.5 对视觉信息的依赖仍然存在

虽然 PhysCaP 强调物理属性和本体感知，但它并不是纯粹的触觉或动力学系统。候选交互、物体身份和任务目标仍可能依赖视觉语言模型或视觉检测模块。因此，如果视觉定位、分割或物体识别出错，物理探索模块也可能选择错误的对象或动作。

### 5.6 Code-as-Policy 的可靠性和安全性问题

代码生成式策略具有灵活性，但部署到真实机器人时还需要考虑：

- 生成代码是否始终满足动作约束；
- 是否可能调用不可执行或危险的动作；
- 如何验证代码的碰撞安全性；
- 规划器和优先级模块之间是否会产生不一致；
- 失败后如何恢复。

摘要未说明是否使用了形式化动作约束、运行时安全检查或故障恢复机制。

## 总体评价

PhysCaP 的重要性在于，它将主动感知、机器人本体感知、物理属性推断与 Code-as-Policy 结合起来，试图解决纯视觉或被动模仿策略无法处理的“外观相似但物理状态不同”问题。对于计算机视觉领域而言，其趣味性并不只在于新增了一个机器人控制模块，更在于它体现了一种扩展视觉理解的方向：通过主动交互获得视觉本身无法直接提供的物理信息，从而使视觉智能体具备更强的可操作性和具身理解能力。

**Key Findings:**

- We present PhysCaP, a Physics-Informed Code-as-Policy agent for active perception in robotic manipulation.
- The results show that existing passive and naive interactive baselines either fail when physical properties are hidden or over-explore, whereas PhysCaP achieves comparable performance with fewer interactions and reduced execution time.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.21031v1)
- [arXiv](https://arxiv.org/abs/2608.21031v1)

---

<a id='2608.20948v1'></a>
## [Neural-Primitive: An Efficient End-to-end Local Planner with Primitive-based Imitation Learning for Autonomous Flight](https://arxiv.org/abs/2608.20948v1)

**Authors:** Zhitao Liu, Guangtong Xu, Zihan Wang, Jialiang Hou, Chao Xu, Fei Gao

**Published:** 2026-08-21

**Categories:** cs.RO, cs.AI

**Abstract:**

Autonomous flight in unknown cluttered environments is hindered by the computation-quality-memory trilemma of onboard trajectory generation. In this paper, we propose an efficient end-to-end local planner via imitation learning. A lightweight offline-primitive-based dataset collection framework is designed to produce safe and high-quality trajectory primitives in non-convex environments. A compact neural network directly maps sensory inputs to polynomial coefficients that inherently encode higher-order dynamical information. The learned policy generates smooth, empirically collision-free and dynamically feasible trajectories in real time without back-end solving. It achieves ultra-fast computation (below 1ms on a standard desktop and average 3.68ms during onboard flight), while maintaining low onboard memory requirements (less than 1.5MiB). Extensive simulation benchmarks demonstrate superiority in both planning latency and target-reaching progress quality. Zero-shot deployment in real-world experiments further validates the robust sim-to-real transfer capability of the proposed method.

**Analysis:**

# 1. 摘要翻译

未知杂乱环境中的自主飞行受到机载轨迹生成在计算量、轨迹质量与内存占用之间的三难困境限制。本文提出一种基于模仿学习的高效端到端局部规划器 Neural-Primitive。作者设计了轻量级、基于离线运动基元的数据采集框架，在非凸环境中生成安全且高质量的轨迹基元；随后利用紧凑神经网络，将传感器输入直接映射为能够内在编码高阶动力学信息的多项式系数。该策略无需后端求解，即可实时生成平滑、经验上无碰撞且动力学可行的轨迹。其计算时间在普通桌面端低于1 ms，机载飞行平均为3.68 ms，模型内存低于1.5 MiB。大量仿真表明，该方法在规划延迟和目标到达质量方面优于多种基线；真实环境中的零样本部署进一步验证了其较强的仿真到现实迁移能力。

# 2. 方法动机分析

**驱动力与痛点：**传统规划器通常采用“建图—搜索—优化”的层级流程，计算延迟会随障碍密度增加；联合优化避障、动力学和目标到达又容易陷入局部最优。离线运动基元虽能预先生成高质量轨迹，但有限库无法覆盖所有初始速度、加速度，在线拼接会产生高阶状态不连续，并带来巨大存储开销。现有学习方法也常需后端优化、边界值求解或轨迹投影，因而并非真正端到端。

**核心假设：**不必让网络直接学习复杂的“避障优化目标”，而可以先用“生成可行基元—剔除碰撞基元—选择最接近目标的基元”构造专家数据，再让网络直接回归完整轨迹参数；只要显式输入当前速度和加速度，便可保证轨迹连续性。

# 3. 方法设计详解

## 3.1 数据集构造

1. 在仿真中随机生成由圆柱、方柱、环等组成的地图，并随机初始化无人机状态、目标位置和目标方向。  
2. 每次重规划时，以当前状态为起点，仅离散采样终端状态，在线生成候选库。对每个终端状态，通过二次规划求解固定时长的五次最小 jerk 多项式轨迹，时长为2 s，并约束位置、速度、加速度及动力学上限。  
3. 在速度对齐坐标系中，将轨迹周围的膨胀占据区域离散为体素，并为每条基元保存其占据体素集合。这样后续碰撞判断只需哈希查表，而不需要反复查询 ESDF。  
4. 将感知点云转换到该坐标系；若任一点落入某基元的占据集合，则标记该基元为碰撞，否则保留。  
5. 在安全基元中按  
   \[
   C=w_tC_{\text{target}}+w_lC_{\text{length}}
   \]
   选择最优者：目标项鼓励前进并避免越过目标，长度项抑制绕行和曲折。轨迹执行一小段后继续重规划；只有最终成功到达目标的整段数据才加入训练集，从而减少失败示范并强化接近目标时的行为。

## 3.2 网络与输出

网络输入为：速度大小1维、加速度3维、目标方向3维和固定数量点云。点云先重采样到666点，经共享 MLP `[128,128,64]` 和最大池化提取全局环境特征，再经 `[64,32]` 压缩。该32维特征与7维状态拼接，经 `[256,512,256]` 融合，最后由 `[128,128,9]` 预测多项式高阶系数。

网络并不预测完整系数：五次轨迹低阶项由当前速度、加速度直接确定，以保证起点状态连续；网络只学习剩余的9个高阶系数，用于表达避障与目标趋近行为。训练采用专家系数的 MSE 损失、AdamW 和余弦退火。

## 3.3 sim-to-real增强

训练时随机降低点云密度，低于666点则用远距离“虚拟点”补齐；同时向状态和点云坐标注入高斯噪声。该设计不仅模拟测量误差，还覆盖真实 LiDAR 中更关键的密度变化。

# 4. 方法对比与创新

本质区别是：网络直接输出可执行的连续多项式，而非路点、终点或优化权重；同时把避障与目标选择解耦到专家数据生成阶段，避免在线联合优化。主要创新包括：**基于策略嵌入的数据采集、显式高阶连续性设计、占据关系加速碰撞标注、密度感知的点云增强**。它尤其适合静态、未知、密集障碍环境中的高速局部导航；不适合需要全局拓扑推理或强动态障碍预测的任务。

# 5. 实验分析

作者通过消融、加速度连续性分析、稀疏/密集仿真对比、未见地图测试和真实飞行验证方法。代表性结果是：仿真密集环境成功率约0.97，规划时间约0.68 ms；真实森林中实现6.10 m/s最高速度、60 m无碰撞飞行，机载平均规划时间3.68 ms。主要优势是低延迟、低存储、轨迹平滑且目标路径较直。局限是依赖仿真专家分布，面对动态障碍、U形死胡同和大尺度局部陷阱能力有限；“经验无碰撞”并非形式化安全保证。

# 6. 实用指南

论文提供项目页面和视频，但文中未明确说明完整代码是否开源。复现关键是：实现五次最小 jerk QP、速度坐标系变换、轨迹占据体素哈希、成功闭环数据筛选，以及666点输入和噪声/密度增强。重要设置包括2 s轨迹时长、20个动力学检查点、输入噪声约状态3%、点云4%、最多训练600轮并早停。迁移到地面机器人或机械臂时，可将输出替换为相应的样条/控制参数，并重新定义动力学约束、碰撞体素和专家代价。

# 7. 总结

**核心思想：**学习直接生成连续可执行轨迹。

**速记版 Pipeline：**
1. 仿真中生成大量可行轨迹候选。  
2. 用点云快速删掉会撞障碍的候选。  
3. 选择最短且最接近目标的安全轨迹作示范。  
4. 网络根据点云、速度、加速度和目标方向直接预测轨迹系数。  
5. 在线滚动执行并周期性重规划。

**Key Findings:**

- In this paper, we propose an efficient end-to-end local planner via imitation learning.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20948v1)
- [arXiv](https://arxiv.org/abs/2608.20948v1)

---

<a id='2608.20946v1'></a>
## [Fast Coordinated Bimanual Motion Planning With Hard Constraints](https://arxiv.org/abs/2608.20946v1)

**Authors:** Borna Paro, Luka Petrović, Ivan Marković

**Published:** 2026-08-21

**Categories:** cs.RO

**Abstract:**

Bimanual manipulation enables complex tasks but introduces added complexity from the high number of degrees of freedom involved. When handling rigid objects, the relative transformation between the two end effectors must remain fixed throughout the motion, manifesting as a nonlinear equality constraint that confines the feasible configuration space to a measure-zero manifold and challenges conventional motion planners. We propose a fast bimanual motion planning pipeline that enforces this hard transformation constraint continuously along the entire path, using a leader-follower parameterization: the leader's configuration is treated as a free variable, while the follower's is determined via inverse kinematics to satisfy the constraint. We extensively evaluate the method in simulation across diverse environments, constraints and bimanual platforms, achieving 19.4x faster planning than prior work while guaranteeing continuous constraint satisfaction. Real-world experiments on a bimanual Kinova Gen3 setup, involving tray transport and elongated-object manipulation, validate direct transfer of planned trajectories to physical hardware.

**Analysis:**

# 1. 摘要翻译

双臂操作能够完成复杂任务，但也因自由度较高而增加了规划难度。当双臂共同抓取刚体时，两个末端执行器之间的相对变换必须在整个运动过程中保持不变。这一要求构成非线性等式约束，使可行构型空间退化为环境空间中的测度为零流形，导致传统运动规划器难以处理。本文提出一种快速双臂运动规划流程，通过“主臂—从臂”参数化连续施加硬相对位姿约束：主臂构型作为自由变量，从臂构型通过逆运动学求解得到。仿真实验覆盖多种环境、约束和双臂平台，规划速度较已有方法提高19.4倍，并保证整条路径连续满足约束。在双Kinova Gen3真实平台上进行的托盘运输和长物体操作实验表明，规划轨迹可以直接迁移到实体机器人执行。

# 2. 方法动机分析

**驱动力：**双臂共同搬运托盘、长杆或大型物体时，末端之间的相对位姿不能仅在起点和终点满足，而必须沿全轨迹保持，否则会造成物体滑落、倾斜或变形。

**现有痛点：**

1. 直接在双臂联合空间随机采样时，满足6维刚性约束的状态几乎不会被采到。
2. 轨迹优化面对非凸、低维流形，容易陷入不可行区域。
3. 投影法需要反复计算约束雅可比并迭代回流形，代价高且对奇异位形敏感。
4. 先规划主臂、再对从臂逐点IK的后处理方法，会产生IK分支跳变；同时从臂可能与障碍物或主臂碰撞。

**核心假设：**只要将一侧机械臂自由采样，另一侧始终通过带连续种子初始化的IK闭合约束，就可以把搜索空间直接限制在约束流形上。

# 3. 方法设计详解

## 3.1 约束建模与流形参数化

设双臂构型为 \(q=(q_L,q_F)\)，末端正运动学为 \(FK_L,FK_F\)，期望相对变换为 \(T\)。硬约束为：

\[
FK_F(q_F)=FK_L(q_L)T.
\]

因此双臂不是在完整的 \(2n\) 维空间中独立运动，而是在维数 \(2n-6\) 的约束流形上运动。论文的关键改动不是“采样后投影”，而是直接构造有效状态：

1. 给定主臂构型 \(q_L\)；
2. 计算从臂目标末端位姿  
   \[
   X_F^*=FK_L(q_L)T;
   \]
3. 以历史从臂构型作为种子，通过IK求解 \(q_F=IK_F(X_F^*,q_F^{seed})\)；
4. 得到有效双臂状态  
   \[
   \sigma(q_L)=(q_L,q_F).
   \]

这相当于用主臂关节变量作为约束流形的坐标。IK种子决定从臂的分支；使用前一状态作为种子，可减少相邻状态间的关节跳变，并在冗余机械臂中维持较稳定的冗余解。

## 3.2 采样、距离度量与有效性检查

论文在OMPL中实现自定义采样器。主臂可以在关节空间采样，也可以先采样末端笛卡尔位姿，再通过IK得到主臂构型。对于超出预计算双臂可达工作空间包围盒的样本，直接拒绝，避免无意义的从臂IK调用。

最近邻采用环绕关节差值：

\[
\Delta q_i=\operatorname{atan2}(\sin\delta_i,\cos\delta_i),
\]

并对从臂关节设置更高权重。其直觉是：优先连接从臂运动较小的状态，从而降低IK分支跳跃和奇异性风险。靠近从臂奇异位形的状态也会被拒绝。

## 3.3 约束感知插值

普通关节线性插值会破坏末端相对位姿，因此边验证也必须在流形上完成。给定两个有效状态 \(q_a,q_b\)：

1. 根据加权构型距离确定插值步数 \(K\)，距离越大，步数越多；
2. 仅对主臂进行线性预测；
3. 每一步都重新计算从臂目标位姿；
4. 用上一步从臂状态热启动IK；
5. 对每个中间状态进行碰撞、关节限制和奇异性检查。

所有中间IK解会被缓存并插入规划树。这样，边验证不仅判断可行性，还顺便提高树在流形上的密度，减少后续搜索成本。

## 3.4 RRT-Connect与后处理

规划器采用RRT-Connect，但其采样、距离和边验证均替换为上述约束版本。找到路径后依次执行：

- **路径简化：**尝试用直接的约束连接替换冗余子路径；
- **不连续修正：**交替将左右臂设为主臂，利用另一种参数化选择更接近前一状态的IK解；仍不连续时插入更多流形状态；
- **轨迹生成：**使用TOPP-RA进行时间参数化，满足关节速度和加速度约束；
- **最终验证：**沿重定时轨迹重新检查相对位姿残差，失败则拒绝轨迹。

# 4. 方法对比与适用性

其本质区别是：主流投影方法在完整空间中“采样后纠正”，本文则通过IK“构造即满足约束”；相比先规划单臂再后处理，本文在搜索阶段就同时考虑双臂碰撞和从臂可行性。创新集中在三点：IK驱动的流形图参数化、约束感知插值，以及将插值IK结果复用于树扩展。

适合刚性双臂协同搬运、闭链操作、托盘运输和长物体抓取，也适用于没有解析IK的机械臂。它不适合相对位姿允许大范围变化、接触关系频繁切换或需要力控制主导的任务。

# 5. 实验分析

作者在KUKA iiwa、Kinova Gen3和UR5上测试，并覆盖无障碍、稀疏障碍和密集障碍环境。相较IK-BiRRT，在线规划平均快19.4倍；在真实Kinova实验中，托盘中的水和直立物体保持稳定，长物体也成功完成双臂运输。

**优势：**硬约束由构造保证、无需离线预计算、适配不同机械臂、能直接检查双臂碰撞。  
**局限：**数值IK是主要瓶颈；IK分支仍可能跳变；大幅不连续只能在后处理中修正，可能降低成功率。

# 6. 实用指南

论文当前未公开代码，作者表示审稿完成后开源。复现需实现双臂正运动学、带种子IK、碰撞检测、约束采样器、RRT-Connect、流形插值和TOPP-RA。关键实现点是：保存并复用上一从臂解、提高从臂距离权重、按距离自适应插值、提前剔除奇异位形，并对最终轨迹逐点检查约束。该框架可迁移到其他双臂或多臂闭链任务，只需替换正运动学、IK和碰撞模型。

# 7. 总结

**核心思想：**用IK构造满足约束的双臂路径。

**速记版Pipeline：**

1. 生成多个起点和终点双臂构型。  
2. 随机改变主臂，并用IK同步求从臂。  
3. 沿每条连接逐步重算从臂，始终保持刚性关系。  
4. 用RRT-Connect搜索并简化路径，修复关节跳变。  
5. 进行时间参数化和全轨迹约束复核。

**Key Findings:**

- We propose a fast bimanual motion planning pipeline that enforces this hard transformation constraint continuously along the entire path, using a leader-follower parameterization: the leader's configuration is treated as a free variable, while the follower's is determined via inverse kinematics to satisfy the constraint.

**Links:**

- [PDF](https://arxiv.org/pdf/2608.20946v1)
- [arXiv](https://arxiv.org/abs/2608.20946v1)

---

