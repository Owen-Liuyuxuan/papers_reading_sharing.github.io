time: 20261020

# Arxiv Computer Vision Papers - 2026-10-20

## Executive Summary

## 当日研究脉络

本期十篇论文共同关心机器人如何从更多经验中获得稳定、可执行的行为。数据来源从机器人车队实操、互联网第一视角视频扩展到飞控在环水下仿真；方法侧则在闭环控制、表示诊断和多模态融合上补足单看最终成功率难以发现的问题。另有两项工作从形态与概率建模切入，分别扩大人形机器人全身操作能力、提高多峰动作策略在小样本与边缘推理条件下的可靠性。

规模化学习方向值得优先关注。**Learning from Fleet Experience** 用失败轨迹末端检索车队成功片段并提高其训练权重，在五项真实操作任务中报告93.3%–99.2%成功率；手机包装任务从79.2%提升至98.3%。**ScaleNav** 将无动作的人类视频先验与有动作的模拟示范分阶段对齐，四足机器人60次室内试验达到83.3%成功率。**AQUAACT** 则把水下遥操作的摇杆、图像和惯性数据配对，并显示模拟示范可降低真实微调需求，但现有策略在长时闭环任务中仍频繁碰撞。三者分别展示车队再利用、视频迁移和同控制链仿真三种扩展经验的路径。

可靠控制也需要表示和评测层面的诊断。**UBD-Bench** 发现多模态联合训练可能提升整体任务表现，却削弱相机单模态分支，且融合前BEV表征的落差尤为明显；在该处蒸馏能同时改善分支探针和完整模型。**SWAP** 将人形全身候选动作生成与测试时可转向选择分开，在仿真五类避障任务上把成功率从直接端到端扩散的17.4%提升到88.3%，同时显著降低碰撞。**VQ-WM** 把离散世界模型规划的邻接关系按控制器实际到达目标所需时间定义，使规划状态更贴合可执行性。这些结果提示，闭环成功、单模态表征质量、碰撞安全和控制可达性应共同评估。

感知与动作表达方面，**RAHF-Net** 根据LiDAR距离让近场使用几何细节、远场更多依赖RGB语义，在KITTI-360取得65.76 mIoU；**SpaRV** 以LiDAR约束稀疏辐射体素地图，并用层级裁剪支持大范围视觉渲染；**Energy-Score Stabilized Attentive Mixture-Density Policy** 用Energy Score/CRPS稳定多模态动作分布，在有限示教下避免确定性回归把绕障路径平均成不可行动作。**Humanoid SRL** 则用额外肢体同时承担操作、支撑与移动，展示形态扩展对可达区域和任务分工的作用。

若时间有限，建议先读 Fleet Experience、ScaleNav 和 SWAP，分别把握车队经验复用、互联网视频迁移和人形闭环避障；关注多模态感知评估者可读 UBD-Bench 与 RAHF-Net，研究长时控制、生成式规划或地图表示者可选择 VQ-WM、Energy-Score Policy 与 SpaRV。总体趋势是把数据规模、动作可执行性和运行中的安全可靠性放在同一闭环问题中检验；当前多数证据仍限定于特定平台、任务分布或模拟条件，跨场景及长期部署仍需验证。

---

## Table of Contents

1. [Learning from Fleet Experience: A Collective Learning Paradigm for Robots](#2610.25736v1)
2. [UBD-Bench: Benchmarking and Analyzing Unimodal Branch Degradation in Multimodal Perception for Autonomous Driving](#2610.25448v1)
3. [Vector-Quantized World Model: Latent Feedback Motion Planning for Robot Manipulation](#2610.25600v1)
4. [SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance](#2610.25711v1)
5. [Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation](#2610.25628v1)
6. [RAHF-Net: Range-Aware Heterogeneous Fusion for LiDAR-Assisted RGB Semantic Segmentation](#2610.25400v1)
7. [Energy-Score Stabilized Attentive Mixture-Density Policy: A Lightweight Approach to Multimodal Robot Motion](#2610.25401v1)
8. [SpaRV: Sparse Radiance Voxels for Scalable Geometrically Consistent LiDAR-Visual Mapping and Rendering](#2610.25665v1)
9. [AQUAACT: A Joystick-Level Underwater Teleoperation Dataset and Closed-Loop Benchmark](#2610.25534v1)
10. [ScaleNav: Scalable Internet Video Pretraining for Language-Conditioned Local Navigation](#2610.25690v1)

---

## Papers

<a id='2610.25736v1'></a>
## [Learning from Fleet Experience: A Collective Learning Paradigm for Robots](https://arxiv.org/abs/2610.25736v1)

**Authors:** Chao Yu, Shu'ang Yu, Hao Lin, Hongzhi Zang, Yuxiang Yan, Chenxi Jiang, Zhen Guo, Chuankang Li, Yihao Chen, Xin Xu, Xu Fu, Jiakai Zhou, Liangzhi Shi, Hongyi Ren, Yiwei Liu, Jing Wu, Yuanqing Wang, Chen Wu, Bo Dai, Chunyu Yang, Hongliang Huang, Shuoyu Yue, Weiqi Yin, Yinuo Chen, Ruihua Chen, Chunyang Zhu, Han Xiao, Yanghao Li, Ruipu Wu, Zhikai Cai, Xiu Chen, Xu Tan, Yuchen Yan, Jialin Zhang, Wen Zeng, Yiming Zhou, Zixiao Huang, Guanghui Ren, Quanlu Zhang, Boxun Li, Maoqing Yao, Yu Wang

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

Experience generated during deployment offers an opportunity for robots to improve policy reliability through learning from their own executions. However, learning through human corrections requires teleoperators to monitor execution and intervene when policies fail, a demand that is difficult to meet as robot fleets grow. We introduce collective experience learning (CEL), a learning paradigm in which deployed robots share their execution experience to improve a common policy. We realize CEL through failure-guided experience retrieval fine-tuning (FERT), which uses failure contexts from robot executions to retrieve relevant trajectory segments from the fleet's shared successful experience and uses their actions as supervision for policy updates, without collecting additional expert demonstrations or corrective actions. A fault-tolerant asynchronous training system (FATS) coordinates experience collection, policy optimization and policy distribution despite local interruptions within the robot fleet. Across five real-world manipulation tasks, the resulting policies achieve success rates of 93.3-99.2%, evaluated over 120 trials per task on robots excluded from training data collection. With collection time per robot held fixed, larger fleets provide more experience and FERT generally achieves higher success rates across the evaluated sizes. On multi-stage phone packaging, learning with 96 robots increases success from 79.2% to 98.3%. These results support collective experience as a practical basis for scalable robot learning and a path towards improving policy reliability during deployment without collecting additional expert corrective actions.

### 论文解读
#### 摘要翻译
机器人部署会自然产生新经验，但常见的在线改进依赖操作员持续监看并提供纠正动作，难以随车队扩大。本文提出集体经验学习（CEL）：多台机器人共享执行数据，共同改进任务策略。算法 FERT 用失败轨迹的上下文，从车队累积的成功轨迹中检索相似片段，并提高相关成功动作的训练权重。配套系统 FATS 让采集、推理和训练异步运行，并支持单机故障后的局部恢复。作者在五项真实机械臂任务上报告，训练后的策略在未参与采集的同型号机器人上达到93.3%–99.2%成功率。
#### 方法动机分析
物体、接触和设备差异会使策略出错，长任务的早期误差还会传到后续步骤。若每次失败都要专家接管，人工动作监督会成为扩展瓶颈。论文假设失败末端附近的视觉与关节状态能在其他机器人成功轨迹中找到相近阶段，成功动作比失败动作更适合作为监督。人仍需标记成败并复位环境，但不必为 FERT 持续示范动作。
#### 方法设计详解
方法从人工示范和多机器人 HG-DAgger 初始化。执行时，成功轨迹入滚动经验库，失败轨迹只作查询。冻结视觉编码器将每个相机的16×16 token 池化为4×4，再与归一化关节状态拼接。查询取失败终点前最多90帧，提前10帧结束，避免混入失败后的状态。算法对查询和成功轨迹逐帧比较，以受约束动态时间规整寻找允许轻微速度变化的子序列匹配；选择成本最低的至多10段，并将每段前后各扩展30帧。动作块按与检索区间的重合度提高采样优先级，历史权重按0.88衰减；训练用基础策略的条件流匹配损失更新，最高学习率2.5×10⁻⁵、全局批量1024，每次训练约7–8小时。FATS 让边缘机器人只等待自己的动作；云端拼接观测、执行记录和结局，并用 worker 身份过滤迟到消息，故障时只恢复对应位置。
#### 方法对比分析
HG-DAgger 依赖专家接管，RLT 使用在线强化学习及纠正动作，RECAP 通过价值估计筛选动作；相比这些基线，FERT 用失败状态作检索键、成功动作作监督，不另训价值网络。相比均匀回放，它把更新集中到当前策略易错环节。其前提是结局标注可靠，且跨执行的相似状态对应可复用行为。
#### 实验分析（精简版）
实验使用 AgiBot G2 双臂机器人，覆盖插花、折衣、充电、凳子装配和手机包装；每项最终策略在两台未参与采集的同型号机器人上测试120次。五项任务平均成功率从初始化的81.0%升至97.3%；手机包装中去掉失败引导检索后，成功率从86.7%降至64.2%。96台车队收集9,834条轨迹，记录故障损失为1.3%机器人小时。证据限于五项任务和单一平台，仍需验证跨平台及长期运行。
#### 实用指南
复现需先完成示教微调与人工门控聚合，再记录成功轨迹、失败上下文和二元结局。关键设定包括查询90帧、端点偏移10帧、检索扩展30帧、最多10段、优先级衰减0.88、每500步同步策略。论文提供项目网站，训练依赖多机器人及云端基础设施。检索思路可启发移动机器人经验复用；用于自动驾驶需重新定义状态、失败和可复用片段，论文未验证这一迁移。
#### 总结
核心思想：失败定位，成功复用
1. 用示范建立初始策略，车队并行执行。
2. 用失败上下文检索相似成功片段。
3. 提高相关动作权重，更新策略并返回车队。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25736v1)
- [arXiv](https://arxiv.org/abs/2610.25736v1)

---

<a id='2610.25448v1'></a>
## [UBD-Bench: Benchmarking and Analyzing Unimodal Branch Degradation in Multimodal Perception for Autonomous Driving](https://arxiv.org/abs/2610.25448v1)

**Authors:** Ang Liu, Xiang Gu, Ruikai Li, Qiyue Wang, Yizhe Fan, Hao Shan, Ziyang Yan, Han Jiang, Zhenning Li, Xiaoshuai Hao, Yilong Ren, Zhiyong Cui, Haiyang Yu

**Published:** 2026-10-18

**Categories:** cs.CV

**Abstract:**

Multimodal perception improves autonomous-driving performance by combining complementary sensor information. However, stronger multimodal predictions do not necessarily imply stronger representations in each modality-specific branch. Such branch-level weaknesses can remain hidden in conventional output-level evaluation. We introduce UBD-Bench, a systematic benchmark for studying unimodal branch degradation (UBD) in multimodal perception. UBD-Bench uses a Paired Frozen-Branch Probe to compare the task-relevant representation quality of the same modality after independent unimodal training and multimodal joint training under matched decoding conditions. Across four autonomous-driving perception tasks and 10 models, we find that Camera branches consistently degrade after multimodal joint training, whereas LiDAR branches exhibit weaker and more model-dependent changes, revealing a pronounced modality asymmetry. Hierarchical probing further shows that Camera-branch degradation extends from the visual backbone through the Camera-to-BEV pathway and is most severe at the pre-fusion Camera BEV representation. Guided by this diagnosis, we introduce unimodal-to-multimodal distillation at the degraded representation. This intervention consistently improves both Camera-branch Probe performance and full-model performance across all four tasks. These results show that multimodal gains can coexist with degraded unimodal representations, highlighting branch-level representation quality as an important dimension for evaluating and improving multimodal perception.

### 论文解读
#### 摘要翻译
多模态感知常因融合相机和激光雷达而提升整体任务表现，但融合模型的最终预测无法说明各模态分支本身是否都学得更好。本文提出 UBD-Bench，用配对冻结分支探针比较独立单模态训练与多模态联合训练得到的同类表示：固定表示网络，只训练结构、初始化相同的任务解码器，并保持训练集划分、优化器设定和迭代预算相同。四类自动驾驶感知任务、十种模型的结果显示，相机分支持续退化，激光雷达分支的变化则依任务和模型而异。作者进一步用单模态到多模态的特征蒸馏缓解这一现象。
#### 方法动机分析
相机提供密集语义与外观信息，激光雷达提供几何结构，两者互补并不保证联合优化时每个分支都被充分利用。常规端到端指标把表示质量、融合模块和解码器混在一起，因而看不出退化发生在哪个分支或处理阶段。作者假设，在固定下游解码条件后，任务指标差异能反映冻结分支对目标任务的支持程度。
#### 方法设计详解
首先构造配对 Probe：分别载入独立相机/雷达模型及融合模型对应支路的权重，冻结从传感输入到探测位置的所有模块，连接相同结构且同初始化的任务 decoder，只训练 decoder，并保持数据划分、优化设定和预算一致。用联合训练支路分数减去独立单模态分数，负差定义为单模态分支退化。基准覆盖3D检测、在线矢量地图、BEV分割和占用预测，采用各任务原有指标。层级 Probe 再把观测位置沿相机到 BEV 路径向前移动，每次训练下游模块，检查退化来自 backbone、图像 neck 还是 BEV 编码器。最后以独立相机模型作教师，在融合模型的预融合 Camera-BEV 特征上施加均方误差蒸馏，训练损失为任务损失加蒸馏项；教师只在训练期使用，因此推理结构不增加模块。
#### 方法对比分析
相比只比较融合模型的最终性能或传感器受损时的鲁棒性，本文把单一模态分支作为受控测量对象，并用逐级 Probe 定位差异，再以针对性蒸馏作干预。相机分支在四项任务和十种模型中均出现负差，而 LiDAR 结果方向不一致；这种模态不对称是其主要发现。Probe 仍只是冻结特征对指定任务的可解码性指标，不等同于通用表示质量。
#### 实验分析（精简版）
CMT 的3D检测中，相机分支 Probe mAP 从36.43降至10.50，差25.93点；相应 LiDAR Probe 从61.64升至61.88。蒸馏后，检测相机 Probe mAP提升20.42点，端到端占用预测 IoU提升4.00点。层级实验发现退化横跨相机 backbone、neck 和 BEV 编码器；极早切分 backbone 的 Probe 会同时压低独立相机参考结果，因此该位置不适合作整体比较。证据支持基准范围内的系统性差异，尚不能说明其他数据集或真实传感器故障中的效果。
#### 实用指南
复现要确保配对双方使用相同 decoder 结构与初始权重、相同数据划分和训练预算，并冻结被测表示；否则差值不能归因于分支来源。按原任务指标评估，层级探针必须逐次重训下游网络。蒸馏只约束融合前相机 BEV 特征，权重 λ_t 按任务设置。该思路可用于诊断机器人中的相机、深度、力觉分支，但需重新选取任务表征和 Probe 位置；本文没有提供机器人实验或跨域迁移验证。
#### 总结
核心思想：测量并修复融合中的支路退化
1. 冻结相机或雷达支路，只训练配对解码器。
2. 比较独立训练与联合训练的分支分数，并逐层定位差异。
3. 将独立相机表征蒸馏到融合支路，检查分支与整体任务是否改善。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25448v1)
- [arXiv](https://arxiv.org/abs/2610.25448v1)

---

<a id='2610.25600v1'></a>
## [Vector-Quantized World Model: Latent Feedback Motion Planning for Robot Manipulation](https://arxiv.org/abs/2610.25600v1)

**Authors:** Xuezhou Xu, Chung-Pang Wang, Zhenzhen Li, Benjamin Riviere

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

Visual world models have shown strong potential for robot planning, yet long-horizon manipulation remains challenging due to compounding prediction errors, large search complexity, and latent plans that the low-level controller cannot reliably reach. In this work, we introduce Vector-Quantized World Model (VQ-WM), a latent feedback motion planning framework that integrates discrete long-horizon search with continuous visual feedback control: a discrete latent predictor models transitions between vector-quantized latent states, allowing the planner to look far ahead and find a shortest path to the goal through discrete search and a continuous learned controller sequentially reaches the planned states through visual feedback, correcting deviations to mitigate compounding errors. Whereas most vector quantization uses Euclidean distance, we use a novel estimator of the first-passage-time as a quasi-metric, which aligns the planning abstraction with the controller's capabilities. We evaluate on the physical robot manipulation of the Tower of Hanoi puzzle: whereas existing state-of-the-art foundation models (VLAs, WAMs) only succeed when the solution trace in their training dataset, our method achieves 100\% success rate with only short-play data, demonstrate its ability to compose long-horizon solutions beyond the short demonstrated trajectories and execute them reliably. Our solution also outperforms strongest world-models by 13.3--85.3 percentage points on simulation benchmarks spanning manipulation and locomotion, demonstrating consistent progress and new capabilities.

### 论文解读
#### 摘要翻译
长时域机器人操作规划容易遇到三个难点：世界模型预测误差随滚动步数累积，连续动作序列搜索成本高，潜空间中的子目标也未必能被低层控制器稳定到达。本文提出 VQ-WM，将规划拆成离散搜索与连续视觉反馈控制。它用控制器闭环到达目标所需的 first-passage time（FPT）构造向量量化状态，使离散状态代表控制器真正可达的目标。规划器在离散状态图上找长路径，控制器再依次跟踪中间目标并纠正偏差。
#### 方法动机分析
已有潜空间规划多依据欧氏距离或直接预测子目标，但两个 latent 靠近并不代表机器人容易从一个到另一个。若目标超出执行策略能力，长计划即使预测正确也可能落空。作者的假设是，控制器完成局部转移所需的调用次数，比潜变量几何距离更适合定义规划抽象。
#### 方法设计详解
方法模块流程按输入到输出展开：冻结视觉编码器把图像变为 latent；训练目标条件控制器 π 和连续 latent 预测器 PZ。对起点 z，控制器朝目标 latent 闭环执行，在 PZ 中模拟，首次进入目标邻域的调用数即 FPT，预算内未到则视作不可达。作者估计数据状态各自覆盖的可达区域，用贪心集合覆盖挑选状态码本，再将每个状态指派给 FPT 最小的码。目标状态由于距离有向，单独选取最容易到达目标的码。随后将轨迹中离散状态变化编码为离散动作，学习动作码本和离散状态转移预测器 PQ。测试时，规划器用 PQ 展开候选图并以广度优先搜索寻找最短路径；若初始到目标足够近，就直接控制。高频控制器逐个追踪计划目标，用视觉反馈纠偏，按估计 FPT 分配执行时间。训练与推理设定为：实验用 V-JEPA 2 作为冻结视觉编码器，真实采集控制频率为30Hz；约五分钟的完整计划可在 NVIDIA RTX 5080 GPU 上一秒内生成。
#### 方法对比分析
相比连续潜空间动作优化，VQ-WM 将高层组合搜索离散化；相比欧氏量化或直接预测子目标，它把控制器可达性写进状态码本。规划器与控制器因此相对独立：添加新的控制输入可改变可达区域与量化，但不要求重新设计图搜索。代价是 FPT 估计受连续模型和控制器准确性影响，且码本建设需要额外闭环模拟。
#### 实验分析（精简版）
真实四圆盘汉诺塔需15步，VQ-WM用不超过五步长的 play 数据训练，H15任务进度达100%；对照的 π0.5 为3%，Cosmos Policy 为5%。模拟 Cube-Hard 得73.3%，比 H-JEPA 的60.0%高13.3点；Scene 得75.3%，比40.7%高34.6点。模拟汉诺塔中，FPT量化峰值成功率约90%，欧氏量化约40%。Walker加入本体输入后，VQ-WM成功率从22.0%升至97.3%。真实结果只基于少量固定配置，尚未检验频繁重规划。
#### 实用指南
从离线图像-动作轨迹先训练目标条件控制器和连续 latent 预测器，再估计 FPT 并构建可达状态码本，最后学习离散动作及转移预测。真实采集使用30Hz控制；汉诺塔评估在6个配置上各做3次。实现时需共同设置到达容差、FPT调用上限、码本大小和转移拒绝类别；码本过小会合并不同构型，过大会降低转移样本覆盖。作者计划探索多任务控制器和重规划频率。用于移动机器人或自动驾驶需添加安全约束，论文未验证该迁移。
#### 总结
核心思想：按控制器可达性离散化规划
1. 用闭环到达时间衡量 latent 状态间的可达性。
2. 以可达区域构造状态码本并学习离散转移图。
3. 搜索长路径，控制器依次到达目标并反馈纠偏。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25600v1)
- [arXiv](https://arxiv.org/abs/2610.25600v1)

---

<a id='2610.25711v1'></a>
## [SWAP: Test-Time Steerable Humanoid Whole-Body Collision Avoidance](https://arxiv.org/abs/2610.25711v1)

**Authors:** Zeji Yi, Guanqi He, Xiaofeng Guo, Guannan Qu, Pieter Abbeel, Yan Duan, Karen Liu, Guanya Shi

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

Humanoid robots operating in human-centric environments must coordinate the entire body to traverse tight, cluttered spaces while maintaining balance. Existing methods often struggle to jointly consider dynamic stability and collision avoidance from high-dimensional perceptual inputs. Moreover, for humanoids, safe behavior is inherently multimodal: the same obstacle may admit multiple valid traversals, which is difficult to represent with a single deterministic policy. To address these challenges, we propose SWAP (Steerable Whole-body Avoidance Policy), a hierarchical framework that separates collision-aware decision making from stabilization. A high-level, perception-conditioned diffusion policy generates multimodal, collision-aware motion proposals, while a low-level whole-body RL controller tracks these proposals and maintains dynamic balance. To efficiently train the high-level policy, we obtain scalable supervision via hindsight labeling: we roll out the low-level policy under randomized commands, place obstacles only in the space that each executed motion did not sweep, so the motion is collision-free by construction, and re-render the perception the robot would have seen. At deployment, Monte Carlo sampling with simple cost signals steers the diffusion model toward safe, task-aligned behaviors without heavy reward tuning.   Experiments in simulation and on a real Unitree G1 humanoid show reliable avoidance with static and dynamic obstacles. Test-time steering further enables customization, improving targeted rewards (e.g., speed, orientation, or posture) by 34% on average while maintaining a high collision avoidance rate. Together, these results support a practical solution to humanoid whole-body collision avoidance with test-time steerability. Project website: https://iscoyizj.github.io/SWAP-steerable-humanoid-oa-website/

### 论文解读
#### 摘要翻译
人形机器人要在拥挤环境中穿行，必须让全身避障与动态平衡同时成立。SWAP提出分层方法：高层感知条件扩散策略提出多种避障动作，低层强化学习控制器跟踪动作并维持平衡。作者用回顾式标注从仿真中自动构造碰撞监督，再在部署时通过候选轨迹采样和代价选择，把策略转向速度、朝向或姿态等目标。模拟与Unitree G1实验展示了静态、动态障碍避让及目标定制能力。
#### 方法动机分析
只规划身体中心会忽略手臂、腿和躯干扫过的空间；而单一确定性输出也难表达绕过同一障碍的侧移、下蹲或转身等多种可行解。主要驱动力与痛点在于同时获得平衡和多解避障；作者的关键假设是，把动作生成与稳定跟踪分开，并保留多模态运动先验，再依据碰撞和任务目标挑选候选动作，能够兼顾自然动作与安全性。训练还需避免昂贵的逐条人工遥操作。
#### 方法设计详解
整体流程先训练低层全身强化学习控制器，使其跟踪基座运动和上身关节命令。高层接收LiDAR点云与本体感觉历史，经PointNet++和GRU提取环境及状态特征，再由DiT扩散模型生成一段全身动作命令。训练标注时，研究者随机化命令并运行低层控制器，计算机器人实际运动扫过的体积；在该体积之外放置障碍，就能把这次运动标为无碰撞，并重新渲染机器人所见画面。扫掠区域内的障碍则提供碰撞负例，训练一个用于评价候选动作的Q函数。部署时，扩散模型一次生成N条候选，按用户代价或碰撞Q值排序，选出更符合当前任务且安全的命令段交由低层执行。真实机器人使用32条候选；感知运行在10Hz，扩散策略20Hz，低层控制50Hz，采用短命令块滚动更新。
#### 方法对比分析
端到端扩散直接输出动作，容易缺少稳定性；单一避障策略则通常把目标固定在训练奖励里。SWAP以强化学习跟踪器承担平衡，以扩散模型保留多种全身动作，再将目标选择放到测试时完成，其创新区别在于把任务目标从训练奖励中解耦，因此无需为每个新速度、朝向或姿态目标重新训练。回顾式标注把仿真执行轨迹转成监督样本，减少人工碰撞标注成本。其优势依赖稳定跟踪器、可信碰撞评价和足够快的批量推理。
#### 实验分析（精简版）
五类仿真任务的平均成功率为88.3±2.5%，碰撞率2.0±0.5%；去掉测试时引导后分别为72.4±3.8%和13.2±3.1%（均值跨3个随机种子）。目标奖励平均提升34%。Unitree G1五种真实场景各测试5次，成功3至4次，说明系统可运行但证据规模有限。未见布局来自相同场景分布族，不能据此推断任意环境中的分布外泛化。
#### 实用指南
论文提供项目网站，但本文所读材料没有给出可核验的代码许可与复现包信息。复现需先训练低层平衡跟踪器，再生成碰撞安全/负例数据，训练感知扩散策略及碰撞评价器，最后测量批量采样时延。论文训练感知编码器约200K步、扩散策略500K步，使用8张L40S GPU；真实设置使用Unitree G1、Livox Mid-360和Jetson Orin，感知10Hz、扩散20Hz、低层控制50Hz。迁移到其他机器人时，必须重新验证本体扫掠几何、传感器遮挡和安全标签质量。
#### 总结
核心思想：生成多解，按目标选安全动作。
1. 训练低层控制器稳定跟踪全身动作。
2. 从仿真执行轨迹自动构造碰撞标签。
3. 扩散模型提出多种候选动作。
4. 用任务代价和碰撞评价选取并执行。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25711v1)
- [arXiv](https://arxiv.org/abs/2610.25711v1)

---

<a id='2610.25628v1'></a>
## [Humanoid SRL: Augmenting Humanoids with Supernumerary Limbs for Coordinated Loco-Manipulation](https://arxiv.org/abs/2610.25628v1)

**Authors:** Fukang Liu, Feiyang Wu, Paul Barsa, Jaehwi Jang, Akshat Jain, Zhigen Zhao, Yipu Chen, Zhaoyuan Gu, Nikhil Vemuri, Anshul Chandaliya, Kaijie Zhu, Sean Lublinsky, Matthew Zhu, Tony G. Chen, Ye Zhao

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

Humanoid robots are envisioned to perform diverse human-level tasks through their anthropomorphic morphology. Their dual-arm configuration enables a broad range of everyday actions. Yet, as in humans, this morphology can constrain loco-manipulation when both arms are occupied or additional contacts are required beyond their reach. Supernumerary robotic limbs (SRLs) offer a way to extend these capabilities, but their integration requires the added limb to be coordinated with whole-body motion and balance. In this study, we develop and integrate a supernumerary arm into a humanoid platform to investigate how morphological augmentation reshapes its feasible action space. Through quantitative analysis and hardware experiments across two configurations, we show how the additional limb expands whole-body interaction capabilities by enabling manipulation and support to be coordinated beyond the constraints of the native dual-arm configuration. To control this augmented morphology, we leverage model-based trajectory optimization (TO) to generate dynamically feasible motions for goal-conditioned reinforcement learning (RL) controllers. The controllers are trained in simulation and transferred to hardware in a zero-shot manner, enabling coordinated whole-body skills for contact-rich tri-arm loco-manipulation. We further show vision-guided autonomous execution on selected tasks using diffusion policies trained from hardware demonstrations. Our findings suggest that humanoid morphology need not be restricted to human-like forms, but can be reimagined as an extensible design space for robotic augmentation.

### 论文解读
#### 摘要翻译
双臂人形机器人面对双手已被占用、目标超出可达范围或还需额外支撑时，仍会受到自身形态限制。本文在Digit人形机器人上增加一条4自由度超额机械臂，并比较颈部和背部两种安装方式。作者用动力学轨迹优化生成动作参考，再训练强化学习控制器协调原生双臂、额外机械臂和全身运动。硬件演示覆盖操作、支撑、行走中交互，并展示了视觉引导的自主任务。
#### 方法动机分析
驱动问题是形态本身限制了能同时进行的接触：双臂搬物时无法腾手开门或稳定另一物体，身体转向也未必能解决空间可达性。额外机械臂只有与全身平衡、环境接触和双臂任务分工共同设计，才可能成为有效增强。作者的假设是物理可行的轨迹参考可补足纯强化学习的样本和奖励设计难题，而学生策略能从仿真特权教师学习到硬件可观测控制。
#### 方法设计详解
研究设计一条可装在颈部或背部的4自由度机械臂，分别扩展前上方或后下方空间。控制流程先以全身动力学轨迹优化生成参考轨迹，约束包括身体动力学、接触与摩擦、关节范围和力矩。优化器输出全身状态、力矩与末端轨迹，作为强化学习的跟踪目标。Learn-to-Teach框架并行训练教师与学生：教师使用仿真特权状态，学生仅使用带噪本体感觉、动作历史和任务末端参考；两者共享训练经验，教师用PPO更新，学生以模仿目标逼近教师。部署时只保留学生，控制Digit全身并零样本迁移到硬件。对于三项选定自主任务，额外加入视觉扩散策略：输入躯干及机械臂相机RGB-D和关节状态，输出短时末端目标、夹爪和控制器选择；每次预测64步，执行32步后根据新观测重规划。硬件学生控制器以200Hz运行。
#### 方法对比分析
与仅优化双臂动作相比，超额机械臂扩大的是可达区域和可分配的接触角色，既能接力搬运，也能在原生双臂持物时操作或支撑。与直接从奖励学习全身策略相比，轨迹优化提供满足接触和力矩约束的参考，教师—学生共训练再处理部署观测受限的问题。颈装与背装具有互补的空间优势，不能视为任一安装在所有任务上都更好。方法适用于额外肢体可提供明确可达或支撑收益的任务。
#### 实验分析（精简版）
采样总可达体积由原生双臂的1.95立方米增至颈装2.94（+50.3%）和背装2.78（+42.3%）。外部撑扶实验中，机器人可承受10厘米扰动；没有SRL支撑时在7.5厘米失衡。L2T相对Asym-PPO平均降低末端跟踪误差11.2%，但每个任务只用一个训练种子。硬件自主演示仅覆盖三项任务，不能证明广泛环境泛化。
#### 实用指南
论文公开了实验视频和数据材料说明，但所读正文未提供可核验的代码许可。复现要准备增强形态动力学模型、接触约束、任务轨迹优化和充分的仿真随机化。硬件为约48千克Digit平台，SRL长83厘米、质量4.3千克、持续负载2千克；RL控制200Hz。迁移到其他平台前应评估装载质量、安装点、碰撞、传感器及控制接口，并按其工作空间重新规划参考轨迹。
#### 总结
核心思想：加一臂，重分配全身接触。
1. 分析新肢体改变的可达与支撑空间。
2. 优化全身可行参考并训练可部署学生控制器。
3. 让三臂分工完成操作、支撑与移动交互。
4. 用视觉策略为低层控制器提供短时任务目标。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25628v1)
- [arXiv](https://arxiv.org/abs/2610.25628v1)

---

<a id='2610.25400v1'></a>
## [RAHF-Net: Range-Aware Heterogeneous Fusion for LiDAR-Assisted RGB Semantic Segmentation](https://arxiv.org/abs/2610.25400v1)

**Authors:** R. D. Lin, Pengcheng Weng, Ge Wang, Han Ding, Fei Wang

**Published:** 2026-10-18

**Categories:** cs.CV

**Abstract:**

This paper studies LiDAR-assisted RGB semantic segmentation, where LiDAR point-cloud features improve dense pixel-wise predictions in the RGB image plane rather than 3D point labels or detections. Existing 2D fusion methods often project point clouds into shallow depth/range maps and apply spatially uniform fusion, discarding rich 3D structure and ignoring the range-dependent reliability of outdoor LiDAR. We propose RAHF-Net, a Range-Aware Heterogeneous Fusion Network. Its Cylindrical-to-Perspective (C2P) module projects pretrained cylindrical voxel features into the image plane, preserving structured 3D semantic priors. A Range-Aware Heterogeneous Fusion (RAHF) module then uses a soft depth-derived range mask to perform LiDAR-guided geometric enhancement in reliable near-field regions and visual-context enhancement in sparse far-field regions. On KITTI-360, RAHF-Net achieves 65.76 mIoU, outperforming representative RGB-only and RGB-LiDAR baselines, including CMX by 1.45 points. Experiments on DELIVER further demonstrate improved robustness under adverse weather, illumination degradation, and sensor corruption.

### 论文解读
#### 摘要翻译
本文研究用LiDAR辅助RGB图像中的逐像素语义分割。已有融合方法常把点云投影成浅层深度图，并在整幅图上统一融合，既忽略点云的三维结构，也忽略激光雷达可靠性随距离变化。RAHF-Net先把预训练柱面点云特征投影到图像平面，再根据距离软掩码分别融合近场几何和远场视觉上下文。在KITTI-360和DELIVER上的实验显示，它提升了分割精度并增强了恶劣成像条件下的表现。
#### 方法动机分析
关键痛点有两点：二维深度投影会压掉三维编码器已经学到的结构信息；近处LiDAR点密、边界清晰，远处点稀疏且更容易受噪声和标定偏差影响，统一融合会在远处引入冲突。作者的动机是让表示保留3D语义，同时让融合强度随距离平滑变化，而不是将几何信息视作全图同等可靠。
#### 方法设计详解
输入是一张RGB图像和已标定的点云。图像由SegFormer-B2提取视觉特征；点云先柱面体素化，再通过预训练Cylinder3D获得稀疏的高层几何语义特征。C2P模块恢复每个非空体素的空间中心，用相机内外参映射到图像像素；多个特征落到同一像素时取最大值，再调整分辨率与通道数以对齐图像分支。网络还将原始点云投影成稀疏深度图，重叠像素保留最近可见点。深度经归一化和sigmoid形成软范围掩码：近处值较大，远处较小，过渡由平滑参数和距离阈值控制。RAHF据此分成两路：近场用LiDAR几何产生注意力以修正RGB边界；远场减少稀疏点云影响，通过全局RGB上下文补足局部视觉信息。两路结果按掩码软混合，解码为相机视角的稠密标签。训练将类别加权交叉熵与Lovasz-Softmax结合，兼顾长尾类别和mIoU目标。
#### 方法对比分析
与仅投影深度或统一跨模态注意力相比，C2P传递的是柱面编码器提取的高层三维特征，RAHF则把近处几何细化和远处视觉推理区分开。方法的创新在于按物理距离异质融合，而不只是增加一个点云输入分支。它面向已有相机-LiDAR配准、且需要图像平面分割输出的系统；不能直接与3D点云分割或目标检测结果等同。
#### 实验分析（精简版）
KITTI-360上RAHF-Net达到65.76 mIoU，比任务匹配的CMX高1.45点。DELIVER恶劣天气与传感器扰动测试为58.05 mIoU，比唯一可用基线高5.08点；欠曝光仍只有37.22，说明几何补充无法完全挽救严重视觉退化。作者未报告完整系统实时性，并明确DELIVER对比不是全面的先进方法排名。
#### 实用指南
论文给出可复现设置：视觉骨干SegFormer-B2、预训练Cylinder3D、4张RTX 3090、每卡batch size 4、AdamW学习率6×10⁻⁵；柱面点云覆盖半径50米与360度视野。实际迁移要确认相机-LiDAR标定和同步，评估点云编码器的计算成本，并重新选择数据相关的距离掩码参数。论文将掩码作为距离先验，而非学习到的传感器不确定度。
#### 总结
核心思想：近处信几何，远处信视觉。
1. 编码图像与柱面点云特征。
2. 将三维特征投影到相机平面并生成距离掩码。
3. 近处融合几何，远处强化视觉上下文。
4. 混合特征并输出像素级语义图。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25400v1)
- [arXiv](https://arxiv.org/abs/2610.25400v1)

---

<a id='2610.25401v1'></a>
## [Energy-Score Stabilized Attentive Mixture-Density Policy: A Lightweight Approach to Multimodal Robot Motion](https://arxiv.org/abs/2610.25401v1)

**Authors:** Sachiya Fujita, Hideyuki Ichiwara, Shigeki Sugano, Tetsuya Ogata

**Published:** 2026-10-18

**Categories:** cs.RO

**Abstract:**

For practical motion generation, it is essential to handle multimodality (multiple actions can be correct for the same observation) while maintaining low inference cost and strong generalization from limited demonstrations. We introduce an attentive mixture-density policy trained with the Energy Score that represents future visuomotor trajectories as explicit probability distributions with a single forward pass. Concretely, we output Gaussian mixture parameters for joint commands (and optionally attention points) and optimize them with the Energy Score (computed as a sum of 1D CRPS under a diagonal covariance assumption) instead of the negative log-likelihood (NLL) to avoid MDN singularities and gradient-scale instabilities. Our architecture couples explicit spatial attention with mixture-density outputs, enabling robust multimodal behavior selection under visual noise and small-data regimes. On an 8-DOF manipulator, the method generates distinct avoidance trajectories and stabilizes learning in obstacle avoidance. Moreover, in a randomized moving-object grasping task, where spatiotemporal multimodality and fast timing decisions are required, the proposed model outperforms Diffusion Policy by 15-25% under several unseen conditions, demonstrating superior generalization and real-time feasibility.

### 论文解读
#### 摘要翻译
机器人在同一观测下可能有多种正确动作，例如从障碍左侧或右侧通过。确定性回归容易输出两种轨迹的平均值；扩散策略虽能生成多解，却需要反复采样，可能难以满足边缘设备的低延迟要求。本文提出GMMSARNN，将注意力式视觉编码、循环状态和高斯混合动作分布结合，并用Energy Score稳定训练。在8自由度机械臂的避障和移动物体抓取实验中，作者报告了低推理耗时和对未见运动模式的适应能力。
#### 方法动机分析
主要痛点是多模态、少样本和实时性很难同时满足：均方误差使确定性策略产生模式平均；MDN的负对数似然可能因方差收缩而数值发散；扩散模型推理又较重。作者的驱动力是保留MDN一次前向、多分支的轻量特性，同时用更稳定的概率评分规则训练，并通过空间注意与循环记忆适应视觉变化和运动时序。
#### 方法设计详解
图像先经CNN和空间注意提取关注点，再与当前关节角输入LSTM，整合视觉和时间状态。输出层预测高斯混合的权重、均值和方差，描述未来关节动作；可选地也为视觉注意点建立独立混合分布。注意点分布与关节动作分开，避免绕障动作改变视觉焦点。训练以Energy Score替代负对数似然。为避免完整协方差计算开销，作者采用对角独立假设，将多维Energy Score拆为一维高斯CRPS项之和，并与辅助图像重建误差共同优化。CRPS比较预测分布与真实值的累积分布差异，能避免MDN方差趋零时的似然奇异。训练时混合分量批量处理；部署只需一次模型前向。实验用4个混合分量；在Jetson Orin NX上推理，策略10Hz更新，并将命令线性插值到50Hz执行。
#### 方法对比分析
与确定性回归相比，高斯混合显式保留左右绕行等离散动作模式；与扩散策略相比，它不需要多轮去噪；与常规MDN相比，Energy Score缓解负对数似然下方差塌缩的训练不稳。代价是对角协方差不显式建模分量内部的维度相关性，且有限混合数可能出现组件失活。该方法适合需要快速多模态输出的机器人模仿学习，不会自动提供安全约束。
#### 实验分析（精简版）
机械臂障碍绕行中，48条示教训练后提议方法成功率约70%，ACT约65%，Diffusion Policy约45%（每模型共200次试验）。移动物体以未见轨迹且2倍速度运行时，GMMSARNN-JK仍成功45%。其Orin NX推理为62.6毫秒，满足10Hz周期。实验规模和任务类型有限，不能推出跨平台泛化。
#### 实用指南
论文报告使用AIREC-Basic 8自由度机械臂，10Hz在线策略、50Hz命令插值；移动目标训练80条示教，混合数M=4。部署时需检查Jetson推理延迟和动作插值平滑度，并覆盖目标速度、相位和转向变化。作者还发现混合权重快速变化会引起振动；Sparsemax可减振，但会削弱快速响应，因此模式切换策略需要结合具体任务调试。
#### 总结
核心思想：用稳定混合分布一次生成多解。
1. 注意力与LSTM提取视觉和时序状态。
2. 高斯混合表示多种未来动作。
3. Energy Score训练分布并避免似然奇异。
4. 低延迟推理并插值执行轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25401v1)
- [arXiv](https://arxiv.org/abs/2610.25401v1)

---

<a id='2610.25665v1'></a>
## [SpaRV: Sparse Radiance Voxels for Scalable Geometrically Consistent LiDAR-Visual Mapping and Rendering](https://arxiv.org/abs/2610.25665v1)

**Authors:** Jianheng Liu, Yunfei Wan, Fangming Cheng, Xiangyu Hong, Zuhao Zou, Haotian Li, Longji Yin, Chunran Zheng, Jiarong Lin, Fu Zhang

**Published:** 2026-10-19

**Categories:** cs.RO, cs.CV

**Abstract:**

Robotic simulation, navigation, and inspection need photorealistic and geometrically accurate maps. Radiance fields achieve photorealism, but building them from robot data raises three challenges. A robot's trajectory serves its task, not reconstruction, so images from few viewpoints leave the geometry under-constrained. Residual errors in camera poses and intrinsics blur the map. The map's GPU memory grows with scene size, which limits its level of detail in large scenes. We present \textbf{SpaRV}, a \textbf{Spa}rse \textbf{R}adiance \textbf{V}oxel map rendered by differentiable ray marching, which lets irregular LiDAR scans supervise it alongside images. LiDAR depth and occupancy anchor the under-constrained geometry, and normal and smoothness regularizers keep it smooth between LiDAR points. To correct camera errors, SpaRV alternates map training with photometric optimization of the camera poses and focal lengths. To bound GPU memory in training and rendering, SpaRV holds one submap at a time at full resolution and the rest at coarser octree levels. Experiments show that SpaRV keeps the geometry consistent away from the captured trajectory and sharpens textures by optimizing the camera parameters. On one 24\,GB GPU, SpaRV reconstructs a university campus captured in 63{,}582 images and simulates other cameras and LiDARs. Our code will be released at \url{https://github.com/hku-mars/SpaRV}.

### 论文解读
#### 摘要翻译
机器人仿真、导航和巡检需要既逼真又几何准确的环境地图。仅凭机器人采集的图像，观测视角往往稀疏且不均匀；相机位姿误差会模糊纹理，大场景的高分辨率辐射场又会超出显存。SpaRV提出稀疏辐射体素地图，以可微光线行进统一渲染相机像素和LiDAR射线，用激光深度与占用信息约束几何，并交替优化地图和相机。八叉树多层级裁剪使其能在单张24GB GPU上处理公里级校园场景。
#### 方法动机分析
作者针对三个工程痛点：图像短基线难以确定几何深度，导致新视角下出现漂浮表面；逐帧相机误差会把同一表面投到不同像素，造成纹理模糊；全场精细体素不适合有限GPU内存。核心思路是让LiDAR沿射线提供度量锚点，用稀疏体素保留局部表面，并根据视点远近分配地图分辨率。
#### 方法设计详解
SpaRV把场景表示成自适应稀疏辐射体素，密度在共享顶点上三线性插值，颜色用球谐表示。八叉树按空间需求细分；世界远场通过坐标压缩映入有限域。渲染时Morton编码查询占用叶节点，空区域可整块跳过，逐射线可微marcher依次穿过体素并进行体渲染。因此，相机像素、非规则LiDAR光束和鱼眼/全景射线都可用同一渲染器。训练损失包括图像颜色与结构相似度、LiDAR深度、法线一致性、相邻密度平滑，以及相机射线首个LiDAR占用体素的终止引导，减少锚点间漂浮几何。地图更新与相机校正交替进行：冻结地图，根据渲染图像残差微调各帧位姿和焦距；再固定相机更新地图。大场景先训练覆盖全域的粗图，再逐子图细化。父级体素保留为粗层；训练和部署共用按屏幕投影尺寸决定的八叉树切面，让视点附近细、远处粗，从而限制GPU常驻体素数。
#### 方法对比分析
与纯图像NeRF或3D Gaussian Splatting相比，SpaRV用LiDAR几何监督改善外推视角下的表面一致性；与只给测量射线加深度损失的方法相比，占用引导和法线平滑还能约束未被LiDAR直接覆盖的区域。与栅格化渲染相比，逐射线行进支持非针孔相机和稀疏LiDAR。其设计把地图、传感器仿真和表面提取统一在可查询的场景场中，但需要LiDAR和相机标定输入。
#### 实验分析（精简版）
Replica八场景外推视角平均PSNR为36.690 dB，高于共享体素表征基线SVRaster的33.347 dB；平均LPIPS为0.077。FAST-LIVO2上相机校正使平均训练视角PSNR提升3.3 dB。公里级Campus地图在单张24GB RTX 4090上以15.6GiB显存渲染68.8FPS；该数字对应论文指定的1280×960飞行序列和部署裁剪设置，不能脱离显存上限与评测分辨率外推。
#### 实用指南
论文使用自定义CUDA光线行进和稀疏Adam；基准实验在24GB RTX 4090上训练20,000步，体素每1,000步调整，相机优化与地图训练交替。复现时，大场景还需设定子图尺寸、显存体素预算和屏幕粒度，并验证传感器标定误差与裁剪切换延迟。适用于已具备LiDAR—相机数据、需要新视角渲染、表面提取或传感器模拟的机器人地图；论文当前地图为静态且无语义标签，动态环境需扩展。
#### 总结
核心思想：用激光锚定几何，按视距分配细节。
1. 编码图像和LiDAR为稀疏辐射体素。
2. 用统一光线行进渲染相机与测距射线。
3. 交替优化几何地图和相机参数。
4. 近处加载细体素，远处保留粗层级。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25665v1)
- [arXiv](https://arxiv.org/abs/2610.25665v1)

---

<a id='2610.25534v1'></a>
## [AQUAACT: A Joystick-Level Underwater Teleoperation Dataset and Closed-Loop Benchmark](https://arxiv.org/abs/2610.25534v1)

**Authors:** Zhenqi Wu, Trung Tien Dong, Aditya Penumarti, Mohamed A. Fouad, Menna A. Fouad, Mingyang Mao, Jane Jaejeong Shin, Xiaomin Lin

**Published:** 2026-10-18

**Categories:** cs.RO

**Abstract:**

Underwater robots support a growing range of work, from inspecting offshore structures and pipelines to surveying reefs and accompanying divers. Most of these tasks, however, are still flown by hand by a pilot, which raises the question of whether they can be carried out autonomously instead. On land, vision-language-action (VLA) models already let robots carry out tasks autonomously from language instructions, which makes them a natural candidate for taking the pilot's place. Underwater, however, there is little data to train such models. We take inspiration from how these tasks are flown today: the pilot watches the camera image and steers with a joystick, whose commands the flight controller turns into motion. We therefore record the pilots and introduce AQUAACT, to our knowledge the first publicly available language-annotated underwater dataset pairing camera images with human joystick commands. It comprises 1.9 hours of real-world and 17.9 hours of simulated demonstrations, both flown by human pilots, together with the vehicle's inertial measurements. Experiments with AQUAACT show that its simulated demonstrations transfer to real data and greatly reduce the real demonstrations a policy needs. A closed-loop benchmark of seven policies in simulation, from imitation-learning baselines to VLA models, each trained with its default recipe, shows that these tasks remain far from solved: no policy completes more than a fifth of the episodes. Pretraining on land data gives large VLA models no advantage over small policies trained from scratch, a gap that land-trained models have yet to close underwater.

### 论文解读

#### 摘要翻译
AQUAACT提出水下机器人语言标注数据集与闭环基准，配对相机、IMU和飞控收到的摇杆命令，含1.9小时真实及17.9小时模拟示范。七种策略的仿真闭环成功率最高仅约五分之一。

#### 方法动机分析
水下巡检和调查仍依赖人工驾驶，训练自主策略需要记录驾驶员真正发出的动作。既有数据多用推进器PWM或机械臂动作。论文假设以摇杆通道为动作、让实机和仿真共用控制链，可低成本扩充并迁移示范。

#### 方法设计详解
BlueROV2 Heavy通过QGroundControl和ArduSub接受摇杆控制。数据保存前视图、六维IMU和四个归一化动作surge、sway、heave、yaw；roll、pitch由飞控稳定。数据15Hz记录，对齐后10Hz导出。

模拟端由Isaac Sim渲染和积分运动、MarineGym提供水动力，ArduSub软件在环飞控以62.5Hz锁步；参数按实艇响应校准，并变化动力学和水下外观。数据含潜水员跟随、结构扫描、牡蛎礁与管线任务及语言指令。

输入输出链条为“任务、图像、IMU→预测四通道摇杆→ArduSub驱动推进器→车辆运动并返回观测”。飞控仍负责姿态稳定，仿真记录器产生与实机同构的数据，保留闭环控制误差。

迁移实验用模拟甲板扫描预训练SmolVLA、π0.5，再以1/4/7/13段真实数据微调并在独立回合测试；遵循默认配方，批量分别为64、16，最多6000步并早停。闭环评测七种策略各跑50回合，以10Hz控制、每五步重规划，按覆盖、环绕和跟踪判定成功。

#### 方法对比分析
AQUAACT以飞控前摇杆输入为动作，并让真实和模拟数据共用ArduSub控制链；这与推进器指令或机械臂动作数据不同，也比简化运动学回放更接近遥操作。它贡献在数据和统一基准，而非新网络，可横向比较七种模仿学习与VLA策略。

其新意是把真实摇杆、同控制链仿真和长任务成功判据放在一起。基准显示小策略也能与大VLA竞争。它适合遥操作示范和仿真迁移研究；推进器级或机械臂控制需另行适配动作空间。

#### 实验分析（精简版）
模拟预训练后，SmolVLA用一段实示范即达到横移相关性0.70、方向准确率0.88，接近公共权重直接训练13段的结果；误差仍较高（0.124对0.100）。闭环中π0.5成功10/50，ACT和Diffusion Policy各9/50；72%回合碰撞，碰撞无关成功率仅4–12%。管线与牡蛎礁无人成功。实验为单种子校准仿真，不能代表真实海域表现。

#### 实用指南
复现需对齐传感器时间戳、保留飞控映射，并按潜水日切分。真实集约1.99小时/53段，模拟集17.9小时/513回合。迁移到新水域须重新校准动力学和成像，重点解决持续跟踪与避碰。数据集公开；代码状态以项目主页为准。

#### 总结
核心思想：让水下策略学习驾驶员摇杆操作。

1. 记录真实机器人上的图像、IMU、语言任务和摇杆输入。
2. 在相同飞控链中校准仿真并扩展驾驶示范。
3. 用模拟预训练辅助真实微调，再以闭环任务检验策略是否能持续完成操作。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25534v1)
- [arXiv](https://arxiv.org/abs/2610.25534v1)

---

<a id='2610.25690v1'></a>
## [ScaleNav: Scalable Internet Video Pretraining for Language-Conditioned Local Navigation](https://arxiv.org/abs/2610.25690v1)

**Authors:** Yihui Yang, Yuanhong Yu, Jiawei Wang, Zhongrui Yu, Jia Shi, Kun Zhan, Ruizhen Hu, Xiaowei Zhou, Yu Liu, Sida Peng

**Published:** 2026-10-19

**Categories:** cs.RO

**Abstract:**

We study language-conditioned local navigation, where a robot must navigate to a text-specified target visible in its initial observation and stop at the described location. End-to-end navigation policies require diverse training data to generalize across scenes, yet real-robot data is costly to collect and simulated data leaves a sim-to-real gap. Internet egocentric videos offer a scalable source of real-world navigation experience. Our key insight is to learn goal-directed visual dynamics from Internet egocentric videos, and use these dynamics to support robot action generation. To this end, we propose ScaleNav, a unified world-action framework with three-stage training. We first perform instruction-conditioned video pretraining on these videos to learn how observations evolve as the camera approaches instructed targets across diverse real-world scenes. Action alignment then uses action-labeled simulated data to train the model to predict executable action chunks. Joint fine-tuning subsequently optimizes all trainable components together for navigation. ScaleNav achieves success rates of 86.5% on a single-view adaptation of Short-Horizon OVON and 94.9% on our HabitatGS-LN benchmark, outperforming the evaluated baselines on both benchmarks. We further deploy and evaluate ScaleNav on a real quadruped robot, with results supporting the model's ability to generalize to real-world navigation. Code and videos are available at https://zju3dv.github.io/ScaleNav/

### 论文解读

#### 摘要翻译
ScaleNav研究机器人如何按自然语言走向初始视野内的目标并停在指定位置。它先从互联网第一视角视频学习人类接近目标时的视觉变化，再用有动作标签的仿真数据教会模型输出机器人动作块。论文在两个仿真基准及Unitree四足机器人上评测，显示这种迁移优于所比较的方法。

#### 方法动机分析
真实机器人动作数据采集成本高，纯仿真又难免与现实有差距；互联网视频场景丰富，却没有机器人动作标签。现有方法常先从视频估计轨迹、几何目标或潜在动作，这些中间标签若有偏差，会传入策略。ScaleNav的核心假设是：视频可先提供“按指令靠近目标时画面怎样变化”的导航先验，再由带动作标签的模拟数据把它转成可执行控制。

#### 方法设计详解
模型采用共享Diffusion Transformer联合预测未来视频潜变量与动作块。输入是语言指令、当前RGB、初始帧及最近历史；动作表示机器人局部坐标的平移Δx、Δy和偏航变化Δψ。VAE视频潜变量和归一化动作分别嵌入成token后拼接，由DiT共同处理，两个输出头分别预测视频与动作的flow速度，去噪后反归一化得到动作。机器人执行一块动作后获取新图像，再预测下一块；预测视频可选解码，实际控制不依赖外部规划器。初始帧作为常驻锚点，最近两块观察和动作构成滑动历史，chunk因果注意力使当前块可参考历史但不访问未来。每块含9帧/9个动作，输入采样为4Hz。

训练分三阶段：第一阶段在18.2K条互联网导航视频上预训练Wan2.2-TI2V-5B，先双向训练2K步，再按chunk因果方式训练8K步；第二阶段用650个场景中的2.6K条模拟动作示范对齐动作头，只训练动作模块和DiT自注意力，同时以10%概率回放互联网视频；第三阶段解冻其余DiT和视频头联合微调，回放比例20%。动作与视频用flow matching损失共同优化。训练使用AdamW、BF16，三阶段全局批量大小64/64/32，学习率峰值为1e-5/1e-5/5e-6。推理采用4步Euler去噪，按4Hz执行动作块。

#### 方法对比分析
ScaleNav的关键差异是直接从无动作视频学习导航视觉动态，不构造伪轨迹或伪动作，再用模拟动作数据连接到机器人控制；与只预测视觉计划、依赖外部搜索器选动作的方法相比，它直接预测可执行动作块。初始帧锚点保留语言中左右、序数等指代所依赖的参照，滑动历史支持闭环推进。它适合目标从初始视野可见的室内局部导航，不能据此替代需要地图规划的长程任务。

#### 实验分析（精简版）
在单视角Short-Horizon OVON上，ScaleNav成功率/路径效率为86.5%/71.3%；在包含157个回合、45个训练外场景的HabitatGS-LN上为94.9%/86.1%。相同模拟动作数据下，去掉互联网视频后成功率分别低5.3和4.5个百分点。Unitree Go2三处室内环境共60次试验，成功率83.3%，无互联网视频版本为65.0%。结果支持视频先验的价值，但实机试验规模有限；测试主要是初始视野可见物体，户外和区域目标尚未验证。

#### 实用指南
复现需保留初始帧与最近两块历史、按4Hz处理512×288图像，并对齐动作块边界。互联网数据先做指令和片段筛选；模拟数据来自650个场景、2.6K条动作轨迹。四步去噪在RTX 4090上的平均推理调用为684ms/动作块，按块内9个动作折算约76ms/动作，尚不含网络、渲染和机器人执行时间，边缘部署需重新评估。方法适用于视野内目标导航；迁移到底盘时要重新定义动作和执行频率。

#### 总结
核心思想：用人类视频教机器人朝目标移动。

1. 从互联网第一视角视频学会目标导向的画面变化。
2. 用模拟动作示范把视觉先验对齐为机器人动作块。
3. 联合微调并在新观测反馈下反复执行，直到停在目标附近。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.25690v1)
- [arXiv](https://arxiv.org/abs/2610.25690v1)

---

