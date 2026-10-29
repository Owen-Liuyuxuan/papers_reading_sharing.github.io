time: 20261029

# Arxiv Computer Vision Papers - 2026-10-29

## Executive Summary

## 今日摘要

本日选读一篇关于多指灵巧手连续位姿控制的研究。作者将抓取数据、逐步降低摩擦的仿真课程、特权策略蒸馏和腕部深度视觉结合，并额外学习独立的残差安全过滤器。在真实 LEAP 手上，状态策略针对四种物体和六种腕部朝向完成 4,800 个连续目标；在 2 cm、0.4 rad 阈值下成功率为 95.2%。加过滤器后成功率为 91.5%，掉落从 1,200 个目标中的 16 次降至 4 次。视觉策略在刀具、酒杯上的感知迁移仍有缺口；所有策略和过滤器按物体单独训练，下游组合操作也依赖脚本化机械臂路径。

---

## Table of Contents

1. [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](#2610.37652v1)

---

## Papers

<a id='2610.37652v1'></a>
## [Pushing Dexterity and Safety for In-Hand Object Pose Reaching That Works Remarkably Well](https://arxiv.org/abs/2610.37652v1)

**Authors:** Ian Chuang, Hiya Shah, Vamsi Chilamkur, Iman Soltani, Claire Tomlin

**Published:** 2026-10-27

**Categories:** cs.RO, cs.LG, eess.SY

**Abstract:**

Dexterous manipulation with multi-fingered hands is the next frontier of robotics, yet learned hand skills remain limited to simple grasps or basic repeated motions, such as rotating an object. As robots attempt more demanding in-hand maneuvers, safety, such as not dropping the object, becomes critical, yet it is rarely measured. We study in-hand object pose reaching, a task that demands diverse and difficult grasps: moving an object to almost any position and orientation within reach of the fingers, at any wrist orientation, through long sequences of consecutive goals. We present a recipe for training such policies for a LEAP hand in simulation without human data, built on a range of insights, from diverse resets and goals via generated grasp datasets to an adaptive friction curriculum. With careful system identification, the policies transfer zero-shot to the real hand, using either motion capture or a single wrist-mounted depth camera, and prove remarkably capable and robust. On top of the task policy, we add a learned residual safety filter with adjustable conservativeness, letting us tune the balance between performance and safety and greatly reduce object drops. In 13.3 hours of real-robot evaluation over 4,800 unseen goals across 4 objects and diverse wrist orientations, the state-based policy reaches 95.2% of goals within 2 cm and 0.4 rad, and the filter cuts its drops by 75% while still reaching 91.5% of goals. The policies can also spin a lightbulb held in an arbitrary pose, reach out-of-distribution goals, withstand fast arm motion and perturbations, and, needing only a wrist camera, serve as portable skills such as screwing in a lightbulb. We release the full recipe as a platform for studying performance, reliability, and safety in high-dimensional dexterous systems. Videos and more are available at https://ian-chuang.github.io/dexsafe/.

### 论文解读

#### 摘要翻译
本文研究多指灵巧手能否把物体连续移动到手掌坐标系的不同位置和姿态，同时减少掉落。作者在仿真中训练 LEAP Hand 位姿到达策略，并在其上叠加可调的残差安全过滤器。真实机器人实验覆盖四种物体、六种腕部朝向和数千个连续目标；过滤器明显减少掉落，但会牺牲部分目标成功率。作者还展示了仅用腕部深度相机进行旋转和组合操作的可能性。

#### 方法动机分析
既有灵巧操作常集中在单轴旋转、固定腕部方向或短序列。真实手内操作还要求物体平移到指尖可支撑的位置，并在长时间运行中控制掉落。作者假设从可抓取姿态构造目标、逐步降低仿真摩擦，能让策略学习更广的操控；单独训练安全过滤器则可调节安全与成功率之间的取舍。

#### 方法设计详解
作者为每种物体生成抓取库：先用 LightningGrasp 产生十万抓取，再用 RRT 扩展到五十万，碰撞过滤后保留约 42–45 万。初始状态和连续目标均从库中抽取。仿真采用经过实机辨识的伺服器模型和延迟，并随机化摩擦、质量、几何、扰动及深度噪声；物理仿真 120 Hz，策略控制 20 Hz。

训练先用 PPO 和特权状态学习教师，再以在线 DAgger 蒸馏为实机可用的状态策略，之后 PPO 微调；最后训练读取腕部 160×120 深度图的视觉策略，并加入辅助位姿预测。课程从高摩擦和较近目标逐步转向现实摩擦及全抓取库目标。任务奖励衡量位置、朝向、转速和保持时间的进展，并惩罚掉落、自碰撞、非指尖接触与剧烈动作。安全过滤器在冻结策略动作上加残差，以存活奖励减去加权修正代价训练；参数 w 控制干预强度，状态过滤器随后被蒸馏为深度输入版本。

#### 方法对比分析
与仅靠任务奖励间接降低掉落相比，独立过滤器可单独调节干预，无需重训任务策略。与基于安全价值函数的过滤器相比，它只需一次网络前向计算，不必在线搜索候选动作；代价是没有形式化安全保证，也可能压制有用动作。视觉策略还把感知误差带入控制闭环，不能与状态策略混为一谈。

#### 实验分析（精简版）
主实验含 4,800 个连续目标。状态策略在 2 cm、0.4 rad 阈值下成功率为 95.2%；过滤后为 91.5%，但掉落从 1,200 个目标中的 16 次降至 4 次。深度策略过滤后成功率从 71.8% 降到 66.7%，掉落由 88 次降至 46 次。视觉位姿误差在刀具和酒杯上有长尾；脚本化翻转、拧灯泡等任务只做有限试验，不能证明自主高层规划。

#### 实用指南
论文称将发布代码和检查点。复现时需重点校准伺服动力学、延迟、碰撞模型和低摩擦范围。抓取库使目标更可达，但不保证每个姿态真实可行。策略与过滤器都按物体训练；迁移到新物体或手型需重新建库、辨识和训练，不能直接当作通用安全层。

#### 总结
核心思想：分开学习灵巧操作与安全修正
1. 从抓取姿态构造多样目标。
2. PPO 学任务，再蒸馏为状态和深度策略。
3. 残差过滤器修正动作，调节安全与成功率。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.37652v1)
- [arXiv](https://arxiv.org/abs/2610.37652v1)

---

