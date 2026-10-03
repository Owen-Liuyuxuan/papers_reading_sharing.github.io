time: 20261003

# Arxiv Computer Vision Papers - 2026-10-03

## Executive Summary

## 今日摘要

本次筛选覆盖 2026 年 10 月 1–3 日提交的 CS.CV/CS.RO 论文：抓取 100 篇，关键词与领域规则保留 41 篇，逐篇评分后选择 1 篇完成 PDF 阅读。值得优先关注的是 QueryArt：它不先尝试分割整块活动部件，而是围绕指定的操作点预测关节类型和轴线，并把单目不可辨识的尺度留给一次查询点深度测量。这个表示让单图运动假设能够转成显式、几何一致的机器人轨迹。

实验覆盖多个真实/合成训练源、两个未见数据集和 Spot 移动操作。论文报告 OOD 整体成功率为 69.08% 与 80.05%，机器人试验为 40/57 次计分成功。值得一起阅读的是方法中的深度归一化消融和实机视角分组结果：成功率会随视角与遮挡变化，而且抓取失败未纳入计分，说明该结果验证了预测几何的可操作性，但仍不能代表完整操作系统的端到端可靠率。研究方向上，交互锚点条件化与尺度不变几何为单视角具身感知提供了可迁移的设计思路。

---

## Table of Contents

1. [Query-Conditioned Articulation Estimation from a Single Image](#2610.01726v1)

---

## Papers

<a id='2610.01726v1'></a>
## [Query-Conditioned Articulation Estimation from a Single Image](https://arxiv.org/abs/2610.01726v1)

**Authors:** Abdelrhman Werby, Fabio Scaparro, Kai O. Arra

**Published:** 2026-10-01

**Categories:** cs.RO

**Abstract:**

Enabling robots to estimate the kinematic parameters of articulated objects unlocks a wide range of capabilities for interaction and manipulation. The estimation has to happen from the information the robot currently observes, often just a single RGB image of an object it has never seen before. Current single-image approaches couple articulation part segmentation with articulation estimation, making their predictions vulnerable to missed detections and incorrect part associations, and they regress metric 3D geometry that a single view fixes only up to scale. We present QueryArt, a model that estimates articulation parameters from a single RGB image, a 2D query point, and camera intrinsics. QueryArt is trained to estimate the 3D articulation geometry relative to the queried point and in units of its depth, which keeps its target identifiable from the image alone. A single depth measurement at the query point then supplies the scale and recovers the metric parameters. We train QueryArt on a curated mixture of synthetic and real-world articulation datasets. We evaluate QueryArt on several benchmarks and compare it against recent baselines. QueryArt outperforms recent baselines on most articulation metrics, including on out-of-distribution data. To demonstrate the model's capabilities in real-world settings, we evaluate QueryArt on a mobile manipulator across 57 manipulation trials spanning 16 object parts and five viewpoint classes, achieving a 70.2% success rate. We provide code and videos at: https://abwerby.github.io/queryart/

### 论文解读

#### 摘要翻译
机器人遇到从未见过的柜门、抽屉等可动部件时，需要在交互前知道应沿什么轴、以何种关节运动。QueryArt 从单张标定 RGB 图和一个操作查询点预测关节类型与运动几何；它把转轴位置表示为相对查询点、并按该点深度归一化的偏移，因此只需一次深度测量就能恢复米制参数。作者在多个数据集和移动机器人操作试验中检验了这种做法。

#### 方法动机分析
现有单图方案常把活动部件分割和关节估计绑在一起，定位错会连带产生错误运动模型；直接回归米制三维位置又超出了单目图像本身可辨识的范围。QueryArt 假设一个交互像素可由人或独立的可供性模型提供，并利用查询点的深度只补足全局尺度，从而把“在哪里操作”与“该点如何运动”分开。

#### 方法设计详解
输入图像由冻结的 DINOv3 ViT-B/16 编码，融合中层与末层特征后形成三尺度特征金字塔。查询点编码结合三种信息：其 Fourier 位置编码、该像素采样到的视觉特征，以及由相机内参构造的标定向量。三者投影后相加，作为 TYPE 与 LINE 两个任务 token 的条件。两层 decoder 先以可变形交叉注意力读取图像特征，再通过 token 自注意力和前馈层更新表示。类型头预测静止、转动或平移；两个独立方向头分别预测转动轴和移动方向。对转动部件，位置头预测查询点到轴线最近点的偏移，并将其正交投影到轴方向的法平面，保证几何一致。偏移按查询深度归一化，预测时再乘以实测深度，即可得到米制轴线；转动和平移方向采用独立预测头。可学习部分约 4.7M 参数。

#### 方法对比分析
与依赖活动部件分割的单图方法相比，查询点使模型无需先恢复完整部件掩码；与仅预测三维点轨迹的方法相比，显式关节类型和轴可直接生成结构一致的运动；与深度点云或多状态重建方法相比，它能在机器人尚未推动物体时提出运动假设。适合有清晰交互点、相机标定和查询点深度的家具操作，无法仅凭 RGB 解决绝对尺度。

#### 实验分析（精简版）
在四个留出数据集上，整体成功率为 60.12%–89.67%；在未参与训练的 HOI! 与 Arti4D 上分别为 69.08% 和 80.05%。整体成功要求关节类型正确、轴误差不超过 10°且诱导运动 RMSE 不超过 0.1 米。Spot 移动操作实验在 57 次计分试验中成功 40 次（70.2%），覆盖 16 个部件和五类视角；但抓取失败不计入该分数，极端视角的表现较弱，因此不能把结果等同于完整端到端成功率。

#### 实用指南
代码与视频由作者公开。训练冻结视觉主干，在一张 RTX 5090 上使用 AdamW、batch size 64、初始学习率 1e-4，最多训练 15 个 epoch，并按验证集早停。复现需准备查询点、相机内参和有效深度标签；迁移到新机器人时，应先校准查询点深度及抓取位置，再验证预测轨迹和安全边界。将语言模型产生的把手候选接入查询模块是可行的系统组合，但并非本文验证过的结果。

#### 总结
核心思想：用查询深度补尺度

1. 输入单张 RGB、交互像素与相机内参。
2. 融合图像金字塔和查询点特征，预测关节类型及轴。
3. 在深度归一化空间估计转轴相对位置。
4. 用查询点实测深度恢复米制几何并生成操作轨迹。

**Links:**

- [PDF](https://arxiv.org/pdf/2610.01726v1)
- [arXiv](https://arxiv.org/abs/2610.01726v1)

---

