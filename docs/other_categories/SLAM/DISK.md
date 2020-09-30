time: 20200930
pdf_source: https://arxiv.org/pdf/2006.13566.pdf

# DISK: Learning local features with policy gradient

这篇paper引用强化学习进行关键点的检测。是2020NIPS paper.

动机上来说直接优化成功的匹配点数量。

## 流程

### 特征提取 $P(F_I | I, \theta_F)$

使用UNet从图片$I$提取输出heatmap以及特征矢量(128维)。这篇paper模仿 [superPoint](SuperPoint:Self-Supervised_Interest_Point_Detection_and_Description.md)的做法，将图片分为由$h\times h$方块组成的网格，然后每一个网格中最多只会有一个keypoint.

而每一个网格内得分最高的点是否为keypoint除了softmax的相对score外还有一个sigmoid的绝对score。$P(p|K^u) = softmax(K^u)_p \cdot \sigma(K_p^u)$

图片的特征集$F_I = \{(p_1, D(p_1)), (p_2, D(p_2)), ...\}$

### 匹配分布 $P(M_{A\leftrightarrow B} | F_A, F_B, \theta_M)$

已知两个特征的特征集，作者计算每两个descriptor之间的$l_2$距离。从而得到距离矩阵$d$.在训练的时候，作者根据$d$的行或列的softmax值来计算从$A \rightarrow B$ 或者 $B \rightarrow A$的匹配概率。

作者指出为了避免学出难以找出一一对应的特征点,一般采用循环一致匹配(Cycle-consistent matching)以及比率测试(ratio test)两个技巧。

- 循环一致匹配要求两个keypoint必须是相互间都是nearest neighbours才能算是一个一个完整的match
- 比率测试是SIFT开始引入的技巧，对于一个keypoint，它与最近点的距离必须与次近点的特征距离大一定的倍率，才能算是正确的match.

但是这两个技巧都是不可导的。作者提出的方案是放松循环一直匹配的要求。这里采取的方案是只要两个keypoint在softmax中互相采样到就算是consistent.

而两个特征成功匹配上的概率可以计算是 $P(i \leftrightarrow j) = P_{A\rightarrow B}(i | d, j) \cdot P_{A \leftarrow B} (j|d,i)$. 因而采样本身不影响梯度的计算.

### Reward $R(M_{A \leftrightarrow B})$

对于正确匹配的点与不正确的点分别给一定量的reward。

如果两个点是有深度标注且重投影距离足够小，则记录为正确的匹配。如果两个点没有深度标注，且距离不大，则不计算reward，其他点记录为不正确的匹配点。

### 梯度

这里采用policy gradient的算法.

目标: 计算$\nabla_{\theta} \underset{M_{A} \rightarrow B}{\mathbb{E}} R\left(M_{A \leftrightarrow B}\right)$

1. 期望表达: $\underset{i,j}{\sum}[P(i\leftrightarrow j|F_A, F_B, \theta_M) \cdot r(i \leftrightarrow j)]$
2. Log-Gradient Trick: $\nabla_\theta P = P\cdot \nabla{\log{P}}$
3. Factorizing $\nabla_\theta P$: $\nabla_\theta P = \nabla_{\theta_M}\log{P(i\leftrightarrow j | F_A, F_B, \theta_M)} + \nabla_{\theta_F}\log{P(F_{A,i} | A, \theta_F) + \nabla_{\theta_F}\log{P(F_{B,j} | B, \theta_F)}}$。这个等式讲的就是$i,j$两个点成功匹配的概率是由距离匹配以及两张图各自的分类结果累乘而得的。

由此，仅仅通过匹配的reward，就可以使整个网络端到端训练。
