pdf_source: https://bmvc2019.org/wp-content/uploads/papers/0865-paper.pdf
code_source: https://github.com/MarvinTeichmann/ConvCRF
short_title: Convolutional CRF
# Convolutional CRFs for Semantic Segmentation

这篇论文提出使用卷积版的Conditional Random Field(CRF)用于优化语义分割的结果。首先回顾(对写review的我此时是新学的) FullCRF的算法，然后提出了ConvCRF的算法以及implementation,[作者代码已开源](https://github.com/MarvinTeichmann/ConvCRF)

## FullCRF

CRF的原意在于让特征相似的点输出相似的值，最后转换为优化一下这个条件概率:
$$
E(\hat{x} | I)=\sum_{i \leq N} \psi_{u}\left(\hat{x}_{i} | I\right)+\sum_{i \neq j \leq N} \psi_{p}\left(\hat{x}_{i}, \hat{x}_{j} | I\right)
$$

第一项为基础全语义分割网络输出的值，第二项体现图片中不同位置的相互影响。FullCRF中，第二项计算公式为
$$
\psi_{p}\left(x_{i}, x_{j} | I\right) :=\mu\left(x_{i}, x_{j}\right) \sum_{m=1}^{M} w^{(m)} k_{G}^{(m)}\left(f_{i}^{I}, f_{j}^{I}\right)
$$
其中$\mu(x_i,x_j) = |x_i \neq x_j|$也就是图中每一个点(除了点i之外).常用的核函数$k$有如以下的高斯函数

$$
k\left(f_{i}^{I}, f_{j}^{I}\right) :=w^{(1)} \exp \left(-\frac{\left|p_{i}-p_{j}\right|^{2}}{2 \theta_{\alpha}^{2}}-\frac{\left|I_{i}-I_{j}\right|^{2}}{2 \theta_{\beta}^{2}}\right)+w^{(2)} \exp \left(-\frac{\left|p_{i}-p_{j}\right|^{2}}{2 \theta_{\gamma}^{2}}\right)
$$

其中$w^{(1)},\theta$等是仅有的可学习参数，直觉而言，就是特征相似者相互影响大，距离近者相互影响大。

最终实现的迭代算法:

![image](res/CRFInference.png)

## ConvCRF

ConvCRF先假设两个曼哈顿距离大于一定阈值$k$的点相互独立，这个$k$称为ConvCRF的filter size.这也就是ConvCRF对前文$\mu(x_i,x_j)$的预设方式

对于位于$x,y$的点它对应的卷积核/CRF核为
$$
k_{g}[b, d x, d y, x, y] :=\exp \left(-\sum_{i=1}^{d} \frac{\left|f_{i}^{(d)}[b, x, y]-f_{i}^{(d)}[b, x-d x, y-d y]\right|^{2}}{2 \dot{\theta}_{i}^{2}}\right)
$$

其中$\theta_i$为可学习的变量$f_i$为特征向量,卷积范围内的每一个pair有一个对应的,$K=\sum^s_{i=1}w_i g_i$结果Q,combined message则由此式子给出

$$
Q[b, c, x, y]=\sum_{d x, d y \leq k} K[b, d x, d y, x, y] \cdot P[b, c, x+d x, y+d y]
$$

作者提到，这个运算操作可以说类似于locally connected layers(every pixel has its own filter),区别在于每一个kernel在channel方向上是一个常数(只负责加权求和整个feature vector而不需要重整feature)。

(题外话，locally connected layer目前有keras implementation但是还没有officail pytorch implementation，[参考](https://discuss.pytorch.org/t/locally-connected-layers/26979/7))
