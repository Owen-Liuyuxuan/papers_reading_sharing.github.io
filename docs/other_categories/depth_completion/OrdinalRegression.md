time: 20200626
pdf_source: http://openaccess.thecvf.com/content_CVPR_2019/papers/Diaz_Soft_Labels_for_Ordinal_Regression_CVPR_2019_paper.pdf

# Soft Labels for Ordinal Regression

与[DORN](dorn.md)一样。采取序数回归的概念对深度进行回归。序数回归的概念在于被回归的值有一定的自然逻辑顺序。


## SORD

序数回归的target:

$$y_{i}=\frac{e^{-\phi\left(r_{t}, r_{i}\right)}}{\sum_{k=1}^{K} e^{-\phi\left(r_{t}, r_{k}\right)}} \quad \forall r_{i} \in \mathcal{Y}$$

类似于交叉熵损失函数的反传:

$$\frac{\partial L}{\partial p_{i}}=-\frac{e^{-\phi\left(r_{t}, r_{i}\right)}}{e^{o_{i}^{\prime}}}=-e^{-\phi\left(r_{t}, r_{i}\right)-o_{i}^{\prime}}$$

对于深度预测，文中提到了三种核函数$\phi$

Square Difference(SD): $\phi\left(r_{t}, r_{i}\right)=\left\|r_{t}-r_{i}\right\|^{2}$

Square Log Difference(SL): $\phi\left(r_{t}, r_{i}\right)=\left\|\log r_{t}-\log r_{i}\right\|^{2}$

Square Invariant Logarithmic Error (SI): $\phi\left(r_{t}, r_{i}\right)=d_{r_{t}, r_{i}}^{2}-\frac{d_{r_{t}, r_{i}}}{n}\left(d_{r_{t}, r_{i}}+\sum_{p^{\prime} \neq p} d_{p^{\prime}}\right)$

SI Loss 源自于这篇 [paper.pdf](https://arxiv.org/pdf/1406.2283.pdf)

$$\begin{aligned}
D\left(y, y^{*}\right) &=\frac{1}{n^{2}} \sum_{i, j}\left(\left(\log y_{i}-\log y_{j}\right)-\left(\log y_{i}^{*}-\log y_{j}^{*}\right)\right)^{2} \\
&=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{n^{2}} \sum_{i, j} d_{i} d_{j}=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{1}{n^{2}}\left(\sum_{i} d_{i}\right)^{2}
\end{aligned}$$

那篇paper提出的一个混合的Loss:

$$L\left(y, y^{*}\right)=\frac{1}{n} \sum_{i} d_{i}^{2}-\frac{\lambda}{n^{2}}\left(\sum_{i} d_{i}\right)^{2}$$