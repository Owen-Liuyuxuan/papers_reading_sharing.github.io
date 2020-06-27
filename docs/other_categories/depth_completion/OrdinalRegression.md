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