time: 20201210
pdf_source: https://openreview.net/pdf?id=YLewtnvKgR7

# Estimating and Evaluating Regression Predictive Uncertainty in Deep Object Detectors 

这篇paper是 ICLR 2021 under-review的一篇paper，分数为 9666. 在讨论不确定性上有一定的价值.


对于预测不确定性，传统的方法是选择让网络预测分支优化一个负log likelihood函数

$$
\mathrm{NLL}=\frac{1}{2 N} \sum_{n=1}^{N}\left(\boldsymbol{z}_{n}-\boldsymbol{\mu}\left(\boldsymbol{x}_{n}, \boldsymbol{\theta}\right)\right)^{\top} \boldsymbol{\Sigma}\left(\boldsymbol{x}_{n}, \boldsymbol{\theta}\right)^{-1}\left(\boldsymbol{z}_{n}-\boldsymbol{\mu}\left(\boldsymbol{x}_{n}, \boldsymbol{\theta}\right)\right)+\log \operatorname{det} \boldsymbol{\Sigma}\left(\boldsymbol{x}_{n}, \boldsymbol{\theta}\right)
$$

作者指出的问题在于这个损失函数会使得预测结果过于不自信，从损失函数上来看，尽管最优值是合理的，但是对高置信度的惩罚很大。

本文提出一个 Energy Score:

$$
\mathrm{ES}=\frac{1}{N} \sum_{n=1}^{N}\left(\frac{1}{M} \sum_{i=1}^{M}\left\|\mathbf{z}_{n, i}-\boldsymbol{z}_{n}\right\|-\frac{1}{2(M-1)} \sum_{i=1}^{M-1}\left\|\mathbf{z}_{n, i}-\mathbf{z}_{n, i+1}\right\|\right)
$$

可以计算发现这个基于采样的损失函数相对而言更鼓励高自信度的结果，而且正负结果的惩罚更均匀。


