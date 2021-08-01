time: 20210730
pdf_source: https://arxiv.org/abs/1901.02731
code_source: https://github.com/kumar-shridhar/PyTorch-BayesianCNN
short_title: Bayesian Neural Network

# A comprehensive guide to bayesian convolutional neural network with variational inference

这篇文章是整合了一个阶段的BNN网络的研究，并用在了CNN上。

[相关的tutorial](https://arxiv.org/abs/2007.06823)

BNN的基本方法论基本上是:
- 训练的时候，需要得到网络权重的分布而不是一个点估计, 目前主流的实际操作是将每一个权重值理解为一个独立的高斯分布
- 推理的时候，可以让网络只取mean值，只输出一个点估计。也可以对全网络多次采样，用蒙特卡洛的方式输出多个点估计，可以使用bagging融合，也可以估计不确定性。
- 权重的高斯分布，在训练时以re-parametrization的形式前传并训练，对于全连接层以及卷积层，有local-reparametrization的形式前传，可以提高效率.

## 理论

已知训练集$D$, 测试集数据$x_{test}$, 目标是输出$y_{test}$的概率分布。 描述为$p(y_{test}|x_{test}, D)$

用全概率公式关于权重展开。

$$p(y_{test} | x_{test}, D) = \int_{\theta} p(y_{test}| x_{test}, \theta') p(\theta'|D) d \theta'$$

$p(\theta'|D)$ 可以理解为各组点估计的网络权重在训练数据集中的likelihood.

这个likelihood可以进一步用贝叶斯公式拆解:

$$
p(\theta | D) = \frac{p(D_y | D_x, \theta) p(\theta)}{\int_\theta p(D_y | D_x, \theta') p(\theta') d \theta'} \propto p(D_y | D_x, \theta) p(\theta)
$$

分母是归一化，分子第一项是训练集上的likelihood,实现上常常可以理解为score或者loss。 第二项是先验分布，可以理解为regularization.


