time: 20191110
pdf_source: https://arxiv.org/pdf/1502.05477.pdf
code_source: https://github.com/pat-coady/trpo
# Trust Region Policy Optimization

这篇论文是经典的强化学习论文，直觉的motivation来说就是在强化学习训练的过程中噪音很大，很容易崩溃，所以这里会尝试限制一个Trust Region，模仿相关优化问题的思路进行控制。其实本文后续有一篇implementation更intuitive的[PPO](https://arxiv.org/abs/1707.06347).

本文在Spinningup上有很优秀的[官方介绍](https://spinningup.openai.com/en/latest/algorithms/trpo.html)，本页主要通过中文翻译+结合论文与其他资料进行补充。

本文有对应的pytorch开源代码,[链接](https://github.com/ikostrikov/pytorch-trpo)

## 数学上的Motivation

为什么强化学习在训练过程中噪音这么严重？数学上可以归纳为以下的公式说明。对于当前以及最优的policy之间的expected cost的差别，准确的公式是:

$$
\eta(\tilde\pi) = \eta(\pi) + \sum_s\rho_{\tilde\pi}(s)\sum_s\tilde\pi(a|s)A_\pi(s,a)
$$

但是由于没有办法用optimal policy进行sample.所以一般实际操作的时候只能用当前的policy对路径进行采样，得到

$$
L_\pi(\tilde\pi) = \eta(\pi) + \sum_s\rho_{\pi}(s)\sum_s\tilde\pi(a|s)A_\pi(s,a)
$$

## 算法数学描述

优化问题描述:

$$
\begin{aligned}
    \theta_{k+1} = &\argmax_\theta \mathcal{L}(\theta_k,\theta) \\
    &s.t. D_{KL}(\theta||\theta_k) \le \delta
\end{aligned}
$$

其中$\mathcal{L}(\theta_k,\theta)$含义与前文基本一致，具体来说是
$$
    \mathcal{L}(\theta_k,\theta) = E_{s,a~\pi_{\theta_k}}[\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}A^{\pi_{\theta_k}}(s,a)]
$$
$$
D_{KL}(\theta||\theta_k) = E_{s~\pi_{\theta_k}}[D_{KL}(\pi_\theta(·|s) ||\pi_{\theta_k}(·|s)  )]
$$
其中KL divergence的定义请查阅[wiki](https://www.wikiwand.com/en/Kullback%E2%80%93Leibler_divergence)
$$
    D_{KL}(P||Q) = \sum_P P(x) ln(\frac{P(x)}{Q(x)})
$$

对上面的问题取泰勒近似，优化问题变为一个带约束的二次方程，设$g$为$L$关于网络参数$\theta$的梯度.

$$
\begin{aligned}
\theta_{k+1} = \argmax_\theta &g^T(\theta - \theta_k) \\
s.t. &\frac{1}{2}(\theta - \theta_k)^TH(\theta - \theta_k) \le\delta
\end{aligned}
$$

根据凸优化的最优解是

$$
    \theta_{k+1} = \theta_k + \sqrt{\frac{2\delta}{g^TH^{-1}g}}H^{-1}g
$$

对于参数数量很大的$\theta$网络，hessian矩阵的计算，尤其是逆的矩阵运算量很大，这里使用的是[conjugate gradient算法](https://www.wikiwand.com/en/Conjugate_gradient_method)，这是一个求解对称矩阵的逆或者求解稀疏线性方程的迭代算法。
下文来自wiki链接中的是求解线性方程$Ax = b$的算法
![image](res/ConjugateGradient.svg)

最后openai spinningup总结了算法如下图

![image](res/TRPO.svg)

### 补充描述：

关于step 9 line search:
就是逐个查看选择实际能取的最长的步长(因为前面有做二次近似)

关于step 10:
常规policy gradient 算法中对value网络的拟合.