time: 20210101
pdf: https://arxiv.org/pdf/2012.11140v1.pdf
# LQF: Linear Quadratic Fine-Tuning

这篇paper提出一个使用神经网络的线性近似模型进行fine-tunning的思路, 使得fine-tunnning的过程一定能有"最优解"，且使得各个超参数的影响可以解释.

## 模型的线性化

$$f^{lin}_w(x) = f_{w_0}(x) + \nabla_w f_{w_0}(x) \cdot (w-w_0)$$


将分类问题改为一个回归问题，即回归一个one-hot vector. 在这个条件下，可以写出权重的闭式解:

$$
    w^* = (J^TJ + \lambda I)^{-1} J (Y - f_0(X))s
$$

由于矩阵大小很大，因而我们不能直接计算这个最优解的，所以还是需要使用随机梯度下降处理。

但是我们可以从这个公式中直观的体现出每一个训练样本以及超参数在Fine-tunning过程中的contribution.

## pre-conditioning

由上式，整个问题变成了一个凸优化问题，使用SGD我们能确保系统最终能收敛到全局最优，但是由于曲率的变化，收敛速度是不稳定的。这个收敛速度取决于hessian 矩阵的大小特征值的比例。

使用高斯牛顿算法，直接考虑二阶信息能大大提升收敛速度，高斯牛顿算法的公式

$$
    w_{t+1} = w_t - H^{-1}(X) G(X)
$$

本文使用"pre-conditioning"这个术语，技术上使用 Fisher Information Matrix 近似二阶导。

