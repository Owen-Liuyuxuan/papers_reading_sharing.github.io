time: 20200102
pdf_source: http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf
code_source: https://github.com/cvxgrp/cvxpylayers
short_title: More on Differentiable Convex Optimization

# Some Papers on Differentiable Convex Optimization
在这个页面计划记录两篇关于可微分凸优化的算法的记录.本文此处会大致记录三篇文章告诉我的一些新的概念,分别是一篇关于凸优化以及Disciplined convex programming的综述[(1)],一篇关于凸优化的导数的paper[(2)],一篇介绍[cvxlayers]的paper[(3)].它们都来自于Stanford Boyd大佬的组。

## Convex optimization and Disciplined Convex Programming

一般凸优化问题定义：
$$
\begin{aligned}
\text { minimize } & f_{0}(x, \theta) \\
& f(x, \theta) \preceq 0 \\
& h(x, \theta)=0
\end{aligned}
$$

综述[(1)]对凸优化问题的特殊情况(线性最优，二次最优，最小二乘)进行分类，同时指出了专用问题专用算法以及非线性凸优化普遍方法之间的关系，指出了非线性凸优化一般方法比如内点法在效率上往往会和线性或二次凸优化算法相近，原因是都利用了局部的convexity.

Disciplined Convex Programming包含两个内容，一个是将符合要求的凸优化问题通过一系列原子操作并转换为标准问题，另一个是一套用于判断用户输入的问题是否为符合要求的DCP的规则系统。在本文作者详细描述了DCP规则系统的算法，并提供了决策树例子。在[cvxlayers]的paper[(3)]中以及代码文档中,我们可以知晓目前库中支持的原子操作(绝大多数的单调函数以及多项式函数)


## 基于KKT condition的 Convex optimization 导数传递


这里采用的思路很接近与[OptNet],paper[(2)]将这个问题的解答拓展到近乎任意凸优化问题，在接近最优的情况下，使用KKT condition进行反传。

KKT条件:
$$
\begin{aligned}
f(\tilde{x}, \theta) & \preceq 0 \\
h(\tilde{x}, \theta) &=0 \\
\tilde{\lambda}_{i} & \geq 0, \quad i=1, \ldots, m \\
\tilde{\lambda}_{i} f_{i}(\tilde{x}, \theta) &=0, \quad i=1, \ldots, m \\
\nabla_{x} L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta) &=0
\end{aligned}
$$

其中$\mathcal{L}$为原函数加上拉格朗日乘子，为:
$$
L(x, \lambda, \nu, \theta)=f_{0}(x, \theta)+\lambda^{T} f(x, \theta)+\nu^{T} h(x, \theta)
$$

再定义
$$
g(z, \theta)=\left[\begin{array}{c}
{\nabla_{x} L(x, \lambda, \nu, \theta)} \\
{\operatorname{diag}(\lambda) f(x, \theta)} \\
{h(x, \theta)}
\end{array}\right]
$$
作者最终给出的导数公式
$$
\mathrm{D}_{z} g(\tilde{z}, \theta)=\left[\begin{array}{ccc}
{\mathrm{D}_{x} \nabla_{x} L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta)} & {\mathrm{D}_{x} f(\tilde{x}, \theta)^{T}} & {\mathrm{D}_{x} h(\tilde{x}, \theta)^{T}} \\
{\operatorname{diag}(\tilde{\lambda}) \mathrm{D}_{x} f(\tilde{x}, \theta)} & {\operatorname{diag}(f(\tilde{x}, \theta))} & {0} \\
{\mathrm{D}_{x} h(\tilde{x}, \theta)} & {0} & {0}
\end{array}\right]
$$

$$
\mathrm{D}_{\theta} g(\tilde{z}, \theta)=\left[\begin{array}{c}
{\mathrm{D}_{\theta} \nabla_{x} L(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta)} \\
{\operatorname{diag}(\tilde{\lambda}) \mathrm{D}_{\theta} f(\tilde{x}, \theta)} \\
{\mathrm{D}_{\theta} h(\tilde{x}, \theta)}
\end{array}\right]
$$

$$
\mathrm{D}_{\theta} s(\theta)=-\mathrm{D}_{z} g(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta)^{-1} \mathrm{D}_{\theta} g(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta) \text { for every } \theta \in Q
$$

考虑二次规划作为本文的特殊情况，带入公式有

$$
\mathrm{D}_{x} g(\tilde{x}, \tilde{\lambda}, \tilde{\nu}, \theta)=\left[\begin{array}{ccc}
{Q} & {G^{T}} & {A^{T}} \\
{\operatorname{diag}(\tilde{\lambda}) G} & {\operatorname{diag}(G \tilde{x}-h)} & {0} \\
{A} & {0} & {0}
\end{array}\right]
$$
$$
\mathrm{D}_{\theta} g(\dot{x}, \tilde{\lambda}, \tilde{\nu}, \theta)=\left[\begin{array}{c}
{\mathrm{d} Q \tilde{x}+\mathrm{D}_{\theta} q+\mathrm{d} G^{T} \tilde{\lambda}+\mathrm{d} A^{T} \tilde{\nu}} \\
{\operatorname{diag}(\lambda)\left(\mathrm{d} G \tilde{x}-\mathrm{D}_{\theta} h\right)} \\
{\mathrm{d} A \tilde{x}-\mathrm{D}_{\theta} b}
\end{array}\right]
$$
最终的公式与[OptNet]是一致的


## Convex Optimization as a Differentiable learnable layer

在2019 NeurIPS的paper[(3)]中, 作者将初始解$x_0$,最优化问题参数$\theta$,最终输出$s$理解为神经网络层的输入、参数以及输出。进一步地，作者糅合前两个章节的idea,使用 Disciplined Convex Programming使得用户在设计优化层的时候有更高的自由度(不再只局限于一次或二次型等特性)，输入与参数在形成成本函数与约束的计算过程中允许按照DCP规则链接大量的原子操作，形成更为复杂而自然的"一般"优化问题输入(注:[cvxpylayers]虽然支持很多别的函数，但是不支持正余弦函数)。使用KKT当优化结果收敛时快速地实现反向传播。

作者在[(3)]的附录中给出了使用cvxpylayers用最优化问题实现ReLU,Softmax，QP等问题的代码。[cvxpylayers]的代码同样也有很多的例程,在RL、控制、网络后优化领域有较强的使用空间。


[OptNet]:./OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.md
[(1)]:https://web.stanford.edu/~boyd/papers/pdf/disc_cvx_prog.pdf
[(2)]:https://arxiv.org/pdf/1804.05098.pdf
[(3)]:http://web.stanford.edu/~boyd/papers/pdf/diff_cvxpy.pdf
[cvxpylayers]:https://github.com/cvxgrp/cvxpylayers