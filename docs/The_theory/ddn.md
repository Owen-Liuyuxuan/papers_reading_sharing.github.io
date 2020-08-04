time: 20200804
pdf_source: https://arxiv.org/pdf/1909.04866.pdf
code_source: https://github.com/anucvml/ddn
short_title: Deep Declarative Networks
# Deep Declarative Networks: A New Hope

这个概念类似于深度均衡paper里面的 [implicit deep learning](../Building_Blocks/MDEQ.md)，也就是网络模块的定义的是由对输出结果的定义来决定的。

[MDEQ](../Building_Blocks/MDEQ.md)考虑的是实现一个无穷深的网络，更多情况下这类网络层使用的是优化问题，如[EPnP](../Building_Blocks/Bpnp.md), [OptNet](../other_categories/others/OptNet_Differentiable_Optimization_as_a_Layer_in_Neural_Networks.md), [SS3D](../3dDetection/Monocular_3D_Object_Detection_and_Box_Fitting_Trained_End-to-End_Using_Intersection-over-Union_Loss.md)

这篇paper则给出一类基于优化的 declarative networks 的求导训练方法。

以无约束的优化问题为例，输入为$x$, 输出为$y$, 
$$
\begin{array}{ll}
\operatorname{minimize} & J(x, y) \\
\text { subject to } & y \in \arg \min _{u \in C} f(x, u)
\end{array}
$$

在最优的点上，$\frac{\partial f}{\partial y} = 0$, 且这个不由$x$变化，再次求导可得

$$
\begin{aligned}
0_{m \times n} &=\mathrm{D}\left(\mathrm{D}_{Y} f(x, y)\right)^{\top} \\
&=\mathrm{D}_{X Y}^{2} f(x, y)+\mathrm{D}_{Y Y}^{2} f(x, y) \mathrm{D} y(x)
\end{aligned}
$$

$$\mathrm{D} y(x)=-\left(\mathrm{D}_{Y Y}^{2} f(x, y)\right)^{-1} \mathrm{D}_{X Y}^{2} f(x, y)$$

对于带等式与不等式的约束，本文分别给出了求导的计算方式。


## 代码

本文开源了前文所有类型的节点的求导方法。代码上只要实现主函数，约束函数以及优化过程(与反向无关)即可。

其次在本文的代码中看到了pytorch自带的一个完整的优化函数 [torch.optim.LBFGS](https://pytorch.org/docs/stable/optim.html#torch.optim.LBFGS)。本文利用这个函数实现了一个很通用的PnP非线性优化