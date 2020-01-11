time: 20200111
code_source: https://github.com/BoChenYS/BPnP
pdf_source: https://arxiv.org/pdf/1909.06043v2.pdf
short_title: backprob through PnP Optimization

# End-to-End Learnable Geometric Vision by Backpropagating PnP Optimization

这篇论文的核心贡献是将PnP 优化过程变为一个可导的部件插入到网络之中，对于更多端到端的模型有很好的帮助。

## 隐函数定理

[Implicit function theorem](https://www.wikiwand.com/en/Implicit_function_theorem) 

对于$f(a^*, b^*) = 0, 且 b^* = g(a^*)$

若其中的雅克比矩阵可导，有
$$
\frac{\partial g}{\partial x_{j}}(\mathbf{x})=-\left[J_{f, \mathbf{y}}(\mathbf{x}, g(\mathbf{x}))\right]_{m \times m}^{-1}\left[\frac{\partial f}{\partial x_{j}}(\mathbf{x}, g(\mathbf{x}))\right]_{m \times 1}
$$

PnP问题，作者选择的描述是，寻找6 DoF位姿，使得3D点在2D平面上的投影的对应点个点在图片中的误差平方和最小，

这里作者选择$f$为误差函数关于输出的导数$\frac{\partial f}{\partial y}$，在pnp优化好后，其中这个导数应为零。

具体计算见代码与论文

其中forward函数opencv完成，backward函数详见代码。整个函数可以作为一个即插即用的部件。且作者有训练的一个sample脚本。

作者在paper第五章节给出了一个object pose estimation的例子。