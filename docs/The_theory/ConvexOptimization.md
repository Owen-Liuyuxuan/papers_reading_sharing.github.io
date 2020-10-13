# Convex Optimization

[boyd's book](https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

本文主要收集在凸优化课程中记下的基础概念与结论。

## 凸函数与凸集

### 基本定义与重要判定
- 凸函数要求函数上，任意两点间的函数值的线性插值要不小于线性插值的函数值. 
 $$
 \forall \theta\in[0,1] \quad f(\theta x + (1-\theta)y) \le \theta f(x) + (1-\theta) f(y)
 $$
- 凸集要求集合内两点连线上的任意点都在该集合内

凸函数的一个充分条件是二次导不小于零，严格来说是对各变量的Hessian Matrix为半正定.

### 常见而重要的凹凸函数函数

- 线性函数同时为凹、凸函数
- 所有的 norm (满足三角形不等式)都是凸函数
- 二次函数 $f(x) = x^TPx + 2q^Tx + r$, if $P \ge 0$ 是凸函数.
- 几何平均 $f(x) = (\Pi^n_{i=1} x_i)^{1/n}$ 是凹函数
- log-sum-exp $f(x) = \log{\sum_i e^{x_i}}$为凸函数
- quadratic over linear $f(x,y) = x^2 / y$为凸函数
- log-det $f(X) = logdet(X)$为凹函数 (证明比较取巧，取定义域内任意两点，证明直线上插值比例为变量的函数都为凹函数，则整个函数为凹函数)
- 矩阵的最大特征值 $f(X) = \lambda_{max}(X) = \underset{||y||_2=1}{max}y^TXy$为凸函数 (由最大值性质)
  
### 保留凸特性的常见操作

- 凸函数的非负权重求和
- 与线性函数的嵌套 $f(Ax + b)$
- 多个凸函数的point-wise maximum.
- 凸函数中部分变量的maximum $g(x) = \underset{y\in \mathcal{A}}{sup}f(x, y)$

## 凸优化问题

$$
\begin{array}{lll}
\underset{x}{\operatorname{minimize}} & f_{0}(x) \\
\text { subject to } & f_{i}(x) \leq 0 & i=1, \ldots, m \\
& h_{i}(x)=0 & i=1, \ldots, p
\end{array}
$$

要求:

- $f_0(x)$为凸函数
- $f_i(x)$为凸函数
- $h_i(x)$为线性函数，也就是只允许$Ax = b$

主要性质:

    凸优化的局部最优等于全局最优，且唯一.

### 将 $L_1$, $L_{\infty}$转换为凸优化问题

$L_{\infty}$问题可以如此转换:
$$
\begin{array}{ll}
\underset{x}{\operatorname{minimize}} & \|x\|_{\infty} \\
\text { subject to } & G x \leq h \\
& A x=b
\end{array}
$$
变成
$$
\begin{array}{ll}
\underset{t, x}{\operatorname{minimize}} & t \\
\text { subject to } & -t \mathbf{1} \leq x \leq t \mathbf{1} \\
& G x \leq h \\
& A x=b
\end{array}
$$

$L_1$:
$$
\begin{array}{ll}
\underset{x}{\operatorname{minimize}} & \|x\|_{1} \\
\text { subject to } & G x \leq h \\
& A x=b
\end{array}
$$

$$
\begin{array}{ll}
\underset{t, x}{\operatorname{minimize}} & \sum_{i} t_{i} \\
\text { subject to } & -t \leq x \leq t \\
& G x \leq h \\
& A x=b
\end{array}
$$

## 对偶与KKT条件

对于一般最优化问题:
$$
\begin{array}{lll}
\underset{x}{\operatorname{minimize}} & f_{0}(x) \\
\text { subject to } & f_{i}(x) \leq 0 & i=1, \ldots, m \\
& h_{i}(x)=0 & i=1, \ldots, p
\end{array}
$$

Lagrangian为:

$$L(x, \lambda, v) = f_0(x) + \sum_i \lambda_i f_i(x) + \sum_i v_ih_i(x)$$

### 对偶函数

$$g(\lambda, v) = \underset{x\in D}{inf} L(x, \lambda, v)$$

无论原来的问题是否是凸优化问题，对偶函数$g$一定为凹函数 (线性函数的inf下界组合)，因而最大化$g$经常是一个凸优化问题.

同时$g(\lambda, v) \le p^*$, 对偶函数可以表征原最优化问题的优化下界。

对偶问题就是通过计算$g$的最大值(同时有约束 $\lambda \ge 0$)，来分析原问题的最优下界。

### 强弱对偶性

对偶问题得到的最优解 $d^* \le p^*$. 对于强对偶问题，则有$d^* = p^*$

最简单常用的的判断原问题为强对偶的条件:

- 原问题是凸优化问题
- 且将不等式$\le$约束改为更强的$<$约束，仍存在可行域.

线性不等式约束不需要满足第二条条件。这两点在大多数现实凸优化问题中都成立，大多数的凸优化问题都是强对偶的。

### KKT 条件

- 主可行性, $f_i(x) \le 0$, $h_i(x) = 0$
- 对偶可行性, $\lambda \ge 0$
- Complementary slackness $\lambda_i^*f_i(x^*) = 0$
- 拉格朗日函数对每一个变量的导数(或者说梯度矢量)为零:

$$\nabla f_0(x) + \sum\lambda_i \nabla f_i(x) + \sum v_i \nabla h_i(x) = 0$$

主要性质:

- KKT条件是任何一个可导的最优化问题的必要条件，也即是说最优解一定满足KKT条件
- KKT条件对凸优化问题来说是最优解的充要条件，也即是说可以用KKT条件求出来的解也一定是全局最优解。

计算与解题:

利用KKT条件直接求解的时候经常会遇到分类讨论，主要问题在于$\lambda_i$是否为零，物理意义上来说就是解是在边界上还是可行域内。可以用这一个反推回各种情况下对系统参数的要求。

## CVX库

python中可以使用cvxpy库对凸优化问题进行建模与求解。理论上来说cvxpy库可以接受比较复杂的问题描述，然后库会尝试将问题转换为标准凸优化问题并调用求解器求解. 老师的建议是用来快速验证一个问题是不是凸优化问题.

## 最优化算法

以最小化为例子

### 牛顿法

承接梯度下降法, 牛顿法通过二阶导辅助确定步长:

$$
\Delta \mathbf{x}_{\mathrm{nt}}=-\nabla^{2} f(\mathbf{x})^{-1} \nabla f(\mathbf{x})
$$

### 内点法

对于不等式约束，可以理解为阶跃到无穷的一个损失项，使用Log惩罚函数软化此约束:

$$
\underset{\mathbf{x}}{\operatorname{minimize}} \quad f_{0}(\mathbf{x})-(1 / t) \sum_{i=1}^{m} \log \left(-f_{i}(\mathbf{x})\right)
$$

- 当t很小的时候，最优化几乎只优化不等式约束。
- 当t很大的时候，最优化几乎只优化原目标函数，但是在边界附近接近于阶跃。

直接来说，当t很大的时候，函数不好优化，因而需要迭代优化.

![image](res/cvx_ipt_barrier.png)

作业一道题就是要求使用barrier method 处理Lasso regression.

### Block Coordinate Descent (BCD)

$$
\mathbf{x}_{i}^{k+1}=\arg \min _{\mathbf{x}_{i} \in \mathcal{X}_{i}} f\left(\mathbf{x}_{1}^{k+1}, \ldots, \mathbf{x}_{i-1}^{k+1}, \mathbf{x}_{i}, \mathbf{x}_{i+1}^{k} \ldots, \mathbf{x}_{N+1}^{k}\right)
$$

固定一部分解，每一步只对其中一部分解进行最优化迭代。


## Geometric programming

概念定义

- monomial: $c x_1^{a_1}x_2^{a_2}...$, 要求 $c > 0$
- Posynomial: $\sum c_k x_1^{a_{1k}}x_2^{a_{2k}}...$

GP 定义:
$$
\begin{array}{lll}
\underset{x}{\operatorname{minimize}} & f_{0}(x) \\
\text { subject to } & f_{i}(x) \leq 1 & i=1, \ldots, m \\
& h_{i}(x)=1 & i=1, \ldots, p
\end{array}
$$
其中 $f_i$ 为posynomials, $g_i$为 monomials.

问题来源:

- 最大化容器体积
- 约束所有墙(地面与侧边墙体)的表面积
- 约束长宽比,高宽比等
$$
\begin{array}{ll}
\text { maximize } & h w d \\
\text { subject to } & 2(h w+h d) \leq A_{\text {wall }}, \quad w d \leq A_{\text {floor }} \\
& \alpha \leq h / w \leq \beta, \quad \gamma \leq w / d \leq \delta
\end{array}
$$

求解方案:

将所有变量$x$ 用 $e^{\tilde{x}}$替换, 对目标函数，约束取$log$

$$
\begin{array}{lll}
\underset{x}{\operatorname{minimize}} & \log{ f_{0}(e^{\tilde{x}})} \\
\text { subject to } & \text{log}f_{i}(e^{\tilde{x}}) \leq 0 & i=1, \ldots, m \\
& \text{log} h_{i}(e^{\tilde{x}})=0 & i=1, \ldots, p
\end{array}
$$

目标函数都变为 log-exp-sum 的形式，因而为凸函数.

## Majorization-Minimization Algorithm (MM Algo.)

对于通用优化问题

$$
\begin{array}{lll}
\underset{x}{\operatorname{minimize}} & f_{0}(x) \\
\text { subject to } & f_{i}(x) \leq 0 & i=1, \ldots, m \\
& h_{i}(x)=0 & i=1, \ldots, p
\end{array}
$$

迭代的优化一个surrogate function.

$$x^{k+1} = \underset{x\in\mathcal{X}}{argmin} u(x, x^k)$$

要求:

- $u$在$f(x^k)$处相等且相切
- $u(x, y) \ge f(x)$

![image](res/cvx_mm_example.png)

作业有一道编码题要求使用MM求解

Expectation-Maximization EM 算法属于 MM
