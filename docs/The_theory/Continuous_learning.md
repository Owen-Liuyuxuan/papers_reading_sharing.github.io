time: 20201106
pdf_source: https://arxiv.org/abs/1612.00796

# Continuous Learning

这个页面介绍 continuous learning 等相关文章，这个实现的task是说: 现有一个模型，在数据集$D_A$上使用一个backbone $\Theta_A$, 一个head $\theta_A$完成了对第一个任务$A$的训练。然后现在我们失去了数据集$D_A$, 以 backbone $\Theta_A$为基础，一个新的head $\theta_B$ 进行训练，实现对任务$B$的训练，依次迭代。

直觉可以判断，如果直接训练$\theta_B$固定$\Theta_A$, 则原来的任务不会忘记，但是网络在$B$上的性能可能会有限。如果$\Theta_A$与$\theta_B$都在$B$上重新训练，则既是是低学习率的fine-tuning 都可能会使得网络完全破坏任务$A$的结果。

Continuous Learning的任务则是设计对策，使得在$B$,乃至$C, D ...$训练时可以同时提升新任务的性能且尽可能减少对旧任务的性能.

## Overcoming catastrophic forgetting in neural networks

[pdf](https://arxiv.org/pdf/1612.00796.pdf)

本文为一个开山鼻祖。首先从直觉来说，在没有旧任务数据的前提下要同时实现旧任务，又能训练新任务，最naive的方法就是设计损失函数

$$
    \mathcal{L} = \mathcal{L}_B + \lambda \mathcal{L}_{dist_{\theta_A}}
$$

其中前者为当前任务训练时的一个损失函数，后者为与原来训练好的参数$\theta_A$的某一个距离函数，约束网络不要离开原来的模型太远。

本文首先自然地进行第一个延伸: **在$\theta_A$中不同的参数需要约束的权重可以是不同的**，因为有重要的参数也有没那么重要的参数.

为了评价的权重参数，本文用概率论对这个问题进行了问题, 首先:

$$
\begin{aligned}
    \log p(\theta | D_A, D_B) &= \log p(D_B | \theta, D_A) + \log p(\theta | D_A) - \log p(D_B | D_A) \\
    &= \log p(D_B | \theta) + \log p (\theta | D_A) + \text{const} \\
    & = \log p (D_B | \theta) + \log p(D_A | \theta) + \log p(\theta) - \log p(D_A) + \text{const} \\
    & = \log p (D_B | \theta) + \log p(D_A | \theta) + \text{const}\\
\end{aligned}
$$
第一项: 对于基于独立同分布的分类问题来说，第一项等同于 对每一个数据， 用网络计算其得到当前标签类的概率. 这个与常用的损失函数Entropy是一致的; 对于基于独立分布且带有高斯噪声的回归问题来说, 第一项则等同于 加上固定权重的MSE 损失.

第二项: 由于在训练第二个任务的时候失去了第一个任务，这个函数变得不可计算了，作者提出使用泰勒展开公式在此前优化结果的最优值$\theta_A^*$附近展开, 使用一个二次函数对其进行近似. 这有点像凸优化/一次优化里面的 [Proximal Gradient Method](First_order_methods_review_and_updates.md).

$$
\begin{aligned}
    \log p(D_A | \theta) &= \log p(D_A | \theta^*_A) + \frac{\partial}{\partial \theta} \log p(D_A | \theta) |_{\theta=\theta_A^*} (\theta - \theta_A^*) + \frac{\partial^2}{\partial\theta^2} \log p(D_A|\theta)(\theta - \theta_A^*)^2\\

    &=  \frac{\partial^2}{\partial\theta^2} \log p(D_A|\theta)(\theta - \theta_A^*)^2 + \text{const}\\
    &= -\mathbb{F}(D_A, \theta_A^*)(\theta - \theta_A^*)^2
\end{aligned}
$$

可以注意到:

- 由于$\theta_A^*$是最优解了，所以一阶导项为零
- 工程技巧，这里简化了Hessian矩阵为对角矩阵,不考虑相关项.
- Fisher Infomation Matrix 近似 [wiki](https://www.wikiwand.com/en/Fisher_information#/Definition)
  
$$
\begin{aligned}
    \mathbb{F} &= - \mathbb{E}_x \left[ \frac{\partial^2}{\partial \theta^2} \log p(x|\theta)\right] =  \frac{\partial^2}{\partial \theta^2} \log p(D|\theta)\\
    &= -\mathbb{E}_x \left[\frac{\frac{\partial^2}{\partial\theta^2}p(x|\theta)}{p(x|\theta)} - \left( \frac{\frac{\partial}{\partial\theta} p(x|\theta)}{p(x|\theta)}\right)^2\right] \\ 
    &= -\mathbb{E}_x\left[\frac{\frac{\partial^2}{\partial\theta^2}p(x|\theta)}{p(x|\theta)} \right] + \mathbb{E}_x\left[\left(\frac{\partial}{\partial\theta} \log p(x|\theta)\right)^2 \right] \\
    &= \frac{\partial^2}{\partial\theta^2}\int p(x|\theta) dx + \left(\frac{\partial}{\partial\theta}\log p(D|\theta) \right)^2 \\
    & = \left(\frac{\partial}{\partial\theta}\log p(D|\theta) \right)^2
\end{aligned}
$$

这个等式的关键意思在于可以使用从数据采样一阶导的平方去近似二阶导的值。

### EWC: Elastic weight consolidation
因而本文提出的 EWC 算法:

在完成一个任务的学习后，记录下它的Fisher information matrix以及当时的模型，在面对新任务的时候，使用损失函数:

$$
\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i\frac{\lambda}{2} F_i(\theta_i - \theta_{A,i}^*)^2
$$

## On Quadratic Penalties in Elastic Weight Consolidation
[pdf](https://arxiv.org/pdf/1712.03847.pdf)

这篇paper接着前文的概念，进一步给出了在多个任务上迭代的公式，也就是当训练完第二个任务之后继续往第三、四、...、$t$个任务时的使用的公式.

这里首先以三个任务为例子:
$$
\log p\left(\theta \mid \mathcal{D}_{A}, \mathcal{D}_{B}, \mathcal{D}_{C}\right)=\log p\left(\mathcal{D}_{C} \mid \theta\right)+\log p\left(\theta \mid \mathcal{D}_{A}, \mathcal{D}_{B}\right)+\text { constant. }
$$

其中

$$

\begin{aligned}
\log p\left(\theta \mid \mathcal{D}_{A}, \mathcal{D}_{B}\right) \approx &\frac{\partial}{\partial \theta}\left[\log p\left(\mathcal{D}_{B} \mid \boldsymbol{\theta}\right)+\log p\left(\boldsymbol{\theta} \mid \mathcal{D}_{A}\right)\right]|_{\boldsymbol{\theta}_{A B}^{*}}\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{A B}^{*}\right) \\
&+\frac{1}{2}\left\{\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{A B}^{*}\right)^{T} \frac{\partial^{2}}{\partial^{2} \boldsymbol{\theta}}\left[\log p\left(\mathcal{D}_{B} \mid \boldsymbol{\theta}\right)+\log p\left(\boldsymbol{\theta} \mid \mathcal{D}_{A}\right)\right] \mid \boldsymbol{\theta}_{A B}^{*}\right.\\
&\left.\times\left(\boldsymbol{\theta}-\boldsymbol{\theta}_{A B}^{*}\right)\right\}+\mathrm{const.}
\end{aligned}

$$

注意这是在$\theta_{AB^*}$展开的. 留意到: 第一项一阶导依然为零，第二项二阶导里面在删去常数(这个常数会被二阶导清除)后可理解为两个 Fisher Information Matrix的相加.

代入得到

$$
\log p\left(\theta \mid \mathcal{D}_{A}, \mathcal{D}_{B}, \mathcal{D}_{C}\right) \approx \log p\left(\mathcal{D}_{C} \mid \theta\right)-\frac{1}{2} \sum\left(\lambda_{A} F_{A, i}+\lambda_{B} F_{B, i}\right)\left(\theta_{i}-\theta_{AB, i}^{*}\right)^{2}+\text { constant. }
$$
作者经过迭代后给出通项公式:

$$
\theta_{T}^{*}=\operatorname{argmin}_{\theta}\left\{-\log p\left(\mathcal{D}_{T} \mid \theta\right)+\frac{1}{2} \sum_{i}\left(\sum_{t<T} \lambda_{t} F_{t, i}+\lambda_{\text {prior }}\right)\left(\theta_{i}-\theta_{S, i}^{*}\right)^{2}\right\}
$$

给出一些关键的直觉:

- 每次的展开都是基于最近一次任务得到的最优解，因而只需要存下一个先前model.
- 随着任务的增加,Fisher matrix会变得越来越大，模型也会越来越难学新的结果。这也是不可避免的.
- 前面任务在后面的任务的head上没有梯度，因而$F_t$ $t\in [0,... \tau-1]$也不会影响新的head.每个任务自己的head只会有一个结果


## Memory Aware Synapses: Learning what (not) to forget
[pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Rahaf_Aljundi_Memory_Aware_Synapses_ECCV_2018_paper.pdf)

这篇paper的意见更为直接，就是说不应该考虑那个权重重要那个权重不重要，关键得看最终的输出是否相似.

设原来的网络的输出为$F(x_k;\theta)$, 经过扰动后的输出变化显然可以由输出值对权重的梯度，一阶近似.

所以每一个权重的 importance weight 就是 
$$
\Omega_{i j}=\frac{1}{N} \sum_{k=1}^{N}\left\|g_{i j}\left(x_{k}\right)\right\|
$$
与之前的设定不同，这里的importance weights. 这里采用更为简单的方案:

$$
    g_{ij}(x_k) = \frac{\partial l_2^2(F(x_k;\theta))}{\partial \theta_{ij}}
$$
具体实现上，就是在每个任务完成后，计算每一个权重的importance weights. 学习新的任务时:

$$
L(\theta)=L_{n}(\theta)+\lambda \sum_{i, j} \Omega_{i j}\left(\theta_{i j}-\theta_{i j}^{*}\right)^{2}
$$
尽管理论比较直白效果甚至比前文提到的更好.