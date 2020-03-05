time: 20200305
pdf_source: https://arxiv.org/pdf/1611.01144.pdf
short_title: Gumbel_softmax; Differentiable Indexing
# Categorical Reparameterization with Gumbel-Softmax

这篇文章的内容已经固化为了pytorch的一个[函数](https://pytorch.org/docs/stable/nn.functional.html#gumbel-softmax)

其作用是允许 Stochastic, Differentiable, Probabilistic Weighted, Indexing.

先用代码解释 gumbel采样可以如何用均匀随机采样表达：

```python

def gumbel(*shape):
    u = np.random.rand(*shape)
    return -np.log(-np.log(u))

def gumbelsoftmax(weights, lmbda=1, N=10000):
    d = len(weights)
    logits = np.log(weights.reshape(d, 1))
    gumbel_noise = gumbel(d*N).reshape(d, N)
    return softmax(( logits + gumbel_noise)/ lmbda, axis=0)
```

通过采样 gumbelsoftmax,得到的分布近似于 

$\frac{weights}{\sum(weights)}$

其中$lmbda$变量理解为温度超参，其作用在于控制采样系统的随机性.


pytorch functional 的代码如下(不必复制使用，这内置于pytorch中):

```python
def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
    # type: (Tensor, float, bool, float, int) -> Tensor
    r"""
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    """

    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret
```

值得注意的是这里使用了

```python
ret = y_hard - y_soft.detach() + y_soft
```
这一个trick使得one-hot的$y\_hard$在forward时是indexing量，但是backward的时候用的是$y\_soft$的梯度。

这篇文章与很多其他内容相关，比如在强化学习中作为一个可以exploit又可以explore的hard indexing. 在网络剪枝中可以作为一个可以学习使用的参量。

比较奇妙的是这篇paper在ICLR发布的时候只是marginally accepted,也只是poster,主要是可能原作者以及Reviewer当时只是在考虑使用在Generative Model上，提升没有那么显著，而没有预知这么多后来的应用。不过后来大家对它的引用以及发挥是很巨大的.

