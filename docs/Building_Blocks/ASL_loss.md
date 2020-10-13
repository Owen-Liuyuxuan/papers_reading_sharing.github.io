time: 20201013
pdf_source: https://arxiv.org/pdf/2009.14119v1.pdf
code_source: https://github.com/Alibaba-MIIL/ASL

# Asymmetric Loss For Multi-Label Classification (ASL Loss)

这篇paper提出了 Asymmetric Loss. 用于解决正负样本不平衡的问题。进一步地提供了概率解释以及自适应版本。


## Focal Loss

$$
L=-y L_{+}-(1-y) L_{-}
$$

$$
\left\{\begin{array}{l}
L_{+}=(1-p)^{\gamma} \log (p) \\
L_{-}=p^{\gamma} \log (1-p)
\end{array}\right.
$$


## Asymmetric Focusing

作者添加多一个变量,提出让负样本与正样本的$\gamma$不对称，且一般来说$\gamma_{-} > \gamma_{+}$

$$
\left\{\begin{array}{l}
L_{+}=(1-p)^{\gamma_{+}} \log (p) \\
L_{-}=p^{\gamma_{-}}\log (1-p)
\end{array}\right.
$$

进一步作者提出 Asymmetric Probability Shifting. 对于负样本中简单的部分直接清除掉.

$$
\begin{aligned}
    &p_m = max(p - m, 0) \\
    &L_{-} = (p_m)^{\gamma_{-}} \log{1 - p_m}
\end{aligned}
$$


## 概率实验

作者定义对预测结果的有效置信度$p_t$

$$
p_{t}=\left\{\begin{array}{ll}
\bar{p} & \text { if } y=1 \\
1-\bar{p} & \text { otherwise }
\end{array}\right.
$$

正负样本概率gap:
$$\Delta p = p_t^+ - p_t^-$$
实验发现ASL Loss可以使得训练过程中正负样本的概率gap比较小。

## 自适应调参

由以上实验，作者发现其实可以动态地调整$\gamma_-$使得概率gap保持在一定的范围内。

$$
\gamma_{-} \leftarrow \gamma_{-}+\lambda\left(\Delta p-\Delta p_{\text {target }}\right)
$$

