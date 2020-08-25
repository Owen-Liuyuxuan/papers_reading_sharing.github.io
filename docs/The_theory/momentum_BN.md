time: 20200825
pdf_source: https://www4.comp.polyu.edu.hk/~cslzhang/paper/conf/ECCV20/ECCV_MBN.pdf
short_title: Momentum Batch Normalization

# Momentum Batch Normalization for Deep Learning with Small Batch Size

这篇paper尝试推导地证明 Batch Normalization 在训练时会引入一定的噪声，且这个噪声的方差与 Batch Size 成反比.然后本文提出了Momentum BN来平衡。

相似的希望在低 Batch下降低噪声的方案本站收录有 [Cross Iteration BN](../Building_Blocks/crossBatchNormalization.md).本文提出的 Momentum BN相较而言更简单，理论分析上各有优劣。

## Noise in BN

假设输入信号符合某一正太分布$~ \mathcal{N}(\mu, \sigma^2)$，那么大小为$m$的mini-batch则可以理解为对这一正太分布进行采样。采样数据对原正太数据的统计特征进行估计时，其结果的均值$\mu_B = \frac{1}{m} \sum^m_{i=1} x_i$也满足一个正太分布，而其结果的方差 $\sigma_B^2 = \frac{1}{m}\sum^m_{i=1}(x_i - \mu_B)^2$也满足一个卡方分布. 可以得到以下两个随机变量

$$\xi_{\mu}=\frac{\mu-\mu_{B}}{\sigma} \sim \mathcal{N}\left(0, \frac{1}{m}\right), \quad \xi_{\sigma}=\frac{\sigma_{B}^{2}}{\sigma^{2}} \sim \frac{1}{m} \chi^{2}(m-1)$$

那么在训练过程中，BN的归一化运算可以重新写为:

$$\widehat{x}=\frac{x-\mu_{B}}{\sigma_{B}}=\frac{x-\mu+\left(\mu-\mu_{B}\right)}{\sigma \frac{\sigma_{B}}{\sigma}}=\frac{\frac{x-\mu}{\sigma}+\xi_{\mu}}{\sqrt{\xi_{\sigma}}}=\frac{\widetilde{x}+\xi_{\mu}}{\sqrt{\xi_{\sigma}}}$$

在概念上，$\tilde{x}$为原数据的真实归一化值， $\xi_{\mu}$为加法项的高斯噪声，而$\xi_{\sigma}$为乘法项的高斯噪声。

## Reduce Noise for BN with Small Batches

$$\mu_{M}^{(n)}=\lambda \mu_{M}^{(n-1)}+(1-\lambda) \mu_{B}, \quad\left(\sigma_{M}^{(n)}\right)^{2}=\lambda\left(\sigma_{M}^{(n-1)}\right)^{2}+(1-\lambda) \sigma_{B}^{2}$$

算法上比较直白，对均值与方差计算一个滑动平均。不难计算滑动平均后归一化运算的高斯噪声也会平滑而下降。

$$\xi_{\mu}=\frac{\mu-\mu_{M}}{\sigma} \sim \mathcal{N}\left(0, \frac{1-\lambda}{m}\right)$$

进一步地，作者指出我们可以通过调整$\lambda$，使BN的噪声等效于指定的Batch Size时的噪声。

本文与[Cross Iteration BN](../Building_Blocks/crossBatchNormalization.md)的区别:

优点：
- 可以通过调节$\lambda$指定对应的batch size
- 计算过程简单,快速.

缺点:
- 没有考虑训练过程中由于权重变化带来的统计特征漂移