time: 20211008
pdf_source: https://arxiv.org/pdf/1312.6114.pdf

# Auto-Encoding Variational Bayes

Variational Auto Encoder (VAE) 是经典的生成模型。其论文的讨论思路和目前大家整理的常用分析思路相比有一定的区别。这里根据各处的资料整理一份完整的理论解答.

已知数据集$X$, 其数据点$x_i$, 受隐变量$z_i$影响。我们的目标是设计一个解码器decoder, 参数记为$\theta$, 我们的目标是使用这个解码器对数据的分布进行建模，成为$p_\theta(x)$. 这个映射受隐变量$z$的影响，进而记这个分布为

$$p_\theta(x) = \int_z{p_\theta(x|z) p_\theta(z) dz}$$

目前这个方程的左边的计算以及优化，从各个角度来看都intractable，而且我们还需要对隐变量进行提取和分析. 这时候我们引入编码器encoder, 其参数记为$\phi$， 这个编码器会建模隐变量的后验分布 $q_\phi(z|x)$.
我们希望这个后验分布与上式展开式时所需要的后验分布$p_\theta(z|x)$几乎一致，也就是

$$q_\phi(z|x) \approx p_\theta(z|x)$$

首先我们尝试更数学地描述上述的$\approx$. 这里我们采用KL Divergence, 作为两个概率分布之间的差的描述.

$$D_{KL}(p||q) = H(p, q) - H(p) = \sum_{x\in X} P(x) \text{log}\left( \frac{P(x)}{Q(x)}\right) = \int{p(x)\text{log}(\frac{p(x)}{q(x)})}dx$$

前面的两个分布的KL Divergence展开如下

$$\begin{aligned}
D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})\right) &=\int q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z} \mid \mathbf{x})} d \mathbf{z} \\
&=\int q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x}) p_{\theta}(\mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} d \mathbf{z} \\
&=\int q_{\Phi}(\mathbf{z} \mid \mathbf{x})\left(\log \left(p_{\theta}(\mathbf{x})\right)+\log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})}\right) d \mathbf{z} \\
&=\log \left(p_{\theta}(\mathbf{x})\right)+\int q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z}, \mathbf{x})} d \mathbf{z} \\
&=\log \left(p_{\theta}(\mathbf{x})\right)+\int q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{x} \mid \mathbf{z}) p_{\theta}(\mathbf{z})} d \mathbf{z} \\
&=\log \left(p_{\theta}(\mathbf{x})\right)+E_{\mathbf{z} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x})}\left(\log \frac{q_{\Phi}(\mathbf{z} \mid \mathbf{x})}{p_{\theta}(\mathbf{z})}-\log \left(p_{\theta}(\mathbf{x} \mid \mathbf{z})\right)\right) \\
&=\log \left(p_{\theta}(\mathbf{x})\right)+D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)-E_{\mathbf{z} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x})}\left(\log \left(p_{\theta}(\mathbf{x} \mid \mathbf{z})\right)\right)
\end{aligned}$$

稍微变形，将我们想优化的内容放一边，可以计算的内容放另一边:

$$\log \left(p_{\theta}(\mathbf{x})\right)-D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})\right)=E_{\mathbf{z} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x})}\left(\log \left(p_{\theta}(\mathbf{x} \mid \mathbf{z})\right)\right)-D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)$$

左边第一项是我们想要最大化的表征解码器对数据分布的建模正确度的似然值，第二项是编码器对隐变量的建模与实际隐变量后验概率的概率模型差异，是我们想要最小化的内容.

右边第一项是由编码器采样的隐变量进行解码得到的数据的似然值，实现起来就是reconstruction loss, 第二项指编码器输出与隐变量本身的先验之间的概率差异，实现起来就是一个限制编码器输出幅值的regularizer.

我们令损失函数$L_{\theta, \Phi}=-\log \left(p_{\theta}(\mathbf{x})\right)+D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})\right)=-E_{\mathbf{z} \sim q_{\Phi}(\mathbf{z} \mid \mathbf{x})}\left(\log \left(p_{\theta}(\mathbf{x} \mid \mathbf{z})\right)\right)+D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z})\right)$

由KL divergence的非负性，可以得到

$$-L_{\theta, \Phi}=\log \left(p_{\theta}(\mathbf{x})\right)-D_{K L}\left(q_{\Phi}(\mathbf{z} \mid \mathbf{x}) \| p_{\theta}(\mathbf{z} \mid \mathbf{x})\right) \leq \log \left(p_{\theta}(\mathbf{x})\right)$$

因而损失函数优化的内容又可以理解为 $\log{p_\theta(x)}$的下限，因而被成为 evidence lower bound (ELBO).

