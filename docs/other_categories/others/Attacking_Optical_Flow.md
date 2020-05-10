time: 20200510
pdf_source: https://arxiv.org/pdf/1910.10053.pdf
code_source: https://github.com/anuragranj/flowattack/
# Attacking Optical Flow


这篇paper follow [Adversarial Patch](adversarialPatch.md)的做法，不同之处在于使用patch去攻击optical flow 而不是单分类网络。

## Approach


优化的指标是最大化被攻击后的光流方向与正确光流方向的夹角。
作者指出现在没有足够大量的 dense　的optical flow　数据集，因而使用现成model生成Pseudo Ground Truth.这也有一个好处在于使得这个方法可以在任意未标注的视频上训练，并攻击任意的光流模型。

$$\hat{p}=\underset{p}{\operatorname{argmin}} \mathbb{E}_{\left(I_{t}, I_{t+1}\right) \sim \mathcal{I}, l \sim \mathcal{L}, \delta \sim \mathcal{T}} \frac{(u, v) \cdot(\tilde{u}, \tilde{v})}{\|(u, v)\| \cdot\|(\tilde{u}, \tilde{v})\|}$$

具体性能可以观看视频

<iframe width="560" height="315" src="https://www.youtube.com/embed/5nQ7loiPmdA" frameborder="0" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

