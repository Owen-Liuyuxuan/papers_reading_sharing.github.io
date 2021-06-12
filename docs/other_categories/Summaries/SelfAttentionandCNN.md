code_source: https://github.com/epfml/attention-cnn
pdf_source: https://arxiv.org/pdf/1911.03584.pdf
time: 20210612
short_title: Self-Attention & CNN
# Summary of Self Attention / Transformer in Vision System (Last update 2021-05-15)

这是一份描述数篇关于在CNN系统中使用self-attention的小统计.

对于在NLP取得很好表现的self-attention机制，本网站在论文[Attention is all you need](../../Building_Blocks/Attention_is_all_you_need.md)有详细的介绍。

2019年上半年，Google 提出了 [Attention Augmented Convolution](../../Building_Blocks/Attention_Augmented_Conv.md), 这是一个类似[Non-local](../../Building_Blocks/Non-local_Neural_Networks.md)模块的思路，借用self-attention的机制加上positional-encoding，设计出一个提供全局attention的模块，这个模块的缺点在于在图片很大的时候会需要一个很大的矩阵乘法，所以这个模块必须只能在多次下采样后使用，且还需要注重显存管理，可以理解为是一个难以scale up的方案。

本文接下来会介绍两篇paper，一篇paper提出了一个轻量级的图片局部attention的模块，用略少于传统Conv的运算与参数，在imagenet和Coco分别得到了与传统CNN几乎一致的性能。另一篇阐述了局部Attention与Convolution的关系，表明Multi-head 局部Attention可以实现传统Convolution的性能。

## Stand-Alone Self-Attention in Vision Models
[pdf](https://arxiv.org/pdf/1906.05909.pdf) [code](https://github.com/leaderj1001/Stand-Alone-Self-Attention)

![image](res/localAttention_compute.png)

$$
y_{i j}=\sum_{a, b \in \mathcal{N}_{k}(i, j)} \operatorname{softmax}_{a b}\left(q_{i j}^{\top} k_{a b}+q_{i j}^{\top} r_{a-i, b-j}\right) v_{a b}
$$

这个模块的设计可以直接CNN，同时运算量并没有显然提升，性能则相近。

作者通过实验发现除了第一个Conv建议用Convolution，模型的其余卷积模块就可以用Attention替代。作者还做了更多的其他实验，证明attention替代CNN是大有可为的。

## On The Relationship Between Self-Attention and Convolution Layers

[pdf](https://arxiv.org/pdf/1911.03584.pdf) [code](https://github.com/epfml/attention-cnn)

本文还有一个[官方网站](https://epfml.github.io/attention-cnn/)以及[官方英文博客](http://jbcordonnier.com/posts/attention-cnn/)

理论结论是局部Attention是CNN的扩展，具体implementation有区别。

## Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

[pdf](https://arxiv.org/pdf/2103.14030.pdf) [code](https://github.com/microsoft/Swin-Transformer)

![image](res/swin_transformer_arch.png)

![image](res/swin_transformer_idea.png)

这篇paper其实有点回到local了，但是更加可靠了，每次只对窗口内的跑transformer, 依靠多层级逐渐提升感受野.

## Scaling Vision Transformers
[pdf](https://arxiv.org/pdf/2106.04560.pdf)

这篇google的paper在JFT-3B以及数千TPU的加持下训练了一个SOTA的Transformer.改进了ViT的架构和训练，减少了内存消耗并提高了模型的准确性.

[BLOG](https://blog.csdn.net/amusi1994/article/details/117827006)

方案

- Decouple weight decay for the head. 让输出头的weight decay更大。
- Save memory by removing the [class] token
- Scale up data
- Memory-efficient optimizers: adafactor optimizer
- Learning-rate schedule: Reciprocal-square root

