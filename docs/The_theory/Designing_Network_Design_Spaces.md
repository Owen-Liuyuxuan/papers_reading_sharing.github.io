time: 20200404
pdf_source: https://arxiv.org/pdf/2003.13678.pdf

# Designing Network Design Spaces

这一篇是何凯明组超越NAS进行思考的一项工作，中文源有相当多的介绍，[知乎](https://zhuanlan.zhihu.com/p/122557226), [CSDN](https://blog.csdn.net/jiaoyangwm/article/details/105245796)

NAS for image classification，包括最近的[EfficientNet](../Building_Blocks/EfficientNet:_Rethinking_Model_Scaling_for_Convolutional_Neural_Network.md)，核心思路是通过搜索或者学习，得到单一一个best model，文章讨论的重点也是在于如何缩小无穷的搜索空间以及如何学习的方法。

本文的核心思路贡献在于对搜索空间进行大幅度采样、统计分析，再进行剪枝分析，得到一些更鲁棒的分析结论，所以称为"designing network design spaces"。

## 搜索空间探索

作者的思路是搭建控制模型的FLOPS在400MFLOPS级别，在分析某一个搜索空间的时候，在其中采样500个模型，快速在imagenet上训练(使用最简单的设定，比如减少数据增强，epoch仅为10)

统计分析这些模型的性能，用来评估这个设计空间的性能，同时根据设定的一些变量与性能的相关性找出修剪或者改进搜索空间的方向。

作者从AnyNetXa逐步缩小空间早AnyNetXe,最后到RegNet的搜索空间。这个过程中有一系列的结论

1. 网络深度最好在20个block，也就是60层附近，
2. bottleneck ratio应该取1，也就是不使用Bottleneck
3. 每次下采样，channel宽度的乘积应当为2.5左右，而不是普遍使用的2(也相近)
4. "activation"也就是卷积层输出tensor的大小与运算时间的相关性超越FLOPS
5. inverted bottleneck以及 depthwise_conv(每一个channel一组卷积参数)对性能是有损害的。
6. squeeze and excitation是有效的

最后将[EfficientNet](../Building_Blocks/EfficientNet:_Rethinking_Model_Scaling_for_Convolutional_Neural_Network.md)的点数锤了一下,且更加轻量.