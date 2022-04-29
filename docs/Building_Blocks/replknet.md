time: 20220429
short_title:RepLKNet
pdf_source:https://arxiv.org/pdf/2203.06717.pdf
code_source: https://github.com/DingXiaoH/RepLKNet-pytorch
# Scaling Up Your Kernels to 31x31: Revisiting Large Kernel Design in CNNs

本文作者有相当完整的[介绍](https://aijishu.com/a/1060000000309144), [MegEngine Code](https://github.com/MegEngine/RepLKNet). 

基本逻辑:

1. 使用大卷积核直接增大感受野以及有效感受野(effective receptive field ERF), 同时使用depthwise conv控制FLOPS复杂度。
2. 更改卷积的CUDA加速代码使得卷积算法充分利用大卷积核后在卷积核维度上的并行度.
3. 模仿[RepVGG](RepVGG.md)的参数重整化方式，增加identity connection以及小卷积核分支。在推理的时候融合成VGG形态的简单结构。



