time: 20200316
pdf_source: https://distill.pub/2016/deconv-checkerboard/
code_source: https://github.com/distillpub/post--deconv-checkerboard

# Deconvolution and Checkerboard Artifacts

这篇paper的"pdf链接"指向的就是这篇paper的官网，很强的可视化效果，讨论的问题是deconvolution以及它造成的棋盘格子效应。

具体原理与可视化在官方的动图中很清晰。这里综合一下本文的结论


1. 在upsampling的时候用deconv会形成棋盘格子,尤其是kernel_size不能整除stride的时候(常见kernel_size=3, stride=2就是如此)
2. 多层深度网络在理论上可以学习权重消除这个棋盘样式，但是这个很难，自然情况下多层网络，二维卷积只会加强这个artifact。
3. 使用resizing 加上convolution能更好的消除artifact
4. deconv的artifact也会说明二维Downsample的时候，如果使用convolution直接下采样，也会在梯度上形成artifact，也就是会形成噪音，有些像素得到的梯度更多有的会少。这会影响GAN的训练以及一般卷积网络的训练。
