pdf_source: https://arxiv.org/pdf/1611.03530.pdf
# Understanding Deep Learning Requires Rethinking Generalization
这篇文章讨论网络的generalization能力。

一个特殊的实验：

    在一个dataset中，将label改为完全随机抽取，实验证明，目前的神经网络能够完全记忆所有的随机label，优化难度并没有显著提升，但是明确可以知道，generalization的结果必然等同于随机(training 与 testing完全不相关)。

所以首先可以知道优化是否顺利与能否generalize没有直接显著的关系

关于regularization技巧：

数据增强、weight delay($l_2$ regularizer)、dropout，经验上都能提升网络的test准确率，但是在fitting random labels的时候都是能达到training accuracy = 100%,test accuracy仍然是10%.

BatchNorm和EarlyStop都是有效的。