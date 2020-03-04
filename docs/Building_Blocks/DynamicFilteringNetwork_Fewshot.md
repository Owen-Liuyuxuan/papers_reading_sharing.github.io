pdf_source: https://www.researchgate.net/publication/326584672_Dynamic_Conditional_Networks_for_Few-Shot_Learning
code_source: https://github.com/ZhaoJ9014/Dynamic-Conditional-Networks.PyTorch
# Dynamic Conditional Networks for Few-Shot Learning

本文使用class的文字 embedding 作为 convolution 的 guidance 来完成few-shot learning的任务。

这里仅介绍其开源的部分,也就是 dynamic filtering network部分.

## Dynamic Filtering Network

![image](https://raw.githubusercontent.com/ZhaoJ9014/Dynamic-Conditional-Networks.PyTorch/master/pub/DCL.png)

算法上为初始化$N$个权重组，embedding输出一个$N$维的分类矢量，将$N$个权重加权求和，作为主线上Conv2D的权重。