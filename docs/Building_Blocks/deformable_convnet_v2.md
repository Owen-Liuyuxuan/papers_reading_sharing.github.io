pdf_source: https://zpascal.net/cvpr2019/Zhu_Deformable_ConvNets_V2_More_Deformable_Better_Results_CVPR_2019_paper.pdf
code_source: https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
short_title: Deformable ConvNet V2
# Deformable ConvNets V2: More Deformable, Better Results

这篇论文一方面讨论了一些评价deformable convolution模块性能的metric,并用一些实验引入。在模块设计上给出deformable convolution的细节.

# 结构解读

本文的关键想法是允许deformable conv模块在同一张图片同一个channel、不同地方有不同的behavior,而原来的conv以及初始的deformable conv都是处处相同的特征。计算公式
$$y(p) = \sum^K_{k=1}w_k\dot x(p + p_k + \Delta p_k) \dot \Delta m_k$$

其中$p, x(p), y(p)$表示位置$p$以及当前位置上的输入、输出feature map。$K$为卷积核的数量，$p_k， w_k$为普通卷积学习到的offset以及权重(在同一图片同一channel处处相同),$\Delta p_k, \Delta m_k$为新版增加的，与当前区域相关的offset以及权重
(offset 形状为 [B, kernel_size\*\*2, H, W])

而$\Delta p_k, \Delta m_k$由当前位置卷积产生，比如使用K=9，对当前位置使用正常卷积，得到27个channel，前18个channel分别对应$\Delta p_k$的两个坐标,后9个channel经过sigmoid激活后变为$\Delta m_k$

## 细节

可以学习本文github中使用cuda辅助帮忙开发pytorch子模块，如果可能可以适当学习。有加速代码运行的潜能。