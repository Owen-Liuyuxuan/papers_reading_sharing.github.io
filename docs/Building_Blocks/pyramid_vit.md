time: 20210714
pdf_source:https://arxiv.org/pdf/2102.12122.pdf
code_source:https://github.com/whai362/PVT
# Pyramid Vision Transformer: A Versatile Backbone for Dense Prediction without Convolutions

这篇paper给出了一个在高分辨率时仍能运行的transformer的backbone,并且能输出dense head.

![image](res/pyramid_vit_arch.png)

在stage $i$, 特征图会变成 $\frac{H_{i-1}W_{i-1}}{P_i^2}$个patches


![image](res/pyramid_vit_sra.png)

对于高分辨率的图，可以采用如图的方式，spatially reduce key和value的维度，以控制在高分辨率时的维度.

## PVTv2: Improved Baselines with Pyramid Vision Transformer
[pdf](https://arxiv.org/pdf/2106.13797.pdf) 

这篇paper 给出三个新的PVT改良:

- overlapping patch embedding
- Zero-padding and depth-wise convolution aided feed forward network
- Linear SRA

![image](res/pyramid_vit_v2_1.png)
![image](res/pyramid_vit_v2_2.png)