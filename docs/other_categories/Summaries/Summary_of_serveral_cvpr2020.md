time: 20200403
short_title: CVPR 2020 clips

# Summaries for sevearl CVPR 2020 papers

## RetinaTrack
[pdf](https://arxiv.org/pdf/2003.13870.pdf)

![image](res/RetinaTrack_arch.png)

特点，不同anchor在更早期features就开始分开，每一个anchor输出一个256维度的features.

对**单一图片**用triplet loss
$$
\mathcal{L}_{B H}(\theta ; X)=\sum_{j=1}^{A} \text { SoftPlus }\left(m+\max _{p=1 \rightarrow A \atop t_{j}=t_{p}} D_{j p}-\min _{\ell=1 \ldots A \atop t_{j} \neq t_{\ell}} D_{j \ell}\right)
$$

核心思路就是相同instance的不同anchor输出相似的embedding，不同instance的不同anchor输出不同的embedding。本文用基础的euclidean distance作为loss

## MUXConv: Information Multiplexing in Convolutional Neural Networks
[pdf](https://arxiv.org/pdf/2003.13880v1.pdf) [code](https://github.com/human-analysis/MUXConv)

![image](res/MUX_spatial.png)
![image](res/MUX_channel.png)

## Structure Aware Single-stage 3D Object Detection from Point Cloud
[pdf](https://www4.comp.polyu.edu.hk/~cslzhang/paper/SA-SSD.pdf) [code](https://github.com/skyhehe123/SA-SSD)

基于MMdetection开发的点云3D 检测，性能很高，重点在于附加task的设计的，能让一个接近于VoxelNet的结构得到很大的提升

![image](res/SASSD_arch.png)