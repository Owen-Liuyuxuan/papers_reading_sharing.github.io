time: 20191209
pdf_source: https://arxiv.org/pdf/1609.03677.pdf
code_source: https://github.com/mrharicot/monodepth
short_title: Unsupervised Mono Depth from stereo
# Unsupervised Monocular Depth Estimation with Left-Right Consistency

这篇论文解决的问题是无监督条件下，仅使用单个相机单次的拍摄图，得到对深度的估计。在训练过程中需要第二个相机同时拍摄的结果来构造损失函数用于训练。可以理解为是使用第二个相机作为监督输入，但是这也节省了数据标定的难度。本文同时有非官方的[pytorch实现]

## inference结构

![image](res/MonoDepth_model.png)

本文神经网络的输入是单张RGB图片，输出是multi-scale的disparity,并且在每个scale是同时输出左图->右图的disparity以及右图->左图的disparity.作者的思路是在多个scale上同时对两个disparity检测的正确性、统一性进行检验。

## Loss结构

定义在scale $s$上的损失值为$C_s$,

损失函数为

$$
C_{s}=\alpha_{a p}\left(C_{a p}^{l}+C_{a p}^{r}\right)+\alpha_{d s}\left(C_{d s}^{l}+C_{d s}^{r}\right)+\alpha_{l r}\left(C_{l r}^{l}+C_{l r}^{r}\right)
$$

### 重构损失

$C_{ap}$指的是左右图相互重构时的误差损失，本文同时采用naive的$L1$距离以及[SSIM距离]的加权求和，代码中可以清楚地留意到作者是如何使用Pytorch原生层以及基本操作实现SSIM的计算，并且允许反传。

$$
C_{a p}^{l}=\frac{1}{N} \sum_{i, j} \alpha \frac{1-\operatorname{SSIM}\left(I_{i j}^{l}, \tilde{I}_{i j}^{l}\right)}{2}+(1-\alpha)\left\|I_{i j}^{l}-\tilde{I}_{i j}^{l}\right\|
$$

### 光滑损失

$C_{ds}$代表的是disparity-smoothness loss，这里的做法是对disparity map的梯度进行惩罚.
$$
C_{d s}^{l}=\frac{1}{N} \sum_{i, j}\left|\partial_{x} d_{i j}^{l}\right| e^{-\left\|\partial_{x} I_{i j}^{l}\right\|}+\left|\partial_{y} d_{i j}^{l}\right| e^{-\left\|\partial_{y} I_{i j}^{l}\right\|}
$$

### 左右差值损失

$C_{lr}$用来表征两个disparity map的自洽性。根据左图的disparity，将左图的点投到右图，这两个对应点的disparity应该是一致的。

$$
C_{l r}^{l}=\frac{1}{N} \sum_{i, j}\left|d_{i j}^{l}-d_{i j+d_{i j}^{l}}^{r}\right|
$$


[pytorch实现]:https://github.com/OniroAI/MonoDepth-PyTorch
[SSIM距离]:https://www.wikiwand.com/en/Structural_similarity#/Algorithm