time: 20200408
pdf_source: https://arxiv.org/pdf/1904.07850.pdf
code_source: https://github.com/xingyizhou/CenterNet
short_title: Detection and Tracking as Point

# Detection and Tracking as Point

本文引入两篇paper，第一篇是与CenterNet撞名的 *Object as Point*, 第二篇是以此为基础的*Tracking as Point*,它是好多篇2D/3D检测的前置paper，其特点是速度很快，不需要特殊的NMS(真正意义地抛却NMS)，且模型扩展性很强——在网络中事实上很多东西都是一个point

## Object as Point

[pdf](https://arxiv.org/pdf/1904.07850.pdf) [code](https://github.com/xingyizhou/CenterNet)

这篇paper的keypoint检测部分与[CornerNet](CornerNet_Detecting_Objects_as_Paired_Keypoints.md)是一致的,但是正样本的定义有区别，最靠近物体中心的点会被标记为正样本，使用一个高斯核，根据物体大小smooth out 分类网络的负样本惩罚.

$$L_{k}=\frac{-1}{N} \sum_{x y c}\left\{\begin{array}{ll}
\left(1-\hat{Y}_{x y c}\right)^{\alpha} \log \left(\hat{Y}_{x y c}\right) & \text { if } Y_{x y c}=1 \\
\left(1-Y_{x y c}\right)^{\beta}\left(\hat{Y}_{x y c}\right)^{\alpha} \log \left(1-\hat{Y}_{x y c}\right) & \text { otherwise }
\end{array}\right.$$

**注意CenterTrack的公式是有bug的，漏一个一个负号,血泪教训**

对于2D框的长宽以及cx cy,不normalize,直接L1loss回归raw pixel coordinates,选择scale loss(也就是说输出的数值会很大，但是loss会比较正常)

$$L_{d e t}=L_{k}+\lambda_{s i z e} L_{s i z e}+\lambda_{o f f} L_{o f f}$$

进行NMS的时候，不需要使用特殊的NMS，作者采用的是更为的Maxpooling

```python
def _nms(heat, kernel=3):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(
        heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep
```

进而作者延伸出更多的application，比如比如借助预测offset以及heatmap的机制进行pose-estimation,同时预测3D信息实现3D object detection。速度快而精确

## Tracking Objects as Points

[pdf](https://arxiv.org/pdf/2004.01177v1.pdf) [code](https://github.com/xingyizhou/CenterTrack)

这篇paper也被称作CenterTrack.

思路是将连续两帧的图片以及上一帧的tracking结果构成的 heatMap作为输入，因此网络的结构与前文一致，仅仅区别在于多4个channel的输入。输出额外多两个channel,也就是两帧ground truth之间的区别，

$$L_{o f f}=\frac{1}{N} \sum_{i=1}^{N}\left|\hat{D}_{\mathbf{p}_{i}^{(t)}}-\left(\mathbf{p}_{i}^{(t-1)}-\mathbf{p}_{i}^{(t)}\right)\right|$$

因而在tracking的时候，按照confidence的排序，将当前位置点与最靠近$p - D_p$的前一帧点贪婪地匹配，如果发现在一定范围内unmatched，就认为产生了一个新的物体。

**比较神奇的是，本文是可以用静态图片进行训练的**,随机将图片平移，scale up，结果惊人地好