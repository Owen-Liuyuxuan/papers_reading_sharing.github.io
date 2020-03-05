time: 20200304
short_title: Recent Collections for Stereo 3D

# Recent Collections for Stereo 3D detection

近期积攒了一系列双目3D检测paper的阅读。这里一次过进行记录,以结果排列为顺序。

这里列出目前有文章可寻的KITTI排行榜(2020.03.04)

| Methods          | Moderate |   Easy  |   Hard  |  Time  |
|------------------|:--------:|:-------:|:-------:|:------:|
| [DSGN]           |  52.18 % | 73.50 % | 45.14 % |  0.67  |
| [Pseudo-LiDAR++] |  42.43 % | 61.11 % | 36.99 % |  0.4 s |
| [ZoomNet]        |  38.64 % | 55.98 % | 30.97 % |  0.3 s |
| [OC Stereo]      |  37.60 % | 55.15 % | 30.25 % | 0.35 s |
| [Pseudo-Lidar]   |  34.05 % | 54.53 % | 28.25 % |  0.4 s |
| [Stereo R-CNN]   |  30.23 % | 47.58 % | 23.72 % |  0.3 s |
| [RT3DStereo]     |  23.28 % | 29.90 % | 18.96 % | 0.08 s |

# Pseudo-Lidar++
[pdf](https://arxiv.org/pdf/1906.06310.pdf)  [code](https://github.com/mileyan/Pseudo_Lidar_V2)

![image](res/plidarpp_arch.png)

这篇paper的主要新意有两点

在深度估计上，作者给出一个新的insight，就是均匀的3D卷积很可能是disparity-based cost volome的一个error source，比如说对于disparity比较高的点，可以smooth out，但是对于disparity比较小的点则不应该同等级别的smooth out(会产生很大误差)。所以作者将disparity cost volume的深度方向求倒数，并线性插值得到depth cost volume,然后在depth cost volume上面做3D卷积。

在后处理上，作者融合了深度补全(depth completion)的思想，由于disparity是离散的，所以会引起很多不应该的误差，进而作者考虑使用低线数的lidar(开源的一个方案线数是4)作为一个ground truth的补偿。这里不进一步展开。

作者在KITTI上提交了两个成绩(PL++),标题下面给出的是春双目而没有GDC的成绩，有GDC的成绩会更高一些。


# ZoomNet
[pdf](https://arxiv.org/pdf/2003.00529.pdf) [code](https://github.com/detectRecog/ZoomNet)

![image](res/zoomnet_arch.png)

使用2D检测先得到两个车子咋图片的位置，然后分别resize,并且调节名义相机参数(zooming)。

中介辅助预测的内容包括disparity, instance segmentation, part location(每一个像素相对于车子中心x, y, z轴的位置，这个一般使用点云和稠密深度图进行标注)。得到点云后将feature 链接，然后用类似于point net的方式预测最终结果。

# OC Stereo
[pdf](https://arxiv.org/pdf/1909.07566.pdf)

![image](res/OC_stereo_arch.png)

这篇paper的想法是使用RoIAlign将左右目两个区域的feature 提出来，然后使用instance seg与Cost volumn计算对应pixel处的disparity。

作者对于RoIAlign前后的segmentation pixel的位置关系做了很细致的解释。

在得到局部RGB点云之后作者使用[AVOD](https://github.com/kujason/avod) 进行3D检测。

# Pseudo-Lidar
[pdf](https://arxiv.org/pdf/1812.07179.pdf) [code](https://github.com/mileyan/pseudo_lidar)

这篇paper理论上来说是pseudo-lidar的第一篇文章

![image](res/plidar_original.png)

思路目前回看比较地直接，双目深度估计方面，作者使用pretrain [PSMNet](../other_categories/others/PSMNet.md).注意这个PSMNet是在sceneflow数据集，以及training set的点云数据作为监督的。lidar 3D检测方面，作者使用[AVOD](https://github.com/kujason/avod)

# Stereo R-CNN
[pdf](https://arxiv.org/pdf/1902.09738.pdf) [code](https://github.com/HKUST-Aerial-Robotics/Stereo-RCNN)

![image](res/StereoRCNN_arch.png)

几个训练细节:

1. positive anchors 的threshold提高了。
2. 多预测一个Keypoint的位置，如下图
3. SSIM，利用双目的disparity，后处理优化深度值。

![image](res/StereoRCNN_keypoints.png)


# RT3D Stereo
[pdf](https://www.mrt.kit.edu/z/publ/download/2019/Koenigshof2019Objects.pdf)
![image](res/RT3DStereo_arch.png)

训练细节:

1. 使用单一一个ResNet解决2D 检测以及语义分割的encoding.用的是[二作作者的同时检测与语义分割网络.pdf](https://arxiv.org/pdf/1905.02285.pdf)
2. Disparity使用的是block matching的传统方法，
3. 作者根据语义分割以及detector结果分割出相关像素，然后聚类，然后以优化凸包的方式得出结果。由于作者没有开源，很多内容有待商榷。




[DSGN]:DSGN.md
[Pseudo-LiDAR++]:#pseudo-lidar
[ZoomNet]:#zoomnet
[OC Stereo]:#oc-stereo
[Pseudo-Lidar]:#pseudo-lidar_1
[Stereo R-CNN]:#stereo-r-cnn
[RT3DStereo]:#rt3d-stereo