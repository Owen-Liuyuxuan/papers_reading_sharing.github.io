time: 20200608
short_title: ICRA 2020 clips

# Summaries for sevearl ICRA 2020 papers

本届ICRA有数篇paper在之前已经有review, ["A General Framework for Uncertainty Estimation in Deep Learning"](../../The_theory/Framework_Uncertainty_Propagation.md), ["FADNet: A Fast and Accurate Network for Disparity Estimation"](../others/FADNet.md) ["Object-Centric Stereo Matching for 3D Object Detection"](../../3dDetection/RecentCollectionForStereo3D.md)


这里继续搜集多篇有趣的ICRA 2020 papers.

## Event-Based Angular Velocity Regression with Spiking Networks

[pdf](https://arxiv.org/pdf/2003.02790.pdf) [code](https://github.com/uzh-rpg/snn_angular_velocity)

这篇paper利用了2018NeurIPS的一篇关于[spiking neural network](http://papers.nips.cc/paper/7415-slayer-spike-layer-error-reassignment-in-time.pdf)的文章，这篇文章提出了SNN的一个训练方法，并且介绍了相关的概念，同时给出了[pytorch库/cuda](https://github.com/bamsumit/slayerPytorch)代码用于加速运算.

本文利用了NIPS paper的这个库，输入为序列的image-like event sequence,输出为序列的三轴角速度，

![image](res/event_snn.png)

仿真数据来自于[esim仿真器](https://github.com/uzh-rpg/rpg_esim)

## Pedestrian Planar LiDAR Pose (PPLP) Network for Oriented Pedestrian Detection Based on Planar LiDAR and Monocular Images

[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8943147) [code](https://github.com/BoomFan/PPLP)

![image](res/PPLP.png)

## CNN Based Road User Detection Using the 3D Radar Cube

[pdf](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8962258) [code](https://github.com/tudelft-iv/RTCnet)

这篇paper调用底层的radar数据，同时使用底层的radar cube数据以及radar target数据，在纯radar的条件下实现了 3D object detection.

![image](res/radar_cube_arch.png)

## PST900: RGB-Thermal Calibration, Dataset and Segmentation Network

[pdf](https://arxiv.org/pdf/1909.10980.pdf) [code](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)

本文提出的主要贡献是 RGB-Thermal的校正(利用一个双目RGB相机得到深度估计，再回投到Thermal上)以及一个语义分割数据集，

## Instance Segmentation of LiDAR Point Clouds

[pdf](http://www.feihuzhang.com/ICRA2020.pdf) [code](https://github.com/feihuzhang/LiDARSeg)

![image](res/instance_seg_lidar.png)
![image](res/lidar_instanceseg_arch.png)

# SegVoxelNet: Exploring Semantic Context and Depth-aware Features for 3D Vehicle Detection from Point Cloud

[pdf](https://arxiv.org/pdf/2002.05316.pdf)

![image](res/segvoxelnet_arch.png)

这篇paper来自于[D4LCN](../../3dDetection/RecentCollectionForMono3D.md)的组。

语义分割BEV Ground Truth来自于bbox直接的投影。Depth Aware的理解是近处、远处的点云分布密度差距较大，将BEV沿着深度轴分成带有重叠部分的几个部分，执行不同的卷积操作。在KITTI上的性能与PointPillars和PointRCNN相近。

##  Radar as a Teacher: Weakly Supervised Vehicle Detection using Radar Labels

[pdf](http://www.robots.ox.ac.uk/~mobile/Papers/Relabel_ICRA2020.pdf)

![image](res/relabel_coteaching.png)

这篇paper建议参考此前NIPS的 [co-teaching的paper](https://arxiv.org/pdf/1804.06872.pdf)

## Self-supervised linear motion deblurring

[pdf](https://arxiv.org/pdf/2002.04070.pdf) [code](https://github.com/ethliup/SelfDeblur)

这篇paper出自KITTI数据集的实验室。这篇paper的主要idea是使用一个reblur module，在线性运动的假设下，利用光流与blurring之间的关系，将一个deblurred的module重新变为blurred，这样就可以形成一个自监督的体系。

本文使用现成的deblur以及光流网络结构，利用前后帧的consistence训练光流，同时对deblur结果提出隐性的要求。前面提到的自监督网络loss可以训练deblur网络，同时对光流的计算提出隐性的要求，本文的reblur是一个非学习可微分模块，因而整个网络可微分，可以端到端自监督学习。

![image](res/self-supervised-deblur.png)

reblur模块方程:

$$\mathbf{B}(\mathbf{x}) \approx \frac{1}{2 N+1} \sum_{i=-N}^{N}\left(\mathcal{W}_{0 \rightarrow i} \circ \mathbf{I}_{0}\right)(\mathbf{x})$$

其中$\mathcal{W}$指的是将原图根据光流进行warping,

![image](res/reblur_warpping.png)


## Fast Panoptic Segmentation Network
[pdf](https://arxiv.org/pdf/1910.03892.pdf)

![image](res/panoptic_arch.png)
![image](res/panoptic_head.png)

