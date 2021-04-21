time:20210421
code_source: https://github.com/nianticlabs/monodepth2
pdf_source: https://arxiv.org/pdf/1806.01260.pdf

# Collections on Monodepth (unsupervised)

## MonoDepth2

[pdf](https://arxiv.org/pdf/1806.01260.pdf) [code](https://github.com/nianticlabs/monodepth2)

MonoDepth2是非监督单目深度估计的一个Baseline， 

![image](res/monodepth2_arch.png)

主要几个思路:

1. 简单的res18以及decoder出深度估计结果.
2. 图片并接直接输出相对pose
3. 用前一帧或者后一帧或者双目的图片重建处当前帧.
4. 重建loss使用 SSIM, 且选择重建的min的loss，重建损失过大的遮挡部分被滤掉了.


## Full Surround Monodepth from Multiple Cameras

[pdf](https://arxiv.org/pdf/2104.00152.pdf)

![image](res/fsm_example.png)

![image](res/fsm_transform.png)

1. 提出做多摄像机的深度估计
2. 考虑时序+空间两个方向的consistency
3. 需要self-occlusion mask去掉相机内车体相关的部分，需要non-overlapping areas去除相机之间不重叠的部分.

## Towards Good Practice for CNN-Based Monocular Depth Estimation

[pdf](https://openaccess.thecvf.com/content_WACV_2020/papers/Fang_Towards_Good_Practice_for_CNN-Based_Monocular_Depth_Estimation_WACV_2020_paper.pdf) [code](https://github.com/zenithfang/supervised_dispnet)




## Monocular Depth Prediction through Continuous 3D Loss

[pdf](https://arxiv.org/pdf/2003.09763.pdf) [code](https://github.com/minghanz/c3d)


