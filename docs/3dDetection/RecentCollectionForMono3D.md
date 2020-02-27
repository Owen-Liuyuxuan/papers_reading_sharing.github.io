time: 20200227
short_title: Recent Collection for Mono3D

# Recent Collections for Mono 3D detection

在IROS2020投稿前后积攒了一系列单目3D检测paper的阅读。这里一次过进行记录,开源在前，未开源在后.

这里列出目前有文章可寻的KITTI排行榜(2020.02.27)


|   Methods  | Moderate |  Easy | Hard | Time |
|----------|:--------:|:------:|:------:|:------:|
| [D4LCN]      | 11.72    | 16.65 | 9.51 | 0.2  |
| [Refined-MPL] | 11.14    | 18.09 | 8.94 | 0.15 |
| [AM3D]       | 10.74    | 16.50 | 9.52 | 0.4  |
| [YOLOMono3D] | 10.59    | 17.18 | 7.09 | 0.05 |
| [RTM3D]      | 10.34    | 14.41 | 8.77 | 0.05 |
| [MonoPair]   | 9.99     | 13.04 | 8.65 | 0.06 |
| [SMOKE]      | 9.76     | 14.03 | 7.84 | 0.03 |
| [M3D-RPN]    | 9.71     | 14.76 | 7.42 | 0.16 |

## D4LCN
[pdf](https://arxiv.org/pdf/1912.04799.pdf)  [code](https://github.com/dingmyu/D4LCN)


这篇paper完全继承了[M3D-RPN]的衣钵，它不同的地方在于，摒弃了M3D-RPN处理缓慢的height-wise convolution,而是使用单目估计深度，然后使用深度作为卷积核的guide, 这个guide类似于这几篇文章的操作:[guidenet](../other_categories/depth_completion/guideNet.md)[DFN](../Building_Blocks/DynanicFilteringNetwork.md)

![image](res/D4LCN.png)

## RTM3D
[pdf](https://arxiv.org/pdf/2001.03343.pdf) [code](https://github.com/Banconxuan/RTM3D)

这篇文章还没有正式开源，但是github就先开着了。这篇文章在技术上有一定的新意，它使用[CenterNet]的架构估计大量的keypoints以及冗余的3D信息，最后通过最优化融合。使用大量冗余信息它不是第一个,前者比如有[SS3D](Monocular_3D_Object_Detection_and_Box_Fitting_Trained_End-to-End_Using_Intersection-over-Union_Loss.md),但是它绕过了anchor使用CenterNet有一定的新意。
![image](res/RTM3D_0.png)
![image](res/RTM3D_2.png)


## MonoPair

这篇文章是实验室大师兄邰磊在阿里的CVPR2020 paper。目前还没有挂上arxiv，所以现在(2020/02/27)还不可以传播图片与PDF。

文章的核心创新是第一个使用场景中不同物体之间的相互约束进行优化的paper。

## SMOKE

[pdf](https://arxiv.org/pdf/2002.10111v1.pdf)

这篇paper的创新点不算特别多。
1. 使用了[CenterNet]的架构进行中心点的估计。
2. 使用了distangling loss, 这个来自于[MonoDIS]
3. 数据增强上使用了shifting等方法，但是只是用来train keypoint热图等结构。属于specialized augmentation for specialized cost.可谓深度调参

![image](res/SMOKE.png)

## YOLOMono3D

不多说了，快上车

<iframe src="//player.bilibili.com/player.html?aid=91364947&cid=156014191&page=1" scrolling="no" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>



[M3D-RPN]:M3D-RPN_Monocular_3D_Region_Proposal_Network_for_Object_Detection.md
[D4LCN]:##D4LCN
[Refined-MPL]:./RefinedMPL.md
[AM3D]:Accurate%20Monocular%20Object%20Detection%20via%20Color-Embedded%203D%20Reconstruction%20for%20Autonomous%20Driving.md
[RTM3D]:##RTM3D
[MonoPair]:##MonoPair
[SMOKE]:##SMOKE
[YOLOMono3D]:##YOLOMono3D
[CenterNet]:../other_categories/object_detection_2D/CenterNet:_Keypoint_Triplets_for_Object_Detection.md
[MonoDIS]:Disentangling_Monocular_3D_Object_Detection.md