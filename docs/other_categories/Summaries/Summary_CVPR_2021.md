time: 20210401
short_title: CVPR 2021 clips

# Summaries for several CVPR 2021 papers

## OTA: Optimal Transport Assignment for Object Detection

[pdf](https://arxiv.org/pdf/2103.14259.pdf) [code](https://github.com/Megvii-BaseDetection/OTA)

OTA 这篇paper提出使用optimal transport assignment 做 ground truth label assignment.

前一年的CVPR一篇paper [ATSS](Summary_of_serveral_cvpr2020.md) 说明RetinaNet和FCOS之间的差距只要在于label的分配上. 这篇paper使用 [最优运输](Collections_StereoMatching_KITTI.md). 

![image](res/OTA_arch.png)

理解上来说，每一个ground truth作为一个 optimal transport 的 supplier 提供 $k$个正样本, 剩下的由background提供负样本. 然后每一个anchor 作为demander. 分配成本由损失函数决定

$$
\begin{aligned}
c_{i j}^{f g}=& L_{c l s}\left(P_{j}^{c l s}(\theta), G_{i}^{c l s}\right)+\\
& \alpha L_{r e g}\left(P_{j}^{b o x}(\theta), G_{i}^{b o x}\right)
\end{aligned}
$$

![image](res/OTA_algorithm.png)

这个assign 过程代码上是屏蔽反传的。


## Boundary IoU: Improving Object-Centric Image Segmentation Evaluation

[pdf](https://arxiv.org/pdf/2103.16562.pdf) [code](https://github.com/bowenc0221/boundary-iou-api)

![image](res/Bounrdary_iou.png)
![image](res/notation.png)

$G_d, P_d$定义为距离ground truth与预测的contours距离在$d$以内的像素.

![image](res/BoundaryIoUCompute.png)

实现上是使用cv2.copyMakeBorder以及cv2.erode。 [核心代码](https://github.com/bowenc0221/boundary-iou-api/blob/master/boundary_iou/utils/boundary_utils.py)


## GrooMeD-NMS: Grouped Mathematically Differentiable NMS for Monocular 3D Object Detection

[pdf](https://arxiv.org/pdf/2103.17202.pdf) [code](https://github.com/abhi1kumar/groomed_nms)

这篇paper提出了一个新的NMS算法，但是仅仅使用了2D等信息，但是仅evaluate在3D算法上.思路是让NMS变得可以训练

![image](res/group_nms_alg.png)
