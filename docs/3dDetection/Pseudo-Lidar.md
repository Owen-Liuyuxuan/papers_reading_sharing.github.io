pdf_source: https://arxiv.org/pdf/1903.09847.pdf
short_title: Pseudo-Lidar with Instance Segmentation
# Monocular 3D Object Detection with Pseudo-LiDAR Point Cloud

这篇论文有多个重要贡献，一是使用单目预测深度图，形成假Lidar数据，并得到人工电云，然后使用[Frustum Pointnet](Frustum_PointNets_for_3D_Object_Detection_from_RGB-D_Data.md)。对得到一个可靠的初始解。

第二由于假Lidar有很多噪点，噪音体现在两个方面，一个是有偏移，影响对距离的估计，一个是有长尾——物体边缘的一些点云会拉得很长，原因是物体边缘处的深度估计不准确。 为了缓解第一个问题，使用2D-3D box约束调整3D box的位置，这里引入了一个loss函数在training过程中处理这个问题，在inference的时候将问题转为优化再提高性能。 为了缓解第二个问题，使用instance mask来代表2D proposal而不是bounding box。相当于用segmentation的像素点级的结果输出

## 2D 3D约束

将3D框转换为8个坐标点，然后转换为图片坐标，求出最小bounding rectangle对应的四个坐标。BBCL就是这四个坐标与2Dbounding box的L1距离。BBCO则是在inference的时候使用global search的方式(多半是模拟退火)有货对应的L1距离。

这是一年前的SOTA