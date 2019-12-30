time:20191230
pdf_source: https://arxiv.org/pdf/1912.08035.pdf
short_title: Mono3d with virtual cameras

# Single-Stage Monocular 3D Object Detection with Virtual Cameras

这篇文章来自于[MonoDIS]的作者组，采用的是virtual camera的方法，得到了相对不错的结果

## virtual camera

![image](res/mono3d_virtualcam.png)
![image](res/mono3d_virtualcam_inference.png)

核心思路，在图中核心区域crop出多个有效区域，然后在里面进行3D detection，重要的有几个insight.

1. 思路与传统的RCNN有相似之处，也就是使用传统方法(根据3D空间遍历或者其他提示)，从原图中crop出有效框再进行分析，区别在于crop出来的每一张子都还可能有多个target
2. 每一个3Ddetection预测的深度是相对于虚拟相机位置的深度，突出一个scale invariance.

virtual camera具体的实现trick较多，超参数很多，若想要复现，这里建议回看论文的第4章节以及第6.2章节。由于欠缺代码，这个结果比较难以确认。

## DL structure

![image](res/mono3d_virtualcam_structure.png)

网络的输入输出与[MonoDIS]是一样的。


[MonoDIS]:./Disentangling_Monocular_3D_Object_Detection.md