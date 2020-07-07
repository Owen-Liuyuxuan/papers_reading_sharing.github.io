time: 20200707
code_source: https://github.com/aim-uofa/AdelaiDet
pdf_source: https://arxiv.org/pdf/2003.05664.pdf
short_title: Conditional Convolutions for Instance Segmentation
# Conditional Convolutions for Instance Segmentation

这篇paper给出了一点性能不错且代码相对规范简洁的One-Stage Instance Segmentation算法。 

## 网络结构

![image](res/CondInst_framework.png)

基于[FCOS](../object_detection_2D/FCOS.md)的检测架构，每一个scale的Shared head会预测分类，centerness等值。每一个被检测出来的物体会使用跟随Head一起估计的Convolution filter，对mask进行融合.

[Dynamic Mask Head代码](https://github.com/aim-uofa/AdelaiDet/blob/master/adet/modeling/condinst/dynamic_mask_head.py)


