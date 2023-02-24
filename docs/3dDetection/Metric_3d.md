time: 20200908

# 3D detection evaluation metric

本文主要尝试综述3D检测的评价方法以及对应的一些code的分析，目标是比较细致的分析，在有新metric提出的过程中会持续更新，先综述的是KITTI与Nuscene两大benchmark的评价分数。

Update 2020.09.08: Add Cityscapes 3D

## Average Precision - Kitti

[Official cpp code](https://github.com/KleinYuan/kitti-eval) <br>
[fast numba code](https://github.com/traveller59/kitti-object-eval-python)

整体思路上来说，KITTI的测试是将2D，BEV以及3D检测的评价过程完全分开。因而三种evaluation中间需要的matching是不相关的，但是由于有相似的搜索思路，因而代码实现上有一定的复用。

### 匹配(true positive判定)

KITTI的matching是从ground truth boxes出发，简单地循环，贪婪地寻找与其IoU(使用什么IoU由当前任务决定)最大的prediction.但是要注意的是，要计算AP,还是要考虑score的高低的。官方的实现与加速实现的思路是一致的。
先考虑所有prediction,计算一遍matching，并记录每一个matching的score，然后在PR曲线上采样41个点，得到41个confidence阈值,对于每一个阈值，滤掉比这个阈值更低的boxes，然后重新match，计算这个点上的recall（这个会与采样点一致） precision.以此刻画pr曲线。

### Easy, Medium, Hard

KITTI一个很特殊的机制在于分辨了 easy, medium与hard的结果。从代码实现上，可以发现这是一个以2D以及遮挡为主要根据的分别。Easy包含2D框高度大于40 pixs，遮挡等级最低的objects；Medium包含2D框高于25 pixs，遮挡等级 0,1的物体；Hard包含2D框高于25 pixs，遮挡等级0-2的物体(也包含前面提及的所有物体)。

### 关于加速

前面提及的Fast numba code相对于原来的cpp代码在代码核心逻辑上差距不大。核心变化主要有两个，第一个是使用numba.cudaGPU加速了3D IoU的计算,本质上是简单的并行运算(单个物体的算法并没有改变)。第二个是利用了numpy以及numba.jit的CPU并行优化，代码中有一个函数的作用是calculate_iou_partly,尽管实际运算量提升了，但是减少python循环的次数，且充分利用numpy以及numba.jit的并行优化。

## nuScenes detection score(NDS) - nuScenes

[pdf](https://arxiv.org/pdf/1903.11027.pdf) [code](https://github.com/nutonomy/nuscenes-devkit)

nuscene 的评价metric是相当独特的，作者的原意是希望有metric分别表达对中心距离、朝向、大小甚至速度等细项的计算结果。最后的NDS单一数值会是多个metric的均值。

### 匹配(true positive判定)

nuscene的matching是从sorted predicted boxes出发，按confidence大小循环(注意这里的排序以及循环是以所有帧的预测bounding box为准，而匹配的时候根据sample_token只在同一帧的gt中进行匹配， sample_token也是这个方法能方便地实现的原因，如果对KITTI结果要用同样的算法，则需要对每一个box都记录自己在哪一帧)，贪婪地寻找尚未被匹配的且与其平面距离最近的ground truth，如果最近的距离在阈值内，则认为是一对true positive。由于这个match已经是从高confidence到低confidence了，所以可以直接计算precisions跟recall曲线。

Thresholds分四次，分别取$[0.5, 1, 2, 4]$,得到的mAP取均值。此外以$threshold=2$时的匹配为准，计算其他 True positive metrics(TP metrics)

```python
"""
    tp: List[Union[0, 1]], len(tp) = num_prediction
    fp: List[Union[1, 0]], len(fp) = num_prediction fp[i] = 1 - tp[i],相当于
    conf: List[float], len(conf) = num_prediction (sorted)
        They recorded whether each predicted object is true positive or false positive, and also its score.
    npos: number of positive ground truth
"""
tp_array = np.cumsum(tp).astype(np.float)
fp_array = np.cumsum(fp).astype(np.float)
conf_array = np.array(conf)

prec = tp_array / (fp_array + tp_array)
rec = tp_array / float(npos)

rec_interp = np.linspace(0, 1, 101)              # sample 0,0.01,0.02,..0.99,1;totally 101 samples
prec = np.interp(rec_interp, rec, prec, right=0) # 1d-interpolate
conf = np.interp(rec_interp, rec, conf, right=0) # np.interp(x_coordinate, x_data, y_data) -> interpolated_y
rec = rec_interp
```

### TP Metrics

论文中给出了五个metrics，分别是

1. 中心点平面直线距离
2. scaled IoU (假设位置与方向正确，predicted whl长方体与gt whl长方体的iou)
3. yaw角差值(radian)
4. 2D速度差值(m/s)
5. 细分类分类准确度(nuscene对部分类别会继续细分)

各个TP值为误差值的在各个recall点上的累积均值的均值。

$$\mathrm{NDS}=\frac{1}{10}\left[5 \mathrm{mAP}+\sum_{\mathrm{mTP} \in \mathrm{TP}}(1-\min (1, \mathrm{mTP}))\right]$$

## Average Precision Weighted by Heading(APH) - Waymo

[页面](https://waymo.com/open/challenges/3d-detection/#)

waymo的算法与KITTI的极度相似，区别在于:

1. Easy/Difficult分辨方法主要是遮挡程度以及box内部点云的数量。因而是一个完全的3D-oriented的分类标准.
2. 每当发现一个true-positive matching, $tp = \frac{\Delta\theta}{\pi}$,相当于只有角度是准确的才能得到完整的一个true-positive,否则会加上一个惩罚权重。而False positive和False negative没有变化。


## mean Detection Score - Cityscapes 3D

[pdf](https://arxiv.org/abs/2006.07864) [code](https://github.com/mcordts/cityscapesScripts)

Cityscapes 基于他原有的数据也发布了一个三维检测数据标注集。与之前的数据最大的不同有二，首先是它仅使用双目数据进行标注，其次是它标注了3个维度的旋转。这个数据集的设计就是为了**评价单目的3D检测的，因而不采用3D IoU而选择了不同的设计**。

其评价指标 mDS 由以下几个参数组成:

- 2D AP: 与图像2D一致, $IoU \geq 0.7$
- Center Distance: 俯瞰图距离 $\mathrm{BEVCD}$.
- Yaw Similarity: yaw角
- Pitch-Roll Similarity: pitch-roll作者认为在无人驾驶场景往往是耦合的，因而要放在一起评判.
- Size Similarity: 大小 whl

有几个点需要注意:

- AP的计算根据代码，在PR曲线上采样了50个点.
- 后面四项的计算，都是depth-dependent的。代码上每5米分一个bin，统计这个区间内的match,根据这个区间的match计算对应四个项目中该bin的score，然后每个项目会对各个bin求平均.
- 后面四项计算的时候，confidence threshold是固定的，2D AP计算的时候会通过变化confidence threshold在一系列的recall值上计算aP.这里的confidence threshold固定为 $c_{w}=\underset{c \in|0,1|}{\operatorname{argmax}} p(c) r(c)$ 作者的intuition是说这个评价方案和现实部署的时候更为一致(我们会直接采用一个平衡好recall and precision的 threshold给出预测).

BEVCD的计算:其中$X_{max}$是100米
$$
\begin{aligned}
\mathrm{BEVCD}&=1-\frac{1}{\mathrm{X}_{\max }^{2}} \int_{0}^{\mathrm{X}_{\max }} k(s) \mathrm{d} s
 \\ 
k(s)&=\frac{1}{\mathrm{N}} \underset{d, g \in \mathrm{D}\left(s, c_{w}\right)}{\sum} \min \left(\mathrm{X}_{\max }, \sqrt{\sum_{i \in|x, y|}\left(d_{i}-g_{i}\right)^{2}}\right)
\end{aligned}
$$

Yaw similarity 计算:

$$
\begin{aligned}
\text { Yawsim }&=\frac{1}{X_{\text {max }}} \int_{0}^{X_{\max }} k(s) \mathrm{d} s
\\
k(s)&=\frac{1}{N} \sum_{d, g \in D\left(s, c_{w}\right)} \frac{1+\cos \left(\Delta_{Y a w}\right)}{2}
\end{aligned}
$$

Pitch-Roll Similarity 计算:

$$
\begin{aligned}
\text { PRSim }&=\frac{1}{X_{\text {max }}} \int_{0}^{X_{\max }} k(s) \mathrm{d} s
\\
k(s)&=\frac{1}{N} \sum_{d, g \in D\left(s, c_{w}\right)} \frac{2+\cos \left(\Delta_{Pitch}\right) + \cos \left(\Delta_{Roll}\right) }{4}
\end{aligned}
$$

Size Similarity 计算:
$$\begin{aligned}
\text { SizeSim }&=\frac{1}{X_{\text {max }}} \int_{0}^{X_{\max }} k(s) \mathrm{d} s
\\
k(s)&=\frac{1}{N} \sum_{d, g \in D\left(s, c_{w}\right)} \prod_{x\in \{l, w, h\}} \min \left(\frac{d_{x}}{g_{x}}, \frac{g_{x}}{d_{x}}\right)
\end{aligned}
$$

$$
DS = AP \times \frac{BEVCD + YawSim + PRSim + SizeSim}{4}
$$