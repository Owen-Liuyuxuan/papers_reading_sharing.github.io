time: 20200608

# Collections of Stereo Matching from KITTI

本文记录了 Stereo Matching 有文章/有code实现的主要paper.将会持续update


|   Methods     | D1-all   | D1-bg| D1-fg| Time |
|---------------|:--------:|:------:|:------:|:------:|
| [CSPN]        |  1.74    | 1.51 | 2.88 | 1.0  |
| [GANet-deep]  |  1.81    | 1.48 | 3.46 | 1.8  |
| [AcfNet]      |  1.89    | 1.51 | 3.80 | 0.48 |
| [AANet+]      |  2.03    | 1.65 | 3.96 | 0.06 |
| [DeepPruner]  |  2.15    | 1.87 | 3.56 | 0.18 |
| [PSMNet]      |  2.32    | 1.86 | 4.62 | 0.21 |
| [FADNet]      |  2.82    | 2.68 | 3.50 | 0.05 |
| [NVStereoNet] |  3.13    | 2.62 | 5.69 | 0.6  |
| [RTS2Net]     |  3.56    | 3.09 | 5.91 | 0.02 |
| [SsSMnet]     |  3.40    | 2.70 | 6.92 | 0.8  |

其中本站已有的文章为[CSPN],[AcfNet],[DeepPruner], [PSMNet], [FADNet], [SsSMnet], [RTS2Net].

Update: 2020.06.08: add RTS2Net
## GANet
[pdf](http://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_GA-Net_Guided_Aggregation_Net_for_End-To-End_Stereo_Matching_CVPR_2019_paper.pdf) [code](https://github.com/feihuzhang/GANet)

![image](res/GANet_arch.png)

Feature Extraction使用的是stacked hourglass network. Cost Volume的形成与[PSMNet]一致。然后接数个SGA模块，以及LGA模块。左图会接上"guidance subnet"使用数个简单卷积生成权重矩阵提供到GA模块中。

GA层对应scanline optimization方法 [ref1](https://core.ac.uk/download/pdf/11134866.pdf) [ref2](https://www.tugraz.at/fileadmin/user_upload/Institute/ICG/Documents/courses/robotvision/2019/RV_StereoMatching.pdf),可以理解为一个动态规划算法,其中本文的$\mathbf{r}$为四个方向的矢量。里面的权重是每一个像素不一致的，通过subnet提供guidance.

Semi-Global Guided Aggregation(SGA) 需要的guidence权重大小为$H\times W \times K\times F(K=5)$，不同disparity使用的权重一致:
$$C_{\mathbf{r}}^{A}(\mathbf{p}, d)=\operatorname{sum}\left\{\begin{array}{l}
\mathbf{w}_{0}(\mathbf{p}, \mathbf{r}) \cdot C(\mathbf{p}, d) \\
\mathbf{w}_{1}(\mathbf{p}, \mathbf{r}) \cdot C_{\mathbf{r}}^{A}(\mathbf{p}-\mathbf{r}, d) \\
\mathbf{w}_{2}(\mathbf{p}, \mathbf{r}) \cdot C_{\mathbf{r}}^{A}(\mathbf{p}-\mathbf{r}, d-1) \\
\mathbf{w}_{3}(\mathbf{p}, \mathbf{r}) \cdot C_{\mathbf{r}}^{A}(\mathbf{p}-\mathbf{r}, d+1) \\
\mathbf{w}_{4}(\mathbf{p}, \mathbf{r}) \cdot \max _{i} C_{\mathbf{r}}^{A}(\mathbf{p}-\mathbf{r}, i)
\end{array}\right.$$
$$\text {s.t.} \quad \sum_{i=0,1,2,3,4} \mathbf{w}_{i}(\mathbf{p}, \mathbf{r})=1$$
$$C^{A}(\mathbf{p}, d)=\max _{\mathbf{r}} C_{\mathbf{r}}^{A}(\mathbf{p}, d)$$

Local Aggregation(LGA),需要的guidence权重大小为$H\times W\times 3K^2 \times F$:
$$\begin{array}{c}
C^{A}(\mathbf{p}, d)=\operatorname{sum}\left\{\begin{array}{l}
\sum_{\mathbf{q} \in N_{\mathrm{p}}} \omega_{0}(\mathbf{p}, \mathbf{q}) \cdot C(\mathbf{q}, d) \\
\sum_{\mathbf{q} \in N_{\mathrm{p}}} \omega_{1}(\mathbf{p}, \mathbf{q}) \cdot C(\mathbf{q}, d-1) \\
\sum_{\mathbf{q} \in N_{\mathrm{p}}} \omega_{2}(\mathbf{p}, \mathbf{q}) \cdot C(\mathbf{q}, d+1)
\end{array}\right. \\
\text { s.t. } \sum_{\mathbf{q} \in N_{\mathrm{p}}} \omega_{0}(\mathbf{p}, \mathbf{q})+\omega_{1}(\mathbf{p}, \mathbf{q})+\omega_{2}(\mathbf{p}, \mathbf{q})=1
\end{array}$$



## AANet
[pdf](https://arxiv.org/pdf/2004.09548.pdf) [code](https://github.com/haofeixu/aanet)

![image](res/AANet_arch.png)

本文使用coorelation的方式生成3D Cost Volume.

Adaptive Intra-Scale Aggregation本质上是分组的可变卷积：

$$\tilde{\boldsymbol{C}}(d, \boldsymbol{p})=\sum_{k=1}^{K^{2}} w_{k} \cdot \boldsymbol{C}\left(d, \boldsymbol{p}+\boldsymbol{p}_{k}+\Delta \boldsymbol{p}_{k}\right) \cdot m_{k}$$

多scale融合，这里采用的是[HRNet](../../Building_Blocks/HRNet.md)的方法
$$\hat{\boldsymbol{C}}^{s}=\sum_{k=1}^{S} f_{k}\left(\tilde{\boldsymbol{C}}^{k}\right), \quad s=1,2, \cdots, S$$

$$f_{k}=\left\{\begin{array}{l}
\mathcal{I}, \quad k=s \\
(s-k) \text { stride }-2\oplus 3 \times 3 \text { convs, } \quad k<s \\
\text { upsampling } \oplus 1 \times 1 \text { conv, } \quad k>s
\end{array}\right.$$


## NVStereoNet

[pdf](http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w14/Smolyanskiy_On_the_Importance_CVPR_2018_paper.pdf)
[code](https://github.com/NVIDIA-AI-IOT/redtail/tree/master/stereoDNN)

![image](res/nv_stereo_arch.png)

损失与[monodepth](../others/Unsupervised_depth_prediction.md)相似

$$L=\lambda_{1} E_{\text {image}}+\lambda_{2} E_{\text {lidar}}+\lambda_{3} E_{l r}+\lambda_{4} E_{d s}$$

$$\begin{aligned}
E_{\text {image}} &=E_{\text {image}}^{l}+E_{\text {image}}^{r} \\
E_{\text {lidar}} &=\left|d_{l}-\bar{d}_{l}\right|+\left|d_{r}-\bar{d}_{r}\right| \\
E_{l r} &=\frac{1}{n} \sum_{i j}\left|d_{i j}^{l}-\tilde{d}_{i j}^{l}\right|+\frac{1}{n} \sum_{i j}\left|d_{i j}^{r}-\tilde{d}_{i j}^{r}\right| \\
E_{d s} &=E_{d s}^{l}+E_{d s}^{r}
\end{aligned}$$

$$\begin{aligned}
E_{\text {image}}^{l} &=\frac{1}{n} \sum_{i, j} \alpha \frac{1-\operatorname{SSIM}\left(I_{i j}^{l}, \tilde{I}_{i j}^{l}\right)}{2}+(1-\alpha) | I_{i j}^{l}-\tilde{I}_{i j}^{l} \\
E_{d s}^{l} &=\frac{1}{n} \sum_{i, j}\left|\partial_{x} d_{i j}^{l}\right| e^{-\left\|\partial_{x} I_{i, j}^{l}\right\|}+\left|\partial_{y} d_{i j}^{l}\right| e^{-\| \partial_{y} I_{i, j}^{l}} \|
\end{aligned}$$


[CSPN]:../../Building_Blocks/SPN_CSPN.md
[AcfNet]:../others/Adaptive_Unimodal_Cost_Volume_Filtering_for_Deep_Stereo_Matching.md
[DeepPruner]:../../Building_Blocks/deepPruner.md
[PSMNet]:../others/PSMNet.md
[FADNet]:../others/FADNet.md
[SsSMnet]:../others/self_supervised_stereo.md
[GANet-deep]:#ganet
[AANet+]:#aanet
[NVStereoNet]:#nvstereonet
[RTS2Net]:Summary_of_ICRA_2020.md