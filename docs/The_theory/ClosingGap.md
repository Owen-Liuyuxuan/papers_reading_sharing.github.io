time: 20200415
pdf_source: https://arxiv.org/pdf/1705.08741.pdf
code_source: https://github.com/eladhoffer/bigBatch
# Train longer, generalize better: closing the generalization gap in large batch training of neural networks

这篇paper从理论以及实验上对训练过程以及Generalization进行了分析。给出了一些有趣的结论以及有趣的module。

##　以前的paper的一些结论

1. 用大batch训练，generalization error会提升
2. 即使网络训练很长时间，这个损失也不会变化
3. 好的generalization对应平滑的minima
4. 小batch能让权重更远离初始值。

## 大batch中模仿小batch的统计数据

### 训练时长

作者分析认为，在训练的初期，在网络不发散的情况下，网络的random-walk会极慢地远离初始值，得到更好的结果。建议的实现方法是尽可能大的不发散的学习率，足够大的学习步长。

### 学习率应正比于batchsize的平方根

这样的原由是对应相同的参数升级协方差

$$\operatorname{cov}(\Delta \mathbf{w}, \Delta \mathbf{w}) \approx \frac{\eta^{2}}{M}\left(\frac{1}{N} \sum_{n=1}^{N} \mathbf{g}_{n} \mathbf{g}_{n}^{\top}\right)$$

### 使用Ghost BatchNorm模仿小batch时数据的统计特性。

根据Tensorflow Batchnorm的[官方API文档](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization?hl=en),设定
```python
layer = tf.keras.layers.BatchNormalization(*args, virtual_batch_size=<any_number>)
```
可以让batchnorm对大batch拆分，分成大小为$<any\_number>$的小batch运行batchnorm

### 训练loss趋于平滑后不要畏惧overfitting， 继续train

作者实验判断的认为是，大batch性能差的主要原因是update的次数不够多(旁白:这个有些时候很真实)。
