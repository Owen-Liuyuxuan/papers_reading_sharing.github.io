pdf_source: https://arxiv.org/pdf/1811.08883.pdf
# Rethinking ImageNet Pre-training

这篇论文来自何凯明的论文讨论了pretraining对detection task的影响，分析了数个要素

## 主要结论
1. pretrained加速收敛
2. imagenet pretrained不一定提升regularization，除非原来数据集量真的很小
3. 当训练任务对位置信息非常敏感时，比如key-point检测，imagenet pretrained用处不大
   
## 其他技术细节
1. Normalization必不可少，但是由于Detection高清图要求高，显存不够，所以如果需要从头开始train batch normalization会因为batch太小影响效果，所以尝试GroupNorm等。
2. 对于数据量足够大的detection task来说，pretrain可以使结果更快收敛，但是random-initialization足够长epoch后得到的结果一般不会差于pretrain，当然要求有GN
3. 使用初始学习率(较大的学习率)，训练更长的时间是有用的，长时间使用低学习率提高准确率经常会导致overfitting

