pdf_source: https://arxiv.org/abs/1904.08377
short_title: Gaze Training - Chen Yuying
# Gaze  Training  by  Modulated  Dropout  Improves  Imitation  Learning

这篇论文源自于实验室师姐。

核心贡献，用encoder-decoder训练一个Gaze_map生成网络(数据来自于人工标注),然后在模仿学习的时候使用Gaze-modulated Dropout,这个模块的思路是在使用dropout的时候，减少gaze_map相关部分的dropout，就像人眼注视单一区域一样。注意这里使用的dropout是tensorflow默认版本的dropout而不是pytorch默认的dropout2d(spatial-dropout).
