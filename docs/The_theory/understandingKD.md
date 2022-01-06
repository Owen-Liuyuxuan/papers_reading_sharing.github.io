time: 20220106
pdf_source: https://arxiv.org/pdf/2012.09816.pdf

# Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning 

这篇paper有微软研究院的一个[blog](https://www.microsoft.com/en-us/research/blog/three-mysteries-in-deep-learning-ensemble-knowledge-distillation-and-self-distillation/)作为介绍.

研究的主要问题以及切入点是为什么把网络ensemble起来，甚至使用自蒸馏，就能提升网络的测试性能。

## 基础结论

文章通过精巧地设计实验得到了数个有意思地结论。

| Conclusions         | Simple Ideas | 
| --------------- | :------: |
| 在深度学习中的模型组合与传统方法(类似[NTK]方法)很不同          |  NTK这种类似线性的模型在模型ensemble时不会有显著性能提升，因而网络的学习机制和NTK还是有很大区别,网络不只是在随机权重中对特征进行选择，而是在训练中真的有为数据提取新的特征   |
| 网络的ensemble不只是降低了预测的variance提高信心 | 网络在图像分类training set都是几乎无损失的，但是仍然能有测试准确度提升。 但是如果数据集是高斯采样的基础数据集，则没有这样的效果。是图像数据集中某些特定的分布特性造成的。|
| hard label对ensemble很重要，甚至是必要 | 如果使用KD 也就是软label来训练单个网络，那么这些网络的组合不会带来测试准确度提升| 

## Multi view Data
本文提出multi-view data这个说法来justify为什么ensemble以及KD有用.在图片上来说就是同一个分类我们可以有多种特征，多种视角来成功分类。Ensemble之后的模型就有了从多个视角，多个途径进行分类的能力，所以提升了准确度。

这个说法初听似乎是在深度学习比较早期就已经有的说法，但是本文设计了一个简单的数据集，具体地刻画了这个multi-view data的表现，并用简单的网络在这个简单的数据集上复现了上述的表现。

### Multi-view Data 的人工构建。
数据集包含四个特征$v_1, v_2, v_3, v_4$，是一个二分类问题。

label == 1
- 80%: $v_1, v_2$都是接近1，而$v_3, v_4$其中之一是0.1
- 10%: $v_1$都是接近1，而$v_3, v_4$其中之一是0.1
- 10%: $v_2$都是接近1，而$v_3, v_4$其中之一是0.1

label == 2
- 80%: $v_3, v_4$都是接近1，而$v_1, v_2$其中之一是0.1
- 10%: $v_3$都是接近1，而$v_1, v_2$其中之一是0.1
- 10%: $v_4$都是接近1，而$v_1, v_2$其中之一是0.1

其中 80%的数据被称为 multi-view data. 因为他们有不止一种方式分类。而 剩下的是 single-view data.

### 每个网络的学习特性, 融合， 蒸馏
作者发现，网络会在$v_1, v_2$中选择一个特征来正确分类label==1，以及$v_3, v_4$中选择一个来正确分类label==2. 因此90%的数据会很快地分类成功。训练一段时间后，这90%的数据就不再提供梯度了。

接着网络会尝试记忆剩余的10%训练样本，但是是依靠网络的噪音记忆能力而不是真正的找到了新的特征依据。 test set的准确率就卡在90左右

模型ensemble之后，由于随机初始化的网络是随机选择主特征的，因而有限个模型的融合，就可以凑齐所以需要的特征，test set准确率上到100%。

而KD的时候由于我们提供的是软label，所以网络实际上是被强行要求学习multi-view.



[NTK]:NTK.md
