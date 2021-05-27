time: 20210527
pdf_source: https://arxiv.org/pdf/2104.14294.pdf
code_source: https://github.com/facebookresearch/dino

# Emerging Properties in Self-Supervised Vision Transformers

[官方博客主页](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training)

![image](https://scontent-hkg4-1.xx.fbcdn.net/v/t39.2365-6/10000000_208749027394474_3359566853088796337_n.gif?_nc_cat=105&ccb=1-3&_nc_sid=ad8a9d&_nc_ohc=6dE7aiqAnfsAX8lno3k&_nc_oc=AQlDxAe-JqLq9DHiiuoK8hiI8bGfwjI_TDWWKbgwVSD625Uv7hiYP5N9Rai5UUfVHoE&_nc_ht=scontent-hkg4-1.xx&oh=b9861fbafae323fa30de496242bb5e84&oe=60D28B47)

这篇paper提出了自监督训练ViT, 给出的性能很高，接近于有监督的数据，且其输出的feature map性能很高。


## Related works

### Moco 

[pdf](https://arxiv.org/pdf/1911.05722.pdf) [code](https://github.com/facebookresearch/moco)

Moco 是一个自监督学习图片分类的框架，其算法如图:

![image](res/moco_arch.png)


### ViT

ViT具体参考[这篇文章](../other_categories/Summaries/Summary_ICLR_2021.md)


## Method

这篇paper的方法和MoCo有一定的相似性，很受Moco的启发，但是它把他用在transformer上.作者发现其transformer里面的分类token可以直接用于前景的segmentation.

![image](res/DINO_alg.png)
![image](res/dino_diag.png)

