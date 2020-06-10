time: 20200610
pdf_source: https://arxiv.org/pdf/2006.04388v1.pdf
code_source: https://github.com/implus/GFocal
short_title: Generalized Focal Loss
# Generalized Focal Loss: Learning Qualified and Distributed Bounding Boxes for Dense Object Detection

这篇paper的初衷是分析IoU Centerness与classification loss的相关问题，在NMS的时候，我们使用的是IoU Centerness和cls Score的乘积，但是训练的时候，cls Score使用focal loss而IoU Centerness被视为回归问题。这就构成了一定的不匹配。

本文借此首先提出了 Quality Focal loss,使得focal loss可以统一到"连续的分类问题",其二本文进一步考虑将这个概念进行扩展到Distribution Focal Loss，可以拟合任意loss，最后提出的Generalized Focal loss融合多种情况。

## Focal loss
$$\mathbf{F L}(p)=-\left(1-p_{t}\right)^{\gamma} \log \left(p_{t}\right), p_{t}=\left\{\begin{aligned}
p, & \text { when } y=1 \\
1-p, & \text { when } y=0
\end{aligned}\right.$$
<div id="focalLoss" ></div>


## Quality Focal Loss

$$\mathbf{Q} \mathbf{F} \mathbf{L}(\sigma)=-|y-\sigma|^{\beta}((1-y) \log (1-\sigma)+y \log (\sigma))$$

<div id="qfocalLoss" ></div>

```python
def quality_focal_loss(
          pred,          # (n, 80)
          label,         # (n) 0, 1-80: 0 is neg, 1-80 is positive
          score,         # (n) reg target 0-1, only positive is good
          weight=None,
          beta=2.0,
          reduction='mean',
          avg_factor=None):
    """
        from https://github.com/implus/GFocal/blob/cc0e72680f16a8abe0770eb531d6baa07a6e511f/mmdet/models/losses/gfocal_loss.py
    """
    # all goes to 0
    pred_sigmoid = pred.sigmoid()
    pt = pred_sigmoid
    zerolabel = pt.new_zeros(pred.shape)
    loss = F.binary_cross_entropy_with_logits(
           pred, zerolabel, reduction='none') * pt.pow(beta)

    label = label - 1
    pos = (label >= 0).nonzero().squeeze(1)
    a = pos
    b = label[pos].long()
    
    # positive goes to bbox quality
    pt = score[a] - pred_sigmoid[a, b]
    loss[a,b] = F.binary_cross_entropy_with_logits(
           pred[a,b], score[a], reduction='none') * pt.pow(beta)

    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

```

## Distribution Focal Loss (DFL)

当我们使用序列的多个分类值(multi-bin)分类时，inference的时候我们使用 $\hat{y}=\sum_{i=0}^{n} P\left(y_{i}\right) y_{i}$。 $$\mathbf{D F L}\left(\mathcal{S}_{i}, \mathcal{S}_{i+1}\right)=-\left(\left(y_{i+1}-y\right) \log \left(\mathcal{S}_{i}\right)+\left(y-y_{i}\right) \log \left(\mathcal{S}_{i+1}\right)\right)$$

其中 $\mathcal{S}_{i}=\frac{y_{i+1}-y}{y_{i+1}-y_{i}}, \mathcal{S}_{i+1}=\frac{y-y_{i}}{y_{i+1}-y_{i}}$,直觉就是按照权重使用loss鼓励multibin ground truth临近的两侧分类点的权重。

```python
def distribution_focal_loss(
            pred,
            label,
            weight=None,
            reduction='mean',
            avg_factor=None):
    """
        from https://github.com/implus/GFocal/blob/cc0e72680f16a8abe0770eb531d6baa07a6e511f/mmdet/models/losses/gfocal_loss.py
    """
    disl = label.long()
    disr = disl + 1

    wl = disr.float() - label
    wr = label - disl.float()

    loss = F.cross_entropy(pred, disl, reduction='none') * wl \
         + F.cross_entropy(pred, disr, reduction='none') * wr
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss
```

## Generalized Focal Loss (GFL)

$$\mathbf{G F L}\left(p_{y_{l}}, p_{y_{r}}\right)=-\left|y-\left(y_{l} p_{y_{l}}+y_{r} p_{y_{r}}\right)\right|^{\beta}\left(\left(y_{r}-y\right) \log \left(p_{y_{l}}\right)+\left(y-y_{l}\right) \log \left(p_{y_{r}}\right)\right)$$


<script>
function focal_loss_y(x, gamma){
    return - Math.pow(1-x, gamma) * Math.log(x)
}

function quality_focal_loss_y(x, y){
    return - Math.pow(y-x, 2) * ((1-y) * Math.log(1-x) + y * Math.log(x))
}


function get_focal_loss_list(p, gamma){
    focal = []
    for (j = 0; j < 98;j++){
        focal.push(focal_loss_y(p[j], gamma))
    }
    return focal
}
function get_quality_focal_loss_list(p, y){
    focal = []
    for (j = 0; j < 98;j++){
        focal.push(quality_focal_loss_y(p[j], y))
    }
    return focal
}
focalLoss = document.getElementById('focalLoss');
var p = [];
for (i = 1; i < 99;i++){
    p.push(i * 0.01);
}
var focal = get_focal_loss_list(p, 0.2)
slider_steps = []
for (i = 0.2; i < 4; i += 0.2){
    slider_steps.push(
        {
            method: 'animate',
            label: Math.floor(i * 100) /100,
            args: [
                {
                    data: [{ x:p, y: get_focal_loss_list(p, i)}],
                },
                {
                transition: {duration: 20},
                frame: {duration: 20, redraw: false},
                }
            ]
        }
    )
}

var qfocal = get_quality_focal_loss_list(p, 0.5)
qslider_steps = []
for (i = 0.02; i < 0.98; i += 0.02){
    qslider_steps.push(
        {
            method: 'animate',
            label: Math.floor(i * 100) /100,
            args: [
                {
                    data: [{ x:p, y: get_quality_focal_loss_list(p, i)}],
                },
                {
                transition: {duration: 20},
                frame: {duration: 20, redraw: false},
                }
            ]
        }
    )
}



Plotly.plot(focalLoss, [{
    x: p,
    y: focal,
}], {
    title: 'Focal Loss for positive samples',
    xaxis: {
        title: 'p'
    },
    yaxis: {
        title: 'loss'
    },
    sliders: [{
    pad: {t: 30},
    currentvalue: {
      xanchor: 'right',
      prefix: 'gamma: ',
      font: {
        color: '#888',
        size: 20
      }
    },
    steps: slider_steps
  }]
});

Plotly.plot(qfocalLoss, [{
    x: p,
    y: qfocal,
}], {
    title: 'Quality Focal Loss with beta=2',
    xaxis: {
        title: 'p'
    },
    yaxis: {
        title: 'loss'
    },
    sliders: [{
    pad: {t: 30},
    currentvalue: {
      xanchor: 'right',
      prefix: 'gt_y: ',
      font: {
        color: '#888',
        size: 20
      }
    },
    steps: qslider_steps
  }]
});

</script>



