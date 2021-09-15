time: 20210605
pdf_source: https://arxiv.org/pdf/2106.01401v1.pdf
code_source: https://github.com/gaopengcuhk/Container
# Container: Context Aggregation Network

这篇paper从临近矩阵的形成方式的角度，将MLP Mixer, CNN以及transformer统一起来.

## Method:

一个残差网络层的计算表达可以抽象地表达为

$$Y = \mathcal{F}(X, \{W_i\}) + X$$

其中$W_i$是可学习的参数.

其中，定义关联矩阵$\mathcal{A} \in \mathbb{R}^{N \times N}$, 代表邻域的关注，那么网络层可以表达为

$$Y = (\mathcal{A}V)W_1 + X$$

其中$V\in\mathbb{R}^{N\times C} = XW_2$是X的一个线性投影.通过引入不同的关联矩阵， 这个模块的拟合capacity可以进一步提升，其中可以采用multi-head的版本

$$Y = \text{Concat}(\mathcal{A}_1V_1, ..., \mathcal{A}_MV_M)W_2 + X$$

### Typical instance of the Context Aggregation Module

**Transformer**: 
$$A_m^{sa} = \text{Softmax}(Q_mK^T_m / \sqrt{C/M})$$

**Depthwise Convolution**:

$$
\mathcal{A}_{m i j}^{\text {conv }}=\left\{\begin{array}{cl}
\operatorname{Ker}[m, 0,|i-j|] & |i-j| \leq k \\
0 & |i-j|>k
\end{array}\right.
$$

这个矩阵的形态是静态的，值是可以学习的.

**MLP-Mixer**:

其计算公式为$X=X + (V^TW_{MLP})^T$, 关联矩阵为

$$A^{mlp} = (W_{MLP})^T$$

因而这个矩阵是完全静态的且完全可学习的。但是完全没有参数共享。
```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., seq_l=196):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        #Uncomment this line for Container-PAM
        #self.static_a = nn.Parameter(torch.Tensor(1, num_heads, 1 + seq_l , 1 + seq_l))
        #trunc_normal_(self.static_a)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).
        permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        #Uncomment this line for Container-PAM
        #attn = attn + self.static_a

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
```