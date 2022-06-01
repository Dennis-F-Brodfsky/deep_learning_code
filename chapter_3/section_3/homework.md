T1
Pytorch 官网的解释：
$$
\ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
l_n = \left( x_n - y_n \right)^2, \\
\ell(x, y) =
\begin{cases}
    \operatorname{mean}(L), &amp;  \text{if reduction} = \text{`mean';}\\
    \operatorname{sum}(L),  &amp;  \text{if reduction} = \text{`sum'.}
\end{cases}
$$
当loss的值出现差异时，需要对应调整学习率大小，使得两种训练方式参数迭代幅度一致。如果reduction是mean（一般在训练中，是对一个batch size的样本求平均），那么学习率就要调大一点；反之如果是sum，那么调小一点；

T2
详见reference

T3
访问网络的参数，就可以执行梯度裁剪、梯度重置、梯度保存等等操作。

Reference:
[MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html)
[HuberLoss](https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html)
