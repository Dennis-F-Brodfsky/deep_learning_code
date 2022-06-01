1-1
proof: $(A^T)^T = A$  
$$
B = A^T = (b_{ij}) \quad then   \\
b_{ji} = a_{ij} \\
C = B^T = (A^T)^T = (c_{ij}) \\
\therefore c_{ij} = b_{ji} = a_{ij} 
$$

1-2
proof: $A^T+B^T = (A+B)^T $ 

$$
A^T + B^T = (a_{ij})^T + (b_{ij})^T \\
= (a_{ij}+b_{ij})^T \\
= (A+B)^T
$$

1-3
$A$ 不一定对称；但是$A+A^T$一定对称：因为$C=A+A^T=(A^T+A)^T=C^T$

假设`torch.tensor.shape` 是 $(d_1, d_2, \cdots, d_n)$
`len(torch.tensor)`返回$d_1$；
`torch.tensor.sum(axis=1)`返回按照张量第一维度，加和所有元素的结果。结果的形状是$(d_1, d_3, \cdots, d_n) $
直接运行`A/A.sum(axis=1)` 会发生RuntimeError，表示张量broadcast失败，如果想正常运行，`sum`函数需要添加`keepdim=True`选项
`torch.norm`对高维张量依然可以运行，返回一个常量；
街道距离: 如果不能斜着走，应该使用曼哈顿距离对应L1范数的情况；欧式距离对应L2范数

**note**
- 只有`dtype`为`torch.float`的张量可以使用`torch.linag.norm`，如果需要传入非float张量，可以该函数dtype选项。
- `torch.linag.norm` 的dim选项，当传入张量是高维张量时，利用dim选项可以计算张量中向量的范数或者矩阵的范数；
- `torch.linag.norm`传入高维张量的情形，详见官网介绍，默认计算Flatten之后张量的2-范数。

Reference:
[`torch.linalg.norm`](https://pytorch.org/docs/stable/generated/torch.linalg.norm.html?highlight=norm#torch.linalg.norm)
