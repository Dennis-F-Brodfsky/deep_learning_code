T1
可以考虑一个情况，向量一阶求导和二阶hessian矩阵求导。前者问题规模是$O(n)$,后者问题的规模是$O(n^2)$；此外二阶导是在一阶导的基础上进行的，开销自然大于一阶导。
T2
再次运行`backward`函数，如果设置了`retain_graph=True`，那么会出现梯度累积的情况；如果采用默认设置，那么会出现RuntimeError，因为Pytorch中，计算图在求导一次之后会被丢弃。
T3
如果计算导数的函数传入向量返回向量，那么不能直接`backward`，如果想计算梯度，需要求sum再backward。原理类似于矩阵求trace再对矩阵自身求导。得到的结果刚好是，对应的梯度。
T4
torch与matplotlib结合，需要注意`torch.Tensor`需要先detach()，再取numpy()，转化为一个普通numpy数组。

Reference
[`torch.Tensor.backward`](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html?highlight=backward#torch.Tensor.backward)
[`torch.Tensor.detach`](https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html?highlight=detach#torch.Tensor.detach)
