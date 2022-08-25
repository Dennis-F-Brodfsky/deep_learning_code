T1
batchsize减小到1，会导致单次读取加快，但是读取次数增加，所以可能不是性能最优的选择。
而且，batchsize为1，要注意和numworker契合，如果小于numworker可能会造成cpu闲置（具体和Pytorch实现方法有关）

T2
考虑利用torch.multiprocessing来并行读取

T3
Reference:
[Torchvision Datasets](https://pytorch.org/vision/stable/datasets.html)
