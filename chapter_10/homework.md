## ch10-sec1

T1
自主性寻找深层次不明显的语言序列特征(这个由RNN encoder部分获取，在第九章的例子中)；非自住性是词元数字表示本身，大部分是出现频率高的词元类（这个来自于target语言的embedding数据）；

T2
code

## ch10-sec2
T1
使用200样本进行实现；发现loss变大；观察拟合效果发现，样本变大，模型倾向于缩减窗口，导致了类似过拟合的情况；观察模型attention weight发现，样本量增加导致模型倾向于使用很近的样本构造attention weight；间接导致窗口宽度的缩小。

T2
ω的价值在于充当正态分布方差的类似的效果，更加准确的统计学意义是，正态分布的精度；ω越小，正态分布尾部更厚，表明模型更多的考虑远处的样本点；omega的引入，

T3
对K核函数，引入超参；核函数正态分布，可以考虑引入精度；方差等；或者引入改变核函数类别的超参数；

T4
code

## ch10-sec3
T1
结果不一样；key值不再包含相同元素，那么最终的attention weight与query的取值和计算方法相关。

T2
考虑引入一个可学习的参数矩阵w，采用dotattention的形式，将w置于query和key之间，使得注意力得分函数可以兼容不同长度的张量。

T3
本人觉得，由于AdditiveAttention具备可学习参数，在大多数情况下，会表现出比DotAttention更加理想的性能，但是会增加其计算量，拖慢运行效率；但是在特殊情况下，如果可学习参数，优化为等价于后者的形式，那么性能会弱于后者，（考虑计算成本）

## ch10-sec4
T1 T2
code; combine requirements in two questions into one code block.

## ch10-sec5
T1
code

T2
研究系数W_o，引入权重衰减，对于权重过小的W_o直接稀疏化，这样筛去不重要的头的注意力的取值。

## ch10-sec6
T1
深度架构堆叠位置编码，根据三角函数加和公式，可能会出现，10000数量不足以独一无二表示位置编码的情况，此时需要适当调整分母。

T2
code

## ch10-sec7
T1
网络更深；参数数量更多；运算量加大；在简单数据集的训练效果提升可能有限；

T2
不是好主意；因为Transformer多头注意力的输入的query，key，value的size都是相同的，在dotAttention节约训练时间且性能差异不是很大的情况下，更换为AdditiveAttention效果差不多。

T3
不太清楚语言模型的具体什么应用？翻译还是情感分析还是？

T4
输入序列很长，根据Transformer的运算复杂度$O(n^2d)$，模型的计算量将会二次方增长而不是线性增长，增加GPU数量进行并行运算来缓解这个问题

T5
根据论文内容，有以下几个大方向，进行transformer模型的优化：
1. Memory： transformer 的 内存优化；
2. 迭代优化：Transformer-XL；
3. 低秩核： Low-Rank Transformer;
4. Random/Fixed/Factorized Patterns Image Transformer;
5. Learnable Patterns: Reformer;Clusterformer;
6. Sparsity: Switch Transformer;

T6
根据VIT模型的思路，它是先将一个大图片分割成若干个小patch，添加位置编码（2维）并且传入FFN层；如果是图像分类任务，那么只需要Transformer的Encoder部分，最后通过一个MLP层输出类别，从而可以在不使用CNN的情况下，实现图像分类。
