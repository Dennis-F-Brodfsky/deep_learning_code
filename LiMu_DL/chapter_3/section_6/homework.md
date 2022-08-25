T1. 
会有溢出问题；
在C++，C等静态语言中，该问题会导致程序运行错误；

T2
值可能为零，导致程序报错

T3
关于softmax溢出问题，可以利用$\textrm{softmax}(o) = \textrm{softmax}(o-a)$来解决，令a是所有o的最大值，那么可以避免溢出问题；
关于CrossEntropy 自变量取零的问题，需要分类讨论，如果自变量取零，则函数默认返回0值。

T4
对数似然最大，并不总是好主意；还要考虑分类错误所造成的成本；

T5
词汇量过多，导致预测类别过多，概率被过度稀释，这就导致预测的概率分布不稀疏，从而概率最大的标签的概率与其他标签的概率差别不是很大，导致分类性能的下降；