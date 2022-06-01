T1
可以写若干循环，采用GridSearchCV的思路，但是缺点是复杂度太高，运行时间太长；

还有若干思路，是利用参数搜索算法，比如BayesOptimSearchCV，Hyperband等等，Pytorch根据官方文档的建议是采样Ray Tune进行调参；

T2
如果训练的Epoch太长，会产生过拟合Overfiting的现象；一般做法是，先训练一个长的Epoch，观察Train-Loss，Train-Metric，Test-Loss，Test-Metric指标，如果出现拐点，那么一般说明可能出现过拟合；比较方便的做法是使用TensorBoard观察实验结果，选取合适的Epoch大小；

解决过拟合的方法有很多，也可以考虑优化模型。优化方法，在后面的章节均有介绍。

Reference
[Ray-Tune 调参](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
