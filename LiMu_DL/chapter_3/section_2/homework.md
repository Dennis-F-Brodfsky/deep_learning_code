T1
初始化权重为零，会造成参数更新对称的问题，引入不对称因素，更容易找到全局最优点。

T2
T3

T4
二阶导，计算复杂度太高？（在神经网络中）
使用拟牛顿法缓解；

T5
保证y_hat 和 y的shape相同；

T6
低学习率，损失函数变动较小；

T7
根据data_iter的实现方法，如果不能整除，那么最后一批样本的数量就是`sample % batch_size`
