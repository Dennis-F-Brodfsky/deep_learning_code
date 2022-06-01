## ch4-sec1

T1 
$$
\begin{equation*}
\begin{split} 
\frac{\partial \textrm{PReLu}(x))}{\partial x} = \displaystyle\left\{\begin{matrix}
1 ,\quad x > 0 \\ 
\alpha, \quad x < 0
\end{matrix}\right.
\end{split}
\end{equation*} $$ subgradient on x=0 is $[\alpha, 1]$

T2
我记得要严格证明的话，需要泛函的知识，就是类似于万能近似定理的那篇论文。就是要证明若干基函数满足某个空间的完备性等等若干性质，然后才能证明 具有激活函数的 MLP可以拟合任意函数。 这里记录一下直观理解。
如果是连续线性函数y=cx，那么可数个c*ReLu(x) (x>n)的部分拼接即可。
如果连续函数在某一点$x_1$上的导数存在跳跃，那么就利用$c_1*\textrm{ReLU}(x) (x>x_0), c_2*\textrm{ReLU}(x) (x>x_1), x_0 < x_1 $ 去逼近跳跃点的导数变化，从而达到近似效果。

T3
$$
\begin{equation*}
\begin{split} 
\tanh(x)+1 & =\frac{\exp(x)-\exp(-x)}{\exp(x)+\exp(-x)} + 1 \\
& =  \frac{2\exp(x)}{\exp(x)+\exp(-x)} \\
& = 2  \ \frac{\exp(2x)}{\exp(2x)+1} \\
& = 2 \ \text{sigmoid}(2x)
\end{split}
\end{equation*} 
$$  在Pytorch验证中，发现浮点数张量 运算存在小误差的问题，如果要比较两浮点张量是否相等，可以考虑`torch.isclose`函数。

## ch4-sec2

T5
等多超参数，意味着超参数空间的高维化，导致超参组合呈现指数级别增长，加大寻找到全局最优的难度；

T6
最聪明的策略是使用最前沿的调参方法，比如HyperBand算法等；或者根据经验判断比较重要的超参数，比如学习率。

## ch4-sec3

和ch4-sec2 作业的前几问，原理相似，故考虑选做第三小节的习题。 Ray-Tune调参代码见code部分

## ch4-sec4

T1 
和多元回归解析解相同$\mathbm{\beta} = (X^TX)^{-1}X^Ty $ 其中，如果X的列等于行，那么X就是一个标准的范德蒙矩阵，其逆矩阵一定存在。

T2 
(1)
见code
(2) 理论上，当回归式的次数超过数据生成的次数之后，train-loss会趋于零，与此同时，val-loss会出现不减反增；现实中，因为模型训练未完成，随机性等原因，train-loss不一定会真的等于零，但是val-loss 的趋势基本符合预期。
(3)
见code

T3 

高次项的参数求解结果太小， 在Pytorch 浮点数运算下，误差惊人； 除了除以i阶乘标准化，考虑正态归一化，或者取对数长尾分布；

T4

不太可能，可以接近零；等于零的模型，没见过。

## ch4-sec5

T1 code
gamma 越小，test的loss越小；
T2 
不一定，正则化做的好未必代表全局最优解。但是保证了模型的泛化能力。

T3
$$ \mathbf{\omega} \leftarrow \mathbf{\omega} - \eta \frac{\partial l(y, \hat{y})}{\partial \omega} - \eta \lambda \mathbf{1} $$

T4
Frobenius范数 $ \left | \left | A \right | \right |_F = \sqrt{tr(A^TA)}$

T5
神经网络使用dropout，图像识别可以有数据增广，文本分析也有类似数据增广，利用Tensorboard 在训练网络的时候观察train-loss和val-loss的变化，将训练轮数控制在val-loss的拐点出现之前。

T6
贝叶斯角度而言，可以引入参数的先验分布来达到正则化的目的，L2正则化，引入了参数的高斯分布先验分布；而L1正则化，则是引入了参数的拉普拉斯分布先验分布。

## ch4-sec6

T1
嘶，暂时没看出有啥差异？ 交换dropout rate，可能会改变mask node的规模？

T2
增加轮数，不适用dropout的模型过拟合的概率高于使用dropout的模型，从test-loss曲线的趋势中，明显得出该结论。

T3

T4
训练过程中，随机dropout结点，相当于无效化训练得到的参数，这是一种浪费，而且，预测引入随机性就是引入不稳定性，得到的不同结果，没有参考价值。

T5
dropout + 权重衰减，通常不会有更好结果；1+1小于2

T6
应用到权重，应该相当于达到了删除两层结点删除边的效果；这种情况下，可能会有类似于dropout的效果，但是会增加维持期望不变的难度；

T7
详见代码，初步思路是，构造一个新Module，在forward的过程中，自动添加均值为零的正态噪声。

## ch4-sec7

T1
维度是n乘以m

T2


T3


T4
中间过程可能生成了一个包含一阶导的新的计算图；这其中会有利用中间结果优化计算的过程；计算时间包含了计算一阶导的时间；

T5
(1) 
可以，涉及到模型并行训练的问题，直观上来讲，可以将模型按层划分、可以垂直于层划分（比如AlexNet）；
(2)
优点是，更新权重时，较少的收到随机因素扰动的影响，而且，由于是多台设备并行，效率更高；缺点是，要考虑通信时间的影响，也就是所谓的时钟时间；

## ch4-sec8

T1
卷积神经网络、循环神经网络；

T2
线性回归类比于一层MLP，而softmax类比为带有softmax activation的一层MLP，出于对称性的考虑，权重应该随机赋予。

T3
一个十分宽松的界是 the egenvalues of product of matrices A,B 介于两个矩阵最大特征值之积和最小特征值之积之间，如果控制这些矩阵的最大特征值和最小特征值的大小，那么就可以避免梯度消失和梯度爆炸的问题。
https://www.researchgate.net/publication/3032531_Eigenvalue_Inequalities_for_Matrix_Product

T4

## ch4-sec10

T1
略
T2
预测价格对数，可以有效解决价格长尾分布的问题，是数据比赛中的常用策略，价格对数一般比价格更加地趋近于正态分布，符合一些模型对于数据分布的假设。
T3
不是很好的做法；有时候缺失值不是随机丢失，比如收入变量，收入较低的人，更不愿意透露自己的收入，如果用均值替代，那么会有误导性。
T4
略
T5
略
T6
如果没有数据标准化，更容易出现梯度爆炸的情况。或者模型收敛更困难。