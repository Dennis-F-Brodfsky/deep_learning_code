## ch11-sec1 

T1 
考虑数学归纳法：
在d=1的时候，很显然如果$y=f(x)$有一个局部最小点，那么由于全局最小值的存在，在局部最小值核全局最小值之间（如果函数连续的话，在MLP情景下，应该是符合的）会有至少一个替代方案，题设成立；
假设d=n-1成立；
d=n的时候，考虑将其中一个维度投影，那么至少存在一个等价方案；那么剩余n-1个维度，就会有至少n-1个替代方案；总和就是至少n个方案；
综上所述；对于一个隐藏层是d维度一个输出的函数，局部最小点的输出在d维空间上，至少有d个替代方案。

T2
(1)
设想一个对称矩阵M的取值，使得特征向量v的特征值λ为正，对于每一个取值都存在一个双射：矩阵元素取负，使得对于任意特征向量v的特征值为-λ；根据题目所给条件，抽取分布是对称的，得到$P(\lambda > 0) = P(\lambda < 0)$

(2)
因为对称抽取不能保证对称矩阵特征值为零的概率是零。即$P(\lambda > 0) = P(\lambda < 0) = \frac{1}{2}(1-P(\lambda = 0)) $

T3
梯度爆炸， 在RNN中； 
调参问题：如何在指数级爆炸的参数空间中，找到最优超参；

T4
1. 这个问题，有关物理学中的稳定稳态，非稳定稳态，和非稳态； 鞍点上平衡球体属于非稳定稳态问题，给予其微小的扰动，导致球体脱离稳态（从鞍点掉落）
2. 在优化中，考虑给学习率添加扰动？或者在修正参数的时候加入微扰常数项？maybe

## ch11-sec2

T1
(1)
可能，一个凸集内所有的点都可以表示为边界点的线性组合
(2)
边界点可以表示为顶点的线性组合

T2
$p \geqslant 1$ 由Stein实变可知，表示的范数满足距离测度的若干性质，其中一条就是两条边距离之和大于等于第三边；可得$\forall \lambda \in (0, 1), x, y \in \mathfrak{B}$， 有$ \left \| \lambda x + (1 - \lambda ) y \right \|_p \leqslant \lambda \left \| x \right \|_p + (1-\lambda ) \left \| y \right \|_p \leqslant 1$. 命题得证

T3
$$ 
\begin{equation*}
\begin{split}
& \because |f(x) - g(x)| = \max(f, g) - \min(f, g)  \\
& \therefore \min(f, g)  = \max(f, g) - | f(x) - g(x) |  \\
\end{split}
\end{equation*}
$$
由于$\max(f, g)$是凸函数，而后面那些减去了一个凸函数，故凹凸性非凸。

T4
Softmax函数求二阶导正好是Categorical Distribution的二阶矩，推导详见第三章作业，或者说是方差；而协方差矩阵都是非负定的，所以命题得证。

T5
高等代数学线性空间那章的常见证明题，
$$ 
\begin{equation*}
\begin{split}
& \because X_1, X_2 \in \mathfrak{X}  \\
& \therefore WX_1 = WX_2 = b \\
& \because \forall \lambda \in (0, 1),  \\
& W(\lambda X_1 + (1-\lambda) X_2) = \lambda b + (1-\lambda) b = b \\
& \therefore \lambda X_1 + (1-\lambda) X_2 \in \mathfrak{X}
\end{split}
\end{equation*}
$$

T6
对于矩阵M的投影的定义是？在书中正文内容没提到，而且一般将投影，想到的都是向量的投影，没太理解题目意思

T7
这应该就是泰勒公式吧， 搞数竞都是直接拿来用的，想不起来咋证明的了？而且记得不需要函数凸的条件吧。有点懵。

T8
(1)
subgradient 做法：结论：$ w' = \sign(w)(w - \frac{\lambda}{2})_{+} $；
参考链接: [subgradient](https://freemind.pluskid.org/machine-learning/sparsity-and-some-basics-of-l1-regularization/)

(2) λ取值，比较常规的做法，当然还是交叉验证。

T9
设空间中，向量及投影到凸集空间的向量之差（表示一个点到某平面距离的那个向量）为$h$；此处默认书中对距离的度量测度是满足希尔伯特空间完备性的。（主要用到一个三角形第三边大于等于两边距离之和的性质）
$$ 
\begin{equation*}
\begin{split}
\left \| \mathrm{Proj}_{\mathfrak{X}}(x) - \mathrm{Proj}_{\mathfrak{X}}(y) \right \| & = \left \| (x - h_x) - (y - h_y) \right \| \\
& \leqslant \left \| x - y \right \| + \left \| h_x - h_y \right \| \\
& \leq \left \| x - y \right \| 
\end{split}
\end{equation*}
$$

## ch11-sec3
T1
code

T2
code

T3
code

T4
code

T5
code

## ch11-sec4
T1
code

T2
对gradient添加噪声采样，就$f(x_1, x_2)=x_1^2 + 2x_2^2$而言，相当于在值函数的基础上，增加了正态分布随机变量$f(x, w) = (x-w_1)^2 + 2(x-w_2)^2$。对这个函数求导并且结合程序实现可知等价。

T3
重复采样，相当于进行boostrap操作，在缩减有效样本量的同时，出现无意义重复的情况，随机性降低。

T4
随机渐变，添加学习率权重。

T5
无穷多个，因为存在周期变化的sinx这一项；考虑对sin函数中的x指定一个映射，不改变global minimum的位置和数值，并且压缩其周期。比如将sinx 替换为2pi*sin(x)。

## ch11-sec5
T1
code

T2
MXnet 略

T3
code

T4
权重更迭翻倍，相当于实际learning rate翻倍

## ch11-sec6
T1
code

T2
code

T3
根据凸性，直接对x求导取零点即可；由于是凸函数，所以Q是一个正定矩阵。
$$
 \frac{\partial h(x)}{\partial x} = Qx + c = 0
$$ 求得 $$ x^{*} = - Q^{-1}c $$，带入原方程得到最小值$$h_{min}(x^{*}) = b - \frac{1}{2} c^TQ^{-1}c $$

## ch11-sec7
T1
因为正交矩阵的一个性质，就是一个向量的范数乘以一个正交矩阵，那么结果得到的范数不变。 原因是$M^TM = I$。而$\left \| Mv \right \|_2 = v^TM^TMv = \left \| v \right \|_2 $。正交矩阵变换相当于旋转变换，欧氏距离旋转不变性，所以扰动不变，

T3
$Mx= \lambda x$，等式中，记$x=(x_j)$，然后在所有x中，选取绝对值最大的一个元$x_i$，展开左边的等式，对第i行，我们有$\sum\limits_{j} M_{ij}x_j = \lambda x_i$，移项相除可得$\mid \lambda - M_{ii} \mid = \mid \sum\limits_{j \neq i} \frac{M_{ij}x_j}{x_i} \leqslant \sum\limits_{j \neq i} \mid M_{ij} \mid $. 得证。

T4
根据定理，可知，矩阵$diag^{-0.5}(M) M diag^{-0.5}(M)$ 经过预处理可以保证特征值误差在可接受范围以内，以1为中心。

T5
原来的训练流程照旧，就是更换一个optimizer即可，对引入的超参数同样做调参处理。

T6
根据其记录历史梯度之和的行为，考虑引入折现因子，使得历史太远的梯度记录会消逝掉。

## ch11 sec8
T1
gamma = 1，记录功能基本失效，退化为普通SGD

T4
gamma如果是在合理范围内，调参意义主要还是以学习率为主。或者更改网络结构。

## ch11 sec9
T1
NASA 案例运行结果 初步显示，ρ变小，收敛速度变慢。

T2
略去gt‘ 节省存储空间；在$\delta x$迭代那个过程中，直接一步到位。$\triangle x_t = \rho \triangle x_{t-1} + (1-\rho)g_t^{'2} = \rho \triangle x_{t-1} + (1-\rho)\frac{\triangle x_{t-1}+\varepsilon}{s_t+\varepsilon} g_t $

T3
并不是，learning-rate调节与ρ超参数有关。

## ch11 sec10
T1 

T2
把v_bias_corr, s_bias_corr并入v，s中，但是代码会不美观。所以官方实现代码中，也是采用了纠正的思路。

T3
接近收敛时，小的学习率使得学习成果更加稳定，优化更加稳定，而不是跳跃且最优解不稳定的。
