T1
(1)
$$ 
\sum_{i} (x_i-b)^2 = (\mathbf{x}-b\mathbf{1})^T(\mathbf{x}-b\mathbf{1}) \\
\frac{\mathrm{d}  (\mathbf{x}-b\mathbf{1})^T(\mathbf{x}-b\mathbf{1})}{\mathrm{d} x} = - 2 \mathbf{1}^T\mathbf{x} + 2 b n = 0 \\
b = \frac{\sum_{i} x_i}{n} = E[x]
$$
(2) 
这个问题，和广义线性方程模型有关，大概含义就是，如果扰动项服从正态分布，那么Link-Function就是g(x)=E(x)。

T2
(1)
objective function:
$$
Loss(\omega) = (y-X\omega)^T(y-X\omega) \\
\max_{\omega} Loss(\omega)
$$
(2)
$$
\frac{\partial Loss(\mathbf{\omega})) }{\partial \mathbf{\omega}} = -2X^Ty+2X^TX\omega
$$
(3)
$$
\hat{\omega} = (X^TX)^{-1}X^Ty
$$
(4)
处理简单回归问题的时候，解析解优于随机梯度下降；或者，实际问题要求严格精确，采用解析解；
- 但是解析解存在问题，当$X$不满足列满秩条件，$X^TX$的逆不存在；
- 亦或者当X是个超级大的矩阵时，计算X^TX的成本十分高昂；
第一个问题，可以考虑引入正则项解决，但是引入正则项之后，一般而言，系数估计是有偏的；
第二个问题，考虑SVD分解，缓解大矩阵计算上的复杂度；

T3
(1)
$$
-\log{P(\boldsymbol{y} \mid \boldsymbol{X}))} = - \sum_{i} \log{P(y_i \mid X_i)} = -\sum_{i}\log( \frac{1}{2} \exp{(-\left | y_i-X_i\omega \right |)}) \\
= \sum_{i} \left | y_i-X_i\omega \right | + const
$$
(2)
没有解析解，但是如果回归系数只含常数项，那么这个常数代表样本数据的中位值。
PS：也可以通过sub-gradient求解，得到解析解。
(3)
不可导凸函数模型数值迭代求解，参考the element of Statistic Learning 相关讨论。

Reference
[Generalize Linear Model](https://zhuanlan.zhihu.com/p/467976558)
