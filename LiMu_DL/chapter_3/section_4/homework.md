T1
(1)
根据原书公式 
$$ \frac{\partial l(\mathbf{y}, \mathbf{\hat{y}}))}{\partial o} = \textrm{softmax}(\mathbf{o}) - \mathbf{y} $$
进一步对$o^T$求导即可得
$$ \frac{\partial^2 l(\mathbf{y}, \mathbf{\hat{y}}))}{\partial o \partial o^T} = (\textrm{softmax}(o)_j(\mathbb{I}(i=j) - \textrm{softmax}(o)_j))_{ij} $$
(2)根据指数分布族的性质，可以得知，softmax函数其实和Categorical分布有关，把Categorical分布函数写成指数分布族的形式如下： 
$$\begin{equation*}
\begin{split} 
&f_X(x\mid \mathbf{\theta})=h(x)\exp{(\eta(\mathbf{\theta}) \cdot \mathbf{T(x)} - A(\mathbf{\eta}))} \\ 
&h(x)=1 \\
&\eta(\mathbf{\theta}) = {\displaystyle {\begin{bmatrix}\log {\dfrac {p_{1}}{p_{k}}}\\[10pt]\vdots \\[5pt]\log {\dfrac {p_{k-1}}{p_{k}}}\\[15pt]0\end{bmatrix}}} \\ 
&\mathbf{T(x)} = \begin{bmatrix}\mathbb{I}(x=1)\\\vdots \\{\mathbb{I}(x=k)}\end{bmatrix} \\
&A(\mathbf{\eta}) = {\displaystyle \log \left(\sum _{i=1}^{k}e^{\eta _{i}}\right)=\log \left(1+\sum _{i=1}^{k-1}e^{\eta _{i}}\right)}
\end{split}
\end{equation*}
$$ 以及指数分布族两个重要定理：
$$ \begin{equation*}
\begin{split} 
&{\displaystyle \operatorname {E} (T_{j})={\frac {\partial A(\eta )}{\partial \eta _{j}}}} \\
&{\displaystyle \operatorname {cov} \left(T_{i},\ T_{j}\right)={\frac {\partial ^{2}A(\eta )}{\partial \eta _{i}\,\partial \eta _{j}}}.} 
\end{split}
\end{equation*}
$$ 其实，softmax函数就是Categorcal的一阶矩，如果是多分类逻辑斯蒂回归的语境下，有$\eta_i = E(y_i \mid \mathbf{x}) $，带入期望表达式中，即可得到熟悉的softmax函数。那么该分布的二阶矩，就是第一问求得的结果。

T2
(1) 
独热编码
潜在问题：
容易过拟合，独热编码假设不同类别是相互独立的，然鹅事实并非如此
如果类别太多了，导致过于稀疏； 
(2)
可以考虑标签平滑，如果是n个标签(1/n, ..., 1/n) 那么，编码结果就是$(\frac{\varepsilon}{n}, \cdots, 1-\varepsilon ,\cdots,  \frac{\varepsilon}{n})$
考虑Encoding-Decoding或者Embedding技术；

T3
题目中的RealSoftmax函数其实是 LSE的二维特殊形式。而经常提到的softmax是LSE的一阶导。
(1) 
Proof
$$ 
\begin{equation*}
\begin{split}
& \textrm{RealSoftmax}(a, b) = b\log(1+\exp(a-b)) = a\log(1+\exp(b-a)) \\
& \textrm{if} \quad b > a \quad \textrm{then} \\
&\textrm{RealSoftmax}(a, b) = b\log(1+\exp(a-b)) \geqslant \max(a,b) \\
&\textrm{else} \\
&\textrm{RealSoftmax}(a, b) = a\log(1+\exp(b-a)) \geqslant \max(a,b)
\end{split}
\end{equation*}
$$ (2) 
Proof
$$
\begin{equation*}
\begin{split}
& \lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) = b\log(1+\exp(\lambda)\exp(a-b)) = a\log(1+\exp(\lambda)\exp(b-a)) \\
& \textrm{if} \quad b > a \quad \textrm{then} \\
&\lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) = b\log(1+\exp(\lambda)\exp(a-b)) \geqslant \max(a,b) \\
&\textrm{else} \\
&\lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) = a\log(1+\exp(\lambda)\exp(b-a)) \geqslant \max(a,b)
\end{split}
\end{equation*} $$ (3)
Proof 
$$ 
\begin{equation*}
\begin{split}
& \textrm{if} \quad b > a \quad \textrm{then} \\
& \lim\limits_{\lambda \to \infty} \lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) =  \lim\limits_{\lambda \to \infty} b\log(1+\exp(\lambda)\exp(a-b)) = b \\
&\textrm{else} \\
& \lim\limits_{\lambda \to \infty} \lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) = \lim\limits_{\lambda \to \infty} a\log(1+\exp(\lambda)\exp(b-a)) = a \\
& \textrm {above all:} \lim\limits_{\lambda \to \infty} \lambda^{-1}\textrm{RealSoftmax}(\lambda a, \lambda b) = \max(a, b)
\end{split}
\end{equation*}
$$ (4) 
Answer: $$ \textrm{softmin}(o)_j = \frac{\exp(-o_j)}{1 + \sum\limits_i^{k-1} \exp(-o_i)} $$ 详情见Pytorch参考文献。
(5)
n维情况对应维基百科中的LSE函数$$ \log(\sum\limits_{i}^n\exp(z_i)) $$

Reference
[Exponential_family](https://en.wikipedia.org/wiki/Exponential_family)
[Softmax function](https://en.wikipedia.org/wiki/Softmax_function)
[`torch.nn.Softmin`](https://pytorch.org/docs/stable/generated/torch.nn.Softmin.html)
[`torch.nn.functional.softmin`](https://pytorch.org/docs/stable/generated/torch.nn.functional.softmin.html)