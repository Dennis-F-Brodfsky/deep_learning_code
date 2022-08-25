## ch6-sec1
T1
根据公式(6.1.7)，令a，b变动范围为零，得到：$$ H_{i,j,d} = \sum\limits_{c=0}^k V_{0,0,c,d}*X_{i,j,c} $$ 本质上依然是针对channel的全连接层；

T2
因为平移不变性本质上相当于在卷积层中共享参数；虽然获得平移不变性，但是参数的减少，带来网络表达能力的潜在降低。

T3
卷积核的大小，让多少像素共享参数；卷积核的平移方式，逐一移动，还是允许跳跃；对图像边缘的处理，是否需要补零；进行卷积运算后，输出的大小是否要一致还是其他控制。

T4
音频看作是一维数据，卷积考虑一维卷积的形式；

T5
当然适用，文本数据经过合适的Tokenize，Embeding转换，等价于一维序列数据；存在一维卷积在nlp处理中的实例；

T6
连续：$$f*g = \int f(x)g(z-x) \mathrm{dx} = \int f(z-x)g(x) \mathrm{d(z-x)} = \int g(x)f(z-x) \mathrm{dx} = g*f$$ 
离散：积分号改成连加号即可

## ch6-sec2
T1
(1) 卷积核可以识别对角线边界线
(2) 图像转置导致结果也转置
(3) kernel转置那么输出的1和-1的顺序颠倒
T2
只能解决二维张量的自动求导问题
T3
将输入矩阵x，根据卷积核的大小和步长分割为若干patch，并展平为一个行向量，再concat, $[p_1; \cdots; p_n]$，将卷积核的元素展平为列向量，然后矩阵乘法，最后结果reshape成对应输出的大小。
T4
(1)`K = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]`
(2)`K = [[1]]`
(3)d+1

## ch6-sec3
T1 T2
[Conv2d 输入输出形状](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)

T3
音频信号步幅为2，说明间隔一个音频提取特征信息

T4
降维

## ch6-sec4
T1
Proof 
假设`A--Conv2d->B--Conv2d->C`,那么
$$ C_{ij} = B_{ij}+ \cdots + B_{i+k_2-1,j+k_2-1} = (A_{ij} + \cdots + A_{i+k_1-1, j+k_1-1}) + \cdots (A_{i+k_2-1, j+k_2-1}+ \cdots +(A_{i+k_2+k_1, k+k_2+k_1})) $$ 相当于一个核为$k_1+k_2-1$的卷积层；反之亦然

T2
(1) $\left \lfloor \frac{h - k_h + p_h + s_h}{s_h} \right \rfloor \left \lfloor \frac{w - k_w + p_w + s_w}{s_w} \right \rfloor c_0c_1$
(2) $\left \lfloor \frac{h - k_h + p_h + s_h}{s_h} \right \rfloor \left \lfloor \frac{w - k_w + p_w + s_w}{s_w} \right \rfloor c_0 + c_ihw + c_oc_ik_hk_w$ 就是核、输入、输出的大小之和；
(3)$c_oc_ik_hk_w $
(4)$ \left \lceil \frac{k_h}{s_h} \right \rceil \left \lceil \frac{k_w}{s_w} \right \rceil c_0c_ik_hk_w $

T3
in_channel,out_channel翻倍，计算成本翻4倍;
如果增加填充，则增加$\frac{p_hp_w}{s_hs_w} c_0c_1 $ 计算数量

T4
$O(c_0c_ih_wh_k)$

T5
浮点数运算存在正常误差

T6
见code

## ch6-sec5
T1 
average-pooling，对应卷积kernel权值相等且平分1的情况，权值矩阵的require_grad=False，并且默认`in_channel=out_channel`。其他的照常。

T2
max-pooling，对应卷积`kernel=torch.where(x==torch.max(x), 1, 0)`，并且默认`in_channel=out_channel`。其他的照常。

T3
$ck_hk_w \left \lfloor \frac{h - k_h + p_h + s_h}{s_h} \right \rfloor \left \lfloor \frac{w - k_w + p_w + s_w}{s_w} \right \rfloor  $

T4
MaxPooling只考虑一类元素；而AvgPooling是相邻所有元素。导致选择strides值会有所不同?

T5
不需要，max-pooling输入取负，输出取负数可以得到min-pooling的效果

T6
GlobalAvgPooling，MaxPooling1D，MaxPooling3D，AvgPooling1D，AvgPooling3D
Softmax不受欢迎是因为，它生成概率分布；图像识别，是为了生成特征，而且Pytorch的CrossEntropyLoss自带Softmax操作；
再者，Softmax针对一维输入，多维情况，统计意义不明。

## ch6-sec6
见code
