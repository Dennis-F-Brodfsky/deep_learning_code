## ch9-sec1
T1
t = t'时，更新门是0，重置门是1.

T2.
调参题

T3
code

T4
功能不完善，删除重置门，那么GRU跳过每个序列，退化为一个普通的全连接层；删除更新门，则会采纳每个短期记忆，相当于普通的RNN层。

## ch9-sec2
T1
调参题

T2
将`load_corpus_time_machine`中的`tokens = tokenize(lines, 'char')`改为`tokens = tokenize(lines, 'word')`

T3
长短期循环网络大于门控循环网络大于常规神经网络
这个主要由同一设备，同一隐藏层设置，同一训练任务的三种记忆循环网络时间确定。具体见code

T4
第一个tanh作用于candidate hidden state，这是基于RNN的实现；第二个tanh作用于短期candidate hidden state和长期记忆C的加和，有可能超过-1到1的范围；除此以外，第二个缓解加和项反向传播梯度爆炸。

T5
作业ch8中，将autoregression中的`self.rnn`改为GRU或者LSTM Cell，再运行该代码，即可完成时间序列预测演示。

## ch9-sec3
T1
code

T2
code；两种双层网络，GRU运行速度更快，且性能以及精准度其实差别不大

T3
目前的数据已经达到1.0 （理论最低困惑度）

T4
优。在于增加了数据量；劣，引入了不同作者，但是作者本身也是信息，不同作者风格不同，如果单纯套用循环神经网络，可能造成这部分信息丢失。

## ch9-sec4
T1
隐藏层分为正向和反向两个变量；形状分别为$(H_1, H_1),(H_2, H_2)$

T2
$$
\begin{align*} 
\begin{split}
\overrightarrow{\mathbf{H}}_t^{(l)} &= \phi(\overrightarrow{\mathbf{H}}_{t}^{(l-1)} \mathbf{W}_{xh}^{(f)} + \overrightarrow{\mathbf{H}}_{t-1}^{(l)} \mathbf{W}_{hh}^{(f)} + \mathbf{b}_h^{(f)}),\\ 
\overleftarrow{\mathbf{H}}_t^{(l)} &= \phi(\overleftarrow{\mathbf{H}}_t^{(l+1)} \mathbf{W}_{xh}^{(b)} + \overleftarrow{\mathbf{H}}_{t+1}^{(l)} \mathbf{W}_{hh}^{(b)} + \mathbf{b}_h^{(b)}), \\
\overrightarrow{\mathbf{H}}_t^{(0)} &= \mathbf{X_t} \\
\overleftarrow{\mathbf{H}}_t^{(L)} &= \mathbf{X_t} 
\end{split}
\end{align*}
$$

T3
个人觉得Embedding + Directional LSTM？ 

## ch9-sec5
T1
num_example 越大，词汇量越多，但是代码中有一个小错误，就是设定`num_example=600`时，实际上读取了601行数据，在code中已修正。

T2
仍然可以一试；虽然直接来看不是好主意；中文分词，可以尝试jieba分词库，日文不太清楚。主要会面临划分方式的多义性。

## ch9-sec6
T1
不是，利用`Flatten()`,`UnSqueeze()`，等改变张量形状的自定义层或者API层，实现不同种类Encoder-Decoder的衔接。

T2
数据降维； 个人感觉GAN那种结构和Encoder-Decoder结构挺像的。（虽然完全不是一回事）

## ch9-sce7
T1
调参题

T2
code;居然loss变小了，而且收敛更快。不加入mask，高估了loss，在相同学习率下，加速了训练，但是这样虽然降低loss，但是引入了杂质信息，实际翻译效果不见得更好。

T3
层数不一致，会有影响，decoder输入要用到相关的state，要保证维度一致；但是隐藏层不一致，解码器的RNN的in-feature要与编码器的outfeature保持一致。

T4
code

T5
code

T6
除了普通线性层，可以考虑其他层，一维卷积层，一维平均池化层；此外尝试一些其他的激活函数（Pytorch分类任务中，由于CrossEntropyLoss，不需要Softmax层），等等。

## ch9-sec8
T1
穷举搜索看成跨度为整个序列长度的束搜索；原因，和贪心搜索是时间跨度为1的束搜索相同：束搜索就是介于贪心搜索和穷举搜索二者之间的搜索方式。

T2
束搜索的束越大，那么程序运行时间越长，但是困惑度不变或者下降

T3
例8.5生成文本，理论上使用了贪心搜索。改进的话，多加入较长的序列取优。
