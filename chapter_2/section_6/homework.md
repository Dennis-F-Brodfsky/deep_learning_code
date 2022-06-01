T1
考虑编写一个py文件，argparse设置M，N, S, F，命令行运行，观察相关结果。
使用方法：将**homework_code_2_6.py**放在工作目录下，运行`Python homework_code_2_6.py [Option]`即可

Option:
- `-M` 是单次抽样的试验次数trial，默认50
- `-N` 是抽样的分组数量，如果是模拟掷色子，则N=6，默认10
- `-S` 是进行试验的数量或者说抽样的次数，默认1000
- `-F` 设置绘图曲线的格式，建议格式数量与`-N`保持一致。否则可能出现某些线不能正常显示的错误，如果没有设置，那么随机抽取`-N`

Example

`Python homework_code_2_6.py -M 50 -N 5 -S 1500 -F b- g- r- c- m-`


T2
$$
0 \leqslant P(\mathcal{A} \cap \mathcal{B}) \leqslant \min{(P(\mathcal{A}) , P(\mathcal{B}))}
$$
当A，B交集为空集时，左边的等号成立；当A包含于B或者B包含于A，右边的等号成立；

$$
\max{(P(\mathcal{A}) , P(\mathcal{B}))} \leqslant P(\mathcal{A} \cup  \mathcal{B}) \leqslant P(\mathcal{A})+P(\mathcal{B})
$$
当A包含于B或者B包含于A，左边的等号成立；当A，B交集是空集时，右边的等号成立

T3
Under Markov-Condition:
$$
P(A, B, C) = P(C \mid A,B)P(A, B) = P(C \mid B)P(B \mid A)P(A)
$$

Reference
[`argparse`](https://docs.python.org/zh-cn/3/library/argparse.html)
[`torch.distributions.multinomial.Multinomial`](https://pytorch.org/docs/stable/distributions.html#multinomial)
