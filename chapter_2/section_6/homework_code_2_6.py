import torch
import argparse
import itertools, random
import matplotlib.pyplot as plt


FMT_LST = [''.join(pair) for pair in itertools.product(['b', 'g', 'r', 'c', 'm', 'y', 'k'], ['-', '--', '-.'])] 


def set_figsize(figsize=(3.5, 2.5)):
    """设置matplotlib的图表大小

    Defined in :numref:`sec_calculus`"""
    # use_svg_display()
    plt.rcParams['figure.figsize'] = figsize
    plt.rcParams['font.sans-serif']=['SimHei']
    plt.rcParams['axes.unicode_minus']=False

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴

    Defined in :numref:`sec_calculus`"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点

    Defined in :numref:`sec_calculus`"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)

parser = argparse.ArgumentParser('Sampler Visualize')
parser.add_argument('-M', default=50, dest='expr_time', type=int)
parser.add_argument('-N', type=int, default=10, dest='group_num')
parser.add_argument('-S', type=int, default=1000, dest='sample_num')
parser.add_argument('-F', type=str, nargs='*', required=False, dest='format')
arg = parser.parse_args()
if not arg.format:
    format = random.sample(FMT_LST, arg.group_num)
else:
    format = arg.format
m = torch.distributions.multinomial.Multinomial(arg.expr_time, torch.ones([arg.group_num]))
counts = m.sample((arg.sample_num,))
cum_sum = counts.cumsum(dim=0)
probits = cum_sum/cum_sum.sum(dim=1, keepdim=True)
plot([probits[:, i].numpy() for i in range(arg.group_num)],
     figsize=(10, 8), fmts=format, xlabel='试验次数', ylabel='概率',
     legend=[f'group_{i+1}' for i in range(arg.group_num)])
plt.show()
plt.close()
