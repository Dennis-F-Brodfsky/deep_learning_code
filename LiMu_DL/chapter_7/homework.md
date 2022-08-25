## ch7-sec1
T1
迭代周期增加，LeNet的val-loss会比AlexNet下降地更明显，因为前者的防止过拟合的措施没有Alexnet多

T2 T3 T5 code

## ch7-sec2
T1 
剩余三层，由于在写VGG-block时，连续添加两个卷积层，打印输出的时候，两个层合并为一个层打印

T2
VGG相较于Alexnet 网络深度更深，参数更多，所以运行时间更长，且需要更多显存。

T3
大小改成96，首先输入输出的大小会发生变化，经过5层maxpooling层的降维，丢失了更多的信息，但是大小减小，降低了显存消耗，同时也提升了训练速度。

T4
见code。

## ch7-sec3
T1
Code

T2
两个1×1卷积层，可能是设计者的针对ImageNet比赛调参的结果，其作用类似于对channel进行两层全连接层的计算；

T3
(1)
1st NiN：11x11x3x96+1x1x96x96+1x1x96x96=53280
2nd NiN：5x5x96x256+1x1x256x256+1x1x256x256=745472
3rd NiN：3x3x256x384+1x1x384x384+1x1x384x384=1179648
4th NiN：3x3x384x10+1x1x10x10+1x1x10x10=34760

(2)
1st NiN：54x54x96+54x54x96+54x54x96
1st MaxPool2d：26x26x96
2st NiN：26x26x256+26x26x256+26x26x256
2nd MaxPool2d：12x12x256
3rd NiN: 12x12x384+12x12x384+12x12x384
3rd MaxPool2d：5x5x384
4th NiN：5x5x10+5x5x10+5x5x10
1st AdaptiveAvgPool2d：1x10

T4
如果中间没有dropout层，所有参数都将起作用，会存在信息瓶颈的问题，且容易过拟合。

## ch7-sec4
T1
Code

T2
32

T3
通过使用1x1的卷积层以及GlobalAvg2代替全连接层降低模型参数个数。

## ch7-sec5
T1
偏置项可以不要，因为自动减去均值，并且除以协方差。而偏置项的均值就是其本身，归零，对参数优化影响忽略不计。而且根据batchnormalize layer的训练公式，模型本身的偏置项和batchnormalize layer的β是重复的，所以没必要加。

T2
详细见code
多gpu下训练，学习率同为0.1，batchsize相等（意味着一个gpu上的batchsize是单cpu的一半）准确率从0.875提高到0.90 ；体现为初始正确率更高。

T3
见code

T4
存疑，二者都有正则化作用，一般是二者选其一，而不会同时出现，如果要同时出现，batchnormal层要在dropout层之前。

T6
[normalization-layer](https://pytorch.org/docs/stable/nn.html#normalization-layers)

T7
其他归一化还可以是，层归一化、局部归一化、权重矩阵归一化；

## ch7-sec6
T1
区别在于inception 4个通道；resnet 2个通道；且其中一个通道是用于防止梯度消失的作用（直接传值，或者仅经过1×1卷积层传值）；第二个区别在于，inception对不同通道的结果按channel叠加，而resnet是取和。

T2
see variant code

T3 
see bottlenet code

T4
see code

T5
复杂性越高，得到的准确率提升边际收益越低；复杂性越高，对硬件存储和内存要求提高；复杂性越高，使得过拟合更容易发生。

## ch7-sec7
T1
Avgpooling 在降维的同时尽可能保留更多原始数据的信息；

T2
DenseNet的优势在于其过渡层使用通道数压缩来降低参数数量

T3
通过引入bottlenet降低Densenet显存占用

T4
[Densenet_code](https://pytorch.org/vision/stable/_modules/torchvision/models/densenet.html#densenet121)

T5
to be continued;但是受到DenseNet的启发，突然有了一个不错的想法。DenseNet提供了神经网络在数据竞赛中，扩展特征的一个方式，就是跨层连接concat，依次替代特征工程，创造出更多的特征。
