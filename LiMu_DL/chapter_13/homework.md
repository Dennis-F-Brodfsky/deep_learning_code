## ch13 sec1
T1
code

T2
提高测试集准确性的本质就是，进行图像增广，使得训练图像的特征分布可以经过增广的某种特征转化的组合逼近测试集的特征分布。根据这个原理就可以推测出，如果数据增广过度，也会带来准确率的下降；是否过度的评判标准，取决于测试集和训练集数据特征分布的差异大小。

T3
[Torchvision ImageTransform](https://pytorch.org/vision/stable/transforms.html) 分为三大类Transform，一个是支持PIL image的transform，一个是只支持Tensor的Transform，第三个是既支持PIL image也支持Tensor的。 有些Transform支持TorchScript预编译。

## ch13 sec2
T1
code

T2
code

T3
code

T4
code

## ch13 sec3
T2
因为给定一个框，书中给了两种定位方法，一个是左上右下坐标确定一个框；还有一个就是给定中心和长宽确定一个框；这两种方法在二维图像中都需要四个数值来确定。

## ch13 sec4
