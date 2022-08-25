## ch5-sec1
T1
如果使用普通Python列表代替Pytorch自定义Sequential，将会不具备Pytorch神经网络所特有的特性，比如整合forward，比如共享参数；比如正常的正向和反向传播；会导致训练性能的异常，需要着重注意。

## ch5-sec2

T1
使用parameters方法遍历即可，Torch Module会自动递归寻找参数；

T2
Pytorch .nn.init模块中，有若干方法，常用有glorot、kami、normal等初始化

T3
调用Parameters，和Parameters.grad获取响应信息，并可视化。类似于Tensorboard的做法；

T4
个人觉得，共享参数一个有利于模型迁移学习；另一个是最大化利用Loss对梯度修正；还有一个就是，共享参数，减小模型规模，使其不至于过于臃肿；

## ch5-sec3

两道code题，详情见code

## ch5-sec4

T1
存储模型参数的好处有：储存check point，如果想继续训练（或者计算机意外crash），直接读取保存好的check point继续训练即可，不用重复训练；或者迁移学习，构建好模型，直接读取别人训练好的，保存模型性能；

T2
先model.load或者load_state_dict读入参数，然后遍历model.children的前两层，保存下来对应的层，自定义一个新模型，第一层取保存好的前两层，构建新模型。注意前两层的self.training设置为false，在训练的时候；或者前两层设置小的learning_rate允许微调。

T3
`model.save(model, 'file_name')`即可保存结构与参数；结构的限制，最好指定对应名称；而且网络的结构最好保持不变；https://pytorch.org/tutorials/beginner/saving_loading_models.html

## ch5-sec5

T1
大矩阵运算GPU快于CPU，如果是小矩阵，那么由于通讯时间的损失，会恰恰相反；

T2
GPU->GPU CPU->CPU读取照常进行
GPU->CPU 需要在`load`函数中加入`map_operation='cpu'`选项；
CPU->GPU 需要在`load`函数中加入`map_operation=lambda storage, loc: storage.cuda()`选项
如果是多GPU训练的模型，读取时会报错，这是因为存入模型的时候，参数的名称被自动加入了前缀'module.'，需要手动删除后，正常读取；
