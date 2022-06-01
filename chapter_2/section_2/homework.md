使用kaggle convid-19 确诊人数数据，

requirement：
pandas；
numpy；
torch；

大致思路：
`.isnull().sum()` 计数缺失值
`np.argmax ` 定位缺失值最大的列数
`.drop` 删除指定列
`torch.tensor` 将DataFrame的Values转换成torch。

**Note**
DataFrame中Object类型数据无法直接转换成tensor.torch。需要数据预处理，如one-hot、pd.getdummy()等encoding操作进行转换。
