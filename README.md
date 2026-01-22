# DIRECTLY-TRAINED-SPIKING-NEURAL-NETWORKS-WITH-ADAPTIVE-PHASE-CODING
文件主要包含shd分类网络的具体实现，以及qkformer的具体修改。

其中qkformer的具体修改,可见model文件中31行开始的具体的LIF神经元代码，仅对原网络神经元函数（class MultiStepLIFNode）进行了重写，qkformer运行环境与原环境相同。

其中shd数据集生成处理方式，以及网络训练范式，见https://github.com/Zhu-Spike-Lab/Robot-SHD，对源代码中的网络结构进行了重写，满足原网络运行环境即可。

其中cifar以及dvs数据集可在网络上搜索下载。

其中SHD与论文中描述的5-mlp结构有较大差距，将数据折叠后，放入了卷积网络中（具体见代码）,该网络准确率可达89.33(高于论文中82的准确率，论文为5层mlp下的结果)，为基础LIF神经元的sota。
