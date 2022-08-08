"""3.2 使用torchvision导入和加载数据集
3.2.1 创建深度学习项目的流程：
准备数据集
创建网络模型
训练网络模型
分析结果 #### 3.2.2 数据准备遵守ETL过程：
提取(extract)、转换(transform)、加载(load)
pytorch中自带的包，能够将ETL过程变得简单 #### 3.2.3 数据的准备：
1.提取：从源数据中获取fashion-mnist图像数据
2.转换：将数据转换成张量的形式
3.加载：将数据封装成对象，使其更容易访问
Fashion-MNIST 与 MNIST数据集在调用上最大的不同就是URL的不同
torch.utils.data.Dataset:一个用于表示数据集的抽象类
torch.utils.data.DataLoader: 包装数据集并提供对底层的访问"""
import torch
import torchvision
import torchvision.transforms as transforms # 可帮助对数据进行转换

train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',   # 数据集在本地的存储位置
    train = True,                   # 数据集用于训练
    download = True,                # 如果本地没有数据，就自动下载
    transform = transforms.Compose([
        transforms.ToTensor()         
    ])                              # 将图像转换成张量
)

train_loader = torch.utils.data.DataLoader(train_set)
# 训练集被打包或加载到数据加载器中，可以以我们期望的格式来访问基础数据；
# 数据加载器使我们能够访问数据并提供查询功能

#3.3 数据集的访问
train_set = torchvision.datasets.FashionMNIST(
    root = './data/FashionMNIST',
    train = True,
    download = True,
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)     # 设置打印行宽
print(len(train_set))
print(train_set.train_labels)
print(train_set.train_labels.bincount())    # bincount:张量中每个值出现的频数


"""要想从训练集对象中访问一个单独的元素，
首先要将一个训练集对象(train_set)传递给python的iter()函数，
以此返回一个表示数据流的对象；然后就可以使用next函数来获取数据流中的元素"""

# 查看单个样本
sample = next(iter(train_set))
print(len(sample))
print(type(sample))

# 将sample解压成图像和标签
image = sample[0]
label = sample[1]
image.shape

# 显示图像和标签
plt.imshow(image.squeeze(), cmap='gray')    # 将[1, 28, 28]->[28,28]
print('label:', label)

# 查看批量样本
batch= next(iter(train_loader))
print(len(batch))
print(type(batch))
images, labels = batch
print(images.shape)
print(labels.shape)

# 画出一批的图像
grid= torchvision.utils.make_grid(images,nrow =10)
print(grid.shape)
plt.figure(figsize=(15, 15))
plt.imshow(np.transpose(grid,(1,2,0)))   # 将张量转换成矩阵
print('labels:', labels)
# 可以通过改变batchsize来显示更多的数据