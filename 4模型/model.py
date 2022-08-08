""" 面向对象编程与pytorch的结合
构建一个神经网络的主要组件是层（pytorch神经网络库中包含了帮助构造层的类）
神经网络中的每一层都有两个主要组成部分：转换和权重（转换代表代码；权重代表数据）
forward方法（前向传输）：张量通过每层的变换向前流动，直到达到输出层
构建神经网络时必须提供前向方法，前向方法即为实际的变换
使用pytorch创建神经网络的步骤：
1.扩展nn.Module基类
2.定义层(layers)为类属性
3.实现前向方法"""
import torch
# CNN网络的建立
import torch.nn as nn
class Network(nn.Module):   #()中加入nn.Module可以使得Network类继承Module基类中的所有功能
    def __init__(self):
        super(Network, self).__init__()     # 对继承的父类的属性进行初始化，使用父类的方法来进行初始化
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)        # 从卷积层传入线性层需要对张量flatten
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    def forward(self, t):
        # implement the forward pass
        return t
    
network = Network()     # 创建网络对象network
network

""" CNN前向方法的实现
前向方法的实现将使用我们在构造函数中定义的所有层
前向方法实际上是输入张量到预测的输出张量的映射 #### 3.8.1 Input Layer
输入层是由输入数据决定的
输入层可以看做是恒等变换 f(x)=x
输入层通常是隐式存在的"""

import torch.nn as nn
import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120,out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)
    def forward(self,t):
        # (1) input layer
        t = t
        # (2) hidden conv layer1
        t = self.conv1(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # (3) hidden conv layer2
        t = self.conv2(t)
        t = F.relu(t)
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        # relu 和 max pooling 都没有权重；激活层和池化层的本质都是操作而非层；层与操作的不同之处在于，层有权重，操作没有
        #（4）hidden linear layer2
        t = t.reshape(-1, 12*4*4)
        t = self.fc1(t)
        t = F.relu(t)
        # (5) hidden linear layer2
        t = self.fc2(t)
        t = F.relu(t)
        # (6) output layer
        t = self.out(t)
        # t= F.softmax(t, dim=1)  # 这里暂不使用softmax，在训练中使用交叉熵损失可隐式的表示softmax
        # 在隐藏层中，通常使用relu作为非线性激活函数
        # 在输出层，有类别要预测时，使用


