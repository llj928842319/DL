import torch
from torch import nn



#查询可用的GPU的数量
torch.cuda.device_count()

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  #@save
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]"""
    devices = [torch.device(f'cuda:{i}')
             for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]


#神经网络可以以指定设备。 下面的代码将模型参数放在GPU上。

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_gpu())
