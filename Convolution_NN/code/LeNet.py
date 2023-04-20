import torch
from torch import nn
import torchvision
from torch.utils import data
from torchvision import transforms


def load_data_fashion_mnist(batch_size, resize=None):  
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    trans = [transforms.ToTensor()]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)
    mnist_train = torchvision.datasets.MNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.MNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)  #获取训练集和测试集

net = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(6, 16, kernel_size=5), nn.Sigmoid(),
                    nn.AvgPool2d(kernel_size=2, stride=2),
                    nn.Flatten(),
                    nn.Linear(16 * 5 * 5, 120), nn.Sigmoid(),
                    nn.Linear(120, 84), nn.Sigmoid(),
                    nn.Linear(84, 10))
'''
AlexNet
net = nn.Sequential(
# 这⾥使⽤⼀个11*11的更⼤窗⼝来捕捉对象。
# 同时，步幅为4，以减少输出的⾼度和宽度。
# 另外，输出通道的数⽬远⼤于LeNet
    nn.Conv2d(1, 96, kernel_size = 11, stride = 4, padding = 1), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    # 减⼩卷积窗⼝，使⽤填充为2来使得输⼊与输出的⾼和宽⼀致，且增⼤输出通道数
    nn.Conv2d(96, 256, kernel_size = 5, padding = 2), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    # 使⽤三个连续的卷积层和较⼩的卷积窗⼝。
# 除了最后的卷积层，输出通道的数量进⼀步增加。
# 在前两个卷积层之后，汇聚层不⽤于减少输⼊的⾼度和宽度
    nn.Conv2d(256, 384, kernel_size = 3, padding = 1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size = 3, padding = 1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size = 3, padding = 1), nn.ReLU(),
    nn.MaxPool2d(kernel_size = 3, stride = 2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p = 0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p = 0.5),
    nn.Linear(4096, 10)
)

'''

'''
VGG-11

def vgg_block(num_conv, in_channels, out_channel):
    layers = []
    for _ in range(num_conv):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1))
        layers.append(nn.ReLU())
        in_channels  = out_channels

    layers.append(nn.MaxPool2d(kernel_size = 2, stide = ))
    return nn.Sequential(*layers)

conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))  (conv_layers, out_channels)

def vgg(conv_arch):
    conv_blks = []
    in_channels = 1

    for (num_convs, out_channels) in  conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(*conv_blks, nn.Flatten(),
    nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(4096, 10))  假设这里是十分类
        
vgg11 = vgg(conv_arch)
'''

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

def accuracy(y_hat, y): 
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

class Accumulator: 
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    



def train(net, epoch, loss, batch, trian_iter, trainer):
  
  metric = Accumulator(3)
  batch_nums = 0
  net.train()
  for  X, y in train_iter:
    trainer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    trainer.step()
    with torch.no_grad():
      metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
    batch_nums += batch
    if batch_nums % 10240 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_nums, len(train_iter.dataset),
        100. * batch_nums/ len(train_iter.dataset), metric[0] / metric[2]))
  print('Train Epoch: {} Accuracy: {:.6f}'.format(epoch, metric[1] / metric[2]) )
  return [metric[0] / metric[2], metric[1] / metric[2]]


net.apply(init_weights)

batch_size, lr, num_epochs = 256, 0.9, 10  #定义批量大小，学习率，训练次数

loss = nn.CrossEntropyLoss()  #标签大于2个时为softmax

trainer = torch.optim.SGD(net.parameters(), lr = lr)  #随机梯度下降  

for epoch in range(1, num_epochs + 1):
  train(net, epoch, loss, batch_size, train_iter, trainer)