#基础依赖
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
    mnist_train = torchvision.datasets.FashionMNIST(
        root="../data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=0),
            data.DataLoader(mnist_test, batch_size, shuffle=False,
                            num_workers=0))


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)  #获取训练集和测试集


net = nn.Sequential(nn.Flatten(), 
                    nn.Linear(784, 256), #输入维度为784，输出为256
                    nn.ReLU(),        #激活函数 relu
                    nn.Linear(256, 10)  #输入为256，输出为10
                    )

def init_weights(m):  #定义初始权重
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)



net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10  #定义批量大小，学习率，训练次数

loss = nn.CrossEntropyLoss(reduction='none')  #标签大于2个时为softmax损失函数

trainer = torch.optim.SGD(net.parameters(), lr=lr)  #随机梯度下降

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
    

def train(net, epoch, loss, batch, trian_iter, trainer):  #训练函数

  metric = Accumulator(3)
  batch_nums = 0
  for X, y in train_iter:
    trainer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.mean().backward()
    trainer.step()
    metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    batch_nums += batch
    #print(batch_nums)
    if batch_nums % 10240 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(    #打印loss
        epoch, batch_nums, len(train_iter.dataset),
        100. * batch_nums/ len(train_iter.dataset), metric[0] / metric[2]))
  print('Train Epoch: {} Accuracy: {:.6f}'.format(epoch, metric[1] / metric[2]) )   #打印准确率
  return metric[0] / metric[2], metric[1] / metric[2]

train_losses_acuuras = []
for epoch in range(1, num_epochs + 1):
    train_losses_acuuras.append(train(net, epoch, loss, batch_size, train_iter, trainer))

train_losses_acuuras = torch.tensor(train_losses_acuuras)

plt.plot(train_losses_acuuras[:,0], label = 'loss')
plt.plot(train_losses_acuuras[:,1], label = 'accuracy')
plt.grid()
plt.legend()
plt.show()