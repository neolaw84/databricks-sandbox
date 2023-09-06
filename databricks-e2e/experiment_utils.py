from collections import namedtuple

import torch

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

MNIST_DIR = '/tmp/data/mnist'
USE_GPU = False

data_transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])

def get_data_loader(MNIST_DIR="/tmp/data/mnist", train=True, shuffle=True, download=True, batch_size=64, num_workers=1):
    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(MNIST_DIR, train=train, download=download,
                        transform=data_transform_fn),
            batch_size=batch_size, shuffle=shuffle, num_workers=1)
    return train_loader

Params = namedtuple('Params', ['batch_size', 'test_batch_size', 'epochs', 'lr', 'momentum', 'seed', 'cuda', 'log_interval'])

class Net(nn.Module):
    def __init__(self, num_conv_channels=[10, 20], last_fc_input=50):
        super(Net, self).__init__()
        self.num_conv_channels = num_conv_channels
        self.last_fc_input = last_fc_input
        self.conv1 = nn.Conv2d(1, num_conv_channels[0], kernel_size=5)
        self.conv2 = nn.Conv2d(num_conv_channels[0], num_conv_channels[1], kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        # 4 x 4 x num_channels[1] = 16 * 20 = 320 (for default)
        self.fc1 = nn.Linear(16 * num_conv_channels[1], last_fc_input)
        self.fc2 = nn.Linear(last_fc_input, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 16 * self.num_conv_channels[1])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
    
def train_epoch(model, data, target, optimizer):
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = model(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    return loss


def test_epoch(model, data_loader, args):
    test_loss = 0
    correct = 0
    for data, target in data_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()      
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(data_loader.dataset)
    return test_loss, correct