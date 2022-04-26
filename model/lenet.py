import torch
import torch.nn as nn
from torchsummary import summary


class leNet(nn.Module):
    def __init__(self):
        super(leNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5))
        self.relu = nn.ReLU()
        self.mpool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5))
        self.f4 = nn.Linear(120*9*9, 84)
        self.f5 = nn.Linear(84, 8)
        self.logM = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.mpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.mpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.f4(x)
        x = self.relu(x)
        x = self.f5(x)
        x = self.logM(x)
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# test = torch.rand((1, 1, 64, 64), device=device)
# print(test)
# net = leNet()
# net = net.to(device=device)
# res = net(test)
# print(res.size())
# summary(net, (1, 32, 32))


