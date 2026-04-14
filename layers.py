import torch.nn as nn


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        # 展平操作
        self.flatten = nn.Flatten()
        # 第一全连接层
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu1 = nn.ReLU()
        # 第二全连接层
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        # 输出层
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        # 最后一层不加激活，CrossEntropyLoss 会处理
        x = self.fc3(x)
        return x
