import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.layer1 = nn.Sequential(nn.Conv2d(3, 20, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(nn.Conv2d(20, 20, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(nn.Conv2d(20, 10, 3, 1, 1), nn.ReLU(), nn.MaxPool2d(2))
        self.layer4 = nn.Sequential(nn.Linear(10 * 12 * 12, 200), nn.ReLU())
        self.layer5 = nn.Sequential(nn.Linear(200, 1), nn.Sigmoid())

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)

        return out
