import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        return F.nll_loss(F.log_softmax(input, dim=1), target)


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(3 * 64 * 64, 6)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(3 * 64 * 64, 100)
        self.fc2 = nn.Linear(100, 6)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
