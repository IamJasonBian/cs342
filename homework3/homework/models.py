import torch
import torch.nn.functional as F
import torch.nn as nn


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)
        self.skip = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                torch.nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(x)
        out = self.relu(out)
        return out

class CNNClassifier(torch.nn.Module):
    def __init__(self, n_input_channels=1, n_output_channels=6, layers=[16, 32, 64, 128]):
        super(CNNClassifier, self).__init__()

        self.in_channels = layers[0]
        self.conv1 = torch.nn.Conv2d(n_input_channels, self.in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(self.in_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        
        layer_blocks = []
        for idx, num_channels in enumerate(layers[1:]):
            strides = 2 if idx == 0 else 1
            layer_blocks.append(ResidualBlock(self.in_channels, num_channels, strides))
            self.in_channels = num_channels
        self.layers = torch.nn.Sequential(*layer_blocks)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1,1))
        self.fc = torch.nn.Linear(self.in_channels, n_output_channels)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class FCN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.resblock1 = ResidualBlock(64, 64)
        
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.resblock2 = ResidualBlock(128, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        
        self.upconv2 = nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x1 = self.resblock1(x1)
        
        x2 = F.max_pool2d(x1, 2)
        x2 = F.relu(self.conv2(x2))
        x2 = self.resblock2(x2)
        
        x3 = self.upconv1(x2)
        
        # Adding skip connection
        x3 = x3 + x1
        
        x4 = self.upconv2(x3)
        
        return x4


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
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
