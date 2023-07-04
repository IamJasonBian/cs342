import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:  # Add 1x1 convolution for channel matching
            self.conv_res = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv_res = None

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv_res is not None:
            residual = self.conv_res(residual)
        out += residual
        out = F.relu(out)
        return out


class CNNClassifier(nn.Module):
    def __init__(self, layers=[16, 32, 64, 128], n_input_channels=1, n_output_channels=6, kernel_size=5, dropout_p=0.5):
        super().__init__()

        self.network = []
        c = n_input_channels
        for l in layers:
            self.network.extend([
                ResidualBlock(c, l),
                nn.BatchNorm2d(l),  # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_p)  # Dropout
            ])
            c = l
        self.network = nn.Sequential(*self.network)
        self.classifier = nn.Linear(c, n_output_channels)

    def forward(self, x):
        x = self.network(x)
        x = x.mean(dim=[2, 3])  # Average pooling across spatial dimensions
        return self.classifier(x)


class FCN(nn.Module):
    def __init__(self, in_channels=3,  num_classes=5):
        super().__init__()

                # Encoder
        self.enc_conv1 = DoubleConv(in_channels, 8)  # from 16
        self.enc_pool1 = nn.MaxPool2d(2)
        self.enc_conv2 = DoubleConv(8, 16)  # from 32
        self.enc_pool2 = nn.MaxPool2d(2)
        self.enc_conv3 = DoubleConv(16, 32)  # from 64
        self.enc_pool3 = nn.MaxPool2d(2)
        self.enc_conv4 = DoubleConv(32, 64)  # from 128
        self.enc_pool4 = nn.MaxPool2d(2)

        # Bridge
        self.bridge_conv = DoubleConv(64, 128)  # from 256

        # Decoder
        self.dec_upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)  # from 256, 128
        self.dec_conv1 = DoubleConv(128, 64)  # from 256, 128
        self.dec_upconv2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)  # from 128, 64
        self.dec_conv2 = DoubleConv(64, 32)  # from 128, 64
        self.dec_upconv3 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)  # from 64, 32
        self.dec_conv3 = DoubleConv(32, 16)  # from 64, 32
        self.dec_upconv4 = nn.ConvTranspose2d(16, 8, kernel_size=2, stride=2)  # from 32, 16
        self.dec_conv4 = DoubleConv(16, 8)  # from 32, 16

        # Output
        self.output_conv = nn.Conv2d(8,  num_classes, kernel_size=1)  # from 16


    def forward(self, x):
        # Encoder
        enc1 = self.enc_conv1(x)
        enc2 = self.enc_conv2(enc1)
        enc3 = self.enc_conv3(enc2)
        enc4 = self.enc_conv4(enc3)

        # Bridge
        bridge = self.bridge_conv(enc4)

        # Decoder
        dec1 = self.dec_upconv1(bridge)
        # Resize to match size of corresponding encoder layer output
        enc4_resized = F.interpolate(enc4, size=dec1.shape[2:])
        dec1 = torch.cat((enc4_resized, dec1), dim=1)
        dec1 = self.dec_conv1(dec1)

        dec2 = self.dec_upconv2(dec1)
        # Resize to match size of corresponding encoder layer output
        enc3_resized = F.interpolate(enc3, size=dec2.shape[2:])
        dec2 = torch.cat((enc3_resized, dec2), dim=1)
        dec2 = self.dec_conv2(dec2)

        dec3 = self.dec_upconv3(dec2)
        # Resize to match size of corresponding encoder layer output
        enc2_resized = F.interpolate(enc2, size=dec3.shape[2:])
        dec3 = torch.cat((enc2_resized, dec3), dim=1)
        dec3 = self.dec_conv3(dec3)

        dec4 = self.dec_upconv4(dec3)
        # Resize to match size of corresponding encoder layer output
        enc1_resized = F.interpolate(enc1, size=dec4.shape[2:])
        dec4 = torch.cat((enc1_resized, dec4), dim=1)
        dec4 = self.dec_conv4(dec4)

        # Output
        output = self.output_conv(dec4)

        return output

    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

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
