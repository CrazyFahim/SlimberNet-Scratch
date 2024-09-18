# Jsut a scratch code from the image on page 4:
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self,in_channels,out_channels,dropout_rate=0.3):
        super(ResNetBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3, 
            padding=1
        )
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1
        )
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # Shortcut connection for residual learning
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        # Save input for residual connection
        identity = x

        # First conv-batchnorm-relu-dropout block
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv-batchnorm block
        out = self.conv2(out)
        out = self.batchnorm2(out)

        # Residual connection
        identity = self.shortcut(identity)
        out += identity

        # Final relu and dropout
        out = self.relu(out)
        out = self.dropout(out)

        return out


class SlumberNet(nn.Module):
    def __init__(self, in_channels=2, num_classes=3, dropout_rate=0.3):
        super(SlumberNet, self).__init__()

        # Block 1-7 with doubling number of filters each time
        self.block1 = ResNetBlock(in_channels, 32, dropout_rate)
        self.block2 = ResNetBlock(32, 64, dropout_rate)
        self.block3 = ResNetBlock(64, 128, dropout_rate)
        self.block4 = ResNetBlock(128, 256, dropout_rate)
        self.block5 = ResNetBlock(256, 512, dropout_rate)
        self.block6 = ResNetBlock(512, 1024, dropout_rate)
        self.block7 = ResNetBlock(1024, 2048, dropout_rate)

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected dense layer with softmax activation for classification
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        # Pass input through ResNet blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)

        # Global Average Pooling
        x = self.global_avg_pool(x)

        # Flatten and pass through final dense layer
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        # Apply softmax for classification probabilities
        x = F.softmax(x, dim=1)

        return x
















