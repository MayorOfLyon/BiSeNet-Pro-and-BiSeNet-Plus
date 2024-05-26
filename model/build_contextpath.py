import torch
from torchvision import models
from torch import nn

class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2,padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))

class stem_block(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__( )
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        
        self.left = ConvBlock(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.right = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.final = ConvBlock(out_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
    def forward(self, input):
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x_left = self.left(x)
        x_right = self.right(x)
        x = torch.cat([x_left, x_right], 1)
        x = self.final(x)
        return x
    
class resnet18(torch.nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.features = models.resnet18(pretrained=pretrained)
        self.conv1 = self.features.conv1
        self.bn1 = self.features.bn1
        self.relu = self.features.relu
        self.maxpool1 = self.features.maxpool
        # self.stem = stem_block(3, 64)
        
        self.layer1 = self.features.layer1
        self.layer2 = self.features.layer2
        self.layer3 = self.features.layer3
        self.layer4 = self.features.layer4 
        
    def forward(self, input):
        # x = self.stem(input)
        x = self.conv1(input)
        x = self.relu(self.bn1(x))
        x = self.maxpool1(x)
        
        feature1 = self.layer1(x)  # 1 / 4
        feature2 = self.layer2(feature1)  # 1 / 8
        feature3 = self.layer3(feature2)  # 1 / 16
        feature4 = self.layer4(feature3)  # 1 / 32
        # global average pooling to build tail
        tail = torch.mean(feature4, 3, keepdim=True)
        tail = torch.mean(tail, 2, keepdim=True)
        return feature3, feature4, tail

def build_contextpath(name):
    model = {
        'resnet18': resnet18(pretrained=True),
        # 'resnet101': resnet101(pretrained=True)
    }
    return model[name]
