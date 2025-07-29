import torch.nn as nn
import torchvision.models as models
import torch

class regressor(nn.Module):
    def __init__(self):
        super(regressor, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)
        #resnet = models.resnet18(weights = None, norm_layer=lambda num_features: nn.GroupNorm(8, num_features))
        #norm_layer=lambda num_features: nn.GroupNorm(8, num_features)
        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:1, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        #self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features

        self.num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2


class registration(nn.Module):
    def __init__(self):
        super(registration, self).__init__()

        # Load the pre-trained ResNet-18 model
        resnet = models.resnet18(weights = None, norm_layer=nn.InstanceNorm2d)

        # Modify the first layer to accept grayscale images
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.weight = nn.Parameter(resnet.conv1.weight[:, 0:2, :, :])

        # Replace the remaining layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # self.dropout = nn.Dropout(0.2)

        # Replace the last fully connected layer
        num_features = resnet.fc.in_features
        self.fc1 = nn.Linear(num_features, 3)
        self.fc2 = nn.Linear(num_features, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        #x = self.dropout(x)

        output1 = self.fc1(x)
        output2 = self.fc2(x)
        return output1, output2
