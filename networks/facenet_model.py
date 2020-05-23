"""
@Author: wei
@Date: 2020-05-14 22:55:45
@LastEditors: wei
@LastEditTime: 2020-05-18 11:18:38
@Description: file content
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18


class FaceNetModel(nn.Module):
    """Model for extracting features"""

    def __init__(self, cfg, embedding_size=128, num_classes=230, pretrained=True):
        """Initializer FaceNet

        Arguments:
            nn {[type]} -- [description]
            embedding_size {[type]} -- [description]
            num_classes {[type]} -- [description]
        """
        super(FaceNetModel, self).__init__()

        self.cfg = cfg
        self.model = resnet18(pretrained)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.embedding_size = embedding_size
        self.model.fc = nn.Linear(512 * 8 * 8, self.embedding_size)
        self.model.classifier = nn.Linear(self.embedding_size, num_classes)
        self.alpha = 10

    def l2_norm(self, x):
        """l2 normalization
        Arguments:
            input {[type]} -- [description]
        """
        input_size = x.size()
        # buffer = torch.pow(x, 2)
        buffer = x * x

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        norm = norm.view(-1, 1)
        new_norm = norm * torch.ones([1, 128]).cuda()
        _output = torch.div(x, new_norm)
        output = _output.view(input_size)

        return output

    def forward(self, x):
        """Forward

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        x = self.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.model.fc(x)
        features = self.l2_norm(x)
        # Multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        features = features * self.alpha

        return features

    def forward_classifier(self, x):
        """Classifier
        """
        features = self.forward(x)
        res = self.model.classifier(features)

        return res
