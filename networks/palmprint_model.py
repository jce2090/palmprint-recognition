'''
@Author: your name
@Date: 2020-05-10 18:01:57
@LastEditors: wei
@LastEditTime: 2020-05-14 12:03:45
@Description: file content
'''
import torch
import torch.nn as nn


def basic_bottle(inplane, outplane, kernel_size, stride, padding):
    """ConvRelu

    Arguments:
        inplane {[type]} -- [description]
        outplane {[type]} -- [description]
        kernel_size {[type]} -- [description]
        padding {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    conv_relu_bn = nn.Sequential(
        nn.Conv2d(inplane, outplane, kernel_size, stride, padding),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(outplane)
    )
    return conv_relu_bn


class PalmPrintModel(nn.Module):
    """Model for Palmprint recognition

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, cfg):
        """Initializer
        """
        super().__init__()
        self.features = nn.Sequential(
            basic_bottle(cfg.inplane, 16, 3, 4, 0),
            nn.MaxPool2d(kernel_size=2, stride=1),
            basic_bottle(16, 32, 5, 2, 0),
            nn.MaxPool2d(kernel_size=2, stride=1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(7200, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh()
        )


    def forward(self, x):
        """Forward

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        # x = torch.sign(x)
        return x

    def _init_weight(self):
        """Initializer module in this Model
        """
        for cls in self.modules():
            if isinstance(cls, nn.Conv2d):
                nn.init.kaiming_normal_(cls.weight, mode='fan_in')
            if isinstance(cls, nn.BatchNorm2d):
                nn.init.constant_(cls.weight, 1)
                nn.init.constant_(cls.bias, 0)
