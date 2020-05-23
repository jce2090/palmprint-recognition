"""
@Author: your name
@Date: 2020-05-14 09:57:03
@LastEditors: wei
@LastEditTime: 2020-05-14 11:53:35
@Description: file content
"""
import torch
import torch.nn as nn


class AlexNetModel(nn.Module):
    """AlexNet for palmprint
    """

    def __init__(self, cfg):
        super().__init__()
        self.inplane = cfg.inplane

        # Feature extract
        self.features = nn.Sequential(nn.Conv2d(self.inplane, 64, 11, stride=4, padding=2),)
        # Avgpool
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # classifier
        self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(256 * 6 * 6, 4096),)
        # hash layer
        self.hash_layer = nn.Linear(4096, cfg.code_length)

    def forward(self, x):
        """Forwar
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        x = self.hash_layer(x)
        x = torch.tanh(x)

        return x
