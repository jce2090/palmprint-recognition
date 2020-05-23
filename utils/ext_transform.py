"""
@Version: 2.0
@Autor: jce2090
@Date: 2020-05-22 00:01:56
@LastEditors: Seven
@LastEditTime: 2020-05-23 08:15:41
"""
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class ToTensor:
    """Extra transform for load palmprint
    """

    def __init__(self, target_type=np.uint8):
        """Initiliaze
        """
        super().__init__()
        self.target_type = target_type

    def __call__(self, sample):
        """transform numpy data to Tensor

        Arguments:
            sample {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        if len(sample[2]) != 1:
            anchor, positive, negative = sample[0], sample[1], sample[2]
            anchor, positive, negative = F.to_tensor(anchor), F.to_tensor(positive), F.to_tensor(negative)
            anchor = anchor - torch.mean(anchor) / 128
            positive = positive - torch.mean(positive) / 128
            negative = negative - torch.mean(negative) / 128

            return [anchor.float(), positive.float(), negative.float()]
        else:
            image, label = sample[0], sample[1]
            image = F.to_tensor(image)
            if isinstance(image, np.ndarray):
                label = torch.from_numpy(label).float()

            return [image, label, sample[2]]


def ext_traindata_transform():
    """Transform for train data
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227), interpolation=Image.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )


def ext_testdata_transform():
    """Transform for test data
    """
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((227, 227), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
