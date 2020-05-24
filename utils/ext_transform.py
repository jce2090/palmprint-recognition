'''
@Author: your name
@Date: 2020-05-11 11:54:03
@LastEditors: wei
@LastEditTime: 2020-05-19 15:15:48
@Description: file content
'''
import torch
import numpy as np
import  torchvision.transforms.functional as F


class ToTensor():
    """Ext transform for load palmprint
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
            img1, img2, img3 = sample[0], sample[1], sample[2]
            img1, img2, img3 = F.to_tensor(img1), F.to_tensor(img2), F.to_tensor(img3)
            img1 = img1 - torch.mean(img1) / 128
            img2 = img2 - torch.mean(img2) / 128
            img3 = img3 - torch.mean(img3) / 128

            return [img1.float(), img2.float(), img3.float()]
        else:
            img, label = sample[0], sample[1]
            img = F.to_tensor(img)
            if isinstance(img, np.ndarray):
                label = torch.from_numpy(label).float()

            return [img, label, sample[2]]
