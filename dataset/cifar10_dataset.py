'''
@Author: wei
@Date: 2020-05-16 15:17:58
@LastEditors: wei
@LastEditTime: 2020-05-16 19:35:21
@Description: file content
'''
import torch
from torch.utils.data import Dataset
from PIL import Image
from .dataset_tool import make_img_list, make_label_list

DATA_ROOT = './data/cifar-10/images'


class Cifar10Dataset(Dataset):
    """Cifar10 for test accuracy

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, cfg, mode, transform):
        """Initilaizer class

        Arguments:
            cfg {[type]} -- [description]
            mode {[type]} -- [description]
            transform {[type]} -- [description]
        """
        self.cfg = cfg
        self.mode = mode
        self.transform = transform

        self.imgs = make_img_list(DATA_ROOT, mode)
        self.labels = make_label_list(DATA_ROOT, mode)
        self._check()

    def __getitem__(self, index):
        """Get single image

        Arguments:
            index {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        img = Image.open(self.imgs[index]).convert('L')
        label = self.labels[index]

        img = img.resize((227, 227))
        sample = [img.copy(), label, index]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return length of images and labels

        Returns:
            [type] -- [description]
        """
        return len(self.imgs)

    def _check(self):
        """check correction
        """
        assert len(self.imgs) == len(self.labels)
