"""
@Author: your name
@Date: 2020-05-10 18:24:09
@LastEditors: wei
@LastEditTime: 2020-05-18 06:08:21
@Description: file content
"""
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from dataset.dataset_tool import make_img_list, make_label_list


class PalmPrintV1Dataset(Dataset):
    """Dataset of palmprint

    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(self, cfg, mode, transform):
        """Initializer dataset

        Arguments:
            cfg {[type]} -- [description]
            mode {[type]} -- [description]
            transform {[type]} -- [description]
        """
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.transform = transform

        # Create image_list and label_list
        self.image_list = make_img_list(cfg.data_root, mode)
        self.label_list = make_label_list(cfg.data_root, mode)
        self.onehot_target = torch.from_numpy(self.label_list).float()
        self._check()

    def __getitem__(self, idx):
        """get single dataset

        Arguments:
            idx {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        img = Image.open(self.image_list[idx]).convert("L")
        img = img.resize((227, 227), resample=Image.BILINEAR)
        label = self.label_list[idx]

        sample = [np.asarray(img).copy(), np.asarray(label).copy(), idx]
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        """Return length of data_list

        Returns:
            [type] -- [description]
        """
        return len(self.image_list)

    def _check(self):
        """Check for consistency in the length of data
        """
        if len(self.image_list) != (len(self.label_list)):
            print("Pls check your dataset")
            exit(0)

    def get_onehot_targets(self):
        """Return onehot labels
        """
        return self.onehot_target
