"""
@Version: 2.0
@Autor: jce2090
@Date: 2020-05-22 00:01:56
@LastEditors: Seven
@LastEditTime: 2020-05-24 00:46:06
"""
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from dataset.dataset_tool import make_triple_img_list


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
        self.image_list, self.interval = make_triple_img_list(cfg.data_root, mode)
        self._check()

    def __getitem__(self, idx):
        """get single dataset

        Arguments:
            idx {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        random_index = [random.randint(0, self.interval - 1) for _ in range(3)]
        random_index_different = random.randint(0, self.cfg.num_classes - 1)
        while random_index_different == idx:
            random_index_different = random.randint(0, self.cfg.num_classes - 1)

        anchor = Image.open(self.image_list[idx][random_index[0]]).convert("L")
        positive = Image.open(self.image_list[idx][random_index[1]]).convert("L")
        negative = Image.open(self.image_list[random_index_different][random_index[2]]).convert("L")

        sample = [np.asarray(anchor).copy(), np.asarray(positive).copy(), np.asarray(negative).copy()]
        if self.transform:
            for i in range(3):
                sample[i] = self.transform(sample[i])

        return sample

    def __len__(self):
        """Return length of data_list

        Returns:
            [type] -- [description]
        """
        return len(self.image_list)

    def _check(self):
        """Check if the length is even.
        """
        assert len(self.image_list) % 2 == 0
