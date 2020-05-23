'''
@Author: your name
@Date: 2020-05-10 18:23:54
@LastEditors: wei
@LastEditTime: 2020-05-12 14:04:09
@Description: file content
'''
import importlib
from torch.utils.data import DataLoader


def find_dataset_using_name(dataset_name):
    """Find dataset using name
    Arguments:
        dataset_name {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dataset_file_name = 'dataset.' + dataset_name + '_dataset'
    dataset_lib = importlib.import_module(dataset_file_name)
    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in dataset_lib.__dict__.items():
        if name.lower() == target_dataset_name.lower():
            dataset = cls

    if dataset is None:
        print('pls check your dataset in this folder')
        exit(0)

    return dataset


def create_dataset(cfg, mode, transform):
    """Create dataset

    Arguments:
        cfg {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    dataset = find_dataset_using_name(cfg.dataset_name)
    instance = dataset(cfg, mode, transform)
    print("Dataset {} {} was created, there are {} images in all".format(cfg.dataset_name, mode, len(instance)))
    dataloader = DataLoader(instance, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)


    return dataloader
