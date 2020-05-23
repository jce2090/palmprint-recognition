'''
@Author: your name
@Date: 2020-05-10 18:40:02
@LastEditors: wei
@LastEditTime: 2020-05-20 11:48:17
@Description: file content
'''
import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


IMG_EXTENSIONS = ['PNG', 'BMP', 'bmp', 'png']


def is_image_file(filename):
    """[summary]

    Arguments:
        filename {[type]} -- [description]
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_img_list(data_root, mode):
    """Make img list"""
    full_data_path = os.path.join(data_root, mode)
    assert os.path.exists(full_data_path)
    images = []
    for root, _, fnames in os.walk(full_data_path):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(root, fname))

    images = sorted(images)
    return images


def make_label_list(data_root, mode):
    """Make label list

    Arguments:
        data_root {[type]} -- [description]
        mode {[type]} -- [description]
    """
    full_label_path = os.path.join(data_root, mode)
    assert os.path.exists(full_label_path)
    labels = []
    for _, _, fnames in os.walk(full_label_path):
        for fname in fnames:
            labels.append(fname.split('_')[1])

    labels = sorted(np.asarray(labels))
    values = np.unique(labels)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # binary encode
    onehot_encoded = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoded.fit_transform(integer_encoded)

    new_labels = []
    for i in range(len(labels)):
        new_labels.append(onehot_encoded[int(labels[i]) - 1])

    return np.asarray(new_labels)


def generate_code(model, dataloader, code_length, device):
    """Generate code to acquire MAP of palmprint

    Arguments:
        model {[type]} -- [description]
        dataloader {[type]} -- [description]
        code_length {[type]} -- [description]
        device {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    model.eval()
    model = model.to(device)
    code = torch.zeros(len(dataloader), code_length)
    with torch.no_grad():
        for imgs, labels, index in dataloader:
            imgs = imgs.to(device, torch.float32), labels.to(device, torch.float32), index.to(device, torch.float32)
            hash_code = model(imgs)
            code[index, :] = hash_code.sign().cpu()
        model.train()

    return code


def make_triple_img_list(data_root, mode):
    """Make triple_img list"""
    full_data_path = os.path.join(data_root, mode)
    assert os.path.exists(full_data_path)
    images = []
    for root, _, fnames in os.walk(full_data_path):
        for fname in fnames:
            if is_image_file(fname):
                images.append(os.path.join(root, fname))

    images = sorted(images)
    triple_images = []
    interval = int(len(images) / 232)

    for i in range(232):
        triple_images.append(images[i*interval: (i+1)*interval])
    return triple_images, interval
# if __name__ == "__main__":
#     print(os.getcwd())
#     data_root = './data/IITD/flip/ROI/'
#     mode = 'train'
#     images = make_img_list(data_root, mode)
#     labels = make_label_list(data_root, mode)
#     print(len(images), len(labels))
#     print(labels[0])
