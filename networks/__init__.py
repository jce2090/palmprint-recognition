'''
@Author: your name
@Date: 2020-05-10 15:31:40
@LastEditors: wei
@LastEditTime: 2020-05-15 04:10:39
@Description: file content
'''
import importlib


def find_model_using_name(model_name):
    """Find model using name
    Arguments:
        model_name {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    model_file_name = 'networks.' + model_name + '_model'
    model_lib = importlib.import_module(model_file_name)
    target_model_name = model_name.replace('_', '') + 'model'
    model = None
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower():
            model = cls

    if model is None:
        print('pls input correct model name')
        exit(0)
    return model


def create_model(cfg):
    """Create model

    Arguments:
        cfg {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    model = find_model_using_name(cfg.model_name)
    instance = model(cfg)
    print('Model {} was created'.format(type(instance).__name__))
    return instance
