'''
@Author: wei
@Date: 2020-05-14 13:00:54
@LastEditors: wei
@LastEditTime: 2020-05-20 15:33:49
@Description: file content
'''
import time
import os
import threading
import psutil
import humanize
import GPUtil as GPU
import torch
from thop import profile

GPUs = GPU.getGPUs()
gpu = GPUs[0]
SHOW_GPU_USAGE_TIME = 60


def print_network(model, cfg):
    """Print basic parameters of model

    Arguments:
        model {[type]} -- [description]
        cfg {[type]} -- [description]
    """
    # print(model)
    print('-----------param and flops of {}------------'.format(cfg.model_name))
    input_image, model = torch.randn(1, 1, 227, 227).cuda(), model.cuda()
    flops, params = profile(model, inputs=(input_image,), verbose=False)
    print('FLOPS: {:.4f}G, PARAMS: {:.4f}M'.format(flops / (1000 ** 3), params / (1000 ** 2)))
    return flops / (1000 ** 3), params / (1000 ** 2)


def worker():
    """Print usage of GPU
    """
    if SHOW_GPU_USAGE_TIME == 0:
        return
    while True:
        process = psutil.Process(os.getpid())
        print("\n Gen RAM Free:" + humanize.naturalsize(psutil.virtual_memory().available),
              "I Proc size:" + humanize.naturalsize(process.memory_info().rss))
        print("GPU RAM Free:{0:.0f}MB, Used:{1:.0f}MB, Util:{2:3.0f}%, Total:{3:.0f}MB, Load:{4:.0f}%".format(gpu.memoryFree, gpu.memoryUsed,
                                                                                                              gpu.memoryUtil*100, gpu.memoryTotal, gpu.load))
        time.sleep(SHOW_GPU_USAGE_TIME)


def monitor():
    """Interval time of monitor

    Arguments:
        interval {[type]} -- [description]
    """
    t = threading.Thread(target=worker, name='Monitor')
    t.start()
