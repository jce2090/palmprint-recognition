"""
@Version: 2.0
@Autor: jce2090
@Date: 2020-05-22 00:01:56
@LastEditors: Seven
@LastEditTime: 2020-05-22 12:38:06
"""
"""
@Author: your name
@Date: 2020-05-09 09:26:01
@LastEditors: wei
@LastEditTime: 2020-05-20 12:01:10
@Description: file content
"""
import os
import argparse
import torch


def get_configuration():
    """[summary]
    """
    parser = argparse.ArgumentParser(description="PyTorch Detector Training")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--log_path", type=str, default="./logs/wei.log")
    parser.add_argument("--training_cfg", type=str, default=None)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--opt-level", type=str, default="O1")
    parser.add_argument("--keep-batchnorm-fp32", type=str, default=None)
    parser.add_argument("--loss-scale", type=str, default=None)
    parser.add_argument("--sync_bn", action="store_true", help="enabling apex sync BN.")
    parser.add_argument("--use_sgd", default=False)
    parser.add_argument("--lr", default=1e-2)
    # train params
    parser.add_argument("--model_name", default="facenet")
    parser.add_argument("--dataset_name", default="palmprintv1")
    parser.add_argument("--data_root", default="./data/IITD/flip/ROI/")
    parser.add_argument("--inplane", default=1)
    parser.add_argument("--epochs", default=5000)
    parser.add_argument("--save_dir", default="./checkpoints/")
    parser.add_argument("--resume", default=False)
    parser.add_argument("--classes", default=2)
    parser.add_argument("--train_img_size", default=(150, 150))
    parser.add_argument("--save_iter", default=3)
    parser.add_argument("--batch_size", default=2)
    parser.add_argument("--num_workers", default=4)
    parser.add_argument("--lamda", default=0.5)
    parser.add_argument("--save_interval", default=1000)
    parser.add_argument("--val_interval", default=100)
    parser.add_argument("--save_path", default="./checkpoints/facenet/checkpoint_2500.pth")
    parser.add_argument("--code_length", default=128)
    parser.add_argument("--monitor", default=True)

    return parser.parse_args()


def setup_training():
    """[summary]
    """
    assert torch.cuda.is_available()
    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
        print("save_path was created")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    print("model parameters setup correctly")
