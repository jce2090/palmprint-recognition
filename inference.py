"""
@Author: wei
@Date: 2020-05-18 20:35:53
@LastEditors: wei
@LastEditTime: 2020-05-20 15:19:05
@Description: file content
"""
import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from utils import PairwiseDistance
from utils import get_configuration
from networks import create_model
from dataset import create_dataset
from networks.yolov3_model import *
from utils.utils import *
from utils.parse_config import *
import torchvision.transforms.functional as F
from PIL import Image
from utils import ToTensor


class InferenceEngine(object):
    """Inference results

    Arguments:
        object {[type]} -- [description]
    """

    def __init__(self, cfg, model, dataloader=None):
        """Initializer model

        Arguments:
            cfg {[type]} -- [description]
            model {[type]} -- [description]
            dataloader {[type]} -- [description]
        """
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataprocess = None
        self.l2_dist = PairwiseDistance()
        self.base_path = "./data/IITD/flip/ROI/test"

    def setup(self):
        """Set up
        """
        print("Initializer inference model")
        checkpoints = torch.load("./checkpoints/facenet/checkpoint_1000.pth")

        pretrained_dict = checkpoints["model"]

        model_dict = self.model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.model.load_state_dict(pretrained_dict)
        self.model.eval()
        self.model.to(self.device)

    def inference(self):
        """Inference results
        """
        total_positive_loss = 0
        total_negative_loss = 0
        n = len(self.dataloader)
        pos_list, neg_list = [], []
        with torch.no_grad():
            for anchor, positive, negative in self.dataloader:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                anchor_embedding = self.model(anchor)
                pos_embedding = self.model(positive)
                neg_embedding = self.model(negative)

                positive_loss = self.l2_dist(anchor_embedding, pos_embedding).detach().cpu().numpy()
                negative_loss = self.l2_dist(anchor_embedding, neg_embedding).detach().cpu().numpy()

                pos_list.extend(positive_loss)
                neg_list.extend(negative_loss)

                total_positive_loss += positive_loss
                total_negative_loss += negative_loss

                print("pos:{}, neg:{}".format(positive_loss, negative_loss))
            print(
                "mean_loss_positive:{}, mean_loss_negative:{}".format(total_positive_loss / n, total_negative_loss / n)
            )
            sns.distplot(pos_list, rug=True, color="r")
            sns.distplot(neg_list, rug=True, color="c")
            plt.show()

    def save_code_base(self, image_path):
        """Save base img

        Arguments:
            img_path {[type]} -- [description]
        """
        img = Image.open(os.path.join(self.base_path, image_path)).convert("L")
        img = img.resize((227, 227))
        anchor = F.to_tensor(img).unsqueeze(0).cuda()
        anchor = anchor - torch.mean(anchor) / 128
        anchor_embedding = self.model(anchor)
        self.anchor_emdedding = anchor_embedding
        print("your palmprint data has been saved")

    def calculate_similarity(self, test_img_path):
        """Calculate

        Arguments:
            test_img_path {[type]} -- [description]
        """
        img = Image.open(os.path.join(self.base_path, test_img_path)).convert("L")
        img = img.resize((227, 227))
        test = F.to_tensor(img).unsqueeze(0).cuda()
        test = test - torch.mean(test) / 128
        test_embedding = self.model(test)

        l2_dist = self.l2_dist(self.anchor_emdedding, test_embedding)
        print(l2_dist.detach().item())

        if l2_dist < 10:
            print("correct")
        else:
            print("wrong")
        print("=" * 55)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    cfg = get_configuration()
    model = create_model(cfg)
    valload = create_dataset(cfg, "train", transform=ToTensor())
    # dataloader = create_dataset(cfg, 'test', transform=ToTensor())
    inference_engie = InferenceEngine(cfg, model, valload)
    inference_engie.setup()
    text = "=" * 80
    print("\033[33m{}\033[0m".format(text))
    # test single image
    # img_path = input("Enter the picture you want to save:")
    # inference_engie.save_code_base(img_path)
    # while True:
    #     test_path = input("Enter the picture you want to test:")
    #     if test_path == "q":
    #         break
    #     inference_engie.calculate_similarity(test_path)

    inference_engie.inference()
