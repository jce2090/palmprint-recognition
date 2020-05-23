"""
@Author: your name
@Date: 2020-05-12 21:53:00
@LastEditors: Seven
@LastEditTime: 2020-05-22 00:59:47
@Description: file content
"""
import os
import time
import logging
from comet_ml import Experiment, ExistingExperiment
import torch
import torch.optim as optim
import numpy as np
import tqdm
from metrice import evaluate
from utils.visualizer import print_network, monitor
from dataset import create_dataset
from networks import create_model
from utils import get_configuration, setup_training, ToTensor, DNHloss
from utils import PairwiseDistance, TripletMarginLoss


cfg = get_configuration()

hyper_parameters = {
    "lr": cfg.lr,
    "batch_size": cfg.batch_size,
    "model_name": cfg.model_name,
    "dataset_name": cfg.dataset_name,
    "resume": cfg.resume,
}

if not cfg.resume:
    experiment = Experiment(
        api_key="c7k4G3pkTcyL6mbvZSkg8LcXs",
        project_name="palmprint",
        workspace="jce1995",
        log_graph=False,
        auto_param_logging=False,
        auto_metric_logging=False,
    )
else:
    previous_experiment = input("pls input previous_experiment api key")
    experiment = ExistingExperiment(api_key="c7k4G3pkTcyL6mbvZSkg8LcXs", previous_experiment=previous_experiment)
experiment.add_tag("wei")
experiment.set_name("{}_{}_lr_{}".format(cfg.model_name, cfg.dataset_name, cfg.lr))
experiment.log_parameters(hyper_parameters)


class TrainEngine(object):
    """Train engine
    """

    def __init__(self, cfg, model, optimizer, criterion, trnload, valload, logger=None):
        """Initialize TrainEngie
        """
        self.cfg = cfg
        self.epochs = cfg.epochs
        self.optim = optimizer
        self.model = model
        self.trainload = trnload
        self.valload = valload
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = criterion
        self.save_dir = cfg.save_dir
        self.logger = logger
        self.save_iter = cfg.save_iter - 1
        self.classes = cfg.classes
        self.resume = cfg.resume
        self.train_img_size = cfg.train_img_size
        self.start_epoch = 0
        # Using comet_ml to record the training process

    def setup(self):
        """Setup essential params
        """
        print("setup training parameters")
        # Calulate params of model
        flops, params = print_network(self.model, self.cfg)
        model_parameters = {
            "Model": self.cfg.model_name,
            "Flops": flops,
            "Params": params,
        }
        logging.info("model: {}, flops: {}, params: {}".format(self.cfg.model_name, flops, params))
        experiment.log_parameters(model_parameters)

        # Restore model
        checkpoints_path = os.path.join("./checkpoints/", self.cfg.model_name)
        if not os.path.exists(checkpoints_path):
            os.makedirs(checkpoints_path)

        if self.resume:
            checkpoints = torch.load(self.cfg.save_path)
            pretrained_dict = checkpoints["model"]
            model_dict = self.model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.model.load_state_dict(pretrained_dict)
            self.start_epoch = checkpoints["epoch"]
            print("Resume Model {} from epoch {}".format(self.model.__class__.__name__, self.start_epoch))
        self.model.to(self.device)

        # Monitor for GPU and memory
        if self.cfg:
            monitor()

    def train(self):
        """Minimum training unit
        """

        with experiment.train():
            for epoch in range(self.start_epoch, self.epochs):
                experiment.log_current_epoch(epoch)
                labels, distances = [], []
                running_loss = 0
                dataprocess = tqdm.tqdm(self.trainload, position=0, leave=True)
                for data_a, data_p, data_n in dataprocess:
                    start_time = time.time()
                    anc_img, pos_img, neg_img = (
                        data_a.to(self.device),
                        data_p.to(self.device),
                        data_n.to(self.device),
                    )
                    # Compute output
                    anc_embed, pos_embed, neg_embed = {
                        self.model(anc_img),
                        self.model(pos_img),
                        self.model(neg_img),
                    }
                    # Choose the hard negative
                    l2_dist = PairwiseDistance().to(self.device)
                    pos_dist = l2_dist.forward(anc_embed, pos_embed)
                    neg_dist = l2_dist.forward(anc_embed, neg_embed)

                    all = (neg_dist - pos_dist < self.cfg.margin).cpu().numpy().flatten()
                    hard_triplets = np.where(all == 1)
                    if len(hard_triplets[0]) == 0:
                        continue
                    anc_hard_embed = anc_embed[hard_triplets].to(self.device)
                    pos_hard_embed = pos_embed[hard_triplets].to(self.device)
                    neg_hard_embed = neg_embed[hard_triplets].to(self.device)

                    anc_hard_img = anc_img[hard_triplets].to(self.device)
                    pos_hard_img = pos_img[hard_triplets].to(self.device)
                    neg_hard_img = neg_img[hard_triplets].to(self.device)

                    anc_img_pred = self.model.forward_classifier(anc_hard_img).to(self.device)
                    pos_img_pred = self.model.forward_classifier(pos_hard_img).to(self.device)
                    neg_img_pred = self.model.forward_classifier(neg_hard_img).to(self.device)
                    # Calculate triplet loss
                    loss = (
                        TripletMarginLoss(margin=self.cfg.margin)
                        .forward(anchor=anc_hard_embed, positive=pos_hard_embed, negative=neg_hard_embed)
                        .to(self.device)
                    )
                    # Backward
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    dists = l2_dist.forward(anc_embed, pos_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.ones(dists.size(0)))

                    dists = l2_dist.forward(anc_embed, neg_embed)
                    distances.append(dists.data.cpu().numpy())
                    labels.append(np.zeros(dists.size(0)))

                    total_time = time.time() - start_time
                    running_loss += loss.detach().cpu().item()

                    experiment.log_metric("loss", loss.detach().item())
                    dataprocess.set_description_str("Epoch:{}".format(epoch))
                    dataprocess.set_postfix_str("mask_loss:{:.4f} {:.4f}s".format(loss.detach().item(), total_time))
                    # print('\r')
                # self.scheduler.step()
                print("current loss {} when epoch{}".format(running_loss / len(dataprocess), epoch))
                labels = np.array([sublabel for label in labels for sublabel in label])
                distances = np.array([subdist for dist in distances for subdist in dist])

                tpr, fpr, accuracy, val, val_std, far = evaluate(distances, labels)
                if epoch % self.cfg.save_interval == 0:

                    checkpoints = {"model": self.model.state_dict(), "epoch": epoch}
                    checkpoints_path = os.path.join(self.cfg.save_dir, self.cfg.model_name)
                    torch.save(checkpoints, "{}/checkpoint_{}.pth".format(checkpoints_path, epoch))
                    print("\n Save Model {} at epoch {}".format(self.model.__class__.__name__, epoch))

                if epoch % self.cfg.val_interval == 0:
                    # MAP = self.val()
                    # if MAP > best_mean_val:
                    #     best_mean_val = MAP

                    #     checkpoints = {
                    #         'model', self.model.state_dict(),
                    #         'epoch', epoch
                    #     }
                    #     torch.save(checkpoints, '{}/best_model'.format(self.cfg.save_path))
                    self.val()

    def val(self):
        """Minimum val
        """
        with experiment.validate():
            self.model.eval()
            positive_distance = 0
            negative_distance = 0
            n = len(self.valload)
            for anchor, positive, negative in self.valload:
                anchor, positive, negative = anchor.to(self.device), positive.to(self.device), negative.to(self.device)

                anchor_embedding, pos_embedding, neg_embedding = (
                    self.model(anchor),
                    self.model(positive),
                    self.model(negative),
                )

                l2_dist = PairwiseDistance()

                pos_dist = l2_dist.forward(anchor_embedding, pos_embedding)
                neg_dist = l2_dist.forward(anchor_embedding, neg_embedding)

                positive_distance += np.sum(pos_dist.detach().cpu().numpy())
                negative_distance += np.sum(neg_dist.detach().cpu().numpy())

            print(
                "mean postive_distance:{}, mean negative_distance:{}".format(
                    positive_distance / n, negative_distance / n
                )
            )
            experiment.log_metric("mean_distance", negative_distance / n - positive_distance / n)
            self.model.train()

    def mean_average_precision(self, code, code_target):
        """Compute average precision

        Arguments:
            code {code predict} -- code predict
            code_target {code target} -- code target

        Returns:
            mean distacne -- mean distance for two binary code
        """
        assert len(code) == len(code_target)
        total_distance = []
        for i in range(len(code)):
            distance = 0
            for j in range(len(code[i])):
                if code[j] != code_target[j]:
                    distance += 1
            distance = distance / self.cfg.code_length
            total_distance.append(distance)

        return np.mean(total_distance)


def train_palmrint_demo(cfg):
    """Palmprint recognition

    Arguments:
        cfg {[type]} -- [description]
    """
    setup_training()
    model = create_model(cfg)
    # Dataset of palmprint
    train_dataloader = create_dataset(cfg, "train", transform=ToTensor())
    val_dataloader = create_dataset(cfg, "test", transform=ToTensor())
    # optimizer
    if cfg.use_sgd:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=1e-6)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    # criterion
    criterion = DNHloss(cfg.lamda)
    # class for train
    engine = TrainEngine(cfg, model, optimizer, criterion, train_dataloader, val_dataloader)
    engine.setup()
    engine.train()


def main():
    """Main function
    """
    cfg = get_configuration()
    # Important parameters
    hyper_parameters = {
        "lr": cfg.lr,
        "batch_size": cfg.batch_size,
        "model_name": cfg.model_name,
        "dataset_name": cfg.dataset_name,
        "resume": cfg.resume,
    }
    logging.basicConfig(
        level=logging.INFO,
        filename=cfg.log_path,
        filemode="a",
        format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
        datefmt="%d-%M-%Y %H:%M:%S",
    )
    logging.info(hyper_parameters)

    print("====> start training")
    train_palmrint_demo(cfg)


if __name__ == "__main__":
    main()
