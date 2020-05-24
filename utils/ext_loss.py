'''
@Author: your name
@Date: 2020-05-12 12:57:28
@LastEditors: wei
@LastEditTime: 2020-05-18 06:04:02
@Description: file content
'''
import torch
import torch.nn as nn


class DNHloss(nn.Module):
    """DNH loss function

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, lamda):
        """Initializer class

        Arguments:
            lamda {[type]} -- [description]
        """
        super(DNHloss, self).__init__()
        self.lamda = lamda

    def forward(self, H, S):
        """Forward H and S

        Arguments:
            H {[type]} -- [description]
            S {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        # Inner product
        theta = H @ H.t() / 2

        # log(1+e^z) may be overflow when z is large.
        # We convert log(1+e^z) to log(1 + e^(-z)) + z.
        metric_loss = (torch.log(1 + torch.exp(-(self.lamda * theta).abs())) + theta.clamp(min=0) - self.lamda * S * theta).mean()
        quantization_loss = self.logcosh(H.abs() - 1).mean()

        loss = metric_loss + self.lamda * quantization_loss

        return loss

    def logcosh(self, x):
        # TODO: loss的含义待补充
        """log cos

        Arguments:
            x {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        return torch.log(torch.cosh(x))


class PairwiseDistance(nn.Module):
    """class for calculating distance

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, smooth=1e-4):
        """Initializer

        Arguments:
            smooth {int} -- [description]
        """
        super(PairwiseDistance, self).__init__()
        self.smooth = smooth

    def forward(self, x1, x2):
        """x1, x2 represent input data

        Arguments:
            x1 {[type]} -- [description]
            x2 {[type]} -- [description]

        Returns:
            [type] -- [description]
        """
        assert x1.size() == x2.size()
        diff = torch.abs(x1 - x2)
        out = torch.pow(diff, 2).sum(dim=1)

        return torch.pow(out + self.smooth, 0.5)


class TripletMarginLoss(nn.Module):
    """Triplet loss

    Arguments:
        nn {[type]} -- [description]
    """
    def __init__(self, margin):
        super(TripletMarginLoss, self).__init__()
        self.margin = margin
        self.pdist = PairwiseDistance()

    def forward(self, anchor, positive, negative):
        d_p = self.pdist(anchor, positive)
        d_n = self.pdist(anchor, negative)

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0)
        loss = torch.mean(dist_hinge)
        return loss
