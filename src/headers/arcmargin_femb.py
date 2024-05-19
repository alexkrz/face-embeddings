import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Source: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
class ArcMarginProduct(nn.Module):
    r"""Implement of large margin arc distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin

        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        # one_hot = torch.zeros(cosine.size(), requires_grad=True, device='cuda')
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
        output = (one_hot * phi) + (
            (1.0 - one_hot) * cosine
        )  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s
        # print(output)

        return output


# Goal: We want to make the above ArcMarginProduct class more general
class ArcMarginHeader(torch.nn.Module):
    """
    ArcMarginHeader class
    Adjusted ArcMarginProduct class from https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 1.0,
        m1: float = 1.0,
        m2: float = 0.0,
        m3: float = 0.0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        self.weight = torch.nn.Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)
        self.epsilon = 1e-6

    # NOTE: We deprecate the original forward pass and introduce new variable names
    # def forward_orig(self, input, label):
    #     # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
    #     logits = F.linear(F.normalize(input), F.normalize(self.weight), bias=None)
    #     logits = logits.clamp(-1 + self.epsilon, 1 - self.epsilon)

    #     # apply arccos to get theta
    #     # NOTE: Looks like the mistake is here, acos is not supposed to be clamped as it can be between 0 and pi
    #     # theta = torch.acos(logits).clamp(-1, 1)
    #     theta = torch.acos(logits)

    #     # add angular margin (m) to theta and transform back by cos
    #     target_logits = torch.cos(self.m1 * theta + self.m2) - self.m3

    #     # derive one-hot encoding for label
    #     one_hot = torch.zeros(logits.size(), device=input.device)
    #     one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

    #     # build the output logits
    #     output = one_hot * target_logits + (1.0 - one_hot) * logits
    #     # feature re-scaling
    #     output *= self.s

    #     return output

    def forward(self, input, label):
        # Cosine similarity between normalized input and normalized weight
        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight), bias=None)
        cos_theta = cos_theta.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)

        # Get the angle (theta) between input and weight vectors
        theta = torch.acos(cos_theta)

        # Apply the angular margin penalty
        cos_theta_m = torch.cos(self.m1 * theta + self.m2) - self.m3

        # One-hot encode labels for the corresponding class weight
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * cos_theta_m + (1 - one_hot) * cos_theta
        output *= self.s  # Scale the output

        return output


class ArcFaceHeader(ArcMarginHeader):
    """
    ArcFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8953658 (CVPR, 2019)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m2=m)


class CosFaceHeader(ArcMarginHeader):
    """
    CosFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8578650 (CVPR, 2018)
    """

    def __init__(self, in_features, out_features, s=1, m=0.35):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m3=m)


class SphereFaceHeader(ArcMarginHeader):
    """
    SphereFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8100196 (CVPR, 2017)
    """

    def __init__(self, in_features, out_features, m=4):
        super().__init__(in_features=in_features, out_features=out_features, s=1, m1=m)


class LinearHeader(torch.nn.Module):
    """LinearHeader class."""

    def __init__(self, in_features, out_features):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

    def forward(self, input, label):
        return self.linear(input)
