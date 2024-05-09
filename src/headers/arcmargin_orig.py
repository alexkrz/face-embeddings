import math

import torch
import torch.nn.functional as F


# CombinedMarginLoss is taken from https://github.com/deepinsight/insightface/blob/master/recognition/arcface_torch/losses.py
class CombinedMarginLoss(torch.nn.Module):
    def __init__(self, s, m1, m2, m3, interclass_filtering_threshold=0):
        super().__init__()
        self.s = s
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3
        self.interclass_filtering_threshold = interclass_filtering_threshold

        # For ArcFace
        self.cos_m = math.cos(self.m2)
        self.sin_m = math.sin(self.m2)
        self.theta = math.cos(math.pi - self.m2)
        self.sinmm = math.sin(math.pi - self.m2) * self.m2
        self.easy_margin = False

    def forward(self, logits, labels):
        index_positive = torch.where(labels != -1)[0]

        if self.interclass_filtering_threshold > 0:
            with torch.no_grad():
                dirty = logits > self.interclass_filtering_threshold
                dirty = dirty.float()
                mask = torch.ones([index_positive.size(0), logits.size(1)], device=logits.device)
                mask.scatter_(1, labels[index_positive], 0)
                dirty[index_positive] *= mask
                tensor_mul = 1 - dirty
            logits = tensor_mul * logits

        target_logit = logits[index_positive, labels[index_positive].view(-1)]

        if self.m1 == 1.0 and self.m3 == 0.0:
            with torch.no_grad():
                target_logit.arccos_()
                logits.arccos_()
                final_target_logit = target_logit + self.m2
                logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
                logits.cos_()
            logits = logits * self.s

        elif self.m3 > 0:
            final_target_logit = target_logit - self.m3
            logits[index_positive, labels[index_positive].view(-1)] = final_target_logit
            logits = logits * self.s
        else:
            raise

        return logits


class CombinedMarginHeader(CombinedMarginLoss):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        s: float = 1.0,
        m1: float = 1.0,
        m2: float = 0.0,
        m3: float = 0.0,
    ):
        super().__init__(s=s, m1=m1, m2=m2, m3=m3)

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.epsilon = 1e-6

    def forward(self, embeddings, labels):
        self.linear.weight = torch.nn.Parameter(F.normalize(self.linear.weight))
        norm_embeddings = F.normalize(embeddings).clamp(-1 + self.epsilon, 1 - self.epsilon)

        logits = self.linear(norm_embeddings)
        loss = super().forward(logits, labels)
        return loss


class ArcFaceHeader(CombinedMarginHeader):
    """
    ArcFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8953658 (CVPR, 2019)
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m2=m)


class CosFaceHeader(CombinedMarginHeader):
    """
    CosFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8578650 (CVPR, 2018)
    """

    def __init__(self, in_features, out_features, s=1, m=0.35):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m3=m)


class SphereFaceHeader(CombinedMarginHeader):
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
