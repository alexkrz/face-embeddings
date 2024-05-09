import torch


class ArcMarginHeader(torch.nn.Module):
    """
    ArcMarginHeader class
    Inspired by ArcMarginProduct implementation: https://github.com/ronghuaiyang/arcface-pytorch/blob/master/models/metrics.py
    Reference: https://ieeexplore.ieee.org/document/8953658
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

        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.normalize = torch.nn.functional.normalize

        self.epsilon = 1e-6

    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)
        self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        logits = self.linear(self.normalize(input)).clamp(-1 + self.epsilon, 1 - self.epsilon)

        # apply arccos to get theta
        theta = torch.acos(logits).clamp(-1, 1)

        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(self.m1 * theta + self.m2) - self.m3

        # derive one-hot encoding for label
        one_hot = torch.zeros(logits.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1.0)

        # build the output logits
        output = one_hot * target_logits + (1.0 - one_hot) * logits
        # feature re-scaling
        output *= self.s

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
