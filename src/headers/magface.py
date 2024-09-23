import torch
import torch.nn.functional as F

from .arcmargin_femb import ArcFaceHeader


class MagFaceHeader(ArcFaceHeader):
    """
    MagFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/9578764
    """

    def __init__(
        self, in_features, out_features, s=64.0, l_a=10, u_a=110, l_m=0.45, u_m=0.8, lambda_g=20
    ):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m=None)

        self.l_a = l_a
        self.u_a = u_a
        self.l_m = l_m
        self.u_m = u_m

        self.lambda_g = lambda_g

    def compute_m(self, a):
        return (self.u_m - self.l_m) / (self.u_a - self.l_a) * (a - self.l_a) + self.l_m

    def compute_g(self, a):
        return torch.mean((1 / self.u_a**2) * a + 1 / a)

    def forward(self, input, label):
        # multiply normed features (input) and normed weights to obtain cosine of theta (logits)

        ########
        #  Replace this part as in ArcFaceHeader
        # self.linear.weight = torch.nn.Parameter(self.normalize(self.linear.weight))
        # logits = self.linear(self.normalize(input)).clamp(-1 + self.epsilon, 1 - self.epsilon)
        ########

        cos_theta = F.linear(F.normalize(input), F.normalize(self.weight), bias=None)
        cos_theta = cos_theta.clamp(-1.0 + self.epsilon, 1.0 - self.epsilon)

        # difference compared to arcface
        a = torch.norm(input, dim=1, keepdim=True).clamp(self.l_a, self.u_a)
        m = self.compute_m(a)
        g = self.compute_g(a)

        # apply arccos to get theta
        theta = torch.acos(cos_theta)

        # add angular margin (m) to theta and transform back by cos
        target_logits = torch.cos(theta + m)

        # One-hot encode labels for the corresponding class weight
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = one_hot * target_logits + (1 - one_hot) * cos_theta

        # Scale the output
        output *= self.s
        return output + self.lambda_g * g
