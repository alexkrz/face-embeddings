from .arcmargin import ArcMarginHeader


class ArcFaceHeader(ArcMarginHeader):
    """
    ArcFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8953658
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m2=m)
