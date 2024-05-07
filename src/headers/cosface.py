from .arcmargin import ArcMarginHeader


class CosFaceHeader(ArcMarginHeader):
    """
    CosFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8578650
    """

    def __init__(self, in_features, out_features, s=1, m=0.35):
        super().__init__(in_features=in_features, out_features=out_features, s=s, m3=m)