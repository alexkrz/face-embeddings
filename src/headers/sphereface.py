from .arcmargin import ArcMarginHeader


class SphereFaceHeader(ArcMarginHeader):
    """
    SphereFaceHeader class
    Reference: https://ieeexplore.ieee.org/document/8100196
    """

    def __init__(self, in_features, out_features, m=4):
        super().__init__(in_features=in_features, out_features=out_features, s=1, m1=m)