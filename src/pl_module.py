import pytorch_lightning as pl
import torch

from src.backbones.build import build_backbone
from src.headers.arcmargin_femb import (
    ArcFaceHeader,
    CosFaceHeader,
    LinearHeader,
    SphereFaceHeader,
)
from src.headers.elasticface import ArcFace, ElasticArcFace
from src.headers.magface import MagFaceHeader

header_dict = {
    "arcface": ArcFaceHeader,
    "arcface-fadi": ArcFace,
    "cosface": CosFaceHeader,
    "elasticface": ElasticArcFace,
    "linear": LinearHeader,
    "magface": MagFaceHeader,
    "sphereface": SphereFaceHeader,
}


class FembModule(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "iresnet50",
        embed_dim: int = 512,
        pretrained_bb: bool = False,
        header="arcface",
        n_classes: int = 10572,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = build_backbone(
            backbone=backbone,
            embed_dim=embed_dim,
            pretrained=pretrained_bb,
        )
        assert header in header_dict.keys()
        self.header = header_dict[header](embed_dim, n_classes)

        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(imgs)
        return feats

    def training_step(self, batch, batch_idx):
        imgs, targets = batch
        feats = self(imgs)
        logits = self.header(feats, targets)
        # logits vector describes the probability for each image to belong to one of n_classes
        loss = self.criterion(logits, targets)
        optimizer_lr = self.optimizers().optimizer.param_groups[0]["lr"]
        log_dict = {
            # "step": float(self.current_epoch),  # Overwrite step to plot epochs on x-axis
            "loss": loss,
            "optimizer_lr": optimizer_lr,
        }
        self.log_dict(log_dict, on_step=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            # Need to optimize over all parameters in the module!
            params=self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
