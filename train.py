import os
import random
from pathlib import Path
from typing import Callable, Iterable, Optional, Tuple

import jsonargparse
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

from src.datamodule import MXFaceDatamodule
from src.pl_module import FembModule
from src.utils import find_max_version


def process_parser_args(
    parser: jsonargparse.ArgumentParser,
) -> Tuple[jsonargparse.Namespace, str, int]:
    cfg = parser.parse_args()
    # Make output directories
    results_dir = Path(cfg.results_dir)
    if not results_dir.exists():
        results_dir.mkdir()
    results_dir = Path(cfg.results_dir) / cfg.method_name
    if not results_dir.exists():
        results_dir.mkdir()
    if cfg.version is None:
        version = find_max_version(results_dir) + 1
    else:
        version = cfg.version
    cfg["version"] = version
    version_dir = results_dir / f"version_{version}"

    # Setup for distributed training
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    print(f"local_rank is: {local_rank}")
    if local_rank == 0:
        version_dir.mkdir(parents=False, exist_ok=False)
        # Save config to version_dir
        parser.save(cfg, version_dir / "config.yaml")
    # IDEA: Instead of results_dir and version one could also output results_dir with subfolder datetimefmt and version=None
    return cfg, str(results_dir), version


def main(
    cfg: jsonargparse.Namespace,
    datamodule: LightningDataModule,
    pl_module: LightningModule,
    results_dir: str,
    version: Optional[int] = None,
):
    # 1. Set fixed seed
    pl.seed_everything(cfg.seed)

    # 2. Assign datamodule and pl_module
    datamodule = datamodule
    # datamodule.setup("fit")
    # print(len(datamodule.train_dataloader().dataset))
    # batch = next(iter(datamodule.train_dataloader()))
    # x, y = batch
    # print(x.shape)
    # print(y)
    pl_module = pl_module

    # 3. Configure loggers
    tensorboard_logger = TensorBoardLogger(results_dir, name=None, version=version, sub_dir="logs")
    csv_logger = CSVLogger(results_dir, name=None, version=version)
    logger = [tensorboard_logger, csv_logger]

    # 4. Configure callbacks
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        save_on_train_epoch_end=True,
        save_weights_only=True,
    )
    progress_bar = TQDMProgressBar(refresh_rate=cfg.pbar_refresh_rate)
    # lr_monitor = LearningRateMonitor(logging_interval="step")  # Log learning rate inside pl_module to avoid problems with CSVLogger
    callbacks = [model_checkpoint, progress_bar]

    # 5. Set up Trainer
    cfg_logger = cfg.trainer.init_args.pop("logger")
    cfg_callbacks = cfg.trainer.init_args.pop("callbacks")
    trainer = Trainer(**cfg.trainer.init_args, logger=logger, callbacks=callbacks)

    # 6. Perform training
    print("Starting training..")
    # print(f"Writing logs to {str(trainer.log_dir)}")  # Print log_dir not working in ddp mode
    trainer.fit(model=pl_module, datamodule=datamodule)


if __name__ == "__main__":
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_argument("--config", action=ActionConfigFile)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results_dir", type=str)
    parser.add_argument("--method_name", type=str)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--datamodule", type=LightningDataModule)
    parser.add_argument("--pl_module", type=LightningModule)
    parser.add_argument("--trainer", type=Trainer)
    # parser.add_class_arguments(MXFaceDatamodule, "datamodule")
    # parser.add_class_arguments(FembModule, "pl_module")
    # parser.add_class_arguments(Trainer, "trainer")
    parser.add_argument("--pbar_refresh_rate", type=int, default=1)

    cfg, results_dir, version = process_parser_args(parser)
    # Instantiate datamodule and pl_module from config
    datamodule = parser.instantiate_classes({"datamodule": cfg.datamodule}).datamodule
    pl_module = parser.instantiate_classes({"pl_module": cfg.pl_module}).pl_module
    main(cfg, datamodule, pl_module, results_dir, version)
