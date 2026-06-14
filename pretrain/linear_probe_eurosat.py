# Copyright (c) András Kalapos.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Linear probing evaluation on EuroSAT.
#
# Two usage modes:
#   1. From a CNN-JEPA checkpoint:
#      python -m pretrain.linear_probe_eurosat --config-name=linear_probe_eurosat \
#          checkpoint_path=artifacts/pretrain_lightly/ijepacnn/.../<checkpoint>.ckpt
#
#   2. From ImageNet supervised pre-training (baseline, no checkpoint needed):
#      python -m pretrain.linear_probe_eurosat --config-name=linear_probe_eurosat \
#          checkpoint_path=null backbone.pretrained_weights=imagenet

import hydra
from omegaconf import DictConfig

import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision import transforms
from torch.utils.data import DataLoader

from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.utils.benchmarking.topk import mean_topk_accuracy

from data.eurosat import EuroSATDataset

import models.convnext  # registers sparse ConvNeXt variants in timm


# ---------------------------------------------------------------------------
# Lightning module
# ---------------------------------------------------------------------------

class LinearProbe(pl.LightningModule):
    """Frozen backbone + single linear head trained on EuroSAT."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # ---- Build backbone ----
        self.backbone = timm.create_model(
            cfg.backbone.name,
            pretrained=cfg.backbone.pretrained_weights == "imagenet",
            num_classes=0,
            **dict(cfg.backbone.get("kwargs", {})),
        )

        # ---- Load CNN-JEPA checkpoint if provided ----
        ckpt_path = cfg.get("checkpoint_path", None)
        if ckpt_path is not None:
            print(f"Loading backbone weights from CNN-JEPA checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            # CNN-JEPA checkpoints store the momentum backbone as backbone_momentum.
            # We use the online encoder (backbone.*) for probing.
            backbone_sd = {
                k.replace("backbone.", "", 1): v
                for k, v in state_dict.items()
                if k.startswith("backbone.") and not k.startswith("backbone_momentum.")
            }
            missing, unexpected = self.backbone.load_state_dict(backbone_sd, strict=False)
            if missing:
                print(f"  Missing keys  : {missing[:5]}{'...' if len(missing) > 5 else ''}")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

        # ---- Freeze backbone ----
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # ---- Linear head ----
        self.classifier = nn.Linear(self.backbone.num_features, cfg.num_classes)

        self.criterion = nn.CrossEntropyLoss()
        self.topk = (1, 5)

    # ---- Forward ----
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        features = features.flatten(start_dim=1)
        return self.classifier(features)

    # ---- Shared step ----
    def _step(self, batch, metric_label):
        x, label = batch
        logits = self(x)
        loss = self.criterion(logits, label)

        _, predicted_classes = logits.topk(max(self.topk))
        topk = mean_topk_accuracy(predicted_classes=predicted_classes, targets=label, k=self.topk)

        self.log(f"{metric_label}/loss", loss, on_epoch=True, prog_bar=True)
        self.log_dict(
            {f"{metric_label}/acc_top{k}": acc for k, acc in topk.items()},
            on_epoch=True, prog_bar=True,
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # ---- Optimizer ----
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.classifier.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
        )

    # ---- Data ----
    def _make_transform(self, augment: bool):
        if augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.cfg.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_NORMALIZE["mean"],
                    std=IMAGENET_NORMALIZE["std"],
                ),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(self.cfg.input_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE["std"],
            ),
        ])

    def train_dataloader(self):
        ds = EuroSATDataset(
            root=self.cfg.data.root,
            split="train",
            transform=self._make_transform(augment=True),
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.cfg.data.num_workers,
        )

    def val_dataloader(self):
        ds = EuroSATDataset(
            root=self.cfg.data.root,
            split="validation",
            transform=self._make_transform(augment=False),
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )

    def test_dataloader(self):
        ds = EuroSATDataset(
            root=self.cfg.data.root,
            split="test",
            transform=self._make_transform(augment=False),
        )
        return DataLoader(
            ds,
            batch_size=self.cfg.optimizer.batch_size,
            shuffle=False,
            num_workers=self.cfg.data.num_workers,
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base="1.2", config_path="configs/", config_name="linear_probe_eurosat.yaml")
def main(cfg: DictConfig):
    pl.seed_everything(cfg.seed)

    model = LinearProbe(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    trainer.fit(model)
    trainer.test(model)


if __name__ == "__main__":
    main()
