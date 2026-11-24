import lightning as L
import segmentation_models_pytorch as smp
import torch
from torch import nn







class CrackSegModel(L.LightningModule):
    def __init__(self, arch: str = "Unet", encoder: str = "resnet34", lr: float = 3e-4):
        super().__init__()
        self.save_hyperparameters()                       # arch/encoder/lr trafią do checkpointu i logów

        self.net = smp.create_model(arch, encoder_name=encoder, encoder_weights="imagenet",
                                    in_channels=3, classes=1)
        self.dice = smp.losses.DiceLoss(mode="binary", from_logits=True)   #strata Dice based on IoU
        self.bce = nn.BCEWithLogitsLoss() # strata Binarna entropia krzyżowa z logitów

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        img, mask = batch
        logits = self.net(img)
        loss = self.dice(logits, mask)+ self.bce(logits, mask)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss


    def _shared_eval(self, batch, stage: str):
        """Wspólna logika walidacji i testu - różni się tylko przedrostkiem metryk (val/ albo test/)."""
        img, mask = batch
        logits = self(img)
        loss = self.dice(logits, mask) + self.bce(logits, mask)

        tp, fp, fn, tn = smp.metrics.get_stats(logits.sigmoid(), mask.long(), mode="binary", threshold=0.5)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}/dice", smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro"), on_epoch=True, prog_bar=True)
        self.log(f"{stage}/iou", smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro"), on_epoch=True)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, "test")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

