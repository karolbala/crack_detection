from pathlib import Path
import lightning as L
from torch.utils.data import DataLoader
from crackseg.data import CrackDataset
from crackseg.transform import get_transforms


'''DataModule '''
class CrackDataModule(L.LightningDataModule):
    def __init__(self, manifest, data_dir, batch_size=8, num_workers=0, crop=384):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.save_hyperparameters()       # zapamiętuje argumenty, później trafią do logów

    def setup(self, stage=None):
        self.train_ds = CrackDataset(self.hparams.manifest, self.hparams.data_dir, "train", transform=get_transforms(train=True, crop=self.hparams.crop))
        self.val_ds = CrackDataset(self.hparams.manifest, self.hparams.data_dir, "val", transform=get_transforms(train=False, crop=self.hparams.crop))
        self.test_ds = CrackDataset(self.hparams.manifest, self.hparams.data_dir, "test", transform=get_transforms(train=False, crop=self.hparams.crop))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, num_workers=self.hparams.num_workers)        #batch size 1 bo nierówne wymiary obrazka

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, num_workers=self.hparams.num_workers)




