import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from crackseg.transform import get_transforms

logger = logging.getLogger(__name__)

class CrackDataset(Dataset):
    """Reads sample pairs listed in manifest"""
    def __init__(self, manifest: Path,data_dir: Path, split : str, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.to_tensor = A.Compose([A.Normalize(), ToTensorV2()])   # zawsze, niezależnie od augmentacji; Normalize = statystyki ImageNet
        samples_df  = pd.read_csv(manifest, sep=',', dtype={"mask_format": str})

        self.split_samples_df = samples_df.loc[samples_df["split"] == split].reset_index(drop=True)

        if self.split_samples_df.empty:                     #check condition where user type in bad split name
            possible_splits = samples_df["split"].unique()
            raise ValueError(f"No samples for split {split} found, avaluable split names: {possible_splits}")

    def __len__(self) -> int:
        '''returns number of samples in split'''
        return len(self.split_samples_df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        '''returns (image, mask): image float32 (3, H, W) normalized, mask float32 (1, H, W) in {0, 1}'''
        row = self.split_samples_df.iloc[idx]
        img_path = self.data_dir / row["image"]   #wczytanie zadanego obrazka
        mask_path = self.data_dir / row["mask"]
        mask_format = row["mask_format"]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        #zamiansa BGR->RGB
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask_format == "gray" or mask_format == "0255":               #binaryzacja masek
            mask = (mask>127).astype(np.float32)
        elif mask_format == "01":
            mask = (mask>0).astype(np.float32)
        else:
            raise ValueError(f"Mask format {mask_format} is not supported.")


        if self.transform is not None:                      # transform = tylko augmentacje, numpy -> numpy
            out = self.transform(image = img, mask = mask)
            img, mask = out["image"], out["mask"]

        out = self.to_tensor(image = img, mask = mask)     # normalizacja + HWC -> CHW
        return out["image"], out["mask"].unsqueeze(0)      # maska (H, W) -> (1, H, W): kanał maski, batch dokłada DataLoader


if __name__ == "__main__":
    from crackseg.logging_utils import setup_logging
    setup_logging()

    manifest = Path("manifests/manifest.csv")
    data_dir = Path("data/raw/deepcrack")
    for s in ("train", "val", "test"):
        ds = CrackDataset(manifest, data_dir, s)
        logger.info("split %s: %d samples", s, len(ds))

    #CrackDataset(manifest, data_dir, "Train")   # literówka: ma rzucić czytelny ValueError


    ds = CrackDataset(manifest, data_dir, "train", transform=get_transforms(train=True))
    img, mask = ds[0]
    logger.info("with transform: img %s %s, mask %s %s, mask values %s",
                tuple(img.shape), img.dtype, tuple(mask.shape), mask.dtype, mask.unique().tolist())

    ds = CrackDataset(manifest, data_dir, "train")       # bez transformu - format wyjścia ma być ten sam
    img, mask = ds[0]
    logger.info("no transform:   img %s %s, mask %s %s, mask values %s",
                tuple(img.shape), img.dtype, tuple(mask.shape), mask.dtype, mask.unique().tolist())
