from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CrackDataset(Dataset):
    """Reads sample pairs listed in manifest"""
    def __init__(self, manifest: Path,data_dir: Path, split : str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        samples_df  = pd.read_csv(manifest, sep=',', dtype={"mask_format": str})

        self.split_samples_df = samples_df.loc[samples_df["split"] == split].reset_index(drop=True)

        if self.split_samples_df.empty:                     #check condition where user type in bad split name
            possible_splits = samples_df["split"].unique()
            raise ValueError(f"No samples for split {split} found, avaluable split names: {possible_splits}")

    def __len__(self) -> int:
        '''returns number of samples in split'''
        return len(self.split_samples_df)

    def __getitem__(self, idx: int) -> torch.Tensor:
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


        if self.transform is not None:
            out = self.transform(img = img, mask = mask)

            return out["image"], out["mask"].unsqueeze(0)
        else:
            return img, mask


if __name__ == "__main__":
    manifest = Path("manifests/manifest.csv")
    data_dir = Path("data/raw/deepcrack")

    for s in ("train", "val", "test"):
        ds = CrackDataset(manifest, data_dir, s)
        print(s, len(ds))

    CrackDataset(manifest, data_dir, "Train")   # literówka: ma rzucić czytelny ValueError