from typing import List, Literal
from pathlib import Path
import sys
import numpy as np
import cv2
import pandas as pd


'''
manifest structure;
id - simple sample (image-mask) identifier (key)
image - relative image path from /data dir
mask - relative mask path from /data dir
split - train / val / test; official split kept
height - sample height
width - sample width
mask_format - how mask endcode class - eg "01", "255"


code should be run from base project directory
'''
FileType = Literal["*.png", "*.jpg", "*.tiff", "*.jpeg"]


def pair_sample_by_id(img_dir: Path, mask_dir: Path, img_filetype: FileType, mask_filetype: FileType) -> List[tuple[Path, Path]]:
    '''Pairs samples by id's, returns warning for orphan samples'''


    pairs = []
    all_img_paths = sorted(img_dir.glob(img_filetype))
    all_mask_paths = sorted(mask_dir.glob(mask_filetype))
    imgs = {i.stem: i for i in all_img_paths}
    masks = {m.stem: m for m in all_mask_paths}

    matching_ids = imgs.keys() & masks.keys()
    img_orphans = imgs.keys() - masks.keys()  # obrazy bez maski
    mask_orphans = masks.keys() - imgs.keys()  # maski bez obrazu

    if img_orphans:
        print(f"WARNING: {len(img_orphans)} images without mask: {sorted(img_orphans)[:10]}", file=sys.stderr)
    if mask_orphans:
        print(f"WARNING: {len(mask_orphans)} masks without image: {sorted(mask_orphans)[:10]}", file=sys.stderr)

    for sample_id in sorted(matching_ids):
        img = imgs[sample_id]
        mask = masks[sample_id]
        pairs.append((sample_id, img, mask))
        print("paired")

    return pairs



def inspect_sample(img: Path, mask: Path) -> dict:
    '''Returns pair metadata (size, mask format etc.)'''

    img_cv2 = cv2.imread(str(img), cv2.IMREAD_UNCHANGED)
    mask_cv2 = cv2.imread(str(mask), cv2.IMREAD_UNCHANGED)
    if img_cv2 is None:
        raise FileNotFoundError(f"failed to load image {img}")
    if mask_cv2 is None:
        raise FileNotFoundError(f"failed to load mask {mask}")


    #size
    img_height, img_width = img_cv2.shape[:2]
    shape_ok = img_cv2.shape[:2] == mask_cv2.shape[:2]


    #number of channels
    img_channels = 1 if img_cv2.ndim == 2 else img_cv2.shape[2]

    #mask format
    values = np.unique(mask_cv2)
    values_set = set(values.tolist())
    if values_set <= {0,1}:
        mask_format = "01"
    elif values_set <= {0,255}:
        mask_format = "0255"
    else:
        mask_format = "gray"


    return {
        "height": img_height,
        "width": img_width,
        "mask_format": mask_format,
        "shape_ok": shape_ok,
        "img_channels": img_channels
    }


def assign_split(ids_list, val_frac = 0.15, test_frac = 0 , seed = 42):  #test docelowo 0 bo mam podzielone na wejściu (tne tylko traina)
    ids_sorted = sorted(ids_list)
    permutations = np.random.default_rng(seed).permutation(len(ids_sorted))
    test = round(test_frac * len(ids_sorted))
    val = round(val_frac * len(ids_sorted))

    shuffled = [ids_sorted[i] for i in permutations]
    test_ids = shuffled[:test]
    val_ids = shuffled[test:test + val]
    train_ids = shuffled[test + val:]

    splits = {i: "test" for i in test_ids}
    splits |= {i: "val" for i in val_ids}   #łączenie słowników
    splits |= {i: "train" for i in train_ids}

    return [splits[i] for i in ids_list]   #returning list for furhter pandas processing (not treating id columnd as index)



def deep_crack(data_dir: Path):   #collector
    '''returns list with complex inforamtion about DeepCrack dataset initial split'''
    out = []
    for split in ("train", "test"):
        pairs = pair_sample_by_id(data_dir / f"{split}_img", data_dir / f"{split}_lab", "*.jpg", "*.png")
        out += [(sid, img, mask, split) for sid, img, mask in pairs]
    return out


def build_manifests(dataset, data_dir: Path, val_frac = 0.15, test_frac = 0, seed = 45):
    '''builds csv with comprehensive inforamtion about dataset'''
    samples = dataset(data_dir)
    rows = [{"dataset": dataset.__name__,
             "id": sid,
             "image": img.relative_to(data_dir).as_posix(),
             "mask": mask.relative_to(data_dir).as_posix(),
             "split": split,
             **inspect_sample(img, mask),} for sid, img, mask, split in samples]    # przez list comprehension rozpakowujemmy każdą sample z datasetu wczytanego przez funkcje collect

    df = pd.DataFrame(rows)
    no_split = df["split"].isna()
    if no_split.all():
        # zbiór bez podziału -> losujemy train/val/test
        df["split"] = assign_split(df["id"].tolist(), val_frac=val_frac, test_frac=test_frac, seed=seed)
    else:
        # oficjalny podział -> val tylko z train, test bez zmian
        is_train = df["split"] == "train"
        df.loc[is_train, "split"] = assign_split(df.loc[is_train, "id"].tolist(), val_frac=val_frac, seed=seed)

    return df.sort_values(["split", "id"]).reset_index(drop=True)





df = build_manifests(deep_crack,Path("data/raw/deepcrack"))
df.to_csv(Path("manifests/manifest.csv"), index=False)