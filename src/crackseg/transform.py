import albumentations as A
from albumentations.pytorch import ToTensorV2




def get_transforms(train: bool, crop: int = 384, scale_limit: float = 0.5) -> A.Compose:                  #brać parametry
    """Augmentations for training, deterministic preprocessing for val/test."""
    if train:
        aug = [
            A.RandomScale(scale_limit=scale_limit, p=1.0),   # skala 0.5-1.5x: odporność na różne GSD!!!!!
            A.PadIfNeeded(min_height=crop, min_width=crop),  # po pomniejszeniu obraz może być < crop
            A.RandomCrop(height=crop, width=crop),
            A.HorizontalFlip(),                         #flip po Y
            A.VerticalFlip(),                           #flip po X
            A.RandomRotate90(),                            #losowa rotacja
            A.RandomBrightnessContrast(),                   #losowe zmiany kontrastu/jasności
        ]
    else:
        aug = []
    return A.Compose([*aug, A.Normalize(), ToTensorV2()])

