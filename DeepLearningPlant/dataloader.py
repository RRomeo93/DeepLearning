"""
Dataset PyTorch pour les herbiers.
Charge les paires (image originale, image segmentée) + masque binaire + label épines.
"""

import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image, ImageFile
from torchvision import transforms
import config

ImageFile.LOAD_TRUNCATED_IMAGES = True


class HerbariumDataset(Dataset):
    """
    Dataset qui renvoie pour chaque spécimen :
      - original  : image non segmentée  (RGB, transformée)
      - segmented : image segmentée       (RGB, transformée)
      - mask      : masque binaire         (1, H, W) — NON normalisé
      - label     : 0 ou 1 (absence / présence d'épines)
    """

    def __init__(self, csv_path, orig_dir, seg_dir, transform=None,
                 img_size=224):
      
        self.df = pd.read_csv(csv_path, header=1)
        self.df = self.df.dropna(subset=["epines"])
       
        self.df = self.df[self.df["epines"].isin([0.0, 1.0])].reset_index(drop=True)

        self.orig_dir = orig_dir
        self.seg_dir = seg_dir
        self.transform = transform
        self.img_size = img_size

        
        self.mask_transform = transforms.Compose([
            transforms.Resize((img_size, img_size),
                               interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

       
        valid = []
        for idx in range(len(self.df)):
            code = str(self.df.iloc[idx]["code"]).replace(".0", "")
            fname = f"{code}.jpg"
            if (os.path.isfile(os.path.join(self.orig_dir, fname)) and
                    os.path.isfile(os.path.join(self.seg_dir, fname))):
                valid.append(idx)
        self.df = self.df.iloc[valid].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        code = str(row["code"]).replace(".0", "")
        fname = f"{code}.jpg"
        label = int(row["epines"])

        img_orig = Image.open(
            os.path.join(self.orig_dir, fname)).convert("RGB")
        img_seg = Image.open(
            os.path.join(self.seg_dir, fname)).convert("RGB")

     
        mask = img_seg.convert("L").point(lambda p: 255 if p > 10 else 0)
        mask = self.mask_transform(mask)  

        if self.transform:
            img_orig = self.transform(img_orig)
            img_seg = self.transform(img_seg)

        return {
            "original": img_orig,
            "segmented": img_seg,
            "mask": mask,
            "label": torch.tensor(label, dtype=torch.long),
            "code": code,
        }


def get_transforms(img_size=224, train=True):
    """Renvoie les transformations pour train ou val."""
    if train:
        return transforms.Compose([
            transforms.Resize((int(img_size * 1.1), int(img_size * 1.1))),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                    saturation=0.3, hue=0.1),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN,
                                  std=config.IMAGENET_STD),
            transforms.RandomErasing(p=0.2),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGENET_MEAN,
                                  std=config.IMAGENET_STD),
        ])


def get_dataloaders(img_size=224, batch_size=None, num_workers=None):
    """
    Crée les DataLoaders train / val avec split 80/20 reproductible.
    Deux datasets séparés : train avec augmentations, val sans.
    """
    bs = batch_size or config.BATCH_SIZE
    nw = num_workers or config.NUM_WORKERS

   
    full_dataset = HerbariumDataset(
        csv_path=config.CSV_PATH,
        orig_dir=config.ORIG_DIR,
        seg_dir=config.SEG_DIR,
        transform=get_transforms(img_size, train=False),
        img_size=img_size,
    )

    n_total = len(full_dataset)
    n_train = int(n_total * config.TRAIN_RATIO)
    n_val = n_total - n_train

    generator = torch.Generator().manual_seed(config.SEED)
    train_subset, val_subset = random_split(
        full_dataset, [n_train, n_val], generator=generator
    )

 
    train_dataset = HerbariumDataset(
        csv_path=config.CSV_PATH,
        orig_dir=config.ORIG_DIR,
        seg_dir=config.SEG_DIR,
        transform=get_transforms(img_size, train=True),
        img_size=img_size,
    )

    train_indices = train_subset.indices
    train_dataset.df = train_dataset.df.iloc[train_indices].reset_index(drop=True)

  
    val_dataset = HerbariumDataset(
        csv_path=config.CSV_PATH,
        orig_dir=config.ORIG_DIR,
        seg_dir=config.SEG_DIR,
        transform=get_transforms(img_size, train=False),
        img_size=img_size,
    )
    val_indices = val_subset.indices
    val_dataset.df = val_dataset.df.iloc[val_indices].reset_index(drop=True)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True,
                               num_workers=nw, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False,
                             num_workers=nw, pin_memory=True)

    print(f"Dataset : {n_total} images | Train : {n_train} | Val : {n_val}")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print(f"Original : {batch['original'].shape}")
    print(f"Segmented: {batch['segmented'].shape}")
    print(f"Mask     : {batch['mask'].shape}")
    print(f"Labels   : {batch['label']}")
