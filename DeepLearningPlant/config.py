"""
Configuration centralisée pour le projet Herbiers & CrossViT.
"""

import os
import torch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(ROOT_DIR, "dataset.csv")
ORIG_DIR = os.path.join(ROOT_DIR, "mission_herbonaute_2000")
SEG_DIR = os.path.join(ROOT_DIR, "mission_herbonaute_2000_seg_black")
CROSSVIT_DIR = os.path.join(os.path.dirname(ROOT_DIR), "CrossViT")
SAVE_DIR = os.path.join(ROOT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 32  
NUM_WORKERS = 0  
EPOCHS = 20 
LR = 5e-5
WEIGHT_DECAY = 0.05
TRAIN_RATIO = 0.8


IMG_SIZE_SMALL = 240   # branche Small (patch 12)
IMG_SIZE_LARGE = 224   # branche Large (patch 16)
PATCH_SIZE_SMALL = 12
PATCH_SIZE_LARGE = 16
NUM_CLASSES = 2        # épines : présence / absence

# Pour la Partie 2 (même résolution) on utilise 224x224, patch 16
IMG_SIZE_SAME = 224
PATCH_SIZE_SAME = 16

#partie3
WEIGHT_EPSILON = 0.1
WEIGHT_GAMMA = 1.0     # gamma pour la puissance (1.0 = linéaire)
WEIGHT_FUNCTION = "linear"  # "linear", "power", "normalized"

# partie4
IOE_QUANTILE = 0.8     
IOE_LOSS_WEIGHT = 1.0  

#pour les image net il faut l'expliquer dans le rapport!!
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
