"""
Partie 4 — Attention Rollout, Heatmaps & IoU.

a) Attention rollout : pour chaque couche, moyenne des têtes, ajout identité,
   renormalisation, produit cumulatif → influence globale patches → [CLS].
b) Heatmaps : superposer la carte d'attention aux images.
c) IoU : binariser la carte (quantile 0.8) et calculer IoU avec le masque plante.
d) IoU dans la loss : perte auxiliaire pour forcer l'attention sur la plante.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import config



class AttentionHook:
    """
    Enregistre les matrices d'attention de chaque Block du ViT
    via des forward hooks.

    Args:
        differentiable: si True, garde les tenseurs sur le device d'origine
                        sans les détacher (permet le backprop à travers l'IoU loss).
    """

    def __init__(self, differentiable=False):
        self.attention_maps = []
        self.hooks = []
        self.differentiable = differentiable

    def register(self, model):
        """Enregistre les hooks sur tous les blocs d'attention."""
        self.attention_maps = []

     
        if hasattr(model, "blocks_1"):
            for block in model.blocks_1:
                h = block.attn.register_forward_hook(self._make_hook())
                self.hooks.append(h)
        elif hasattr(model, "crossvit"):
     
            for ms_block in model.crossvit.blocks:
                for branch_blocks in ms_block.blocks:
                    for block in branch_blocks:
                        h = block.attn.register_forward_hook(self._make_hook())
                        self.hooks.append(h)

    def _make_hook(self):
        def hook_fn(module, input, output):

            x = input[0]
            B, N, C = x.shape
            qkv = module.qkv(x).reshape(B, N, 3, module.num_heads,
                                          C // module.num_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            attn = (q @ k.transpose(-2, -1)) * module.scale
            attn = attn.softmax(dim=-1)  
            if self.differentiable:
                self.attention_maps.append(attn)
            else:
                self.attention_maps.append(attn.detach().cpu())

        return hook_fn

    def clear(self):
        self.attention_maps = []

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []



def attention_rollout(attention_maps, discard_ratio=0.0):
    """
    Calcule l'attention rollout à partir d'une liste de matrices d'attention.

    Pour CrossViT : sépare les branches par taille, calcule le rollout séparément,
    et fusionne les résultats.

    Args:
        attention_maps : liste de tenseurs (B, heads, N, N)
        discard_ratio  : ratio d'attention à ignorer (pour nettoyer)

    Returns:
        rollout : (B, N) — attention du [CLS] vers chaque patch
    """
    if not attention_maps:
        raise ValueError("attention_maps est vide")

    
    sizes = [attn.shape[-1] for attn in attention_maps]
    unique_sizes = sorted(set(sizes))

    if len(unique_sizes) > 1:
  
        print(f" Détection CrossViT avec {len(unique_sizes)} branches : {unique_sizes}")

        branch_results = []
        for branch_size in unique_sizes:
           
            branch_attn = [attn for attn in attention_maps if attn.shape[-1] == branch_size]
            if branch_attn:
            
                branch_rollout = _compute_rollout_single_branch(branch_attn, discard_ratio)
                branch_results.append(branch_rollout)


        cls_attention = min(branch_results, key=lambda x: x.shape[-1])
        print(f" Utilisation de la branche Large avec {cls_attention.shape[-1]} patches")

    else:
    
        cls_attention = _compute_rollout_single_branch(attention_maps, discard_ratio)

    return cls_attention


def _compute_rollout_single_branch(attention_maps, discard_ratio=0.0):
    """
    Calcule l'attention rollout pour une seule branche (taille uniforme).
    """
    result = None

    for attn in attention_maps:
     
        attn_heads_mean = attn.mean(dim=1)

        if discard_ratio > 0:
        
            flat = attn_heads_mean.view(attn_heads_mean.size(0), -1)
            threshold = torch.quantile(flat, discard_ratio, dim=1,
                                        keepdim=True)
            threshold = threshold.unsqueeze(-1)
            attn_heads_mean = attn_heads_mean * (
                attn_heads_mean > threshold).float()

        
        I = torch.eye(attn_heads_mean.size(-1), device=attn_heads_mean.device).unsqueeze(0)
        I = I.expand_as(attn_heads_mean)
        attn_with_id = attn_heads_mean + I

   
        attn_with_id = attn_with_id / attn_with_id.sum(dim=-1, keepdim=True)

      
        if result is None:
            result = attn_with_id
        else:
            result = attn_with_id @ result


    cls_attention = result[:, 0, 1:]  

    return cls_attention


def rollout_to_heatmap(cls_attention, img_size, patch_size):
    """
    Convertit l'attention rollout en heatmap de la taille de l'image.

    Args:
        cls_attention : (B, N) ou (N,) attention par patch
        img_size      : taille de l'image (int)
        patch_size    : taille du patch (int)

    Returns:
        heatmap : (B, img_size, img_size) ou (img_size, img_size)
    """
    grid_size = img_size // patch_size
    squeeze = False
    if cls_attention.dim() == 1:
        cls_attention = cls_attention.unsqueeze(0)
        squeeze = True

    
    heatmap = cls_attention.view(-1, 1, grid_size, grid_size)

    
    heatmap = F.interpolate(heatmap, size=(img_size, img_size),
                             mode="bilinear", align_corners=False)
    heatmap = heatmap.squeeze(1) 


    flat = heatmap.view(heatmap.shape[0], -1)
    h_min = flat.min(dim=1)[0].view(-1, 1, 1)
    h_max = flat.max(dim=1)[0].view(-1, 1, 1)
    heatmap = (heatmap - h_min) / (h_max - h_min + 1e-8)

    if squeeze:
        heatmap = heatmap.squeeze(0)

    return heatmap



# IoU

def compute_iou(attention_heatmap, plant_mask, quantile=None):
    """
    Calcule l'IoU entre la carte d'attention binarisée et le masque plante.

    Args:
        attention_heatmap : (H, W) ou (B, H, W) — valeurs entre 0 et 1
        plant_mask        : (H, W) ou (B, H, W) — binaire (0 ou 1)
        quantile          : seuil de binarisation (défaut: config.IOE_QUANTILE)

    Returns:
        iou : scalaire ou (B,)
    """
    q = quantile if quantile is not None else config.IOE_QUANTILE

    squeeze = False
    if attention_heatmap.dim() == 2:
        attention_heatmap = attention_heatmap.unsqueeze(0)
        plant_mask = plant_mask.unsqueeze(0)
        squeeze = True

 
    if plant_mask.shape[-2:] != attention_heatmap.shape[-2:]:
        plant_mask = F.interpolate(
            plant_mask.unsqueeze(1).float(),
            size=attention_heatmap.shape[-2:],
            mode="nearest"
        ).squeeze(1)

 
    B = attention_heatmap.shape[0]
    att_binary = torch.zeros_like(attention_heatmap)
    for i in range(B):
        threshold = torch.quantile(attention_heatmap[i].float(), q)
        att_binary[i] = (attention_heatmap[i] >= threshold).float()

    plant_binary = (plant_mask > 0.5).float()

   
    intersection = (att_binary * plant_binary).sum(dim=(-2, -1))
    union = ((att_binary + plant_binary) > 0).float().sum(dim=(-2, -1))
    iou = intersection / (union + 1e-8)

    if squeeze:
        iou = iou.squeeze(0)

    return iou


def differentiable_iou(attention_heatmap, plant_mask):
    """
    IoU différentiable (soft) pour utiliser dans la loss.

    Utilise des valeurs continues au lieu de binariser.

    Args:
        attention_heatmap : (B, H, W) — valeurs entre 0 et 1
        plant_mask        : (B, 1, H, W) ou (B, H, W) — binaire


    """
    if plant_mask.dim() == 4:
        plant_mask = plant_mask.squeeze(1)


    if plant_mask.shape[-2:] != attention_heatmap.shape[-2:]:
        plant_mask = F.interpolate(
            plant_mask.unsqueeze(1).float(),
            size=attention_heatmap.shape[-2:],
            mode="nearest"
        ).squeeze(1)

    plant_mask = plant_mask.float()

 
    intersection = (attention_heatmap * plant_mask).sum(dim=(-2, -1))
    union = (attention_heatmap + plant_mask -
             attention_heatmap * plant_mask).sum(dim=(-2, -1))
    iou = intersection / (union + 1e-8)

    return iou



class IoUAwareLoss(nn.Module):


    def __init__(self, lambda_iou=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.lambda_iou = (lambda_iou if lambda_iou is not None
                           else config.IOE_LOSS_WEIGHT)

    def forward(self, logits, labels, attention_heatmap=None,
                plant_mask=None):
 
        loss_ce = self.ce(logits, labels)

        if attention_heatmap is not None and plant_mask is not None:
            iou = differentiable_iou(attention_heatmap, plant_mask)
            loss_iou = 1.0 - iou.mean()
            return loss_ce + self.lambda_iou * loss_iou, loss_ce, loss_iou

        return loss_ce, loss_ce, torch.tensor(0.0)


def visualize_attention(image, heatmap, mask=None, title="",
                         save_path=None):
    """
    Superpose la heatmap d'attention sur l'image.

    Args:
        image    : (3, H, W) tensor ou (H, W, 3) numpy
        heatmap  : (H, W) numpy ou tensor
        mask     : (H, W) masque plante optionnel
        title    : titre du plot
        save_path: chemin de sauvegarde
    """
    if isinstance(image, torch.Tensor):
       
        mean = torch.tensor(config.IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(config.IMAGENET_STD).view(3, 1, 1)
        image = image * std + mean
        image = image.clamp(0, 1).permute(1, 2, 0).numpy()

    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.numpy()

    n_plots = 3 if mask is not None else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))

    axes[0].imshow(image)
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(image)
    axes[1].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[1].set_title("Attention Rollout")
    axes[1].axis("off")

    if mask is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.squeeze().numpy()
        axes[2].imshow(mask, cmap="gray")
        axes[2].set_title("Masque plante")
        axes[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sauvé: {save_path}")
    plt.close()


if __name__ == "__main__":

    N = 197 
    attn_maps = [torch.rand(1, 4, N, N).softmax(dim=-1) for _ in range(3)]

    cls_attn = attention_rollout(attn_maps)
    print(f"CLS attention shape: {cls_attn.shape}") 

    heatmap = rollout_to_heatmap(cls_attn, img_size=224, patch_size=16)
    print(f"Heatmap shape: {heatmap.shape}")  

   
    mask = torch.zeros(1, 224, 224)
    mask[:, 50:200, 50:200] = 1.0
    iou = compute_iou(heatmap, mask)
    print(f"IoU: {iou.item():.4f}")
