"""
Partie 1 — CrossViT de base avec 4 configurations (A, B, C1, C2).

Utilise le VisionTransformer d'IBM/CrossViT avec poids pré-entraînés ImageNet.
On adapte la tête de classification pour 2 classes (épines oui/non).

Configurations :
  A  : les deux branches reçoivent l'image NON segmentée
  B  : les deux branches reçoivent l'image segmentée
  C1 : segmentée → Large (224, patch 16), non segmentée → Small (240, patch 12)
  C2 : segmentée → Small (240, patch 12), non segmentée → Large (224, patch 16)
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CrossViT"))

from models.crossvit import VisionTransformer, _model_urls
import config


def create_crossvit(pretrained=True, num_classes=2):
    """
    Crée un CrossViT-small (patch 12/16, embed 192/384) pré-entraîné ImageNet
    et remplace les têtes de classification pour `num_classes` classes.
    """
    model = VisionTransformer(
        img_size=[240, 224],
        patch_size=[12, 16],
        embed_dim=[192, 384],
        depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        num_heads=[6, 6],
        mlp_ratio=[4, 4, 1],
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )

    if pretrained:
        state_dict = torch.hub.load_state_dict_from_url(
            _model_urls["crossvit_small_224"], map_location="cpu"
        )
        model.load_state_dict(state_dict, strict=False)



    embed_dims = [192, 384]
    model.head = nn.ModuleList([
        nn.Linear(embed_dims[i], num_classes)
        for i in range(2)
    ])

    return model


class CrossViTHerbarium(nn.Module):
    """
    Wrapper autour du CrossViT pour gérer les 4 configurations A/B/C1/C2.

    Le CrossViT original prend UNE seule image et la resize en interne
    pour les deux branches (240 pour Small, 224 pour Large).

    Ici, on modifie forward_features pour injecter des images différentes
    dans chaque branche selon la config.
    """

    def __init__(self, mode="A", pretrained=True):
        """
        Args:
            mode: "A", "B", "C1" ou "C2"
            pretrained: charger les poids ImageNet
        """
        super().__init__()
        assert mode in ("A", "B", "C1", "C2"), f"Mode inconnu: {mode}"
        self.mode = mode
        self.crossvit = create_crossvit(pretrained=pretrained)

    def forward(self, original, segmented, **kwargs):
        """
        Args:
            original  : (B, 3, H, W) image non segmentée
            segmented : (B, 3, H, W) image segmentée
        Returns:
            logits    : (B, num_classes)
        """
   
   
        if self.mode == "A":
         
            img_small, img_large = original, original
        elif self.mode == "B":
        
            img_small, img_large = segmented, segmented
        elif self.mode == "C1":
          
            img_small, img_large = original, segmented
        elif self.mode == "C2":
       
            img_small, img_large = segmented, original

     
        img_small = F.interpolate(img_small, size=(240, 240), mode="bicubic",
                                   align_corners=False)
        img_large = F.interpolate(img_large, size=(224, 224), mode="bicubic",
                                   align_corners=False)

       
        B = img_small.shape[0]
        xs = []
        for i, img in enumerate([img_small, img_large]):
            tmp = self.crossvit.patch_embed[i](img)
            cls_tokens = self.crossvit.cls_token[i].expand(B, -1, -1)
            tmp = torch.cat((cls_tokens, tmp), dim=1)
            tmp = tmp + self.crossvit.pos_embed[i]
            tmp = self.crossvit.pos_drop(tmp)
            xs.append(tmp)

        for blk in self.crossvit.blocks:
            xs = blk(xs)

        xs = [self.crossvit.norm[i](x) for i, x in enumerate(xs)]
        cls_tokens = [x[:, 0] for x in xs]

        logits = [self.crossvit.head[i](cls_tokens[i]) for i in range(2)]
        logits = torch.mean(torch.stack(logits, dim=0), dim=0)

        return logits

    def get_attention_maps(self):
        """Renvoie les matrices d'attention de chaque bloc (pour rollout)."""
   
        pass


if __name__ == "__main__":

    model = CrossViTHerbarium(mode="C1", pretrained=False)
    model.eval()
    x_orig = torch.randn(2, 3, 224, 224)
    x_seg = torch.randn(2, 3, 224, 224)
    out = model(x_orig, x_seg)
    print(f"Mode C1 — Output shape: {out.shape}")  

    for m in ("A", "B", "C2"):
        model_m = CrossViTHerbarium(mode=m, pretrained=False)
        model_m.eval()
        out = model_m(x_orig, x_seg)
        print(f"Mode {m} — Output shape: {out.shape}")
