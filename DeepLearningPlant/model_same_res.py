"""
Partie 2 — CrossViT avec deux branches de MÊME résolution.

Les deux branches ont le même patch size (16) et la même image size (224).
- Branche 1 : image non segmentée
- Branche 2 : image segmentée

Les poids NE sont PAS partagés entre les branches.
La fusion croisée (cross-attention) est identique au CrossViT original.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CrossViT"))

from models.crossvit import (
    PatchEmbed, CrossAttentionBlock, MultiScaleBlock,
    _compute_num_patches, VisionTransformer
)
from timm.models.layers import DropPath, trunc_normal_
from timm.models.vision_transformer import Block, Mlp

import config


class CrossViTSameResolution(nn.Module):
    """
    CrossViT modifié : deux branches avec la même résolution de patches.
    Branche 1 (non segmentée) et Branche 2 (segmentée) ont toutes deux
    une image 224x224 avec des patches de 16x16.

    Architecture identique au CrossViT original, mais les deux branches
    ont les mêmes dimensions (embed_dim peut différer pour plus de capacité).
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=2, embed_dim=256, depth=4, num_heads=8,
                 mlp_ratio=4.0, drop_rate=0.0, attn_drop_rate=0.0,
                 drop_path_rate=0.1):
        super().__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        num_patches = (img_size // patch_size) ** 2

  
        self.patch_embed_1 = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token_1 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_1 = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim))

       
        self.patch_embed_2 = PatchEmbed(
            img_size=img_size, patch_size=patch_size,
            in_chans=in_chans, embed_dim=embed_dim
        )
        self.cls_token_2 = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_2 = nn.Parameter(
            torch.zeros(1, 1 + num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

      
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

  
        self.blocks_1 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        self.blocks_2 = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                  drop_path=dpr[i], norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])


        n_cross = depth // 2
        self.cross_attn_1to2 = nn.ModuleList([
            CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate, norm_layer=nn.LayerNorm,
                has_mlp=False
            )
            for _ in range(n_cross)
        ])

        self.cross_attn_2to1 = nn.ModuleList([
            CrossAttentionBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=True, drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=drop_path_rate, norm_layer=nn.LayerNorm,
                has_mlp=False
            )
            for _ in range(n_cross)
        ])

   
        self.norm_1 = nn.LayerNorm(embed_dim)
        self.norm_2 = nn.LayerNorm(embed_dim)
        self.head_1 = nn.Linear(embed_dim, num_classes)
        self.head_2 = nn.Linear(embed_dim, num_classes)


        trunc_normal_(self.pos_embed_1, std=0.02)
        trunc_normal_(self.pos_embed_2, std=0.02)
        trunc_normal_(self.cls_token_1, std=0.02)
        trunc_normal_(self.cls_token_2, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, original, segmented, patch_weights=None, **kwargs):
        """
        Args:
            original  : (B, 3, H, W) — image non segmentée
            segmented : (B, 3, H, W) — image segmentée
            patch_weights : (B, N) — poids optionnels par patch (Partie 3)
        Returns:
            logits : (B, num_classes)
        """
        B = original.shape[0]

    
        original = F.interpolate(original, size=(self.img_size, self.img_size),
                                  mode="bicubic", align_corners=False)
        segmented = F.interpolate(segmented, size=(self.img_size, self.img_size),
                                   mode="bicubic", align_corners=False)


        x1 = self.patch_embed_1(original)
        x2 = self.patch_embed_2(segmented)

    
        cls1 = self.cls_token_1.expand(B, -1, -1)
        cls2 = self.cls_token_2.expand(B, -1, -1)
        x1 = torch.cat((cls1, x1), dim=1) + self.pos_embed_1
        x2 = torch.cat((cls2, x2), dim=1) + self.pos_embed_2
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

 
        if patch_weights is not None:
            
            pw = patch_weights.unsqueeze(-1)
            x1[:, 1:, :] = x1[:, 1:, :] * pw  # pondérer les patch tokens
            x2[:, 1:, :] = x2[:, 1:, :] * pw

  
        cross_idx = 0
        for i in range(len(self.blocks_1)):
            x1 = self.blocks_1[i](x1)
            x2 = self.blocks_2[i](x2)

     
            if (i + 1) % 2 == 0 and cross_idx < len(self.cross_attn_1to2):
             
                combined_for_1 = torch.cat(
                    [x1[:, 0:1, :], x2[:, 1:, :]], dim=1)
                new_cls1 = self.cross_attn_2to1[cross_idx](combined_for_1)
                x1 = torch.cat([new_cls1, x1[:, 1:, :]], dim=1)

             
                combined_for_2 = torch.cat(
                    [x2[:, 0:1, :], x1[:, 1:, :]], dim=1)
                new_cls2 = self.cross_attn_1to2[cross_idx](combined_for_2)
                x2 = torch.cat([new_cls2, x2[:, 1:, :]], dim=1)

                cross_idx += 1


        x1 = self.norm_1(x1)
        x2 = self.norm_2(x2)
        cls1 = x1[:, 0]
        cls2 = x2[:, 0]

        logits1 = self.head_1(cls1)
        logits2 = self.head_2(cls2)
        logits = (logits1 + logits2) / 2.0

        return logits

    def forward_with_attention(self, original, segmented, patch_weights=None):
        """
        Forward pass qui renvoie aussi les matrices d'attention
        pour l'attention rollout (Partie 4).
        """
        B = original.shape[0]
        attention_maps_1 = []
        attention_maps_2 = []

        original = F.interpolate(original, size=(self.img_size, self.img_size),
                                  mode="bicubic", align_corners=False)
        segmented = F.interpolate(segmented, size=(self.img_size, self.img_size),
                                   mode="bicubic", align_corners=False)

        x1 = self.patch_embed_1(original)
        x2 = self.patch_embed_2(segmented)

        cls1 = self.cls_token_1.expand(B, -1, -1)
        cls2 = self.cls_token_2.expand(B, -1, -1)
        x1 = torch.cat((cls1, x1), dim=1) + self.pos_embed_1
        x2 = torch.cat((cls2, x2), dim=1) + self.pos_embed_2
        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

        if patch_weights is not None:
            pw = patch_weights.unsqueeze(-1)
            x1[:, 1:, :] = x1[:, 1:, :] * pw
            x2[:, 1:, :] = x2[:, 1:, :] * pw

        cross_idx = 0
        for i in range(len(self.blocks_1)):
            x1 = self.blocks_1[i](x1)
            x2 = self.blocks_2[i](x2)

       
            attention_maps_1.append(
                self.blocks_1[i].attn.attn_weights
                if hasattr(self.blocks_1[i].attn, 'attn_weights') else None
            )
            attention_maps_2.append(
                self.blocks_2[i].attn.attn_weights
                if hasattr(self.blocks_2[i].attn, 'attn_weights') else None
            )

            if (i + 1) % 2 == 0 and cross_idx < len(self.cross_attn_1to2):
                combined_for_1 = torch.cat(
                    [x1[:, 0:1, :], x2[:, 1:, :]], dim=1)
                new_cls1 = self.cross_attn_2to1[cross_idx](combined_for_1)
                x1 = torch.cat([new_cls1, x1[:, 1:, :]], dim=1)

                combined_for_2 = torch.cat(
                    [x2[:, 0:1, :], x1[:, 1:, :]], dim=1)
                new_cls2 = self.cross_attn_1to2[cross_idx](combined_for_2)
                x2 = torch.cat([new_cls2, x2[:, 1:, :]], dim=1)
                cross_idx += 1

        x1 = self.norm_1(x1)
        x2 = self.norm_2(x2)
        cls1 = x1[:, 0]
        cls2 = x2[:, 0]
        logits1 = self.head_1(cls1)
        logits2 = self.head_2(cls2)
        logits = (logits1 + logits2) / 2.0

        return logits, attention_maps_1, attention_maps_2


if __name__ == "__main__":
    model = CrossViTSameResolution()
    model.eval()
    x_orig = torch.randn(2, 3, 224, 224)
    x_seg = torch.randn(2, 3, 224, 224)
    out = model(x_orig, x_seg)
    print(f"Same resolution — Output shape: {out.shape}") 
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
