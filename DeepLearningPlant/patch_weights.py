

import torch
import torch.nn.functional as F
import config


def compute_patch_ratios(mask, patch_size):
    """
    Calcule le ratio de pixels plante par patch.

    Args:
        mask       : (B, 1, H, W) masque binaire (0 ou 1)
        patch_size : taille du patch (ex: 16)

    Returns:
        ratios : (B, N) ratio de pixels plante par patch
                 N = (H // patch_size) * (W // patch_size)
    """
   
    ratios = F.avg_pool2d(mask.float(), kernel_size=patch_size,
                           stride=patch_size)
    B = ratios.shape[0]
    ratios = ratios.view(B, -1)  
    return ratios


def compute_patch_weights(mask, patch_size, method="linear",
                           epsilon=None, gamma=None):
    """
    Calcule les poids par patch à partir du masque.

    Args:
        mask       : (B, 1, H, W) masque binaire
        patch_size : taille du patch
        method     : "linear", "power" ou "normalized"
        epsilon    : petit epsilon pour éviter les poids nuls
        gamma      : exposant pour la méthode puissance

    Returns:
        weights : (B, N) poids par patch
    """
    eps = epsilon if epsilon is not None else config.WEIGHT_EPSILON
    gam = gamma if gamma is not None else config.WEIGHT_GAMMA

    ratios = compute_patch_ratios(mask, patch_size)  

    if method == "linear":
        weights = eps + ratios
    elif method == "power":
        weights = (eps + ratios) ** gam
    elif method == "normalized":
        weights = eps + ratios
     
        mean_w = weights.mean(dim=1, keepdim=True)
        weights = weights / (mean_w + 1e-8)
    elif method == "log":
       
        weights = torch.log(1.0 + ratios + eps)
    elif method == "sigmoid":

        alpha = gam if gam != 1.0 else 10.0
        weights = torch.sigmoid(alpha * (ratios - 0.5))
    else:
        raise ValueError(f"Méthode inconnue: {method}")

    return weights


class PatchWeightedLoss(torch.nn.Module):
    """
    Perte CrossEntropy pondérée par les poids de patch.

    Au lieu d'une simple CE, on peut pondérer la contribution
    de chaque échantillon du batch par le poids moyen de ses patches plante.
    Plus un échantillon a de plante visible, plus il pèse.
    """

    def __init__(self):
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")

    def forward(self, logits, labels, patch_weights=None):
        """
        Args:
            logits        : (B, C)
            labels        : (B,)
            patch_weights : (B, N) poids par patch (optionnel)
        """
        loss = self.ce(logits, labels)  

        if patch_weights is not None:
           
            sample_weights = patch_weights.mean(dim=1)  
         
            sample_weights = sample_weights / (sample_weights.mean() + 1e-8)
            loss = loss * sample_weights

        return loss.mean()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

 
    mask = torch.zeros(1, 1, 224, 224)
 
    mask[:, :, :, 112:] = 1.0

    patch_size = 16
    for method in ("linear", "power", "normalized"):
        w = compute_patch_weights(mask, patch_size, method=method)
        grid_size = 224 // patch_size  # 14
        w_grid = w.view(1, grid_size, grid_size)
        print(f"{method:>12s} — min={w.min():.3f}  max={w.max():.3f}  "
              f"mean={w.mean():.3f}")

    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(mask[0, 0], cmap="gray")
    axes[0].set_title("Masque")
    for i, method in enumerate(("linear", "power", "normalized")):
        w = compute_patch_weights(mask, patch_size, method=method)
        w_grid = w.view(grid_size, grid_size).numpy()
        axes[i + 1].imshow(w_grid, cmap="hot", vmin=0)
        axes[i + 1].set_title(f"Poids ({method})")
    plt.tight_layout()
    plt.savefig("patch_weights_demo.png", dpi=100)
    print("Sauvé dans patch_weights_demo.png")
