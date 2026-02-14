"""
Script d'évaluation — Génère tous les résultats pour le rapport.

Usage :
  # Évaluer un modèle sauvegardé + générer heatmaps + calculer IoU
  python evaluate.py --checkpoint checkpoints/part1_A_best.pth --part 1 --mode A

  # Générer le tableau comparatif de toutes les configs
  python evaluate.py --compare-all

  # Générer les heatmaps pour le modèle same_res
  python evaluate.py --checkpoint checkpoints/part2_same_res_best.pth --part 2 --heatmaps
"""

import argparse
import os
import sys
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score, accuracy_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

import config
from dataloader import get_dataloaders, HerbariumDataset, get_transforms
from model import CrossViTHerbarium
from model_same_res import CrossViTSameResolution
from patch_weights import compute_patch_weights
from attention_rollout import (
    AttentionHook, attention_rollout, rollout_to_heatmap,
    compute_iou, visualize_attention
)


def parse_args():
    p = argparse.ArgumentParser(description="Évaluation Herbiers & CrossViT")
    p.add_argument("--checkpoint", type=str, default=None,
                    help="Chemin du checkpoint")
    p.add_argument("--part", type=int, default=1, choices=[1, 2, 5])
    p.add_argument("--mode", type=str, default="A",
                    choices=["A", "B", "C1", "C2"])
    p.add_argument("--heatmaps", action="store_true",
                    help="Générer les heatmaps d'attention")
    p.add_argument("--n-heatmaps", type=int, default=10,
                    help="Nombre de heatmaps à générer")
    p.add_argument("--compare-all", action="store_true",
                    help="Comparer toutes les configs")
    p.add_argument("--use-patch-weights", action="store_true")
    p.add_argument("--weight-fn", type=str, default="linear")
    p.add_argument("--gamma", type=float, default=1.0)
    return p.parse_args()


def load_model(args):
    """Charge le modèle depuis un checkpoint."""
    if args.part == 1:
        model = CrossViTHerbarium(mode=args.mode, pretrained=False)
    else:
        model = CrossViTSameResolution(
            img_size=config.IMG_SIZE_SAME,
            patch_size=config.PATCH_SIZE_SAME,
            num_classes=config.NUM_CLASSES,
        )

    if args.checkpoint and os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=config.DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Checkpoint chargé: {args.checkpoint}")
        print(f"  Epoch: {ckpt.get('epoch', '?')}, "
              f"Val F1: {ckpt.get('val_f1', '?'):.4f}")
    else:
        print("Pas de checkpoint fourni — modèle non entraîné.")

    return model


@torch.no_grad()
def full_evaluation(model, loader, device, args, patch_size=16):
    """Évaluation complète avec métriques détaillées."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    for batch in loader:
        original = batch["original"].to(device)
        segmented = batch["segmented"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        pw = None
        if args.use_patch_weights:
            pw = compute_patch_weights(
                mask, patch_size, method=args.weight_fn,
                gamma=args.gamma
            )

        if args.part == 1:
            logits = model(original, segmented)
        else:
            logits = model(original, segmented, patch_weights=pw)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": acc,
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "labels": all_labels,
        "probabilities": all_probs,
    }


@torch.no_grad()
def compute_attention_iou(model, loader, device, args, patch_size=16,
                           n_samples=None):
    """
    Calcule l'IoU entre les heatmaps d'attention et les masques plante
    sur le set de validation.
    """
    model.eval()
    ious = []
    count = 0

    
    hook = AttentionHook()
    hook.register(model)

    for batch in loader:
        original = batch["original"].to(device)
        segmented = batch["segmented"].to(device)
        mask = batch["mask"].to(device)

        hook.clear()

        if args.part == 1:
            logits = model(original, segmented)
        else:
            logits = model(original, segmented)

        if len(hook.attention_maps) > 0:
            cls_attn = attention_rollout(hook.attention_maps)
            img_size = config.IMG_SIZE_SAME if args.part != 1 else 224
            heatmap = rollout_to_heatmap(cls_attn, img_size, patch_size)

            mask_resized = F.interpolate(
                mask, size=(img_size, img_size), mode="nearest"
            ).squeeze(1)

            for i in range(heatmap.shape[0]):
                iou = compute_iou(heatmap[i], mask_resized[i])
                ious.append(iou.item())

        count += original.shape[0]
        if n_samples and count >= n_samples:
            break

    hook.remove()

    if ious:
        return {
            "mean_iou": np.mean(ious),
            "std_iou": np.std(ious),
            "all_ious": ious,
        }
    return {"mean_iou": 0.0, "std_iou": 0.0, "all_ious": []}


@torch.no_grad()
def generate_heatmaps(model, loader, device, args, n_samples=10,
                       patch_size=16, save_dir=None):
    """Génère et sauvegarde les heatmaps d'attention."""
    model.eval()
    save_dir = save_dir or os.path.join(config.RESULTS_DIR, "heatmaps")
    os.makedirs(save_dir, exist_ok=True)

    hook = AttentionHook()
    hook.register(model)
    count = 0

    for batch in loader:
        if count >= n_samples:
            break

        original = batch["original"].to(device)
        segmented = batch["segmented"].to(device)
        mask = batch["mask"].to(device)
        codes = batch["code"]

        hook.clear()

        if args.part == 1:
            logits = model(original, segmented)
        else:
            logits = model(original, segmented)

        if len(hook.attention_maps) > 0:
            cls_attn = attention_rollout(hook.attention_maps)
            img_size = config.IMG_SIZE_SAME if args.part != 1 else 224
            heatmap = rollout_to_heatmap(cls_attn, img_size, patch_size)

            preds = logits.argmax(dim=1).cpu()

            for i in range(min(original.shape[0], n_samples - count)):
                code = codes[i]
                pred = "epines" if preds[i] == 1 else "sans_epines"

                # IoU pour cette image
                mask_i = F.interpolate(
                    mask[i:i+1], size=(img_size, img_size), mode="nearest"
                ).squeeze()
                iou = compute_iou(heatmap[i].cpu(), mask_i.cpu())

                title = (f"{code} — Pred: {pred} — "
                         f"IoU: {iou.item():.3f}")

                visualize_attention(
                    original[i].cpu(),
                    heatmap[i].cpu(),
                    mask=mask[i].cpu(),
                    title=title,
                    save_path=os.path.join(save_dir, f"{code}_heatmap.png")
                )
                count += 1

    hook.remove()
    print(f"{count} heatmaps sauvées dans {save_dir}")


def compare_all_configs():
    """
    Compare toutes les configurations et génère un tableau récapitulatif.
    Lit les fichiers history JSON dans results/.
    """
    results = {}
    results_dir = config.RESULTS_DIR

    if not os.path.exists(results_dir):
        print("Pas de résultats trouvés. Lancez d'abord les entraînements.")
        return

    for fname in os.listdir(results_dir):
        if fname.endswith("_history.json"):
            exp_name = fname.replace("_history.json", "")
            with open(os.path.join(results_dir, fname)) as f:
                history = json.load(f)

            best_epoch = np.argmax(history["val_f1"])
            results[exp_name] = {
                "best_epoch": int(best_epoch) + 1,
                "best_val_acc": history["val_acc"][best_epoch],
                "best_val_f1": history["val_f1"][best_epoch],
                "final_train_loss": history["train_loss"][-1],
                "final_val_loss": history["val_loss"][-1],
            }

    if not results:
        print("Aucun résultat trouvé.")
        return


    print(f"\n{'='*80}")
    print(f"{'Expérience':<30} {'Epoch':>6} {'Val Acc':>10} "
          f"{'Val F1':>10} {'Train Loss':>12} {'Val Loss':>10}")
    print(f"{'='*80}")

    for name, r in sorted(results.items()):
        print(f"{name:<30} {r['best_epoch']:>6} "
              f"{r['best_val_acc']:>10.4f} {r['best_val_f1']:>10.4f} "
              f"{r['final_train_loss']:>12.4f} {r['final_val_loss']:>10.4f}")

    print(f"{'='*80}\n")

    summary_path = os.path.join(results_dir, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Résumé sauvé: {summary_path}")


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    names = list(results.keys())
    accs = [results[n]["best_val_acc"] for n in names]
    f1s = [results[n]["best_val_f1"] for n in names]

    x = np.arange(len(names))
    axes[0].bar(x, accs, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha="right")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Validation Accuracy")
    axes[0].set_ylim(0, 1)

    axes[1].bar(x, f1s, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha="right")
    axes[1].set_ylabel("F1-Score")
    axes[1].set_title("Validation F1-Score")
    axes[1].set_ylim(0, 1)

    plt.suptitle("Comparaison des configurations")
    plt.tight_layout()
    plot_path = os.path.join(results_dir, "comparison_chart.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Graphique comparatif sauvé: {plot_path}")


def main():
    args = parse_args()
    device = config.DEVICE

    if args.compare_all:
        compare_all_configs()
        return

  
    model = load_model(args)
    model = model.to(device)


    img_size = config.IMG_SIZE_SAME if args.part in (2, 5) else 224
    patch_size = config.PATCH_SIZE_SAME if args.part in (2, 5) else 16
    _, val_loader = get_dataloaders(img_size=img_size)

  
    print("\n--- Évaluation ---")
    metrics = full_evaluation(model, val_loader, device, args, patch_size)
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"F1-Score : {metrics['f1_score']:.4f}")
    print(f"Matrice de confusion :")
    print(np.array(metrics["confusion_matrix"]))

  
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))
    cm = np.array(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Sans épines", "Épines"])
    disp.plot(ax=ax, cmap="Blues")
    exp_name = f"part{args.part}_{args.mode}" if args.part == 1 else f"part{args.part}_same_res"
    cm_path = os.path.join(config.RESULTS_DIR, f"{exp_name}_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Matrice de confusion sauvée: {cm_path}")

    
    if args.heatmaps:
        print("\n--- Génération des heatmaps ---")
        heatmap_dir = os.path.join(config.RESULTS_DIR, f"heatmaps_{exp_name}")
        generate_heatmaps(
            model, val_loader, device, args,
            n_samples=args.n_heatmaps, patch_size=patch_size,
            save_dir=heatmap_dir
        )

   
        print("\n--- Calcul IoU (attention vs masque plante) ---")
        iou_results = compute_attention_iou(
            model, val_loader, device, args, patch_size=patch_size
        )
        print(f"IoU moyen : {iou_results['mean_iou']:.4f} "
              f"(+/- {iou_results['std_iou']:.4f})")

    
        iou_path = os.path.join(config.RESULTS_DIR, f"{exp_name}_iou.json")
        with open(iou_path, "w") as f:
            json.dump({
                "mean_iou": iou_results["mean_iou"],
                "std_iou": iou_results["std_iou"],
            }, f, indent=2)


if __name__ == "__main__":
    main()
