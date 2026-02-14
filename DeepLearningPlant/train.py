"""

# A METTRE DANS LE README§!!
Script d'entraînement principal — couvre les 5 objectifs du projet.

Usage :
  # Partie 1 — Configurations A/B/C1/C2
  python train.py --part 1 --mode A
  python train.py --part 1 --mode B
  python train.py --part 1 --mode C1
  python train.py --part 1 --mode C2

  # Partie 2 — Même résolution (sans pondération)
  python train.py --part 2

  # Partie 2 — Même résolution + pondération par patch (Partie 3)
  python train.py --part 2 --use-patch-weights --weight-fn linear

  # Partie 5 — Même résolution + loss IoU
  python train.py --part 5

  # Partie 5 — Même résolution + pondération + loss IoU
  python train.py --part 5 --use-patch-weights --weight-fn power --gamma 0.5
"""

import argparse
import os
import sys
import json
import time

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

import config
from dataloader import get_dataloaders, get_transforms, HerbariumDataset
from model import CrossViTHerbarium
from model_same_res import CrossViTSameResolution
from patch_weights import compute_patch_weights, PatchWeightedLoss
from attention_rollout import (
    AttentionHook, attention_rollout, rollout_to_heatmap,
    compute_iou, IoUAwareLoss
)


def parse_args():
    p = argparse.ArgumentParser(description="Entraînement Herbiers & CrossViT")
    p.add_argument("--part", type=int, default=1, choices=[1, 2, 5],
                    help="Partie du projet (1, 2 ou 5)")
    p.add_argument("--mode", type=str, default="A",
                    choices=["A", "B", "C1", "C2"],
                    help="Config CrossViT (Partie 1 uniquement)")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--use-patch-weights", action="store_true",
                    help="Activer la pondération par patch (Partie 3)")
    p.add_argument("--use-iou-loss", action="store_true",
                    help="Activer la loss IoU (Partie 2 ou 5)")
    p.add_argument("--weight-fn", type=str, default="linear",
                    choices=["linear", "power", "normalized", "log", "sigmoid"])
    p.add_argument("--gamma", type=float, default=1.0,
                    help="Gamma pour la pondération puissance")
    p.add_argument("--lambda-iou", type=float, default=1.0,
                    help="Lambda pour la loss IoU")
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    return p.parse_args()


def create_model(args):
    """Crée le modèle selon la partie choisie."""
    if args.part == 1:
        model = CrossViTHerbarium(mode=args.mode, pretrained=args.pretrained)
        name = f"part1_{args.mode}"
    elif args.part in (2, 5):
        model = CrossViTSameResolution(
            img_size=config.IMG_SIZE_SAME,
            patch_size=config.PATCH_SIZE_SAME,
            num_classes=config.NUM_CLASSES,
        )
        suffix = ""
        if args.use_patch_weights:
            suffix += f"_pw_{args.weight_fn}"
        if args.use_iou_loss or args.part == 5:
            suffix += "_iou"
        name = f"part{args.part}_same_res{suffix}"
    else:
        raise ValueError(f"Partie inconnue: {args.part}")

    return model, name


def train_one_epoch(model, loader, optimizer, criterion, device, args,
                     epoch, patch_size=16):
    """Entraîne le modèle pour une époque."""
    model.train()
    total_loss = 0
    total_loss_ce = 0
    total_loss_iou = 0
    all_preds = []
    all_labels = []

    use_iou = args.use_iou_loss or args.part == 5


    hook = None
    if use_iou:
        hook = AttentionHook(differentiable=True)
        hook.register(model)

    pbar = tqdm(loader, desc=f"  Train epoch {epoch+1}", leave=False,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}")
    for batch in pbar:
        original = batch["original"].to(device)
        segmented = batch["segmented"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()

   
        if hook is not None:
            hook.clear()

    
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


        if use_iou:
            attention_heatmap = None
            try:
                if hook.attention_maps:
                    cls_attn = attention_rollout(hook.attention_maps)
                    img_size = config.IMG_SIZE_SAME if args.part in (2, 5) else 224
                    attention_heatmap = rollout_to_heatmap(
                        cls_attn, img_size=img_size, patch_size=patch_size
                    )
            except Exception as e:
                print(f"\n Erreur attention rollout: {e}")
                attention_heatmap = None

            loss, loss_ce, loss_iou = criterion(
                logits, labels, attention_heatmap=attention_heatmap, plant_mask=mask
            )
            total_loss_ce += loss_ce.item()
            total_loss_iou += loss_iou.item() if isinstance(loss_iou, torch.Tensor) else loss_iou
        elif args.use_patch_weights:
            loss = criterion(logits, labels, patch_weights=pw)
        else:
            loss = criterion(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix_str(f"{loss.item():.4f}")

 
    if hook is not None:
        hook.remove()

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")

    if use_iou:
        avg_loss_ce = total_loss_ce / len(loader)
        avg_loss_iou = total_loss_iou / len(loader)
        print(f"    [CE: {avg_loss_ce:.4f}, IoU: {avg_loss_iou:.4f}]", end="")

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device, args, patch_size=16):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, desc="  Val", leave=False)
    for batch in pbar:
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

        if args.use_iou_loss or args.part == 5:
            loss, _, _ = criterion(logits, labels)
        elif args.use_patch_weights:
            loss = criterion(logits, labels, patch_weights=pw)
        else:
            loss = criterion(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, acc, f1, cm


def main():
    args = parse_args()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.SEED)

    device = config.DEVICE
    print(f"Device: {device}")

 
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

 
    img_size = config.IMG_SIZE_SAME if args.part in (2, 5) else 224
    train_loader, val_loader = get_dataloaders(
        img_size=img_size,
        batch_size=args.batch_size
    )


    model, exp_name = create_model(args)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Expérience : {exp_name}")
    print(f"Paramètres entraînables : {n_params:,}")

 
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )


    if args.use_iou_loss or args.part == 5:
        criterion = IoUAwareLoss(lambda_iou=args.lambda_iou)
    elif args.use_patch_weights:
        criterion = PatchWeightedLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
    }
    best_f1 = 0.0
    patch_size = config.PATCH_SIZE_SAME if args.part in (2, 5) else 16

    print(f"\n{'='*60}")
    print(f"Début de l'entraînement — {args.epochs} époques")
    print(f"{'='*60}\n")

    for epoch in range(args.epochs):
        t0 = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args,
            epoch, patch_size
        )

        val_loss, val_acc, val_f1, val_cm = evaluate(
            model, val_loader, criterion, device, args, patch_size
        )

        scheduler.step()
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch + 1:3d}/{args.epochs} "
              f"({elapsed:.1f}s) | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  "
              f"F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  "
              f"F1: {val_f1:.4f}")

     
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_path = os.path.join(config.SAVE_DIR, f"{exp_name}_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
            }, save_path)
            print(f"  → Meilleur modèle sauvé (F1: {val_f1:.4f})")

   
    print(f"\n{'='*60}")
    print(f"Entraînement terminé — Meilleur F1: {best_f1:.4f}")
    print(f"Matrice de confusion (dernière époque):")
    print(val_cm)
    print(f"{'='*60}\n")


    history_path = os.path.join(config.RESULTS_DIR, f"{exp_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Historique sauvé: {history_path}")

    
    plot_curves(history, exp_name)


def plot_curves(history, exp_name):
    """Trace et sauvegarde les courbes d'apprentissage."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))


    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True)

  
    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    axes[1].grid(True)


    axes[2].plot(epochs, history["train_f1"], label="Train")
    axes[2].plot(epochs, history["val_f1"], label="Val")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1-Score")
    axes[2].set_title("F1-Score")
    axes[2].legend()
    axes[2].grid(True)

    plt.suptitle(f"Courbes d'apprentissage — {exp_name}")
    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, f"{exp_name}_curves.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Courbes sauvées: {save_path}")


if __name__ == "__main__":
    main()
