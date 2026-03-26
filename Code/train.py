"""
train.py — Boucle d'entraînement optimisée.

Inclut :
  - AdamW optimizer avec weight decay
  - CosineAnnealingWarmRestarts scheduler
  - Label Smoothing sur CrossEntropyLoss
  - Mixup training intégré
  - Mixed Precision (AMP) sur CUDA
  - Early stopping sur val_loss
  - Sauvegarde du meilleur modèle
"""

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
import time

import config
from model import InstrumentCNN
from dataset import get_train_val_loaders


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss pour Mixup : combinaison pondérée des deux labels."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp):
    """Entraîne le modèle pendant une epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in tqdm(train_loader, desc="  Train", leave=False):
        specs, labels_a, labels_b, lam = batch
        specs = specs.to(device, non_blocking=True)
        labels_a = labels_a.to(device, non_blocking=True)
        labels_b = labels_b.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(specs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)

        if use_amp:
            scaler.scale(loss).backward()
            # Gradient clipping pour la stabilité
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        running_loss += loss.item() * specs.size(0)

        # Accuracy (basée sur labels_a, le label principal du mixup)
        _, predicted = outputs.max(1)
        total += labels_a.size(0)
        correct += predicted.eq(labels_a).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, val_loader, criterion, device, use_amp):
    """Évalue le modèle sur le set de validation."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for specs, labels in tqdm(val_loader, desc="  Val  ", leave=False):
        specs = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            outputs = model(specs)
            loss = criterion(outputs, labels)

        running_loss += loss.item() * specs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def train_model():
    """
    Boucle d'entraînement complète.
    
    Returns:
        model: le meilleur modèle entraîné
    """
    device = config.DEVICE
    use_amp = config.USE_AMP

    # Créer le dossier pour sauvegarder les modèles
    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    best_model_path = config.MODEL_DIR / "best_model.pt"

    # Modèle
    model = InstrumentCNN(num_classes=config.NUM_CLASSES).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[Train] Modèle: {param_count:,} paramètres")
    print(f"[Train] Device: {device} | AMP: {use_amp}")

    # DataLoaders
    train_loader, val_loader = get_train_val_loaders()

    # Loss avec Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)

    # Optimiseur AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler CosineAnnealingWarmRestarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.COSINE_T0,
        T_mult=config.COSINE_T_MULT,
    )

    # Mixed Precision scaler
    scaler = GradScaler(enabled=use_amp)

    # Early stopping
    best_val_loss = float("inf")
    best_val_acc = 0.0
    patience_counter = 0

    print(f"\n[Train] Début de l'entraînement ({config.NUM_EPOCHS} epochs max)")
    print("-" * 70)
    start_time = time.time()

    for epoch in range(1, config.NUM_EPOCHS + 1):
        epoch_start = time.time()
        current_lr = optimizer.param_groups[0]["lr"]

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Validation
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, use_amp
        )

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Logging
        print(
            f"Epoch {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}% | "
            f"LR: {current_lr:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Sauvegarde du meilleur modèle (basé sur val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            patience_counter = 0

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_acc": best_val_acc,
            }, best_model_path)
            print(f"  → Meilleur modèle sauvegardé (val_loss: {best_val_loss:.4f}, val_acc: {best_val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.EARLY_STOP_PATIENCE:
                print(f"\n[Train] Early stopping après {epoch} epochs (patience={config.EARLY_STOP_PATIENCE})")
                break

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"[Train] Terminé en {total_time:.1f}s")
    print(f"[Train] Meilleur modèle : val_loss={best_val_loss:.4f}, val_acc={best_val_acc:.1f}%")

    # Charger le meilleur modèle
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[Train] Modèle chargé depuis epoch {checkpoint['epoch']}")

    return model


if __name__ == "__main__":
    train_model()
