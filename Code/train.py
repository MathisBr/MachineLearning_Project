"""
train.py — Entraînement du modèle avec validation et sélection du meilleur modèle.

Stratégies appliquées pour combattre le sur-apprentissage (section 3.3 du sujet) :
  - BatchNorm dans chaque bloc convolutionnel
  - Dropout (0.4) dans la partie dense
  - SpecAugment sur les données d'entraînement
  - Early stopping si la val_loss stagne (patience = 10 epochs)
  - ReduceLROnPlateau pour adapter le taux d'apprentissage

Optimiseur : Adam — robuste, adaptatif, standard pour l'audio classification.

Usage:
    python train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import inspect

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

import config
from dataset import build_train_val_loaders
from model import build_model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    epoch_idx: int | None = None,
    total_epochs: int | None = None,
) -> tuple[float, float]:
    """
    Effectue une epoch d'entraînement.

    Returns:
        (train_loss, train_accuracy)
    """
    model.train()

    total_loss     = 0.0
    correct_preds  = 0
    total_samples  = 0

    batch_iterator = loader
    if tqdm is not None:
        batch_desc = f"Epoch {epoch_idx}/{total_epochs}" if epoch_idx is not None and total_epochs is not None else "Train"
        batch_iterator = tqdm(
            loader,
            total=len(loader),
            desc=batch_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )

    for spectrograms, labels in batch_iterator:
        spectrograms = spectrograms.to(config.DEVICE)
        labels       = labels.to(config.DEVICE)

        optimizer.zero_grad()

        logits = model(spectrograms)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss    += loss.item() * spectrograms.size(0)
        predictions    = logits.argmax(dim=1)
        correct_preds += (predictions == labels).sum().item()
        total_samples += spectrograms.size(0)

        if tqdm is not None:
            running_loss = total_loss / total_samples
            running_acc = correct_preds / total_samples
            batch_iterator.set_postfix(loss=f"{running_loss:.4f}", acc=f"{running_acc:.2%}")

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> tuple[float, float]:
    """
    Évalue le modèle sur un ensemble (validation ou test).
    Désactive le calcul de gradient pour accélérer l'évaluation (section 3.4).

    Returns:
        (val_loss, val_accuracy)
    """
    model.eval()

    total_loss    = 0.0
    correct_preds = 0
    total_samples = 0

    for spectrograms, labels in loader:
        spectrograms = spectrograms.to(config.DEVICE)
        labels       = labels.to(config.DEVICE)

        logits       = model(spectrograms)
        loss         = criterion(logits, labels)

        total_loss    += loss.item() * spectrograms.size(0)
        predictions    = logits.argmax(dim=1)
        correct_preds += (predictions == labels).sum().item()
        total_samples += spectrograms.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct_preds / total_samples
    return avg_loss, accuracy


class EarlyStopping:
    """Arrête l'entraînement si la val_loss ne s'améliore plus après `patience` epochs."""

    def __init__(self, patience: int, model_save_path: str):
        self.patience        = patience
        self.model_save_path = model_save_path
        self.best_val_loss   = float("inf")
        self.epochs_no_improve = 0
        self.should_stop     = False

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """
        Met à jour l'état de l'early stopping.

        Returns:
            True si le modèle s'est amélioré (meilleur checkpoint sauvegardé).
        """
        if val_loss < self.best_val_loss:
            self.best_val_loss     = val_loss
            self.epochs_no_improve = 0
            torch.save(model.state_dict(), self.model_save_path)
            return True

        self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            self.should_stop = True
        return False


def train(metadata_path: str = None) -> None:
    """
    Pipeline d'entraînement complet.

    Args:
        metadata_path: Chemin vers train_metadata.json (optionnel, utilise config par défaut).
    """
    if metadata_path is None:
        metadata_path = str(Path(config.CACHE_DIR) / "train_metadata.json")

    print("=" * 55)
    print("    ENTRAÎNEMENT — Reconnaissance d'instruments")
    print("=" * 55)
    print(f"Device      : {config.DEVICE}")
    print(f"Batch size  : {config.BATCH_SIZE}")
    print(f"Epochs max  : {config.NUM_EPOCHS}")
    print(f"LR initiale : {config.LEARNING_RATE}")
    print(f"Dropout     : {config.DROPOUT_RATE}")
    print(f"Patience    : {config.PATIENCE}")
    print()

    # ── Données ──────────────────────────────────────────────────────────────
    train_loader, val_loader = build_train_val_loaders(metadata_path)

    # ── Modèle, optimiseur, critère ──────────────────────────────────────────
    model     = build_model()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    # Poids inverses des frequences de classe (rare -> poids plus fort).
    train_counts = torch.tensor([637.0, 682.0, 721.0, 778.0])
    class_weights = (1.0 / train_counts)
    class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
    class_weights = class_weights.to(config.DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Compatibilité PyTorch: certaines versions n'acceptent pas `verbose`.
    scheduler_kwargs = {
        "mode": "min",
        "patience": config.LR_PATIENCE,
        "factor": config.LR_FACTOR,
    }
    if "verbose" in inspect.signature(torch.optim.lr_scheduler.ReduceLROnPlateau).parameters:
        scheduler_kwargs["verbose"] = True

    # ReduceLROnPlateau : réduit le LR si la val_loss stagne
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        **scheduler_kwargs,
    )

    early_stopping = EarlyStopping(
        patience=config.PATIENCE,
        model_save_path=config.MODEL_PATH,
    )

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8} | {'Best':>5}")
    print("-" * 63)

    epoch_iterator = range(1, config.NUM_EPOCHS + 1)
    if tqdm is not None:
        epoch_iterator = tqdm(
            epoch_iterator,
            total=config.NUM_EPOCHS,
            desc="Epochs",
            unit="epoch",
            dynamic_ncols=True,
            leave=False,
        )

    for epoch in epoch_iterator:
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            epoch_idx=epoch,
            total_epochs=config.NUM_EPOCHS,
        )
        val_loss,   val_acc   = evaluate(model, val_loader, criterion)

        scheduler.step(val_loss)
        is_best = early_stopping.step(val_loss, model)

        best_marker = "✓" if is_best else " "
        if tqdm is not None:
            epoch_iterator.set_postfix(
                train_loss=f"{train_loss:.4f}",
                val_loss=f"{val_loss:.4f}",
                val_acc=f"{val_acc:.2%}",
            )
        row = (
            f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2%} "
            f"| {val_loss:>8.4f} | {val_acc:>8.2%} | {best_marker:>5}"
        )
        if tqdm is not None:
            tqdm.write(row)
        else:
            print(row)

        if early_stopping.should_stop:
            if tqdm is not None:
                epoch_iterator.close()
                tqdm.write(f"\n⚡ Early stopping à l'epoch {epoch}.")
            else:
                print(f"\n⚡ Early stopping à l'epoch {epoch}.")
            break

    print(f"\n✓ Meilleur modèle sauvegardé : {config.MODEL_PATH}")
    print(f"  Meilleure val_loss : {early_stopping.best_val_loss:.4f}")


if __name__ == "__main__":
    train()
