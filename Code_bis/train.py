"""
train.py — Entraînement + Évaluation finale CNN IRMAS
======================================================
Lancer :
    python train.py

Sorties :
    ../Models/<MODEL_NAME>.pt   meilleur modèle (val_loss)
    viz_spectrogramme.png
    viz_distribution.png
    viz_courbes.png
    training_log.json
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import librosa
from tqdm import tqdm

from dataset import (
    INSTRUMENTS, IDX_TO_LABEL, LABEL_TO_IDX,
    SAMPLE_RATE,
    get_dataloaders, build_val_transform,
    rms_normalize,
)

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTES — modifier ici pour changer de config
# ══════════════════════════════════════════════════════════════════════════

MODEL_NAME   = "MonkI"               # nom du fichier de sortie
MODEL_EXT    = ".pt"                 # .pt ou .pth, au choix

TRAIN_DIR    = "../Dataset/TrainData"
TEST_DIR     = "../Dataset/TestData"
MODELS_DIR   = "../Models"

EPOCHS       = 30
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10                    # early stopping
VAL_RATIO    = 0.25
SEED         = 42
DROPOUT      = 0.3

# ══════════════════════════════════════════════════════════════════════════
#  ARCHITECTURE CNN
#  ⚠ Doit rester synchronisée avec eval.py
#    conv_layers : 4 blocs 64→128→256→256
#    classifier  : Flatten → Dropout → Linear(4096,256) → ReLU → Dropout → Linear(256,4)
# ══════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class InstrumentCNN(nn.Module):
    def __init__(self, n_classes: int = 4, dropout: float = DROPOUT):
        super().__init__()
        self.conv_layers = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.conv_layers(x)))

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ══════════════════════════════════════════════════════════════════════════
#  VISUALISATIONS
# ══════════════════════════════════════════════════════════════════════════

def viz_spectrogrammes(out_path: str = "viz_spectrogramme.png"):
    transform = build_val_transform()
    root      = Path(TRAIN_DIR)
    emojis    = {"gac": "🎸", "org": "🎻", "pia": "🎹", "voi": "🎤"}

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle("Log-Mel Spectrogramme par instrument (1 exemple)", fontsize=13)

    for ax, inst in zip(axes, INSTRUMENTS):
        wav_file = sorted((root / inst).glob("*.wav"))[0]
        waveform, sr = librosa.load(str(wav_file), sr=SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)
        waveform = rms_normalize(waveform)
        mel = transform(waveform)

        ax.imshow(mel[0].numpy(), aspect="auto", origin="lower", cmap="magma")
        ax.set_title(f"{emojis.get(inst,'')} {inst.upper()}", fontsize=12)
        ax.set_xlabel("Frames temporelles")
        ax.set_ylabel("Bandes Mel")
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[viz] Spectrogrammes → {out_path}")


def viz_distribution(stats: dict, out_path: str = "viz_distribution.png"):
    x      = np.arange(len(INSTRUMENTS))
    width  = 0.35
    trains = [stats[i]["train"] for i in INSTRUMENTS]
    vals   = [stats[i]["val"]   for i in INSTRUMENTS]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0B1F2E")
    ax.set_facecolor("#0D2D42")

    bars1 = ax.bar(x - width/2, trains, width, label="Train", color="#28AAE1", alpha=0.85)
    bars2 = ax.bar(x + width/2, vals,   width, label="Val",   color="#00D4FF", alpha=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels([i.upper() for i in INSTRUMENTS], fontsize=12, color="white")
    ax.set_ylabel("Nombre de clips", color="white")
    ax.set_title("Distribution des classes — Train vs Val", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0D2D42", labelcolor="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#28AAE1")
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(int(bar.get_height())), ha="center", va="bottom",
                fontsize=9, color="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Distribution → {out_path}")


def viz_courbes(history: dict, out_path: str = "viz_courbes.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0B1F2E")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0D2D42")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#28AAE1")

    ax1.plot(epochs, history["train_loss"], color="#28AAE1", label="Train loss", linewidth=2)
    ax1.plot(epochs, history["val_loss"],   color="#00D4FF", label="Val loss",   linewidth=2, linestyle="--")
    if history.get("best_epoch"):
        ax1.axvline(history["best_epoch"], color="#f39c12", linestyle=":", linewidth=1.5,
                    label=f"Best epoch {history['best_epoch']}")
    ax1.set_title("Loss", color="white")
    ax1.set_xlabel("Époque", color="white")
    ax1.legend(facecolor="#0B1F2E", labelcolor="white")

    ax2.plot(epochs, history["train_acc"], color="#28AAE1", label="Train acc", linewidth=2)
    ax2.plot(epochs, history["val_acc"],   color="#00D4FF", label="Val acc",   linewidth=2, linestyle="--")
    ax2.axhline(0.70, color="#2ecc71", linestyle=":", linewidth=1.5, label="Objectif 70%")
    if history.get("best_epoch"):
        ax2.axvline(history["best_epoch"], color="#f39c12", linestyle=":", linewidth=1.5)
    ax2.set_title("Accuracy", color="white")
    ax2.set_xlabel("Époque", color="white")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax2.legend(facecolor="#0B1F2E", labelcolor="white")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"[viz] Courbes → {out_path}")


# ══════════════════════════════════════════════════════════════════════════
#  BOUCLES TRAIN / VAL
# ══════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for mel, labels in tqdm(loader, desc="  Train", leave=False, ncols=80):
        mel, labels = mel.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(mel)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(dim=1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for mel, labels in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            mel, labels = mel.to(device), labels.to(device)
            logits      = model(mel)
            loss        = criterion(logits, labels)
            total_loss += loss.item() * len(labels)
            correct    += (logits.argmax(dim=1) == labels).sum().item()
            total      += len(labels)
    return total_loss / total, correct / total


# ══════════════════════════════════════════════════════════════════════════
#  ÉVALUATION FINALE SUR LE TEST SET IRMAS
#
#  Structure TestData/ :
#    fichier.wav
#    fichier.txt  ← liste des instruments présents (parmi les 11 IRMAS)
#
#  Métrique : 1 si l'instrument prédit est dans la liste du .txt, 0 sinon
#  On prédit par fenêtres glissantes de 3s + vote majoritaire
# ══════════════════════════════════════════════════════════════════════════

def parse_test_labels(txt_path: Path) -> set[str]:
    """
    Lit le fichier .txt associé à un fichier test IRMAS.
    Retourne un set des instruments présents (ex: {'pia', 'voi'}).
    Seuls nos 4 instruments nous intéressent.
    """
    labels = set()
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            instrument = line.strip().lower()
            if instrument in LABEL_TO_IDX:
                labels.add(instrument)
    return labels


def predict_file(model, wav_path: Path, transform, device, window_s: float = 3.0) -> str:
    """
    Prédit l'instrument dominant dans un fichier de longueur variable.
    Découpe en fenêtres de window_s secondes, vote majoritaire.
    Retourne la clé de l'instrument prédit (ex: 'pia').
    """
    waveform, sr = librosa.load(str(wav_path), sr=SAMPLE_RATE, mono=True)
    waveform = torch.from_numpy(waveform).float().unsqueeze(0)
    waveform = rms_normalize(waveform)

    win_n  = int(window_s * SAMPLE_RATE)
    total  = waveform.shape[1]

    # Padding si trop court
    if total < win_n:
        pad      = torch.zeros(1, win_n - total)
        waveform = torch.cat([waveform, pad], dim=1)
        total    = win_n

    votes = {inst: 0 for inst in INSTRUMENTS}

    model.eval()
    with torch.no_grad():
        start = 0
        while start + win_n <= total:
            chunk  = waveform[:, start:start + win_n]
            mel    = transform(chunk).unsqueeze(0).to(device)   # (1, 1, 128, T)
            logits = model(mel).squeeze(0)                       # (4,)
            probs  = torch.sigmoid(logits)
            winner = INSTRUMENTS[int(probs.argmax())]
            votes[winner] += 1
            start += win_n                                       # pas = fenêtre (sans overlap)

    return max(votes, key=votes.get)


def evaluate_test_set(model, test_dir: str, transform, device) -> dict:
    """
    Évalue le modèle sur tout le test set IRMAS.
    Retourne un dict avec le score global et les détails par fichier.

    Métrique officielle :
        score = 1 si instrument prédit ∈ liste des instruments présents
        score = 0 sinon
    """
    test_root = Path(test_dir)
    wav_files = sorted(test_root.glob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"Aucun .wav dans : {test_root}")

    scores        = []
    details       = []
    per_class     = {inst: {"correct": 0, "total": 0} for inst in INSTRUMENTS}

    print(f"\n  Évaluation sur {len(wav_files)} fichiers test...")

    for wav_path in tqdm(wav_files, desc="  Test ", ncols=80):
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            # Pas de labels → on skip
            continue

        true_labels = parse_test_labels(txt_path)
        if not true_labels:
            # Aucun de nos 4 instruments dans ce fichier → skip
            continue

        predicted = predict_file(model, wav_path, transform, device)
        hit       = 1 if predicted in true_labels else 0
        scores.append(hit)

        details.append({
            "file":        wav_path.name,
            "predicted":   predicted,
            "true_labels": list(true_labels),
            "hit":         hit,
        })

        # Stats par classe (on compte sur le label prédit vs la vérité)
        for inst in true_labels:
            if inst in per_class:
                per_class[inst]["total"] += 1
                if predicted == inst:
                    per_class[inst]["correct"] += 1

    global_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "global_score":  round(global_score, 4),
        "n_evaluated":   len(scores),
        "n_correct":     sum(scores),
        "per_class":     per_class,
        "details":       details,
    }


# ══════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'═'*55}")
    print(f"  SoundID — Entraînement CNN IRMAS")
    print(f"  Modèle  : {MODEL_NAME}{MODEL_EXT}")
    print(f"  Device  : {device}")
    print(f"  Epochs  : {EPOCHS}  |  Batch : {BATCH_SIZE}")
    print(f"{'═'*55}\n")

    # ── Dossier Models ───────────────────────────────────────────────────
    models_dir = Path(MODELS_DIR)
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / f"{MODEL_NAME}{MODEL_EXT}"

    # ── Dataloaders ──────────────────────────────────────────────────────
    print("[1/5] Chargement du dataset...")
    train_loader, val_loader, stats = get_dataloaders(
        train_dir=TRAIN_DIR,
        batch_size=BATCH_SIZE,
        val_ratio=VAL_RATIO,
        seed=SEED,
        num_workers=0,
    )

    print(f"\n  {'Classe':<6}  {'Train':>6}  {'Val':>6}")
    print(f"  {'-'*22}")
    total_train, total_val = 0, 0
    for inst in INSTRUMENTS:
        t, v = stats[inst]["train"], stats[inst]["val"]
        print(f"  {inst.upper():<6}  {t:>6}  {v:>6}")
        total_train += t
        total_val   += v
    print(f"  {'TOTAL':<6}  {total_train:>6}  {total_val:>6}\n")

    # ── Visualisations initiales ─────────────────────────────────────────
    print("[2/5] Visualisations initiales...")
    viz_spectrogrammes()
    viz_distribution(stats)

    # ── Modèle ───────────────────────────────────────────────────────────
    print("\n[3/5] Initialisation du modèle...")
    model     = InstrumentCNN(n_classes=len(INSTRUMENTS), dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )
    print(f"  Paramètres entraînables : {model.count_params():,}\n")

    # ── Entraînement ─────────────────────────────────────────────────────
    print("[4/5] Entraînement...\n")

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "best_epoch": None, "best_val_loss": float("inf"),
    }
    best_val_loss    = float("inf")
    patience_counter = 0
    t_start          = time.time()

    for epoch in range(1, EPOCHS + 1):
        print(f"Époque {epoch:3d}/{EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))

        elapsed = time.time() - t_start
        print(
            f"  loss  train={train_loss:.4f}  val={val_loss:.4f}"
            f"  |  acc  train={train_acc:.1%}  val={val_acc:.1%}"
            f"  [{elapsed:.0f}s]"
        )

        if val_loss < best_val_loss:
            best_val_loss            = val_loss
            history["best_epoch"]    = epoch
            history["best_val_loss"] = round(best_val_loss, 4)
            patience_counter         = 0
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Meilleur modèle sauvegardé → {model_path}")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping à l'époque {epoch}.")
                break
            print(f"  patience {patience_counter}/{PATIENCE}")

        print()

    # ── Résumé entraînement ──────────────────────────────────────────────
    best_idx     = history["best_epoch"] - 1
    best_val_acc = history["val_acc"][best_idx]
    total_time   = time.time() - t_start

    print(f"\n{'═'*55}")
    print(f"  Entraînement terminé en {total_time:.0f}s")
    print(f"  Meilleure époque   : {history['best_epoch']}")
    print(f"  Meilleure val loss : {history['best_val_loss']:.4f}")
    print(f"  Meilleure val acc  : {best_val_acc:.1%}")
    print(f"{'═'*55}\n")

    viz_courbes(history)

    # ── Évaluation finale sur le test set ────────────────────────────────
    print("[5/5] Évaluation finale sur le test set IRMAS...")

    # Recharger le meilleur modèle sauvegardé
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    test_transform = build_val_transform()

    test_results = evaluate_test_set(model, TEST_DIR, test_transform, device)

    print(f"\n{'═'*55}")
    print(f"  RÉSULTATS TEST SET")
    print(f"{'─'*55}")
    print(f"  Score global  : {test_results['global_score']:.1%}"
          f"  ({test_results['n_correct']}/{test_results['n_evaluated']})")
    if test_results['global_score'] >= 0.70:
        print(f"  ✓ Objectif 70% ATTEINT !")
    else:
        print(f"  ✗ Objectif 70% non atteint")
    print(f"{'─'*55}")
    print(f"  Détail par classe (sur instances où cet instrument est présent) :")
    for inst, s in test_results["per_class"].items():
        if s["total"] > 0:
            acc = s["correct"] / s["total"]
            print(f"    {inst.upper()}  {s['correct']:3d}/{s['total']:3d}  ({acc:.1%})")
    print(f"{'═'*55}\n")

    # ── Sauvegarde log complet ───────────────────────────────────────────
    log = {
        "model_name":    MODEL_NAME,
        "model_path":    str(model_path),
        "device":        str(device),
        "epochs_run":    len(history["train_loss"]),
        "batch_size":    BATCH_SIZE,
        "lr":            LR,
        "weight_decay":  WEIGHT_DECAY,
        "n_params":      model.count_params(),
        "history":       history,
        "test_results":  {
            k: v for k, v in test_results.items() if k != "details"
        },
    }
    with open("training_log.json", "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print("[log] Historique + résultats test → training_log.json")


if __name__ == "__main__":
    main()