"""
config.py — Hyperparamètres centralisés pour le pipeline IRMAS.

Détection automatique du device :
    1. CUDA (NVIDIA RTX)
    2. CPU (fallback)
"""

import os
import torch
from pathlib import Path

# ─── Chemins ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "Dataset"

TRAIN_DIRS = {
    "gac": DATA_DIR / "Train_gac" / "gac",
    "org": DATA_DIR / "Train_org" / "org",
    "pia": DATA_DIR / "Train_pia" / "pia",
    "voi": DATA_DIR / "Train_voi" / "voi",
}

TEST_DIR = DATA_DIR / "Test"
CACHE_DIR = PROJECT_ROOT / "cache"
MODEL_DIR = PROJECT_ROOT / "Models"

# ─── Classes ────────────────────────────────────────────────────────────────
CLASSES = ["gac", "org", "pia", "voi"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# ─── Audio ──────────────────────────────────────────────────────────────────
SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
F_MIN = 20
F_MAX = 8000

# ─── Training ──────────────────────────────────────────────────────────────
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
VAL_SPLIT = 0.15          # 15% pour la validation
EARLY_STOP_PATIENCE = 15
LABEL_SMOOTHING = 0.1
NUM_WORKERS = 0

# ─── Augmentation ──────────────────────────────────────────────────────────
FREQ_MASK_PARAM = 20      # SpecAugment : masquage fréquentiel
TIME_MASK_PARAM = 30      # SpecAugment : masquage temporel
MIXUP_ALPHA = 0.3         # Mixup : paramètre beta distribution

# ─── Scheduler ─────────────────────────────────────────────────────────────
COSINE_T0 = 10            # CosineAnnealingWarmRestarts : période initiale
COSINE_T_MULT = 2         # Multiplicateur de période

# ─── Device ────────────────────────────────────────────────────────────────
def get_device():
    """Détecte le meilleur device disponible : CUDA → CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] CUDA détecté : {torch.cuda.get_device_name(0)}")
        return device

    print("[Device] CPU uniquement")
    return torch.device("cpu")


DEVICE = get_device()
USE_AMP = DEVICE.type == "cuda"  # Mixed Precision uniquement sur CUDA


def print_config():
    """Affiche la configuration courante."""
    print("=" * 60)
    print("  Configuration IRMAS Pipeline")
    print("=" * 60)
    print(f"  Device        : {DEVICE}")
    print(f"  Mixed Prec.   : {USE_AMP}")
    print(f"  Batch size    : {BATCH_SIZE}")
    print(f"  Learning rate : {LEARNING_RATE}")
    print(f"  Epochs (max)  : {NUM_EPOCHS}")
    print(f"  N_MELS        : {N_MELS}")
    print(f"  N_FFT         : {N_FFT}")
    print(f"  Hop length    : {HOP_LENGTH}")
    print(f"  Val split     : {VAL_SPLIT}")
    print(f"  Mixup alpha   : {MIXUP_ALPHA}")
    print(f"  Label smooth. : {LABEL_SMOOTHING}")
    print(f"  Num workers   : {NUM_WORKERS}")
    print("=" * 60)
