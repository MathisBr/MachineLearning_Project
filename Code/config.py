"""
config.py — Hyperparamètres centralisés du projet.
Modifier ce fichier pour ajuster les paramètres d'entraînement.
"""

import torch
from pathlib import Path

# ── Chemins ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR    = PROJECT_ROOT / "Dataset"      # Sous-dossiers : gac/, org/, pia/, voi/
TEST_DIR     = PROJECT_ROOT / "Dataset" / "Test"  # Fichiers .wav + fichiers d'annotations
NPY_DIR      = PROJECT_ROOT / "Dataset_npy"  # Version pré-convertie en .npy (optionnelle)
CACHE_DIR    = PROJECT_ROOT / "data" / "cache"    # Spectrogrammes pré-calculés (.pt)
MODEL_PATH   = PROJECT_ROOT / "best_model.pt"      # Meilleur modèle sauvegardé

# ── Classes ──────────────────────────────────────────────────────────────────
INSTRUMENT_CLASSES = ["gac", "org", "pia", "voi"]
NUM_CLASSES        = len(INSTRUMENT_CLASSES)
CLASS_TO_IDX       = {cls: i for i, cls in enumerate(INSTRUMENT_CLASSES)}
IDX_TO_CLASS       = {i: cls for i, cls in enumerate(INSTRUMENT_CLASSES)}

# ── Audio ────────────────────────────────────────────────────────────────────
SAMPLE_RATE   = 22050   # Hz — taux standard pour la musique
TRAIN_SAMPLES = 66150   # 3 secondes × 22050 Hz (durée fixe pour le train)

# ── Mel Spectrogram ───────────────────────────────────────────────────────────
# Choix justifié : Log-Mel spectrogramme = meilleur rapport performance/complexité
# selon la littérature IRMAS (Chen et al. 2024, Castel-Branco et al. 2020).
N_FFT       = 1024   # Fenêtre FFT — bonne résolution fréquentielle pour la musique
HOP_LENGTH  = 512    # Décalage = 50% overlap — standard pour l'audio musical
N_MELS      = 64     # Bandes Mel — compromis taille/expressivité pour 4 classes
F_MIN       = 20.0   # Fréquence min : limite basse de l'audition humaine (20 Hz)
F_MAX = 11025.0  # = Nyquist pour 22050 Hz
# ── SpecAugment (data augmentation) ──────────────────────────────────────────
# Recommandé par le sujet pour combattre le sur-apprentissage.
FREQ_MASK_PARAM = 8   # Masque max sur l'axe fréquentiel (sur 64 bandes)
TIME_MASK_PARAM = 30  # Masque max sur l'axe temporel

# ── Entraînement ─────────────────────────────────────────────────────────────
VALIDATION_SPLIT = 0.25   # 75% train / 25% validation (recommandé par le sujet)
BATCH_SIZE       = 32     # Optimal d'après Stanford CS230 sur IRMAS
LEARNING_RATE    = 0.001 # Adam — taux initial standard
WEIGHT_DECAY     = 0.0001   # Régularisation L2 légère
NUM_EPOCHS       = 50
PATIENCE         = 10     # Early stopping : arrêt si pas d'amélioration après N epochs
LR_PATIENCE      = 5      # ReduceLROnPlateau : réduction du LR après N epochs stagnants
LR_FACTOR        = 0.5    # Facteur de réduction du LR

# ── Modèle ────────────────────────────────────────────────────────────────────
DROPOUT_RATE     = 0.4    # Optimal selon la littérature (0.25–0.5)
ADAPTIVE_POOL_H  = 4      # Taille de sortie de l'AdaptiveAvgPool2d
ADAPTIVE_POOL_W  = 4      # → tenseur (C, 4, 4) avant flatten

# ── Système ──────────────────────────────────────────────────────────────────
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0    # 0 est recommandé sur Windows pour les petits fichiers (.pt)
RANDOM_SEED = 42
