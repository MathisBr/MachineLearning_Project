"""
dataset.py — IRMAS Dataset & DataLoaders
=========================================
Structure attendue :
    TrainData/
        gac/  [gac][cla]0518__1.wav  [gac][cla]0518__2.wav  ...
        org/  [org][nod]0112__1.wav  ...
        pia/  [pia][cla]0001__1.wav  ...
        voi/  [voi][nod]0789__1.wav  ...

Deux structures internes :
    by_instrument : { "gac": ["0518", "0231", ...], "org": [...], ... }
    by_source     : { "gac_0518": [(Path, label), ...], ... }

Split : stratifié par morceau source, 75/25 par instrument.
        Tous les clips d'un même morceau restent du même côté.
        shuffle=True dans le DataLoader : ordre aléatoire à chaque époque.

Choix techniques retenus :
    - Log-Mel Spectrogram, 1 canal, 128 bandes, sr=16kHz
    - Normalisation RMS du volume (avant Mel)
    - Normalisation par instance mean=0 std=1 (après Mel)
    - SpecAugment en train uniquement (FreqMask=20, TimeMask=30)
    - Label depuis le sous-dossier (source fiable, pas le filename)
    - CrossEntropyLoss en train, Sigmoid en inférence

Usage :
    from dataset import get_dataloaders, INSTRUMENTS
    train_loader, val_loader, stats = get_dataloaders("../Dataset/TrainData")
"""

import random
import re
from pathlib import Path

import torch
import librosa
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ══════════════════════════════════════════════════════════════════════════
#  CONSTANTES
# ══════════════════════════════════════════════════════════════════════════

INSTRUMENTS  = ["gac", "org", "pia", "voi"]
LABEL_TO_IDX = {inst: i for i, inst in enumerate(INSTRUMENTS)}
IDX_TO_LABEL = {i: inst for i, inst in enumerate(INSTRUMENTS)}

SAMPLE_RATE      = 16_000
N_MELS           = 128
N_FFT            = 1024
HOP_LENGTH       = 512
FREQ_MASK_PARAM  = 20
TIME_MASK_PARAM  = 30

# Pattern : [gac][cla]0518__2.wav → "0518"
_SOURCE_RE = re.compile(r'(\d+)__\d+\.wav$')


# ══════════════════════════════════════════════════════════════════════════
#  EXTRACTION DE L'ID SOURCE
# ══════════════════════════════════════════════════════════════════════════

def extract_source_id(filename: str) -> str:
    """
    [gac][cla]0518__2.wav → '0518'
    Fallback : stem complet si pattern non trouvé.
    """
    m = _SOURCE_RE.search(filename)
    return m.group(1) if m else Path(filename).stem


# ══════════════════════════════════════════════════════════════════════════
#  CONSTRUCTION DES DEUX DICOS
# ══════════════════════════════════════════════════════════════════════════

def build_dicts(train_dir: str) -> tuple[dict, dict]:
    """
    Construit les deux structures depuis les sous-dossiers TrainData/.

    by_instrument : { "gac": ["0518", "0231", ...], ... }
        Une liste d'IDs sources uniques par instrument.
        Utilisation : savoir quels morceaux existent par classe,
                      pour faire le split par morceau.

    by_source : { "gac_0518": [(Path, 0), (Path, 0), ...], ... }
        Tous les clips d'un morceau, avec leur label int.
        Utilisation : récupérer les clips après le split.

    Le label vient du nom du sous-dossier (gac/ → 0) — plus fiable
    que parser le nom de fichier.
    """
    root          = Path(train_dir)
    by_instrument: dict[str, list[str]]               = {inst: [] for inst in INSTRUMENTS}
    by_source:     dict[str, list[tuple[Path, int]]]   = {}

    for inst in INSTRUMENTS:
        folder = root / inst
        if not folder.exists():
            raise FileNotFoundError(f"Dossier introuvable : {folder}")

        label    = LABEL_TO_IDX[inst]
        seen_ids: set[str] = set()

        for wav in sorted(folder.glob("*.wav")):
            source_id = extract_source_id(wav.name)
            key       = f"{inst}_{source_id}"

            # Enregistrer l'ID source une seule fois dans by_instrument
            if source_id not in seen_ids:
                by_instrument[inst].append(source_id)
                seen_ids.add(source_id)

            # Ajouter le clip dans by_source
            if key not in by_source:
                by_source[key] = []
            by_source[key].append((wav, label))

    for inst in INSTRUMENTS:
        if not by_instrument[inst]:
            raise ValueError(f"Aucun .wav trouvé dans : {root / inst}")

    return by_instrument, by_source


# ══════════════════════════════════════════════════════════════════════════
#  SPLIT STRATIFIÉ PAR MORCEAU SOURCE
# ══════════════════════════════════════════════════════════════════════════

def stratified_source_split(
    by_instrument: dict[str, list[str]],
    by_source:     dict[str, list[tuple[Path, int]]],
    val_ratio:     float = 0.25,
    seed:          int   = 42,
) -> tuple[list, list]:
    """
    Split 75/25 stratifié par instrument ET par morceau source.

    Pour chaque instrument :
      1. Prendre la liste de ses IDs sources (1 ID = 1 morceau entier)
      2. Shuffler ces IDs avec seed fixé (reproductible)
      3. Couper à val_ratio → val_ids, le reste → train_ids
      4. Récupérer tous les clips via by_source

    Garanties :
      - Équilibre des classes : chaque instrument contribue ~75/25
      - Anti-leakage : tous les clips d'un morceau restent du même côté
      - Reproductible : seed fixé une fois, split identique à chaque run
      - Aléatoire à chaque époque : géré par shuffle=True dans le DataLoader
                                    (pas ici — le split est fixé)
    """
    rng = random.Random(seed)

    train_files: list[tuple[Path, int]] = []
    val_files:   list[tuple[Path, int]] = []

    for inst in INSTRUMENTS:
        source_ids = by_instrument[inst].copy()
        rng.shuffle(source_ids)

        n_val     = max(1, int(len(source_ids) * val_ratio))
        val_ids   = source_ids[:n_val]
        train_ids = source_ids[n_val:]

        for sid in train_ids:
            train_files += by_source[f"{inst}_{sid}"]
        for sid in val_ids:
            val_files   += by_source[f"{inst}_{sid}"]

    return train_files, val_files


# ══════════════════════════════════════════════════════════════════════════
#  RMS NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
#  RMS NORMALIZATION & MEL PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def rms_normalize(waveform: torch.Tensor, target_rms: float = 0.1) -> torch.Tensor:
    """
    Normalise le volume à un RMS cible.
    Sans ça, le modèle peut apprendre la différence d'amplitude entre
    enregistrements plutôt que les caractéristiques de l'instrument.
    """
    rms = waveform.pow(2).mean().sqrt()
    if rms > 1e-8:
        waveform = waveform * (target_rms / rms)
    return waveform


def _base_mel_pipeline(
    waveform: torch.Tensor,
    mel_tf,
    amp_to_db,
) -> torch.Tensor:
    """
    Partie commune train + val :
    RMS norm → MelSpectrogram → AmplitudeToDB → normalisation par instance
    """
    waveform = rms_normalize(waveform)
    mel      = mel_tf(waveform)                        # (1, N_MELS, T)
    mel      = amp_to_db(mel)                          # log scale (dB)
    mel      = (mel - mel.mean()) / (mel.std() + 1e-6) # mean=0, std=1
    return mel                                         # (1, 128, T)


# ══════════════════════════════════════════════════════════════════════════
#  CUSTOM TRANSFORMS (librosa-based)
# ══════════════════════════════════════════════════════════════════════════

class MelSpectrogramTransform:
    def __init__(self, sr, n_fft, hop_length, n_mels):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
    
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, num_samples)
        wav = waveform.squeeze(0).numpy()
        mel = librosa.feature.melspectrogram(
            y=wav, sr=self.sr, n_fft=self.n_fft, 
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        return torch.from_numpy(mel).float().unsqueeze(0)  # (1, n_mels, time)

class AmplitudeToDBTransform:
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # Convertir en dB
        mel_db = 20 * torch.log10(torch.clamp(mel, min=1e-9))
        return mel_db

class FrequencyMaskingTransform:
    def __init__(self, freq_mask_param):
        self.freq_mask_param = freq_mask_param
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (1, n_mels, time)
        if random.random() < 0.5:  # Apply masking 50% of the time
            mask_width = random.randint(0, self.freq_mask_param)
            mask_start = random.randint(0, mel.shape[1] - mask_width) if mask_width > 0 else 0
            mel = mel.clone()
            mel[:, mask_start:mask_start+mask_width, :] = 0
        return mel

class TimeMaskingTransform:
    def __init__(self, time_mask_param):
        self.time_mask_param = time_mask_param
    
    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (1, n_mels, time)
        if random.random() < 0.5:  # Apply masking 50% of the time
            mask_width = random.randint(0, self.time_mask_param)
            mask_start = random.randint(0, mel.shape[2] - mask_width) if mask_width > 0 else 0
            mel = mel.clone()
            mel[:, :, mask_start:mask_start+mask_width] = 0
        return mel


# ══════════════════════════════════════════════════════════════════════════
#  TRANSFORMS BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def build_train_transform():
    """
    Pipeline train : base + SpecAugment.

    SpecAugment applique aléatoirement :
      - FrequencyMasking : efface jusqu'à 20 bandes de fréquences
      - TimeMasking      : efface jusqu'à 30 frames temporelles
    Force le modèle à ne pas surapprendre un pattern fréquentiel ou
    temporel fixe. Appliqué APRES la normalisation.
    """
    mel_tf    = MelSpectrogramTransform(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    amp_to_db = AmplitudeToDBTransform()
    freq_mask = FrequencyMaskingTransform(freq_mask_param=FREQ_MASK_PARAM)
    time_mask = TimeMaskingTransform(time_mask_param=TIME_MASK_PARAM)

    def transform(waveform: torch.Tensor) -> torch.Tensor:
        mel = _base_mel_pipeline(waveform, mel_tf, amp_to_db)
        mel = freq_mask(mel)
        mel = time_mask(mel)
        return mel

    return transform


def build_val_transform():
    """
    Pipeline val/test : base uniquement, sans SpecAugment.
    Reproductible : même entrée → même sortie à chaque appel.
    Utilisé aussi par eval.py et train.py pour le test set.
    """
    mel_tf    = MelSpectrogramTransform(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
    amp_to_db = AmplitudeToDBTransform()

    def transform(waveform: torch.Tensor) -> torch.Tensor:
        return _base_mel_pipeline(waveform, mel_tf, amp_to_db)

    return transform


# ══════════════════════════════════════════════════════════════════════════
#  DATASET PYTORCH
# ══════════════════════════════════════════════════════════════════════════

class IRMASDataset(Dataset):
    """
    Dataset PyTorch pour IRMAS.
    Charge chaque WAV à la volée et applique la transform.

    Label : entier 0-3 issu du sous-dossier (ground truth).
    Val   : le label sert à calculer l'accuracy sur la val.
    Test  : le label vient des .txt (géré dans train.py).
    """

    def __init__(self, files: list[tuple[Path, int]], transform):
        self.files     = files
        self.transform = transform

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.files[idx]

        waveform, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        waveform = torch.from_numpy(waveform).float().unsqueeze(0)

        return self.transform(waveform), label


# ══════════════════════════════════════════════════════════════════════════
#  STATS POUR VISUALISATION
# ══════════════════════════════════════════════════════════════════════════

def dataset_stats(
    by_instrument: dict,
    train_files:   list,
    val_files:     list,
) -> dict:
    """
    Stats détaillées par instrument pour les visualisations de train.py.
    """
    stats = {
        inst: {
            "train":   0,
            "val":     0,
            "sources": len(by_instrument[inst]),
        }
        for inst in INSTRUMENTS
    }
    for _, label in train_files:
        stats[IDX_TO_LABEL[label]]["train"] += 1
    for _, label in val_files:
        stats[IDX_TO_LABEL[label]]["val"] += 1
    return stats


# ══════════════════════════════════════════════════════════════════════════
#  FONCTION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════

def get_dataloaders(
    train_dir:   str,
    batch_size:  int   = 64,
    val_ratio:   float = 0.25,
    seed:        int   = 42,
    num_workers: int   = 0,
) -> tuple[DataLoader, DataLoader, dict]:
    """
    Pipeline complet → (train_loader, val_loader, stats).

    Flux :
      build_dicts()             → by_instrument + by_source
      stratified_source_split() → train_files + val_files
      IRMASDataset()            → avec transforms respectives
      DataLoader()              → shuffle=True en train (aléatoire / époque)
    """
    by_instrument, by_source = build_dicts(train_dir)
    train_files, val_files   = stratified_source_split(
        by_instrument, by_source, val_ratio, seed
    )
    stats = dataset_stats(by_instrument, train_files, val_files)

    pin_mem = torch.cuda.is_available()  # Activer pin_memory seulement si CUDA
    
    train_loader = DataLoader(
        IRMASDataset(train_files, build_train_transform()),
        batch_size=batch_size,
        shuffle=True,           # ordre aléatoire à chaque époque
        num_workers=num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        IRMASDataset(val_files, build_val_transform()),
        batch_size=batch_size,
        shuffle=False,          # reproductible pour la validation
        num_workers=num_workers,
        pin_memory=pin_mem,
    )

    return train_loader, val_loader, stats


# ══════════════════════════════════════════════════════════════════════════
#  QUICK CHECK  —  python dataset.py ../Dataset/TrainData
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # ── Device check ──
    print(f"\n{'='*60}")
    print(f"Device Info:")
    print(f"  CPU   : {torch.get_num_threads()} threads")
    print(f"  GPU   : {'CUDA available' if torch.cuda.is_available() else 'NO CUDA'}")
    if torch.cuda.is_available():
        print(f"          Device: {torch.cuda.get_device_name(0)}")
        print(f"          Count : {torch.cuda.device_count()}")
    print(f"{'='*60}\n")

    train_dir = sys.argv[1] if len(sys.argv) > 1 else "../Dataset/TrainData"
    print(f"Scan : {train_dir}\n")

    by_instrument, by_source = build_dicts(train_dir)

    print("Morceaux sources par instrument :")
    for inst in INSTRUMENTS:
        ids = by_instrument[inst]
        print(f"  {inst.upper():4s}  {len(ids):4d} morceaux  ex: {ids[:4]}")

    print(f"\nTotal clés by_source : {len(by_source)}")
    first_key = next(iter(by_source))
    print(f"Exemple by_source['{first_key}'] : {len(by_source[first_key])} clips")

    train_loader, val_loader, stats = get_dataloaders(train_dir, batch_size=4)

    print(f"\n{'Classe':<6}  {'Morceaux':>8}  {'Train clips':>12}  {'Val clips':>10}")
    print("─" * 44)
    for inst in INSTRUMENTS:
        s = stats[inst]
        print(f"{inst.upper():<6}  {s['sources']:>8}  {s['train']:>12}  {s['val']:>10}")

    mel, labels = next(iter(train_loader))
    print(f"\nBatch shape : {mel.shape}")
    print(f"Labels      : {[IDX_TO_LABEL[l.item()] for l in labels]}")
    print(f"Mel range   : [{mel.min():.3f}, {mel.max():.3f}]")
    print(f"Mel mean    : {mel.mean():.3f}  std : {mel.std():.3f}")
    print("\nOK")