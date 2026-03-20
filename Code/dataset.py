"""
dataset.py — Datasets PyTorch pour l'entraînement, la validation et le test.

Les spectrogrammes sont chargés depuis le cache (.pt) pré-calculé par preprocess.py.
L'augmentation SpecAugment est appliquée uniquement pendant l'entraînement.
"""

import json
import random
import torch
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path

import config


LABEL_ALIASES = {
    # IRMAS peut fournir "gel" (guitare electrique) alors que le train utilise "gac".
    "gel": "gac",
}


def normalize_instrument_label(raw_label: str) -> str:
    """Normalise un label d'annotation test vers l'espace des classes du modele."""
    label = raw_label.strip().lower()
    return LABEL_ALIASES.get(label, label)


class SpecAugment(torch.nn.Module):
    """
    Augmentation SpecAugment (Park et al., 2019) via torchaudio.
    Applique des masques aléatoires sur les axes fréquentiel et temporel.
    Recommandé par le sujet pour améliorer la généralisation.
    """

    def __init__(self):
        super().__init__()
        self.freq_masking = T.FrequencyMasking(freq_mask_param=config.FREQ_MASK_PARAM)
        self.time_masking = T.TimeMasking(time_mask_param=config.TIME_MASK_PARAM)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        spectrogram = self.freq_masking(spectrogram)
        spectrogram = self.time_masking(spectrogram)
        return spectrogram


class InstrumentTrainDataset(Dataset):
    """
    Dataset pour l'entraînement et la validation.
    Charge les spectrogrammes depuis le cache et applique l'augmentation
    SpecAugment si `augment=True`.
    """

    def __init__(self, metadata: list[dict], augment: bool = False):
        """
        Args:
            metadata: Liste de dicts avec clés 'cache_path' et 'label_idx'.
            augment:  Active SpecAugment (uniquement pour le sous-ensemble train).
        """
        self.metadata = metadata
        self.augment  = augment
        self.spec_augment = SpecAugment() if augment else None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        item      = self.metadata[index]
        log_mel   = torch.load(item["cache_path"], weights_only=True)  # (1, N_MELS, T)
        label_idx = item["label_idx"]

        if self.augment and self.spec_augment is not None:
            log_mel = self.spec_augment(log_mel)

        return log_mel, label_idx


class InstrumentTestDataset(Dataset):
    """
    Dataset pour l'évaluation sur l'ensemble de test.
    Les fichiers ont des longueurs variables → pas de padding.
    Le modèle utilise AdaptiveAvgPool2d pour absorber la variabilité.
    """

    def __init__(self, metadata: list[dict]):
        """
        Args:
            metadata: Liste de dicts avec clés 'cache_path' et 'annotation_path'.
        """
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[str]]:
        item    = self.metadata[index]
        log_mel = torch.load(item["cache_path"], weights_only=True)  # (1, N_MELS, T)

        # Lecture des instruments présents dans le fichier d'annotation
        instruments_present = []
        annotation_path = item.get("annotation_path")
        if annotation_path and Path(annotation_path).exists():
            with open(annotation_path, "r") as f:
                for line in f:
                    instrument = normalize_instrument_label(line)
                    if instrument:
                        instruments_present.append(instrument)

        return log_mel, instruments_present


def build_train_val_loaders(
    metadata_path: str,
) -> tuple[DataLoader, DataLoader]:
    """
    Construit les DataLoaders d'entraînement et de validation.
    Réalise un split 75/25 aléatoire et reproductible.

    Args:
        metadata_path: Chemin vers train_metadata.json.

    Returns:
        (train_loader, val_loader)
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    if len(metadata) == 0:
        raise ValueError(
            "Aucun exemple d'entraînement trouvé dans les métadonnées. "
            "Vérifiez les chemins TRAIN_DIR/NPY_DIR et relancez le pré-traitement."
        )

    # Mélange reproductible avant split
    random.seed(config.RANDOM_SEED)
    random.shuffle(metadata)

    split_idx    = int(len(metadata) * (1 - config.VALIDATION_SPLIT))
    train_meta   = metadata[:split_idx]
    val_meta     = metadata[split_idx:]

    train_counts = torch.tensor([637.0, 682.0, 721.0, 778.0])
    sample_weights = []
    for item in train_meta:
        label = item["label_idx"]
        sample_weights.append(1.0 / train_counts[label].item())

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_dataset = InstrumentTrainDataset(train_meta, augment=True)
    val_dataset   = InstrumentTrainDataset(val_meta,   augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE.type == "cuda"),
    )

    print(f"Train : {len(train_dataset)} exemples | Val : {len(val_dataset)} exemples")
    return train_loader, val_loader


def build_test_loader(metadata_path: str) -> DataLoader:
    """
    Construit le DataLoader de test.
    batch_size=1 obligatoire car les spectrogrammes ont des tailles variables.

    Args:
        metadata_path: Chemin vers test_metadata.json.

    Returns:
        test_loader avec batch_size=1.
    """
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    test_dataset = InstrumentTestDataset(metadata)

    # batch_size=1 car les tenseurs de test ont des dimensions temporelles différentes
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 0 workers pour éviter les problèmes de pickling
        # Evite que le collate par defaut transforme list[str] en structure imbriquee.
        collate_fn=lambda batch: batch[0],
    )

    print(f"Test  : {len(test_dataset)} exemples")
    return test_loader
