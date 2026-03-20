"""
dataset.py — Datasets PyTorch (train/val/test) + DataLoaders.

Inclut :
  - Split stratifié train/val
  - Augmentation SpecAugment (FrequencyMasking + TimeMasking) — train only
  - Mixup intégré dans le collate_fn
  - Weighted RandomSampler pour compenser le déséquilibre des classes
  - TestDataset pour fichiers de longueur variable avec annotations multi-label
"""

import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path

import config


# ─── Augmentation SpecAugment ──────────────────────────────────────────────

class SpecAugment:
    """Applique FrequencyMasking + TimeMasking sur un spectrogramme."""

    def __init__(self, freq_mask=config.FREQ_MASK_PARAM,
                 time_mask=config.TIME_MASK_PARAM):
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask)

    def __call__(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec


# ─── Train/Val Dataset ─────────────────────────────────────────────────────

class IRMASTrainDataset(Dataset):
    """
    Dataset pour l'entraînement/validation.
    Charge les spectrogrammes pré-calculés depuis le cache.
    """

    def __init__(self, file_paths, labels, augment=False):
        """
        Args:
            file_paths: liste de chemins vers les fichiers .pt
            labels: liste des indices de classe correspondants
            augment: si True, applique SpecAugment
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.spec_augment = SpecAugment() if augment else None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        spec = torch.load(self.file_paths[idx], weights_only=True)

        if self.augment and self.spec_augment is not None:
            spec = self.spec_augment(spec)

        label = self.labels[idx]
        return spec, label


# ─── Test Dataset ──────────────────────────────────────────────────────────

class IRMASTestDataset(Dataset):
    """
    Dataset pour le test.
    Charge les spectrogrammes pré-calculés et les annotations multi-label.
    batch_size=1 obligatoire car longueurs variables.
    """

    def __init__(self):
        self.cache_dir = config.CACHE_DIR / "test"
        self.test_dir = config.TEST_DIR

        # Lister les fichiers de test (un .pt = un .wav)
        self.files = sorted(self.cache_dir.glob("*.pt"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        pt_path = self.files[idx]
        spec = torch.load(pt_path, weights_only=True)

        # Charger les annotations depuis le .txt correspondant
        txt_path = self.test_dir / (pt_path.stem + ".txt")
        annotations = self._load_annotations(txt_path)

        return spec, annotations, pt_path.stem

    @staticmethod
    def _load_annotations(txt_path):
        """
        Charge les annotations multi-label depuis un fichier .txt.
        Format : une ligne par instrument (e.g. 'pia\\t\\n')
        
        Returns:
            set des instruments présents parmi {gac, org, pia, voi}
        """
        annotations = set()
        if txt_path.exists():
            with open(txt_path, "r") as f:
                for line in f:
                    instr = line.strip().replace("\t", "")
                    if instr in config.CLASS_TO_IDX:
                        annotations.add(instr)
        return annotations


# ─── Collate function avec Mixup ──────────────────────────────────────────

def mixup_collate_fn(batch, alpha=config.MIXUP_ALPHA):
    """
    Collate function avec Mixup pour le training.
    Mélange les paires d'exemples avec un ratio lambda ~ Beta(alpha, alpha).
    """
    specs, labels = zip(*batch)

    # Pad les spectrogrammes à la même taille temporelle dans le batch
    max_time = max(s.shape[-1] for s in specs)
    padded_specs = []
    for s in specs:
        if s.shape[-1] < max_time:
            pad_size = max_time - s.shape[-1]
            s = torch.nn.functional.pad(s, (0, pad_size))
        padded_specs.append(s)

    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Assurer que lam >= 0.5

        batch_size = specs_tensor.size(0)
        index = torch.randperm(batch_size)

        mixed_specs = lam * specs_tensor + (1 - lam) * specs_tensor[index]
        labels_a = labels_tensor
        labels_b = labels_tensor[index]

        return mixed_specs, labels_a, labels_b, lam
    else:
        return specs_tensor, labels_tensor, labels_tensor, 1.0


def standard_collate_fn(batch):
    """Collate function standard (sans Mixup) pour la validation."""
    specs, labels = zip(*batch)

    max_time = max(s.shape[-1] for s in specs)
    padded_specs = []
    for s in specs:
        if s.shape[-1] < max_time:
            pad_size = max_time - s.shape[-1]
            s = torch.nn.functional.pad(s, (0, pad_size))
        padded_specs.append(s)

    specs_tensor = torch.stack(padded_specs)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    return specs_tensor, labels_tensor


# ─── Fonctions de création des DataLoaders ─────────────────────────────────

def get_train_val_datasets():
    """
    Crée les datasets train et validation à partir du cache.
    Split stratifié 85/15.
    
    Returns:
        (train_dataset, val_dataset)
    """
    all_paths = []
    all_labels = []

    for class_name in config.CLASSES:
        class_cache_dir = config.CACHE_DIR / "train" / class_name
        pt_files = sorted(class_cache_dir.glob("*.pt"))

        for pt_path in pt_files:
            all_paths.append(pt_path)
            all_labels.append(config.CLASS_TO_IDX[class_name])

    # Split stratifié
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels,
        test_size=config.VAL_SPLIT,
        stratify=all_labels,
        random_state=42,
    )

    train_dataset = IRMASTrainDataset(train_paths, train_labels, augment=True)
    val_dataset = IRMASTrainDataset(val_paths, val_labels, augment=False)

    print(f"[Dataset] Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_dataset, val_dataset


def get_train_val_loaders():
    """
    Crée les DataLoaders train et validation.
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset, val_dataset = get_train_val_datasets()

    # WeightedRandomSampler pour compenser le déséquilibre
    labels = train_dataset.labels
    class_counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    class_weights = 1.0 / (class_counts + 1e-6)
    sample_weights = [class_weights[label] for label in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=mixup_collate_fn,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        collate_fn=standard_collate_fn,
        persistent_workers=config.NUM_WORKERS > 0,
    )

    return train_loader, val_loader


def get_test_loader():
    """
    Crée le DataLoader pour le test (batch_size=1, pas d'augmentation).
    
    Returns:
        test_loader
    """
    test_dataset = IRMASTestDataset()
    print(f"[Dataset] Test: {len(test_dataset)}")

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # Pas de multiprocessing pour le test
    )

    return test_loader
