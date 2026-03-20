"""
model.py — Architecture CNN pour la classification d'instruments.

Choix architecturaux justifiés par la littérature :
  - Convolutions 2D sur Log-Mel spectrogramme (Chen et al. 2024, meilleure performance)
  - BatchNorm2d entre conv et activation (réduit le sur-apprentissage sur IRMAS)
  - Dropout 0.4 (optimal selon Stanford CS230 sur IRMAS)
  - AdaptiveAvgPool2d pour gérer les entrées de taille variable (fichiers test)
  - Modèle compact : 4 classes seulement, dataset de taille modeste → éviter
    un modèle trop grand qui mémoriserait les données d'entraînement

Architecture :
  Conv Block 1 : Conv2d(1→32, 3×3)   → BN → ReLU → MaxPool2d(2×2)
  Conv Block 2 : Conv2d(32→64, 3×3)  → BN → ReLU → MaxPool2d(2×2)
  Conv Block 3 : Conv2d(64→128, 3×3) → BN → ReLU → MaxPool2d(2×2)
  AdaptiveAvgPool2d(4, 4)             → Flatten : 128 × 4 × 4 = 2048 features
  Dropout(0.4)
  FC(2048 → 256) → ReLU → Dropout(0.4)
  FC(256 → NUM_CLASSES)
"""

import torch
import torch.nn as nn

import config


class ConvBlock(nn.Module):
    """
    Bloc convolutionnel : Conv2d → BatchNorm2d → ReLU → MaxPool2d.
    La BatchNorm est placée AVANT l'activation (recommandation du sujet, section 3.3).
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: tuple = (2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class InstrumentCNN(nn.Module):
    """
    CNN compact pour la reconnaissance d'instruments de musique.
    Supporte des entrées de tailles variables grâce à AdaptiveAvgPool2d.
    """

    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        dropout_rate: float = config.DROPOUT_RATE,
    ):
        super().__init__()

        # ── Partie convolutionnelle ──────────────────────────────────────────
        self.conv_layers = nn.Sequential(
            ConvBlock(in_channels=1,   out_channels=32),   # (1, 64, T) → (32, 32, T//2)
            ConvBlock(in_channels=32,  out_channels=64),   # → (64, 16, T//4)
            ConvBlock(in_channels=64,  out_channels=128),  # → (128, 8, T//8)
        )

        # ── Pooling adaptatif ────────────────────────────────────────────────
        # Fixe la sortie à (128, 4, 4) quelle que soit la taille temporelle de l'entrée.
        # Indispensable pour les fichiers test à longueur variable.
        self.adaptive_pool = nn.AdaptiveMaxPool2d(
            (config.ADAPTIVE_POOL_H, config.ADAPTIVE_POOL_W)
        )

        # ── Partie dense ─────────────────────────────────────────────────────
        flatten_size = 128 * config.ADAPTIVE_POOL_H * config.ADAPTIVE_POOL_W

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(flatten_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tenseur de forme (B, 1, N_MELS, T).

        Returns:
            Logits de forme (B, NUM_CLASSES).
        """
        features = self.conv_layers(x)
        pooled   = self.adaptive_pool(features)
        logits   = self.classifier(pooled)
        return logits

    def count_parameters(self) -> int:
        """Retourne le nombre total de paramètres entraînables."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model() -> InstrumentCNN:
    """Instancie le modèle et l'affiche avec le nombre de paramètres."""
    model = InstrumentCNN(dropout_rate=config.DROPOUT_RATE)
    model = model.to(config.DEVICE)
    print(f"Modèle : InstrumentCNN | Paramètres : {model.count_parameters():,}")
    return model
