"""
model.py — Architecture CNN pour la reconnaissance d'instruments.

Architecture VGG-inspired avec améliorations modernes :
  - 4 blocs Conv2d doublés + BatchNorm + ReLU + MaxPool
  - AdaptiveAvgPool2d(4,4) pour gérer les longueurs variables
  - Dropout stratifié dans le classifieur
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Bloc de 2 convolutions avec BatchNorm, ReLU et MaxPool."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.block(x)


class InstrumentCNN(nn.Module):
    """
    CNN profond pour la classification d'instruments de musique.

    Input:  (batch, 1, n_mels, time)  — Log-Mel spectrogram
    Output: (batch, num_classes)      — logits pour chaque classe

    Architecture :
        Conv Block 1:  1  → 32  channels, /2 spatial
        Conv Block 2:  32 → 64  channels, /2 spatial
        Conv Block 3:  64 → 128 channels, /2 spatial
        Conv Block 4: 128 → 256 channels, /2 spatial
        AdaptiveAvgPool2d(4, 4)
        FC: 4096 → 512 → num_classes
    """

    def __init__(self, num_classes=4):
        super().__init__()

        # Feature extractor
        self.features = nn.Sequential(
            ConvBlock(1, 32),      # (1, 128, T) → (32, 64, T/2)
            ConvBlock(32, 64),     # (32, 64, T/2) → (64, 32, T/4)
            ConvBlock(64, 128),    # (64, 32, T/4) → (128, 16, T/8)
            ConvBlock(128, 256),   # (128, 16, T/8) → (256, 8, T/16)
        )

        # Pooling adaptatif pour gérer les longueurs variables
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifieur
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    """Compte le nombre total de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Sanity check
    model = InstrumentCNN(num_classes=4)
    print(f"Paramètres: {count_parameters(model):,}")

    # Test avec un batch
    x = torch.randn(2, 1, 128, 130)  # batch=2, 1 channel, 128 mels, 130 frames
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")

    # Test avec une entrée de taille différente (test set)
    x2 = torch.randn(1, 1, 128, 300)
    out2 = model(x2)
    print(f"Input:  {x2.shape}")
    print(f"Output: {out2.shape}")
