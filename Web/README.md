# 🎵 SoundID
### Reconnaissance automatique d'instruments · INF3043 · ESIEA

> Déposez un fichier audio ou enregistrez depuis votre micro.  
> Un CNN PyTorch identifie l'instrument dominant en quelques secondes.

**Classes reconnues** — guitare acoustique `gac` · orgue `org` · piano `pia` · voix humaine `voi`

---

## Vue d'ensemble

SoundID est une application web fullstack composée de trois éléments indépendants :

| Fichier | Rôle |
|---------|------|
| `esiea_instrument_detector.html` | Interface SPA — zéro dépendance, s'ouvre directement dans le navigateur |
| `api.py` | Relay FastAPI — reçoit le POST, appelle `eval.py`, renvoie le JSON |
| `eval.py` | Inférence pure — charge le `.pth`, découpe l'audio, vote, sort le résultat |

```
Navigateur  ──POST /predict──▶  api.py  ──subprocess──▶  eval.py
                                                              │
                                                         charge .pth
                                                         fenêtres 3s
                                                         vote majoritaire
                                                              │
Navigateur  ◀───────JSON────────api.py  ◀──────stdout────────┘
```

---

## Setup

```bash
# 1 — Environnement virtuel
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate

# 2 — Dépendances
pip install -r requirements.txt

# 3 — Lancer l'API
uvicorn api:app --reload --port 5000
```

Placer les poids entraînés dans :
```
../Models/
    LilDrill.pth
    Brrrevet.pth
    Silver.pth
    MonkI.pth
```

Ouvrir `esiea_instrument_detector.html` dans le navigateur, onglet **Analyse** → renseigner `http://localhost:5000` → **Tester →**.

---

## Fonctionnalités de l'API

### `GET /health`

Vérifie que le serveur est vivant. Utilisé par le bouton **Tester →** de l'interface.

```json
{ "status": "ok" }
```

---

### `POST /predict`

Point d'entrée principal. Accepte un fichier audio et des paramètres d'inférence, retourne les scores par instrument.

**Corps de la requête** — `multipart/form-data`

| Champ | Type | Valeurs acceptées | Défaut |
|-------|------|-------------------|--------|
| `audio` | File | `.wav` `.mp3` `.ogg` `.flac` | — |
| `model` | string | `LilDrill` `Brrrevet` `Silver` `MonkI` | `LilDrill` |
| `binary_mode` | string | `"true"` `"false"` | `"false"` |

**Réponse** — `application/json`

```json
{
  "predicted_instrument": "pia",
  "confidence": 0.84,
  "scores": {
    "gac": 0.05,
    "org": 0.07,
    "pia": 0.84,
    "voi": 0.04
  },
  "metadata": {
    "model_name":      "Silver",
    "binary_mode":     false,
    "n_windows":       7,
    "duration_s":      19.2,
    "processed_in_ms": 312.4
  }
}
```

**Pipeline interne de `eval.py`**

1. Chargement du `.pth` depuis `../Models/<model_name>.pth`
2. Resample → 16 kHz mono
3. Découpage en fenêtres de **3 secondes** (overlap 50%) — padding silence si trop court
4. Pour chaque fenêtre : Log-Mel Spectrogram → normalisation → CNN → logits
5. **Vote majoritaire** sur les fenêtres → instrument prédit
6. Moyenne des logits → Sigmoid → scores continus pour le barplot
7. JSON imprimé sur `stdout`, capturé par `api.py`

**Mode binaire** — ne modifie pas l'inférence. Formate uniquement la sortie : instrument prédit = `1.0`, tous les autres = `0.0`.

**Gestion des erreurs**

| Code HTTP | Cause |
|-----------|-------|
| `400` | Modèle inconnu dans le champ `model` |
| `422` | Fichier audio illisible ou corrompu |
| `500` | `eval.py` a planté — détail dans le body |
| `504` | `eval.py` a dépassé le timeout de 60s |

---

## Utilisation de l'IA générative

Ce projet a utilisé **Claude (Anthropic)** tout au long du développement.

L'approche a été **très guidée** : chaque choix technique (architecture CNN, stratégie de fenêtrage, split anti-leakage, choix de FastAPI, structure du contrat API) a fait l'objet d'échanges et de validation avant implémentation. Claude a servi de sparring partner pour challenger les décisions — notamment sur les pièges du dataset IRMAS (data leakage inter-clips, domain shift train/test) et sur les stratégies d'agrégation.

En revanche, la **rédaction du code** de `esiea_instrument_detector.html` et de `api.py` / `eval.py` a été entièrement produite par l'IA à partir des spécifications fournies, sans copier-coller de sources extérieures.

> Sections concernées : `esiea_instrument_detector.html` (intégralité), `api.py` (intégralité), `eval.py` (intégralité), `README.md` (intégralité).
