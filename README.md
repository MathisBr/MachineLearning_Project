PROJET MACHINE LEARNING - RENDU

1) Participants
- Alystan GOUGEON--BRIHAULT
- Mathis BREVET
- Alexandra CULLMANN
- Lucas HARDY

2) Description du travail realise
Ce projet implemente une pipeline de classification d'instruments de musique sur IRMAS (classes: gac, org, pia, voi).

Contenu principal:
- Preprocessing audio vers spectrogrammes Mel log (dossier Code/preprocess.py)
- Entrainement d'un CNN PyTorch avec validations (Code/train.py, Code/model.py, Code/dataset.py)
- Evaluation du modele sur le jeu de test (Code/evaluate.py)
- Point d'entree pipeline (Code/main.py)
- API web de prediction et script d'inference (Web/api.py, Web/eval.py)

Organisation des donnees:
- Dataset/ : donnees source (train/test)
- cache/   : spectrogrammes pre-calcules (.pt)
- Models/  : checkpoints (ex: best_model.pt)

Execution (exemple):
1. Installer les dependances:
   pip install -r requirements.txt
2. Lancer la page web en une commande (ouvre le navigateur automatiquement):
   py launch_web.py
3. Lancer la pipeline complete:
   py Code/main.py
4. Variante evaluation seule:
   py Code/main.py --eval-only
5. Variante web API (manuel):
   cd Web
   uvicorn api:app --reload --port 5000

3) Utilisation de l'IA generative
- Utilisee: Oui
- But:
  - Recherche documentaire pour se renseigner sur l'etat de l'art (classification audio, CNN sur spectrogrammes Mel, bonnes pratiques d'evaluation).
  - Aide methodologique pour identifier la meilleure facon de repondre au probleme (choix pipeline preprocess/train/eval, metriques, validation).
  - Production assistee d'un script de recherche d'hyperparametres pour proposer des valeurs optimales dans Code/config.py.
  - la rédaction du code de esiea_instrument_detector.html a été entièrement produite par l'IA à partir des spécifications fournies.

4) Le script
Objectif:
Trouver automatiquement de meilleures valeurs pour les hyperparametres definis dans Code/config.py (ex: LEARNING_RATE, WEIGHT_DECAY, BATCH_SIZE, MIXUP_ALPHA, LABEL_SMOOTHING, COSINE_T0, COSINE_T_MULT).

Principe general:
1. Un espace de recherche a ete defini pour les hyperparametres principaux.
2. Des essais automatiques ont ete executes (plusieurs configurations testees).
3. Pour chaque essai, le modele a ete entraine puis evalue sur validation.
4. Les resultats ont ete compares a partir des metriques obtenues.
5. Les meilleurs reglages identifies ont servi a fixer les valeurs retenues dans Code/config.py.

Trace de l'utilisation IA sur cette partie:
L'IA a ete utilisee pour aider a structurer ce script de recherche (logique d'exploration, suivi des essais, et comparaison des metriques) afin d'identifier plus rapidement des parametres performants.
