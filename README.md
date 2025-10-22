# Exercice 2 : Rapports avec visualisations

Ce projet est la correction de l'exercice 2, qui étend le pipeline CML de l'exercice 1 en y ajoutant des rapports visuels.

## Objectif

L'objectif est d'enrichir le rapport CML en y incluant des graphiques générés après l'entraînement du modèle, tels que la matrice de confusion et l'importance des features.

## Structure du projet

La structure du projet de l'exercice 1 est étendue avec les éléments suivants :

```
.
├── .github/
│   └── workflows/
│       └── cml.yml          # Workflow mis à jour
├── models/
│   └── iris_model.pkl
├── reports/
│   ├── confusion_matrix.png # Graphique généré
│   └── feature_importance.png # Graphique généré
├── train.py                 # Script d'entraînement (inchangé)
├── evaluate.py              # NOUVEAU: Script de génération des graphiques
├── requirements.txt         # Mis à jour avec matplotlib et seaborn
├── metrics.json
└── README.md
```

## Nouveautés de l'Exercice 2

### 1. `evaluate.py`

Ce nouveau script est responsable de la génération des visualisations :

- **Charge** le modèle `iris_model.pkl` précédemment entraîné.
- **Génère et sauvegarde** une image de la **matrice de confusion** dans `reports/confusion_matrix.png`.
- **Génère et sauvegarde** un graphique de l'**importance des features** dans `reports/feature_importance.png`.

### 2. `requirements.txt`

Le fichier a été mis à jour pour inclure les bibliothèques de visualisation :

- `matplotlib`
- `seaborn`

### 3. `.github/workflows/cml.yml`

Le workflow a été modifié pour intégrer la nouvelle étape d'évaluation :

- **Nouvelle étape `Generate visualizations`** : Exécute le script `evaluate.py` après l'entraînement.
- **Rapport CML enrichi** : Le rapport inclut maintenant les métriques JSON ainsi que les deux images générées (matrice de confusion et importance des features) en utilisant la commande `cml publish`.

## Résultat attendu

Lorsqu'une Pull Request est créée, le commentaire CML généré sera beaucoup plus riche. Il contiendra :

- Les métriques de performance (accuracy, etc.) au format JSON.
- L'image de la matrice de confusion.
- L'image du graphique d'importance des features.

Cela permet une analyse beaucoup plus rapide et visuelle de la performance du modèle directement depuis l'interface de GitHub.

## Instructions d'utilisation

Le processus est similaire à celui de l'exercice 1. Après avoir poussé le code et créé une Pull Request, le workflow s'exécutera et publiera le rapport enrichi.

### Test local

Pour tester l'ensemble du processus localement :

```bash
# Installer les nouvelles dépendances
pip install -r requirements.txt

# Étape 1: Entraîner le modèle
python train.py

# Étape 2: Générer les visualisations
python evaluate.py

# Vérifier les fichiers générés
ls -l reports/
```

Vous devriez voir `confusion_matrix.png` et `feature_importance.png` dans le dossier `reports/`.
