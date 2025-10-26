# Guide de Démarrage Rapide - RegNetY Breast Cancer Detection

Ce guide vous aidera à démarrer rapidement avec le projet de détection du cancer du sein.

## 📦 Installation Rapide

```bash
# 1. Cloner le repository
git clone https://github.com/MohAitMesskine/detection-cancer-sein-regnet.git
cd detection-cancer-sein-regnet

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

## 📊 Préparation des Données

### Option 1 : Dataset déjà organisé en dossiers

Si vos données sont déjà organisées par classe :

```bash
# Diviser en train/val/test
python src/prepare_data.py split \
    --source_dir /path/to/organized/data \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### Option 2 : Dataset plat avec fichier CSV

Si toutes les images sont dans un seul dossier :

```bash
# Organiser par classe
python src/prepare_data.py organize \
    --source_dir /path/to/flat/images \
    --output_dir data/organized \
    --labels_file labels.csv

# Puis diviser
python src/prepare_data.py split \
    --source_dir data/organized \
    --output_dir data
```

### Vérifier la structure

```bash
python src/prepare_data.py verify --data_dir data
```

Structure attendue :
```
data/
├── train/
│   ├── benign/
│   └── malignant/
├── val/
│   ├── benign/
│   └── malignant/
└── test/
    ├── benign/
    └── malignant/
```

## 🎓 Entraînement Rapide

### Entraînement de base (recommandé pour débuter)

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_400mf \
    --epochs 50 \
    --batch_size 32
```

### Entraînement avec GPU

```bash
# PyTorch utilisera automatiquement CUDA s'il est disponible
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_800mf \
    --epochs 100 \
    --batch_size 16
```

### Entraînement sur CPU (plus lent)

```bash
# Réduire batch_size et epochs
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_400mf \
    --epochs 20 \
    --batch_size 8
```

## 📈 Évaluation

```bash
python src/evaluate.py \
    --data_dir data/test \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf \
    --output_dir results/evaluation
```

Résultats générés :
- `evaluation_metrics.json` - Métriques de performance
- `confusion_matrix.png` - Matrice de confusion
- `roc_curve.png` - Courbe ROC

## 🔮 Prédiction

### Sur une image

```bash
python src/predict.py \
    --input test_image.png \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

### Sur un dossier

```bash
python src/predict.py \
    --input /path/to/images/ \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

## 🎨 Choix du Modèle

| Modèle | Cas d'Usage | Commande |
|--------|-------------|----------|
| regnet_y_400mf | GPU limité, petit dataset | `--model_name regnet_y_400mf` |
| regnet_y_800mf | Bon équilibre | `--model_name regnet_y_800mf` |
| regnet_y_1_6gf | Haute précision | `--model_name regnet_y_1_6gf` |
| regnet_y_3_2gf | GPU puissant | `--model_name regnet_y_3_2gf` |

## ⚙️ Paramètres Courants

### Batch Size
- **32** : Standard, bon équilibre
- **16** : Si erreur de mémoire GPU
- **8** : CPU ou GPU très limité
- **64** : GPU puissant (RTX 3090+)

### Learning Rate
- **0.001** : Défaut, bon pour la plupart des cas
- **0.0001** : Fine-tuning plus précis
- **0.01** : Convergence rapide (risque d'instabilité)

### Epochs
- **20-30** : Tests rapides
- **50** : Standard
- **100** : Entraînement complet
- **200+** : Fine-tuning maximal

## 🐛 Problèmes Courants

### Erreur : CUDA out of memory

```bash
# Solution 1 : Réduire batch size
python src/train.py --batch_size 16

# Solution 2 : Utiliser CPU
export CUDA_VISIBLE_DEVICES=""
python src/train.py
```

### Erreur : No module named 'torch'

```bash
# Réinstaller PyTorch
pip install torch torchvision
```

### Dataset non trouvé

```bash
# Vérifier la structure
python src/prepare_data.py verify --data_dir data

# Vérifier les chemins
ls -la data/train
ls -la data/val
```

### Modèle ne converge pas

```bash
# Réduire learning rate
python src/train.py --learning_rate 0.0001

# Augmenter dropout
python src/train.py --dropout_rate 0.6
```

## 📊 Suivi de l'Entraînement

Pendant l'entraînement, vous verrez :
```
Epoch 1 [Train]: 100%|████| 50/50 [00:30<00:00, loss: 0.4523, acc: 78.50%]
Epoch 1 [Val]:   100%|████| 10/10 [00:05<00:00, loss: 0.5234, acc: 75.20%]

Epoch 1 Summary:
  Train Loss: 0.4523, Train Acc: 78.50%
  Val Loss: 0.5234, Val Acc: 75.20%
  New best model saved with validation accuracy: 75.20%
```

Les fichiers sauvegardés :
- `models/best_checkpoint.pth` - Meilleur modèle (plus haute accuracy)
- `models/last_checkpoint.pth` - Dernier modèle (dernière epoch)
- `results/training_history.json` - Historique d'entraînement
- `results/training_history.png` - Graphiques de performance

## 🎯 Prochaines Étapes

1. **Entraîner votre modèle** avec les paramètres par défaut
2. **Évaluer les performances** sur le test set
3. **Ajuster les hyperparamètres** si nécessaire
4. **Faire des prédictions** sur de nouvelles images

## 📚 Documentation Complète

- **README.md** - Documentation principale
- **docs/RAPPORT_TECHNIQUE.md** - Rapport technique détaillé
- **src/** - Code source avec commentaires

## 💡 Conseils

1. **Commencez petit** : Testez avec `regnet_y_400mf` et 20 epochs
2. **Vérifiez les données** : Utilisez `prepare_data.py verify`
3. **Surveillez l'entraînement** : Regardez les courbes de loss
4. **Sauvegardez régulièrement** : Les checkpoints sont automatiques
5. **Testez les prédictions** : Validez sur quelques images manuellement

## 🔗 Liens Utiles

- Repository : https://github.com/MohAitMesskine/detection-cancer-sein-regnet
- Dataset Kaggle : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
- PyTorch Docs : https://pytorch.org/docs/

## ❓ Besoin d'Aide ?

Ouvrez une issue sur GitHub avec :
- Description du problème
- Commande exécutée
- Message d'erreur complet
- Configuration système (OS, GPU, RAM)
