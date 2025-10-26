# Dataset Information

## Source

Ce projet utilise le dataset **Breast Cancer Detection** de Kaggle.

- **URL** : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
- **Description** : Images de tumeurs mammaires classées en bénignes et malignes
- **Format** : PNG/JPG

## Structure Attendue

Le dataset doit être organisé dans la structure suivante :

```
data/
├── train/
│   ├── benign/          # Images de tumeurs bénignes (entraînement)
│   │   ├── image_001.png
│   │   ├── image_002.png
│   │   └── ...
│   └── malignant/       # Images de tumeurs malignes (entraînement)
│       ├── image_001.png
│       ├── image_002.png
│       └── ...
├── val/
│   ├── benign/          # Images de tumeurs bénignes (validation)
│   │   └── ...
│   └── malignant/       # Images de tumeurs malignes (validation)
│       └── ...
└── test/
    ├── benign/          # Images de tumeurs bénignes (test)
    │   └── ...
    └── malignant/       # Images de tumeurs malignes (test)
        └── ...
```

## Classes

- **benign** (0) : Tumeurs bénignes (non cancéreuses)
- **malignant** (1) : Tumeurs malignes (cancéreuses)

## Téléchargement et Préparation

### Méthode 1 : Via Kaggle CLI

```bash
# 1. Installer Kaggle CLI
pip install kaggle

# 2. Configurer les credentials Kaggle
# Créer ~/.kaggle/kaggle.json avec vos clés API
# Télécharger depuis : https://www.kaggle.com/settings -> Create New API Token

# 3. Télécharger le dataset
kaggle datasets download -d hayder17/breast-cancer-detection

# 4. Décompresser
unzip breast-cancer-detection.zip -d data/raw/

# 5. Organiser et diviser le dataset
python src/prepare_data.py split \
    --source_dir data/raw \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15
```

### Méthode 2 : Téléchargement manuel

1. Aller sur https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
2. Cliquer sur "Download"
3. Décompresser dans `data/raw/`
4. Utiliser le script de préparation :

```bash
python src/prepare_data.py split \
    --source_dir data/raw \
    --output_dir data
```

## Statistiques Recommandées

### Division du Dataset

- **Training** : 70% des données
- **Validation** : 15% des données
- **Test** : 15% des données

### Distribution des Classes

Idéalement, maintenir un équilibre 50/50 entre les deux classes, ou au minimum 30/70.

Si déséquilibre important, considérer :
- Weighted loss function
- Over-sampling de la classe minoritaire
- Under-sampling de la classe majoritaire
- Augmentation de données ciblée

## Format des Images

- **Type** : RGB (3 canaux)
- **Taille** : Variable (sera redimensionnée à 224×224 lors du chargement)
- **Format** : PNG, JPG, ou JPEG
- **Normalisation** : Automatique (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

## Vérification du Dataset

Pour vérifier que votre dataset est correctement organisé :

```bash
python src/prepare_data.py verify --data_dir data
```

Sortie attendue :
```
Verifying dataset in: data
============================================================

TRAIN SET:
  benign: XXX images
  malignant: XXX images

VAL SET:
  benign: XXX images
  malignant: XXX images

TEST SET:
  benign: XXX images
  malignant: XXX images
============================================================
```

## Notes Importantes

1. **Ne pas commiter les données** : Les images sont exclues par `.gitignore`
2. **Sauvegardes** : Garder une copie du dataset brut dans `data/raw/`
3. **Reproductibilité** : Le script de split utilise un seed fixe (42) par défaut
4. **Taille** : Le dataset peut être volumineux (plusieurs GB)

## Utilisation dans le Code

Les scripts du projet chargent automatiquement les données depuis :
- `data/train/` pour l'entraînement
- `data/val/` pour la validation
- `data/test/` pour l'évaluation finale

Exemple :
```bash
python src/train.py --train_dir data/train --val_dir data/val
python src/evaluate.py --data_dir data/test --checkpoint models/best_checkpoint.pth
```

## Alternative : Dataset Custom

Si vous utilisez un autre dataset de cancer du sein :

1. Organiser les images dans la même structure (benign/malignant)
2. Utiliser le script de préparation
3. Ajuster les hyperparamètres si nécessaire

## Droits et Licence

Respecter les conditions d'utilisation du dataset Kaggle original.

## Contact

Pour toute question sur le dataset, se référer à :
- Page Kaggle : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
- Issues GitHub : https://github.com/MohAitMesskine/detection-cancer-sein-regnet/issues
