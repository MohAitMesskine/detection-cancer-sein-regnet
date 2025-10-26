# Detection du Cancer du Sein avec RegNetY

Ce projet implémente un modèle RegNetY (Designing Network Design Spaces) pour la détection du cancer du sein à partir d'images médicales. Le modèle utilise l'architecture RegNetY pré-entraînée sur ImageNet et fine-tunée sur le dataset de cancer du sein de Kaggle.

## 📋 Description du Projet

Le cancer du sein est l'un des cancers les plus courants chez les femmes dans le monde. La détection précoce est cruciale pour améliorer les chances de survie. Ce projet utilise l'apprentissage profond (deep learning) pour classifier automatiquement les images de tumeurs mammaires en deux catégories :
- **Bénignes** : Tumeurs non cancéreuses
- **Malignes** : Tumeurs cancéreuses

### Architecture RegNetY

RegNetY est une famille d'architectures de réseaux de neurones convolutifs (CNN) développée par Facebook AI Research. Ces modèles sont conçus pour être efficaces en termes de calcul tout en maintenant une haute précision. RegNetY utilise des connexions résiduelles et une structure de réseau optimisée.

## 🎯 Dataset

Ce projet utilise le dataset **Breast Cancer Detection** disponible sur Kaggle :
- **Source** : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
- **Format** : Images PNG/JPG
- **Classes** : 2 (Bénigne, Maligne)

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- CUDA (optionnel, pour l'entraînement GPU)
- 8 GB RAM minimum (16 GB recommandé)

### Étapes d'installation

1. Cloner le repository :
```bash
git clone https://github.com/MohAitMesskine/detection-cancer-sein-regnet.git
cd detection-cancer-sein-regnet
```

2. Créer un environnement virtuel :
```bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
detection-cancer-sein-regnet/
├── src/
│   ├── model.py          # Implémentation du modèle RegNetY
│   ├── dataset.py        # Chargement et prétraitement des données
│   ├── train.py          # Script d'entraînement
│   ├── evaluate.py       # Script d'évaluation
│   ├── predict.py        # Script de prédiction
│   └── utils.py          # Fonctions utilitaires
├── data/
│   ├── train/            # Données d'entraînement
│   │   ├── benign/
│   │   └── malignant/
│   └── val/              # Données de validation
│       ├── benign/
│       └── malignant/
├── models/               # Modèles sauvegardés
├── results/              # Résultats et visualisations
├── docs/                 # Documentation
├── requirements.txt      # Dépendances Python
└── README.md            # Ce fichier
```

## 📊 Préparation des Données

1. Télécharger le dataset depuis Kaggle :
```bash
# Installer Kaggle CLI si nécessaire
pip install kaggle

# Configurer les credentials Kaggle (créer ~/.kaggle/kaggle.json)
# Télécharger le dataset
kaggle datasets download -d hayder17/breast-cancer-detection
unzip breast-cancer-detection.zip -d data/
```

2. Organiser les données :
```
data/
├── train/
│   ├── benign/      # Images bénignes pour l'entraînement
│   └── malignant/   # Images malignes pour l'entraînement
└── val/
    ├── benign/      # Images bénignes pour la validation
    └── malignant/   # Images malignes pour la validation
```

## 🎓 Entraînement du Modèle

### Entraînement de base

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_400mf \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 0.001
```

### Options avancées

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_800mf \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --dropout_rate 0.5 \
    --pretrained \
    --augment \
    --output_dir results \
    --checkpoint_dir models
```

### Reprendre l'entraînement

```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --resume models/last_checkpoint.pth
```

## 📈 Évaluation du Modèle

```bash
python src/evaluate.py \
    --data_dir data/test \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf \
    --output_dir results/evaluation
```

Cette commande génère :
- Métriques de performance (accuracy, precision, recall, F1-score, AUC)
- Matrice de confusion
- Courbe ROC

## 🔮 Prédiction sur Nouvelles Images

### Image unique

```bash
python src/predict.py \
    --input path/to/image.png \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

### Dossier d'images

```bash
python src/predict.py \
    --input path/to/images/ \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

## 🎨 Variantes de Modèles RegNetY

Le projet supporte plusieurs variantes de RegNetY :

| Modèle | Paramètres | FLOPS | Utilisation |
|--------|-----------|-------|-------------|
| regnet_y_400mf | 4.3M | 0.4 GFLOPs | Petit dataset, GPU limité |
| regnet_y_800mf | 6.3M | 0.8 GFLOPs | Équilibre performance/vitesse |
| regnet_y_1_6gf | 11.2M | 1.6 GFLOPs | Haute précision |
| regnet_y_3_2gf | 19.4M | 3.2 GFLOPs | Très haute précision |

## 📊 Résultats Attendus

Les résultats typiques sur le dataset de cancer du sein :
- **Accuracy** : 85-95%
- **Precision** : 85-95%
- **Recall** : 85-95%
- **F1-Score** : 85-95%
- **AUC-ROC** : 0.90-0.98

*Note : Les résultats réels dépendent de la qualité des données, des hyperparamètres et de la variance aléatoire.*

## 🔧 Configuration Recommandée

### Pour l'entraînement

- **GPU** : NVIDIA GPU avec 4+ GB VRAM (recommandé)
- **RAM** : 16 GB
- **Stockage** : 10 GB d'espace libre

### Hyperparamètres recommandés

```python
model_name = 'regnet_y_400mf'  # ou regnet_y_800mf pour plus de précision
batch_size = 32                 # Réduire si mémoire GPU limitée
learning_rate = 0.001           # Taux d'apprentissage initial
epochs = 50                     # Nombre d'époques
dropout_rate = 0.5              # Régularisation
weight_decay = 1e-4             # Régularisation L2
```

## 📚 Documentation Technique

### Augmentation des Données

Le projet utilise plusieurs techniques d'augmentation :
- Flip horizontal et vertical
- Rotation aléatoire (±20°)
- Ajustement de la luminosité, contraste et saturation
- Normalisation (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### Architecture du Modèle

```python
RegNetY (pre-trained on ImageNet)
  ├── Stem (convolution initiale)
  ├── Stage 1-4 (blocs RegNet)
  ├── Global Average Pooling
  └── Classifier personnalisé:
      ├── Dropout(0.5)
      ├── Linear(in_features → 512)
      ├── ReLU
      ├── Dropout(0.25)
      └── Linear(512 → 2)
```

## 🐛 Dépannage

### Erreur de mémoire GPU

```bash
# Réduire la taille du batch
python src/train.py --batch_size 16

# Ou utiliser le CPU
export CUDA_VISIBLE_DEVICES=""
python src/train.py
```

### Problèmes d'installation

```bash
# Installer PyTorch avec CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Ou sans CUDA
pip install torch torchvision
```

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier LICENSE pour plus de détails.

## 👥 Auteurs

- Mohamed Ait Messekine

## 🙏 Remerciements

- Dataset : [Hayder17](https://www.kaggle.com/hayder17) pour le dataset Breast Cancer Detection
- Architecture : Facebook AI Research pour RegNet
- Framework : PyTorch et torchvision

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

## 🔗 Références

1. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). Designing Network Design Spaces. CVPR 2020.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.
3. Dataset Kaggle : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection

## 📝 Citations

Si vous utilisez ce code dans vos recherches, veuillez citer :

```bibtex
@misc{regnet-breast-cancer,
  author = {Mohamed Ait Messekine},
  title = {Detection du Cancer du Sein avec RegNetY},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/MohAitMesskine/detection-cancer-sein-regnet}
}
```