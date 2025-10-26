# Exemple d'Utilisation - RegNetY Breast Cancer Detection

Ce fichier montre des exemples concrets d'utilisation du projet.

## Table des Matières
1. [Installation](#installation)
2. [Préparation des Données](#préparation-des-données)
3. [Entraînement](#entraînement)
4. [Évaluation](#évaluation)
5. [Prédiction](#prédiction)
6. [Utilisation Avancée](#utilisation-avancée)

---

## Installation

```bash
# Cloner le repository
git clone https://github.com/MohAitMesskine/detection-cancer-sein-regnet.git
cd detection-cancer-sein-regnet

# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

---

## Préparation des Données

### Scénario 1 : Dataset déjà organisé

Si vos données sont déjà dans des dossiers benign/ et malignant/ :

```bash
# Structure existante :
# raw_data/
#   ├── benign/
#   └── malignant/

# Diviser en train/val/test
python src/prepare_data.py split \
    --source_dir raw_data \
    --output_dir data \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42
```

### Scénario 2 : Dataset avec fichier CSV

Si vous avez un fichier CSV avec les labels :

```bash
# CSV format: image_name,label
# example.csv:
#   image_001.png,0
#   image_002.png,1
#   ...

# Organiser par classe
python src/prepare_data.py organize \
    --source_dir raw_images/ \
    --output_dir data/organized \
    --labels_file labels.csv

# Puis diviser
python src/prepare_data.py split \
    --source_dir data/organized \
    --output_dir data
```

### Vérification

```bash
# Vérifier la structure
python src/prepare_data.py verify --data_dir data

# Sortie attendue :
# TRAIN SET:
#   benign: 350 images
#   malignant: 350 images
# VAL SET:
#   benign: 75 images
#   malignant: 75 images
# TEST SET:
#   benign: 75 images
#   malignant: 75 images
```

---

## Entraînement

### Exemple 1 : Entraînement de Base

```bash
# Configuration minimale
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_400mf \
    --epochs 50 \
    --batch_size 32
```

Sortie pendant l'entraînement :
```
Using device: cuda
Creating data loaders...
Training samples: 700
Validation samples: 150
Creating regnet_y_400mf model...

Starting training...
Epoch 0 [Train]: 100%|████| 22/22 [00:15<00:00, loss: 0.6234, acc: 65.00%]
Epoch 0 [Val]:   100%|████| 5/5 [00:02<00:00, loss: 0.5892, acc: 70.00%]

Epoch 0 Summary:
  Train Loss: 0.6234, Train Acc: 65.00%
  Val Loss: 0.5892, Val Acc: 70.00%
  New best model saved with validation accuracy: 70.00%
...
```

### Exemple 2 : Entraînement avec Hyperparamètres Personnalisés

```bash
# Configuration avancée
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_800mf \
    --epochs 100 \
    --batch_size 16 \
    --learning_rate 0.0001 \
    --weight_decay 1e-4 \
    --dropout_rate 0.5 \
    --image_size 224 \
    --pretrained \
    --augment \
    --output_dir results \
    --checkpoint_dir models \
    --num_workers 4
```

### Exemple 3 : Reprendre un Entraînement

```bash
# Reprendre depuis le dernier checkpoint
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --resume models/last_checkpoint.pth \
    --epochs 100
```

### Exemple 4 : Entraînement sur CPU

```bash
# Forcer l'utilisation du CPU
export CUDA_VISIBLE_DEVICES=""
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --model_name regnet_y_400mf \
    --epochs 30 \
    --batch_size 8
```

---

## Évaluation

### Exemple 1 : Évaluation Basique

```bash
python src/evaluate.py \
    --data_dir data/test \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf \
    --output_dir results/evaluation
```

Sortie :
```
Using device: cuda
Creating regnet_y_400mf model...
Loading checkpoint from models/best_checkpoint.pth...
Creating data loader...
Evaluation samples: 150
Getting predictions...

Calculating metrics...

Evaluation Results:
  Accuracy: 0.9333
  Precision: 0.9286
  Recall: 0.9387
  F1-Score: 0.9336
  AUC-ROC: 0.9654

Evaluation complete! Results saved to results/evaluation
```

Fichiers générés :
- `results/evaluation/evaluation_metrics.json`
- `results/evaluation/confusion_matrix.png`
- `results/evaluation/roc_curve.png`

### Exemple 2 : Évaluation avec Batch Size Personnalisé

```bash
python src/evaluate.py \
    --data_dir data/test \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf \
    --batch_size 64 \
    --output_dir results/evaluation
```

---

## Prédiction

### Exemple 1 : Prédiction sur une Image

```bash
python src/predict.py \
    --input test_images/sample_001.png \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

Sortie :
```
Using device: cuda
Loading regnet_y_400mf model...
Processing 1 image(s)...

Image: sample_001.png
  Prediction: Malignant
  Confidence: 94.52%
  Probabilities:
    Benign: 5.48%
    Malignant: 94.52%
```

### Exemple 2 : Prédiction sur un Dossier

```bash
python src/predict.py \
    --input test_images/ \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

Sortie :
```
Processing 10 image(s)...

Image: image_001.png
  Prediction: Benign
  Confidence: 89.23%
  ...

Image: image_002.png
  Prediction: Malignant
  Confidence: 92.15%
  ...
```

### Exemple 3 : Prédiction avec Modèle Personnalisé

```bash
python src/predict.py \
    --input my_image.jpg \
    --checkpoint models/regnet_y_800mf_best.pth \
    --model_name regnet_y_800mf \
    --image_size 224
```

---

## Utilisation Avancée

### Utilisation Programmatique en Python

```python
# script_personnalise.py
import torch
from src.model import get_model
from src.utils import load_checkpoint
from src.dataset import get_data_transforms
from PIL import Image

# 1. Charger le modèle
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = get_model(model_name='regnet_y_400mf', pretrained=False, device=device)
load_checkpoint('models/best_checkpoint.pth', model)
model.eval()

# 2. Préparer l'image
transforms = get_data_transforms(image_size=224, augment=False)['val']
image = Image.open('test_image.png').convert('RGB')
image_tensor = transforms(image).unsqueeze(0).to(device)

# 3. Faire la prédiction
with torch.no_grad():
    output = model(image_tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, predicted_class].item()

# 4. Afficher le résultat
class_names = ['Benign', 'Malignant']
print(f"Prediction: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

### Entraînement avec Monitoring Personnalisé

```python
# train_custom.py
import torch
from torch.utils.data import DataLoader
from src.model import get_model
from src.dataset import BreastCancerDataset, get_data_transforms
import wandb  # Pour le monitoring

# Initialiser wandb
wandb.init(project="breast-cancer-detection")

# Créer le modèle
model = get_model(model_name='regnet_y_400mf')

# Créer les data loaders
transforms = get_data_transforms()
train_dataset = BreastCancerDataset('data/train', transform=transforms['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Boucle d'entraînement personnalisée
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(50):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Logger dans wandb
        wandb.log({"loss": loss.item()})
```

### Ensemble de Modèles

```python
# ensemble.py
import torch
from src.model import get_model
from src.utils import load_checkpoint

# Charger plusieurs modèles
models = []
for model_name in ['regnet_y_400mf', 'regnet_y_800mf']:
    model = get_model(model_name=model_name, pretrained=False)
    load_checkpoint(f'models/{model_name}_best.pth', model)
    model.eval()
    models.append(model)

# Prédiction par ensemble
def ensemble_predict(image_tensor):
    predictions = []
    for model in models:
        with torch.no_grad():
            output = model(image_tensor)
            predictions.append(torch.softmax(output, dim=1))
    
    # Moyenne des prédictions
    ensemble_pred = torch.stack(predictions).mean(dim=0)
    return ensemble_pred

# Utiliser l'ensemble
image_tensor = ...  # Votre image
prediction = ensemble_predict(image_tensor)
```

---

## Conseils et Astuces

### 1. Optimiser les Performances

```bash
# Augmenter le batch size si GPU puissant
python src/train.py --batch_size 64

# Réduire le batch size si OOM error
python src/train.py --batch_size 8

# Utiliser plusieurs workers pour le chargement des données
python src/train.py --num_workers 8
```

### 2. Debugging

```bash
# Entraîner sur un petit subset pour tester
# (Créer un petit dataset dans data/debug/)
python src/train.py \
    --train_dir data/debug/train \
    --val_dir data/debug/val \
    --epochs 5 \
    --batch_size 4
```

### 3. Comparaison de Modèles

```bash
# Entraîner plusieurs modèles
for model in regnet_y_400mf regnet_y_800mf regnet_y_1_6gf; do
    python src/train.py \
        --model_name $model \
        --checkpoint_dir models/$model
done

# Évaluer tous les modèles
for model in regnet_y_400mf regnet_y_800mf regnet_y_1_6gf; do
    python src/evaluate.py \
        --checkpoint models/$model/best_checkpoint.pth \
        --model_name $model \
        --output_dir results/$model
done
```

---

## Troubleshooting

### Problème : CUDA out of memory
**Solution** : Réduire batch_size ou utiliser un modèle plus petit

```bash
python src/train.py --batch_size 8 --model_name regnet_y_400mf
```

### Problème : Dataset non trouvé
**Solution** : Vérifier la structure avec verify

```bash
python src/prepare_data.py verify --data_dir data
```

### Problème : Modèle ne converge pas
**Solution** : Ajuster learning rate

```bash
python src/train.py --learning_rate 0.0001
```

---

## Ressources Supplémentaires

- [README.md](README.md) - Documentation complète
- [docs/RAPPORT_TECHNIQUE.md](docs/RAPPORT_TECHNIQUE.md) - Rapport technique détaillé
- [docs/GUIDE_DEMARRAGE.md](docs/GUIDE_DEMARRAGE.md) - Guide de démarrage rapide
- [CONTRIBUTING.md](CONTRIBUTING.md) - Guide de contribution

---

**Bon coding ! 🚀**
