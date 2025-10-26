# Models Directory

Ce répertoire contient les modèles entraînés et les checkpoints.

## Structure

```
models/
├── best_checkpoint.pth      # Meilleur modèle (plus haute accuracy sur validation)
├── last_checkpoint.pth      # Dernier checkpoint (dernière époque)
└── README.md               # Ce fichier
```

## Checkpoints

### Format des Checkpoints

Les checkpoints PyTorch (.pth) contiennent :

```python
{
    'epoch': int,                    # Numéro de l'époque
    'model_state_dict': OrderedDict, # Poids du modèle
    'optimizer_state_dict': dict,    # État de l'optimiseur
    'val_acc': float,                # Accuracy de validation
    'best_val_acc': float,          # Meilleure accuracy
    'history': dict                  # Historique d'entraînement
}
```

### Chargement d'un Checkpoint

```python
import torch
from src.model import get_model
from src.utils import load_checkpoint

# Créer le modèle
model = get_model(model_name='regnet_y_400mf', pretrained=False, device='cuda')

# Charger le checkpoint
checkpoint = load_checkpoint('models/best_checkpoint.pth', model)

# Le modèle est maintenant prêt pour l'inférence
model.eval()
```

### Utilisation

**Pour l'évaluation** :
```bash
python src/evaluate.py \
    --data_dir data/test \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

**Pour la prédiction** :
```bash
python src/predict.py \
    --input image.png \
    --checkpoint models/best_checkpoint.pth \
    --model_name regnet_y_400mf
```

**Pour reprendre l'entraînement** :
```bash
python src/train.py \
    --train_dir data/train \
    --val_dir data/val \
    --resume models/last_checkpoint.pth
```

## Modèles Pré-entraînés

Les modèles utilisent les poids ImageNet comme point de départ :
- RegNetY-400MF : https://download.pytorch.org/models/regnet_y_400mf-...
- RegNetY-800MF : https://download.pytorch.org/models/regnet_y_800mf-...

Ces poids sont téléchargés automatiquement par torchvision lors du premier entraînement.

## Taille des Modèles

| Modèle | Taille du Checkpoint | Paramètres |
|--------|---------------------|------------|
| RegNetY-400MF | ~20 MB | 4.3M |
| RegNetY-800MF | ~30 MB | 6.3M |
| RegNetY-1.6GF | ~50 MB | 11.2M |
| RegNetY-3.2GF | ~90 MB | 19.4M |

## Gestion des Checkpoints

### Sauvegarde Automatique

Durant l'entraînement, deux types de checkpoints sont sauvegardés :

1. **best_checkpoint.pth** : Sauvegardé quand la validation accuracy s'améliore
2. **last_checkpoint.pth** : Sauvegardé à chaque époque

### Nettoyage

Pour économiser l'espace disque :
```bash
# Garder uniquement le meilleur modèle
rm models/last_checkpoint.pth

# Garder les modèles de différentes epochs
mv models/best_checkpoint.pth models/model_epoch50_acc95.pth
```

## Versioning

Bonne pratique pour versionner les modèles :
```bash
# Exemple de nommage
models/
├── regnet_y_400mf_acc_92.5_20231015.pth
├── regnet_y_800mf_acc_94.2_20231020.pth
└── regnet_y_400mf_final.pth
```

## Notes Importantes

1. **Ne pas commiter les checkpoints** : Les fichiers .pth sont exclus par `.gitignore`
2. **Sauvegarde** : Faire des backups réguliers des meilleurs modèles
3. **Compatibilité** : Les checkpoints sont spécifiques à l'architecture du modèle
4. **Device** : Les checkpoints peuvent être chargés sur CPU ou GPU

## Partage des Modèles

Pour partager vos modèles entraînés :

1. Télécharger sur un service de stockage (Google Drive, Dropbox, etc.)
2. Ajouter le lien dans le README principal
3. Inclure les informations de performance

Exemple :
```
Modèle : RegNetY-400MF
Accuracy : 93.5%
Precision : 92.8%
Recall : 94.1%
F1-Score : 93.4%
AUC-ROC : 0.96
Lien : https://drive.google.com/...
```

## Dépannage

### Erreur de chargement

```python
# Si erreur "unexpected key in state_dict"
checkpoint = torch.load('models/best_checkpoint.pth')
# Vérifier les clés
print(checkpoint.keys())
```

### Incompatibilité de version PyTorch

```bash
# Sauvegarder pour compatibilité
torch.save(checkpoint, 'model.pth', _use_new_zipfile_serialization=False)
```

### Modèle trop volumineux

```bash
# Sauvegarder seulement les poids
torch.save(model.state_dict(), 'model_weights_only.pth')
```
