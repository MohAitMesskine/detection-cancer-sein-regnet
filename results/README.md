# Results Directory

Ce répertoire contient les résultats d'entraînement et d'évaluation du modèle.

## Structure

```
results/
├── training_history.json    # Historique d'entraînement (loss, accuracy)
├── training_history.png     # Graphiques de performance
├── evaluation/
│   ├── evaluation_metrics.json  # Métriques d'évaluation
│   ├── confusion_matrix.png     # Matrice de confusion
│   └── roc_curve.png           # Courbe ROC
└── README.md
```

## Fichiers Générés

### Training

**training_history.json** : Historique d'entraînement
```json
{
    "train_loss": [0.5, 0.4, 0.3, ...],
    "train_acc": [75, 80, 85, ...],
    "val_loss": [0.6, 0.5, 0.4, ...],
    "val_acc": [70, 75, 80, ...]
}
```

**training_history.png** : Graphiques montrant l'évolution de :
- Loss (entraînement et validation)
- Accuracy (entraînement et validation)

### Evaluation

**evaluation_metrics.json** : Métriques de performance
```json
{
    "accuracy": 0.935,
    "precision": 0.928,
    "recall": 0.941,
    "f1_score": 0.934,
    "auc": 0.96
}
```

**confusion_matrix.png** : Matrice de confusion visualisant :
- Vrais Positifs (TP)
- Vrais Négatifs (TN)
- Faux Positifs (FP)
- Faux Négatifs (FN)

**roc_curve.png** : Courbe ROC (Receiver Operating Characteristic) avec AUC

## Notes

- Les fichiers PNG et JSON sont exclus du versioning (.gitignore)
- Les résultats sont régénérés à chaque entraînement/évaluation
- Sauvegarder les résultats importants avec des noms explicites
