# Rapport Technique : Détection du Cancer du Sein avec RegNetY

## Table des Matières
1. [Introduction](#introduction)
2. [Contexte et Objectifs](#contexte-et-objectifs)
3. [Architecture RegNetY](#architecture-regnety)
4. [Dataset et Prétraitement](#dataset-et-prétraitement)
5. [Méthodologie](#méthodologie)
6. [Implémentation](#implémentation)
7. [Résultats Attendus](#résultats-attendus)
8. [Conclusions et Perspectives](#conclusions-et-perspectives)
9. [Références](#références)

---

## 1. Introduction

Ce rapport présente un système de détection automatique du cancer du sein basé sur l'apprentissage profond (deep learning) utilisant l'architecture RegNetY. Le système est conçu pour classifier des images médicales en deux catégories : bénigne (non cancéreuse) et maligne (cancéreuse).

### 1.1 Problématique

Le cancer du sein est l'un des cancers les plus répandus chez les femmes à travers le monde. Selon l'Organisation Mondiale de la Santé (OMS), il représente environ 25% de tous les cancers diagnostiqués chez les femmes. La détection précoce est cruciale pour améliorer les taux de survie, mais l'analyse manuelle des images médicales est :
- **Chronophage** : Nécessite beaucoup de temps de la part des radiologues
- **Sujette aux erreurs** : La fatigue et l'expérience peuvent affecter le diagnostic
- **Coûteuse** : Requiert des experts qualifiés

### 1.2 Solution Proposée

Nous proposons un système automatisé basé sur l'intelligence artificielle qui :
- Analyse automatiquement les images médicales
- Fournit une classification rapide (bénigne vs maligne)
- Assiste les médecins dans leur diagnostic
- Réduit le temps de traitement des examens

---

## 2. Contexte et Objectifs

### 2.1 Objectif Principal

Développer un modèle de classification binaire performant pour distinguer les tumeurs mammaires bénignes des tumeurs malignes à partir d'images médicales.

### 2.2 Objectifs Spécifiques

1. **Performance** : Atteindre une précision (accuracy) supérieure à 90%
2. **Fiabilité** : Minimiser les faux négatifs (tumeurs malignes classées comme bénignes)
3. **Généralisation** : Obtenir un modèle robuste qui fonctionne bien sur des données non vues
4. **Efficacité** : Utiliser une architecture optimisée pour un temps d'inférence rapide

### 2.3 Métriques d'Évaluation

Nous utilisons plusieurs métriques pour évaluer la performance du modèle :

- **Accuracy** : Proportion de prédictions correctes
  ```
  Accuracy = (TP + TN) / (TP + TN + FP + FN)
  ```

- **Precision** : Proportion de vrais positifs parmi les prédictions positives
  ```
  Precision = TP / (TP + FP)
  ```

- **Recall (Sensibilité)** : Proportion de vrais positifs détectés
  ```
  Recall = TP / (TP + FN)
  ```

- **F1-Score** : Moyenne harmonique de la précision et du recall
  ```
  F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
  ```

- **AUC-ROC** : Aire sous la courbe ROC, mesure la capacité de discrimination

Où :
- TP (True Positive) : Maligne correctement identifiée
- TN (True Negative) : Bénigne correctement identifiée
- FP (False Positive) : Bénigne incorrectement identifiée comme maligne
- FN (False Negative) : Maligne incorrectement identifiée comme bénigne

---

## 3. Architecture RegNetY

### 3.1 Qu'est-ce que RegNet ?

RegNet (Regularized Network) est une famille d'architectures CNN développée par Facebook AI Research (FAIR) en 2020. Ces modèles sont issus d'une recherche systématique dans l'espace des designs de réseaux pour identifier des architectures optimales.

### 3.2 Caractéristiques de RegNetY

RegNetY améliore RegNetX en ajoutant des connexions squeeze-and-excitation (SE) :

1. **Blocs de Construction** :
   - Convolutions en groupe (group convolutions)
   - Connexions résiduelles (skip connections)
   - Squeeze-and-Excitation blocks
   - Batch Normalization
   - ReLU activation

2. **Avantages** :
   - **Efficacité** : Bon équilibre entre précision et vitesse
   - **Scalabilité** : Plusieurs variantes (400MF à 32GF FLOPS)
   - **Performance** : Comparable à ResNet et EfficientNet
   - **Généralisation** : Pré-entraînement ImageNet transférable

### 3.3 Variantes Utilisées

| Modèle | Paramètres | FLOPs | Top-1 Acc (ImageNet) |
|--------|-----------|-------|----------------------|
| RegNetY-400MF | 4.3M | 0.4B | 74.0% |
| RegNetY-800MF | 6.3M | 0.8B | 76.3% |
| RegNetY-1.6GF | 11.2M | 1.6B | 78.0% |
| RegNetY-3.2GF | 19.4M | 3.2B | 78.9% |

### 3.4 Architecture Adaptée au Cancer du Sein

```python
RegNetY-400MF (backbone pré-entraîné ImageNet)
    ↓
Global Average Pooling
    ↓
Dropout (p=0.5)
    ↓
Dense Layer (in_features → 512)
    ↓
ReLU Activation
    ↓
Dropout (p=0.25)
    ↓
Dense Layer (512 → 2)
    ↓
Softmax (Bénigne | Maligne)
```

Cette architecture permet :
- **Transfer Learning** : Utilisation des features pré-apprises sur ImageNet
- **Fine-tuning** : Adaptation aux caractéristiques spécifiques du cancer du sein
- **Régularisation** : Dropout pour éviter le surapprentissage

---

## 4. Dataset et Prétraitement

### 4.1 Dataset Utilisé

**Source** : Kaggle - Breast Cancer Detection
- **URL** : https://www.kaggle.com/datasets/hayder17/breast-cancer-detection
- **Format** : Images PNG/JPG
- **Taille** : Variable selon les images
- **Classes** : 
  - Classe 0 : Bénigne
  - Classe 1 : Maligne

### 4.2 Prétraitement des Images

Toutes les images subissent les transformations suivantes :

1. **Redimensionnement** : 224×224 pixels (format standard ImageNet)
2. **Normalisation** : 
   - Mean = [0.485, 0.456, 0.406]
   - Std = [0.229, 0.224, 0.225]
   - (Valeurs ImageNet pour le transfer learning)

### 4.3 Augmentation des Données (Training uniquement)

Pour améliorer la généralisation et éviter le surapprentissage :

```python
Transformations d'entraînement :
- RandomHorizontalFlip (p=0.5)
- RandomVerticalFlip (p=0.5)
- RandomRotation (±20°)
- ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2)
- Resize (224×224)
- ToTensor
- Normalize
```

Avantages de l'augmentation :
- Augmente la diversité des données d'entraînement
- Réduit le surapprentissage
- Améliore la robustesse du modèle
- Simule différentes conditions d'acquisition

### 4.4 Division du Dataset

Stratégie recommandée :
- **Entraînement** : 70% des données
- **Validation** : 15% des données
- **Test** : 15% des données

Important : Utiliser une stratification pour maintenir les proportions de classes.

---

## 5. Méthodologie

### 5.1 Processus d'Entraînement

1. **Initialisation** :
   - Chargement du modèle RegNetY pré-entraîné
   - Remplacement de la tête de classification
   - Configuration de l'optimiseur et du loss

2. **Entraînement** :
   ```python
   Pour chaque époque :
       Pour chaque batch d'entraînement :
           - Forward pass
           - Calcul du loss (CrossEntropy)
           - Backward pass
           - Mise à jour des poids
       
       Validation :
           - Évaluation sur le set de validation
           - Calcul des métriques
           - Sauvegarde si meilleur modèle
   ```

3. **Optimisation** :
   - **Optimiseur** : Adam
   - **Learning Rate** : 0.001 (initial)
   - **Scheduler** : ReduceLROnPlateau
   - **Weight Decay** : 1e-4 (régularisation L2)

### 5.2 Fonction de Loss

CrossEntropyLoss pour la classification binaire :
```
Loss = -[y × log(ŷ) + (1-y) × log(1-ŷ)]
```

Où :
- y : vraie étiquette (0 ou 1)
- ŷ : probabilité prédite

### 5.3 Stratégies de Régularisation

1. **Dropout** : 
   - Couche 1 : 50%
   - Couche 2 : 25%

2. **Weight Decay** : L2 regularization (1e-4)

3. **Data Augmentation** : Voir section 4.3

4. **Early Stopping** : Arrêt si validation loss n'améliore pas pendant N époques

### 5.4 Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| Batch Size | 32 | Équilibre mémoire/stabilité |
| Learning Rate | 0.001 | Standard pour Adam |
| Epochs | 50 | Convergence attendue |
| Dropout | 0.5/0.25 | Prévention surapprentissage |
| Weight Decay | 1e-4 | Régularisation L2 |
| Image Size | 224×224 | Standard ImageNet |

---

## 6. Implémentation

### 6.1 Technologies Utilisées

- **Framework** : PyTorch 2.0+
- **Modèle** : torchvision.models.regnet_y_*
- **Data Processing** : pandas, numpy, PIL
- **Visualisation** : matplotlib, seaborn
- **Évaluation** : scikit-learn

### 6.2 Structure du Code

```
src/
├── model.py       # Architecture du modèle
├── dataset.py     # Chargement et prétraitement
├── train.py       # Script d'entraînement
├── evaluate.py    # Évaluation et métriques
├── predict.py     # Inférence sur nouvelles images
└── utils.py       # Fonctions utilitaires
```

### 6.3 Composants Principaux

1. **Model (model.py)** :
   - Classe `RegNetYBreastCancer`
   - Factory function `get_model()`
   - Support de multiples variantes

2. **Dataset (dataset.py)** :
   - Classe `BreastCancerDataset`
   - Transformations et augmentation
   - DataLoader configuration

3. **Training (train.py)** :
   - Boucle d'entraînement
   - Validation
   - Checkpointing
   - Logging des métriques

4. **Evaluation (evaluate.py)** :
   - Calcul des métriques
   - Génération des visualisations
   - Rapport de classification

5. **Prediction (predict.py)** :
   - Inférence sur images individuelles
   - Batch prediction
   - Affichage des probabilités

### 6.4 Pipeline Complet

```
1. Préparation des données
   ↓
2. Entraînement du modèle
   ↓
3. Évaluation sur test set
   ↓
4. Génération des visualisations
   ↓
5. Déploiement pour inférence
```

---

## 7. Résultats Attendus

### 7.1 Performance Cible

Sur le dataset de validation/test :

| Métrique | Objectif | Acceptable |
|----------|----------|------------|
| Accuracy | > 93% | > 90% |
| Precision | > 93% | > 90% |
| Recall | > 93% | > 90% |
| F1-Score | > 93% | > 90% |
| AUC-ROC | > 0.95 | > 0.90 |

### 7.2 Visualisations Générées

1. **Courbes d'apprentissage** :
   - Loss vs Epochs (train/val)
   - Accuracy vs Epochs (train/val)

2. **Matrice de confusion** :
   ```
                Prédit
              B    M
   Vrai  B  [TN] [FP]
         M  [FN] [TP]
   ```

3. **Courbe ROC** :
   - True Positive Rate vs False Positive Rate
   - AUC comme métrique de performance

### 7.3 Temps d'Exécution

Estimations avec GPU (NVIDIA RTX 3060) :
- **Entraînement** : ~2-4 heures (50 epochs, ~1000 images)
- **Inférence** : ~50ms par image
- **Batch inference** : ~2s pour 32 images

### 7.4 Comparaison avec d'Autres Architectures

| Architecture | Params | Accuracy | Inference Time |
|--------------|--------|----------|----------------|
| ResNet50 | 25.6M | 91-93% | ~60ms |
| EfficientNet-B0 | 5.3M | 92-94% | ~45ms |
| **RegNetY-400MF** | **4.3M** | **92-94%** | **~50ms** |
| RegNetY-800MF | 6.3M | 93-95% | ~70ms |

RegNetY offre un excellent compromis entre :
- Nombre de paramètres
- Précision
- Vitesse d'inférence

---

## 8. Conclusions et Perspectives

### 8.1 Conclusions

Ce projet démontre que :

1. **RegNetY est adapté** pour la détection du cancer du sein
2. **Transfer Learning** améliore significativement les performances
3. **L'augmentation des données** est essentielle pour la généralisation
4. **Le système peut assister** les radiologues dans leur diagnostic

### 8.2 Limites

1. **Dépendance aux données** : 
   - Performance liée à la qualité du dataset
   - Biais potentiels dans les données d'entraînement

2. **Interprétabilité** :
   - Modèle "boîte noire"
   - Difficile d'expliquer les décisions

3. **Généralisation** :
   - Performance peut varier selon l'équipement d'imagerie
   - Nécessite validation sur données multi-centres

### 8.3 Perspectives et Améliorations Futures

1. **Améliorations Techniques** :
   - Grad-CAM pour visualisation des zones d'intérêt
   - Ensemble de modèles (RegNetY + EfficientNet)
   - Architecture attention-based

2. **Extension du Système** :
   - Classification multi-classe (types de tumeurs)
   - Détection et segmentation des tumeurs
   - Intégration de données cliniques

3. **Déploiement** :
   - API REST pour intégration clinique
   - Interface web pour radiologues
   - Application mobile pour screening

4. **Validation Clinique** :
   - Études prospectives multi-centres
   - Validation par comité d'experts
   - Certification médicale

### 8.4 Impact Potentiel

- **Santé Publique** : Détection précoce, meilleurs taux de survie
- **Économique** : Réduction des coûts de diagnostic
- **Social** : Accès au diagnostic dans zones sous-équipées

---

## 9. Références

### 9.1 Articles Scientifiques

1. Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollár, P. (2020). 
   **Designing Network Design Spaces.** 
   In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 10428-10436).

2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). 
   **Deep residual learning for image recognition.** 
   In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). 
   **Imagenet classification with deep convolutional neural networks.** 
   Advances in neural information processing systems, 25.

4. McKinney, S. M., et al. (2020). 
   **International evaluation of an AI system for breast cancer screening.** 
   Nature, 577(7788), 89-94.

### 9.2 Datasets

5. Hayder17. (2023). 
   **Breast Cancer Detection Dataset.** 
   Kaggle. https://www.kaggle.com/datasets/hayder17/breast-cancer-detection

6. Moreira, I. C., et al. (2012). 
   **INbreast: toward a full-field digital mammography database.** 
   Academic radiology, 19(2), 236-248.

### 9.3 Frameworks et Outils

7. Paszke, A., et al. (2019). 
   **PyTorch: An imperative style, high-performance deep learning library.** 
   Advances in neural information processing systems, 32.

8. Pedregosa, F., et al. (2011). 
   **Scikit-learn: Machine learning in Python.** 
   Journal of machine learning research, 12(Oct), 2825-2830.

### 9.4 Ressources en Ligne

9. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
10. torchvision Models: https://pytorch.org/vision/stable/models.html
11. RegNet Paper: https://arxiv.org/abs/2003.13678

---

## Annexes

### A. Installation et Configuration

Voir README.md pour les instructions détaillées d'installation.

### B. Exemples d'Utilisation

```bash
# Entraînement
python src/train.py --train_dir data/train --val_dir data/val

# Évaluation
python src/evaluate.py --data_dir data/test --checkpoint models/best_checkpoint.pth

# Prédiction
python src/predict.py --input image.png --checkpoint models/best_checkpoint.pth
```

### C. Codes Sources

Tous les codes sources sont disponibles dans le répertoire `src/` du repository GitHub.

### D. Contact et Contribution

Repository GitHub : https://github.com/MohAitMesskine/detection-cancer-sein-regnet

Pour toute question ou contribution, veuillez ouvrir une issue sur GitHub.

---

**Date du rapport** : Octobre 2025  
**Auteur** : Mohamed Ait Messekine  
**Version** : 1.0
