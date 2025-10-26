# Guide de Contribution

Merci de votre intérêt pour contribuer à ce projet de détection du cancer du sein avec RegNetY !

## Comment Contribuer

### 1. Signaler des Bugs

Si vous trouvez un bug, veuillez :
- Vérifier que le bug n'a pas déjà été signalé dans les [Issues](https://github.com/MohAitMesskine/detection-cancer-sein-regnet/issues)
- Créer une nouvelle issue avec :
  - Un titre descriptif
  - Une description détaillée du problème
  - Les étapes pour reproduire le bug
  - Votre environnement (OS, Python version, GPU/CPU)
  - Les messages d'erreur complets

### 2. Proposer des Améliorations

Pour proposer une nouvelle fonctionnalité :
- Ouvrir une issue avec le tag "enhancement"
- Décrire la fonctionnalité souhaitée
- Expliquer pourquoi elle serait utile
- Proposer une implémentation si possible

### 3. Soumettre des Pull Requests

#### Processus

1. **Fork** le repository
2. **Clone** votre fork :
   ```bash
   git clone https://github.com/VOTRE_USERNAME/detection-cancer-sein-regnet.git
   ```
3. **Créer une branche** pour votre contribution :
   ```bash
   git checkout -b feature/ma-nouvelle-fonctionnalite
   ```
4. **Faire vos modifications** en suivant les guidelines ci-dessous
5. **Tester** vos modifications
6. **Commit** vos changements :
   ```bash
   git commit -m "Description claire des changements"
   ```
7. **Push** vers votre fork :
   ```bash
   git push origin feature/ma-nouvelle-fonctionnalite
   ```
8. **Créer une Pull Request** sur GitHub

#### Guidelines de Code

**Style Python**
- Suivre PEP 8
- Utiliser des noms de variables descriptifs
- Ajouter des docstrings pour les fonctions et classes
- Limiter les lignes à 100 caractères

**Exemple de docstring** :
```python
def ma_fonction(param1, param2):
    """
    Description brève de la fonction.
    
    Args:
        param1 (type): Description du paramètre 1.
        param2 (type): Description du paramètre 2.
        
    Returns:
        type: Description du retour.
    """
    pass
```

**Organisation du Code**
- Garder les fonctions courtes et ciblées
- Éviter la duplication de code
- Utiliser les imports relatifs dans le package src/
- Ajouter des commentaires pour le code complexe

**Tests**
- Tester les nouvelles fonctionnalités
- S'assurer que les fonctionnalités existantes fonctionnent toujours
- Documenter comment tester votre contribution

### 4. Types de Contributions Recherchées

Nous accueillons favorablement les contributions suivantes :

#### Code
- **Optimisations** : Améliorer la vitesse ou l'efficacité
- **Nouvelles architectures** : Support pour d'autres modèles (EfficientNet, Vision Transformer, etc.)
- **Fonctionnalités** : 
  - Grad-CAM pour visualisation
  - Support multi-GPU
  - Mixed precision training
  - Model ensembling
  - API REST
  - Interface web

#### Documentation
- **Traductions** : Anglais, Espagnol, Arabe
- **Tutoriels** : Guides d'utilisation, exemples
- **Améliorations** : Clarification, corrections, ajouts

#### Données
- **Nouveaux datasets** : Support pour d'autres datasets de cancer du sein
- **Data augmentation** : Nouvelles techniques d'augmentation
- **Preprocessing** : Méthodes de prétraitement améliorées

#### Recherche
- **Expérimentations** : Partager les résultats d'hyperparamètres
- **Benchmarks** : Comparaison avec d'autres approches
- **Analyses** : Études d'erreurs, analyse des prédictions

### 5. Standards de Qualité

Avant de soumettre une PR, vérifiez que :

- [ ] Le code suit les conventions Python (PEP 8)
- [ ] Les docstrings sont présents et clairs
- [ ] Le code a été testé
- [ ] La documentation est à jour
- [ ] Les commits sont clairs et descriptifs
- [ ] Pas de fichiers générés/temporaires (utiliser .gitignore)

### 6. Review Process

1. Un maintainer reviewera votre PR
2. Des changements peuvent être demandés
3. Une fois approuvée, la PR sera mergée
4. Votre contribution sera créditée

### 7. Code de Conduite

Ce projet adhère aux principes suivants :

- **Respect** : Traiter tous les contributeurs avec respect
- **Bienveillance** : Être constructif dans les critiques
- **Collaboration** : Travailler ensemble vers l'objectif commun
- **Inclusion** : Accueillir les contributions de tous niveaux

Comportements inacceptables :
- Harcèlement ou discrimination
- Trolling ou insultes
- Spam ou publicité non sollicitée

### 8. Reconnaissance des Contributions

Tous les contributeurs seront :
- Listés dans le fichier CONTRIBUTORS.md
- Mentionnés dans les release notes
- Crédités dans les publications académiques le cas échéant

### 9. Questions ?

Pour toute question :
- Ouvrir une [Discussion](https://github.com/MohAitMesskine/detection-cancer-sein-regnet/discussions)
- Commenter une issue existante
- Contacter les maintainers

### 10. Ressources Utiles

- [PEP 8 Style Guide](https://pep8.org/)
- [Git Handbook](https://guides.github.com/introduction/git-handbook/)
- [How to Write a Git Commit Message](https://chris.beams.io/posts/git-commit/)
- [PyTorch Documentation](https://pytorch.org/docs/)

## Exemples de Contributions

### Exemple 1 : Ajouter une Nouvelle Augmentation

```python
# Dans src/dataset.py
transforms.RandomGrayscale(p=0.1)  # Ajouter cette ligne
```

### Exemple 2 : Améliorer la Documentation

```markdown
## Nouveau Section

Description de la nouvelle fonctionnalité...
```

### Exemple 3 : Corriger un Bug

```python
# Avant
if accuracy > best_accuracy:  # Bug : comparison incorrecte

# Après  
if val_accuracy > best_val_accuracy:  # Correction
```

## Merci !

Merci de contribuer à améliorer la détection du cancer du sein avec l'IA. Chaque contribution, petite ou grande, fait une différence !

---

**Contact** : Ouvrir une issue sur GitHub pour toute question.
