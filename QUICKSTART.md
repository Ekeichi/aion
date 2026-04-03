# JEPA Digital Twin - Démarrage Rapide

## Installation (1 minute)

```bash
pip install -r requirements.txt
```

## Test Rapide (2 minutes)

Vérifier que tout fonctionne :

```bash
python quick_test.py
```

Résultat attendu :
```
Epoch 5/5 - Train Loss: 0.090, Val Loss: 0.073
Quick test completed successfully!
```

## Entraînement Complet (30-60 minutes sur CPU)

```bash
python train.py
```

Cela va :
- Générer ou charger les données synthétiques
- Entraîner le modèle JEPA pour 100 epochs
- Sauvegarder le meilleur modèle dans `models/best_model.pt`
- Créer des visualisations dans `logs/`

## Évaluation et Inférence

```bash
python inference.py
```

Affiche les métriques sur le test set et génère des visualisations.

## Exemple d'Utilisation

```bash
python example_usage.py
```

Montre comment utiliser le modèle entraîné pour comparer différentes actions.

## Structure des Données d'Entrée

Le modèle prend en entrée :

**x_t (contexte)** : Fenêtre de 14 jours × 12 features
- Features repos : HRV, FC repos, sommeil (durée, qualité)
- Features séance : FC moyenne, dérive FC, allure, économie, cadence, durée, dénivelé, RPE

**a_t (action)** : 8 features décrivant la séance planifiée
- Type : [repos, facile, modéré, intensif] (one-hot)
- Durée planifiée
- Intensité planifiée
- Dénivelé planifié

**y_t (cible)** : État futur (même format que x_t mais pour un seul jour)

## Architecture JEPA

```
Contexte (14 jours)  →  Enc_x (Transformer)  →  s_x  ─┐
                                                         ├→ Prédicteur → ŝ_y → Loss
Action planifiée     →  Enc_action (MLP)     →  e_a  ─┘              ↓
                                                                       vs
État futur           →  Enc_y (MLP + stop-grad) → s_y ───────────────┘
```

**Point clé** : Le gradient ne passe PAS par Enc_y (stop-gradient). Cela force le modèle à apprendre la vraie dynamique.

## Fichiers Importants

| Fichier | Description |
|---------|-------------|
| `config.py` | Tous les hyperparamètres |
| `data_generator.py` | Génération de données synthétiques |
| `jepa_model.py` | Modèle JEPA complet |
| `train.py` | Script d'entraînement |
| `inference.py` | Évaluation et visualisation |
| `SUMMARY.md` | Résumé détaillé de l'implémentation |
| `IMPLEMENTATION.md` | Guide d'implémentation complet |

## Données Synthétiques

Les données générées simulent :
- 10 athlètes avec profils physiologiques différents
- 180 jours par athlète (6 mois)
- Patterns réalistes :
  - Accumulation de fatigue après entraînement
  - Récupération sur jours de repos
  - Variation HRV/FC corrélée à la fatigue
  - Séances variées avec RPE cohérent

## Utiliser Vos Données Réelles

1. Créer un CSV avec les colonnes :
   ```
   athlete_id, day, hrv_rmssd, hr_rest, sleep_duration, sleep_quality,
   hr_mean, hr_drift, pace_mean, pace_hr_ratio, cadence_mean,
   duration, elevation_gain, rpe
   ```

2. Modifier le chemin CSV dans `train.py` :
   ```python
   csv_path = Path('votre_fichier.csv')
   ```

3. Lancer l'entraînement normalement

## Résultats Attendus (après 100 epochs)

- **Train Loss** : ~0.02-0.05
- **Val Loss** : ~0.05-0.08
- **Cosine Similarity** : >0.7

La loss mesure la distance MSE en espace latent entre prédiction et cible.

## Prochaines Extensions

1. **Têtes de tâches** : Ajouter des prédicteurs de fatigue, risque, performance
2. **Multi-horizon** : Prédire à τ = 1, 3, 7 jours
3. **Optimisation d'actions** : Rechercher la meilleure planification
4. **Données réelles** : Intégrer vos données d'athlètes

## Troubleshooting

**Problème** : `ModuleNotFoundError: No module named 'torch'`
**Solution** : `pip install -r requirements.txt`

**Problème** : Données manquantes
**Solution** : Lancer `python data_generator.py` pour générer les données

**Problème** : Loss ne diminue pas
**Solution** :
- Vérifier que les données sont bien normalisées
- Réduire le learning rate dans `config.py`
- Augmenter le nombre d'epochs

**Problème** : Overfitting (val loss augmente)
**Solution** :
- Augmenter le dropout dans `config.py`
- Augmenter le weight decay
- Générer plus de données

## Support

Pour questions/issues :
- Consulter `IMPLEMENTATION.md` pour détails techniques
- Consulter `README.md` pour théorie JEPA
- Examiner les logs dans `logs/`

---

**Tout est prêt - Bon entraînement ! 🏃‍♂️💨**
