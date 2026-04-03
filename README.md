# Digital twin athlète (course à pied) — approche JEPA

Ce document regroupe la synthèse des échanges sur l’application d’un schéma **JEPA** (Joint-Embedding Predictive Architecture) pour un **jumeau numérique** d’un coureur, en lien avec le diagramme `jepa_training_pipeline.svg`.

---

## 1. Schéma JEPA (rappel)

- **`x_t`** : état / contexte à l’instant (ou la fenêtre) considéré(e).
- **`a_t`** : action ou planification (séance, consigne, intervention).
- **`y_{t+τ}`** : état futur à l’horizon `τ`.
- **`Enc_x`**, **`Enc_action`**, **`Enc_y`** : encodeurs (ex. Transformer 1D pour la série, MLP pour l’action structurée).
- **`Prédicteur`** : apprend une représentation prédite `ŝ_y` à partir de `(s_x, ê_a)` (ex. MLP profond `f(s_x, ê_a)`).
- **Perte** : distance dans l’espace latent, typiquement `‖ŝ_y − s_y‖²`.
- **`stop-gradient` sur `Enc_y`** : la cible `s_y` est traitée comme **cible figée** (pas de rétropropagation à travers `Enc_y`), pour stabiliser l’apprentissage et éviter les solutions triviales.

Interprétation : le modèle n’apprend pas à recopier pixel à pixel le futur, mais une **dynamique en représentation latente** : « si je fais cette action dans ce contexte, l’état futur plausible se projette ici ».

---

## 2. Objectif produit

Construire un **digital twin** capable de :

- encoder l’**historique récent** et le **contexte d’entraînement** ;
- prédire en latent un **état futur** cohérent après une action planifiée ;
- éventuellement alimenter des **têtes métier** (fatigue, risque, performance probable) après pré-entraînement.

---

## 3. Granularité du temps : deux lectures

### 3.1 Lecture « intra-séance » (possible)

- **`x_t`** peut être une **fenêtre glissante pendant l’activité** (ex. quelques secondes à quelques minutes) : allure, FC, cadence, puissance, dénivelé, etc.
- **`a_t`** : consigne ou plan sur l’horizon court (allure cible, durée d’intervalle, récupération).
- **`y_{t+τ}`** : même type de signaux, décalés de `τ` dans le futur.

### 3.2 Lecture retenue pour la spec détaillée : **un timestep = un jour**

Ici, **`x_t` n’est pas un vecteur unique** : c’est une **série temporelle sur une fenêtre glissante**, par exemple les **14 derniers jours**. Chaque jour est un vecteur de features.

Forme typique : **matrice `(L, F)`**, par exemple **`(14, 12)`** — 14 jours, 12 features — qui alimente un **Transformer 1D** sur l’axe temps.

---

## 4. Features par jour (un timestep)

### 4.1 Signaux de repos (matin, avant activité)

| Feature           | Description                                      |
|-------------------|--------------------------------------------------|
| `hrv_rmssd`       | Variabilité cardiaque (ex. RMSSD), ms            |
| `hr_rest`         | FC au repos, bpm                                 |
| `sleep_duration`  | Durée de sommeil, h                              |
| `sleep_quality`   | Score subjectif 1–5 ou issu de la montre       |

### 4.2 Signaux de séance (agrégats du jour ; **0 si jour de repos**)

| Feature           | Description                                      |
|-------------------|--------------------------------------------------|
| `hr_mean`         | FC moyenne (éventuellement normalisée)          |
| `hr_drift`        | Dérive FC sur la séance (fatigue intra-effort)   |
| `pace_mean`       | Allure moyenne, m/s (ou équivalent)              |
| `pace_hr_ratio`   | Proxy d’économie de course                       |
| `cadence_mean`    | Cadence, spm                                     |
| `duration`        | Durée de la séance, min                          |
| `elevation_gain`  | Dénivelé positif, m                              |
| `rpe`             | RPE post-séance, 1–10                            |

Exemple de structure (ordre indicatif) :

```python
x_t[j] = [
    hrv_rmssd, hr_rest, sleep_duration, sleep_quality,
    hr_mean, hr_drift, pace_mean, pace_hr_ratio,
    cadence_mean, duration, elevation_gain, rpe,
]
# → vecteur de dimension ~12
```

**`x_t` complet** : empilement des `j = 0 … L-1` → tenseur **`(14, 12)`** pour une fenêtre de 14 jours.

---

## 5. Subtilités importantes

### 5.1 Repos : zéros ≠ « effort nul mesuré »

Les **jours de repos** mettent souvent les features **séance** à **0**. Il ne faut pas que le modèle confonde :

- **pas de séance** (structure du jour) ;
- **séance avec effort réellement nul** (rare / ambigu).

**Pistes** (souvent combinées) :

- **Masque explicite** sur les canaux « séance » (ou groupe de features) pour indiquer *non observé / non applicable* vs valeur numérique.
- **Embedding « type de jour »** : repos / facile / modéré / intensif (ou au minimum binaire repos / entraînement).

Si des jours sont **incomplets** (séance sans RPE, etc.), il faut distinguer **donnée manquante** et **repos** : le masque et/ou un code « missing » aide.

### 5.2 Normalisation **par athlète**

Une même valeur (ex. FC 145 bpm) n’a pas le même sens selon le niveau. On normalise **sur l’historique de l’athlète** (souvent z-score par feature).

**Note** : sur des données avec **queues lourdes** ou **jours exceptionnels** (compét, maladie), des statistiques **robustes** (médiane, IQR, MAD) peuvent compléter ou remplacer moyenne/écart-type pour limiter l’influence des outliers.

---

## 6. Cohérence avec `a_t` et `y_{t+τ}` à l’échelle journalière

Avec un **`x_t`** journalier en fenêtre glissante :

- **`a_t`** : décision de **planification** (volume/intensité prévus, type de séance le lendemain ou sur les prochains jours), encodable par un **MLP** sur une structure fixe (comme sur le schéma).
- **`y_{t+τ}`** : **même type de représentation** que l’état futur après **`τ` jours** (features agrégées par jour), pour prédire la dynamique à moyen terme (charge, fatigue, sommeil, etc.).

Le diagramme JEPA (encodeurs + prédicteur + perte latente + `stop-gradient` sur la branche cible) reste **inchangé** ; seule l’**échelle de temps** des tenseurs change par rapport à une version purement intra-séance.

---

## 7. Bonnes pratiques (rappel)

- **Personnalisation** : modèle partagé + **embedding ou adaptation par athlète** si les volumes de données le permettent.
- **Causalité / fuite** : séparer clairement variables **actionnables** (`a_t`) et signaux **passifs** ; validation **temporelle** (split par date / blocs de séances), pas de futur dans les features passées.
- **Sécurité** : le twin ne remplace pas un avis médical ; **bornes** et **alertes** pour toute recommandation d’entraînement.

---

## 8. Fichier graphique associé

- `jepa_training_pipeline.svg` : illustration du flux `x_t`, `a_t`, `y_{t+τ}` → encodeurs → représentations `s_x`, `ê_a`, `s_y` → prédicteur → `ŝ_y` → perte `‖ŝ_y − s_y‖²`, avec mention explicite du **backprop** sur `Enc_x`, `Enc_action`, `Prédicteur` et du **blocage du gradient** sur `Enc_y`.

---

*Document de synthèse — projet AION_explr.*
