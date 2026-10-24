Trained on **503 Premier League players** (model: `linear`, 5-fold cross-validation; every player is scored by a model that never saw them).

| Metric | Value |
|---|---|
| Cross-validated R² (log value) | 0.745 |
| Mean absolute error | €7.9m |
| Median absolute error | 30% |
| 80% prediction interval | -46% to +76% of the point estimate |

**Model comparison** (all features, cross-validated R²)

| Model | R² |
|---|---|
| linear | 0.745 |
| ridge | 0.748 |
| gbm | 0.741 |

**Ablation study** (drop one feature group, retrain, measure the change in R²)

| Feature group removed | R² | Change |
|---|---|---|
| age | 0.615 | -0.130 |
| league_stats | 0.747 | +0.001 |
| club_strength | 0.711 | -0.034 |
| season_activity | 0.618 | -0.127 |
| contract_caps | 0.725 | -0.020 |
| position | 0.750 | +0.004 |
| world_cup | 0.746 | +0.000 |
