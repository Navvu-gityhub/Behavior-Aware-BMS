# Unsupervised detection study

Source: `continuous_model_training_data.csv` — 2682 rows, 34 cells.
Contamination assumed: 0.05.

## Separability

```
Cluster separability: SEPARABLE
  best k = 6, silhouette = 0.5414
  null (uniform, same shape, p95) = 0.1534   margin = +0.3881 (needs +0.10)
  2682 samples, 6 features
    k=2: real +0.4201  null +0.2878
    k=3: real +0.4454  null +0.2119
    k=4: real +0.5262  null +0.1620
    k=5: real +0.5189  null +0.1518
    k=6: real +0.5414  null +0.1534
```

## Segment profile

```
 cluster  n_rows    share
       0     453 0.168904
       1     816 0.304251
       2     132 0.049217
       3     256 0.095451
       4     606 0.225951
       5     419 0.156227
```

DBSCAN: eps=0.212, noise share 0.062

## isolation_forest

```
rule-flag agreement not computable on a cycle-level frame: the boolean flags exist on row-level telemetry, which this frame aggregates away. Run against pipeline telemetry to compare.

Fade association: NOT_ASSOCIATED
  flagged   median fade -0.005221 (n=135)
  unflagged median fade +0.000101 (n=1624)
  Mann-Whitney U = 83250, p = 1.0000, 20 cells
  Flagged cycles are not followed by faster fade. These are statistical outliers; on this evidence they are not evidence of damage, and must not be reported as harmful behaviour.
```

## local_outlier_factor

```
rule-flag agreement not computable on a cycle-level frame: the boolean flags exist on row-level telemetry, which this frame aggregates away. Run against pipeline telemetry to compare.

Fade association: NOT_ASSOCIATED
  flagged   median fade -0.002545 (n=135)
  unflagged median fade +0.000000 (n=2342)
  Mann-Whitney U = 147735, p = 0.8999, 29 cells
  Flagged cycles are not followed by faster fade. These are statistical outliers; on this evidence they are not evidence of damage, and must not be reported as harmful behaviour.
```

Detector overlap: 34 of 135 in common.
