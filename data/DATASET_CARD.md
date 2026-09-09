# Dataset Card — combined_dataset.csv

Generated: 2026-09-06T08:46:30+00:00

## Sources

1. **L-HSAB** (Mulki et al., 2019 — ACL Anthology W19-3512). Levantine Twitter dataset, originally 3-class (normal/abusive/hate), naturally imbalanced (~8% hate).
2. **training_ds1.csv** — 1,100 rows, already binary, artificially balanced (~59% positive). Source paper not confirmed.
3. **Arabic_Tweets_dataset.csv** — already binary, artificially balanced (~59% positive). Source not academically confirmed — treated as a lower-confidence, supplementary source.

## Label definition

Binary: `1` = hate/offensive, `0` = not_hate.

**Important:** for L-HSAB, both the `abusive` and `hate` classes are mapped to the positive class (see `LHSAB_LABEL_MAP` in `config.py`). This means the resulting label represents offensive/abusive language broadly, not strictly identity-targeted hate speech. State this explicitly wherever model scope is described.

## Cleanup decisions applied

- Cross-source label conflicts: 4 distinct texts had disagreeing labels across sources; 8 rows were dropped as unreliable.
- Exact duplicate texts (post-conflict-removal): 93 rows dropped.

## Final composition

- Total rows: 20,034
- Overall hate rate: 0.527

| Source | Rows | Hate rate |
|---|---|---|
| Arabic_Tweets_dataset | 13,188 | 0.587 |
| LHSAB | 5,751 | 0.378 |
| training_ds1 | 1,095 | 0.587 |

## Known caveat carried into modeling

This dataset mixes one naturally-distributed source (LHSAB, ~8% positive) with two artificially-balanced sources (~59% positive each). Any train/val/test split MUST stratify by `dataset` in addition to `label` (see `src/split_data.py`), otherwise evaluation metrics will not reflect performance on the natural distribution.
