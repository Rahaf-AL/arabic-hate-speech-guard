"""
Data Ingestion & Merging

Loads the three raw source files, standardizes them to a common
(text, label, dataset) schema, merges them, resolves cross-source label
conflicts, drops exact duplicates, and writes:
  - data/raw/combined_dataset.csv   (the merged working file)
  - data/DATASET_CARD.md            (dataset documentation)
"""
# %%
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-source loaders: each returns a DataFrame with columns
# ["text", "label", "dataset"].
# ---------------------------------------------------------------------------
# %%
def load_lhsab(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    df = df.rename(columns={"Tweet": "text", "Class": "label"})
    df["label"] = df["label"].map(config.LHSAB_LABEL_MAP)

    unmapped = df["label"].isna().sum()
    if unmapped:
        logger.warning("LHSAB: %d rows had an unrecognized Class value and were dropped.", unmapped)
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)
    df["dataset"] = "LHSAB"
    return df[["text", "label", "dataset"]]


def load_ds1(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df["dataset"] = "training_ds1"
    return df[["text", "label", "dataset"]]


def load_arabic_tweets(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    df, n_dropped = drop_exact_duplicates(df)
    if n_dropped:
        logger.info("Arabic_Tweets_dataset: dropped %d within-source exact duplicate texts.", n_dropped)
    df["dataset"] = "Arabic_Tweets_dataset"
    return df[["text", "label", "dataset"]]


# ---------------------------------------------------------------------------
# Cleanup logic shared across sources
# ---------------------------------------------------------------------------

def drop_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    return df, before - len(df)


def resolve_cross_source_conflicts(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """Drop any text that appears with more than one distinct label anywhere
    in the merged data (i.e. different sources — or the same source —
    disagree on it). These are treated as unreliable and removed."""
    label_counts_per_text = df.groupby("text")["label"].nunique()
    conflict_texts = label_counts_per_text[label_counts_per_text > 1].index
    n_conflicts = len(conflict_texts)

    before = len(df)
    df = df[~df["text"].isin(conflict_texts)].reset_index(drop=True)
    n_dropped = before - len(df)
    return df, n_conflicts, n_dropped


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_dataset_card(df: pd.DataFrame, n_conflict_texts: int, n_conflict_rows: int,
                        n_exact_dupes: int, path: Path) -> None:
    hate_rate_overall = df["label"].mean()
    per_source_counts = df["dataset"].value_counts()
    per_source_rates = df.groupby("dataset")["label"].mean()

    lines = [
        "# Dataset Card — combined_dataset.csv",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "## Sources",
        "",
        "1. **L-HSAB** (Mulki et al., 2019 — ACL Anthology W19-3512). "
        "Levantine Twitter dataset, originally 3-class (normal/abusive/hate), "
        "naturally imbalanced (~8% hate).",
        "2. **training_ds1.csv** — 1,100 rows, already binary, artificially "
        "balanced (~59% positive). Source paper not confirmed.",
        "3. **Arabic_Tweets_dataset.csv** — already binary, artificially "
        "balanced (~59% positive). Source not academically confirmed — "
        "treated as a lower-confidence, supplementary source.",
        "",
        "## Label definition",
        "",
        "Binary: `1` = hate/offensive, `0` = not_hate.",
        "",
        "**Important:** for L-HSAB, both the `abusive` and `hate` classes are "
        "mapped to the positive class (see `LHSAB_LABEL_MAP` in `config.py`). "
        "This means the resulting label represents offensive/abusive language "
        "broadly, not strictly identity-targeted hate speech. State this "
        "explicitly wherever model scope is described.",
        "",
        "## Cleanup decisions applied",
        "",
        f"- Cross-source label conflicts: {n_conflict_texts} distinct texts had "
        f"disagreeing labels across sources; {n_conflict_rows} rows were dropped "
        "as unreliable.",
        f"- Exact duplicate texts (post-conflict-removal): {n_exact_dupes} rows dropped.",
        "",
        "## Final composition",
        "",
        f"- Total rows: {len(df):,}",
        f"- Overall hate rate: {hate_rate_overall:.3f}",
        "",
        "| Source | Rows | Hate rate |",
        "|---|---|---|",
    ]
    for source in per_source_counts.index:
        lines.append(f"| {source} | {per_source_counts[source]:,} | {per_source_rates[source]:.3f} |")

    lines += [
        "",
        "## Known caveat carried into modeling",
        "",
        "This dataset mixes one naturally-distributed source (LHSAB, ~8% "
        "positive) with two artificially-balanced sources (~59% positive "
        "each). Any train/val/test split MUST stratify by `dataset` in "
        "addition to `label` (see `src/split_data.py`), otherwise evaluation "
        "metrics will not reflect performance on the natural distribution.",
        "",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Dataset card written to %s", path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    logger.info("Loading raw sources...")
    lhsab = load_lhsab(config.RAW_LHSAB_PATH)
    ds1 = load_ds1(config.RAW_DS1_PATH)
    art = load_arabic_tweets(config.RAW_ARABIC_TWEETS_PATH)

    combined = pd.concat([lhsab, ds1, art], ignore_index=True)
    logger.info("Combined shape before cross-source cleanup: %s", combined.shape)

    combined, n_conflict_texts, n_conflict_rows = resolve_cross_source_conflicts(combined)
    logger.info("Cross-source conflicting texts: %d (dropped %d rows)", n_conflict_texts, n_conflict_rows)

    combined, n_exact_dupes = drop_exact_duplicates(combined)
    logger.info("Dropped %d cross-source exact duplicate rows", n_exact_dupes)

    logger.info("Final shape: %s", combined.shape)
    logger.info("Label distribution:\n%s", combined["label"].value_counts(normalize=True).rename(index=config.ID2LABEL))
    logger.info("Rows per source:\n%s", combined["dataset"].value_counts())
    logger.info("Hate rate per source:\n%s", combined.groupby("dataset")["label"].mean())

    combined.to_csv(config.COMBINED_RAW_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved merged dataset to %s", config.COMBINED_RAW_PATH)

    write_dataset_card(combined, n_conflict_texts, n_conflict_rows, n_exact_dupes, config.DATASET_CARD_PATH)


if __name__ == "__main__":
    main()

