"""
Exploratory Data Analysis

Runs lightweight statistical and lexical analysis on the merged dataset and
saves the resulting plots under reports/eda/. This script only produces
figures and console/log output — it does not generate the written EDA
report; that is a separate step once the results here have been reviewed.
"""
# %%
import logging
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# A small, fixed Arabic stopword list (no external download required).
ARABIC_STOPWORDS = {
    "من", "الى", "إلى", "عن", "على", "في", "و", "او", "أو", "ثم", "ان", "أن",
    "إن", "كان", "كانت", "يكون", "هذا", "هذه", "ذلك", "تلك", "هو", "هي",
    "هم", "انت", "أنت", "انا", "أنا", "نحن", "لا", "ما", "لم", "لن", "لكن",
    "مع", "بعد", "قبل", "عند", "كل", "بعض", "غير", "كما", "حتى", "اذا", "إذا",
    "التي", "الذي", "الذين", "هناك", "هنالك", "قد", "لقد", "له", "لها",
    "لهم", "به", "بها", "فيه", "فيها", "ذا", "الا", "إلا",
}

# Unicode ranges covering the majority of emoji blocks.
EMOJI_PATTERN = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF]"
)
LATIN_PATTERN = re.compile(r"[A-Za-z]")
ARABIC_PATTERN = re.compile(r"[؀-ۿ]")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")  # same character repeated 3+ times
REPEATED_PUNCT_PATTERN = re.compile(r"[!؟?]{2,}")
URL_PATTERN = re.compile(r"https?://\S+")
MENTION_PATTERN = re.compile(r"@\w+")
HASHTAG_PATTERN = re.compile(r"#(\w+)")


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.COMBINED_RAW_PATH)
    logger.info("Loaded %s rows from %s", len(df), config.COMBINED_RAW_PATH)
    return df


def basic_overview(df: pd.DataFrame) -> None:
    logger.info("Shape: %s", df.shape)
    logger.info("Missing values:\n%s", df.isna().sum())
    logger.info("Exact duplicate texts: %d", df["text"].duplicated().sum())


def plot_class_distribution(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    counts = df["label"].value_counts().sort_index()
    labels = [config.ID2LABEL[i] for i in counts.index]
    axes[0].bar(labels, counts.values, color=["steelblue", "tomato"])
    axes[0].set_title("Overall class distribution")
    for i, v in enumerate(counts.values):
        axes[0].text(i, v, f"{v:,}", ha="center", va="bottom")

    source_label = pd.crosstab(df["dataset"], df["label"])
    source_label.plot(kind="bar", stacked=True, ax=axes[1], color=["steelblue", "tomato"])
    axes[1].set_title("Class distribution per source")
    axes[1].legend(["not_hate", "hate"])

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved class distribution plot to %s", out_path)


def tokenized_lengths(df: pd.DataFrame) -> pd.Series | None:
    try:
        from transformers import AutoTokenizer
    except ImportError:
        logger.warning("transformers not installed — skipping tokenized length analysis.")
        return None

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    lengths = df["text"].astype(str).apply(lambda t: len(tokenizer.encode(t, add_special_tokens=True)))
    return lengths


def plot_token_length_distribution(lengths: pd.Series, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lengths.clip(upper=lengths.quantile(0.99)).hist(bins=50, ax=ax, color="steelblue", edgecolor="white")
    for pct, style in [(0.90, "--"), (0.95, "-.")]:
        ax.axvline(lengths.quantile(pct), color="red", linestyle=style, label=f"{int(pct * 100)}th pct")
    ax.set_title("MARBERT token count distribution")
    ax.set_xlabel("Tokens per example")
    ax.legend()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved token length plot to %s", out_path)
    logger.info(
        "Token length percentiles: median=%.0f, 90th=%.0f, 95th=%.0f, 99th=%.0f, max=%.0f",
        lengths.median(), lengths.quantile(0.90), lengths.quantile(0.95),
        lengths.quantile(0.99), lengths.max(),
    )


def character_level_features(df: pd.DataFrame) -> pd.DataFrame:
    text = df["text"].astype(str)
    features = pd.DataFrame({
        "label": df["label"],
        "has_emoji": text.str.contains(EMOJI_PATTERN),
        "has_elongation": text.apply(lambda t: bool(ELONGATION_PATTERN.search(t))),
        "has_repeated_punct": text.str.contains(REPEATED_PUNCT_PATTERN),
        "has_url": text.str.contains(URL_PATTERN),
        "has_mention": text.str.contains(MENTION_PATTERN),
        "has_hashtag": text.apply(lambda t: bool(HASHTAG_PATTERN.search(t))),
        "latin_char_ratio": text.apply(lambda t: len(LATIN_PATTERN.findall(t)) / max(len(t), 1)),
        "arabic_char_ratio": text.apply(lambda t: len(ARABIC_PATTERN.findall(t)) / max(len(t), 1)),
    })
    return features


def plot_character_features(features: pd.DataFrame, out_path: Path) -> None:
    flags = ["has_emoji", "has_elongation", "has_repeated_punct", "has_url", "has_mention", "has_hashtag"]
    rates = features.groupby("label")[flags].mean().rename(index=config.ID2LABEL)

    fig, ax = plt.subplots(figsize=(9, 5))
    rates.T.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Character-level signal rate by class")
    ax.set_ylabel("Fraction of examples")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved character-level features plot to %s", out_path)


def low_arabic_ratio_check(features: pd.DataFrame, threshold: float = 0.5) -> int:
    n_low = (features["arabic_char_ratio"] < threshold).sum()
    logger.info("Texts with Arabic character ratio below %.0f%%: %d", threshold * 100, n_low)
    return n_low


def canonicalize(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text)
    text = ELONGATION_PATTERN.sub(r"\1", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def near_duplicate_check(df: pd.DataFrame) -> int:
    canonical = df["text"].astype(str).apply(canonicalize)
    group_sizes = canonical.groupby(canonical).transform("size")
    n_groups_with_dupes = (group_sizes > 1).sum()
    logger.info("Rows sharing a canonicalized form with at least one other row: %d", n_groups_with_dupes)
    return n_groups_with_dupes


def hashtag_label_correlation(df: pd.DataFrame, min_count: int = 20, skew_threshold: float = 0.9) -> pd.DataFrame:
    hashtag_rows = []
    for text, label in zip(df["text"].astype(str), df["label"]):
        for tag in HASHTAG_PATTERN.findall(text):
            hashtag_rows.append((tag.lower(), label))

    if not hashtag_rows:
        logger.info("No hashtags found.")
        return pd.DataFrame(columns=["hashtag", "count", "hate_rate"])

    tags_df = pd.DataFrame(hashtag_rows, columns=["hashtag", "label"])
    stats = tags_df.groupby("hashtag")["label"].agg(["count", "mean"]).rename(columns={"mean": "hate_rate"})
    flagged = stats[(stats["count"] >= min_count) & ((stats["hate_rate"] >= skew_threshold) | (stats["hate_rate"] <= 1 - skew_threshold))]
    flagged = flagged.sort_values("count", ascending=False)
    logger.info("Hashtags strongly correlated with one class (count>=%d): %d\n%s", min_count, len(flagged), flagged.head(20))
    return flagged.reset_index()


def main() -> None:
    df = load_data()
    basic_overview(df)

    plot_class_distribution(df, config.EDA_REPORT_DIR / "class_distribution.png")

    lengths = tokenized_lengths(df)
    if lengths is not None:
        plot_token_length_distribution(lengths, config.EDA_REPORT_DIR / "token_length_distribution.png")

    features = character_level_features(df)
    plot_character_features(features, config.EDA_REPORT_DIR / "character_features.png")
    low_arabic_ratio_check(features)

    near_duplicate_check(df)
    hashtag_label_correlation(df)


if __name__ == "__main__":
    main()

# %%
