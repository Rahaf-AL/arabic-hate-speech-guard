"""
Data Splitting & Dataset Preparation

Splits the cleaned dataset into train/val/test, and provides a PyTorch Dataset
class that tokenizes the text for MARBERT model.
"""
 # %%
import logging
import sys
from pathlib import Path
from transformers import AutoTokenizer
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tokenized length analysis 
# ---------------------------------------------------------------------------
 # %%
def tokenized_lengths(df: pd.DataFrame) -> pd.Series | None:

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    lengths = df["text"].astype(str).apply(lambda t: len(tokenizer.encode(t, add_special_tokens=True)))
    return lengths


def print_tokenization_samples(df: pd.DataFrame, n: int = 3) -> None:

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    sample = df["text"].astype(str).sample(min(n, len(df)), random_state=config.RANDOM_SEED)
    for text in sample:
        print("text :", text)
        print("token:", tokenizer.tokenize(text))
        print("-" * 40)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
 # %%
def make_strat_key(df: pd.DataFrame) -> pd.Series:
    return df["dataset"] + "_" + df["label"].astype(str)


def stratified_split(df: pd.DataFrame):
    strat_key = make_strat_key(df)

    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - config.TRAIN_RATIO),
        stratify=strat_key,
        random_state=config.RANDOM_SEED,
    )

    val_share = config.VAL_RATIO / (config.VAL_RATIO + config.TEST_RATIO)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_share),
        stratify=make_strat_key(temp_df),
        random_state=config.RANDOM_SEED,
    )

    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def print_split_report(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        composition = pd.crosstab(split_df["dataset"], split_df["label"], normalize="index").round(3)
        logger.info("%s: %d rows\n%s", name, len(split_df), composition)


def report_token_lengths(df: pd.DataFrame) -> None:
    lengths = tokenized_lengths(df)
    if lengths is None:
        return
    logger.info(
        "Token length: median=%.0f, 90th=%.0f, 95th=%.0f, 99th=%.0f, max=%.0f",
        lengths.median(), lengths.quantile(0.90), lengths.quantile(0.95),
        lengths.quantile(0.99), lengths.max(),
    )


def run_split() -> None:
    df = pd.read_csv(config.COMBINED_CLEAN_PATH)
    logger.info("Loaded %d rows from %s", len(df), config.COMBINED_CLEAN_PATH)

    report_token_lengths(df)
    print_tokenization_samples(df)

    train_df, val_df, test_df = stratified_split(df)
    print_split_report(train_df, val_df, test_df)

    train_df.to_csv(config.TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(config.VAL_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(config.TEST_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved train/val/test splits to %s", config.SPLITS_DIR)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
 # %%
class HateSpeechDataset:
    """Tokenizes Arabic text for MARBERT and returns tensors ready for
    training."""

    def __init__(self, texts, labels, tokenizer, max_length: int):
        self.texts = list(texts)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoding.items()}
        item["labels"] = self.labels[idx]
        return item


def build_datasets():
   
    if config.MAX_SEQ_LENGTH is None:
        raise ValueError(
          
        )

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    datasets = {}
    for name, path in [("train", config.TRAIN_PATH), ("val", config.VAL_PATH), ("test", config.TEST_PATH)]:
        split_df = pd.read_csv(path)
        datasets[name] = HateSpeechDataset(
            texts=split_df["text"],
            labels=split_df["label"],
            tokenizer=tokenizer,
            max_length=config.MAX_SEQ_LENGTH,
        )
    return datasets


def main() -> None:
    run_split()


if __name__ == "__main__":
    main()
# %%
