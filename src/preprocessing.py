# %%
"""
Text Preprocessing

Cleans and normalizes the dataset.
"""
import logging
import re
import sys
from pathlib import Path
 
import pandas as pd
import pyarabic.araby as araby
 
sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)
 
URL_PATTERN = re.compile(r"https?://\S+")
NON_ARABIC_PATTERN = re.compile(
    r"[A-Za-z0-9\n،\(\)\{\}\[\]\\/+ـ_=%٪$#@!:;\-*&÷×'؛<>\"^`|~]"
)
ALEF_VARIANTS_PATTERN = re.compile("[إأآٱ]")
 # %%
 
def clean_text(text: str) -> str:
    text = araby.strip_tashkeel(text)
    text = araby.strip_tatweel(text)
 
    text = re.sub(r"\.", " ", text)
    text = URL_PATTERN.sub(" ", text)
    text = NON_ARABIC_PATTERN.sub(" ", text)
 
    text = ALEF_VARIANTS_PATTERN.sub("ا", text)
    text = re.sub(r"\?", "؟", text)
 
    return " ".join(text.split())
 
 
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text_raw"] = df["text"]
    df["text"] = df["text_raw"].astype(str).apply(clean_text)
 
    before = len(df)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    logger.info("Dropped %d rows that became empty after cleaning.", before - len(df))
 
    before = len(df)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    logger.info("Dropped %d duplicate rows created by cleaning.", before - len(df))
 
    return df[["text_raw", "text", "label", "dataset"]]
 
 
def print_before_after_samples(df: pd.DataFrame, n: int = 5) -> None:
    sample = df.sample(min(n, len(df)), random_state=config.RANDOM_SEED)
    for _, row in sample.iterrows():
        print("befor :", row["text_raw"], row["label"])
        print("after :", row["text"],row["label"])
        print("-" * 40)
 
 
def main() -> None:
    df = pd.read_csv(config.COMBINED_RAW_PATH)
    logger.info("Loaded %d rows from %s", len(df), config.COMBINED_RAW_PATH)
 
    df_clean = preprocess(df)
    logger.info("Final shape after preprocessing: %s", df_clean.shape)
 
    print_before_after_samples(df_clean)
 
    df_clean.to_csv(config.COMBINED_CLEAN_PATH, index=False, encoding="utf-8-sig")
    logger.info("Saved cleaned dataset to %s", config.COMBINED_CLEAN_PATH)
 
 
if __name__ == "__main__":
    main()

