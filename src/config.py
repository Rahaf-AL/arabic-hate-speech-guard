"""
Central configuration for the Arabic Hate Speech Detection project.

"""
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Random seed used for any splitting/sampling across the project.
RANDOM_SEED = 42

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"

RAW_LHSAB_PATH = RAW_DIR / "L-HSAB"
RAW_DS1_PATH = RAW_DIR / "training_ds1.csv"
RAW_ARABIC_TWEETS_PATH = RAW_DIR / "Arabic_Tweets_dataset.csv"

COMBINED_RAW_PATH = RAW_DIR / "combined_dataset.csv"
COMBINED_CLEAN_PATH = PROCESSED_DIR / "combined_clean.csv"
TRAIN_PATH = SPLITS_DIR / "train.csv"
VAL_PATH = SPLITS_DIR / "val.csv"
TEST_PATH = SPLITS_DIR / "test.csv"

# Reports and documentation artifacts
REPORTS_DIR = PROJECT_ROOT / "reports"
EDA_REPORT_DIR = REPORTS_DIR / "eda"
DATASET_CARD_PATH = DATA_DIR / "DATASET_CARD.md"

# Label schema
LABEL2ID = {"not_hate": 0, "hate": 1}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

# Maps L-HSAB's original 3-class scheme to the binary schema above.
LHSAB_LABEL_MAP = {"hate": 1, "abusive": 1, "normal": 0}

# Model
MODEL_NAME = "UBC-NLP/MARBERT"

# Maximum sequence length for tokenization, set after analyzing the actual
# tokenized length distribution of the cleaned corpus 
MAX_SEQ_LENGTH = 40

# Train / validation / test split ratios (must sum to 1.0)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Runtime environment
# ---------------------------------------------------------------------------
try:
    import google.colab  # noqa: F401
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

# On Colab, model outputs go to a Drive folder so they survive session
# disconnects. Locally, they go to the project's own models/ folder.
if IS_COLAB:
    COLAB_DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/arabic-hate-speech-mlops")
    MODELS_DIR = COLAB_DRIVE_PROJECT_DIR / "models"
else:
    MODELS_DIR = PROJECT_ROOT / "models"

CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
BEST_MODEL_DIR = MODELS_DIR / "best_model"
TRAIN_REPORT_DIR = REPORTS_DIR / "train"

# ---------------------------------------------------------------------------
# Training hyperparameters (defaults; overridden during hyperparameter search)
# ---------------------------------------------------------------------------
LEARNING_RATE = 2e-5
BATCH_SIZE = 16
NUM_EPOCHS = 4
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1

# ---------------------------------------------------------------------------
# LoRA (parameter-efficient fine-tuning), used as an alternative to full
# fine-tuning
# ---------------------------------------------------------------------------
USE_LORA = False
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.1

# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
HPO_N_TRIALS = 15

# ---------------------------------------------------------------------------
# Experiment tracking
# ---------------------------------------------------------------------------
WANDB_PROJECT = "arabic-hate-speech-marbert"

# Create the project directory structure if it doesn't already exist.
for _dir in (RAW_DIR, PROCESSED_DIR, SPLITS_DIR, EDA_REPORT_DIR,
             MODELS_DIR, CHECKPOINTS_DIR, BEST_MODEL_DIR, TRAIN_REPORT_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
