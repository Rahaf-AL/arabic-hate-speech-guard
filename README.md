# Arabic Hate Speech Detection — واعــي (Wa'ei)

A hate speech classifier for Arabic text (Modern Standard Arabic and dialects), built by fine-tuning **MARBERT** — a BERT model pretrained specifically on Arabic — with **LoRA** for parameter-efficient fine-tuning. The project covers the full pipeline: data collection and cleaning, training with class-weighted loss and hyperparameter search, evaluation against a held-out test set, and an interactive Arabic-language demo app.

The trained model is published on the Hugging Face Hub: [`rahaf088/arabic-hate-speech-marbert-lora`](https://huggingface.co/rahaf088/arabic-hate-speech-marbert-lora).

## Why this project

Most hate speech detection tools are built for English and perform poorly on Arabic, where a single idea can be written in Modern Standard Arabic or any of several dialects, often mixed in the same sentence. This project trains a classifier specifically on Arabic text to flag content directed at a person or group based on ethnicity, nationality, religion, disability, or social class.

## How it works, end to end

```
Raw datasets (L-HSAB, ds1, Arabic Tweets)
        │
        ▼
  Combine + clean (src/preprocess.py)
        │
        ▼
  Stratified train / val / test split (src/split_dataset.py)
        │
        ▼
  Fine-tune MARBERT + LoRA, class-weighted loss (src/train.py)
        │
        ▼
  Evaluate on held-out test set (src/evaluate.py)
        │
        ▼
  Serve predictions through a Gradio app (src/app.py)
```

### 1. Data

Three Arabic datasets are combined into one corpus:

- **L-HSAB** — originally labeled `hate` / `abusive` / `normal`. `abusive` and `hate` are merged into a single `hate` class, `normal` becomes `not_hate`, giving a binary label.
- `training_ds1.csv`
- `Arabic_Tweets_dataset.csv`

### 2. Preprocessing (`src/preprocess.py`)

Each piece of text goes through:

- Removing diacritics (tashkeel) and letter elongation (tatweel)
- Stripping URLs and non-Arabic characters (Latin letters, digits, punctuation)
- Normalizing alef variants (`إ أ آ ٱ` → `ا`) so the same word isn't treated as several different tokens
- Normalizing question marks to the Arabic form (`؟`)
- Dropping rows that become empty after cleaning, and duplicate rows created by cleaning


### 3. Training (`src/train.py`)

- **Base model:** `UBC-NLP/MARBERT`
- **Fine-tuning method:** LoRA (Low-Rank Adaptation) — instead of updating all of MARBERT's parameters, LoRA freezes the base model and trains small additional matrices, which is much cheaper in memory and compute while reaching comparable accuracy. Configurable to fall back to full fine-tuning.
- **Class imbalance:** the dataset has more `not_hate` than `hate` examples, so the loss function (`WeightedTrainer`, a custom `Trainer` subclass) weights each class inversely to its frequency, so the model isn't biased toward the majority class.
- **Hyperparameter search:** an Optuna-backed search (`run_hpo`) tunes learning rate, batch size, epoch count, weight decay, warmup ratio, and (when using LoRA) the LoRA rank — optimizing for validation macro-F1.

### 4. Evaluation (`src/evaluate.py`)

Loads one or more trained model variants (full fine-tune, LoRA, HPO-tuned) and evaluates each on the **test set only** — data none of them saw during training or checkpoint selection. For each variant it reports:

- Accuracy, precision, recall, macro-F1
- Per-class F1 (`not_hate`, `hate`)
- A confusion matrix plot
- A `comparison.csv` ranking all evaluated variants by test F1

### 6. Inference & demo app (`src/inference.py`, `src/app.py`)

`inference.py` is the shared prediction logic: it loads the tokenizer and model (auto-detecting whether the model directory is a LoRA adapter or a full fine-tune), cleans the input text with the same preprocessing used during training, and returns a probability for each class.

`app.py` wraps this in **"واعــي" (Wa'ei)** — an Arabic-first [Gradio](https://gradio.app) interface: a right-to-left layout, a text box for the input sentence, and a result panel showing the probability of each class as a bar plus a plain-language verdict ("✓ normal text" / "⚠ hateful speech"). It includes a light/dark theme toggle and defaults to loading the model straight from the Hugging Face Hub, so it runs with no local model files.

There's also a small classroom-monitoring demo (`src/monitor.py`) that raises an alert if the same person sends more than a configurable number of flagged messages.

## Project structure

```
arabic-hate-speech-mlops/
├── src/
│   ├── data_ingest.py       # loads and combines the raw datasets
│   ├── preprocessing.py        # text cleaning (clean_text, preprocess)
│   ├── split_dataset.py     # stratified train/val/test split + PyTorch Dataset
│   ├── train.py             # LoRA/full fine-tuning, class weighting, HPO
│   ├── evaluate.py          # test-set evaluation, confusion matrices
│   ├── inference.py         # shared load + predict helpers
│   ├── app.py               # Gradio demo app ("واعــي")
│   ├── monitor.py           # classroom monitoring demo
│   ├── eda.py                # exploratory data analysis
│   └── config.py            # single source of truth for paths & hyperparameters
├── data/                    # raw / processed / split datasets (not versioned)
├── models/                  # trained checkpoints (not versioned)
├── reports/                 # EDA, evaluation, and training reports/plots
├── Dockerfile
└── requirements.txt
```

## Running it locally


```bash
pip install -r requirements.txt

# Train (optionally with LoRA and a hyperparameter search)
python src/train.py --use-lora --hpo

# Evaluate every trained variant against the test set
python src/evaluate.py --all

# Launch the demo app — pulls the trained model from the Hugging Face Hub by default
python src/app.py
```



## Author

Rahaf Alruwaithi —
Model on Hugging Face: [rahaf088/arabic-hate-speech-marbert-lora](https://huggingface.co/rahaf088/arabic-hate-speech-marbert-lora)
