"""
Model Training

Fine-tunes MARBERT for binary Arabic hate speech classification: class-
weighted loss to handle the residual imbalance, optional LoRA fine-tuning,
hyperparameter search via Optuna, and live experiment tracking via
Weights & Biases. Saves the best model and training-curve plots.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from split_dataset import build_datasets  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    logger.warning("wandb not installed — training will run without live tracking.")


def wandb_is_ready() -> bool:
    if not WANDB_AVAILABLE:
        return False
    try:
        return bool(wandb.api.api_key)
    except Exception:
        return False


WANDB_READY = wandb_is_ready()


# ---------------------------------------------------------------------------
# Class weights, to counter the residual imbalance in the training set
# ---------------------------------------------------------------------------

def compute_class_weights(labels) -> torch.Tensor:
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(use_lora: bool = None):
    use_lora = config.USE_LORA if use_lora is None else use_lora
    model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)

    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=config.LORA_R,
            lora_alpha=config.LORA_ALPHA,
            lora_dropout=config.LORA_DROPOUT,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model


# ---------------------------------------------------------------------------
# Trainer with class-weighted loss
# ---------------------------------------------------------------------------

class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = nn.CrossEntropyLoss(weight=weight)(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(eval_pred) -> dict:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)

    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    _, _, f1_per_class, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f1_not_hate": f1_per_class[0],
        "f1_hate": f1_per_class[1],
    }


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_args(output_dir, learning_rate, batch_size, epochs,
                         weight_decay, warmup_ratio, run_name: str) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        seed=config.RANDOM_SEED,
        logging_steps=50,
        save_total_limit=2,
        report_to=["wandb"] if WANDB_READY else [],
        run_name=run_name,
    )


# ---------------------------------------------------------------------------
# Hyperparameter search (Optuna)
# ---------------------------------------------------------------------------

def hp_space(trial) -> dict:
    space = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
        "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [8, 16, 32]),
        "num_train_epochs": trial.suggest_int("num_train_epochs", 2, 5),
        "weight_decay": trial.suggest_float("weight_decay", 0.0, 0.1),
        "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
    }
    if config.USE_LORA:
        space["lora_r"] = trial.suggest_categorical("lora_r", [4, 8, 16])
    return space


def run_hpo(datasets, class_weights) -> dict:
    training_args = build_training_args(
        output_dir=config.CHECKPOINTS_DIR / "hpo",
        learning_rate=config.LEARNING_RATE,
        batch_size=config.BATCH_SIZE,
        epochs=config.NUM_EPOCHS,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        run_name="hpo-search",
    )

    trainer = WeightedTrainer(
        model=None,
        model_init=lambda: build_model(),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    best_run = trainer.hyperparameter_search(
        direction="maximize",
        backend="optuna",
        hp_space=hp_space,
        n_trials=config.HPO_N_TRIALS,
        compute_objective=lambda metrics: metrics["eval_f1"],
    )
    logger.info("Best hyperparameters found: %s", best_run.hyperparameters)
    return best_run.hyperparameters


# ---------------------------------------------------------------------------
# Final training run
# ---------------------------------------------------------------------------

def plot_training_curves(log_history: list, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    train_loss = [(e["epoch"], e["loss"]) for e in log_history if "loss" in e]
    eval_f1 = [(e["epoch"], e["eval_f1"]) for e in log_history if "eval_f1" in e]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if train_loss:
        epochs, losses = zip(*train_loss)
        axes[0].plot(epochs, losses, marker="o")
        axes[0].set_title("Training loss")
        axes[0].set_xlabel("Epoch")

    if eval_f1:
        epochs, scores = zip(*eval_f1)
        axes[1].plot(epochs, scores, marker="o", color="tomato")
        axes[1].set_title("Validation Macro-F1")
        axes[1].set_xlabel("Epoch")

    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved training curves to %s", out_path)


def run_final_training(datasets, class_weights, hyperparams: dict = None):
    hyperparams = hyperparams or {}
    training_args = build_training_args(
        output_dir=config.CHECKPOINTS_DIR,
        learning_rate=hyperparams.get("learning_rate", config.LEARNING_RATE),
        batch_size=hyperparams.get("per_device_train_batch_size", config.BATCH_SIZE),
        epochs=int(hyperparams.get("num_train_epochs", config.NUM_EPOCHS)),
        weight_decay=hyperparams.get("weight_decay", config.WEIGHT_DECAY),
        warmup_ratio=hyperparams.get("warmup_ratio", config.WARMUP_RATIO),
        run_name="final-training",
    )

    trainer = WeightedTrainer(
        model=build_model(),
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["val"],
        compute_metrics=compute_metrics,
        class_weights=class_weights,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer.train()

    trainer.save_model(str(config.BEST_MODEL_DIR))
    logger.info("Saved best model to %s", config.BEST_MODEL_DIR)

    plot_training_curves(trainer.state.log_history, config.TRAIN_REPORT_DIR / "training_curves.png")
    return trainer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train MARBERT for Arabic hate speech detection.")
    parser.add_argument("--hpo", action="store_true", help="Run hyperparameter search before final training.")
    parser.add_argument("--use-lora", action="store_true", help="Fine-tune with LoRA instead of full fine-tuning.")
    args = parser.parse_args()

    set_seed(config.RANDOM_SEED)
    if args.use_lora:
        config.USE_LORA = True

    os.environ.setdefault("WANDB_PROJECT", config.WANDB_PROJECT)

    datasets = build_datasets()
    class_weights = compute_class_weights(datasets["train"].labels)
    logger.info("Class weights: %s", class_weights.tolist())

    best_hyperparams = run_hpo(datasets, class_weights) if args.hpo else None
    run_final_training(datasets, class_weights, best_hyperparams)


if __name__ == "__main__":
    main()