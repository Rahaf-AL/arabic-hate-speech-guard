"""
Model Evaluation
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from split_dataset import build_datasets  # noqa: E402
from train import compute_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loading 
# ---------------------------------------------------------------------------

def load_model(model_dir):
    
    model_dir = str(model_dir)

    from peft import PeftConfig, PeftModel
    try:
        PeftConfig.from_pretrained(model_dir)
        is_lora_adapter = True
    except Exception:
        is_lora_adapter = False

    if is_lora_adapter:
        base_model = AutoModelForSequenceClassification.from_pretrained(config.MODEL_NAME, num_labels=2)
        model = PeftModel.from_pretrained(base_model, model_dir)
        model = model.merge_and_unload()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2)

    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def build_eval_args() -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(config.CHECKPOINTS_DIR / "eval_tmp"),
        per_device_eval_batch_size=config.BATCH_SIZE,
        report_to=[],
    )


def evaluate_model(model_dir: Path, test_dataset):
    model = load_model(model_dir)
    trainer = Trainer(model=model, args=build_eval_args(), compute_metrics=compute_metrics)

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids
    return predictions.metrics, preds, labels


def plot_confusion_matrix(labels, preds, name: str, out_path: Path) -> None:
    cm = confusion_matrix(labels, preds)
    class_names = [config.ID2LABEL[0], config.ID2LABEL[1]]

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], class_names)
    ax.set_yticks([0, 1], class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {name}")

    threshold = cm.max() / 2
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > threshold else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def run_evaluation(model_dir: Path, name: str, test_dataset) -> dict:
    logger.info("Evaluating '%s' from %s", name, model_dir)
    metrics, preds, labels = evaluate_model(model_dir, test_dataset)

    plot_confusion_matrix(labels, preds, name, config.EVAL_REPORT_DIR / f"confusion_matrix_{name}.png")

    metrics_path = config.EVAL_REPORT_DIR / f"metrics_{name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Saved metrics to %s", metrics_path)

    return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved model variants on the test set.")
    parser.add_argument("--model-dir", type=str, help="Path to a single model directory to evaluate.")
    parser.add_argument("--name", type=str, help="Label for the model given via --model-dir.")
    parser.add_argument("--all", action="store_true",
                         help="Evaluate every variant in config.MODEL_VARIANTS that exists on disk.")
    args = parser.parse_args()

    datasets = build_datasets()
    test_dataset = datasets["test"]

    if args.all:
        targets = {name: path for name, path in config.MODEL_VARIANTS.items() if Path(path).exists()}
        if not targets:
            raise FileNotFoundError(
                "No model variants found under config.MODEL_VARIANTS. "
                "Back up a trained model to one of those paths first."
            )
    elif args.model_dir:
        targets = {args.name or Path(args.model_dir).name: Path(args.model_dir)}
    else:
        targets = {"best_model": config.BEST_MODEL_DIR}

    results = {name: run_evaluation(path, name, test_dataset) for name, path in targets.items()}

    comparison = pd.DataFrame(results).T
    comparison = comparison.sort_values("test_f1", ascending=False)
    comparison_path = config.EVAL_REPORT_DIR / "comparison.csv"
    comparison.to_csv(comparison_path)
    logger.info("Comparison across variants:\n%s", comparison)
    logger.info("Saved comparison table to %s", comparison_path)


if __name__ == "__main__":
    main()