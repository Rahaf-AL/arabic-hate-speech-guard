"""
Inference App

Gradio interface for the Arabic hate speech classifier. applies the same cleaning
used during training, and returns a hate/not_hate probability for any
Arabic text typed in.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from evaluate import load_model  # noqa: E402
from preprocessing  import clean_text  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

EXAMPLES = [
    "صباح الخير يا جماعة، يومكم سعيد ان شاء الله",
    "والله ما شاء الله عليك شغل رائع ومجهود تشكر عليه",
    "روح يا حقير ما تسوى شي",
]


def load_classifier(model_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = load_model(model_dir)
    model.eval()
    return tokenizer, model


def build_predict_fn(tokenizer, model):
    def predict(text: str) -> dict:
        if not text or not text.strip():
            return {config.ID2LABEL[0]: 0.0, config.ID2LABEL[1]: 0.0}

        cleaned = clean_text(text)
        inputs = tokenizer(
            cleaned,
            truncation=True,
            padding="max_length",
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        return {config.ID2LABEL[i]: probs[i] for i in range(len(probs))}

    return predict


def build_interface(predict_fn):
    import gradio as gr

    return gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(
            label="النص",
            placeholder="اكتبي نصًا عربيًا هنا...",
            rtl=True,
            lines=3,
        ),
        outputs=gr.Label(label="التصنيف", num_top_classes=2),
        title="مصنّف الخطاب العدائي العربي",
        description="نموذج MARBERT مُدرَّب لتصنيف النص العربي إلى hate أو not_hate.",
        examples=EXAMPLES,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the hate speech classifier demo.")
    parser.add_argument("--model-dir", type=str, default=str(config.BEST_MODEL_DIR),
                         help="Path to the model directory to serve.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    tokenizer, model = load_classifier(Path(args.model_dir))
    predict_fn = build_predict_fn(tokenizer, model)
    demo = build_interface(predict_fn)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)


if __name__ == "__main__":
    main()
