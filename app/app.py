"""
Inference App

Gradio interface for the Arabic hate speech classifier. Loads a trained
model ( LoRA), applies the same cleaning used during training, and returns
a hate/not_hate probability for any Arabic text typed in.
"""
import argparse
import sys
from pathlib import Path
 
sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from inference import classify_text, load_classifier  # noqa: E402
 
EXAMPLES = [
    "صباح الخير يا جماعة، يومكم سعيد ان شاء الله",
    "والله ما شاء الله عليك شغل رائع ومجهود تشكر عليه",
    "روح يا حقير ما تسوى شي",
]
 
CUSTOM_CSS = """
:root {
    --app-bg: #FAF8F3;
    --app-surface: #FFFFFF;
    --app-border: #E7E2D6;
    --app-text: #24301F;
    --app-muted: #6B7566;
    --app-safe: #2F8F5B;
    --app-safe-soft: #E7F5EC;
    --app-alert: #B8492F;
    --app-alert-soft: #FBEAE4;
}
.gradio-container { background: var(--app-bg) !important; }
#header-block { text-align: center; padding: 8px 0 4px; }
#header-block h1 { color: var(--app-text); margin-bottom: 4px; }
#header-block p { color: var(--app-muted); margin-top: 0; }
#result-panel {
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: 12px;
    padding: 18px 20px;
}
.result-empty { color: var(--app-muted); font-size: 14px; }
.result-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.result-row:last-child { margin-bottom: 0; }
.result-label { width: 90px; font-weight: 600; font-size: 14px; color: var(--app-text); }
.result-bar-track { flex: 1; height: 10px; border-radius: 999px; background: var(--app-border); overflow: hidden; }
.result-bar-fill { height: 100%; border-radius: 999px; }
.result-bar-fill.safe { background: var(--app-safe); }
.result-bar-fill.alert { background: var(--app-alert); }
.result-pct { width: 52px; text-align: left; font-family: monospace; font-size: 13px; color: var(--app-muted); }
.result-verdict { margin-top: 12px; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 14px; }
.result-verdict.safe { background: var(--app-safe-soft); color: var(--app-safe); }
.result-verdict.alert { background: var(--app-alert-soft); color: var(--app-alert); }
"""
 
 
def render_result_html(probs: dict) -> str:
    if probs is None:
        return '<div class="result-empty">اكتبي نصًا واضغطي "صنّفي النص" لرؤية النتيجة.</div>'
 
    not_hate_pct = probs[config.ID2LABEL[0]] * 100
    hate_pct = probs[config.ID2LABEL[1]] * 100
    verdict_class = "alert" if hate_pct >= 50 else "safe"
    verdict_text = "⚠ تصنيف: رسالة إساءة" if verdict_class == "alert" else "✓ تصنيف: كلام طبيعي"
 
    return f"""
    <div class="result-row">
        <span class="result-label">غير عدائي</span>
        <span class="result-bar-track"><span class="result-bar-fill safe" style="width:{not_hate_pct:.1f}%"></span></span>
        <span class="result-pct">{not_hate_pct:.1f}%</span>
    </div>
    <div class="result-row">
        <span class="result-label">عدائي</span>
        <span class="result-bar-track"><span class="result-bar-fill alert" style="width:{hate_pct:.1f}%"></span></span>
        <span class="result-pct">{hate_pct:.1f}%</span>
    </div>
    <div class="result-verdict {verdict_class}">{verdict_text}</div>
    """
 
 
def build_predict_fn(tokenizer, model):
    def predict(text: str) -> str:
        if not text or not text.strip():
            return render_result_html(None)
        return render_result_html(classify_text(tokenizer, model, text))
 
    return predict
 
 
def build_interface(predict_fn):
    import gradio as gr
 
    theme = gr.themes.Soft(primary_hue="green", neutral_hue="stone")
 
    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="مصنّف الرسائل المسيئة باللغة العربية") as demo:
        with gr.Column(elem_id="header-block"):
            gr.Markdown("# مصنّف الرسائل المسيئة باللغة العربية")
            gr.Markdown("أداة ذكاء اصطناعي تقرأ أي نص عربي وتحدد إذا كان يحتوي على عبارات إساءة .")
 
        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="النص",
                    rtl=True,
                    lines=4,
                )
                submit_btn = gr.Button("صنّف النص", variant="primary")
                gr.Examples(examples=EXAMPLES, inputs=text_input, label="أمثلة جاهزة")
 
            with gr.Column(scale=1, elem_id="result-panel"):
                gr.Markdown("**النتيجة**")
                result_html = gr.HTML(render_result_html(None))
 
        submit_btn.click(fn=predict_fn, inputs=text_input, outputs=result_html)
        text_input.submit(fn=predict_fn, inputs=text_input, outputs=result_html)
 
    return demo
 
 
def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the hate speech classifier demo.")
    parser.add_argument("--model-dir", type=str, default=str(config.BEST_MODEL_DIR),
                         help="Local model directory or Hugging Face Hub repo id to serve.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()
 
    tokenizer, model = load_classifier(args.model_dir)
    predict_fn = build_predict_fn(tokenizer, model)
    demo = build_interface(predict_fn)
    demo.launch(server_name="0.0.0.0", server_port=7860, share=args.share)
 
 
if __name__ == "__main__":
    main()
 