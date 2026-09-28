"""
Inference App

Gradio interface for the Arabic hate speech classifier.
"""

import argparse
import os
import sys
from pathlib import Path
 
import gradio as gr
 
sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from inference import classify_text, load_classifier  # noqa: E402
 

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700&display=swap');
 
:root {
    /* Light-green palette throughout (background, borders, accents) instead
       of the earlier warm-beige tone — the alert color stays red/orange on
       purpose so a flagged result still stands out against all the green. */
    --app-bg: #F1F8F4;
    --app-surface: #FFFFFF;
    --app-border: #D7E8DC;
    --app-text: #1F3D2B;
    --app-muted: #5C7268;
    --app-safe: #2F8F5B;
    --app-safe-soft: #E3F5EA;
    --app-alert: #B8492F;
    --app-alert-soft: #FBEAE4;
}
/* Dark-mode palette, applied when our own toggle button adds .dark-mode to
   <html> (see the script below) — every rule above reads these same
   variable names, so nothing else needs duplicating for dark mode. */
:root.dark-mode {
    --app-bg: #10231A;
    --app-surface: #16301F;
    --app-border: #234631;
    --app-text: #E9F5EE;
    --app-muted: #9FB8AA;
    --app-safe: #4CC788;
    --app-safe-soft: #1E3C2B;
    --app-alert: #E4795A;
    --app-alert-soft: #3B211A;
}
/* Fill the actual laptop/desktop viewport instead of leaving a narrow card
   floating with large empty margins on the sides and a dead zone below it —
   the content block is centered vertically within the full window height. */
html, body {
    background: var(--app-bg) !important;
    min-height: 100vh;
}
.gradio-container {
    background: var(--app-bg) !important;
    min-height: 100vh !important;
    max-width: 700px !important;
    margin: 0 auto !important;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    /* The whole layout reads right-to-left, so the input card (the first
       thing a visitor should notice) lands on the right and the result
       card on the left — the natural reading order for Arabic. */
    direction: rtl !important;
    font-family: 'Tajawal', 'Segoe UI', Tahoma, sans-serif !important;
}
#header-block { position: relative; text-align: center; padding: 20px 0 8px; }
#header-block h1 { color: var(--app-text); margin-bottom: 4px; font-size: 28px; }
#header-block p { color: var(--app-muted); margin-top: 0; font-size: 15px; }
 
/* Our own light/dark toggle button — a fixed circle in the header corner,
   styled to match the rest of the app instead of Gradio's default icon. */
#theme-toggle-btn {
    position: absolute;
    top: 4px;
    right: 0;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    border: 1px solid var(--app-border);
    background: var(--app-surface);
    font-size: 16px;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
}
 
/* Give the input side the same "card" look as the result panel, so the two
   halves of the layout read as a matched pair instead of one styled box
   next to plain Gradio defaults. */
#input-panel, #result-panel {
    background: var(--app-surface);
    border: 1px solid var(--app-border);
    border-radius: 12px;
    padding: 18px 20px;
}
#input-panel { display: flex; flex-direction: column; gap: 12px; margin-bottom: 16px; }
 
/* Section titles ("النص" / "النتيجة") were barely visible before — same
   pale gray as the placeholder text. Make them a clear visual anchor: bigger,
   bold, and set off with a short accent underline. */
#panel-title {
    font-weight: 700;
    color: var(--app-text);
    margin: 0 0 8px;
    font-size: 16px;
    padding-bottom: 6px;
    border-bottom: 2px solid var(--app-safe-soft);
    text-align: right;
}
/* Gradio's Markdown component wraps content in a ".prose" block that sets
   its own left alignment regardless of the page's RTL direction — override
   it directly so "النص" / "النتيجة" sit on the right like the rest of the
   Arabic layout. */
#panel-title.prose, #panel-title .prose { text-align: right !important; }
 
/* Force a light textbox regardless of the visitor's OS/browser dark-mode
   setting — Gradio's default theme otherwise switches the textarea to a
   dark background, which clashed with the rest of the light page. */
textarea, input[type="text"] {
    background: var(--app-surface) !important;
    color: var(--app-text) !important;
    border-color: var(--app-border) !important;
}
 
/* The footer settings/branding row ("Built with Gradio", "Use via API") — a
   plain, stable selector, safe to hide by CSS alone. */
footer { display: none !important; }
 
.result-empty { color: var(--app-muted); font-size: 14px; padding: 6px 0; }
.result-row { display: flex; align-items: center; gap: 10px; margin-bottom: 10px; }
.result-row:last-child { margin-bottom: 0; }
.result-label { width: 90px; font-weight: 600; font-size: 14px; color: var(--app-text); }
.result-bar-track { flex: 1; height: 10px; border-radius: 999px; background: var(--app-border); overflow: hidden; }
.result-bar-fill { height: 100%; border-radius: 999px; transition: width 0.3s ease; }
.result-bar-fill.safe { background: var(--app-safe); }
.result-bar-fill.alert { background: var(--app-alert); }
.result-pct { width: 52px; text-align: left; font-family: monospace; font-size: 13px; color: var(--app-muted); }
.result-verdict { margin-top: 12px; padding: 8px 12px; border-radius: 8px; font-weight: 600; font-size: 14px; text-align: center; }
.result-verdict.safe { background: var(--app-safe-soft); color: var(--app-safe); }
.result-verdict.alert { background: var(--app-alert-soft); color: var(--app-alert); }
"""
 
# Two things injected into <head>:
# 1) Gradio's own corner icon (a light/dark-mode toggle it injects
#    automatically next to the title) doesn't have a stable class name across
#    Gradio versions, so CSS alone can't target it reliably — this scans every
#    button after the page loads and hides any whose accessible name mentions
#    theme/dark/light/fullscreen, catching it however Gradio renders it.
# 2) Our own replacement toggle (the #theme-toggle-btn in the header) flips a
#    .dark-mode class on <html>, which the CSS variables above key off of, and
#    remembers the choice in localStorage so it persists across visits.
HIDE_GRADIO_CHROME_JS = """
<script>
window.addEventListener('load', () => {
    const keywords = ['theme', 'dark', 'light', 'fullscreen', 'settings'];
    document.querySelectorAll('button').forEach((btn) => {
        if (btn.id === 'theme-toggle-btn') return;
        const name = (btn.getAttribute('aria-label') || btn.getAttribute('title') || '').toLowerCase();
        if (keywords.some((k) => name.includes(k))) {
            btn.style.display = 'none';
        }
    });
 
    const applyTheme = (isDark) => {
        document.documentElement.classList.toggle('dark-mode', isDark);
        const btn = document.getElementById('theme-toggle-btn');
        if (btn) btn.textContent = isDark ? '☀️' : '🌙';
    };
 
    let saved = null;
    try { saved = localStorage.getItem('appTheme'); } catch (e) {}
    applyTheme(saved === 'dark');
 
    window.toggleAppTheme = () => {
        const isDark = !document.documentElement.classList.contains('dark-mode');
        applyTheme(isDark);
        try { localStorage.setItem('appTheme', isDark ? 'dark' : 'light'); } catch (e) {}
    };
});
</script>
"""
 
 
def render_result_html(probs: dict) -> str:
    if probs is None:
        return '<div class="result-empty">.</div>'
 
    not_hate_pct = probs[config.ID2LABEL[0]] * 100
    hate_pct = probs[config.ID2LABEL[1]] * 100
    verdict_class = "alert" if hate_pct >= 50 else "safe"
    verdict_text = "⚠ تصنيف: خطاب عدائي" if verdict_class == "alert" else "✓ تصنيف: كلام طبيعي"
 
    return f"""
    <div class="result-row">
        <span class="result-label">كلام طبيعي</span>
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
    # title stays on Blocks() (that's still where Gradio 6 reads it from);
    # theme/css move to demo.launch() below per the Gradio 6 deprecation
    # warning ("parameters have been moved from the Blocks constructor to
    # the launch() method").
    with gr.Blocks(title="واعــي", head=HIDE_GRADIO_CHROME_JS) as demo:
        with gr.Column(elem_id="header-block"):
            gr.HTML(
                '<button id="theme-toggle-btn" onclick="toggleAppTheme()" '
                'title="تبديل الوضع الفاتح/الداكن">🌙</button>'
            )
            gr.Markdown("# واعــي")
            gr.Markdown("هذا النموذج مُعدّ للتعرف على خطاب الكراهية الموجّه ضد عرق أو جنسية أو دين أو إعاقة أو طبقة اجتماعية معينة")
 
        # Stacked vertically (input card, then the result card right below
        # it) instead of side by side — a single reading path top to bottom.
        with gr.Column(elem_id="input-panel"):
            gr.Markdown("النص", elem_id="panel-title")
            text_input = gr.Textbox(
                show_label=False,
                placeholder="أدخل نصًا عربيًا هنا...",
                rtl=True,
                lines=5,
                autofocus=True,
            )
            submit_btn = gr.Button("صنّف النص", variant="primary")
 
        with gr.Column(elem_id="result-panel"):
            gr.Markdown("النتيجة", elem_id="panel-title")
            result_html = gr.HTML(render_result_html(None))
 
        submit_btn.click(fn=predict_fn, inputs=text_input, outputs=result_html)
        text_input.submit(fn=predict_fn, inputs=text_input, outputs=result_html)
 
    return demo
 

def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the hate speech classifier demo.")
    parser.add_argument(
        "--model-dir", type=str,
        default=os.environ.get("MODEL_DIR", "rahaf088/arabic-hate-speech-marbert-lora"),
        help="Local model directory or Hugging Face Hub repo id to serve.",
    )
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()
 
    tokenizer, model = load_classifier(args.model_dir)
    predict_fn = build_predict_fn(tokenizer, model)
    demo = build_interface(predict_fn)
 
    # Render/Railway assign the port to listen on dynamically via the PORT
    # env var — Docker/local runs don't set it, so 7860 stays the default.
    port = int(os.environ.get("PORT", 7860))
 
    theme = gr.themes.Soft(primary_hue="green", neutral_hue="stone")
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=args.share,
        theme=theme,
        css=CUSTOM_CSS,
    )



 
if __name__ == "__main__":
    main()
 