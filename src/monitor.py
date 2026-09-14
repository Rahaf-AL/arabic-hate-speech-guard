"""
Classroom Monitoring Demo

Demonstrates an anti-bullying use case for the classifier: scans a log of
student messages, tracks how many hate-flagged messages each student has
sent, and raises an alert once a student crosses the configured threshold
— so a counselor reviews a short alert list instead of every message.

The bundled sample log is fictional, for demonstration only.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import config  # noqa: E402
from inference import classify_text, load_classifier  # noqa: E402

# Fictional example log: student_id, text. Used only when no --log-file is
# given, to demonstrate the monitoring flow end to end.
SAMPLE_LOG = pd.DataFrame([
    {"student_id": "student_1", "text": "صباح الخير، جاهز للاختبار اليوم؟"},
    {"student_id": "student_2", "text": "روح يا غبي محد يبيك بالمجموعة"},
    {"student_id": "student_1", "text": "ممكن أحد يشرح لي الواجب؟"},
    {"student_id": "student_2", "text": "انت فاشل وما تفهم شي, اطلع من هنا"},
    {"student_id": "student_3", "text": "شكرا على المساعدة أمس"},
    {"student_id": "student_2", "text": "كرهتك وكل اللي معك, ما تسوى شي"},
    {"student_id": "student_1", "text": "متى موعد تسليم المشروع؟"},
    {"student_id": "student_3", "text": "بالتوفيق للجميع في الاختبار"},
])


def scan_messages(tokenizer, model, log: pd.DataFrame) -> pd.DataFrame:
    """Classifies every message and returns the log with a flagged column."""
    log = log.copy()
    log["is_flagged"] = log["text"].apply(
        lambda t: classify_text(tokenizer, model, t)[config.ID2LABEL[1]] >= 0.5
    )
    return log


def find_alerts(scanned_log: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Returns one row per student whose flagged-message count reached the
    threshold, with the count and the flagged messages themselves."""
    flagged = scanned_log[scanned_log["is_flagged"]]
    counts = flagged.groupby("student_id").size().rename("flagged_count")

    alerts = counts[counts >= threshold].reset_index()
    alerts["messages"] = alerts["student_id"].apply(
        lambda sid: " | ".join(flagged.loc[flagged["student_id"] == sid, "text"])
    )
    return alerts


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan a student message log for repeated hate speech.")
    parser.add_argument("--log-file", type=str, default=None,
                         help="CSV with columns 'student_id' and 'text'. Uses a fictional sample log if omitted.")
    parser.add_argument("--model-dir", type=str, default=str(config.BEST_MODEL_DIR),
                         help="Local model directory or Hugging Face Hub repo id.")
    parser.add_argument("--threshold", type=int, default=config.BULLYING_ALERT_THRESHOLD,
                         help="Number of flagged messages from one student that triggers an alert.")
    args = parser.parse_args()

    log = pd.read_csv(args.log_file) if args.log_file else SAMPLE_LOG
    print(f"Scanning {len(log)} messages from {log['student_id'].nunique()} students...")

    tokenizer, model = load_classifier(args.model_dir)
    scanned = scan_messages(tokenizer, model, log)
    alerts = find_alerts(scanned, args.threshold)

    scanned_path = config.MONITOR_REPORT_DIR / "scanned_log.csv"
    alerts_path = config.MONITOR_REPORT_DIR / "alerts.csv"
    scanned.to_csv(scanned_path, index=False, encoding="utf-8-sig")
    alerts.to_csv(alerts_path, index=False, encoding="utf-8-sig")

    print(f"Flagged messages: {scanned['is_flagged'].sum()} / {len(scanned)}")
    if alerts.empty:
        print(f"No student reached the alert threshold ({args.threshold} flagged messages).")
    else:
        print(f"\n⚠ {len(alerts)} student(s) reached the alert threshold:")
        for _, row in alerts.iterrows():
            print(f"  - {row['student_id']}: {row['flagged_count']} flagged messages")

    print(f"\nFull scan saved to {scanned_path}")
    print(f"Alert list saved to {alerts_path}")


if __name__ == "__main__":
    main()
