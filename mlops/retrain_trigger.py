"""
Monitors eval scores over time and triggers a retraining pipeline run when
the current production model's score drops meaningfully vs. its own
historical baseline -- e.g. due to data drift (new drug names, updated
clinical guidelines) that the fine-tune hasn't seen.

Intended to run on a schedule (cron / Airflow / GitHub Actions cron) via:
    python mlops/retrain_trigger.py --current_score 0.81

Where --current_score comes from eval/evaluate.py's output (wire the two
together in your CI/CD: run evaluate.py, parse its pass rate, pass it here).
"""
import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SCORE_HISTORY_PATH = Path("mlops/score_history.json")


def load_history() -> list[dict]:
    if not SCORE_HISTORY_PATH.exists():
        return []
    return json.loads(SCORE_HISTORY_PATH.read_text())


def save_history(history: list[dict]) -> None:
    SCORE_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SCORE_HISTORY_PATH.write_text(json.dumps(history, indent=2))


def get_baseline_score(history: list[dict]) -> float | None:
    """Baseline = best score seen in the last 10 recorded evaluations.
    Using "best of recent window" (not just "most recent") avoids
    ratcheting the baseline down after a single bad-but-not-yet-triggering
    score."""
    if not history:
        return None
    recent = history[-10:]
    return max(r["score"] for r in recent)


def trigger_retraining_pipeline(reason: str) -> None:
    """Kicks off the retraining pipeline. Here we shell out to the existing
    training + registry scripts; in a real deployment this would instead
    enqueue a job in your orchestrator (Airflow DAG trigger, Step Functions
    execution, GitHub Actions workflow_dispatch, etc.) rather than running
    synchronously in this process."""
    log.warning("TRIGGERING RETRAINING PIPELINE. Reason: %s", reason)
    # Example of what a real trigger might shell out to (commented out --
    # actually running this requires GPU infra + a prepared dataset path):
    #
    # subprocess.run([
    #     "accelerate", "launch", "--config_file", "finetune/ds_config.json",
    #     "finetune/train_qlora.py",
    #     "--base_model", "meta-llama/Llama-3-13b",
    #     "--dataset", "data/processed/medical_sft.jsonl",
    #     "--output_dir", f"checkpoints/medical-lora-13b-retrain-{datetime.now():%Y%m%d}",
    # ], check=True)
    log.info(
        "(placeholder) In production this would enqueue a retraining job "
        "in your orchestrator rather than block this process."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--current_score", type=float, required=True,
        help="Latest eval pass rate/accuracy, e.g. from eval/evaluate.py",
    )
    args = parser.parse_args()

    history = load_history()
    baseline = get_baseline_score(history)

    history.append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "score": args.current_score,
    })
    save_history(history)

    if baseline is None:
        log.info("No baseline yet (first recorded score: %.4f) -- nothing to compare against.", args.current_score)
        return

    drop = baseline - args.current_score
    log.info(
        "Current score=%.4f  baseline=%.4f  drop=%.4f  threshold=%.4f",
        args.current_score, baseline, drop, settings.retrain_trigger_score_drop,
    )

    if drop >= settings.retrain_trigger_score_drop:
        trigger_retraining_pipeline(
            reason=f"Eval score dropped {drop:.4f} vs. recent baseline {baseline:.4f}"
        )
    else:
        log.info("Score within acceptable range -- no retraining triggered.")


if __name__ == "__main__":
    main()
