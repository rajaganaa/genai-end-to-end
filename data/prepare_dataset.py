"""
Builds an instruction-tuning dataset (JSONL, {"instruction","input","output"})
from one or more medical QA sources for QLoRA fine-tuning.

This is a *pipeline*, not a data source: point it at your licensed /
de-identified corpora. Public benchmarks like MedQA/PubMedQA are shown as
examples of the expected schema -- swap in your own loaders as needed.

Usage:
    python data/prepare_dataset.py \
        --output data/processed/medical_sft.jsonl \
        --sources medqa,pubmedqa,internal_notes
"""
import argparse
import json
import logging
import re
from pathlib import Path
from typing import Iterator

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

SYSTEM_PREAMBLE = (
    "You are a clinical decision-support assistant. You provide "
    "evidence-based information to help clinicians and patients, but you "
    "never present a definitive diagnosis, you always note uncertainty, "
    "and you always recommend professional medical evaluation for concerning "
    "symptoms."
)


def clean_text(text: str) -> str:
    """Basic cleaning: collapse whitespace, strip control chars, drop
    obvious PHI-like patterns (very naive placeholder -- real pipelines
    should use a proper de-identification tool e.g. Philter/NLM-Scrubber)."""
    text = re.sub(r"\s+", " ", text).strip()
    # naive SSN / phone patterns -- replace with proper PHI scrubber in prod
    text = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED-SSN]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[REDACTED-PHONE]", text)
    return text


def load_medqa(path: str) -> Iterator[dict]:
    """Expected raw format: JSONL with {question, options, answer, explanation}."""
    p = Path(path)
    if not p.exists():
        log.warning("MedQA source not found at %s, skipping", path)
        return
    with open(p, "r") as f:
        for line in f:
            row = json.loads(line)
            yield {
                "instruction": clean_text(row["question"]),
                "input": "",
                "output": clean_text(
                    f"{row['answer']}. {row.get('explanation', '')}"
                ),
            }


def load_pubmedqa(path: str) -> Iterator[dict]:
    """Expected raw format: JSONL with {question, context, long_answer}."""
    p = Path(path)
    if not p.exists():
        log.warning("PubMedQA source not found at %s, skipping", path)
        return
    with open(p, "r") as f:
        for line in f:
            row = json.loads(line)
            yield {
                "instruction": clean_text(row["question"]),
                "input": clean_text(row.get("context", "")),
                "output": clean_text(row["long_answer"]),
            }


def load_internal_notes(path: str) -> Iterator[dict]:
    """Placeholder loader for de-identified internal clinical notes,
    already run through your org's PHI de-identification pipeline
    *before* reaching this script. This script does NOT perform
    HIPAA-grade de-identification -- that must happen upstream."""
    p = Path(path)
    if not p.exists():
        log.warning("Internal notes source not found at %s, skipping", path)
        return
    with open(p, "r") as f:
        for line in f:
            row = json.loads(line)
            yield {
                "instruction": clean_text(row["prompt"]),
                "input": "",
                "output": clean_text(row["completion"]),
            }


LOADERS = {
    "medqa": ("data/raw/medqa.jsonl", load_medqa),
    "pubmedqa": ("data/raw/pubmedqa.jsonl", load_pubmedqa),
    "internal_notes": ("data/raw/internal_notes.jsonl", load_internal_notes),
}


def format_for_sft(row: dict) -> dict:
    """Converts to the chat/instruction format consumed by train_qlora.py."""
    user_turn = row["instruction"]
    if row["input"]:
        user_turn += f"\n\nContext:\n{row['input']}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PREAMBLE},
            {"role": "user", "content": user_turn},
            {"role": "assistant", "content": row["output"]},
        ]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sources", default="medqa,pubmedqa",
        help="comma-separated: medqa,pubmedqa,internal_notes",
    )
    parser.add_argument(
        "--max_examples", type=int, default=None,
        help="cap total examples (useful for quick smoke tests)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_written = 0
    seen = set()  # dedup exact-duplicate instructions
    with open(out_path, "w") as out_f:
        for source_name in args.sources.split(","):
            source_name = source_name.strip()
            if source_name not in LOADERS:
                log.warning("Unknown source '%s', skipping", source_name)
                continue
            path, loader_fn = LOADERS[source_name]
            count = 0
            for row in loader_fn(path):
                key = row["instruction"].lower()
                if key in seen or not row["instruction"] or not row["output"]:
                    continue
                seen.add(key)
                out_f.write(json.dumps(format_for_sft(row)) + "\n")
                count += 1
                n_written += 1
                if args.max_examples and n_written >= args.max_examples:
                    break
            log.info("Loaded %d examples from %s", count, source_name)
            if args.max_examples and n_written >= args.max_examples:
                break

    log.info("Wrote %d total examples to %s", n_written, out_path)


if __name__ == "__main__":
    main()
