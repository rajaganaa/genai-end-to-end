"""
Two export paths for a trained LoRA adapter:

1. --mode vllm_lora (default): leaves the adapter separate from the base
   model. vLLM's --enable-lora flag loads both and does the merge at
   inference time per-request. This is what we use in serving/vllm_server.sh
   because it lets us hot-swap or A/B test adapters without re-deploying
   the (much larger) base model.

2. --mode merged: physically merges LoRA weights into the base model and
   saves a single standalone checkpoint. Useful if you need a
   single-artifact deployment (e.g. shipping to an edge device, or a
   serving stack without LoRA support).
"""
import argparse
import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--mode", choices=["vllm_lora", "merged"], default="vllm_lora"
    )
    args = parser.parse_args()

    if args.mode == "vllm_lora":
        # Nothing to merge -- just validate the adapter loads cleanly and
        # copy it into a clean directory vLLM can point --lora-modules at.
        log.info("Validating adapter loads against base model (no merge)...")
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16, device_map="cpu"
        )
        PeftModel.from_pretrained(base, args.adapter)  # raises if incompatible
        import shutil
        shutil.copytree(args.adapter, args.output_dir, dirs_exist_ok=True)
        log.info(
            "Adapter validated and staged at %s. Point vLLM's "
            "--lora-modules at this directory.", args.output_dir
        )
        return

    # --- merged mode ---
    log.info("Loading base model in fp16 for merging...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    log.info("Loading and merging LoRA adapter: %s", args.adapter)
    merged = PeftModel.from_pretrained(base, args.adapter)
    merged = merged.merge_and_unload()  # folds LoRA deltas into base weights

    log.info("Saving merged model to %s", args.output_dir)
    merged.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    log.info("Done. This directory is now a standalone HF model checkpoint.")


if __name__ == "__main__":
    main()
