"""
QLoRA fine-tuning for a 13B-70B base model on the medical instruction
dataset produced by data/prepare_dataset.py.

Why QLoRA here:
  - 4-bit NF4 quantization of the frozen base model cuts memory ~4x vs fp16
  - Only small LoRA adapter matrices are trained (~0.1-1% of params)
  - Makes 13B trainable on a single 40-80GB GPU, and 70B trainable on
    4-8 GPUs with ZeRO-3 (see ds_config.json), vs. needing 8x80GB+ for
    full fine-tuning of 70B.

Usage (single/multi-GPU via accelerate):
    accelerate launch --config_file finetune/ds_config.json \
        finetune/train_qlora.py \
        --base_model meta-llama/Llama-3-13b \
        --dataset data/processed/medical_sft.jsonl \
        --output_dir checkpoints/medical-lora-13b
"""

import argparse
import logging

import mlflow
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from trl import SFTTrainer

from config.settings import settings

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class MLflowLoggingCallback(TrainerCallback):
    """Logs HF Trainer metrics (loss, eval_loss, learning_rate) to MLflow
    as training progresses, so runs are comparable in the MLflow UI without
    needing to grep through TensorBoard logs after the fact."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        # Filter to numeric metrics only; step is used as the MLflow "step" axis
        numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
        if numeric_logs:
            mlflow.log_metrics(numeric_logs, step=state.global_step)


def build_quant_config() -> BitsAndBytesConfig:
    """4-bit NF4 quantization config -- the core of QLoRA's memory savings."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # extra ~0.4 bits/param saved
    )


def build_lora_config(target_modules: list[str]) -> LoraConfig:
    return LoraConfig(
        r=16,  # rank -- 16 is a solid default for 13B-70B SFT
        lora_alpha=32,  # scaling factor, typically 2x rank
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )


def format_chat_example(example: dict, tokenizer) -> dict:
    """Applies the model's chat template to our {"messages": [...]} rows."""
    text = tokenizer.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--target_modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="comma-separated attention/MLP projection layers to adapt",
    )
    parser.add_argument(
        "--run_name",
        default=None,
        help="MLflow run name; defaults to '<base_model>-<timestamp>' if unset",
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)

    run_name = (
        args.run_name
        or f"{args.base_model.split('/')[-1]}-{args.output_dir.split('/')[-1]}"
    )

    # Everything inside this `with` block is one comparable MLflow run:
    # hyperparameters, per-step metrics (via MLflowLoggingCallback), and
    # the final adapter as a logged artifact. This is what makes runs
    # diffable in the MLflow UI instead of scattered across log files.
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "base_model": args.base_model,
                "epochs": args.epochs,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum_steps": args.grad_accum_steps,
                "learning_rate": args.learning_rate,
                "max_seq_length": args.max_seq_length,
                "lora_r": 16,
                "lora_alpha": 32,
                "target_modules": args.target_modules,
                "dataset": args.dataset,
            }
        )

        log.info("Loading tokenizer and base model: %s", args.base_model)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            quantization_config=build_quant_config(),
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        # Required prep step for k-bit training: casts norm layers to fp32,
        # enables gradient checkpointing hooks correctly, etc.
        model = prepare_model_for_kbit_training(model)

        lora_config = build_lora_config(args.target_modules.split(","))
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()  # sanity check: should be ~0.1-1%

        trainable, total = model.get_nb_trainable_parameters()
        mlflow.log_params(
            {
                "trainable_params": trainable,
                "total_params": total,
                "trainable_pct": round(100 * trainable / total, 4),
            }
        )

        log.info("Loading dataset: %s", args.dataset)
        dataset = load_dataset("json", data_files=args.dataset, split="train")
        dataset = dataset.map(
            lambda ex: format_chat_example(ex, tokenizer),
            remove_columns=dataset.column_names,
        )
        # Hold out a small slice for eval loss monitoring during training
        split = dataset.train_test_split(test_size=0.05, seed=42)
        mlflow.log_params(
            {"train_size": len(split["train"]), "eval_size": len(split["test"])}
        )

        training_args = TrainingArguments(
            output_dir=args.output_dir,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.grad_accum_steps,
            learning_rate=args.learning_rate,
            bf16=True,
            gradient_checkpointing=True,  # trade compute for memory -- needed at this scale
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=3,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to=[
                "tensorboard"
            ],  # per-step curves; MLflow gets the same via callback below
        )

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=split["train"],
            eval_dataset=split["test"],
            dataset_text_field="text",
            max_seq_length=args.max_seq_length,
            tokenizer=tokenizer,
            callbacks=[MLflowLoggingCallback()],
        )

        log.info("Starting training...")
        trainer.train()

        log.info("Saving final LoRA adapter to %s", args.output_dir)
        trainer.model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Log final eval metrics explicitly (in addition to the per-step
        # callback) so they're easy to find as the run's headline numbers.
        final_metrics = trainer.evaluate()
        mlflow.log_metrics(
            {
                f"final_{k}": v
                for k, v in final_metrics.items()
                if isinstance(v, (int, float))
            }
        )

        # Log the adapter directory as an MLflow artifact so it's retrievable
        # from the MLflow UI/API even if the local checkpoints/ dir is later
        # cleaned up -- this is what mlops/model_registry.py registers from.
        mlflow.log_artifacts(args.output_dir, artifact_path="lora_adapter")

        run_id = mlflow.active_run().info.run_id
        log.info(
            "MLflow run complete: run_id=%s (use with mlops/model_registry.py to register)",
            run_id,
        )


if __name__ == "__main__":
    main()
