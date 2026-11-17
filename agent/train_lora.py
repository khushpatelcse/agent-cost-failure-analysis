# train_lora - LoRA fine-tune of Llama on the planning (and answer) data
#
# Run on an OSC GPU node through osc/train.sbatch, not on a login node.
#
#   python train_lora.py \
#       --train data/plan_train.jsonl data/answer_train.jsonl \
#       --val data/plan_val.jsonl \
#       --out outputs/f1-lora

import argparse

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--val", nargs="+")
    parser.add_argument("--out", default="outputs/f1-lora")
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    train_ds = load_dataset("json", data_files=args.train, split="train")
    val_ds = (load_dataset("json", data_files=args.val, split="train")
              if args.val else None)
    print(f"train rows: {len(train_ds)}  val rows: {len(val_ds) if val_ds else 0}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,   # older transformers (<4.56): torch_dtype=
    )

    # LoRA: train small adapter matrices instead of all 3B weights.
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        lora_dropout=0.05,
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )

    config = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,      # float < 1 = fraction of steps (transformers 5)
        bf16=True,
        gradient_checkpointing=True,
        # Answer prompts include tool JSON and can get long. Anything past
        # this is cut off, so keep it above your longest example.
        max_length=args.max_length,
        logging_steps=10,
        eval_strategy="epoch" if val_ds is not None else "no",
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    trainer.train()

    # Saves only the adapter (tens of MB), not a full copy of Llama.
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"adapter saved to {args.out}")


if __name__ == "__main__":
    main()
