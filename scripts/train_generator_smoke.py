#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.generator_dataset import (
    GeneratorSFTDataset,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "generator"
    / "generator_train.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "generator_smoke"
)


def main():
    print("=" * 80)
    print("OntGQA Generator LoRA Smoke Training")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    print("\nLoading base model...")

    model = (
        AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
        )
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nLoRA parameters:")
    model.print_trainable_parameters()

    print("\nLoading Generator dataset...")

    train_dataset = GeneratorSFTDataset(
        jsonl_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=512,
    )

    print(
        f"Training samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Filtered samples: "
        f"{train_dataset.filtered_count}"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR
        ),

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=1e-4,

        max_steps=20,

        bf16=True,
        fp16=False,

        logging_strategy="steps",
        logging_steps=1,

        save_strategy="no",

        report_to=[],

        remove_unused_columns=False,

        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(
        "\nStarting smoke training...\n"
    )

    train_result = trainer.train()

    print("\n" + "=" * 80)
    print("Smoke Training Finished")
    print("=" * 80)

    print(
        f"Train loss   : "
        f"{train_result.training_loss:.6f}"
    )

    print(
        f"Train runtime: "
        f"{train_result.metrics.get('train_runtime', 0):.2f} s"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()