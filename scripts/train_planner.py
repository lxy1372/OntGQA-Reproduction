#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.planner_dataset import PlannerSFTDataset


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "planner"
    / "planner_train.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
)

FINAL_DIR = (
    OUTPUT_DIR
    / "final"
)


def main():
    print("加载 tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("加载 Qwen2-1.5B-Instruct...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )

    # Planner 使用轻量 LoRA 微调
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    print("\nLoRA 参数量：")
    model.print_trainable_parameters()

    dataset = PlannerSFTDataset(
        data_path=TRAIN_DATA_PATH,
        tokenizer=tokenizer,
        max_length=256,
    )

    print(
        f"\n训练样本数：{len(dataset)}"
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,

        num_train_epochs=3,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        learning_rate=1e-4,
        bf16=True,

        logging_steps=20,

        save_strategy="epoch",
        save_total_limit=2,

        report_to="none",
        remove_unused_columns=False,

        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    print("\n开始训练 Planner...\n")

    trainer.train()

    # 保存最终 LoRA Adapter
    FINAL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        FINAL_DIR
    )

    tokenizer.save_pretrained(
        FINAL_DIR
    )

    print("\nPlanner 训练完成。")
    print(
        f"LoRA Adapter 已保存到：{FINAL_DIR}"
    )


if __name__ == "__main__":
    main()