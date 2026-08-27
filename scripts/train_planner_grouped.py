#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.planner_grouped_dataset import (
    PlannerGroupedCollator,
    PlannerGroupedDataset,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "planner"
    / "planner_train_grouped.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "planner_grouped_lora_lr1e4"
)

FINAL_DIR = (
    OUTPUT_DIR
    / "final"
)


class MultiPositivePlannerTrainer(Trainer):
    """
    Planner 多正例训练器。

    对每个问题 q 的合法 Type Pair 集合 T_q：

        L_q =
        -1 / |T_q|
        * sum(log P(tau | q))

    同一问题的多个合法 Type Pair 等权，
    不同问题之间同样等权。
    """

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
    ):
        group_sizes = inputs.pop(
            "group_sizes"
        )

        labels = inputs[
            "labels"
        ]

        outputs = model(
            **inputs
        )

        logits = outputs.logits

        shift_logits = logits[
            :,
            :-1,
            :
        ].contiguous()

        shift_labels = labels[
            :,
            1:
        ].contiguous()

        vocab_size = (
            shift_logits.size(-1)
        )

        token_losses = F.cross_entropy(
            shift_logits.view(
                -1,
                vocab_size,
            ),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        )

        token_losses = token_losses.view(
            shift_labels.shape
        )

        valid_token_mask = (
            shift_labels != -100
        )

        # 一个完整 Type Pair 的序列级负对数似然
        sequence_losses = (
            token_losses
            * valid_token_mask
        ).sum(
            dim=1
        )

        question_losses = []

        offset = 0

        for group_size in group_sizes.tolist():
            group_losses = (
                sequence_losses[
                    offset:
                    offset + group_size
                ]
            )

            question_losses.append(
                group_losses.mean()
            )

            offset += group_size

        if offset != sequence_losses.size(0):
            raise RuntimeError(
                "group_sizes 与展开后的 Type Pair "
                "数量不一致。"
            )

        loss = torch.stack(
            question_losses
        ).mean()

        if return_outputs:
            return (
                loss,
                outputs,
            )

        return loss


def main():
    print("=" * 90)
    print("OntGQA Multi-Positive Planner LoRA Training")
    print("=" * 90)

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

    print(
        "\nLoading grouped Planner dataset..."
    )

    train_dataset = PlannerGroupedDataset(
        jsonl_path=TRAIN_PATH,
        tokenizer=tokenizer,
        max_length=256,
    )

    print(
        f"Training questions: "
        f"{len(train_dataset)}"
    )

    data_collator = (
        PlannerGroupedCollator(
            tokenizer=tokenizer,
            pad_to_multiple_of=8,
        )
    )

    training_args = TrainingArguments(
        output_dir=str(
            OUTPUT_DIR
        ),

        num_train_epochs=3,

        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,

        # 全参数微调变为 LoRA 后，
        # 使用更适合 LoRA 的学习率。
        learning_rate=1e-4,

        bf16=True,
        fp16=False,

        max_grad_norm=1.0,

        logging_strategy="steps",
        logging_steps=20,

        save_strategy="epoch",
        save_total_limit=2,

        report_to=[],

        remove_unused_columns=False,

        seed=42,
    )

    trainer = MultiPositivePlannerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(
        "\nStarting Planner training...\n"
    )

    train_result = trainer.train()

    print(
        "\nSaving final Planner adapter..."
    )

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

    print("\n" + "=" * 90)
    print("Planner Training Finished")
    print("=" * 90)

    print(
        f"Train loss   : "
        f"{train_result.training_loss:.6f}"
    )

    print(
        f"Train runtime: "
        f"{train_result.metrics.get('train_runtime', 0):.2f} s"
    )

    print(
        f"Epoch        : "
        f"{train_result.metrics.get('epoch', 0):.2f}"
    )

    print(
        f"Adapter path : "
        f"{FINAL_DIR}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()