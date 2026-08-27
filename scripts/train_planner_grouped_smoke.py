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
    / "planner_grouped_smoke"
)


class MultiPositivePlannerTrainer(Trainer):
    """
    Planner 多正例训练器。

    对每个 Type Pair 计算完整输出序列的负对数似然，
    然后先在同一问题的合法 Type Pairs 内求平均，
    最后在 batch 中的问题之间求平均。

    对应：
        L_q = -1/|T_q| * sum(log P(tau | q))
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

        # Causal LM：
        # token t 的 logits 用来预测 token t+1。
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

        # 为每一个 token 分别计算交叉熵，
        # Prompt 和 padding 的 label=-100 会被忽略。
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

        # 每一个 Type Pair 的 loss 使用目标序列 token NLL 之和，
        # 对应 -log P(tau | q)。
        sequence_losses = (
            token_losses
            * valid_token_mask
        ).sum(
            dim=1
        )

        # Collator 已经将不同问题的 Type Pairs 展平。
        # 这里按照 group_sizes 恢复问题边界。
        question_losses = []

        offset = 0

        for group_size in group_sizes.tolist():
            group_sequence_losses = (
                sequence_losses[
                    offset:
                    offset + group_size
                ]
            )

            # 同一问题的全部合法 Type Pair 等权平均。
            question_loss = (
                group_sequence_losses.mean()
            )

            question_losses.append(
                question_loss
            )

            offset += group_size

        if offset != sequence_losses.size(0):
            raise RuntimeError(
                "group_sizes 与展平后的 "
                "Type Pair 数量不一致。"
            )

        # 不同问题之间等权平均。
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
    print("OntGQA Multi-Positive Planner LoRA Smoke Training")
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

    train_dataset = (
        PlannerGroupedDataset(
            jsonl_path=TRAIN_PATH,
            tokenizer=tokenizer,
            max_length=256,
        )
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

        # 这里的 batch_size 指“问题数”，
        # 不是展平后的 Type Pair 数。
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

    trainer = MultiPositivePlannerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print(
        "\nStarting smoke training...\n"
    )

    train_result = trainer.train()

    print("\n" + "=" * 90)
    print("Smoke Training Finished")
    print("=" * 90)

    print(
        f"Train loss   : "
        f"{train_result.training_loss:.6f}"
    )

    print(
        f"Train runtime: "
        f"{train_result.metrics.get('train_runtime', 0):.2f} s"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()