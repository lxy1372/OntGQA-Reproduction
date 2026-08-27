#!/usr/bin/env python3

import sys
from pathlib import Path

import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.judge_grouped_dataset import (
    JudgeGroupedCollator,
    JudgeGroupedDataset,
)


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train_grouped.jsonl"
)


def get_scores(
    model,
    input_ids,
    attention_mask,
    lengths,
    yes_token_id,
    no_token_id,
):
    """计算 YES/NO log probability 和 signed margin。"""

    causal_lm = model.get_base_model()

    transformer = causal_lm.model

    hidden_outputs = transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )

    hidden_states = (
        hidden_outputs.last_hidden_state
    )

    batch_indices = torch.arange(
        hidden_states.shape[0],
        device=hidden_states.device,
    )

    last_positions = (
        lengths - 1
    )

    last_hidden_states = hidden_states[
        batch_indices,
        last_positions,
        :
    ]

    next_token_logits = (
        causal_lm.lm_head(
            last_hidden_states
        )
    )

    log_probs = torch.log_softmax(
        next_token_logits.float(),
        dim=-1,
    )

    yes_log_probs = log_probs[
        :,
        yes_token_id
    ]

    no_log_probs = log_probs[
        :,
        no_token_id
    ]

    margins = (
        yes_log_probs
        - no_log_probs
    )

    return (
        yes_log_probs,
        no_log_probs,
        margins,
    )


def main():
    print("=" * 100)
    print("OntGQA Judge Margin-InfoNCE Gradient Direction Check")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    yes_ids = tokenizer.encode(
        "YES",
        add_special_tokens=False,
    )

    no_ids = tokenizer.encode(
        "NO",
        add_special_tokens=False,
    )

    if (
        len(yes_ids) != 1
        or len(no_ids) != 1
    ):
        raise ValueError(
            "YES/NO must each be a single token."
        )

    yes_token_id = yes_ids[0]
    no_token_id = no_ids[0]

    dataset = JudgeGroupedDataset(
        jsonl_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=640,
    )

    collator = JudgeGroupedCollator(
        tokenizer=tokenizer,
    )

    # 使用与上一个实验相同类型的
    # 1 positive + 1 negative 问题。
    sample_index = None

    for index, record in enumerate(
        dataset.data
    ):
        if (
            len(record["positives"]) == 1
            and len(record["negatives"]) == 1
        ):
            sample_index = index
            break

    if sample_index is None:
        raise RuntimeError(
            "No 1-positive / 1-negative sample found."
        )

    batch = collator(
        [
            dataset[sample_index]
        ]
    )

    print()

    print(
        f"Question ID : "
        f"{batch['sample_id']}"
    )

    print(
        f"Question    : "
        f"{batch['question']}"
    )

    print(
        f"Positive    : "
        f"{batch['candidate_names'][0]}"
    )

    print(
        f"Negative    : "
        f"{batch['candidate_names'][1]}"
    )

    print()
    print("Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
    )

    model.config.use_cache = False

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
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

    # 仅用于观察单步梯度方向。
    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter: (
                parameter.requires_grad
            ),
            model.parameters(),
        ),
        lr=1e-3,
    )

    input_ids = batch[
        "input_ids"
    ].to(
        model.device
    )

    attention_mask = batch[
        "attention_mask"
    ].to(
        model.device
    )

    lengths = batch[
        "lengths"
    ].to(
        model.device
    )

    num_positive = int(
        batch[
            "num_positive"
        ].item()
    )

    num_negative = int(
        batch[
            "num_negative"
        ].item()
    )

    if (
        num_positive != 1
        or num_negative != 1
    ):
        raise RuntimeError(
            "This diagnostic expects exactly "
            "one positive and one negative."
        )

    # ============================================================
    # 训练前
    # ============================================================

    model.eval()

    with torch.no_grad():
        (
            yes_before,
            no_before,
            margin_before,
        ) = get_scores(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        )

    positive_margin_before = (
        margin_before[0].item()
    )

    negative_margin_before = (
        margin_before[1].item()
    )

    print()
    print("=" * 100)
    print("Before one Margin-InfoNCE step")
    print("=" * 100)

    print(
        f"Positive YES logP : "
        f"{yes_before[0].item():.6f}"
    )

    print(
        f"Positive NO  logP : "
        f"{no_before[0].item():.6f}"
    )

    print(
        f"Positive margin   : "
        f"{positive_margin_before:.6f}"
    )

    print()

    print(
        f"Negative YES logP : "
        f"{yes_before[1].item():.6f}"
    )

    print(
        f"Negative NO  logP : "
        f"{no_before[1].item():.6f}"
    )

    print(
        f"Negative margin   : "
        f"{negative_margin_before:.6f}"
    )

    # ============================================================
    # Margin-based Listwise InfoNCE
    # ============================================================

    model.train()

    optimizer.zero_grad()

    (
        _,
        _,
        margins,
    ) = get_scores(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        lengths=lengths,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
    )

    positive_margins = (
        margins[
            :num_positive
        ]
    )

    negative_margins = (
        margins[
            num_positive:
        ]
    )

    # 对每一个 positive：
    #
    # -log[
    #   exp(m_pos)
    #   /
    #   (
    #       exp(m_pos)
    #       +
    #       sum exp(m_neg)
    #   )
    # ]
    negative_log_partition = (
        torch.logsumexp(
            negative_margins,
            dim=0,
        )
    )

    loss_per_positive = (
        torch.logaddexp(
            positive_margins,
            negative_log_partition.expand_as(
                positive_margins
            ),
        )
        - positive_margins
    )

    loss = (
        loss_per_positive.mean()
    )

    loss.backward()

    optimizer.step()

    # ============================================================
    # 训练后
    # ============================================================

    model.eval()

    with torch.no_grad():
        (
            yes_after,
            no_after,
            margin_after,
        ) = get_scores(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            lengths=lengths,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        )

    positive_margin_after = (
        margin_after[0].item()
    )

    negative_margin_after = (
        margin_after[1].item()
    )

    positive_delta = (
        positive_margin_after
        - positive_margin_before
    )

    negative_delta = (
        negative_margin_after
        - negative_margin_before
    )

    print()
    print("=" * 100)
    print("After one Margin-InfoNCE step")
    print("=" * 100)

    print(
        f"Loss               : "
        f"{loss.item():.6f}"
    )

    print()

    print(
        f"Positive YES logP  : "
        f"{yes_after[0].item():.6f}"
    )

    print(
        f"Positive NO  logP  : "
        f"{no_after[0].item():.6f}"
    )

    print(
        f"Positive margin    : "
        f"{positive_margin_after:.6f}"
    )

    print(
        f"Positive margin Δ  : "
        f"{positive_delta:+.6f}"
    )

    print()

    print(
        f"Negative YES logP  : "
        f"{yes_after[1].item():.6f}"
    )

    print(
        f"Negative NO  logP  : "
        f"{no_after[1].item():.6f}"
    )

    print(
        f"Negative margin    : "
        f"{negative_margin_after:.6f}"
    )

    print(
        f"Negative margin Δ  : "
        f"{negative_delta:+.6f}"
    )

    print()
    print("=" * 100)
    print("Interpretation")
    print("=" * 100)

    print(
        "Desired direction:"
    )

    print(
        "  Positive margin Δ > 0"
    )

    print(
        "  Negative margin Δ < 0"
    )

    print()

    print(
        f"Observed positive Δ : "
        f"{positive_delta:+.6f}"
    )

    print(
        f"Observed negative Δ : "
        f"{negative_delta:+.6f}"
    )

    print()

    if (
        positive_delta > 0
        and negative_delta < 0
    ):
        print(
            "Result: Margin-InfoNCE moves both "
            "positive and negative candidates "
            "in the correct Judge direction."
        )

    else:
        print(
            "Result: Margin-InfoNCE did not move "
            "both classes in the expected direction."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()