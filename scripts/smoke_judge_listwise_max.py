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


def main():
    print("=" * 100)
    print("OntGQA Listwise Judge Maximum-Group Smoke Test")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    yes_token_ids = tokenizer.encode(
        "YES",
        add_special_tokens=False,
    )

    no_token_ids = tokenizer.encode(
        "NO",
        add_special_tokens=False,
    )

    if (
        len(yes_token_ids) != 1
        or len(no_token_ids) != 1
    ):
        raise ValueError(
            "YES/NO 不是单 token，"
            "无法按照当前 Judge 定义计算概率。"
        )

    yes_token_id = (
        yes_token_ids[0]
    )

    no_token_id = (
        no_token_ids[0]
    )

    dataset = JudgeGroupedDataset(
        jsonl_path=DATA_PATH,
        tokenizer=tokenizer,
        max_length=640,
    )

    collator = JudgeGroupedCollator(
        tokenizer=tokenizer,
    )

    # 找出候选最多的问题。
    max_index = max(
        range(len(dataset)),
        key=lambda index: (
            len(
                dataset.data[
                    index
                ][
                    "positives"
                ]
            )
            +
            len(
                dataset.data[
                    index
                ][
                    "negatives"
                ]
            )
        ),
    )

    batch = collator(
        [
            dataset[
                max_index
            ]
        ]
    )

    print()
    print(
        f"Question ID       : "
        f"{batch['sample_id']}"
    )

    print(
        f"Positive          : "
        f"{batch['num_positive'].item()}"
    )

    print(
        f"Negative          : "
        f"{batch['num_negative'].item()}"
    )

    print(
        f"Input shape       : "
        f"{tuple(batch['input_ids'].shape)}"
    )

    print(
        f"Real tokens       : "
        f"{batch['attention_mask'].sum().item()}"
    )

    print(
        f"Max real length   : "
        f"{batch['lengths'].max().item()}"
    )

    print("\nLoading model...")

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

    model.train()

    # Listwise 单题可能包含大量候选，
    # 开启 gradient checkpointing 降低激活显存。
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    model.enable_input_require_grads()

    print()
    model.print_trainable_parameters()

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
        num_positive
        + num_negative
        != input_ids.shape[0]
    ):
        raise RuntimeError(
            "正负候选数量与 batch 大小不一致。"
        )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print("Running forward...")

    # PeftModel 内部的 LoRA 已经注入基础 Transformer。
    # 这里只调用 Transformer 得到 hidden states，
    # 不让 CausalLM 为每个序列位置生成完整词表 logits。
    causal_lm = (
        model.get_base_model()
    )

    transformer = (
        causal_lm.model
    )

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

    # Prompt 使用右侧 padding，
    # lengths - 1 才是每条 Prompt 最后一个真实 token。
    last_positions = (
        lengths - 1
    )

    last_hidden_states = (
        hidden_states[
            batch_indices,
            last_positions,
            :
        ]
    )

    # 只对 202 个最后位置计算词表 logits。
    next_token_logits = (
        causal_lm.lm_head(
            last_hidden_states
        )
    )

    # Eq.(4)：YES / NO 的 next-token log probability。
    log_probs = torch.log_softmax(
        next_token_logits.float(),
        dim=-1,
    )

    yes_log_probs = (
        log_probs[
            :,
            yes_token_id
        ]
    )

    no_log_probs = (
        log_probs[
            :,
            no_token_id
        ]
    )

    positive_yes = (
        yes_log_probs[
            :num_positive
        ]
    )

    negative_no = (
        no_log_probs[
            num_positive:
        ]
    )

    if (
        positive_yes.numel()
        != num_positive
    ):
        raise RuntimeError(
            "Positive YES score 数量错误。"
        )

    if (
        negative_no.numel()
        != num_negative
    ):
        raise RuntimeError(
            "Negative NO score 数量错误。"
        )

    # ------------------------------------------------------------
    # 论文 Eq.(9)
    #
    # 对于每个 positive a+：
    #
    # -log[
    #   exp(l_yes(a+))
    #   /
    #   (
    #       exp(l_yes(a+))
    #       +
    #       sum_{a- in Nq} exp(l_no(a-))
    #   )
    # ]
    #
    # 使用 logsumexp / logaddexp 做数值稳定计算。
    # ------------------------------------------------------------

    negative_log_partition = (
        torch.logsumexp(
            negative_no,
            dim=0,
        )
    )

    loss_per_positive = (
        torch.logaddexp(
            positive_yes,
            negative_log_partition.expand_as(
                positive_yes
            ),
        )
        - positive_yes
    )

    loss = (
        loss_per_positive.mean()
    )

    print()
    print(
        f"Loss              : "
        f"{loss.item():.6f}"
    )

    print(
        f"Positive YES mean : "
        f"{positive_yes.mean().item():.6f}"
    )

    print(
        f"Positive YES min  : "
        f"{positive_yes.min().item():.6f}"
    )

    print(
        f"Positive YES max  : "
        f"{positive_yes.max().item():.6f}"
    )

    print(
        f"Negative NO mean  : "
        f"{negative_no.mean().item():.6f}"
    )

    print(
        f"Negative NO min   : "
        f"{negative_no.min().item():.6f}"
    )

    print(
        f"Negative NO max   : "
        f"{negative_no.max().item():.6f}"
    )

    forward_peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print(
        f"Forward peak VRAM : "
        f"{forward_peak:.2f} GiB"
    )

    print()
    print("Running backward...")

    loss.backward()

    total_grad_sq = 0.0

    trainable_with_grad = 0

    for parameter in model.parameters():
        if (
            parameter.requires_grad
            and parameter.grad
            is not None
        ):
            trainable_with_grad += 1

            grad_norm = (
                parameter.grad
                .detach()
                .float()
                .norm(2)
                .item()
            )

            total_grad_sq += (
                grad_norm
                * grad_norm
            )

    total_grad_norm = (
        total_grad_sq ** 0.5
    )

    backward_peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    print()
    print("=" * 100)
    print("Smoke Test Summary")
    print("=" * 100)

    print(
        f"Question ID             : "
        f"{batch['sample_id']}"
    )

    print(
        f"Candidates              : "
        f"{input_ids.shape[0]}"
    )

    print(
        f"Positive / Negative     : "
        f"{num_positive} / {num_negative}"
    )

    print(
        f"Loss                    : "
        f"{loss.item():.6f}"
    )

    print(
        f"Trainable tensors w/grad: "
        f"{trainable_with_grad}"
    )

    print(
        f"Gradient norm           : "
        f"{total_grad_norm:.6f}"
    )

    print(
        f"Peak VRAM               : "
        f"{backward_peak:.2f} GiB"
    )

    print(
        f"Reserved VRAM           : "
        f"{reserved:.2f} GiB"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()