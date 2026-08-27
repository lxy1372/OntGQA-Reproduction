#!/usr/bin/env python3

import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
from torch.utils.data import DataLoader
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

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train_grouped.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "judge_margin_infonce_lora"
)

FINAL_DIR = (
    OUTPUT_DIR
    / "final"
)


SEED = 42

NUM_EPOCHS = 3

LEARNING_RATE = 1e-4

GRADIENT_ACCUMULATION_STEPS = 4

MAX_GRAD_NORM = 1.0

MAX_LENGTH = 640

LOGGING_STEPS = 20


def set_seed(seed):
    """固定随机种子。"""

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def get_yes_no_token_ids(
    tokenizer,
):
    """获取 YES / NO 的单 token ID。"""

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
            "YES/NO 不是单 token，"
            "无法使用当前 Judge scoring。"
        )

    return (
        yes_ids[0],
        no_ids[0],
    )


def compute_candidate_margins(
    model,
    input_ids,
    attention_mask,
    lengths,
    yes_token_id,
    no_token_id,
):
    """
    计算一个问题全部候选的 signed margin：

        m(a)
        =
        log P(YES | q,a,w)
        -
        log P(NO | q,a,w)

    只对每条 Prompt 最后一个真实 token
    计算词表 logits，避免生成所有位置的完整词表 logits。
    """

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

    next_token_logits = (
        causal_lm.lm_head(
            last_hidden_states
        )
    )

    # 使用 float32 计算概率，
    # 提高 log_softmax 数值稳定性。
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

    return margins


def compute_margin_infonce_loss(
    margins,
    num_positive,
    num_negative,
):
    """
    Margin-based listwise InfoNCE。

    对一个问题 q：

        Y_q = positive candidates
        N_q = negative candidates

    每个正例都与同一问题全部负例对比：

        L_q
        =
        1/|Y_q|
        sum_{a+ in Y_q}
        -log
        [
            exp(m(a+))
            /
            (
                exp(m(a+))
                +
                sum_{a- in N_q}
                exp(m(a-))
            )
        ]

    其中：

        m(a)
        =
        log P(YES)
        -
        log P(NO)
    """

    if (
        num_positive <= 0
        or num_negative <= 0
    ):
        raise ValueError(
            "Each Judge question must contain "
            "both positive and negative candidates."
        )

    if (
        num_positive
        + num_negative
        != margins.numel()
    ):
        raise RuntimeError(
            "Candidate count does not match margins."
        )

    positive_margins = margins[
        :num_positive
    ]

    negative_margins = margins[
        num_positive:
    ]

    # log sum exp(m_neg)
    negative_log_partition = (
        torch.logsumexp(
            negative_margins,
            dim=0,
        )
    )

    # log(
    #     exp(m_pos)
    #     +
    #     sum exp(m_neg)
    # )
    denominator = (
        torch.logaddexp(
            positive_margins,
            negative_log_partition.expand_as(
                positive_margins
            ),
        )
    )

    loss_per_positive = (
        denominator
        - positive_margins
    )

    loss = (
        loss_per_positive.mean()
    )

    return loss


def save_adapter(
    model,
    tokenizer,
    output_path,
):
    """保存 LoRA Adapter 和 tokenizer。"""

    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(
        output_path
    )

    tokenizer.save_pretrained(
        output_path
    )


def main():
    set_seed(
        SEED
    )

    print("=" * 100)
    print("OntGQA Margin-InfoNCE Judge LoRA Training")
    print("=" * 100)

    print()
    print(
        f"Model               : "
        f"{MODEL_PATH}"
    )

    print(
        f"Training data       : "
        f"{TRAIN_PATH}"
    )

    print(
        f"Epochs              : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Learning rate       : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    print(
        f"Max prompt length   : "
        f"{MAX_LENGTH}"
    )

    print()

    # ============================================================
    # Tokenizer
    # ============================================================

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = (
            tokenizer.eos_token
        )

    (
        yes_token_id,
        no_token_id,
    ) = get_yes_no_token_ids(
        tokenizer
    )

    print(
        f"YES token id        : "
        f"{yes_token_id}"
    )

    print(
        f"NO token id         : "
        f"{no_token_id}"
    )

    # ============================================================
    # Dataset
    # ============================================================

    print()
    print(
        "Loading grouped Judge dataset..."
    )

    train_dataset = (
        JudgeGroupedDataset(
            jsonl_path=TRAIN_PATH,
            tokenizer=tokenizer,
            max_length=MAX_LENGTH,
        )
    )

    data_collator = (
        JudgeGroupedCollator(
            tokenizer=tokenizer,
        )
    )

    generator = torch.Generator()

    generator.manual_seed(
        SEED
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        generator=generator,
    )

    print(
        f"Training questions  : "
        f"{len(train_dataset)}"
    )

    # ============================================================
    # Model
    # ============================================================

    print()
    print("Loading base model...")

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

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={
            "use_reentrant": False,
        }
    )

    model.enable_input_require_grads()

    model.train()

    print()
    print("LoRA parameters:")

    model.print_trainable_parameters()

    # ============================================================
    # Optimizer / Scheduler
    # ============================================================

    trainable_parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    steps_per_epoch = math.ceil(
        len(train_loader)
        / GRADIENT_ACCUMULATION_STEPS
    )

    total_optimizer_steps = (
        steps_per_epoch
        * NUM_EPOCHS
    )

    # 与此前训练保持一致：
    # 无 warmup，线性衰减至 0。
    scheduler = (
        torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: max(
                0.0,
                1.0
                - (
                    current_step
                    / total_optimizer_steps
                ),
            ),
        )
    )

    print()

    print(
        f"Questions / epoch   : "
        f"{len(train_loader)}"
    )

    print(
        f"Optimizer steps/epoch: "
        f"{steps_per_epoch}"
    )

    print(
        f"Total optimizer steps: "
        f"{total_optimizer_steps}"
    )

    # ============================================================
    # Training
    # ============================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    global_optimizer_step = 0

    total_question_steps = 0

    all_losses = []

    logging_losses = []

    start_time = time.time()

    print()
    print("=" * 100)
    print("Starting training")
    print("=" * 100)

    for epoch_index in range(
        NUM_EPOCHS
    ):
        epoch_number = (
            epoch_index + 1
        )

        epoch_losses = []

        epoch_start = time.time()

        for (
            question_index,
            batch,
        ) in enumerate(
            train_loader,
            start=1,
        ):
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

            margins = (
                compute_candidate_margins(
                    model=model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    lengths=lengths,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                )
            )

            loss = (
                compute_margin_infonce_loss(
                    margins=margins,
                    num_positive=num_positive,
                    num_negative=num_negative,
                )
            )

            if not torch.isfinite(
                loss
            ):
                raise RuntimeError(
                    f"Non-finite loss at "
                    f"{batch['sample_id']}: "
                    f"{loss.item()}"
                )

            raw_loss = float(
                loss.detach().item()
            )

            epoch_losses.append(
                raw_loss
            )

            all_losses.append(
                raw_loss
            )

            logging_losses.append(
                raw_loss
            )

            # 一个问题对应一个 L_q。
            # 每 4 个问题进行一次 optimizer step。
            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            total_question_steps += 1

            should_update = (
                question_index
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            if should_update:
                grad_norm = (
                    torch.nn.utils.clip_grad_norm_(
                        trainable_parameters,
                        MAX_GRAD_NORM,
                    )
                )

                optimizer.step()

                scheduler.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_optimizer_step += 1

                if (
                    global_optimizer_step
                    % LOGGING_STEPS
                    == 0
                ):
                    mean_logging_loss = (
                        sum(
                            logging_losses
                        )
                        / len(
                            logging_losses
                        )
                    )

                    current_lr = (
                        scheduler.get_last_lr()[
                            0
                        ]
                    )

                    elapsed = (
                        time.time()
                        - start_time
                    )

                    peak_vram = (
                        torch.cuda.max_memory_allocated()
                        / 1024**3
                    )

                    print(
                        f"epoch "
                        f"{epoch_number}/{NUM_EPOCHS} | "
                        f"step "
                        f"{global_optimizer_step}/"
                        f"{total_optimizer_steps} | "
                        f"loss "
                        f"{mean_logging_loss:.6f} | "
                        f"grad_norm "
                        f"{float(grad_norm):.4f} | "
                        f"lr "
                        f"{current_lr:.8f} | "
                        f"peak_vram "
                        f"{peak_vram:.2f} GiB | "
                        f"time "
                        f"{elapsed:.1f}s"
                    )

                    logging_losses = []

        # 2156 可以整除 4；
        # 若以后数据改变，这里处理最后一个不足 4 的梯度组。
        remainder = (
            len(train_loader)
            % GRADIENT_ACCUMULATION_STEPS
        )

        if remainder != 0:
            correction = (
                GRADIENT_ACCUMULATION_STEPS
                / remainder
            )

            for parameter in trainable_parameters:
                if parameter.grad is not None:
                    parameter.grad.mul_(
                        correction
                    )

            grad_norm = (
                torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    MAX_GRAD_NORM,
                )
            )

            optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

            global_optimizer_step += 1

        epoch_runtime = (
            time.time()
            - epoch_start
        )

        epoch_mean_loss = (
            sum(epoch_losses)
            / len(epoch_losses)
        )

        print()
        print("-" * 100)

        print(
            f"Epoch {epoch_number} finished"
        )

        print(
            f"Mean question loss : "
            f"{epoch_mean_loss:.6f}"
        )

        print(
            f"Runtime            : "
            f"{epoch_runtime:.2f} s"
        )

        print("-" * 100)
        print()

        epoch_dir = (
            OUTPUT_DIR
            / f"epoch_{epoch_number}"
        )

        save_adapter(
            model=model,
            tokenizer=tokenizer,
            output_path=epoch_dir,
        )

    # ============================================================
    # Save final
    # ============================================================

    print(
        "Saving final Judge adapter..."
    )

    save_adapter(
        model=model,
        tokenizer=tokenizer,
        output_path=FINAL_DIR,
    )

    total_runtime = (
        time.time()
        - start_time
    )

    mean_train_loss = (
        sum(all_losses)
        / len(all_losses)
    )

    peak_vram_gib = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )

    print()
    print("=" * 100)
    print("Judge Training Finished")
    print("=" * 100)

    print(
        f"Training questions     : "
        f"{len(train_dataset)}"
    )

    print(
        f"Epochs                 : "
        f"{NUM_EPOCHS}"
    )

    print(
        f"Optimizer steps        : "
        f"{global_optimizer_step}"
    )

    print(
        f"Mean question loss     : "
        f"{mean_train_loss:.6f}"
    )

    print(
        f"Total runtime          : "
        f"{total_runtime:.2f} s"
    )

    print(
        f"Peak VRAM              : "
        f"{peak_vram_gib:.2f} GiB"
    )

    print(
        f"Final adapter          : "
        f"{FINAL_DIR}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()