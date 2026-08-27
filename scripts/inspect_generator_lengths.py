#!/usr/bin/env python3

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "generator"
    / "generator_train.jsonl"
)


def percentile(values, ratio):
    """计算整数序列的简单百分位数。"""

    if not values:
        return 0

    values = sorted(values)

    index = int(
        (len(values) - 1)
        * ratio
    )

    return values[index]


def main():
    print("=" * 80)
    print("OntGQA Generator Length Inspection")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    records = []

    with open(
        DATA_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            records.append(
                json.loads(line)
            )

    prompt_lengths = []
    target_lengths = []
    full_lengths = []

    over_256 = 0
    over_512 = 0
    over_1024 = 0
    over_2048 = 0

    longest_samples = []

    for record in records:
        prompt_messages = [
            {
                "role": "user",
                "content": record["prompt"],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": record["prompt"],
            },
            {
                "role": "assistant",
                "content": record["target"],
            },
        ]

        prompt_text = (
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        full_text = (
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        prompt_ids = tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        target_ids = tokenizer(
            record["target"],
            add_special_tokens=False,
        )["input_ids"]

        full_ids = tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        prompt_length = len(
            prompt_ids
        )

        target_length = len(
            target_ids
        )

        full_length = len(
            full_ids
        )

        prompt_lengths.append(
            prompt_length
        )

        target_lengths.append(
            target_length
        )

        full_lengths.append(
            full_length
        )

        if full_length > 256:
            over_256 += 1

        if full_length > 512:
            over_512 += 1

        if full_length > 1024:
            over_1024 += 1

        if full_length > 2048:
            over_2048 += 1

        longest_samples.append(
            (
                full_length,
                target_length,
                len(record["answers"]),
                record["id"],
                record["question"],
            )
        )

    longest_samples.sort(
        reverse=True
    )

    total = len(
        records
    )

    print(
        f"\nSamples              : "
        f"{total}"
    )

    print("\nFull sequence length:")

    print(
        f"Min                  : "
        f"{min(full_lengths)}"
    )

    print(
        f"Median               : "
        f"{percentile(full_lengths, 0.50)}"
    )

    print(
        f"P90                  : "
        f"{percentile(full_lengths, 0.90)}"
    )

    print(
        f"P95                  : "
        f"{percentile(full_lengths, 0.95)}"
    )

    print(
        f"P99                  : "
        f"{percentile(full_lengths, 0.99)}"
    )

    print(
        f"Max                  : "
        f"{max(full_lengths)}"
    )

    print("\nTarget length:")

    print(
        f"Median               : "
        f"{percentile(target_lengths, 0.50)}"
    )

    print(
        f"P90                  : "
        f"{percentile(target_lengths, 0.90)}"
    )

    print(
        f"P95                  : "
        f"{percentile(target_lengths, 0.95)}"
    )

    print(
        f"P99                  : "
        f"{percentile(target_lengths, 0.99)}"
    )

    print(
        f"Max                  : "
        f"{max(target_lengths)}"
    )

    print("\nLength thresholds:")

    print(
        f"> 256                : "
        f"{over_256}/{total} "
        f"({over_256 / total * 100:.2f}%)"
    )

    print(
        f"> 512                : "
        f"{over_512}/{total} "
        f"({over_512 / total * 100:.2f}%)"
    )

    print(
        f"> 1024               : "
        f"{over_1024}/{total} "
        f"({over_1024 / total * 100:.2f}%)"
    )

    print(
        f"> 2048               : "
        f"{over_2048}/{total} "
        f"({over_2048 / total * 100:.2f}%)"
    )

    print("\nTop 10 longest samples:")

    for (
        full_length,
        target_length,
        answer_count,
        sample_id,
        question,
    ) in longest_samples[:10]:

        print("\n" + "-" * 80)

        print(
            f"ID            : "
            f"{sample_id}"
        )

        print(
            f"Question      : "
            f"{question}"
        )

        print(
            f"Answers       : "
            f"{answer_count}"
        )

        print(
            f"Target tokens : "
            f"{target_length}"
        )

        print(
            f"Full tokens   : "
            f"{full_length}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()