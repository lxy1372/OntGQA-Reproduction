#!/usr/bin/env python3

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from transformers import AutoTokenizer


DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train.jsonl"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)


def load_records():
    """读取 Judge 训练数据。"""

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

    return records


def main():
    print("=" * 100)
    print("OntGQA Judge Listwise Data Inspection")
    print("=" * 100)

    records = load_records()

    tokenizer = (
        AutoTokenizer.from_pretrained(
            MODEL_PATH,
            local_files_only=True,
        )
    )

    groups = defaultdict(
        list
    )

    for record in records:
        groups[
            record["id"]
        ].append(
            record
        )

    total_positive = 0
    total_negative = 0

    balanced_questions = 0
    unbalanced_questions = []

    candidate_counts = []
    positive_counts = []
    negative_counts = []

    max_candidate_question = None
    max_candidate_count = 0

    max_token_question = None
    max_total_tokens = 0

    total_tokens_per_question = []

    max_single_prompt_tokens = 0

    for sample_id, group in groups.items():
        positives = [
            record
            for record in group
            if record["target"] == "YES"
        ]

        negatives = [
            record
            for record in group
            if record["target"] == "NO"
        ]

        num_positive = len(
            positives
        )

        num_negative = len(
            negatives
        )

        num_candidates = len(
            group
        )

        total_positive += (
            num_positive
        )

        total_negative += (
            num_negative
        )

        positive_counts.append(
            num_positive
        )

        negative_counts.append(
            num_negative
        )

        candidate_counts.append(
            num_candidates
        )

        if (
            num_positive
            == num_negative
            and num_positive > 0
        ):
            balanced_questions += 1

        else:
            unbalanced_questions.append(
                {
                    "id": sample_id,
                    "positive": (
                        num_positive
                    ),
                    "negative": (
                        num_negative
                    ),
                }
            )

        if (
            num_candidates
            > max_candidate_count
        ):
            max_candidate_count = (
                num_candidates
            )

            max_candidate_question = (
                sample_id
            )

        question_total_tokens = 0

        for record in group:
            messages = [
                {
                    "role": "user",
                    "content": (
                        record["prompt"]
                    ),
                }
            ]

            prompt_text = (
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            token_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
            )[
                "input_ids"
            ]

            token_count = len(
                token_ids
            )

            question_total_tokens += (
                token_count
            )

            max_single_prompt_tokens = max(
                max_single_prompt_tokens,
                token_count,
            )

        total_tokens_per_question.append(
            question_total_tokens
        )

        if (
            question_total_tokens
            > max_total_tokens
        ):
            max_total_tokens = (
                question_total_tokens
            )

            max_token_question = (
                sample_id
            )

    print()
    print("=" * 100)
    print("Basic Statistics")
    print("=" * 100)

    print(
        f"Total records                  : "
        f"{len(records)}"
    )

    print(
        f"Unique questions               : "
        f"{len(groups)}"
    )

    print(
        f"Positive records               : "
        f"{total_positive}"
    )

    print(
        f"Negative records               : "
        f"{total_negative}"
    )

    print()

    print(
        f"Balanced questions             : "
        f"{balanced_questions}/{len(groups)}"
    )

    print(
        f"Unbalanced questions           : "
        f"{len(unbalanced_questions)}"
    )

    if unbalanced_questions:
        print(
            "\nFirst 10 unbalanced questions:"
        )

        for item in (
            unbalanced_questions[:10]
        ):
            print(
                f"  {item['id']}: "
                f"YES={item['positive']}, "
                f"NO={item['negative']}"
            )

    print()
    print("=" * 100)
    print("Candidate Statistics")
    print("=" * 100)

    print(
        f"Avg candidates/question        : "
        f"{statistics.mean(candidate_counts):.2f}"
    )

    print(
        f"Median candidates/question     : "
        f"{statistics.median(candidate_counts):.2f}"
    )

    print(
        f"Max candidates/question        : "
        f"{max(candidate_counts)}"
    )

    print(
        f"Max candidate question         : "
        f"{max_candidate_question}"
    )

    print()

    print(
        f"Avg positives/question         : "
        f"{statistics.mean(positive_counts):.2f}"
    )

    print(
        f"Max positives/question         : "
        f"{max(positive_counts)}"
    )

    print(
        f"Avg negatives/question         : "
        f"{statistics.mean(negative_counts):.2f}"
    )

    print(
        f"Max negatives/question         : "
        f"{max(negative_counts)}"
    )

    print()

    count_distribution = Counter(
        candidate_counts
    )

    print(
        "Largest candidate-count groups:"
    )

    for count in sorted(
        count_distribution,
        reverse=True,
    )[:15]:
        print(
            f"{count:3d} candidates : "
            f"{count_distribution[count]} questions"
        )

    print()
    print("=" * 100)
    print("Token Statistics")
    print("=" * 100)

    sorted_token_totals = sorted(
        total_tokens_per_question
    )

    def percentile(
        values,
        ratio,
    ):
        index = int(
            (len(values) - 1)
            * ratio
        )

        return values[
            index
        ]

    print(
        f"Max single prompt tokens       : "
        f"{max_single_prompt_tokens}"
    )

    print(
        f"Avg total tokens/question      : "
        f"{statistics.mean(total_tokens_per_question):.2f}"
    )

    print(
        f"Median total tokens/question   : "
        f"{statistics.median(total_tokens_per_question):.2f}"
    )

    print(
        f"P90 total tokens/question      : "
        f"{percentile(sorted_token_totals, 0.90)}"
    )

    print(
        f"P95 total tokens/question      : "
        f"{percentile(sorted_token_totals, 0.95)}"
    )

    print(
        f"P99 total tokens/question      : "
        f"{percentile(sorted_token_totals, 0.99)}"
    )

    print(
        f"Max total tokens/question      : "
        f"{max_total_tokens}"
    )

    print(
        f"Max token question             : "
        f"{max_token_question}"
    )

    print()
    print("=" * 100)

    if (
        len(unbalanced_questions)
        == 0
    ):
        print(
            "All questions contain balanced "
            "YES/NO supervision."
        )

    else:
        print(
            "Some questions are not balanced. "
            "Do not start listwise training yet."
        )

    print("=" * 100)


if __name__ == "__main__":
    main()