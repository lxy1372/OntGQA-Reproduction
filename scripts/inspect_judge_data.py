#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader


WEBQSP_PATHS = [
    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "train-00000-of-00002.parquet",

    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "train-00001-of-00002.parquet",
]


def load_all_samples():
    """读取完整 WebQSP 训练集。"""

    samples = []

    for path in WEBQSP_PATHS:
        loader = WebQSPDataLoader(path)

        for row_group_index in range(
            loader.num_row_groups()
        ):
            samples.extend(
                loader.load_row_group(
                    row_group_index
                )
            )

    return samples


def main():
    print("=" * 80)
    print("WebQSP Judge Data Inspection")
    print("=" * 80)

    samples = load_all_samples()

    total_answers = 0
    total_answer_entities = 0

    total_unique_answers = 0
    total_unique_answer_entities = 0

    zero_answer_entity_questions = 0

    answer_count_distribution = {}

    samples_by_answer_count = []

    for sample in samples:
        answers = sample.answers
        answer_entities = sample.answer_entities

        unique_answers = set(
            answers
        )

        unique_answer_entities = set(
            answer_entities
        )

        total_answers += len(
            answers
        )

        total_answer_entities += len(
            answer_entities
        )

        total_unique_answers += len(
            unique_answers
        )

        total_unique_answer_entities += len(
            unique_answer_entities
        )

        if not answer_entities:
            zero_answer_entity_questions += 1

        count = len(
            unique_answer_entities
        )

        answer_count_distribution[count] = (
            answer_count_distribution.get(
                count,
                0,
            )
            + 1
        )

        samples_by_answer_count.append(
            (
                count,
                sample.sample_id,
                sample.question,
                answers,
                answer_entities,
            )
        )

    print(
        f"\nQuestions                    : "
        f"{len(samples)}"
    )

    print(
        f"Total answer strings         : "
        f"{total_answers}"
    )

    print(
        f"Total answer entities        : "
        f"{total_answer_entities}"
    )

    print(
        f"Unique answers per question  : "
        f"{total_unique_answers}"
    )

    print(
        f"Unique a_entity per question : "
        f"{total_unique_answer_entities}"
    )

    print(
        f"Questions without a_entity   : "
        f"{zero_answer_entity_questions}"
    )

    print("\nAnswer-entity count distribution:")

    for count in sorted(
        answer_count_distribution
    ):
        print(
            f"{count:3d} answers : "
            f"{answer_count_distribution[count]} questions"
        )

    print("\nTop 10 questions with most answer entities:")

    samples_by_answer_count.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    for (
        count,
        sample_id,
        question,
        answers,
        answer_entities,
    ) in samples_by_answer_count[:10]:

        print("\n" + "-" * 80)

        print(
            f"id       : {sample_id}"
        )

        print(
            f"question : {question}"
        )

        print(
            f"count    : {count}"
        )

        print(
            f"answer   : {answers[:20]}"
        )

        print(
            f"a_entity : {answer_entities[:20]}"
        )

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()