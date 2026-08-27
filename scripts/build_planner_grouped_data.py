#!/usr/bin/env python3

import json
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "planner"
    / "planner_train.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "planner"
    / "planner_train_grouped.jsonl"
)


def build_prompt(question):
    """构造论文中的 Planner 指令。"""

    return (
        "<TASK: TYPE_PAIRS>\n"
        "Please generate a valid type pair that can be helpful "
        "for answering the following question:\n"
        f"Question: {question}"
    )


def build_target(
    head_type,
    tail_type,
):
    """构造 Planner 的标准 Type Pair 输出。"""

    return (
        f"<PAIR> {head_type} "
        f"<SEP> {tail_type} </PAIR>"
    )


def main():
    print("=" * 90)
    print("OntGQA Planner Grouped Supervision Builder")
    print("=" * 90)

    grouped = OrderedDict()

    input_records = 0

    with open(
        INPUT_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            record = json.loads(
                line
            )

            input_records += 1

            sample_id = record[
                "id"
            ]

            question = record[
                "question"
            ]

            head_type = record[
                "head_type"
            ]

            tail_type = record[
                "tail_type"
            ]

            type_pair = (
                head_type,
                tail_type,
            )

            if sample_id not in grouped:
                grouped[
                    sample_id
                ] = {
                    "id": sample_id,
                    "question": question,
                    "type_pairs": [],
                    "_seen": set(),
                }

            # 同一问题中重复出现的 Type Pair 只保留一次
            if (
                type_pair
                not in grouped[
                    sample_id
                ][
                    "_seen"
                ]
            ):
                grouped[
                    sample_id
                ][
                    "_seen"
                ].add(
                    type_pair
                )

                grouped[
                    sample_id
                ][
                    "type_pairs"
                ].append(
                    type_pair
                )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_pairs = 0
    max_pairs = 0

    pair_count_distribution = {}

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for record in grouped.values():
            type_pairs = record[
                "type_pairs"
            ]

            pair_count = len(
                type_pairs
            )

            total_pairs += (
                pair_count
            )

            max_pairs = max(
                max_pairs,
                pair_count,
            )

            pair_count_distribution[
                pair_count
            ] = (
                pair_count_distribution.get(
                    pair_count,
                    0,
                )
                + 1
            )

            output_record = {
                "id": record[
                    "id"
                ],
                "question": record[
                    "question"
                ],
                "prompt": build_prompt(
                    record[
                        "question"
                    ]
                ),
                "type_pairs": [
                    [
                        head_type,
                        tail_type,
                    ]
                    for (
                        head_type,
                        tail_type
                    ) in type_pairs
                ],
                "targets": [
                    build_target(
                        head_type,
                        tail_type,
                    )
                    for (
                        head_type,
                        tail_type
                    ) in type_pairs
                ],
            }

            f.write(
                json.dumps(
                    output_record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    num_questions = len(
        grouped
    )

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Original supervision rows : "
        f"{input_records}"
    )

    print(
        f"Grouped questions         : "
        f"{num_questions}"
    )

    print(
        f"Unique type pairs         : "
        f"{total_pairs}"
    )

    if num_questions > 0:
        print(
            f"Avg pairs/question        : "
            f"{total_pairs / num_questions:.2f}"
        )

    print(
        f"Max pairs/question        : "
        f"{max_pairs}"
    )

    print()
    print(
        "Pair-count distribution:"
    )

    for pair_count in sorted(
        pair_count_distribution
    ):
        print(
            f"{pair_count:2d} pairs : "
            f"{pair_count_distribution[pair_count]} questions"
        )

    print()
    print(
        f"Output file               : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()