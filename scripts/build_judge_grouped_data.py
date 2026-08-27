#!/usr/bin/env python3

import json
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train_grouped.jsonl"
)


def main():
    print("=" * 100)
    print("OntGQA Judge Grouped Training Data Builder")
    print("=" * 100)

    groups = OrderedDict()

    total_records = 0

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

            total_records += 1

            sample_id = record["id"]

            if sample_id not in groups:
                groups[sample_id] = {
                    "id": sample_id,
                    "question": record[
                        "question"
                    ],
                    "positives": [],
                    "negatives": [],
                }

            candidate_record = {
                "candidate": record[
                    "candidate"
                ],
                "evidence_paths": record[
                    "evidence_paths"
                ],
                "prompt": record[
                    "prompt"
                ],
            }

            if record["target"] == "YES":
                groups[
                    sample_id
                ][
                    "positives"
                ].append(
                    candidate_record
                )

            elif record["target"] == "NO":
                groups[
                    sample_id
                ][
                    "negatives"
                ].append(
                    candidate_record
                )

            else:
                raise ValueError(
                    f"Unexpected target: "
                    f"{record['target']}"
                )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_positive = 0
    total_negative = 0

    min_positive = None
    max_positive = 0

    min_negative = None
    max_negative = 0

    unbalanced = []

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for record in groups.values():
            num_positive = len(
                record["positives"]
            )

            num_negative = len(
                record["negatives"]
            )

            if (
                num_positive == 0
                or num_negative == 0
            ):
                raise RuntimeError(
                    f"{record['id']} "
                    f"does not contain both "
                    f"positive and negative samples."
                )

            if (
                num_positive
                != num_negative
            ):
                unbalanced.append(
                    record["id"]
                )

            total_positive += (
                num_positive
            )

            total_negative += (
                num_negative
            )

            if min_positive is None:
                min_positive = (
                    num_positive
                )
            else:
                min_positive = min(
                    min_positive,
                    num_positive,
                )

            max_positive = max(
                max_positive,
                num_positive,
            )

            if min_negative is None:
                min_negative = (
                    num_negative
                )
            else:
                min_negative = min(
                    min_negative,
                    num_negative,
                )

            max_negative = max(
                max_negative,
                num_negative,
            )

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    num_questions = len(
        groups
    )

    print()
    print("=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        f"Original records       : "
        f"{total_records}"
    )

    print(
        f"Grouped questions      : "
        f"{num_questions}"
    )

    print(
        f"Positive candidates    : "
        f"{total_positive}"
    )

    print(
        f"Negative candidates    : "
        f"{total_negative}"
    )

    print()

    print(
        f"Avg positive/question  : "
        f"{total_positive / num_questions:.2f}"
    )

    print(
        f"Avg negative/question  : "
        f"{total_negative / num_questions:.2f}"
    )

    print(
        f"Positive range         : "
        f"{min_positive} - {max_positive}"
    )

    print(
        f"Negative range         : "
        f"{min_negative} - {max_negative}"
    )

    print()

    print(
        f"Unbalanced questions   : "
        f"{len(unbalanced)}"
    )

    print(
        f"Output file            : "
        f"{OUTPUT_PATH}"
    )

    print()

    first_record = next(
        iter(
            groups.values()
        )
    )

    print("=" * 100)
    print("First grouped question")
    print("=" * 100)

    print(
        f"ID       : "
        f"{first_record['id']}"
    )

    print(
        f"Question : "
        f"{first_record['question']}"
    )

    print(
        f"YES      : "
        f"{len(first_record['positives'])}"
    )

    print(
        f"NO       : "
        f"{len(first_record['negatives'])}"
    )

    print()

    print("First YES candidate:")
    print(
        first_record[
            "positives"
        ][0][
            "candidate"
        ]
    )

    print()

    print("First NO candidate:")
    print(
        first_record[
            "negatives"
        ][0][
            "candidate"
        ]
    )

    print("=" * 100)


if __name__ == "__main__":
    main()