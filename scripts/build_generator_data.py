#!/usr/bin/env python3

import json
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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generator"
    / "generator_train.jsonl"
)


def load_all_samples():
    """读取完整 WebQSP 训练集。"""

    samples = []

    for path in WEBQSP_PATHS:
        loader = WebQSPDataLoader(
            path
        )

        for row_group_index in range(
            loader.num_row_groups()
        ):
            samples.extend(
                loader.load_row_group(
                    row_group_index
                )
            )

    return samples


def deduplicate_answers(answers):
    """按原始顺序去除重复答案。"""

    unique_answers = []
    seen = set()

    for answer in answers:
        if answer in seen:
            continue

        seen.add(answer)
        unique_answers.append(
            answer
        )

    return unique_answers


def build_prompt(question):
    """构造论文中的 Generator 指令。"""

    return (
        "<TASK: ANSWERS>\n"
        "Please generate ALL correct answer entities "
        "for the following question:\n"
        f"Question: {question}"
    )


def main():
    print("=" * 80)
    print("OntGQA Generator Supervision Builder")
    print("=" * 80)

    samples = load_all_samples()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    records = []

    total_answers = 0
    max_answers = 0
    questions_without_answer = 0

    for sample in samples:
        answers = deduplicate_answers(
            sample.answers
        )

        if not answers:
            questions_without_answer += 1
            continue

        total_answers += len(
            answers
        )

        max_answers = max(
            max_answers,
            len(answers),
        )

        # 论文只规定输出所有正确答案，
        # 没有规定具体的多答案序列化格式。
        # 这里统一使用 JSON 数组，方便训练和后续稳定解析。
        target = json.dumps(
            answers,
            ensure_ascii=False,
        )

        records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "answers": answers,
                "prompt": build_prompt(
                    sample.question
                ),
                "target": target,
            }
        )

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)

    print(
        f"Input questions          : "
        f"{len(samples)}"
    )

    print(
        f"Generator samples        : "
        f"{len(records)}"
    )

    print(
        f"Questions without answer : "
        f"{questions_without_answer}"
    )

    print(
        f"Total unique answers     : "
        f"{total_answers}"
    )

    if records:
        print(
            f"Avg answers/question     : "
            f"{total_answers / len(records):.2f}"
        )

    print(
        f"Max answers/question     : "
        f"{max_answers}"
    )

    print(
        f"Output file              : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()