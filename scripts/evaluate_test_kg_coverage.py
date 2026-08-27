#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader


TEST_PATHS = [
    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "test-00000-of-00002.parquet",

    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "test-00001-of-00002.parquet",
]


def load_all_samples():
    """读取完整 WebQSP test split。"""

    samples = []

    for path in TEST_PATHS:
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


def get_graph_entities(
    graph_triples,
):
    """提取当前问题子图中出现的全部实体。"""

    entities = set()

    for head, _, tail in graph_triples:
        entities.add(head)
        entities.add(tail)

    return entities


def main():
    print("=" * 90)
    print("WebQSP Test KG Gold Answer Coverage")
    print("=" * 90)

    samples = load_all_samples()

    total_questions = len(
        samples
    )

    any_gold_present = 0
    all_gold_present = 0
    no_gold_present = 0
    partial_questions = 0

    total_gold_answers = 0
    total_gold_answers_present = 0

    oracle_precision_sum = 0.0
    oracle_recall_sum = 0.0
    oracle_f1_sum = 0.0

    for sample in samples:
        graph_entities = get_graph_entities(
            sample.graph_triples
        )

        gold_answers = set(
            sample.answer_entities
        )

        present_answers = (
            gold_answers
            & graph_entities
        )

        missing_answers = (
            gold_answers
            - graph_entities
        )

        total_gold_answers += len(
            gold_answers
        )

        total_gold_answers_present += len(
            present_answers
        )

        if present_answers:
            any_gold_present += 1

        if not present_answers:
            no_gold_present += 1

        elif not missing_answers:
            all_gold_present += 1

        else:
            partial_questions += 1

        # 假设模型能够完美找出图中全部 Gold，
        # 且一个错误答案都不预测。
        #
        # 该值代表当前 KG coverage 对答案集合指标
        # 所施加的理论上限。
        if present_answers:
            precision = 1.0

            recall = (
                len(present_answers)
                / len(gold_answers)
            )

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )
        else:
            precision = 0.0
            recall = 0.0
            f1 = 0.0

        oracle_precision_sum += precision
        oracle_recall_sum += recall
        oracle_f1_sum += f1

    question_any_coverage = (
        any_gold_present
        / total_questions
        * 100
    )

    question_full_coverage = (
        all_gold_present
        / total_questions
        * 100
    )

    question_zero_coverage = (
        no_gold_present
        / total_questions
        * 100
    )

    answer_entity_coverage = (
        total_gold_answers_present
        / total_gold_answers
        * 100
    )

    oracle_macro_precision = (
        oracle_precision_sum
        / total_questions
        * 100
    )

    oracle_macro_recall = (
        oracle_recall_sum
        / total_questions
        * 100
    )

    oracle_macro_f1 = (
        oracle_f1_sum
        / total_questions
        * 100
    )

    print()
    print("=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Test questions                  : "
        f"{total_questions}"
    )

    print(
        f"At least one Gold in graph      : "
        f"{any_gold_present}/{total_questions} "
        f"({question_any_coverage:.2f}%)"
    )

    print(
        f"All Gold answers in graph       : "
        f"{all_gold_present}/{total_questions} "
        f"({question_full_coverage:.2f}%)"
    )

    print(
        f"Partial Gold coverage           : "
        f"{partial_questions}/{total_questions} "
        f"({partial_questions / total_questions * 100:.2f}%)"
    )

    print(
        f"No Gold answer in graph         : "
        f"{no_gold_present}/{total_questions} "
        f"({question_zero_coverage:.2f}%)"
    )

    print()

    print(
        f"Gold answer entities            : "
        f"{total_gold_answers}"
    )

    print(
        f"Gold entities present in graph  : "
        f"{total_gold_answers_present}"
    )

    print(
        f"Gold entity-level coverage      : "
        f"{answer_entity_coverage:.2f}%"
    )

    print()

    print(
        "KG-only Oracle Upper Bound"
    )

    print(
        f"Oracle Hit@1 upper bound        : "
        f"{question_any_coverage:.2f}%"
    )

    print(
        f"Oracle Macro Precision          : "
        f"{oracle_macro_precision:.2f}%"
    )

    print(
        f"Oracle Macro Recall             : "
        f"{oracle_macro_recall:.2f}%"
    )

    print(
        f"Oracle Macro F1                 : "
        f"{oracle_macro_f1:.2f}%"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()