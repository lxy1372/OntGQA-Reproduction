#!/usr/bin/env python3

import json
import re
import string
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.generator import QwenGenerator
from src.judge import QwenJudge
from src.ontology import OntologyGraph
from src.pipeline import OntGQAPipeline
from src.planner import QwenPlanner
from src.retriever import OntologyRetriever


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

ONTOLOGY_PATH = (
    PROJECT_ROOT
    / "data"
    / "ontology"
    / "ontology_graph_freebase.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "Qwen2-1.5B-Instruct"
)

PLANNER_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
    / "final"
)

JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_lora"
    / "final"
)

GENERATOR_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "generator_lora"
    / "final"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pipeline_test.jsonl"
)


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


def normalize_answer(text):
    """
    按 WebQSP / RoG evaluator 的答案规范化规则处理字符串。

    处理顺序：
    1. 小写
    2. 删除英文标点
    3. 删除 a / an / the
    4. 合并多余空格
    """

    if text is None:
        return ""

    text = str(
        text
    ).lower()

    # 删除英文标点符号
    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    # 删除英文冠词
    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    # 合并多余空格
    text = " ".join(
        text.split()
    )

    return text


def normalize_answer_list(
    answers,
):
    """规范化答案列表并去重，同时保留原始顺序。"""

    normalized = []
    seen = set()

    for answer in answers:
        value = normalize_answer(
            answer
        )

        if not value:
            continue

        if value in seen:
            continue

        seen.add(
            value
        )

        normalized.append(
            value
        )

    return normalized


def calculate_paper_metrics(
    predicted_answers,
    gold_answers,
):
    """
    计算 OntGQA 论文口径的单题 Hit@1 和集合 F1。

    Hit@1：
    只使用排名第一的预测答案。

    F1：
    对规范化后的预测答案集合和 Gold 集合计算
    Precision、Recall 和 F1。
    """

    normalized_predictions = (
        normalize_answer_list(
            predicted_answers
        )
    )

    normalized_gold = (
        normalize_answer_list(
            gold_answers
        )
    )

    prediction_set = set(
        normalized_predictions
    )

    gold_set = set(
        normalized_gold
    )

    # Hit@1 只判断最高排名预测
    if normalized_predictions:
        top1 = normalized_predictions[
            0
        ]

        hit_at_1 = (
            top1 in gold_set
        )
    else:
        top1 = None
        hit_at_1 = False

    true_positive = len(
        prediction_set
        & gold_set
    )

    if prediction_set:
        precision = (
            true_positive
            / len(prediction_set)
        )
    else:
        precision = 0.0

    if gold_set:
        recall = (
            true_positive
            / len(gold_set)
        )
    else:
        recall = 0.0

    if (
        precision
        + recall
        > 0
    ):
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
        f1 = 0.0

    return {
        "top1": top1,
        "hit_at_1": hit_at_1,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "normalized_predictions": (
            normalized_predictions
        ),
        "normalized_gold": (
            normalized_gold
        ),
    }


def main():
    print("=" * 100)
    print("OntGQA WebQSP Test Evaluation")
    print("=" * 100)

    samples = load_all_samples()

    print(
        f"\nTest questions: "
        f"{len(samples)}"
    )

    if len(samples) != 1628:
        raise RuntimeError(
            "WebQSP test 样本数量不是 1628，"
            "请检查两个 parquet 文件。"
        )

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    print("\nLoading Planner...")

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=PLANNER_ADAPTER_PATH,
    )

    print("Loading Judge...")

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=JUDGE_ADAPTER_PATH,
        margin_threshold=1.0,
    )

    print("Loading Generator...")

    generator = QwenGenerator(
        model_path=MODEL_PATH,
        adapter_path=GENERATOR_ADAPTER_PATH,
    )

    pipeline = OntGQAPipeline(
        ontology=ontology,
        planner=planner,
        judge=judge,
        generator=generator,
        retriever_class=OntologyRetriever,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hit_at_1_count = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    source_counter = Counter()
    backoff_counter = Counter()

    judge_questions = 0
    judge_hit_at_1 = 0

    generator_questions = 0
    generator_hit_at_1 = 0

    total_predicted_answers = 0

    output_records = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        result = pipeline.answer(
            question=sample.question,
            topic_entities=sample.topic_entities,
            graph_triples=sample.graph_triples,
        )

        predicted_answers = result[
            "answers"
        ]

        gold_answers = (
            sample.answer_entities
        )

        metrics = (
            calculate_paper_metrics(
                predicted_answers=predicted_answers,
                gold_answers=gold_answers,
            )
        )

        if metrics[
            "hit_at_1"
        ]:
            hit_at_1_count += 1

        macro_precision_sum += (
            metrics["precision"]
        )

        macro_recall_sum += (
            metrics["recall"]
        )

        macro_f1_sum += (
            metrics["f1"]
        )

        answer_source = result[
            "answer_source"
        ]

        source_counter[
            answer_source
        ] += 1

        if answer_source == "judge":
            judge_questions += 1

            if metrics[
                "hit_at_1"
            ]:
                judge_hit_at_1 += 1

        elif answer_source == "generator":
            generator_questions += 1

            if metrics[
                "hit_at_1"
            ]:
                generator_hit_at_1 += 1

        backoff_reason = result.get(
            "backoff_reason"
        )

        if backoff_reason:
            backoff_counter[
                backoff_reason
            ] += 1

        total_predicted_answers += len(
            metrics[
                "normalized_predictions"
            ]
        )

        output_records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": gold_answers,
                "predicted_answers": predicted_answers,
                "normalized_gold": metrics[
                    "normalized_gold"
                ],
                "normalized_predictions": metrics[
                    "normalized_predictions"
                ],
                "top1_prediction": metrics[
                    "top1"
                ],
                "hit_at_1": metrics[
                    "hit_at_1"
                ],
                "precision": metrics[
                    "precision"
                ],
                "recall": metrics[
                    "recall"
                ],
                "f1": metrics[
                    "f1"
                ],
                "answer_source": (
                    answer_source
                ),
                "backoff_reason": (
                    backoff_reason
                ),
                "num_candidates": result.get(
                    "num_candidates",
                    0,
                ),
                "predicted_type_pairs": [
                    list(pair)
                    for pair in result.get(
                        "predicted_type_pairs",
                        [],
                    )
                ],
                "valid_type_pairs": [
                    list(pair)
                    for pair in result.get(
                        "valid_type_pairs",
                        [],
                    )
                ],
            }
        )

        print(
            f"\rProcessing: "
            f"{index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for record in output_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    total_questions = len(
        samples
    )

    hit_at_1 = (
        hit_at_1_count
        / total_questions
        * 100
    )

    macro_precision = (
        macro_precision_sum
        / total_questions
        * 100
    )

    macro_recall = (
        macro_recall_sum
        / total_questions
        * 100
    )

    macro_f1 = (
        macro_f1_sum
        / total_questions
        * 100
    )

    print("\n" + "=" * 100)
    print("Paper-aligned Summary")
    print("=" * 100)

    print(
        f"Test questions                 : "
        f"{total_questions}"
    )

    print()

    print(
        f"Hit@1                          : "
        f"{hit_at_1_count}/{total_questions} "
        f"({hit_at_1:.2f}%)"
    )

    print(
        f"Macro Precision                : "
        f"{macro_precision:.2f}%"
    )

    print(
        f"Macro Recall                   : "
        f"{macro_recall:.2f}%"
    )

    print(
        f"Macro F1                       : "
        f"{macro_f1:.2f}%"
    )

    print()

    print(
        f"Judge answers                  : "
        f"{judge_questions}"
    )

    if judge_questions > 0:
        print(
            f"Judge Hit@1                    : "
            f"{judge_hit_at_1}/{judge_questions} "
            f"({judge_hit_at_1 / judge_questions * 100:.2f}%)"
        )

    print(
        f"Generator backoffs             : "
        f"{generator_questions}"
    )

    if generator_questions > 0:
        print(
            f"Generator Hit@1                : "
            f"{generator_hit_at_1}/{generator_questions} "
            f"({generator_hit_at_1 / generator_questions * 100:.2f}%)"
        )

    print()

    print(
        f"Backoff: no valid type pair    : "
        f"{backoff_counter['no_valid_type_pair']}"
    )

    print(
        f"Backoff: no candidate          : "
        f"{backoff_counter['no_retrieved_candidate']}"
    )

    print(
        f"Backoff: all rejected          : "
        f"{backoff_counter['all_candidates_rejected']}"
    )

    print()

    print(
        f"Avg predicted answers          : "
        f"{total_predicted_answers / total_questions:.2f}"
    )

    print(
        f"Result file                    : "
        f"{RESULT_PATH}"
    )

    print()

    print("-" * 100)
    print("Paper reference: Qwen2-1.5B")
    print("Hit@1 = 89.93%")
    print("F1    = 76.27%")
    print("-" * 100)

    print("=" * 100)


if __name__ == "__main__":
    main()