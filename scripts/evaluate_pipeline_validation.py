#!/usr/bin/env python3

import json
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


VALID_PATH = (
    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "validation-00000-of-00001.parquet"
)

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
    / "pipeline_validation.jsonl"
)


def load_all_samples(loader):
    """读取完整 validation 数据集。"""

    samples = []

    for row_group_index in range(
        loader.num_row_groups()
    ):
        samples.extend(
            loader.load_row_group(
                row_group_index
            )
        )

    return samples


def calculate_metrics(
    predicted_answers,
    gold_answers,
):
    """计算单个问题的答案集合指标。"""

    predicted = set(
        predicted_answers
    )

    gold = set(
        gold_answers
    )

    true_positive = len(
        predicted & gold
    )

    false_positive = len(
        predicted - gold
    )

    false_negative = len(
        gold - predicted
    )

    hit = (
        true_positive > 0
    )

    exact_match = (
        predicted == gold
    )

    if predicted:
        precision = (
            true_positive
            / len(predicted)
        )
    else:
        precision = 0.0

    if gold:
        recall = (
            true_positive
            / len(gold)
        )
    else:
        recall = 0.0

    if precision + recall > 0:
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
        "hit": hit,
        "exact_match": exact_match,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
    }


def main():
    print("=" * 90)
    print("OntGQA End-to-End Pipeline Validation")
    print("=" * 90)

    loader = WebQSPDataLoader(
        VALID_PATH
    )

    samples = load_all_samples(
        loader
    )

    print(
        f"\nValidation samples: "
        f"{len(samples)}"
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

    hit_count = 0
    exact_match_count = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    micro_tp = 0
    micro_fp = 0
    micro_fn = 0

    total_predicted_answers = 0

    source_counter = Counter()
    backoff_counter = Counter()

    generator_questions = 0
    generator_hit_count = 0

    judge_questions = 0
    judge_hit_count = 0

    result_records = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        result = pipeline.answer(
            question=sample.question,
            topic_entities=sample.topic_entities,
            graph_triples=sample.graph_triples,
        )

        predicted_answers = (
            result["answers"]
        )

        gold_answers = (
            sample.answer_entities
        )

        metrics = calculate_metrics(
            predicted_answers=predicted_answers,
            gold_answers=gold_answers,
        )

        if metrics["hit"]:
            hit_count += 1

        if metrics["exact_match"]:
            exact_match_count += 1

        macro_precision_sum += (
            metrics["precision"]
        )

        macro_recall_sum += (
            metrics["recall"]
        )

        macro_f1_sum += (
            metrics["f1"]
        )

        micro_tp += metrics["tp"]
        micro_fp += metrics["fp"]
        micro_fn += metrics["fn"]

        total_predicted_answers += len(
            set(predicted_answers)
        )

        answer_source = result[
            "answer_source"
        ]

        source_counter[
            answer_source
        ] += 1

        backoff_reason = result.get(
            "backoff_reason"
        )

        if backoff_reason is not None:
            backoff_counter[
                backoff_reason
            ] += 1

        if answer_source == "generator":
            generator_questions += 1

            if metrics["hit"]:
                generator_hit_count += 1

        elif answer_source == "judge":
            judge_questions += 1

            if metrics["hit"]:
                judge_hit_count += 1

        result_records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": gold_answers,
                "predicted_answers": predicted_answers,
                "answer_source": answer_source,
                "backoff_reason": backoff_reason,
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
                "num_candidates": result.get(
                    "num_candidates",
                    0,
                ),
                "hit": metrics["hit"],
                "exact_match": (
                    metrics["exact_match"]
                ),
                "precision": (
                    metrics["precision"]
                ),
                "recall": (
                    metrics["recall"]
                ),
                "f1": metrics["f1"],
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
        for record in result_records:
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

    macro_precision = (
        macro_precision_sum
        / total_questions
    )

    macro_recall = (
        macro_recall_sum
        / total_questions
    )

    macro_f1 = (
        macro_f1_sum
        / total_questions
    )

    if (
        micro_tp
        + micro_fp
        > 0
    ):
        micro_precision = (
            micro_tp
            / (
                micro_tp
                + micro_fp
            )
        )
    else:
        micro_precision = 0.0

    if (
        micro_tp
        + micro_fn
        > 0
    ):
        micro_recall = (
            micro_tp
            / (
                micro_tp
                + micro_fn
            )
        )
    else:
        micro_recall = 0.0

    if (
        micro_precision
        + micro_recall
        > 0
    ):
        micro_f1 = (
            2
            * micro_precision
            * micro_recall
            / (
                micro_precision
                + micro_recall
            )
        )
    else:
        micro_f1 = 0.0

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Input questions             : "
        f"{total_questions}"
    )

    print(
        f"Hit questions               : "
        f"{hit_count}/{total_questions} "
        f"({hit_count / total_questions * 100:.2f}%)"
    )

    print(
        f"Exact Match                 : "
        f"{exact_match_count}/{total_questions} "
        f"({exact_match_count / total_questions * 100:.2f}%)"
    )

    print()

    print(
        f"Macro Precision             : "
        f"{macro_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall                : "
        f"{macro_recall * 100:.2f}%"
    )

    print(
        f"Macro F1                    : "
        f"{macro_f1 * 100:.2f}%"
    )

    print()

    print(
        f"Micro Precision             : "
        f"{micro_precision * 100:.2f}%"
    )

    print(
        f"Micro Recall                : "
        f"{micro_recall * 100:.2f}%"
    )

    print(
        f"Micro F1                    : "
        f"{micro_f1 * 100:.2f}%"
    )

    print()

    print(
        f"Judge answers               : "
        f"{judge_questions}"
    )

    print(
        f"Judge question hits         : "
        f"{judge_hit_count}/{judge_questions}"
        if judge_questions > 0
        else "Judge question hits         : 0/0"
    )

    print(
        f"Generator backoffs          : "
        f"{generator_questions}"
    )

    print(
        f"Generator question hits     : "
        f"{generator_hit_count}/{generator_questions}"
        if generator_questions > 0
        else "Generator question hits     : 0/0"
    )

    print()

    print(
        f"Backoff: no valid type pair : "
        f"{backoff_counter['no_valid_type_pair']}"
    )

    print(
        f"Backoff: no candidate       : "
        f"{backoff_counter['no_retrieved_candidate']}"
    )

    print(
        f"Backoff: all rejected       : "
        f"{backoff_counter['all_candidates_rejected']}"
    )

    print()

    print(
        f"Avg predicted answers       : "
        f"{total_predicted_answers / total_questions:.2f}"
    )

    print(
        f"Result file                 : "
        f"{RESULT_PATH}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()