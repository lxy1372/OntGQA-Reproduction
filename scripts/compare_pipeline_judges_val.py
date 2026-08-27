#!/usr/bin/env python3

import gc
import re
import string
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.generator import QwenGenerator
from src.judge import QwenJudge
from src.ontology import OntologyGraph
from src.pipeline import OntGQAPipeline
from src.planner import QwenPlanner
from src.retriever import OntologyRetriever


VALIDATION_PATH = (
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

GENERATOR_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "generator_lora"
    / "final"
)

JUDGE_CONFIGS = [
    (
        "Old YES/NO-SFT Judge",
        PROJECT_ROOT
        / "outputs"
        / "judge_lora"
        / "final",
    ),
    (
        "Margin-InfoNCE Epoch 1",
        PROJECT_ROOT
        / "outputs"
        / "judge_margin_infonce_lora"
        / "epoch_1",
    ),
]


def load_samples():
    loader = WebQSPDataLoader(
        VALIDATION_PATH
    )

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


def normalize_answer(text):
    """与正式 WebQSP evaluator 保持一致。"""

    if text is None:
        return ""

    text = str(text).lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text


def normalize_list(
    answers,
):
    normalized = []
    seen = set()

    for answer in answers:
        answer = normalize_answer(
            answer
        )

        if not answer:
            continue

        if answer in seen:
            continue

        seen.add(answer)
        normalized.append(
            answer
        )

    return normalized


def calculate_metrics(
    predictions,
    gold_answers,
):
    predictions = normalize_list(
        predictions
    )

    gold_answers = normalize_list(
        gold_answers
    )

    prediction_set = set(
        predictions
    )

    gold_set = set(
        gold_answers
    )

    hit1 = (
        bool(predictions)
        and predictions[0]
        in gold_set
    )

    if prediction_set:
        precision = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                prediction_set
            )
        )
    else:
        precision = 0.0

    if gold_set:
        recall = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                gold_set
            )
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

    return (
        hit1,
        precision,
        recall,
        f1,
    )


def evaluate_pipeline(
    name,
    judge_adapter_path,
    samples,
    ontology,
):
    print()
    print("=" * 100)
    print(
        f"Evaluating: {name}"
    )
    print("=" * 100)

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=PLANNER_ADAPTER_PATH,
    )

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=judge_adapter_path,
        margin_threshold=1.0,
    )

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

    total = len(
        samples
    )

    hit1_count = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    judge_questions = 0
    judge_hit1 = 0

    generator_questions = 0
    generator_hit1 = 0

    no_valid_type_pair = 0
    no_retrieved_candidate = 0
    all_candidates_rejected = 0

    total_predicted_answers = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        result = pipeline.answer(
            question=sample.question,
            topic_entities=sample.topic_entities,
            graph_triples=sample.graph_triples,
        )

        predictions = result[
            "answers"
        ]

        (
            hit1,
            precision,
            recall,
            f1,
        ) = calculate_metrics(
            predictions=predictions,
            gold_answers=(
                sample.answer_entities
            ),
        )

        if hit1:
            hit1_count += 1

        macro_precision_sum += (
            precision
        )

        macro_recall_sum += (
            recall
        )

        macro_f1_sum += (
            f1
        )

        total_predicted_answers += len(
            normalize_list(
                predictions
            )
        )

        source = result[
            "answer_source"
        ]

        if source == "judge":
            judge_questions += 1

            if hit1:
                judge_hit1 += 1

        elif source == "generator":
            generator_questions += 1

            if hit1:
                generator_hit1 += 1

            reason = result[
                "backoff_reason"
            ]

            if (
                reason
                == "no_valid_type_pair"
            ):
                no_valid_type_pair += 1

            elif (
                reason
                == "no_retrieved_candidate"
            ):
                no_retrieved_candidate += 1

            elif (
                reason
                == "all_candidates_rejected"
            ):
                all_candidates_rejected += 1

        print(
            f"\r{name}: "
            f"{index}/{total}",
            end="",
            flush=True,
        )

    print()

    metrics = {
        "name": name,

        "hit1": (
            hit1_count
            / total
        ),

        "hit1_count": (
            hit1_count
        ),

        "macro_precision": (
            macro_precision_sum
            / total
        ),

        "macro_recall": (
            macro_recall_sum
            / total
        ),

        "macro_f1": (
            macro_f1_sum
            / total
        ),

        "judge_questions": (
            judge_questions
        ),

        "judge_hit1": (
            judge_hit1
        ),

        "generator_questions": (
            generator_questions
        ),

        "generator_hit1": (
            generator_hit1
        ),

        "no_valid_type_pair": (
            no_valid_type_pair
        ),

        "no_retrieved_candidate": (
            no_retrieved_candidate
        ),

        "all_candidates_rejected": (
            all_candidates_rejected
        ),

        "avg_predicted_answers": (
            total_predicted_answers
            / total
        ),
    }

    del pipeline
    del planner
    del judge
    del generator

    gc.collect()

    torch.cuda.empty_cache()

    return metrics


def print_result(
    metrics,
    total,
):
    print()
    print("-" * 100)
    print(
        metrics[
            "name"
        ]
    )
    print("-" * 100)

    print(
        f"Hit@1                    : "
        f"{metrics['hit1_count']}/{total} "
        f"({metrics['hit1'] * 100:.2f}%)"
    )

    print(
        f"Macro Precision          : "
        f"{metrics['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Macro Recall             : "
        f"{metrics['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Macro F1                 : "
        f"{metrics['macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        f"Judge answers            : "
        f"{metrics['judge_questions']}"
    )

    if (
        metrics[
            "judge_questions"
        ]
        > 0
    ):
        print(
            f"Judge Hit@1              : "
            f"{metrics['judge_hit1']}/"
            f"{metrics['judge_questions']} "
            f"({metrics['judge_hit1'] / metrics['judge_questions'] * 100:.2f}%)"
        )

    print(
        f"Generator backoffs       : "
        f"{metrics['generator_questions']}"
    )

    if (
        metrics[
            "generator_questions"
        ]
        > 0
    ):
        print(
            f"Generator Hit@1          : "
            f"{metrics['generator_hit1']}/"
            f"{metrics['generator_questions']} "
            f"({metrics['generator_hit1'] / metrics['generator_questions'] * 100:.2f}%)"
        )

    print()

    print(
        f"Backoff no valid pair    : "
        f"{metrics['no_valid_type_pair']}"
    )

    print(
        f"Backoff no candidate     : "
        f"{metrics['no_retrieved_candidate']}"
    )

    print(
        f"Backoff all rejected     : "
        f"{metrics['all_candidates_rejected']}"
    )

    print(
        f"Avg predicted answers    : "
        f"{metrics['avg_predicted_answers']:.2f}"
    )


def main():
    print("=" * 100)
    print("OntGQA Full Pipeline Judge A/B on WebQSP Validation")
    print("=" * 100)

    samples = load_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    results = []

    for (
        name,
        judge_adapter_path,
    ) in JUDGE_CONFIGS:

        result = evaluate_pipeline(
            name=name,
            judge_adapter_path=(
                judge_adapter_path
            ),
            samples=samples,
            ontology=ontology,
        )

        results.append(
            result
        )

    print()
    print("=" * 100)
    print("Full Pipeline Results")
    print("=" * 100)

    for result in results:
        print_result(
            metrics=result,
            total=len(samples),
        )

    print()
    print("=" * 100)
    print("Compact Comparison")
    print("=" * 100)

    print(
        f"{'Judge':30s} "
        f"{'Hit@1':>9s} "
        f"{'MacroP':>9s} "
        f"{'MacroR':>9s} "
        f"{'MacroF1':>9s} "
        f"{'AvgAns':>9s}"
    )

    print("-" * 100)

    for result in results:
        print(
            f"{result['name'][:30]:30s} "
            f"{result['hit1'] * 100:8.2f}% "
            f"{result['macro_precision'] * 100:8.2f}% "
            f"{result['macro_recall'] * 100:8.2f}% "
            f"{result['macro_f1'] * 100:8.2f}% "
            f"{result['avg_predicted_answers']:9.2f}"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()