#!/usr/bin/env python3

import gc
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.judge import QwenJudge
from src.ontology import OntologyGraph
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


JUDGE_CONFIGS = [
    (
        "Old YES/NO-SFT",
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
    (
        "Margin-InfoNCE Epoch 2",
        PROJECT_ROOT
        / "outputs"
        / "judge_margin_infonce_lora"
        / "epoch_2",
    ),
    (
        "Margin-InfoNCE Epoch 3",
        PROJECT_ROOT
        / "outputs"
        / "judge_margin_infonce_lora"
        / "epoch_3",
    ),
]


def load_samples():
    """读取完整 validation split。"""

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


def deduplicate_paths(
    paths,
):
    """去除重复 reasoning paths。"""

    unique_paths = []
    seen = set()

    for path in paths:
        if path in seen:
            continue

        seen.add(
            path
        )

        unique_paths.append(
            path
        )

    return unique_paths


def build_fixed_candidate_pool(
    samples,
    ontology,
):
    """固定 Planner 与 Retriever，构造公平 Judge 候选池。"""

    print()
    print("Loading Planner...")

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=PLANNER_ADAPTER_PATH,
    )

    cases = []

    no_valid_plan = 0
    no_candidate = 0

    valid_plan_questions = 0
    retriever_gold_questions = 0

    total_candidates = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        planner_result = (
            planner.generate(
                question=sample.question,
                top_k=3,
                num_beams=3,
            )
        )

        predicted_pairs = (
            planner_result[
                "predicted_type_pairs"
            ][:3]
        )

        valid_pairs = []

        for (
            head_type,
            tail_type,
        ) in predicted_pairs:

            if (
                ontology.has_type(
                    head_type
                )
                and ontology.has_type(
                    tail_type
                )
            ):
                valid_pairs.append(
                    (
                        head_type,
                        tail_type,
                    )
                )

        if not valid_pairs:
            no_valid_plan += 1

            print(
                f"\rBuilding candidate pool: "
                f"{index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        valid_plan_questions += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(
            list
        )

        for (
            head_type,
            tail_type,
        ) in valid_pairs:

            result = retriever.retrieve(
                topic_entities=(
                    sample.topic_entities
                ),
                head_type=head_type,
                tail_type=tail_type,
            )

            for (
                candidate,
                paths,
            ) in (
                result
                .candidate_to_paths
                .items()
            ):
                candidate_to_paths[
                    candidate
                ].extend(
                    paths
                )

        for candidate in list(
            candidate_to_paths.keys()
        ):
            candidate_to_paths[
                candidate
            ] = deduplicate_paths(
                candidate_to_paths[
                    candidate
                ]
            )

        if not candidate_to_paths:
            no_candidate += 1

            print(
                f"\rBuilding candidate pool: "
                f"{index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        gold_answers = set(
            sample.answer_entities
        )

        candidate_set = set(
            candidate_to_paths.keys()
        )

        if (
            candidate_set
            & gold_answers
        ):
            retriever_gold_questions += 1

        total_candidates += len(
            candidate_set
        )

        cases.append(
            {
                "id": (
                    sample.sample_id
                ),
                "question": (
                    sample.question
                ),
                "gold_answers": (
                    gold_answers
                ),
                "candidate_to_paths": (
                    dict(
                        candidate_to_paths
                    )
                ),
            }
        )

        print(
            f"\rBuilding candidate pool: "
            f"{index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

    del planner

    gc.collect()

    torch.cuda.empty_cache()

    return {
        "cases": cases,
        "input_questions": len(
            samples
        ),
        "valid_plan_questions": (
            valid_plan_questions
        ),
        "no_valid_plan": (
            no_valid_plan
        ),
        "no_candidate": (
            no_candidate
        ),
        "retriever_gold_questions": (
            retriever_gold_questions
        ),
        "total_candidates": (
            total_candidates
        ),
    }


def calculate_set_metrics(
    predictions,
    gold_answers,
):
    """计算单个问题的答案集合 Precision / Recall / F1。"""

    prediction_set = set(
        predictions
    )

    gold_set = set(
        gold_answers
    )

    if not prediction_set:
        precision = 0.0
    else:
        precision = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                prediction_set
            )
        )

    if not gold_set:
        recall = 0.0
    else:
        recall = (
            len(
                prediction_set
                & gold_set
            )
            / len(
                gold_set
            )
        )

    if (
        precision
        + recall
        == 0
    ):
        f1 = 0.0

    else:
        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return (
        precision,
        recall,
        f1,
    )


def evaluate_checkpoint(
    name,
    adapter_path,
    cases,
):
    """评估一个 Judge checkpoint。"""

    print()
    print("=" * 100)
    print(
        f"Evaluating: {name}"
    )
    print("=" * 100)

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=adapter_path,
        margin_threshold=1.0,
    )

    retriever_gold_questions = 0

    gold_survived_questions = 0

    raw_top1_gold = 0
    raw_top1_given_gold = 0

    accepted_top1_gold = 0
    accepted_top1_given_gold = 0

    all_rejected = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    total_accepted = 0

    macro_precision_sum = 0.0
    macro_recall_sum = 0.0
    macro_f1_sum = 0.0

    gold_margins = []
    nongold_margins = []

    for index, case in enumerate(
        cases,
        start=1,
    ):
        gold_answers = case[
            "gold_answers"
        ]

        candidate_to_paths = case[
            "candidate_to_paths"
        ]

        retrieved_gold = bool(
            set(
                candidate_to_paths.keys()
            )
            & gold_answers
        )

        if retrieved_gold:
            retriever_gold_questions += 1

        judged = []

        for candidate in sorted(
            candidate_to_paths
        ):
            result = judge.judge(
                question=case[
                    "question"
                ],
                candidate=candidate,
                evidence_paths=(
                    candidate_to_paths[
                        candidate
                    ]
                ),
                max_paths=3,
            )

            margin = float(
                result[
                    "margin"
                ]
            )

            accepted = bool(
                result[
                    "accepted"
                ]
            )

            is_gold = (
                candidate
                in gold_answers
            )

            if is_gold:
                gold_margins.append(
                    margin
                )
            else:
                nongold_margins.append(
                    margin
                )

            if (
                is_gold
                and accepted
            ):
                true_positive += 1

            elif (
                not is_gold
                and accepted
            ):
                false_positive += 1

            elif (
                is_gold
                and not accepted
            ):
                false_negative += 1

            else:
                true_negative += 1

            judged.append(
                {
                    "candidate": (
                        candidate
                    ),
                    "margin": (
                        margin
                    ),
                    "accepted": (
                        accepted
                    ),
                    "is_gold": (
                        is_gold
                    ),
                }
            )

        judged.sort(
            key=lambda item: (
                item[
                    "margin"
                ]
            ),
            reverse=True,
        )

        # ========================================================
        # Raw ranking
        # ========================================================

        raw_top1_is_gold = bool(
            judged
            and judged[
                0
            ][
                "is_gold"
            ]
        )

        if raw_top1_is_gold:
            raw_top1_gold += 1

        if (
            retrieved_gold
            and raw_top1_is_gold
        ):
            raw_top1_given_gold += 1

        # ========================================================
        # Thresholded answers
        # ========================================================

        accepted = [
            item
            for item in judged
            if item[
                "accepted"
            ]
        ]

        total_accepted += len(
            accepted
        )

        if not accepted:
            all_rejected += 1

        accepted_answers = [
            item[
                "candidate"
            ]
            for item in accepted
        ]

        accepted_gold = [
            item
            for item in accepted
            if item[
                "is_gold"
            ]
        ]

        if accepted_gold:
            gold_survived_questions += 1

        accepted_top1_is_gold = bool(
            accepted
            and accepted[
                0
            ][
                "is_gold"
            ]
        )

        if accepted_top1_is_gold:
            accepted_top1_gold += 1

        if (
            retrieved_gold
            and accepted_top1_is_gold
        ):
            accepted_top1_given_gold += 1

        (
            question_precision,
            question_recall,
            question_f1,
        ) = calculate_set_metrics(
            predictions=accepted_answers,
            gold_answers=gold_answers,
        )

        macro_precision_sum += (
            question_precision
        )

        macro_recall_sum += (
            question_recall
        )

        macro_f1_sum += (
            question_f1
        )

        print(
            f"\r{name}: "
            f"{index}/{len(cases)}",
            end="",
            flush=True,
        )

    print()

    num_cases = len(
        cases
    )

    candidate_precision = (
        true_positive
        / (
            true_positive
            + false_positive
        )
        if (
            true_positive
            + false_positive
        ) > 0
        else 0.0
    )

    candidate_recall = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    candidate_f1 = (
        2
        * candidate_precision
        * candidate_recall
        / (
            candidate_precision
            + candidate_recall
        )
        if (
            candidate_precision
            + candidate_recall
        ) > 0
        else 0.0
    )

    macro_precision = (
        macro_precision_sum
        / num_cases
    )

    macro_recall = (
        macro_recall_sum
        / num_cases
    )

    macro_f1 = (
        macro_f1_sum
        / num_cases
    )

    mean_gold_margin = (
        sum(
            gold_margins
        )
        / len(
            gold_margins
        )
    )

    mean_nongold_margin = (
        sum(
            nongold_margins
        )
        / len(
            nongold_margins
        )
    )

    result = {
        "name": name,

        "retriever_gold": (
            retriever_gold_questions
        ),

        "gold_survived": (
            gold_survived_questions
        ),

        "raw_top1": (
            raw_top1_gold
        ),

        "raw_top1_given_gold": (
            raw_top1_given_gold
        ),

        "accepted_top1": (
            accepted_top1_gold
        ),

        "accepted_top1_given_gold": (
            accepted_top1_given_gold
        ),

        "all_rejected": (
            all_rejected
        ),

        "avg_accepted": (
            total_accepted
            / num_cases
        ),

        "candidate_precision": (
            candidate_precision
        ),

        "candidate_recall": (
            candidate_recall
        ),

        "candidate_f1": (
            candidate_f1
        ),

        "macro_precision": (
            macro_precision
        ),

        "macro_recall": (
            macro_recall
        ),

        "macro_f1": (
            macro_f1
        ),

        "mean_gold_margin": (
            mean_gold_margin
        ),

        "mean_nongold_margin": (
            mean_nongold_margin
        ),
    }

    del judge

    gc.collect()

    torch.cuda.empty_cache()

    return result


def print_result(
    result,
    num_cases,
):
    """打印一个 checkpoint 的结果。"""

    retriever_gold = result[
        "retriever_gold"
    ]

    print()
    print("-" * 100)
    print(
        result[
            "name"
        ]
    )
    print("-" * 100)

    print(
        f"Gold survived                   : "
        f"{result['gold_survived']}/"
        f"{retriever_gold} "
        f"({result['gold_survived'] / retriever_gold * 100:.2f}%)"
    )

    print(
        f"Raw Top-1 given Gold            : "
        f"{result['raw_top1_given_gold']}/"
        f"{retriever_gold} "
        f"({result['raw_top1_given_gold'] / retriever_gold * 100:.2f}%)"
    )

    print(
        f"Accepted Top-1 given Gold       : "
        f"{result['accepted_top1_given_gold']}/"
        f"{retriever_gold} "
        f"({result['accepted_top1_given_gold'] / retriever_gold * 100:.2f}%)"
    )

    print(
        f"Accepted Top-1 / candidate-q    : "
        f"{result['accepted_top1']}/"
        f"{num_cases} "
        f"({result['accepted_top1'] / num_cases * 100:.2f}%)"
    )

    print()

    print(
        f"All candidates rejected         : "
        f"{result['all_rejected']}"
    )

    print(
        f"Avg accepted answers            : "
        f"{result['avg_accepted']:.2f}"
    )

    print()

    print(
        f"Candidate Precision             : "
        f"{result['candidate_precision'] * 100:.2f}%"
    )

    print(
        f"Candidate Recall                : "
        f"{result['candidate_recall'] * 100:.2f}%"
    )

    print(
        f"Candidate F1                    : "
        f"{result['candidate_f1'] * 100:.2f}%"
    )

    print()

    print(
        f"Question Macro Precision        : "
        f"{result['macro_precision'] * 100:.2f}%"
    )

    print(
        f"Question Macro Recall           : "
        f"{result['macro_recall'] * 100:.2f}%"
    )

    print(
        f"Question Macro F1               : "
        f"{result['macro_f1'] * 100:.2f}%"
    )

    print()

    print(
        f"Mean Gold margin                : "
        f"{result['mean_gold_margin']:.4f}"
    )

    print(
        f"Mean non-Gold margin            : "
        f"{result['mean_nongold_margin']:.4f}"
    )

    print(
        f"Margin separation               : "
        f"{result['mean_gold_margin'] - result['mean_nongold_margin']:.4f}"
    )


def main():
    print("=" * 100)
    print("OntGQA Judge Checkpoint Comparison on WebQSP Validation")
    print("=" * 100)

    samples = load_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    pool = build_fixed_candidate_pool(
        samples=samples,
        ontology=ontology,
    )

    cases = pool[
        "cases"
    ]

    print()
    print("=" * 100)
    print("Fixed Candidate Pool")
    print("=" * 100)

    print(
        f"Input questions           : "
        f"{pool['input_questions']}"
    )

    print(
        f"Valid Planner questions   : "
        f"{pool['valid_plan_questions']}"
    )

    print(
        f"No valid Planner pair     : "
        f"{pool['no_valid_plan']}"
    )

    print(
        f"Candidate questions       : "
        f"{len(cases)}"
    )

    print(
        f"No Retriever candidate    : "
        f"{pool['no_candidate']}"
    )

    print(
        f"Retriever Gold questions  : "
        f"{pool['retriever_gold_questions']}"
    )

    print(
        f"Total candidates          : "
        f"{pool['total_candidates']}"
    )

    results = []

    for (
        name,
        adapter_path,
    ) in JUDGE_CONFIGS:

        result = evaluate_checkpoint(
            name=name,
            adapter_path=adapter_path,
            cases=cases,
        )

        results.append(
            result
        )

    print()
    print("=" * 100)
    print("Checkpoint Results")
    print("=" * 100)

    for result in results:
        print_result(
            result=result,
            num_cases=len(cases),
        )

    print()
    print("=" * 100)
    print("Compact Comparison")
    print("=" * 100)

    print(
        f"{'Model':30s} "
        f"{'GoldSurv':>9s} "
        f"{'RawTop1':>9s} "
        f"{'AccTop1':>9s} "
        f"{'MacroF1':>9s} "
        f"{'CandF1':>9s}"
    )

    print("-" * 100)

    for result in results:
        retriever_gold = (
            result[
                "retriever_gold"
            ]
        )

        gold_survival = (
            result[
                "gold_survived"
            ]
            / retriever_gold
            * 100
        )

        raw_top1 = (
            result[
                "raw_top1_given_gold"
            ]
            / retriever_gold
            * 100
        )

        accepted_top1 = (
            result[
                "accepted_top1_given_gold"
            ]
            / retriever_gold
            * 100
        )

        print(
            f"{result['name'][:30]:30s} "
            f"{gold_survival:8.2f}% "
            f"{raw_top1:8.2f}% "
            f"{accepted_top1:8.2f}% "
            f"{result['macro_f1'] * 100:8.2f}% "
            f"{result['candidate_f1'] * 100:8.2f}%"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()