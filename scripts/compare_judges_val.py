#!/usr/bin/env python3

import gc
import json
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

OLD_JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_lora"
    / "final"
)

NEW_JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_margin_infonce_lora"
    / "final"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "judge_side_by_side_val.json"
)


def load_validation_samples():
    """读取完整 WebQSP validation split。"""

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


def deduplicate_paths(paths):
    """去除重复 reasoning paths。"""

    unique_paths = []
    seen = set()

    for path in paths:
        if path in seen:
            continue

        seen.add(path)
        unique_paths.append(path)

    return unique_paths


def build_fixed_candidate_pool(
    samples,
    ontology,
):
    """
    使用固定 Planner + Retriever 构造一次候选池。

    后续两个 Judge 都在完全相同的候选上评估。
    """

    print("\nLoading Planner...")

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
        planner_result = planner.generate(
            question=sample.question,
            top_k=3,
            num_beams=3,
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

            retrieval_result = (
                retriever.retrieve(
                    topic_entities=(
                        sample.topic_entities
                    ),
                    head_type=head_type,
                    tail_type=tail_type,
                )
            )

            for (
                candidate,
                paths,
            ) in (
                retrieval_result
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

        retrieved_candidates = set(
            candidate_to_paths.keys()
        )

        if (
            retrieved_candidates
            & gold_answers
        ):
            retriever_gold_questions += 1

        total_candidates += len(
            candidate_to_paths
        )

        cases.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": gold_answers,
                "candidate_to_paths": dict(
                    candidate_to_paths
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

    # Planner 已经不再需要，释放显存。
    del planner

    gc.collect()

    torch.cuda.empty_cache()

    pool_stats = {
        "total_questions": len(
            samples
        ),
        "valid_plan_questions": (
            valid_plan_questions
        ),
        "no_valid_plan": (
            no_valid_plan
        ),
        "candidate_questions": len(
            cases
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

    return (
        cases,
        pool_stats,
    )


def evaluate_judge(
    name,
    adapter_path,
    cases,
):
    """在固定候选池上评估一个 Judge。"""

    print()
    print("=" * 100)
    print(f"Evaluating {name}")
    print("=" * 100)

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=adapter_path,
        margin_threshold=1.0,
    )

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    retriever_gold_questions = 0

    gold_survived_questions = 0

    raw_top1_gold = 0
    accepted_top1_gold = 0

    raw_top1_given_gold = 0
    accepted_top1_given_gold = 0

    all_rejected = 0

    retrieved_gold_all_rejected = 0

    total_accepted = 0

    gold_margins = []
    nongold_margins = []

    per_question = {}

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

        # ----------------------------------------
        # Raw ranking
        # ----------------------------------------

        raw_top1_is_gold = bool(
            judged
            and judged[0][
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

        # ----------------------------------------
        # Thresholded ranking
        # ----------------------------------------

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
            and accepted[0][
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

        if (
            retrieved_gold
            and not accepted_gold
        ):
            retrieved_gold_all_rejected += 1

        per_question[
            case["id"]
        ] = {
            "retriever_gold": (
                retrieved_gold
            ),
            "gold_survived": bool(
                accepted_gold
            ),
            "raw_top1_gold": (
                raw_top1_is_gold
            ),
            "accepted_top1_gold": (
                accepted_top1_is_gold
            ),
            "num_accepted": len(
                accepted
            ),
        }

        print(
            f"\r{name}: "
            f"{index}/{len(cases)}",
            end="",
            flush=True,
        )

    print()

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

    mean_gold_margin = (
        sum(gold_margins)
        / len(gold_margins)
    )

    mean_nongold_margin = (
        sum(nongold_margins)
        / len(nongold_margins)
    )

    metrics = {
        "name": name,

        "retriever_gold_questions": (
            retriever_gold_questions
        ),

        "gold_survived_questions": (
            gold_survived_questions
        ),

        "raw_top1_gold": (
            raw_top1_gold
        ),

        "accepted_top1_gold": (
            accepted_top1_gold
        ),

        "raw_top1_given_gold": (
            raw_top1_given_gold
        ),

        "accepted_top1_given_gold": (
            accepted_top1_given_gold
        ),

        "all_rejected": (
            all_rejected
        ),

        "retrieved_gold_all_rejected": (
            retrieved_gold_all_rejected
        ),

        "total_accepted": (
            total_accepted
        ),

        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "tn": true_negative,

        "precision": (
            candidate_precision
        ),

        "recall": (
            candidate_recall
        ),

        "f1": (
            candidate_f1
        ),

        "mean_gold_margin": (
            mean_gold_margin
        ),

        "mean_nongold_margin": (
            mean_nongold_margin
        ),

        "per_question": (
            per_question
        ),
    }

    # 释放当前 Judge。
    del judge

    gc.collect()

    torch.cuda.empty_cache()

    return metrics


def print_judge_summary(
    metrics,
    candidate_questions,
):
    """打印单个 Judge 结果。"""

    retriever_gold = metrics[
        "retriever_gold_questions"
    ]

    print()
    print("-" * 100)
    print(metrics["name"])
    print("-" * 100)

    print(
        f"Gold survived                 : "
        f"{metrics['gold_survived_questions']}/"
        f"{retriever_gold} "
        f"({metrics['gold_survived_questions'] / retriever_gold * 100:.2f}%)"
    )

    print()

    print(
        f"Raw Top-1 Gold                : "
        f"{metrics['raw_top1_gold']}/"
        f"{candidate_questions} "
        f"({metrics['raw_top1_gold'] / candidate_questions * 100:.2f}%)"
    )

    print(
        f"Raw Top-1 given Gold retrieved: "
        f"{metrics['raw_top1_given_gold']}/"
        f"{retriever_gold} "
        f"({metrics['raw_top1_given_gold'] / retriever_gold * 100:.2f}%)"
    )

    print()

    print(
        f"Accepted Top-1 Gold           : "
        f"{metrics['accepted_top1_gold']}/"
        f"{candidate_questions} "
        f"({metrics['accepted_top1_gold'] / candidate_questions * 100:.2f}%)"
    )

    print(
        f"Accepted Top-1 given Gold     : "
        f"{metrics['accepted_top1_given_gold']}/"
        f"{retriever_gold} "
        f"({metrics['accepted_top1_given_gold'] / retriever_gold * 100:.2f}%)"
    )

    print()

    print(
        f"All candidates rejected       : "
        f"{metrics['all_rejected']}"
    )

    print(
        f"Retrieved Gold all rejected   : "
        f"{metrics['retrieved_gold_all_rejected']}"
    )

    print(
        f"Avg accepted candidates       : "
        f"{metrics['total_accepted'] / candidate_questions:.2f}"
    )

    print()

    print(
        f"TP / FP / FN / TN             : "
        f"{metrics['tp']} / "
        f"{metrics['fp']} / "
        f"{metrics['fn']} / "
        f"{metrics['tn']}"
    )

    print(
        f"Candidate Precision           : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Candidate Recall              : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"Candidate F1                  : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print()

    print(
        f"Mean Gold margin              : "
        f"{metrics['mean_gold_margin']:.4f}"
    )

    print(
        f"Mean non-Gold margin          : "
        f"{metrics['mean_nongold_margin']:.4f}"
    )

    print(
        f"Margin separation             : "
        f"{metrics['mean_gold_margin'] - metrics['mean_nongold_margin']:.4f}"
    )


def compare_question_ranking(
    old_metrics,
    new_metrics,
):
    """比较两个 Judge 的问题级 Top-1 排序。"""

    old_results = old_metrics[
        "per_question"
    ]

    new_results = new_metrics[
        "per_question"
    ]

    retrieved_gold_ids = [
        sample_id
        for sample_id, item
        in old_results.items()
        if item[
            "retriever_gold"
        ]
    ]

    raw_both = 0
    raw_old_only = 0
    raw_new_only = 0
    raw_neither = 0

    accepted_both = 0
    accepted_old_only = 0
    accepted_new_only = 0
    accepted_neither = 0

    for sample_id in retrieved_gold_ids:
        old_raw = old_results[
            sample_id
        ][
            "raw_top1_gold"
        ]

        new_raw = new_results[
            sample_id
        ][
            "raw_top1_gold"
        ]

        if old_raw and new_raw:
            raw_both += 1
        elif old_raw:
            raw_old_only += 1
        elif new_raw:
            raw_new_only += 1
        else:
            raw_neither += 1

        old_accepted = old_results[
            sample_id
        ][
            "accepted_top1_gold"
        ]

        new_accepted = new_results[
            sample_id
        ][
            "accepted_top1_gold"
        ]

        if (
            old_accepted
            and new_accepted
        ):
            accepted_both += 1

        elif old_accepted:
            accepted_old_only += 1

        elif new_accepted:
            accepted_new_only += 1

        else:
            accepted_neither += 1

    return {
        "retrieved_gold_questions": len(
            retrieved_gold_ids
        ),

        "raw_both": (
            raw_both
        ),
        "raw_old_only": (
            raw_old_only
        ),
        "raw_new_only": (
            raw_new_only
        ),
        "raw_neither": (
            raw_neither
        ),

        "accepted_both": (
            accepted_both
        ),
        "accepted_old_only": (
            accepted_old_only
        ),
        "accepted_new_only": (
            accepted_new_only
        ),
        "accepted_neither": (
            accepted_neither
        ),
    }


def main():
    print("=" * 100)
    print("OntGQA Old vs Margin-InfoNCE Judge Fair Comparison")
    print("=" * 100)

    samples = load_validation_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    (
        cases,
        pool_stats,
    ) = build_fixed_candidate_pool(
        samples=samples,
        ontology=ontology,
    )

    print()
    print("=" * 100)
    print("Fixed Candidate Pool")
    print("=" * 100)

    print(
        f"Input questions             : "
        f"{pool_stats['total_questions']}"
    )

    print(
        f"Valid Planner questions     : "
        f"{pool_stats['valid_plan_questions']}"
    )

    print(
        f"No valid Planner pair       : "
        f"{pool_stats['no_valid_plan']}"
    )

    print(
        f"Questions with candidates   : "
        f"{pool_stats['candidate_questions']}"
    )

    print(
        f"No Retriever candidate      : "
        f"{pool_stats['no_candidate']}"
    )

    print(
        f"Retriever Gold questions    : "
        f"{pool_stats['retriever_gold_questions']}"
    )

    print(
        f"Total candidates            : "
        f"{pool_stats['total_candidates']}"
    )

    print(
        f"Avg candidates / candidate-q: "
        f"{pool_stats['total_candidates'] / pool_stats['candidate_questions']:.2f}"
    )

    # ============================================================
    # Old Judge
    # ============================================================

    old_metrics = evaluate_judge(
        name="Old YES/NO-SFT Judge",
        adapter_path=(
            OLD_JUDGE_ADAPTER_PATH
        ),
        cases=cases,
    )

    # ============================================================
    # New Judge
    # ============================================================

    new_metrics = evaluate_judge(
        name="New Margin-InfoNCE Judge",
        adapter_path=(
            NEW_JUDGE_ADAPTER_PATH
        ),
        cases=cases,
    )

    candidate_questions = (
        pool_stats[
            "candidate_questions"
        ]
    )

    print()
    print("=" * 100)
    print("Fair Side-by-Side Summary")
    print("=" * 100)

    print_judge_summary(
        old_metrics,
        candidate_questions,
    )

    print_judge_summary(
        new_metrics,
        candidate_questions,
    )

    comparison = (
        compare_question_ranking(
            old_metrics,
            new_metrics,
        )
    )

    print()
    print("=" * 100)
    print("Top-1 Pairwise Comparison")
    print("=" * 100)

    print(
        f"Retriever has Gold questions : "
        f"{comparison['retrieved_gold_questions']}"
    )

    print()
    print("Raw margin ranking:")

    print(
        f"  Both correct               : "
        f"{comparison['raw_both']}"
    )

    print(
        f"  Old only correct           : "
        f"{comparison['raw_old_only']}"
    )

    print(
        f"  New only correct           : "
        f"{comparison['raw_new_only']}"
    )

    print(
        f"  Neither correct            : "
        f"{comparison['raw_neither']}"
    )

    print()
    print("After margin > 1 threshold:")

    print(
        f"  Both correct               : "
        f"{comparison['accepted_both']}"
    )

    print(
        f"  Old only correct           : "
        f"{comparison['accepted_old_only']}"
    )

    print(
        f"  New only correct           : "
        f"{comparison['accepted_new_only']}"
    )

    print(
        f"  Neither correct            : "
        f"{comparison['accepted_neither']}"
    )

    # 去掉无法 JSON 序列化或没必要保存的大字段。
    output = {
        "candidate_pool": (
            pool_stats
        ),

        "old_judge": {
            key: value
            for key, value
            in old_metrics.items()
            if key != "per_question"
        },

        "new_judge": {
            key: value
            for key, value
            in new_metrics.items()
            if key != "per_question"
        },

        "top1_comparison": (
            comparison
        ),
    }

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print(
        f"Result file                  : "
        f"{RESULT_PATH}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()