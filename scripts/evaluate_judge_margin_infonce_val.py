#!/usr/bin/env python3

import sys
from collections import defaultdict
from pathlib import Path


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

# 继续使用之前表现最好的旧 Planner。
PLANNER_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
    / "final"
)

# 新训练的 Margin-InfoNCE Judge。
JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_margin_infonce_lora"
    / "final"
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


def merge_candidate_paths(
    candidate_to_paths,
    new_candidate_to_paths,
):
    """合并多个 Type Pair 产生的候选证据。"""

    for (
        candidate,
        paths,
    ) in new_candidate_to_paths.items():
        candidate_to_paths[
            candidate
        ].extend(
            paths
        )


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


def main():
    print("=" * 100)
    print("OntGQA Margin-InfoNCE Judge Evaluation on WebQSP Validation")
    print("=" * 100)

    samples = load_validation_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    print()
    print(
        f"Validation questions: "
        f"{len(samples)}"
    )

    print("\nLoading Planner...")

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=PLANNER_ADAPTER_PATH,
    )

    print("Loading new Judge...")

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=JUDGE_ADAPTER_PATH,
        margin_threshold=1.0,
    )

    total_questions = len(
        samples
    )

    evaluated_questions = 0
    skipped_questions = 0

    no_valid_plan = 0
    no_candidates = 0

    # Retriever 问题级统计
    retriever_gold_questions = 0

    # Judge 问题级统计
    judge_gold_survived_questions = 0

    # 至少有一个 accepted candidate 的问题
    questions_with_accepted = 0

    # accepted candidates 中 Top-1 是 Gold
    judge_top1_gold = 0

    # Retriever 已找到 Gold，
    # 但 Judge Top-1 排错的问题。
    retrieved_gold_top1_miss = 0

    # Retriever 已找到 Gold，
    # 但 Judge 把全部 Gold 拒绝。
    retrieved_gold_all_rejected = 0

    total_candidates_before = 0
    total_candidates_after = 0

    # Candidate-level confusion matrix
    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    retrieved_gold_candidates = 0
    accepted_gold_candidates = 0

    # Margin 诊断
    positive_margins = []
    negative_margins = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        # ========================================================
        # Planner
        # ========================================================

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
            skipped_questions += 1
            no_valid_plan += 1

            print(
                f"\rProcessing: "
                f"{index}/{total_questions}",
                end="",
                flush=True,
            )

            continue

        # ========================================================
        # Retriever
        # ========================================================

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

            merge_candidate_paths(
                candidate_to_paths,
                retrieval_result.candidate_to_paths,
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
            skipped_questions += 1
            no_candidates += 1

            print(
                f"\rProcessing: "
                f"{index}/{total_questions}",
                end="",
                flush=True,
            )

            continue

        evaluated_questions += 1

        gold_answers = set(
            sample.answer_entities
        )

        retrieved_candidates = set(
            candidate_to_paths.keys()
        )

        retrieved_gold = (
            retrieved_candidates
            & gold_answers
        )

        if retrieved_gold:
            retriever_gold_questions += 1

        total_candidates_before += len(
            retrieved_candidates
        )

        retrieved_gold_candidates += len(
            retrieved_gold
        )

        # ========================================================
        # Judge
        # ========================================================

        judged_candidates = []

        for candidate in sorted(
            candidate_to_paths
        ):
            judge_result = judge.judge(
                question=sample.question,
                candidate=candidate,
                evidence_paths=(
                    candidate_to_paths[
                        candidate
                    ]
                ),
                max_paths=3,
            )

            is_gold = (
                candidate in gold_answers
            )

            accepted = bool(
                judge_result[
                    "accepted"
                ]
            )

            margin = float(
                judge_result[
                    "margin"
                ]
            )

            if is_gold:
                positive_margins.append(
                    margin
                )
            else:
                negative_margins.append(
                    margin
                )

            if (
                is_gold
                and accepted
            ):
                true_positive += 1
                accepted_gold_candidates += 1

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

            judged_candidates.append(
                {
                    "candidate": candidate,
                    "margin": margin,
                    "accepted": accepted,
                    "is_gold": is_gold,
                }
            )

        # 与当前 pipeline 完全一致：
        # 按 Judge margin 从高到低排序。
        judged_candidates.sort(
            key=lambda item: (
                item[
                    "margin"
                ]
            ),
            reverse=True,
        )

        accepted_candidates = [
            item
            for item in judged_candidates
            if item[
                "accepted"
            ]
        ]

        total_candidates_after += len(
            accepted_candidates
        )

        if accepted_candidates:
            questions_with_accepted += 1

        accepted_gold = [
            item
            for item in accepted_candidates
            if item[
                "is_gold"
            ]
        ]

        if accepted_gold:
            judge_gold_survived_questions += 1

        if (
            accepted_candidates
            and accepted_candidates[
                0
            ][
                "is_gold"
            ]
        ):
            judge_top1_gold += 1

        if retrieved_gold:
            if not accepted_gold:
                retrieved_gold_all_rejected += 1

            if (
                not accepted_candidates
                or not accepted_candidates[
                    0
                ][
                    "is_gold"
                ]
            ):
                retrieved_gold_top1_miss += 1

        print(
            f"\rProcessing: "
            f"{index}/{total_questions}",
            end="",
            flush=True,
        )

    print()

    # ============================================================
    # Metrics
    # ============================================================

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

    avg_before = (
        total_candidates_before
        / evaluated_questions
        if evaluated_questions > 0
        else 0.0
    )

    avg_after = (
        total_candidates_after
        / evaluated_questions
        if evaluated_questions > 0
        else 0.0
    )

    reduction = (
        1.0
        - (
            avg_after
            / avg_before
        )
        if avg_before > 0
        else 0.0
    )

    mean_positive_margin = (
        sum(
            positive_margins
        )
        / len(
            positive_margins
        )
        if positive_margins
        else 0.0
    )

    mean_negative_margin = (
        sum(
            negative_margins
        )
        / len(
            negative_margins
        )
        if negative_margins
        else 0.0
    )

    print()
    print("=" * 100)
    print("New Margin-InfoNCE Judge Summary")
    print("=" * 100)

    print(
        f"Input questions                  : "
        f"{total_questions}"
    )

    print(
        f"Evaluated questions              : "
        f"{evaluated_questions}"
    )

    print(
        f"Skipped questions                : "
        f"{skipped_questions}"
    )

    print(
        f"  No valid Planner pair          : "
        f"{no_valid_plan}"
    )

    print(
        f"  No Retriever candidate         : "
        f"{no_candidates}"
    )

    print()

    print(
        f"Retriever Gold questions         : "
        f"{retriever_gold_questions}/"
        f"{evaluated_questions} "
        f"({retriever_gold_questions / evaluated_questions * 100:.2f}%)"
    )

    print(
        f"Judge Gold survived              : "
        f"{judge_gold_survived_questions}/"
        f"{retriever_gold_questions} "
        f"({judge_gold_survived_questions / retriever_gold_questions * 100:.2f}%)"
    )

    print(
        f"Judge end-to-end Gold recall     : "
        f"{judge_gold_survived_questions}/"
        f"{evaluated_questions} "
        f"({judge_gold_survived_questions / evaluated_questions * 100:.2f}%)"
    )

    print()

    print(
        f"Questions with accepted answers  : "
        f"{questions_with_accepted}/"
        f"{evaluated_questions} "
        f"({questions_with_accepted / evaluated_questions * 100:.2f}%)"
    )

    print(
        f"Judge Top-1 Gold                 : "
        f"{judge_top1_gold}/"
        f"{evaluated_questions} "
        f"({judge_top1_gold / evaluated_questions * 100:.2f}%)"
    )

    if retriever_gold_questions > 0:
        print(
            f"Top-1 given Retriever has Gold   : "
            f"{judge_top1_gold}/"
            f"{retriever_gold_questions} "
            f"({judge_top1_gold / retriever_gold_questions * 100:.2f}%)"
        )

    print(
        f"Retriever Gold -> all rejected   : "
        f"{retrieved_gold_all_rejected}"
    )

    print(
        f"Retriever Gold -> Top-1 miss     : "
        f"{retrieved_gold_top1_miss}"
    )

    print()

    print(
        f"Avg candidates before Judge      : "
        f"{avg_before:.2f}"
    )

    print(
        f"Avg candidates after Judge       : "
        f"{avg_after:.2f}"
    )

    print(
        f"Candidate reduction              : "
        f"{reduction * 100:.2f}%"
    )

    print()

    print(
        f"Retrieved Gold candidates        : "
        f"{retrieved_gold_candidates}"
    )

    print(
        f"Accepted Gold candidates         : "
        f"{accepted_gold_candidates}"
    )

    print()

    print(
        f"TP / FP / FN / TN                : "
        f"{true_positive} / "
        f"{false_positive} / "
        f"{false_negative} / "
        f"{true_negative}"
    )

    print(
        f"Candidate Precision              : "
        f"{candidate_precision * 100:.2f}%"
    )

    print(
        f"Candidate Recall                 : "
        f"{candidate_recall * 100:.2f}%"
    )

    print(
        f"Candidate F1                     : "
        f"{candidate_f1 * 100:.2f}%"
    )

    print()

    print(
        f"Mean Gold margin                 : "
        f"{mean_positive_margin:.4f}"
    )

    print(
        f"Mean non-Gold margin             : "
        f"{mean_negative_margin:.4f}"
    )

    print(
        f"Mean margin separation           : "
        f"{mean_positive_margin - mean_negative_margin:.4f}"
    )

    print()

    print("-" * 100)
    print("Old YES/NO-SFT Judge reference")
    print("-" * 100)

    print(
        "Retriever Gold questions        : "
        "181/235 (77.02%)"
    )

    print(
        "Judge Gold survived             : "
        "170/181 (93.92%)"
    )

    print(
        "Judge end-to-end Gold recall    : "
        "170/235 (72.34%)"
    )

    print(
        "Avg candidates before Judge     : "
        "90.32"
    )

    print(
        "Avg candidates after Judge      : "
        "11.28"
    )

    print(
        "Candidate reduction             : "
        "87.51%"
    )

    print(
        "TP / FP / FN / TN               : "
        "896 / 1754 / 133 / 18442"
    )

    print(
        "Candidate Precision             : "
        "33.81%"
    )

    print(
        "Candidate Recall                : "
        "87.07%"
    )

    print(
        "Candidate F1                    : "
        "48.71%"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()