#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.judge import QwenJudge
from src.ontology import OntologyGraph
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

JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_lora"
    / "final"
)

PLANNER_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "planner_retriever_validation.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "results"
    / "judge_validation.jsonl"
)


def load_all_samples(loader):
    """读取完整 validation 数据。"""

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


def load_planner_results(path):
    """读取之前保存的 Planner + Retriever validation 结果。"""

    results = {}

    with open(
        path,
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

            results[
                record["id"]
            ] = record

    return results


def merge_candidate_paths(
    candidate_to_paths,
    new_candidate_to_paths,
):
    """合并不同 Type Pair 检索得到的候选及证据路径。"""

    for candidate, paths in new_candidate_to_paths.items():
        candidate_to_paths[candidate].extend(
            paths
        )


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


def gold_answer_in_graph(sample):
    """检查当前 RoG 子图是否至少包含一个 Gold Answer。"""

    graph_entities = set()

    for head, _, tail in sample.graph_triples:
        graph_entities.add(head)
        graph_entities.add(tail)

    return any(
        answer in graph_entities
        for answer in sample.answer_entities
    )


def main():
    print("=" * 90)
    print("OntGQA Judge Full Validation")
    print("=" * 90)

    loader = WebQSPDataLoader(
        VALID_PATH
    )

    samples = load_all_samples(
        loader
    )

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    planner_results = load_planner_results(
        PLANNER_RESULT_PATH
    )

    print(
        f"\nValidation samples: "
        f"{len(samples)}"
    )

    print(
        f"Cached Planner results: "
        f"{len(planner_results)}"
    )

    print("\nLoading Judge...")

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=JUDGE_ADAPTER_PATH,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluated_questions = 0
    skipped_questions = 0

    retrieved_questions = 0
    survived_questions = 0

    total_candidates = 0
    total_accepted = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    total_retrieved_gold = 0
    total_accepted_gold = 0

    output_records = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        if not gold_answer_in_graph(sample):
            skipped_questions += 1

            output_records.append(
                {
                    "id": sample.sample_id,
                    "question": sample.question,
                    "gold_answers": sample.answer_entities,
                    "status": "skipped",
                    "reason": "gold_answer_not_in_graph",
                }
            )

            print(
                f"\rProcessing: {index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        evaluated_questions += 1

        cached_result = planner_results.get(
            sample.sample_id
        )

        if cached_result is None:
            raise RuntimeError(
                f"Planner result missing: "
                f"{sample.sample_id}"
            )

        valid_pairs = [
            tuple(pair)
            for pair in cached_result.get(
                "valid_type_pairs",
                [],
            )
        ]

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(
            list
        )

        # 使用之前 validation 中真实 Planner Top-3
        # 产生的合法 Type Pairs 重新构造候选及证据路径。
        for head_type, tail_type in valid_pairs:
            retrieval_result = retriever.retrieve(
                topic_entities=sample.topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            merge_candidate_paths(
                candidate_to_paths,
                retrieval_result.candidate_to_paths,
            )

        # 合并不同 plan 产生的重复 evidence paths
        for candidate in candidate_to_paths:
            candidate_to_paths[candidate] = (
                deduplicate_paths(
                    candidate_to_paths[
                        candidate
                    ]
                )
            )

        candidates = set(
            candidate_to_paths.keys()
        )

        gold_answers = set(
            sample.answer_entities
        )

        retrieved_gold = (
            candidates
            & gold_answers
        )

        if retrieved_gold:
            retrieved_questions += 1

        total_retrieved_gold += len(
            retrieved_gold
        )

        total_candidates += len(
            candidates
        )

        accepted_candidates = []

        # Judge 对每个 Retriever candidate 独立判断
        for candidate in sorted(
            candidate_to_paths
        ):
            paths = candidate_to_paths[
                candidate
            ]

            judge_result = judge.judge(
                question=sample.question,
                candidate=candidate,
                evidence_paths=paths,
                max_paths=3,
            )

            accepted = judge_result[
                "accepted"
            ]

            margin = judge_result[
                "margin"
            ]

            is_gold = (
                candidate
                in gold_answers
            )

            if accepted:
                total_accepted += 1

                accepted_candidates.append(
                    {
                        "candidate": candidate,
                        "margin": margin,
                        "is_gold": is_gold,
                    }
                )

            if accepted and is_gold:
                true_positive += 1

            elif accepted and not is_gold:
                false_positive += 1

            elif not accepted and is_gold:
                false_negative += 1

            else:
                true_negative += 1

        accepted_candidates.sort(
            key=lambda item: (
                item["margin"]
            ),
            reverse=True,
        )

        accepted_gold = {
            item["candidate"]
            for item in accepted_candidates
            if item["is_gold"]
        }

        total_accepted_gold += len(
            accepted_gold
        )

        if accepted_gold:
            survived_questions += 1

        output_records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.answer_entities,
                "valid_type_pairs": [
                    list(pair)
                    for pair in valid_pairs
                ],
                "num_candidates": len(
                    candidates
                ),
                "retrieved_gold": sorted(
                    retrieved_gold
                ),
                "num_accepted": len(
                    accepted_candidates
                ),
                "accepted_gold": sorted(
                    accepted_gold
                ),
                "accepted_candidates": (
                    accepted_candidates
                ),
                "status": "evaluated",
            }
        )

        print(
            f"\rProcessing: {index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

    with open(
        OUTPUT_PATH,
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

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Input questions               : "
        f"{len(samples)}"
    )

    print(
        f"Evaluated questions           : "
        f"{evaluated_questions}"
    )

    print(
        f"Skipped questions             : "
        f"{skipped_questions}"
    )

    print(
        f"Gold retrieved                : "
        f"{retrieved_questions}"
    )

    print(
        f"Gold survived Judge           : "
        f"{survived_questions}"
    )

    if evaluated_questions > 0:
        retrieval_recall = (
            retrieved_questions
            / evaluated_questions
        )

        end_to_end_recall = (
            survived_questions
            / evaluated_questions
        )

        avg_candidates = (
            total_candidates
            / evaluated_questions
        )

        avg_accepted = (
            total_accepted
            / evaluated_questions
        )

        print(
            f"Retriever question recall     : "
            f"{retrieval_recall * 100:.2f}%"
        )

        print(
            f"Judge end-to-end recall       : "
            f"{end_to_end_recall * 100:.2f}%"
        )

        print(
            f"Avg candidates before Judge   : "
            f"{avg_candidates:.2f}"
        )

        print(
            f"Avg candidates after Judge    : "
            f"{avg_accepted:.2f}"
        )

    if retrieved_questions > 0:
        survival_rate = (
            survived_questions
            / retrieved_questions
        )

        print(
            f"Question survival after Judge : "
            f"{survival_rate * 100:.2f}%"
        )

    if total_candidates > 0:
        reduction_rate = (
            1
            - total_accepted
            / total_candidates
        )

        print(
            f"Candidate reduction           : "
            f"{reduction_rate * 100:.2f}%"
        )

    print(
        f"Retrieved Gold candidates     : "
        f"{total_retrieved_gold}"
    )

    print(
        f"Accepted Gold candidates      : "
        f"{total_accepted_gold}"
    )

    print(
        f"Candidate TP                  : "
        f"{true_positive}"
    )

    print(
        f"Candidate FP                  : "
        f"{false_positive}"
    )

    print(
        f"Candidate FN                  : "
        f"{false_negative}"
    )

    print(
        f"Candidate TN                  : "
        f"{true_negative}"
    )

    if (
        true_positive
        + false_positive
        > 0
    ):
        precision = (
            true_positive
            / (
                true_positive
                + false_positive
            )
        )
    else:
        precision = 0.0

    if (
        true_positive
        + false_negative
        > 0
    ):
        recall = (
            true_positive
            / (
                true_positive
                + false_negative
            )
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

    print(
        f"Judge candidate precision     : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Judge candidate recall        : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Judge candidate F1            : "
        f"{f1 * 100:.2f}%"
    )

    print(
        f"Result file                   : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()