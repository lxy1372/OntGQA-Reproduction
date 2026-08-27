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


def merge_candidate_paths(
    candidate_to_paths,
    new_candidate_to_paths,
):
    """合并多个 Type Pair 检索得到的候选及证据路径。"""

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
    """判断当前 RoG 子图中是否至少包含一个 Gold Answer。"""

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
    print("OntGQA Planner + Retriever + Judge Validation")
    print("=" * 90)

    loader = WebQSPDataLoader(
        VALID_PATH
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
    )

    samples = loader.load_first_n(5)

    evaluated_questions = 0
    skipped_questions = 0

    retrieved_questions = 0
    judge_recovered_questions = 0

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        print("\n" + "=" * 90)
        print(
            f"[{index}] {sample.question}"
        )

        print(
            f"Gold answers: "
            f"{sample.answer_entities}"
        )

        # 当前问题子图没有答案时，Retriever/Judge 无法公平评价
        if not gold_answer_in_graph(sample):
            skipped_questions += 1

            print(
                "Status: skipped "
                "(Gold Answer 不在当前 RoG 子图中)"
            )

            continue

        evaluated_questions += 1

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

        valid_pairs = [
            pair
            for pair in predicted_pairs
            if (
                ontology.has_type(pair[0])
                and ontology.has_type(pair[1])
            )
        ]

        print("\nPlanner Top-3:")

        for rank, pair in enumerate(
            predicted_pairs,
            start=1,
        ):
            valid_flag = (
                pair in valid_pairs
            )

            print(
                f"{rank}: "
                f"<{pair[0]}, {pair[1]}> "
                f"{'[valid]' if valid_flag else '[invalid]'}"
            )

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(list)

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

        # 合并不同 Type Pair 产生的重复证据路径
        for candidate in candidate_to_paths:
            candidate_to_paths[candidate] = (
                deduplicate_paths(
                    candidate_to_paths[candidate]
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

        print(
            f"\nRetrieved candidates: "
            f"{len(candidates)}"
        )

        print(
            f"Retrieved gold: "
            f"{sorted(retrieved_gold)}"
        )

        if retrieved_gold:
            retrieved_questions += 1

        accepted_candidates = []

        # Judge 对 Retriever 的每一个候选独立判断
        for candidate, paths in candidate_to_paths.items():
            result = judge.judge(
                question=sample.question,
                candidate=candidate,
                evidence_paths=paths,
                max_paths=3,
            )

            accepted = result[
                "accepted"
            ]

            is_gold = (
                candidate in gold_answers
            )

            if accepted:
                accepted_candidates.append(
                    (
                        candidate,
                        result["margin"],
                        is_gold,
                    )
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
            key=lambda item: item[1],
            reverse=True,
        )

        accepted_gold = {
            candidate
            for candidate, _, is_gold
            in accepted_candidates
            if is_gold
        }

        if accepted_gold:
            judge_recovered_questions += 1

        print(
            f"\nJudge accepted: "
            f"{len(accepted_candidates)}"
        )

        print(
            f"Accepted gold: "
            f"{sorted(accepted_gold)}"
        )

        print("\nTop accepted candidates:")

        if not accepted_candidates:
            print("None")

        for (
            candidate,
            margin,
            is_gold,
        ) in accepted_candidates[:10]:
            flag = (
                "GOLD"
                if is_gold
                else "NON-GOLD"
            )

            print(
                f"{candidate} | "
                f"margin={margin:.4f} | "
                f"{flag}"
            )

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Input questions            : "
        f"{len(samples)}"
    )

    print(
        f"Evaluated questions        : "
        f"{evaluated_questions}"
    )

    print(
        f"Skipped questions          : "
        f"{skipped_questions}"
    )

    print(
        f"Gold retrieved             : "
        f"{retrieved_questions}"
    )

    print(
        f"Gold survived Judge        : "
        f"{judge_recovered_questions}"
    )

    print(
        f"Candidate TP               : "
        f"{true_positive}"
    )

    print(
        f"Candidate FP               : "
        f"{false_positive}"
    )

    print(
        f"Candidate FN               : "
        f"{false_negative}"
    )

    print(
        f"Candidate TN               : "
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
        f"Judge candidate precision  : "
        f"{precision * 100:.2f}%"
    )

    print(
        f"Judge candidate recall     : "
        f"{recall * 100:.2f}%"
    )

    print(
        f"Judge candidate F1         : "
        f"{f1 * 100:.2f}%"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()