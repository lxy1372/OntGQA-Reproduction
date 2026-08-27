#!/usr/bin/env python3

import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph
from src.planner import QwenPlanner
from src.retriever import OntologyRetriever


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

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
    / "final"
)


def load_all_samples():
    """读取完整 WebQSP 训练集。"""

    samples = []

    for path in WEBQSP_PATHS:
        loader = WebQSPDataLoader(path)

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
    """合并不同 Type Pair 检索得到的候选答案。"""

    for candidate, paths in new_candidate_to_paths.items():
        candidate_to_paths[candidate].extend(
            paths
        )


def main():
    print("=" * 90)
    print("Judge Candidate Inspection with Planner Top-3")
    print("=" * 90)

    samples = load_all_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )

    questions_with_prediction = 0
    questions_with_valid_plan = 0
    questions_with_positive = 0
    questions_without_positive = 0

    total_predicted_pairs = 0
    total_valid_pairs = 0

    total_candidates = 0
    total_positive_candidates = 0
    total_negative_candidates = 0

    total_gold_answers = 0

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
            planner_result["predicted_type_pairs"][:3]
        )

        total_predicted_pairs += len(
            predicted_pairs
        )

        if predicted_pairs:
            questions_with_prediction += 1

        valid_pairs = []

        for head_type, tail_type in predicted_pairs:
            if (
                ontology.has_type(head_type)
                and ontology.has_type(tail_type)
            ):
                valid_pairs.append(
                    (
                        head_type,
                        tail_type,
                    )
                )

        total_valid_pairs += len(
            valid_pairs
        )

        if valid_pairs:
            questions_with_valid_plan += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(list)

        for head_type, tail_type in valid_pairs:
            result = retriever.retrieve(
                topic_entities=sample.topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            merge_candidate_paths(
                candidate_to_paths,
                result.candidate_to_paths,
            )

        candidates = set(
            candidate_to_paths.keys()
        )

        gold_answers = set(
            sample.answer_entities
        )

        positive_candidates = (
            candidates
            & gold_answers
        )

        negative_candidates = (
            candidates
            - gold_answers
        )

        total_gold_answers += len(
            gold_answers
        )

        total_candidates += len(
            candidates
        )

        total_positive_candidates += len(
            positive_candidates
        )

        total_negative_candidates += len(
            negative_candidates
        )

        if positive_candidates:
            questions_with_positive += 1
        else:
            questions_without_positive += 1

        print(
            f"\rProcessing: {index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

    print("\n" + "=" * 90)
    print("Summary")
    print("=" * 90)

    print(
        f"Input questions             : "
        f"{len(samples)}"
    )

    print(
        f"Questions with prediction   : "
        f"{questions_with_prediction}"
    )

    print(
        f"Questions with valid plan   : "
        f"{questions_with_valid_plan}"
    )

    print(
        f"Questions with positive     : "
        f"{questions_with_positive}"
    )

    print(
        f"Questions without positive  : "
        f"{questions_without_positive}"
    )

    print(
        f"Predicted type pairs        : "
        f"{total_predicted_pairs}"
    )

    print(
        f"Valid ontology type pairs   : "
        f"{total_valid_pairs}"
    )

    print(
        f"Gold answer entities        : "
        f"{total_gold_answers}"
    )

    print(
        f"Retrieved candidates        : "
        f"{total_candidates}"
    )

    print(
        f"Positive candidates         : "
        f"{total_positive_candidates}"
    )

    print(
        f"Negative candidates         : "
        f"{total_negative_candidates}"
    )

    if len(samples) > 0:
        print(
            f"Avg positive/question       : "
            f"{total_positive_candidates / len(samples):.2f}"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()