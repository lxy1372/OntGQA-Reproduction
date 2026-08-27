#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
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

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
    / "final"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "planner_retriever_validation.jsonl"
)


def load_all_samples(loader):
    """读取 Parquet 文件中的全部样本。"""

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


def gold_answer_in_graph(sample):
    """判断至少一个 Gold Answer 是否存在于当前问题子图中。"""

    graph_entities = set()

    for head, _, tail in sample.graph_triples:
        graph_entities.add(head)
        graph_entities.add(tail)

    return any(
        answer in graph_entities
        for answer in sample.answer_entities
    )


def main():
    print("=" * 80)
    print("OntGQA Planner + Retriever Validation")
    print("=" * 80)

    loader = WebQSPDataLoader(
        VALID_PATH
    )

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )

    samples = load_all_samples(
        loader
    )

    print(
        f"\nValidation samples: "
        f"{len(samples)}"
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluated_count = 0
    skipped_count = 0

    recovered_count = 0

    total_predicted_pairs = 0
    total_valid_pairs = 0

    total_candidates = 0
    total_paths = 0

    results = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        # 当前 RoG 子图中不存在 Gold Answer 时无法评价 Retriever
        if not gold_answer_in_graph(sample):
            skipped_count += 1

            results.append(
                {
                    "id": sample.sample_id,
                    "question": sample.question,
                    "gold_answers": sample.answer_entities,
                    "status": "skipped",
                }
            )

            print(
                f"\rProcessing: {index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        evaluated_count += 1

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

        # 只保留官方本体中真实存在的实体类型
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

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(list)

        pair_results = []

        # 分别使用 Top-3 中合法的 Type Pair 进行检索
        for head_type, tail_type in valid_pairs:
            retrieval_result = retriever.retrieve(
                topic_entities=sample.topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            for (
                candidate,
                paths,
            ) in retrieval_result.candidate_to_paths.items():
                candidate_to_paths[candidate].extend(
                    paths
                )

            pair_results.append(
                {
                    "head_type": head_type,
                    "tail_type": tail_type,
                    "num_paths": retrieval_result.num_paths,
                    "num_candidates": len(
                        retrieval_result.candidates
                    ),
                }
            )

        candidates = set(
            candidate_to_paths.keys()
        )

        gold_answers = set(
            sample.answer_entities
        )

        recovered_gold = (
            candidates
            & gold_answers
        )

        recovered = bool(
            recovered_gold
        )

        if recovered:
            recovered_count += 1

        path_count = sum(
            len(paths)
            for paths in candidate_to_paths.values()
        )

        total_candidates += len(
            candidates
        )

        total_paths += path_count

        results.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "topic_entities": sample.topic_entities,
                "gold_answers": sample.answer_entities,
                "predicted_type_pairs": [
                    list(pair)
                    for pair in predicted_pairs
                ],
                "valid_type_pairs": [
                    list(pair)
                    for pair in valid_pairs
                ],
                "pair_results": pair_results,
                "num_candidates": len(candidates),
                "num_paths": path_count,
                "recovered_gold": sorted(
                    recovered_gold
                ),
                "recovered": recovered,
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
        RESULT_PATH,
        "w",
        encoding="utf-8",
    ) as f:
        for result in results:
            f.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print(
        f"Input samples             : "
        f"{len(samples)}"
    )

    print(
        f"Evaluated                 : "
        f"{evaluated_count}"
    )

    print(
        f"Skipped                   : "
        f"{skipped_count}"
    )

    print(
        f"Predicted type pairs      : "
        f"{total_predicted_pairs}"
    )

    print(
        f"Valid ontology type pairs : "
        f"{total_valid_pairs}"
    )

    print(
        f"Gold recovered            : "
        f"{recovered_count}"
    )

    if evaluated_count > 0:
        retrieval_recall = (
            recovered_count
            / evaluated_count
            * 100
        )

        avg_candidates = (
            total_candidates
            / evaluated_count
        )

        avg_paths = (
            total_paths
            / evaluated_count
        )

        print(
            f"Retrieval recall          : "
            f"{retrieval_recall:.2f}%"
        )

        print(
            f"Avg candidates            : "
            f"{avg_candidates:.2f}"
        )

        print(
            f"Avg evidence paths        : "
            f"{avg_paths:.2f}"
        )

    print(
        f"Result file               : "
        f"{RESULT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()