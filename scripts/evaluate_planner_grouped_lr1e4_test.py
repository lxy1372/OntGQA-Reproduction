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

ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_grouped_lora_lr1e4"
    / "final"
)

RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "planner_grouped_lr1e4_test.jsonl"
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


def build_outgoing_index(
    graph_triples,
):
    """按照头实体建立正向邻接索引。"""

    outgoing = defaultdict(list)

    for head, relation, tail in graph_triples:
        outgoing[head].append(
            (
                relation,
                tail,
            )
        )

    return outgoing


def find_shortest_paths(
    outgoing,
    topic_entities,
    target_entity,
):
    """搜索 Topic Entity 到 Gold Answer 的 1-hop / 2-hop 最短路径。"""

    one_hop_paths = []

    for topic_entity in topic_entities:
        for relation, tail in outgoing.get(
            topic_entity,
            [],
        ):
            if tail == target_entity:
                one_hop_paths.append(
                    (
                        (
                            topic_entity,
                            relation,
                            tail,
                        ),
                    )
                )

    if one_hop_paths:
        return one_hop_paths

    two_hop_paths = []

    for topic_entity in topic_entities:
        for first_relation, middle_entity in outgoing.get(
            topic_entity,
            [],
        ):
            for second_relation, tail_entity in outgoing.get(
                middle_entity,
                [],
            ):
                if tail_entity == target_entity:
                    two_hop_paths.append(
                        (
                            (
                                topic_entity,
                                first_relation,
                                middle_entity,
                            ),
                            (
                                middle_entity,
                                second_relation,
                                tail_entity,
                            ),
                        )
                    )

    return two_hop_paths


def align_path_to_ontology(
    ontology,
    path,
):
    """将 Gold reasoning path 映射为 ontology-consistent Type Pair。"""

    signatures = []

    for _, relation, _ in path:
        signature = (
            ontology.get_relation_signature(
                relation
            )
        )

        if signature is None:
            return None

        signatures.append(
            signature
        )

    for index in range(
        len(signatures) - 1
    ):
        if (
            signatures[index][1]
            != signatures[index + 1][0]
        ):
            return None

    return (
        signatures[0][0],
        signatures[-1][1],
    )


def get_oracle_type_pairs(
    sample,
    ontology,
):
    """构造当前问题全部合法 Oracle Type Pairs。"""

    outgoing = build_outgoing_index(
        sample.graph_triples
    )

    oracle_pairs = set()

    for gold_answer in set(
        sample.answer_entities
    ):
        paths = find_shortest_paths(
            outgoing=outgoing,
            topic_entities=sample.topic_entities,
            target_entity=gold_answer,
        )

        for path in paths:
            type_pair = align_path_to_ontology(
                ontology=ontology,
                path=path,
            )

            if type_pair is not None:
                oracle_pairs.add(
                    type_pair
                )

    return oracle_pairs


def main():
    print("=" * 100)
    print("OntGQA Multi-Positive Planner lr=1e-4 Evaluation on WebQSP Test")
    print("=" * 100)

    samples = load_all_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    print(
        f"\nTest questions: "
        f"{len(samples)}"
    )

    print("\nLoading Planner...")

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )

    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluable_questions = 0
    unavailable_questions = 0

    top1_hits = 0
    top2_hits = 0
    top3_hits = 0

    total_predictions = 0
    valid_predictions = 0

    invalid_top1 = 0
    questions_with_valid_prediction = 0

    retriever_gold_recovered = 0

    total_candidates = 0
    total_paths = 0

    output_records = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        oracle_pairs = get_oracle_type_pairs(
            sample=sample,
            ontology=ontology,
        )

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

            total_predictions += 1

            valid = (
                ontology.has_type(
                    head_type
                )
                and ontology.has_type(
                    tail_type
                )
            )

            if valid:
                valid_predictions += 1

                valid_pairs.append(
                    (
                        head_type,
                        tail_type,
                    )
                )

        if valid_pairs:
            questions_with_valid_prediction += 1

        if predicted_pairs:
            first_pair = (
                predicted_pairs[0]
            )

            if (
                not ontology.has_type(
                    first_pair[0]
                )
                or not ontology.has_type(
                    first_pair[1]
                )
            ):
                invalid_top1 += 1

        if oracle_pairs:
            evaluable_questions += 1

            if (
                len(predicted_pairs) >= 1
                and predicted_pairs[0]
                in oracle_pairs
            ):
                top1_hits += 1

            if any(
                pair in oracle_pairs
                for pair in predicted_pairs[:2]
            ):
                top2_hits += 1

            if any(
                pair in oracle_pairs
                for pair in predicted_pairs[:3]
            ):
                top3_hits += 1

        else:
            unavailable_questions += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidates = set()
        path_count = 0

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

            candidates.update(
                retrieval_result.candidates
            )

            path_count += (
                retrieval_result.num_paths
            )

        gold_answers = set(
            sample.answer_entities
        )

        recovered_gold = (
            candidates
            & gold_answers
        )

        if recovered_gold:
            retriever_gold_recovered += 1

        total_candidates += len(
            candidates
        )

        total_paths += path_count

        output_records.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": (
                    sample.answer_entities
                ),
                "oracle_type_pairs": [
                    list(pair)
                    for pair in sorted(
                        oracle_pairs
                    )
                ],
                "predicted_type_pairs": [
                    list(pair)
                    for pair in predicted_pairs
                ],
                "valid_type_pairs": [
                    list(pair)
                    for pair in valid_pairs
                ],
                "retrieved_gold": sorted(
                    recovered_gold
                ),
                "num_candidates": len(
                    candidates
                ),
                "num_paths": path_count,
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

    print("\n" + "=" * 100)
    print("New Planner Summary")
    print("=" * 100)

    print(
        f"Test questions                    : "
        f"{total_questions}"
    )

    print(
        f"Evaluable with Oracle pair        : "
        f"{evaluable_questions}/{total_questions} "
        f"({evaluable_questions / total_questions * 100:.2f}%)"
    )

    print(
        f"Unavailable Oracle pair           : "
        f"{unavailable_questions}"
    )

    print()

    print(
        f"Planner Top-1 pair accuracy       : "
        f"{top1_hits}/{evaluable_questions} "
        f"({top1_hits / evaluable_questions * 100:.2f}%)"
    )

    print(
        f"Planner Top-2 pair accuracy       : "
        f"{top2_hits}/{evaluable_questions} "
        f"({top2_hits / evaluable_questions * 100:.2f}%)"
    )

    print(
        f"Planner Top-3 pair accuracy       : "
        f"{top3_hits}/{evaluable_questions} "
        f"({top3_hits / evaluable_questions * 100:.2f}%)"
    )

    print()

    print(
        f"Questions with valid prediction   : "
        f"{questions_with_valid_prediction}/{total_questions} "
        f"({questions_with_valid_prediction / total_questions * 100:.2f}%)"
    )

    print(
        f"Invalid Top-1 prediction          : "
        f"{invalid_top1}/{total_questions} "
        f"({invalid_top1 / total_questions * 100:.2f}%)"
    )

    if total_predictions > 0:
        print(
            f"Ontology-valid predicted pairs    : "
            f"{valid_predictions}/{total_predictions} "
            f"({valid_predictions / total_predictions * 100:.2f}%)"
        )

    print()

    print(
        f"Planner -> Retriever Gold recall  : "
        f"{retriever_gold_recovered}/{total_questions} "
        f"({retriever_gold_recovered / total_questions * 100:.2f}%)"
    )

    print(
        f"Avg Retriever candidates          : "
        f"{total_candidates / total_questions:.2f}"
    )

    print(
        f"Avg Retriever evidence paths      : "
        f"{total_paths / total_questions:.2f}"
    )

    print()

    print("-" * 100)
    print("Old independent-SFT Planner")
    print("-" * 100)
    print("Top-1 pair accuracy              : 65.48%")
    print("Top-2 pair accuracy              : 72.97%")
    print("Top-3 pair accuracy              : 75.16%")
    print("Ontology-valid predicted pairs   : 72.45%")
    print("Planner -> Retriever Gold recall : 75.68%")

    print()

    print("-" * 100)
    print("Grouped Planner, lr=2e-5")
    print("-" * 100)
    print("Top-1 pair accuracy              : 17.23%")
    print("Top-2 pair accuracy              : 26.58%")
    print("Top-3 pair accuracy              : 28.84%")
    print("Ontology-valid predicted pairs   : 36.91%")
    print("Planner -> Retriever Gold recall : 38.51%")

    print()

    print(
        f"Result file                      : "
        f"{RESULT_PATH}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()