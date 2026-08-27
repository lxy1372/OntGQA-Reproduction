#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph


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

PIPELINE_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pipeline_test.jsonl"
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


def load_pipeline_results():
    """读取已经保存的 Planner test 输出。"""

    results = {}

    with open(
        PIPELINE_RESULT_PATH,
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
    print("OntGQA Planner Evaluation on WebQSP Test")
    print("=" * 100)

    samples = load_all_samples()

    pipeline_results = (
        load_pipeline_results()
    )

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    total_questions = len(
        samples
    )

    evaluable_questions = 0
    unavailable_questions = 0

    top1_hits = 0
    top2_hits = 0
    top3_hits = 0

    questions_with_prediction = 0
    questions_with_valid_prediction = 0

    total_predictions = 0
    valid_predictions = 0

    invalid_top1 = 0

    oracle_pair_count = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        record = pipeline_results.get(
            sample.sample_id
        )

        if record is None:
            raise RuntimeError(
                f"Missing pipeline result: "
                f"{sample.sample_id}"
            )

        oracle_pairs = (
            get_oracle_type_pairs(
                sample=sample,
                ontology=ontology,
            )
        )

        if not oracle_pairs:
            unavailable_questions += 1

            print(
                f"\rProcessing: "
                f"{index}/{total_questions}",
                end="",
                flush=True,
            )

            continue

        evaluable_questions += 1

        oracle_pair_count += len(
            oracle_pairs
        )

        predicted_pairs = [
            tuple(pair)
            for pair in record.get(
                "predicted_type_pairs",
                [],
            )[:3]
        ]

        if predicted_pairs:
            questions_with_prediction += 1

        valid_pair_flags = []

        for head_type, tail_type in predicted_pairs:
            total_predictions += 1

            valid = (
                ontology.has_type(
                    head_type
                )
                and ontology.has_type(
                    tail_type
                )
            )

            valid_pair_flags.append(
                valid
            )

            if valid:
                valid_predictions += 1

        if any(
            valid_pair_flags
        ):
            questions_with_valid_prediction += 1

        if (
            predicted_pairs
            and not valid_pair_flags[0]
        ):
            invalid_top1 += 1

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

        print(
            f"\rProcessing: "
            f"{index}/{total_questions}",
            end="",
            flush=True,
        )

    print()

    print("\n" + "=" * 100)
    print("Summary")
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

    if evaluable_questions > 0:
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
            f"Questions with prediction         : "
            f"{questions_with_prediction}/{evaluable_questions}"
        )

        print(
            f"Questions with valid prediction   : "
            f"{questions_with_valid_prediction}/{evaluable_questions}"
        )

        print(
            f"Invalid Top-1 prediction          : "
            f"{invalid_top1}/{evaluable_questions} "
            f"({invalid_top1 / evaluable_questions * 100:.2f}%)"
        )

    print()

    if total_predictions > 0:
        print(
            f"Ontology-valid predicted pairs    : "
            f"{valid_predictions}/{total_predictions} "
            f"({valid_predictions / total_predictions * 100:.2f}%)"
        )

    if evaluable_questions > 0:
        print(
            f"Avg Oracle pairs/question         : "
            f"{oracle_pair_count / evaluable_questions:.2f}"
        )

    print()

    print("-" * 100)

    print(
        "Predicted Planner -> Retriever Gold recall : 75.68%"
    )

    print(
        "Oracle Planner    -> Retriever Gold recall : 95.21%"
    )

    print("-" * 100)

    print("=" * 100)


if __name__ == "__main__":
    main()