#!/usr/bin/env python3

import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph
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
    """搜索 Topic Entity 到 Gold Answer 的 1-hop 或 2-hop 最短路径。"""

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
        signature = ontology.get_relation_signature(
            relation
        )

        if signature is None:
            return None

        signatures.append(
            signature
        )

    # 完整 relation path 在 ontology 中需要保持类型连续
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
    """根据 Gold reasoning paths 构造 Oracle Type Pair 集合。"""

    outgoing = build_outgoing_index(
        sample.graph_triples
    )

    type_pairs = set()

    found_gold_path = False

    for gold_answer in set(
        sample.answer_entities
    ):
        paths = find_shortest_paths(
            outgoing=outgoing,
            topic_entities=sample.topic_entities,
            target_entity=gold_answer,
        )

        if paths:
            found_gold_path = True

        for path in paths:
            type_pair = align_path_to_ontology(
                ontology=ontology,
                path=path,
            )

            if type_pair is not None:
                type_pairs.add(
                    type_pair
                )

    return (
        type_pairs,
        found_gold_path,
    )


def get_graph_entities(
    graph_triples,
):
    """提取当前问题子图中的全部实体。"""

    entities = set()

    for head, _, tail in graph_triples:
        entities.add(
            head
        )
        entities.add(
            tail
        )

    return entities


def main():
    print("=" * 100)
    print("OntGQA Oracle-Type Retriever Evaluation on WebQSP Test")
    print("=" * 100)

    samples = load_all_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    total_questions = len(
        samples
    )

    kg_has_gold = 0

    questions_with_gold_path = 0
    questions_without_gold_path = 0

    questions_with_oracle_pair = 0
    questions_without_oracle_pair = 0

    oracle_recovered = 0

    total_candidates = 0
    total_paths = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        gold_answers = set(
            sample.answer_entities
        )

        graph_entities = get_graph_entities(
            sample.graph_triples
        )

        if (
            gold_answers
            & graph_entities
        ):
            kg_has_gold += 1

        (
            oracle_pairs,
            found_gold_path,
        ) = get_oracle_type_pairs(
            sample=sample,
            ontology=ontology,
        )

        if found_gold_path:
            questions_with_gold_path += 1
        else:
            questions_without_gold_path += 1

        if not oracle_pairs:
            questions_without_oracle_pair += 1

            print(
                f"\rProcessing: "
                f"{index}/{total_questions}",
                end="",
                flush=True,
            )

            continue

        questions_with_oracle_pair += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidates = set()
        path_count = 0

        for (
            head_type,
            tail_type,
        ) in oracle_pairs:

            result = retriever.retrieve(
                topic_entities=sample.topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            candidates.update(
                result.candidates
            )

            path_count += (
                result.num_paths
            )

        total_candidates += len(
            candidates
        )

        total_paths += path_count

        if (
            candidates
            & gold_answers
        ):
            oracle_recovered += 1

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
        f"Test questions                  : "
        f"{total_questions}"
    )

    print(
        f"KG contains >=1 Gold            : "
        f"{kg_has_gold}/{total_questions} "
        f"({kg_has_gold / total_questions * 100:.2f}%)"
    )

    print()

    print(
        f"Questions with Gold path        : "
        f"{questions_with_gold_path}/{total_questions} "
        f"({questions_with_gold_path / total_questions * 100:.2f}%)"
    )

    print(
        f"Questions without Gold path     : "
        f"{questions_without_gold_path}"
    )

    print()

    print(
        f"Questions with Oracle pair      : "
        f"{questions_with_oracle_pair}/{total_questions} "
        f"({questions_with_oracle_pair / total_questions * 100:.2f}%)"
    )

    print(
        f"Questions without Oracle pair   : "
        f"{questions_without_oracle_pair}"
    )

    print()

    print(
        f"Oracle Retriever Gold recovered : "
        f"{oracle_recovered}/{total_questions} "
        f"({oracle_recovered / total_questions * 100:.2f}%)"
    )

    if questions_with_oracle_pair > 0:
        conditional_recall = (
            oracle_recovered
            / questions_with_oracle_pair
            * 100
        )

        print(
            f"Recall given Oracle pair        : "
            f"{conditional_recall:.2f}%"
        )

        print(
            f"Avg Oracle candidates           : "
            f"{total_candidates / questions_with_oracle_pair:.2f}"
        )

        print(
            f"Avg Oracle evidence paths       : "
            f"{total_paths / questions_with_oracle_pair:.2f}"
        )

    print()

    print("-" * 100)

    print(
        "Current predicted-Planner Retriever recall: 75.68%"
    )

    print("-" * 100)

    print("=" * 100)


if __name__ == "__main__":
    main()