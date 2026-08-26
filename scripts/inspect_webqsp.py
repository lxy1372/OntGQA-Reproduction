#!/usr/bin/env python3

import sys
from collections import defaultdict, deque
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph


WEBQSP_PATH = (
    PROJECT_ROOT
    / "data"
    / "webqsp"
    / "train-00000-of-00002.parquet"
)

ONTOLOGY_PATH = (
    PROJECT_ROOT
    / "data"
    / "ontology"
    / "ontology_graph_freebase.json"
)


def build_outgoing_index(graph_triples):
    """按照头实体建立知识图谱正向邻接索引。"""

    outgoing = defaultdict(list)

    for head, relation, tail in graph_triples:
        outgoing[head].append(
            (relation, tail)
        )

    return outgoing


def find_shortest_path(
    graph_triples,
    start_entities,
    target_entities,
    max_depth=2,
):
    """
    在知识图谱中搜索从起始实体到目标实体的最短有向路径。

    仅用于数据检查和调试，不参与正式推理。
    """

    outgoing = build_outgoing_index(graph_triples)

    targets = set(target_entities)

    queue = deque()

    for start in start_entities:
        queue.append(
            (
                start,
                [],
            )
        )

    visited = {
        (start, 0)
        for start in start_entities
    }

    while queue:
        current_entity, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for relation, tail in outgoing.get(current_entity, []):
            new_path = path + [
                (
                    current_entity,
                    relation,
                    tail,
                )
            ]

            if tail in targets:
                return new_path

            state = (
                tail,
                len(new_path),
            )

            if state not in visited:
                visited.add(state)

                queue.append(
                    (
                        tail,
                        new_path,
                    )
                )

    return None


def main():
    print("=" * 80)
    print("RoG-WebQSP Data Compatibility Test")
    print("=" * 80)

    loader = WebQSPDataLoader(WEBQSP_PATH)
    ontology = OntologyGraph(ONTOLOGY_PATH)

    sample = loader.get_sample(0)

    print("\n[1] Basic information")
    print(f"id             : {sample.sample_id}")
    print(f"question       : {sample.question}")
    print(f"answers        : {sample.answers}")
    print(f"topic_entities : {sample.topic_entities}")
    print(f"answer_entities: {sample.answer_entities}")
    print(f"graph triples  : {len(sample.graph_triples)}")

    # 统计知识图谱中的 relation
    graph_relations = {
        relation
        for _, relation, _ in sample.graph_triples
    }

    matched_relations = {
        relation
        for relation in graph_relations
        if ontology.has_relation(relation)
    }

    missing_relations = (
        graph_relations
        - matched_relations
    )

    print("\n[2] Relation alignment")

    print(
        f"Unique graph relations : "
        f"{len(graph_relations)}"
    )

    print(
        f"Matched in ontology    : "
        f"{len(matched_relations)}"
    )

    print(
        f"Missing from ontology  : "
        f"{len(missing_relations)}"
    )

    if graph_relations:
        coverage = (
            len(matched_relations)
            / len(graph_relations)
            * 100
        )
    else:
        coverage = 0.0

    print(
        f"Ontology coverage       : "
        f"{coverage:.2f}%"
    )

    if missing_relations:
        print("\nFirst 20 missing relations:")

        for relation in sorted(missing_relations)[:20]:
            print(relation)

    # 统计 Topic Entity 的直接邻居
    outgoing_index = build_outgoing_index(
        sample.graph_triples
    )

    print("\n[3] Topic entity outgoing edges")

    for topic_entity in sample.topic_entities:
        edges = outgoing_index.get(
            topic_entity,
            []
        )

        print(
            f"\nTopic entity: {topic_entity}"
        )

        print(
            f"Outgoing edges: {len(edges)}"
        )

        for relation, tail in edges[:20]:
            ontology_signature = (
                ontology.get_relation_signature(
                    relation
                )
            )

            print(
                f"{topic_entity}"
                f" --{relation}--> "
                f"{tail}"
            )

            print(
                f"    Ontology signature: "
                f"{ontology_signature}"
            )

    # 搜索 Topic Entity 到 Gold Answer 的最短路径
    print("\n[4] Shortest gold path within 2 hops")

    shortest_path = find_shortest_path(
        graph_triples=sample.graph_triples,
        start_entities=sample.topic_entities,
        target_entities=sample.answer_entities,
        max_depth=2,
    )

    if shortest_path is None:
        print("No gold path found within 2 hops.")

    else:
        print(
            f"Path length: "
            f"{len(shortest_path)}"
        )

        print("\nReasoning path:")

        for head, relation, tail in shortest_path:
            print(
                f"{head}"
                f" --{relation}--> "
                f"{tail}"
            )

        relation_path = [
            relation
            for _, relation, _ in shortest_path
        ]

        print("\nRelation path:")

        print(
            " -> ".join(relation_path)
        )

        # 检查 gold path 中每个 relation 的本体签名
        print("\nOntology signatures:")

        for relation in relation_path:
            signature = (
                ontology.get_relation_signature(
                    relation
                )
            )

            print(
                f"{relation}"
                f" -> "
                f"{signature}"
            )

        # 根据路径首尾 relation 获取 head/tail type
        first_relation = relation_path[0]
        last_relation = relation_path[-1]

        first_signature = (
            ontology.get_relation_signature(
                first_relation
            )
        )

        last_signature = (
            ontology.get_relation_signature(
                last_relation
            )
        )

        if (
            first_signature is not None
            and last_signature is not None
        ):
            head_type = first_signature[0]
            tail_type = last_signature[1]

            print("\nDerived endpoint type pair:")

            print(
                f"<{head_type}, {tail_type}>"
            )

    print("\n" + "=" * 80)
    print("Compatibility test finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()