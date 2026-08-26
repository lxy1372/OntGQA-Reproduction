#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ontology import OntologyGraph


ONTOLOGY_PATH = (
    PROJECT_ROOT
    / "data"
    / "ontology"
    / "ontology_graph_freebase.json"
)


def main():
    print("=" * 80)
    print("OntGQA Ontology Graph Test")
    print("=" * 80)

    ontology = OntologyGraph(ONTOLOGY_PATH)

    # 查看本体图整体统计信息
    print("\n[1] Ontology statistics")

    stats = ontology.stats()

    for key, value in stats.items():
        print(f"{key}: {value}")

    # 测试指定 Head Type 的出边查询
    test_head_type = "american_football.football_coach"

    print("\n[2] Test head type")
    print(f"Head type: {test_head_type}")

    outgoing_edges = ontology.get_outgoing_edges(test_head_type)

    print(f"Number of outgoing edges: {len(outgoing_edges)}")

    for relation, tail_type in outgoing_edges[:10]:
        print(
            f"{test_head_type}"
            f" --{relation}--> "
            f"{tail_type}"
        )

    # 测试 R+(Head Type)
    print("\n[3] R+(Head Type)")

    outgoing_relations = ontology.get_outgoing_relations(
        test_head_type
    )

    print(
        f"Number of outgoing relations: "
        f"{len(outgoing_relations)}"
    )

    for relation in sorted(outgoing_relations)[:10]:
        print(relation)

    # 测试指定 Tail Type 的入边查询
    test_tail_type = "american_football.football_team"

    print("\n[4] Test tail type")
    print(f"Tail type: {test_tail_type}")

    incoming_edges = ontology.get_incoming_edges(test_tail_type)

    print(f"Number of incoming edges: {len(incoming_edges)}")

    for relation, head_type in incoming_edges[:10]:
        print(
            f"{head_type}"
            f" --{relation}--> "
            f"{test_tail_type}"
        )

    # 测试 R-(Tail Type)
    print("\n[5] R-(Tail Type)")

    incoming_relations = ontology.get_incoming_relations(
        test_tail_type
    )

    print(
        f"Number of incoming relations: "
        f"{len(incoming_relations)}"
    )

    for relation in sorted(incoming_relations)[:10]:
        print(relation)

    # 测试 Relation -> (Head Type, Tail Type) 查询
    test_relation = (
        "american_football.football_coach."
        "current_team_head_coached"
    )

    print("\n[6] Relation signature")
    print(f"Relation: {test_relation}")

    signature = ontology.get_relation_signature(
        test_relation
    )

    print(f"Signature: {signature}")

    # 测试类型和关系是否存在
    print("\n[7] Existence checks")

    print(
        f"Has head type: "
        f"{ontology.has_type(test_head_type)}"
    )

    print(
        f"Has tail type: "
        f"{ontology.has_type(test_tail_type)}"
    )

    print(
        f"Has relation: "
        f"{ontology.has_relation(test_relation)}"
    )

    print("\n" + "=" * 80)
    print("Ontology module test finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()