#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph
from src.retriever import OntologyRetriever


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


def print_path(path):
    """打印 reasoning path。"""

    parts = []

    for head, relation, tail in path:
        parts.append(
            f"{head} --{relation}--> {tail}"
        )

    print("\n".join(parts))


def main():
    print("=" * 80)
    print("OntGQA Retriever Test")
    print("=" * 80)

    # 加载本体图和第一条 WebQSP 样本
    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    loader = WebQSPDataLoader(
        WEBQSP_PATH
    )

    sample = loader.get_sample(0)

    print("\n[1] Question")
    print(sample.question)

    print("\nTopic entities:")
    print(sample.topic_entities)

    print("\nGold answers:")
    print(sample.answer_entities)

    # 使用当前样本已验证的类型规划测试 Retriever
    head_type = "people.person"
    tail_type = "people.person"

    print("\n[2] Type plan")
    print(
        f"<{head_type}, {tail_type}>"
    )

    retriever = OntologyRetriever(
        ontology=ontology,
        graph_triples=sample.graph_triples,
    )

    result = retriever.retrieve(
        topic_entities=sample.topic_entities,
        head_type=head_type,
        tail_type=tail_type,
    )

    print("\n[3] Ontology constraints")

    outgoing_relations = (
        ontology.get_outgoing_relations(
            head_type
        )
    )

    incoming_relations = (
        ontology.get_incoming_relations(
            tail_type
        )
    )

    print(
        f"R+({head_type}) size: "
        f"{len(outgoing_relations)}"
    )

    print(
        f"R-({tail_type}) size: "
        f"{len(incoming_relations)}"
    )

    print("\n[4] Retrieval statistics")

    print(
        f"1-hop paths       : "
        f"{len(result.one_hop_paths)}"
    )

    print(
        f"2-hop paths       : "
        f"{len(result.two_hop_paths)}"
    )

    print(
        f"Total paths       : "
        f"{result.num_paths}"
    )

    print(
        f"Unique candidates : "
        f"{len(result.candidates)}"
    )

    # 检查 Gold Answer 是否进入候选集合
    gold_answers = set(
        sample.answer_entities
    )

    retrieved_candidates = set(
        result.candidates
    )

    recovered_gold = (
        gold_answers
        & retrieved_candidates
    )

    print("\n[5] Gold answer coverage")

    print(
        f"Gold answers     : "
        f"{gold_answers}"
    )

    print(
        f"Recovered gold   : "
        f"{recovered_gold}"
    )

    print(
        f"Gold recovered   : "
        f"{bool(recovered_gold)}"
    )

    # 查看 Gold Answer 对应的 evidence paths
    print("\n[6] Gold evidence paths")

    for gold_answer in sample.answer_entities:
        paths = result.candidate_to_paths.get(
            gold_answer,
            [],
        )

        print(
            f"\nCandidate: {gold_answer}"
        )

        print(
            f"Evidence paths: {len(paths)}"
        )

        for index, path in enumerate(
            paths[:10],
            start=1,
        ):
            print(
                f"\nPath {index}:"
            )

            print_path(path)

    print("\n" + "=" * 80)
    print("Retriever test finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()