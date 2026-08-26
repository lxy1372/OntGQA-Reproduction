#!/usr/bin/env python3

import sys
from collections import defaultdict, deque
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

NUM_SAMPLES = 20
MAX_DEPTH = 2


def build_outgoing_index(graph_triples):
    """按照头实体建立知识图谱正向邻接索引。"""

    outgoing = defaultdict(list)

    for head, relation, tail in graph_triples:
        outgoing[head].append(
            (relation, tail)
        )

    return outgoing


def find_shortest_gold_path(
    graph_triples,
    topic_entities,
    answer_entities,
    max_depth=2,
):
    """搜索 Topic Entity 到任一 Gold Answer 的最短有向路径。"""

    outgoing = build_outgoing_index(
        graph_triples
    )

    targets = set(answer_entities)

    queue = deque()

    for topic_entity in topic_entities:
        queue.append(
            (
                topic_entity,
                [],
            )
        )

    visited = set(topic_entities)

    while queue:
        current_entity, path = queue.popleft()

        if len(path) >= max_depth:
            continue

        for relation, tail in outgoing.get(
            current_entity,
            [],
        ):
            new_path = path + [
                (
                    current_entity,
                    relation,
                    tail,
                )
            ]

            if tail in targets:
                return new_path

            if tail in visited:
                continue

            visited.add(tail)

            queue.append(
                (
                    tail,
                    new_path,
                )
            )

    return None


def derive_type_pair(
    ontology,
    reasoning_path,
):
    """根据 reasoning path 首尾关系推导端点类型。"""

    if not reasoning_path:
        return None

    first_relation = reasoning_path[0][1]
    last_relation = reasoning_path[-1][1]

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
        first_signature is None
        or last_signature is None
    ):
        return None

    head_type = first_signature[0]
    tail_type = last_signature[1]

    return (
        head_type,
        tail_type,
    )


def count_unconstrained_paths(
    graph_triples,
    topic_entities,
):
    """统计不使用本体边界约束时的 1-hop 和 2-hop 路径数量。"""

    outgoing = build_outgoing_index(
        graph_triples
    )

    one_hop_count = 0
    two_hop_count = 0

    candidates = set()

    for topic_entity in topic_entities:
        first_edges = outgoing.get(
            topic_entity,
            [],
        )

        for _, middle_entity in first_edges:
            one_hop_count += 1
            candidates.add(middle_entity)

            second_edges = outgoing.get(
                middle_entity,
                [],
            )

            for _, tail_entity in second_edges:
                two_hop_count += 1
                candidates.add(tail_entity)

    return (
        one_hop_count,
        two_hop_count,
        len(candidates),
    )


def format_path(path):
    """将 reasoning path 转换为简短字符串。"""

    if not path:
        return ""

    result = []

    for head, relation, tail in path:
        result.append(
            f"{head} --{relation}--> {tail}"
        )

    return " | ".join(result)


def main():
    print("=" * 100)
    print("OntGQA Oracle Retriever Evaluation")
    print("=" * 100)

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    loader = WebQSPDataLoader(
        WEBQSP_PATH
    )

    samples = loader.load_first_n(
        NUM_SAMPLES
    )

    evaluated_count = 0
    recovered_count = 0
    no_gold_path_count = 0
    no_type_pair_count = 0

    total_unconstrained_paths = 0
    total_constrained_paths = 0

    total_unconstrained_candidates = 0
    total_constrained_candidates = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        print("\n" + "-" * 100)

        print(
            f"[{index}] "
            f"{sample.sample_id}"
        )

        print(
            f"Question: "
            f"{sample.question}"
        )

        print(
            f"Gold: "
            f"{sample.answer_entities}"
        )

        # 从真实图中寻找最短 Gold Path，仅用于构造 Oracle Type Pair
        gold_path = find_shortest_gold_path(
            graph_triples=sample.graph_triples,
            topic_entities=sample.topic_entities,
            answer_entities=sample.answer_entities,
            max_depth=MAX_DEPTH,
        )

        if gold_path is None:
            no_gold_path_count += 1

            print(
                "Gold path: "
                "not found within 2 hops"
            )

            continue

        print(
            f"Gold path: "
            f"{format_path(gold_path)}"
        )

        type_pair = derive_type_pair(
            ontology=ontology,
            reasoning_path=gold_path,
        )

        if type_pair is None:
            no_type_pair_count += 1

            print(
                "Type pair: "
                "cannot be aligned to ontology"
            )

            continue

        head_type, tail_type = type_pair

        print(
            f"Oracle type pair: "
            f"<{head_type}, {tail_type}>"
        )

        # 统计不使用本体约束时的原始路径规模
        (
            unconstrained_one_hop,
            unconstrained_two_hop,
            unconstrained_candidates,
        ) = count_unconstrained_paths(
            graph_triples=sample.graph_triples,
            topic_entities=sample.topic_entities,
        )

        unconstrained_paths = (
            unconstrained_one_hop
            + unconstrained_two_hop
        )

        # 使用 Oracle Type Pair 执行 OntGQA Retriever
        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        result = retriever.retrieve(
            topic_entities=sample.topic_entities,
            head_type=head_type,
            tail_type=tail_type,
        )

        constrained_paths = (
            result.num_paths
        )

        constrained_candidates = (
            len(result.candidates)
        )

        gold_set = set(
            sample.answer_entities
        )

        candidate_set = set(
            result.candidates
        )

        recovered_gold = (
            gold_set
            & candidate_set
        )

        recovered = bool(
            recovered_gold
        )

        evaluated_count += 1

        if recovered:
            recovered_count += 1

        total_unconstrained_paths += (
            unconstrained_paths
        )

        total_constrained_paths += (
            constrained_paths
        )

        total_unconstrained_candidates += (
            unconstrained_candidates
        )

        total_constrained_candidates += (
            constrained_candidates
        )

        print(
            f"Paths: "
            f"{unconstrained_paths} "
            f"-> "
            f"{constrained_paths}"
        )

        print(
            f"Candidates: "
            f"{unconstrained_candidates} "
            f"-> "
            f"{constrained_candidates}"
        )

        print(
            f"Recovered: "
            f"{recovered}"
        )

        if recovered_gold:
            print(
                f"Recovered gold: "
                f"{sorted(recovered_gold)}"
            )

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        f"Input samples             : "
        f"{len(samples)}"
    )

    print(
        f"Evaluated samples         : "
        f"{evaluated_count}"
    )

    print(
        f"No gold path              : "
        f"{no_gold_path_count}"
    )

    print(
        f"No ontology type pair     : "
        f"{no_type_pair_count}"
    )

    print(
        f"Gold recovered            : "
        f"{recovered_count}"
    )

    if evaluated_count > 0:
        recall = (
            recovered_count
            / evaluated_count
            * 100
        )

        avg_unconstrained_paths = (
            total_unconstrained_paths
            / evaluated_count
        )

        avg_constrained_paths = (
            total_constrained_paths
            / evaluated_count
        )

        avg_unconstrained_candidates = (
            total_unconstrained_candidates
            / evaluated_count
        )

        avg_constrained_candidates = (
            total_constrained_candidates
            / evaluated_count
        )

        print(
            f"Oracle retrieval recall   : "
            f"{recall:.2f}%"
        )

        print(
            f"Avg paths before gating   : "
            f"{avg_unconstrained_paths:.2f}"
        )

        print(
            f"Avg paths after gating    : "
            f"{avg_constrained_paths:.2f}"
        )

        print(
            f"Avg candidates before     : "
            f"{avg_unconstrained_candidates:.2f}"
        )

        print(
            f"Avg candidates after      : "
            f"{avg_constrained_candidates:.2f}"
        )

        if total_unconstrained_paths > 0:
            path_reduction = (
                1
                - total_constrained_paths
                / total_unconstrained_paths
            ) * 100

            print(
                f"Path reduction            : "
                f"{path_reduction:.2f}%"
            )

    print("=" * 100)


if __name__ == "__main__":
    main()