#!/usr/bin/env python3

import json
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.ontology import OntologyGraph


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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "planner"
    / "planner_train.jsonl"
)

MAX_DEPTH = 2


def build_outgoing_index(graph_triples):
    """按照头实体建立知识图谱正向邻接索引。"""

    outgoing = defaultdict(list)

    for head, relation, tail in graph_triples:
        outgoing[head].append(
            (relation, tail)
        )

    return outgoing


def load_all_samples():
    """读取全部 WebQSP 训练样本。"""

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


def deduplicate_paths(paths):
    """按照完整三元组序列去除重复路径。"""

    unique_paths = []
    seen = set()

    for path in paths:
        if path in seen:
            continue

        seen.add(path)
        unique_paths.append(path)

    return unique_paths


def find_shortest_paths_to_target(
    outgoing,
    topic_entities,
    target_entity,
    max_depth=2,
):
    """搜索 Topic Entity 到指定 Gold Answer 的全部最短有向路径。"""

    one_hop_paths = []

    # 搜索全部 1-hop Gold Path
    for topic_entity in topic_entities:
        for relation, tail in outgoing.get(
            topic_entity,
            [],
        ):
            if tail != target_entity:
                continue

            one_hop_paths.append(
                (
                    (
                        topic_entity,
                        relation,
                        tail,
                    ),
                )
            )

    # 存在 1-hop 路径时不再使用更长路径
    if one_hop_paths:
        return deduplicate_paths(
            one_hop_paths
        )

    if max_depth < 2:
        return []

    two_hop_paths = []

    # 不存在 1-hop Gold Path 时搜索全部 2-hop Gold Path
    for topic_entity in topic_entities:
        for first_relation, middle_entity in outgoing.get(
            topic_entity,
            [],
        ):
            for second_relation, tail_entity in outgoing.get(
                middle_entity,
                [],
            ):
                if tail_entity != target_entity:
                    continue

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

    return deduplicate_paths(
        two_hop_paths
    )


def align_path_to_ontology(
    ontology,
    path,
):
    """
    将完整 Relation Path 对齐到本体图。

    相邻关系的类型签名必须连续：
        r1: A -> B
        r2: B -> C
        ...

    对齐成功后返回：
        (head_type, tail_type), None

    对齐失败时返回：
        None, failure_reason
    """

    if not path:
        return None, "empty_path"

    signatures = []

    for _, relation, _ in path:
        signature = ontology.get_relation_signature(
            relation
        )

        if signature is None:
            return None, "missing_relation"

        signatures.append(
            signature
        )

    # 检查完整 Relation Path 在本体图中的类型连续性
    for index in range(
        len(signatures) - 1
    ):
        current_tail_type = (
            signatures[index][1]
        )

        next_head_type = (
            signatures[index + 1][0]
        )

        if current_tail_type != next_head_type:
            return None, "type_discontinuity"

    head_type = signatures[0][0]
    tail_type = signatures[-1][1]

    return (
        head_type,
        tail_type,
    ), None


def build_planner_prompt(question):
    """构造 Planner 指令输入。"""

    return (
        "<TASK: TYPE_PAIRS>\n"
        "Please generate a valid type pair that can be helpful "
        "for answering the following question:\n"
        f"Question: {question}"
    )


def build_planner_target(
    head_type,
    tail_type,
):
    """构造 Planner 目标输出。"""

    return (
        f"<PAIR> {head_type} "
        f"<SEP> {tail_type} </PAIR>"
    )


def main():
    print("=" * 100)
    print("OntGQA Planner Supervision Builder")
    print("=" * 100)

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    samples = load_all_samples()

    print(
        f"\nTraining questions: "
        f"{len(samples)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions_with_supervision = 0
    questions_without_gold_path = 0
    questions_without_type_pair = 0
    questions_with_multiple_pairs = 0

    one_hop_questions = 0
    two_hop_questions = 0

    total_gold_paths = 0
    ontology_consistent_paths = 0

    paths_with_missing_relation = 0
    paths_with_type_discontinuity = 0

    type_pair_frequency = defaultdict(int)

    output_records = []

    for sample in samples:
        outgoing = build_outgoing_index(
            sample.graph_triples
        )

        all_gold_paths = []

        # 分别搜索每个 Gold Answer 的最短路径
        for answer_entity in sample.answer_entities:
            paths = find_shortest_paths_to_target(
                outgoing=outgoing,
                topic_entities=sample.topic_entities,
                target_entity=answer_entity,
                max_depth=MAX_DEPTH,
            )

            all_gold_paths.extend(
                paths
            )

        all_gold_paths = deduplicate_paths(
            all_gold_paths
        )

        if not all_gold_paths:
            questions_without_gold_path += 1
            continue

        total_gold_paths += len(
            all_gold_paths
        )

        path_lengths = {
            len(path)
            for path in all_gold_paths
        }

        if 1 in path_lengths:
            one_hop_questions += 1

        if 2 in path_lengths:
            two_hop_questions += 1

        type_pairs = set()

        # 仅保留能够完整对齐到本体图的 Gold Path
        for path in all_gold_paths:
            (
                type_pair,
                failure_reason,
            ) = align_path_to_ontology(
                ontology=ontology,
                path=path,
            )

            if type_pair is not None:
                ontology_consistent_paths += 1

                type_pairs.add(
                    type_pair
                )

                continue

            if failure_reason == "missing_relation":
                paths_with_missing_relation += 1

            elif failure_reason == "type_discontinuity":
                paths_with_type_discontinuity += 1

        if not type_pairs:
            questions_without_type_pair += 1
            continue

        questions_with_supervision += 1

        if len(type_pairs) > 1:
            questions_with_multiple_pairs += 1

        # 每个问题内部按照 Type Pair 去重
        for head_type, tail_type in sorted(
            type_pairs
        ):
            record = {
                "id": sample.sample_id,
                "question": sample.question,
                "head_type": head_type,
                "tail_type": tail_type,
                "prompt": build_planner_prompt(
                    sample.question
                ),
                "target": build_planner_target(
                    head_type,
                    tail_type,
                ),
            }

            output_records.append(
                record
            )

            type_pair_frequency[
                (
                    head_type,
                    tail_type,
                )
            ] += 1

    # 重新生成 Planner supervision 文件
    with open(
        OUTPUT_PATH,
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

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)

    print(
        f"Input questions                  : "
        f"{len(samples)}"
    )

    print(
        f"Questions with supervision       : "
        f"{questions_with_supervision}"
    )

    print(
        f"Questions without gold path      : "
        f"{questions_without_gold_path}"
    )

    print(
        f"Questions without type pair      : "
        f"{questions_without_type_pair}"
    )

    print(
        f"1-hop questions                  : "
        f"{one_hop_questions}"
    )

    print(
        f"2-hop questions                  : "
        f"{two_hop_questions}"
    )

    print(
        f"Total shortest gold paths        : "
        f"{total_gold_paths}"
    )

    print(
        f"Ontology-consistent gold paths   : "
        f"{ontology_consistent_paths}"
    )

    print(
        f"Paths with missing relation      : "
        f"{paths_with_missing_relation}"
    )

    print(
        f"Paths with type discontinuity    : "
        f"{paths_with_type_discontinuity}"
    )

    print(
        f"Questions with >1 type pair      : "
        f"{questions_with_multiple_pairs}"
    )

    print(
        f"Planner supervisions             : "
        f"{len(output_records)}"
    )

    print(
        f"Unique type pairs globally       : "
        f"{len(type_pair_frequency)}"
    )

    print(
        f"Output file                      : "
        f"{OUTPUT_PATH}"
    )

    print("\nTop 20 type pairs:")

    sorted_pairs = sorted(
        type_pair_frequency.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    for (
        (head_type, tail_type),
        count,
    ) in sorted_pairs[:20]:
        print(
            f"{count:4d}  "
            f"<{head_type}, {tail_type}>"
        )

    print("=" * 100)


if __name__ == "__main__":
    main()