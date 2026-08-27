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
    / "planner_validation.jsonl"
)


def build_outgoing_index(graph_triples):
    """按照头实体建立知识图谱正向邻接索引。"""

    outgoing = defaultdict(list)

    for head, relation, tail in graph_triples:
        outgoing[head].append(
            (relation, tail)
        )

    return outgoing


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


def find_shortest_paths(
    outgoing,
    topic_entities,
    target_entity,
):
    """搜索 Topic Entity 到指定答案的全部 1-hop 或 2-hop 最短路径。"""

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


def path_to_type_pair(
    ontology,
    path,
):
    """将完整 Relation Path 对齐到本体 Type Pair。"""

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

    # 检查相邻关系对应的实体类型是否连续
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
            return None

    return (
        signatures[0][0],
        signatures[-1][1],
    )


def get_gold_type_pairs(
    sample,
    ontology,
):
    """根据 Gold Answer 和本体图生成 Oracle Type Pair 集合。"""

    outgoing = build_outgoing_index(
        sample.graph_triples
    )

    gold_pairs = set()

    for answer_entity in sample.answer_entities:
        paths = find_shortest_paths(
            outgoing=outgoing,
            topic_entities=sample.topic_entities,
            target_entity=answer_entity,
        )

        for path in paths:
            type_pair = path_to_type_pair(
                ontology=ontology,
                path=path,
            )

            if type_pair is not None:
                gold_pairs.add(
                    type_pair
                )

    return gold_pairs


def type_pair_to_list(type_pair):
    """将 Type Pair 转换为可序列化列表。"""

    return [
        type_pair[0],
        type_pair[1],
    ]


def main():
    print("=" * 80)
    print("OntGQA Planner Validation")
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

    top1_hits = 0
    top3_hits = 0

    no_prediction_count = 0

    invalid_prediction_count = 0
    total_prediction_count = 0

    results = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        gold_pairs = get_gold_type_pairs(
            sample,
            ontology,
        )

        # 当前问题子图无法构造 Gold Type Pair 时跳过评价
        if not gold_pairs:
            skipped_count += 1

            results.append(
                {
                    "id": sample.sample_id,
                    "question": sample.question,
                    "gold_answers": sample.answer_entities,
                    "gold_type_pairs": [],
                    "predicted_type_pairs": [],
                    "top1_hit": None,
                    "top3_hit": None,
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

        result = planner.generate(
            question=sample.question,
            top_k=3,
            num_beams=3,
        )

        predicted_pairs = (
            result["predicted_type_pairs"]
        )

        if not predicted_pairs:
            no_prediction_count += 1

        # 统计模型生成的实体类型是否存在于官方本体中
        for head_type, tail_type in predicted_pairs[:3]:
            total_prediction_count += 1

            if (
                not ontology.has_type(head_type)
                or not ontology.has_type(tail_type)
            ):
                invalid_prediction_count += 1

        top1_hit = (
            len(predicted_pairs) > 0
            and predicted_pairs[0] in gold_pairs
        )

        top3_hit = any(
            pair in gold_pairs
            for pair in predicted_pairs[:3]
        )

        if top1_hit:
            top1_hits += 1

        if top3_hit:
            top3_hits += 1

        results.append(
            {
                "id": sample.sample_id,
                "question": sample.question,
                "gold_answers": sample.answer_entities,
                "gold_type_pairs": [
                    type_pair_to_list(pair)
                    for pair in sorted(gold_pairs)
                ],
                "predicted_type_pairs": [
                    type_pair_to_list(pair)
                    for pair in predicted_pairs[:3]
                ],
                "raw_outputs": result["raw_outputs"],
                "top1_hit": top1_hit,
                "top3_hit": top3_hit,
                "status": "evaluated",
            }
        )

        print(
            f"\rProcessing: {index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

    # 保存逐题评价结果
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
        f"Input samples              : "
        f"{len(samples)}"
    )

    print(
        f"Evaluated                  : "
        f"{evaluated_count}"
    )

    print(
        f"Skipped                    : "
        f"{skipped_count}"
    )

    print(
        f"No parsed prediction       : "
        f"{no_prediction_count}"
    )

    if evaluated_count > 0:
        top1_accuracy = (
            top1_hits
            / evaluated_count
            * 100
        )

        top3_accuracy = (
            top3_hits
            / evaluated_count
            * 100
        )

        print(
            f"Top-1                      : "
            f"{top1_hits}/{evaluated_count} "
            f"({top1_accuracy:.2f}%)"
        )

        print(
            f"Top-3                      : "
            f"{top3_hits}/{evaluated_count} "
            f"({top3_accuracy:.2f}%)"
        )

    if total_prediction_count > 0:
        valid_prediction_count = (
            total_prediction_count
            - invalid_prediction_count
        )

        valid_prediction_rate = (
            valid_prediction_count
            / total_prediction_count
            * 100
        )

        print(
            f"Valid ontology predictions : "
            f"{valid_prediction_count}/"
            f"{total_prediction_count} "
            f"({valid_prediction_rate:.2f}%)"
        )

    print(
        f"Result file                : "
        f"{RESULT_PATH}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()