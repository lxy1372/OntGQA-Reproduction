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

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "judge"
    / "judge_train.jsonl"
)

MAX_EVIDENCE_PATHS = 3


def load_all_samples():
    """读取完整 WebQSP 训练集。"""

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
    """去除完全重复的 reasoning paths。"""

    unique_paths = []
    seen = set()

    for path in paths:
        if path in seen:
            continue

        seen.add(path)
        unique_paths.append(path)

    return unique_paths


def merge_candidate_paths(
    candidate_to_paths,
    new_candidate_to_paths,
):
    """合并不同 Type Pair 检索得到的候选及证据路径。"""

    for candidate, paths in new_candidate_to_paths.items():
        candidate_to_paths[candidate].extend(
            paths
        )


def clean_candidate_paths(
    candidate_to_paths,
):
    """对每个候选答案的 reasoning paths 去重。"""

    cleaned = {}

    for candidate, paths in candidate_to_paths.items():
        cleaned[candidate] = deduplicate_paths(
            paths
        )

    return cleaned


def select_shortest_paths(
    paths,
    max_paths=3,
):
    """保留少量最短 evidence paths。"""

    if not paths:
        return []

    minimum_hops = min(
        len(path)
        for path in paths
    )

    shortest_paths = [
        path
        for path in paths
        if len(path) == minimum_hops
    ]

    return shortest_paths[:max_paths]


def textualize_path(path):
    """将 reasoning path 转换为 Judge 使用的文本形式。"""

    parts = []

    for head, relation, tail in path:
        parts.append(
            f"{head} --{relation}--> {tail}"
        )

    return " | ".join(parts)


def build_evidence_text(paths):
    """构造候选答案对应的 evidence paths 文本。"""

    selected_paths = select_shortest_paths(
        paths,
        max_paths=MAX_EVIDENCE_PATHS,
    )

    textualized_paths = [
        textualize_path(path)
        for path in selected_paths
    ]

    if not textualized_paths:
        return (
            "None",
            [],
        )

    evidence_text = "\n".join(
        f"{index}. {path}"
        for index, path in enumerate(
            textualized_paths,
            start=1,
        )
    )

    return (
        evidence_text,
        textualized_paths,
    )


def build_judge_prompt(
    question,
    candidate,
    evidence_text,
):
    """构造 Judge 的 YES/NO 判断指令。"""

    return (
        "You are a strict judge for knowledge-graph QA. "
        "Given a QUESTION, a CANDIDATE answer, and its "
        "EVIDENCE PATHS, decide whether the candidate is "
        "a correct final answer to the question. "
        'Return strictly one token: "YES" or "NO". '
        "No explanations.\n"
        f"Question: {question}\n"
        f"Candidate: {candidate}\n"
        f"Evidence paths:\n{evidence_text}"
    )


def build_record(
    sample,
    candidate,
    paths,
    target,
):
    """构造单条 Judge 监督数据。"""

    (
        evidence_text,
        evidence_paths,
    ) = build_evidence_text(
        paths
    )

    return {
        "id": sample.sample_id,
        "question": sample.question,
        "candidate": candidate,
        "evidence_paths": evidence_paths,
        "prompt": build_judge_prompt(
            question=sample.question,
            candidate=candidate,
            evidence_text=evidence_text,
        ),
        "target": target,
    }


def negative_sort_key(
    candidate,
    candidate_to_paths,
):
    """
    对负候选排序。

    优先选择路径更短、证据路径更多的候选，
    使负样本具有更强的迷惑性。
    """

    paths = candidate_to_paths[candidate]

    minimum_hops = min(
        len(path)
        for path in paths
    )

    return (
        minimum_hops,
        -len(paths),
        candidate,
    )


def main():
    print("=" * 100)
    print("OntGQA Judge Supervision Builder")
    print("=" * 100)

    samples = load_all_samples()

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=ADAPTER_PATH,
    )

    print(
        f"\nTraining questions: "
        f"{len(samples)}"
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    questions_with_prediction = 0
    questions_without_prediction = 0

    questions_with_valid_plan = 0
    questions_without_valid_plan = 0

    questions_with_positive = 0
    questions_without_positive = 0
    questions_without_negative = 0

    total_predicted_pairs = 0
    total_valid_pairs = 0

    positive_count = 0
    negative_count = 0

    output_records = []

    for index, sample in enumerate(
        samples,
        start=1,
    ):
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

        if predicted_pairs:
            questions_with_prediction += 1
        else:
            questions_without_prediction += 1

        # 过滤 Planner 生成的非法 ontology type
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

        if not valid_pairs:
            questions_without_valid_plan += 1

            print(
                f"\rProcessing: {index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        questions_with_valid_plan += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(list)

        # 按 Planner Top-3 中合法的 Type Pair 进行真实检索
        for head_type, tail_type in valid_pairs:
            retrieval_result = retriever.retrieve(
                topic_entities=sample.topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            merge_candidate_paths(
                candidate_to_paths,
                retrieval_result.candidate_to_paths,
            )

        candidate_to_paths = clean_candidate_paths(
            candidate_to_paths
        )

        gold_answers = set(
            sample.answer_entities
        )

        # Retriever 召回且属于 Gold Answer 的候选为正样本
        positive_candidates = sorted(
            candidate
            for candidate in candidate_to_paths
            if candidate in gold_answers
        )

        if not positive_candidates:
            questions_without_positive += 1

            print(
                f"\rProcessing: {index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        # 其余 Retriever 候选作为负样本池
        negative_candidates = [
            candidate
            for candidate in candidate_to_paths
            if candidate not in gold_answers
        ]

        if not negative_candidates:
            questions_without_negative += 1

            print(
                f"\rProcessing: {index}/{len(samples)}",
                end="",
                flush=True,
            )

            continue

        questions_with_positive += 1

        # 优先选取更具有迷惑性的负样本
        negative_candidates.sort(
            key=lambda candidate: negative_sort_key(
                candidate,
                candidate_to_paths,
            )
        )

        # 每个 positive 配一个 negative，保持 1:1 平衡
        for positive_index, positive_candidate in enumerate(
            positive_candidates
        ):
            negative_candidate = negative_candidates[
                positive_index
                % len(negative_candidates)
            ]

            positive_record = build_record(
                sample=sample,
                candidate=positive_candidate,
                paths=candidate_to_paths[
                    positive_candidate
                ],
                target="YES",
            )

            negative_record = build_record(
                sample=sample,
                candidate=negative_candidate,
                paths=candidate_to_paths[
                    negative_candidate
                ],
                target="NO",
            )

            output_records.append(
                positive_record
            )

            output_records.append(
                negative_record
            )

            positive_count += 1
            negative_count += 1

        print(
            f"\rProcessing: {index}/{len(samples)}",
            end="",
            flush=True,
        )

    print()

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
        f"Input questions                 : "
        f"{len(samples)}"
    )

    print(
        f"Questions with prediction       : "
        f"{questions_with_prediction}"
    )

    print(
        f"Questions without prediction    : "
        f"{questions_without_prediction}"
    )

    print(
        f"Questions with valid plan       : "
        f"{questions_with_valid_plan}"
    )

    print(
        f"Questions without valid plan    : "
        f"{questions_without_valid_plan}"
    )

    print(
        f"Questions with positive         : "
        f"{questions_with_positive}"
    )

    print(
        f"Questions without positive      : "
        f"{questions_without_positive}"
    )

    print(
        f"Questions without negative      : "
        f"{questions_without_negative}"
    )

    print(
        f"Predicted type pairs            : "
        f"{total_predicted_pairs}"
    )

    print(
        f"Valid ontology type pairs       : "
        f"{total_valid_pairs}"
    )

    print(
        f"Positive samples                : "
        f"{positive_count}"
    )

    print(
        f"Negative samples                : "
        f"{negative_count}"
    )

    print(
        f"Total Judge samples             : "
        f"{len(output_records)}"
    )

    print(
        f"Output file                     : "
        f"{OUTPUT_PATH}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()