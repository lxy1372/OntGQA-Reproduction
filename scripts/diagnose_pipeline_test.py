#!/usr/bin/env python3

import json
import re
import string
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

PIPELINE_RESULT_PATH = (
    PROJECT_ROOT
    / "results"
    / "pipeline_test.jsonl"
)


def load_all_samples():
    """读取完整 WebQSP test split。"""

    samples = []

    for path in TEST_PATHS:
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


def load_pipeline_results():
    """读取已经完成的端到端 test 预测。"""

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

            record = json.loads(line)

            results[
                record["id"]
            ] = record

    return results


def normalize_answer(text):
    """使用当前 WebQSP evaluator 的规范化规则。"""

    if text is None:
        return ""

    text = str(text).lower()

    text = "".join(
        character
        for character in text
        if character not in string.punctuation
    )

    text = re.sub(
        r"\b(a|an|the)\b",
        " ",
        text,
    )

    text = " ".join(
        text.split()
    )

    return text


def normalize_set(answers):
    """将答案列表规范化为集合。"""

    return {
        normalize_answer(answer)
        for answer in answers
        if normalize_answer(answer)
    }


def normalize_list(answers):
    """规范化答案并保持排名顺序。"""

    normalized = []
    seen = set()

    for answer in answers:
        answer = normalize_answer(
            answer
        )

        if not answer:
            continue

        if answer in seen:
            continue

        seen.add(answer)
        normalized.append(answer)

    return normalized


def get_graph_entities(graph_triples):
    """提取当前问题子图中的全部实体。"""

    entities = set()

    for head, _, tail in graph_triples:
        entities.add(
            normalize_answer(head)
        )

        entities.add(
            normalize_answer(tail)
        )

    return entities


def merge_candidate_paths(
    candidate_to_paths,
    new_candidate_to_paths,
):
    """合并多个 Type Pair 的 Retriever 结果。"""

    for candidate, paths in new_candidate_to_paths.items():
        candidate_to_paths[candidate].extend(
            paths
        )


def main():
    print("=" * 100)
    print("OntGQA WebQSP Test Pipeline Diagnosis")
    print("=" * 100)

    samples = load_all_samples()

    pipeline_results = (
        load_pipeline_results()
    )

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    total = len(samples)

    if total != 1628:
        raise RuntimeError(
            f"Unexpected test size: {total}"
        )

    kg_has_gold = 0

    questions_with_valid_plan = 0
    questions_with_candidates = 0

    retriever_gold_questions = 0

    final_any_gold = 0
    final_top1_gold = 0

    ranking_loss_questions = 0

    judge_questions = 0
    judge_any_gold = 0
    judge_top1_gold = 0

    generator_questions = 0
    generator_any_gold = 0
    generator_top1_gold = 0

    retriever_gold_but_final_miss = 0
    retriever_gold_but_top1_miss = 0

    total_candidates = 0

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

        gold_set = normalize_set(
            sample.answer_entities
        )

        graph_entities = get_graph_entities(
            sample.graph_triples
        )

        if gold_set & graph_entities:
            kg_has_gold += 1

        valid_pairs = [
            tuple(pair)
            for pair in record.get(
                "valid_type_pairs",
                [],
            )
        ]

        if valid_pairs:
            questions_with_valid_plan += 1

        retriever = OntologyRetriever(
            ontology=ontology,
            graph_triples=sample.graph_triples,
        )

        candidate_to_paths = defaultdict(
            list
        )

        for head_type, tail_type in valid_pairs:
            retrieval_result = (
                retriever.retrieve(
                    topic_entities=(
                        sample.topic_entities
                    ),
                    head_type=head_type,
                    tail_type=tail_type,
                )
            )

            merge_candidate_paths(
                candidate_to_paths,
                retrieval_result.candidate_to_paths,
            )

        candidate_set = normalize_set(
            candidate_to_paths.keys()
        )

        if candidate_set:
            questions_with_candidates += 1

        total_candidates += len(
            candidate_set
        )

        retriever_hit = bool(
            candidate_set
            & gold_set
        )

        if retriever_hit:
            retriever_gold_questions += 1

        predictions = normalize_list(
            record[
                "predicted_answers"
            ]
        )

        prediction_set = set(
            predictions
        )

        any_gold = bool(
            prediction_set
            & gold_set
        )

        top1_gold = (
            bool(predictions)
            and predictions[0] in gold_set
        )

        if any_gold:
            final_any_gold += 1

        if top1_gold:
            final_top1_gold += 1

        if any_gold and not top1_gold:
            ranking_loss_questions += 1

        if retriever_hit and not any_gold:
            retriever_gold_but_final_miss += 1

        if retriever_hit and not top1_gold:
            retriever_gold_but_top1_miss += 1

        source = record[
            "answer_source"
        ]

        if source == "judge":
            judge_questions += 1

            if any_gold:
                judge_any_gold += 1

            if top1_gold:
                judge_top1_gold += 1

        elif source == "generator":
            generator_questions += 1

            if any_gold:
                generator_any_gold += 1

            if top1_gold:
                generator_top1_gold += 1

        print(
            f"\rProcessing: "
            f"{index}/{total}",
            end="",
            flush=True,
        )

    print()

    print("\n" + "=" * 100)
    print("Stage-wise Diagnosis")
    print("=" * 100)

    print(
        f"Test questions                     : "
        f"{total}"
    )

    print()

    print(
        f"KG contains >=1 Gold               : "
        f"{kg_has_gold}/{total} "
        f"({kg_has_gold / total * 100:.2f}%)"
    )

    print(
        f"Planner has valid Type Pair        : "
        f"{questions_with_valid_plan}/{total} "
        f"({questions_with_valid_plan / total * 100:.2f}%)"
    )

    print(
        f"Retriever has candidates           : "
        f"{questions_with_candidates}/{total} "
        f"({questions_with_candidates / total * 100:.2f}%)"
    )

    print(
        f"Retriever contains >=1 Gold        : "
        f"{retriever_gold_questions}/{total} "
        f"({retriever_gold_questions / total * 100:.2f}%)"
    )

    print()

    print(
        f"Final prediction contains Gold     : "
        f"{final_any_gold}/{total} "
        f"({final_any_gold / total * 100:.2f}%)"
    )

    print(
        f"Final Hit@1                        : "
        f"{final_top1_gold}/{total} "
        f"({final_top1_gold / total * 100:.2f}%)"
    )

    print(
        f"Gold present but not ranked first  : "
        f"{ranking_loss_questions}/{total} "
        f"({ranking_loss_questions / total * 100:.2f}%)"
    )

    print()

    print(
        f"Retriever Gold -> final no Gold    : "
        f"{retriever_gold_but_final_miss}"
    )

    print(
        f"Retriever Gold -> final top1 miss  : "
        f"{retriever_gold_but_top1_miss}"
    )

    print()

    print(
        f"Judge questions                    : "
        f"{judge_questions}"
    )

    if judge_questions:
        print(
            f"Judge contains Gold                : "
            f"{judge_any_gold}/{judge_questions} "
            f"({judge_any_gold / judge_questions * 100:.2f}%)"
        )

        print(
            f"Judge Hit@1                        : "
            f"{judge_top1_gold}/{judge_questions} "
            f"({judge_top1_gold / judge_questions * 100:.2f}%)"
        )

    print()

    print(
        f"Generator questions                : "
        f"{generator_questions}"
    )

    if generator_questions:
        print(
            f"Generator contains Gold            : "
            f"{generator_any_gold}/{generator_questions} "
            f"({generator_any_gold / generator_questions * 100:.2f}%)"
        )

        print(
            f"Generator Hit@1                    : "
            f"{generator_top1_gold}/{generator_questions} "
            f"({generator_top1_gold / generator_questions * 100:.2f}%)"
        )

    print()

    print(
        f"Avg Retriever candidates/question  : "
        f"{total_candidates / total:.2f}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()