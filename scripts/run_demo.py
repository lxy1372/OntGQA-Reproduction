#!/usr/bin/env python3

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import WebQSPDataLoader
from src.generator import QwenGenerator
from src.judge import QwenJudge
from src.ontology import OntologyGraph
from src.pipeline import OntGQAPipeline
from src.planner import QwenPlanner
from src.retriever import OntologyRetriever


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

PLANNER_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "planner_lora"
    / "final"
)

JUDGE_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "judge_lora"
    / "final"
)

GENERATOR_ADAPTER_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "generator_lora"
    / "final"
)

NUM_DEMO_SAMPLES = 10


def main():
    print("=" * 100)
    print("OntGQA Reproduction Demo")
    print("=" * 100)

    print("\nLoading ontology...")

    ontology = OntologyGraph(
        ONTOLOGY_PATH
    )

    print("Loading Planner...")

    planner = QwenPlanner(
        model_path=MODEL_PATH,
        adapter_path=PLANNER_ADAPTER_PATH,
    )

    print("Loading Judge...")

    judge = QwenJudge(
        model_path=MODEL_PATH,
        adapter_path=JUDGE_ADAPTER_PATH,
        margin_threshold=1.0,
    )

    print("Loading Generator...")

    generator = QwenGenerator(
        model_path=MODEL_PATH,
        adapter_path=GENERATOR_ADAPTER_PATH,
    )

    pipeline = OntGQAPipeline(
        ontology=ontology,
        planner=planner,
        judge=judge,
        generator=generator,
        retriever_class=OntologyRetriever,
    )

    loader = WebQSPDataLoader(
        VALID_PATH
    )

    samples = loader.load_first_n(
        NUM_DEMO_SAMPLES
    )

    hit_count = 0

    for index, sample in enumerate(
        samples,
        start=1,
    ):
        result = pipeline.answer(
            question=sample.question,
            topic_entities=sample.topic_entities,
            graph_triples=sample.graph_triples,
        )

        predicted_answers = set(
            result["answers"]
        )

        gold_answers = set(
            sample.answer_entities
        )

        hit = bool(
            predicted_answers
            & gold_answers
        )

        if hit:
            hit_count += 1

        print("\n" + "=" * 100)

        print(
            f"[{index}] Question:"
        )

        print(
            sample.question
        )

        print(
            "\nTopic Entity:"
        )

        print(
            sample.topic_entities
        )

        print(
            "\nPlanner Top-3 Type Pairs:"
        )

        predicted_pairs = result.get(
            "predicted_type_pairs",
            [],
        )

        valid_pairs = set(
            result.get(
                "valid_type_pairs",
                [],
            )
        )

        if not predicted_pairs:
            print("None")

        for rank, pair in enumerate(
            predicted_pairs,
            start=1,
        ):
            status = (
                "valid"
                if pair in valid_pairs
                else "invalid"
            )

            print(
                f"{rank}. "
                f"<{pair[0]}, {pair[1]}> "
                f"[{status}]"
            )

        print(
            "\nRetrieved Candidates:"
        )

        print(
            result.get(
                "num_candidates",
                0,
            )
        )

        print(
            "\nAnswer Source:"
        )

        print(
            result["answer_source"]
        )

        if (
            result["answer_source"]
            == "generator"
        ):
            print(
                "\nBackoff Reason:"
            )

            print(
                result["backoff_reason"]
            )

        print(
            "\nPredicted Answers:"
        )

        if result["answers"]:
            for answer in result[
                "answers"
            ]:
                print(
                    f"- {answer}"
                )
        else:
            print("None")

        print(
            "\nGold Answers:"
        )

        for answer in sample.answer_entities:
            print(
                f"- {answer}"
            )

        print(
            "\nResult:"
        )

        print(
            "HIT"
            if hit
            else "MISS"
        )

    print("\n" + "=" * 100)
    print("Demo Summary")
    print("=" * 100)

    print(
        f"Questions : "
        f"{len(samples)}"
    )

    print(
        f"Hits      : "
        f"{hit_count}/{len(samples)}"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()