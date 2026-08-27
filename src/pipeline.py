from collections import defaultdict


class OntGQAPipeline:
    """OntGQA 完整推理流程。"""

    def __init__(
        self,
        ontology,
        planner,
        judge,
        generator,
        retriever_class,
    ):
        self.ontology = ontology
        self.planner = planner
        self.judge = judge
        self.generator = generator
        self.retriever_class = retriever_class

    @staticmethod
    def deduplicate_paths(paths):
        """去除重复 reasoning paths。"""

        unique_paths = []
        seen = set()

        for path in paths:
            if path in seen:
                continue

            seen.add(path)

            unique_paths.append(
                path
            )

        return unique_paths

    @staticmethod
    def merge_candidate_paths(
        candidate_to_paths,
        new_candidate_to_paths,
    ):
        """合并不同 Type Pair 产生的候选及证据路径。"""

        for (
            candidate,
            paths,
        ) in new_candidate_to_paths.items():

            candidate_to_paths[
                candidate
            ].extend(
                paths
            )

    def filter_type_pairs(
        self,
        predicted_pairs,
    ):
        """过滤 Planner 生成的非法 ontology types。"""

        valid_pairs = []

        for (
            head_type,
            tail_type,
        ) in predicted_pairs:

            if (
                self.ontology.has_type(
                    head_type
                )
                and self.ontology.has_type(
                    tail_type
                )
            ):
                valid_pairs.append(
                    (
                        head_type,
                        tail_type,
                    )
                )

        return valid_pairs

    def retrieve_candidates(
        self,
        graph_triples,
        topic_entities,
        type_pairs,
    ):
        """根据 Planner Type Pairs 执行本体约束检索。"""

        retriever = self.retriever_class(
            ontology=self.ontology,
            graph_triples=graph_triples,
        )

        candidate_to_paths = defaultdict(
            list
        )

        pair_results = []

        for (
            head_type,
            tail_type,
        ) in type_pairs:

            result = retriever.retrieve(
                topic_entities=topic_entities,
                head_type=head_type,
                tail_type=tail_type,
            )

            self.merge_candidate_paths(
                candidate_to_paths,
                result.candidate_to_paths,
            )

            pair_results.append(
                {
                    "head_type": head_type,
                    "tail_type": tail_type,
                    "num_candidates": len(
                        result.candidates
                    ),
                    "num_paths": (
                        result.num_paths
                    ),
                }
            )

        # 不同 Type Pair 可能产生重复路径
        for candidate in candidate_to_paths:
            candidate_to_paths[
                candidate
            ] = self.deduplicate_paths(
                candidate_to_paths[
                    candidate
                ]
            )

        return (
            candidate_to_paths,
            pair_results,
        )

    def judge_candidates(
        self,
        question,
        candidate_to_paths,
    ):
        """使用 Judge 对 Retriever 候选逐个判断。"""

        judged_candidates = []

        for candidate in sorted(
            candidate_to_paths
        ):
            paths = candidate_to_paths[
                candidate
            ]

            result = self.judge.judge(
                question=question,
                candidate=candidate,
                evidence_paths=paths,
                max_paths=3,
            )

            judged_candidates.append(
                {
                    "candidate": candidate,
                    "margin": result[
                        "margin"
                    ],
                    "accepted": result[
                        "accepted"
                    ],
                    "evidence_text": result[
                        "evidence_text"
                    ],
                }
            )

        judged_candidates.sort(
            key=lambda item: (
                item["margin"]
            ),
            reverse=True,
        )

        accepted_candidates = [
            item
            for item in judged_candidates
            if item["accepted"]
        ]

        return (
            judged_candidates,
            accepted_candidates,
        )

    def generator_backoff(
        self,
        question,
        reason,
    ):
        """Retriever/Judge 无可用答案时执行生成式回退。"""

        generator_result = (
            self.generator.generate(
                question=question
            )
        )

        return {
            "question": question,
            "answers": (
                generator_result[
                    "answers"
                ]
            ),
            "answer_source": "generator",
            "backoff_reason": reason,
            "generator_raw_output": (
                generator_result[
                    "raw_output"
                ]
            ),
        }

    def answer(
        self,
        question,
        topic_entities,
        graph_triples,
    ):
        """
        执行完整 OntGQA 推理。

        Question
          -> Planner
          -> Ontology-constrained Retriever
          -> Judge
          -> Generative Backoff
        """

        # 1. Planner 预测 Top-3 Type Pairs
        planner_result = (
            self.planner.generate(
                question=question,
                top_k=3,
                num_beams=3,
            )
        )

        predicted_pairs = (
            planner_result[
                "predicted_type_pairs"
            ][:3]
        )

        # 2. 过滤不属于官方 ontology 的类型
        valid_pairs = (
            self.filter_type_pairs(
                predicted_pairs
            )
        )

        # Planner 没有产生任何有效计划
        if not valid_pairs:
            result = (
                self.generator_backoff(
                    question=question,
                    reason="no_valid_type_pair",
                )
            )

            result.update(
                {
                    "predicted_type_pairs": (
                        predicted_pairs
                    ),
                    "valid_type_pairs": [],
                    "num_candidates": 0,
                    "accepted_candidates": [],
                }
            )

            return result

        # 3. Ontology-guided Retriever
        (
            candidate_to_paths,
            pair_results,
        ) = self.retrieve_candidates(
            graph_triples=graph_triples,
            topic_entities=topic_entities,
            type_pairs=valid_pairs,
        )

        # Retriever 没有找到任何候选
        if not candidate_to_paths:
            result = (
                self.generator_backoff(
                    question=question,
                    reason="no_retrieved_candidate",
                )
            )

            result.update(
                {
                    "predicted_type_pairs": (
                        predicted_pairs
                    ),
                    "valid_type_pairs": (
                        valid_pairs
                    ),
                    "pair_results": (
                        pair_results
                    ),
                    "num_candidates": 0,
                    "accepted_candidates": [],
                }
            )

            return result

        # 4. Judge 逐个判断候选
        (
            judged_candidates,
            accepted_candidates,
        ) = self.judge_candidates(
            question=question,
            candidate_to_paths=(
                candidate_to_paths
            ),
        )

        # 所有 Retriever 候选都被 Judge 拒绝
        if not accepted_candidates:
            result = (
                self.generator_backoff(
                    question=question,
                    reason="all_candidates_rejected",
                )
            )

            result.update(
                {
                    "predicted_type_pairs": (
                        predicted_pairs
                    ),
                    "valid_type_pairs": (
                        valid_pairs
                    ),
                    "pair_results": (
                        pair_results
                    ),
                    "num_candidates": len(
                        candidate_to_paths
                    ),
                    "judged_candidates": (
                        judged_candidates
                    ),
                    "accepted_candidates": [],
                }
            )

            return result

        # 5. 至少一个候选通过 Judge
        answers = [
            item["candidate"]
            for item in accepted_candidates
        ]

        return {
            "question": question,
            "answers": answers,
            "answer_source": "judge",
            "backoff_reason": None,
            "predicted_type_pairs": (
                predicted_pairs
            ),
            "valid_type_pairs": (
                valid_pairs
            ),
            "pair_results": (
                pair_results
            ),
            "num_candidates": len(
                candidate_to_paths
            ),
            "judged_candidates": (
                judged_candidates
            ),
            "accepted_candidates": (
                accepted_candidates
            ),
        }