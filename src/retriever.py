#!/usr/bin/env python3

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.ontology import OntologyGraph


KGTriple = Tuple[str, str, str]
ReasoningPath = Tuple[KGTriple, ...]


@dataclass
class RetrievalResult:
    """单个类型规划对应的检索结果。"""

    head_type: str
    tail_type: str

    one_hop_paths: List[ReasoningPath]
    two_hop_paths: List[ReasoningPath]

    candidate_to_paths: Dict[str, List[ReasoningPath]]

    @property
    def candidates(self) -> List[str]:
        """获取去重后的候选答案。"""

        return list(self.candidate_to_paths.keys())

    @property
    def num_paths(self) -> int:
        """获取检索出的 reasoning path 总数。"""

        return (
            len(self.one_hop_paths)
            + len(self.two_hop_paths)
        )


class OntologyRetriever:
    """基于本体类型约束进行知识图谱路径检索。"""

    def __init__(
        self,
        ontology: OntologyGraph,
        graph_triples: List[List[str]],
    ):
        self.ontology = ontology

        # 将原始三元组统一转换为不可变 tuple
        self.graph_triples: List[KGTriple] = [
            (head, relation, tail)
            for head, relation, tail in graph_triples
        ]

        # 按头实体建立知识图谱正向邻接索引
        self.outgoing_by_entity: Dict[
            str,
            List[Tuple[str, str]]
        ] = defaultdict(list)

        self._build_graph_index()

    def _build_graph_index(self) -> None:
        """建立知识图谱正向邻接索引。"""

        for head, relation, tail in self.graph_triples:
            self.outgoing_by_entity[head].append(
                (relation, tail)
            )

    def retrieve(
        self,
        topic_entities: List[str],
        head_type: str,
        tail_type: str,
    ) -> RetrievalResult:
        """
        根据 Head-Tail Type Pair 检索 1-hop 和 2-hop reasoning paths。

        1-hop：
            第一跳关系属于 R+(head_type)

        2-hop：
            第一跳关系属于 R+(head_type)
            第二跳关系属于 R-(tail_type)
        """

        allowed_first_relations = (
            self.ontology.get_outgoing_relations(
                head_type
            )
        )

        allowed_last_relations = (
            self.ontology.get_incoming_relations(
                tail_type
            )
        )

        one_hop_paths = self._retrieve_one_hop(
            topic_entities=topic_entities,
            allowed_relations=allowed_first_relations,
        )

        two_hop_paths = self._retrieve_two_hop(
            topic_entities=topic_entities,
            allowed_first_relations=allowed_first_relations,
            allowed_last_relations=allowed_last_relations,
        )

        # 合并并去除完全重复的 reasoning path
        all_paths = self._deduplicate_paths(
            one_hop_paths + two_hop_paths
        )

        candidate_to_paths = self._group_by_candidate(
            all_paths
        )

        return RetrievalResult(
            head_type=head_type,
            tail_type=tail_type,
            one_hop_paths=self._deduplicate_paths(
                one_hop_paths
            ),
            two_hop_paths=self._deduplicate_paths(
                two_hop_paths
            ),
            candidate_to_paths=candidate_to_paths,
        )

    def _retrieve_one_hop(
        self,
        topic_entities: List[str],
        allowed_relations: set[str],
    ) -> List[ReasoningPath]:
        """检索满足 Head Type 约束的 1-hop 路径。"""

        paths = []

        for topic_entity in topic_entities:
            for relation, tail in self.outgoing_by_entity.get(
                topic_entity,
                [],
            ):
                if relation not in allowed_relations:
                    continue

                path = (
                    (
                        topic_entity,
                        relation,
                        tail,
                    ),
                )

                paths.append(path)

        return paths

    def _retrieve_two_hop(
        self,
        topic_entities: List[str],
        allowed_first_relations: set[str],
        allowed_last_relations: set[str],
    ) -> List[ReasoningPath]:
        """检索满足首尾本体边界约束的 2-hop 路径。"""

        paths = []

        for topic_entity in topic_entities:
            first_edges = self.outgoing_by_entity.get(
                topic_entity,
                [],
            )

            for first_relation, middle_entity in first_edges:
                if first_relation not in allowed_first_relations:
                    continue

                second_edges = self.outgoing_by_entity.get(
                    middle_entity,
                    [],
                )

                for second_relation, tail_entity in second_edges:
                    if second_relation not in allowed_last_relations:
                        continue

                    path = (
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

                    paths.append(path)

        return paths

    @staticmethod
    def _deduplicate_paths(
        paths: List[ReasoningPath],
    ) -> List[ReasoningPath]:
        """按照完整三元组序列去除重复 reasoning path。"""

        unique_paths = []
        seen = set()

        for path in paths:
            if path in seen:
                continue

            seen.add(path)
            unique_paths.append(path)

        return unique_paths

    @staticmethod
    def _group_by_candidate(
        paths: List[ReasoningPath],
    ) -> Dict[str, List[ReasoningPath]]:
        """按照终点实体聚合 candidate 及其 evidence paths。"""

        candidate_to_paths = defaultdict(list)

        for path in paths:
            if not path:
                continue

            candidate = path[-1][2]

            candidate_to_paths[candidate].append(
                path
            )

        return dict(candidate_to_paths)