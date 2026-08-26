#!/usr/bin/env python3

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# 本体图边：(关系, 相邻实体类型)
OntologyEdge = Tuple[str, str]

# 关系签名：(头实体类型, 尾实体类型)
RelationSignature = Tuple[str, str]


class OntologyGraph:
    """
    Freebase 本体图加载与索引。

    每条本体三元组格式为：
        [head_type, relation, tail_type]

    加载后建立三个主要索引：
        1. head_type -> [(relation, tail_type), ...]
        2. tail_type -> [(relation, head_type), ...]
        3. relation -> (head_type, tail_type)
    """

    def __init__(self, ontology_path: str | Path):
        self.ontology_path = Path(ontology_path)

        # 原始本体三元组
        self.triples: List[Tuple[str, str, str]] = []

        # 按头实体类型建立正向索引
        self.outgoing_by_type: Dict[str, List[OntologyEdge]] = defaultdict(list)

        # 按尾实体类型建立反向索引
        self.incoming_by_type: Dict[str, List[OntologyEdge]] = defaultdict(list)

        # 按关系建立类型签名索引
        self.relation_to_types: Dict[str, RelationSignature] = {}

        # 本体图中出现的全部实体类型
        self.entity_types: Set[str] = set()

        self._load()

    def _load(self) -> None:
        """加载本体文件并建立索引。"""

        if not self.ontology_path.exists():
            raise FileNotFoundError(
                f"Ontology file not found: {self.ontology_path}"
            )

        with open(self.ontology_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise TypeError(
                f"Expected top-level ontology data to be a list, "
                f"got {type(data)}"
            )

        for index, item in enumerate(data):
            if not isinstance(item, list) or len(item) != 3:
                raise ValueError(
                    f"Invalid ontology triple at index {index}: {item}"
                )

            head_type, relation, tail_type = item

            if not all(
                isinstance(value, str)
                for value in (head_type, relation, tail_type)
            ):
                raise TypeError(
                    f"Ontology triple contains non-string value "
                    f"at index {index}: {item}"
                )

            # 当前发布的本体中，每个 relation 对应唯一的类型签名
            if relation in self.relation_to_types:
                old_signature = self.relation_to_types[relation]

                if old_signature != (head_type, tail_type):
                    raise ValueError(
                        f"Relation has multiple ontology signatures: "
                        f"{relation}\n"
                        f"Old: {old_signature}\n"
                        f"New: {(head_type, tail_type)}"
                    )

            triple = (head_type, relation, tail_type)

            self.triples.append(triple)

            self.outgoing_by_type[head_type].append(
                (relation, tail_type)
            )

            self.incoming_by_type[tail_type].append(
                (relation, head_type)
            )

            self.relation_to_types[relation] = (
                head_type,
                tail_type
            )

            self.entity_types.add(head_type)
            self.entity_types.add(tail_type)

    def get_outgoing_edges(
        self,
        head_type: str
    ) -> List[OntologyEdge]:
        """
        获取指定头实体类型的全部出边。

        返回格式：
            [(relation, tail_type), ...]
        """

        return list(self.outgoing_by_type.get(head_type, []))

    def get_outgoing_relations(
        self,
        head_type: str
    ) -> Set[str]:
        """
        获取指定头实体类型允许的全部出关系。

        对应论文中的 R+(head_type)。
        """

        return {
            relation
            for relation, _ in self.outgoing_by_type.get(head_type, [])
        }

    def get_incoming_edges(
        self,
        tail_type: str
    ) -> List[OntologyEdge]:
        """
        获取指向指定尾实体类型的全部入边。

        返回格式：
            [(relation, head_type), ...]
        """

        return list(self.incoming_by_type.get(tail_type, []))

    def get_incoming_relations(
        self,
        tail_type: str
    ) -> Set[str]:
        """
        获取可以进入指定尾实体类型的全部关系。

        对应论文中的 R-(tail_type)。
        """

        return {
            relation
            for relation, _ in self.incoming_by_type.get(tail_type, [])
        }

    def get_relation_signature(
        self,
        relation: str
    ) -> Optional[RelationSignature]:
        """
        获取关系对应的类型签名。

        返回：
            (head_type, tail_type)

        关系不存在时返回 None。
        """

        return self.relation_to_types.get(relation)

    def has_relation(self, relation: str) -> bool:
        """判断关系是否存在于本体图中。"""

        return relation in self.relation_to_types

    def has_type(self, entity_type: str) -> bool:
        """判断实体类型是否存在于本体图中。"""

        return entity_type in self.entity_types

    def get_all_types(self) -> Set[str]:
        """获取本体图中的全部实体类型。"""

        return set(self.entity_types)

    def get_all_relations(self) -> Set[str]:
        """获取本体图中的全部关系。"""

        return set(self.relation_to_types.keys())

    def stats(self) -> dict:
        """获取本体图基本统计信息。"""

        return {
            "num_triples": len(self.triples),
            "num_relations": len(self.relation_to_types),
            "num_entity_types": len(self.entity_types),
            "num_head_types": len(self.outgoing_by_type),
            "num_tail_types": len(self.incoming_by_type),
        }