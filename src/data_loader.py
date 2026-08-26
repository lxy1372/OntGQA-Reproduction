#!/usr/bin/env python3

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pyarrow.parquet as pq


Triple = List[str]


@dataclass
class WebQSPSample:
    """单条 RoG-WebQSP 样本。"""

    sample_id: str
    question: str
    answers: List[str]
    topic_entities: List[str]
    answer_entities: List[str]
    graph_triples: List[Triple]
    choices: List


class WebQSPDataLoader:
    """读取本地 RoG-WebQSP Parquet 数据。"""

    def __init__(self, parquet_path: str | Path):
        self.parquet_path = Path(parquet_path)

        if not self.parquet_path.exists():
            raise FileNotFoundError(
                f"WebQSP parquet file not found: {self.parquet_path}"
            )

        self.parquet_file = pq.ParquetFile(self.parquet_path)

    def num_row_groups(self) -> int:
        """获取 Parquet 文件中的 row group 数量。"""

        return self.parquet_file.num_row_groups

    def load_row_group(self, row_group_index: int) -> List[WebQSPSample]:
        """读取指定 row group 中的全部样本。"""

        if row_group_index < 0 or row_group_index >= self.num_row_groups():
            raise IndexError(
                f"Invalid row group index: {row_group_index}"
            )

        table = self.parquet_file.read_row_group(row_group_index)
        rows = table.to_pylist()

        return [
            self._convert_row(row)
            for row in rows
        ]

    def load_first_n(self, n: int) -> List[WebQSPSample]:
        """按顺序读取前 n 条样本。"""

        if n <= 0:
            return []

        samples = []

        for row_group_index in range(self.num_row_groups()):
            row_group_samples = self.load_row_group(row_group_index)

            remaining = n - len(samples)

            samples.extend(row_group_samples[:remaining])

            if len(samples) >= n:
                break

        return samples

    def get_sample(self, index: int) -> WebQSPSample:
        """按照文件中的全局序号读取单条样本。"""

        if index < 0:
            raise IndexError("Sample index must be non-negative.")

        current_index = 0

        for row_group_index in range(self.num_row_groups()):
            row_group_samples = self.load_row_group(row_group_index)

            next_index = current_index + len(row_group_samples)

            if index < next_index:
                return row_group_samples[index - current_index]

            current_index = next_index

        raise IndexError(f"Sample index out of range: {index}")

    @staticmethod
    def _convert_row(row: dict) -> WebQSPSample:
        """将 Parquet 原始记录转换为统一的数据结构。"""

        graph = row.get("graph") or []

        graph_triples = []

        for triple in graph:
            if not isinstance(triple, list) or len(triple) != 3:
                raise ValueError(
                    f"Invalid graph triple: {triple}"
                )

            head, relation, tail = triple

            graph_triples.append([
                head,
                relation,
                tail,
            ])

        return WebQSPSample(
            sample_id=row.get("id") or "",
            question=row.get("question") or "",
            answers=list(row.get("answer") or []),
            topic_entities=list(row.get("q_entity") or []),
            answer_entities=list(row.get("a_entity") or []),
            graph_triples=graph_triples,
            choices=list(row.get("choices") or []),
        )