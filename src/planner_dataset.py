#!/usr/bin/env python3

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PlannerSFTDataset(Dataset):
    """Planner 监督微调数据集。"""

    def __init__(
        self,
        data_path,
        tokenizer,
        max_length=256,
    ):
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.records = []

        with open(
            self.data_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.records.append(
                    json.loads(line)
                )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]

        prompt = record["prompt"]
        target = record["target"]

        # 构造只有用户输入的文本
        prompt_messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        # 构造包含目标答案的完整对话
        full_messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": target,
            },
        ]

        full_text = (
            self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        # 分别编码 Prompt 和完整训练文本
        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_encoding["input_ids"]
        attention_mask = full_encoding["attention_mask"]

        # 训练时只计算 Assistant 输出部分的损失
        labels = input_ids.copy()

        prompt_length = min(
            len(prompt_ids),
            len(labels),
        )

        labels[:prompt_length] = (
            [-100] * prompt_length
        )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }