import json

import torch
from torch.utils.data import Dataset


class GeneratorSFTDataset(Dataset):
    """OntGQA Generator 的监督微调数据集。"""

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []

        self.filtered_count = 0

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                record = json.loads(
                    line
                )

                full_messages = [
                    {
                        "role": "user",
                        "content": record["prompt"],
                    },
                    {
                        "role": "assistant",
                        "content": record["target"],
                    },
                ]

                full_text = (
                    self.tokenizer.apply_chat_template(
                        full_messages,
                        tokenize=False,
                        add_generation_prompt=False,
                    )
                )

                full_ids = self.tokenizer(
                    full_text,
                    add_special_tokens=False,
                    truncation=False,
                )["input_ids"]

                # 超长多答案问题不做截断，
                # 直接从 Generator 训练集中排除。
                if len(full_ids) > self.max_length:
                    self.filtered_count += 1
                    continue

                self.data.append(
                    record
                )

    def __len__(self):
        return len(
            self.data
        )

    def __getitem__(
        self,
        index,
    ):
        record = self.data[
            index
        ]

        prompt_messages = [
            {
                "role": "user",
                "content": record["prompt"],
            }
        ]

        full_messages = [
            {
                "role": "user",
                "content": record["prompt"],
            },
            {
                "role": "assistant",
                "content": record["target"],
            },
        ]

        prompt_text = (
            self.tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        full_text = (
            self.tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        prompt_encoding = self.tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=False,
        )

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = full_encoding[
            "input_ids"
        ]

        attention_mask = full_encoding[
            "attention_mask"
        ]

        prompt_length = min(
            len(
                prompt_encoding[
                    "input_ids"
                ]
            ),
            len(input_ids),
        )

        labels = input_ids.copy()

        # Generator 只学习答案部分，
        # Prompt token 不参与语言模型损失。
        labels[:prompt_length] = (
            [-100]
            * prompt_length
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