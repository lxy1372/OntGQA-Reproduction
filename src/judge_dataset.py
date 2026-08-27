import json

import torch
from torch.utils.data import Dataset


class JudgeSFTDataset(Dataset):
    """OntGQA Judge 的监督微调数据集。"""

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=512,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.data = []

        with open(
            jsonl_path,
            "r",
            encoding="utf-8",
        ) as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                self.data.append(
                    json.loads(line)
                )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        record = self.data[index]

        prompt = record["prompt"]
        target = record["target"]

        prompt_messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

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

        # Prompt 末尾包含 assistant generation 标记，
        # 后续对应的位置才参与 Judge 的 YES/NO 损失计算。
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
            truncation=True,
            max_length=self.max_length,
        )

        full_encoding = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = full_encoding[
            "input_ids"
        ]

        attention_mask = full_encoding[
            "attention_mask"
        ]

        prompt_length = min(
            len(prompt_encoding["input_ids"]),
            len(input_ids),
        )

        labels = input_ids.copy()

        # Question、Candidate 和 Evidence 只作为输入，
        # 不参与语言模型损失。
        labels[:prompt_length] = [
            -100
        ] * prompt_length

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