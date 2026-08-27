import json

import torch
from torch.utils.data import Dataset


class PlannerGroupedDataset(Dataset):
    """按问题组织的 Planner 多正例监督数据集。"""

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=256,
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

                record = json.loads(
                    line
                )

                if not record["targets"]:
                    continue

                self.data.append(
                    record
                )

    def __len__(self):
        return len(
            self.data
        )

    def encode_target(
        self,
        prompt,
        target,
    ):
        """编码同一问题下的一个合法 Type Pair。"""

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
            len(
                prompt_encoding[
                    "input_ids"
                ]
            ),
            len(input_ids),
        )

        labels = input_ids.copy()

        # Planner 只学习 Type Pair 输出，
        # Prompt 部分不参与语言模型损失。
        labels[:prompt_length] = (
            [-100]
            * prompt_length
        )

        return {
            "input_ids": input_ids,
            "attention_mask": (
                attention_mask
            ),
            "labels": labels,
        }

    def __getitem__(
        self,
        index,
    ):
        record = self.data[
            index
        ]

        encoded_targets = []

        for target in record[
            "targets"
        ]:
            encoded_targets.append(
                self.encode_target(
                    prompt=record[
                        "prompt"
                    ],
                    target=target,
                )
            )

        return {
            "id": record["id"],
            "question": record[
                "question"
            ],
            "targets": record[
                "targets"
            ],
            "encoded_targets": (
                encoded_targets
            ),
        }


class PlannerGroupedCollator:
    """
    将一个 batch 中每个问题的多个合法 Type Pair 展平。

    例如：
        Q1 -> 3 个 Pair
        Q2 -> 1 个 Pair

    展平后模型一次前向处理 4 条序列，
    同时保留 group_sizes=[3, 1]，
    后续 loss 再按问题重新求平均。
    """

    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of=8,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = (
            pad_to_multiple_of
        )

    def __call__(
        self,
        features,
    ):
        flat_features = []
        group_sizes = []

        for feature in features:
            encoded_targets = feature[
                "encoded_targets"
            ]

            group_sizes.append(
                len(encoded_targets)
            )

            flat_features.extend(
                encoded_targets
            )

        max_length = max(
            len(
                feature[
                    "input_ids"
                ]
            )
            for feature in flat_features
        )

        if (
            self.pad_to_multiple_of
            is not None
        ):
            multiple = (
                self.pad_to_multiple_of
            )

            max_length = (
                (
                    max_length
                    + multiple
                    - 1
                )
                // multiple
                * multiple
            )

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        pad_token_id = (
            self.tokenizer.pad_token_id
        )

        if pad_token_id is None:
            pad_token_id = (
                self.tokenizer.eos_token_id
            )

        for feature in flat_features:
            input_ids = feature[
                "input_ids"
            ]

            attention_mask = feature[
                "attention_mask"
            ]

            labels = feature[
                "labels"
            ]

            padding_length = (
                max_length
                - len(input_ids)
            )

            batch_input_ids.append(
                input_ids
                + [
                    pad_token_id
                ]
                * padding_length
            )

            batch_attention_mask.append(
                attention_mask
                + [0]
                * padding_length
            )

            batch_labels.append(
                labels
                + [-100]
                * padding_length
            )

        return {
            "input_ids": torch.tensor(
                batch_input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                batch_attention_mask,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                batch_labels,
                dtype=torch.long,
            ),
            "group_sizes": torch.tensor(
                group_sizes,
                dtype=torch.long,
            ),
        }