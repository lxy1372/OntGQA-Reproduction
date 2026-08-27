#!/usr/bin/env python3

import json

import torch
from torch.utils.data import Dataset


class JudgeGroupedDataset(Dataset):
    """
    按问题组织的 OntGQA Judge 数据集。

    每一个 dataset item 对应一个问题：

        q
        ├── positives = Y_q
        └── negatives = N_q

    Listwise Judge 训练不再把 YES / NO 当作普通
    Causal LM target，而是直接读取 Prompt 结束位置
    对 YES / NO token 的概率。
    """

    def __init__(
        self,
        jsonl_path,
        tokenizer,
        max_length=640,
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

                positives = record[
                    "positives"
                ]

                negatives = record[
                    "negatives"
                ]

                if not positives:
                    raise ValueError(
                        f"{record['id']} "
                        f"has no positive candidates."
                    )

                if not negatives:
                    raise ValueError(
                        f"{record['id']} "
                        f"has no negative candidates."
                    )

                self.data.append(
                    record
                )

    def __len__(
        self,
    ):
        return len(
            self.data
        )

    def encode_prompt(
        self,
        prompt,
    ):
        """
        编码一个 Judge Prompt。

        Listwise loss 需要 Prompt 结束位置的
        next-token YES / NO probability，
        因此这里不附加 YES 或 NO target。
        """

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )

        encoding = self.tokenizer(
            text,
            add_special_tokens=False,
            truncation=False,
        )

        input_ids = encoding[
            "input_ids"
        ]

        attention_mask = encoding[
            "attention_mask"
        ]

        sequence_length = len(
            input_ids
        )

        # 不允许像旧版 max_length=512 那样
        # 静默截断 evidence。
        if (
            sequence_length
            > self.max_length
        ):
            raise ValueError(
                f"Judge prompt length "
                f"{sequence_length} exceeds "
                f"max_length={self.max_length}."
            )

        return {
            "input_ids": input_ids,
            "attention_mask": (
                attention_mask
            ),
            "length": (
                sequence_length
            ),
        }

    def encode_candidates(
        self,
        candidates,
    ):
        """编码一个正例或负例候选集合。"""

        encoded = []

        for candidate_record in candidates:
            encoded_prompt = (
                self.encode_prompt(
                    candidate_record[
                        "prompt"
                    ]
                )
            )

            encoded.append(
                {
                    "candidate": (
                        candidate_record[
                            "candidate"
                        ]
                    ),
                    "input_ids": (
                        encoded_prompt[
                            "input_ids"
                        ]
                    ),
                    "attention_mask": (
                        encoded_prompt[
                            "attention_mask"
                        ]
                    ),
                    "length": (
                        encoded_prompt[
                            "length"
                        ]
                    ),
                }
            )

        return encoded

    def __getitem__(
        self,
        index,
    ):
        record = self.data[
            index
        ]

        positives = (
            self.encode_candidates(
                record[
                    "positives"
                ]
            )
        )

        negatives = (
            self.encode_candidates(
                record[
                    "negatives"
                ]
            )
        )

        return {
            "id": record[
                "id"
            ],
            "question": record[
                "question"
            ],
            "positives": positives,
            "negatives": negatives,
        }


class JudgeGroupedCollator:
    """
    将一个问题内的所有正负候选组成一个 batch。

    训练时使用：

        per_device_train_batch_size = 1

    因此一个 DataLoader batch 就对应一个问题。

    候选排列顺序：

        [全部 positive, 全部 negative]
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
        if len(features) != 1:
            raise ValueError(
                "JudgeGroupedCollator requires "
                "per_device_train_batch_size=1."
            )

        feature = features[
            0
        ]

        positives = feature[
            "positives"
        ]

        negatives = feature[
            "negatives"
        ]

        all_candidates = (
            positives
            + negatives
        )

        num_positive = len(
            positives
        )

        num_negative = len(
            negatives
        )

        if (
            num_positive == 0
            or num_negative == 0
        ):
            raise ValueError(
                "Each Judge question must contain "
                "both positive and negative candidates."
            )

        max_length = max(
            item[
                "length"
            ]
            for item
            in all_candidates
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

        pad_token_id = (
            self.tokenizer.pad_token_id
        )

        if pad_token_id is None:
            pad_token_id = (
                self.tokenizer.eos_token_id
            )

        batch_input_ids = []
        batch_attention_mask = []
        lengths = []

        candidate_names = []

        for item in all_candidates:
            input_ids = item[
                "input_ids"
            ]

            attention_mask = item[
                "attention_mask"
            ]

            padding_length = (
                max_length
                - len(
                    input_ids
                )
            )

            # 使用右侧 padding。
            #
            # 后续 Trainer 不能直接取 logits[:, -1, :]，
            # 必须根据 lengths 找每条序列最后一个真实 token。
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

            lengths.append(
                len(
                    input_ids
                )
            )

            candidate_names.append(
                item[
                    "candidate"
                ]
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

            # 每条 Prompt 的真实长度。
            # Trainer 根据 length-1 取得
            # next-token YES/NO logits。
            "lengths": torch.tensor(
                lengths,
                dtype=torch.long,
            ),

            "num_positive": torch.tensor(
                num_positive,
                dtype=torch.long,
            ),

            "num_negative": torch.tensor(
                num_negative,
                dtype=torch.long,
            ),

            "sample_id": (
                feature[
                    "id"
                ]
            ),

            "question": (
                feature[
                    "question"
                ]
            ),

            "candidate_names": (
                candidate_names
            ),
        }