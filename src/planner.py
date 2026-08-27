#!/usr/bin/env python3

import re
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


TypePair = Tuple[str, str]


class QwenPlanner:
    """基于 Qwen2-Instruct 和 LoRA Adapter 的 Type Pair Planner。"""

    def __init__(
        self,
        model_path: str | Path,
        adapter_path: Optional[str | Path] = None,
        device: str = "cuda",
        torch_dtype=torch.bfloat16,
    ):
        self.model_path = str(model_path)

        self.adapter_path = (
            str(adapter_path)
            if adapter_path is not None
            else None
        )

        self.device = device
        self.torch_dtype = torch_dtype

        # 加载基础模型 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            local_files_only=True,
        )

        # 加载 Qwen2 基础模型
        if self.device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map={"": 0},
                local_files_only=True,
            )

        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                local_files_only=True,
            )

            self.model.to(
                self.device
            )

        # 加载 Planner LoRA Adapter
        if self.adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                self.adapter_path,
                local_files_only=True,
            )

        self.model.eval()

        # Beam Search 不使用采样参数
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @staticmethod
    def build_prompt(
        question: str,
    ) -> str:
        """构造 Planner 指令输入。"""

        return (
            "<TASK: TYPE_PAIRS>\n"
            "Please generate a valid type pair that can be helpful "
            "for answering the following question:\n"
            f"Question: {question}"
        )

    @staticmethod
    def parse_type_pairs(
        text: str,
    ) -> List[TypePair]:
        """从生成文本中解析 Type Pair。"""

        pattern = re.compile(
            r"<PAIR>\s*(.*?)\s*<SEP>\s*(.*?)\s*</PAIR>",
            flags=re.DOTALL,
        )

        pairs = []

        for match in pattern.findall(
            text
        ):
            head_type = match[0].strip()
            tail_type = match[1].strip()

            if not head_type or not tail_type:
                continue

            pairs.append(
                (
                    head_type,
                    tail_type,
                )
            )

        # 去除重复 Type Pair，同时保持生成顺序
        unique_pairs = []
        seen = set()

        for pair in pairs:
            if pair in seen:
                continue

            seen.add(
                pair
            )

            unique_pairs.append(
                pair
            )

        return unique_pairs

    def generate(
        self,
        question: str,
        top_k: int = 3,
        num_beams: int = 3,
        max_new_tokens: int = 64,
    ) -> dict:
        """生成单个问题对应的 Top-k Type Pairs。"""

        if top_k > num_beams:
            raise ValueError(
                "top_k cannot be greater than num_beams."
            )

        prompt = self.build_prompt(
            question
        )

        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        model_inputs = self.tokenizer(
            [text],
            return_tensors="pt",
        )

        if self.device == "cuda":
            model_inputs = {
                key: value.to(
                    self.model.device
                )
                for key, value
                in model_inputs.items()
            }

        with torch.no_grad():
            outputs = self.model.generate(
                **model_inputs,
                do_sample=False,
                num_beams=num_beams,
                num_return_sequences=top_k,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_length = (
            model_inputs[
                "input_ids"
            ].shape[1]
        )

        decoded_outputs = []

        for output_ids in outputs:
            generated_ids = (
                output_ids[
                    input_length:
                ]
            )

            decoded_text = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()

            decoded_outputs.append(
                decoded_text
            )

        predicted_pairs = []
        seen_pairs = set()

        for decoded_text in decoded_outputs:
            pairs = self.parse_type_pairs(
                decoded_text
            )

            for pair in pairs:
                if pair in seen_pairs:
                    continue

                seen_pairs.add(
                    pair
                )

                predicted_pairs.append(
                    pair
                )

        return {
            "question": question,
            "prompt": prompt,
            "raw_outputs": decoded_outputs,
            "predicted_type_pairs": predicted_pairs,
        }