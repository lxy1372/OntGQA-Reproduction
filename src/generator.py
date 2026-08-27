import json

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class QwenGenerator:
    """OntGQA Generative Backoff 推理模块。"""

    def __init__(
        self,
        model_path,
        adapter_path=None,
        device="cuda",
        torch_dtype=torch.bfloat16,
    ):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        model_device = (
            {"": 0}
            if device.startswith("cuda")
            else None
        )

        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
                torch_dtype=torch_dtype,
                device_map=model_device,
            )
        )

        if adapter_path is not None:
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                local_files_only=True,
            )

        self.model.eval()

        # 使用确定性生成时关闭采样参数，
        # 避免 transformers 输出无意义警告。
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None

    @staticmethod
    def build_prompt(question):
        """构造 Generator 指令。"""

        return (
            "<TASK: ANSWERS>\n"
            "Please generate ALL correct answer entities "
            "for the following question:\n"
            f"Question: {question}"
        )

    @staticmethod
    def parse_answers(text):
        """
        解析模型输出的 JSON 答案数组。

        标准输出格式：
        ["answer1", "answer2"]
        """

        text = text.strip()

        # 优先直接按照完整 JSON 解析
        try:
            data = json.loads(
                text
            )

            if isinstance(
                data,
                list,
            ):
                answers = []

                for item in data:
                    if not isinstance(
                        item,
                        str,
                    ):
                        continue

                    item = item.strip()

                    if item:
                        answers.append(
                            item
                        )

                return list(
                    dict.fromkeys(
                        answers
                    )
                )

        except json.JSONDecodeError:
            pass

        # 模型偶尔可能在 JSON 数组前后生成少量额外文本，
        # 此时提取最外层数组重新解析。
        left = text.find("[")
        right = text.rfind("]")

        if (
            left != -1
            and right != -1
            and right > left
        ):
            json_text = text[
                left:right + 1
            ]

            try:
                data = json.loads(
                    json_text
                )

                if isinstance(
                    data,
                    list,
                ):
                    answers = []

                    for item in data:
                        if not isinstance(
                            item,
                            str,
                        ):
                            continue

                        item = item.strip()

                        if item:
                            answers.append(
                                item
                            )

                    return list(
                        dict.fromkeys(
                            answers
                        )
                    )

            except json.JSONDecodeError:
                pass

        return []

    def generate(
        self,
        question,
        max_new_tokens=256,
    ):
        """根据问题直接生成答案实体列表。"""

        prompt = self.build_prompt(
            question
        )

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

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=False,
        )

        inputs = {
            key: value.to(
                self.model.device
            )
            for key, value in inputs.items()
        }

        input_length = (
            inputs["input_ids"].shape[1]
        )

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.pad_token_id
                ),
                eos_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )

        generated_ids = output_ids[
            0,
            input_length:,
        ]

        raw_output = (
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            )
            .strip()
        )

        answers = self.parse_answers(
            raw_output
        )

        return {
            "question": question,
            "raw_output": raw_output,
            "answers": answers,
        }