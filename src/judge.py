import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


class QwenJudge:
    """OntGQA Judge 推理模块。"""

    def __init__(
        self,
        model_path,
        adapter_path=None,
        device="cuda",
        torch_dtype=torch.bfloat16,
        margin_threshold=1.0,
    ):
        self.device = device
        self.margin_threshold = margin_threshold

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

        self.yes_token_ids = (
            self.tokenizer.encode(
                "YES",
                add_special_tokens=False,
            )
        )

        self.no_token_ids = (
            self.tokenizer.encode(
                "NO",
                add_special_tokens=False,
            )
        )

        if (
            len(self.yes_token_ids) != 1
            or len(self.no_token_ids) != 1
        ):
            raise ValueError(
                "当前 tokenizer 中 YES/NO "
                "不是单 token，无法直接计算 "
                "Judge log-prob margin。"
            )

        self.yes_token_id = (
            self.yes_token_ids[0]
        )

        self.no_token_id = (
            self.no_token_ids[0]
        )

    @staticmethod
    def textualize_path(path):
        """将 reasoning path 转换为文本。"""

        parts = []

        for head, relation, tail in path:
            parts.append(
                f"{head} --{relation}--> {tail}"
            )

        return " | ".join(parts)

    def build_evidence_text(
        self,
        paths,
        max_paths=3,
    ):
        """保留少量最短 evidence paths。"""

        if not paths:
            return "None"

        minimum_hops = min(
            len(path)
            for path in paths
        )

        shortest_paths = [
            path
            for path in paths
            if len(path) == minimum_hops
        ]

        shortest_paths = (
            shortest_paths[:max_paths]
        )

        textualized_paths = [
            self.textualize_path(path)
            for path in shortest_paths
        ]

        return "\n".join(
            f"{index}. {path_text}"
            for index, path_text in enumerate(
                textualized_paths,
                start=1,
            )
        )

    @staticmethod
    def build_prompt(
        question,
        candidate,
        evidence_text,
    ):
        """构造 Judge 判断指令。"""

        return (
            "You are a strict judge for knowledge-graph QA. "
            "Given a QUESTION, a CANDIDATE answer, and its "
            "EVIDENCE PATHS, decide whether the candidate is "
            "a correct final answer to the question. "
            'Return strictly one token: "YES" or "NO". '
            "No explanations.\n"
            f"Question: {question}\n"
            f"Candidate: {candidate}\n"
            f"Evidence paths:\n{evidence_text}"
        )

    def score_prompt(
        self,
        prompt,
    ):
        """
        计算 YES 和 NO 的首 token log-prob。

        margin = log P(YES) - log P(NO)

        按论文主实验设置，
        仅当 margin > 1.0 时接受候选答案。
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

        with torch.no_grad():
            outputs = self.model(
                **inputs
            )

        next_token_logits = (
            outputs.logits[
                0,
                -1,
                :
            ]
        )

        log_probs = torch.log_softmax(
            next_token_logits,
            dim=-1,
        )

        yes_log_prob = (
            log_probs[
                self.yes_token_id
            ].item()
        )

        no_log_prob = (
            log_probs[
                self.no_token_id
            ].item()
        )

        margin = (
            yes_log_prob
            - no_log_prob
        )

        accepted = (
            margin
            > self.margin_threshold
        )

        return {
            "yes_log_prob": yes_log_prob,
            "no_log_prob": no_log_prob,
            "margin": margin,
            "threshold": self.margin_threshold,
            "accepted": accepted,
        }

    def judge(
        self,
        question,
        candidate,
        evidence_paths,
        max_paths=3,
    ):
        """判断单个候选答案是否通过 Judge。"""

        evidence_text = (
            self.build_evidence_text(
                evidence_paths,
                max_paths=max_paths,
            )
        )

        prompt = self.build_prompt(
            question=question,
            candidate=candidate,
            evidence_text=evidence_text,
        )

        result = self.score_prompt(
            prompt
        )

        result.update(
            {
                "candidate": candidate,
                "evidence_text": (
                    evidence_text
                ),
            }
        )

        return result