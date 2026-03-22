import torch
from .terminal import Name
from typing import List, Dict, Optional, Tuple, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
from .voices import TTS
import re
import json

llmMsgs = List[Dict[str, str]]

# @dataclass
# class PromptTable:
#     instruction: str

# @dataclass
# class HeldItem:
#     name: str
#     spatial_weight: float
#     description: str
#     pointers: List[Tuple[str, int]]

class TextModelRunner:
    def __repr__(self):
        objId = hex(id(self))
        short = '0x..' + objId[-4:]
        return f'<{short}:{Name(self)}>'

    _SharedModels = {}

    def __init__(
        self,
        model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        device: Optional[str] = None
    ):

        self.model = TextModelRunner._SharedModels.setdefault(
            f'm_{model}',
            AutoModelForCausalLM.from_pretrained(model)
        )
        self.tokenizer = TextModelRunner._SharedModels.setdefault(
            f't_{tokenizer}',
            AutoTokenizer.from_pretrained(tokenizer)
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.TTS = TTS()

    def token_count(self, messages: llmMsgs) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True))

    def instructions(self, messages: llmMsgs):
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return tokens  # dict with input_ids, attention_mask, etc.

    @staticmethod
    def _safe_json_extract(text: str) -> Optional[str]:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start:end + 1]

        try:
            json.loads(candidate)
            return candidate
        except Exception:
            pass

        repaired = candidate

        open_braces = repaired.count("{")
        close_braces = repaired.count("}")
        if open_braces > close_braces:
            repaired += "}" * (open_braces - close_braces)

        repaired = re.sub(r",\s*}", "}", repaired)
        repaired = re.sub(r",\s*]", "]", repaired)

        try:
            json.loads(repaired)
            return repaired
        except Exception:
            return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.strip()
        matches = list(re.finditer(r'[.!?]', text))
        if matches:
            return text[:matches[-1].end()]
        else:
            return text.rstrip(".") + "..."

    def think(
            self,
            instructions,  # llmMsgs OR tokenized dict
            normalize_decoded: bool = False,
            max_new_tokens: Optional[int] = 128,
            json_retry: int = 2,
            wrap_role: Optional[str] = None,
            json_mode: bool = False
        ) -> Any:

        def wrap(output):
            if wrap_role:
                return [{"role": wrap_role, "content": output}]
            return output

        # 🧠 Accept llmMsgs OR pre-tokenized input
        if isinstance(instructions, list):
            instructions = self.tokenizer.apply_chat_template(
                instructions,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

        for attempt in range(json_retry):

            outputs = self.model.generate(
                **instructions,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.6 if json_mode else 0.7,
                top_p=0.9,
                repetition_penalty=1.15 if json_mode else 1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            generated = outputs[0][instructions["input_ids"].shape[-1]:]
            decoded = self.tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()

            # 🔒 Empty guard
            if not decoded:
                continue

            # 🔷 JSON MODE (strict)
            if json_mode:
                json_result = TextModelRunner._safe_json_extract(decoded)
                if json_result is not None:
                    try:
                        return wrap(json.loads(json_result))
                    except Exception:
                        continue

                # final attempt fallback
                if attempt == json_retry - 1:
                    return wrap({"error": "Failed to produce valid JSON"})

            # 🔶 NORMAL MODE
            else:
                if normalize_decoded:
                    json_result = TextModelRunner._safe_json_extract(decoded)
                    if json_result is not None:
                        try:
                            return wrap(json.loads(json_result))
                        except Exception:
                            pass

                    if attempt == json_retry - 1:
                        return wrap(TextModelRunner._normalize_text(decoded))

                else:
                    return wrap(decoded)

        # 🚨 Absolute fallback
        return wrap({"error": "Empty generation"} if json_mode else "...")

    # TODO : Batch optimization
    # def generate_batch(
    #     self,
    #     batch_messages: List[List[Dict[str, str]]],
    #     max_new_tokens: int = 128
    # ) -> List[str]:
    #     """
    #     Robust batch generation for chat templates.
    #     Works for single or multi-message batches, single or multi-dimensional tensors.
    #     """

    #     # Apply chat template
    #     inputs = self.tokenizer.apply_chat_template(
    #         batch_messages,
    #         padding=True,
    #         return_tensors="pt"
    #     )

    #     # Normalize: ensure a dict with at least 'input_ids'
    #     if isinstance(inputs, torch.Tensor):
    #         inputs = {"input_ids": inputs}

    #     # Move tensors to device
    #     for k, v in inputs.items():
    #         if not isinstance(v, torch.Tensor):
    #             inputs[k] = torch.tensor(v, device=self.device)
    #         else:
    #             inputs[k] = v.to(self.device)

    #     # Ensure batch dimension
    #     if inputs["input_ids"].dim() == 1:
    #         inputs["input_ids"] = inputs["input_ids"].unsqueeze(0)

    #     batch_size = inputs["input_ids"].shape[0]
    #     input_lengths = [inputs["input_ids"].shape[1]] * batch_size

    #     # Generate
    #     with torch.no_grad():
    #         outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

    #     # Decode generated tokens only
    #     decoded_outputs: List[str] = []
    #     for idx, output in enumerate(outputs):
    #         start_idx = input_lengths[idx]
    #         # SAFE slicing: works for 1D or 2D tensors
    #         gen_tokens = output[start_idx:]
    #         decoded_outputs.append(self.tokenizer.decode(gen_tokens, skip_special_tokens=True))

    #     return decoded_outputs


    # def optimize(
    #         self
    #     ):
    #     '''
    #     Self-optimization (self-reflection tick) phase sum and putting short-term
    #     memory into long-term memory if it's important enough.
    #     '''

    #     IMWEIGH = self.importance_weight

    #     # Short-Term Memory is optimized past this point
    #     short_importance_bias = IMWEIGH(200, self.mind._short)
    #     # if thresh(short_importance_bias, self.memory.short)
    #     if self.token_count(self.memory_short.messages) >= short_importance_bias:
    #         short_memory = self.MemoryBank(
    #             [
    #                 {
    #                     "role": "system",
    #                     "content": (
    #                         f"You are {self.role}. You are now {self.age} cycles old. "
    #                         f"Between 0 and 2, the importance of these memories is {self.mind._short}."
    #                         "\n"
    #                         "These are your short-term memories."
    #                         "\n"
    #                         "There are now too many of them. "
    #                         "Your objective: SUMMARIZE and EXTRACT "
    #                         "lessons, beliefs, and important events from "
    #                         "your short term memories "
    #                         "in as few words as possible, "
    #                         "and return a JSON."
    #                         "\n"
    #                         "Respond ONLY in JSON:"
    #                         "\n"
    #                         "{'summary': '...', 'keep_long_term': '...'}"
    #                         "\n"
    #                         "short-term memories to follow."
    #                     )
    #                 }
    #             ],
    #             self.memory_short.messages
    #         )

        # # Long-Term Memory is optimized past this point
        # long_importance_bias = IMWEIGH(800, self.mind._long, 1.0)

        # # Historical-Term Memory
        # historical_importance_bias = IMWEIGH(500, self.mind._long, 1.0)

        # # Beliefs-Term Memory
        # beliefs_importance_bias = IMWEIGH(200, self.mind._beliefs, 1.0)

        # # Feelings-Term Memory
        # feelings_memory = IMWEIGH(100, self.mind._short)

        # return