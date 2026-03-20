import torch
from .terminal import Name
from typing import List, Dict, Optional, Any
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
import random

@dataclass
class PromptTable:
    instruction: str

# 🧠 Analysts (Intuitive + Thinking)
# INTJ – The Architect
# INTP – The Logician
# ENTJ – The Commander
# ENTP – The Debater
# 🧭 Diplomats (Intuitive + Feeling)
# INFJ – The Advocate
# INFP – The Mediator
# ENFJ – The Protagonist
# ENFP – The Campaigner
# 🛠️ Sentinels (Observant + Judging)
# ISTJ – The Logistician
# ISFJ – The Defender
# ESTJ – The Executive
# ESFJ – The Consul
# 🎨 Explorers (Observant + Perceiving)
# ISTP – The Virtuoso
# ISFP – The Adventurer
# ESTP – The Entrepreneur
# ESFP – The Entertainer

@dataclass
class Mind:
    # Example:
    # agent.memory_weights = {
    #     "short": 2.5,   # very reactive
    #     "long": 1.0,    # forget history easily
    #     "spatial": 3.0, # strong environmental awareness
    #     "agents": 0.5   # doesn't care about others
    # }
    seed      : Any   = field(default_factory=lambda: random.randint(1, 100))
    _short    : float = 1.0
    _long     : float = 2.0
    _beliefs  : float = 2.2
    _agents   : float = 2.5
    _jobs     : float = 2.0
    variance  : float = 0.75

    # Apply jitter based on variance
    def __post_init__(self):
        mbti = [
            "INTJ", "INTP", "ENTJ", "ENTP",
            "INFJ", "INFP", "ENFJ", "ENFP",
            "ISTJ", "ISFJ", "ESTJ", "ESFJ",
            "ISTP", "ISFP", "ESTP", "ESFP"
        ]
        rng = random.Random(self.seed)
        def jitter(value: float) -> float:
            return value * rng.uniform(1 - self.variance, 1 + self.variance)
        self._short   = jitter(self._short)
        self._long    = jitter(self._long)
        self._beliefs = jitter(self._beliefs)
        self._agents  = jitter(self._agents)
        self._jobs    = jitter(self._jobs)
        self.mbti     = random.choice(mbti)

    def as_awareness_strings(self):
        return (
            f"[Assistant Importance Bias] "
            f"[short-term-memory]: {self._short}, "
            f"[long-term-memory]: {self._long}, "
            f"[beliefs]: {self._beliefs}, "
            f"[jobs]: {self._jobs}, "
            f"[mbti]: {self.mbti}"
        )

class TextModelRunner:
    def __repr__(self):
        objId = hex(id(self))
        short = '0x..'+objId[-4:]
        return f'<{short}:{Name(self)}>'
    
    _SharedModels = {}
    
    def __init__(
            self, 
            model:str="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            tokenizer:str="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            device: Optional[str] = None
        ):

        # Get individual's mental weight traits
        # for dealing with memory management
        self.mind = Mind()

        # These will have different instructions for each
        self.memory = {
            bank : []
            for bank in [
                'short',    # "Short Term Memory"
                'long',     # "Long Term Memory"
                'beliefs',
                'dialogue', # Communication "dialogue" itself
                # 'spatial',  # "Spatial Awareness" (Coordinate System)
                # NOTE : Spatial awareness will be supplied by Grid ...
                'agents',   # Relationships/thoughts of other agents
                'jobs'      # Tasking and Mission
            ]
        }

        self.mind_awareness = {"role": "system", "content": ""}
        self.instruct = {"role": "system", "content": "You are an individual in a virtual world."}

        # Get the shared model
        self.model     = TextModelRunner._SharedModels.setdefault(f'm_{model}', AutoModelForCausalLM.from_pretrained(model))
        self.tokenizer = TextModelRunner._SharedModels.setdefault(f't_{tokenizer}', AutoTokenizer.from_pretrained(model))

        # Initialize the device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def token_count(self, messages: List[Dict[str, str]]) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True))

    
    def triage(
            self,
            messages: List[Dict[str, str]],
            instruct: Optional[str] = None,
            max_new_tokens: int = 40 
        ):

        instruct = instruct or self.instruct

        all_instructions = [
            self.instruct,
            *messages
        ]
        # TODO : Apply weight based summarization and weight based trimming...

        # while self.token_count(all_instructions) > 2048:
        #     messages = messages[:1]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        # decoded ...

        # print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
        # print(outputs)

        # messages: list
        # emotions: list[str]
        # beliefs: list[str]



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


class AgentAI:



    def __init__(self):
        pass