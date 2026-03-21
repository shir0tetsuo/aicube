import torch
from .terminal import Name
from typing import List, Dict, Optional, Tuple, Any
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass, field
from .voices import TTS
import random
import json

llmMsgs = List[Dict[str, str]]

@dataclass
class PromptTable:
    instruction: str


@dataclass
class HeldItem:
    name: str
    spatial_weight: float
    description: str
    pointers: List[Tuple[str, int]]


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
            # 🧠 Analysts (Intuitive + Thinking)
            "INTJ", # – The Architect
            "INTP", # – The Logician
            "ENTJ", # – The Commander
            "ENTP", # – The Debater
            # 🧭 Diplomats (Intuitive + Feeling)
            "INFJ", # – The Advocate
            "INFP", # – The Mediator
            "ENFJ", # – The Protagonist
            "ENFP", # – The Campaigner
            # 🛠️ Sentinels (Observant + Judging)
            "ISTJ", # – The Logistician
            "ISFJ", # – The Defender
            "ESTJ", # – The Executive
            "ESFJ", # – The Consul
            # 🎨 Explorers (Observant + Perceiving)
            "ISTP", # – The Virtuoso
            "ISFP", # – The Adventurer
            "ESTP", # – The Entrepreneur
            "ESFP"  # – The Entertainer
        ]
        rng = random.Random(self.seed)
        def jitter(value: float) -> float:
            return value * rng.uniform(1 - self.variance, 1 + self.variance)
        self._short   = jitter(self._short)
        self._long    = jitter(self._long)
        self._beliefs = jitter(self._beliefs)
        self._agents  = jitter(self._agents)
        self._jobs    = jitter(self._jobs)
        self.mbti     = rng.choice(mbti)

    def as_awareness_strings(self):
        return (
            f"[Assistant Importance Bias] "
            f"short_term_memory={self._short}, "
            f"long_term_memory={self._long}, "
            f"beliefs={self._beliefs}, "
            f"jobs={self._jobs}"
            # f"[mbti]: {self.mbti}"
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

        # Get the shared model
        self.model     = TextModelRunner._SharedModels.setdefault(f'm_{model}', AutoModelForCausalLM.from_pretrained(model))
        self.tokenizer = TextModelRunner._SharedModels.setdefault(f't_{tokenizer}', AutoTokenizer.from_pretrained(model))

        # Initialize the device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def token_count(self, messages: llmMsgs) -> int:
        return len(self.tokenizer.apply_chat_template(messages, tokenize=True))
    
    def instructions(self, messages: llmMsgs):
        '''
        Provides tokenized inputs for `self.model`
        
        ---
        >>> inputs = self.instructions(all_instructions)
            outputs = self.model.generate(**inputs, ...)
        '''
        tokens = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        return tokens
    
    def think(
            self, 
            instructions, 
            max_new_tokens:Optional[int] = None
        ) -> llmMsgs:
        outputs = self.model.generate(**instructions, max_new_tokens=max_new_tokens)
        generated = outputs[0][instructions["input_ids"].shape[-1]:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    # def triage(
    #         self,
    #         messages: List[Dict[str, str]],
    #         instruct: Optional[str] = None,
    #         max_new_tokens: int = 40 
    #     ):

    #     mind = self.mind.as_awareness_strings()
    #     instruct = instruct or self.instruct

        # task ->
        # observe environment -> 
        # think ->
        # feel ->
        # think about what to say ->
        # speak

        # all_instructions = [
        #     self.instruct,
        #     {"role": "system", "content": mind},
        #     *messages
        # ]

        # while self.token_count(all_instructions) > 2048:
        #     messages = messages[:1]


        # inputs = self.instructions(all_instructions)
        # outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

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


class AutoAgent(TextModelRunner):

    # TODO : Triggers

    @dataclass
    class MemoryBank:
        instruct: llmMsgs = field(default_factory=list)
        messages: llmMsgs = field(default_factory=list)

    @property
    def role(self):
        objId = self.__objId
        short = objId[-4:]
        return f'{self.societal_role}_{short}'

    def __init__(
            self, 

            # TextModelRunner
            model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            device = None

        ):
        super().__init__(model, tokenizer, device)

        self.__objId = hex(id(self))

        # Get individual's mental weight traits
        # for dealing with memory management
        # NOTE : Maybe we make this adjustable reward
        # based on how they feel about things
        self.mind = Mind()
        self.mbti = self.mind.mbti
        
        # Personality -> MBTI + Mind as weights? ->
        # "I am __role__! I ..."
        self.personality = {"role": "system", "content": ""}

        self.societal_role = self.think(self.instructions())

        # cycles age old
        self.age = 0

        self.init_memories()

        # self-reflection average period
        # (and also re-evaluate this value)
        self.reflection_tick = 50

        # These will have different instructions for each
        # self.memory = {
        #     bank : AutoAgent.MemoryBank(self.societal_role)
        #     for bank in [
        #         'ailments', # "Health Conditions"
        #         'short',    # "Short Term Memory"
        #         'long',     # "Long Term Memory"
        #         'history',  # "Long-Long Term Memory, Foundational"
        #         'beliefs',
        #         'feelings', # Agent felt <feeling> after <...> (SMALL BANK)
        #         'dialogue', # Communication "dialogue" itself
        #         # 'spatial',  # "Spatial Awareness" (Coordinate System)
        #         # NOTE : Spatial awareness will be supplied by Grid ...
        #         'agents',   # Relationships/thoughts of other agents
        #         'jobs'      # Tasking and Mission
        #     ]
        # }

        

        self.mind_awareness = {"role": "system", "content": ""}
        self.instruct = {"role": "system", "content": "You are an individual in a virtual world."}

    def init_memories(self):
        self.memory_short = AutoAgent.MemoryBank()
        return

    @staticmethod
    def importance_weight(tokens:int, importance_bias:float, multiplier:float=1.25):
        return importance_bias * multiplier * tokens

    def optimize(
            self
        ):
        '''
        Self-optimization (self-reflection tick) phase sum and putting short-term
        memory into long-term memory if it's important enough.
        '''

        IMWEIGH = self.importance_weight

        # Short-Term Memory is optimized past this point
        short_importance_bias = IMWEIGH(200, self.mind._short)
        # if thresh(short_importance_bias, self.memory.short)
        if self.token_count(self.memory_short.messages) >= short_importance_bias:
            short_memory = self.MemoryBank(
                [
                    {
                        "role": "system",
                        "content": (
                            f"You are {self.role}. You are now {self.age} cycles old."
                            "\n"
                            "These are your short-term memories."
                            "\n"
                            "There are now too many of them. "
                            "Your objective: SUMMARIZE and EXTRACT "
                            "lessons, beliefs, and important events from "
                            "your short term memories "
                            "in as few words as possible, "
                            "and return a JSON."
                            "\n"
                            "Respond ONLY in JSON:"
                            "\n"
                            "{'summary': '...', 'keep_long_term': '...'}"
                            "\n"
                            "short-term memories to follow."
                        )
                    }
                ],
                self.memory_short.messages
            )

        # Long-Term Memory is optimized past this point
        long_importance_bias = IMWEIGH(800, self.mind._long, 1.0)

        # Historical-Term Memory
        historical_importance_bias = IMWEIGH(300, self.mind._long, 1.0)

        # Beliefs-Term Memory
        beliefs_importance_bias = IMWEIGH(200, self.mind._beliefs, 1.0)

        # Feelings-Term Memory
        feelings_memory = IMWEIGH(100, self.mind._short)

        return
    
    # NOTE : TEST !
    def build_context(self, incoming):

        weighted = []

        def sample(bank: AutoAgent.MemoryBank, weight: float):
            if not bank.messages:
                return []
            k = max(1, int(len(bank.messages) * weight * 0.2))
            return bank.messages[-k:]

        weighted += sample(self.memory_short, self.mind._short)

        return [
            self.instruct,
            {"role": "system", "content": self.mind.as_awareness_strings()},
            *weighted,
            *incoming
        ]

    def triage(
            self,
            messages:List[Dict[str, str]],
            instruct: Optional[str] = None,
            max_new_tokens: int = 40
        ):
        # TODO : Apply weight based summarization and weight based trimming...

        all_instructions = [

        ]

        # tasks -> observe environment, self-state
        # think -> feel, tone -> think about what to say ->
        # speak -> action

        return