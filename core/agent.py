from .terminal import Name
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from .llm import TextModelRunner
import random
import json

llmMsgs = List[Dict[str, str]]


class AutoAgent(TextModelRunner):

    MBTI = [
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

    # TODO : Triggers
    @dataclass
    class MemoryBank:
        '''
        Agent memory bank ensures messages can be pruned
        while persisting the original prompt instructions.

        :param instruct: The original instruction prompt for
            the memory set.
        :type instruct: `llmMsgs` (`List[Dict[str, str]]`)
        :param messages: The LLM message chain.
        :type messages: `llmMsgs` (`List[Dict[str, str]]`)
        
        ---
        Type `llmMsgs` (`List[Dict[str, str]]`) is a collection
            of LLM messages in a format such as:

        >>> {
                "role": "system", 
                "content": "Your role is now ..., This is a virtual environment..."
            }
        
        ---
        >>> m = MemoryBank()
            bool(m.empty) # True
        
        >>> m.last # {...}
        '''
        instruct: llmMsgs = field(default_factory=list)
        messages: llmMsgs = field(default_factory=list)
        weight: float = field(default=1.0)
        max_tokens: int = field(default=100)

        @property
        def empty(self):
            return True if (len(self.messages)==0) else False
        
        @property
        def last(self) -> Dict[str, str]:
            return self.messages[-1] if self.messages else {}

    @property
    def role(self):
        objId = self.__objId
        short = objId[-4:]
        return f'<{self.societal_role}:{short}>'
    
    @staticmethod
    def jitter(rng:random.Random, value: float, variance: float):
        return value * rng.uniform(1 - variance, 1 + variance)

    def __init__(
            self, 

            # TextModelRunner
            model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            tokenizer = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            device = None,

            seed: int = field(default_factory=lambda: random.randint(1, 100)),

            # PRETEXTS
            societal_role: Optional[str] = None,
            societal_role_description: Optional[str] = None,
            personality: Optional[str] = None,

        ):
        # Initialize LLM
        super().__init__(model, tokenizer, device)

        # Randomness
        self.seed = seed
        self.rng  = random.Random(self.seed)
        self.variance = 0.75
        
        self.__objId = hex(id(self))  # Hex ID of this object

        # based on the NEED of the environment
        # and maybe later the personality
        # in place of ASSISTANT, USER, and SYSTEM ...
        self.societal_role = societal_role

        # NOTE : Maybe we make this adjustable reward?
        # based on how they feel about things
        self.mbti = self.rng.choice(self.MBTI)

        jitter = lambda value: (self.jitter(self.rng, value, self.variance))
        self.personality = AutoAgent.MemoryBank(
            instruct=[
                {
                    'role': 'system', 'content': (
                        'You are a virtual agent in a virtual environment.\n\n'
                        f'{self.mind.as_awareness_strings()}\n\n'
                        f'{personality}\n\n'
                        f'Your ROLE in this virtual environment:\n'
                        f'{self.role}\nYou will then be known as: {self.role}\n'
                        f'{societal_role_description}\n\n'
                    )
                }
            ],
            weight=jitter(2.0)
        )
        # -- NOTE
        # ALL MEMORY (huge, persistent)
        #             ↓
        #     build_context()
        #             ↓
        #   WORKING MEMORY (limited, focused)
        #             ↓
        #        LLM inference
        #             ↓
        #       NEW THOUGHTS
        #             ↓
        #       stored back into memory

        # IDENTITY → BIASES → EXPERIENCE → RECENT → SOCIAL → GOALS → INPUT
        
        # -> These are funneled to individual processes
        #    and summarized to produce responses/TTS
        # Such as:
        # -- X MemoryBank at 1024 Tokens ->
        #    "Summarize. You are allotted X tokens."
        # -> Sentences get cleaned, removing unused parts
        #    "Which of these elements would you like to
        #    instruct yourself to respond with?" & biases.

        #     # tasks -> observe environment, self-state
        #     # think -> feel, tone -> think about what to say ->
        #     # speak -> action
        # -- NOTE
        self.memory_feelings      =AutoAgent.MemoryBank()                     # <ID> was feeling <...> about <...>
        self.memory_short         =AutoAgent.MemoryBank( weight=jitter(1.0) ) # Short Term Memory
        self.memory_long          =AutoAgent.MemoryBank( weight=jitter(2.0) ) # Long Term Memory
        self.memory_historic      =AutoAgent.MemoryBank( weight=jitter(2.1) ) # Historic Term Memory
        self.memory_ailments      =AutoAgent.MemoryBank()                     # "Disability", Ailment
        self.memory_beliefs       =AutoAgent.MemoryBank( weight=jitter(2.0) ) # "Belief" System (Few words, re-evaluated)
        self.memory_dialogue      =AutoAgent.MemoryBank( weight=jitter(1.5) ) # Dialogues between Agents/User, temporary
        self.memory_spatial       =AutoAgent.MemoryBank()                     # Direct spatial information from Grid (Does it align with job task?)
        self.memory_relationships =AutoAgent.MemoryBank( weight=jitter(2.5) ) # Relationships to Other Agents
        self.memory_jobs          =AutoAgent.MemoryBank( weight=jitter(2.0) ) # Tasking and Mission

        # cycles age old
        self.age = 0

        # self-reflection average period
        # (and also re-evaluate this value)
        self.reflection_tick = 50

    def tokenalloc(
            banks: List[MemoryBank],
            total_tokens: int = 2048,
            reserve_tokens: int = 256,
            min_tokens: int = 32
        ) -> None:
        """
        Mutates MemoryBank.max_tokens based on weight distribution.
        """

        available_tokens = max(0, total_tokens - reserve_tokens)

        active_banks = [b for b in banks if not b.empty]

        if not active_banks:
            return

        total_weight = sum(b.weight for b in active_banks)

        # Fallback: equal distribution
        if total_weight == 0:
            equal = available_tokens // len(active_banks)
            for b in active_banks:
                b.max_tokens = equal
            return

        # First pass: proportional allocation
        for b in active_banks:
            proportion = b.weight / total_weight
            tokens = int(proportion * available_tokens)
            b.max_tokens = max(min_tokens, tokens)

        # Optional: normalize if we overshot due to min_tokens
        total_allocated = sum(b.max_tokens for b in active_banks)

        if total_allocated > available_tokens:
            scale = available_tokens / total_allocated
            for b in active_banks:
                b.max_tokens = int(b.max_tokens * scale)
   