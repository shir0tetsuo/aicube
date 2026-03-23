from .terminal import Name
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from .llm import TextModelRunner
import random
import json
from .terminal import cprint
import math

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

            seed: Optional[Any] = None,

            # PRETEXTS
            societal_role: Optional[str] = None,
            societal_role_description: Optional[str] = None,
            personality: Optional[str] = None,

        ):
        # Initialize LLM
        super().__init__(model, tokenizer, device)

        # Randomness
        self.seed = seed if seed is not None else random.randint(1, 100)
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
                        # f'{self.mind.as_awareness_strings()}\n\n'
                        f'{personality}\n\n'
                        f'Your ROLE in this virtual environment:\n'
                        f'{self.role}\nYou will thus be known as: {self.role}\n'
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
        # self.memory_spatial       =AutoAgent.MemoryBank()                     # Direct spatial information from Grid (Does it align with job task?)
        self.memory_relationships =AutoAgent.MemoryBank( weight=jitter(2.5) ) # Relationships to Other Agents
        self.memory_jobs          =AutoAgent.MemoryBank( weight=jitter(2.0) ) # Tasking and Mission

        # cycles age old
        self.age = 0

        # self-reflection average period
        # (and also re-evaluate this value)
        self.reflection_tick = 50

    @staticmethod
    def tokenalloc(
            banks: List[MemoryBank],
            total_tokens: int = 2048,
            reserve_tokens: int = 256,  # Maybe get the token count of incoming instructs
            min_tokens: int = 32
        ) -> None:
        """
        Mutates `MemoryBank.max_tokens` based on weight distribution.
        """

        available_tokens = max(0, total_tokens - reserve_tokens)

        active_banks = [b for b in banks if (b.messages or b.instruct)]

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

    def _trim_optimization(self, bank:MemoryBank, longterm_bank:MemoryBank, bank_description:str='memories'):
        '''Squeeze current memory banks into a sum
        with JSON instructions for what should be
        passed into long-term memory or historic-term;
        Interpret any feelings, etc.'''

        tokens = self.token_count(bank.messages)
        if tokens < bank.max_tokens:
            return

        bank_importance = math.ceil(bank.weight)+1

        bank.instruct = [
            {
                "role": "system",
                "content": (
                    f'You are {self.role}. You are now {self.age} cycles old. '
                    f"Between 0 and {bank_importance}, the importance of these memories "
                    f"is {bank.weight}.\n\n"
                    f"These are your {bank_description}. There are now too many of them.\n\n"
                    "Your objective: SUMMARIZE in as few words as possible and EXTRACT "
                    f"lessons, beliefs if any, and important events if any from your {bank_description}.\n\n"
                    "For 'feelings', quickly summarize such as:\n"
                    f"{self.role} was feeling ... about ...\n\n"
                    # f"Your token limit: {bank.max_tokens}, try to keep it under this.\n\n"
                    # "Respond ONLY in JSON:\n\n"
                    # "{'summary': '...', 'keep_long_term': '...', 'feelings': '...'}"
                    "Respond ONLY in valid JSON with this exact schema:\n\n"
                    "{\n"
                    '"summary": "short compressed memory",\n'
                    '"keep_long_term": "critical facts only",\n'
                    '"feelings": "emotional state summary"\n'
                    "}"
                )
            }
        ]

        reorganized_thoughts = self.think(
            self.instructions(bank.instruct + bank.messages), 
            normalize_decoded=True, 
            # max_new_tokens=bank.max_tokens, 
            json_mode=True
        )
        if not isinstance(reorganized_thoughts, dict) or "error" in reorganized_thoughts:
            cprint('ERROR: AutoAgent did not respond with JSON.', fg='#ffffff', bg='#ff0000')
            while self.token_count(bank.messages) > bank.max_tokens:
                bank.messages.pop(0)
        else:
            summary = reorganized_thoughts.get('summary')
            longterm = reorganized_thoughts.get('keep_long_term')
            feelings = reorganized_thoughts.get('feelings')
            bank.messages = [{"role": self.role, "content": summary}]
            longterm_bank.messages.append({"role": self.role, "content": longterm})
            self.memory_feelings.messages.append({"role": self.role, "content": feelings})

        return
    
    def _bank_token_count(self, bank: MemoryBank) -> int:
        return self.token_count(bank.instruct + bank.messages)

    # TODO
    # [✔] Memory system
    # [✔] Token attention system
    # [✔] Identity anchoring
    # [✔] LLM inference wrapper
    # [✔] Context builder
    # [ ] Memory write-back
    # [ ] Reflection loop  ← NEXT
    # [ ] Multi-agent interaction
    def update(self, spatial:MemoryBank, instruct: Optional[llmMsgs] = None, dialogue:llmMsgs=[]):

        # Spatial Awareness => Incoming

        # SPATIAL : - Others around.
        #           - Neighboring tiles.
        #           - Initiated actions/activities, extra data.
        #           - Held Items

        TOKENCOUNT = self._bank_token_count

        finite = {
            'spatial': spatial,
            'identity': self.personality,
            'ailments': self.memory_ailments
        }

        finite_tokens = sum([TOKENCOUNT(bank) for bank in finite.values()])

        weighed = {
            'feels': self.memory_feelings,
            'short': self.memory_short,
            'long': self.memory_long,
            'hist': self.memory_historic,
            'beliefs': self.memory_beliefs,
            'dialogue': self.memory_dialogue,
            'relationships': self.memory_relationships,
            'jobs': self.memory_jobs
        }

        if dialogue:
            weighed['dialogue'].messages.extend(dialogue)


        instruct = instruct or [
            {
                "role": "system", 
                "content": (
                    f"You are {self.role}. Continue the conversation."
                ) if dialogue else (
                    f"You are {self.role}. You must decide what to do next. "
                    
                    # Continue Job
                    "Figure out what your task is. If you are currently performing a job, "
                    'let your "action" = "CONTINUE"\n\n'

                    # Continue Conversation
                    "If you wish to continue a conversation, "
                    'let your "action" = "CONVERSE"\n\n'

                    # Movement
                    "If you wish to navigate in the world, "
                    'let your "action" = "MOVE (DIRECTION)"'
                    "and replace (DIRECTION) with UP, DOWN, LEFT, or RIGHT.\n\n"

                    # Idle (increase time until next tick)
                    "If you wish to randomly explore or let time pass by, "
                    'let your "action" = "IDLE"\n\n'

                    "Respond ONLY in valid JSON with this exact schema:\n\n"
                    "{\n"
                    '"action": "next action",\n'
                    # '"short_term_memory": "",\n'
                    '"thoughts": "personal thoughts",\n' # to be split into short term
                    '"feelings": "emotional state summary",\n'
                    '"say": "said aloud"'
                    "}\n\n"
                    
                )
            }
        ]

        instruct_tokens = self.token_count(instruct)

        self.tokenalloc(
            [bank for bank in weighed.values()],
            reserve_tokens=finite_tokens + instruct_tokens + 128,
            min_tokens=32
        )

        # Optimization Layer
        self._trim_optimization(weighed['feels'], weighed['long'], 'feelings')
        self._trim_optimization(weighed['short'], weighed['long'], 'short term memories')
        self._trim_optimization(weighed['long'], weighed['hist'], 'long term memories')
        self._trim_optimization(weighed['beliefs'], weighed['beliefs'], 'beliefs')
        self._trim_optimization(weighed['dialogue'], weighed['long'], 'memories of recent conversation')
        self._trim_optimization(weighed['relationships'], weighed['long'], 'relationship memories')
        self._trim_optimization(weighed['jobs'], weighed['jobs'], 'memories of jobs')
        

        brain = self.MemoryBank(
            instruct=instruct,
            messages=[
                *finite['identity'].instruct,
                *weighed['hist'].messages,
                *weighed['beliefs'].messages,
                *weighed['jobs'].messages,
                *finite['spatial'].instruct,
                *finite['spatial'].messages,
                *finite['ailments'].messages,
                *weighed['feels'].messages,
                *weighed['short'].messages,
                *weighed['long'].messages,
                *weighed['relationships'].messages,
                *weighed['dialogue'].messages
            ]
        )
 
        tokens = self.instructions(brain.instruct + brain.messages)
        result = self.think(
            tokens, 
            normalize_decoded = True if dialogue else False,
            json_mode = False if dialogue else True,
            wrap_role = self.role if dialogue else None
        )
        
        return result