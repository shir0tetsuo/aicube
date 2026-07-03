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
            return len(self.messages) == 0
        
        @property
        def last(self) -> Dict[str, str]:
            return self.messages[-1] if self.messages else {}

    @property
    def role(self):
        short = self.__objId[-4:]
        return f'<{self.societal_role}:{short}>'
    
    @staticmethod
    def jitter(rng: random.Random, value: float, variance: float):
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
        self.rng = random.Random(self.seed)
        self.variance = 0.75
        
        self.__objId = hex(id(self))  # Hex ID of this object

        # based on the NEED of the environment
        # and maybe later the personality
        # in place of ASSISTANT, USER, and SYSTEM ...
        self.societal_role = societal_role

        # NOTE : Maybe we make this adjustable reward?
        # based on how they feel about things
        self.mbti = self.rng.choice(self.MBTI)

        jitter = lambda v: self.jitter(self.rng, v, self.variance)

        self.personality = AutoAgent.MemoryBank(
            instruct=[{
                'role': 'system',
                'content': (
                    'You are a virtual agent in a virtual environment.\n\n'
                    f'{personality}\n\n'
                    f'Your ROLE:\n{self.role}\nKnown as: {self.role}\n'
                    f'{societal_role_description}\n\n'
                )
            }],
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

    # ---------------- TOKEN UTILS ----------------

    def _ensure_token_cache(self, msg):
        if "_tokens" not in msg:
            msg["_tokens"] = self.token_count([msg])
        return msg["_tokens"]

    def _ensure_token_cache_many(self, messages):
        for m in messages:
            self._ensure_token_cache(m)

    def _bank_token_count(self, bank):
        self._ensure_token_cache_many(bank.messages)
        return sum(m["_tokens"] for m in bank.messages) + self.token_count(bank.instruct)

    # ---------------- TOKEN ALLOCATION ----------------

    @staticmethod
    def tokenalloc(banks, total_tokens=2048, reserve_tokens=256, min_tokens=32):
        available = max(0, total_tokens - reserve_tokens)
        active = [b for b in banks if (b.messages or b.instruct)]

        if not active:
            return

        total_weight = sum(b.weight for b in active)

        if total_weight == 0:
            equal = available // len(active)
            for b in active:
                b.max_tokens = equal
            return

        for b in active:
            tokens = int((b.weight / total_weight) * available)
            b.max_tokens = max(min_tokens, tokens)

        total_alloc = sum(b.max_tokens for b in active)
        if total_alloc > available:
            scale = available / total_alloc
            for b in active:
                b.max_tokens = int(b.max_tokens * scale)

    # ---------------- MICRO SUMMARIZER ----------------

    def _summarize_to_one_sentence(self, content: str) -> str:
        prompt = [
            {"role": "system", "content": (
                "Summarize into EXACTLY one short sentence (max 20 words). "
                "Preserve key facts. No explanation."
            )},
            {"role": "user", "content": content}
        ]

        result = self.think(prompt, normalize_decoded=True)

        if isinstance(result, list):
            result = result[0].get("content", "")

        if not isinstance(result, str):
            return ""

        return result.strip()

    # ---------------- PROMOTION ----------------

    def _promote_before_trim(self, source, target, promote_ratio=0.25):
        if not source.messages:
            return

        self._ensure_token_cache_many(source.messages)

        scored = []
        for i, msg in enumerate(source.messages):
            recency = (i + 1) / len(source.messages)
            length = msg["_tokens"]
            score = recency * 0.7 + math.log1p(length) * 0.3
            scored.append((score, msg))

        scored.sort(reverse=True, key=lambda x: x[0])
        k = max(1, int(len(scored) * promote_ratio))
        promoted = [msg for _, msg in scored[:k]]

        existing = set(m["content"] for m in target.messages)

        for msg in promoted:
            summary = self._summarize_to_one_sentence(msg["content"])
            if not summary or summary in existing:
                continue

            new_msg = {"role": self.role, "content": summary}
            self._ensure_token_cache(new_msg)
            target.messages.append(new_msg)
            existing.add(summary)

        source.messages = [m for m in source.messages if m not in promoted]

    # ---------------- TRIM ----------------

    def hybrid_trim_tokens(self, bank, keep_recent=10):
        messages = bank.messages
        if not messages:
            return messages

        self._ensure_token_cache_many(messages)

        total = sum(m["_tokens"] for m in messages)
        if total <= bank.max_tokens:
            return messages

        indexed = list(enumerate(messages))
        recent = indexed[-keep_recent:]
        remaining = indexed[:-keep_recent]

        selected = []
        current = 0

        # Keep recent (from newest backwards)
        for idx, msg in reversed(recent):
            t = msg["_tokens"]
            if current + t <= bank.max_tokens:
                selected.append((idx, msg))
                current += t

        selected.reverse()

        self.rng.shuffle(remaining)

        for idx, msg in remaining:
            t = msg["_tokens"]
            if current + t <= bank.max_tokens:
                selected.append((idx, msg))
                current += t
            if current >= bank.max_tokens:
                break

        selected.sort(key=lambda x: x[0])
        return [m for _, m in selected]

    # ---------------- TRIM OPT ----------------

    def _trim_optimization(self, bank, longterm_bank, desc="memories"):
        self._ensure_token_cache_many(bank.messages)
        tokens = sum(m["_tokens"] for m in bank.messages)

        if tokens < bank.max_tokens:
            return

        bank.instruct = [{
            "role": "system",
            "content": (
                f"You are {self.role}. These {desc} are too large.\n"
                "Summarize and extract key facts.\n"
                "Respond JSON:\n"
                '{"summary":"","keep_long_term":"","feelings":""}'
            )
        }]

        result = self.think(
            self.instructions(bank.instruct + bank.messages),
            normalize_decoded=True,
            json_mode=True
        )

        if not isinstance(result, dict):
            self._promote_before_trim(bank, longterm_bank)
            bank.messages = self.hybrid_trim_tokens(bank)
            return

        summary = result.get("summary")
        longterm = result.get("keep_long_term")
        feelings = result.get("feelings")

        if summary:
            new_msg = {"role": self.role, "content": summary}
            self._ensure_token_cache(new_msg)
            bank.messages = [new_msg]

        if longterm:
            msg = {"role": self.role, "content": longterm}
            self._ensure_token_cache(msg)
            longterm_bank.messages.append(msg)

        if feelings:
            msg = {"role": self.role, "content": feelings}
            self._ensure_token_cache(msg)
            self.memory_feelings.messages.append(msg)

    # ---------------- UPDATE ----------------

    def update(self, spatial, instruct=None, dialogue=None):

        if dialogue is None:
            dialogue = []

        TOKENCOUNT = self._bank_token_count

        finite = {
            'spatial': spatial,
            'identity': self.personality,
            'ailments': self.memory_ailments
        }

        finite_tokens = sum(TOKENCOUNT(b) for b in finite.values())

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
            list(weighed.values()),
            reserve_tokens=finite_tokens + instruct_tokens + 128
        )

        for k, b in weighed.items():
            self._trim_optimization(b, self.memory_long, k)

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

        return self.think(
            tokens,
            normalize_decoded=bool(dialogue),
            json_mode=not dialogue,
            wrap_role=self.role if dialogue else None
        )