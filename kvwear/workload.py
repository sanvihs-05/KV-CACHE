"""Synthetic but realistic multi-turn chat workload.

Design choices (they matter for a fair experiment):

* Every policy sees the *identical* token trace. Assistant turns in the history are
  scripted text from the generator, not the model's own output. The model still decodes
  `gen_tokens` per turn (that is what we time), but the decoded tokens are thrown away.
  This keeps the trace independent of the policy, even when quantized KV slightly changes
  the model's output. Quantization's effect on quality is measured separately (quality.py).
* Conversation kinds:
    - doc_qa : user pastes one of a few *shared* documents, then asks 3-7 questions
               (the "10-page document, 5 questions" case; shared docs give cross-user reuse)
    - debug  : user pastes a one-off error log in some turn, then a few follow-ups
    - general: short chit-chat, often abandoned after 1-2 turns
* `concurrency` conversations are open at once and their turns interleave randomly,
  so a conversation's KV has to survive several other requests before it is reused.
  That pressure is what pushes KV out of RAM and onto the SSD.
* A prompt longer than `max_context_tokens` drops its oldest turns, which invalidates
  every later block (new prefix -> new hashes). That is a real source of never-reused KV.

Where never-reused KV comes from here: the last turn of every conversation, abandoned
conversations, and blocks invalidated by history truncation.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Protocol

from .config import WorkloadConfig

# ---------------------------------------------------------------- tokenizers


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...


class FakeTokenizer:
    """One token per whitespace word, stable ids. For tests and the `sim` engine."""

    vocab_size = 50_000

    def encode(self, text: str) -> list[int]:
        import zlib
        return [10 + zlib.crc32(w.encode()) % (self.vocab_size - 10) for w in text.split()]


class HFTokenizer:
    def __init__(self, name: str):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)

    def encode(self, text: str) -> list[int]:
        return self.tok(text, add_special_tokens=False)["input_ids"]


# ---------------------------------------------------------------- text generators

_COMMON = (
    "the a of to and in is that for it as with on be by this are from at or an which "
    "can will we not have has more also when each other into used than its two these "
    "data system model value user time file memory process example result method case "
    "first new number then should could would because however between through while "
    "performance cache request server client storage write read update table page block "
    "layer network latency throughput error config version release build test module "
    "function class object field index query response thread queue buffer device driver"
).split()
_TECH = (
    "allocation scheduler partition replica checkpoint gradient tensor kernel batch "
    "embedding attention decoder encoder token sequence quantization compression "
    "eviction admission policy threshold heuristic workload benchmark baseline variance"
).split()
_LOG_CLASSES = ["com.app.Service", "org.db.Pool", "io.net.Client", "app.worker.Task", "core.cache.Store"]
_LOG_ERRORS = ["NullPointerException", "TimeoutError", "ConnectionResetError", "KeyError",
               "OutOfMemoryError", "IllegalStateException", "PermissionDenied"]
_QUESTIONS = [
    "Can you summarize section {n} in simple words?",
    "What does the part about {w} mean?",
    "How does {w} relate to {w2} in this?",
    "Is there a problem with the {w} step?",
    "Give me an example of {w}.",
    "Why is {w} important here?",
    "What should I change to make {w} faster?",
    "Thanks. And what about {w2}?",
]

SYSTEM_PROMPT = (
    "You are a helpful, careful assistant running locally on the user's laptop. "
    "Answer clearly and concisely. When the user shares a document or a log, refer to "
    "specific parts of it. If you are not sure, say so. Use short paragraphs and "
    "bullet points where it helps. Never invent facts that are not in the provided text. "
)


def _words(rng: random.Random, n: int) -> str:
    out = []
    for i in range(n):
        out.append(rng.choice(_TECH) if rng.random() < 0.15 else rng.choice(_COMMON))
        if i % 14 == 13:
            out[-1] += "."
    return " ".join(out)


def _log_text(rng: random.Random, n_lines: int) -> str:
    lines = []
    for i in range(n_lines):
        cls = rng.choice(_LOG_CLASSES)
        lines.append(
            f"2026-10-{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:"
            f"{rng.randint(0, 59):02d} ERROR [worker-{rng.randint(1, 16)}] {rng.choice(_LOG_ERRORS)} "
            f"at {cls}.handle({cls.split('.')[-1]}.java:{rng.randint(10, 900)})"
        )
    return "\n".join(lines)


def _fit(tok: Tokenizer, make_text, target: int) -> list[int]:
    """Generate text until it tokenizes to >= target tokens, then cut to exactly target."""
    n = max(4, target)
    while True:
        ids = tok.encode(make_text(n))
        if len(ids) >= target:
            return ids[:target]
        n = int(n * 1.5) + 4


# ---------------------------------------------------------------- data model


@dataclass
class Request:
    index: int
    conv_id: int
    turn: int                 # 1-based
    kind: str
    has_doc: bool
    prompt_ids: list[int]
    gen_tokens: int
    is_last_turn: bool        # ground truth: hidden from real policies, used by oracle/analysis


@dataclass
class Conversation:
    conv_id: int
    kind: str
    num_turns: int
    doc_id: int | None
    fixed_ids: list[int]                       # system (+ shared doc)
    user_ids: list[list[int]] = field(default_factory=list)
    reply_ids: list[list[int]] = field(default_factory=list)


@dataclass
class Workload:
    requests: list[Request]
    conversations: list[Conversation]

    @property
    def prompt_tokens(self) -> int:
        return sum(len(r.prompt_ids) for r in self.requests)

    @property
    def gen_tokens(self) -> int:
        return sum(r.gen_tokens for r in self.requests)


def _pick_kind(rng: random.Random, mix: dict) -> str:
    kinds, weights = zip(*mix.items())
    return rng.choices(kinds, weights=weights, k=1)[0]


def generate_workload(cfg: WorkloadConfig, tok: Tokenizer) -> Workload:
    rng = random.Random(cfg.seed)
    text_rng = random.Random(cfg.seed * 7919 + 1)

    sys_ids = _fit(tok, lambda n: (SYSTEM_PROMPT * (n // 60 + 1)), cfg.system_prompt_tokens)
    docs = []
    for d in range(cfg.num_shared_docs):
        drng = random.Random(10_000 + d)   # documents are the same across seeds/policies
        target = drng.randint(*cfg.doc_tokens)
        docs.append(_fit(tok, lambda n, r=drng: "Document:\n" + _words(r, n), target))

    user_tag = tok.encode("\nUser:")
    asst_tag = tok.encode("\nAssistant:")

    convs: list[Conversation] = []
    for c in range(cfg.num_conversations):
        kind = _pick_kind(rng, cfg.mix)
        lo, hi = cfg.turns[kind]
        n_turns = rng.randint(lo, hi)
        doc_id = rng.randrange(cfg.num_shared_docs) if kind == "doc_qa" and docs else None
        fixed = list(sys_ids) + (list(docs[doc_id]) if doc_id is not None else [])
        conv = Conversation(c, kind, n_turns, doc_id, fixed)
        log_turn = rng.randint(1, n_turns) if kind == "debug" else -1
        for t in range(1, n_turns + 1):
            q = rng.choice(_QUESTIONS).format(n=rng.randint(1, 9), w=rng.choice(_TECH), w2=rng.choice(_COMMON))
            u_len = rng.randint(*cfg.user_tokens)
            u = user_tag + _fit(tok, lambda n: q + " " + _words(text_rng, n), u_len)
            if t == log_turn:
                l_len = rng.randint(*cfg.log_tokens)
                u = u + _fit(tok, lambda n: "Here is the error log:\n" + _log_text(text_rng, n // 20 + 1), l_len)
            r_len = rng.randint(*cfg.reply_tokens)
            a = asst_tag + _fit(tok, lambda n: _words(text_rng, n), r_len)
            conv.user_ids.append(u)
            conv.reply_ids.append(a)
        convs.append(conv)

    # interleave turns of `concurrency` open conversations
    requests: list[Request] = []
    pending = list(range(len(convs)))
    active: list[int] = []
    next_turn = {c.conv_id: 1 for c in convs}
    while pending or active:
        while pending and len(active) < cfg.concurrency:
            active.append(pending.pop(0))
        cid = rng.choice(active)
        conv = convs[cid]
        t = next_turn[cid]
        prompt = _build_prompt(conv, t, asst_tag, cfg.max_context_tokens)
        requests.append(Request(len(requests), cid, t, conv.kind, conv.doc_id is not None,
                                prompt, cfg.gen_tokens, t == conv.num_turns))
        next_turn[cid] = t + 1
        if t == conv.num_turns:
            active.remove(cid)
    return Workload(requests, convs)


def _build_prompt(conv: Conversation, turn: int, asst_tag: list[int], budget: int) -> list[int]:
    history = [(conv.user_ids[i], conv.reply_ids[i]) for i in range(turn - 1)]
    current = conv.user_ids[turn - 1] + asst_tag

    def length() -> int:
        return len(conv.fixed_ids) + sum(len(u) + len(a) for u, a in history) + len(current)

    while history and length() > budget:
        history.pop(0)                      # drop the oldest turn -> later blocks get new hashes
    ids = list(conv.fixed_ids)
    for u, a in history:
        ids += u + a
    ids += current
    return ids[-budget:] if len(ids) > budget else ids


def turn_gap_stats(wl: Workload) -> tuple[dict, list[int]]:
    """Training statistics for the reuse predictor.

    Returns ({(has_doc, turn): (reached, continued)}, gaps) where gaps are the numbers of
    requests between consecutive turns of the same conversation.
    """
    counts: dict = {}
    last_idx: dict[int, int] = {}
    gaps: list[int] = []
    for r in wl.requests:
        key = (r.has_doc, r.turn)
        reached, cont = counts.get(key, (0, 0))
        counts[key] = (reached + 1, cont + (0 if r.is_last_turn else 1))
        if r.conv_id in last_idx:
            gaps.append(r.index - last_idx[r.conv_id])
        last_idx[r.conv_id] = r.index
    return counts, gaps
