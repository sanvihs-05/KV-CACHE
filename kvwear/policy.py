"""Offloading policies = SSD *admission rule* + block *codec*.

The store keeps blocks in RAM (LRU). A policy decides which blocks also get written to
the SSD, and in what format:

  admission     when a block is written to the SSD
  ---------     -------------------------------------------------------------
  none          never (RAM only; a miss means recomputing the KV) - fewest writes, slowest
  always        write-through: every new block is written when it is created
                (how LMCache-style disk backends and FlexLLMGen behave) - the usual baseline
  evict         write-back: write a block only when RAM evicts it
  reuse         reuse-aware write-back: when RAM evicts a block, write it only if it is
                predicted to be read again (see ReuseAware below)
  oracle        knows the future trace and writes only blocks that are read again.
                Not implementable; it is the upper bound on writes saved.

Spec grammar:  "<admission>[@<threshold>]/<codec>"   e.g. "reuse@0.4/int4+zstd"
Named presets: see PRESETS.
"""
from __future__ import annotations

import bisect
from collections import OrderedDict
from dataclasses import dataclass

from .codecs import Codec, make_codec

PRESETS = {
    "no_ssd": "none/fp16",
    "write_through": "always/fp16",
    "write_through_int4": "always/int4",
    "spill": "evict/fp16",
    "spill_fp16_zstd": "evict/fp16+zstd",
    "spill_int8": "evict/int8",
    "spill_int4": "evict/int4",
    "reuse": "reuse@0.5/fp16",
    "reuse_int8": "reuse@0.5/int8",
    "reuse_int4": "reuse@0.5/int4",
    "oracle": "oracle/fp16",
    "oracle_int4": "oracle/int4",
}


@dataclass
class RequestContext:
    index: int
    conv_id: int
    turn: int
    has_doc: bool


class Admission:
    name = "base"
    eager = False                      # True: write on creation (write-through)

    def observe_request(self, ctx: RequestContext, hashes: list[str]) -> None:
        pass

    def on_lookup(self, h: str, hit: bool) -> None:
        pass

    def on_lookup_done(self) -> None:
        """The current request has finished reading its prefix."""

    def should_write(self, h: str) -> bool:
        return True

    def on_drop(self, h: str) -> None:
        pass

    def stats(self) -> dict:
        return {}


class NoSSD(Admission):
    name = "none"

    def should_write(self, h):
        return False


class WriteThrough(Admission):
    name = "always"
    eager = True


class SpillOnEvict(Admission):
    name = "evict"


# --------------------------------------------------------------------------- reuse-aware


class ContinuationModel:
    """P(a conversation sends another turn | its turn index, has a doc, idle time).

    Fitted on a *training* trace (different random seed from the evaluated one), like a
    learned admission filter (cf. Flashield, NSDI'19: DRAM filters what goes to flash).

      p_cont(turn, has_doc) = P(turn t is not the last)            (Laplace-smoothed)
      S(g)                  = P(gap between turns > g requests)    (empirical survival)
      P(return | idle g)    = p*S(g) / (p*S(g) + (1-p))            (Bayes)

    On a real device `g` would be wall-clock time since the user's last message.
    """

    def __init__(self, counts: dict | None = None, gaps: list[int] | None = None, prior: float = 0.6):
        self.prior = prior
        self.table: dict = {}
        for (has_doc, turn), (reached, cont) in (counts or {}).items():
            self.table[(has_doc, turn)] = (cont + 1) / (reached + 2)
        self.max_turn = {hd: max([t for (h, t) in self.table if h == hd], default=0) for hd in (False, True)}
        g = sorted(gaps or [])
        self._gaps = g

    def p_cont(self, turn: int, has_doc: bool) -> float:
        if (has_doc, turn) in self.table:
            return self.table[(has_doc, turn)]
        if turn > self.max_turn.get(has_doc, 0) and self.max_turn.get(has_doc, 0) > 0:
            return 0.05                 # longer than any training conversation: probably done
        return self.prior

    def survival(self, idle: int) -> float:
        if idle <= 0 or not self._gaps:
            return 1.0
        n_le = bisect.bisect_right(self._gaps, idle)
        return max(1.0 - n_le / len(self._gaps), 0.01)

    def p_return(self, turn: int, has_doc: bool, idle: int) -> float:
        p = self.p_cont(turn, has_doc)
        s = self.survival(idle)
        return p * s / (p * s + (1 - p)) if p < 1 else 1.0


class ReuseAware(Admission):
    """Write a block to SSD on RAM eviction only if it is likely to be read again.

    score(block) =
      1.0  if it is shared by >= `shared_min_convs` conversations (system prompt, shared docs)
      1.0  if it was dropped before and then requested again ("proven" via a ghost list:
           the filter corrects its own mistakes)
      0.0  if no conversation's current context contains it any more (dead: the history
           was truncated or the block belongs to an old prefix)
      else max over live owner conversations of P(conversation returns), from ContinuationModel
    Write iff score >= threshold. Sweeping the threshold traces the writes-vs-speed curve.
    """

    name = "reuse"

    def __init__(self, threshold: float = 0.5, model: ContinuationModel | None = None,
                 shared_min_convs: int = 2, ghost_capacity: int = 200_000):
        self.threshold = threshold
        self.model = model or ContinuationModel()
        self.shared_min = shared_min_convs
        self.ghost: OrderedDict[str, None] = OrderedDict()
        self.ghost_capacity = ghost_capacity
        self.proven: set[str] = set()
        self.block_convs: dict[str, set[int]] = {}
        self.conv_live: dict[int, set[str]] = {}
        self.conv_state: dict[int, tuple[int, bool, int]] = {}
        self.now = 0
        self.n_ghost_hits = 0
        self.n_dead = 0
        self.n_shared = 0

    def observe_request(self, ctx, hashes):
        self.now = ctx.index
        for h in hashes:
            s = self.block_convs.setdefault(h, set())
            if len(s) < self.shared_min:
                s.add(ctx.conv_id)
        self.conv_live[ctx.conv_id] = set(hashes)
        self.conv_state[ctx.conv_id] = (ctx.turn, ctx.has_doc, ctx.index)

    def on_lookup(self, h, hit):
        if not hit and h in self.ghost:
            del self.ghost[h]
            self.proven.add(h)
            self.n_ghost_hits += 1

    def score(self, h: str) -> float:
        if h in self.proven:
            return 1.0
        convs = self.block_convs.get(h, ())
        if len(convs) >= self.shared_min:
            self.n_shared += 1
            return 1.0
        best = 0.0
        live = False
        for c in convs:
            if h in self.conv_live.get(c, ()):
                live = True
                turn, has_doc, last = self.conv_state[c]
                best = max(best, self.model.p_return(turn, has_doc, self.now - last))
        if not live:
            self.n_dead += 1
        return best

    def should_write(self, h):
        return self.score(h) >= self.threshold

    def on_drop(self, h):
        self.ghost[h] = None
        if len(self.ghost) > self.ghost_capacity:
            self.ghost.popitem(last=False)

    def stats(self):
        return {"ghost_hits": self.n_ghost_hits, "dead_scored": self.n_dead,
                "shared_scored": self.n_shared, "threshold": self.threshold}


class Oracle(Admission):
    """Writes a block iff the (pre-computed) trace looks it up again later."""

    name = "oracle"

    def __init__(self, future: dict[str, list[int]]):
        self.future = future
        self.now = 0
        self.in_lookup = False

    def observe_request(self, ctx, hashes):
        self.now = ctx.index
        self.in_lookup = True

    def on_lookup_done(self):
        self.in_lookup = False

    def should_write(self, h):
        uses = self.future.get(h)
        if not uses:
            return False
        # while the current request is still reading its prefix, its own use is still ahead
        pos = bisect.bisect_left(uses, self.now) if self.in_lookup else bisect.bisect_right(uses, self.now)
        return pos < len(uses)


# --------------------------------------------------------------------------- factory


@dataclass
class Policy:
    name: str
    spec: str
    admission: Admission
    codec: Codec


def make_policy(name_or_spec: str, *, continuation: ContinuationModel | None = None,
                future: dict[str, list[int]] | None = None) -> Policy:
    spec = PRESETS.get(name_or_spec, name_or_spec)
    adm_part, _, codec_part = spec.partition("/")
    adm_name, _, thr = adm_part.partition("@")
    codec = make_codec(codec_part or "fp16")
    if adm_name == "none":
        adm: Admission = NoSSD()
    elif adm_name == "always":
        adm = WriteThrough()
    elif adm_name == "evict":
        adm = SpillOnEvict()
    elif adm_name == "reuse":
        adm = ReuseAware(float(thr) if thr else 0.5, continuation)
    elif adm_name == "oracle":
        if future is None:
            raise ValueError("oracle policy needs the future trace")
        adm = Oracle(future)
    else:
        raise ValueError(f"unknown admission '{adm_name}' in policy '{name_or_spec}'")
    return Policy(name_or_spec, spec, adm, codec)
