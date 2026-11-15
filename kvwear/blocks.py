"""Prefix-chained block hashing (same idea as vLLM / LMCache prefix caching).

A KV block covers `block_tokens` consecutive tokens. Its KV values depend on every
token before it, so its identity is a hash of the *whole prefix*, chained block by block.
Two requests that start with the same tokens (e.g. the same system prompt) share the
same leading block hashes. Only full blocks are cached.
"""
from __future__ import annotations

import hashlib
from array import array


def block_hashes(token_ids: list[int], block_tokens: int) -> list[str]:
    out: list[str] = []
    prev = b""
    n_full = len(token_ids) // block_tokens
    for b in range(n_full):
        chunk = array("i", token_ids[b * block_tokens:(b + 1) * block_tokens]).tobytes()
        h = hashlib.blake2b(prev + chunk, digest_size=16).digest()
        out.append(h.hex())
        prev = h
    return out
