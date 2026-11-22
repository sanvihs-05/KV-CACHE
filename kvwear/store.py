"""Tiered KV block store: RAM (LRU) in front of an SSD folder (LRU), governed by a Policy.

Blocks are numpy float16 arrays [layers, 2, kv_heads, block_tokens, head_dim], keyed by
their prefix-chained hash. The SSD tier writes one file per block into `offload_dir`;
those writes are what wear the drive. Writes are asynchronous by default (a background
writer thread with back-pressure), as in real offloading systems, so write cost only
shows up in latency when the disk can't keep up.
"""
from __future__ import annotations

import os
import shutil
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .codecs import Codec, EncodedBlock
from .policy import Policy, RequestContext

MARKER = ".kvwear_offload"


def prepare_offload_dir(path: str | os.PathLike) -> Path:
    """Create (or safely empty) the offload folder. Refuses to wipe a folder it didn't create."""
    root = Path(os.path.expanduser(str(path)))
    if root.exists():
        if (root / MARKER).exists():
            shutil.rmtree(root)
        elif any(root.iterdir()):
            raise RuntimeError(f"{root} is not empty and was not created by kvwear; refusing to wipe it")
    root.mkdir(parents=True, exist_ok=True)
    (root / MARKER).write_text("created by kvwear; safe to delete\n")
    return root


def dir_size(path: str | os.PathLike) -> int:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(dirpath, f))
            except OSError:
                pass
    return total


@dataclass
class TierStats:
    writes: int = 0
    bytes_written: int = 0          # encoded bytes sent to the filesystem
    raw_bytes_written: int = 0      # fp16 bytes before encoding
    reads: int = 0
    bytes_read: int = 0
    evictions: int = 0
    write_s: float = 0.0            # encode + write, in the writer thread
    encode_s: float = 0.0
    read_s: float = 0.0             # read + decode, on the critical path
    backpressure_s: float = 0.0
    peak_bytes: int = 0


class RamTier:
    def __init__(self, capacity_bytes: int):
        self.capacity = capacity_bytes
        self.blocks: OrderedDict[str, np.ndarray] = OrderedDict()
        self.used = 0
        self.evictions = 0

    def __contains__(self, h):
        return h in self.blocks

    def get(self, h):
        a = self.blocks.get(h)
        if a is not None:
            self.blocks.move_to_end(h)
        return a

    def put(self, h, arr) -> list[tuple[str, np.ndarray]]:
        if h in self.blocks:
            self.blocks.move_to_end(h)
            return []
        self.blocks[h] = arr
        self.used += arr.nbytes
        evicted = []
        while self.used > self.capacity and self.blocks:
            eh, ea = self.blocks.popitem(last=False)
            self.used -= ea.nbytes
            self.evictions += 1
            evicted.append((eh, ea))
        return evicted


class SsdTier:
    def __init__(self, root: Path, capacity_bytes: int, codec: Codec, *, async_writes=True,
                 max_pending_bytes=512 << 20, fsync=False, bypass_page_cache=False, read_encoded=False):
        self.root, self.capacity, self.codec = root, capacity_bytes, codec
        self.read_encoded = read_encoded          # True: hand SSD hits back still quantized (GPU decodes)
        self.fsync, self.bypass = fsync or bypass_page_cache, bypass_page_cache
        self.index: OrderedDict[str, int] = OrderedDict()   # hash -> bytes on disk (estimate while pending)
        self.pending: dict[str, np.ndarray] = {}
        self.pending_bytes = 0
        self.max_pending = max_pending_bytes
        self.used = 0
        self.stats = TierStats()
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="kv-writer") if async_writes else None
        self.errors: list[BaseException] = []

    def path(self, h: str) -> Path:
        return self.root / h[:2] / f"{h}.kvb"

    def __contains__(self, h):
        with self.lock:
            return h in self.index

    # -------------------------------------------------------------- write
    def write(self, h: str, arr: np.ndarray) -> None:
        with self.cv:
            if h in self.index:
                return
            if self.pool is not None:
                t0 = time.perf_counter()
                while self.pending_bytes > self.max_pending and not self.errors:
                    self.cv.wait(timeout=1.0)
                self.stats.backpressure_s += time.perf_counter() - t0
            self.index[h] = arr.nbytes
            self.used += arr.nbytes
            self._evict_locked()
            if self.pool is not None:
                self.pending[h] = arr
                self.pending_bytes += arr.nbytes
        if self.pool is not None:
            self.pool.submit(self._do_write, h, arr)
        else:
            self._do_write(h, arr)

    def _do_write(self, h: str, arr: np.ndarray) -> None:
        try:
            t0 = time.perf_counter()
            data = arr.data if isinstance(arr, EncodedBlock) else self.codec.encode(arr)
            t1 = time.perf_counter()
            p = self.path(h)
            p.parent.mkdir(exist_ok=True)
            with open(p, "wb") as f:
                f.write(data)
                if self.fsync:
                    f.flush()
                    os.fsync(f.fileno())
                    if self.bypass and hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
            t2 = time.perf_counter()
            with self.cv:
                s = self.stats
                s.writes += 1
                s.bytes_written += len(data)
                s.raw_bytes_written += arr.nbytes
                s.encode_s += t1 - t0
                s.write_s += t2 - t0
                if self.pool is not None:
                    self.pending.pop(h, None)
                    self.pending_bytes -= arr.nbytes
                if h in self.index:
                    self.used += len(data) - self.index[h]
                    self.index[h] = len(data)
                    s.peak_bytes = max(s.peak_bytes, self.used)
                    self._evict_locked()
                else:                          # evicted while it was being written
                    self._unlink(h)
                self.cv.notify_all()
        except BaseException as e:             # surface writer errors in the main thread
            with self.cv:
                self.errors.append(e)
                self.pending.pop(h, None)
                self.cv.notify_all()

    def _evict_locked(self):
        while self.used > self.capacity and len(self.index) > 1:
            eh, nb = self.index.popitem(last=False)
            self.used -= nb
            self.stats.evictions += 1
            if eh not in self.pending:
                self._unlink(eh)

    def _unlink(self, h):
        try:
            os.remove(self.path(h))
        except FileNotFoundError:
            pass

    # -------------------------------------------------------------- read
    def read(self, h: str) -> np.ndarray | None:
        with self.lock:
            if h not in self.index:
                return None
            self.index.move_to_end(h)
            arr = self.pending.get(h)
        if arr is not None:                    # still in the write queue: serve from memory
            return arr
        t0 = time.perf_counter()
        try:
            with open(self.path(h), "rb") as f:
                data = bytearray(os.fstat(f.fileno()).st_size)   # writable: decoders use it in place
                f.readinto(data)
        except FileNotFoundError:
            return None
        out = EncodedBlock(data) if self.read_encoded else Codec.decode(data)
        with self.lock:
            self.stats.reads += 1
            self.stats.bytes_read += len(data)
            self.stats.read_s += time.perf_counter() - t0
        return out

    def check(self):
        if self.errors:
            raise RuntimeError("KV writer thread failed") from self.errors[0]

    def close(self) -> float:
        t0 = time.perf_counter()
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        self.check()
        return time.perf_counter() - t0


@dataclass
class StoreStats:
    requests: int = 0
    blocks_requested: int = 0
    ram_hits: int = 0
    ssd_hits: int = 0
    misses: int = 0
    blocks_put: int = 0
    ssd_admitted: int = 0
    ssd_dropped: int = 0
    lookup_s: float = 0.0
    extra: dict = field(default_factory=dict)


class TieredKVStore:
    def __init__(self, policy: Policy, offload_dir, *, ram_bytes: int, ssd_bytes: int,
                 async_writes=True, max_pending_bytes=512 << 20, fsync=False, bypass_page_cache=False,
                 read_encoded=False):
        self.policy = policy
        self.adm = policy.admission
        self.root = prepare_offload_dir(offload_dir)
        self.ram = RamTier(ram_bytes)
        self.ssd = SsdTier(self.root, ssd_bytes, policy.codec, async_writes=async_writes,
                           max_pending_bytes=max_pending_bytes, fsync=fsync,
                           bypass_page_cache=bypass_page_cache, read_encoded=read_encoded)
        self.stats = StoreStats()

    def begin_request(self, ctx: RequestContext, hashes: list[str]) -> None:
        self.ssd.check()
        self.stats.requests += 1
        self.stats.blocks_requested += len(hashes)
        self.adm.observe_request(ctx, hashes)

    def lookup_prefix(self, hashes: list[str]) -> list[np.ndarray]:
        """Longest run of leading blocks available in RAM or SSD (SSD hits are promoted)."""
        t0 = time.perf_counter()
        out: list[np.ndarray] = []
        for i, h in enumerate(hashes):
            a = self.ram.get(h)
            if a is not None:
                self.stats.ram_hits += 1
                self.adm.on_lookup(h, True)
                out.append(a)
                continue
            a = self.ssd.read(h)
            if a is not None:
                self.stats.ssd_hits += 1
                self.adm.on_lookup(h, True)
                out.append(a)
                self._ram_insert(h, a)
                continue
            # first miss ends the usable prefix; tell the policy about every missing block
            for mh in hashes[i:]:
                if mh not in self.ram and mh not in self.ssd:
                    self.stats.misses += 1
                    self.adm.on_lookup(mh, False)
            break
        self.adm.on_lookup_done()
        self.stats.lookup_s += time.perf_counter() - t0
        return out

    def put(self, h: str, arr: np.ndarray) -> None:
        """Offer a freshly computed block."""
        self.stats.blocks_put += 1
        if self.adm.eager and h not in self.ssd:
            self.ssd.write(h, arr)
            self.stats.ssd_admitted += 1
        self._ram_insert(h, arr)

    def _ram_insert(self, h, arr):
        for eh, ea in self.ram.put(h, arr):
            if eh in self.ssd:
                continue                       # already persisted (inclusive tiers)
            if self.adm.eager:
                continue                       # write-through already handled it
            if self.adm.should_write(eh):
                self.ssd.write(eh, ea)
                self.stats.ssd_admitted += 1
            else:
                self.adm.on_drop(eh)
                self.stats.ssd_dropped += 1

    def close(self) -> float:
        return self.ssd.close()

    def summary(self) -> dict:
        s, t = self.stats, self.ssd.stats
        return {
            "policy": self.policy.name, "policy_spec": self.policy.spec, "codec": self.policy.codec.name,
            "requests": s.requests, "blocks_requested": s.blocks_requested,
            "ram_hits": s.ram_hits, "ssd_hits": s.ssd_hits, "misses": s.misses,
            "blocks_put": s.blocks_put, "ssd_admitted": s.ssd_admitted, "ssd_dropped": s.ssd_dropped,
            "ram_evictions": self.ram.evictions,
            "app_ssd_writes": t.writes, "app_bytes_written": t.bytes_written,
            "app_raw_bytes_written": t.raw_bytes_written,
            "app_ssd_reads": t.reads, "app_bytes_read": t.bytes_read, "ssd_evictions": t.evictions,
            "ssd_peak_bytes": t.peak_bytes, "write_thread_s": t.write_s, "encode_s": t.encode_s,
            "ssd_read_s": t.read_s, "backpressure_s": t.backpressure_s, "lookup_s": s.lookup_s,
            **{f"policy_{k}": v for k, v in self.adm.stats().items()},
        }
