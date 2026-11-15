"""Linux-side write accounting: /proc/diskstats for the block device behind a folder.

In WSL2 the Linux root filesystem is an ext4 virtual disk (ext4.vhdx) shown as /dev/sdX.
Sectors written to that device = what Linux sent to the virtual disk. Comparing it with the
SSD's own SMART counter (read on Windows) shows the extra writes added by the
virtualization layer. Diskstats sectors are always 512 bytes.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path


def device_for_path(path: str | os.PathLike) -> str | None:
    """Kernel block device name (e.g. 'sdc') that stores `path`, or None if unknown."""
    if not hasattr(os, "major"):          # not Linux (e.g. testing on Windows)
        return None
    try:
        st = os.stat(os.path.expanduser(str(path)))
    except OSError:
        return None
    major, minor = os.major(st.st_dev), os.minor(st.st_dev)
    sys_link = Path(f"/sys/dev/block/{major}:{minor}")
    if sys_link.exists():
        return sys_link.resolve().name
    # fallback: match major:minor directly in /proc/diskstats
    for line in _read_lines():
        p = line.split()
        if len(p) > 3 and int(p[0]) == major and int(p[1]) == minor:
            return p[2]
    return None


def _read_lines() -> list[str]:
    try:
        return Path("/proc/diskstats").read_text().splitlines()
    except OSError:
        return []


def read_counters(device: str | None) -> dict | None:
    """{'sectors_written', 'writes_completed', 'sectors_read', 'reads_completed'} for device."""
    if not device:
        return None
    for line in _read_lines():
        p = line.split()
        if len(p) >= 10 and p[2] == device:
            return {
                "reads_completed": int(p[3]), "sectors_read": int(p[5]),
                "writes_completed": int(p[7]), "sectors_written": int(p[9]),
            }
    return None


class DiskCounter:
    """Snapshot diskstats before/after a run. No-op (None values) off Linux."""

    def __init__(self, path):
        self.device = device_for_path(path)
        self.before = read_counters(self.device)

    def delta(self, do_sync: bool = True) -> dict:
        if do_sync and hasattr(os, "sync"):
            os.sync()
        after = read_counters(self.device)
        if not (self.before and after):
            return {"linux_device": self.device, "linux_bytes_written": None, "linux_bytes_read": None}
        return {
            "linux_device": self.device,
            "linux_bytes_written": 512 * (after["sectors_written"] - self.before["sectors_written"]),
            "linux_bytes_read": 512 * (after["sectors_read"] - self.before["sectors_read"]),
            "linux_write_ios": after["writes_completed"] - self.before["writes_completed"],
        }


class DirSizeMonitor(threading.Thread):
    """Samples a folder's size in the background (week-1 check: does the offload folder grow?)."""

    def __init__(self, path, interval: float = 1.0, verbose: bool = False):
        super().__init__(daemon=True)
        self.path, self.interval, self.verbose = path, interval, verbose
        self.peak = 0
        self.samples: list[tuple[float, int]] = []
        self._halt = threading.Event()      # not `_stop`: Thread uses that name internally

    def run(self):
        from .store import dir_size
        t0 = time.time()
        while not self._halt.is_set():
            size = dir_size(self.path)
            self.peak = max(self.peak, size)
            self.samples.append((time.time() - t0, size))
            if self.verbose:
                print(f"[offload-dir] t={time.time() - t0:6.1f}s size={size / 2**20:9.1f} MiB", flush=True)
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)
