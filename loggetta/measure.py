"""Observation: what a run actually used, kept distinct from what the plan estimated.

* device, allocator view: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` (measured);
* device, driver view: this process's ``used_memory`` from ``nvidia-smi --query-compute-apps``, sampled on a thread
  (measured; the difference to the allocator's reserved peak is the CUDA context and library workspaces);
* host: ``VmHWM`` (peak resident set) and ``VmRSS`` from ``/proc/self/status`` (measured);
* provenance: git commits of the source trees actually imported, distribution versions, the command line.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time


class DriverMemorySampler:
    """Peak driver-reported device memory of THIS process, and its peak anonymous / file-backed host RSS, sampled
    every ``interval`` seconds. VmHWM alone overstates what a run NEEDS from the host: memory-mapped checkpoint shards
    count in it as file-backed pages the kernel can reclaim; ``RssAnon`` is the part that cannot be."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.peak, self.samples = interval, 0, 0
        self.anon_peak, self.file_peak = 0, 0
        self._stop = threading.Event()
        self._smi = shutil.which("nvidia-smi")
        self._t = threading.Thread(target=self._run, daemon=True)

    def _read(self):
        out = subprocess.run([self._smi, "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        pid = str(os.getpid())
        for line in out.splitlines():
            p, _, mib = line.partition(",")
            if p.strip() == pid and mib.strip().isdigit():
                return int(mib.strip()) << 20
        return None

    def _run(self):
        while not self._stop.is_set():
            try:
                v = self._read() if self._smi else None
            except (subprocess.SubprocessError, OSError):
                v = None
            if v is not None:
                self.peak, self.samples = max(self.peak, v), self.samples + 1
            st = proc_status()
            self.anon_peak = max(self.anon_peak, st.get("RssAnon", 0))
            self.file_peak = max(self.file_peak, st.get("RssFile", 0))
            self._stop.wait(self.interval)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t.is_alive():
            self._t.join(timeout=15)


def proc_status() -> dict:
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, v = line.partition(":")
                if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile"):
                    out[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return out


def git_commit(path) -> dict:
    """``{"commit", "branch", "dirty"}`` for the git checkout that TRACKS ``path``, or ``{}`` (an installed wheel, or a
    file that merely sits inside some enclosing repository -- reporting that repository's commit would be false)."""
    d = os.path.dirname(os.path.abspath(path))
    try:
        tracked = subprocess.run(["git", "-C", d, "ls-files", "--error-unmatch", os.path.abspath(path)],
                                 capture_output=True, text=True, timeout=10)
        if tracked.returncode:
            return {}
        sha = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if sha.returncode:
            return {}
        dirty = subprocess.run(["git", "-C", d, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30).stdout.strip()
        branch = subprocess.run(["git", "-C", d, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                text=True, timeout=10).stdout.strip()
        return {"commit": sha.stdout.strip(), "branch": branch, "dirty": bool(dirty)}
    except (subprocess.SubprocessError, OSError):
        return {}


def provenance(modules=("experts4bit_qlora", "nf4_grouped", __package__.split(".")[0])) -> dict:
    out = {"argv": list(sys.argv), "python": sys.version.split()[0], "sources": {}}
    for name in modules:
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        out["sources"][name] = {"file": getattr(mod, "__file__", None), **git_commit(mod.__file__)}
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("torch", "triton", "transformers", "bitsandbytes", "accelerate", "experts4bit-qlora",
                 "grouped-nf4-gemm"):
        try:
            out.setdefault("versions", {})[dist] = version(dist)
        except PackageNotFoundError:
            pass
    out["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out
