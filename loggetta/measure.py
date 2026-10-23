"""Observation: what a run actually used, kept distinct from what the plan estimated.

* device, allocator view: ``torch.cuda.max_memory_allocated`` / ``max_memory_reserved`` (measured);
* device, driver view: this process's ``used_memory`` from ``nvidia-smi --query-compute-apps``, sampled on a thread
  (measured; the difference to the allocator's reserved peak is the CUDA context and library workspaces);
* host: ``VmHWM`` (peak resident set) and ``VmRSS`` from ``/proc/self/status`` (measured);
* provenance: git commits of the source trees actually imported, distribution versions, the command line.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time


class DriverMemorySampler:
    """Peak driver-reported device memory of THIS process, and its peak anonymous / file-backed host RSS, sampled
    every ``interval`` seconds. VmHWM alone overstates what a run NEEDS from the host: memory-mapped checkpoint shards
    count in it as file-backed pages the kernel can reclaim. What cannot be reclaimed is ``RssAnon`` plus ``RssShmem``,
    which is where pinned (cudaHostAlloc) memory is accounted -- measured: a 512 MiB pinned tensor adds 511 MiB of
    RssShmem and nothing to RssAnon."""

    def __init__(self, interval: float = 0.25):
        self.interval, self.peak, self.samples = interval, 0, 0
        #: nvidia-smi reads attempted and reads that failed: with no sample, they say why the driver peak is unknown
        self.reads, self.read_errors = 0, 0
        self.anon_peak, self.file_peak, self.shmem_peak, self.required_peak = 0, 0, 0, 0
        #: the anonymous host peak since the last :meth:`mark` (a phase's own peak, e.g. serving after a load)
        self.phase_anon_peak = 0
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
            v = None
            if self._smi:
                self.reads += 1
                try:
                    v = self._read()
                except (subprocess.SubprocessError, OSError):
                    self.read_errors += 1
            if v is not None:
                self.peak, self.samples = max(self.peak, v), self.samples + 1
            st = proc_status()
            self.anon_peak = max(self.anon_peak, st.get("RssAnon", 0))
            self.phase_anon_peak = max(self.phase_anon_peak, st.get("RssAnon", 0))
            self.file_peak = max(self.file_peak, st.get("RssFile", 0))
            self.shmem_peak = max(self.shmem_peak, st.get("RssShmem", 0))
            # pinned host memory (cudaHostAlloc) is accounted as RssShmem; mapped checkpoint shards as RssFile
            self.required_peak = max(self.required_peak, st.get("RssAnon", 0) + st.get("RssShmem", 0))
            self._stop.wait(self.interval)

    def unmeasured_reason(self) -> str | None:
        """Why the driver-reported process peak is unknown (no sample was taken), or None when it was sampled."""
        if self.samples:
            return None
        if not self._smi:
            return "no nvidia-smi on PATH"
        if self.reads and self.read_errors == self.reads:
            return f"all {self.reads} nvidia-smi reads failed"
        return (f"nvidia-smi answered {self.reads} time(s) but never listed this process's GPU memory (WSL2, and some "
                "containers, do not report per-process memory)")

    def mark(self, anon_now: int = 0):
        """Start a new phase: :attr:`phase_anon_peak` restarts from ``anon_now``. A load can peak higher than the
        serving that follows it (a repack's host buffers, handed back before serving), so serving's own growth needs
        its own peak."""
        self.phase_anon_peak = anon_now

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
                if k in ("VmHWM", "VmRSS", "RssAnon", "RssFile", "RssShmem"):
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


def _source_file(name):
    """Where ``name`` WOULD be imported from, without importing it (so provenance can be taken before any import)."""
    import importlib.util

    mod = sys.modules.get(name)
    if mod is not None:
        return getattr(mod, "__file__", None)
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    return spec.origin if spec else None


#: experts4bit-qlora switches that change a run's numerics or speed: the ONLY E4B_* variables whose VALUES a receipt
#: records or hashes. Every other set E4B_* variable is recorded by NAME only, in the record and in the hash alike, since
#: the prefix also carries plumbing whose values identify a host (E4B_RENT_SSH_HOST, E4B_HOST), a path or a private
#: model, or are secrets (E4B_TOKEN), and a hash over a low-entropy value is a guessing oracle. An allowlist, not a
#: denylist: a knob the hash must separate goes on this list. The one list test_renameable reads.
E4B_VALUE_KNOBS = (
    "E4B_ABSMAX_DQ", "E4B_CHUNKED_LM_LOSS", "E4B_CHUNKED_EVAL_LOSS", "E4B_TRAIN_FUSE_QKV",
    "E4B_FUSE_T1_GLUE", "E4B_FUSE_T1_GLUE_R2", "E4B_FUSE_ROUTER_EPI",
    "E4B_PAGED_BUCKETS", "E4B_PAGED_BULK_KV", "E4B_PAGED_CHUNK_TOKENS", "E4B_PAGED_DECODE_LOOKAHEAD", "E4B_PAGED_DRAM_GB",
    "E4B_PAGED_FUSE_QKV", "E4B_PAGED_GRAPHS", "E4B_PAGED_HOT_ROWS", "E4B_PAGED_KV_GROUPS", "E4B_PAGED_LAST_LOGITS",
    "E4B_PAGED_MAX_PREFILL_TOKENS", "E4B_PAGED_MAX_QUEUE", "E4B_PAGED_MAX_SEQS", "E4B_PAGED_MAX_TOKENS",
    "E4B_PAGED_MAX_TOKENS_PER_SEQ", "E4B_PAGED_PLACEMENT", "E4B_PAGED_PREFILL_ATTN", "E4B_PAGED_PREFILL_GRAPH",
    "E4B_PAGED_SPEC", "E4B_PAGED_SPEC_K", "E4B_PAGED_TORCH_THREADS", "E4B_PAGED_VRAM_GB",
    "E4B_SERVE_ATTN_INT4", "E4B_SERVE_ATTN_INT8", "E4B_SERVE_EXP_INT4",
    # route and kernel switches: each picks a different function for the same shape (K25 plan and LUT, the small-M
    # routes, int4 tiles, the dense and head GEMVs, the fused glue), so two runs differing only here measure differently
    "E4B_K25_PLAN", "E4B_K25_LUT", "E4B_K_SLOTS", "E4B_NF4_GROUPED_SMALLM", "E4B_NF4_T1_DEVICE_GROUPING", "E4B_NF4_SMALLM_BW",
    "E4B_INT4_DECODE_A16", "E4B_INT4_GROUPED_SMALLM", "E4B_INT4_LEAN_GLUE", "E4B_INT4_PREFILL", "E4B_INT4_TILE_PROGRAMS",
    "E4B_INT4_WIDE_TILES", "E4B_INT4_KEEP_NF4", "E4B_INT4_GPTQ_DAMP", "E4B_ATTN_INT4_SMALLM", "E4B_ATTN_INT4_WIDE",
    "E4B_ATTN_SPLIT_CAP", "E4B_INT8_HEAD_BW", "E4B_DENSE_GEMV_BW", "E4B_INFER_GEMV", "E4B_MXFP4_GEMV",
    "E4B_MXFP4_GROUPED_SMALLM", "E4B_DECODE_FASTPATH", "E4B_FUSED_RMSNORM", "E4B_FUSED_ROPE", "E4B_FUSED_KV_APPEND",
    "E4B_KV_STEP_SELECT", "E4B_BATCHED_KV_WRITE", "E4B_BATCHED_PAD_WASTE_LIMIT", "E4B_COMBINE_CHUNK", "E4B_GROUPING",
    "E4B_K19_PREFILL_BIG_M", "E4B_PREFILL_LEAN_DISPATCH", "E4B_ROUTER_EPI_CAST", "E4B_PIPELINED_TRAFFIC",
    "E4B_TRITON_PREBIND", "E4B_DENSE_MLP_LORA_FUSE", "E4B_CKPT_OFFLOAD", "E4B_EMPTY_CACHE", "E4B_MOE_KEEP_LAYERS",
    "E4B_HOT_PER_LAYER", "E4B_VRAM_FRACTION", "E4B_RESIDENCY", "E4B_OFFLOAD_PIN_SLAB", "E4B_REFERENCE_EXPERT_ORDER",
    # grouped-nf4-gemm's own switches (all but GNF4_NATIVE_CACHE, a build-cache directory: by name only)
    "GNF4_ALLOW_UNVERIFIED_V5", "GNF4_ATTN_COMPUTE", "GNF4_ATTN_F32_PRECISION", "GNF4_DECODED_MAX_BYTES",
    "GNF4_DECODE_PLAN", "GNF4_F32_FUSE_COMBINE", "GNF4_GEMV_BW", "GNF4_GEMV_BW_DECODE", "GNF4_GEMV_BW_PLAN",
    "GNF4_GEMV_DOTPAD", "GNF4_GEMV_FUSED_REDUCE", "GNF4_GEMV_SPLITK", "GNF4_GEMV_VEC_LOADS", "GNF4_GEMV_WIDE_LOADS",
    "GNF4_HOST_REUSE", "GNF4_PDL", "GNF4_PDL_MAX_ROWS", "GNF4_PINNED_RING", "GNF4_PINNED_RING_SLOTS",
    "GNF4_PINNED_RING_SLOT_INTS", "GNF4_PIN_ARENA_INTS", "GNF4_PREFILL_TILE_D", "GNF4_PREFILL_TILE_RULE",
    "GNF4_TRAIN_GEMM", "GNF4_TRITON_PREBIND",
    # more switches that change the route, the numbers or the run's shape: the fused SwiGLU / combine / prefill glue
    # (0 swaps the kernel), the speculative head, gpt-oss's expert-LoRA route, the arena poison (debug), remote code,
    # the stop ids, the int4 calibration and GPTQ settings, a safety bypass, the device, the serving token limits and
    # warmup, and the timing-affecting debug flags. A path-valued one (the spec head, the int4 assignment, the
    # calibration source) is still recorded by its name only (see e4b_env)
    "E4B_FUSE_SWIGLU", "E4B_FUSE_COMBINE", "E4B_FUSE_PREFILL_GLUE", "E4B_PAGED_SPEC_HEAD", "E4B_GPTOSS_EXPERT_LORA",
    "E4B_ARENA_POISON", "E4B_TRUST_REMOTE_CODE", "E4B_PAGED_EOS_IDS", "E4B_CALIB_NSEQ", "E4B_CALIB_LAYERS_PER_PASS",
    "E4B_INT4_GPTQ_DEVICE", "E4B_INT4_HESSIAN_BUDGET_GB", "E4B_INT4_DUMP_ALLOW_UNKNOWN_REVISION", "E4B_PAGED_DEVICE",
    "E4B_MAX_NEW_TOKENS", "E4B_MAX_INPUT_TOKENS", "E4B_WARMUP_TOKENS", "E4B_LOAD_SYNC_DEBUG", "E4B_PROFILE_PHASE",
    "E4B_OFFLOAD_STATS", "E4B_INT4_ASSIGNMENT", "E4B_CALIB_SOURCE",
    "E4B_PAGED_ADAPTER_ACCEPT_UNVERIFIED_BASE", "E4B_PAGED_ADAPTER_ALLOW_UNVERIFIED_ARENA",
)
#: a name with one of these as a whole "_"-separated word names a credential: its value is NEVER recorded or hashed, even
#: if it is ever added to E4B_VALUE_KNOBS by mistake (a test also keeps it off that list). "TOKENS" (a count) is not one
SECRET_WORDS = frozenset({"TOKEN", "KEY", "APIKEY", "SECRET", "PASSWORD", "PASSWD", "AUTH", "CREDENTIAL", "CREDENTIALS"})


def is_secret_name(name: str) -> bool:
    return bool(SECRET_WORDS & set(name.split("_")))


#: every other switch the backends read, recorded by NAME only, each with why its value stays out. Between them, this
#: table and E4B_VALUE_KNOBS classify every E4B_/GNF4_ name in the installed backends' sources, and a test fails on a
#: new one until someone decides whether its value is recorded (default: recorded, unless a path, an id or a secret)
NAME_ONLY = {
    "E4B_TOKEN": "a secret: the HTTP server's bearer token",
    "E4B_HOST": "the HTTP server's bind address",
    "E4B_PORT": "the HTTP server's port",
    "E4B_QUEUE_MAX": "the HTTP server's queue limit (serving plumbing, not the engine)",
    "E4B_REQUEST_TIMEOUT_S": "the HTTP server's request timeout (serving plumbing, not the engine)",
    "E4B_RECEIPTS_PATH": "a path: where the server writes its receipts",
    "E4B_ADAPTERS": "paths: the server's adapter directories",
    "E4B_MODEL_ID": "an id: the served model, recorded in the receipt's model field by its label",
    "E4B_PAGED_MODEL": "an id or path: the served model, recorded in the receipt's model field by its label",
    "E4B_PAGED_REVISION": "an id: the served revision, recorded in the receipt's model field",
    "E4B_PAGED_SERVED_NAME": "an id: the name the server answers to",
    "E4B_PAGED_ARENA": "a path: the NF4 arena (a serve receipt names it with its sha256)",
    "E4B_PAGED_CALIB": "a path: the calibration file (a serve receipt names it with its sha256)",
    "E4B_PAGED_ADAPTER": "a path: the served adapter directory",
    "E4B_PAGED_STEP_TRACE": "a path: the step-trace sink",
    "E4B_PAGED_TRACE": "a path: the request-trace sink",
    "E4B_OFFLOAD_ARENA": "a path: the offload arena",
    "E4B_EXPERT_PROFILE": "a path: the routing profile",
    "E4B_HOT_PROFILE": "a path: the hot-expert profile",
    "E4B_INT4_ARTIFACT_DIR": "a path: the int4 artifact directory",
    "E4B_INT4_DUMP_ARTIFACT_DIR": "a path: where int4 artifacts are dumped",
    "E4B_INT4_EXPECTED_FINGERPRINT": "an id: the expected int4 artifact fingerprint (the artifact is checked against it)",
    "E4B_SERVE_ATTN_INT4_ARTIFACT": "a path: the int4 attention artifact",
    "E4B_SERVE_ATTN_INT4_CALIB": "a path: the int4 attention calibration",
    "E4B_SERVE_ATTN_INT4_DUMP": "a path: where the int4 attention calibration is dumped",
    "E4B_SERVE_ATTN_INT4_FINGERPRINT": "an id: the expected int4 attention fingerprint",
    "E4B_SERVE_DENSE_INT4_CALIB": "a path: the dense int4 calibration",
    "E4B_SERVE_EXP_INT4_CALIB": "a path: the expert int4 calibration",
    "E4B_SERVE_LMHEAD_INT4_CALIB": "a path: the LM-head int4 calibration",
    "GNF4_NATIVE_CACHE": "a path: the native build cache",
}


def e4b_env(environ=None) -> dict:
    """The backends' environment a run starts under (experts4bit-qlora's E4B_* and grouped-nf4-gemm's GNF4_*),
    comparable across receipts without carrying identifiers: the sorted NAMES of every set variable, the sha256 of its
    sorted lines (``name=value`` for a switch in ``E4B_VALUE_KNOBS``, the bare name for any other), and the VALUES of
    those switches. A listed switch whose value is nonetheless a path is recorded and hashed by its name only."""
    from .shareable import label

    env = ({k: v for k, v in os.environ.items() if k.startswith(("E4B_", "GNF4_"))} if environ is None
           else {k: v for k, v in environ.items() if k.startswith(("E4B_", "GNF4_"))})
    vals = {k: label(env[k]) for k in E4B_VALUE_KNOBS if k in env and not is_secret_name(k)}
    lines = sorted(f"{k}={vals[k]}" if k in vals else k for k in env)
    return {"names": sorted(env), "sha256": hashlib.sha256("\n".join(lines).encode()).hexdigest(), "values": vals}


def backend_env_names() -> list:
    """The sorted NAMES (never values) of every set E4B_* or GNF4_* variable: the two backends' switches."""
    return sorted(k for k in os.environ if k.startswith(("E4B_", "GNF4_")))


def provenance(modules=("experts4bit_qlora", "nf4_grouped", __package__.split(".")[0])) -> dict:
    """Commits, versions and argv. Call it at PROCESS START, before the measured code is imported: a commit read later
    can name code that changed on disk after the process loaded it. :func:`changed_since` checks for that."""
    out = {"argv": list(sys.argv), "python": sys.version.split()[0], "sources": {}}
    for name in modules:
        f = _source_file(name)
        if f:
            out["sources"][name] = {"file": f, **git_commit(f)}
    from importlib.metadata import PackageNotFoundError, version

    for dist in ("torch", "triton", "transformers", "bitsandbytes", "accelerate", "experts4bit-qlora",
                 "grouped-nf4-gemm"):
        try:
            out.setdefault("versions", {})[dist] = version(dist)
        except PackageNotFoundError:
            pass
    # the allocator setting the run starts under (None when unset): PyTorch's caching allocator reads it at CUDA init,
    # and it moves the reserved-but-unallocated bytes the planner learns its reserve from. Recorded only; nothing reads it
    out["pytorch_cuda_alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    out["e4b_env"] = e4b_env()       # experts4bit-qlora switches (E4B_ABSMAX_DQ moved s/step 39% unrecorded); recorded only
    out["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return out


def changed_since(prov: dict) -> list:
    """Source trees whose commit or dirty state differs now from ``prov``: the run's code may not be what it names."""
    out = []
    for name, src in prov.get("sources", {}).items():
        now = git_commit(src["file"]) if src.get("file") else {}
        if (now.get("commit"), now.get("dirty")) != (src.get("commit"), src.get("dirty")):
            out.append(f"{name}: {src.get('commit', 'untracked')[:10]}{'+dirty' if src.get('dirty') else ''} at start, "
                       f"{now.get('commit', 'untracked')[:10]}{'+dirty' if now.get('dirty') else ''} at end")
    return out


def decode_step_b1(rows) -> dict | None:
    """The single-stream decode step from an experts4bit-qlora step trace (``E4B_PAGED_STEP_TRACE``, one row per
    engine step; ``experts4bit_qlora.engines.step_trace``): the steps that decoded exactly one row and ran no prefill.
    ``step_ms`` is the step's whole host time (inputs, replay or eager step, the token sync, bookkeeping), so a decode
    step and its tokens per second are what one sequence sees. ``device_ms`` is the decode's device time
    (``gpu.dec_issue - gpu.dec_prep``) where the trace resolved its events. None when the trace has no such step."""
    steps = [r for r in rows if r.get("decode_rows") == 1 and not r.get("prefill_tokens")
             and isinstance(r.get("step_ms"), (int, float))]
    if not steps:
        return None

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    dev = [r["gpu"]["dec_issue"] - r["gpu"]["dec_prep"] for r in steps
           if isinstance(r.get("gpu"), dict) and {"dec_issue", "dec_prep"} <= set(r["gpu"])]
    return {"decode_step_ms_b1": round(median([r["step_ms"] for r in steps]), 4),
            "decode_device_ms_b1": round(median(dev), 4) if dev else None,
            "decode_steps_b1": len(steps),
            "decode_buckets_b1": sorted({r.get("bucket") for r in steps}, key=lambda b: (b is None, b))}
