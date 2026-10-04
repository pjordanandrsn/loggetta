"""Hardware inventory: what this machine offers, and how each number is known.

Every value is a :class:`Fact` carrying its source:

* ``reported`` -- the driver or the kernel said so (``nvidia-smi``, ``/proc``, cgroup files);
* ``measured`` -- this package timed or allocated something to get it;
* ``inferred`` -- derived from other facts or a stated default (it says from what);
* ``user`` -- the caller supplied it, and it overrides discovery.

Nothing here imports torch: an inventory must be cheap enough to take before deciding whether to load anything.
Discovery that fails yields ``None`` with a note, never a guess. Multiple GPUs are a list from the start; no code
below assumes there is exactly one.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import asdict, dataclass, field, replace

GiB = 1 << 30


@dataclass(frozen=True)
class Fact:
    value: object
    source: str            # "reported" | "measured" | "inferred" | "user"
    note: str = ""

    def __bool__(self):
        return self.value is not None


def _fact(value, source, note=""):
    return Fact(value, source if value is not None else "unknown", note)


@dataclass(frozen=True)
class GPU:
    index: int
    vendor: str
    name: str
    uuid: str | None
    compute_capability: Fact
    memory_total: Fact
    #: free right now, as the driver reports it -- other processes' allocations already subtracted
    memory_free: Fact
    driver: Fact
    pcie_gen_max: Fact
    pcie_width_max: Fact
    pcie_gen_current: Fact
    pcie_width_current: Fact


@dataclass(frozen=True)
class Host:
    cpu_model: Fact
    #: CPUs this process may use: a cgroup quota beats the visible core count
    cpus: Fact
    memory_total: Fact
    #: what THIS process tree can still allocate: the cgroup limit minus usage with reclaimable page cache added back,
    #: else MemAvailable. ``free``/psutil report the host, which a container cannot have.
    memory_available: Fact
    memory_limit: Fact


@dataclass(frozen=True)
class HardwareProfile:
    gpus: tuple
    host: Host
    platform: str
    notes: tuple = ()

    def gpu(self, index: int = 0) -> GPU | None:
        for g in self.gpus:
            if g.index == index:
                return g
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict, *, origin: str | None = None) -> "HardwareProfile":
        """Rebuild a profile saved with :meth:`to_dict` (``inspect --json``), e.g. to plan for another machine.
        ``origin`` is recorded in ``notes`` so a plan says it was made for hardware it did not probe itself."""
        def fact(x):
            v = x["value"]
            return Fact(tuple(v) if isinstance(v, list) else v, x["source"], x.get("note", ""))

        gpus = tuple(GPU(**{k: (fact(v) if isinstance(v, dict) and "source" in v else v) for k, v in g.items()})
                     for g in d["gpus"])
        host = Host(**{k: fact(v) for k, v in d["host"].items()})
        notes = tuple(d.get("notes", ())) + ((f"profile loaded from {origin}, not probed by this process",)
                                             if origin else ())
        return cls(gpus=gpus, host=host, platform=d["platform"], notes=notes)


# ---------------------------------------------------------------------------------------------------------------- GPU

_SMI_FIELDS = ("index", "uuid", "name", "compute_cap", "memory.total", "memory.free", "driver_version",
               "pcie.link.gen.max", "pcie.link.width.max", "pcie.link.gen.current", "pcie.link.width.current")


def _num(s):
    s = s.strip()
    if s in ("", "[N/A]", "N/A", "[Not Supported]"):
        return None
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def parse_nvidia_smi(csv_text: str) -> list:
    """Parse ``nvidia-smi --query-gpu=<_SMI_FIELDS> --format=csv,noheader,nounits`` output into :class:`GPU` rows."""
    gpus = []
    for line in csv_text.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) != len(_SMI_FIELDS):
            continue
        row = dict(zip(_SMI_FIELDS, cols))
        cc = row["compute_cap"]
        cap = tuple(int(x) for x in cc.split(".")) if cc and cc[0].isdigit() else None
        mib = lambda k: (_num(row[k]) * (1 << 20)) if _num(row[k]) is not None else None  # noqa: E731
        pcie_note = "the current link can train down at idle; read it under load before trusting it"
        gpus.append(GPU(
            index=int(row["index"]), vendor="nvidia", name=row["name"], uuid=row["uuid"] or None,
            compute_capability=_fact(cap, "reported", "nvidia-smi compute_cap"),
            memory_total=_fact(mib("memory.total"), "reported", "nvidia-smi"),
            memory_free=_fact(mib("memory.free"), "reported", "nvidia-smi, at probe time"),
            driver=_fact(row["driver_version"] or None, "reported"),
            pcie_gen_max=_fact(_num(row["pcie.link.gen.max"]), "reported"),
            pcie_width_max=_fact(_num(row["pcie.link.width.max"]), "reported"),
            pcie_gen_current=_fact(_num(row["pcie.link.gen.current"]), "reported", pcie_note),
            pcie_width_current=_fact(_num(row["pcie.link.width.current"]), "reported", pcie_note),
        ))
    return gpus


def _probe_gpus(notes: list) -> tuple:
    smi = shutil.which("nvidia-smi")
    if smi is None:
        notes.append("no nvidia-smi on PATH: no NVIDIA GPU discovered (other vendors are not probed yet)")
        return ()
    try:
        out = subprocess.run([smi, f"--query-gpu={','.join(_SMI_FIELDS)}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError) as e:
        notes.append(f"nvidia-smi failed: {e}")
        return ()
    vis = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpus = parse_nvidia_smi(out)
    if vis is not None:
        notes.append(f"CUDA_VISIBLE_DEVICES={vis!r}: indices here are nvidia-smi's, not the process's")
    return tuple(gpus)


# --------------------------------------------------------------------------------------------------------------- host

def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def cgroup_memory(root: str = "/sys/fs/cgroup") -> dict:
    """``{"limit", "usage", "reclaimable", "version"}`` for this container, or ``{}`` when unlimited/unreadable.

    v2 (``memory.max``) and v1 (``memory/memory.limit_in_bytes``) both occur in practice. The page cache is counted
    in usage by both and is reclaimable, so it is reported separately rather than treated as used.
    """
    layouts = (("v2", "memory.max", "memory.current", "memory.stat", "file"),
               ("v1", "memory/memory.limit_in_bytes", "memory/memory.usage_in_bytes", "memory/memory.stat", "cache"))
    for ver, lim, cur, stat, key in layouts:
        raw = _read(os.path.join(root, lim))
        if raw is None:
            continue
        if raw == "max" or not raw.isdigit() or int(raw) >= (1 << 62):
            return {}
        usage = _read(os.path.join(root, cur))
        cache = 0
        for line in (_read(os.path.join(root, stat)) or "").splitlines():
            k, _, v = line.partition(" ")
            if k == key and v.isdigit():
                cache = int(v)
        return {"limit": int(raw), "usage": int(usage) if usage and usage.isdigit() else None,
                "reclaimable": cache, "version": ver}
    return {}


def cgroup_cpus(root: str = "/sys/fs/cgroup"):
    raw = _read(os.path.join(root, "cpu.max"))                         # v2: "<quota> <period>" or "max <period>"
    if raw:
        q, _, p = raw.partition(" ")
        if q.isdigit() and p.isdigit():
            return int(q) / int(p)
    q, p = _read(os.path.join(root, "cpu/cpu.cfs_quota_us")), _read(os.path.join(root, "cpu/cpu.cfs_period_us"))
    if q and p and q.lstrip("-").isdigit() and int(q) > 0 and p.isdigit():
        return int(q) / int(p)
    return None


def _meminfo() -> dict:
    out = {}
    for line in (_read("/proc/meminfo") or "").splitlines():
        k, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[k] = int(parts[0]) * 1024
    return out


def _probe_host(root: str = "/sys/fs/cgroup") -> Host:
    mi = _meminfo()
    cg = cgroup_memory(root)
    cpu_model = None
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break
    quota = cgroup_cpus(root)
    visible = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    if quota is not None:
        cpus = Fact(min(quota, visible or quota), "reported", f"cgroup CPU quota {quota:g} (visible cores {visible})")
    else:
        cpus = _fact(visible, "reported", "scheduler affinity")
    if cg:
        avail = cg["limit"] - (cg["usage"] or 0) + cg["reclaimable"]
        mem_avail = Fact(max(0, avail), "reported",
                         f"cgroup {cg['version']}: limit - usage + reclaimable page cache "
                         f"({cg['reclaimable'] / GiB:.1f} GiB of it is page cache)")
        limit = Fact(cg["limit"], "reported", f"cgroup {cg['version']} memory limit")
    else:
        mem_avail = _fact(mi.get("MemAvailable"), "reported", "/proc/meminfo MemAvailable")
        limit = Fact(None, "unknown", "no cgroup memory limit found")
    return Host(cpu_model=_fact(cpu_model, "reported"), cpus=cpus,
                memory_total=_fact(mi.get("MemTotal"), "reported", "/proc/meminfo (the host's, not the container's)"),
                memory_available=mem_avail, memory_limit=limit)


def probe(*, gpu_memory: dict | None = None, ram_available: int | None = None) -> HardwareProfile:
    """Take the inventory. ``gpu_memory`` (``{index: bytes}``) and ``ram_available`` override discovery as ``user`` facts."""
    notes: list = []
    gpus = list(_probe_gpus(notes))
    for i, g in enumerate(gpus):
        if gpu_memory and g.index in gpu_memory:
            gpus[i] = replace(g, memory_free=Fact(int(gpu_memory[g.index]), "user", "caller-supplied budget"))
    host = _probe_host()
    if ram_available is not None:
        host = replace(host, memory_available=Fact(int(ram_available), "user", "caller-supplied"))
    return HardwareProfile(gpus=tuple(gpus), host=host, platform=f"{platform.system()} {platform.machine()}",
                           notes=tuple(notes))


def describe(hw: HardwareProfile) -> str:
    """The inventory as text, each value with its source."""
    def gb(f):
        return "unknown" if f.value is None else f"{f.value / GiB:.1f} GiB"
    lines = ["Hardware"]
    if not hw.gpus:
        lines.append("  GPU: none discovered")
    for g in hw.gpus:
        cc = g.compute_capability.value
        lines.append(f"  GPU {g.index}: {g.name}  (sm_{cc[0]}{cc[1]}, driver {g.driver.value})" if cc else
                     f"  GPU {g.index}: {g.name}")
        lines.append(f"    memory: {gb(g.memory_total)} total, {gb(g.memory_free)} free now [{g.memory_free.source}]")
        if g.pcie_gen_max:
            lines.append(f"    PCIe: gen{g.pcie_gen_current.value} x{g.pcie_width_current.value} now, "
                         f"gen{g.pcie_gen_max.value} x{g.pcie_width_max.value} max [reported]")
    h = hw.host
    lines.append(f"  CPU: {h.cpu_model.value}; {h.cpus.value:g} usable [{h.cpus.note}]")
    lines.append(f"  RAM: {gb(h.memory_available)} available to this process [{h.memory_available.source}: "
                 f"{h.memory_available.note}]; limit {gb(h.memory_limit)}; host total {gb(h.memory_total)}")
    lines += [f"  note: {n}" for n in hw.notes]
    return "\n".join(lines)
