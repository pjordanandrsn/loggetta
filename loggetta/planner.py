"""The planner: (topology, hardware, workload, constraints, backends, receipts) -> ExecutionPlan. Pure policy; runs
nothing.

Policy lives here and nowhere below: the device and host budgets, the headroom kept free, the runtime overhead
charged on top of a backend's estimate, which candidates are feasible, the order an objective tries them in, and
what to suggest when none fits. Mechanism costs come from the backends, which ask the packages that own them.

Deterministic: the same inputs give the same plan, byte for byte (``ExecutionPlan.to_json``).
"""
from __future__ import annotations

import inspect
import json
import math
from dataclasses import replace

from .backends import BACKENDS
from .plan import Candidate, Constraints, ExecutionPlan, MemoryLine, Workload

GiB, MiB = 1 << 30, 1 << 20

#: Charged when no receipt on file measured these for this GPU + driver + torch. Labelled "inferred" in every plan.
DEFAULT_CUDA_CONTEXT = 512 * MiB
DEFAULT_HOST_BASELINE = 3 * GiB
HEADROOM_MIN, HEADROOM_FRAC = 512 * MiB, 0.05
OBJECTIVES = ("speed", "min_vram", "min_ram")


def headroom_policy(budget: int) -> int:
    """Device memory left unused for allocator fragmentation and transient spikes: max(512 MiB, 5% of the budget)."""
    return max(HEADROOM_MIN, int(HEADROOM_FRAC * budget))


DEFAULT_RESERVE_FRAC = 0.20


def host_headroom_policy(budget: int) -> int:
    """Host memory a serve plan leaves unused: max(1 GiB, 5% of the budget). The server's CPU tier allocates compute
    buffers no estimate prices (+0.72 GiB measured during generation on OLMoE-1B-7B, RTX A2000 seat, two solver runs)."""
    return max(GiB, int(HEADROOM_FRAC * budget))


#: What a receipt may teach the planner. A receipt whose registration licenses only some uses names them in
#: ``licensed_for`` (lane SV6, experts4bit-qlora#1275: the allocator reserve and the CUDA context on its card class, and
#: nothing more); a receipt without the field teaches every use, as receipts always have.
OVERHEAD_USES = ("context", "reserve", "residual", "host_growth", "baseline", "capacity")


def licensed(obs, use):
    """Whether receipt ``obs`` may teach the planner ``use`` (one of ``OVERHEAD_USES``)."""
    scope = obs.get("licensed_for")
    return scope is None or use in scope


def _receipt_backend(obs, backends):
    """The backend whose plan a receipt ran. Receipts that do not say (written before plans named their backend, or
    imported from bench records) belong to the backend listed first, the one every early receipt came from."""
    ran = ((obs.get("plan") or {}).get("selected") or {}).get("backend") or obs.get("backend")
    return ran or (backends[0].NAME if backends else None)


def _receipt_residency(obs, backends):
    """Where a receipt's frozen weights lived, asked of the backend whose plan it ran (the first backend that can say,
    for a receipt without its plan)."""
    setup = obs.get("setup") or {}
    ran = ((obs.get("plan") or {}).get("selected") or {}).get("backend")
    for b in backends:
        if ran is None or b.NAME == ran:
            where = getattr(b, "residency", lambda s: None)(setup)
            if where is not None or ran is not None:
                return where
    return None


def _overheads(hardware, gpu, observations, backends=()):
    """Runtime costs the backends' estimates exclude, preferring measurements from receipts for this GPU + driver:
    the CUDA context (device), the allocator's reserved-but-unallocated blocks as a fraction of the allocator peak
    (device), and the process's own host baseline (anonymous RSS after load, from a run whose frozen weights were all on
    the device). Returns (device lines, host lines, reserve fraction, its line meta)."""
    key = (gpu.name if gpu else None, gpu.driver.value if gpu else None)
    for obs in observations:
        m = obs.get("measured", {})
        g = obs.get("hardware", {}).get("gpu", {})
        if (g.get("name"), g.get("driver")) != key or m.get("cuda_context_bytes") is None \
                or not licensed(obs, "context"):
            continue
        src = f"receipt {obs.get('run_id')}"
        alloc, reserved = m.get("device_peak_bytes"), m.get("device_reserved_peak_bytes")
        reserve_ok = bool(alloc and reserved and licensed(obs, "reserve"))
        frac = (reserved - alloc) / alloc if reserve_ok else DEFAULT_RESERVE_FRAC
        base = m.get("host_anon_after_load_bytes") if _receipt_residency(obs, backends) == "device" \
            and licensed(obs, "baseline") else None
        hb = m.get("host_baseline_bytes") if licensed(obs, "baseline") else None
        return ([MemoryLine("CUDA context + library workspaces", "device", max(0, int(m["cuda_context_bytes"])),
                            "measured", f"{src}: driver-reported process peak minus allocator reserved peak")],
                [MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host",
                            int(base or hb or DEFAULT_HOST_BASELINE),
                            "measured" if (base or hb) else "inferred",
                            f"{src}: anonymous RSS after load" if base else src)],
                frac, ("measured", f"{src}: reserved peak / allocated peak - 1 = {frac:.3f}") if reserve_ok else
                ("inferred", f"default {DEFAULT_RESERVE_FRAC:.0%} of the allocator estimate; {src} does not give it"))
    return ([MemoryLine("CUDA context + library workspaces", "device", DEFAULT_CUDA_CONTEXT, "inferred",
                        "default; no receipt on file measured this GPU + driver")],
            [MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host", DEFAULT_HOST_BASELINE,
                        "inferred", "default; no receipt on file measured this host")],
            DEFAULT_RESERVE_FRAC, ("inferred", f"default {DEFAULT_RESERVE_FRAC:.0%} of the allocator estimate; "
                                               "no receipt on file measured this GPU + driver"))


#: PCIe payload bandwidth per lane, GB/s, by generation (after line coding): the theoretical ceiling, never achieved
PCIE_LANE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def reserve_fraction(gpu, setup, observations, default, model=None, kind="train", key=(), budget_fields=()):
    """Allocator reserve slack (reserved peak / allocated peak - 1) for one candidate, from receipts. Returns
    (fraction, basis, source).

    Slack depends on the allocation pattern (offload's per-layer staging leaves more cached blocks), on the model and on
    the GPU: OLMoE's resident slack measured 0.222 on an RTX A2000 and 0.150 on an RTX 5090. In order:

    1. a receipt for this GPU, this residency + kernel and this model: those with the candidate's whole setup first,
       then those that differ from it only in ``budget_fields`` (the tier budgets the planner sizes: the same workload
       shape), then any; of several, the largest slack: measured;
    2. training only: this model + setup measured on ANOTHER GPU, scaled by an anchor model measured with the same setup
       on both GPUs (slack(model, here) = slack(model, there) x slack(anchor, here) / slack(anchor, there)): a stated
       transfer. Not for serving: its slack is 0.06-1.5%, so the anchor ratio divides one measurement's noise by
       another's (a 2.8% OLMoE slack on an RTX 5090 came out as 4.0 GiB of reserve on an RTX 4090, and gpt-oss-20b's
       as 10.4 GiB, through a ~14x ratio of two Qwen3-30B slacks under 1%);
    3. this GPU + setup, another model: measured, but for a different model;
    4. the largest slack measured on this GPU, conservative (a smaller borrowed figure would make the unmeasured
       candidate look cheaper than the measured one);
    5. serving only: the largest slack measured for this setup on any GPU (heuristic: borrowed across cards; all-VRAM
       serving measured 0.06-1.46% on two cards, where training slack moved 8-39% with model and card);
    6. ``default``.

    Only receipts of the same workload ``kind`` count: a server allocates its pools once, a trainer churns activations
    every step, so one's slack says nothing about the other's. Receipts without a kind are training receipts. ``key``
    names the setup fields that separate allocation patterns (the backend's ``SLACK_KEYS``): measured, a server's
    tiered placement leaves 15% slack where its all-VRAM placement leaves 1-2%.

    A receipt scoped by ``licensed_for`` is same-setup evidence (lane SV6's licence, experts4bit-qlora#1275): its slack
    counts only for a candidate whose whole setup it ran, never borrowed for another.
    """
    def frac(o):
        m = o["measured"]
        return m["device_reserved_peak_bytes"] / m["device_peak_bytes"] - 1

    def gname(o):
        return o.get("hardware", {}).get("gpu", {}).get("name")

    def mname(o):
        return o.get("model", {}).get("model")

    def sj(s, drop=()):
        return json.dumps({k: v for k, v in (s or {}).items() if k not in drop}, sort_keys=True, default=list)

    whole, shape = sj(setup), sj(setup, budget_fields)
    usable = [o for o in observations if o.get("status") in ("OK", None) and licensed(o, "reserve")
              and (o.get("licensed_for") is None or sj(o.get("setup")) == whole)
              and o.get("measured", {}).get("device_peak_bytes")
              and o.get("measured", {}).get("device_reserved_peak_bytes")
              and o.get("workload", {}).get("kind", "train") == kind]
    same_setup = [o for o in usable if {k: o.get("setup", {}).get(k) for k in key} == {k: setup.get(k) for k in key}]
    same_setup.sort(key=lambda o: sj(o.get("setup")) != whole)

    def tier(group):
        """Among receipts with the same key fields: the candidate's whole setup, else its shape (only the budgets
        differ), else any; of several, the largest slack (repeats differ, and a setting the setup does not name can
        change it). Returns (group, how it matched)."""
        for match, how in ((lambda o: sj(o.get("setup")) == whole, "setup"),
                           (lambda o: sj(o.get("setup"), budget_fields) == shape, "shape, other tier budgets")):
            top = [o for o in group if match(o)]
            if top and (how == "setup" or budget_fields):
                return top, how
        return group, "key fields"

    def pick(group):
        return max(tier(group)[0], key=frac)

    here = [o for o in same_setup if gname(o) == gpu.name]
    exact = [o for o in here if model and mname(o) == model]
    if exact:
        top, how = tier(exact)
        e = max(top, key=frac)
        where = "this GPU, setup and model" if how == "setup" else f"this GPU and model, same {how}"
        return frac(e), "measured", f"receipt {e.get('run_id')} ({where}; largest of {len(top)}) = {frac(e):.3f}"
    if model and kind != "serve":
        for there in (o for o in same_setup if mname(o) == model and gname(o) != gpu.name):
            for anchor_here in here:
                anchor_there = next((o for o in same_setup if gname(o) == gname(there) and mname(o) == mname(anchor_here)), None)
                if anchor_there and frac(anchor_there) > 0:
                    f = frac(there) * frac(anchor_here) / frac(anchor_there)
                    return f, "heuristic", (f"transferred: {frac(there):.3f} measured for this model on {gname(there)} "
                                            f"(receipt {there.get('run_id')}) x anchor {mname(anchor_here)} "
                                            f"{frac(anchor_here):.3f} here / {frac(anchor_there):.3f} there = {f:.3f}")
    if here:
        h = pick(here)
        return frac(h), "measured", (f"receipt {h.get('run_id')} (this GPU and setup, model {mname(h)}) = {frac(h):.3f}")
    same_gpu = [o for o in usable if gname(o) == gpu.name]
    if same_gpu:
        worst = max(same_gpu, key=frac)
        return frac(worst), "measured", (f"no receipt for this setup; the largest slack measured on this GPU, receipt "
                                         f"{worst.get('run_id')} = {frac(worst):.3f} (conservative)")
    if same_setup and kind == "serve":              # training slack varies 8-39% by model and card; serving's did not
        worst = max(same_setup, key=frac)
        return frac(worst), "heuristic", (f"no receipt on this GPU; the largest slack measured for this setup on any GPU, "
                                          f"receipt {worst.get('run_id')} on {gname(worst)} = {frac(worst):.3f} "
                                          "(borrowed across cards, conservative)")
    return default[0], default[1], default[2]


_GIB_RE = r"([0-9.]+) (GiB|MiB)"


def _oom_numbers(error: str):
    """``(tried, capacity)`` in bytes from a CUDA out-of-memory message (``Tried to allocate X``, ``total capacity of
    Y``); None where absent."""
    import re

    def grab(pat):
        m = re.search(pat, error or "")
        return int(float(m.group(1)) * (GiB if m.group(2) == "GiB" else 1 << 20)) if m else None
    return grab(r"Tried to allocate " + _GIB_RE), grab(r"total capacity of " + _GIB_RE)


def usable_capacity(gpu, observations):
    """``(bytes, run_id)``: the smallest device capacity a CUDA out-of-memory message reported for this GPU class, or
    None. The driver's total over-states what a process gets: lane SV5's RTX 4090 reports 24,564 MiB and gave the
    process 23.52 GiB, and a plan sized against 24 GiB ran out of memory."""
    best = None
    for o in observations:
        if o.get("hardware", {}).get("gpu", {}).get("name") != gpu.name or not licensed(o, "capacity"):
            continue
        _tried, cap = _oom_numbers(o.get("error") or "")
        if cap and (best is None or cap < best[0]):
            best = (cap, o.get("run_id"))
    return best


def learned_serve_overheads(backend, topology, setup, observations, key):
    """Two overheads a serve plan learns from serve receipts of THIS model, when there are any:

    * the allocator residual: a receipt's measured allocator peak minus what the backend estimates today for that
      receipt's own setup. Measured 0.15-0.21 GiB on every serve run so far (two models, two cards, both placements),
      a runtime term no item prices. The largest such residual is charged (conservative);
    * host growth while serving: the anonymous host memory a run gained after load (the CPU tier's compute buffers,
      +0.72 GiB on OLMoE under the solver), from receipts with the candidate's key fields; the largest is charged.

    Returns MemoryLine-shaped tuples ``(name, where, bytes, basis, detail)``."""
    from .plan import Workload

    out, best, best_key, grow = [], None, None, None
    for o in observations:
        if o.get("workload", {}).get("kind") != "serve" or o.get("status") not in ("OK", "OOM") \
                or o.get("model", {}).get("model") != topology.model:
            continue
        m, rs = o.get("measured", {}), dict(o.get("setup", {}))
        if o.get("status") == "OOM":
            # an out-of-memory run's allocator peak plus the allocation that failed is a LOWER bound on that setup's
            # need: charged where it exceeds today's estimate, like a residual (lane SV5)
            tried, _cap = _oom_numbers(o.get("error") or "")
            if not (m.get("device_peak_bytes") and tried):
                continue
            m = {"device_peak_bytes": m["device_peak_bytes"] + tried}
        if m.get("device_peak_bytes") and licensed(o, "residual"):
            try:
                raw, _, refusals = backend.estimate(topology, rs, Workload(kind="serve"))
            except (TypeError, ValueError, KeyError):
                raw, refusals = (), ("not estimable here",)
            if not refusals:
                r = m["device_peak_bytes"] - sum(x[2] for x in raw if x[1] == "device")
                if best is None or r > best[0]:
                    best = (r, o.get("run_id"))
                if all(str(rs.get(k)) == str(setup.get(k)) for k in key) and (best_key is None or r > best_key[0]):
                    best_key = (r, o.get("run_id"))          # same placement / graphs / prefill graph: their pools too
        # serving's own peak where the receipt has one: a load can peak above everything after it (the int4 repack's
        # host buffers, handed back before serving), and that is the load's to price, not growth while serving
        peak = m.get("host_anon_serving_peak_bytes") or m.get("host_anon_peak_bytes")      # (absent on an OOM)
        if peak and m.get("host_anon_after_load_bytes") and licensed(o, "host_growth") \
                and all(rs.get(k) == setup.get(k) for k in key):
            g = peak - m["host_anon_after_load_bytes"]
            if grow is None or g > grow[0]:
                grow = (g, o.get("run_id"))
    if best_key is not None:
        best, scope = best_key, f"this model with the same {', '.join(key)}"
    else:
        scope = "this model (none with the same " + ", ".join(key) + ")"
    if best and best[0] > 0:
        out.append(("allocator residual (runtime buffers no item prices)", "device", int(best[0]), "measured",
                    f"receipt {best[1]}: its allocator peak minus today's estimate of its own setup (largest on file "
                    f"for {scope})"))
    if grow and grow[0] > 0:
        out.append(("host growth while serving", "host", int(grow[0]), "measured",
                    f"receipt {grow[1]}: anonymous host memory gained after load (largest on file for this model and "
                    f"{', '.join(key)})"))
    return out


def link_bandwidth(gpu, observations):
    """(GB/s, basis, source) for host-to-device copies: a receipt's measurement for this GPU, else the PCIe ceiling."""
    for obs in observations:
        g = obs.get("hardware", {}).get("gpu", {})
        v = obs.get("measured", {}).get("link_h2d_gbps")
        if v and g.get("name") == gpu.name:
            return float(v), "measured", f"pinned host-to-device copy, receipt {obs.get('run_id')}"
    gen, width = gpu.pcie_gen_max.value, gpu.pcie_width_max.value
    if gen in PCIE_LANE_GBPS and width:
        return PCIE_LANE_GBPS[gen] * width, "inferred", f"theoretical PCIe gen{gen} x{width} ceiling"
    return None, "unknown", "no PCIe facts and no measurement"


def _observed(observations, model, setup, workload, gpu):
    for obs in observations:
        if (obs.get("model", {}).get("model") == model and obs.get("setup") == setup
                and obs.get("workload", {}).get("tokens_per_microbatch") == workload.tokens_per_microbatch
                and obs.get("hardware", {}).get("gpu", {}).get("name") == (gpu.name if gpu else None)
                and obs.get("status") == "OK"):
            return obs
    return None


def _hw_dict(hardware, gpu) -> dict:
    h = hardware.host
    g = None if gpu is None else {
        "index": gpu.index, "name": gpu.name, "uuid": gpu.uuid, "compute_capability": gpu.compute_capability.value,
        "driver": gpu.driver.value, "memory_total": gpu.memory_total.value, "memory_free": gpu.memory_free.value,
        "memory_free_source": gpu.memory_free.source,
        "pcie": [gpu.pcie_gen_current.value, gpu.pcie_width_current.value, gpu.pcie_gen_max.value,
                 gpu.pcie_width_max.value]}
    return {"gpu": g, "gpus": len(hardware.gpus), "cpu": h.cpu_model.value, "cpus": h.cpus.value,
            "ram_available": h.memory_available.value, "ram_available_source": h.memory_available.source,
            "ram_limit": h.memory_limit.value, "platform": hardware.platform}


def _rank(objective, setup, dev, host, backend):
    if objective == "min_vram":
        return (dev, host), "objective min_vram: least device memory first"
    if objective == "min_ram":
        return (host, dev), "objective min_ram: least host memory first"
    return backend.speed_rank(setup), "objective speed: ordered by measured evidence (see Why)"


def resolve_data(workload: Workload, profile: dict | None):
    """What the training data's profile means for the plan: ``steps`` from ``epochs``, whether the data covers what
    the plan reads, and the examples packing splits or truncates. Pure arithmetic on the profile.

    Concatenated packing reads tokens; isolated packing reads whole packed rows (the profile's packing at this seq).
    Returns ``(workload, reasons, warnings, refusal)``; ``refusal`` is ``(reasons, suggestions)`` or None."""
    if profile is None or workload.kind != "train":
        return workload, [], [], None
    from .data import longer_than

    if workload.data is not None and profile.get("options") != workload.data:
        raise ValueError("the data profile was made with other dataset options than the workload's")
    isolated = profile.get("packing_mode") == "isolated"
    tokens = profile["tokens"]
    if isolated:
        packed = profile.get("packed") or {}
        if packed.get("seq_len") != workload.seq_len:
            raise ValueError(f"the data profile was packed for seq {packed.get('seq_len')}, not {workload.seq_len}: "
                             "profile the data again at this seq")
        supply, per_step, unit = packed["rows"], workload.micro_batch * workload.grad_accum, "packed rows"
        per_step_text = "rows per optimizer step (micro-batch x grad-accum)"
        step_parts = "micro-batch or grad-accum"
    else:
        supply, per_step, unit = tokens, workload.tokens_per_microbatch * workload.grad_accum, "tokens"
        per_step_text = "tokens per optimizer step (seq x micro-batch x grad-accum)"
        step_parts = "seq, micro-batch or grad-accum"
    reasons, warnings = [], []
    if workload.epochs is not None:
        workload = replace(workload, steps=max(1, int(workload.epochs * supply // per_step)))
        reasons.append(f"steps {workload.steps}: {workload.epochs:g} epoch(s) of {supply:,} {unit} at {per_step:,} "
                       f"{per_step_text}")
    needed = workload.steps * per_step
    repeat = bool((workload.data or {}).get("repeat"))
    allowed = math.inf if repeat else (math.ceil(workload.epochs) if workload.epochs is not None else 1)
    passes = needed / supply
    held = "the dataset holds" if not isolated else "the dataset fills"
    reasons.append(f"data: this plan reads {needed:,} {unit} of the {supply:,} {held} ({passes:.2f} passes"
                   + (", repetition allowed" if repeat else "") + ")")
    if isolated:
        reasons.append(f"isolated packing: {supply:,} rows of {workload.seq_len:,} tokens, {packed['efficiency']:.0%} "
                       "filled; each example attends only to itself, and its first token is never a target")
    if profile.get("loss_mode") == "assistant":
        share = profile["loss_tokens"] / tokens
        reasons.append(f"loss on {profile['loss_tokens']:,} of {tokens:,} tokens ({share:.0%}): {profile['loss']}; "
                       "gradients are averaged over trained tokens across the whole optimizer step")
    if isolated:
        if packed["truncated_examples"]:
            warnings.append(f"{packed['truncated_examples']:,} of {profile['rows']:,} examples are longer than seq "
                            f"{workload.seq_len:,} tokens: isolated packing truncates them, dropping "
                            f"{packed['truncated_tokens']:,} tokens ({packed['truncated_loss_tokens']:,} of them trained)")
    else:
        sure, most = longer_than(profile, workload.seq_len)
        if most:
            count = f"{sure:,}" if sure == most else f"{sure:,}-{most:,}"
            warnings.append(f"{count} of {profile['rows']:,} examples are longer than seq {workload.seq_len:,} tokens: "
                            "packing always splits them across rows, so their later tokens train on a cut context")
    refusal = None
    if passes > allowed:
        if supply < per_step:
            formula = "micro-batch x grad-accum" if isolated else "seq x micro-batch x grad-accum"
            why = (f"the dataset {'packs into' if isolated else 'holds'} {supply:,} {unit}, less than one optimizer "
                   f"step reads ({per_step:,} = {formula})")
        else:
            why = (f"the dataset {'packs into' if isolated else 'holds'} {supply:,} {unit}; this plan reads {needed:,} "
                   f"({passes:.2f} passes) and repeating it is not allowed")
        suggestions = []
        if supply >= per_step:
            suggestions.append(f"--steps {supply // per_step} reads it once")
        suggestions += ["--epochs N reads it N times, on purpose", "--repeat-data allows repeating it"]
        if supply < per_step:
            suggestions.insert(0, f"a shorter {step_parts}")
        refusal = ([why], suggestions)
    return workload, reasons, warnings, refusal


#: bytes per element of a packed row's attention mask: transformers' boolean [rows, 1, seq, seq] mask, plus the bf16
#: bias SDPA's memory-efficient kernel can turn it into (flash attention cannot take a mask). Measured at seq 2048 on
#: an RTX A2000 (evidence/2026-10-07-a2000-packing-ab): the allocator peak moved by exactly the boolean mask, 1 B per
#: element; the bias is charged as well, conservatively, until a longer-seq receipt shows whether it meets the peak
PACKED_MASK_BYTES = 3


def data_memory_lines(workload: Workload, profile: dict | None) -> list:
    """Device memory the data's packing costs beyond what a backend prices: isolated packing's attention masks."""
    if profile is None or workload.kind != "train" or profile.get("packing_mode") != "isolated":
        return []
    n = workload.micro_batch * workload.seq_len * workload.seq_len * PACKED_MASK_BYTES
    return [MemoryLine("packed-example attention masks", "device", n, "heuristic",
                       f"micro-batch x seq^2 x {PACKED_MASK_BYTES} B: a boolean mask transformers builds for packed "
                       "rows, and the bf16 bias SDPA's memory-efficient kernel makes of it; flash attention cannot run "
                       "with a mask, so attention is slower (not modelled)")]


def plan(topology, hardware, workload: Workload, constraints: Constraints = Constraints(), *, backends=None,
         observations=(), data_profile: dict | None = None) -> ExecutionPlan:
    """The ExecutionPlan for ``workload`` on ``hardware``. ``data_profile`` is the training data's profile
    (``data.encode_dataset``), read before planning; with it the plan states what the data holds and refuses data that
    cannot cover the plan, and ``epochs`` resolve to steps."""
    backends = BACKENDS if backends is None else backends
    if constraints.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    gpu = hardware.gpu(constraints.device)
    workload, reasons, warnings, data_refusal = resolve_data(workload, data_profile)
    data_lines = data_memory_lines(workload, data_profile)
    if workload.data is not None:
        warnings.append("Data preparation buffers and temporary token-file disk space are not included in the "
                        "training memory estimate; data is validated before "
                        + ("planning and checked again before model loading" if data_profile else "model loading")
                        + ". A feasible plan is an estimate, not a guarantee against out-of-memory errors.")
    dev_budget = constraints.vram_budget
    dev_src = "user"
    if dev_budget is None:
        dev_budget = (gpu.memory_free.value or 0) if gpu else 0
        dev_src = f"{gpu.memory_free.source}: free now" if gpu else "no GPU"
    elif gpu and gpu.memory_free.value is not None and dev_budget > gpu.memory_free.value:
        warnings.append(f"device budget {dev_budget / GiB:.2f} GiB exceeds what the driver reports free now "
                        f"({gpu.memory_free.value / GiB:.2f} GiB): another process holds the difference")
    cap = usable_capacity(gpu, observations or ()) if gpu else None
    if cap:
        others = max(0, (gpu.memory_total.value or 0) - (gpu.memory_free.value or 0)) if gpu.memory_free.value is not None else 0
        if dev_budget > cap[0] - others:
            dev_budget = cap[0] - others
            dev_src += (f"; capped at the {cap[0] / GiB:.2f} GiB a CUDA process got on this GPU class "
                        f"(receipt {cap[1]}'s out-of-memory message)")
    if gpu and gpu.memory_total.value and gpu.memory_free.value is not None:
        used = gpu.memory_total.value - gpu.memory_free.value
        if used > 1 * GiB:
            warnings.append(f"{used / GiB:.2f} GiB of {gpu.name} is already in use by other processes")
    host_budget = constraints.ram_budget
    host_src = "user"
    if host_budget is None:
        host_budget = hardware.host.memory_available.value or 0
        host_src = f"{hardware.host.memory_available.source}: available now"
    headroom = constraints.headroom if constraints.headroom is not None else headroom_policy(dev_budget)
    host_headroom = host_headroom_policy(host_budget) if workload.kind == "serve" else 0
    budget = {"device": int(dev_budget), "device_source": dev_src, "host": int(host_budget),
              "host_source": host_src, "headroom": int(headroom)}
    if host_headroom:
        budget["host_headroom"] = int(host_headroom)
    # each backend says whether it can plan for this description at all (its own admission), before anything is priced
    from .model import Refused

    if isinstance(topology, Refused):                  # every backend that described it refused: their own words
        refusals = [(b, topology.reasons.get(b.NAME) or b.refusal(topology.description)) for b in backends]
        owner = next((b for b in backends if b.NAME == topology.backend), backends[0] if backends else None)
        topology = topology.description
    else:
        refusals = [(b, b.refusal(topology)) for b in backends]
        owner = None
    admitted = [b for b, why in refusals if why is None]
    owner = admitted[0] if admitted else (owner or (backends[0] if backends else None))
    if owner is not None:
        warnings += getattr(owner, "planning_warnings", lambda *a: [])(topology)
    common = dict(model=owner.summary(topology) if owner else {"model": getattr(topology, "model", None)},
                  hardware=_hw_dict(hardware, gpu), workload=workload, constraints=constraints, budget=budget,
                  data_profile=data_profile)

    def refuse(why, **extra):
        return ExecutionPlan(status="refused", selected=None, alternatives=extra.pop("alternatives", ()),
                             reasons=tuple(reasons), warnings=tuple(warnings),
                             refusal={"reasons": list(why), **extra}, **common)

    if data_refusal is not None:                # the cheapest answer: nothing about the machine or model changes it
        return refuse(data_refusal[0], suggestions=data_refusal[1])
    if gpu is None:
        return refuse([f"no GPU at index {constraints.device}: every installed backend needs one"])
    if backends and not admitted:
        return refuse([why if len(refusals) == 1 else f"{b.NAME}: {why}" for b, why in refusals])
    usable = [b for b in admitted if workload.kind in b.WORKLOADS]
    if not usable and admitted:
        names = " and ".join(f"the {b.NAME} backend" for b in admitted)
        return refuse([f"{names} described this model but {'does' if len(admitted) == 1 else 'do'} not plan "
                       f"{workload.kind!r} workloads yet ({owner.summary(topology).get('summary', '')})"])
    if data_profile is not None and data_profile.get("packing_mode") == "isolated" and workload.kind == "train":
        cannot = [(b, getattr(b, "isolation_refusal", lambda t: None)(topology)) for b in usable]
        if usable and all(why for _b, why in cannot):
            return refuse([why if len(cannot) == 1 else f"{b.NAME}: {why}" for b, why in cannot],
                          suggestions=["--packing concat (attention then crosses examples within a row)"])
        usable = [b for b, why in cannot if not why]
    if not usable:
        return refuse([f"no installed backend plans {workload.kind!r} workloads yet "
                       f"(backends: {', '.join(b.NAME for b in backends)}; they plan {sorted({w for b in backends for w in b.WORKLOADS})})"])

    dev_over, host_over, reserve_frac, reserve_meta = _overheads(hardware, gpu, observations, backends)
    link_gbps, link_basis, link_src = link_bandwidth(gpu, observations)
    cands, statuses, own_fracs = [], {}, {}
    for b in usable:
        st = b.probe(gpu)
        statuses[b.NAME] = st
        if not st.available:
            reasons.append(f"backend {b.NAME} unavailable: {st.reason}")
            continue
        reasons += b.policy_notes(topology, constraints)
        wanted = getattr(b, "KERNELS_FOR", {}).get(workload.kind)
        for k, (ok, why) in sorted(st.kernels.items()):
            if not ok and (wanted is None or k in wanted):
                reasons.append(f"{b.NAME}: kernel {k} not usable here: {why}")
        slack_key = getattr(b, "SLACK_KEYS", {}).get(workload.kind, ())
        # a receipt written before the backend grew a setup field ran with that field's default: read it so, or a
        # key that names the field would never match the receipts that predate it
        key_defaults = getattr(b, "SLACK_DEFAULTS", {}).get(workload.kind, {})
        # allocation patterns are a backend's own: a dense trainer must not borrow a MoE offload run's reserve slack
        own = [o for o in observations if _receipt_backend(o, backends) == b.NAME]
        # the fallback slack too: the CUDA context and host baseline are the GPU's, the reserve is the backend's
        _d, _h, own_frac, own_meta = _overheads(hardware, gpu, own, backends)
        own_fracs[b.NAME] = own_frac
        b_obs = [{**o, "setup": {**key_defaults, **o["setup"]}} if key_defaults and isinstance(o.get("setup"), dict)
                 and o.get("workload", {}).get("kind", "train") == workload.kind else o for o in own]
        learned_cache = {}

        def price(setup):
            raw, unmodelled, refusals = b.estimate(topology, setup, workload)
            raw = list(raw) + [(x.name, x.where, x.bytes, x.basis, x.detail) for x in data_lines]
            # matched against receipts as they are read (defaults filled): a backend release whose setup lacks a key
            # field ran with its default too, so the candidate reads the same way
            keyed = {**key_defaults, **setup}
            alloc = sum(r[2] for r in raw if r[1] == "device")
            default = (own_frac, *own_meta)
            if workload.kind != "train":
                default = (DEFAULT_RESERVE_FRAC, "inferred", f"default {DEFAULT_RESERVE_FRAC:.0%} of the allocator "
                           f"estimate; no {workload.kind} receipt on file measured this GPU")
            frac, fbasis, fsrc = reserve_fraction(gpu, keyed, b_obs, default, model=topology.model,
                                                  kind=workload.kind, key=slack_key,
                                                  budget_fields=getattr(b, "BUDGET_FIELDS", {}).get(workload.kind, ()))
            reserve = MemoryLine("allocator reserve (cached, unallocated blocks)", "device", int(frac * alloc),
                                 fbasis, fsrc)
            learned = ()
            if workload.kind == "serve":
                ck = (keyed.get("placement"), tuple(keyed.get(k) for k in slack_key))
                if ck not in learned_cache:
                    learned_cache[ck] = learned_serve_overheads(b, topology, keyed, b_obs, slack_key)
                learned = tuple(MemoryLine(*x) for x in learned_cache[ck])
            lines = tuple(MemoryLine(*r) for r in raw) + (reserve,) + learned + tuple(dev_over) + tuple(host_over)
            dev = sum(ln.bytes for ln in lines if ln.where == "device")
            host = sum(ln.bytes for ln in lines if ln.where == "host")
            return lines, dev, host, unmodelled, refusals

        def candidate_headroom(setup, device_bytes):
            return max(headroom, getattr(b, "minimum_headroom", lambda *a: 0)(setup, device_bytes))

        def side_fits(setup, side):
            lines, dev, host, _, refusals = price(setup)
            if refusals:
                return False
            return dev + candidate_headroom(setup, dev) <= dev_budget if side == "device" else host + host_headroom <= host_budget

        for setup in b.candidates(topology, workload, constraints, st):
            # knobs a backend asks the planner to size: the largest value its side of the budget allows (policy).
            # Two passes: a knob sized before a later one is filled sees the later one at zero (the VRAM tier sized
            # with no DRAM tier also pays for NVMe-only items), so the second pass re-sizes each with the others set.
            knobs = getattr(b, "fill_knobs", lambda *a: [])(topology, setup, workload)
            for _pass in range(2 if len(knobs) > 1 else 1):
                for field, side, hi in knobs:
                    if field in constraints.fixed:
                        continue
                    lo, top = 0.0, float(hi)
                    if side_fits({**setup, field: top}, side):
                        lo = top
                    else:
                        for _ in range(24):
                            mid = (lo + top) / 2
                            lo, top = (mid, top) if side_fits({**setup, field: mid}, side) else (lo, mid)
                    setup = {**setup, field: int(lo * 1000) / 1000}
            for field, side, _hi in knobs:
                if field not in constraints.fixed:
                    reasons.append(f"{b.NAME}: {field} sized to {setup[field]:.3f}, the largest the {side} budget allows")
            if hasattr(b, "resolve"):              # fields the candidate left to the mechanism, now concrete
                setup = b.resolve(topology, setup, workload)
            lines, dev, host, unmodelled, refusals = price(setup)
            rejected = list(refusals)
            required_headroom = candidate_headroom(setup, dev)
            bounds = ({"device_headroom_bytes": required_headroom} if required_headroom != headroom else {})
            link = sum(ln.bytes for ln in lines if ln.where == "link") * workload.grad_accum
            if link and link_gbps:
                bounds.update({"link_bytes_per_step": link, "link_gbps": link_gbps, "link_gbps_basis": link_basis,
                          "link_gbps_source": link_src, "s_per_step_lower_bound": link / (link_gbps * 1e9)})
            if not refusals and "s_per_step_lower_bound" in bounds and constraints.target_s_per_step is not None and \
                    bounds["s_per_step_lower_bound"] > constraints.target_s_per_step:
                rejected.append(f"host-to-device traffic alone needs >= {bounds['s_per_step_lower_bound']:.2f} s/step "
                                f"({link / 1e9:.1f} GB over <= {link_gbps:.1f} GB/s, {link_basis}); target is "
                                f"{constraints.target_s_per_step:g} s/step")
            if not refusals:
                if dev + required_headroom > dev_budget:
                    rejected.append(f"device {dev / GiB:.2f} + headroom {required_headroom / GiB:.2f} GiB > budget "
                                    f"{dev_budget / GiB:.2f} GiB")
                if host + host_headroom > host_budget:
                    rejected.append(f"host {host / GiB:.2f}" + (f" + headroom {host_headroom / GiB:.2f}" if host_headroom
                                                                else "") + f" GiB > budget {host_budget / GiB:.2f} GiB")
            rank, note = _rank(constraints.objective, setup, dev, host, b)
            obs = _observed(observations, topology.model, setup, workload, gpu)
            if obs:
                note += (f"; measured before: device peak {obs['measured']['device_peak_bytes'] / GiB:.2f} GiB "
                         f"(receipt {obs.get('run_id')})")
            # the planner charges the context, allocator reserve and process baseline itself: drop the backend's note
            unmodelled = tuple(u for u in unmodelled if not u.startswith("CUDA context"))
            cands.append(Candidate(backend=b.NAME, setup=setup, lines=lines, device_bytes=dev, host_bytes=host,
                                   label_text=b.label(setup),
                                   feasible=not rejected, rejected=tuple(rejected), unmodelled=tuple(unmodelled),
                                   rank=rank, rank_note=note, bounds=bounds))
    if not cands:
        return refuse(["no backend produced a candidate"])
    feasible = sorted((c for c in cands if c.feasible), key=lambda c: (c.rank, c.backend))
    infeasible = sorted((c for c in cands if not c.feasible), key=lambda c: (c.device_bytes, c.host_bytes))
    perf = {"statement": "not predicted: no performance model is calibrated for this backend yet; the ordering "
                         "below rests on measured relative speeds, not on a time estimate", "estimate": None}

    if not feasible:
        valid = [c for c in infeasible if not any("refus" in r or "cannot" in r or "must be" in r for r in c.rejected)
                 and all(r.startswith(("device ", "host ", "host-to-device traffic")) for r in c.rejected)]
        closest = min(valid or infeasible, key=lambda c: (c.device_bytes, c.host_bytes))
        closest_backend = next(b for b in usable if b.NAME == closest.backend)
        warnings += getattr(closest_backend, "scope_warnings", lambda *a: [])(topology, closest.setup, workload)
        budget["headroom"] = closest.bounds.get("device_headroom_bytes", headroom)
        why = [f"{closest.label()}: " + "; ".join(closest.rejected)]
        if not valid:
            why = sorted({r for c in infeasible for r in c.rejected})
        suggestions = _suggest(topology, workload, constraints, budget, closest, usable, statuses, dev_over, host_over,
                               own_fracs.get(closest.backend, reserve_frac))
        return ExecutionPlan(status="refused", selected=None, alternatives=tuple(infeasible), reasons=tuple(reasons),
                             warnings=tuple(warnings), performance=perf,
                             refusal={"reasons": why, "closest": closest.label(),
                                      "closest_lines": [f"{ln.where:6s} {ln.name:37s} {ln.bytes / GiB:6.2f} GiB [{ln.basis}]"
                                                        for ln in closest.lines],
                                      "suggestions": suggestions}, **common)

    sel = feasible[0]
    budget["headroom"] = sel.bounds.get("device_headroom_bytes", headroom)
    st = statuses[sel.backend]
    b = next(x for x in usable if x.NAME == sel.backend)
    kw = {"topology": topology} if "topology" in inspect.signature(b.explain).parameters else {}
    reasons += b.explain(sel, feasible, infeasible, budget, st, constraints, workload, **kw)
    perf = getattr(b, "performance", lambda *a: None)(topology, sel.setup, workload, gpu, observations, st) or perf
    warnings += getattr(b, "plan_warnings", lambda *a: [])(sel.setup, gpu)
    warnings += getattr(b, "scope_warnings", lambda *a: [])(topology, sel.setup, workload)
    return ExecutionPlan(status="feasible", selected=sel, alternatives=tuple(feasible[1:]) + tuple(infeasible),
                         reasons=tuple(reasons), warnings=tuple(warnings), performance=perf,
                         provenance={"backend_versions": st.versions}, **common)


def _suggest(topology, workload, constraints, budget, closest, backends, statuses, dev_over, host_over,
             reserve_frac=DEFAULT_RESERVE_FRAC):
    if workload.kind == "serve":
        return _suggest_serve(topology, workload, budget, closest, backends, dev_over, reserve_frac)
    out = []
    dev_short = closest.device_bytes + budget["headroom"] - budget["device"]
    if dev_short > 0:
        out.append(f"{dev_short / GiB:.2f} GiB more device memory (a larger --vram budget, or free what other "
                   "processes hold)")
    if closest.host_bytes > budget["host"]:
        out.append(f"{(closest.host_bytes - budget['host']) / GiB:.2f} GiB more host memory")
    for b in backends:
        for words, setup in getattr(b, "relaxed_candidates", lambda *a: [])(topology, workload, constraints,
                                                                            statuses[b.NAME]):
            raw, _, refusals = b.estimate(topology, setup, workload)
            dev = int((1 + reserve_frac) * sum(r[2] for r in raw if r[1] == "device")) + sum(x.bytes for x in dev_over)
            host = sum(r[2] for r in raw if r[1] == "host") + sum(x.bytes for x in host_over)
            margin = max(budget["headroom"], getattr(b, "minimum_headroom", lambda *a: 0)(setup, dev))
            if not refusals and dev + margin <= budget["device"] and host <= budget["host"]:
                out.append(f"{words}: {dev / GiB:.2f} GiB device + {host / GiB:.2f} GiB host fits")
                break
        else:
            continue
        break
    # the largest power-of-two token count at which the closest setup fits the device budget
    b = next((x for x in backends if x.NAME == closest.backend), None)
    if b is not None and dev_short > 0:
        t = workload.tokens_per_microbatch
        while t > 16:
            t //= 2
            w = Workload(**{**workload.__dict__, "seq_len": t, "micro_batch": 1})
            raw, _, refusals = b.estimate(topology, closest.setup, w)
            dev = int((1 + reserve_frac) * sum(r[2] for r in raw if r[1] == "device")) + sum(x.bytes for x in dev_over)
            if not refusals and dev + budget["headroom"] <= budget["device"]:
                out.append(f"reduce tokens per micro-batch to {t} (seq x micro-batch) with the same setup")
                break
        else:
            out.append("no token count makes the closest setup fit: its fixed weights alone exceed the budget")
    return out


def _suggest_serve(topology, workload, budget, closest, backends, dev_over, reserve_frac):
    """The largest context (per sequence) and the largest concurrency that fit with the closest setup, by search."""
    out = []
    b = next((x for x in backends if x.NAME == closest.backend), None)
    short = closest.device_bytes + budget["headroom"] - budget["device"]
    if short > 0:
        out.append(f"{short / GiB:.2f} GiB more device memory (a larger --vram budget, or free what other processes hold)")
    if b is None:
        return out

    def fits(ctx, seqs):
        w = Workload(**{**workload.__dict__, "context_len": ctx, "concurrency": seqs})
        raw, _, refusals = b.estimate(topology, {**closest.setup, "max_tokens_per_seq": ctx, "max_seqs": seqs}, w)
        dev = int((1 + reserve_frac) * sum(r[2] for r in raw if r[1] == "device")) + sum(x.bytes for x in dev_over)
        return not refusals and dev + budget["headroom"] <= budget["device"]

    ctx, seqs = closest.setup["max_tokens_per_seq"], closest.setup["max_seqs"]
    lo, hi = 0, -(-ctx // 16)                  # in 16-token KV blocks: the pool's granularity
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fits(16 * mid, seqs) else (lo, mid)
    if lo >= 1 and 16 * lo < ctx:
        out.append(f"context {16 * lo} tokens per sequence at concurrency {seqs} (requested {ctx})")
    n = seqs
    while n > 1 and not fits(ctx, n):
        n //= 2
    if n >= 1 and fits(ctx, n) and n < seqs:
        out.append(f"concurrency {n} at context {ctx} (requested {seqs})")
    if not fits(16, 1):
        out.append("the weights alone exceed the budget: no context or concurrency fits")
    return out
