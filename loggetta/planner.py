"""The planner: (topology, hardware, workload, constraints, backends) -> ExecutionPlan. Pure policy; runs nothing.

Policy lives here and nowhere below: the device and host budgets, the headroom kept free, the runtime overhead
charged on top of a backend's estimate, which candidates are feasible, the order an objective tries them in, and
what to suggest when none fits. Mechanism costs come from the backends, which ask the packages that own them.

Deterministic: the same inputs give the same plan, byte for byte (``ExecutionPlan.to_json``).
"""
from __future__ import annotations

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


def _overheads(hardware, gpu, observations):
    """Runtime costs the backends' estimates exclude, preferring measurements from receipts for this GPU + driver:
    the CUDA context (device), the allocator's reserved-but-unallocated blocks as a fraction of the allocator peak
    (device), and the process's own host baseline. Returns (device lines, host lines, reserve fraction, its line meta)."""
    key = (gpu.name if gpu else None, gpu.driver.value if gpu else None)
    for obs in observations:
        m = obs.get("measured", {})
        g = obs.get("hardware", {}).get("gpu", {})
        if (g.get("name"), g.get("driver")) != key or m.get("cuda_context_bytes") is None:
            continue
        src = f"receipt {obs.get('run_id')}"
        alloc, reserved = m.get("device_peak_bytes"), m.get("device_reserved_peak_bytes")
        frac = (reserved - alloc) / alloc if alloc and reserved else DEFAULT_RESERVE_FRAC
        base = m.get("host_anon_after_load_bytes") if obs.get("setup", {}).get("expert_residency") == "device" else None
        return ([MemoryLine("CUDA context + library workspaces", "device", max(0, int(m["cuda_context_bytes"])),
                            "measured", f"{src}: driver-reported process peak minus allocator reserved peak")],
                [MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host",
                            int(base or m.get("host_baseline_bytes") or DEFAULT_HOST_BASELINE),
                            "measured" if (base or m.get("host_baseline_bytes")) else "inferred",
                            f"{src}: anonymous RSS after load" if base else src)],
                frac, ("measured", f"{src}: reserved peak / allocated peak - 1 = {frac:.3f}"))
    return ([MemoryLine("CUDA context + library workspaces", "device", DEFAULT_CUDA_CONTEXT, "inferred",
                        "default; no receipt on file measured this GPU + driver")],
            [MemoryLine("process baseline (torch, CUDA, libraries, model objects)", "host", DEFAULT_HOST_BASELINE,
                        "inferred", "default; no receipt on file measured this host")],
            DEFAULT_RESERVE_FRAC, ("inferred", f"default {DEFAULT_RESERVE_FRAC:.0%} of the allocator estimate; "
                                               "no receipt on file measured this GPU + driver"))


#: PCIe payload bandwidth per lane, GB/s, by generation (after line coding): the theoretical ceiling, never achieved
PCIE_LANE_GBPS = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938, 6: 7.563}


def reserve_fraction(gpu, setup, observations, default):
    """Allocator reserve slack (reserved peak / allocated peak - 1) for a candidate: from a receipt on this GPU with the
    same expert residency and kernel if one exists (slack depends on the allocation pattern: offload's per-layer staging
    leaves more cached blocks), else any receipt on this GPU, else ``default``. Returns (fraction, basis, source)."""
    same_gpu = [o for o in observations if o.get("hardware", {}).get("gpu", {}).get("name") == gpu.name
                and o.get("measured", {}).get("device_peak_bytes") and o.get("measured", {}).get("device_reserved_peak_bytes")]
    for pool, how in (([o for o in same_gpu if {k: o.get("setup", {}).get(k) for k in ("expert_residency", "expert_kernel")}
                        == {k: setup.get(k) for k in ("expert_residency", "expert_kernel")}], "same residency and kernel"),
                      (same_gpu, "same GPU, different setup")):
        if pool:
            m = pool[0]["measured"]
            frac = m["device_reserved_peak_bytes"] / m["device_peak_bytes"] - 1
            return frac, "measured", f"receipt {pool[0].get('run_id')} ({how}): reserved / allocated peak - 1 = {frac:.3f}"
    return default[0], default[1], default[2]


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


def _model_dict(topo) -> dict:
    return {"model": topo.model, "model_type": topo.model_type, "revision": topo.revision,
            "convention": topo.convention, "summary": topo.summary(), "n_layers": topo.n_layers,
            "moe_layers": len(topo.expert_stacks), "n_experts": topo.n_experts, "top_k": topo.top_k,
            "expert_params": topo.expert_numel, "dense_params": topo.dense_numel,
            "loader_refusal": topo.loader_refusal, "provenance": topo.provenance}


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


def plan(topology, hardware, workload: Workload, constraints: Constraints = Constraints(), *, backends=None,
         observations=()) -> ExecutionPlan:
    backends = BACKENDS if backends is None else backends
    if constraints.objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    gpu = hardware.gpu(constraints.device)
    warnings, reasons = [], []
    dev_budget = constraints.vram_budget
    dev_src = "user"
    if dev_budget is None:
        dev_budget = (gpu.memory_free.value or 0) if gpu else 0
        dev_src = f"{gpu.memory_free.source}: free now" if gpu else "no GPU"
    elif gpu and gpu.memory_free.value is not None and dev_budget > gpu.memory_free.value:
        warnings.append(f"device budget {dev_budget / GiB:.2f} GiB exceeds what the driver reports free now "
                        f"({gpu.memory_free.value / GiB:.2f} GiB): another process holds the difference")
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
    budget = {"device": int(dev_budget), "device_source": dev_src, "host": int(host_budget),
              "host_source": host_src, "headroom": int(headroom)}
    common = dict(model=_model_dict(topology), hardware=_hw_dict(hardware, gpu), workload=workload,
                  constraints=constraints, budget=budget)

    def refuse(why, **extra):
        return ExecutionPlan(status="refused", selected=None, alternatives=extra.pop("alternatives", ()),
                             reasons=tuple(reasons), warnings=tuple(warnings),
                             refusal={"reasons": list(why), **extra}, **common)

    if gpu is None:
        return refuse([f"no GPU at index {constraints.device}: every installed backend needs one"])
    if topology.loader_refusal:
        return refuse([f"the model-family layer cannot load this model: {topology.loader_refusal}"])
    usable = [b for b in backends if workload.kind in b.WORKLOADS]
    if not usable:
        return refuse([f"no installed backend plans {workload.kind!r} workloads yet "
                       f"(backends: {', '.join(b.NAME for b in backends)}; they plan {sorted({w for b in backends for w in b.WORKLOADS})})"])

    dev_over, host_over, reserve_frac, reserve_meta = _overheads(hardware, gpu, observations)
    link_gbps, link_basis, link_src = link_bandwidth(gpu, observations)
    cands, statuses = [], {}
    for b in usable:
        st = b.probe(gpu)
        statuses[b.NAME] = st
        if not st.available:
            reasons.append(f"backend {b.NAME} unavailable: {st.reason}")
            continue
        reasons += b.policy_notes(topology, constraints)
        for k, (ok, why) in sorted(st.kernels.items()):
            if not ok:
                reasons.append(f"{b.NAME}: kernel {k} not usable here: {why}")
        for setup in b.candidates(topology, workload, constraints, st):
            raw, unmodelled, refusals = b.estimate(topology, setup, workload)
            alloc = sum(r[2] for r in raw if r[1] == "device")
            frac, fbasis, fsrc = reserve_fraction(gpu, setup, observations, (reserve_frac, *reserve_meta))
            reserve = MemoryLine("allocator reserve (cached, unallocated blocks)", "device", int(frac * alloc),
                                 fbasis, fsrc)
            lines = tuple(MemoryLine(*r) for r in raw) + (reserve,) + tuple(dev_over) + tuple(host_over)
            dev = sum(ln.bytes for ln in lines if ln.where == "device")
            host = sum(ln.bytes for ln in lines if ln.where == "host")
            rejected = list(refusals)
            bounds = {}
            link = sum(ln.bytes for ln in lines if ln.where == "link") * workload.grad_accum
            if link and link_gbps:
                bounds = {"link_bytes_per_step": link, "link_gbps": link_gbps, "link_gbps_basis": link_basis,
                          "link_gbps_source": link_src, "s_per_step_lower_bound": link / (link_gbps * 1e9)}
            if not refusals and bounds and constraints.target_s_per_step is not None and \
                    bounds["s_per_step_lower_bound"] > constraints.target_s_per_step:
                rejected.append(f"host-to-device traffic alone needs >= {bounds['s_per_step_lower_bound']:.2f} s/step "
                                f"({link / 1e9:.1f} GB over <= {link_gbps:.1f} GB/s, {link_basis}); target is "
                                f"{constraints.target_s_per_step:g} s/step")
            if not refusals:
                if dev + headroom > dev_budget:
                    rejected.append(f"device {dev / GiB:.2f} + headroom {headroom / GiB:.2f} GiB > budget "
                                    f"{dev_budget / GiB:.2f} GiB")
                if host > host_budget:
                    rejected.append(f"host {host / GiB:.2f} GiB > budget {host_budget / GiB:.2f} GiB")
            rank, note = _rank(constraints.objective, setup, dev, host, b)
            obs = _observed(observations, topology.model, setup, workload, gpu)
            if obs:
                note += (f"; measured before: device peak {obs['measured']['device_peak_bytes'] / GiB:.2f} GiB "
                         f"(receipt {obs.get('run_id')})")
            # the planner charges the context, allocator reserve and process baseline itself: drop the backend's note
            unmodelled = tuple(u for u in unmodelled if not u.startswith("CUDA context"))
            cands.append(Candidate(backend=b.NAME, setup=setup, lines=lines, device_bytes=dev, host_bytes=host,
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
        why = [f"{closest.label()}: " + "; ".join(closest.rejected)]
        if not valid:
            why = sorted({r for c in infeasible for r in c.rejected})
        suggestions = _suggest(topology, workload, constraints, budget, closest, usable, statuses, dev_over, host_over,
                               reserve_frac)
        return ExecutionPlan(status="refused", selected=None, alternatives=tuple(infeasible), reasons=tuple(reasons),
                             warnings=tuple(warnings), performance=perf,
                             refusal={"reasons": why, "closest": closest.label(),
                                      "closest_lines": [f"{ln.where:6s} {ln.name:37s} {ln.bytes / GiB:6.2f} GiB [{ln.basis}]"
                                                        for ln in closest.lines],
                                      "suggestions": suggestions}, **common)

    sel = feasible[0]
    st = statuses[sel.backend]
    b = next(x for x in usable if x.NAME == sel.backend)
    reasons += _explain(sel, feasible, infeasible, budget, st, b, constraints)
    if sel.setup["expert_residency"] == "host" and gpu.pcie_width_current.value and gpu.pcie_width_max.value and \
            gpu.pcie_width_current.value < gpu.pcie_width_max.value:
        warnings.append(f"host-resident experts stream over PCIe, and the driver reports the link at "
                        f"x{gpu.pcie_width_current.value} of x{gpu.pcie_width_max.value} right now")
    return ExecutionPlan(status="feasible", selected=sel, alternatives=tuple(feasible[1:]) + tuple(infeasible),
                         reasons=tuple(reasons), warnings=tuple(warnings), performance=perf,
                         provenance={"backend_versions": st.versions}, **common)


def _explain(sel, feasible, infeasible, budget, status, backend, constraints):
    s, out = sel.setup, []
    resident = [c for c in feasible + infeasible if c.setup["expert_residency"] == "device"
                and not any("cannot" in r or "refus" in r for r in c.rejected)]
    if s["expert_residency"] == "device":
        out.append(f"experts resident on the device: {sel.device_bytes / GiB:.2f} GiB estimated + "
                   f"{budget['headroom'] / GiB:.2f} GiB headroom fits the {budget['device'] / GiB:.2f} GiB budget")
    else:
        need = min((c.device_bytes for c in resident), default=None)
        out.append("experts host-backed (pinned, streamed one layer at a time): "
                   + (f"the cheapest resident setup needs {need / GiB:.2f} GiB + headroom, over the "
                      f"{budget['device'] / GiB:.2f} GiB budget" if need is not None and constraints.objective == "speed"
                      else f"chosen by objective {constraints.objective}"))
    out.append(f"expert kernel {s['expert_kernel']}: {backend.describe_kernel(s, status)}")
    if constraints.objective == "speed":
        for axis, ev in backend.SPEED_EVIDENCE.items():
            out.append(f"ordering, {axis}: {ev}")
    if s["attn_4bit"]:
        out.append("attention stored in NF4 because no bf16-attention setup fit")
    if constraints.fixed:
        out.append(f"fixed by the caller: {constraints.fixed}")
    out.append("activation policy: every decoder layer checkpointed (recomputed in backward); "
               + (f"MoE activations kept in {s['keep_moe_layers']} layers" if s.get("keep_moe_layers")
                  else "no MoE activations kept (keep_moe_layers is a dial the planner does not choose yet)"))
    return out


def _suggest(topology, workload, constraints, budget, closest, backends, statuses, dev_over, host_over,
             reserve_frac=DEFAULT_RESERVE_FRAC):
    out = []
    dev_short = closest.device_bytes + budget["headroom"] - budget["device"]
    if dev_short > 0:
        out.append(f"{dev_short / GiB:.2f} GiB more device memory (a larger --vram budget, or free what other "
                   "processes hold)")
    if closest.host_bytes > budget["host"]:
        out.append(f"{(closest.host_bytes - budget['host']) / GiB:.2f} GiB more host memory")
    if constraints.expert_residency and "host" not in constraints.expert_residency:
        relaxed = Constraints(**{**constraints.__dict__, "expert_residency": None})
        for b in backends:
            for setup in b.candidates(topology, workload, relaxed, statuses[b.NAME]):
                if setup["expert_residency"] != "host":
                    continue
                raw, _, refusals = b.estimate(topology, setup, workload)
                dev = int((1 + reserve_frac) * sum(r[2] for r in raw if r[1] == "device")) + sum(x.bytes for x in dev_over)
                host = sum(r[2] for r in raw if r[1] == "host") + sum(x.bytes for x in host_over)
                if not refusals and dev + budget["headroom"] <= budget["device"] and host <= budget["host"]:
                    out.append(f"allow host-backed experts: {dev / GiB:.2f} GiB device + {host / GiB:.2f} GiB host fits")
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
