"""Command line: ``inspect`` (the machine), ``plan`` (an ExecutionPlan; loads nothing), ``execute`` (a saved plan,
through its backend, to an ExecutionReceipt) and ``train`` (plan + execute in one step). The program name is taken
from argv, never spelled here."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

GiB = 1 << 30


def _gib(s):
    return None if s is None else int(float(s) * GiB)


def _parse_fixed(items):
    out = {}
    for it in items or ():
        k, _, v = it.partition("=")
        if not _:
            raise SystemExit(f"--fix expects key=value, got {it!r}")
        low = v.lower()
        out[k] = True if low == "true" else False if low == "false" else int(v) if v.lstrip("-").isdigit() else v
    return out


def _common(p):
    p.add_argument("model", help="hub id or local snapshot directory")
    p.add_argument("--revision")
    p.add_argument("--dataset", help="Hub dataset ID or local JSON/JSONL/CSV/Parquet/TXT file")
    p.add_argument("--dataset-config", help="Hub dataset configuration")
    p.add_argument("--dataset-revision", help="Hub dataset revision; use a commit SHA for reproducibility")
    p.add_argument("--split", default="train", help="dataset split (default: train)")
    p.add_argument("--format", choices=("auto", "text", "alpaca", "chat"), default="auto")
    p.add_argument("--text-field", default="text")
    p.add_argument("--messages-field", default="messages")
    p.add_argument("--instruction-field", default="instruction")
    p.add_argument("--input-field", default="input")
    p.add_argument("--output-field", default="output")
    p.add_argument("--shuffle-data", action="store_true", help="shuffle rows deterministically with --seed")
    p.add_argument("--repeat-data", action="store_true", help="explicitly allow repeating a short dataset")
    p.add_argument("--packing", choices=("auto", "concat", "isolated"), default="auto",
                   help="auto = isolated for chat and alpaca (each example attends only to itself), concat for text")
    p.add_argument("--loss", choices=("auto", "all", "assistant"), default="auto",
                   help="which tokens train: auto = assistant turns (chat) or the response (alpaca), every token for text")
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--workload", default="train", choices=("train", "serve"))
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--steps", type=int, help="optimizer steps (default 20; with --epochs the planner derives them)")
    p.add_argument("--epochs", type=float, help="passes over --dataset; the planner derives --steps from its tokens")
    p.add_argument("--context", type=int, help="serve: tokens per sequence, prompt + output (default 4096)")
    p.add_argument("--concurrency", type=int, help="serve: sequences decoded together (default 1)")
    p.add_argument("--optimizer", default="adamw", choices=("adamw", "adamw_8bit"))
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--vram", help="device budget in GiB (default: free now)")
    p.add_argument("--ram", help="host budget in GiB (default: available now)")
    p.add_argument("--headroom", help="device headroom in GiB (default: max(0.5, 5%% of budget))")
    p.add_argument("--experts", choices=("any", "device", "host"), default="any",
                   help="where frozen experts may live")
    p.add_argument("--fix", action="append", metavar="FIELD=VALUE",
                   help="fix a backend setup field (expert mode), e.g. --fix expert_kernel=reference")
    p.add_argument("--objective", default="speed", choices=("speed", "min_vram", "min_ram"))
    p.add_argument("--target-s-per-step", type=float,
                   help="refuse setups whose host-to-device traffic alone provably exceeds this step time")
    p.add_argument("--observations", help="directory of earlier receipts: measured evidence the plan learns from")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument("--hardware", help="plan for a saved hardware profile (inspect --json) instead of probing this machine")


def _plan(a):
    """Cheapest checks first: the workload and the dataset (read and tokenized, no weights), then the model's config,
    then the plan."""
    from . import Constraints, Workload, data as data_mod, describe_model, plan, probe
    from .execution import load_observations

    data = None
    if a.dataset:
        source = a.dataset
        if Path(source).expanduser().exists() or source.startswith((".", "~", "/")):
            source = str(Path(source).expanduser().absolute())
        data = data_mod.TrainingData(source=source, format=a.format, split=a.split, config=a.dataset_config,
                                     revision=a.dataset_revision, text_field=a.text_field,
                                     messages_field=a.messages_field, instruction_field=a.instruction_field,
                                     input_field=a.input_field, output_field=a.output_field,
                                     shuffle=a.shuffle_data, repeat=a.repeat_data, loss=a.loss,
                                     packing=a.packing).to_dict()
    elif (a.dataset_config or a.dataset_revision or a.format != "auto" or a.split != "train"
          or a.shuffle_data or a.repeat_data or a.loss != "auto" or a.packing != "auto" or a.text_field != "text"
          or a.messages_field != "messages"
          or a.instruction_field != "instruction" or a.input_field != "input" or a.output_field != "output"):
        raise ValueError("dataset options require --dataset; omit them all for the Alpaca demonstration")
    if a.epochs is not None and a.steps is not None:
        raise ValueError("use --steps or --epochs, not both: with --epochs the planner derives the steps")
    w = Workload(kind=a.workload, seq_len=a.seq, micro_batch=a.micro_batch, grad_accum=a.grad_accum,
                 steps=a.steps if a.steps is not None else 20, epochs=a.epochs,
                 optimizer=a.optimizer, context_len=a.context or (4096 if a.workload == "serve" else None),
                 concurrency=a.concurrency or (1 if a.workload == "serve" else None),
                 data=data, learning_rate=a.learning_rate)
    profile = None
    if data is not None:                     # every row validated and tokenized before the model is looked at
        tokenizer = data_mod.load_tokenizer(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code)
        with data_mod.encode_dataset(tokenizer, data_mod.TrainingData.from_dict(data), seq_len=a.seq) as encoded:
            profile = encoded.profile
    topo = describe_model(a.model, revision=a.revision, trust_remote_code=a.trust_remote_code)
    if a.hardware:
        from .hardware import HardwareProfile

        with open(a.hardware) as f:
            hw = HardwareProfile.from_dict(json.load(f), origin=a.hardware)
    else:
        hw = probe()
    c = Constraints(device=a.device, vram_budget=_gib(a.vram), ram_budget=_gib(a.ram), headroom=_gib(a.headroom),
                    expert_residency=None if a.experts == "any" else (a.experts,), fixed=_parse_fixed(a.fix),
                    objective=a.objective, target_s_per_step=a.target_s_per_step)
    return plan(topo, hw, w, c, observations=load_observations(a.observations), data_profile=profile)


def main(argv=None) -> int:
    from .measure import provenance

    prov = {**provenance(), "taken": "at process start, before the measured code was imported"}
    ap = argparse.ArgumentParser(description="Turn a workload, this machine and constraints into an inspectable "
                                             "execution plan; execute it through its backend; keep the receipt.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    pi = sub.add_parser("inspect", help="the hardware inventory, and a model's topology if one is given")
    pi.add_argument("model", nargs="?")
    pi.add_argument("--json", action="store_true")
    pp = sub.add_parser("plan", help="the ExecutionPlan: what should run here, why, and what lost; loads no weights")
    _common(pp)
    pp.add_argument("--json", action="store_true")
    pp.add_argument("--out", help="write the plan as JSON here")
    pp.add_argument("-v", "--verbose", action="store_true")
    pe = sub.add_parser("execute", help="run a saved feasible plan (plan --out) through its backend; write a receipt")
    pe.add_argument("plan", help="a plan written by plan --out")
    pe.add_argument("--out", default="receipts", help="receipt directory")
    pe.add_argument("--seed", type=int, default=0)
    pe.add_argument("--adapter-out", help="fresh adapter directory (default: OUT/RUN_ID/adapter)")
    pt = sub.add_parser("train", help="plan, then execute a feasible plan through its backend and write a receipt")
    _common(pt)
    pt.add_argument("--out", default="receipts", help="receipt directory")
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--adapter-out", help="fresh adapter directory (default: OUT/RUN_ID/adapter)")
    a = ap.parse_args(argv)

    if a.cmd == "inspect":
        from .hardware import describe, probe

        hw = probe()
        if a.json:
            print(json.dumps(hw.to_dict(), indent=1, default=str))
        else:
            print(describe(hw))
        if a.model:
            from .model import describe_model

            t = describe_model(a.model)
            print(json.dumps(t.to_dict(), indent=1, default=str) if a.json else "\nModel\n  " + t.summary())
        return 0

    if a.cmd == "execute":
        from .plan import ExecutionPlan

        with open(a.plan) as f:
            p = ExecutionPlan.from_dict(json.load(f))
    else:
        try:
            p = _plan(a)
        except (ValueError, OSError) as e:      # OSError: a dataset, tokenizer or hardware file that cannot be read
            print(f"invalid plan: {e}", file=sys.stderr)
            return 2
    if a.cmd == "plan":
        if a.out:
            with open(a.out, "w") as f:
                f.write(p.to_json())
        print(p.to_json() if a.json else p.render(verbose=a.verbose))
        return 0 if p.status == "feasible" else 2

    print(p.render())
    if p.status != "feasible":
        return 2
    from .execution import PlanNotExecutable, execute, summarize

    print(f"\nExecuting the selected plan through backend {p.selected.backend}...")
    try:
        receipt = execute(p, out_dir=a.out, adapter_dir=a.adapter_out, seed=a.seed, prov=prov)
    except PlanNotExecutable as e:
        print(f"not executable: {e}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as e:
        print(f"execution failed: {e}", file=sys.stderr)
        return 1
    print("\n" + summarize(receipt))
    print(f"\nreceipt: {a.out}/{receipt['run_id']}.json")
    return 0 if receipt["status"] == "OK" else 1


if __name__ == "__main__":
    sys.exit(main())
