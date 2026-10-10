"""Turn a workload, a machine, and constraints into an inspectable execution plan.

Loggetta owns planning, ExecutionPlan, and ExecutionReceipt; experts4bit-qlora executes supported plans.
Plan first. Execute through the backend. Measure. Feed the receipt back:

    from loggetta import Constraints, Workload, describe_model, execute, load_observations, plan, probe
    evidence = load_observations("receipts")                 # earlier ExecutionReceipts, if any
    p = plan(describe_model("allenai/OLMoE-1B-7B-0924"), probe(), Workload(seq_len=512), observations=evidence)
    print(p.render())                                        # the ExecutionPlan: choice, why, what lost, or a refusal
    receipt = execute(p, out_dir="receipts")                 # the backend runs it; the receipt is the next evidence

Planning is pure policy and loads no weights; ``execute`` only dispatches to the backend and writes the receipt.

Serialized schemas remain implementation-neutral: ``execution-plan/1`` and ``execution-receipt/1``.
"""
from .data import TrainingData
from .execution import PlanNotExecutable, execute, load_observations
from .hardware import HardwareProfile, probe
from .model import describe_model
from .plan import Constraints, ExecutionPlan, Workload
from .planner import plan

__all__ = ["HardwareProfile", "probe", "describe_model", "Constraints", "ExecutionPlan", "Workload", "plan",
           "execute", "load_observations", "PlanNotExecutable"]
__version__ = "0.5.0"


def load_adapter(directory, *, device="cuda"):
    """Rebuild the recorded runtime setup and return a model with the saved adapters loaded."""
    import json
    from pathlib import Path

    manifest = json.loads((Path(directory).expanduser() / "adapter_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("backend") == "dense":
        from .backends.dense_adapters import load_adapter as load

        return load(directory, device=device)
    from .backends.experts4bit_adapters import load_adapter as load

    return load(directory, device=device)


__all__ += ["TrainingData", "load_adapter"]
