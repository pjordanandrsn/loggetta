"""Turn a workload, a machine, and constraints into an inspectable execution plan.

Loggetta owns planning, ExecutionPlan, and ExecutionReceipt; experts4bit-qlora executes supported plans.

    from loggetta import probe, describe_model, plan, Workload, Constraints
    p = plan(describe_model("allenai/OLMoE-1B-7B-0924"), probe(), Workload(seq_len=512))
    print(p.render())

Serialized schemas remain implementation-neutral: ``execution-plan/1`` and ``execution-receipt/1``.
"""
from .hardware import HardwareProfile, probe
from .model import describe_model
from .plan import Constraints, ExecutionPlan, Workload
from .planner import plan

__all__ = ["HardwareProfile", "probe", "describe_model", "Constraints", "ExecutionPlan", "Workload", "plan"]
__version__ = "0.0.1"
