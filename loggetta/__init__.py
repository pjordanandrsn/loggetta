"""A planning layer above experts4bit-qlora and grouped-nf4-gemm: decide how a MoE workload should run on this
machine -- and say why, or why not -- before loading anything.

    from loggetta import probe, describe_model, plan, Workload, Constraints
    p = plan(describe_model("allenai/OLMoE-1B-7B-0924"), probe(), Workload(seq_len=512))
    print(p.render())

The package name is a working name; nothing serialized carries it (schemas are ``execution-plan/1`` and
``execution-receipt/1``), so renaming the package renames nothing else.
"""
from .hardware import HardwareProfile, probe
from .model import describe_model
from .plan import Constraints, ExecutionPlan, Workload
from .planner import plan

__all__ = ["HardwareProfile", "probe", "describe_model", "Constraints", "ExecutionPlan", "Workload", "plan"]
__version__ = "0.0.1"
