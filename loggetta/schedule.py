"""The learning-rate schedule: one documented recipe, a pure function of the workload.

``cosine``: linear warmup from ``1/warmup`` of the peak to the peak over the first ``warmup`` steps, then cosine decay
to :data:`COSINE_FLOOR` of the peak at the last step. ``constant``: the peak throughout, after the warmup if one is
asked for. The warmup defaults to :data:`WARMUP_FRACTION` of the steps (at least one) under ``cosine`` and to none
under ``constant``. Step ``s`` (0-based) trains at ``learning_rate * lr_factor(s, ...)``; no step trains at zero.
"""
from __future__ import annotations

import math

SCHEDULES = ("constant", "cosine")
WARMUP_FRACTION = 0.03
COSINE_FLOOR = 0.1


def warmup_steps(workload) -> int:
    """The warmup a workload resolves to."""
    if workload.warmup_steps is not None:
        return min(workload.warmup_steps, workload.steps)
    return max(1, round(WARMUP_FRACTION * workload.steps)) if workload.lr_schedule == "cosine" else 0


def lr_factor(step: int, steps: int, warmup: int, schedule: str, floor: float = COSINE_FLOOR) -> float:
    """The fraction of the peak learning rate step ``step`` of ``steps`` trains at."""
    if step < warmup:
        return (step + 1) / warmup
    if schedule == "constant":
        return 1.0
    decay = steps - warmup
    progress = (step - warmup) / (decay - 1) if decay > 1 else 0.0
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def learning_rates(workload) -> list:
    """The learning rate of every step of ``workload``."""
    warm = warmup_steps(workload)
    return [workload.learning_rate * lr_factor(s, workload.steps, warm, workload.lr_schedule)
            for s in range(workload.steps)]


def describe(workload) -> str:
    """The schedule in words, for the plan and the receipt."""
    warm, lr = warmup_steps(workload), workload.learning_rate
    head = f"learning rate {lr:g}"
    if workload.lr_schedule == "constant":
        return head + (f" after a linear warmup over {warm} step(s)" if warm else "")
    return head + f": linear warmup over {warm} step(s), then cosine decay to {lr * COSINE_FLOOR:g}"
