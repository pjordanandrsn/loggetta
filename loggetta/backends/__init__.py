"""Execution backends: the packages that know HOW to run what the planner selects.

Each module here is the planner's adapter to one backend. It describes a model its own way (``describe``,
``refusal``, ``summary``), answers the planner's questions (``probe``, ``candidates``, ``policy_notes``, ``estimate``,
``speed_rank``, ``label``, ``explain``, ``describe_kernel``, ``residency``, ``plan_warnings``, ``relaxed_candidates``;
for serving ``fill_knobs`` and ``resolve``) by asking the backend's own packages, and hands a feasible plan back to
them (``executor``, ``run_tag``). A plain tuple, not a plugin registry: two backends today (experts4bit, dense), each providing that
interface."""
from . import dense, experts4bit

#: asked in this order: a MoE model is experts4bit's, and the dense backend describes what it refuses
BACKENDS = (experts4bit, dense)
