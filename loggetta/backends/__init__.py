"""Execution backends: the packages that know HOW to run what the planner selects.

Each module here is the planner's adapter to one backend. It answers the planner's questions (``probe``,
``candidates``, ``policy_notes``, ``estimate``, ``speed_rank``, ``label``, ``explain``, ``describe_kernel``; for
serving ``fill_knobs`` and ``resolve``) by asking the backend's own packages, and hands a feasible plan back to them
(``executor``, ``run_tag``). A plain list, not a plugin registry: there is one backend today, and that interface is
what a second one would have to provide. Generalize when that second backend exists, not before."""
from . import experts4bit

BACKENDS = (experts4bit,)
