"""Execution backends. A plain list, not a plugin registry: there is one backend today, and the interface it exposes
(``probe``, ``candidates``, ``policy_notes``, ``estimate``, ``speed_rank``, ``describe_kernel``, plus an executor) is what a second
one would have to provide. Generalize when that second backend exists, not before."""
from . import experts4bit

BACKENDS = (experts4bit,)
