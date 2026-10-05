"""The former name of :mod:`.execution`, kept so existing imports keep working.

Execution here is orchestration around a backend call; the runtime that loads and runs the model is the backend's.
"""
from .execution import (RECEIPT_SCHEMA, PlanNotExecutable, check_here, compare, execute,  # noqa: F401
                        load_observations, summarize)
