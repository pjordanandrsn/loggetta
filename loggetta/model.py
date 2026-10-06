"""Model identity and topology, from the model-family layer that owns it.

The planner never parses a config itself: structure, admission and family conventions belong to the package that
loads the model. Each backend asks its own package (``describe``) and says whether it can plan for the answer
(``refusal``); this module is the one place that asks them.
"""
from __future__ import annotations


class NoModelProvider(RuntimeError):
    pass


def describe_model(model, *, revision=None, trust_remote_code=False, backends=None):
    """The model's topology without its weights (config.json at most, plus a meta-device module tree).

    Backends are asked in order, and the first description its backend can plan for is returned. When none can, the
    first description is returned anyway: its refusal is the plan's answer. The topology is the backend's own type
    (today experts4bit-qlora's ``MoETopology``)."""
    from .backends import BACKENDS

    first, missing = None, []
    for b in BACKENDS if backends is None else backends:
        try:
            topology = b.describe(model, revision=revision, trust_remote_code=trust_remote_code)
        except NoModelProvider as e:
            missing.append(e)
            continue
        if b.refusal(topology) is None:
            return topology
        if first is None:
            first = topology
    if first is None:
        if len(missing) == 1:
            raise missing[0]
        raise NoModelProvider("; ".join(str(e) for e in missing) or "no backend is installed to describe a model")
    return first
