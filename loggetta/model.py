"""Model identity and topology, from the model-family layer that owns it.

The planner never parses a config itself: structure, admission and family conventions belong to the package that
loads the model. Each backend asks its own package (``describe``) and says whether it can plan for the answer
(``refusal``); this module is the one place that asks them.
"""
from __future__ import annotations


class NoModelProvider(RuntimeError):
    pass


class Refused:
    """What :func:`describe_model` returns when more than one backend described the model and every one refused it:
    the first description (its attributes read through, so callers that expect it still work) and each backend's own
    refusal, by backend name, so a plan can say why each one cannot take the model."""

    def __init__(self, first, backend: str, reasons: dict):
        self._first, self.backend, self.reasons = first, backend, reasons

    @property
    def description(self):
        return self._first

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._first, name)


def describe_model(model, *, revision=None, trust_remote_code=False, backends=None):
    """The model's topology without its weights (config.json at most, plus a meta-device module tree).

    Backends are asked in order, and the first description its backend can plan for is returned. When none can, the
    first description is returned anyway: its refusal is the plan's answer. The topology is the backend's own type
    (today experts4bit-qlora's ``MoETopology``)."""
    from .backends import BACKENDS

    described, missing = [], []
    for b in BACKENDS if backends is None else backends:
        try:
            topology = b.describe(model, revision=revision, trust_remote_code=trust_remote_code)
        except NoModelProvider as e:
            missing.append(e)
            continue
        why = b.refusal(topology)
        if why is None:
            return topology
        described.append((b.NAME, topology, why))
    if not described:
        if len(missing) == 1:
            raise missing[0]
        raise NoModelProvider("; ".join(str(e) for e in missing) or "no backend is installed to describe a model")
    if len(described) == 1:
        return described[0][1]
    return Refused(described[0][1], described[0][0], {name: why for name, _t, why in described})
