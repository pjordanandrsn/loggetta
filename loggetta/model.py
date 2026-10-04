"""Model identity and topology, from the model-family layer that owns it.

The planner never parses a config itself: structure, admission and family conventions belong to the package
that loads the model (today experts4bit-qlora's ``describe_moe``). This module is the one place that asks.
"""
from __future__ import annotations


class NoModelProvider(RuntimeError):
    pass


def describe_model(model, *, revision=None, trust_remote_code=False):
    """The model's topology without its weights (config.json at most, plus a meta-device module tree)."""
    try:
        from experts4bit_qlora.arch.topology import describe_moe
    except ImportError as e:
        raise NoModelProvider("no model-family provider installed: experts4bit-qlora with arch.topology is "
                              f"required to describe a model ({e})") from e
    return describe_moe(model, revision=revision, trust_remote_code=trust_remote_code)
