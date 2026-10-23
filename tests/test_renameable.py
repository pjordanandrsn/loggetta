"""The working name must stay a label: no serialized format, schema, env var or code path may depend on it.

Renaming the umbrella project = renaming the package directory and the packaging metadata. These tests fail the
day a protocol string, env var or import starts to depend on the current name.
"""
import pathlib
import re

import loggetta

PKG = pathlib.Path(loggetta.__file__).parent
NAME = PKG.name


def test_no_code_file_spells_the_package_name():
    hits = []
    for f in PKG.rglob("*.py"):
        if f.name == "__init__.py" and f.parent == PKG:
            continue                                   # the top-level docstring's usage example
        for i, line in enumerate(f.read_text().splitlines(), 1):
            if re.search(NAME, line, re.I):
                hits.append(f"{f.relative_to(PKG)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_schemas_are_generic():
    from loggetta.plan import PLAN_SCHEMA
    from loggetta.execution import RECEIPT_SCHEMA

    for s in (PLAN_SCHEMA, RECEIPT_SCHEMA):
        assert NAME.lower() not in s.lower()


def test_no_environment_variable_is_read():
    # configuration is typed objects and CLI flags; an env var would be a second, renaming-hostile surface. The two
    # exceptions are variables OTHER tools read, which the probes must read the same way: CUDA_VISIBLE_DEVICES (torch),
    # CC (Triton's compiler choice, read so the C-compiler probe tests the compiler Triton will actually use) and
    # PYTORCH_CUDA_ALLOC_CONF (torch's allocator; recorded in a receipt's provenance, never acted on)
    for f in PKG.rglob("*.py"):
        text = f.read_text()
        for allowed in ('os.environ.get("CUDA_VISIBLE_DEVICES")', 'os.environ.get("CC")',
                        'os.environ.get("PYTORCH_CUDA_ALLOC_CONF")',
                        # measure.e4b_env: experts4bit-qlora's switches, recorded only. The prefix scan yields NAMES; a
                        # VALUE is kept only for measure.E4B_VALUE_KNOBS (test_e4b_env_* in test_execution.py hold that)
                        '{k: v for k, v in os.environ.items() if k.startswith(("E4B_", "GNF4_"))}',
                        # experts4bit_serve.measure_serve: a WRITE, handing a serve plan's settings to experts4bit-qlora,
                        # whose PagedServeConfig.from_env reads only the process environment
                        'os.environ.update(env)',
                        # measure.backend_env_names: NAMES only, so the serve executor can refuse inherited switches
                        'sorted(k for k in os.environ if k.startswith(("E4B_", "GNF4_")))'):
            text = text.replace(allowed, "")
        assert "os.environ" not in text, f
