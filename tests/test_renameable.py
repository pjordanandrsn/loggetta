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
    from loggetta.runtime import RECEIPT_SCHEMA

    for s in (PLAN_SCHEMA, RECEIPT_SCHEMA):
        assert NAME.lower() not in s.lower()


def test_no_environment_variable_is_read():
    # configuration is typed objects and CLI flags; an env var would be a second, renaming-hostile surface
    for f in PKG.rglob("*.py"):
        text = f.read_text()
        assert "os.environ" not in text.replace('os.environ.get("CUDA_VISIBLE_DEVICES")', ""), f
