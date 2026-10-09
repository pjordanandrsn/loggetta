"""Every receipt an evidence document cites is in git.

``receipts/`` is gitignored (run output stays local), so a receipt meant as evidence has to be added with ``git add -f``.
When that step was missed (#49's granite re-run), the README still cited the receipt, a plan-vs-driver table had a row
built from it, and nothing failed. This test reads every Markdown file under ``evidence/`` and requires each path that
names a ``receipts/`` entry to be tracked: the path itself, the path plus ``.json``, or a directory holding tracked files.
Paths resolve against the document's own directory and against ``evidence/`` (the plan-vs-driver tables cite runs by
their path under it).
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EVIDENCE = ROOT / "evidence"
#: a path with at least one directory before ``receipts/``; a bare ``receipts/`` is the directory name, not a citation
CITED = re.compile(r"[\w.\-]+(?:/[\w.\-…]+)*/receipts/[\w.\-…]*")


def _tracked() -> set:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "evidence"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return {f for f in out.decode().split("\0") if f}


def _resolves(cited: str, doc: Path, tracked: set) -> bool:
    if "…" in cited:                                    # an elided path in prose names no single file
        return True
    for base in (doc.parent, EVIDENCE):
        rel = (base / cited).resolve().relative_to(ROOT).as_posix().rstrip("/")
        if rel in tracked or rel + ".json" in tracked or any(f.startswith(rel + "/") for f in tracked):
            return True
    return False


def test_every_cited_receipt_is_in_git():
    tracked = _tracked()
    missing = []
    for doc in sorted(EVIDENCE.rglob("*.md")):
        if doc.relative_to(ROOT).as_posix() not in tracked:
            continue                                    # an untracked scratch document is not evidence
        for n, line in enumerate(doc.read_text().splitlines(), 1):
            for cited in CITED.findall(line):
                if not _resolves(cited, doc, tracked):
                    missing.append(f"{doc.relative_to(ROOT)}:{n}: {cited}")
    assert not missing, "cited receipts that are not in git (receipts/ is gitignored: add them with git add -f):\n" + \
        "\n".join(missing)
