"""Every run an evidence log names has its receipt committed, or an explicit note says why not.

``receipts/`` is gitignored (run output stays local), so a receipt meant as evidence has to be added with ``git add -f``.
That step was missed twice (#49's granite re-run, #48's absmax arms) and both times a reader found it, not CI. This test
reads every committed log under ``evidence/`` for the two ways a run names its receipt -- loggetta's CLI line
``receipt: <path>/<run_id>.json`` and ``bench/train_residual.py``'s ``run <run_id> status ...`` -- and requires a committed
``<run_id>.json`` anywhere under ``evidence/``, or a line ``no receipt: <run_id>`` (backticks allowed) in a Markdown file
in the log's evidence directory saying why. A run that stopped before writing one names none and needs nothing.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
RECEIPT_LINE = re.compile(r"receipt: \S*?([\w.\-]+)\.json\b")
RUN_LINE = re.compile(r"^run (\S+) status\b", re.M)
NO_RECEIPT = re.compile(r"no receipt: `?([\w.\-]+)`?", re.I)


def _tracked() -> list:
    try:
        out = subprocess.run(["git", "ls-files", "-z", "evidence"], cwd=ROOT, capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    return [f for f in out.decode().split("\0") if f]


def _evidence_dir(path: str) -> str:
    parts = path.split("/")
    return "/".join(parts[:2]) if len(parts) > 2 else parts[0]      # evidence/<dir>, or evidence/ for top-level files


def test_every_run_an_evidence_log_names_has_a_committed_receipt_or_a_note():
    tracked = _tracked()
    receipts = {Path(f).stem for f in tracked if f.endswith(".json")}
    notes = {}
    for f in tracked:
        if f.endswith(".md"):
            notes.setdefault(_evidence_dir(f), set()).update(NO_RECEIPT.findall((ROOT / f).read_text()))
    missing = []
    for f in tracked:
        if not f.endswith((".log", ".log.txt")):
            continue
        text = (ROOT / f).read_text(errors="replace")
        for run_id in sorted(set(RECEIPT_LINE.findall(text)) | set(RUN_LINE.findall(text))):
            if run_id not in receipts and run_id not in notes.get(_evidence_dir(f), ()):
                missing.append(f"{f}: {run_id}")
    assert not missing, ("runs named in evidence logs with no committed receipt (receipts/ is gitignored: git add -f "
                         "it, or write 'no receipt: <run_id>' and why in the directory's README):\n" + "\n".join(missing))
