"""Read D7 with the immutable merged DQ7 reducer, retaining source checksums; never imports planner observations."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from loggetta.dense_calibration import read

DQ7_SHA = "4ca5a3494e746b9aaa812270e03f4d0c1ec050a9"
D7_REGISTRATION_SHA = "a4fbb266eda101a5d1a27434929213fb542770d9"


def run(directory, e4b):
    e4b, directory = Path(e4b).resolve(), Path(directory).resolve()
    sha = subprocess.check_output(["git", "-C", str(e4b), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(e4b), "status", "--porcelain"], text=True).strip()
    if sha != DQ7_SHA or dirty:
        raise ValueError("D7 requires the clean immutable merged DQ7 reducer checkout")
    path = e4b/"bench/dq7/dq7_reduce.py"
    verdict = json.loads(subprocess.check_output([sys.executable, str(path), str(directory)], text=True))
    sources = sorted(directory.glob("read-*.json"))
    rows = [json.loads(p.read_text()) for p in sources]
    report = read(rows, verdict)
    report.update(dq7_verdict=verdict, registration_commit=D7_REGISTRATION_SHA, reducer_commit=sha,
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [directory/"proof.json", *sources]},
                  reducer_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("receipts")
    parser.add_argument("--e4b", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = run(args.receipts, args.e4b)
    with open(args.out, "x") as f:
        f.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"verdict": report["verdict"], "out": args.out, "reserve_import_licensed": False}))
