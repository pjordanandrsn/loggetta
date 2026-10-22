#!/usr/bin/env bash
# A first run on a fresh machine: a new virtual environment, the torch build that matches this NVIDIA driver installed
# FIRST, then this package, `inspect`, `plan` and a short training run on the smallest supported MoE.
#
#   bash first_run.sh [DIR]          DIR: a new directory for the environment, the plan, the receipt and the summary
#                                    (default ./first-run); it must not exist yet
#
# Environment (all optional):
#   PACKAGE_SPEC   what to install (default: the PyPI release; a version pin, a wheel or a local path all work)
#   TORCH_INDEX    a PyTorch index tag (cu130, cu126, ...) instead of the one read from the driver, or "pypi" for the
#                  torch PyPI resolves by default (the first-use matrix's default-torch cell)
#   MODEL, REVISION, STEPS, SEQ   the training run (default allenai/OLMoE-1B-7B-0924 at a pinned revision, 10, 512)
#   PYTHON         the interpreter that makes the environment (default python3; 3.10 or later)
#
# Every step's exit status and seconds go to DIR/first_run.json. The script exits 0 when the run trained, or with a
# NAMED refusal (its reason printed and recorded), never a bare traceback: 3 = no torch build fits this machine,
# 4 = the plan was refused, 5 = an install step failed, 6 = the run did not train.
set -uo pipefail

DIR=${1:-./first-run}
PACKAGE_SPEC=${PACKAGE_SPEC:-loggetta}
MODEL=${MODEL:-allenai/OLMoE-1B-7B-0924}
REVISION=${REVISION:-7f1c97f440f06ce36705e4f2b843edb5925f4498}
STEPS=${STEPS:-10}
SEQ=${SEQ:-512}
PYTHON=${PYTHON:-python3}

if [ -e "$DIR" ]; then
  echo "refusing: $DIR already exists; a first run starts from a new directory" >&2
  exit 2
fi
mkdir -p "$DIR" || exit 2
DIR=$(cd "$DIR" && pwd)
LOG="$DIR/steps.tsv"
: > "$LOG"
T0=$(date +%s)

# step NAME CMD...: run CMD, append "NAME<TAB>rc<TAB>seconds" to the log, and return its exit status
step() {
  local name=$1; shift
  local s; s=$(date +%s)
  echo "== $name: $*" >&2
  "$@"
  local rc=$?
  printf '%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - s ))" >> "$LOG"
  return $rc
}

# finish CODE REASON: write first_run.json from the step log, print the reason, exit CODE
finish() {
  local code=$1 reason=${2:-}
  "$PYTHON" - "$DIR" "$code" "$reason" "$T0" "${TAG:-}" "$MODEL" "$REVISION" "$STEPS" "$SEQ" "$PACKAGE_SPEC" <<'PY'
import json, os, platform, subprocess, sys, time

d, code, reason, t0, tag, model, rev, steps, seq, spec = sys.argv[1:]
rows = []
for line in open(os.path.join(d, "steps.tsv")):
    name, rc, secs = line.rstrip("\n").split("\t")
    rows.append({"step": name, "exit": int(rc), "seconds": int(secs)})
receipts = sorted(f for f in os.listdir(os.path.join(d, "receipts")) if f.endswith(".json")) \
    if os.path.isdir(os.path.join(d, "receipts")) else []
receipt = None
if receipts:
    with open(os.path.join(d, "receipts", receipts[-1])) as f:
        r = json.load(f)
    receipt = {"run_id": r.get("run_id"), "status": r.get("status"),
               "s_per_step_median": (r.get("measured") or {}).get("s_per_step_median")}
pyexe = os.path.join(d, "venv", "bin", "python")
versions = {}
if os.path.exists(pyexe):
    out = subprocess.run([pyexe, "-m", "pip", "list", "--format=json"], capture_output=True, text=True).stdout
    try:
        versions = {p["name"].lower(): p["version"] for p in json.loads(out)
                    if p["name"].lower() in ("torch", "loggetta", "experts4bit-qlora", "grouped-nf4-gemm",
                                             "transformers", "bitsandbytes")}
    except ValueError:
        pass
try:
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True, timeout=60).stdout.strip()
except (OSError, subprocess.SubprocessError):
    smi = ""
summary = {"schema": "first-run/1", "exit": int(code), "refusal": reason or None,
           "trained": int(code) == 0, "seconds_to_trained": int(time.time()) - int(t0) if int(code) == 0 else None,
           "torch_index": tag or None, "gpu": smi or None, "platform": platform.platform(),
           "workload": {"model": model, "revision": rev, "steps": int(steps), "seq": int(seq)},
           "package_spec": spec, "versions": versions, "steps": rows, "receipt": receipt}
with open(os.path.join(d, "first_run.json"), "w") as f:
    json.dump(summary, f, indent=1, sort_keys=True)
PY
  if [ "$code" -eq 0 ]; then
    echo "first run: trained; summary in $DIR/first_run.json" >&2
  else
    echo "first run refused ($code): $reason" >&2
    echo "summary in $DIR/first_run.json" >&2
  fi
  exit "$code"
}

"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null \
  || { echo "refusing: $PYTHON is not Python 3.10 or later" >&2; exit 2; }

VPY="$DIR/venv/bin/python"
step venv "$PYTHON" -m venv "$DIR/venv" || finish 5 "could not create a virtual environment with $PYTHON"
step pip_upgrade "$VPY" -m pip install -q --upgrade pip || finish 5 "could not upgrade pip in the new environment"

# 1. Read the driver BEFORE any torch exists: the package alone (its modules import only the standard library at
#    load time), asked which PyTorch build this driver runs; then removed, so the full install below resolves cleanly.
TAG=${TORCH_INDEX:-}
if [ -z "$TAG" ]; then
  step inspect_driver_install "$VPY" -m pip install -q --no-deps "$PACKAGE_SPEC" \
    || finish 5 "could not install $PACKAGE_SPEC (without dependencies) to read the driver"
  TAG=$("$VPY" -m loggetta inspect --torch-index 2> "$DIR/torch_index.err")
  rc=$?
  printf '%s\t%s\t%s\n' "torch_index" "$rc" "0" >> "$LOG"
  if [ "$rc" -eq 3 ]; then
    finish 3 "$(tr '\n' ' ' < "$DIR/torch_index.err")"
  elif [ "$rc" -ne 0 ] || [ -z "$TAG" ]; then
    # a release from before `inspect --torch-index` exits 2 on the unknown option: that is not "no build fits"
    finish 5 "the installed package cannot name a torch build (inspect --torch-index exited $rc; releases before it \
lack the option; set TORCH_INDEX to a build tag, e.g. cu126): $(tail -2 "$DIR/torch_index.err" | tr '\n' ' ')"
  fi
  step inspect_driver_uninstall "$VPY" -m pip uninstall -q -y loggetta || finish 5 "could not remove the bootstrap install"
fi

# 2. torch FIRST, from its own index, so the package's dependency on torch is already satisfied by the right build
if [ "$TAG" = pypi ]; then
  step torch "$VPY" -m pip install -q torch || finish 5 "could not install torch from PyPI"
else
  step torch "$VPY" -m pip install -q torch --index-url "https://download.pytorch.org/whl/$TAG" \
    || finish 5 "could not install torch from the $TAG index"
fi

# 3. the package and its stack
step install "$VPY" -m pip install -q "$PACKAGE_SPEC" || finish 5 "could not install $PACKAGE_SPEC"

# 4. inspect (the torch it installed is asked whether it can use the GPU), then plan, then train
step inspect "$VPY" -m loggetta inspect > "$DIR/inspect.txt" 2>&1
cat "$DIR/inspect.txt" >&2
step plan "$VPY" -m loggetta plan "$MODEL" --revision "$REVISION" --seq "$SEQ" --micro-batch 1 --steps "$STEPS" \
  --out "$DIR/plan.json" > "$DIR/plan.txt" 2>&1
rc=$?
cat "$DIR/plan.txt" >&2
if [ "$rc" -ne 0 ]; then
  why=$(grep -m3 -iE 'refus|cannot|reason|not feasible' "$DIR/plan.txt" | tr '\n' ' ')
  [ -n "$why" ] || why=$(tail -3 "$DIR/plan.txt" | tr '\n' ' ')
  finish 4 "the plan was refused (exit $rc): $why"
fi
step execute "$VPY" -m loggetta execute "$DIR/plan.json" --out "$DIR/receipts" > "$DIR/execute.txt" 2>&1
rc=$?
tail -25 "$DIR/execute.txt" >&2
[ "$rc" -eq 0 ] || finish 6 "the run did not train (exit $rc): $(tail -3 "$DIR/execute.txt" | tr '\n' ' ')"
finish 0 ""
