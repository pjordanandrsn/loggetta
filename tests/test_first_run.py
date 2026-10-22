"""The first run's torch choice (T1): which PyTorch build a driver runs, read before torch is installed, and the
script's refusals. No GPU, no network."""
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from loggetta import cli, hardware as H

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "first_run.sh"


@pytest.mark.parametrize("driver, cc, tag", [
    ("580.65.06", (8, 9), "cu130"), ("590.10", None, "cu130"),           # CUDA 13 needs 580
    ("575.57.08", (8, 9), "cu126"), ("525.60.13", (8, 6), "cu126"),     # the newest torch a 12.x driver runs
    ("580.65.06", (12, 0), "cu130"),                                    # Blackwell on a 13.x driver
    ("575.57.08", (12, 0), "cu129"), ("570.86.10", (10, 0), "cu129"),   # Blackwell: CUDA 12.8 or later, not cu126
])
def test_the_build_fits_the_driver_and_the_architecture(driver, cc, tag):
    got, why = H.torch_build_for_driver(driver, cc)
    assert got == tag and why.startswith(f"driver {driver.split('.')[0]} runs CUDA")
    if cc and cc >= (10, 0):
        assert "needs a build for CUDA 12.8 or later" in why


@pytest.mark.parametrize("driver, cc, match", [
    ("470.256.02", (8, 6), "older than every supported torch build needs: the oldest needs driver 525"),
    ("565.57.01", (12, 0), "compute capability 12.0 (Blackwell) needs driver 570 or later; this one is 565"),
    (None, None, "could not be read"), ("n/a", (8, 9), "could not be read"),
])
def test_no_build_fits_is_a_named_reason(driver, cc, match):
    tag, why = H.torch_build_for_driver(driver, cc)
    assert tag is None and match in why


def test_every_build_is_a_known_cuda_with_the_drivers_minimum():
    for tag, cuda, need in H.TORCH_BUILDS:
        assert tag == f"cu{cuda[0]}{cuda[1]}" and need == H.MIN_DRIVER_FOR_CUDA[cuda[0]]
    assert any(cuda >= H.BLACKWELL_MIN_CUDA and need <= H.BLACKWELL_MIN_DRIVER for _, cuda, need in H.TORCH_BUILDS)


def gpu(driver="575.57.08", vendor="nvidia", cc=(8, 9)):
    f = lambda v: H.Fact(v, "reported")  # noqa: E731
    return H.GPU(index=0, vendor=vendor, name="Test GPU", uuid=None, compute_capability=f(cc),
                 memory_total=f(24 << 30), memory_free=f(24 << 30), driver=f(driver), pcie_gen_max=f(4),
                 pcie_width_max=f(16), pcie_gen_current=f(4), pcie_width_current=f(16))


HOST = H.probe(torch=False).host                   # read once, before any test replaces probe


def profile(gpus):
    return H.HardwareProfile(gpus=tuple(gpus), host=HOST, platform="test", notes=("a note",), torch=None)


@pytest.mark.parametrize("gpus, rc, out, err", [
    ([gpu("575.57.08")], 0, "cu126", ""),
    ([gpu("580.65.06")], 0, "cu130", ""),
    ([gpu("470.256.02")], 3, "", "no torch build fits GPU 0 (Test GPU): driver 470 is older"),
    ([gpu("575.57.08", cc=(12, 0))], 0, "cu129", ""),
    ([], 3, "", "no NVIDIA GPU discovered"),
])
def test_inspect_torch_index_prints_only_the_tag_or_refuses_by_name(monkeypatch, capsys, gpus, rc, out, err):
    seen = {}

    def probe(**kw):
        seen.update(kw)
        return profile(gpus)

    monkeypatch.setattr(H, "probe", probe)
    assert cli.main(["inspect", "--torch-index"]) == rc
    got = capsys.readouterr()
    assert got.out.strip() == out and err in got.err
    assert seen == {"torch": False}                # torch is not asked: it is not installed yet


def test_inspect_prints_the_build_for_the_driver():
    hw = replace(profile([gpu("575.57.08")]), torch=H.TorchCuda(version=None, cuda=None, available=None,
                                                                device_count=None, note=""))
    text = H.describe(hw)
    assert ("torch build for this driver: cu126 (driver 575 runs CUDA 12.6 builds (needs 525 or later)); install it "
            "first: pip install torch --index-url https://download.pytorch.org/whl/cu126") in text


def test_the_script_refuses_an_existing_directory_and_parses(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 2 and "already exists; a first run starts from a new directory" in r.stderr
    assert list(tmp_path.iterdir()) == []          # nothing written into it


STUB = r'''#!/usr/bin/env bash
# A stand-in python for first_run.sh: real python for -c / - (the version check, the summary); a simulated venv, pip and
# package CLI otherwise, driven by STUB_* variables. No network, no GPU.
if [ "${1:-}" = "-c" ] || [ "${1:-}" = "-" ]; then exec "$REAL_PY" "$@"; fi
if [ "${1:-}" = "-m" ]; then
  mod=$2; shift 2
  case "$mod" in
    venv) mkdir -p "$1/bin" && ln -s "$STUB_SELF" "$1/bin/python"; exit 0;;
    pip) [ "${1:-}" = "list" ] && echo "[]"; exit 0;;
    loggetta)
      case "${1:-} ${2:-}" in
        "inspect --torch-index") echo cu126; exit 0;;
        "inspect "*) echo Hardware; exit 0;;
      esac
      if [ "$1" = plan ]; then
        out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
        echo '{}' > "$out"; [ "${STUB_PLAN_RC:-0}" = 0 ] || echo "NOT FEASIBLE under the requested constraints."
        exit "${STUB_PLAN_RC:-0}"
      fi
      if [ "$1" = execute ]; then
        out=""; while [ $# -gt 0 ]; do [ "$1" = --out ] && out=$2; shift; done
        mkdir -p "$out"
        if [ "${STUB_STATUS:-OK}" = UNMEASURED ]; then
          printf '%s' '{"run_id": "r1", "status": "UNMEASURED", "measured": {"driver_samples": 0, "driver_unmeasured": "nvidia-smi answered 3 time(s) but never listed this process"}}' > "$out/r1.json"
          echo "device check NOT RUN (status UNMEASURED): nvidia-smi answered 3 time(s) but never listed this process"
        else
          printf '%s' "{\"run_id\": \"r1\", \"status\": \"${STUB_STATUS:-OK}\", \"measured\": {\"driver_samples\": 12}}" > "$out/r1.json"
        fi
        exit "${STUB_EXEC_RC:-0}"
      fi;;
  esac
fi
echo "stub python: unhandled $*" >&2; exit 99
'''


def stub_run(tmp_path, **env):
    import json
    import os
    import sys

    stub = tmp_path / "python"
    stub.write_text(STUB)
    stub.chmod(0o755)
    out = tmp_path / "run"
    e = {**os.environ, "PYTHON": str(stub), "REAL_PY": sys.executable, "STUB_SELF": str(stub), **env}
    r = subprocess.run(["bash", str(SCRIPT), str(out)], capture_output=True, text=True, env=e)
    return r, json.loads((out / "first_run.json").read_text())


@pytest.mark.parametrize("env, code, trained, check", [
    ({}, 0, True, "checked"),
    ({"STUB_STATUS": "UNMEASURED", "STUB_EXEC_RC": "4"}, 7, True, "not run"),     # WSL2: trained, device check not run
    ({"STUB_STATUS": "ALARM", "STUB_EXEC_RC": "1"}, 6, False, "checked"),
])
def test_first_run_reads_executes_exit_codes(tmp_path, env, code, trained, check):
    r, summary = stub_run(tmp_path, **env)
    assert r.returncode == code, r.stderr
    assert summary["exit"] == code and summary["trained"] is trained and summary["torch_index"] == "cu126"
    assert summary["receipt"]["device_check"] == check
    if code == 7:
        assert "trained; the device check did not run (status UNMEASURED)" in r.stderr
        assert summary["receipt"]["driver_unmeasured"].startswith("nvidia-smi answered 3 time(s)")
        assert summary["seconds_to_trained"] is not None


def test_first_run_names_a_refused_plan(tmp_path):
    r, summary = stub_run(tmp_path, STUB_PLAN_RC="2")
    assert r.returncode == 4 and summary["trained"] is False
    assert summary["refusal"].startswith("the plan was refused (exit 2): NOT FEASIBLE")


@pytest.mark.parametrize("spec, recorded", [
    ("loggetta", "loggetta"),                                   # a PyPI spec is not a path: recorded as given
    ("loggetta==0.7.0", "loggetta==0.7.0"),
    ("/srv/wheels/loggetta-0.7.0-py3-none-any.whl", "local: loggetta-0.7.0-py3-none-any.whl"),
])
def test_first_run_never_records_a_local_path(tmp_path, spec, recorded):
    """Users share first_run.json: an absolute PACKAGE_SPEC would carry their home directory and username with it."""
    _r, summary = stub_run(tmp_path, PACKAGE_SPEC=spec)
    assert summary["package_spec"] == recorded


def test_first_run_records_an_existing_relative_path_by_name(tmp_path, monkeypatch):
    wheel = tmp_path / "loggetta-0.7.0-py3-none-any.whl"
    wheel.write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    _r, summary = stub_run(tmp_path, PACKAGE_SPEC="./loggetta-0.7.0-py3-none-any.whl")
    assert summary["package_spec"] == "local: loggetta-0.7.0-py3-none-any.whl"


def test_a_bare_name_is_a_requirement_even_beside_a_directory_of_that_name(tmp_path, monkeypatch):
    (tmp_path / "loggetta").mkdir()
    monkeypatch.chdir(tmp_path)
    _r, summary = stub_run(tmp_path, PACKAGE_SPEC="loggetta")
    assert summary["package_spec"] == "loggetta"
