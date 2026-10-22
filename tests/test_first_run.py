"""The first run's torch choice (T1): which PyTorch build a driver runs, read before torch is installed, and the
script's refusals. No GPU, no network."""
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from loggetta import cli, hardware as H

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "first_run.sh"


@pytest.mark.parametrize("driver, tag", [
    ("580.65.06", "cu130"), ("590.10", "cu130"),       # CUDA 13 needs 580
    ("575.57.08", "cu126"), ("525.60.13", "cu126"),    # a CUDA 12 build runs on 525 and later
])
def test_the_newest_build_the_driver_runs(driver, tag):
    got, why = H.torch_build_for_driver(driver)
    assert got == tag and why.startswith(f"driver {driver.split('.')[0]} runs CUDA")


@pytest.mark.parametrize("driver, match", [
    ("470.256.02", "older than every supported torch build needs: the oldest, cu126, needs driver 525"),
    (None, "could not be read"), ("n/a", "could not be read"),
])
def test_no_build_fits_is_a_named_reason(driver, match):
    tag, why = H.torch_build_for_driver(driver)
    assert tag is None and match in why


def test_every_build_names_a_known_cuda_and_the_list_is_newest_first():
    majors = [m for _, m in H.TORCH_BUILDS]
    assert all(m in H.MIN_DRIVER_FOR_CUDA for m in majors) and majors == sorted(majors, reverse=True)


def gpu(driver="575.57.08", vendor="nvidia"):
    f = lambda v: H.Fact(v, "reported")  # noqa: E731
    return H.GPU(index=0, vendor=vendor, name="Test GPU", uuid=None, compute_capability=f((8, 9)),
                 memory_total=f(24 << 30), memory_free=f(24 << 30), driver=f(driver), pcie_gen_max=f(4),
                 pcie_width_max=f(16), pcie_gen_current=f(4), pcie_width_current=f(16))


HOST = H.probe(torch=False).host                   # read once, before any test replaces probe


def profile(gpus):
    return H.HardwareProfile(gpus=tuple(gpus), host=HOST, platform="test", notes=("a note",), torch=None)


@pytest.mark.parametrize("gpus, rc, out, err", [
    ([gpu("575.57.08")], 0, "cu126", ""),
    ([gpu("580.65.06")], 0, "cu130", ""),
    ([gpu("470.256.02")], 3, "", "no torch build fits GPU 0 (Test GPU): driver 470 is older"),
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
    assert ("torch build for this driver: cu126 (driver 575 runs CUDA 12.x builds (needs 525 or later)); install it "
            "first: pip install torch --index-url https://download.pytorch.org/whl/cu126") in text


def test_the_script_refuses_an_existing_directory_and_parses(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    r = subprocess.run(["bash", str(SCRIPT), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 2 and "already exists; a first run starts from a new directory" in r.stderr
    assert list(tmp_path.iterdir()) == []          # nothing written into it
