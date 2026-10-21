"""torch's CUDA facts: inspect reports them, and plan refuses a CUDA plan that the installed torch cannot run.

The failure this guards: ``pip install loggetta`` pulled torch 2.14.1+cu130 (CUDA 13.0 needs driver 580) onto a driver
575 machine. nvidia-smi listed the GPU, so inspect and plan said feasible, and execute died in
``torch.cuda.set_device`` ("driver too old"). torch is mocked here: its answer is stated text, as nvidia-smi's is.
"""
import json
import types
from dataclasses import replace

import pytest

import loggetta
import loggetta.hardware as h
from loggetta.hardware import (GPU, Fact, HardwareProfile, Host, TorchCuda, describe, parse_torch_probe, probe,
                               torch_cuda_problem)

GiB = 1 << 30
INSTALL = f"then pip install {loggetta.__name__}"   # the package's working name, never spelled
SMI_575 = "0, GPU-01aa, NVIDIA RTX A2000 12GB, 8.6, 12282, 12000, 575.64.05, 3, 16, 3, 8\n"
#: torch 2.14.1+cu130 on driver 575, as the child prints it (the warning is torch's own text)
TOO_OLD = json.dumps({"version": "2.14.1+cu130", "cuda": "13.0", "available": False, "device_count": 0,
                      "warnings": [("CUDA initialization: The NVIDIA driver on your system is too old (found version "
                                    "12090). Please update your GPU driver by downloading and installing a new version "
                                    "from the URL: http://www.nvidia.com/Download/index.aspx")]})
WORKS = json.dumps({"version": "2.14.1+cu128", "cuda": "12.8", "available": True, "device_count": 1, "warnings": []})


def _gpu(driver="575.64.05"):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return GPU(index=0, vendor="nvidia", name="NVIDIA RTX A2000 12GB", uuid=None, compute_capability=r((8, 6)),
               memory_total=r(12 * GiB), memory_free=r(12 * GiB), driver=r(driver), pcie_gen_max=r(4),
               pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))


def _hw(torch=None, driver="575.64.05"):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(_gpu(driver),), host=host, platform="Linux x86_64", torch=torch)


# ------------------------------------------------------------------------------------------------ the child's answer

def test_parse_torch_probe_keeps_torchs_own_warning():
    t = parse_torch_probe("some import chatter\n" + TOO_OLD + "\n")
    assert (t.version, t.cuda, t.available, t.device_count) == ("2.14.1+cu130", "13.0", False, 0)
    assert t.note.startswith("CUDA initialization: The NVIDIA driver on your system is too old")
    w = parse_torch_probe(WORKS)
    assert (w.available, w.device_count, w.note) == (True, 1, "")


def test_an_unreadable_answer_is_unknown_never_false():
    t = parse_torch_probe("Traceback (most recent call last):\nImportError: libcudart.so.13", "2.14.1+cu130")
    assert t.available is None and t.version == "2.14.1+cu130" and "could not be read" in t.note
    assert torch_cuda_problem(t, _gpu()) is None                 # not asked is not proof: no refusal


def test_probe_asks_torch_only_with_an_nvidia_gpu(monkeypatch):
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=SMI_575 if "nvidia-smi" in cmd[0] else TOO_OLD, stderr="", returncode=0)

    monkeypatch.setattr(h.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(h.subprocess, "run", run)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "2.14.1+cu130")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    hw = probe()
    assert hw.torch == replace(parse_torch_probe(TOO_OLD), visible_devices="")   # as the child saw it
    assert calls[-1][1:3] == ["-c", h._TORCH_PROBE]              # a child process: this one never imports torch
    assert probe(torch=False).torch is None


def test_probe_never_asks_torch_without_an_nvidia_gpu(monkeypatch):
    """No GPU discovered (no nvidia-smi, no AMD tool, no /dev/kfd): no torch child, and no torch facts."""
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        raise AssertionError(f"nothing should run: {cmd}")

    monkeypatch.setattr(h.shutil, "which", lambda name: None)
    monkeypatch.setitem(__import__("sys").modules, "amdsmi", None)
    real_exists = h.os.path.exists
    monkeypatch.setattr(h.os.path, "exists", lambda p: False if p == "/dev/kfd" else real_exists(p))
    monkeypatch.setattr(h.subprocess, "run", run)
    hw = probe()
    assert hw.gpus == () and hw.torch is None and calls == []


def test_execute_and_the_dq10_check_take_gpu_facts_without_asking_torch(monkeypatch):
    """execute's device check and the DQ10 policy read only GPU facts, so they never spawn the torch child."""
    import loggetta.hardware
    from loggetta.execution import PlanNotExecutable, check_here

    seen = []

    def fake_probe(**kw):
        seen.append(kw)
        return HardwareProfile(gpus=(), host=_hw().host, platform="Linux x86_64")

    monkeypatch.setattr(loggetta.hardware, "probe", fake_probe)
    plan = types.SimpleNamespace(hardware={"gpu": {"name": "x"}}, constraints=types.SimpleNamespace(device=0))
    with pytest.raises(PlanNotExecutable):
        check_here(plan)
    assert seen == [{"torch": False}]
    import inspect

    import loggetta.dense_policy
    assert "probe(torch=False).gpu(plan.constraints.device)" in inspect.getsource(loggetta.dense_policy)


def test_probe_records_torch_not_installed(monkeypatch):
    from importlib.metadata import PackageNotFoundError

    def missing(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(h.shutil, "which", lambda name: "/usr/bin/nvidia-smi" if name == "nvidia-smi" else None)
    monkeypatch.setattr(h.subprocess, "run", lambda cmd, **kw: types.SimpleNamespace(stdout=SMI_575))
    monkeypatch.setattr("importlib.metadata.version", missing)
    t = probe().torch
    assert t.version is None and t.note == "torch is not installed"
    assert torch_cuda_problem(t, _gpu()) is None                 # the backends say what is missing


def test_a_profile_with_torch_round_trips_and_an_older_one_still_loads():
    hw = _hw(parse_torch_probe(TOO_OLD))
    back = HardwareProfile.from_dict(json.loads(json.dumps(hw.to_dict())))
    assert back.torch == hw.torch
    old = hw.to_dict()
    del old["torch"]                                             # saved by 0.6.1
    assert HardwareProfile.from_dict(old).torch is None


# ----------------------------------------------------------------------------------------------------------- inspect

def test_inspect_reports_torchs_cuda_facts_and_the_remedy():
    text = describe(_hw(parse_torch_probe(TOO_OLD)))
    assert "GPU 0: NVIDIA RTX A2000 12GB  (sm_86, driver 575.64.05)" in text
    assert "torch: 2.14.1+cu130 (built for CUDA 13.0); torch.cuda.is_available() False" in text
    assert "torch says: CUDA initialization: The NVIDIA driver on your system is too old" in text
    assert "a CUDA plan is refused here until torch can use the GPU" in text and "fresh virtual environment" in text
    ok = describe(_hw(parse_torch_probe(WORKS)))
    assert "torch.cuda.is_available() True, 1 device(s)" in ok and "refused" not in ok


# -------------------------------------------------------------------------------------------------------------- plan

def test_the_problem_names_torch_the_driver_and_the_order_of_the_fix():
    reason, remedies = torch_cuda_problem(parse_torch_probe(TOO_OLD), _gpu())
    assert reason.startswith("torch 2.14.1+cu130 (built for CUDA 13.0) cannot use GPU 0 (NVIDIA RTX A2000 12GB, "
                             "driver 575.64.05): torch.cuda.is_available() is False; torch says: CUDA initialization")
    assert reason.endswith("CUDA 13.x needs driver 580 or later on Linux")
    assert "a torch build for CUDA 12.x (driver 575 runs one) FIRST" in remedies[0]
    assert remedies[0].endswith(INSTALL)
    assert "pip counts torch 2.14.1 as already satisfied" in remedies[1]
    assert remedies[2] == "or update the NVIDIA driver to 580 or later, which runs this build as installed"
    cpu = torch_cuda_problem(TorchCuda("2.14.1+cpu", None, False, 0), _gpu())
    assert "(a CPU-only build)" in cpu[0] and "a CUDA build of torch" in cpu[1][0] and len(cpu[1]) == 2
    assert torch_cuda_problem(parse_torch_probe(WORKS), _gpu()) is None


def test_a_driver_that_meets_the_build_is_not_told_to_reinstall_torch():
    """Driver 590 runs a cu130 build, so is_available False has another cause (CUDA_VISIBLE_DEVICES, a container without
    GPU access): the refusal says the driver meets the minimum, shows the variable and torch's note, and gives no
    reinstall remedy."""
    t = replace(parse_torch_probe(json.dumps({"version": "2.14.1+cu130", "cuda": "13.0", "available": False,
                                              "device_count": 0, "warnings": ["CUDA initialization: CUDA driver "
                                                                              "initialization failed."]})),
                visible_devices="")
    reason, remedies = torch_cuda_problem(t, _gpu(driver="590.44.01"))
    assert reason.endswith("torch says: CUDA initialization: CUDA driver initialization failed; driver 590 meets "
                           "CUDA 13.x's minimum (580), so the build is not the cause; CUDA_VISIBLE_DEVICES is ''")
    assert ".;" not in reason
    assert remedies[0].startswith("make GPU 0 visible to this process: check CUDA_VISIBLE_DEVICES (now '')")
    assert not any("fresh virtual environment" in r or "pip" in r for r in remedies)
    unset = torch_cuda_problem(replace(t, visible_devices=None), _gpu(driver="590.44.01"))
    assert "CUDA_VISIBLE_DEVICES is unset" in unset[0]
    mismatch = torch_cuda_problem(replace(t, visible_devices=None), _gpu())   # driver 575: the build is the cause
    assert "fresh virtual environment" in mismatch[1][0] and "meets" not in mismatch[0]
    edge = torch_cuda_problem(t, _gpu(driver="580.65.06"))                   # exactly the minimum meets it
    assert "driver 580 meets CUDA 13.x's minimum (580)" in edge[0]
    below = torch_cuda_problem(t, _gpu(driver="579.99"))
    assert "CUDA 13.x needs driver 580 or later on Linux" in below[0] and "meets" not in below[0]
    quiet = torch_cuda_problem(replace(t, note=""), _gpu(driver="590.44.01"))  # no torch message: none is cited
    assert quiet[1][1] == "reinstalling torch will not change it" and "torch says" not in quiet[0]


@pytest.fixture(scope="module")
def topo():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import describe_model

    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def test_plan_refuses_a_cuda_plan_torch_cannot_run(topo):
    from loggetta import Workload, plan

    p = plan(topo, _hw(parse_torch_probe(TOO_OLD)), Workload())
    assert p.status == "refused" and p.selected is None
    assert p.refusal["reasons"] == [torch_cuda_problem(parse_torch_probe(TOO_OLD), _gpu())[0]]
    assert any(s.endswith(INSTALL) for s in p.refusal["suggestions"])
    assert p.hardware["torch"]["available"] is False
    text = p.render()
    assert "NOT FEASIBLE" in text and "torch.cuda.is_available() is False" in text and "already satisfied" in text


def test_the_normal_path_is_unchanged(topo):
    """A torch that can use the GPU, and a profile without torch facts, plan exactly as before."""
    from loggetta import Workload, plan

    before = plan(topo, _hw(None), Workload())
    works = plan(topo, _hw(parse_torch_probe(WORKS)), Workload())
    assert before.status == "feasible" and "torch" not in before.hardware
    a, b = json.loads(before.to_json()), json.loads(works.to_json())
    assert b["hardware"].pop("torch")["available"] is True
    assert a == b
    unknown = plan(topo, _hw(parse_torch_probe("garbage", "2.14.1+cu130")), Workload())
    assert unknown.status == "feasible"
    assert any("could not ask torch 2.14.1+cu130" in w for w in unknown.warnings)
