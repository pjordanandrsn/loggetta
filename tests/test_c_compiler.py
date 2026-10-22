"""A machine without a working C compiler is refused by name at plan time, before any download.

Found by a first-use shakedown (explore, not a reading): on the CUDA runtime image, which ships no gcc, a first run
installed torch, planned and downloaded a 13.8 GB checkpoint, then died in training with Triton's "Failed to find C
compiler" traceback. Triton compiles a small helper against Python.h the first time a kernel runs, and the training and
serving kernels are Triton kernels. The probe checks the compiler the way Triton uses it, not by name.
"""
import os
import shutil
import sysconfig

import pytest

from loggetta import hardware
from loggetta.hardware import GPU, Fact, HardwareProfile, Host, c_compiler_problem, probe_c_compiler

GiB = 1 << 30
HAVE_CC = bool(os.environ.get("CC") or shutil.which("gcc") or shutil.which("clang"))


def nvidia():
    r = lambda v: Fact(v, "reported")  # noqa: E731
    return GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 9)),
               memory_total=r(24 * GiB), memory_free=r(24 * GiB), driver=r("560.35.03"), pcie_gen_max=r(4),
               pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))


def host(cc=None):
    r = lambda v: Fact(v, "reported")  # noqa: E731
    kw = {} if cc is None else {"c_compiler": cc}
    return Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB), **kw)


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler on this machine")
def test_a_working_compiler_is_recorded_by_name_only(monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    f = probe_c_compiler()
    assert f.value in ("gcc", "clang") and f.source == "reported" and "/" not in f.value


def test_a_compiler_that_cannot_build_is_refused_with_its_exit(monkeypatch):
    false = shutil.which("false")
    if not false:
        pytest.skip("no false(1) here")
    monkeypatch.setenv("CC", false)
    f = probe_c_compiler()
    assert f.value is None and f.source == "reported" and "false could not build a shared object" in f.note


def test_a_cc_that_does_not_exist_is_refused(monkeypatch):
    monkeypatch.setenv("CC", "/nonexistent/bin/cc")
    f = probe_c_compiler()
    assert f.value is None and "cc could not be run" in f.note and "/nonexistent" not in f.note


def test_an_empty_cc_counts_as_set_as_it_does_for_triton(monkeypatch):
    monkeypatch.setenv("CC", "")
    f = probe_c_compiler()
    assert f.value is None and "CC is set but empty" in f.note


def test_the_compilers_diagnostics_never_carry_a_path(monkeypatch, tmp_path):
    """gcc's errors name the temp file, the include chain and, in a venv, a home directory. Plans, inspect and
    first_run.json are shared, so the recorded note keeps only basenames. (The home path is built at run time.)"""
    home = os.sep + os.path.join("home", "someone", ".venv", "include")
    fake = tmp_path / "fakecc"
    fake.write_text("#!/bin/sh\n"
                    f"echo \"In file included from $1:1:\" >&2\n"
                    f"echo \"{home}/Python.h:12:10: fatal error: pyconfig.h: No such file or directory\" >&2\n"
                    "exit 1\n")
    fake.chmod(0o755)
    monkeypatch.setenv("CC", str(fake))
    f = probe_c_compiler()
    assert f.value is None and f.note.startswith("fakecc could not build")
    assert "/" not in f.note and "someone" not in f.note and str(tmp_path) not in f.note
    assert "Python.h:12:10: fatal error: pyconfig.h" in f.note and "probe.c:1" in f.note      # the diagnosis survives


def test_no_compiler_at_all_is_refused(monkeypatch):
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(shutil, "which", lambda name: None)
    f = probe_c_compiler()
    assert f.value is None and f.note.startswith("no C compiler")


@pytest.mark.skipif(not HAVE_CC, reason="no C compiler on this machine")
def test_a_compiler_without_python_headers_is_refused(monkeypatch, tmp_path):
    """gcc on PATH but no Python.h: a which-check would pass; the build does not."""
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setattr(sysconfig, "get_paths", lambda *a, **k: {"include": str(tmp_path)})
    monkeypatch.setattr(sysconfig, "get_config_var", lambda *a, **k: None)
    f = probe_c_compiler()
    if f.value is not None:
        pytest.skip("this compiler finds Python.h on its own default path")
    assert "Python.h" in f.note


def test_only_a_definite_absence_refuses():
    gpu = nvidia()
    assert c_compiler_problem(host(), gpu) is None                                        # not probed: unknown
    assert c_compiler_problem(host(Fact("gcc", "reported")), gpu) is None                  # works
    why, fixes = c_compiler_problem(host(Fact(None, "reported", "no C compiler: ...")), gpu)
    assert why.startswith("no working C compiler") and "Triton" in why
    assert any("build-essential" in f and "python3-dev" in f for f in fixes)
    assert c_compiler_problem(host(Fact(None, "reported", "x")), None) is None             # no GPU: not this refusal


def test_a_profile_saved_before_the_field_loads_as_not_probed():
    hw = HardwareProfile(gpus=(nvidia(),), host=host(), platform="Linux x86_64")
    d = hw.to_dict()
    del d["host"]["c_compiler"]
    back = HardwareProfile.from_dict(d)
    assert back.host.c_compiler.source == "unknown" and c_compiler_problem(back.host, back.gpus[0]) is None


def test_the_probe_runs_only_when_an_nvidia_gpu_is_found(monkeypatch):
    monkeypatch.setattr(hardware, "_probe_gpus", lambda notes: iter(()))
    monkeypatch.setattr(hardware, "probe_c_compiler", lambda: pytest.fail("probed a machine with no NVIDIA GPU"))
    assert hardware.probe(torch=False).host.c_compiler.source == "unknown"


@pytest.fixture(scope="module")
def topo():
    tr = pytest.importorskip("transformers")
    pytest.importorskip("experts4bit_qlora.recipe")
    from loggetta import describe_model

    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def test_the_plan_refuses_by_name_before_anything_is_loaded(topo):
    from loggetta import Workload, plan

    hw = HardwareProfile(gpus=(nvidia(),), host=host(Fact(None, "reported", "no C compiler: CC is unset ...")),
                         platform="Linux x86_64")
    p = plan(topo, hw, Workload(seq_len=512))
    assert p.status == "refused" and p.refusal["reasons"][0].startswith("no working C compiler")
    ok = plan(topo, HardwareProfile(gpus=(nvidia(),), host=host(Fact("gcc", "reported")), platform="Linux x86_64"),
              Workload(seq_len=512))
    assert ok.status == "feasible"
