"""A plan records the backend environment switches its estimate read, and execute warns when the running process
differs: experts4bit-qlora's training estimate reads ``E4B_CHUNKED_LM_LOSS``, so a plan written in one process and run in
another can price a loss the run does not take."""
import pytest

tr = pytest.importorskip("transformers")
pytest.importorskip("experts4bit_qlora.recipe")

from loggetta import Workload, describe_model, plan  # noqa: E402
from loggetta.backends import experts4bit  # noqa: E402
from loggetta.execution import check_estimate_env  # noqa: E402
from loggetta.hardware import GPU, Fact, HardwareProfile, Host  # noqa: E402
from loggetta.plan import ExecutionPlan  # noqa: E402

GiB = 1 << 30


def hw():
    r = lambda v: Fact(v, "reported")  # noqa: E731
    gpu = GPU(index=0, vendor="nvidia", name="Test GPU", uuid=None, compute_capability=r((8, 6)),
              memory_total=r(12 * GiB), memory_free=r(12 * GiB), driver=r("575.64.05"), pcie_gen_max=r(4),
              pcie_width_max=r(16), pcie_gen_current=r(4), pcie_width_current=r(16))
    host = Host(cpu_model=r("test cpu"), cpus=r(8), memory_total=r(64 * GiB), memory_available=r(40 * GiB),
                memory_limit=r(64 * GiB))
    return HardwareProfile(gpus=(gpu,), host=host, platform="Linux x86_64")


@pytest.fixture(scope="module")
def topo():
    cfg = tr.Qwen3MoeConfig(hidden_size=1024, intermediate_size=2048, moe_intermediate_size=768, num_experts=128,
                            num_experts_per_tok=4, num_hidden_layers=8, num_attention_heads=8, num_key_value_heads=4,
                            head_dim=128, vocab_size=32000, max_position_embeddings=4096, decoder_sparse_step=1)
    return describe_model(cfg)


def test_a_plan_records_the_switches_its_estimate_read(topo, monkeypatch):
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: {"E4B_CHUNKED_LM_LOSS": "auto"})
    p = plan(topo, hw(), Workload(seq_len=512))
    assert p.provenance["estimate_env"] == {"E4B_CHUNKED_LM_LOSS": "auto"}
    assert ExecutionPlan.from_dict(p.to_dict()).provenance["estimate_env"] == {"E4B_CHUNKED_LM_LOSS": "auto"}


def test_a_backend_release_without_the_accessor_records_nothing(topo, monkeypatch):
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: None)
    p = plan(topo, hw(), Workload(seq_len=512))
    assert "estimate_env" not in p.provenance
    assert check_estimate_env(p, experts4bit, log=lambda *_: pytest.fail("no warning without a record")) is None


def test_execute_warns_when_the_running_process_differs(topo, monkeypatch):
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: {"E4B_CHUNKED_LM_LOSS": "auto"})
    p = plan(topo, hw(), Workload(seq_len=512))
    said = []
    assert check_estimate_env(p, experts4bit, log=said.append)["differs"] == [] and not said
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: {"E4B_CHUNKED_LM_LOSS": "0"})
    out = check_estimate_env(p, experts4bit, log=said.append)
    assert out["differs"] == ["E4B_CHUNKED_LM_LOSS"]
    assert out["planned"] == {"E4B_CHUNKED_LM_LOSS": "auto"} and out["running"] == {"E4B_CHUNKED_LM_LOSS": "0"}
    assert len(said) == 1 and "WARNING" in said[0] and "planned 'auto', running '0'" in said[0]


def test_an_unset_switch_matches_a_running_backend_that_reports_none(topo, monkeypatch):
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: {"E4B_CHUNKED_LM_LOSS": None})
    p = plan(topo, hw(), Workload(seq_len=512))
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: None)
    said = []
    assert check_estimate_env(p, experts4bit, log=said.append)["differs"] == [] and not said   # unset == unknown-absent
    monkeypatch.setattr(experts4bit, "estimate_env", lambda: {"E4B_CHUNKED_LM_LOSS": "1"})
    assert check_estimate_env(p, experts4bit, log=said.append)["differs"] == ["E4B_CHUNKED_LM_LOSS"] and said
