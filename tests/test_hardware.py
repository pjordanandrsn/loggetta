"""Hardware inventory parsing: driver output and cgroup files, from stated text (no device needed)."""
from loggetta.hardware import cgroup_cpus, cgroup_memory, describe, parse_nvidia_smi, probe

SMI = ("0, GPU-01aa, NVIDIA RTX A2000 12GB, 8.6, 12282, 5299, 575.64.05, 3, 16, 3, 8\n"
       "1, GPU-02bb, NVIDIA GeForce RTX 5090, 12.0, 32607, 32000, 580.10, 5, 16, [N/A], [N/A]\n")


def test_parse_nvidia_smi_two_gpus():
    a, b = parse_nvidia_smi(SMI)
    assert a.name == "NVIDIA RTX A2000 12GB" and a.compute_capability.value == (8, 6)
    assert a.memory_total.value == 12282 << 20 and a.memory_free.value == 5299 << 20
    assert a.memory_free.source == "reported" and a.pcie_width_current.value == 8
    assert b.compute_capability.value == (12, 0)
    assert b.pcie_gen_current.value is None and b.pcie_gen_current.source == "unknown"


def test_cgroup_v1_counts_page_cache_as_reclaimable(tmp_path):
    m = tmp_path / "memory"
    m.mkdir()
    (m / "memory.limit_in_bytes").write_text("30064771072\n")
    (m / "memory.usage_in_bytes").write_text("28682104832\n")
    (m / "memory.stat").write_text("cache 20000000000\nrss 8000000000\n")
    cg = cgroup_memory(str(tmp_path))
    assert cg == {"limit": 30064771072, "usage": 28682104832, "reclaimable": 20000000000, "version": "v1"}


def test_cgroup_v2_and_unlimited(tmp_path):
    (tmp_path / "memory.max").write_text("max\n")
    assert cgroup_memory(str(tmp_path)) == {}
    (tmp_path / "memory.max").write_text("1073741824\n")
    (tmp_path / "memory.current").write_text("536870912\n")
    (tmp_path / "memory.stat").write_text("anon 1\nfile 1000\n")
    assert cgroup_memory(str(tmp_path))["reclaimable"] == 1000


def test_cgroup_cpu_quota(tmp_path):
    c = tmp_path / "cpu"
    c.mkdir()
    (c / "cpu.cfs_quota_us").write_text("200000\n")
    (c / "cpu.cfs_period_us").write_text("100000\n")
    assert cgroup_cpus(str(tmp_path)) == 2.0
    (tmp_path / "cpu.max").write_text("max 100000\n")
    assert cgroup_cpus(str(tmp_path)) == 2.0                    # v2 unlimited falls through to v1


def test_probe_never_raises_and_labels_every_fact():
    hw = probe()
    text = describe(hw)
    assert "RAM" in text
    for g in hw.gpus:
        assert g.memory_total.source in ("reported", "unknown")


def test_user_overrides_are_labelled_user():
    hw = probe(ram_available=1 << 30)
    assert hw.host.memory_available.source == "user"


def test_a_saved_profile_round_trips_and_says_where_it_came_from():
    import json

    from loggetta.hardware import HardwareProfile

    hw = probe()
    hw = HardwareProfile(gpus=tuple(parse_nvidia_smi(SMI)), host=hw.host, platform=hw.platform)
    back = HardwareProfile.from_dict(json.loads(json.dumps(hw.to_dict())), origin="seat.json")
    assert back.gpus == hw.gpus and back.host == hw.host
    assert back.gpus[0].compute_capability.value == (8, 6)
    assert any("seat.json" in n for n in back.notes)
