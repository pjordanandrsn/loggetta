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


# ------------------------------------------------------------------------------------------- AMD: recorded, not planned

ROCM_SMI = ('{"card0": {"Card Series": "AMD Instinct MI300X", "GFX Version": "gfx942", '
            '"VRAM Total Memory (B)": "206141652992", "VRAM Total Used Memory (B)": "301989888", '
            '"Unique ID": "0x1a2b"}, "card1": {"Card Series": "Radeon RX 7900 XTX", "GFX Version": "gfx1100", '
            '"VRAM Total Memory (B)": "25753026560", "VRAM Total Used Memory (B)": "0"}, '
            '"system": {"Driver version": "6.10.5"}}')


def test_parse_rocm_smi_records_the_arch_and_never_a_capability():
    from loggetta.hardware import parse_rocm_smi

    a, b = parse_rocm_smi(ROCM_SMI)
    assert (a.vendor, a.name, a.arch, a.index) == ("amd", "AMD Instinct MI300X", "gfx942", 0)
    assert a.memory_total.value == 206141652992 and a.memory_free.value == 206141652992 - 301989888
    assert a.compute_capability.value is None and "gfx number" in a.compute_capability.note
    assert a.driver.value == "6.10.5" and a.uuid == "0x1a2b"
    assert (b.arch, b.index, b.memory_free.value) == ("gfx1100", 1, 25753026560)
    assert parse_rocm_smi("not json") == []


def test_parse_amdsmi_tolerates_missing_answers():
    from loggetta.hardware import parse_amdsmi

    a, b = parse_amdsmi([{"asic": {"market_name": "AMD Instinct MI300X", "target_graphics_version": "gfx942"},
                          "vram": {"vram_total": 196592, "vram_used": 288}, "uuid": "u-0",
                          "driver": {"driver_version": "6.10.5"}},
                         {"asic": None, "vram": None, "uuid": None, "driver": None}])
    assert (a.vendor, a.arch, a.memory_total.value, a.memory_free.value) == ("amd", "gfx942", 196592 << 20,
                                                                             (196592 - 288) << 20)
    assert a.compute_capability.value is None
    assert (b.vendor, b.name, b.memory_total.source) == ("amd", "AMD GPU", "unknown")


def test_probe_without_nvidia_smi_records_an_amd_gpu(monkeypatch):
    """Before: no nvidia-smi meant no GPU, so every plan said 'no GPU' on an AMD box."""
    import loggetta.hardware as h

    real_which = h.shutil.which
    monkeypatch.setattr(h.shutil, "which", lambda name: "/opt/rocm/bin/rocm-smi" if name == "rocm-smi" else
                        None if name == "nvidia-smi" else real_which(name))
    monkeypatch.setitem(__import__("sys").modules, "amdsmi", None)          # amdsmi not importable: the CLI answers
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: __import__("types").SimpleNamespace(stdout=ROCM_SMI))
    hw = probe()
    assert [g.vendor for g in hw.gpus] == ["amd", "amd"] and hw.gpus[0].arch == "gfx942"
    assert any("rocm-smi" in n for n in hw.notes)
    assert "AMD gfx942: not yet supported" in describe(hw)


def test_an_amd_profile_round_trips_with_its_arch():
    import json

    from loggetta.hardware import HardwareProfile, parse_rocm_smi

    p = probe()
    amd = HardwareProfile(gpus=tuple(parse_rocm_smi(ROCM_SMI)), host=p.host, platform=p.platform)
    back = HardwareProfile.from_dict(json.loads(json.dumps(amd.to_dict())))
    assert back.gpus[0].arch == "gfx942" and back.gpus[0].vendor == "amd"
