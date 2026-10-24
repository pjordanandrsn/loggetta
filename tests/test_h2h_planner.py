"""bench/h2h_planner.py, the planner head-to-head's driver: Accelerate's table is read as printed, a receipt that is not
the split's is refused, and every term is graded measured / estimated."""
import hashlib
import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPLIT = json.loads((ROOT / "evidence" / "planner-split" / "SPLIT-v3.json").read_text())
#: estimate-memory's table as Accelerate 1.15.0 printed it for ERNIE-4.5-21B-A3B-PT (bench/h2h-planner/PREREG's smoke)
STDOUT = """Loading pretrained config for `baidu/ERNIE-4.5-21B-A3B-PT` from `transformers`...
┌──────────────────────────────────────────────────────┐
│Memory Usage for loading `baidu/ERNIE-4.5-21B-A3B-PT` │
├───────┬─────────────┬──────────┬─────────────────────┤
│ dtype │Largest Layer│Total Size│ Training using Adam │
├───────┼─────────────┼──────────┼─────────────────────┤
│float16│   1.48 GB   │ 40.65 GB │      162.61 GB      │
│  int4 │  378.83 MB  │ 10.16 GB │         N/A         │
└───────┴─────────────┴──────────┴─────────────────────┘
"""


def _tool():
    spec = importlib.util.spec_from_file_location("h2h_planner", ROOT / "bench" / "h2h_planner.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_accelerates_table_is_read_as_printed():
    t = _tool()
    s = t.parse_accelerate(STDOUT)
    assert set(s) == {"float16", "int4"}
    assert s["float16"]["printed"] == {"largest_layer": "1.48 GB", "total": "40.65 GB", "training_adam": "162.61 GB"}
    assert s["float16"]["total_bytes"] == round(40.65 * (1 << 30)) and s["int4"]["largest_layer_bytes"] == round(378.83 * (1 << 20))
    assert s["int4"]["training_adam_bytes"] is None                   # "N/A" stays absent, never zero


def test_a_table_that_is_not_there_is_refused():
    t = _tool()
    with pytest.raises(t.Refused, match="no size table"):
        t.parse_accelerate("Loading pretrained config...\nsomething else\n")


def test_a_receipt_that_is_not_the_splits_is_refused():
    t = _tool()
    row = SPLIT["held_out"][-1]
    with pytest.raises(t.Refused, match="not the split's sha256"):
        t.read_row(row, SPLIT["pinned_at"], lambda commit, path: b'{"not": "it"}')


def test_a_receipt_that_is_the_splits_is_read_at_the_pin():
    t = _tool()
    raw = b'{"comparison": {}}'
    row = {**SPLIT["held_out"][-1], "sha256": hashlib.sha256(raw).hexdigest()}
    seen = []
    assert t.read_row(row, SPLIT["pinned_at"], lambda commit, path: (seen.append((commit, path)), raw)[1]) == {"comparison": {}}
    assert seen == [(SPLIT["pinned_at"][row["repo"]], row["file"])]


def test_every_term_is_graded_measured_over_estimated():
    t = _tool()
    g = t.grade({"device_driver": {"estimated": 100, "measured": 80}, "host": {"estimated": 10, "measured": 11},
                 "device_allocator": {"estimated": 0, "measured": 5}, "note": "not a term"})
    assert g == {"device_driver": 0.8, "host": 1.1, "device_allocator": None}


def test_the_split_has_the_registered_21_rows():
    assert len(SPLIT["held_out"]) == 21 and SPLIT["pinned_at"]["experts4bit-qlora"].startswith("a4b2b98f")


def test_a_release_that_is_not_installed_reads_not_read_today():
    t = _tool()
    have = {"loggetta": "0.7.0", "experts4bit-qlora": "0.57.0", "grouped-nf4-gemm": None, "accelerate": "1.15.0"}.get
    with pytest.raises(t.Refused, match="not read today: experts4bit-qlora wants 0.58.0, installed 0.57.0; grouped-nf4-gemm"):
        t.check_versions(None, have, lambda d: "index")
    got = t.check_versions({"loggetta": "0.7.0"}, have, lambda d: "index")
    assert got == {"loggetta": {"version": "0.7.0", "installed_from": "index"}}


def test_the_registered_releases_are_always_checked():
    t = _tool()
    assert t.REGISTERED == {"loggetta": "0.7.0", "experts4bit-qlora": "0.58.0", "grouped-nf4-gemm": "0.51.0",
                            "accelerate": "1.15.0"}
    with pytest.raises(t.Refused, match="not read today"):
        t.check_versions(version_of=lambda d: None, source_of=lambda d: None)       # nothing installed: no default pass


@pytest.mark.parametrize("source", ["editable", "local file"])
def test_a_checkout_reporting_the_release_version_is_not_the_release(source):
    t = _tool()
    with pytest.raises(t.Refused, match=f"experts4bit-qlora 0.58.0 is installed from a {source} install, not a release"):
        t.check_versions({"experts4bit-qlora": "0.58.0"}, lambda d: "0.58.0", lambda d: source)


#: two distinct commit ids, assembled at run time
TAGGED, OTHER = "a1" * 20, "b2" * 20


def _vcs(commit, url="https://example.invalid/r.git"):
    return lambda d: {"url": url, "vcs_info": {"vcs": "git", "commit_id": commit}}


def test_a_git_install_at_the_tags_commit_is_the_release():
    t = _tool()
    seen = []

    def tag_of(url, v):
        seen.append((url, v))
        return TAGGED

    def at_tag(d, v):
        return t.vcs_at_tag(d, v, _vcs(TAGGED), tag_of)
    got = t.check_versions({"grouped-nf4-gemm": "0.51.0"}, lambda d: "0.51.0", lambda d: "vcs", at_tag)
    assert got == {"grouped-nf4-gemm": {"version": "0.51.0", "installed_from": "vcs", "commit": TAGGED}}
    assert seen == [("https://example.invalid/r.git", "0.51.0")]


@pytest.mark.parametrize("direct, tag, why", [
    (_vcs(OTHER), TAGGED, f"is at {OTHER[:12]}, not tag v0.51.0's {TAGGED[:12]}"),
    (_vcs(TAGGED), None, "has no tag v0.51.0"),
    (lambda d: {"url": "https://example.invalid/r.git", "vcs_info": {"vcs": "git"}}, TAGGED, "records no commit"),
])
def test_a_git_install_at_any_other_commit_is_not_the_release(direct, tag, why):
    t = _tool()

    def at_tag(d, v):
        return t.vcs_at_tag(d, v, direct, lambda url, v: tag)
    with pytest.raises(t.Refused, match=re.escape(f"grouped-nf4-gemm 0.51.0 is a git install that {why}, not a release")):
        t.check_versions({"grouped-nf4-gemm": "0.51.0"}, lambda d: "0.51.0", lambda d: "vcs", at_tag)


def test_an_index_wheel_never_asks_for_a_tag():
    t = _tool()

    def at_tag(d, v):
        raise AssertionError("an index wheel is the release as installed")
    assert t.check_versions({"loggetta": "0.7.0"}, lambda d: "0.7.0", lambda d: "index", at_tag)["loggetta"] == {
        "version": "0.7.0", "installed_from": "index"}


def test_the_tag_is_read_peeled():
    t = _tool()

    class R:
        stdout = f"{OTHER}\trefs/tags/v0.51.0\n{TAGGED}\trefs/tags/v0.51.0^{{}}\n"
    import subprocess
    orig = subprocess.run
    try:
        subprocess.run = lambda *a, **k: R()
        assert t.tag_commit("u", "0.51.0") == TAGGED                                  # the annotated tag's commit
        R.stdout = f"{TAGGED}\trefs/tags/v0.51.0\n"
        assert t.tag_commit("u", "0.51.0") == TAGGED                                  # a lightweight tag
        R.stdout = ""
        assert t.tag_commit("u", "0.51.0") is None
    finally:
        subprocess.run = orig


def test_accelerate_runs_from_the_checked_interpreter():
    import sys
    t = _tool()
    calls = []

    class R:
        returncode, stdout, stderr = 0, STDOUT, ""
    got = t.run_accelerate("baidu/ERNIE-4.5-21B-A3B-PT", "1.15.0", run=lambda argv, **k: (calls.append(argv), R())[1])
    argv = calls[0]
    assert argv[0] == sys.executable and argv[1:3] == ["-m", "accelerate.commands.accelerate_cli"]
    assert argv[3:] == ["estimate-memory", "baidu/ERNIE-4.5-21B-A3B-PT", "--library_name", "transformers",
                        "--dtypes", "float16", "int4"]
    assert got["argv"] == argv and got["version"] == "1.15.0"
    assert got["sizes"]["float16"]["printed"]["total"] == "40.65 GB"


def test_an_editable_install_reads_as_editable(tmp_path, monkeypatch):
    from importlib import metadata

    t = _tool()

    class Dist:
        def read_text(self, name):
            return json.dumps({"dir_info": {"editable": True}, "url": "file:///x"}) if name == "direct_url.json" else None
    monkeypatch.setattr(metadata, "distribution", lambda d: Dist())
    assert t.installed_from("experts4bit-qlora") == "editable"

    class Wheel:
        def read_text(self, name):
            return None
    monkeypatch.setattr(metadata, "distribution", lambda d: Wheel())
    assert t.installed_from("experts4bit-qlora") == "index"


def _cmp(drv=(100, 90), alloc=(90, 80), host=None):
    c = {"device_driver": {"estimated": drv[0], "measured": drv[1]}, "device_allocator": {"estimated": alloc[0], "measured": alloc[1]}}
    if host:
        c["host"] = {"estimated": host[0], "measured": host[1]}
    return c


@pytest.mark.parametrize("cmp, host_row, want", [
    (_cmp(), False, "COVERED"),
    (_cmp(drv=(100, 101)), False, "UNDER"),
    (_cmp(alloc=(90, 91)), False, "SHORT"),
    (_cmp(drv=(100, 101), alloc=(90, 91)), False, "UNDER+SHORT"),
    (_cmp(host=(10, 11)), True, "HOST_UNDER"),
    (_cmp(host=(10, 11)), False, "COVERED"),          # a host term on a device row is never ruled (HO3's rule)
    (_cmp(host=(10, 10)), True, "COVERED"),            # at the estimate exactly is covered
    (_cmp(), True, "VOID: no host"),                   # a host row without its host term voids, never reads covered
    ({"device_driver": {"estimated": 100, "measured": 90}}, False, "VOID: no device_allocator"),
])
def test_each_rows_outcome_is_the_planners_own_rule(cmp, host_row, want):
    assert _tool().outcome(cmp, host_row) == want
