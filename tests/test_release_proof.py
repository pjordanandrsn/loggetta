"""bench/release_proof.py: an install reads as the release only when its bytes are the published wheel's. Wheels are
built here, "installed" by unpacking them the way pip lays them out, and served through stand-in fetchers; one network
test reads the real Accelerate 1.15.0 wheel against this environment's install."""
import base64
import hashlib
import importlib.util
import json
import os
import pathlib
import zipfile
from importlib import metadata

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAME, VER = "fakerel", "0.58.0"
DI = f"{NAME}-{VER}.dist-info"
SRC = {"fakerel/__init__.py": b"RELEASE = True\n", "fakerel/cli.py": b"def main():\n    return 0\n"}


def _tool():
    spec = importlib.util.spec_from_file_location("release_proof", ROOT / "bench" / "release_proof.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h(data):
    return "sha256=" + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()


def _wheel(path, files=None, tag="py3-none-any", extra=None):
    """A wheel at ``path`` holding ``files`` (default SRC) plus its dist-info, with its own RECORD."""
    files = dict(SRC if files is None else files)
    files[f"{DI}/METADATA"] = f"Metadata-Version: 2.1\nName: {NAME}\nVersion: {VER}\n".encode()
    files[f"{DI}/WHEEL"] = f"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: {tag}\n".encode()
    files[f"{DI}/entry_points.txt"] = b"[console_scripts]\nfakerel = fakerel.cli:main\n"
    files.update(extra or {})
    record = "".join(f"{p},{_h(d)},{len(d)}\n" for p, d in files.items()) + f"{DI}/RECORD,,\n"
    with zipfile.ZipFile(path, "w") as z:
        for p, d in files.items():
            z.writestr(p, d)
        z.writestr(f"{DI}/RECORD", record)
    return path


def _install(whl, site, extra_record=()):
    """Unpack ``whl`` into ``site`` as pip does: the wheel's files, INSTALLER and REQUESTED, a __pycache__ .pyc and a
    launcher outside site-packages, all listed in the installed RECORD."""
    site.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(whl) as z:
        z.extractall(site)
    rows = [r for r in (site / DI / "RECORD").read_text().splitlines() if not r.startswith(f"{DI}/RECORD,")]
    for name, data in ((f"{DI}/INSTALLER", b"pip\n"), (f"{DI}/REQUESTED", b""), ("fakerel/__pycache__/cli.cpython-312.pyc", b"\0pyc")):
        (site / name).parent.mkdir(parents=True, exist_ok=True)
        (site / name).write_bytes(data)
        rows.append(f"{name},{_h(data)},{len(data)}" if not name.endswith(".pyc") else f"{name},,")
    rows.append(f"../../../bin/fakerel,{_h(b'#!py')},4")
    rows += list(extra_record)
    rows.append(f"{DI}/RECORD,,")
    (site / DI / "RECORD").write_text("\n".join(rows) + "\n")
    return metadata.PathDistribution(site / DI)


def _pypi(*wheels):
    """Stand-in fetchers serving PyPI's JSON for (path, sha256-as-listed) wheels and their bytes."""
    blobs = {f"https://files.example/{p.name}": p.read_bytes() for p, _ in wheels}
    doc = {"urls": [{"filename": p.name, "packagetype": "bdist_wheel", "url": f"https://files.example/{p.name}",
                     "digests": {"sha256": listed or hashlib.sha256(p.read_bytes()).hexdigest()}} for p, listed in wheels]
           + [{"filename": f"{NAME}-{VER}.tar.gz", "packagetype": "sdist", "url": "x", "digests": {"sha256": "0"}}]}

    def get(url):
        return json.dumps(doc).encode() if url == f"https://pypi.org/pypi/{NAME}/{VER}/json" else None

    def fetch_to(url, dest):
        pathlib.Path(dest).write_bytes(blobs[url])
        return hashlib.sha256(blobs[url]).hexdigest()
    return get, fetch_to


def _prove(t, dist, get, fetch_to):
    return t.prove(NAME, VER, get, fetch_to, lambda d: dist)


def test_the_published_wheel_as_installed_is_the_release(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site")
    got = _prove(t, dist, *_pypi((whl, None)))
    assert got == {"dist": NAME, "version": VER, "wheel": whl.name, "sha256": hashlib.sha256(whl.read_bytes()).hexdigest(),
                   "files": 5, "record": "RECORD equal", "record_equal": True, "first_diff": None}


def test_a_locally_built_wheel_at_the_release_version_is_refused(tmp_path):
    t = _tool()
    published = _wheel(tmp_path / "pub" / f"{NAME}-{VER}-py3-none-any.whl" if (tmp_path / "pub").mkdir() is None else None)
    local = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl", {**SRC, "fakerel/__init__.py": b"RELEASE = False\n"})
    dist = _install(local, tmp_path / "site")                   # same name, same version, other bytes
    with pytest.raises(t.ReleaseProofError, match="fakerel/__init__.py differs from the published wheel's") as e:
        _prove(t, dist, *_pypi((published, None)))
    assert e.value.first_diff == "fakerel/__init__.py"


def test_a_file_changed_on_disk_after_install_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site")
    (tmp_path / "site" / "fakerel" / "cli.py").write_bytes(b"def main():\n    return 1\n")   # the installed RECORD still agrees
    with pytest.raises(t.ReleaseProofError, match="fakerel/cli.py differs"):
        _prove(t, dist, *_pypi((whl, None)))


def test_a_published_file_missing_from_the_install_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site")
    (tmp_path / "site" / "fakerel" / "cli.py").unlink()
    with pytest.raises(t.ReleaseProofError, match="fakerel/cli.py is in the published wheel but not installed") as e:
        _prove(t, dist, *_pypi((whl, None)))
    assert e.value.first_diff == "fakerel/cli.py"


def test_an_added_module_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    (tmp_path / "site" / "fakerel").mkdir(parents=True)
    (tmp_path / "site" / "fakerel" / "extra.py").write_bytes(b"import os\n")
    dist = _install(whl, tmp_path / "site", [f"fakerel/extra.py,{_h(b'import os')},10", "evil.pth,,"])
    with pytest.raises(t.ReleaseProofError, match="evil.pth is installed but not in the published wheel") as e:
        _prove(t, dist, *_pypi((whl, None)))
    assert e.value.first_diff == "evil.pth"          # sorted first; an unhashed entry, absent from disk, is not let through


def test_bytecode_is_allowed_only_under_pycache(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    assert _prove(t, _install(whl, tmp_path / "a"), *_pypi((whl, None)))["record_equal"]        # __pycache__/*.pyc: fine
    dist = _install(whl, tmp_path / "b", ["fakerel/cli.pyc,,"])
    with pytest.raises(t.ReleaseProofError, match="fakerel/cli.pyc is installed but not in the published wheel"):
        _prove(t, dist, *_pypi((whl, None)))


def test_a_launcher_is_allowed_only_when_entry_points_names_it(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site", [f"../../../bin/evil,{_h(b'#!sh')},4"])
    with pytest.raises(t.ReleaseProofError, match=r"\.\./\.\./\.\./bin/evil is installed but not in the published wheel"):
        _prove(t, dist, *_pypi((whl, None)))
    assert t.generated_by_pip("../../../bin/fakerel", {"fakerel"}) and not t.generated_by_pip("../../../bin/fakerel", set())


def test_a_download_that_is_not_the_listed_sha256_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site")
    with pytest.raises(t.ReleaseProofError, match="as downloaded is not the sha256 PyPI lists"):
        _prove(t, dist, *_pypi((whl, "0" * 64)))


def test_the_wheel_is_picked_by_the_installed_tag(tmp_path):
    t = _tool()
    tag = "cp312-cp312-manylinux_2_24_x86_64"
    mine = _wheel(tmp_path / f"{NAME}-{VER}-{tag}.whl", tag=tag)
    (tmp_path / "o").mkdir()
    other = _wheel(tmp_path / "o" / f"{NAME}-{VER}-cp312-cp312-macosx_14_0_arm64.whl", {**SRC, "fakerel/cli.py": b"mac\n"},
                   tag="cp312-cp312-macosx_14_0_arm64")
    dist = _install(mine, tmp_path / "site")
    assert _prove(t, dist, *_pypi((other, None), (mine, None)))["wheel"] == mine.name
    with pytest.raises(t.ReleaseProofError, match="0 published wheels carry the installed tags"):
        _prove(t, dist, *_pypi((other, None)))


def test_compressed_tag_sets_expand():
    assert _tool().wheel_tags("x-1-py2.py3-none-any.whl") == {"py2-none-any", "py3-none-any"}


def test_a_version_not_on_pypi_or_not_installed_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl")
    dist = _install(whl, tmp_path / "site")
    with pytest.raises(t.ReleaseProofError, match="is not on PyPI"):
        t.prove(NAME, VER, lambda url: None, None, lambda d: dist)
    with pytest.raises(t.ReleaseProofError, match="installed at 0.58.0, not 0.58.1"):
        t.prove(NAME, "0.58.1", None, None, lambda d: dist)


def test_a_wheel_with_data_entries_is_refused(tmp_path):
    t = _tool()
    whl = _wheel(tmp_path / f"{NAME}-{VER}-py3-none-any.whl", extra={f"{NAME}-{VER}.data/scripts/x": b"#!sh\n"})
    dist = _install(whl, tmp_path / "site")
    with pytest.raises(t.ReleaseProofError, match=r"has \.data entries"):
        _prove(t, dist, *_pypi((whl, None)))


def test_download_streams_to_disk_and_hashes(tmp_path):
    t = _tool()
    src = tmp_path / "blob"
    src.write_bytes(os.urandom(3 * t.CHUNK + 7))
    dest = tmp_path / "dest"
    assert t.download(src.as_uri(), dest) == hashlib.sha256(src.read_bytes()).hexdigest()
    assert dest.read_bytes() == src.read_bytes()


def test_the_cli_exits_nonzero_with_the_reason(capsys):
    t = _tool()
    assert t.main([]) == 2
    assert t.main(["no-such-dist-for-release-proof==1.0"]) == 1
    assert "refused: no-such-dist-for-release-proof is not installed" in capsys.readouterr().err


def test_the_real_accelerate_1_15_0_wheel_matches_its_install():
    """One network test: skipped where Accelerate 1.15.0 is not installed from an index or PyPI is unreachable, unless
    RELEASE_PROOF_NETWORK=1 (on a box), where neither may skip."""
    t = _tool()
    must = os.environ.get("RELEASE_PROOF_NETWORK") == "1"
    try:
        d = metadata.distribution("accelerate")
        fit = d.version == "1.15.0" and not d.read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        fit = False
    if not fit and not must:
        pytest.skip("Accelerate 1.15.0 is not installed from an index here")
    try:
        got = t.prove("accelerate", "1.15.0")
    except OSError as e:
        if must:
            raise
        pytest.skip(f"PyPI unreachable: {e}")
    assert got["wheel"] == "accelerate-1.15.0-py3-none-any.whl" and got["record_equal"] and got["files"] > 50


def test_the_prereg_names_this_files_sha256():
    import re
    text = (ROOT / "bench" / "h2h-planner" / "PREREG-h2h-planner.md").read_text()
    named = re.search(r"`bench/release_proof\.py` \(sha256 `([0-9a-f]{64})`", text).group(1)
    assert named == hashlib.sha256((ROOT / "bench" / "release_proof.py").read_bytes()).hexdigest()
