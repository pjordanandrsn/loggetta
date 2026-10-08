"""D7 reader falsification; all receipt bytes in this file are test fixtures."""
import copy

import pytest

from loggetta.dense_calibration import read


def fixture():
    rows = []
    for subject in ("qwen3_14b", "llama31_8b", "qwen3_32b"):
        for placement in ("device", "stream"):
            for seq in ((2048, 4096) if subject == "qwen3_32b" else (512, 2048, 4096)):
                a = 1100 if seq == 4096 else 1000
                rows.append({"backend": "dense", "status": "OK", "run_id": f"fixture-{subject}-{placement}-{seq}",
                    "dq7": {"subject": subject, "placement": placement, "seq": seq},
                    "measured": {"device_peak_bytes": a, "device_reserved_peak_bytes": a+a//10},
                    "comparison": {"device_allocator": {"estimated": a, "measured": a}}})
    return rows, {"verdict": "NEVER_UNDER", "pass_licensed": True}


def test_six_holdouts_with_exact_fractions_and_original_20_percent_reporting():
    result = read(*fixture())
    assert result["verdict"] == "CALIBRATION_PASS" and len(result["cohorts"]) == 6
    for cohort in result["cohorts"]:
        assert cohort["fraction"] == {"numerator": 1, "denominator": 10}
        assert all(row["seq"] < 4096 for row in cohort["derivation"])
        h = cohort["holdout"]
        assert h["derived_reserve_bytes"] == 110 and h["derived_total_bytes"] == 1210
        assert h["inferred_20_percent_reserve_bytes"] == 220 and h["inferred_20_percent_total_bytes"] == 1320
        assert not cohort["reserve_import_licensed"]
    assert not result["reserve_import_licensed"]


def test_one_byte_holdout_failure_does_not_raise_the_derived_fraction():
    rows, verdict = fixture()
    held = next(row for row in rows if row["dq7"]["seq"] == 4096)
    held["measured"]["device_reserved_peak_bytes"] += 1
    result = read(rows, verdict)
    assert result["verdict"] == "CALIBRATION_FAIL"
    first = result["cohorts"][0]
    assert first["fraction"] == {"numerator": 1, "denominator": 10}
    assert first["holdout"]["cached_block_residual_bytes"] == 1


def test_allocator_overestimate_cannot_hide_a_cached_block_underestimate():
    rows, verdict = fixture()
    held = next(row for row in rows if row["dq7"]["seq"] == 4096)
    held["comparison"]["device_allocator"]["estimated"] += 100
    held["measured"]["device_reserved_peak_bytes"] = 1250
    result = read(rows, verdict)
    first = result["cohorts"][0]
    assert first["holdout"]["total_residual_bytes"] < 0
    assert first["holdout"]["cached_block_residual_bytes"] > 0
    assert result["verdict"] == "CALIBRATION_FAIL"


def test_one_model_or_placement_never_changes_another_cohort():
    rows, verdict = fixture()
    baseline = read(rows, verdict)
    changed = copy.deepcopy(rows)
    changed[0]["measured"]["device_reserved_peak_bytes"] += 50
    result = read(changed, verdict)
    assert result["cohorts"][1:] == baseline["cohorts"][1:]


@pytest.mark.parametrize("verdict", [{"verdict": "ANCHOR_MISS", "pass_licensed": False},
                                    {"verdict": "ESTIMATE_UNDER", "pass_licensed": False},
                                    {"verdict": "NEVER_UNDER"}])
def test_unlicensed_dq7_never_fits_anything(verdict):
    assert read(fixture()[0], verdict)["verdict"] == "DQ7_UNLICENSED"


@pytest.mark.parametrize("field,value", [("device_peak_bytes", 0), ("device_peak_bytes", True),
                                       ("device_reserved_peak_bytes", 999), ("device_reserved_peak_bytes", None),
                                       ("device_peak_bytes", 1001)])
def test_invalid_or_inconsistent_bytes_are_void(field, value):
    rows, verdict = fixture()
    rows[0]["measured"][field] = value
    assert read(rows, verdict)["verdict"] == "VOID"


def test_missing_duplicate_and_changed_partition_are_void():
    rows, verdict = fixture()
    assert read(rows[:-1], verdict)["verdict"] == "VOID"
    assert read(rows+rows[:1], verdict)["verdict"] == "VOID"
    rows[0]["dq7"]["seq"] = 1024
    assert read(rows, verdict)["verdict"] == "VOID"


def test_fraction_charge_rounds_up_at_a_real_byte_boundary():
    rows, verdict = fixture()
    held = next(row for row in rows if row["dq7"]["seq"] == 4096)
    held["comparison"]["device_allocator"]["estimated"] = 1101
    h = read(rows, verdict)["cohorts"][0]["holdout"]
    assert h["derived_reserve_bytes"] == 111 and h["derived_total_bytes"] == 1212
