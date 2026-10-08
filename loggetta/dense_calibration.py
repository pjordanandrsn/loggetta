"""D7: a registered reserve read with held-out rows; no planner coefficients are fitted here."""
from __future__ import annotations

from fractions import Fraction

SUBJECTS = ("qwen3_14b", "llama31_8b", "qwen3_32b")


def ceil_fraction(value: Fraction) -> int:
    return -(-value.numerator // value.denominator)


def read(receipts, dq7_result):
    """Grade D7's fixed six holdouts against derivation-only cached-block fractions.

    ``dq7_result`` must be recomputed by the pinned DQ7 reducer before calling this function.
    This produces a report, never a licensed observation. The later scoped import requires separate review.
    """
    if dq7_result.get("verdict") != "NEVER_UNDER" or dq7_result.get("pass_licensed") is not True:
        return {"schema": "dense-d7/1", "verdict": "DQ7_UNLICENSED", "cohorts": [], "reserve_import_licensed": False}
    expected = {(subject, placement, seq) for subject in SUBJECTS for placement in ("device", "stream")
                for seq in ((2048, 4096) if subject == "qwen3_32b" else (512, 2048, 4096))}
    rows = {}
    for receipt in receipts:
        tag = receipt.get("dq7", {})
        key = tag.get("subject"), tag.get("placement"), tag.get("seq")
        if key not in expected or key in rows or receipt.get("status") != "OK" or receipt.get("backend") != "dense":
            return {"schema": "dense-d7/1", "verdict": "VOID", "cause": "unregistered, duplicate or failed arm",
                    "reserve_import_licensed": False}
        m = receipt.get("measured", {})
        a, r = m.get("device_peak_bytes"), m.get("device_reserved_peak_bytes")
        e = receipt.get("comparison", {}).get("device_allocator", {}).get("estimated")
        if (any(not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in (a, r, e)) or r < a
                or a != receipt.get("comparison", {}).get("device_allocator", {}).get("measured")
                or not receipt.get("run_id")):
            return {"schema": "dense-d7/1", "verdict": "VOID", "cause": "missing or invalid memory bytes",
                    "reserve_import_licensed": False}
        rows[key] = receipt
    if set(rows) != expected:
        return {"schema": "dense-d7/1", "verdict": "VOID", "cause": "incomplete registered arms",
                "reserve_import_licensed": False}
    cohorts = []
    for subject in SUBJECTS:
        for placement in ("device", "stream"):
            derivation = [row for (name, where, seq), row in rows.items()
                          if name == subject and where == placement and seq < 4096]
            def slack(row):
                m = row["measured"]
                return Fraction(m["device_reserved_peak_bytes"] - m["device_peak_bytes"], m["device_peak_bytes"])
            fraction = max(map(slack, derivation))
            holdout = rows[subject, placement, 4096]
            m = holdout["measured"]
            a, r = m["device_peak_bytes"], m["device_reserved_peak_bytes"]
            e = holdout["comparison"]["device_allocator"]["estimated"]
            charge = ceil_fraction(e*fraction)
            total = ceil_fraction(e*(1+fraction))
            cached = r-a
            passed = total >= r and charge >= cached
            cohorts.append({"subject": subject, "placement": placement, "verdict": "CALIBRATION_PASS" if passed else "CALIBRATION_FAIL",
                "fraction": {"numerator": fraction.numerator, "denominator": fraction.denominator},
                "derivation": [{"run_id": row["run_id"], "seq": row["dq7"]["seq"],
                                "allocated_bytes": row["measured"]["device_peak_bytes"],
                                "reserved_bytes": row["measured"]["device_reserved_peak_bytes"]} for row in derivation],
                "holdout": {"run_id": holdout["run_id"], "seq": 4096, "allocated_bytes": a, "reserved_bytes": r,
                            "allocator_estimate_bytes": e, "derived_reserve_bytes": charge, "derived_total_bytes": total,
                            "inferred_20_percent_reserve_bytes": ceil_fraction(Fraction(e, 5)),
                            "inferred_20_percent_total_bytes": ceil_fraction(Fraction(6*e, 5)),
                            "total_residual_bytes": r-total, "cached_block_residual_bytes": cached-charge},
                "reserve_import_licensed": False})
    return {"schema": "dense-d7/1", "verdict": "CALIBRATION_PASS" if all(row["verdict"] == "CALIBRATION_PASS" for row in cohorts)
            else "CALIBRATION_FAIL", "cohorts": cohorts, "reserve_import_licensed": False,
            "note": "A report only. A passing cohort requires a separately reviewed scoped observation import."}
