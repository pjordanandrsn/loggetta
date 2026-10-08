# D7 — dense allocator reserve from held-out receipts

This registers a later calibration read. It changes no estimate, default, receipt licence or execution gate.
DQ7 in experts4bit-qlora PR1359 measures the unchanged planner through Loggetta merge
`1dd862f73433b5d9fdd42d835dac7504726574ae`. D7 reads those receipts only after the registration is reviewed.
Synthetic architecture-only checkpoints provide memory evidence, with no pretrained loss or quality claim.

## Partition fixed before capacity data

For each of Qwen3-14B, Llama-3.1-8B and Qwen3-32B, independently for resident and streamed placement:

- Derivation: all available DQ7 sequence rungs below 4096 (512 and 2048 for the two out-of-sample subjects;
  2048 only for the Qwen3-32B anchor).
- Holdout: the 4096-token rung. Six holdouts total. Holdouts never contribute to the fitted reserve.

Qwen3-32B remains an in-sample anchor for the activation estimate; its reserve holdout is a separate question.
No subject or placement borrows another's reserve. No 5090 reserve transfers to the 4090 reading.

## Preconditions and derivation

The complete DQ7 proof and 16-arm reading must pass its registered reducer, including the strict anchor
reproduction/attribution rule. Missing arms, wrong pins, setup changes, real-token violations or an unlicensed
DQ7 verdict stop D7. Return to review with the cause; do not repair a failed estimate by fitting its failure.

Read integer training-phase `device_peak_bytes` A and `device_reserved_peak_bytes` R from each original receipt.
Both must be positive with R >= A. Preserve the originals and their checksums. For each model/placement cohort,
derive f = max((R - A) / A) over its derivation receipts. Keep exact integer numerator/denominator pairs;
ceil any final byte charge. Do not derive a context, activation, residual or host-memory correction.

Hypothesis: this own-receipt fraction replaces the inferred 20% reserve conservatively at the held-out rung.
A fraction >= 20% is a valid finding but provides no licence to reduce the existing reserve.

## Held-out gates and permitted scope

Rebuild the held-out plan with only the cohort's derivation receipts. Its allocator estimate E is unchanged.
The DQ4 allocator brackets must still pass without calibration. Apply the candidate reserve with an explicit
source label; the held-out gate is ceil(E * (1 + f)) >= R_holdout, in bytes. One byte below fails.
Also require ceil(E * f) >= R_holdout - A_holdout, so a high allocator estimate cannot hide an under-reserved
cached-block term. These two gates are both required for every cohort; publish every held-out residual.

A passing cohort licenses reserve only, scoped to its dense topology, GPU/card capacity, driver/runtime pins,
exact base/placement/rank/alpha/adapter dtype/targets/attention/loss setup, default allocator, micro-batch1,
grad-accum1 and the registered sequence range through 4096. A different topology, card, setup or larger sequence
keeps the inferred 20% reserve. The implementation must explicitly enforce that scope, including fallback paths.
No unscoped observation import, model borrowing, card transfer, fitted activation coefficient or context correction
is licensed. The cohort's derivation sources and held-out receipts accompany any register row or quotation.

If a cohort fails, keep its 20% reserve and report CALIBRATION_FAIL. No additional margin fitted on the holdout,
partition change, redraw or narrowed success quote is permitted by this registration. A new proposal returns to review.

## Implementation and release boundary

The later implementation PR includes mutation tests for one-byte holdout failure, leaked holdouts, wrong model/card/
sequence/runtime/setup, missing or invalid peak fields and every observation fallback. It preserves the DQ4 brackets
and native-backend calibration behavior. Raw receipts and a reading are separate from the implementation PR.

Removing the dense executor development opt-in additionally requires the independently registered 24 GB RTX4090
proof/capacity reading requested by the maintainer. That gate-removal PR updates docs/DENSE.md, TRAINING.md and README,
and adds evidence-linked estimate register rows. The maintainer cuts the release after independent review.
This registration acquires no compute and authorizes no further draw.
