# The frozen-expert digest under experts4bit-qlora's double-quantized absmax (#47)

Five arms of `loggetta train allenai/OLMoE-1B-7B-0924 --seq 512 --micro-batch 2 --steps 2` on an RTX A2000 12GB. Four
use the released pair from PyPI, loggetta 0.3.1 + experts4bit-qlora 0.50.0, in a clean venv (`freeze.txt`). The fifth
uses that venv with this branch's `loggetta` on `PYTHONPATH`. Each arm's `run.log` is the command's whole output; the
receipts are as written. Correctness only: two steps per arm, about 10 minutes of GPU in all.

| arm | experts | kernel | `E4B_ABSMAX_DQ` | loggetta | result |
|---|---|---|---|---|---|
| A_default | device | grouped_nf4 | unset (compressed) | 0.3.1 | **`AbsmaxCompressedError` at `_expert_digest`, before step 1** |
| B_dq0 | device | grouped_nf4 | `0` | 0.3.1 | OK; frozen bytes unchanged; adapters moved |
| C_host | host | grouped_nf4 | unset | 0.3.1 | OK; frozen bytes unchanged; adapters moved |
| D_reference | device | reference | unset | 0.3.1 | OK; frozen bytes unchanged; adapters moved |
| E_fixed_default | device | grouped_nf4 | unset (compressed) | this branch | OK; frozen bytes unchanged; adapters moved |

- **E is A's setup with the fix.** It trains, and its digest covers the double-quantized payload.
- **E's training peak is lower than B's,** 5.047 against 5.316 GiB, with the same load peak (5.259 GiB). That fits the
  absmax being compressed in E and not in B. The receipts do not record the compression itself.

## Correction: the receipts are now committed

As merged in #48, this directory held each arm's `run.log` but not its receipt. loggetta's `.gitignore` excludes
`receipts/`, and the receipts were not force-added. They survived on the box and are committed here unchanged, with
nothing re-run. Arm A has no receipt: it stopped before step 1.

- `arms/B_dq0/receipts/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T115515Z-9f6cdf6a.json`
- `arms/C_host/receipts/OLMoE-1B-7B-0924-host-grouped_nf4-t1024-20261009T115656Z-77afe9fe.json`
- `arms/D_reference/receipts/OLMoE-1B-7B-0924-device-reference-t1024-20261009T115838Z-e6184d69.json`
- `arms/E_fixed_default/receipts/OLMoE-1B-7B-0924-device-grouped_nf4-t1024-20261009T120044Z-af805ae9.json`

`tests/test_evidence_citations.py` checks that every path listed here is in git.
