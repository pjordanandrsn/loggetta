# The planner head-to-head at the release: raw outputs and their public copies

The read registered in `bench/h2h-planner/PREREG-h2h-planner.md`: `python bench/h2h_planner.py evidence/planner-split/SPLIT-v3.json
--e4b E4B_CHECKOUT --out DIR`, run on 2026-10-24 from loggetta's merge commit `8a0b3be6acbb0fa2c91c2f586ffe509e9171d1f5`. It ran
on a development host's CPU, in a fresh venv holding only PyPI wheels of loggetta 0.7.0, experts4bit-qlora 0.58.0, grouped-nf4-gemm
0.51.0 and Accelerate 1.15.0, and its own release proof held all four as the published bytes. USD 0.

| file | what it is |
|---|---|
| `h2h-planner.public.json` | the driver's `h2h-planner.json` with its local paths made tokens |
| `accelerate.public.json` | the driver's `accelerate.json`, the same way |
| `STORE-ONLY.sha256` | the sha256 of the raw pair and the run's log, kept in the private receipt store |

**The raw files, by sha256** (also in `STORE-ONLY.sha256`):
- `h2h-planner.json`: `cead8935422a89b0d5fddaa863db89107c236868d5ad7aa19e838ed361265856`
- `accelerate.json`: `e6a2e2671ba997fd923cbfad85d3794c9d403ee947de997ff6fde10548dd2492`

**The token map** (`bench/h2h-planner/project_public.py`). A token replaces exactly these paths:
- `<python>` is the interpreter Accelerate ran under (its `argv[0]`);
- `<site-packages>` is the venv's site-packages directory, the rest of each import origin kept;
- `<pycache>` is the fresh bytecode prefix the release proofs ran under.

Every other value is byte-for-byte the raw file's, in the driver's own JSON layout. `tests/test_h2h_public.py` holds that: a
projection that changes anything but a path fails it. With the raw pair to hand (`H2H_RAW_DIR`), it also re-derives these files
from the raw ones and checks their sha256 against this list.
