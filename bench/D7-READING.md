# Running the registered D7 reserve read

The reader implements [D7-PREREG.md](D7-PREREG.md), merged as
`a4fbb266eda101a5d1a27434929213fb542770d9`. It reports the fixed derivation/holdout read and licenses no planner
observations. The dense default remains 20%; the executor development opt-in remains required.

Use the fetched original DQ7 `tc1/receipts` directory and a clean e4b checkout at DQ7 merge
`4ca5a3494e746b9aaa812270e03f4d0c1ec050a9`. The reader reruns that exact reducer before deriving anything.
It refuses a changed source checkout and preserves SHA-256 of every source receipt and proof plus the reducer source.
Output creation is exclusive, so a report cannot overwrite an earlier reading.

```sh
PYTHONPATH=. python bench/read_dense_reserve.py /path/to/dq7/tc1/receipts \
  --e4b /path/to/clean/e4b-dq7-merge-checkout --out /path/to/fresh/d7-read.json
```

All six seq4096 holdouts remain outside the derivation. Fractions use exact integer pairs, and byte charges round
up. Every holdout reports allocator estimate, allocated/reserved peaks, derived reserve/total, unchanged 20% charge/
total and both residuals. Missing, duplicate, inconsistent or invalid byte fields are VOID; an unlicensed DQ7 verdict
is DQ7_UNLICENSED. A valid failed holdout is CALIBRATION_FAIL and cannot increase the derived fraction.

A later observation-import PR must enforce the registered topology/card/runtime/setup/sequence scope in every planner
fallback and quote original derivation and held-out sources. A report marked CALIBRATION_PASS alone is insufficient
to import a reserve. No context, host, activation or performance correction is made by this reader.
