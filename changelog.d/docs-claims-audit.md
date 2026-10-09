### Docs: claims brought back in line with the committed evidence

A claims audit checked about 330 statements in the README, PYPI.md and docs/ against the receipts. This fixes the ones
that drifted:
- RESULTS number drifts and roundings, including the 6.64 → 5.64 replan figure, which comes from the original receipt.
- The prefill-graph pool at 30B is +0.56 GiB with NF4 experts. The +3.3 GiB cited before was the int4 stack. The
  planner's explain string is fixed too.
- ARCHITECTURE:
  - `train.py` reaches the grouped kernel through its arena path.
  - gnf4 0.39.0 first shipped the route interface.
  - There are two backends.
  - The interface list is the main one, not all of it.
  - Training never borrows another model's reserve.
  - Serving reads a second setup field (`placement`).
- SESSION-REPORT is left as dated and gains a correction note.
- README/PYPI say "estimates" for the OLMoE 0.2 GiB figure, and scope the no-plan-under-peak claim to today's planner.
