# Evidence

Each directory here holds one reading: its receipts, the files they cite, and a README that says what was measured and
what it supports. `planner-split/` is the planner's frozen split of those receipts into derivation and held-out rows.

## The card a receipt ran on

A plan and its receipt record which GPU they were made on, so `execute` can refuse to run a plan on a different card.

- **From the first release after 0.7.0:** they record a card id, `hardware.gpu.uuid_hash` (and `plan.hardware.gpu.uuid_hash`
  inside a receipt). It is a keyed hash of the GPU's UUID under a random per-install key that is never written out. It is
  stable for one card on one install and cannot be linked across installs, so a shared receipt no longer names the card.
- **Up to and including 0.7.0:** plans and receipts recorded the GPU's UUID itself, at `hardware.gpu.uuid` and
  `plan.hardware.gpu.uuid`. Receipts of that age committed here are in one of two states:
  - some of the RTX A2000 readings (in `*a2000*` directories) still carry it, left as committed;
  - later ones had such a file moved out of the public tree, with its sha256 kept in the directory's
    `STORE-ONLY.sha256` and, where a reading needs it, a public copy with the UUID removed.

`execute` still accepts a plan saved before card ids and compares its UUID as before.
