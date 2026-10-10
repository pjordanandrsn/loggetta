# HO1: held-out cards

Registration: [`bench/HO1-PREREG.md`](../../bench/HO1-PREREG.md).

`rows.json` is the row set. It was written by `python bench/ho1_rows.py --e4b E4B --commit
d6d27ef5cb1509a576ec78c1e5e1a71987429b3b --json evidence/2026-10-10-heldout-cards/rows.json`. That gives 509 rows, 219 of
them primary (215 on an RTX 5090, 4 on an H100). Each row carries its setup and its two measured peaks, plus why it is
not primary where it is not. There are no plans here yet. They follow in a later change, after the registration is
reviewed.
