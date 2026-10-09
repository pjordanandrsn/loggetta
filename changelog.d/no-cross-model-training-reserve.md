### Training plans no longer borrow another model's reserve slack (#43)

- **What changed.** A training plan's allocator reserve comes from receipts for this model on this GPU (or this model
  elsewhere through an anchor, as before). A model with none gets a conservative bound: the largest training slack
  measured on this GPU, and never below the 20% default. It is labelled `policy`. A sibling model's number for the same
  setup is no longer used. Serving keeps its cross-model tiers: its slack is 0.06–1.5%, and every serving plan sat
  over its driver peak (#29).
- **Why.** granite-3.1-3b-a800m, with its siblings' receipts on file, was planned with OLMoE's measured 0.222 (its own:
  0.228) and sat under its driver peak at 1.014.
- **What it costs.** In sample (`evidence/2026-10-09-moe-plan-vs-driver-no-borrow`), no plan is under its driver peak.
  granite-3.1 goes from 1.014 to 0.898 and granite-4.0-h-tiny from 0.896 to 0.796. A model nothing measured is priced
  at the card's worst training slack.
- **The rest of granite's gap is attributed** (`evidence/2026-10-09-a2000-granite-train-residual`). Its training peak is
  the loss, where three fp32 logits-sized tensors are live: 12 B per logit, against the 10 B that experts4bit-qlora's
  MoE estimate prices. That is the miss loggetta's dense estimate fixed in #27. The adapter gradients the estimate
  prices are not live at a loss peak and offset most of it: +23.5 MiB on today's stack.
- **Dense plans are unchanged:** byte-identical on 216 cases, with and without all 62 committed receipts on file.
- **Tests.** `test_training_never_borrows_another_models_slack` (fails before, passes after). Two reserve tests'
  receipts now name the model they stand for.
