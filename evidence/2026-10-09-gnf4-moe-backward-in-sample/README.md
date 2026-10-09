# The grouped_nf4 MoE backward line, in sample (#43)

`in-sample.md` and `in-sample.json` are `bench/gnf4_terms_in_sample.py`'s output: every committed MoE training receipt
with an allocated peak, priced with its own setup and workload. Run on CPU with experts4bit-qlora 0.50.0,
grouped-nf4-gemm 0.43.0 and transformers 5.18.0; configs from the Hugging Face Hub, no weights.

**This is in-sample.** The term was attributed on these receipts (#44 re-ran three of them on the A2000), so agreement
here says nothing about another card, shape or model.

- **OLMoE, `grouped_nf4`, both placements, both cards.** Before: 195–219 MiB under the allocated peak. After:
  90–114 MiB over. The new line is 308.8 MiB: the padded delta is bounded at `E x T` = 65,536 rows (640 MiB), where
  #44's replay saw 51,136 rows (500.9 MiB). The overshoot is the price of a bound in shape.
- **OLMoE, reference kernel.** No line; unchanged at −9.7 MiB.
- **Qwen3-30B-A3B on the RTX 5090.** No line: at its vocabulary the loss branch (1.56 GB) is larger than the
  backward branch. Unchanged, and over already.
- **granite-3.1-3b-a800m and granite-4.0-h-tiny.** No line, for the same reason. They stay 56 and 69 MiB under the
  allocated peak. This term does not explain that gap.

"Before" matches #43's "today's allocator est." column for every receipt the two share.
