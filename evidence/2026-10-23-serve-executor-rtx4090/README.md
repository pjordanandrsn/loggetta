# Serve executor GPU proof · RTX 4090 · 2026-10-23

One rented RTX 4090 (driver 550.78) ran `loggetta execute` on a serve plan for `allenai/OLMoE-1B-7B-0924`
(all experts in VRAM, decode graphs, 4096-token context, one sequence). loggetta was at `1be0c2f6`, with
experts4bit-qlora 0.57.0 and grouped-nf4-gemm 0.50.0 installed from PyPI ([versions.txt](versions.txt)).
This is a smoke test of the executor path, not a performance reading.

| Check | Result |
| :--- | :--- |
| `bench/serve_validate.py` (the bench path) | exit 0 |
| `loggetta plan --workload serve`, then `loggetta execute` | feasible; exit 0; receipt status OK |
| Single-stream decode step (traced) | 4.422 ms median over 63 steps; 144.2 tokens/s |
| PyTorch allocator peak | 5.26 GB (plan estimate without context and reserve: 5.87 GB) |
| Driver-reported process peak | not sampled on this host |
| Paths in the receipt | none |
| `execute` with an inherited `E4B_K25_PLAN` set | refused by name, value not printed, no receipt ([refusal.txt](refusal.txt)) |

Files: [receipt.json](receipt.json) (the receipt as written, renamed) with its step trace
[steps.jsonl](steps.jsonl), [smoke_verdict.json](smoke_verdict.json) (every check the smoke script made),
[gpu.txt](gpu.txt).

The run used `--allow-development-executor`, which the serve executor then required. Its
`provenance.sources.loggetta.version` is null because the version list did not include loggetta itself; the receipt
binds loggetta by the sha256 of `loggetta/__init__.py`, which matches `1be0c2f6`. The change that adds this directory
lifts the opt-in for serve plans and records loggetta's version.
