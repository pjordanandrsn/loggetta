### MoE plans against measured driver peaks (report only)

`bench/plan_vs_driver.py` reads every committed receipt with a plan total and a driver peak, and can replan the
training receipts with today's code (`evidence/2026-10-08-moe-plan-vs-driver`).

- **Serving plans hold:** all 22 are over their driver peak, including SV5–SV7's registered RTX 4090 plans.
- **MoE training plans on the RTX A2000 do not:** today's OLMoE plans sit under the driver peak (driver / plan up to 1.105).
  The allocator estimate is 0.213 GiB under the allocated peak on both placements, and no MoE training residual is
  learned.

No code, coefficient or default changed.
