### Dense training's frozen-parameter digest takes a rank-0 parameter

`dense_train.frozen_digest` hashed each frozen first- and last-layer parameter through a byte view. A byte view of a
rank-0 tensor wider than a byte refuses (`self.dim() cannot be 0 to view Float as Byte`), so a model with a scalar
frozen parameter could not be checked. It now flattens a contiguous copy before the view. For a ranked tensor the bytes
are the same, so every existing digest is unchanged: `tests/test_frozen_digest_rank0.py` pins today's ranked digests (RNG-free fixtures) in
fp32, bf16 and fp16, and checks a rank-0 parameter in each against an independently computed digest.
