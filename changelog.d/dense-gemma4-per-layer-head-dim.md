### Dense plans: gemma4's per-layer head_dim is read per layer, and its layer differences are refused by name

transformers 5.20 declares gemma4's `head_dim` per layer (`per_layer_config`) and raises on the global attribute. The
dense backend read it globally, so describing `google/gemma-4-E2B-it` failed with `AmbiguousGlobalPerLayerAttributeError`.
It now reads each layer's `head_dim`, classifies that layer's projections at that width, and then refuses with each
difference that pricing, which assumes identical decoder layers, cannot represent: the per-layer head_dim (256 in 28
layers, 512 in 7), layers 15-34 without `k_proj`/`v_proj` (cross-layer KV reuse), and the wider MLP in those layers.
Tested on the model's own `config.json` at a pinned revision. gemma4 is still not plannable; the refusal now says why
(#78).
