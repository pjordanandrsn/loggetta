### Dense full-logit loss: price the live backward buffers

The pinned CPU/CUDA census observes three distinct fp32 logits-sized tensors together during log-softmax backward. Price twelve bytes per logit instead of ten, a source-derived missing tensor term. Retain the activation coefficient, chunked-loss bytes, inferred reserve, failed DQ7 verdict and execution opt-in. The resulting surplus remains visible; no historical receipt becomes a new capacity pass.
