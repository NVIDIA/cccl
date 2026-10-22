# Torch-order full-model follow-up protocol

The retained fast-exp consumer failed native generation at batch 1. The original
regular-exp consumer passed batch 1 but failed batch 16 at both fixed lengths.
Torch-only replay matched native logits exactly. A common one-warp, bit-reversed
lane mapping follows the native descending XOR reduction tree and regular expf;
56 fixed FP32/BF16/masked normalization cases match native FP32 output bitwise.
The independent batch-16 diagnostic matches all native tokens and all-step logits.
These consumer changes apply to both A and P, and are not PR performance gains.
Keep both prior failures and unexecuted protocols/plans visible.

Use the same pinned Qwen3-0.6B checkpoint, all 28 BF16 layers, original fixed text,
batches 1 and 16, lengths 128 and 512, and 32 complete greedy-generation steps.
Every worker checks native tokens, exact native-versus-A logits, exact P-versus-A
logits, the independent FP64-normalization baseline-only error budget, fresh KV
state, stable weight addresses/versions and 160 calls per layer per timed case.
No prompt, precision or tolerance change is permitted to hide a prior failure.

After two finite resident APPA/PAAP screens pass, run eight fresh-process quartets
per batch, alternating APPA/PAAP. Each worker uses two warmups and five complete
generations per case; retain every sample, artifact hash and NVML observation.
The full quartet is the resampling unit: 10,000 bootstrap draws, seed 11504,
95% interval and 1% useful-speedup threshold. A foreign GPU1 process invalidates
the entire quartet; replace only whole groups with distinct IDs. Publish all
lengths, default controls and negative results. One-warp consumer parity does
not establish a wide-block or model speedup; micro results remain separate.
