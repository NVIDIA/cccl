# Original-consumer full-model follow-up protocol

The retained __expf model native gate failed before P. The original expf
consumer preserved native Qwen3 tokens at both frozen prompt lengths in the
independent diagnostic; the Torch-only wrapper matched all native logits exactly.
The retained failure and its predeclared, unexecuted model plans remain archived.

Use the independently rebuilt original expf consumer for model screening and
formal model estimates. Preserve the same checkpoint, all 28 BF16 layers, prompt
text, batch1/batch16, lengths128/512, 32 greedy tokens, fresh KV state, native-token
gate, all-step logits, baseline-only FP64-normalization budget and exact P-vs-A
logit requirement. Do not change the source API or substitute prompts to bypass
the retained failure. Each screening process owns one resident weight set and
records parameter addresses/versions across its APPA/PAAP arms.

Only after both batch screens pass, run eight balanced independent-process
APPA/PAAP quartets per batch. Use five timed complete generations per case,
two warmups, eight whole-quartet resampling units, 10,000 bootstrap draws and
fixed seed11504. Retain the 1% useful model speedup threshold. Native correctness
runs in every worker; native latency context is recorded in the independent
resident screens rather than repeatedly timed in every formal A/P worker.
Any foreign selected-GPU process invalidates the entire quartet. Both lengths,
all negative and invalid cases remain reported; no micro-to-model extrapolation.
