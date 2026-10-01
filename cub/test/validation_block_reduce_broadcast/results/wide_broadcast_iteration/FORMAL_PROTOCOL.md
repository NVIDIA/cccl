# Frozen formal follow-up protocol

Screening palindromes select a candidate; their measurements are retained but excluded
from formal estimates. Compare the candidate to the same upstream-header control in
fresh independently built workers. Freeze both source commits, consumers, flags and
checkpoint hashes before formal timing. No source edits during sampling.

For each consumer and row size, run eight balanced quartets, alternating APPA/PAAP.
The unit of resampling is a whole quartet, never an inner CUDA-event repetition.
Use all 20 samples per worker and report quartet ratios and medians in Markdown.
Predeclare a 1% minimum useful normalization speedup and a 1% default-path regression
bound. Use 10,000 fixed-seed bootstrap draws over the eight log-ratio quartets.
Declare stable improvement for a case only when the two-sided 95% interval lower
bound exceeds 1.01; declare model improvement by the same criterion. Results whose
interval crosses 1 or falls below the 1% practical bound are reported explicitly.
All scheduled widths/row sizes, failed workers and invalidated groups remain visible.
No general claim when only selected widths improve.

Any observed foreign process on the selected GPU invalidates its entire quartet.
No partial quartet substitution. A rerun is a separately labelled complete quartet.
Keep raw observations, source/library/input/output/checkpoint hashes and commands.

Full-checkpoint resident screening loads all 28 BF16 Qwen3 layers, uses native greedy
HF generation, fresh KV state, exact 32 tokens against native attention, and all-step
logit checks against the independent FP64 normalization reference with a baseline-only
frozen budget. Both batch sizes 1 and 16 and prompt lengths 128/512 remain disclosed.
The independent FP64 reference is for normalization; it is not a full FP64 network.
Native Torch attention timings are context, not an interchangeable upstream-CUB control.
Formal model runs require the same validation in separate fresh workers.

Common register-retention/__expf or generation-harness changes apply equally to both
arms and cannot be attributed to the PR. Original rereading-expf results remain in
the report alongside the new consumer. A kernel-level useful gain does not establish
a model-level gain.

For the sink exponential follow-up, use eight balanced APPA/PAAP quartets per
return-LSE mode in fresh processes. Keep all 16 shape/sink cases from each worker,
including no-sink controls. The same 1% practical threshold and whole-quartet
bootstrap apply. Compare to the already-correct published sink implementation.
A Qwen3 checkpoint without trained sinks supplies a no-sink regression only;
its timings cannot establish a learned-sink model speedup.
