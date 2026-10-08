# Native Thor full-model resident validation

The complete pinned Qwen3-0.6B, all 28 BF16 layers, batches1/16, lengths128/512 and 32 greedy steps pass the original native tokens and all-step logits exactly. Both APPA and PAAP sequences complete, with two warmups and five timed generations per arm/case. Every case uses fresh KV and retains the independent baseline FP64 normalization budget, stable weights and 160 normalization calls per layer.

These resident-process values are a screening check. They are not fresh-process confidence intervals or a model speedup claim. The faithful one-warp consumer preserves native normalization order and does not exercise the wide-block optimization. Thor SM110 and H20 observations are never pooled.

| Batch | Input tokens | A median of arm medians (ms) | P median of arm medians (ms) | Native token/logit exact | A/P timed arms |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 | 128 | 741.775 | 737.679 | pass | 4/4 |
| 1 | 512 | 725.155 | 723.011 | pass | 4/4 |
| 16 | 128 | 788.634 | 788.460 | pass | 4/4 |
| 16 | 512 | 2321.345 | 2321.921 | pass | 4/4 |

Eight independent whole APPA/PAAP quartets per batch were predeclared separately. Their first entry did not acquire the shared performance lock and naturally exited before launching any worker. Those formal model measurements remain unexecuted while another task owns the GPU; no samples or thresholds are substituted.

Exact frozen harness strings are in results/thor_model_resident/harness-sources.json.
Repository hooks normalize two log files; original text/hashes are retained in
ORIGINAL_RAW_TEXT.json with PUBLIC_FORMATTING.json. Unexecuted formal plan admissions remain separate from these resident results.
Timing JSON samples are unchanged.
