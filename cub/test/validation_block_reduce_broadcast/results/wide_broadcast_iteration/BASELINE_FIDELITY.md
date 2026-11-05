# Native model baseline investigation

The fixed v2 plan passed 29 CTest entries, nine sanitizer runs and 24 independent
consumer screening workers. Its 37th worker naturally failed before the candidate:
the upstream-CUB consumer's 32 greedy tokens differed from native Qwen3 attention.
The remaining batch-16 and native-FA regression workers were not executed.
No model timing or native-equivalence claim is accepted from this failed gate.

The original harness and protocol stay frozen in the failed-plan record. A separate
finite diagnostic compares native N, a Torch-only wrapper T, upstream-CUB A, and
an independent FP64 normalization G, using the same checkpoint and fixed prompts.
All 32 teacher-forced logits use N's continuation. Top-1 diagnosis excludes both
configured EOS IDs, matching the original min_new_tokens=32 generation rule.

| Hypothesis | Prediction | Falsifier | Current status |
| :--- | :--- | :--- | :--- |
| Wrapper changed attention geometry or masks | T differs from N | T and N have exact all-step logits and tokens | Measured; see model-baseline-diagnose-legacy.json and model-baseline-diagnose-retained.json |
| Consumer numerical path changes a close token decision | T=N; A differs; the first differing prediction has a small measured logit margin | A differs substantially from G or the mismatch is present in T | Measured; see model-baseline-diagnose-legacy.json and model-baseline-diagnose-retained.json |
| Register retention plus __expf introduces the native mismatch | Original expf consumer retains N's tokens where the retained consumer differs | Original expf consumer shows the same mismatch | Measured; see model-baseline-diagnose-legacy.json and model-baseline-diagnose-retained.json |

Installed Torch2.7 PersistentSoftmax.cuh and pinned upstream SoftMax.cu use
std::exp and division by the sum. The reciprocal-multiply explanation is therefore
rejected from source evidence; no patch is justified by that explanation.

If the baseline cannot preserve native tokens, disclose that limitation and keep
consumer-pipeline timing distinct from native production inference. Do not silently
remove the native gate, swap prompts, or attribute a common harness correction to
ReduceBroadcast.

Measured outcome: T and N have exact all-step logits and all 32 tokens at both
lengths. Original expf A keeps N's tokens at both 128 and 512. Retained __expf A
first diverges at zero-based token 14 for length128: native IDs785/334 have logits
16.875/16.75, while A rounds both to16.875. The retained model native gate remains
failed; its positive micro timings do not establish native-model fidelity.

The next full-model screen uses the original expf consumer and independently
rebuilt libraries, retaining the same prompts, tokens, native gate, all-step
FP64-normalization reference and baseline-only budget. No API change or looser
accuracy check is used to bypass the failed retained gate.
