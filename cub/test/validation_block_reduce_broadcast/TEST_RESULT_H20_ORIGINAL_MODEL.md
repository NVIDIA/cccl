<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Original-consumer and complete-model follow-up

Original regular-exp normalization improves 1.0271× at 8,192 rows and 1.0251× at
65,536 rows for width 128. Both intervals exceed the fixed 1% useful-gain threshold.
Other widths do not meet that threshold. All 64 fresh performance workers exited
naturally; before/after and 0.5-second monitoring observe no foreign GPU1 process.
Sources, binaries, flags, seed, warmups and event sampling are unchanged.

The host/source are the same as the [65,536-row report](TEST_RESULT_H20_65536_BROADCAST.md).
Latencies are medians of process medians in microseconds. Intervals use eight whole
APPA/PAAP quartets, 10,000 bootstrap draws and seed 11504.

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| legacy-65536 | first-thread/32 | 8 | 50.152 | 50.205 | 0.9995× | 0.9989–1.0001 | within default bound |
| legacy-65536 | first-thread/128 | 8 | 45.427 | 45.491 | 0.9994× | 0.9985–1.0001 | within default bound |
| legacy-65536 | first-thread/256 | 8 | 108.352 | 108.404 | 0.9998× | 0.9993–1.0002 | within default bound |
| legacy-65536 | first-thread/512 | 8 | 233.154 | 233.154 | 1.0000× | 0.9999–1.0001 | within default bound |
| legacy-65536 | normalization/32 | 8 | 44.745 | 44.655 | 1.0025× | 1.0015–1.0033 | no stable >1% gain |
| legacy-65536 | normalization/128 | 8 | 102.122 | 99.620 | 1.0251× | 1.0245–1.0256 | stable >1% gain |
| legacy-65536 | normalization/256 | 8 | 204.388 | 202.653 | 1.0088× | 1.0085–1.0091 | no stable >1% gain |
| legacy-65536 | normalization/512 | 8 | 429.658 | 427.362 | 1.0052× | 1.0051–1.0053 | no stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 7.324 | 7.413 | 0.9970× | 0.9897–1.0051 | default bound not met |
| legacy-8192 | first-thread/128 | 8 | 7.432 | 7.486 | 0.9981× | 0.9910–1.0059 | within default bound |
| legacy-8192 | first-thread/256 | 8 | 12.950 | 13.032 | 0.9974× | 0.9932–1.0018 | within default bound |
| legacy-8192 | first-thread/512 | 8 | 25.547 | 25.650 | 0.9987× | 0.9962–1.0014 | within default bound |
| legacy-8192 | normalization/32 | 8 | 7.284 | 7.291 | 1.0019× | 0.9951–1.0091 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 14.273 | 13.927 | 1.0271× | 1.0233–1.0314 | stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 26.667 | 26.477 | 1.0089× | 1.0068–1.0113 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 53.682 | 53.242 | 1.0095× | 1.0082–1.0107 | no stable >1% gain |

## Default control uncertainty

The 8,192-row width-32 default control has ratio 0.9970× and interval
0.9897–1.0051. It does not establish the predeclared 1% regression bound. The
interval also contains parity; it is not evidence that default code changed.
The corresponding independently compiled instructions/encodings are identical,
as are all 26 default kernels already reported. The initial failed-bound result
remains visible. The one predeclared validation has now completed all 32 workers
naturally, with unchanged sources, binaries, flags and samples. Its width-32
default result is 0.9938× (95% CI 0.9885–0.9992), also failing to establish the
1% default bound. This separate result neither replaces nor pools the initial
valid estimate. All other default controls meet the bound. CPU-only resource
inspection confirms identical register/shared/local/stack/constant usage at all
four widths; it does not override either failed-bound estimate.

## Complete-model screening

The pinned Qwen3-0.6B checkpoint passes both faithful common-consumer resident
screens. Each runs APPA then PAAP with all 28 BF16 layers, original prompt lengths
128/512, 32 greedy tokens, fresh KV state, two warmups and five complete timed
generations per arm/case. A follows native Torch reduction order through the
one-warp mapping; P uses the opt-in API against the same consumer. The original
retained-exp and batch-16 regular-exp native-token failures remain archived.

| Batch | Resident arms | Fixed lengths | Native 32-step tokens | Native all-step logits | P-vs-A logits | Calls/layer/case | Weight addresses/versions |
| ---: | :--- | :--- | :--- | :--- | :--- | ---: | :--- |
| 1 | APPA + PAAP | 128 / 512 | exact | exact | exact | 160 | stable |
| 16 | APPA + PAAP | 128 / 512 | exact | exact | exact | 160 | stable |

Both processes naturally exit zero. Source/diff/library/harness/common hashes
match the frozen manifest; checkpoint files and native Torch sources were checked
before screening. Every arm passes the unchanged baseline-only FP64-normalization
error budget inside the BF16 network. Screens are excluded from formal estimates.

The formal follow-up was interrupted by loss of the H20 SSH connection at
2026-10-01 14:48:50 UTC. The controller returned transport exit 255; no signal or
shutdown command was sent. Before disconnection, 16 batch-1 remaining workers
were observed complete and the first batch-16 worker had started. Only the first
14 remaining workers had been backed up locally, together with the earlier 16.
The final remote state and checkpoint cleanup cannot be verified.

All 30 backed-up workers pass source/library/harness/PID, native generation and
exact-logit gates. Seven complete batch-1 quartets are auditable; the last partial
quartet is retained and excluded. The prescribed eight-quartet estimate is
incomplete, and no interval or model gain is claimed:

| Batch / input tokens | Complete quartets | A median (ms) | P median (ms) | A/P | Result |
| :--- | ---: | ---: | ---: | ---: | :--- |
| 1 / 128 | 7 | 701.821 | 697.600 | 1.0003× | incomplete; no formal speedup claim |
| 1 / 512 | 7 | 705.297 | 701.397 | 0.9987× | incomplete; no formal speedup claim |
| 16 / 128 and 512 | 0 | — | — | — | formal records not recovered |

The earlier complete resident batch-1/batch-16 equivalence screens remain valid.
This one-warp consumer does not exercise wider-block optimization. The user
moved remaining work to RTX 5080 and Thor; their independent runs will be
reported by architecture and will not complete the H20 timing estimate by pooling
samples. Two task-owned H20 checkpoint copies have unverified cleanup because
that machine is inaccessible. No other task's files or environments were modified.

## Fixed default-control validation

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| legacy-8192-validation2 | first-thread/32 | 8 | 7.309 | 7.409 | 0.9938× | 0.9885–0.9992 | default bound not met |
| legacy-8192-validation2 | first-thread/128 | 8 | 7.425 | 7.479 | 0.9966× | 0.9913–1.0026 | within default bound |
| legacy-8192-validation2 | first-thread/256 | 8 | 12.944 | 13.023 | 0.9959× | 0.9932–0.9987 | within default bound |
| legacy-8192-validation2 | first-thread/512 | 8 | 25.553 | 25.650 | 0.9984× | 0.9969–1.0000 | within default bound |
| legacy-8192-validation2 | normalization/32 | 8 | 7.272 | 7.291 | 1.0004× | 0.9947–1.0077 | no stable >1% gain |
| legacy-8192-validation2 | normalization/128 | 8 | 14.269 | 13.951 | 1.0250× | 1.0224–1.0279 | stable >1% gain |
| legacy-8192-validation2 | normalization/256 | 8 | 26.641 | 26.484 | 1.0079× | 1.0064–1.0097 | no stable >1% gain |
| legacy-8192-validation2 | normalization/512 | 8 | 53.729 | 53.244 | 1.0094× | 1.0086–1.0103 | no stable >1% gain |

## Evidence

[Raw records](results/original_consumer_model_screens/evidence-index.json),
[the model manifest](results/original_consumer_model_screens/formal-model-torch-order-manifest.json)
and [exact harness snapshots](results/original_consumer_model_screens/harness-sources.json)
are committed. The micro status is an immutable 64-worker subset of the full
finite 66-worker controller. Its full status preserves both subsequent model
screens and all observations. Original CUDA consumer sources remain in `normalization.cu`, `bindings.cpp`
and `build.py`; the exact fresh-worker benchmark wrappers are in this snapshot.

The [shutdown snapshot index](results/h20_shutdown/RAW_INDEX.json) preserves the
completed default-validation queue, all backed-up model records, the incomplete
quartet, controller observations and the transport-loss checkpoint.
