<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native RTX5080 complete-model formal validation

The original Qwen/Qwen3-0.6B checkpoint at
c1899de289a04d12100db370d81485cdf75e47ca runs all28 BF16 layers with fresh KV
state, B1/B16, input lengths128/512 and32 deterministic generated tokens. All
64 fresh workers preserve exact native generated tokens and teacher-forced
logits at every step, independently fixed FP64 normalization error budgets,
160 consumer calls per layer per case and resident weight addresses/versions.
Native Transformers5.14.1 source and all seven checkpoint files are hashed.

This Torch-order integration exercises the unchanged one-warp path. It verifies
complete-model correctness and reports its measured timing; it cannot establish
a model speedup from the wider-block optimization. Latencies below are ms.
The initial startup attempt naturally failed when Triton's C helper could not
find stdlib.h. Its full log/status are retained. The subsequent jobs use existing
read-only headers through process-local C_INCLUDE_PATH and private caches;
no installed environment or model arithmetic was changed.

Each workload has eight balanced fresh-process APPA/PAAP whole quartets.
Every worker naturally exits zero. Frozen source/diff/library/harness/common/PID
and input/output hashes pass, with zero foreign GPU observations or invalid
quartets. The finite queue uses the shared performance lock and visible parent
CUDA reservation. CPU-only downloads continue; observed load remains recorded.
Intervals use10000 whole-quartet bootstrap draws, seed11504. Useful improvement
requires lower95%CI>1.01; default controls require lower95%CI>=0.99.
This is a separate native SM120 measurement using Torch2.13.0+cu132 and GPU
GPU-99386c0c-e54e-e92e-9cf1-3cd91b6397f7, with no H20/Thor/screen pooling.

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| model-torch-order-b1 | model/128 | 8 | 462.562 | 465.099 | 0.9864× | 0.9670–1.0036 | no stable >1% gain |
| model-torch-order-b1 | model/512 | 8 | 466.214 | 469.129 | 1.0025× | 0.9802–1.0278 | no stable >1% gain |
| model-torch-order-b16 | model/128 | 8 | 499.997 | 497.443 | 1.0057× | 0.9948–1.0178 | no stable >1% gain |
| model-torch-order-b16 | model/512 | 8 | 766.965 | 767.565 | 0.9993× | 0.9940–1.0049 | no stable >1% gain |


[Raw index](results/sm120_model_formal/RAW_INDEX.json) includes every actual worker
output/log, status, frozen manifest and exact measured harness-source snapshot.
Required repository formatting preserves changed original text and its SHA.
