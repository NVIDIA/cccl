<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native Thor broadcast validation

Five of twelve normalization cases establish a useful improvement on native
Thor SM110. All twelve unchanged first-thread controls satisfy the predeclared
1% regression bound. Other widths retain their negative results. These results
are separate from H20 and do not override its unresolved original-exp width32
control interval. No complete-model improvement is inferred from microbenchmarks.

The immutable reduction control is 0909f9ee05c632af18fba74a366b657370a12aee;
production candidate is 76db7e3f3ab0d665fc87282b0160a321e51e698e. Eight
independent native extensions are built with explicit compute110/sm110, Torch
2.13.0+cu130, CUDA13.0, -O3 and C++17. The baseline and candidate consumers use
identical arithmetic and inputs within each workload. No environment, clock,
power setting or other process was changed. GPU UUID is
GPU-a7c66ad2-6dbb-0ab8-c1a2-37ba6dba3600.

Each workload has eight balanced fresh-process APPA/PAAP quartets, for 96 workers.
Every worker naturally exits zero. Source/diff/library/harness/common/PID checks
and every quartet's input/output hashes pass. Before/after and 0.5-second GPU
monitoring observe no foreign CUDA process: zero invalid quartets. The entire
finite queue holds the shared performance lock and a visible idle parent CUDA
reservation. CPU-only model downloads continue and system load is recorded.
Screening runs are excluded. Intervals use 10,000 whole-quartet bootstrap draws,
seed11504; the lower95% bound must exceed1.01 for useful improvement and remain
at least0.99 for default controls. Process-median latencies are microseconds.

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-8192 | first-thread/32 | 8 | 22.582 | 22.604 | 0.9998× | 0.9993–1.0004 | within default bound |
| integer-8192 | first-thread/128 | 8 | 22.683 | 22.645 | 1.0017× | 0.9990–1.0044 | within default bound |
| integer-8192 | first-thread/256 | 8 | 42.181 | 42.198 | 0.9996× | 0.9988–1.0004 | within default bound |
| integer-8192 | first-thread/512 | 8 | 93.646 | 93.512 | 1.0004× | 0.9995–1.0012 | within default bound |
| integer-8192 | normalization/32 | 8 | 26.711 | 26.755 | 0.9988× | 0.9983–0.9994 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 37.982 | 35.799 | 1.0613× | 1.0587–1.0636 | stable >1% gain |
| integer-8192 | normalization/256 | 8 | 75.891 | 72.168 | 1.0524× | 1.0518–1.0531 | stable >1% gain |
| integer-8192 | normalization/512 | 8 | 162.258 | 154.253 | 1.0522× | 1.0519–1.0525 | stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 22.602 | 22.586 | 1.0006× | 0.9996–1.0016 | within default bound |
| legacy-8192 | first-thread/128 | 8 | 28.890 | 28.889 | 0.9991× | 0.9979–1.0002 | within default bound |
| legacy-8192 | first-thread/256 | 8 | 59.475 | 59.464 | 1.0002× | 0.9999–1.0006 | within default bound |
| legacy-8192 | first-thread/512 | 8 | 120.340 | 120.363 | 0.9998× | 0.9989–1.0005 | within default bound |
| legacy-8192 | normalization/32 | 8 | 27.099 | 26.839 | 1.0102× | 1.0076–1.0129 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 39.049 | 36.023 | 1.0839× | 1.0827–1.0852 | stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 77.913 | 77.919 | 1.0000× | 0.9998–1.0002 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 166.116 | 166.074 | 1.0000× | 0.9993–1.0007 | no stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 22.596 | 22.606 | 0.9999× | 0.9988–1.0011 | within default bound |
| retained-8192 | first-thread/128 | 8 | 28.744 | 28.764 | 0.9988× | 0.9974–1.0001 | within default bound |
| retained-8192 | first-thread/256 | 8 | 59.481 | 59.470 | 1.0001× | 0.9998–1.0004 | within default bound |
| retained-8192 | first-thread/512 | 8 | 120.414 | 120.239 | 1.0007× | 0.9996–1.0018 | within default bound |
| retained-8192 | normalization/32 | 8 | 27.121 | 26.866 | 1.0080× | 1.0052–1.0110 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 39.112 | 36.099 | 1.0830× | 1.0813–1.0847 | stable >1% gain |
| retained-8192 | normalization/256 | 8 | 77.924 | 77.921 | 1.0000× | 0.9998–1.0001 | no stable >1% gain |
| retained-8192 | normalization/512 | 8 | 166.120 | 166.056 | 1.0000× | 0.9992–1.0008 | no stable >1% gain |


The standalone faithful Torch-order probes also pass56/56 native FP32 and BF16
bitwise matches in each A/P process, using full fixed widths, rounded/masked
inputs and the unchanged independent FP64 normalization error budget. Four
original-exp default kernels have identical CUDA instructions and encodings in
independently rebuilt binaries. Original-exp uses regular expf; retained uses
stored exponentials and __expf in both arms. Integer normalization remains a
separate bounded count workload. Shared consumer arithmetic choices are excluded
from the PR improvement.

The first extra original-exp CPU build ran before its source transfer completed
and naturally failed with a missing bindings file. That log is preserved; the
corrected transfer is hash-verified and both completed builds exit zero. The
first read-only auditor assumed one common.py hash across all supplied harnesses
and rejected the manifest's unused model helper. Its corrected version checks
each executing script against its own exact frozen common.py hash; all other
source, artifact, completeness, ordering and GPU interference gates are retained.
Neither correction changes measured code, data, quartets, seeds or thresholds.

The complete original28-layer Qwen model passes native tokens and all-step logits
exactly for batches1/16 and lengths128/512. All16 resident timed arms finish
naturally. The [model table](TEST_RESULT_THOR_MODEL_RESIDENT.md) retains those
screening timings separately; fresh-process model confidence intervals remain
unexecuted because another finite queue holds the GPU. No model speedup is
claimed and no H20 model sample is pooled here.

[Raw data index](results/thor_8192/RAW_INDEX.json),
[formal manifest](results/thor_8192/thor-formal-micro-manifest.json),
[default instruction comparison](results/thor_8192/thor-default-sass-comparison.json)
and [exact harness snapshots](results/thor_8192/harness-sources.json) are committed.

Required repository hooks normalize EOF/trailing whitespace in 18 copied log and
metadata files. Their original byte strings and original/published hashes remain
in [the raw text snapshot](results/thor_8192/ORIGINAL_RAW_TEXT.json) and
[formatting record](results/thor_8192/PUBLIC_FORMATTING.json). Original local
archives are byte-verified; all formal timing JSON values remain unchanged.
