<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native RTX5080 broadcast validation

1 of twelve normalization cases establish a useful improvement on native
RTX5080 SM120. 2 of twelve unchanged first-thread controls satisfy the
predeclared 1% regression bound. Every negative and unresolved result remains
in the table. These independent results do not override the H20 original-exp
width32 default interval; no complete-model gain is inferred from microbenchmarks.

Reduction control is 0909f9ee05c632af18fba74a366b657370a12aee and production
candidate is 76db7e3f3ab0d665fc87282b0160a321e51e698e. Eight independently built
extensions use explicit compute120/sm120, Torch2.13.0+cu132, C++17 and -O3.
The read-only existing overlay's NVCC13.4/headers13.2 mismatch is resolved by
extracting four official, SHA-verified CUDA13.2.51 components into this task's
private directory. Existing environments, power/clock settings and other
processes are unchanged. GPU UUID is GPU-99386c0c-e54e-e92e-9cf1-3cd91b6397f7.

Each workload has eight balanced fresh-process APPA/PAAP whole quartets, for96
workers. Every worker naturally exits zero. Source/diff/library/harness/common/
PID and input/output hash checks pass. Before/after plus0.5-second GPU sampling
observe no foreign CUDA process: zero invalid quartets. The entire finite queue
holds the shared performance lock and a visible idle parent CUDA reservation.
CPU-only downloads continue and load is recorded. Screening data are excluded.
Intervals use10000 whole-quartet bootstrap draws, seed11504; lower95%CI must
exceed1.01 for useful improvement and remain at least0.99 for default controls.
Process-median latencies below are microseconds.

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-8192 | first-thread/32 | 8 | 7.695 | 7.666 | 1.0035× | 0.9532–1.0730 | default bound not met |
| integer-8192 | first-thread/128 | 8 | 7.702 | 7.717 | 1.0131× | 0.9599–1.0750 | default bound not met |
| integer-8192 | first-thread/256 | 8 | 8.781 | 8.863 | 0.9688× | 0.9288–1.0012 | default bound not met |
| integer-8192 | first-thread/512 | 8 | 16.499 | 16.538 | 0.9956× | 0.9900–0.9995 | within default bound |
| integer-8192 | normalization/32 | 8 | 7.657 | 7.677 | 1.0161× | 0.9681–1.0841 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 8.500 | 8.127 | 0.9993× | 0.8327–1.1474 | no stable >1% gain |
| integer-8192 | normalization/256 | 8 | 12.635 | 12.590 | 0.9893× | 0.9669–1.0080 | no stable >1% gain |
| integer-8192 | normalization/512 | 8 | 24.689 | 22.658 | 1.0897× | 1.0891–1.0905 | stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 7.670 | 7.629 | 0.9500× | 0.8663–1.0120 | default bound not met |
| legacy-8192 | first-thread/128 | 8 | 8.842 | 8.767 | 1.0164× | 0.9862–1.0491 | default bound not met |
| legacy-8192 | first-thread/256 | 8 | 12.714 | 12.655 | 1.0113× | 0.9829–1.0383 | default bound not met |
| legacy-8192 | first-thread/512 | 8 | 22.858 | 22.861 | 0.9768× | 0.9278–1.0049 | default bound not met |
| legacy-8192 | normalization/32 | 8 | 7.645 | 7.439 | 1.0053× | 0.9626–1.0398 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 8.725 | 8.717 | 0.9905× | 0.9419–1.0375 | no stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 14.599 | 14.569 | 1.0021× | 0.9729–1.0321 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 28.852 | 28.872 | 0.9838× | 0.9522–1.0007 | no stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 7.450 | 7.674 | 0.9509× | 0.8901–0.9886 | default bound not met |
| retained-8192 | first-thread/128 | 8 | 8.649 | 8.709 | 0.9880× | 0.9763–0.9973 | default bound not met |
| retained-8192 | first-thread/256 | 8 | 12.623 | 12.596 | 0.9868× | 0.9617–1.0013 | default bound not met |
| retained-8192 | first-thread/512 | 8 | 22.797 | 22.821 | 1.0005× | 0.9977–1.0039 | within default bound |
| retained-8192 | normalization/32 | 8 | 7.408 | 7.526 | 0.9615× | 0.8979–1.0048 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 8.639 | 8.632 | 0.9737× | 0.9228–1.0053 | no stable >1% gain |
| retained-8192 | normalization/256 | 8 | 14.536 | 14.535 | 1.0320× | 0.9970–1.0974 | no stable >1% gain |
| retained-8192 | normalization/512 | 8 | 28.857 | 28.839 | 1.0005× | 1.0002–1.0008 | no stable >1% gain |


Both independent Torch-order probes match native FP32 and BF16 output bitwise
for56/56 cases, including rounded/masked values and unchanged independent FP64
error budgets. Four original-exp default kernels have identical complete CUDA
instructions and encodings in separately built A/P binaries. Regular expf,
retained __expf and integer count normalization remain distinct workloads;
shared A/P consumer arithmetic is excluded from the PR gain.

Four initial CPU build entries naturally failed before any GPU sampling: Ninja
was absent from the process PATH, host include ordering was incomplete, the
borrowed compiler/header versions differed, then shared library headers were
missing. Their original logs and failed statuses are retained. The final eight
CPU builds all exit zero using only private process configuration and matching
components. No measured source, input, seed, tolerance or statistic is changed.

[Raw index](results/sm120_8192/RAW_INDEX.json),
[formal manifest](results/sm120_8192/sm120-formal-micro-manifest.json),
[default instruction comparison](results/sm120_8192/sm120-default-sass-comparison.json)
and [exact harness sources](results/sm120_8192/sm120-micro-harness-sources.json)
are committed. Original local archives are byte-verified; any formatting changes
required by repository hooks retain the original text and its SHA separately.
