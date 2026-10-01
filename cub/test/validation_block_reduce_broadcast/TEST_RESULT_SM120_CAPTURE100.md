<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Native RTX5080 single-replay timing follow-up

This distinct scope has 5/12 useful normalization gains
and 12/12 default controls meeting the predeclared regression bound.
The original host-submission results, including ten unresolved default bounds,
remain in TEST_RESULT_SM120_BROADCAST.md and are not replaced or pooled.

Each event surrounds one replay of a graph containing100 real consumer calls,
with20 events/2000 timed calls per case. Graph warmup topology differs from the
original scope. The separate six-worker baseline-only diagnosis is retained;
it compares timing methods and provides no A/P gain claim. Sources, compiled
libraries, inputs, seeds and correctness budgets remain unchanged. All twelve
separately built default kernels have identical full instructions/encodings.
Latencies below are microseconds.

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
| integer-8192 | first-thread/32 | 8 | 4.431 | 4.432 | 0.9998× | 0.9979–1.0019 | within default bound |
| integer-8192 | first-thread/128 | 8 | 4.546 | 4.551 | 0.9995× | 0.9985–1.0006 | within default bound |
| integer-8192 | first-thread/256 | 8 | 7.319 | 7.323 | 0.9992× | 0.9959–1.0025 | within default bound |
| integer-8192 | first-thread/512 | 8 | 14.728 | 14.721 | 1.0003× | 0.9997–1.0009 | within default bound |
| integer-8192 | normalization/32 | 8 | 4.579 | 4.593 | 0.9966× | 0.9949–0.9984 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 5.775 | 5.515 | 1.0467× | 1.0448–1.0483 | stable >1% gain |
| integer-8192 | normalization/256 | 8 | 11.129 | 10.357 | 1.0747× | 1.0743–1.0751 | stable >1% gain |
| integer-8192 | normalization/512 | 8 | 22.892 | 21.187 | 1.0806× | 1.0799–1.0812 | stable >1% gain |
| legacy-8192 | first-thread/32 | 8 | 4.464 | 4.469 | 0.9997× | 0.9981–1.0020 | within default bound |
| legacy-8192 | first-thread/128 | 8 | 6.152 | 6.160 | 0.9984× | 0.9974–0.9992 | within default bound |
| legacy-8192 | first-thread/256 | 8 | 10.707 | 10.707 | 1.0003× | 0.9997–1.0009 | within default bound |
| legacy-8192 | first-thread/512 | 8 | 21.847 | 21.844 | 1.0003× | 0.9999–1.0010 | within default bound |
| legacy-8192 | normalization/32 | 8 | 4.696 | 4.681 | 1.0044× | 1.0025–1.0064 | no stable >1% gain |
| legacy-8192 | normalization/128 | 8 | 7.283 | 6.788 | 1.0724× | 1.0718–1.0730 | stable >1% gain |
| legacy-8192 | normalization/256 | 8 | 13.318 | 13.318 | 1.0000× | 0.9998–1.0004 | no stable >1% gain |
| legacy-8192 | normalization/512 | 8 | 27.697 | 27.709 | 0.9996× | 0.9989–1.0002 | no stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 4.461 | 4.463 | 0.9997× | 0.9988–1.0005 | within default bound |
| retained-8192 | first-thread/128 | 8 | 6.150 | 6.149 | 1.0013× | 0.9987–1.0050 | within default bound |
| retained-8192 | first-thread/256 | 8 | 10.705 | 10.704 | 1.0007× | 0.9996–1.0022 | within default bound |
| retained-8192 | first-thread/512 | 8 | 21.752 | 21.748 | 1.0008× | 1.0001–1.0018 | within default bound |
| retained-8192 | normalization/32 | 8 | 4.687 | 4.636 | 1.0115× | 1.0082–1.0141 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 7.276 | 6.778 | 1.0733× | 1.0724–1.0743 | stable >1% gain |
| retained-8192 | normalization/256 | 8 | 13.310 | 13.313 | 0.9998× | 0.9994–1.0002 | no stable >1% gain |
| retained-8192 | normalization/512 | 8 | 27.606 | 27.593 | 1.0001× | 0.9996–1.0006 | no stable >1% gain |


[Raw index](results/sm120_capture100/RAW_INDEX.json) includes every actual worker
output/log, status, frozen manifest and exact measured harness-source snapshot.
Required repository formatting preserves changed original text and its SHA.
