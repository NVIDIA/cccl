<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# H20 wide-broadcast iteration

The wider generic broadcast now folds warp aggregates once in thread 0 and publishes
through the existing shared prefix. Supported integral operators use full-warp reduction
with identity-padded aggregates. Blocks of at most four warps retain the earlier ordered
fold. Floating-point and custom operators preserve the original fold order; default
Reduce/Sum code and storage size are unchanged.

At 8,192 rows, retained-exponential normalization improves 1.0438× at width 128,
1.0137× at width 256 and 1.0161× at width 512. Integer count normalization improves
1.0640×, 1.0375× and 1.0523× respectively. All six intervals exceed the predeclared
1% useful-gain threshold. Neither width-32 workload establishes that gain. These are
complete consumer measurements; full-model speedup remains unproven.

## Source and validation

Candidate source: `76db7e3f3ab0d665fc87282b0160a321e51e698e`.
Control: [`0909f9ee05c632af18fba74a366b657370a12aee`](https://github.com/0z5a/cccl/tree/validation/h20-broadcast-control), the same review-tree control
with the original upstream warp-reduction header. Both builds use clean source trees,
identical consumer sources/flags, SM90, CUDA 12.8.93 and Torch 2.7.0+cu128. Measurements
use H20 GPU 1, UUID `GPU-8ee84e7d-143f-dd29-1097-85943783e027`, driver 580.105.08.

| Validation | Result |
| :--- | :--- |
| Targeted CTest | 29/29 passed |
| Existing Reduce/Sum | 1,200 cases; 50,400 assertions |
| Broadcast | 168 cases; 14,208 assertions |
| Unsupported algorithms | Three expected compilation failures |
| memcheck / racecheck / synccheck | 65/512/1024-thread configurations: zero errors, hazards or warnings |
| Independent default-kernel comparison | All 26 CUDA kernels have identical instructions and encodings: 8 original, 10 retained, 8 integer |
| Retained normalization | 22 ragged/masked fixtures per worker against independent FP64 normalization |
| Integer normalization | 22 ragged/zero-padded count fixtures per worker; exact outputs |
| Pinned source pre-commit hooks | Passed |

Every lane is compared exactly to the original GPU Reduce result. Coverage includes
partial final warps, oversized valid counts, three-dimensional blocks, floating/custom
values, associative noncommutative affine composition, and uint32 plus/min/max/and/or/xor
with high-bit inputs and identity padding. Invalid counts beyond the physical block are
clamped. Sanitizer child processes may exit between the NVML query and the process-tree
snapshot; the three raw functional flags and their classification remain recorded.
No timing quartet receives an exception to the contamination rule.

## Independent formal measurements

Each workload uses eight fresh-process APPA/PAAP quartets, balanced by alternating
order. Each worker contributes the median of all 20 CUDA-event batches, with 100 complete
consumer replays per batch after warmup. Each quartet ratio divides its two A medians'
geometric mean by its two P medians' geometric mean. The estimate is the geometric mean
of eight quartet ratios; the two-sided 95% interval resamples whole log-ratio quartets
10,000 times with seed 11504. The useful-gain threshold is 1%; the default-path regression
bound is 1%. Screening measurements are excluded.

All 64 workers exited naturally with code zero. Before/after GPU snapshots and 0.5-second
observations show no foreign process on the selected GPU. Those observations are sampled,
not a claim of continuous detection. The audit checks source/diff/library/harness/common
hashes and worker PID against the frozen manifest; all input and output hashes match
within each quartet. Every scheduled width and default control appears below.

Latency columns are medians of all process medians (µs for consumers, ms for models).

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-8192 | first-thread/32 | 8 | 7.199 | 7.149 | 1.0040× | 1.0005–1.0074 | within default bound |
| integer-8192 | first-thread/128 | 8 | 7.019 | 6.961 | 1.0050× | 1.0021–1.0081 | within default bound |
| integer-8192 | first-thread/256 | 8 | 10.565 | 10.514 | 1.0025× | 1.0004–1.0046 | within default bound |
| integer-8192 | first-thread/512 | 8 | 20.042 | 19.998 | 1.0011× | 0.9999–1.0024 | within default bound |
| integer-8192 | normalization/32 | 8 | 7.355 | 7.302 | 1.0012× | 0.9982–1.0046 | no stable >1% gain |
| integer-8192 | normalization/128 | 8 | 9.052 | 8.478 | 1.0640× | 1.0615–1.0669 | stable >1% gain |
| integer-8192 | normalization/256 | 8 | 15.829 | 15.247 | 1.0375× | 1.0357–1.0395 | stable >1% gain |
| integer-8192 | normalization/512 | 8 | 31.316 | 29.759 | 1.0523× | 1.0515–1.0530 | stable >1% gain |
| retained-8192 | first-thread/32 | 8 | 7.498 | 7.480 | 0.9993× | 0.9934–1.0051 | within default bound |
| retained-8192 | first-thread/128 | 8 | 7.474 | 7.480 | 0.9986× | 0.9927–1.0042 | within default bound |
| retained-8192 | first-thread/256 | 8 | 13.038 | 13.039 | 1.0004× | 0.9972–1.0036 | within default bound |
| retained-8192 | first-thread/512 | 8 | 25.684 | 25.677 | 1.0007× | 0.9995–1.0019 | within default bound |
| retained-8192 | normalization/32 | 8 | 7.140 | 7.081 | 1.0091× | 1.0032–1.0150 | no stable >1% gain |
| retained-8192 | normalization/128 | 8 | 9.042 | 8.658 | 1.0438× | 1.0394–1.0486 | stable >1% gain |
| retained-8192 | normalization/256 | 8 | 16.714 | 16.497 | 1.0137× | 1.0112–1.0164 | stable >1% gain |
| retained-8192 | normalization/512 | 8 | 34.926 | 34.388 | 1.0161× | 1.0147–1.0175 | stable >1% gain |

The retained consumer stores exponentials in registers and uses __expf identically in
both arms. The original consumer rereads inputs and uses expf in both arms. Such common
consumer differences are excluded from the PR speedup. Integer count normalization
consumes bounded int32 counts and writes FP32 probabilities; it is a separate workload,
not a replacement for Qwen attention. SASS equality confirms that apparent small default
control differences are measurement variation rather than a changed default algorithm.

## Complete-model gate and follow-up

The retained-exp model baseline fails native generation. Original expf screening
passes batch 1 and fails batch 16 before P; these failures remain recorded. The
faithful common consumer following native Torch order now passes complete batch1
and batch16 resident APPA/PAAP screens, with exact native 32-step tokens/logits
and stable weights at both fixed lengths. Independent model quartets remain pending.
The [original-consumer/model report](TEST_RESULT_H20_ORIGINAL_MODEL.md) records
those gates and the unchanged regular-exp microbenchmarks: width128 improves
1.0271×/1.0251× at 8,192/65,536 rows. The original 8,192-row width32 default
interval does not establish the 1% bound; a single separate validation is predeclared.

The [65,536-row follow-up](TEST_RESULT_H20_65536_BROADCAST.md) records four further
stable consumer gains, all default controls, nine invalidated quartets and nine
clean replacements. Kernel results do not establish model speedup. Task-owned
weights will be cleaned after required model workers naturally exit.

## Evidence and reproduction

[Raw worker records, samples and observations](results/wide_broadcast_iteration/formal-v2-primary-plan.status.json),
[the frozen manifest](results/wide_broadcast_iteration/formal-v2-manifest.json),
[the audited estimates](results/wide_broadcast_iteration/formal-v2-primary-audited-summary.json)
and [the evidence index](results/wide_broadcast_iteration/evidence-index.json) are committed.
The [source snapshots](results/wide_broadcast_iteration/harness-sources.json) preserve the
exact consumer and analysis inputs; formatting a snapshot would change its frozen hash.
Extract them into a fresh scratch directory, then use each consumer's build.py and
benchmark.py against the control and candidate source trees:

```python
import hashlib
import json
from pathlib import Path

snapshot = Path("cub/test/validation_block_reduce_broadcast/results/wide_broadcast_iteration/harness-sources.json")
for file in json.loads(snapshot.read_text())["files"]:
    content = file["content"].encode()
    assert hashlib.sha256(content).hexdigest() == file["sha256"]
    destination = Path("h20-reproduce") / file["relative_path"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
```

```sh
python h20-reproduce/retained/build.py --source CONTROL --variant baseline --build BUILD_A --output build-A.json
python h20-reproduce/retained/build.py --source CANDIDATE --variant patch --build BUILD_P --output build-P.json
CUDA_VISIBLE_DEVICES=1 python h20-reproduce/retained/benchmark.py --source CONTROL --variant baseline --build BUILD_A --rows 8192 --output A.json
CUDA_VISIBLE_DEVICES=1 python h20-reproduce/retained/benchmark.py --source CANDIDATE --variant patch --build BUILD_P --rows 8192 --output P.json
```

Use the integer subdirectory for the integer workload. The archived finite controller
plan retains the exact original commands and quartet order. Recompute the committed
primary estimate from the raw records:

```sh
python h20-reproduce/summarize_formal_iteration.py --status cub/test/validation_block_reduce_broadcast/results/wide_broadcast_iteration/formal-v2-primary-plan.status.json --manifest cub/test/validation_block_reduce_broadcast/results/wide_broadcast_iteration/formal-v2-manifest.json --output cub/test/validation_block_reduce_broadcast/results/wide_broadcast_iteration/recomputed
```
