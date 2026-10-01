<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# H20 validation: opt-in BlockReduce broadcast

`ReduceBroadcast` returns the aggregate in every thread for `BLOCK_REDUCE_WARP_REDUCTIONS`. `Reduce` and `Sum` retain their thread-0 contract and original folding path. The broadcast uses the existing warp-aggregate storage and the same reduction order.

The 128-wide normalization consumer measures 1.0251–1.0297×, while the 512-wide consumer measures 0.9022–0.9132×. The unchanged first-thread consumer has identical SASS. Full-model ratios are 0.9900× and 1.0002×, with quartet ranges crossing 1×; these runs demonstrate no stable model speedup. The broadcast cost is explicitly opt-in.

The [oversized valid-count follow-up](TEST_RESULT_H20_VALID_COUNT.md) adds a partial-warp correction and stronger noncommutative coverage. The original timing records below retain their recorded source; all 12 measured consumer kernels are unchanged by that correction.

## Source and machine

- Patch source: `e3af129906d20c3d3317668e9aa30e2fe4a1217c`; control source: `0909f9ee05c632af18fba74a366b657370a12aee`.
- The control restores upstream header blob `6872acdff4464ae7b3d99d071443ed4a8ec76415` in the review-head tree. Both arms use the same remaining CCCL dependencies and identical consumer sources.
- NVIDIA H20 SM90, GPU 0 `GPU-edf64e5f-21ef-a1f2-6601-e8620b5664ff`; driver 580.105.08; CUDA/NVCC 12.8.93; GCC 11; Torch 2.7.0+cu128. GPU 1 was reserved by an existing model worker.
- Existing environments were retained. CMake 3.30.4, Catch2 3.12.0, NVTX `df7511fb9aa336df1d13d60d740e37ed1909b279`, and formatting tools used private task directories.
- Consumer builds use `-O3 -std=c++17`, SM90 and line information, with no fast-math flag. The selected CUB/libcu++/Thrust headers precede toolkit headers.

## Correctness and default path

| Check | Result |
| :--- | :--- |
| Targeted CTest entries | 29/29 passed |
| Existing Reduce/Sum tests | 1,200 cases, 50,400 assertions |
| New broadcast tests | 84 cases, 2,880 GPU fixtures, 11,520 assertions |
| Unsupported algorithms | Three expected compilation failures: raking, commutative raking, nondeterministic warp reductions |
| Compute Sanitizer | memcheck, racecheck and synccheck: zero errors; racecheck: zero hazards/warnings |
| Default consumers | All eight SM90 kernels have identical instructions and encodings to the upstream-header control |
| Ragged/masked normalization | 22 fixtures per process; FP64 reference, rtol 2e-6 and atol 2e-7; baseline/patch output hashes match |
| Full checkpoint output | Exact prefill logits and all 32 generated tokens for both prompt lengths in every formal process |
| Formatting and type checks | Repository pre-commit hooks passed, including pinned secret scan, clang-format, Ruff, mypy and CUB classification; shellcheck had no applicable files |

Broadcast tests cover X dimensions 1/7/32/65/128/256, Y/Z dimensions 1 or 2, full/partial tiles, valid counts 1/7/31/33/N−1/N/N+3 (bounded where needed), int32/float/double, custom values, plus/max and an associative noncommutative operator. Every lane must exactly equal the original GPU `Reduce` result.

A repeated build exposed a cancellation-sensitive sequential CPU reference with random signed floating inputs (seed 2757115478; CPU 1.018465996 versus GPU 1.018451214). The numeric floating fixture now uses positive random inputs while retaining the exact GPU comparison. The original failing seed and the complete suite pass; signed masked consumer inputs remain covered.

## Timing protocol

Each workload runs 16 fresh processes on GPU 0 in four quartets: APPA, PAAP, APPA, PAAP. A is the upstream-header control; P is the patch. All 48 formal processes exited naturally with code zero. Every worker records PID, source and loaded-library hashes, input/output hashes, and GPU state before/after.

For each quartet, the ratio is the geometric mean of its two A process medians divided by the geometric mean of its two P process medians. Reported speedup is the geometric mean of the four quartet ratios; the range retains all four. Latency columns are medians of the eight process medians per arm. Values above 1× indicate a faster patch. These four quartets are descriptive measurements, with no statistical significance claim.

The normalization consumer performs max reduction, exponential sum reduction, normalization and output writes for a complete row per CTA. Both arms evaluate `expf` in the sum and output loops to support arbitrary widths. The control publishes thread-0 max/sum through shared memory and a barrier; P uses explicit broadcasts. The default first-thread consumer performs sum/max and writes only from thread 0. Five side-stream warmups precede capture, then five graph warmups precede 20 CUDA-event batches of 100 complete-kernel replays. All 20 samples are included; resident input/output addresses stay fixed.

## Complete normalization consumer

| Rows | Width / CTA threads | A (µs) | P (µs) | Speedup | Quartet range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 32 | 7.319 | 7.263 | 1.0062× | 0.9991–1.0132× |
| 8,192 | 128 | 14.382 | 13.909 | 1.0297× | 1.0259–1.0380× |
| 8,192 | 256 | 26.732 | 27.000 | 0.9889× | 0.9862–0.9919× |
| 8,192 | 512 | 53.841 | 59.652 | 0.9022× | 0.9014–0.9032× |
| 65,536 | 32 | 45.441 | 45.186 | 1.0057× | 1.0031–1.0074× |
| 65,536 | 128 | 102.158 | 99.683 | 1.0251× | 1.0248–1.0255× |
| 65,536 | 256 | 204.508 | 207.155 | 0.9870× | 0.9867–0.9879× |
| 65,536 | 512 | 429.600 | 470.429 | 0.9132× | 0.9130–0.9135× |

## Unchanged first-thread consumer

| Rows | CTA threads | A (µs) | P (µs) | Speedup | Quartet range |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8,192 | 32 | 7.519 | 7.447 | 1.0036× | 0.9948–1.0135× |
| 8,192 | 128 | 7.519 | 7.452 | 1.0043× | 0.9979–1.0129× |
| 8,192 | 256 | 13.079 | 12.999 | 1.0008× | 0.9974–1.0089× |
| 8,192 | 512 | 25.694 | 25.640 | 1.0012× | 0.9991–1.0038× |
| 65,536 | 32 | 50.660 | 50.620 | 1.0003× | 0.9990–1.0013× |
| 65,536 | 128 | 45.525 | 45.491 | 1.0002× | 0.9979–1.0013× |
| 65,536 | 256 | 108.687 | 108.640 | 1.0000× | 0.9993–1.0004× |
| 65,536 | 512 | 233.680 | 233.626 | 1.0000× | 0.9997–1.0002× |

## Full Qwen3-0.6B checkpoint E2E

Pinned revision `c1899de289a04d12100db370d81485cdf75e47ca`; all seven checkpoint files passed SHA-256 verification. The complete 28-layer BF16 model retains eager attention and its native causal-mask construction, with only FP32 softmax replaced by the CUB consumer. The reference is the same full checkpoint using the upstream-header CUB integration.

Each process runs two warmup generations and five timed generations per prompt length, with a fresh KV cache and 32 greedy steps per generation. CUDA synchronization brackets the complete generation; weights remain resident at stable addresses. Each case records 160 normalization calls per layer: 4,480 actual normalization launches, containing 8,960 CUB reductions. Every timed generation exactly matches the reference prefill logits and all output tokens.

| Input / output tokens | A (ms) | P (ms) | Speedup | Quartet range |
| ---: | ---: | ---: | ---: | ---: |
| 128 / 32 | 623.889 | 629.444 | 0.9900× | 0.9473–1.0301× |
| 512 / 32 | 646.566 | 637.591 | 1.0002× | 0.9780–1.0191× |

## Evidence and reproduction

[Every measured sample and process record](results/h20_measurements.json) and [build, checkpoint and test evidence hashes](results/h20_evidence_index.json) are committed alongside this report. The complete validation checkpoint was removed from local and remote task directories after all model workers exited (1,519,182,365 bytes per copy).

Build this same consumer against each source tree with the existing Torch/CUDA runtime:

```sh
HARNESS=cub/test/validation_block_reduce_broadcast
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 python "$HARNESS/build.py" \
  --source /path/to/control --variant baseline --build /path/to/build/control --output control-build.json
CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=4 python "$HARNESS/build.py" \
  --source /path/to/patch --variant patch --build /path/to/build/patch --output patch-build.json
CUDA_VISIBLE_DEVICES=0 python "$HARNESS/benchmark.py" --source /path/to/patch \
  --variant patch --build /path/to/build/patch --rows 8192 --output patch-normalization.json
CUDA_VISIBLE_DEVICES=0 python "$HARNESS/model_e2e.py" --source /path/to/control \
  --variant baseline --build /path/to/build/control --model /path/to/pinned/checkpoint \
  --write-reference --reference reference.pt --output control-model.json
CUDA_VISIBLE_DEVICES=0 python "$HARNESS/model_e2e.py" --source /path/to/patch \
  --variant patch --build /path/to/build/patch --model /path/to/pinned/checkpoint \
  --reference reference.pt --output patch-model.json
```

Repeat each command in fresh processes with the quartet orders above and both row counts. The first-thread control is included in every benchmark process; the same harness must be used for both builds.
