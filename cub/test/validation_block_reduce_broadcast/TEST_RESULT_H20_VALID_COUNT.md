<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# H20 follow-up: oversized valid counts

`ReduceBroadcast` documents that `num_valid >= BLOCK_THREADS` includes every physical
thread. On a partial final warp, forwarding the oversized count to the internal
reduction incorrectly included nonexistent lanes. The full-tile path now passes
`BLOCK_THREADS`. The default `Reduce` and `Sum` paths retain their implementation.

An associative, noncommutative affine-composition fixture checks every input's
order modulo 2^32. It exposed the defect at 65 and 260 threads with `N+3` valid
items. The default comparison uses a count within its physical block size;
every broadcast lane must exactly equal that result.

| Check on H20 SM90 / CUDA 12.8 | Result |
| :--- | :--- |
| Targeted CTest | 29/29 passed; all workers exited naturally |
| Existing Reduce/Sum | 1,200 cases, 50,400 assertions |
| Broadcast, including affine composition | 96 cases, 11,904 assertions |
| Unsupported algorithms | Three expected compilation failures passed |
| Partial-warp memcheck / racecheck / synccheck | Both 65-thread and 260-thread configurations passed; zero errors, hazards or warnings |
| Original complete normalization consumer | All 12 CUDA kernels have identical instructions and encodings before/after this correction |
| Repository checks | Applicable pre-commit hooks passed with the pinned tools |

The [existing speed table and full-checkpoint E2E](TEST_RESULT_H20_BROADCAST.md)
remain measurements of their recorded source and binaries. This correction does
not supply a new speed claim: the measured normalization calls already use the
physical block size, and their CUDA instructions are unchanged. Further broadcast
performance candidates are screened separately; stable model speedup remains
unproven.

[CTest, sanitizer, source-hash and binary comparison records](results/valid_count_followup/public-clamp-validation.json)
and the accompanying raw logs are retained in `results/valid_count_followup/`.
Published log copies normalize trailing whitespace for repository checks; original
log hashes and the unmodified private logs remain preserved.
