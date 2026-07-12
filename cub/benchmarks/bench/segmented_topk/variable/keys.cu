// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Baseline-backend variable-segment keys-only top-k benchmark (the plain/default keys benchmark). A base build runs the
// library's automatic selector; a tuning variant forces the baseline backend and sweeps its worker knobs. The baseline
// backend has no determinism / tie-break, so this build stays non-deterministic (TUNE_REQUIREMENT 0). The cluster
// backend is tuned separately in `keys.cluster.cu`; the `device` reference is reachable via -DTUNE_BACKEND=2.

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

#ifndef TUNE_BACKEND
#  if TUNE_BASE
#    define TUNE_BACKEND 3 // automatic: the library's production selector, for base/benchmark builds
#  else
#    define TUNE_BACKEND 0 // baseline: the backend this file tunes
#  endif
#endif

#ifndef TUNE_REQUIREMENT
#  define TUNE_REQUIREMENT 0 // baseline backend: non-deterministic only (a non-zero override trips the static_assert)
#endif

#include "keys_common.cuh"
