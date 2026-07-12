// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Cluster-backend variable-segment keys-only top-k benchmark. A base build runs the library's production (automatic)
// selector; a tuning variant forces the cluster backend and sweeps its per-block knobs. This build stays on the
// non-deterministic path (TUNE_REQUIREMENT 0) -- the determinism / tie-break requirement sweep lives in the indexed
// benchmark (`indexed.cluster.cu`). The baseline backend is benchmarked separately in the plain `keys.cu`.

// %RANGE% TUNE_CLUSTER_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_CLUSTER_MIN_BLOCKS_PER_SM mbs 1:2:1
// %RANGE% TUNE_CLUSTER_MIN_CHUNKS_PER_BLOCK mcb 1:2:1
// %RANGE% TUNE_CLUSTER_CHUNK_KIB ckib 8:32:1
// %RANGE% TUNE_CLUSTER_LOAD_ALIGN_BYTES_POW2 la 4:7:1
// %RANGE% TUNE_CLUSTER_PIPELINE_STAGES ps 2:16:1
// %RANGE% TUNE_CLUSTER_BITS_PER_PASS bpp 8:11:1
// %RANGE% TUNE_CLUSTER_HIST_IPT hipt 1:24:1
// %RANGE% TUNE_CLUSTER_TIEBREAK_IPT tipt 1:24:1
// %RANGE% TUNE_CLUSTER_COPY_IPT cipt 1:24:1

#ifndef TUNE_BACKEND
#  if TUNE_BASE
#    define TUNE_BACKEND 3 // automatic: the library's production selector, for base/benchmark builds
#  else
#    define TUNE_BACKEND 1 // cluster: the backend this file tunes
#  endif
#endif

#ifndef TUNE_REQUIREMENT
#  define TUNE_REQUIREMENT 0 // cluster tuned non-deterministic; requirement sweep lives in indexed.cluster.cu
#endif

#include "keys_common.cuh"
