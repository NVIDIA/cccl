// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <nvbench_helper.cuh>

using value_types       = nvbench::type_list<int32_t, int64_t, float, double>;
using op_t              = cub::detail::arg_max;
using some_offset_types = nvbench::type_list<int32_t>;

// %RANGE% TUNE_S_ITEMS ipsw 1:16:1
// %RANGE% TUNE_SW_THREADS_POW2 tpsw 1:4:1
// %RANGE% TUNE_S_LOAD sld 0:2:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

#if !TUNE_BASE
#  define TUNE_L_ITEMS 16
#  define TUNE_M_ITEMS 16

#  define TUNE_MW_THREADS_POW2 (TUNE_SW_THREADS_POW2 + 1)

#  define TUNE_SW_THREADS (1 << TUNE_SW_THREADS_POW2)
#  define TUNE_MW_THREADS (1 << TUNE_MW_THREADS_POW2)

#  define SMALL_SEGMENT_SIZE  TUNE_S_ITEMS* TUNE_SW_THREADS
#  define MEDIUM_SEGMENT_SIZE TUNE_M_ITEMS* TUNE_MW_THREADS
#  define LARGE_SEGMENT_SIZE  TUNE_L_ITEMS* TUNE_THREADS

#  if (LARGE_SEGMENT_SIZE <= SMALL_SEGMENT_SIZE) || (LARGE_SEGMENT_SIZE <= MEDIUM_SEGMENT_SIZE)
#    error Large segment size must be larger than small and medium segment sizes
#  endif

#  if (MEDIUM_SEGMENT_SIZE <= SMALL_SEGMENT_SIZE)
#    error Medium segment size must be larger than small one
#  endif

#  if TUNE_S_LOAD == 0
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_S_LOAD == 1
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_LDG
#  else // TUNE_S_LOAD == 2
#    define TUNE_S_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_S_LOAD
#endif

#include "variable_base.cuh"
