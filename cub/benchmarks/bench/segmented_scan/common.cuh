// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_scan.cuh>

#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_BLOCK_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_BLOCK_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_BLOCK_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_BLOCK_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <int ThreadsPerBlock, int ItemsPerThread, int MaxSegmentsPerBlock>
struct policy_selector_t
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const -> cub::SegmentedScanPolicy
  {
    return cub::SegmentedScanPolicy{cub::SegmentedScanBlockPolicy{
      ThreadsPerBlock,
      ItemsPerThread,
      TUNE_BLOCK_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_BLOCK_STORE_ALGORITHM,
      cub::BLOCK_SCAN_WARP_SCANS,
      MaxSegmentsPerBlock}};
  }
};
#endif // TUNE_BASE

#if (_CCCL_CUDA_COMPILER(NVCC, >=, 12, 1))
using benched_value_types = all_types;
#else
// WAR for excessive time CTK 12.0 CICC takes to compile these benchmarks for int128_t
#  ifdef TUNE_T
static_assert(!cuda::std::is_integral_v<TUNE_T> || sizeof(TUNE_T) < 16);
using benched_value_types = nvbench::type_list<TUNE_T>;
#  else
using benched_value_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t, float, double, complex32>;
#  endif
#endif
