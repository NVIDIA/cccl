//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_driver.cuh>
#include <cub/device/device_radix_sort.cuh>

#include <cccl/c/radix_sort.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
struct agent_radix_sort_downsweep_policy
{
  int items_per_thread;
  int block_threads;
  int radix_bits;
  cub::BlockLoadAlgorithm load_algorithm;
  cub::CacheLoadModifier load_modifier;
  cub::RadixRankAlgorithm rank_algorithm;
  cub::BlockScanAlgorithm scan_algorithm;
};

struct agent_radix_sort_upsweep_policy
{
  int items_per_thread;
  int block_threads;
  int radix_bits;
  cub::CacheLoadModifier load_modifier;
};

struct agent_radix_sort_onesweep_policy
{
  int items_per_thread;
  int block_threads;
  int rank_num_parts;
  int radix_bits;
  cub::RadixRankAlgorithm rank_algorithm;
  cub::BlockScanAlgorithm scan_algorithm;
  cub::RadixSortStoreAlgorithm store_algorithm;
};

struct agent_radix_sort_histogram_policy
{
  int block_threads;
  int items_per_thread;
  int num_parts;
  int radix_bits;
};

struct agent_radix_sort_exclusive_sum_policy
{
  int block_threads;
  int radix_bits;
};

struct agent_scan_policy
{
  int items_per_thread;
  int block_threads;
  cub::BlockLoadAlgorithm load_algorithm;
  cub::CacheLoadModifier load_modifier;
  cub::BlockStoreAlgorithm store_algorithm;
  cub::BlockScanAlgorithm scan_algorithm;
};

struct radix_sort_runtime_tuning_policy
{
  agent_radix_sort_histogram_policy histogram;
  agent_radix_sort_exclusive_sum_policy exclusive_sum;
  agent_radix_sort_onesweep_policy onesweep;
  agent_scan_policy scan;
  agent_radix_sort_downsweep_policy downsweep;
  agent_radix_sort_downsweep_policy alt_downsweep;
  agent_radix_sort_upsweep_policy upsweep;
  agent_radix_sort_upsweep_policy alt_upsweep;
  agent_radix_sort_downsweep_policy single_tile;

  agent_radix_sort_histogram_policy Histogram() const
  {
    return histogram;
  }

  agent_radix_sort_exclusive_sum_policy ExclusiveSum() const
  {
    return exclusive_sum;
  }

  agent_radix_sort_onesweep_policy Onesweep() const
  {
    return onesweep;
  }

  agent_scan_policy Scan() const
  {
    return scan;
  }

  agent_radix_sort_downsweep_policy Downsweep() const
  {
    return downsweep;
  }

  agent_radix_sort_downsweep_policy AltDownsweep() const
  {
    return alt_downsweep;
  }

  agent_radix_sort_upsweep_policy Upsweep() const
  {
    return upsweep;
  }

  agent_radix_sort_upsweep_policy AltUpsweep() const
  {
    return alt_upsweep;
  }

  agent_radix_sort_downsweep_policy SingleTile() const
  {
    return single_tile;
  }
};

radix_sort_runtime_tuning_policy get_policy(int cc, int key_size)
{
  return {{
    256,
  }};
}

} // namespace radix_sort

CCCL_C_API CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_type_info key_t,
  cccl_type_info value_t,
  cccl_op_t decomposer,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;
  try
  {
    const char* name = "test";

    const int cc      = cc_major * 10 + cc_minor;
    const auto policy = radix_sort::get_policy(cc, output_keys_it.value_type.size);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CCCL_C_API CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_double_buffer_t d_keys,
  cccl_double_buffer_t d_values,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  CUstream stream);

CCCL_C_API CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* bld_ptr);
