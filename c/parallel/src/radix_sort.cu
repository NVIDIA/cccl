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

#include "kernels/operators.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/radix_sort.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
struct agent_radix_sort_downsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;
};

struct agent_radix_sort_upsweep_policy
{
  int items_per_thread;
  int block_threads;
  int radix_bits;
};

struct agent_radix_sort_onesweep_policy
{
  int block_threads;
  int items_per_thread;
  int rank_num_parts;
  int radix_bits;
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
  int block_threads;
  int items_per_thread;
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

std::pair<int, int>
reg_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  int items_per_thread = std::max(1, nominal_4_byte_items_per_thread * 4 / std::max(4, key_size));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(cub::detail::max_smem_per_block / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

std::pair<int, int>
mem_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  int items_per_thread =
    std::max(1, std::min(nominal_4_byte_items_per_thread * 4 / key_size, nominal_4_byte_items_per_thread * 2));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(cub::detail::max_smem_per_block / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

radix_sort_runtime_tuning_policy get_policy(int cc, int key_size)
{
  constexpr int onesweep_radix_bits                                        = 8;
  const int primary_radix_bits                                             = (key_size > 1) ? 7 : 5;
  const int single_tile_radix_bits                                         = (key_size > 1) ? 6 : 5;
  const auto [onesweep_items_per_thread, onesweep_block_threads]           = reg_bound_scaling(256, 21);
  const auto [scan_items_per_thread, scan_block_threads]                   = mem_bound_scaling(512, 13);
  const auto [downsweep_items_per_thread, downsweep_block_threads]         = mem_bound_scaling(160, 39);
  const auto [alt_downsweep_items_per_thread, alt_downsweep_block_threads] = mem_bound_scaling(256, 16);
  const auto [single_tile_items_per_thread, single_tile_block_threads]     = mem_bound_scaling(256, 19);

  return
  {
    {256, 8, std::max(1, 1 * 4 / std::max(key_size, 4)), onesweep_radix_bits}, {256, onesweep_radix_bits},
      {onesweep_items_per_thread, onesweep_block_threads, 1, onesweep_radix_bits},
      {scan_block_threads, scan_items_per_thread},
      {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
      {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
      {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
      {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
    {
      single_tile_block_threads, single_tile_items_per_thread, single_tile_radix_bits
    }
  }
};
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

    const int cc             = cc_major * 10 + cc_minor;
    const auto policy        = radix_sort::get_policy(cc, key_t.size);
    const auto key_cpp       = cccl_type_enum_to_name(key_t.type);
    const std::string op_src = make_kernel_user_binary_operator(key_cpp, decomposer);

    constexpr std::string_view src_template = R"XXX(
#include <cub/device/dispatch/kernels/radix_sort.cuh>
struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) values_storage_t {{
  char data[{2}];
}};
struct agent_histogram_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {4};
  static constexpr int BLOCK_THREADS = {5};
  static constexpr int RADIX_BITS = {6};
  static constexpr int NUM_PARTS = {7};
}};
struct agent_exclusive_sum_policy_t {{
  static constexpr int BLOCK_THREADS = {8};
  static constexpr int RADIX_BITS = {9};
}};
struct agent_onesweep_policy_t {{
  static constexpr int BLOCK_THREADS = {10};
  static constexpr int RANK_NUM_PARTS = {11};
  static constexpr int RADIX_BITS = {12};
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM       = cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM       = cub::BLOCK_SCAN_WARP_SCANS;
  static constexpr cub::RadixSortStoreAlgorithm STORE_ALGORITHM = cub::RADIX_SORT_STORE_DIRECT;
}};
struct agent_scan_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {13};
  static constexpr int BLOCK_THREADS = {14};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = cub::LOAD_DEFAULT;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM   = cub::BLOCK_SCAN_RAKING_MEMOIZE;
  struct detail
  {
    using delay_constructor_t = detail::default_delay_constructor_t<{15}>;
  };
}};
struct agent_downsweep_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {16};
  static constexpr int BLOCK_THREADS = {17};
  static constexpr int RADIX_BITS = {18};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_BASIC;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
}};
struct agent_alt_downsweep_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {19};
  static constexpr int BLOCK_THREADS = {20};
  static constexpr int RADIX_BITS = {21};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_RAKING_MEMOIZE;
}};
struct agent_single_tile_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {22};
  static constexpr int BLOCK_THREADS = {23};
  static constexpr int RADIX_BITS = {24};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
}};
struct device_scan_policy {{
  struct ActivePolicy {{
    using HistogramPolicy = agent_histogram_policy_t;
    using ExclusiveSumPolicy = agent_exclusive_sum_policy_t;
    using OnesweepPolicy = agent_onesweep_policy_t;
    using ScanPolicy = agent_scan_policy_t;
    using DownsweepPolicy = agent_downsweep_policy_t;
    using AltDownsweepPolicy = agent_alt_downsweep_policy_t;
    using UpsweepPolicy = agent_downsweep_policy_t;
    using AltUpsweepPolicy = agent_alt_downsweep_policy_t;
    using SingleTilePolicy = agent_single_tile_policy_t;
  }};
}};
)XXX";

    const std::string src = std::format(
      src_template,
      key_t.size, // 0
      key_t.alignment, // 1
      value_t.size, // 2
      value_t.alignment, // 3
      policy.histogram.items_per_thread, // 4
      policy.histogram.block_threads, // 5
      policy.histogram.radix_bits, // 6
      policy.histogram.num_parts, // 7
      policy.exclusive_sum.block_threads, // 8
      policy.exclusive_sum.radix_bits, // 9
      policy.onesweep.block_threads, // 10
      policy.onesweep.rank_num_parts, // 11
      policy.onesweep.radix_bits, // 12
      policy.onesweep.items_per_thread, // 13
      policy.onesweep.block_threads, // 14
      key_cpp, // 15
      policy.downsweep.items_per_thread, // 16
      policy.downsweep.block_threads, // 17
      policy.downsweep.radix_bits, // 18
      policy.alt_downsweep.items_per_thread, // 19
      policy.alt_downsweep.block_threads, // 20
      policy.alt_downsweep.radix_bits, // 21
      policy.single_tile.items_per_thread, // 22
      policy.single_tile.block_threads, // 23
      policy.single_tile.radix_bits // 24
    );
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
