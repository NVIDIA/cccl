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

#include <format>
#include <vector>

#include "cccl/c/types.h"
#include "cub/util_type.cuh"
#include "kernels/operators.h"
#include "util/context.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/radix_sort.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

using OffsetT = unsigned long long;
static_assert(std::is_same_v<cub::detail::choose_offset_t<OffsetT>, OffsetT>, "OffsetT must be unsigned long long");

namespace radix_sort
{
struct agent_radix_sort_downsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_upsweep_policy
{
  int block_threads;
  int items_per_thread;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_onesweep_policy
{
  int block_threads;
  int items_per_thread;
  int rank_num_parts;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
};

struct agent_radix_sort_histogram_policy
{
  int block_threads;
  int items_per_thread;
  int num_parts;
  int radix_bits;

  int BlockThreads() const
  {
    return block_threads;
  }
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

  int BlockThreads() const
  {
    return block_threads;
  }

  int ItemsPerThread() const
  {
    return items_per_thread;
  }
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
  bool is_onesweep;

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

  bool IsOnesweep() const
  {
    return is_onesweep;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int RadixBits(PolicyT policy)
  {
    return policy.radix_bits;
  }

  template <typename PolicyT>
  CUB_RUNTIME_FUNCTION static constexpr int BlockThreads(PolicyT policy)
  {
    return policy.block_threads;
  }
};

std::pair<int, int>
reg_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  assert(key_size > 0);
  int items_per_thread = std::max(1, nominal_4_byte_items_per_thread * 4 / std::max(4, key_size));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(int{cub::detail::max_smem_per_block} / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

std::pair<int, int>
mem_bound_scaling(int nominal_4_byte_block_threads, int nominal_4_byte_items_per_thread, int key_size)
{
  assert(key_size > 0);
  int items_per_thread =
    std::max(1, std::min(nominal_4_byte_items_per_thread * 4 / key_size, nominal_4_byte_items_per_thread * 2));
  int block_threads =
    std::min(nominal_4_byte_block_threads,
             cuda::ceil_div(int{cub::detail::max_smem_per_block} / (key_size * items_per_thread), 32) * 32);

  return {items_per_thread, block_threads};
}

radix_sort_runtime_tuning_policy get_policy(int /*cc*/, int key_size)
{
  // TODO: we hardcode some of these values in order to make sure that the radix_sort tests do not fail due to the
  // memory op assertions. This will be fixed after https://github.com/NVIDIA/cccl/issues/3570 is resolved.
  constexpr int onesweep_radix_bits = 8;
  const int primary_radix_bits      = (key_size > 1) ? 7 : 5;
  const int single_tile_radix_bits  = (key_size > 1) ? 6 : 5;

  const agent_radix_sort_histogram_policy histogram_policy{
    256, 8, std::max(1, 1 * 4 / std::max(key_size, 4)), onesweep_radix_bits};
  constexpr agent_radix_sort_exclusive_sum_policy exclusive_sum_policy{256, onesweep_radix_bits};

  const auto [onesweep_items_per_thread, onesweep_block_threads] = reg_bound_scaling(256, 21, key_size);
  // const auto [scan_items_per_thread, scan_block_threads]         = mem_bound_scaling(512, 23, key_size);
  const int scan_items_per_thread = 5;
  const int scan_block_threads    = 512;
  // const auto [downsweep_items_per_thread, downsweep_block_threads] = mem_bound_scaling(160, 39, key_size);
  const int downsweep_items_per_thread = 5;
  const int downsweep_block_threads    = 160;
  // const auto [alt_downsweep_items_per_thread, alt_downsweep_block_threads] = mem_bound_scaling(256, 16, key_size);
  const int alt_downsweep_items_per_thread                             = 5;
  const int alt_downsweep_block_threads                                = 256;
  const auto [single_tile_items_per_thread, single_tile_block_threads] = mem_bound_scaling(256, 19, key_size);

  constexpr bool is_onesweep = false;

  return {histogram_policy,
          exclusive_sum_policy,
          {onesweep_block_threads, onesweep_items_per_thread, 1, onesweep_radix_bits},
          {scan_block_threads, scan_items_per_thread},
          {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
          {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
          {downsweep_block_threads, downsweep_items_per_thread, primary_radix_bits},
          {alt_downsweep_block_threads, alt_downsweep_items_per_thread, primary_radix_bits - 1},
          {single_tile_block_threads, single_tile_items_per_thread, single_tile_radix_bits},
          is_onesweep};
};

std::string get_single_tile_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortSingleTileKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

std::string get_upsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortUpsweepKernel<{0}, {1}, {2}, {3}, {4}, {5}>",
    chained_policy_t,
    alt_digit_bits ? "true" : "false",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    offset_t,
    "op_wrapper");
}

std::string get_scan_bins_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  return std::format("cub::detail::radix_sort::RadixSortScanBinsKernel<{0}, {1}>", chained_policy_t, offset_t);
}

std::string get_downsweep_kernel_name(
  std::string_view chained_policy_t,
  bool alt_digit_bits,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortDownsweepKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    chained_policy_t,
    alt_digit_bits ? "true" : "false",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

std::string get_histogram_kernel_name(
  std::string_view chained_policy_t, cccl_sort_order_t sort_order, std::string_view key_t, std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortHistogramKernel<{0}, {1}, {2}, {3}, {4}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    offset_t,
    "op_wrapper");
}

std::string get_exclusive_sum_kernel_name(std::string_view chained_policy_t, std::string_view offset_t)
{
  return std::format("cub::detail::radix_sort::DeviceRadixSortExclusiveSumKernel<{0}, {1}>", chained_policy_t, offset_t);
}

std::string get_onesweep_kernel_name(
  std::string_view chained_policy_t,
  cccl_sort_order_t sort_order,
  std::string_view key_t,
  std::string_view value_t,
  std::string_view offset_t)
{
  return std::format(
    "cub::detail::radix_sort::DeviceRadixSortOnesweepKernel<{0}, {1}, {2}, {3}, {4}, int, int, {5}>",
    chained_policy_t,
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    key_t,
    value_t,
    offset_t,
    "op_wrapper");
}

template <auto* GetPolicy>
struct dynamic_radix_sort_policy_t
{
  using MaxPolicy = dynamic_radix_sort_policy_t;

  template <typename F>
  cudaError_t Invoke(int device_ptx_version, F& op)
  {
    return op.template Invoke<radix_sort_runtime_tuning_policy>(GetPolicy(device_ptx_version, key_size));
  }

  uint64_t key_size;
};

struct radix_sort_kernel_source
{
  cccl_device_radix_sort_build_result_t& build;

  CUkernel RadixSortSingleTileKernel() const
  {
    return build.single_tile_kernel;
  }

  CUkernel RadixSortUpsweepKernel() const
  {
    return build.upsweep_kernel;
  }

  CUkernel RadixSortAltUpsweepKernel() const
  {
    return build.alt_upsweep_kernel;
  }

  CUkernel DeviceRadixSortScanBinsKernel() const
  {
    return build.scan_bins_kernel;
  }

  CUkernel RadixSortDownsweepKernel() const
  {
    return build.downsweep_kernel;
  }

  CUkernel RadixSortAltDownsweepKernel() const
  {
    return build.alt_downsweep_kernel;
  }

  CUkernel RadixSortHistogramKernel() const
  {
    return build.histogram_kernel;
  }

  CUkernel RadixSortExclusiveSumKernel() const
  {
    return build.exclusive_sum_kernel;
  }

  CUkernel RadixSortOnesweepKernel() const
  {
    return build.onesweep_kernel;
  }

  std::size_t KeySize() const
  {
    return build.key_type.size;
  }

  std::size_t ValueSize() const
  {
    return build.value_type.size;
  }
};

} // namespace radix_sort

CUresult cccl_device_radix_sort_build_ex(
  cccl_device_radix_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult error = CUDA_SUCCESS;
  try
  {
    const char* name = "test";

    const int cc       = cc_major * 10 + cc_minor;
    const auto policy  = radix_sort::get_policy(cc, input_keys_it.value_type.size);
    const auto key_cpp = cccl_type_enum_to_name(input_keys_it.value_type.type);
    const auto value_cpp =
      input_values_it.type == cccl_iterator_kind_t::CCCL_POINTER && input_values_it.state == nullptr
        ? "cub::NullType"
        : cccl_type_enum_to_name(input_values_it.value_type.type);
    const std::string op_src =
      (decomposer.name == nullptr || (decomposer.name != nullptr && decomposer.name[0] == '\0'))
        ? "using op_wrapper = cub::detail::identity_decomposer_t;"
        : make_kernel_user_unary_operator(key_cpp, decomposer_return_type, decomposer);
    constexpr std::string_view chained_policy_t = "device_radix_sort_policy";

    constexpr std::string_view src_template = R"XXX(
#include <cub/device/dispatch/kernels/radix_sort.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>

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
  static constexpr int ITEMS_PER_THREAD = {10};
  static constexpr int BLOCK_THREADS = {11};
  static constexpr int RANK_NUM_PARTS = {12};
  static constexpr int RADIX_BITS = {13};
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM       = cub::RADIX_RANK_MATCH_EARLY_COUNTS_ANY;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM       = cub::BLOCK_SCAN_WARP_SCANS;
  static constexpr cub::RadixSortStoreAlgorithm STORE_ALGORITHM = cub::RADIX_SORT_STORE_DIRECT;
}};
struct agent_scan_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {14};
  static constexpr int BLOCK_THREADS = {15};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM   = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER     = cub::LOAD_DEFAULT;
  static constexpr cub::BlockStoreAlgorithm STORE_ALGORITHM = cub::BLOCK_STORE_WARP_TRANSPOSE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM   = cub::BLOCK_SCAN_RAKING_MEMOIZE;
  struct detail
  {{
    using delay_constructor_t = cub::detail::default_delay_constructor_t<{16}>;
  }};
}};
struct agent_downsweep_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {17};
  static constexpr int BLOCK_THREADS = {18};
  static constexpr int RADIX_BITS = {19};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_BASIC;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
}};
struct agent_alt_downsweep_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {20};
  static constexpr int BLOCK_THREADS = {21};
  static constexpr int RADIX_BITS = {22};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_RAKING_MEMOIZE;
}};
struct agent_single_tile_policy_t {{
  static constexpr int ITEMS_PER_THREAD = {23};
  static constexpr int BLOCK_THREADS = {24};
  static constexpr int RADIX_BITS = {25};
  static constexpr cub::BlockLoadAlgorithm LOAD_ALGORITHM = cub::BLOCK_LOAD_DIRECT;
  static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_LDG;
  static constexpr cub::RadixRankAlgorithm RANK_ALGORITHM = cub::RADIX_RANK_MEMOIZE;
  static constexpr cub::BlockScanAlgorithm SCAN_ALGORITHM = cub::BLOCK_SCAN_WARP_SCANS;
}};
struct {26} {{
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
{27};
)XXX";

    std::string offset_t;
    check(nvrtcGetTypeName<OffsetT>(&offset_t));

    const std::string src = std::format(
      src_template,
      input_keys_it.value_type.size, // 0
      input_keys_it.value_type.alignment, // 1
      input_values_it.value_type.size, // 2
      input_values_it.value_type.alignment, // 3
      policy.histogram.items_per_thread, // 4
      policy.histogram.block_threads, // 5
      policy.histogram.radix_bits, // 6
      policy.histogram.num_parts, // 7
      policy.exclusive_sum.block_threads, // 8
      policy.exclusive_sum.radix_bits, // 9
      policy.onesweep.items_per_thread, // 10
      policy.onesweep.block_threads, // 11
      policy.onesweep.rank_num_parts, // 12
      policy.onesweep.radix_bits, // 13
      policy.scan.items_per_thread, // 14
      policy.scan.block_threads, // 15
      offset_t, // 16
      policy.downsweep.items_per_thread, // 17
      policy.downsweep.block_threads, // 18
      policy.downsweep.radix_bits, // 19
      policy.alt_downsweep.items_per_thread, // 20
      policy.alt_downsweep.block_threads, // 21
      policy.alt_downsweep.radix_bits, // 22
      policy.single_tile.items_per_thread, // 23
      policy.single_tile.block_threads, // 24
      policy.single_tile.radix_bits, // 25
      chained_policy_t, // 26
      op_src // 27
    );

#if false // CCCL_DEBUGGING_SWITCH
    fflush(stderr);
    printf("\nCODE4NVRTC BEGIN\n%sCODE4NVRTC END\n", src.c_str());
    fflush(stdout);
#endif

    std::string single_tile_kernel_name =
      radix_sort::get_single_tile_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
    std::string upsweep_kernel_name =
      radix_sort::get_upsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, offset_t);
    std::string alt_upsweep_kernel_name =
      radix_sort::get_upsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, offset_t);
    std::string scan_bins_kernel_name = radix_sort::get_scan_bins_kernel_name(chained_policy_t, offset_t);
    std::string downsweep_kernel_name =
      radix_sort::get_downsweep_kernel_name(chained_policy_t, false, sort_order, key_cpp, value_cpp, offset_t);
    std::string alt_downsweep_kernel_name =
      radix_sort::get_downsweep_kernel_name(chained_policy_t, true, sort_order, key_cpp, value_cpp, offset_t);
    std::string histogram_kernel_name =
      radix_sort::get_histogram_kernel_name(chained_policy_t, sort_order, key_cpp, offset_t);
    std::string exclusive_sum_kernel_name = radix_sort::get_exclusive_sum_kernel_name(chained_policy_t, offset_t);
    std::string onesweep_kernel_name =
      radix_sort::get_onesweep_kernel_name(chained_policy_t, sort_order, key_cpp, value_cpp, offset_t);
    std::string single_tile_kernel_lowered_name;
    std::string upsweep_kernel_lowered_name;
    std::string alt_upsweep_kernel_lowered_name;
    std::string scan_bins_kernel_lowered_name;
    std::string downsweep_kernel_lowered_name;
    std::string alt_downsweep_kernel_lowered_name;
    std::string histogram_kernel_lowered_name;
    std::string exclusive_sum_kernel_lowered_name;
    std::string onesweep_kernel_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    std::vector<const char*> args = {
      arch.c_str(), cub_path, thrust_path, libcudacxx_path, ctk_path, "-rdc=true", "-dlto", "-DCUB_DISABLE_CDP"};

    cccl::detail::extend_args_with_build_config(args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};
    appender.append_operation(decomposer);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{src.c_str(), name})
        ->add_expression({single_tile_kernel_name})
        ->add_expression({upsweep_kernel_name})
        ->add_expression({alt_upsweep_kernel_name})
        ->add_expression({scan_bins_kernel_name})
        ->add_expression({downsweep_kernel_name})
        ->add_expression({alt_downsweep_kernel_name})
        ->add_expression({histogram_kernel_name})
        ->add_expression({exclusive_sum_kernel_name})
        ->add_expression({onesweep_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({single_tile_kernel_name, single_tile_kernel_lowered_name})
        ->get_name({upsweep_kernel_name, upsweep_kernel_lowered_name})
        ->get_name({alt_upsweep_kernel_name, alt_upsweep_kernel_lowered_name})
        ->get_name({scan_bins_kernel_name, scan_bins_kernel_lowered_name})
        ->get_name({downsweep_kernel_name, downsweep_kernel_lowered_name})
        ->get_name({alt_downsweep_kernel_name, alt_downsweep_kernel_lowered_name})
        ->get_name({histogram_kernel_name, histogram_kernel_lowered_name})
        ->get_name({exclusive_sum_kernel_name, exclusive_sum_kernel_lowered_name})
        ->get_name({onesweep_kernel_name, onesweep_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
        ->finalize_program();

    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(
      cuLibraryGetKernel(&build_ptr->single_tile_kernel, build_ptr->library, single_tile_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->upsweep_kernel, build_ptr->library, upsweep_kernel_lowered_name.c_str()));
    check(
      cuLibraryGetKernel(&build_ptr->alt_upsweep_kernel, build_ptr->library, alt_upsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->scan_bins_kernel, build_ptr->library, scan_bins_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->downsweep_kernel, build_ptr->library, downsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->alt_downsweep_kernel, build_ptr->library, alt_downsweep_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->histogram_kernel, build_ptr->library, histogram_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->exclusive_sum_kernel, build_ptr->library, exclusive_sum_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(&build_ptr->onesweep_kernel, build_ptr->library, onesweep_kernel_lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;
    build_ptr->key_type   = input_keys_it.value_type;
    build_ptr->value_type = input_values_it.value_type;
    build_ptr->order      = sort_order;
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

template <cub::SortOrder Order>
CUresult cccl_device_radix_sort_impl(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  if (cccl_iterator_kind_t::CCCL_POINTER != d_keys_in.type || cccl_iterator_kind_t::CCCL_POINTER != d_values_in.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_keys_out.type
      || cccl_iterator_kind_t::CCCL_POINTER != d_values_out.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_radix_sort(): radix sort input must be a pointer\n");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  CUresult error = CUDA_SUCCESS;
  bool pushed    = false;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    indirect_arg_t key_arg_in{d_keys_in};
    indirect_arg_t key_arg_out{d_keys_out};
    cub::DoubleBuffer<indirect_arg_t> d_keys_buffer(
      *static_cast<indirect_arg_t**>(&key_arg_in), *static_cast<indirect_arg_t**>(&key_arg_out));

    indirect_arg_t val_arg_in{d_values_in};
    indirect_arg_t val_arg_out{d_values_out};
    cub::DoubleBuffer<indirect_arg_t> d_values_buffer(
      *static_cast<indirect_arg_t**>(&val_arg_in), *static_cast<indirect_arg_t**>(&val_arg_out));

    auto exec_status = cub::DispatchRadixSort<
      Order,
      indirect_arg_t,
      indirect_arg_t,
      OffsetT,
      indirect_arg_t,
      radix_sort::dynamic_radix_sort_policy_t<&radix_sort::get_policy>,
      radix_sort::radix_sort_kernel_source,
      cub::detail::CudaDriverLauncherFactory>::
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_keys_buffer,
        d_values_buffer,
        num_items,
        begin_bit,
        end_bit,
        is_overwrite_okay,
        stream,
        decomposer,
        {build},
        cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        {d_keys_in.value_type.size});

    *selector = d_keys_buffer.selector;
    error     = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  if (pushed)
  {
    CUcontext dummy;
    cuCtxPopCurrent(&dummy);
  }

  return error;
}

CUresult cccl_device_radix_sort(
  cccl_device_radix_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  cccl_op_t decomposer,
  uint64_t num_items,
  int begin_bit,
  int end_bit,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  auto radix_sort_impl =
    (build.order == CCCL_ASCENDING)
      ? cccl_device_radix_sort_impl<cub::SortOrder::Ascending>
      : cccl_device_radix_sort_impl<cub::SortOrder::Descending>;
  return radix_sort_impl(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    decomposer,
    num_items,
    begin_bit,
    end_bit,
    is_overwrite_okay,
    selector,
    stream);
}

CUresult cccl_device_radix_sort_build(
  cccl_device_radix_sort_build_result_t* build,
  cccl_sort_order_t sort_order,
  cccl_iterator_t input_keys_it,
  cccl_iterator_t input_values_it,
  cccl_op_t decomposer,
  const char* decomposer_return_type,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_radix_sort_build_ex(
    build,
    sort_order,
    input_keys_it,
    input_values_it,
    decomposer,
    decomposer_return_type,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
}

CUresult cccl_device_radix_sort_cleanup(cccl_device_radix_sort_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_radix_sort_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
