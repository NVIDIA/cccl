//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cub/detail/choose_offset.cuh> // cub::detail::choose_offset_t
#include <cub/detail/launcher/cuda_driver.cuh> // cub::detail::CudaDriverLauncherFactory
#include <cub/device/dispatch/dispatch_segmented_sort.cuh> // cub::DispatchSegmentedSort
#include <cub/device/dispatch/kernels/segmented_sort.cuh> // DeviceSegmentedSort kernels
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh> // policy_hub
#include <cub/thread/thread_load.cuh> // cub::LoadModifier

#include <exception> // std::exception
#include <format> // std::format
#include <string> // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::is_same_v

#include <stdio.h> // printf

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/runtime_policy.h"
#include "util/types.h"
#include <cccl/c/segmented_sort.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nlohmann/json.hpp>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>

struct device_segmented_sort_policy;
using OffsetT = long;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>, "OffsetT must be long");

// check we can map OffsetT to ::cuda::std::int64_t
static_assert(std::is_signed_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(::cuda::std::int64_t));

namespace segmented_sort
{

// Runtime policy structure for segmented sort
struct segmented_sort_runtime_policy
{
  int partitioning_threshold;
  int large_segment_radix_bits;
  int segments_per_small_block;
  int segments_per_medium_block;
  int small_policy_items_per_tile;
  int medium_policy_items_per_tile;

  // Required methods for SegmentedSortPolicyWrapper
  constexpr int PartitioningThreshold() const
  {
    return partitioning_threshold;
  }
  constexpr int LargeSegmentRadixBits() const
  {
    return large_segment_radix_bits;
  }
  constexpr int SegmentsPerSmallBlock() const
  {
    return segments_per_small_block;
  }
  constexpr int SegmentsPerMediumBlock() const
  {
    return segments_per_medium_block;
  }
  constexpr int SmallPolicyItemsPerTile() const
  {
    return small_policy_items_per_tile;
  }
  constexpr int MediumPolicyItemsPerTile() const
  {
    return medium_policy_items_per_tile;
  }

  // Additional methods expected by SegmentedSortPolicyWrapper
  constexpr void CheckLoadModifierIsNotLDG() const {} // No-op validation
  constexpr void CheckLoadAlgorithmIsNotStriped() const {} // No-op validation
  constexpr void CheckStoreAlgorithmIsNotStriped() const {} // No-op validation

  // Policy accessor methods
  constexpr int BlockThreads(int /* large_segment_policy */) const
  {
    return 256;
  } // Default block size
  constexpr int LargeSegment() const
  {
    return 0;
  } // Return index for large segment policy
  constexpr auto SmallAndMediumSegmentedSort() const
  {
    return *this;
  } // Return policy for small/medium segments

  using MaxPolicy = segmented_sort_runtime_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<segmented_sort_runtime_policy>(*this);
  }
};

// Function to create runtime policy from JSON
segmented_sort_runtime_policy from_json(const nlohmann::json& j)
{
  return segmented_sort_runtime_policy{
    .partitioning_threshold       = j["PartitioningThreshold"].get<int>(),
    .large_segment_radix_bits     = j["LargeSegmentRadixBits"].get<int>(),
    .segments_per_small_block     = j["SegmentsPerSmallBlock"].get<int>(),
    .segments_per_medium_block    = j["SegmentsPerMediumBlock"].get<int>(),
    .small_policy_items_per_tile  = j["SmallPolicyItemsPerTile"].get<int>(),
    .medium_policy_items_per_tile = j["MediumPolicyItemsPerTile"].get<int>()};
}

std::string get_device_segmented_sort_fallback_kernel_name(
  std::string_view /* key_iterator_t */,
  std::string_view /* value_iterator_t */,
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  /*
  template <SortOrder Order,             // 0 (ascending)
            typename ChainedPolicyT,     // 1
            typename KeyT,               // 2
            typename ValueT,             // 3
            typename BeginOffsetIteratorT, // 4
            typename EndOffsetIteratorT,   // 5
            typename OffsetT>              // 6
   DeviceSegmentedSortFallbackKernel(...);
  */
  return std::format(
    "cub::detail::segmented_sort::DeviceSegmentedSortFallbackKernel<cub::SortOrder::Ascending, {0}, {1}, {2}, {3}, "
    "{4}, {5}>",
    chained_policy_t, // 0
    key_t, // 1
    value_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t); // 5
}

std::string get_device_segmented_sort_kernel_small_name(
  std::string_view /* key_iterator_t */,
  std::string_view /* value_iterator_t */,
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  /*
  template <SortOrder Order,             // 0 (ascending)
            typename ChainedPolicyT,     // 1
            typename KeyT,               // 2
            typename ValueT,             // 3
            typename BeginOffsetIteratorT, // 4
            typename EndOffsetIteratorT,   // 5
            typename OffsetT>              // 6
   DeviceSegmentedSortKernelSmall(...);
  */
  return std::format(
    "cub::detail::segmented_sort::DeviceSegmentedSortKernelSmall<cub::SortOrder::Ascending, {0}, {1}, {2}, {3}, {4}, "
    "{5}>",
    chained_policy_t, // 0
    key_t, // 1
    value_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t); // 5
}

std::string get_device_segmented_sort_kernel_large_name(
  std::string_view /* key_iterator_t */,
  std::string_view /* value_iterator_t */,
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  /*
  template <SortOrder Order,             // 0 (ascending)
            typename ChainedPolicyT,     // 1
            typename KeyT,               // 2
            typename ValueT,             // 3
            typename BeginOffsetIteratorT, // 4
            typename EndOffsetIteratorT,   // 5
            typename OffsetT>              // 6
   DeviceSegmentedSortKernelLarge(...);
  */
  return std::format(
    "cub::detail::segmented_sort::DeviceSegmentedSortKernelLarge<cub::SortOrder::Ascending, {0}, {1}, {2}, {3}, {4}, "
    "{5}>",
    chained_policy_t, // 0
    key_t, // 1
    value_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t); // 5
}

struct segmented_sort_kernel_source
{
  cccl_device_segmented_sort_build_result_t& build;

  CUkernel SegmentedSortFallbackKernel() const
  {
    return build.segmented_sort_fallback_kernel;
  }
  CUkernel SegmentedSortKernelSmall() const
  {
    return build.segmented_sort_kernel_small;
  }
  CUkernel SegmentedSortKernelLarge() const
  {
    return build.segmented_sort_kernel_large;
  }
};

struct partition_kernel_source
{
  cccl_device_segmented_sort_build_result_t& build;

  CUkernel ThreeWayPartitionInitKernel() const
  {
    return build.three_way_partition_init_kernel;
  }
  CUkernel ThreeWayPartitionKernel() const
  {
    return build.three_way_partition_kernel;
  }

  std::size_t OffsetSize() const
  {
    return build.offset_type.size;
  }
};

struct segmented_sort_runtime_tuning_policy
{
  cub::detail::RuntimeRadixSortDownsweepAgentPolicy large_segment;
  cub::detail::RuntimeSmallAndMediumSegmentedSortAgentPolicy small_and_medium_segment;

  auto LargeSegment() const
  {
    return large_segment;
  }
  auto SmallAndMediumSegmentedSort() const
  {
    return small_and_medium_segment;
  }

  using MaxPolicy = segmented_sort_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<segmented_sort_runtime_tuning_policy>(*this);
  }
};

struct partition_runtime_tuning_policy
{
  cub::detail::RuntimeThreeWayPartitionAgentPolicy three_way_partition;

  auto ThreeWayPartition() const
  {
    return three_way_partition;
  }

  using MaxPolicy = partition_runtime_tuning_policy;

  template <typename F>
  cudaError_t Invoke(int, F& op)
  {
    return op.template Invoke<partition_runtime_tuning_policy>(*this);
  }
};
} // namespace segmented_sort

struct segmented_sort_keys_input_iterator_tag;
struct segmented_sort_keys_output_iterator_tag;
struct segmented_sort_values_input_iterator_tag;
struct segmented_sort_values_output_iterator_tag;
struct segmented_sort_start_offset_iterator_tag;
struct segmented_sort_end_offset_iterator_tag;

CUresult cccl_device_segmented_sort_build(
  cccl_device_segmented_sort_build_result_t* build_ptr,
  cccl_iterator_t keys_in_it,
  cccl_iterator_t keys_out_it,
  cccl_iterator_t values_in_it,
  cccl_iterator_t values_out_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  CUresult error = CUDA_SUCCESS;

  if (keys_in_it.value_type.type != keys_out_it.value_type.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_segmented_sort_build(): keys_in_it and keys_out_it must have the same type\n ");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  if (values_in_it.value_type.type != values_out_it.value_type.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_segmented_sort_build(): values_in_it and values_out_it must have the same type\n ");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  try
  {
    const char* name = "device_segmented_sort";

    const int cc = cc_major * 10 + cc_minor;

    const auto [keys_in_iterator_name, keys_in_iterator_src] =
      get_specialization<segmented_sort_keys_input_iterator_tag>(template_id<input_iterator_traits>(), keys_in_it);

    const auto [keys_out_iterator_name, keys_out_iterator_src] =
      get_specialization<segmented_sort_keys_output_iterator_tag>(
        template_id<output_iterator_traits>(), keys_out_it, keys_out_it.value_type);

    const bool keys_only = values_in_it.type == cccl_iterator_kind_t::CCCL_POINTER && values_in_it.state == nullptr;

    std::string values_in_iterator_name, values_in_iterator_src;
    std::string values_out_iterator_name, values_out_iterator_src;

    if (!keys_only)
    {
      const auto [vi_name, vi_src] = get_specialization<segmented_sort_values_input_iterator_tag>(
        template_id<input_iterator_traits>(), values_in_it);
      values_in_iterator_name = vi_name;
      values_in_iterator_src  = vi_src;

      const auto [vo_name, vo_src] = get_specialization<segmented_sort_values_output_iterator_tag>(
        template_id<output_iterator_traits>(), values_out_it, values_in_it.value_type);
      values_out_iterator_name = vo_name;
      values_out_iterator_src  = vo_src;
    }
    else
    {
      values_in_iterator_name  = "cub::NullType*";
      values_out_iterator_name = "cub::NullType*";
      values_in_iterator_src   = "";
      values_out_iterator_src  = "";
    }

    const auto [start_offset_iterator_name, start_offset_iterator_src] =
      get_specialization<segmented_sort_start_offset_iterator_tag>(
        template_id<input_iterator_traits>(), start_offset_it);

    const auto [end_offset_iterator_name, end_offset_iterator_src] =
      get_specialization<segmented_sort_end_offset_iterator_tag>(template_id<input_iterator_traits>(), end_offset_it);

    const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_UINT64);

    const std::string key_t   = cccl_type_enum_to_name(keys_in_it.value_type.type);
    const std::string value_t = keys_only ? "cub::NullType" : cccl_type_enum_to_name(values_in_it.value_type.type);

    const std::string dependent_definitions_src = std::format(
      R"XXX(
struct __align__({1}) storage_t {{
  char data[{0}];
}};
{2}
{3}
{4}
{5}
{6}
{7}
)XXX",
      keys_in_it.value_type.size, // 0
      keys_in_it.value_type.alignment, // 1
      keys_in_iterator_src, // 2
      keys_out_iterator_src, // 3
      values_in_iterator_src, // 4
      values_out_iterator_src, // 5
      start_offset_iterator_src, // 6
      end_offset_iterator_src); // 7

    // Runtime parameter tuning
    const std::string ptx_arch = std::format("-arch=compute_{}{}", cc_major, cc_minor);

    constexpr size_t ptx_num_args      = 5;
    const char* ptx_args[ptx_num_args] = {ptx_arch.c_str(), cub_path, thrust_path, libcudacxx_path, "-rdc=true"};

    static constexpr std::string_view policy_wrapper_expr_tmpl =
      R"XXXX(cub::detail::segmented_sort::MakeSegmentedSortPolicyWrapper(cub::detail::segmented_sort::policy_hub<{0}, {1}>::MaxPolicy::ActivePolicy{{}}))XXXX";

    const auto policy_wrapper_expr = std::format(
      policy_wrapper_expr_tmpl,
      key_t, // 0
      value_t); // 1

    static constexpr std::string_view ptx_query_tu_src_tmpl = R"XXXX(
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
{0}
{1}
)XXXX";

    const auto ptx_query_tu_src =
      std::format(ptx_query_tu_src_tmpl, jit_template_header_contents, dependent_definitions_src);

    nlohmann::json runtime_policy = get_policy(policy_wrapper_expr, ptx_query_tu_src, ptx_args);

    auto segmented_sort_policy = segmented_sort::from_json(runtime_policy);

    // Extract sub-policy information if available
    std::string small_and_medium_policy_str;
    if (runtime_policy.contains("SmallAndMediumSegmentedSort"))
    {
      auto sub_policy          = runtime_policy["SmallAndMediumSegmentedSort"];
      auto block_threads       = sub_policy["BlockThreads"].get<int>();
      auto segments_per_medium = sub_policy["SegmentsPerMediumBlock"].get<int>();
      auto segments_per_small  = sub_policy["SegmentsPerSmallBlock"].get<int>();

      small_and_medium_policy_str = std::format(
        R"XXX(
    // Small and Medium Segment Policy
    static constexpr int SMALL_MEDIUM_BLOCK_THREADS = {0};
    static constexpr int SMALL_MEDIUM_SEGMENTS_PER_MEDIUM_BLOCK = {1};
    static constexpr int SMALL_MEDIUM_SEGMENTS_PER_SMALL_BLOCK = {2};)XXX",
        block_threads,
        segments_per_medium,
        segments_per_small);
    }

    // Build the policy structure manually
    const std::string segmented_sort_policy_str = std::format(
      R"XXX(
    static constexpr int PARTITIONING_THRESHOLD = {0};
    static constexpr int LARGE_SEGMENT_RADIX_BITS = {1};
    static constexpr int SEGMENTS_PER_SMALL_BLOCK = {2};
    static constexpr int SEGMENTS_PER_MEDIUM_BLOCK = {3};
    static constexpr int SMALL_POLICY_ITEMS_PER_TILE = {4};
    static constexpr int MEDIUM_POLICY_ITEMS_PER_TILE = {5};{6}
    using MaxPolicy = cub::detail::segmented_sort::policy_hub<{7}, {8}>::MaxPolicy;
)XXX",
      segmented_sort_policy.partitioning_threshold, // 0
      segmented_sort_policy.large_segment_radix_bits, // 1
      segmented_sort_policy.segments_per_small_block, // 2
      segmented_sort_policy.segments_per_medium_block, // 3
      segmented_sort_policy.small_policy_items_per_tile, // 4
      segmented_sort_policy.medium_policy_items_per_tile, // 5
      small_and_medium_policy_str, // 6
      key_t, // 7
      value_t); // 8

    // agent_policy_t is to specify parameters like policy_hub does in dispatch_segmented_sort.cuh
    constexpr std::string_view program_preamble_template = R"XXX(
#include <cub/device/dispatch/kernels/segmented_sort.cuh>
{0}
{1}
struct device_segmented_sort_policy {{
  struct ActivePolicy {{
    {2}
  }};
}};
)XXX";

    std::string final_src = std::format(
      program_preamble_template,
      jit_template_header_contents, // 0
      dependent_definitions_src, // 1
      segmented_sort_policy_str); // 2

    std::string segmented_sort_fallback_kernel_name = segmented_sort::get_device_segmented_sort_fallback_kernel_name(
      keys_in_iterator_name,
      values_in_iterator_name,
      start_offset_iterator_name,
      end_offset_iterator_name,
      key_t,
      value_t);

    std::string segmented_sort_kernel_small_name = segmented_sort::get_device_segmented_sort_kernel_small_name(
      keys_in_iterator_name,
      values_in_iterator_name,
      start_offset_iterator_name,
      end_offset_iterator_name,
      key_t,
      value_t);

    std::string segmented_sort_kernel_large_name = segmented_sort::get_device_segmented_sort_kernel_large_name(
      keys_in_iterator_name,
      values_in_iterator_name,
      start_offset_iterator_name,
      end_offset_iterator_name,
      key_t,
      value_t);

    std::string segmented_sort_fallback_kernel_lowered_name;
    std::string segmented_sort_kernel_small_lowered_name;
    std::string segmented_sort_kernel_large_lowered_name;

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    constexpr size_t num_args  = 9;
    const char* args[num_args] = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-DCUB_DISABLE_CDP",
      "-std=c++20"};

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // Collect all LTO-IRs to be linked.
    nvrtc_ltoir_list ltoir_list;
    nvrtc_ltoir_list_appender appender{ltoir_list};

    // add iterator definitions
    appender.add_iterator_definition(keys_in_it);
    appender.add_iterator_definition(keys_out_it);
    if (!keys_only)
    {
      appender.add_iterator_definition(values_in_it);
      appender.add_iterator_definition(values_out_it);
    }
    appender.add_iterator_definition(start_offset_it);
    appender.add_iterator_definition(end_offset_it);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
        ->add_expression({segmented_sort_fallback_kernel_name})
        ->add_expression({segmented_sort_kernel_small_name})
        ->add_expression({segmented_sort_kernel_large_name})
        ->compile_program({args, num_args})
        ->get_name({segmented_sort_fallback_kernel_name, segmented_sort_fallback_kernel_lowered_name})
        ->get_name({segmented_sort_kernel_small_name, segmented_sort_kernel_small_lowered_name})
        ->get_name({segmented_sort_kernel_large_name, segmented_sort_kernel_large_lowered_name})
        ->link_program()
        ->add_link_list(ltoir_list)
        ->finalize_program();

    // populate build struct members
    cuLibraryLoadData(&build_ptr->library, result.data.get(), nullptr, nullptr, 0, nullptr, nullptr, 0);
    check(cuLibraryGetKernel(&build_ptr->segmented_sort_fallback_kernel,
                             build_ptr->library,
                             segmented_sort_fallback_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->segmented_sort_kernel_small, build_ptr->library, segmented_sort_kernel_small_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->segmented_sort_kernel_large, build_ptr->library, segmented_sort_kernel_large_lowered_name.c_str()));

    build_ptr->cc         = cc;
    build_ptr->cubin      = (void*) result.data.release();
    build_ptr->cubin_size = result.size;
    // Use the runtime policy extracted via from_json
    build_ptr->runtime_policy = new segmented_sort::segmented_sort_runtime_policy{segmented_sort_policy};
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_sort_build(): %s\n", exc.what());
    fflush(stdout);
    error = CUDA_ERROR_UNKNOWN;
  }

  return error;
}

CUresult cccl_device_segmented_sort(
  cccl_device_segmented_sort_build_result_t build,
  void* d_temp_storage,
  size_t* temp_storage_bytes,
  cccl_iterator_t d_keys_in,
  cccl_iterator_t d_keys_out,
  cccl_iterator_t d_values_in,
  cccl_iterator_t d_values_out,
  int64_t num_items,
  int64_t num_segments,
  cccl_iterator_t start_offset_in,
  cccl_iterator_t end_offset_in,
  CUstream stream)
{
  bool pushed    = false;
  CUresult error = CUDA_SUCCESS;
  try
  {
    pushed = try_push_context();

    CUdevice cu_device;
    check(cuCtxGetDevice(&cu_device));

    // Create DoubleBuffer structures for keys and values
    // CUB will handle keys-only vs key-value sorting internally
    auto d_keys_double_buffer = cub::DoubleBuffer<indirect_arg_t>(
      static_cast<indirect_arg_t*>(d_keys_in.state), static_cast<indirect_arg_t*>(d_keys_out.state));
    auto d_values_double_buffer = cub::DoubleBuffer<indirect_arg_t>(
      static_cast<indirect_arg_t*>(d_values_in.state), static_cast<indirect_arg_t*>(d_values_out.state));

    auto exec_status = cub::DispatchSegmentedSort<
      cub::SortOrder::Ascending,
      indirect_arg_t, // KeyT
      indirect_arg_t, // ValueT
      OffsetT, // OffsetT
      indirect_iterator_t, // BeginOffsetIteratorT
      indirect_iterator_t, // EndOffsetIteratorT
      segmented_sort::segmented_sort_runtime_tuning_policy, // PolicyHub
      segmented_sort::segmented_sort_kernel_source, // KernelSource
      segmented_sort::partition_runtime_tuning_policy, // PartitionPolicyHub
      segmented_sort::partition_kernel_source, // PartitionKernelSource
      cub::detail::CudaDriverLauncherFactory>:: // KernelLaunchFactory
      Dispatch(
        d_temp_storage,
        *temp_storage_bytes,
        d_keys_double_buffer,
        d_values_double_buffer,
        num_items,
        num_segments,
        indirect_iterator_t{start_offset_in},
        indirect_iterator_t{end_offset_in},
        true, // is_overwrite_okay
        stream,
        /* kernel_source */ {build},
        /* partition_kernel_source */ {build},
        /* launcher_factory */ cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        /* policy */ *reinterpret_cast<segmented_sort::segmented_sort_runtime_policy*>(build.runtime_policy),
        /* partition_policy */
        *reinterpret_cast<segmented_sort::partition_runtime_tuning_policy*>(build.partition_runtime_policy));

    error = static_cast<CUresult>(exec_status);
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_sort(): %s\n", exc.what());
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

CUresult cccl_device_segmented_sort_cleanup(cccl_device_segmented_sort_build_result_t* build_ptr)
{
  try
  {
    if (build_ptr == nullptr)
    {
      return CUDA_ERROR_INVALID_VALUE;
    }

    // allocation behind cubin is owned by unique_ptr with delete[] deleter now
    std::unique_ptr<char[]> cubin(reinterpret_cast<char*>(build_ptr->cubin));

    // Clean up the runtime policy
    delete static_cast<segmented_sort::segmented_sort_runtime_policy*>(build_ptr->runtime_policy);
    check(cuLibraryUnload(build_ptr->library));
  }
  catch (const std::exception& exc)
  {
    fflush(stderr);
    printf("\nEXCEPTION in cccl_device_segmented_sort_cleanup(): %s\n", exc.what());
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  return CUDA_SUCCESS;
}
