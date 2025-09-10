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
struct device_three_way_partition_policy;
using OffsetT = long;
static_assert(std::is_same_v<cub::detail::choose_signed_offset_t<OffsetT>, OffsetT>, "OffsetT must be long");

// check we can map OffsetT to cuda::std::int64_t
static_assert(std::is_signed_v<OffsetT>);
static_assert(sizeof(OffsetT) == sizeof(cuda::std::int64_t));

namespace segmented_sort
{
std::string get_device_segmented_sort_fallback_kernel_name(
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t,
  cccl_sort_order_t sort_order)
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
    "cub::detail::segmented_sort::DeviceSegmentedSortFallbackKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    chained_policy_t, // 0
    key_t, // 1
    value_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t); // 5
}

std::string get_device_segmented_sort_kernel_small_name(
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t,
  cccl_sort_order_t sort_order)
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
    "cub::detail::segmented_sort::DeviceSegmentedSortKernelSmall<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
    chained_policy_t, // 0
    key_t, // 1
    value_t, // 2
    start_offset_iterator_t, // 3
    end_offset_iterator_t, // 4
    offset_t); // 5
}

std::string get_device_segmented_sort_kernel_large_name(
  std::string_view start_offset_iterator_t,
  std::string_view end_offset_iterator_t,
  std::string_view key_t,
  std::string_view value_t,
  cccl_sort_order_t sort_order)
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
    "cub::detail::segmented_sort::DeviceSegmentedSortKernelLarge<{0}, {1}, {2}, {3}, {4}, {5}, {6}>",
    (sort_order == CCCL_ASCENDING) ? "cub::SortOrder::Ascending" : "cub::SortOrder::Descending",
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

  std::size_t KeySize() const
  {
    return build.key_type.size;
  }

  using LargeSegmentsSelectorT = cub::detail::segmented_sort::LargeSegmentsSelectorT<OffsetT, void*, void*>;
  using SmallSegmentsSelectorT = cub::detail::segmented_sort::SmallSegmentsSelectorT<OffsetT, void*, void*>;

  auto LargeSegmentsSelector(
    OffsetT offset, indirect_iterator_t begin_offset_iterator, indirect_iterator_t end_offset_iterator) const
  {
    return LargeSegmentsSelectorT(
      offset, *reinterpret_cast<void**>(begin_offset_iterator.ptr), *reinterpret_cast<void**>(end_offset_iterator.ptr));
  }

  auto SmallSegmentsSelector(
    OffsetT offset, indirect_iterator_t begin_offset_iterator, indirect_iterator_t end_offset_iterator) const
  {
    return SmallSegmentsSelectorT(
      offset, *reinterpret_cast<void**>(begin_offset_iterator.ptr), *reinterpret_cast<void**>(end_offset_iterator.ptr));
  }
};

std::string get_three_way_partition_init_kernel_name()
{
  constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";

  constexpr std::string_view num_selected_it_t = "cub::detail::segmented_sort::local_segment_index_t*";

  return std::format("cub::detail::three_way_partition::DeviceThreeWayPartitionInitKernel<{0}, {1}>",
                     scan_tile_state_t, // 0
                     num_selected_it_t); // 1
}

std::string
get_three_way_partition_kernel_name(std::string_view start_offset_iterator_t, std::string_view end_offset_iterator_t)
{
  std::string chained_policy_t;
  check(nvrtcGetTypeName<device_three_way_partition_policy>(&chained_policy_t));

  constexpr std::string_view input_it_t =
    "thrust::counting_iterator<cub::detail::segmented_sort::local_segment_index_t>";
  constexpr std::string_view first_out_it_t  = "cub::detail::segmented_sort::local_segment_index_t*";
  constexpr std::string_view second_out_it_t = "cub::detail::segmented_sort::local_segment_index_t*";
  constexpr std::string_view unselected_out_it_t =
    "thrust::reverse_iterator<cub::detail::segmented_sort::local_segment_index_t*>";
  constexpr std::string_view num_selected_it_t = "cub::detail::segmented_sort::local_segment_index_t*";
  constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";
  std::string offset_t;
  check(nvrtcGetTypeName<OffsetT>(&offset_t));

  std::string select_first_part_op_t = std::format(
    "cub::detail::segmented_sort::LargeSegmentsSelectorT<{0}, {1}, {2}>",
    offset_t, // 0
    start_offset_iterator_t, // 1
    end_offset_iterator_t); // 2

  std::string select_second_part_op_t = std::format(
    "cub::detail::segmented_sort::SmallSegmentsSelectorT<{0}, {1}, {2}>",
    offset_t, // 0
    start_offset_iterator_t, // 1
    end_offset_iterator_t); // 2

  constexpr std::string_view per_partition_offset_t = "cub::detail::three_way_partition::per_partition_offset_t";
  constexpr std::string_view streaming_context_t =
    "cub::detail::three_way_partition::streaming_context_t<cub::detail::segmented_sort::global_segment_offset_t>";

  return std::format(
    "cub::detail::three_way_partition::DeviceThreeWayPartitionKernel<{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, "
    "{10}>",
    chained_policy_t, // 0 (ChainedPolicyT)
    input_it_t, // 1 (InputIteratorT)
    first_out_it_t, // 2 (FirstOutputIteratorT)
    second_out_it_t, // 3 (SecondOutputIteratorT)
    unselected_out_it_t, // 4 (UnselectedOutputIteratorT)
    num_selected_it_t, // 5 (NumSelectedIteratorT)
    scan_tile_state_t, // 6 (ScanTileStateT)
    select_first_part_op_t, // 7 (SelectFirstPartOp)
    select_second_part_op_t, // 8 (SelectSecondPartOp)
    per_partition_offset_t, // 9 (OffsetT)
    streaming_context_t); // 10 (StreamingContextT)
}

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
  cub::detail::RuntimeSubWarpMergeSortAgentPolicy small_segment;
  cub::detail::RuntimeSubWarpMergeSortAgentPolicy medium_segment;
  int partitioning_threshold;

  auto LargeSegment() const
  {
    return large_segment;
  }

  auto SmallSegment() const
  {
    return small_segment;
  }

  auto MediumSegment() const
  {
    return medium_segment;
  }

  void CheckLoadModifierIsNotLDG() const
  {
    if (large_segment.LoadModifier() == cub::CacheLoadModifier::LOAD_LDG)
    {
      throw std::runtime_error("The memory consistency model does not apply to texture accesses");
    }
  }

  void CheckLoadAlgorithmIsNotStriped() const
  {
    if (large_segment.LoadAlgorithm() == cub::BLOCK_LOAD_STRIPED
        || medium_segment.LoadAlgorithm() == cub::WARP_LOAD_STRIPED
        || small_segment.LoadAlgorithm() == cub::WARP_LOAD_STRIPED)
    {
      throw std::runtime_error("Striped load will make this algorithm unstable");
    }
  }

  void CheckStoreAlgorithmIsNotStriped() const
  {
    if (medium_segment.StoreAlgorithm() == cub::WARP_STORE_STRIPED
        || small_segment.StoreAlgorithm() == cub::WARP_STORE_STRIPED)
    {
      throw std::runtime_error("Striped stores will produce unsorted results");
    }
  }

  int PartitioningThreshold() const
  {
    return partitioning_threshold;
  }

  int LargeSegmentRadixBits() const
  {
    return large_segment.RadixBits();
  }

  int SegmentsPerSmallBlock() const
  {
    return small_segment.SegmentsPerBlock();
  }

  int SegmentsPerMediumBlock() const
  {
    return medium_segment.SegmentsPerBlock();
  }

  int SmallPolicyItemsPerTile() const
  {
    return small_segment.ItemsPerTile();
  }

  int MediumPolicyItemsPerTile() const
  {
    return medium_segment.ItemsPerTile();
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

std::string get_three_way_partition_policy_delay_constructor(const nlohmann::json& partition_policy)
{
  auto dc_json                = partition_policy["ThreeWayPartitionPolicyDelayConstructor"];
  auto delay_constructor_type = dc_json["type"].get<std::string>();

  if (delay_constructor_type == "fixed_delay_constructor_t")
  {
    auto delay            = dc_json["delay"].get<int>();
    auto l2_write_latency = dc_json["l2_write_latency"].get<int>();
    return std::format("cub::detail::fixed_delay_constructor_t<{}, {}>", delay, l2_write_latency);
  }
  else if (delay_constructor_type == "no_delay_constructor_t")
  {
    auto l2_write_latency = dc_json["l2_write_latency"].get<int>();
    return std::format("cub::detail::no_delay_constructor_t<{}>", l2_write_latency);
  }
  throw std::runtime_error("Invalid delay constructor type: " + delay_constructor_type);
}

std::string inject_delay_constructor_into_three_way_policy(
  const std::string& three_way_partition_policy_str, const std::string& delay_constructor_type)
{
  // Insert before the final closing of the struct (right before the sequence "};")
  const std::string needle = "};";
  const auto pos           = three_way_partition_policy_str.rfind(needle);
  if (pos == std::string::npos)
  {
    return three_way_partition_policy_str; // unexpected; return as-is
  }
  const std::string insertion =
    std::format("\n  struct detail {{ using delay_constructor_t = {}; }}; \n", delay_constructor_type);
  std::string out = three_way_partition_policy_str;
  out.insert(pos, insertion);
  return out;
}
} // namespace segmented_sort

struct segmented_sort_keys_input_iterator_tag;
struct segmented_sort_keys_output_iterator_tag;
struct segmented_sort_values_input_iterator_tag;
struct segmented_sort_values_output_iterator_tag;
struct segmented_sort_start_offset_iterator_tag;
struct segmented_sort_end_offset_iterator_tag;

CUresult cccl_device_segmented_sort_build(
  cccl_device_segmented_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
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

      // For STORAGE values, ensure pointer types in iterator names/sources use items_storage_t*
      if (values_in_it.value_type.type == cccl_type_enum::CCCL_STORAGE)
      {
        auto replace_all = [](std::string& s, const std::string& from, const std::string& to) {
          if (from.empty())
          {
            return;
          }
          size_t pos = 0;
          while ((pos = s.find(from, pos)) != std::string::npos)
          {
            s.replace(pos, from.length(), to);
            pos += to.length();
          }
        };
        replace_all(values_in_iterator_src, "storage_t", "items_storage_t");
        replace_all(values_out_iterator_src, "storage_t", "items_storage_t");
        replace_all(values_in_iterator_name, "storage_t", "items_storage_t");
        replace_all(values_out_iterator_name, "storage_t", "items_storage_t");
      }
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

    const auto offset_t = cccl_type_enum_to_name(cccl_type_enum::CCCL_INT64);

    const std::string key_t = cccl_type_enum_to_name(keys_in_it.value_type.type);
    const std::string value_t =
      keys_only ? "cub::NullType" : cccl_type_enum_to_name<items_storage_t>(values_in_it.value_type.type);

    const std::string dependent_definitions_src = std::format(
      R"XXX(
struct __align__({1}) storage_t {{
  char data[{0}];
}};
struct __align__({3}) items_storage_t {{
  char data[{2}];
}};
{4}
{5}
{6}
{7}
{8}
{9}
)XXX",
      keys_in_it.value_type.size, // 0
      keys_in_it.value_type.alignment, // 1
      values_in_it.value_type.size, // 2
      values_in_it.value_type.alignment, // 3
      keys_in_iterator_src, // 4
      keys_out_iterator_src, // 5
      values_in_iterator_src, // 6
      values_out_iterator_src, // 7
      start_offset_iterator_src, // 8
      end_offset_iterator_src); // 9

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
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>
{0}
{1}
)XXXX";

    const auto ptx_query_tu_src =
      std::format(ptx_query_tu_src_tmpl, jit_template_header_contents, dependent_definitions_src);

    nlohmann::json runtime_policy = get_policy(policy_wrapper_expr, ptx_query_tu_src, ptx_args);

    using cub::detail::RuntimeRadixSortDownsweepAgentPolicy;
    auto [large_segment_policy, large_segment_policy_str] =
      RuntimeRadixSortDownsweepAgentPolicy::from_json(runtime_policy, "LargeSegmentPolicy");

    using cub::detail::RuntimeSubWarpMergeSortAgentPolicy;
    auto [small_segment_policy, small_segment_policy_str] =
      RuntimeSubWarpMergeSortAgentPolicy::from_json(runtime_policy, "SmallSegmentPolicy");

    auto [medium_segment_policy, medium_segment_policy_str] =
      RuntimeSubWarpMergeSortAgentPolicy::from_json(runtime_policy, "MediumSegmentPolicy");

    auto partitioning_threshold = runtime_policy["PartitioningThreshold"].get<int>();

    static constexpr std::string_view partition_policy_wrapper_expr_tmpl =
      R"XXXX(cub::detail::three_way_partition::MakeThreeWayPartitionPolicyWrapper(cub::detail::three_way_partition::policy_hub<{0}, {1}>::MaxPolicy::ActivePolicy{{}}))XXXX";
    const auto partition_policy_wrapper_expr = std::format(
      partition_policy_wrapper_expr_tmpl,
      "::cuda::std::uint32_t", // This is local_segment_index_t defined in segmented_sort.cuh
      "::cuda::std::int32_t"); // This is per_partition_offset_t defined in segmented_sort.cuh

    nlohmann::json partition_policy = get_policy(partition_policy_wrapper_expr, ptx_query_tu_src, ptx_args);

    using cub::detail::RuntimeThreeWayPartitionAgentPolicy;
    auto [three_way_partition_policy, three_way_partition_policy_str] =
      RuntimeThreeWayPartitionAgentPolicy::from_json(partition_policy, "ThreeWayPartitionPolicy");

    const std::string three_way_partition_policy_delay_constructor =
      segmented_sort::get_three_way_partition_policy_delay_constructor(partition_policy);

    // Inject delay constructor alias into the ThreeWayPartitionPolicy struct string
    const std::string injected_three_way_partition_policy_str =
      segmented_sort::inject_delay_constructor_into_three_way_policy(
        three_way_partition_policy_str, three_way_partition_policy_delay_constructor);

    constexpr std::string_view program_preamble_template = R"XXX(
#include <cub/device/dispatch/kernels/segmented_sort.cuh>
#include <cub/device/dispatch/kernels/three_way_partition.cuh>
#include <thrust/iterator/counting_iterator.h> // used in three_way_partition kernel
#include <thrust/iterator/reverse_iterator.h> // used in three_way_partition kernel
#include <cub/detail/choose_offset.cuh> // used in three_way_partition kernel
{0}
{1}
struct device_segmented_sort_policy {{
  struct ActivePolicy {{
    {2}
    {3}
    {4}
  }};
}};
struct device_three_way_partition_policy {{
  struct ActivePolicy {{
    {5}
  }};
}};
)XXX";

    std::string final_src = std::format(
      program_preamble_template,
      jit_template_header_contents, // 0
      dependent_definitions_src, // 1
      large_segment_policy_str, // 2
      small_segment_policy_str, // 3
      medium_segment_policy_str, // 4
      injected_three_way_partition_policy_str); // 5

    std::string segmented_sort_fallback_kernel_name = segmented_sort::get_device_segmented_sort_fallback_kernel_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string segmented_sort_kernel_small_name = segmented_sort::get_device_segmented_sort_kernel_small_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string segmented_sort_kernel_large_name = segmented_sort::get_device_segmented_sort_kernel_large_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string three_way_partition_init_kernel_name = segmented_sort::get_three_way_partition_init_kernel_name();

    std::string three_way_partition_kernel_name =
      segmented_sort::get_three_way_partition_kernel_name(start_offset_iterator_name, end_offset_iterator_name);

    std::string segmented_sort_fallback_kernel_lowered_name;
    std::string segmented_sort_kernel_small_lowered_name;
    std::string segmented_sort_kernel_large_lowered_name;
    std::string three_way_partition_init_kernel_lowered_name;
    std::string three_way_partition_kernel_lowered_name;

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
        ->add_expression({three_way_partition_init_kernel_name})
        ->add_expression({three_way_partition_kernel_name})
        ->compile_program({args, num_args})
        ->get_name({segmented_sort_fallback_kernel_name, segmented_sort_fallback_kernel_lowered_name})
        ->get_name({segmented_sort_kernel_small_name, segmented_sort_kernel_small_lowered_name})
        ->get_name({segmented_sort_kernel_large_name, segmented_sort_kernel_large_lowered_name})
        ->get_name({three_way_partition_init_kernel_name, three_way_partition_init_kernel_lowered_name})
        ->get_name({three_way_partition_kernel_name, three_way_partition_kernel_lowered_name})
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
    check(cuLibraryGetKernel(&build_ptr->three_way_partition_init_kernel,
                             build_ptr->library,
                             three_way_partition_init_kernel_lowered_name.c_str()));
    check(cuLibraryGetKernel(
      &build_ptr->three_way_partition_kernel, build_ptr->library, three_way_partition_kernel_lowered_name.c_str()));

    build_ptr->cc          = cc;
    build_ptr->cubin       = (void*) result.data.release();
    build_ptr->cubin_size  = result.size;
    build_ptr->key_type    = keys_in_it.value_type;
    build_ptr->offset_type = cccl_type_info{sizeof(OffsetT), alignof(OffsetT), cccl_type_enum::CCCL_INT64};
    // Use the runtime policy extracted via from_json
    build_ptr->runtime_policy = new segmented_sort::segmented_sort_runtime_tuning_policy{
      large_segment_policy, small_segment_policy, medium_segment_policy, partitioning_threshold};
    build_ptr->partition_runtime_policy =
      new segmented_sort::partition_runtime_tuning_policy{three_way_partition_policy};
    build_ptr->order = sort_order;
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

template <cub::SortOrder Order>
CUresult cccl_device_segmented_sort_impl(
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
  bool is_overwrite_okay,
  int* selector,
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
    indirect_arg_t key_arg_in{d_keys_in};
    indirect_arg_t key_arg_out{d_keys_out};
    cub::DoubleBuffer<indirect_arg_t> d_keys_double_buffer(
      *static_cast<indirect_arg_t**>(&key_arg_in), *static_cast<indirect_arg_t**>(&key_arg_out));

    indirect_arg_t val_arg_in{d_values_in};
    indirect_arg_t val_arg_out{d_values_out};
    cub::DoubleBuffer<indirect_arg_t> d_values_double_buffer(
      *static_cast<indirect_arg_t**>(&val_arg_in), *static_cast<indirect_arg_t**>(&val_arg_out));

    auto exec_status = cub::DispatchSegmentedSort<
      Order,
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
        start_offset_in,
        end_offset_in,
        is_overwrite_okay,
        stream,
        /* kernel_source */ {build},
        /* partition_kernel_source */ {build},
        /* launcher_factory */ cub::detail::CudaDriverLauncherFactory{cu_device, build.cc},
        /* policy */ *reinterpret_cast<segmented_sort::segmented_sort_runtime_tuning_policy*>(build.runtime_policy),
        /* partition_policy */
        *reinterpret_cast<segmented_sort::partition_runtime_tuning_policy*>(build.partition_runtime_policy));

    if (selector != nullptr)
    {
      *selector = d_keys_double_buffer.selector;
    }

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
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  auto segmented_sort_impl =
    (build.order == CCCL_ASCENDING)
      ? cccl_device_segmented_sort_impl<cub::SortOrder::Ascending>
      : cccl_device_segmented_sort_impl<cub::SortOrder::Descending>;

  return segmented_sort_impl(
    build,
    d_temp_storage,
    temp_storage_bytes,
    d_keys_in,
    d_keys_out,
    d_values_in,
    d_values_out,
    num_items,
    num_segments,
    start_offset_in,
    end_offset_in,
    is_overwrite_okay,
    selector,
    stream);
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
    delete static_cast<segmented_sort::segmented_sort_runtime_tuning_policy*>(build_ptr->runtime_policy);
    delete static_cast<segmented_sort::partition_runtime_tuning_policy*>(build_ptr->partition_runtime_policy);
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
