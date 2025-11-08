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
#include <cub/detail/ptx-json-parser.h>
#include <cub/device/dispatch/dispatch_segmented_sort.cuh> // cub::DispatchSegmentedSort
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh> // DeviceSegmentedSort kernels
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh> // policy_hub
#include <cub/thread/thread_load.cuh> // cub::LoadModifier

#include <exception> // std::exception
#include <format> // std::format
#include <string> // std::string
#include <string_view> // std::string_view
#include <type_traits> // std::is_same_v

#include "jit_templates/templates/input_iterator.h"
#include "jit_templates/templates/operation.h"
#include "jit_templates/templates/output_iterator.h"
#include "jit_templates/traits.h"
#include "util/context.h"
#include "util/errors.h"
#include "util/indirect_arg.h"
#include "util/types.h"
#include <cccl/c/segmented_sort.h>
#include <cccl/c/types.h> // cccl_type_info
#include <nlohmann/json.hpp>
#include <nvrtc/command_list.h>
#include <nvrtc/ltoir_list_appender.h>
#include <util/build_utils.h>

struct device_segmented_sort_policy;
struct device_three_way_partition_policy;
using OffsetT = ptrdiff_t;
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
  check(cccl_type_name_from_nvrtc<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

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
  check(cccl_type_name_from_nvrtc<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

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
  check(cccl_type_name_from_nvrtc<device_segmented_sort_policy>(&chained_policy_t));

  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

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

struct selector_state_t
{
  OffsetT threshold;
  void* begin_offsets;
  void* end_offsets;
  cub::detail::segmented_sort::global_segment_offset_t base_segment_offset;

  void initialize(OffsetT offset, indirect_iterator_t begin_offset_iterator, indirect_iterator_t end_offset_iterator)
  {
    threshold = offset;
    // If offsets are raw device pointers, unwrap the stored pointer-to-pointer
    // from the iterator state so device code can index it directly.
    begin_offsets       = *static_cast<void**>(begin_offset_iterator.ptr);
    end_offsets         = *static_cast<void**>(end_offset_iterator.ptr);
    base_segment_offset = 0;
  }
};

cccl_op_t make_segments_selector_op(
  OffsetT offset,
  cccl_iterator_t begin_offset_iterator,
  cccl_iterator_t end_offset_iterator,
  const char* selector_op_name,
  const char* comparison,
  const char** compile_args,
  size_t num_compile_args,
  const char** lto_opts,
  size_t num_lto_opts)
{
  cccl_op_t selector_op{};
  auto selector_op_state = std::make_unique<selector_state_t>();
  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  const std::string code = std::format(
    R"XXX(
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh>

extern "C" __device__ void {0}(void* state_ptr, const void* arg_ptr, void* result_ptr)
{{
  using cub::detail::segmented_sort::local_segment_index_t;
  using cub::detail::segmented_sort::global_segment_offset_t;

  struct state_t {{
    {1} threshold;
    void* begin_offsets;
    void* end_offsets;
    global_segment_offset_t base_segment_offset;
  }};

  auto* st = static_cast<state_t*>(state_ptr);
  const local_segment_index_t sid = *static_cast<const local_segment_index_t*>(arg_ptr);
  const global_segment_offset_t index = st->base_segment_offset + static_cast<global_segment_offset_t>(sid);
  const {2} begin = static_cast<const {2}*>(st->begin_offsets)[index];
  const {3} end   = static_cast<const {3}*>(st->end_offsets)[index];
  const bool pred       = (end - begin) {4} st->threshold;
  *static_cast<bool*>(result_ptr) = pred;
}}
)XXX",
    selector_op_name,
    offset_t,
    cccl_type_enum_to_name(begin_offset_iterator.value_type.type),
    cccl_type_enum_to_name(end_offset_iterator.value_type.type),
    comparison);

  selector_op.type = cccl_op_kind_t::CCCL_STATEFUL;
  selector_op.name = selector_op_name;
  auto [lto_size, lto_buf] =
    begin_linking_nvrtc_program(static_cast<uint32_t>(num_lto_opts), lto_opts)
      ->add_program(nvrtc_translation_unit{code.c_str(), selector_op_name})
      ->compile_program({compile_args, num_compile_args})
      ->get_program_ltoir();
  selector_op.code      = lto_buf.release();
  selector_op.code_size = lto_size;
  selector_op.code_type = CCCL_OP_LTOIR;
  selector_op.size      = sizeof(selector_state_t);
  selector_op.alignment = alignof(selector_state_t);
  selector_op.state     = selector_op_state.get();

  selector_op_state->initialize(offset, begin_offset_iterator, end_offset_iterator);
  selector_op_state.release();

  return selector_op;
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

  using LargeSegmentsSelectorT = indirect_arg_t;
  using SmallSegmentsSelectorT = indirect_arg_t;

  indirect_arg_t LargeSegmentsSelector(
    OffsetT offset, indirect_iterator_t begin_offset_iterator, indirect_iterator_t end_offset_iterator) const
  {
    static_cast<selector_state_t*>(build.large_segments_selector_op.state)
      ->initialize(offset, begin_offset_iterator, end_offset_iterator);
    return indirect_arg_t(build.large_segments_selector_op);
  }

  indirect_arg_t SmallSegmentsSelector(
    OffsetT offset, indirect_iterator_t begin_offset_iterator, indirect_iterator_t end_offset_iterator) const
  {
    static_cast<selector_state_t*>(build.small_segments_selector_op.state)
      ->initialize(offset, begin_offset_iterator, end_offset_iterator);
    return indirect_arg_t(build.small_segments_selector_op);
  }

  void SetSegmentOffset(indirect_arg_t& selector, long long base_segment_offset) const
  {
    auto* st                = static_cast<selector_state_t*>(selector.ptr);
    st->base_segment_offset = base_segment_offset;
  }
};

std::string get_three_way_partition_init_kernel_name()
{
  static constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";

  static constexpr std::string_view num_selected_it_t = "cub::detail::segmented_sort::local_segment_index_t*";

  return std::format("cub::detail::three_way_partition::DeviceThreeWayPartitionInitKernel<{0}, {1}>",
                     scan_tile_state_t, // 0
                     num_selected_it_t); // 1
}

std::string get_three_way_partition_kernel_name(std::string_view large_selector_t, std::string_view small_selector_t)
{
  std::string chained_policy_t;
  check(cccl_type_name_from_nvrtc<device_three_way_partition_policy>(&chained_policy_t));

  static constexpr std::string_view input_it_t =
    "thrust::counting_iterator<cub::detail::segmented_sort::local_segment_index_t>";
  static constexpr std::string_view first_out_it_t  = "cub::detail::segmented_sort::local_segment_index_t*";
  static constexpr std::string_view second_out_it_t = "cub::detail::segmented_sort::local_segment_index_t*";
  static constexpr std::string_view unselected_out_it_t =
    "thrust::reverse_iterator<cub::detail::segmented_sort::local_segment_index_t*>";
  static constexpr std::string_view num_selected_it_t = "cub::detail::segmented_sort::local_segment_index_t*";
  static constexpr std::string_view scan_tile_state_t = "cub::detail::three_way_partition::ScanTileStateT";
  std::string offset_t;
  check(cccl_type_name_from_nvrtc<OffsetT>(&offset_t));

  static constexpr std::string_view per_partition_offset_t = "cub::detail::three_way_partition::per_partition_offset_t";
  static constexpr std::string_view streaming_context_t =
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
    large_selector_t, // 7 (SelectFirstPartOp)
    small_selector_t, // 8 (SelectSecondPartOp)
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

  cub::CacheLoadModifier LargeSegmentLoadModifier() const
  {
    return large_segment.LoadModifier();
  }

  cub::BlockLoadAlgorithm LargeSegmentLoadAlgorithm() const
  {
    return large_segment.LoadAlgorithm();
  }

  cub::WarpLoadAlgorithm MediumSegmentLoadAlgorithm() const
  {
    return medium_segment.LoadAlgorithm();
  }

  cub::WarpLoadAlgorithm SmallSegmentLoadAlgorithm() const
  {
    return small_segment.LoadAlgorithm();
  }

  cub::WarpStoreAlgorithm MediumSegmentStoreAlgorithm() const
  {
    return medium_segment.StoreAlgorithm();
  }

  cub::WarpStoreAlgorithm SmallSegmentStoreAlgorithm() const
  {
    return small_segment.StoreAlgorithm();
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
  auto delay_ctor_info = partition_policy["DelayConstructor"];

  std::string delay_ctor_params;
  for (auto&& param : delay_ctor_info["params"])
  {
    delay_ctor_params.append(to_string(param) + ", ");
  }
  delay_ctor_params.erase(delay_ctor_params.size() - 2); // remove last ", "

  return std::format("cub::detail::{}<{}>", delay_ctor_info["name"].get<std::string>(), delay_ctor_params);
}

std::string inject_delay_constructor_into_three_way_policy(
  const std::string& three_way_partition_policy_str, const std::string& delay_constructor_type)
{
  // Insert before the final closing of the struct (right before the sequence "};")
  static constexpr std::string_view needle = "};";
  const auto pos                           = three_way_partition_policy_str.rfind(needle);
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
struct segmented_sort_large_selector_tag;
struct segmented_sort_small_selector_tag;

CUresult cccl_device_segmented_sort_build_ex(
  cccl_device_segmented_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t keys_in_it,
  cccl_iterator_t values_in_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path,
  cccl_build_config* config)
{
  CUresult error = CUDA_SUCCESS;

  if (cccl_iterator_kind_t::CCCL_POINTER != keys_in_it.type || cccl_iterator_kind_t::CCCL_POINTER != values_in_it.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_segmented_sort_build(): keys_in_it and values_in_it must be a pointer\n ");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  if (cccl_iterator_kind_t::CCCL_POINTER != start_offset_it.type
      || cccl_iterator_kind_t::CCCL_POINTER != end_offset_it.type)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_segmented_sort_build(): start_offset_it and end_offset_it must be a pointer\n ");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

  try
  {
    const char* name = "device_segmented_sort";

    const int cc = cc_major * 10 + cc_minor;

    const auto [keys_in_iterator_name, keys_in_iterator_src] =
      get_specialization<segmented_sort_keys_input_iterator_tag>(template_id<input_iterator_traits>(), keys_in_it);

    const bool keys_only = values_in_it.type == cccl_iterator_kind_t::CCCL_POINTER && values_in_it.state == nullptr;

    std::string values_in_iterator_name, values_in_iterator_src;

    if (!keys_only)
    {
      const auto [vi_name, vi_src] = get_specialization<segmented_sort_values_input_iterator_tag>(
        template_id<input_iterator_traits>(), values_in_it);
      values_in_iterator_name = vi_name;
      values_in_iterator_src  = vi_src;
    }
    else
    {
      values_in_iterator_name = "cub::NullType*";
      values_in_iterator_src  = "";
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

    const std::string arch = std::format("-arch=sm_{0}{1}", cc_major, cc_minor);

    std::vector<const char*> selector_compilation_args = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-DCUB_DISABLE_CDP",
      "-std=c++20"};

    cccl::detail::extend_args_with_build_config(selector_compilation_args, config);

    constexpr size_t num_lto_args   = 2;
    const char* lopts[num_lto_args] = {"-lto", arch.c_str()};

    // TODO: we currently compile each selector op separately from the main TU.
    // We do this because we need to pass the selector ops to
    // DispatchThreeWayPartition eventually. This causes increased compilation
    // times, which might be avoidable.
    cccl_op_t large_selector_op = segmented_sort::make_segments_selector_op(
      0,
      start_offset_it,
      end_offset_it,
      "cccl_large_segments_selector_op",
      ">",
      selector_compilation_args.data(),
      selector_compilation_args.size(),
      lopts,
      num_lto_args);
    cccl_op_t small_selector_op = segmented_sort::make_segments_selector_op(
      0,
      start_offset_it,
      end_offset_it,
      "cccl_small_segments_selector_op",
      "<",
      selector_compilation_args.data(),
      selector_compilation_args.size(),
      lopts,
      num_lto_args);

    cccl_type_info selector_result_t{sizeof(bool), alignof(bool), cccl_type_enum::CCCL_BOOLEAN};
    cccl_type_info selector_input_t{
      sizeof(cub::detail::segmented_sort::local_segment_index_t),
      alignof(cub::detail::segmented_sort::local_segment_index_t),
      cccl_type_enum::CCCL_UINT32};

    const auto [large_selector_name, large_selector_src] = get_specialization<segmented_sort_large_selector_tag>(
      template_id<user_operation_traits>(), large_selector_op, selector_result_t, selector_input_t);

    const auto [small_selector_name, small_selector_src] = get_specialization<segmented_sort_small_selector_tag>(
      template_id<user_operation_traits>(), small_selector_op, selector_result_t, selector_input_t);

    const auto segmented_sort_policy_hub_expr = std::format(
      "cub::detail::segmented_sort::policy_hub<{0}, {1}>",
      key_t, // 0
      value_t); // 1

    static constexpr std::string_view three_way_partition_policy_hub_expr =
      "cub::detail::three_way_partition::policy_hub<cub::detail::segmented_sort::local_segment_index_t, "
      "cub::detail::three_way_partition::per_partition_offset_t>";

    const std::string final_src = std::format(
      R"XXX(
#include <cub/device/dispatch/kernels/kernel_segmented_sort.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_sort.cuh>
#include <cub/device/dispatch/kernels/kernel_three_way_partition.cuh>
#include <cub/device/dispatch/tuning/tuning_three_way_partition.cuh>

{0}

#include <thrust/iterator/counting_iterator.h> // used in three_way_partition kernel
#include <thrust/iterator/reverse_iterator.h> // used in three_way_partition kernel
#include <cub/detail/choose_offset.cuh> // used in three_way_partition kernel

struct __align__({2}) storage_t {{
  char data[{1}];
}};
struct __align__({4}) items_storage_t {{
  char data[{3}];
}};
{5}
{6}
{7}
{8}
{9}
{10}
using device_segmented_sort_policy = {11}::MaxPolicy;
using device_three_way_partition_policy = {12}::MaxPolicy;

#include <cub/detail/ptx-json/json.h>
__device__ consteval auto& segmented_sort_policy_generator() {{
  return ptx_json::id<ptx_json::string("device_segmented_sort_policy")>()
    = cub::detail::segmented_sort::SegmentedSortPolicyWrapper<device_segmented_sort_policy::ActivePolicy>::EncodedPolicy();
}}
__device__ consteval auto& three_way_partition_policy_generator() {{
  return ptx_json::id<ptx_json::string("device_three_way_partition_policy")>()
    = cub::detail::three_way_partition::ThreeWayPartitionPolicyWrapper<device_three_way_partition_policy::ActivePolicy>::EncodedPolicy();
}}
)XXX",
      jit_template_header_contents, // 0
      keys_in_it.value_type.size, // 1
      keys_in_it.value_type.alignment, // 2
      values_in_it.value_type.size, // 3
      values_in_it.value_type.alignment, // 4
      keys_in_iterator_src, // 5
      values_in_iterator_src, // 6
      start_offset_iterator_src, // 7
      end_offset_iterator_src, // 8
      large_selector_src, // 9
      small_selector_src, // 10
      segmented_sort_policy_hub_expr, // 11
      three_way_partition_policy_hub_expr); // 12

    std::vector<const char*> args = {
      arch.c_str(),
      cub_path,
      thrust_path,
      libcudacxx_path,
      ctk_path,
      "-rdc=true",
      "-dlto",
      "-DCUB_DISABLE_CDP",
      "-DCUB_ENABLE_POLICY_PTX_JSON",
      "-std=c++20"};

    cccl::detail::extend_args_with_build_config(args, config);

    std::string segmented_sort_fallback_kernel_name = segmented_sort::get_device_segmented_sort_fallback_kernel_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string segmented_sort_kernel_small_name = segmented_sort::get_device_segmented_sort_kernel_small_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string segmented_sort_kernel_large_name = segmented_sort::get_device_segmented_sort_kernel_large_name(
      start_offset_iterator_name, end_offset_iterator_name, key_t, value_t, sort_order);

    std::string three_way_partition_init_kernel_name = segmented_sort::get_three_way_partition_init_kernel_name();

    std::string three_way_partition_kernel_name =
      segmented_sort::get_three_way_partition_kernel_name(large_selector_name, small_selector_name);

    std::string segmented_sort_fallback_kernel_lowered_name;
    std::string segmented_sort_kernel_small_lowered_name;
    std::string segmented_sort_kernel_large_lowered_name;
    std::string three_way_partition_init_kernel_lowered_name;
    std::string three_way_partition_kernel_lowered_name;

    // Collect all LTO-IRs to be linked.
    nvrtc_linkable_list linkable_list;
    nvrtc_linkable_list_appender appender{linkable_list};

    // add iterator definitions
    appender.add_iterator_definition(keys_in_it);
    if (!keys_only)
    {
      appender.add_iterator_definition(values_in_it);
    }
    appender.add_iterator_definition(start_offset_it);
    appender.add_iterator_definition(end_offset_it);

    appender.append_operation(large_selector_op);
    appender.append_operation(small_selector_op);

    nvrtc_link_result result =
      begin_linking_nvrtc_program(num_lto_args, lopts)
        ->add_program(nvrtc_translation_unit{final_src.c_str(), name})
        ->add_expression({segmented_sort_fallback_kernel_name})
        ->add_expression({segmented_sort_kernel_small_name})
        ->add_expression({segmented_sort_kernel_large_name})
        ->add_expression({three_way_partition_init_kernel_name})
        ->add_expression({three_way_partition_kernel_name})
        ->compile_program({args.data(), args.size()})
        ->get_name({segmented_sort_fallback_kernel_name, segmented_sort_fallback_kernel_lowered_name})
        ->get_name({segmented_sort_kernel_small_name, segmented_sort_kernel_small_lowered_name})
        ->get_name({segmented_sort_kernel_large_name, segmented_sort_kernel_large_lowered_name})
        ->get_name({three_way_partition_init_kernel_name, three_way_partition_init_kernel_lowered_name})
        ->get_name({three_way_partition_kernel_name, three_way_partition_kernel_lowered_name})
        ->link_program()
        ->add_link_list(linkable_list)
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

    nlohmann::json runtime_policy =
      cub::detail::ptx_json::parse("device_segmented_sort_policy", {result.data.get(), result.size});

    using cub::detail::RuntimeRadixSortDownsweepAgentPolicy;
    auto large_segment_policy = RuntimeRadixSortDownsweepAgentPolicy::from_json(runtime_policy, "LargeSegmentPolicy");

    using cub::detail::RuntimeSubWarpMergeSortAgentPolicy;
    auto small_segment_policy = RuntimeSubWarpMergeSortAgentPolicy::from_json(runtime_policy, "SmallSegmentPolicy");

    auto medium_segment_policy = RuntimeSubWarpMergeSortAgentPolicy::from_json(runtime_policy, "MediumSegmentPolicy");

    int partitioning_threshold = runtime_policy["PartitioningThreshold"].get<int>();
    nlohmann::json partition_policy =
      cub::detail::ptx_json::parse("device_three_way_partition_policy", {result.data.get(), result.size});

    using cub::detail::RuntimeThreeWayPartitionAgentPolicy;
    auto three_way_partition_policy =
      RuntimeThreeWayPartitionAgentPolicy::from_json(partition_policy, "ThreeWayPartitionPolicy");

    build_ptr->cc                         = cc;
    build_ptr->large_segments_selector_op = large_selector_op;
    build_ptr->small_segments_selector_op = small_selector_op;
    build_ptr->cubin                      = (void*) result.data.release();
    build_ptr->cubin_size                 = result.size;
    build_ptr->key_type                   = keys_in_it.value_type;
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

CUresult cccl_device_segmented_sort_build(
  cccl_device_segmented_sort_build_result_t* build_ptr,
  cccl_sort_order_t sort_order,
  cccl_iterator_t keys_in_it,
  cccl_iterator_t values_in_it,
  cccl_iterator_t start_offset_it,
  cccl_iterator_t end_offset_it,
  int cc_major,
  int cc_minor,
  const char* cub_path,
  const char* thrust_path,
  const char* libcudacxx_path,
  const char* ctk_path)
{
  return cccl_device_segmented_sort_build_ex(
    build_ptr,
    sort_order,
    keys_in_it,
    values_in_it,
    start_offset_it,
    end_offset_it,
    cc_major,
    cc_minor,
    cub_path,
    thrust_path,
    libcudacxx_path,
    ctk_path,
    nullptr);
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
  uint64_t num_items,
  uint64_t num_segments,
  cccl_iterator_t start_offset_in,
  cccl_iterator_t end_offset_in,
  bool is_overwrite_okay,
  int* selector,
  CUstream stream)
{
  if (selector == nullptr)
  {
    fflush(stderr);
    printf("\nERROR in cccl_device_segmented_sort(): selector cannot be nullptr\n");
    fflush(stdout);
    return CUDA_ERROR_UNKNOWN;
  }

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

    *selector = d_keys_double_buffer.selector;
    error     = static_cast<CUresult>(exec_status);
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
  uint64_t num_items,
  uint64_t num_segments,
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

    // Clean up the selector op states
    std::unique_ptr<segmented_sort::selector_state_t> large_state(
      static_cast<segmented_sort::selector_state_t*>(build_ptr->large_segments_selector_op.state));
    std::unique_ptr<segmented_sort::selector_state_t> small_state(
      static_cast<segmented_sort::selector_state_t*>(build_ptr->small_segments_selector_op.state));

    // Clean up the selector op code buffers
    std::unique_ptr<char[]> large_code(const_cast<char*>(build_ptr->large_segments_selector_op.code));
    std::unique_ptr<char[]> small_code(const_cast<char*>(build_ptr->small_segments_selector_op.code));

    // Clean up the runtime policies
    std::unique_ptr<segmented_sort::segmented_sort_runtime_tuning_policy> rtp(
      static_cast<segmented_sort::segmented_sort_runtime_tuning_policy*>(build_ptr->runtime_policy));
    std::unique_ptr<segmented_sort::partition_runtime_tuning_policy> prtp(
      static_cast<segmented_sort::partition_runtime_tuning_policy*>(build_ptr->partition_runtime_policy));
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
