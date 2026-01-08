// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/agent/agent_warp_segmented_scan.cuh>
#include <cub/device/device_segmented_scan.cuh>

#include <thrust/tabulate.h>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename AccumT>
struct policy_hub_t
{
  template <int Nominal4ByteBlockThreads,
            int Nominal4ByteItemsPerThread,
            typename ComputeT,
            cub::BlockLoadAlgorithm LoadAlgorithm,
            cub::CacheLoadModifier LoadModifier,
            cub::BlockStoreAlgorithm StoreAlgorithm,
            cub::BlockScanAlgorithm ScanAlgorithm,
            int SegmentsPerBlock = 1>
  using agent_policy_t = cub::detail::segmented_scan::agent_segmented_scan_policy_t<
    Nominal4ByteBlockThreads,
    Nominal4ByteItemsPerThread,
    ComputeT,
    LoadAlgorithm,
    LoadModifier,
    StoreAlgorithm,
    ScanAlgorithm,
    SegmentsPerBlock,
    cub::detail::MemBoundScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, ComputeT>>;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using SegmentedScanPolicyT =
      agent_policy_t<TUNE_THREADS,
                     TUNE_ITEMS,
                     AccumT,
                     TUNE_LOAD_ALGORITHM,
                     TUNE_LOAD_MODIFIER,
                     TUNE_STORE_ALGORITHM,
                     cub::BLOCK_SCAN_WARP_SCANS,
                     TUNE_SEGMENTS_PER_BLOCK>;
  };

  using MaxPolicy = policy_t;
};
#else

template <typename ComputeT, int NumSegmentsPerWorkUnit>
using segmented_scan_compute_t =
  ::cuda::std::conditional_t<NumSegmentsPerWorkUnit == 1, ComputeT, ::cuda::std::tuple<bool, ComputeT>>;

template <typename AccumT, int SegmentsPerWorkUnit, int ItemsPerThread>
struct user_policy_hub_t
{
  template <int Nominal4ByteBlockThreads,
            int Nominal4ByteItemsPerThread,
            typename ComputeT,
            cub::BlockLoadAlgorithm LoadAlgorithm,
            cub::CacheLoadModifier LoadModifier,
            cub::BlockStoreAlgorithm StoreAlgorithm,
            cub::BlockScanAlgorithm ScanAlgorithm,
            int _SegmentsPerBlock = 1>
  using block_level_agent_policy_t = cub::detail::segmented_scan::agent_segmented_scan_policy_t<
    Nominal4ByteBlockThreads,
    Nominal4ByteItemsPerThread,
    ComputeT,
    LoadAlgorithm,
    LoadModifier,
    StoreAlgorithm,
    ScanAlgorithm,
    _SegmentsPerBlock,
    cub::detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                 Nominal4ByteItemsPerThread,
                                 segmented_scan_compute_t<ComputeT, _SegmentsPerBlock>>>;
  template <int Nominal4ByteBlockThreads,
            int Nominal4ByteItemsPerThread,
            typename ComputeT,
            cub::WarpLoadAlgorithm LoadAlgorithm,
            cub::CacheLoadModifier LoadModifier,
            cub::WarpStoreAlgorithm StoreAlgorithm,
            int _SegmentsPerWarp = 1>
  using warp_level_agent_policy_t = cub::detail::segmented_scan::agent_warp_segmented_scan_policy_t<
    Nominal4ByteBlockThreads,
    Nominal4ByteItemsPerThread,
    ComputeT,
    LoadAlgorithm,
    LoadModifier,
    StoreAlgorithm,
    _SegmentsPerWarp,
    cub::detail::MemBoundScaling<Nominal4ByteBlockThreads,
                                 Nominal4ByteItemsPerThread,
                                 segmented_scan_compute_t<ComputeT, _SegmentsPerWarp>>>;

  using base_block_level_policy_t =
    typename cub::detail::segmented_scan::policy_hub<void, void, AccumT, void, void>::MaxPolicy::segmented_scan_policy_t;

  using base_warp_level_policy_t = typename cub::detail::segmented_scan::policy_hub<void, void, AccumT, void, void>::
    MaxPolicy::warp_segmented_scan_policy_t;

  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using segmented_scan_policy_t = block_level_agent_policy_t<
      128,
      ItemsPerThread,
      AccumT,
      base_block_level_policy_t::load_algorithm,
      base_block_level_policy_t::load_modifier,
      base_block_level_policy_t::store_algorithm,
      base_block_level_policy_t::scan_algorithm,
      SegmentsPerWorkUnit>;

    using warp_segmented_scan_policy_t = warp_level_agent_policy_t<
      128,
      ItemsPerThread,
      AccumT,
      base_warp_level_policy_t::load_algorithm,
      base_warp_level_policy_t::load_modifier,
      base_warp_level_policy_t::store_algorithm,
      SegmentsPerWorkUnit>;
  };

  using MaxPolicy = policy_t;
};

#endif // TUNE_BASE

template <typename OffsetT>
struct to_offsets_functor
{
  OffsetT elements;
  OffsetT segment_size;

  __host__ __device__ __forceinline__ OffsetT operator()(size_t i) const
  {
    return cuda::std::min(elements, static_cast<OffsetT>(i) * segment_size);
  }
};

template <typename T, typename OffsetT, int segments_per_block, int items_per_thread>
static void basic(
  nvbench::state& state,
  nvbench::
    type_list<T, OffsetT, std::integral_constant<int, segments_per_block>, std::integral_constant<int, items_per_thread>>)
{
  using init_t         = T;
  using wrapped_init_t = cub::detail::InputValue<init_t>;
  using accum_t        = cuda::std::__accumulator_t<op_t, init_t, T>;
  using input_it_t     = const T*;
  using output_it_t    = T*;
  using offset_t       = cub::detail::choose_offset_t<OffsetT>;
  using offset_it      = const offset_t*;

#if !TUNE_BASE
  using policy_t   = policy_hub_t<accum_t>;
  using dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    input_it_t,
    output_it_t,
    offset_it,
    offset_it,
    offset_it,
    op_t,
    wrapped_init_t,
    accum_t,
    cub::ForceInclusive::Yes,
    offset_t,
    policy_t>;
#else
  using dispatch_t = cub::detail::segmented_scan::dispatch_segmented_scan<
    input_it_t,
    output_it_t,
    offset_it,
    offset_it,
    offset_it,
    op_t,
    wrapped_init_t,
    accum_t,
    cub::ForceInclusive::Yes,
    offset_t,
    user_policy_hub_t<accum_t, segments_per_block, items_per_thread>>;
#endif

  const auto elements     = static_cast<offset_t>(state.get_int64("Elements{io}"));
  const auto segment_size = static_cast<offset_t>(state.get_int64("SegmentSize{io}"));
  const auto num_segments = cuda::ceil_div(elements, segment_size);
  auto& summary           = state.add_summary("NumSegments");
  summary.set_string("name", "Number of Segments");
  summary.set_int64("value", num_segments);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements);

  thrust::device_vector<offset_t> offsets(num_segments + 1, thrust::no_init);
  thrust::tabulate(offsets.begin(), offsets.end(), to_offsets_functor<offset_t>{elements, segment_size});

  T* d_input          = thrust::raw_pointer_cast(input.data());
  T* d_output         = thrust::raw_pointer_cast(output.data());
  offset_it d_offsets = thrust::raw_pointer_cast(offsets.data());

  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<T>(elements);

  size_t tmp_size;
  dispatch_t::dispatch(
    nullptr,
    tmp_size,
    d_input,
    d_output,
    num_segments,
    d_offsets,
    d_offsets + 1,
    d_offsets,
    op_t{},
    wrapped_init_t{T{}},
    0 /* stream */);

  thrust::device_vector<nvbench::uint8_t> tmp(tmp_size, thrust::no_init);
  nvbench::uint8_t* d_tmp = thrust::raw_pointer_cast(tmp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    dispatch_t::dispatch(
      thrust::raw_pointer_cast(tmp.data()),
      tmp_size,
      d_input,
      d_output,
      num_segments,
      d_offsets,
      d_offsets + 1,
      d_offsets,
      op_t{},
      wrapped_init_t{T{}},
      launch.get_stream());
  });
}

#ifdef TUNE_T
using bench_types = nvbench::type_list<TUNE_T>;
#else
using bench_types = all_types;
#endif

#ifdef TUNE_SEGMENTS_PER_BLOCK
using segments_per_block = nvbench::type_list<std::integral_constant<int, TUNE_SEGMENTS_PER_BLOCK>>;
#else
using segments_per_block =
  nvbench::type_list<std::integral_constant<int, 1>,
                     std::integral_constant<int, 2>,
                     std::integral_constant<int, 3>,
                     std::integral_constant<int, 4>,
                     std::integral_constant<int, 6>,
                     std::integral_constant<int, 8>,
                     std::integral_constant<int, 16>>;

#endif

#ifdef TUNE_ITEMS
using itps = nvbench::type_list<std::integral_constant<int, TUNE_ITEMS>>;
#else
using itps =
  nvbench::type_list<std::integral_constant<int, 9>,
                     std::integral_constant<int, 7>,
                     std::integral_constant<int, 5>,
                     std::integral_constant<int, 3>,
                     std::integral_constant<int, 1>>;
#endif
NVBENCH_BENCH_TYPES(basic, NVBENCH_TYPE_AXES(bench_types, offset_types, segments_per_block, itps))
  .set_name("base")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}", "SegmentsPerBlock{ct}", "ItemsPerThread{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {51, 123, 233, 513, 1337, 4417});
