// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/agent/agent_thread_segmented_scan.cuh>
#include <cub/agent/agent_warp_segmented_scan.cuh>
#include <cub/device/device_segmented_scan.cuh>

#include <thrust/tabulate.h>

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

#if !TUNE_BASE
#  if TUNE_TRANSPOSE == 0
#    define TUNE_BLOCK_LOAD_ALGORITHM  cub::BLOCK_LOAD_DIRECT
#    define TUNE_BLOCK_STORE_ALGORITHM cub::BLOCK_STORE_DIRECT
#    define TUNE_WARP_LOAD_ALGORITHM   cub::WARP_LOAD_DIRECT
#    define TUNE_WARP_STORE_ALGORITHM  cub::WARP_STORE_DIRECT
#  else // TUNE_TRANSPOSE == 1
#    define TUNE_BLOCK_LOAD_ALGORITHM  cub::BLOCK_LOAD_WARP_TRANSPOSE
#    define TUNE_BLOCK_STORE_ALGORITHM cub::BLOCK_STORE_WARP_TRANSPOSE
#    define TUNE_WARP_LOAD_ALGORITHM   cub::WARP_LOAD_TRANSPOSE
#    define TUNE_WARP_STORE_ALGORITHM  cub::WARP_STORE_TRANSPOSE
#  endif // TUNE_TRANSPOSE

#  if TUNE_LOAD == 0
#    define TUNE_LOAD_MODIFIER cub::LOAD_DEFAULT
#  elif TUNE_LOAD == 1
#    define TUNE_LOAD_MODIFIER cub::LOAD_CA
#  endif // TUNE_LOAD

template <typename ComputeT, int NumSegmentsPerWorkUnit>
using segmented_scan_compute_t =
  ::cuda::std::conditional_t<NumSegmentsPerWorkUnit == 1, ComputeT, ::cuda::std::tuple<ComputeT, bool>>;

template <int Nominal4ByteBlockThreads,
          int Nominal4ByteItemsPerThread,
          typename ComputeT,
          cub::BlockLoadAlgorithm LoadAlgorithm,
          cub::CacheLoadModifier LoadModifier,
          cub::BlockStoreAlgorithm StoreAlgorithm,
          cub::BlockScanAlgorithm ScanAlgorithm,
          int MaxSegmentsPerBlock = 1>
using block_level_agent_policy_t = cub::detail::segmented_scan::agent_segmented_scan_policy_t<
  Nominal4ByteBlockThreads,
  Nominal4ByteItemsPerThread,
  ComputeT,
  LoadAlgorithm,
  LoadModifier,
  StoreAlgorithm,
  ScanAlgorithm,
  MaxSegmentsPerBlock,
  cub::detail::NoScaling<Nominal4ByteBlockThreads,
                         Nominal4ByteItemsPerThread,
                         segmented_scan_compute_t<ComputeT, MaxSegmentsPerBlock>>>;

template <int Nominal4ByteBlockThreads,
          int Nominal4ByteItemsPerThread,
          typename ComputeT,
          cub::WarpLoadAlgorithm LoadAlgorithm,
          cub::CacheLoadModifier LoadModifier,
          cub::WarpStoreAlgorithm StoreAlgorithm,
          int MaxSegmentsPerWarp = 1>
using warp_level_agent_policy_t = cub::detail::segmented_scan::agent_warp_segmented_scan_policy_t<
  Nominal4ByteBlockThreads,
  Nominal4ByteItemsPerThread,
  ComputeT,
  LoadAlgorithm,
  LoadModifier,
  StoreAlgorithm,
  MaxSegmentsPerWarp,
  cub::detail::NoScaling<Nominal4ByteBlockThreads,
                         Nominal4ByteItemsPerThread,
                         segmented_scan_compute_t<ComputeT, MaxSegmentsPerWarp>>>;

template <int Nominal4ByteBlockThreads,
          int Nominal4ByteItemsPerThread,
          typename ComputeT,
          cub::CacheLoadModifier LoadModifier>
using thread_level_agent_policy_t = cub::detail::segmented_scan::agent_thread_segmented_scan_policy_t<
  Nominal4ByteBlockThreads,
  Nominal4ByteItemsPerThread,
  ComputeT,
  LoadModifier,
  cub::detail::NoScaling<Nominal4ByteBlockThreads, Nominal4ByteItemsPerThread, ComputeT>>;

template <typename AccumT>
struct policy_hub_t
{
  struct policy_t : cub::ChainedPolicy<300, policy_t, policy_t>
  {
    using block_segmented_scan_policy_t = block_level_agent_policy_t<
      TUNE_THREADS,
      TUNE_ITEMS,
      AccumT,
      TUNE_BLOCK_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_BLOCK_STORE_ALGORITHM,
      cub::BLOCK_SCAN_WARP_SCANS,
      TUNE_MAX_SEGMENTS_PER_BLOCK>;

    using warp_segmented_scan_policy_t = warp_level_agent_policy_t<
      TUNE_THREADS,
      TUNE_ITEMS,
      AccumT,
      TUNE_WARP_LOAD_ALGORITHM,
      TUNE_LOAD_MODIFIER,
      TUNE_WARP_STORE_ALGORITHM,
      // IMPORTANT: Make sure not to hurt occupancy
      // since shared memory is allocated per warp, scale it down so that total amount of shared memory
      // per CTA is the same as in block-level segmented scan
      ::cuda::ceil_div(TUNE_MAX_SEGMENTS_PER_BLOCK* cub::detail::warp_threads, TUNE_THREADS)>;

    using thread_segmented_scan_policy_t =
      thread_level_agent_policy_t<TUNE_THREADS, TUNE_ITEMS, AccumT, TUNE_LOAD_MODIFIER>;
  };

  using MaxPolicy = policy_t;
};
#endif // TUNE_BASE

template <typename OffsetT>
struct to_offsets_functor
{
  OffsetT elements;
  OffsetT segment_size;
  OffsetT wobble;

  __host__ __device__ __forceinline__ OffsetT operator()(size_t i) const
  {
    const auto fixed_size_value = static_cast<OffsetT>(i) * segment_size;
    const auto correction       = ((i & 1) ? wobble : OffsetT{0});

    return cuda::std::min(elements, fixed_size_value + correction);
  }
};

template <size_t Wobble = 0, typename T, typename OffsetT>
static void bench_impl(nvbench::state& state, nvbench::type_list<T, OffsetT>)
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
    cub::ForceInclusive::No,
    offset_t>;
#endif

  const auto elements     = static_cast<offset_t>(state.get_int64("Elements{io}"));
  const auto segment_size = static_cast<offset_t>(state.get_int64("SegmentSize{io}"));
  const auto num_segments = cuda::ceil_div(elements, segment_size);
  auto& summary           = state.add_summary("NumSegments");
  summary.set_string("name", "Number of Segments");
  summary.set_int64("value", num_segments);

  thrust::device_vector<T> input = generate(elements);
  thrust::device_vector<T> output(elements, thrust::no_init);

  thrust::device_vector<offset_t> offsets(num_segments + 1, thrust::no_init);
  thrust::tabulate(offsets.begin(), offsets.end(), to_offsets_functor<offset_t>{elements, segment_size, Wobble});

  const T* d_input    = thrust::raw_pointer_cast(input.data());
  T* d_output         = thrust::raw_pointer_cast(output.data());
  offset_it d_offsets = thrust::raw_pointer_cast(offsets.data());

  state.add_element_count(elements, "Elements");
  state.add_global_memory_reads<T>(elements);
  state.add_global_memory_reads<offset_t>(num_segments + 1);
  state.add_global_memory_writes<T>(elements);

  int num_segments_per_worker = static_cast<int>(state.get_int64("SegmentsPerWorker{io}"));

  if (num_segments_per_worker < 1)
  {
    state.skip("Number of segments parameter is not positive");
    return;
  }

  cub::detail::segmented_scan::worker worker_choice = [](auto token) {
    if (token == "block")
    {
      return cub::detail::segmented_scan::worker::block;
    }
    else if (token == "warp")
    {
      return cub::detail::segmented_scan::worker::warp;
    }
    else if (token == "thread")
    {
      return cub::detail::segmented_scan::worker::thread;
    }
    else
    {
      throw std::runtime_error("Unrecognized value of Worker{io} axis value. Expected 'block', 'wapr', or 'thread'.");
    }
  }(state.get_string("Worker{io}"));

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
    num_segments_per_worker,
    worker_choice,
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
      num_segments_per_worker,
      worker_choice,
      launch.get_stream());
  });
}

template <typename T, typename OffsetT>
static void fixed_segment_size_bench(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  return bench_impl<0, T, OffsetT>(state, tl);
}

template <typename T, typename OffsetT>
static void varying_segment_size_bench(nvbench::state& state, nvbench::type_list<T, OffsetT> tl)
{
  return bench_impl<1, T, OffsetT>(state, tl);
}

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

NVBENCH_BENCH_TYPES(fixed_segment_size_bench, NVBENCH_TYPE_AXES(benched_value_types, offset_types))
  .set_name("fixed_size_segments")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {51, 123, 233, 513, 1337, 4417})
  .add_int64_axis("SegmentsPerWorker{io}", {1})
  .add_string_axis("Worker{io}", {"block"});

NVBENCH_BENCH_TYPES(varying_segment_size_bench, NVBENCH_TYPE_AXES(benched_value_types, offset_types))
  .set_name("varying_size_segments")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(18, 26, 4))
  .add_int64_axis("SegmentSize{io}", {51, 123, 233, 513, 1337, 4417})
  .add_int64_axis("SegmentsPerWorker{io}", {1})
  .add_string_axis("Worker{io}", {"block"});
