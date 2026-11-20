// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/detail/vector_base.h>
#include <thrust/device_vector.h>

#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <cstddef>
#include <cstdint>

#include <nvbench_helper.cuh>

#include "policy_selector.cuh"

#ifdef TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t>;
#endif

// Generates the input of a variable size segmented reduction from the "Elements{io}" and "MaxSegmentSize" axes and
// enables the throughput calculations of the benchmark state.
template <typename T, typename OffsetT, typename OutputT>
struct variable_problem
{
  thrust::device_vector<OffsetT> segment_offsets;
  thrust::device_vector<T> in;
  thrust::device_vector<OutputT> out;
  std::size_t num_segments;

  _CCCL_HOST_API explicit variable_problem(nvbench::state& state)
      : variable_problem(state,
                         static_cast<std::size_t>(state.get_int64("Elements{io}")),
                         static_cast<std::size_t>(state.get_int64("MaxSegmentSize")))
  {}

  _CCCL_HOST_API const T* d_in() const
  {
    return thrust::raw_pointer_cast(in.data());
  }

  _CCCL_HOST_API OutputT* d_out()
  {
    return thrust::raw_pointer_cast(out.data());
  }

  _CCCL_HOST_API const OffsetT* d_begin_offsets() const
  {
    return thrust::raw_pointer_cast(segment_offsets.data());
  }

  _CCCL_HOST_API const OffsetT* d_end_offsets() const
  {
    return d_begin_offsets() + 1;
  }

private:
  _CCCL_HOST_API variable_problem(nvbench::state& state, std::size_t elements, std::size_t max_segment_size)
      : segment_offsets(generate.uniform.segment_offsets(elements, 1, max_segment_size))
      , in(generate(elements))
      , out(segment_offsets.size() - 1, thrust::default_init)
      , num_segments(segment_offsets.size() - 1)
  {
    state.add_element_count(elements);
    state.add_global_memory_reads<T>(elements, "Size");
    state.add_global_memory_writes<OutputT>(num_segments);
    state.add_global_memory_reads<OffsetT>(num_segments + 1);
  }
};

// Public API, no max segment size guarantee
template <typename T, typename OffsetT>
_CCCL_HOST_API void variable_segmented_reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  static constexpr bool is_argmax = std::is_same_v<op_t, cub::detail::arg_max>;

  using output_t = cuda::std::conditional_t<is_argmax, cub::KeyValuePair<int, T>, T>;

  variable_problem<T, OffsetT, output_t> problem(state);

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(policy_selector<output_t>{})
#endif
    );
    if constexpr (is_argmax)
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceSegmentedReduce::ArgMax,
        "Segmented ArgMax failed",
        problem.d_in(),
        problem.d_out(),
        static_cast<::cuda::std::int64_t>(problem.num_segments),
        problem.d_begin_offsets(),
        problem.d_end_offsets(),
        env);
    }
    else
    {
      _CCCL_TRY_RUNTIME_API(
        cub::DeviceSegmentedReduce::Sum,
        "Segmented sum failed",
        problem.d_in(),
        problem.d_out(),
        static_cast<::cuda::std::int64_t>(problem.num_segments),
        problem.d_begin_offsets(),
        problem.d_end_offsets(),
        env);
    }
  });
}

// Internal dispatch with a max segment size guarantee
template <typename T, typename OffsetT>
_CCCL_HOST_API void variable_with_guarantee_segmented_reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  static constexpr bool is_argmin = std::is_same_v<op_t, cub::detail::arg_min>;
  static constexpr bool is_argmax = std::is_same_v<op_t, cub::detail::arg_max>;

  using raw_input_it_t = const T*;
  using output_t       = cuda::std::conditional_t<(is_argmin || is_argmax), cuda::std::pair<int, T>, T>;
  using output_it_t    = output_t*;
  using accum_t        = output_t;
  using init_value_t =
    cuda::std::conditional_t<(is_argmin || is_argmax), cub::detail::reduce::empty_problem_init_t<accum_t>, T>;
  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t*;
  using end_offset_it_t   = const offset_t*;

  // Retrieve axis parameters
  const auto max_segment_size        = static_cast<std::size_t>(state.get_int64("MaxSegmentSize"));
  const auto guaranteed_max_seg_size = static_cast<std::size_t>(state.get_int64("GuaranteedMaxSegSize"));

  // skip if max_segment_size > guaranteed_max_seg_size
  if (guaranteed_max_seg_size != 0 && max_segment_size > guaranteed_max_seg_size)
  {
    state.skip("max_segment_size > guaranteed_max_seg_size");
    return;
  }

  variable_problem<T, offset_t, output_t> problem(state);
  const auto num_segments           = problem.num_segments;
  raw_input_it_t d_raw_in           = problem.d_in();
  output_it_t d_out                 = problem.d_out();
  begin_offset_it_t d_begin_offsets = problem.d_begin_offsets();
  end_offset_it_t d_end_offsets     = problem.d_end_offsets();

  // Create wrapped iterator for argmin/argmax operations
  [[maybe_unused]] auto d_indexed_in = cuda::make_transform_iterator(
    cuda::counting_iterator<::cuda::std::int64_t>(0),
    cub::detail::segmented_reduce::generate_idx_value<raw_input_it_t, T>(d_raw_in, 1));
  using arg_index_input_iterator_t = decltype(d_indexed_in);

  auto d_in = [&] {
    if constexpr (is_argmin || is_argmax)
    {
      return d_indexed_in;
    }
    else
    {
      return d_raw_in;
    }
  }();

  // Allocate temporary storage
  std::size_t temp_size{};
  using override_offset_t = cuda::std::conditional_t<(is_argmin || is_argmax), int, cub::detail::use_default>;

  // TODO(bgruber): rewrite this to use the public CUB API directly. But in order to do this, we need to expose the
  // guaranteed_max_seg_size at the public API
  auto dispatch_segmented_reduce = [&](void* d_temp_storage, std::size_t& temp_storage_bytes, cudaStream_t stream) {
    return cub::detail::segmented_reduce::dispatch<accum_t, override_offset_t>(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<::cuda::std::int64_t>(num_segments),
      d_begin_offsets,
      d_end_offsets,
      op_t{},
      init_value_t{},
      guaranteed_max_seg_size,
      stream);
  };
  _CCCL_TRY_RUNTIME_API(dispatch_segmented_reduce, "Segmented reduce failed", nullptr, temp_size, nullptr /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    _CCCL_TRY_RUNTIME_API(
      dispatch_segmented_reduce, "Segmented reduce failed", temp_storage, temp_size, launch.get_stream());
  });
}
