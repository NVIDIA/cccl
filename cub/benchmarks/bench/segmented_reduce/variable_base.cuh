// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/device/dispatch/dispatch_segmented_reduce.cuh>

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include <nvbench_helper.cuh>

#if TUNE_T
using value_types = nvbench::type_list<TUNE_T>;
#else
using value_types = nvbench::type_list<int32_t, int64_t, float, double>;
#endif

#ifdef TUNE_OffsetT
using some_offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using some_offset_types = nvbench::type_list<int32_t>;
#endif

template <typename T, typename OffsetT>
void variable_segmented_reduce(nvbench::state& state, nvbench::type_list<T, OffsetT>)
{
  static constexpr bool is_argmin = std::is_same_v<op_t, cub::detail::arg_min>;
  static constexpr bool is_argmax = std::is_same_v<op_t, cub::detail::arg_max>;

  using raw_input_it_t = const T*;
  using output_t       = cuda::std::conditional_t<(is_argmin || is_argmax), cuda::std::pair<int, T>, T>;
  using output_it_t    = output_t*;
  using accum_t        = output_t;
  using init_t =
    cuda::std::conditional_t<(is_argmin || is_argmax), cub::detail::reduce::empty_problem_init_t<accum_t>, T>;
  using offset_t          = OffsetT;
  using begin_offset_it_t = const offset_t*;
  using end_offset_it_t   = const offset_t*;

  // Retrieve axis parameters
  const auto elements                = static_cast<std::size_t>(state.get_int64("Elements{io}"));
  const auto max_segment_size        = static_cast<std::size_t>(state.get_int64("MaxSegmentSize"));
  const auto guaranteed_max_seg_size = static_cast<std::size_t>(state.get_int64("GuaranteedMaxSegSize"));

  // skip if max_segment_size > guaranteed_max_seg_size
  if (guaranteed_max_seg_size != 0 && max_segment_size > guaranteed_max_seg_size)
  {
    state.skip("max_segment_size > guaranteed_max_seg_size");
    return;
  }

  const auto min_segment_size     = 1;
  const auto max_segment_size_log = static_cast<offset_t>(std::log2(max_segment_size));

  // Generate segment offsets
  thrust::device_vector<offset_t> segment_offsets =
    generate.uniform.segment_offsets(elements, min_segment_size, max_segment_size);
  const auto num_segments = segment_offsets.size() - 1;

  // Generate input data
  thrust::device_vector<T> in = generate(elements);

  thrust::device_vector<output_t> out(num_segments, thrust::default_init);

  raw_input_it_t d_raw_in           = thrust::raw_pointer_cast(in.data());
  output_it_t d_out                 = thrust::raw_pointer_cast(out.data());
  begin_offset_it_t d_begin_offsets = thrust::raw_pointer_cast(segment_offsets.data());
  end_offset_it_t d_end_offsets     = d_begin_offsets + 1;

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

  // Enable throughput calculations
  state.add_element_count(elements);
  state.add_global_memory_reads<T>(elements, "Size");
  state.add_global_memory_writes<output_t>(num_segments);
  state.add_global_memory_reads<offset_t>(num_segments + 1);

  // Allocate temporary storage
  std::size_t temp_size{};
  using override_offset_t = cuda::std::conditional_t<(is_argmin || is_argmax), int, cub::detail::use_default>;

  cub::detail::segmented_reduce::dispatch<accum_t, override_offset_t>(
    nullptr,
    temp_size,
    d_in,
    d_out,
    static_cast<::cuda::std::int64_t>(num_segments),
    d_begin_offsets,
    d_end_offsets,
    op_t{},
    init_t{},
    guaranteed_max_seg_size,
    nullptr /* stream */);

  thrust::device_vector<nvbench::uint8_t> temp(temp_size, thrust::no_init);
  auto* temp_storage = thrust::raw_pointer_cast(temp.data());

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    cub::detail::segmented_reduce::dispatch<accum_t, override_offset_t>(
      temp_storage,
      temp_size,
      d_in,
      d_out,
      static_cast<::cuda::std::int64_t>(num_segments),
      d_begin_offsets,
      d_end_offsets,
      op_t{},
      init_t{},
      guaranteed_max_seg_size,
      launch.get_stream());
  });
}

NVBENCH_BENCH_TYPES(variable_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_default")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 16, 1))
  .add_int64_axis("GuaranteedMaxSegSize", {0});

// Small segments: 1-16 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_small_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(1, 4, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(1, 4, 1));

// Medium segments: 32-256 items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_medium_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(5, 8, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(5, 8, 1));

// Large segments: 512+ items per segment
NVBENCH_BENCH_TYPES(variable_segmented_reduce, NVBENCH_TYPE_AXES(value_types, some_offset_types))
  .set_name("variable_large_dynamic")
  .set_type_axes_names({"T{ct}", "OffsetT{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4))
  .add_int64_power_of_two_axis("MaxSegmentSize", nvbench::range(9, 16, 1))
  .add_int64_power_of_two_axis("GuaranteedMaxSegSize", nvbench::range(9, 16, 1));
