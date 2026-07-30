// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

#include "common.cuh"

// Indexed (arg-top-k) variant: each key carries a segment-local index as its value payload. The input values are
// produced by a counting iterator that restarts at 0 for every segment, so indices are not (pre-)materialized in global
// memory
template <typename KeyT, typename IndexT, int MaxSegmentSize, int K>
void decode_style_variable_topk_indexed(
  nvbench::state& state, nvbench::type_list<KeyT, IndexT, nvbench::enum_type<MaxSegmentSize>, nvbench::enum_type<K>>)
{
  if constexpr (K > MaxSegmentSize)
  {
    state.skip("K > MaxSegmentSize.");
    return;
  }

  const auto num_segments                                         = static_cast<int>(state.get_int64("NumSegments"));
  const thrust::device_vector<cuda::std::int64_t> d_segment_sizes = generate(
    static_cast<std::size_t>(num_segments),
    bit_entropy::_1_000,
    static_cast<cuda::std::int64_t>(K),
    static_cast<cuda::std::int64_t>(MaxSegmentSize));
  const auto input_elements  = thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end());
  const auto output_elements = static_cast<std::size_t>(num_segments) * K;
  const auto total_num_items = cuda::args::immediate{static_cast<cuda::std::int64_t>(input_elements)};

  auto in_keys_buffer = gen_data<MaxSegmentSize, K>(
    num_segments, string_to_pattern(state.get_string("Pattern")), thrust::raw_pointer_cast(d_segment_sizes.data()));
  auto out_keys_buffer    = thrust::device_vector<KeyT>(output_elements, thrust::no_init);
  auto out_indices_buffer = thrust::device_vector<IndexT>(output_elements, thrust::no_init);

  auto segment_sizes_param = cuda::args::deferred_sequence{
    thrust::raw_pointer_cast(d_segment_sizes.data()), cuda::args::bounds<1, MaxSegmentSize>()};
  auto k_param            = cuda::args::constant<K>{};
  auto select_direction   = cuda::args::constant<cub::detail::topk::select::max>{};
  auto num_segments_param = cuda::args::immediate{static_cast<cuda::std::int64_t>(num_segments)};

  auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(in_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(MaxSegmentSize));
  auto d_keys_out = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(out_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(K));

  // Input values: every segment maps to the same counting iterator starting at 0, so values are segment-local indices.
  auto d_indices_in  = cuda::make_constant_iterator(cuda::make_counting_iterator(IndexT{0}));
  auto d_indices_out = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(out_indices_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(K));

  state.add_element_count(input_elements, "NumElements");
  state.add_global_memory_reads<KeyT>(input_elements, "InputKeys");
  state.add_global_memory_reads<cuda::std::int64_t>(num_segments, "SegmentSizes");
  state.add_global_memory_writes<KeyT>(output_elements, "OutputKeys");
  state.add_global_memory_writes<IndexT>(output_elements, "OutputIndices");

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    // TODO(bgruber): call the public API once available
    _CCCL_TRY_CUDA_API(
      cub::detail::batched_topk::dispatch_with_env,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      d_indices_in,
      d_indices_out,
      segment_sizes_param,
      k_param,
      select_direction,
      num_segments_param,
      total_num_items,
      env);
  });
}

// Index type is a compile-time axis: i32 for now, extensible to i64.
using index_type_list = nvbench::type_list<cuda::std::int32_t>;

NVBENCH_BENCH_TYPES(decode_style_variable_topk_indexed,
                    NVBENCH_TYPE_AXES(key_type_list, index_type_list, max_segment_size_list, k_list))
  .set_name("decode_style_variable_topk_indexed")
  .set_type_axes_names({"KeyT{ct}", "IndexT{ct}", "MaxSegmentSize{ct}", "K{ct}"})
  .add_int64_axis("NumSegments", {1, 2, 4, 8, 16, 32})
  .add_string_axis("Pattern", valid_patterns);
