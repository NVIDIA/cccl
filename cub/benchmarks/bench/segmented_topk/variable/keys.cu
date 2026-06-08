// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__argument_>
#include <cuda/iterator>

#include <nvbench_helper.cuh>

#include "common.cuh"

template <typename KeyT, int MaxSegmentSize, int K>
void decode_style_variable_topk_keys(
  nvbench::state& state, nvbench::type_list<KeyT, nvbench::enum_type<MaxSegmentSize>, nvbench::enum_type<K>>)
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
  const auto total_num_items = ::cuda::__argument::__immediate{static_cast<cuda::std::int64_t>(input_elements)};

  auto in_keys_buffer = gen_data<MaxSegmentSize, K>(
    num_segments, string_to_pattern(state.get_string("Pattern")), thrust::raw_pointer_cast(d_segment_sizes.data()));
  auto out_keys_buffer = thrust::device_vector<KeyT>(output_elements, thrust::no_init);

  auto segment_sizes_param = ::cuda::__argument::__immediate_sequence{
    thrust::raw_pointer_cast(d_segment_sizes.data()), ::cuda::__argument::__bounds<1, MaxSegmentSize>()};
  auto k_param            = ::cuda::__argument::__constant<K>{};
  auto select_direction   = ::cuda::__argument::__constant<cub::detail::topk::select::max>{};
  auto num_segments_param = ::cuda::__argument::__immediate{static_cast<cuda::std::int64_t>(num_segments)};

  auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(in_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(MaxSegmentSize));
  auto d_keys_out = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(out_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(K));

  state.add_element_count(input_elements, "NumElements");
  state.add_global_memory_reads<KeyT>(input_elements, "InputKeys");
  state.add_global_memory_reads<cuda::std::int64_t>(num_segments, "SegmentSizes");
  state.add_global_memory_writes<KeyT>(output_elements, "OutputKeys");

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    // TODO(bgruber): call the public API once available
    _CCCL_TRY_CUDA_API(
      cub::detail::batched_topk::dispatch_with_env,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes_param,
      k_param,
      select_direction,
      num_segments_param,
      total_num_items,
      env);
  });
}

NVBENCH_BENCH_TYPES(decode_style_variable_topk_keys, NVBENCH_TYPE_AXES(key_type_list, max_segment_size_list, k_list))
  .set_name("decode_style_variable_topk_keys")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "K{ct}"})
  .add_int64_axis("NumSegments", {1, 2, 4, 8, 16, 32})
  .add_string_axis("Pattern", valid_patterns);
