// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk_cluster.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__argument_>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>

#include <algorithm>
#include <vector>

#include <nvbench_helper.cuh>

#include "common.cuh"

enum class topk_backend
{
  baseline,
  cluster,
  device,
};

inline constexpr topk_backend selected_backend = topk_backend::baseline;

// Env-based dispatch over the selected backend. The cluster and baseline backends route through their respective
// `dispatch_with_env` entry points (temporary storage is allocated from the memory resource carried by `env`); the
// device backend issues one `cub::DeviceTopK::MaxKeys` per segment, reading the host-side segment sizes.
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename SelectDirectionParamT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename HostSegSizeT,
          typename EnvT>
CUB_RUNTIME_FUNCTION static cudaError_t batched_topk_keys(
  KeyInputItItT d_keys_in,
  KeyOutputItItT d_keys_out,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  SelectDirectionParamT select_direction,
  NumSegmentsParameterT num_segments,
  TotalNumItemsGuaranteeT total_num_items,
  [[maybe_unused]] const HostSegSizeT* h_segment_sizes,
  EnvT env)
{
  if constexpr (selected_backend == topk_backend::cluster)
  {
    return cub::detail::batched_topk_cluster::dispatch_with_env(
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      select_direction,
      num_segments,
      total_num_items,
      env);
  }
  else if constexpr (selected_backend == topk_backend::device)
  {
    using num_segments_val_t = typename ::cuda::__argument::__traits<NumSegmentsParameterT>::element_type;
    const auto num_segs      = cub::detail::params::get_param(num_segments, num_segments_val_t{0});

    // The per-segment device backend uses the unsorted / not-guaranteed-determinism fast path. Layer the requirement
    // on top of the benchmark environment (which carries the stream and the caching memory resource).
    auto seg_env = cuda::std::execution::env{
      env,
      cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                               cuda::execution::output_ordering::unsorted)};

    for (num_segments_val_t i = 0; i < num_segs; ++i)
    {
      const auto k_value  = cub::detail::params::get_param(k, i);
      const auto seg_size = h_segment_sizes[i];
      if (const auto err = cub::DeviceTopK::MaxKeys(
            d_keys_in[i],
            d_keys_out[i],
            static_cast<cuda::std::int64_t>(seg_size),
            static_cast<cuda::std::int64_t>(k_value),
            seg_env);
          err != cudaSuccess)
      {
        return err;
      }
    }
    return cudaSuccess;
  }
  else
  {
    return cub::detail::batched_topk::dispatch_with_env(
      d_keys_in,
      d_keys_out,
      static_cast<cub::NullType**>(nullptr),
      static_cast<cub::NullType**>(nullptr),
      segment_sizes,
      k,
      select_direction,
      num_segments,
      total_num_items,
      env);
  }
}

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

  auto segment_sizes_param = ::cuda::__argument::__deferred_sequence{
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

  // Host copy of segment sizes — consumed only by the per-segment device backend.
  std::vector<cuda::std::int64_t> h_segment_sizes(static_cast<std::size_t>(num_segments));
  thrust::copy(d_segment_sizes.begin(), d_segment_sizes.end(), h_segment_sizes.begin());

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    // TODO(bgruber): call the public API once available
    _CCCL_TRY_CUDA_API(
      batched_topk_keys,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      segment_sizes_param,
      k_param,
      select_direction,
      num_segments_param,
      total_num_items,
      h_segment_sizes.data(),
      env);
  });
}

NVBENCH_BENCH_TYPES(decode_style_variable_topk_keys, NVBENCH_TYPE_AXES(key_type_list, max_segment_size_list, k_list))
  .set_name("decode_style_variable_topk_keys")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "K{ct}"})
  .add_int64_axis("NumSegments", {1, 2, 4, 8, 16, 32})
  .add_string_axis("Pattern", valid_patterns);
