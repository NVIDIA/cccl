// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk_cluster.cuh>

#include <cuda/__argument_>
#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>

#include <algorithm>
#include <vector>

#include <nvbench_helper.cuh>

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1

enum class topk_backend
{
  baseline,
  cluster,
  device,
};

inline constexpr topk_backend selected_backend = topk_backend::baseline;

#if !TUNE_BASE
struct tuned_policy_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability) const
    -> cub::detail::batched_topk::batched_topk_policy
  {
    // Single-entry policy chain driven by the tuning knobs.
    constexpr auto store_alg = cub::BLOCK_STORE_WARP_TRANSPOSE;
#  if TUNE_BLOCK_LOAD_ALGORITHM == 0
    constexpr auto load_alg = cub::BLOCK_LOAD_DIRECT;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 1
    constexpr auto load_alg = cub::BLOCK_LOAD_WARP_TRANSPOSE;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 2
    constexpr auto load_alg = cub::BLOCK_LOAD_VECTORIZE;
#  endif
    return cub::detail::batched_topk::batched_topk_policy{{{
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
    }}};
  }
};
#endif // !TUNE_BASE

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
      d_keys_in, d_keys_out, segment_sizes, k, select_direction, num_segments, total_num_items, env);
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

template <typename KeyT, int MaxSegmentSize, int MaxNumSelected>
void fixed_seg_size_topk_keys(
  nvbench::state& state,
  nvbench::type_list<KeyT, nvbench::enum_type<MaxSegmentSize>, nvbench::enum_type<MaxNumSelected>>)
{
  // Retrieve axis parameters
  const auto max_elements      = static_cast<size_t>(state.get_int64("Elements{io}"));
  const auto segment_size      = static_cast<::cuda::std::ptrdiff_t>(MaxSegmentSize);
  const auto selected_elements = static_cast<::cuda::std::ptrdiff_t>(MaxNumSelected);
  const auto num_segments      = ::cuda::std::max<std::size_t>(1, (max_elements / segment_size));
  const auto elements          = num_segments * segment_size;
  const auto total_num_items   = ::cuda::__argument::__immediate{static_cast<::cuda::std::int64_t>(elements)};
  const bit_entropy entropy    = str_to_entropy(state.get_string("Entropy"));

  // Skip workloads where k exceeds the segment size
  if (selected_elements >= segment_size)
  {
    state.skip("Skipping workload where K >= SegmentSize.");
    return;
  }

  thrust::device_vector<KeyT> in_keys_buffer = generate(elements, entropy);
  thrust::device_vector<KeyT> out_keys_buffer(selected_elements * num_segments, thrust::no_init);
  auto d_keys_in_ptr  = thrust::raw_pointer_cast(in_keys_buffer.data());
  auto d_keys_out_ptr = thrust::raw_pointer_cast(out_keys_buffer.data());
  auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  auto d_keys_out     = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), selected_elements);

  auto segment_sizes      = ::cuda::__argument::__constant<MaxSegmentSize>{};
  auto k                  = ::cuda::__argument::__constant<MaxNumSelected>{};
  auto select_direction   = ::cuda::__argument::__constant<cub::detail::topk::select::max>{};
  auto num_segments_param = ::cuda::__argument::__immediate{static_cast<::cuda::std::int64_t>(num_segments)};

  state.add_element_count(elements, "NumElements");
  state.add_element_count(segment_size, "SegmentSize");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements * num_segments, "OutputKeys");

  // Host copy of segment sizes — all entries equal MaxSegmentSize for fixed-size segments. Consumed only by the
  // per-segment device backend.
  std::vector<cuda::std::int64_t> h_segment_sizes(num_segments, static_cast<cuda::std::int64_t>(MaxSegmentSize));

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(
      alloc,
      launch
#if !TUNE_BASE
      ,
      cuda::execution::tune(tuned_policy_selector{})
#endif // !TUNE_BASE
    );
    // TODO(bgruber): call the public API once available
    _CCCL_TRY_CUDA_API(
      batched_topk_keys,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      segment_sizes,
      k,
      select_direction,
      num_segments_param,
      total_num_items,
      h_segment_sizes.data(),
      env);
  });
}

using key_type_list          = nvbench::type_list<float>;
using segment_size_type_list = nvbench::type_list<uint32_t>;
using out_offset_type_list   = nvbench::type_list<uint32_t>;

using segment_size_        = nvbench::type_list<uint32_t>;
using out_offset_type_list = nvbench::type_list<uint32_t>;

using small_segment_size_list = nvbench::enum_type_list<64, 128, 256, 512, 1024>;
using small_k_list            = nvbench::enum_type_list<8, 16, 32, 128, 512, 1024>;

NVBENCH_BENCH_TYPES(fixed_seg_size_topk_keys, NVBENCH_TYPE_AXES(key_type_list, small_segment_size_list, small_k_list))
  .set_name("small")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "MaxNumSelected{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(28, 28, 4))
  .add_string_axis("Entropy", {"1.000", "0.544", "0.201", "0.000"});
