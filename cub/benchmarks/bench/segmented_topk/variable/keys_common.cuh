// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared implementation for the variable-segment-size keys-only top-k benchmarks. The two thin TUs that include it --
// `keys.cu` (baseline) and `keys.cluster.cu` (cluster) -- only set defaults and list their `%RANGE%` knobs; everything
// else (backend/requirement selection, the tuned `topk_backend_selector`, the dispatch wrapper including the
// per-segment `device` reference, the nvbench body and its registration) lives here so the backends stay in lock-step.
// Each includer must define `TUNE_BACKEND` (0 baseline / 1 cluster / 2 device / 3 automatic) and `TUNE_REQUIREMENT`
// (0 non-det / 1 det + prefer-smaller-index / 2 det + prefer-larger-index) before including.

#pragma once

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check so these benchmarks compile for the full
// configuration space (including deterministic / large-segment requests, which only the SM90+ cluster backend serves)
// across all target architectures, including pre-SM90. See CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in
// cub/device/device_batched_topk.cuh. Must precede the CUB includes below.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_batched_topk.cuh>
#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/__execution/env.h>

#include <algorithm>
#include <vector>

#include <nvbench_helper.cuh>

#include "common.cuh"

#ifndef TUNE_BACKEND
#  error "keys_common.cuh requires the includer to define TUNE_BACKEND (0 baseline/1 cluster/2 device/3 automatic)"
#endif
#ifndef TUNE_REQUIREMENT
#  error "keys_common.cuh requires the includer to define TUNE_REQUIREMENT (0 non-det / 1 det+smaller / 2 det+larger)"
#endif
#if TUNE_BACKEND < 0 || TUNE_BACKEND > 3
#  error "keys_common.cuh: TUNE_BACKEND must be 0 (baseline), 1 (cluster), 2 (device), or 3 (automatic)"
#endif
#if TUNE_REQUIREMENT < 0 || TUNE_REQUIREMENT > 2
#  error "keys_common.cuh: TUNE_REQUIREMENT must be 0 (non-det), 1 (det+smaller), or 2 (det+larger)"
#endif

enum class topk_backend
{
  baseline,
  cluster,
  device,
  automatic,
};

// Which backend this build benchmarks. `automatic` (the default) issues no `tune` override, leaving the choice to the
// library's arch/size selector -- convenient here since variable segment sizes can exceed the baseline backend's
// coverage (forcing baseline is only valid for coverable sizes). A tuning variant forces one concrete backend so its
// sub-policy knobs (below) take effect; `device` issues one `cub::DeviceTopK` call per segment and has no knobs.
inline constexpr topk_backend selected_backend =
#if TUNE_BACKEND == 0
  topk_backend::baseline;
#elif TUNE_BACKEND == 1
  topk_backend::cluster;
#elif TUNE_BACKEND == 2
  topk_backend::device;
#else
  topk_backend::automatic;
#endif

// The determinism / tie-break requirement this build issues. Only the cluster and automatic backends honor a
// deterministic result set / concrete tie-break; the baseline and device backends must stay non-deterministic (enforced
// by the static_assert below). The three values cover the distinct behaviors we benchmark: no guarantee, and
// deterministic with either index preference.
inline constexpr auto selected_determinism =
#if TUNE_REQUIREMENT == 0
  cuda::execution::determinism::__determinism_t::__not_guaranteed;
#else
  cuda::execution::determinism::__determinism_t::__gpu_to_gpu;
#endif

inline constexpr auto selected_tie_break =
#if TUNE_REQUIREMENT == 1
  cuda::execution::tie_break::__tie_break_t::__prefer_smaller_index;
#elif TUNE_REQUIREMENT == 2
  cuda::execution::tie_break::__tie_break_t::__prefer_larger_index;
#else
    cuda::execution::tie_break::__tie_break_t::__unspecified;
#endif

// The baseline requirement is enforced inside `topk_backend_selector` (instantiated only for a forced baseline/cluster
// backend). The device backend bypasses that selector (one cub::DeviceTopK call per segment), so its requirement is
// guarded here; the cluster and automatic backends honor determinism/tie-break, so both allow non-defaults.
static_assert(selected_backend != topk_backend::device
                || (selected_determinism == cuda::execution::determinism::__determinism_t::__not_guaranteed
                    && selected_tie_break == cuda::execution::tie_break::__tie_break_t::__unspecified),
              "The device backend does not honor determinism/tie-break requirements; keep TUNE_REQUIREMENT at 0 "
              "(non-deterministic) for it.");

// Policy selector threaded through the public API's tuning environment when a concrete backend is forced (not
// `automatic`). Its `.backend` pins the backend for this build. In a base build both sub-policies are the defaults;
// in a tuning variant only the forced backend's sub-policy is driven by this build's TUNE_* macros (the other stays
// default), so each backend's benchmark sweeps only its own knobs.
template <class KeyT, class ValueT, class OffsetT, cuda::std::int64_t MaxK>
struct topk_backend_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability cc) const
    -> cub::detail::batched_topk::topk_policy
  {
    // The baseline backend cannot honor a deterministic result set / concrete tie-break request. The `sizeof(KeyT) ==
    // 0` dependent term defers the check to instantiation, which happens only for a forced baseline/cluster backend
    // (not the automatic/device builds that never instantiate this selector).
    static_assert(
      selected_backend == topk_backend::cluster
        || (selected_determinism == cuda::execution::determinism::__determinism_t::__not_guaranteed
            && selected_tie_break == cuda::execution::tie_break::__tie_break_t::__unspecified)
        || sizeof(KeyT) == 0,
      "The baseline backend cannot honor a deterministic result set or a concrete tie-break preference; "
      "force the cluster backend or request the non-deterministic defaults.");
#if !TUNE_BASE && TUNE_BACKEND == 0
    constexpr auto store_alg = cub::BLOCK_STORE_WARP_TRANSPOSE;
#  if TUNE_BLOCK_LOAD_ALGORITHM == 0
    constexpr auto load_alg = cub::BLOCK_LOAD_DIRECT;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 1
    constexpr auto load_alg = cub::BLOCK_LOAD_WARP_TRANSPOSE;
#  elif TUNE_BLOCK_LOAD_ALGORITHM == 2
    constexpr auto load_alg = cub::BLOCK_LOAD_VECTORIZE;
#  endif
    const auto baseline = cub::detail::batched_topk::baseline_topk_policy{{{
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
      cub::detail::batched_topk::worker_policy{TUNE_THREADS_PER_BLOCK, TUNE_ITEMS_PER_THREAD, load_alg, store_alg},
    }}};
#else
    const auto baseline =
      cub::detail::batched_topk::baseline_policy_selector_from_types<KeyT, ValueT, OffsetT, MaxK>{}(cc);
#endif

#if !TUNE_BASE && TUNE_BACKEND == 1
    const auto cluster = cub::detail::batched_topk::cluster_topk_policy{
      /*threads_per_block=*/TUNE_CLUSTER_THREADS_PER_BLOCK,
      /*min_blocks_per_sm=*/TUNE_CLUSTER_MIN_BLOCKS_PER_SM,
      /*min_chunks_per_block=*/TUNE_CLUSTER_MIN_CHUNKS_PER_BLOCK,
      /*chunk_bytes=*/(TUNE_CLUSTER_CHUNK_KIB) * 1024,
      /*load_align_bytes=*/(1 << (TUNE_CLUSTER_LOAD_ALIGN_BYTES_POW2)),
      /*pipeline_stages=*/TUNE_CLUSTER_PIPELINE_STAGES,
      /*single_block_max_seg_size=*/8 * 1024,
      /*bits_per_pass=*/TUNE_CLUSTER_BITS_PER_PASS,
      /*histogram_items_per_thread=*/TUNE_CLUSTER_HIST_IPT,
      /*tie_break_items_per_thread=*/TUNE_CLUSTER_TIEBREAK_IPT,
      /*copy_items_per_thread=*/TUNE_CLUSTER_COPY_IPT,
      /*max_blocks_per_cluster=*/0,
      /*max_chunk_slots_per_block=*/0};
#else
    const auto cluster = cub::detail::batched_topk::cluster_policy_selector{}(cc);
#endif

    constexpr auto backend =
      (selected_backend == topk_backend::cluster)
        ? cub::detail::batched_topk::topk_algorithm::cluster
        : cub::detail::batched_topk::topk_algorithm::baseline;
    return cub::detail::batched_topk::topk_policy{backend, baseline, cluster};
  }
};

// Env-based dispatch over the selected backend. `automatic`/`baseline`/`cluster` all route through the public
// `cub::DeviceBatchedTopK` API (the latter two add a `tune`d `topk_backend_selector` that forces the backend; temp
// storage comes from the memory resource carried by `env`); the `device` backend issues one `cub::DeviceTopK::MaxKeys`
// per segment, reading the host-side segment sizes.
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename NumSegmentsParameterT,
          typename HostSegSizeT,
          typename EnvT>
_CCCL_HOST_API static cudaError_t batched_topk_keys(
  KeyInputItItT d_keys_in,
  KeyOutputItItT d_keys_out,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] const HostSegSizeT* h_segment_sizes,
  EnvT env)
{
  if constexpr (selected_backend == topk_backend::device)
  {
    using num_segments_val_t = typename ::cuda::args::__traits<NumSegmentsParameterT>::element_type;
    const auto num_segs      = cub::detail::params::get_param(num_segments, num_segments_val_t{0});

    // The per-segment device backend uses the unsorted / not-guaranteed-determinism fast path. Layer the requirement
    // on top of the benchmark environment (which carries the stream and the caching memory resource).
    const auto seg_env = cuda::std::execution::env{
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
    // The determinism / tie-break / ordering requirement this benchmark issues; the library honors it whether or not we
    // additionally force a backend.
    const auto req_env = cuda::std::execution::env{
      env,
      cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<selected_determinism>{},
                               cuda::execution::tie_break::__tie_break_holder_t<selected_tie_break>{},
                               cuda::execution::output_ordering::unsorted)};
    if constexpr (selected_backend == topk_backend::automatic)
    {
      // No `tune` override: the library's own selector picks the backend (arch/size crossover) -- the usual behavior.
      return cub::DeviceBatchedTopK::MaxKeys(d_keys_in, d_keys_out, segment_sizes, k, num_segments, req_env);
    }
    else
    {
      using key_t          = cub::detail::it_value_t<cub::detail::it_value_t<KeyInputItItT>>;
      constexpr auto max_k = ::cuda::args::__traits<KParamT>::highest;
      const auto full_env  = cuda::std::execution::env{
        req_env, cuda::execution::tune(topk_backend_selector<key_t, cub::NullType, cuda::std::int64_t, max_k>{})};
      return cub::DeviceBatchedTopK::MaxKeys(d_keys_in, d_keys_out, segment_sizes, k, num_segments, full_env);
    }
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

  const auto num_segments                                     = static_cast<int>(state.get_int64("NumSegments"));
  const thrust::device_vector<segment_size_t> d_segment_sizes = generate(
    static_cast<std::size_t>(num_segments),
    bit_entropy::_1_000,
    static_cast<segment_size_t>(K),
    static_cast<segment_size_t>(MaxSegmentSize));
  const auto input_elements  = thrust::reduce(d_segment_sizes.begin(), d_segment_sizes.end());
  const auto output_elements = static_cast<std::size_t>(num_segments) * K;

  auto in_keys_buffer = gen_data<MaxSegmentSize, K>(
    num_segments, string_to_pattern(state.get_string("Pattern")), thrust::raw_pointer_cast(d_segment_sizes.data()));
  auto out_keys_buffer = thrust::device_vector<KeyT>(output_elements, thrust::no_init);

  const auto segment_sizes_param = cuda::args::deferred_sequence{
    thrust::raw_pointer_cast(d_segment_sizes.data()), cuda::args::bounds<1, MaxSegmentSize>()};
  const auto k_param            = cuda::args::constant<K>{};
  const auto num_segments_param = cuda::args::immediate{static_cast<cuda::std::int64_t>(num_segments)};

  const auto d_keys_in = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(in_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(MaxSegmentSize));
  const auto d_keys_out = cuda::make_strided_iterator(
    cuda::make_counting_iterator(thrust::raw_pointer_cast(out_keys_buffer.data())),
    static_cast<cuda::std::ptrdiff_t>(K));

  state.add_element_count(input_elements, "NumElements");
  state.add_global_memory_reads<KeyT>(input_elements, "InputKeys");
  state.add_global_memory_reads<segment_size_t>(num_segments, "SegmentSizes");
  state.add_global_memory_writes<KeyT>(output_elements, "OutputKeys");

  // Host copy of segment sizes — consumed only by the per-segment device backend.
  std::vector<segment_size_t> h_segment_sizes(static_cast<std::size_t>(num_segments));
  thrust::copy(d_segment_sizes.begin(), d_segment_sizes.end(), h_segment_sizes.begin());

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_CUDA_API(
      batched_topk_keys,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      segment_sizes_param,
      k_param,
      num_segments_param,
      h_segment_sizes.data(),
      env);
  });
}

NVBENCH_BENCH_TYPES(decode_style_variable_topk_keys, NVBENCH_TYPE_AXES(key_type_list, max_segment_size_list, k_list))
  .set_name("decode_style_variable_topk_keys")
  .set_type_axes_names({"KeyT{ct}", "MaxSegmentSize{ct}", "K{ct}"})
  .add_int64_axis("NumSegments", {1, 2, 4, 8, 16, 32})
  .add_string_axis("Pattern", valid_patterns);
