// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared implementation for the variable-segment-size indexed (arg-top-k) benchmarks. The two thin TUs that include it
// -- `indexed.cu` (baseline) and `indexed.cluster.cu` (cluster) -- only set defaults and list their `%RANGE%` knobs;
// everything else (backend/requirement selection, the tuned `topk_backend_selector`, the dispatch wrapper, the nvbench
// body and its registration) lives here so the backends stay in lock-step. Each includer must define `TUNE_BACKEND`
// (0 baseline / 1 cluster / 2 automatic) and `TUNE_REQUIREMENT` (0 non-det / 1 det + prefer-smaller-index / 2 det +
// prefer-larger-index) before including.

#pragma once

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check so these benchmarks compile for the full
// configuration space (including deterministic / large-segment requests, which only the SM90+ cluster backend serves)
// across all target architectures, including pre-SM90. See _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in
// cub/device/dispatch/dispatch_batched_topk.cuh. Must precede the CUB includes below.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_batched_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__execution/determinism.h>
#include <cuda/__execution/output_ordering.h>
#include <cuda/__execution/require.h>
#include <cuda/__execution/tie_break.h>
#include <cuda/__execution/tune.h>
#include <cuda/argument>
#include <cuda/iterator>
#include <cuda/std/cstdint>

#include <nvbench_helper.cuh>

#include "common.cuh"

#ifndef TUNE_BACKEND
#  error "indexed_common.cuh requires the includer to define TUNE_BACKEND (0 baseline / 1 cluster / 2 automatic)"
#endif
#ifndef TUNE_REQUIREMENT
#  error \
    "indexed_common.cuh requires the includer to define TUNE_REQUIREMENT (0 non-det / 1 det+smaller / 2 det+larger)"
#endif

enum class topk_backend
{
  baseline,
  cluster,
  automatic,
};

// Which backend this build benchmarks. `automatic` (the default) issues no `tune` override, leaving the choice to the
// library's arch/size selector -- convenient here since variable segment sizes can exceed the baseline backend's
// coverage (forcing baseline is only valid for coverable sizes). A tuning variant forces one concrete backend so its
// sub-policy knobs (below) take effect.
inline constexpr topk_backend selected_backend =
#if TUNE_BACKEND == 0
  topk_backend::baseline;
#elif TUNE_BACKEND == 1
  topk_backend::cluster;
#else
  topk_backend::automatic;
#endif

// The determinism / tie-break requirement this build issues. Only the cluster and automatic backends honor a
// deterministic result set / concrete tie-break preference; the baseline backend must stay on the non-deterministic
// path (enforced by the static_assert below). The three requirement values collapse the API's determinism/tie-break
// combinations to the distinct behaviors we benchmark: no guarantee, and deterministic with either index preference.
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

// The baseline backend ignores determinism/tie-break, so require the defaults there to avoid a silently ignored
// selection. The cluster backend implements them, and automatic honors them via the library's selector, so both allow
// non-defaults.
static_assert(selected_backend == topk_backend::cluster || selected_backend == topk_backend::automatic
                || (selected_determinism == cuda::execution::determinism::__determinism_t::__not_guaranteed
                    && selected_tie_break == cuda::execution::tie_break::__tie_break_t::__unspecified),
              "Only the cluster and automatic backends honor determinism/tie-break requirements; keep TUNE_REQUIREMENT "
              "at 0 (non-deterministic) for the baseline backend.");

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
      /*copy_items_per_thread=*/TUNE_CLUSTER_COPY_IPT};
#else
    const auto cluster = cub::detail::batched_topk::cluster_policy_selector{}(cc);
#endif

    constexpr auto backend =
      (selected_backend == topk_backend::cluster)
        ? cub::detail::batched_topk::topk_backend::cluster
        : cub::detail::batched_topk::topk_backend::baseline;
    return cub::detail::batched_topk::topk_policy{backend, baseline, cluster};
  }
};

// Env-based dispatch over the selected backend. `automatic` routes through the public `cub::DeviceBatchedTopK` API with
// no backend override, so the library's own selector chooses; a forced backend routes through the same API but adds a
// `tune`d `topk_backend_selector` that pins the backend. Either way the determinism/tie-break/ordering requirement is
// layered on top of the benchmark environment (which carries the stream and the caching memory resource).
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParamT,
          typename KParamT,
          typename NumSegmentsParameterT,
          typename EnvT>
CUB_RUNTIME_FUNCTION static cudaError_t batched_topk_indexed(
  KeyInputItItT d_keys_in,
  KeyOutputItItT d_keys_out,
  ValueInputItItT d_values_in,
  ValueOutputItItT d_values_out,
  SegmentSizeParamT segment_sizes,
  KParamT k,
  NumSegmentsParameterT num_segments,
  EnvT env)
{
  auto req_env = cuda::std::execution::env{
    env,
    cuda::execution::require(cuda::execution::determinism::__determinism_holder_t<selected_determinism>{},
                             cuda::execution::tie_break::__tie_break_holder_t<selected_tie_break>{},
                             cuda::execution::output_ordering::unsorted)};
  if constexpr (selected_backend == topk_backend::automatic)
  {
    // No `tune` override: the library's own selector picks the backend (arch/size crossover) -- the usual behavior.
    return cub::DeviceBatchedTopK::MaxPairs(
      d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k, num_segments, req_env);
  }
  else
  {
    using key_t          = cub::detail::it_value_t<cub::detail::it_value_t<KeyInputItItT>>;
    using value_t        = cub::detail::it_value_t<cub::detail::it_value_t<ValueInputItItT>>;
    constexpr auto max_k = ::cuda::args::__traits<KParamT>::highest;
    auto full_env        = cuda::std::execution::env{
      req_env, cuda::execution::tune(topk_backend_selector<key_t, value_t, cuda::std::int64_t, max_k>{})};
    return cub::DeviceBatchedTopK::MaxPairs(
      d_keys_in, d_keys_out, d_values_in, d_values_out, segment_sizes, k, num_segments, full_env);
  }
}

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
  auto out_keys_buffer    = thrust::device_vector<KeyT>(output_elements, thrust::no_init);
  auto out_indices_buffer = thrust::device_vector<IndexT>(output_elements, thrust::no_init);

  auto segment_sizes_param = cuda::args::deferred_sequence{
    thrust::raw_pointer_cast(d_segment_sizes.data()), cuda::args::bounds<1, MaxSegmentSize>()};
  auto k_param            = cuda::args::constant<K>{};
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
  state.add_global_memory_reads<segment_size_t>(num_segments, "SegmentSizes");
  state.add_global_memory_writes<KeyT>(output_elements, "OutputKeys");
  state.add_global_memory_writes<IndexT>(output_elements, "OutputIndices");

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_CUDA_API(
      batched_topk_indexed,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      d_indices_in,
      d_indices_out,
      segment_sizes_param,
      k_param,
      num_segments_param,
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
