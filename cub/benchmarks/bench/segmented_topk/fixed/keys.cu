// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check so this benchmark compiles for the full
// configuration space (including deterministic / large-segment requests, which only the SM90+ cluster backend serves)
// across all target architectures, including pre-SM90. See _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in
// cub/device/dispatch/dispatch_batched_topk.cuh. Must precede the CUB includes below.
#define _CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

#include <cub/detail/choose_offset.cuh>
#include <cub/device/device_batched_topk.cuh>
#include <cub/device/device_topk.cuh>
#include <cub/device/dispatch/dispatch_batched_topk.cuh>

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

// %RANGE% TUNE_ITEMS_PER_THREAD ipt 1:24:1
// %RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32
// %RANGE% TUNE_BLOCK_LOAD_ALGORITHM ld 0:2:1
// %RANGE% TUNE_BACKEND backend 0:1:1

enum class topk_backend
{
  baseline,
  cluster,
  device,
  automatic,
};

// Which backend this build benchmarks. `automatic` (the default) issues no `tune` override, leaving the choice to the
// library's arch/size selector. `baseline`/`cluster` force one of the two DeviceBatchedTopK backends via the `tune`d
// selector below; `device` is a reference that issues one `cub::DeviceTopK` call per segment. Autotuning sweeps the
// baseline and cluster backends (the `%RANGE% ... 0:1:1` above); only baseline knobs are exposed here, so the cluster
// backend uses its default sub-policy and `device` has no knobs. Override with -DTUNE_BACKEND=0/1 (force a backend),
// =2 (device reference), or =3 (automatic).
#ifndef TUNE_BACKEND
#  if TUNE_BASE
#    define TUNE_BACKEND 3 // automatic: the library's production selector, for base/benchmark builds
#  else
#    define TUNE_BACKEND 0 // force baseline when an actively-tuned variant does not sweep the backend
#  endif
#endif

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

// Determinism / tie-break requirement benchmarked by the cluster backend (a single combination for now).
inline constexpr auto selected_determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
inline constexpr auto selected_tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;

// The baseline/device backends ignore these requirements, so require the defaults there to avoid a silent mismatch. The
// cluster backend implements them, and automatic honors them via the library's selector, so both allow non-defaults.
static_assert(selected_backend == topk_backend::cluster || selected_backend == topk_backend::automatic
                || (selected_determinism == cuda::execution::determinism::__determinism_t::__not_guaranteed
                    && selected_tie_break == cuda::execution::tie_break::__tie_break_t::__unspecified),
              "Only the cluster and automatic backends honor determinism/tie-break requirements; keep "
              "selected_determinism and selected_tie_break at their defaults for the baseline/device backends.");

// Policy selector threaded through the public API's tuning environment when a concrete backend is forced (not
// `automatic`). Its `.backend` pins the backend for this build. In a TUNE_BASE build the forced backend uses the
// default sub-policies; otherwise the baseline knobs come from the TUNE_* macros. The cluster sub-policy is always the
// default (no cluster knobs are exposed here).
template <class KeyT, class ValueT, class OffsetT, cuda::std::int64_t MaxK>
struct topk_backend_selector
{
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr auto operator()(cuda::compute_capability cc) const
    -> cub::detail::batched_topk::topk_policy
  {
#if TUNE_BASE
    const auto baseline =
      cub::detail::batched_topk::baseline_policy_selector_from_types<KeyT, ValueT, OffsetT, MaxK>{}(cc);
#else
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
#endif // TUNE_BASE
    const auto cluster = cub::detail::batched_topk::cluster_policy_selector{}(cc);
    constexpr auto backend =
      (selected_backend == topk_backend::cluster)
        ? cub::detail::batched_topk::topk_backend::cluster
        : cub::detail::batched_topk::topk_backend::baseline;
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
CUB_RUNTIME_FUNCTION static cudaError_t batched_topk_keys(
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
    // The determinism / tie-break / ordering requirement this benchmark issues; the library honors it whether or not we
    // additionally force a backend.
    auto req_env = cuda::std::execution::env{
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
      auto full_env        = cuda::std::execution::env{
        req_env, cuda::execution::tune(topk_backend_selector<key_t, cub::NullType, cuda::std::int64_t, max_k>{})};
      return cub::DeviceBatchedTopK::MaxKeys(d_keys_in, d_keys_out, segment_sizes, k, num_segments, full_env);
    }
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

  auto segment_sizes = ::cuda::args::constant<MaxSegmentSize>{};
  auto k             = ::cuda::args::constant<MaxNumSelected>{};

  state.add_element_count(elements, "NumElements");
  state.add_element_count(segment_size, "SegmentSize");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements * num_segments, "OutputKeys");

  // Host copy of segment sizes — all entries equal MaxSegmentSize for fixed-size segments. Consumed only by the
  // per-segment device backend. Segment sizes fit in a signed 32-bit integer (the library caps them at 2^21).
  std::vector<cuda::std::int32_t> h_segment_sizes(num_segments, static_cast<cuda::std::int32_t>(MaxSegmentSize));

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    auto env = cub_bench_env(alloc, launch);
    _CCCL_TRY_CUDA_API(
      batched_topk_keys,
      "batched topk failed",
      d_keys_in,
      d_keys_out,
      segment_sizes,
      k,
      ::cuda::args::immediate{static_cast<::cuda::std::int64_t>(num_segments)},
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
