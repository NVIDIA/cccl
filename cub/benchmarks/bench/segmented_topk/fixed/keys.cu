// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defer the unsupported-architecture diagnosis to the dispatch's runtime check so this benchmark compiles for the full
// configuration space (including deterministic / large-segment requests, which only the SM90+ cluster backend serves)
// across all target architectures, including pre-SM90. See CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT in
// cub/device/device_batched_topk.cuh. Must precede the CUB includes below.
#define CUB_DISABLE_TOPK_UNSUPPORTED_ARCH_ASSERT

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

// Which backend this build benchmarks. A base build forces nothing (`automatic`, no `tune` override): the speedup
// reference. A tuning build forces `TUNE_BACKEND`; the `%RANGE%` sweep covers baseline and cluster, but only baseline
// knobs are exposed, so the cluster backend uses its default sub-policy. `device` (-DTUNE_BACKEND=2, one
// `cub::DeviceTopK` call per segment) is checked before TUNE_BASE so it stays reachable on the base target.
#ifndef TUNE_BACKEND
#  define TUNE_BACKEND 0 // baseline: the swept default (a base build ignores this and measures `automatic`)
#endif

// The tuning harness builds one architecture per GPU (and the cluster backend needs SM90+); base builds are exempt
// and rely on the runtime unsupported-arch fallback above.
#if !TUNE_BASE
#  if _CCCL_PP_COUNT(__CUDA_ARCH_LIST__) != 1
#    error "When tuning, the top-k benchmarks must be compiled for a single architecture"
#  endif
#  if TUNE_BACKEND == 1 && (__CUDA_ARCH_LIST__) < 900
#    error "Cannot tune the cluster backend below sm90"
#  endif
#endif // !TUNE_BASE

inline constexpr topk_backend selected_backend =
#if TUNE_BACKEND == 2
  topk_backend::device;
#elif TUNE_BASE
  topk_backend::automatic;
#elif TUNE_BACKEND == 0
  topk_backend::baseline;
#elif TUNE_BACKEND == 1
  topk_backend::cluster;
#else
  topk_backend::automatic;
#endif

// Determinism / tie-break requirement benchmarked by the cluster backend (a single combination for now).
inline constexpr auto selected_determinism = cuda::execution::determinism::__determinism_t::__not_guaranteed;
inline constexpr auto selected_tie_break   = cuda::execution::tie_break::__tie_break_t::__unspecified;

// The baseline requirement is enforced inside `topk_backend_selector` (instantiated only for a forced baseline/cluster
// backend). The device backend bypasses that selector (one cub::DeviceTopK call per segment), so its requirement is
// guarded here; the cluster and automatic backends honor determinism/tie-break, so both allow non-defaults.
static_assert(selected_backend != topk_backend::device
                || (selected_determinism == cuda::execution::determinism::__determinism_t::__not_guaranteed
                    && selected_tie_break == cuda::execution::tie_break::__tie_break_t::__unspecified),
              "The device backend does not honor determinism/tie-break requirements; keep selected_determinism and "
              "selected_tie_break at their defaults for it.");

// Policy selector passed to the tuning environment; instantiated only for a forced baseline/cluster backend (base,
// `automatic`, and `device` builds never construct it). The struct is still compiled everywhere, so the baseline knob
// branch stays gated on `!TUNE_BASE && (TUNE_BACKEND == 0 || 1)` -- a base/device/automatic build defines no TUNE_*
// knobs. The cluster sub-policy is always the default (no cluster knobs are exposed here).
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
#if !TUNE_BASE && (TUNE_BACKEND == 0 || TUNE_BACKEND == 1)
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
    const auto baseline = cub::detail::batched_topk::make_baseline_policy();
#endif
    const auto cluster = cub::detail::batched_topk::make_cluster_policy();
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
  const auto d_keys_in_ptr  = thrust::raw_pointer_cast(in_keys_buffer.data());
  const auto d_keys_out_ptr = thrust::raw_pointer_cast(out_keys_buffer.data());
  const auto d_keys_in      = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_in_ptr), segment_size);
  const auto d_keys_out = cuda::make_strided_iterator(cuda::make_counting_iterator(d_keys_out_ptr), selected_elements);

  const auto segment_sizes = ::cuda::args::constant<MaxSegmentSize>{};
  const auto k             = ::cuda::args::constant<MaxNumSelected>{};

  state.add_element_count(elements, "NumElements");
  state.add_element_count(segment_size, "SegmentSize");
  state.add_element_count(selected_elements, "NumSelectedElements");
  state.add_global_memory_reads<KeyT>(elements, "InputKeys");
  state.add_global_memory_writes<KeyT>(selected_elements * num_segments, "OutputKeys");

  // Host copy of segment sizes — all entries equal MaxSegmentSize for fixed-size segments. Consumed only by the
  // per-segment device backend. Segment sizes fit in a signed 32-bit integer (the library caps them at 2^21).
  const std::vector<cuda::std::int32_t> h_segment_sizes(num_segments, static_cast<cuda::std::int32_t>(MaxSegmentSize));

  caching_allocator_t alloc;
  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](nvbench::launch& launch) {
    const auto env = cub_bench_env(alloc, launch);
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
