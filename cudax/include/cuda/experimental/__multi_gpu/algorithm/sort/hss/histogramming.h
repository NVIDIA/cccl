// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_HISTOGRAMMING_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_HISTOGRAMMING_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_transform.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__container/make_buffer_with_pool.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__launch/launch.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/sample.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__random/philox_engine.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#if _CCCL_CTK_BELOW(12, 9)
#  include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#endif // CUDA 12.8-

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/bucket_count_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/ideal_rank_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/merge_k_way.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Single-thread kernel that draws sample keys from the union of splitter intervals.
//!
//! Runs on exactly one grid thread (all others return immediately). It walks the per-splitter
//! sampling intervals `__I_j`, and for each interval `[lo, hi)` locates the corresponding sorted
//! sub-range of the local keys. Implements the per-round sampling of the paper's
//! histogram-sort-with-sampling loop.
//!
//! @param[in] __config The launch configuration used to compute the thread's grid rank.
//! @param[in] __gen The Philox random engine used for sampling.
//! @param[in] __prob The per-key sampling probability.
//! @param[in] __begin Pointer to the first local key; advanced internally across intervals.
//! @param[in] __end Pointer one past the last local key.
//! @param[in] __I_j The span of per-splitter `[lo, hi]` sampling intervals.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[out] __samples The span the drawn sample keys are written into.
//! @param[out] __samples_size Receives the number of keys actually drawn.
//
// TODO(jfaibussowit):
//
// Parallelize with multiple threads (but not too many!). __I_j is O(p-1), so in
// practice at the absolute max a few thousand if you are running on the worlds
// largest supercomputers.
template <class _Config, class _Tp, class _BinaryOp>
_CCCL_KERNEL_ATTRIBUTES void __sample_probes_kernel(
  _Config __config,
  ::cuda::std::philox4x64 __gen,
  const double __prob,
  const _Tp* __begin,
  const _Tp* const __end,
  const ::cuda::std::span<const ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>> __I_j,
  _BinaryOp __cmp,
  const ::cuda::std::span<_Tp> __samples,
  ::cuda::std::size_t* const __samples_size)
{
  if (cuda::gpu_thread.rank(cuda::grid, __config) != 0)
  {
    // Just in case
    return;
  }

  auto __samples_it = __samples.begin();

  // By value so that load from global memory happens only once
  for (const auto [__lo, __hi] : __I_j)
  {
    // Sample from the union of splitter intervals. Splitter intervals are disjoint or
    // identical; lo_it skips an identical interval already covered by an earlier splitter.
    const auto __last  = __hi.has_value() ? ::cuda::std::lower_bound(__begin, __end, *__hi, __cmp) : __end;
    const auto __first = __lo.has_value() ? ::cuda::std::lower_bound(__begin, __last, *__lo, __cmp) : __begin;

    _CCCL_ASSERT(__first <= __last, "Inputs are not sorted for binary search");

    const auto __num_samples       = ::cuda::std::ceil(static_cast<double>(__last - __first) * __prob);
    const auto __remaining_samples = __samples.end() - __samples_it;
    const auto __n                 = ::cuda::std::min(
      static_cast<::cuda::std::uint64_t>(__num_samples), static_cast<::cuda::std::uint64_t>(__remaining_samples));

    __samples_it = ::cuda::std::sample(__first, __last, __samples_it, __n, __gen);
    __begin      = __last;
  }

  *__samples_size = static_cast<::cuda::std::size_t>(__samples_it - __samples.begin());
}

//! @brief Launch the sampling kernel to draw sample keys from a rank's local input.
template <class _Tp, class _Env, class _BinaryOp>
template <class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__sample_probes(
  _InputRange&& __input,
  const __buffer_type<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>& __I_j,
  double __sampling_probability,
  ::cuda::std::uint64_t __sample_seed,
  const _BinaryOp& __cmp,
  __buffer_type<_Tp>* __samples,
  __buffer_type<::cuda::std::size_t>* __sample_size)
{
  constexpr auto __config =
    ::cuda::make_config(::cuda::make_hierarchy(::cuda::block_dims<1>(), ::cuda::grid_dims<1>()));

  _CCCL_VERIFY(__sampling_probability > 0, "Cannot have 0 probably of picking elements");
  _CCCL_VERIFY(__sampling_probability <= 1., "Cannot have >1 probably of picking elements");

  ::cuda::launch(
    // All inputs should be on the same stream here
    __I_j.stream(),
    __config,
    ::cuda::experimental::__detail::__hss_sort::
      __sample_probes_kernel<::cuda::std::remove_cvref_t<decltype(__config)>, _Tp, _BinaryOp>,
    ::cuda::std::philox4x64{__sample_seed},
    __sampling_probability,
    ::cuda::std::to_address(::cuda::std::ranges::begin(__input)),
    ::cuda::std::to_address(::cuda::std::ranges::end(__input)),
    __I_j,
    __cmp,
    ::cuda::std::span<_Tp>{*__samples},
    __sample_size->data());
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

//! @brief Interval-narrowing functor that tightens one splitter's `[L, U]` rank bracket.
template <class _Tp>
struct __update_intervals_fn
{
  const _Tp* const __probes_begin;
  const ::cuda::std::uint64_t* const __hist_begin;
  const ::cuda::std::size_t __num_probes;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::
    tuple<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>, _Bracket<_Tp>, _Bracket<_Tp>>
    operator()(const ::cuda::std::uint64_t __target, _Bracket<_Tp> __L_i, _Bracket<_Tp> __U_i) const noexcept
  {
    // global_rank = number of input keys strictly less than probes[j]
    //             = prefix sum of per-bucket counts up to bucket j.
    ::cuda::std::uint64_t __global_rank = 0;

    for (::cuda::std::size_t __j = 0; __j < __num_probes; ++__j)
    {
      __global_rank += __hist_begin[__j];

      if (__global_rank == __target)
      {
        // Exact match, we have managed to find the perfect splitter value
        __L_i = __U_i = _Bracket<_Tp>{__global_rank, __probes_begin[__j]};
        break;
      }

      if ((__global_rank < __target) && (__global_rank > __L_i.__rank))
      {
        // We undershot the target, so we can raise our lower bound
        __L_i = _Bracket<_Tp>{__global_rank, __probes_begin[__j]};
      }
      else if ((__global_rank > __target) && (__global_rank < __U_i.__rank))
      {
        // Overshot the target but can lower the upper bound
        __U_i = _Bracket<_Tp>{__global_rank, __probes_begin[__j]};
      }
    }

    return ::cuda::std::make_tuple(::cuda::std::make_pair(__L_i.__key, __U_i.__key), __L_i, __U_i);
  }
};

// Rank designated as the collective root for the HSS sampling phase.
inline constexpr ::cuda::std::int32_t __ROOT_RANK = 0;

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange>
[[nodiscard]]
_CCCL_HOST_API ::cuda::std::pair<
  ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::__per_comm_sampling_scratch_type>,
  ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::__per_comm_histogramming_result_type>>
_HSSSorter<_Tp, _Env, _BinaryOp>::__allocate_histogramming_buffers(
  const __local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs)
{
  const auto __comm_size        = __setup.__comm_size;
  const auto __N                = __setup.__N;
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  ::std::vector<__per_comm_sampling_scratch_type> __local_scratch;
  ::std::vector<__per_comm_histogramming_result_type> __local_hist_results;

  __local_scratch.reserve(__num_local_inputs);
  __local_hist_results.reserve(__num_local_inputs);

  for (auto&& [__comm, __env] : ::cuda::std::ranges::views::zip(__comms, __envs))
  {
    const auto __stream  = ::cuda::get_stream(__env);
    const auto __n_split = __comm_size - 1;

    auto&& __resource =
      ::cuda::experimental::__detail::__resource_from_env(__env, __comm.logical_device().underlying_device());
    auto&& __buffer_env = ::cuda::experimental::__detail::__sanitize_buffer_env(__env);

    {
#if _CCCL_CTK_AT_LEAST(12, 9)
      auto __probe_counts =
        ::cuda::make_pinned_buffer<::cuda::std::uint64_t>(__stream, /*__size=*/::cuda::std::size_t{1}, ::cuda::no_init);
#else // ^^^ CUDA 12.9+ ^^^ / vvv CUDA 12.8- vvv
      auto __probe_counts = ::cuda::make_buffer<::cuda::std::uint64_t>(
        __stream,
        ::cuda::mr::legacy_pinned_memory_resource{},
        /*__size=*/::cuda::std::size_t{1},
        ::cuda::no_init);
#endif // ^^^ CUDA 12.8- ^^^

      __local_scratch.emplace_back(__per_comm_sampling_scratch_type{
        /*__I_j=*/
        __buffer_type<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>{
          ::cuda::make_buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>(
            __stream,
            __resource,
            __n_split,
            ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>{},
            __buffer_env)},
        /*__samples=*/
        __buffer_type<_Tp>{__stream, __resource, __buffer_env},
        /*__samples_size=*/
        __buffer_type<::cuda::std::size_t>{
          __stream,
          __resource,
          /*__size=*/::cuda::std::size_t{__comm.rank() == __ROOT_RANK ? __comm_size : 1},
          ::cuda::no_init,
          __buffer_env},
        ::cuda::std::move(__probe_counts)});
    }

    __local_hist_results.emplace_back(__per_comm_histogramming_result_type{
      /*__splitters=*/
      __per_comm_splitters_type{
        /*__Ls=*/__buffer_type<_Bracket<_Tp>>{::cuda::make_buffer<_Bracket<_Tp>>(
          __stream, __resource, __n_split, _Bracket<_Tp>{0, ::cuda::std::nullopt}, __buffer_env)},
        /*__Us=*/
        __buffer_type<_Bracket<_Tp>>{::cuda::make_buffer<_Bracket<_Tp>>(
          __stream, __resource, __n_split, _Bracket<_Tp>{__N, ::cuda::std::nullopt}, __buffer_env)},
        /*__probes=*/
        __buffer_type<_Tp>{__stream, __resource, __buffer_env}},
      /*__hist=*/__buffer_type<::cuda::std::uint64_t>{__stream, __resource, __buffer_env}});
  }

  return ::cuda::std::make_pair(::cuda::std::move(__local_scratch), ::cuda::std::move(__local_hist_results));
}

//! @brief Gather each rank's samples to the root, merge them, and broadcast the shared probes.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__gather_merge_broadcast(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  const _BinaryOp& __cmp,
  ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch,
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results,
  ::std::vector<::cuda::std::size_t>* __root_recvcounts,
  ::std::vector<::cuda::std::size_t>* __root_displs,
  ::cuda::std::optional<__buffer_type<_Tp>>* __root_all_samples)
{
  // The root needs to know how big everyones sample vectors are so it can build its combined
  // sample vector
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __scratch] : ::cuda::std::ranges::views::zip(__comms, *__local_scratch))
    {
      auto* const __ptr = __scratch.__samples_size.data();

      __comm.gather(__guard, __ptr, __ptr, /*__count=*/1, __ROOT_RANK, __scratch.__samples_size.stream());
    }
  }

  for (auto&& [__comm, __env, __scratch] : ::cuda::std::ranges::views::zip(__comms, __envs, *__local_scratch))
  {
    auto& __samples_size               = __scratch.__samples_size;
    ::cuda::std::uint64_t& __sendcount = __scratch.__sample_sendcount;

    if (__comm.rank() == __ROOT_RANK)
    {
      // __root_recvcounts, __root_displs, and __root_all_samples are hoisted out of the
      // sampling loop and passed in by pointer so their allocations are paid once per sort
      // rather than once per round. __root_displs is filled via back_inserter, so it must
      // start each round empty; the recvcounts resize and the all_samples resize below reuse
      // the existing storage (sizes are invariant / monotonically shrinking across rounds, so
      // neither reallocates after round one).
      __root_recvcounts->resize(__samples_size.size());

      ::cuda::copy_bytes(
        __samples_size.stream(),
        __samples_size,
        *__root_recvcounts,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __root_displs->resize(__root_recvcounts->size());

      // Defer until the last possible moment
      __samples_size.stream().sync();

      __sendcount = (*__root_recvcounts)[__ROOT_RANK];

      // recvcounts is likely relatively small (it is on the order of O(ranks)).
      ::cuda::std::exclusive_scan(
        __root_recvcounts->begin(), __root_recvcounts->end(), __root_displs->begin(), ::cuda::std::size_t{0});

      const auto __all_recv = __root_displs->back() + __root_recvcounts->back();
      // First round engages the optional; later rounds reuse the allocation via resize
      // (samples shrink monotonically, so this never grows after round one).
      if (__root_all_samples->has_value())
      {
        (*__root_all_samples)->resize_discard(__samples_size.stream(), __all_recv, ::cuda::no_init);
      }
      else
      {
        __root_all_samples->emplace(
          __samples_size.stream(),
          __samples_size.memory_resource(),
          __all_recv,
          ::cuda::no_init,
          ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
      }
    }
    else
    {
      // Non-root, __samples_size.size() should be == 1
      ::cuda::copy_bytes(
        __samples_size.stream(),
        __samples_size,
        ::cuda::std::span{&__sendcount, ::cuda::std::size_t{1}},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __samples_size.stream().sync();
    }
  }

  // Gather all samples to the root so it can build the global sampling vector
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __scratch] : ::cuda::std::ranges::views::zip(__comms, *__local_scratch))
    {
      __comm.gather_v(
        __guard,
        __scratch.__samples.data(),
        __scratch.__sample_sendcount,
        __comm.rank() == __ROOT_RANK ? __root_all_samples->value().data() : nullptr,
        __root_recvcounts->data(),
        __root_displs->data(),
        __ROOT_RANK,
        __scratch.__samples.stream());
    }
  }

  // Root merges the p sorted runs into one sorted probe set
  for (auto&& [__comm, __env, __hist_result, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__local_hist_results, *__local_scratch))
  {
    if (__comm.rank() == __ROOT_RANK)
    {
      auto& __probes = __hist_result.__splitters.__probes;

      __merge_k_way(__comm, __env, __root_all_samples->value(), *__root_recvcounts, *__root_displs, __cmp, &__probes);

      __scratch.__probe_counts.front() = __probes.size();
      break;
    }
  }

  // Extremely painful stuff here. We need to send the probe count, but we can only use NCCL,
  // and NCCL only provides device transport (even though they definitely have host-host
  // transport available internally).
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __scratch] : ::cuda::std::ranges::views::zip(__comms, *__local_scratch))
    {
      auto* const __ptr = __scratch.__probe_counts.data();

      __comm.broadcast(__guard, __ptr, __ptr, /*__count=*/1, __ROOT_RANK, __scratch.__probe_counts.stream());
    }
  }

  {
    ::cuda::std::optional<::cuda::std::size_t> __probe_count = ::cuda::std::nullopt;

    for (auto&& [__comm, __hist_result, __scratch] :
         ::cuda::std::ranges::views::zip(__comms, *__local_hist_results, *__local_scratch))
    {
      auto& __probes = __hist_result.__splitters.__probes;

      // Slight set of micro-optimizations here: the probe counts are all the same and are
      // broadcast by the root.
      //
      // So in the case we have multiple local GPUs, we can do 2 optimizations:
      //
      // 1. One of the local GPUs is the root, so we can immediately know the value, and the
      //    rest can reuse it.
      // 2. Only the first of the local GPUs actuallys needs to synchronize to get the count on
      //    the host, the rest can just reuse it.
      if (__comm.rank() == __ROOT_RANK)
      {
        __probe_count = __probes.size();
        continue;
      }

      if (!__probe_count.has_value())
      {
        __scratch.__probe_counts.stream().sync();
        __probe_count = __scratch.__probe_counts.front();
      }

      __probes.resize_discard(__probes.stream(), *__probe_count, ::cuda::no_init);
    }
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __hist_result] : ::cuda::std::ranges::views::zip(__comms, *__local_hist_results))
    {
      auto& __probes    = __hist_result.__splitters.__probes;
      auto* const __ptr = __probes.data();

      __comm.broadcast(__guard, __ptr, __ptr, __probes.size(), __ROOT_RANK, __probes.stream());
    }
  }
}

//! @brief Compute and globally reduce the per-probe histogram of the local keys.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__compute_histogram(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __range_of_local_keys,
  const _BinaryOp& __cmp,
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results)
{
  for (auto&& [__comm, __env, __keys, __hist_result] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_local_keys, *__local_hist_results))
  {
    auto& __hist               = __hist_result.__hist;
    auto& __probes             = __hist_result.__splitters.__probes;
    const auto __num_probes    = __probes.size();
    const auto __num_buckets   = __num_probes + 1;
    const auto* __keys_first   = ::cuda::std::to_address(::cuda::std::ranges::begin(__keys));
    const auto* __probes_first = __probes.data();

    __hist.resize_discard(__hist.stream(), __num_buckets, ::cuda::no_init);

    auto __op = __bucket_count_fn<::cuda::std::remove_cvref_t<decltype(__keys_first)>,
                                  ::cuda::std::remove_cvref_t<decltype(__probes_first)>,
                                  _BinaryOp>{
      __keys_first, ::cuda::std::to_address(::cuda::std::ranges::end(__keys)), __probes_first, __num_probes, __cmp};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      __hist.data(),
      __num_buckets,
      ::cuda::std::move(__op),
      __env);
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __hist_result] : ::cuda::std::ranges::views::zip(__comms, *__local_hist_results))
    {
      auto& __hist      = __hist_result.__hist;
      auto* const __ptr = __hist.data();

      __comm.all_reduce(__guard, __ptr, __ptr, __hist.size(), ::cuda::std::plus<>{}, __hist.stream());
    }
  }
}

//! @brief Tighten each per-splitter `[L, U]` rank bracket from the all-reduced probe histogram.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__update_intervals(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  ::cuda::std::uint64_t __N,
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results,
  ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch)
{
  for (auto&& [__comm, __env, __hist_result, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__local_hist_results, *__local_scratch))
  {
    auto& __splitters          = __hist_result.__splitters;
    const auto& __hist         = __hist_result.__hist;
    const auto __comm_size     = __comm.size();
    const auto __num_splitters = __scratch.__I_j.size();
    auto* const __I_j_begin    = __scratch.__I_j.data();
    auto* const __Ls_begin     = __splitters.__Ls.data();
    auto* const __Us_begin     = __splitters.__Us.data();

    auto __in = ::cuda::std::make_tuple(
      ::cuda::make_transform_iterator(
        ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
      __Ls_begin,
      __Us_begin);
    auto __out = ::cuda::std::make_tuple(__I_j_begin, __Ls_begin, __Us_begin);
    auto __op  = __update_intervals_fn<_Tp>{__splitters.__probes.data(), __hist.data(), __splitters.__probes.size()};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::std::move(__in),
      ::cuda::std::move(__out),
      __num_splitters,
      ::cuda::std::move(__op),
      __env);
  }
}

template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
[[nodiscard]]
_CCCL_HOST_API ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::__per_comm_histogramming_result_type>
_HSSSorter<_Tp, _Env, _BinaryOp>::__histogramming_phase(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const _BinaryOp& __cmp)
{
  const auto __comm_size                       = __setup.__comm_size;
  const auto __N                               = __setup.__N;
  auto [__local_scratch, __local_hist_results] = __allocate_histogramming_buffers(__setup, __comms, __envs);

  // Root-only scratch for __gather_merge_broadcast, hoisted out of the sampling loop so their
  // allocations are paid once per sort instead of once per round.
  //
  // recvcounts/displs are O(comm_size) host vectors of invariant size; all_samples is the
  // device buffer holding the gathered sample keys, which shrinks monotonically across
  // rounds.
  ::std::vector<::cuda::std::size_t> __root_recvcounts;
  ::std::vector<::cuda::std::size_t> __root_displs;
  ::cuda::std::optional<__buffer_type<_Tp>> __root_all_samples;

  constexpr double __eps = 0.02; // 2% tolerance

  // Note: K is small, on the order of ~1-10
  const auto __K = ::cuda::std::max(
    static_cast<::cuda::std::int32_t>(::cuda::std::ceil(::cuda::std::log10(::cuda::std::log10(__comm_size) / __eps))),
    1);
  const auto __s_j_interior = 2. * ::cuda::std::log(__comm_size) / __eps;

  for (::cuda::std::int32_t __j = 1; __j <= __K; ++__j)
  {
    const auto __s_j  = ::cuda::std::pow(__s_j_interior, static_cast<double>(__j) / static_cast<double>(__K));
    const auto __prob = ::cuda::std::min(__s_j * static_cast<double>(__comm_size) / static_cast<double>(__N), 1.);

    for (auto&& [__comm, __input, __scratch, __n_local] :
         ::cuda::std::ranges::views::zip(__comms, __local_inputs, __local_scratch, __setup.__local_original_sizes))
    {
      // Each iteration we sample the union of splitter intervals, \gamma_j with a probability
      // of __prob. For the first iteration, \gamma_j is the entire array, but for previous
      // iterations it's impossible for us to tell (on the host), because:
      //
      // 1. We can't inspect the updated intervals __I_j, and
      // 2. We can't count how many of our elements actually lie within those updated
      //    intervals.
      //
      // So instead we use the fact that each round the number of samples must decrease,
      // because I_j gets tightened and we sample an increasingly smaller region. Therefore,
      // the high-water mark for the samples is the previous round's sample vector size.
      const auto __estimate = ::cuda::std::max(
        __j == 1 ? static_cast<::cuda::std::size_t>(::cuda::std::ceil(__n_local * __prob))
                 : __scratch.__sample_sendcount,
        ::cuda::std::size_t{1});
      // 0x129381294235245ULL chosen randomly, by random dice roll
      const auto __seed = (static_cast<::cuda::std::uint64_t>(__j) * 0x129381294235245ULL)
                        ^ static_cast<::cuda::std::uint64_t>(__comm.rank());

      __scratch.__samples.resize_discard(__scratch.__samples.stream(), __estimate, ::cuda::no_init);
      __sample_probes(__input, __scratch.__I_j, __prob, __seed, __cmp, &__scratch.__samples, &__scratch.__samples_size);
    }

    __gather_merge_broadcast(
      __comms,
      __envs,
      __cmp,
      &__local_scratch,
      &__local_hist_results,
      &__root_recvcounts,
      &__root_displs,
      &__root_all_samples);

    __compute_histogram(__comms, __envs, __local_inputs, __cmp, &__local_hist_results);

    // Tighten brackets and rebuild intervals
    __update_intervals(__comms, __envs, __N, &__local_hist_results, &__local_scratch);
  }

  // NRVO does not apply to structured bindings, so need explicit move here
  return ::cuda::std::move(__local_hist_results);
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_HISTOGRAMMING_H
