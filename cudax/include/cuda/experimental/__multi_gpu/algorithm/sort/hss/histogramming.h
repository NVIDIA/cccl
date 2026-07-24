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
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <vector>

#if _CCCL_CTK_BELOW(12, 9)
#  include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#endif // CUDA 12.8-

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/bucket_count_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/ideal_rank_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/merge_k_way.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sample_probes.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/traits.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Interval-narrowing functor that tightens one splitter's `[L, U]` rank bracket.
//!
//! Drives the CUB `DeviceTransform` in `__update_intervals`. For a single splitter it receives
//! its target rank `Ni/p` (from `__ideal_rank_fn`) and its current `L` / `U` brackets, then walks
//! the all-reduced probe histogram accumulating the global rank (the prefix sum of per-bucket
//! counts). On an exact hit it collapses both brackets onto the found probe; otherwise it raises
//! `L` toward the target and lowers `U` toward the target on an overshoot. It returns the
//! tightened `(interval, L, U)`, where the interval is the pair of the two brackets' realized
//! keys and delimits the next round's sampling range.
//!
//! `__hist_begin` holds the globally reduced per-probe counts and `__probes_begin` the
//! matching probe keys.
//!
//! @tparam _Tp The key (value) type.
//! @tparam _Bracket The `__bracket_type` (rank plus optional key) used for the `L` / `U` ends.
template <class _Tp, class _Bracket>
struct __update_intervals_fn
{
  const _Tp* __probes_begin;
  const ::cuda::std::uint64_t* __hist_begin;
  ::cuda::std::size_t __num_probes;

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE constexpr ::cuda::std::
    tuple<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>, _Bracket, _Bracket>
    operator()(const _Tup& __tup) const noexcept
  {
    auto [__target, __L_i, __U_i] = __tup;
    // global_rank = number of input keys strictly less than probes[j]
    //             = prefix sum of per-bucket counts up to bucket j.
    ::cuda::std::uint64_t __global_rank = 0;

    for (::cuda::std::size_t __j = 0; __j < __num_probes; ++__j)
    {
      __global_rank += __hist_begin[__j];

      if (__global_rank == __target)
      {
        // Exact match, we have managed to find the perfect splitter value
        __L_i = __U_i = _Bracket{__global_rank, __probes_begin[__j]};
        break;
      }

      if ((__global_rank < __target) && (__global_rank > __L_i.__rank))
      {
        // We undershot the target, so we can raise our lower bound
        __L_i = _Bracket{__global_rank, __probes_begin[__j]};
      }
      else if ((__global_rank > __target) && (__global_rank < __U_i.__rank))
      {
        // Overshot the target but can lower the upper bound
        __U_i = _Bracket{__global_rank, __probes_begin[__j]};
      }
    }

    return ::cuda::std::make_tuple(::cuda::std::pair{__L_i.__key, __U_i.__key}, __L_i, __U_i);
  }
};

// Rank designated as the collective root for the HSS sampling phase.
inline constexpr ::cuda::std::int32_t __ROOT_RANK = 0;

template <class _Traits, class _CommRange, class _EnvRange>
[[nodiscard]]
_CCCL_HOST_API ::cuda::std::pair<::std::vector<typename _Traits::__per_comm_splitters_type>,
                                 ::std::vector<typename _Traits::__per_comm_sampling_scratch_type>>
__allocate_histogramming_buffers(
  const typename _Traits::__local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs)
{
  using _Tp      = typename _Traits::__value_type;
  using _Bracket = typename _Traits::__bracket_type;

  const auto __comm_size        = __setup.__comm_size;
  const auto __N                = __setup.__N;
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);
  // __per_comm_splitters (Ls/Us/probes) must survive the inner scratch block in __execute because
  // __data_exchange consumes them after the sampling K-loop finishes.
  ::std::vector<typename _Traits::__per_comm_splitters_type> __local_splitters;
  ::std::vector<typename _Traits::__per_comm_sampling_scratch_type> __local_scratch;

  __local_splitters.reserve(__num_local_inputs);
  __local_scratch.reserve(__num_local_inputs);

  for (auto&& [__comm, __env, __resource] : ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources))
  {
    const auto __stream  = ::cuda::get_stream(__env);
    const auto __n_split = __comm_size - 1;

    __local_splitters.emplace_back(typename _Traits::__per_comm_splitters_type{
      /*__Ls=*/__buffer_of<_Traits, _Bracket>{__stream, __resource, __n_split, _Bracket{0, ::cuda::std::nullopt}, __env},
      /*__Us=*/
      __buffer_of<_Traits, _Bracket>{__stream, __resource, __n_split, _Bracket{__N, ::cuda::std::nullopt}, __env},
      /*__probes=*/__buffer_of<_Traits, _Tp>{__stream, __resource, __env}});

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

      __local_scratch.emplace_back(typename _Traits::__per_comm_sampling_scratch_type{
        /*__I_j=*/
        __buffer_of<_Traits, ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>{
          __stream,
          __resource,
          __n_split,
          ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>{},
          __env},
        /*__samples=*/__buffer_of<_Traits, _Tp>{__stream, __resource, __env},
        /*__samples_size=*/
        __buffer_of<_Traits, ::cuda::std::size_t>{
          __stream, __resource, /*__size=*/__comm.rank() == __ROOT_RANK ? __comm_size : 1, ::cuda::no_init, __env},
        /*__hist=*/__buffer_of<_Traits, ::cuda::std::uint64_t>{__stream, __resource, __env},
        ::cuda::std::move(__probe_counts)});
    }
  }

  return ::cuda::std::make_pair(::cuda::std::move(__local_splitters), ::cuda::std::move(__local_scratch));
}

//! @brief Gather each rank's samples to the root, merge them, and broadcast the shared probes.
//!
//! Collective heart of one sampling round. The root-only scratch pointers are hoisted in by
//! the caller so their storage is reused across rounds (samples shrink  monotonically, so
//! nothing reallocates after round one).
//!
//! Reads `__comms`, `__envs`, `__cmp`, and each scratch's `__samples` / `__samples_size`; mutates
//! `*__local_scratch` (`__sample_sendcount`, `__probe_counts`), `*__local_splitters`
//! (`__probes`), `*__root_recvcounts`, `*__root_displs`, and `*__root_all_samples`.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value and buffer types.
//!
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[in,out] __local_scratch The per-comm sampling scratch; sample send counts and probe
//!                counts are written.
//! @param[in,out] __local_splitters The per-comm splitter state whose `__probes` receive the
//!                merged, broadcast probe set.
//! @param[in,out] __root_recvcounts The root's per-rank sample recvcounts, resized and refilled.
//! @param[in,out] __root_displs The root's exclusive-scan sample displacements, cleared and
//!                rebuilt each round.
//! @param[in,out] __root_all_samples The root's gathered-sample buffer, engaged on the first
//!                round and resized thereafter.
template <class _Traits, class _CommRange, class _EnvRange, class _BinaryOp>
_CCCL_HOST_API void __gather_merge_broadcast(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  const _BinaryOp& __cmp,
  ::std::vector<typename _Traits::__per_comm_sampling_scratch_type>* __local_scratch,
  ::std::vector<typename _Traits::__per_comm_splitters_type>* __local_splitters,
  ::std::vector<::cuda::std::size_t>* __root_recvcounts,
  ::std::vector<::cuda::std::size_t>* __root_displs,
  ::cuda::std::optional<__buffer_of<_Traits, typename _Traits::__value_type>>* __root_all_samples)
{
  using _Tp = typename _Traits::__value_type;

  // The root needs to know how big everyones sample vectors are so it can build its combined
  // sample vector
  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __scratch] : ::cuda::std::ranges::views::zip(__comms, *__local_scratch))
    {
      auto* const __ptr = __scratch.__samples_size.data();

      __comm.gather(__guard, __ptr, __ptr, /*__count=*/1, __ROOT_RANK, __scratch.__samples_size.__get().stream());
    }
  }

  for (auto&& [__comm, __env, __scratch] : ::cuda::std::ranges::views::zip(__comms, __envs, *__local_scratch))
  {
    auto& __samples_size               = __scratch.__samples_size.__get();
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

      __root_displs->clear();
      __root_displs->reserve(__root_recvcounts->size() + 1);

      // Defer until the last possible moment
      __samples_size.stream().sync();

      // __root_recv_counts[__root_rank] is exactly the root's send counts as well
      __sendcount = (*__root_recvcounts)[__ROOT_RANK];

      // recvcounts is likely relatively small (it is on the order of O(ranks)).
      ::cuda::std::exclusive_scan(
        __root_recvcounts->begin(),
        __root_recvcounts->end(),
        ::cuda::std::back_inserter(*__root_displs),
        ::cuda::std::size_t{0});
      // displs.back() is the LAST rank's offset (exclusive scan), so the total element count
      // = displs.back() + __root_recvcounts.back(). all_samples is vector<T>, sized in
      // ELEMENTS.
      const auto __all_recv = __root_displs->back() + __root_recvcounts->back();

      // First round engages the optional; later rounds reuse the allocation via resize
      // (samples shrink monotonically, so this never grows after round one).
      if (__root_all_samples->has_value())
      {
        ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(**__root_all_samples, __all_recv);
      }
      else
      {
        __root_all_samples->emplace(
          __samples_size.stream(), __samples_size.memory_resource(), __all_recv, ::cuda::no_init, __env);
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
        __scratch.__samples.__get().stream());
    }
  }

  // Root merges the p sorted runs into one sorted probe set
  for (auto&& [__comm, __env, __splitters, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__local_splitters, *__local_scratch))
  {
    if (__comm.rank() == __ROOT_RANK)
    {
      ::cuda::experimental::__detail::__hss_sort::__merge_k_way<_Traits>(
        __comm, __env, __root_all_samples->value(), *__root_recvcounts, *__root_displs, __cmp, &__splitters.__probes);

      __scratch.__probe_counts.front() = __splitters.__probes.size();
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

  for (auto&& [__comm, __splitters, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, *__local_splitters, *__local_scratch))
  {
    if (__comm.rank() != __ROOT_RANK)
    {
      // Wait for comm so we can access probe_counts.front()
      __scratch.__probe_counts.stream().sync();
      ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(
        __splitters.__probes, __scratch.__probe_counts.front());
    }
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __splitters, __scratch] :
         ::cuda::std::ranges::views::zip(__comms, *__local_splitters, *__local_scratch))
    {
      auto* const __ptr = __splitters.__probes.data();

      __comm.broadcast(
        __guard, __ptr, __ptr, __scratch.__probe_counts.front(), __ROOT_RANK, __splitters.__probes.__get().stream());
    }
  }
}

//! @brief Compute and globally reduce the per-probe histogram of the local keys.
//!
//! For each communicator it resizes `__scratch.__hist` to `num_probes + 1` buckets and runs one
//! CUB `DeviceTransform` over a counting iterator, driving a `__bucket_count_fn` seeded with the
//! rank's sorted local keys and the shared probe set, so bucket `b` receives the count of local
//! keys falling between consecutive probes. It then all-reduces the per-comm histograms
//! element-wise so every rank ends with the same global per-probe counts, which
//! `__update_intervals` consumes.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value and buffer types.
//!
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[in] __range_of_local_keys The range of per-rank sorted local key ranges.
//! @param[in] __local_splitters The per-comm splitter state supplying the probe keys.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[in,out] __local_scratch The per-comm scratch whose `__hist` is filled and all-reduced.
template <class _Traits, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp>
_CCCL_HOST_API void __compute_histogram(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __range_of_local_keys,
  const ::std::vector<typename _Traits::__per_comm_splitters_type>& __local_splitters,
  const _BinaryOp& __cmp,
  ::std::vector<typename _Traits::__per_comm_sampling_scratch_type>* __local_scratch)
{
  for (auto&& [__comm, __env, __keys, __splitters, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_local_keys, __local_splitters, *__local_scratch))
  {
    auto& __probes            = __splitters.__probes;
    auto& __hist              = __scratch.__hist;
    const auto __num_probes   = __probes.size();
    const auto __num_buckets  = __num_probes + 1;
    const auto __keys_first   = ::cuda::std::ranges::begin(__keys);
    const auto __probes_first = __probes.begin();

    ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(__hist, __num_buckets);

    auto __op =
      __bucket_count_fn<::cuda::std::remove_cvref_t<decltype(__keys_first)>,
                        ::cuda::std::remove_cvref_t<decltype(__probes_first)>,
                        _BinaryOp>{__keys_first, ::cuda::std::ranges::end(__keys), __probes_first, __num_probes, __cmp};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      __hist.begin(),
      __num_buckets,
      ::cuda::std::move(__op),
      __env);
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __scratch] : ::cuda::std::ranges::views::zip(__comms, *__local_scratch))
    {
      auto&& __hist     = __scratch.__hist;
      auto* const __ptr = __hist.data();

      __comm.all_reduce(__guard, __ptr, __ptr, __hist.size(), ::cuda::std::plus<>{}, __hist.__get().stream());
    }
  }
}

//! @brief Tighten each per-splitter `[L, U]` rank bracket from the all-reduced probe histogram.
//!
//! For every communicator this runs one CUB `DeviceTransform` over the splitter indices. Each
//! invocation of `__update_intervals_fn` walks the local probe histogram, accumulates the
//! global rank (the prefix sum of per-bucket counts), and narrows the bracket toward the ideal
//! rank `Ni/p` supplied by `__ideal_rank_fn`. The realized keys of the updated `__Ls` / `__Us`
//! brackets delimit the next round's sampling interval `__I_j`.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value, bracket, and buffer
//!                 types for this sort.
//!
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[in] __N The global key count.
//! @param[in,out] __local_splitters The per-comm bracket state whose `__Ls` / `__Us` / `__I_j`
//!                are tightened.
//! @param[in,out] __local_scratch The per-comm scratch whose `__I_j` interval column is
//!                rewritten from the updated brackets.
template <class _Traits, class _CommRange, class _EnvRange>
_CCCL_HOST_API void __update_intervals(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  ::cuda::std::uint64_t __N,
  ::std::vector<typename _Traits::__per_comm_splitters_type>* __local_splitters,
  ::std::vector<typename _Traits::__per_comm_sampling_scratch_type>* __local_scratch)
{
  using _Tp      = typename _Traits::__value_type;
  using _Bracket = typename _Traits::__bracket_type;

  for (auto&& [__comm, __env, __splitters, __scratch] :
       ::cuda::std::ranges::views::zip(__comms, __envs, *__local_splitters, *__local_scratch))
  {
    auto& __probes             = __splitters.__probes;
    auto& __histogram          = __scratch.__hist;
    auto& __Ls                 = __splitters.__Ls;
    auto& __Us                 = __splitters.__Us;
    auto& __I_j                = __scratch.__I_j;
    const auto __comm_size     = __comm.size();
    const auto __num_splitters = __I_j.size();
    const auto __I_j_begin     = __I_j.begin();
    const auto __Ls_begin      = __Ls.begin();
    const auto __Us_begin      = __Us.begin();

    auto __in = ::cuda::make_zip_iterator(
      ::cuda::make_transform_iterator(
        ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
      __Ls_begin,
      __Us_begin);
    auto __out = ::cuda::make_zip_iterator(__I_j_begin, __Ls_begin, __Us_begin);
    auto __op  = __update_intervals_fn<_Tp, _Bracket>{__probes.data(), __histogram.data(), __probes.size()};

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

//! @brief Run the full HSS histogramming (sampling) phase and return the finalized brackets.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value, bracket, and buffer
//!                 types.
//!
//! @param[in] __setup The local-setup result supplying resources, comm size, and `N`.
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[in] __local_inputs The range of per-rank sorted local key ranges.
//! @param[in] __cmp The comparator defining the sorted order.
//!
//! @return The per-comm splitter-state vector with finalized brackets and shared probe sets.
template <class _Traits, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp>
[[nodiscard]] _CCCL_HOST_API ::std::vector<typename _Traits::__per_comm_splitters_type> __histogramming_phase(
  const typename _Traits::__local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const _BinaryOp& __cmp)
{
  const auto __comm_size = __setup.__comm_size;
  const auto __N         = __setup.__N;
  auto [__local_splitters, __local_scratch] =
    ::cuda::experimental::__detail::__hss_sort::__allocate_histogramming_buffers<_Traits>(__setup, __comms, __envs);

  // Root-only scratch for __gather_merge_broadcast, hoisted out of the sampling loop so their
  // allocations are paid once per sort instead of once per round.
  //
  // recvcounts/displs are O(comm_size) host vectors of invariant size; all_samples is the
  // device buffer holding the gathered sample keys, which shrinks monotonically across
  // rounds. __gather_merge_broadcast reuses their storage each round (see the reuse handling
  // there).
  ::std::vector<::cuda::std::size_t> __root_recvcounts;
  ::std::vector<::cuda::std::size_t> __root_displs;
  ::cuda::std::optional<__buffer_of<_Traits, typename _Traits::__value_type>> __root_all_samples;

  // Note: K is small, on the order of ~1-10
  const auto __K = ::cuda::std::max(
    static_cast<::cuda::std::int32_t>(::cuda::std::ceil(::cuda::std::log10(::cuda::std::log10(__comm_size) / __eps))),
    1);
  const auto __s_j_interior = 2. * ::cuda::std::log(__comm_size) / __eps;

  for (::cuda::std::int32_t __j = 1; __j <= __K; ++__j)
  {
    const auto __s_j  = ::cuda::std::pow(__s_j_interior, static_cast<double>(__j) / static_cast<double>(__K));
    const auto __prob = ::cuda::std::min(__s_j * static_cast<double>(__comm_size) / static_cast<double>(__N), 1.);

    for (auto&& [__input, __scratch, __n_local] :
         ::cuda::std::ranges::views::zip(__local_inputs, __local_scratch, __setup.__local_original_sizes))
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

      ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(__scratch.__samples, __estimate);
      ::cuda::experimental::__detail::__hss_sort::__sample_probes<_Traits>(
        __input, __scratch.__I_j, __prob, __cmp, &__scratch.__samples, &__scratch.__samples_size);
    }

    ::cuda::experimental::__detail::__hss_sort::__gather_merge_broadcast<_Traits>(
      __comms,
      __envs,
      __cmp,
      &__local_scratch,
      &__local_splitters,
      &__root_recvcounts,
      &__root_displs,
      &__root_all_samples);

    ::cuda::experimental::__detail::__hss_sort::__compute_histogram<_Traits>(
      __comms, __envs, __local_inputs, __local_splitters, __cmp, &__local_scratch);

    // Tighten brackets and rebuild intervals
    ::cuda::experimental::__detail::__hss_sort::__update_intervals<_Traits>(
      __comms, __envs, __N, &__local_splitters, &__local_scratch);
  }

  return __local_splitters;
}
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_HISTOGRAMMING_H
