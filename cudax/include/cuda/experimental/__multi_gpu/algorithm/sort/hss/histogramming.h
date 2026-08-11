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
#include <cuda/std/__random/philox_engine.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__tuple_dir/tie.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/bucket_count_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/ideal_rank_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/merge_k_way.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Single-thread kernel that draws sample keys from the union of splitter intervals.
//!
//! Runs on exactly one grid thread (all others return immediately). It walks the per-splitter
//! brackets, whose key endpoints `[L.key, U.key)` are the sampling intervals, and for each one
//! locates the corresponding sorted sub-range of the local keys. Implements the per-round
//! sampling of the paper's histogram-sort-with-sampling loop.
//!
//! @param[in] __config The launch configuration used to compute the thread's grid rank.
//! @param[in] __gen The Philox random engine used for sampling.
//! @param[in] __prob The per-key sampling probability.
//! @param[in] __begin Pointer to the first local key; advanced internally across intervals.
//! @param[in] __end Pointer one past the last local key.
//! @param[in] __brackets The span of per-splitter `[L, U]` brackets to sample between.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[out] __samples The span the drawn sample keys are written into.
//! @param[out] __samples_size Receives the number of keys actually drawn.
//
// TODO(jfaibussowit):
//
// Parallelize with multiple threads (but not too many!). __brackets is O(p-1), so in
// practice at the absolute max a few thousand if you are running on the worlds
// largest supercomputers.
template <class _Config, class _Tp, class _BinaryOp>
_CCCL_KERNEL_ATTRIBUTES void __sample_probes_kernel(
  _Config __config,
  ::cuda::std::philox4x64 __gen,
  const double __prob,
  const _Tp* __begin,
  const _Tp* const __end,
  const ::cuda::std::span<const _Splitter<_Tp>> __I_j,
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
  for (const auto [__L_i, __U_i] : __I_j)
  {
    const auto& __lo = __L_i.__key;
    const auto& __hi = __U_i.__key;

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

//! @brief Launch the sampling kernel to draw sample keys from each rank's local input.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _InputIterRange, class _SizeTRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__local_sampling(
  _CommRange&& __comms,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  ::cuda::std::int32_t __j,
  double __sampling_probability,
  const _BinaryOp& __cmp,
  const ::std::vector<__per_comm_histogramming_result_type>& __local_hist_results,
  ::cuda::std::span<const ::cuda::std::size_t> __cap_displs,
  ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch)
{
  constexpr auto __launch_config =
    ::cuda::make_config(::cuda::make_hierarchy(::cuda::block_dims<1>(), ::cuda::grid_dims<1>()));

  _CCCL_ASSERT(__sampling_probability > 0, "Cannot have 0 probably of picking elements");
  _CCCL_ASSERT(__sampling_probability <= 1., "Cannot have >1 probably of picking elements");

  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);
  auto __comm_it                = ::cuda::std::ranges::begin(__comms);
  auto __input_it               = ::cuda::std::ranges::begin(__input_iters);
  auto __num_items_it           = ::cuda::std::ranges::begin(__num_items_range);
  auto& __scratch               = *__local_scratch;

  for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs;
       (void) ++__idx, (void) ++__comm_it, (void) ++__input_it, (void) ++__num_items_it)
  {
    const auto __rank   = __comm_it->rank();
    auto& __all_samples = __scratch[__idx].__all_samples;

    // Sized to hold every rank's slot at full capacity. The slots are ragged once the kernels
    // run, so the gather that follows sends only the true counts.
    __all_samples.resize_discard(__all_samples.stream(), __cap_displs.back(), ::cuda::no_init);

    // 0x129381294235245ULL chosen randomly, by random dice roll
    const auto __seed =
      (static_cast<::cuda::std::uint64_t>(__j) * 0x129381294235245ULL) ^ static_cast<::cuda::std::uint64_t>(__rank);

    const auto& __I_j  = __local_hist_results[__idx].__splitters.__I_j;
    const auto* __keys = ::cuda::std::to_address(*__input_it);

    ::cuda::launch(
      // All inputs should be on the same stream here
      __I_j.stream(),
      __launch_config,
      ::cuda::experimental::__detail::__hss_sort::
        __sample_probes_kernel<::cuda::std::remove_cvref_t<decltype(__launch_config)>, _Tp, _BinaryOp>,
      ::cuda::std::philox4x64{__seed},
      __sampling_probability,
      __keys,
      __keys + *__num_items_it,
      __I_j,
      __cmp,
      // Each rank samples directly into its own slot of `__all_samples` so that we can
      // allgather them in place later
      __all_samples.subspan(__cap_displs[__rank], __cap_displs[__rank + 1] - __cap_displs[__rank]),
      // Likewise, write to our rank slot for an inplace allgather
      __scratch[__idx].__samples_size.data() + __rank);
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT

//! @brief Interval-narrowing functor that tightens one splitter's `[L, U]` rank bracket.
template <class _Tp>
struct __update_intervals_fn
{
  const _Tp* const __probes_begin;
  const ::cuda::std::uint64_t* const __hist_begin;
  const ::cuda::std::size_t __num_probes;

  [[nodiscard]] _CCCL_DEVICE_API constexpr _Splitter<_Tp>
  operator()(const ::cuda::std::uint64_t __target, _Splitter<_Tp> __splitter) const noexcept
  {
    auto& [__L_i, __U_i] = __splitter;

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

    return __splitter;
  }
};

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

  auto __comm_it = ::cuda::std::ranges::begin(__comms);
  auto __env_it  = ::cuda::std::ranges::begin(__envs);

  for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it, (void) ++__env_it)
  {
    const auto __stream  = ::cuda::get_stream(*__env_it);
    const auto __n_split = __comm_size - 1;

    auto&& __resource =
      ::cuda::experimental::__detail::__resource_from_env(*__env_it, __comm_it->logical_device().underlying_device());
    auto&& __buffer_env = ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it);

    __local_scratch.emplace_back(__per_comm_sampling_scratch_type{
      /*__all_samples=*/
      __resizable_buffer_type<_Tp>{__stream, __resource, __buffer_env},
      /*__samples_size=*/
      __resizable_buffer_type<::cuda::std::size_t>{__stream, __resource, __comm_size, ::cuda::no_init, __buffer_env}});

    __local_hist_results.emplace_back(__per_comm_histogramming_result_type{
      /*__splitters=*/
      __per_comm_splitters_type{
        // The starting bracket for every splitter is the whole array: rank 0 below and rank __N
        // above, neither yet realized by a key. A keyless bracket pair is also the "sample
        // everything" interval the first sampling round wants.
        /*__I_j=*/
        __resizable_buffer_type<_Splitter<_Tp>>{::cuda::make_buffer<_Splitter<_Tp>>(
          __stream,
          __resource,
          __n_split,
          _Splitter<_Tp>{/*__Ls=*/_Bracket<_Tp>{0, ::cuda::std::nullopt},
                         /*__Us*/ _Bracket<_Tp>{__N, ::cuda::std::nullopt}},
          __buffer_env)},
        /*__probes=*/
        __resizable_buffer_type<_Tp>{__stream, __resource, __buffer_env}},
      /*__hist=*/__resizable_buffer_type<::cuda::std::uint64_t>{__stream, __resource, __buffer_env}});
  }

  return ::cuda::std::make_pair(::cuda::std::move(__local_scratch), ::cuda::std::move(__local_hist_results));
}

//! @brief Exchange every rank's true sample count and read it back onto the host.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__exchange_sample_counts(
  _CommRange&& __comms,
  ::cuda::std::span<::cuda::std::size_t> __h_recvcounts,
  ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch)
{
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  // Every rank needs everyone's true sample counts because all_gather_v takes the counts on
  // the host. The slot layout itself needs no communication, it comes from the capacity bounds
  // that we already computed in the sampling phase.
  //
  // We can't just infer this from the (already gathered) global offsets and sampling
  // probability because the actual number of probes is probabilistic. If we switched to a
  // fixed sampling regime then we could deduce everyones sizes without communication here.
  //
  // I am fairly certain this would still be correct w.r.t. the paper. The only property that
  // needs to hold there is that every interval is sampled with a particular density. The
  // sampling probability just bounds the density but there should be no reason it cannot be
  // concrete.
  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto&& __guard = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it)
    {
      auto& __samples_size = (*__local_scratch)[__idx].__samples_size;
      auto* const __ptr    = __samples_size.data();

      __comm_it->all_gather(__guard, __ptr + __comm_it->rank(), __ptr, /*__count=*/1, __samples_size.stream());
    }
  }

  // We need to copy only once here because the all gather above ensures all ranks have the
  // same samples-size entries
  const auto& __samples_size = __local_scratch->front().__samples_size;

  ::cuda::copy_bytes(
    __samples_size.stream(),
    __samples_size,
    __h_recvcounts,
    ::cuda::copy_configuration{::cuda::std::ranges::begin(__comms)->logical_device().underlying_device(),
                               ::cuda::host_memory_location,
                               ::cuda::source_access_order::stream});

  // Need to sync here because __gather_merge_probes() needs these on the host for comms. Could
  // potentially move this sync there
  __samples_size.stream().sync();
}

//! @brief Gather the sample keys onto every rank and merge them into each rank's probe set.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__gather_and_merge_probes(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  const _BinaryOp& __cmp,
  ::cuda::std::span<const ::cuda::std::size_t> __h_recvcounts,
  ::cuda::std::span<const ::cuda::std::size_t> __h_cap_displs,
  ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch,
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results)
{
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  // Gather all samples onto every rank so each one can build the global sampling vector
  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto&& __guard = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it)
    {
      auto& __all_samples = (*__local_scratch)[__idx].__all_samples;
      const auto __rank   = __comm_it->rank();

      // A rank that overfills its slot means the capacity bound was too tight and the sampling
      // kernel truncated, which silently lowers the sampling density.
      _CCCL_VERIFY(__h_recvcounts[__rank] <= (__h_cap_displs[__rank + 1] - __h_cap_displs[__rank]),
                   "We have sampled more items than the rank's slot can hold, "
                   "this is an uncaught buffer overrun");

      __comm_it->all_gather_v(
        __guard,
        __all_samples.data() + __h_cap_displs[__rank],
        __h_recvcounts[__rank],
        __all_samples.data(),
        __h_recvcounts.data(),
        __h_cap_displs.data(),
        __all_samples.stream());
    }
  }

  // Every rank merges the p sorted runs into one sorted probe set
  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto __env_it  = ::cuda::std::ranges::begin(__envs);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs;
         (void) ++__idx, (void) ++__comm_it, (void) ++__env_it)
    {
      auto& __probes = (*__local_hist_results)[__idx].__splitters.__probes;

      __merge_k_way(
        *__comm_it,
        *__env_it,
        (*__local_scratch)[__idx].__all_samples,
        __h_recvcounts,
        __h_cap_displs.first(__h_recvcounts.size()),
        __cmp,
        &__probes);
    }
  }
}

//! @brief Compute and globally reduce the per-probe histogram of the local keys.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__compute_histogram(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __key_iters,
  _SizeTRange&& __num_items_range,
  const _BinaryOp& __cmp,
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results)
{
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  {
    auto __comm_it      = ::cuda::std::ranges::begin(__comms);
    auto __env_it       = ::cuda::std::ranges::begin(__envs);
    auto __keys_it      = ::cuda::std::ranges::begin(__key_iters);
    auto __num_items_it = ::cuda::std::ranges::begin(__num_items_range);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs;
         (void) ++__idx, (void) ++__comm_it, (void) ++__env_it, (void) ++__keys_it, (void) ++__num_items_it)
    {
      auto& __hist               = (*__local_hist_results)[__idx].__hist;
      auto& __probes             = (*__local_hist_results)[__idx].__splitters.__probes;
      const auto __num_probes    = __probes.size();
      const auto __num_buckets   = __num_probes + 1;
      const auto* __keys_first   = ::cuda::std::to_address(*__keys_it);
      const auto* __probes_first = __probes.data();

      __hist.resize_discard(__hist.stream(), __num_buckets, ::cuda::no_init);

      auto __op =
        __bucket_count_fn<::cuda::std::remove_cvref_t<decltype(__keys_first)>,
                          ::cuda::std::remove_cvref_t<decltype(__probes_first)>,
                          _BinaryOp>{__keys_first, __keys_first + *__num_items_it, __probes_first, __num_probes, __cmp};

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm_it->logical_device(),
        CUB_NS_QUALIFIER::DeviceTransform::Transform,
        ::cuda::counting_iterator<::cuda::std::uint64_t>{},
        __hist.data(),
        __num_buckets,
        ::cuda::std::move(__op),
        *__env_it);
    }
  }

  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto&& __guard = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it)
    {
      auto& __hist      = (*__local_hist_results)[__idx].__hist;
      auto* const __ptr = __hist.data();

      __comm_it->all_reduce(__guard, __ptr, __ptr, __hist.size(), ::cuda::std::plus<>{}, __hist.stream());
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
  ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results)
{
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  auto __comm_it = ::cuda::std::ranges::begin(__comms);
  auto __env_it  = ::cuda::std::ranges::begin(__envs);

  for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it, (void) ++__env_it)
  {
    const auto __comm_size = __comm_it->size();
    auto& __splitters      = (*__local_hist_results)[__idx].__splitters;
    const auto& __hist     = (*__local_hist_results)[__idx].__hist;
    auto& __I_j            = __splitters.__I_j;
    auto& __probes         = __splitters.__probes;

    // Each splitter's bracket is narrowed against its own ideal rank and written straight back
    // over itself. The bracket keys double as the next round's sampling interval, so there is
    // nothing further to project out.
    auto __in = ::cuda::make_transform_iterator(
      ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size});
    auto __op = __update_intervals_fn<_Tp>{__probes.data(), __hist.data(), __probes.size()};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm_it->logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::std::make_tuple(::cuda::std::move(__in), __I_j.data()),
      __I_j.data(),
      __I_j.size(),
      ::cuda::std::move(__op),
      *__env_it);
  }
}

template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
[[nodiscard]]
_CCCL_HOST_API ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::__per_comm_histogramming_result_type>
_HSSSorter<_Tp, _Env, _BinaryOp>::__histogramming_phase(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  const _BinaryOp& __cmp)
{
  const auto __comm_size                       = __setup.__comm_size;
  const auto __N                               = __setup.__N;
  auto [__local_scratch, __local_hist_results] = __allocate_histogramming_buffers(__setup, __comms, __envs);

  // Host scratch for the sample gather, hoisted out of the sampling loop so it is sized once
  // per sort instead of once per round. Every rank all-gathers the same counts, so one set of
  // vectors serves every local comm.
  //
  // __cap_displs holds where each rank's slot starts in the combined sample buffer, with a
  // trailing total so that slot r spans [__cap_displs[r], __cap_displs[r + 1]). It is derived
  // on the host with no communication: round one from the per-rank input sizes, later rounds
  // from the previous round's counts (samples shrink monotonically as the brackets tighten).
  // __recvcounts is how much of each slot the sampling kernels actually filled.
  ::std::vector<::cuda::std::size_t> __host_scratch((2 * __comm_size) + 1);
  auto __recvcounts = ::cuda::std::span<::cuda::std::size_t>{__host_scratch.data(), __comm_size};
  auto __cap_displs = ::cuda::std::span<::cuda::std::size_t>{__host_scratch.data() + __comm_size, __comm_size + 1};

  _CCCL_ASSERT(__recvcounts.size() + __cap_displs.size() == __host_scratch.size(), "");

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

    __cap_displs[0] = 0;
    for (::cuda::std::int32_t __r = 0; __r < __comm_size; ++__r)
    {
      // Bound every rank's sample count for this round and lay the slots out back to back.
      //
      // Each iteration we sample the union of splitter intervals, `\gamma_j` with probability
      // `__prob`. For the first iteration, `\gamma_j` is the entire array, so the ranks input
      // size bounds it. After that, the host cannot measure `\gamma_j` because:
      //
      // 1. We can't inspect the updated splitter brackets, and
      // 2. We can't count how many of our elements actually lie within those updated
      //    intervals.
      //
      // But the brackets only ever tighten, so each rounds samples a smaller region than the
      // last. The previous round's count is therefore an upper bound on this one. Both bounds
      // are true upper bounds, so a slot can never overflow.
      //
      // The kernel writes straight into this rank's slot of the combined sample buffer, so there
      // is no separate per-rank sample buffer to gather out of later.
      const auto __cap =
        (__j == 1) ? static_cast<::cuda::std::size_t>(::cuda::std::ceil(__setup.__all_local_sizes[__r] * __prob))
                   : __recvcounts[__r];

      __cap_displs[__r + 1] = __cap_displs[__r] + ::cuda::std::max(__cap, ::cuda::std::size_t{1});
    }

    __local_sampling(
      __comms,
      __input_iters,
      __num_items_range,
      __j,
      __prob,
      __cmp,
      __local_hist_results,
      __cap_displs,
      &__local_scratch);

    __exchange_sample_counts(__comms, /*mut ref*/ __recvcounts, &__local_scratch);

    __gather_and_merge_probes(
      __comms, __envs, __cmp, __recvcounts, __cap_displs, &__local_scratch, &__local_hist_results);

    __compute_histogram(__comms, __envs, __input_iters, __num_items_range, __cmp, &__local_hist_results);

    // Tighten the brackets, which are also the next round's sampling intervals
    __update_intervals(__comms, __envs, __N, &__local_hist_results);
  }

  // NRVO does not apply to structured bindings, so need explicit move here
  return ::cuda::std::move(__local_hist_results);
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_HISTOGRAMMING_H
