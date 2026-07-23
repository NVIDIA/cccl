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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__container/buffer.h>
#include <cuda/__container/make_buffer_with_pool.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/__nvtx/nvtx.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cmath/exponential_functions.h>
#include <cuda/std/__cmath/logarithms.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#if _CCCL_CTK_BELOW(12, 9)
#  include <cuda/__memory_resource/legacy_pinned_memory_resource.h>
#endif // CUDA 12.8-

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/bracket.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/buffer.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/ideal_rank_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/sample_probes.h>
#include <cuda/experimental/__multi_gpu/concepts.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
namespace __detail
{
// The following functors replace what would naturally be written as
// function-local lambdas at their use sites inside _Sorter. They are hoisted to
// named, namespace-scope types on purpose: NVCC cannot name a function-local
// closure as a `__global__` kernel template argument when it generates the host
// registration stub, emitting "insufficient contextual information to determine
// type" in *.cudafe1.stub.c. A named functor is nameable there. They live
// outside _Sorter, and depend only on the types they actually use, so they are
// not re-instantiated per _Env/_BinaryOp.

// Interval-narrowing functor for __update_intervals. Given (target_rank, L, U),
// walks the local probe histogram and tightens the [L, U] bracket, returning
// the updated interval and brackets.
template <class _Tp, class _Bracket>
struct __update_intervals_fn
{
  const _Tp* __probes_begin;
  const ::cuda::std::uint64_t* __hist_begin;
  ::cuda::std::size_t __num_probes;

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::
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

// Splitter-selection functor used by __data_exchange (via the lazy __splitter_it
// transform_iterator). Given (target_rank, L, U), returns the bracket endpoint key closest to
// the target rank, realizing an unset (unbounded) endpoint from the probe extrema. This is HSS
// step (5) of Section 4.2.2: "Once the histogramming phase finishes, the key ranked closest to
// Ni/p among the keys seen so far is set as the ith splitter." The (L, U) bracket it consumes
// is the Table 1 notation L(i)/U(i) (ranks of the largest sample key below / smallest sample
// key above the ideal rank Ni/p), whose realized keys delimit the splitter interval I(i) =
// [I(L(i)), I(U(i))].
template <class _Tp, class _Probe>
struct __finalize_splitters_fn
{
  _Probe __first_probe;
  _Probe __last_probe;

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp operator()(const _Tup& __tup) const noexcept
  {
    const auto [__target_rank, __L_i, __U_i] = __tup;
    // Pick the bracket endpoint closest to the target rank. The chosen key becomes the
    // splitter used by data exchange.
    const bool __use_L = (__target_rank - __L_i.__rank) <= (__U_i.__rank - __target_rank);

    if (__use_L)
    {
      return __L_i.__key.has_value() ? *__L_i.__key : *__first_probe;
    }
    return __U_i.__key.has_value() ? *__U_i.__key : *__last_probe;
  }
};

// Per-bucket count functor shared by __local_histogram and __data_exchange. Both phases walk a
// sorted key range against a monotonically increasing bucket index, and for bucket b emit the
// number of keys landing in [split[b - 1], split[b]) (the boundary buckets are half-open at
// the range ends). Because CUB drives the Transform with a monotonically increasing bucket
// index and a fixed stride, we cache the previous upper bound in __lo: the next bucket's lower
// bound can never precede it, so each call only needs to search forward from __lo. This is why
// operator() is mutable -- it advances the __lo cursor across calls.
template <class _KeyIt, class _SplitIt, class _Cmp>
struct __bucket_count_fn
{
  mutable _KeyIt __lo; // mutable cursor: last computed upper bound, reused as the next lower bound
  _KeyIt __keys_last;
  _SplitIt __split_it;
  ::cuda::std::uint64_t __num_splitters;
  _Cmp __cmp;

  // operator() is intentionally non-const: it advances the __lo cursor across successive
  // monotonic bucket invocations.
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr ::cuda::std::uint64_t
  operator()(::cuda::std::uint64_t __bucket) const noexcept
  {
    auto __hi = __keys_last;
    // This reordering takes advantage of the fact that we cached __lo previously. In
    // that case, __lo will already have raised the lower bound of the search. We now
    // just need to raise the *upper* bound of the search. We cannot do that generally by
    // caching (because the operators are called from left to right), but we *can* bound
    // the search for __lo.
    if (__bucket != __num_splitters)
    {
      __hi = ::cuda::std::lower_bound(__lo, __keys_last, __split_it[__bucket], __cmp);
    }

    if (__bucket != 0)
    {
      __lo = ::cuda::std::lower_bound(__lo, __hi, __split_it[__bucket - 1], __cmp);
    }

    const auto __ret = static_cast<::cuda::std::uint64_t>(__hi - __lo);
    // This caching of __lo relies on the fact that CUB calls these operators with
    // monotonically increasing bucket count (i.e. the stride is always adding blockIdx.x). If
    // that doesn't happen, then this caching is wrong.
    __lo = __hi;
    return __ret;
  }
};

template <class _Tp, class _Env, class _BinaryOp>
struct _Sorter
{
  using __resource_type = __resource_type_for<_Env>;

  // Alias the de-nested infrastructure types back into _Sorter so the phase methods below can
  // keep referring to them by their short, resource-free names. These will be replaced by
  // fully-qualified uses as the phases are pulled out into their own headers.
  template <class _Up>
  using __buffer = ::cuda::experimental::__detail::__buffer<_Up, __resource_type>;

  using _Bracket = ::cuda::experimental::__detail::__bracket<_Tp>;

  static constexpr double __EPS     = 0.02; // 2% tolerance
  static constexpr auto __ROOT_RANK = 0;

  // TODO(jfaibussowit):
  //
  // Horrifically inefficient!
  template <class _Comm>
  _CCCL_HOST_API static void __merge_k_way(
    const _Comm& __comm,
    const _Env& __env,
    const __buffer<_Tp>& __data,
    const ::std::vector<::cuda::std::size_t>& __counts,
    const ::std::vector<::cuda::std::size_t>& __displs,
    const _BinaryOp& __cmp,
    __buffer<_Tp>* __ret)
  {
    if (__counts.size() < 2)
    {
      // TODO(jfaibussowit):
      //
      // Handle properly
      _CCCL_VERIFY(__displs.empty() || __displs.front() == 0, "Nonzero displacement for first entry");
      // 0 or 1 inputs, we just copy directly, nothing to merge
      *__ret = __data;
      return;
    }

    const auto __total = __counts.back() + __displs.back();

    ::cuda::experimental::__detail::__resize_for_overwrite(*__ret, __total);

    auto __tmp_buf = __ret->__make_empty_like(__total);

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
      __data.begin() + __displs[0],
      __counts[0],
      __data.begin() + __displs[1],
      __counts[1],
      __ret->begin(),
      __cmp,
      __env);

    ::cuda::std::size_t __merged_size = __counts[0] + __counts[1];

    for (::cuda::std::size_t __i = 2; __i < __displs.size(); ++__i)
    {
      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
        __ret->begin(),
        __merged_size,
        __data.begin() + __displs[__i],
        __counts[__i],
        __tmp_buf.begin(),
        __cmp,
        __env);

      __ret->__get().swap(__tmp_buf);
      __merged_size += __counts[__i];
    }
  }

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __gather_merge_broadcast(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__buffer<_Tp>>& __local_samples,
    const _BinaryOp& __cmp,
    ::std::vector<__buffer<::cuda::std::size_t>>* __local_samples_size,
    ::std::vector<::cuda::std::size_t>* __local_sendcounts,
    ::std::vector<__buffer<_Tp>>* __local_probes,
    ::std::vector<::cuda::buffer<::cuda::std::uint64_t, ::cuda::mr::device_accessible, ::cuda::mr::host_accessible>>*
      __local_probe_counts,
    ::std::vector<::cuda::std::size_t>* __root_recvcounts,
    ::std::vector<::cuda::std::size_t>* __root_displs,
    ::cuda::std::optional<__buffer<_Tp>>* __root_all_samples)
  {
    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __samples_size] :
           ::cuda::std::ranges::views::zip(__comms, __streams, *__local_samples_size))
      {
        auto* const __ptr = __samples_size.data();

        __comm.gather(__guard, __ptr, __ptr, /*__count=*/1, __ROOT_RANK, __stream);
      }
    }

    for (auto&& [__comm, __stream, __env, __samples_size, __sendcount] :
         ::cuda::std::ranges::views::zip(__comms, __streams, __envs, *__local_samples_size, *__local_sendcounts))
    {
      if (__comm.rank() == __ROOT_RANK)
      {
        // __root_recvcounts, __root_displs, and __root_all_samples are hoisted out of the
        // sampling loop and passed in by pointer so their allocations are paid once per sort
        // rather than once per round. __root_displs is filled via back_inserter, so it must
        // start each round empty; the recvcounts resize and the all_samples resize below reuse
        // the existing storage (sizes are invariant / monotonically shrinking across rounds,
        // so neither reallocates after round one).
        __root_recvcounts->resize(__samples_size.size());

        ::cuda::copy_bytes(
          __stream,
          __samples_size.__get(),
          *__root_recvcounts,
          ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                     ::cuda::host_memory_location,
                                     ::cuda::source_access_order::stream});

        __root_displs->clear();
        __root_displs->reserve(__root_recvcounts->size() + 1);

        // Defer until the last possible moment
        __stream.sync();

        // __root_recv_counts[__ROOT_RANK] is exactly the root's send counts as
        // well
        __sendcount = (*__root_recvcounts)[__ROOT_RANK];

        // recvcounts is likely relatively small (it is on the order of
        // O(ranks)). It may be faster to just do the exclusive scan on the host
        // than to copy to device, scan, and copy back
        ::cuda::std::exclusive_scan(
          __root_recvcounts->begin(),
          __root_recvcounts->end(),
          ::cuda::std::back_inserter(*__root_displs),
          ::cuda::std::size_t{0});
        // displs.back() is the LAST rank's offset (exclusive scan), so the total
        // element count = displs.back() + __root_recvcounts.back(). all_samples
        // is vector<T>, sized in ELEMENTS.
        const auto __all_recv = __root_displs->back() + __root_recvcounts->back();

        // First round engages the optional; later rounds reuse the allocation via resize
        // (samples shrink monotonically, so this never grows after round one).
        if (__root_all_samples->has_value())
        {
          ::cuda::experimental::__detail::__resize_for_overwrite(**__root_all_samples, __all_recv);
        }
        else
        {
          __root_all_samples->emplace(
            __stream, __samples_size.__get().memory_resource(), __all_recv, ::cuda::no_init, __env);
        }
      }
      else
      {
        // Non-root, __samples_size.size() should be == 1
        ::cuda::copy_bytes(
          __stream,
          __samples_size.__get(),
          ::cuda::std::span{&__sendcount, ::cuda::std::size_t{1}},
          ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                     ::cuda::host_memory_location,
                                     ::cuda::source_access_order::stream});
        __stream.sync();
      }
    }

    // ---- 3. Variable-count gather of the actual sample keys ----
    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __samples, __sendcount] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_samples, *__local_sendcounts))
      {
        __comm.gather_v(
          __guard,
          __samples.data(),
          __sendcount,
          __comm.rank() == __ROOT_RANK ? __root_all_samples->value().data() : nullptr,
          __root_recvcounts->data(),
          __root_displs->data(),
          __ROOT_RANK,
          __stream);
      }
    }

    // ---- 4. Root merges the p sorted runs into one sorted probe set ----
    for (auto&& [__comm, __env, __probes, __probe_count] :
         ::cuda::std::ranges::views::zip(__comms, __envs, *__local_probes, *__local_probe_counts))
    {
      if (__comm.rank() == __ROOT_RANK)
      {
        __merge_k_way(__comm, __env, __root_all_samples->value(), *__root_recvcounts, *__root_displs, __cmp, &__probes);

        __probe_count.front() = __probes.size();
        break;
      }
    }

    // Extremely painful stuff here. We need to send the probe count, but we can
    // only use NCCL, and NCCL only provides device transport (even though they
    // definitely have host-host transport available internally).
    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __probe_count] :
           ::cuda::std::ranges::views::zip(__comms, __streams, *__local_probe_counts))
      {
        auto* const __ptr = __probe_count.data();

        __comm.broadcast(__guard, __ptr, __ptr, 1, __ROOT_RANK, __stream);
      }
    }

    for (auto&& [__comm, __stream, __probes, __probe_count] :
         ::cuda::std::ranges::views::zip(__comms, __streams, *__local_probes, *__local_probe_counts))
    {
      if (__comm.rank() != __ROOT_RANK)
      {
        // wait for comm
        __stream.sync();
        ::cuda::experimental::__detail::__resize_for_overwrite(__probes, __probe_count.front());
      }
    }

    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __probes, __probe_count] :
           ::cuda::std::ranges::views::zip(__comms, __streams, *__local_probes, *__local_probe_counts))
      {
        auto* const __ptr = __probes.data();

        __comm.broadcast(__guard, __ptr, __ptr, __probe_count.front(), __ROOT_RANK, __stream);
      }
    }
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __local_histogram(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __range_of_local_keys,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    const _BinaryOp& __cmp,
    ::std::vector<__buffer<::cuda::std::uint64_t>>* __local_hist)
  {
    for (auto&& [__comm, __env, __keys, __probes, __hist] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_local_keys, __local_probes, *__local_hist))
    {
      const auto __num_probes   = __probes.size();
      const auto __num_buckets  = __num_probes + 1;
      const auto __keys_first   = ::cuda::std::ranges::begin(__keys);
      const auto __probes_first = __probes.begin();

      ::cuda::experimental::__detail::__resize_for_overwrite(__hist, __num_buckets);

      auto __op = ::cuda::experimental::__detail::__bucket_count_fn<
        ::cuda::std::remove_cvref_t<decltype(__keys_first)>,
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
  }

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __update_intervals(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    const ::std::vector<__buffer<::cuda::std::uint64_t>>& __local_hist,
    ::cuda::std::uint64_t __N,
    ::std::vector<__buffer<_Bracket>>* __local_Ls,
    ::std::vector<__buffer<_Bracket>>* __local_Us,
    ::std::vector<__buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>>* __local_I_js)
  {
    for (auto&& [__comm, __env, __probes, __histogram, __Ls, __Us, __I_j] : ::cuda::std::ranges::views::zip(
           __comms, __envs, __local_probes, __local_hist, *__local_Ls, *__local_Us, *__local_I_js))
    {
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

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __data_exchange(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__resource_type>& __resources,
    const ::std::vector<__buffer<_Bracket>>& __local_Ls,
    const ::std::vector<__buffer<_Bracket>>& __local_Us,
    const ::std::vector<__buffer<_Tp>>& __local_probes,
    ::cuda::std::uint64_t __N,
    _BinaryOp __cmp,
    _InputRange&& __local_inputs)
  {
    ::std::vector<__buffer<::cuda::std::size_t>> __local_send_counts;
    ::std::vector<__buffer<::cuda::std::size_t>> __local_recv_counts;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts;

    ::std::vector<__buffer<_Tp>> __local_recvd;

    const auto __comm_size        = ::cuda::std::ranges::begin(__comms)->size();
    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    __local_send_counts.reserve(__num_local_inputs);
    __local_recv_counts.reserve(__num_local_inputs);
    __local_h_send_counts.reserve(__num_local_inputs);
    __local_h_recv_counts.reserve(__num_local_inputs);

    __local_recvd.reserve(__num_local_inputs);

    for (auto&& [__comm, __env, __stream, __resource, __input, __Ls, __Us, __probes] : ::cuda::std::ranges::views::zip(
           __comms, __envs, __streams, __resources, __local_inputs, __local_Ls, __local_Us, __local_probes))
    {
      auto& __send_counts = __local_send_counts.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

      const auto __input_begin = ::cuda::std::ranges::begin(__input);

      // Lazily reconstruct the finalized splitters (HSS Section 4.2.2 step (5), "the key
      // ranked closest to Ni/p ... is set as the ith splitter") on the fly instead of
      // materializing them. Rather than a separate Transform launch writing a splitter buffer
      // that this kernel then reads, we fuse __finalize_splitters_fn into the data-exchange
      // kernel through a transform_iterator: each __splitter_it[d] evaluates finalize on
      // demand, eliminating one Transform launch and the splitter buffer. The ideal rank Ni/p
      // (the center of the Section 2 / Table 1 target range Ti) is supplied per-splitter by
      // __ideal_rank_fn.
      auto __splitter_it = ::cuda::make_transform_iterator(
        ::cuda::make_zip_iterator(
          ::cuda::make_transform_iterator(
            ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm.size()}),
          __Ls.begin(),
          __Us.begin()),
        __finalize_splitters_fn<_Tp, ::cuda::std::remove_cvref_t<decltype(__probes.begin())>>{
          __probes.begin(), __probes.end() - 1});

      // Route this rank's local keys to destination ranks via the splitter keys: the Data
      // Exchange phase, HSS Section 3.1 step (3), "a key in range [S(i), S(i + 1)) goes to
      // processor i". HSS reuses this phase unchanged (Section 3.3), so bucket d receives the keys
      // in [S(d - 1), S(d)) and its count becomes the send metadata. The send displacements are the
      // exclusive prefix-sum of these counts (buckets are contiguous and non-overlapping), so we
      // recompute them on the host below instead of emitting a second device column here.
      auto __op = ::cuda::experimental::__detail::__bucket_count_fn<
        ::cuda::std::remove_cvref_t<decltype(__input_begin)>,
        ::cuda::std::remove_cvref_t<decltype(__splitter_it)>,
        _BinaryOp>{__input_begin, ::cuda::std::ranges::end(__input), __splitter_it, __Ls.size(), __cmp};

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceTransform::Transform,
        ::cuda::counting_iterator<::cuda::std::uint64_t>{},
        __send_counts.begin(),
        __comm_size,
        ::cuda::std::move(__op),
        __env);
    }

    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __send_counts] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_send_counts))
      {
        auto& __recv_counts    = __local_recv_counts.emplace_back(__send_counts.__make_empty_like());
        auto* const __send_ptr = __send_counts.data();
        auto* const __recv_ptr = __recv_counts.data();

        __comm.all_to_all(__guard, __send_ptr, __recv_ptr, /*__count=*/1, __stream);
      }
    }

    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs;

    __local_h_send_displs.reserve(__num_local_inputs);
    __local_h_recv_displs.reserve(__num_local_inputs);
    for (auto&& [__comm, __stream, __resource, __env, __send_counts, __recv_counts] : ::cuda::std::ranges::views::zip(
           __comms, __streams, __resources, __envs, __local_send_counts, __local_recv_counts))
    {
      auto& __h_send_counts = __local_h_send_counts.emplace_back(__send_counts.size());
      auto& __h_recv_counts = __local_h_recv_counts.emplace_back(__recv_counts.size());

      auto& __h_send_displs = __local_h_send_displs.emplace_back();
      auto& __h_recv_displs = __local_h_recv_displs.emplace_back();

      // The send/recv displacements are just the exclusive prefix-sums of the
      // corresponding counts, and both are consumed only on the host (below and in the
      // all_to_all_v). counts is small (O(ranks)), so we scan on the host after the
      // sync instead of paying a device scan plus a D2H copy of the result.
      __h_send_displs.reserve(__send_counts.size());
      __h_recv_displs.reserve(__recv_counts.size());

      ::cuda::copy_bytes(
        __stream,
        __send_counts.__get(),
        __h_send_counts,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __stream,
        __recv_counts.__get(),
        __h_recv_counts,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __stream.sync();

      // Host counts are only valid post-sync, so scan the displacements here.
      ::cuda::std::exclusive_scan(
        __h_send_counts.begin(),
        __h_send_counts.end(),
        ::cuda::std::back_inserter(__h_send_displs),
        ::cuda::std::size_t{0});
      ::cuda::std::exclusive_scan(
        __h_recv_counts.begin(),
        __h_recv_counts.end(),
        ::cuda::std::back_inserter(__h_recv_displs),
        ::cuda::std::size_t{0});

      const auto __total_recv = __h_recv_displs.back() + __h_recv_counts.back();

      __local_recvd.emplace_back(__stream, __resource, __total_recv, ::cuda::no_init, __env);
    }

    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm,
                   __stream,
                   __input,
                   __recvd,
                   __h_send_counts,
                   __h_send_displs,
                   __h_recv_counts,
                   __h_recv_displs] :
           ::cuda::std::ranges::views::zip(
             __comms,
             __streams,
             __local_inputs,
             __local_recvd,
             __local_h_send_counts,
             __local_h_send_displs,
             __local_h_recv_counts,
             __local_h_recv_displs))
      {
        __comm.all_to_all_v(
          __guard,
          ::cuda::std::to_address(::cuda::std::ranges::begin(__input)),
          __h_send_counts.data(),
          __h_send_displs.data(),
          __recvd.data(),
          __h_recv_counts.data(),
          __h_recv_displs.data(),
          __stream);
      }
    }

    // --- Phase C: merge the p received sorted runs into the final local output
    // ---
    for (auto&& [__comm, __env, __stream, __recvd, __h_recv_counts, __h_recv_displs, __inputs] :
         ::cuda::std::ranges::views::zip(
           __comms, __envs, __streams, __local_recvd, __local_h_recv_counts, __local_h_recv_displs, __local_inputs))
    {
      // TODO(jfaibussowit):
      //
      // Don't use __tmp and instead write directly to __inputs
      auto __tmp = __buffer<_Tp>{__recvd.__make_empty_like(0)};

      __merge_k_way(__comm, __env, __recvd, __h_recv_counts, __h_recv_displs, __cmp, &__tmp);

      ::cuda::experimental::__detail::__resize_for_overwrite(__inputs, __tmp.size());

      ::cuda::copy_bytes(
        __stream,
        __tmp.__get(),
        ::cuda::std::span<_Tp>{::cuda::std::to_address(::cuda::std::ranges::begin(__inputs)), __tmp.size()},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   __comm.logical_device().underlying_device(),
                                   ::cuda::source_access_order::stream});
    }
  }

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __rebalance_to_original_counts(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const ::std::vector<::cuda::stream_ref>& __streams,
    const ::std::vector<__resource_type>& __resources,
    const ::std::vector<::cuda::std::size_t>& __local_original_sizes,
    const ::std::vector<__buffer<::cuda::std::uint64_t>>& __local_desired_offsets,
    ::cuda::std::uint64_t __N,
    _InputRange&& __local_inputs)
  {
    // The splitter exchange already produced a globally sorted, approximately
    // balanced distribution. Rebalance only corrects the rank ranges from that
    // current distribution back to the original per-rank sizes. Desired offsets
    // are the exclusive scan of the original sizes; the CURRENT offsets are
    // measured here -- the actual post-exchange per-rank sizes, all-gathered
    // and exclusive-scanned -- exactly as the reference does. (Duplicate
    // splitter keys can route an unpredictable share of records to a single
    // rank, so the current distribution must be measured, not predicted from
    // the splitter positions.)
    const auto __comm_size = ::cuda::std::ranges::begin(__comms)->size();

    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts;
    ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs;

    ::std::vector<__buffer<_Tp>> __local_rebalanced;

    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    __local_h_send_counts.reserve(__num_local_inputs);
    __local_h_send_displs.reserve(__num_local_inputs);
    __local_h_recv_counts.reserve(__num_local_inputs);
    __local_h_recv_displs.reserve(__num_local_inputs);
    __local_rebalanced.reserve(__num_local_inputs);

    // ---- Measure the realized post-exchange distribution: all-gather each
    // rank's current __input size, then exclusive-scan into current offsets
    // (length __comm_size, current_offsets[0] == 0).
    ::std::vector<__buffer<::cuda::std::uint64_t>> __local_current_sizes;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __local_current_offsets;
    __local_current_sizes.reserve(__num_local_inputs);
    __local_current_offsets.reserve(__num_local_inputs);

    for (auto&& [__comm, __env, __stream, __resource, __input] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __streams, __resources, __local_inputs))
    {
      const auto __n_current = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));
      auto& __sizes = __local_current_sizes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);
      ::cuda::copy_bytes(
        __stream,
        ::cuda::std::span{&__n_current, ::cuda::std::size_t{1}},
        __sizes.__get().subspan(__comm.rank(), 1),
        ::cuda::copy_configuration{::cuda::host_memory_location,
                                   __comm.logical_device().underlying_device(),
                                   ::cuda::source_access_order::during_api_call});
    }

    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      for (auto&& [__comm, __stream, __sizes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __local_current_sizes))
      {
        auto* const __ptr = __sizes.data();

        __comm.all_gather(__guard, __ptr + __comm.rank(), __ptr, 1, __stream);
      }
    }

    for (auto&& [__comm, __env, __stream, __resource, __sizes] :
         ::cuda::std::ranges::views::zip(__comms, __envs, __streams, __resources, __local_current_sizes))
    {
      auto& __offsets = __local_current_offsets.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum,
        __sizes.begin(),
        __offsets.begin(),
        __comm_size,
        __env);
    }

    for (auto&& [__comm, __env, __stream, __resource, __current_offsets, __desired_offsets, __original_size] :
         ::cuda::std::ranges::views::zip(
           __comms,
           __envs,
           __streams,
           __resources,
           __local_current_offsets,
           __local_desired_offsets,
           __local_original_sizes))
    {
      auto __send_counts = ::cuda::make_buffer<::cuda::std::size_t>(
        __stream,
        __resource,
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
      auto __send_displs = ::cuda::make_buffer<::cuda::std::size_t>(
        __stream,
        __resource,
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
      auto __recv_counts = ::cuda::make_buffer<::cuda::std::size_t>(
        __stream,
        __resource,
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
      auto __recv_displs = ::cuda::make_buffer<::cuda::std::size_t>(
        __stream,
        __resource,
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(__env));

      auto __out = ::cuda::make_zip_iterator(
        __send_counts.begin(), __send_displs.begin(), __recv_counts.begin(), __recv_displs.begin());

      auto __op = [__rank = __comm.rank(),
                   __comm_size,
                   __N,
                   __current_offsets = __current_offsets.data(),
                   __desired_offsets = __desired_offsets.data()] _CCCL_HOST_DEVICE(::cuda::std::uint64_t __peer) {
        // current_offsets[i] is the start of rank i's current
        // (post-exchange) bucket; desired offsets are the original final
        // buckets. Intersect the two global element intervals to derive
        // both send and receive metadata directly.
        const auto __my_src_begin = __current_offsets[__rank];
        const auto __my_src_end   = __rank + 1 == __comm_size ? __N : __current_offsets[__rank + 1];

        const auto __peer_dst_begin = __desired_offsets[__peer];
        const auto __peer_dst_end   = __peer + 1 == __comm_size ? __N : __desired_offsets[__peer + 1];

        const auto __send_begin = ::cuda::std::max(__my_src_begin, __peer_dst_begin);
        const auto __send_end   = ::cuda::std::min(__my_src_end, __peer_dst_end);

        const auto __my_dst_begin = __desired_offsets[__rank];
        const auto __my_dst_end   = __rank + 1 == __comm_size ? __N : __desired_offsets[__rank + 1];

        const auto __peer_src_begin = __current_offsets[__peer];
        const auto __peer_src_end   = __peer + 1 == __comm_size ? __N : __current_offsets[__peer + 1];

        const auto __recv_begin = ::cuda::std::max(__peer_src_begin, __my_dst_begin);
        const auto __recv_end   = ::cuda::std::min(__peer_src_end, __my_dst_end);

        const auto __send_count =
          __send_begin < __send_end
            ? static_cast<::cuda::std::size_t>(__send_end - __send_begin)
            : ::cuda::std::size_t{0};
        const auto __recv_count =
          __recv_begin < __recv_end
            ? static_cast<::cuda::std::size_t>(__recv_end - __recv_begin)
            : ::cuda::std::size_t{0};

        return ::cuda::std::tuple{
          __send_count,
          __send_count == 0 ? ::cuda::std::size_t{0} : static_cast<::cuda::std::size_t>(__send_begin - __my_src_begin),
          __recv_count,
          __recv_count == 0 ? ::cuda::std::size_t{0} : static_cast<::cuda::std::size_t>(__recv_begin - __my_dst_begin)};
      };

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceTransform::Transform,
        ::cuda::counting_iterator<::cuda::std::uint64_t>{},
        ::cuda::std::move(__out),
        __comm_size,
        ::cuda::std::move(__op),
        __env);
      auto& __h_send_counts = __local_h_send_counts.emplace_back(__send_counts.size());
      auto& __h_send_displs = __local_h_send_displs.emplace_back(__send_displs.size());
      auto& __h_recv_counts = __local_h_recv_counts.emplace_back(__recv_counts.size());
      auto& __h_recv_displs = __local_h_recv_displs.emplace_back(__recv_displs.size());

      ::cuda::copy_bytes(
        __stream,
        __send_counts,
        __h_send_counts,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __stream,
        __send_displs,
        __h_send_displs,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __stream,
        __recv_counts,
        __h_recv_counts,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __stream,
        __recv_displs,
        __h_recv_displs,
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __local_rebalanced.emplace_back(__stream, __resource, __original_size, ::cuda::no_init, __env);
    }

    // Sync for HtoD transfers
    for (auto __stream : __streams)
    {
      __stream.sync();
    }

    {
      auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

      // The rebalance exchange is the only communication in this phase. It
      // moves already globally sorted contiguous rank intervals into the exact
      // original per-rank sizes.
      for (auto&& [__comm, __stream, __input, __out, __h_send_counts, __h_send_displs, __h_recv_counts, __h_recv_displs] :
           ::cuda::std::ranges::views::zip(
             __comms,
             __streams,
             __local_inputs,
             __local_rebalanced,
             __local_h_send_counts,
             __local_h_send_displs,
             __local_h_recv_counts,
             __local_h_recv_displs))
      {
        __comm.all_to_all_v(
          __guard,
          ::cuda::std::to_address(::cuda::std::ranges::begin(__input)),
          __h_send_counts.data(),
          __h_send_displs.data(),
          __out.data(),
          __h_recv_counts.data(),
          __h_recv_displs.data(),
          __stream);
      }
    }

    for (auto&& [__comm, __stream, __input, __out] :
         ::cuda::std::ranges::views::zip(__comms, __streams, __local_inputs, __local_rebalanced))
    {
      // This resize is safe only so long as the user promises to free their allocation on the
      // stream that they passed us. For thrust/cuda containers, this is vacuously true
      ::cuda::experimental::__detail::__resize_for_overwrite(__input, __out.size());

      ::cuda::copy_bytes(
        __stream,
        __out.__get(),
        ::cuda::std::span<_Tp>{::cuda::std::to_address(::cuda::std::ranges::begin(__input)), __out.size()},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   __comm.logical_device().underlying_device(),
                                   ::cuda::source_access_order::stream});
    }
  }

  template <class _Policy, class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __execute(
    const __result_policy_base<_Policy>&,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __local_inputs,
    _BinaryOp __cmp)
  {
    static_assert(::cuda::std::ranges::sized_range<_CommRange>);

    // Could use ::cuda::std::invocable here, but it is overkill (compile-time wise). We know
    // that get_stream_t is a normal CPO and normally callable.
    static_assert(::cuda::std::__is_callable_v<::cuda::get_stream_t, ::cuda::std::ranges::range_value_t<_EnvRange>>,
                  "Environment must contain a stream");

    static_assert(::cuda::std::same_as<_Policy, distributed_t>,
                  "Only distributed results are currently supported. Please open an issue at "
                  "github.com/NVIDIA/cccl/issue requesting support for your specified policy.");

    // We take number of comms to be the local input
    const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

    if (!__num_local_inputs)
    {
      // We have no inputs, so... nothing to do
      return;
    }

    const auto __comm_size = static_cast<::cuda::std::size_t>(::cuda::std::ranges::begin(__comms)->size());

    ::std::vector<::cuda::stream_ref> __streams;
    ::std::vector<__resource_type> __resources;
    ::std::vector<__buffer<::cuda::std::uint64_t>> __all_local_offsets;
    ::std::vector<::cuda::std::size_t> __local_original_sizes;

    // TODO (jfaibussowit): maybe can combine some of these
    __streams.reserve(__num_local_inputs);
    __resources.reserve(__num_local_inputs);
    __all_local_offsets.reserve(__num_local_inputs);
    __local_original_sizes.reserve(__num_local_inputs);

    ::cuda::std::uint64_t __N = 0;

    {
      ::std::vector<__buffer<::cuda::std::uint64_t>> __all_local_sizes;

      __all_local_sizes.reserve(__num_local_inputs);

      for (auto&& [__comm, __env, __input] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_inputs))
      {
        const auto __n_local = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));
        const auto __stream  = __streams.emplace_back(::cuda::get_stream(__env));
        auto& __resource     = __resources.emplace_back(
          ::cuda::experimental::__detail::__resource_from_env(__env, __comm.logical_device().underlying_device()));
        auto& __sizes =
          __all_local_sizes.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env).__get();

        __local_original_sizes.push_back(__n_local);
        ::cuda::copy_bytes(
          __stream,
          ::cuda::std::span{&__n_local, ::cuda::std::size_t{1}},
          __sizes.subspan(__comm.rank(), 1),
          ::cuda::copy_configuration{::cuda::host_memory_location,
                                     __comm.logical_device().underlying_device(),
                                     ::cuda::source_access_order::during_api_call});
      }

      {
        auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

        for (auto&& [__comm, __stream, __sizes] :
             ::cuda::std::ranges::views::zip(__comms, __streams, __all_local_sizes))
        {
          auto* const __ptr = __sizes.data();

          __comm.all_gather(__guard, __ptr + __comm.rank(), __ptr, 1, __stream);
        }
      }

      bool __N_computed = false;

      for (auto&& [__comm, __stream, __resource, __env, __input, __sizes] :
           ::cuda::std::ranges::views::zip(__comms, __streams, __resources, __envs, __local_inputs, __all_local_sizes))
      {
        auto& __offsets = __all_local_offsets.emplace_back(__stream, __resource, __comm_size, ::cuda::no_init, __env);

        __CUDAX_MULTI_GPU_DISPATCH(
          __comm.logical_device(),
          CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum,
          __sizes.begin(),
          __offsets.begin(),
          __sizes.size(),
          __env);

        if (!__N_computed)
        {
          ::cuda::std::uint64_t __last_offset = 0;
          ::cuda::std::uint64_t __last_size   = 0;

          // The desired-offset scan already encodes the global extent: N =
          // offset[p - 1] + size[p - 1].
          ::cuda::copy_bytes(
            __stream,
            __offsets.__get().subspan(__comm_size - 1, 1),
            ::cuda::std::span{&__last_offset, ::cuda::std::size_t{1}},
            ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                       ::cuda::host_memory_location,
                                       ::cuda::source_access_order::stream});
          ::cuda::copy_bytes(
            __stream,
            __sizes.__get().subspan(__comm_size - 1, 1),
            ::cuda::std::span{&__last_size, ::cuda::std::size_t{1}},
            ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                       ::cuda::host_memory_location,
                                       ::cuda::source_access_order::stream});

          __stream.sync();
          __N          = __last_offset + __last_size;
          __N_computed = true;
        }

        __CUDAX_MULTI_GPU_DISPATCH(
          __comm.logical_device(),
          CUB_NS_QUALIFIER::DeviceMergeSort::SortKeys,
          ::cuda::std::ranges::begin(__input),
          ::cuda::std::ranges::size(__input),
          __cmp,
          __env);
      }
    }

    if (__comm_size == 1 || __N == 0)
    {
      return;
    }

    {
      ::std::vector<__buffer<_Bracket>> __local_Ls;
      ::std::vector<__buffer<_Bracket>> __local_Us;
      ::std::vector<__buffer<_Tp>> __local_probes;

      __local_Ls.reserve(__num_local_inputs);
      __local_Us.reserve(__num_local_inputs);
      __local_probes.reserve(__num_local_inputs);

      {
        ::std::vector<__buffer<_Tp>> __local_samples;
        ::std::vector<__buffer<::cuda::std::size_t>> __local_samples_sizes;
        ::std::vector<__buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>> __local_I_js;
        ::std::vector<__buffer<::cuda::std::uint64_t>> __local_hist;
        ::std::vector<::cuda::std::size_t> __local_sample_sendcounts(__num_local_inputs);

        __local_samples.reserve(__num_local_inputs);
        __local_samples_sizes.reserve(__num_local_inputs);
        __local_I_js.reserve(__num_local_inputs);
        __local_hist.reserve(__num_local_inputs);

        // Allocated once and reused across sampling rounds. It is fully overwritten each round
        // (root writes it before the broadcast), so storage-only reuse is safe. Pinned
        // allocation is expensive and serializes the stream, so we avoid re-allocating it K
        // times.
        ::std::vector<::cuda::buffer<::cuda::std::uint64_t, ::cuda::mr::device_accessible, ::cuda::mr::host_accessible>>
          __local_probe_counts;

        __local_probe_counts.reserve(__num_local_inputs);

        // Root-only scratch for __gather_merge_broadcast, hoisted out of the sampling loop so their
        // allocations are paid once per sort instead of once per round. recvcounts/displs are
        // O(comm_size) host vectors of invariant size; all_samples is the device buffer holding the
        // gathered sample keys, which shrinks monotonically across rounds. __gather_merge_broadcast
        // reuses their storage each round (see the reuse handling there).
        ::std::vector<::cuda::std::size_t> __root_recvcounts;
        ::std::vector<::cuda::std::size_t> __root_displs;
        ::cuda::std::optional<__buffer<_Tp>> __root_all_samples;

        for (auto&& [__comm, __stream, __resource, __env] :
             ::cuda::std::ranges::views::zip(__comms, __streams, __resources, __envs))
        {
          const auto __n_split = __comm_size - 1;
          // TODO(jfaibussowit): these can all be done in parallel on separate
          // utility streams.
          __local_Ls.emplace_back(__stream, __resource, __n_split, _Bracket{0, ::cuda::std::nullopt}, __env);
          __local_Us.emplace_back(__stream, __resource, __n_split, _Bracket{__N, ::cuda::std::nullopt}, __env);
          __local_I_js.emplace_back(
            __stream,
            __resource,
            __n_split,
            ::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>{},
            __env);
          __local_probes.emplace_back(__stream, __resource, __env);
          __local_samples.emplace_back(__stream, __resource, __env);
          // differing sizes for NCCL inplace
          __local_samples_sizes.emplace_back(
            __stream,
            __resource,
            /*__size=*/__comm.rank() == __ROOT_RANK ? __comm_size : 1,
            ::cuda::no_init,
            __env);
          __local_hist.emplace_back(__stream, __resource, __env);
#if _CCCL_CTK_AT_LEAST(12, 9)
          __local_probe_counts.emplace_back(::cuda::make_pinned_buffer<::cuda::std::uint64_t>(
            __streams.front(), /*__size=*/::cuda::std::size_t{1}, ::cuda::no_init));
#else // ^^^ CUDA 12.9+ ^^^ / vvv CUDA 12.8- vvv
          __local_probe_counts.emplace_back(::cuda::make_buffer<::cuda::std::uint64_t>(
            __streams.front(),
            ::cuda::mr::legacy_pinned_memory_resource{},
            /*__size=*/::cuda::std::size_t{1},
            ::cuda::no_init));
#endif // ^^^ CUDA 12.8- ^^^
        }

        // Note: K is small, on the order of ~1-10
        const auto __K = ::cuda::std::max(
          static_cast<::cuda::std::int32_t>(
            ::cuda::std::ceil(::cuda::std::log10(::cuda::std::log10(__comm_size) / __EPS))),
          1);
        const auto __s_j_interior = 2. * ::cuda::std::log(__comm_size) / __EPS;

        for (::cuda::std::int32_t __j = 1; __j <= __K; ++__j)
        {
          const auto __s_j  = ::cuda::std::pow(__s_j_interior, static_cast<double>(__j) / static_cast<double>(__K));
          const auto __prob = ::cuda::std::min(__s_j * static_cast<double>(__comm_size) / static_cast<double>(__N), 1.);

          for (auto&& [__stream, __input, __I_j, __samples, __sample_size, __n_local, __previous_sendcount] :
               ::cuda::std::ranges::views::zip(
                 __streams,
                 __local_inputs,
                 __local_I_js,
                 __local_samples,
                 __local_samples_sizes,
                 __local_original_sizes,
                 __local_sample_sendcounts))
          {
            // Each iteration we sample the union of splitter intervals, \gamma_j with a
            // probability of __prob. For the first iteration, \gamma_j is the entire array, but for
            // previous iterations it's impossible for us to tell (on the host), because:
            //
            // 1. We can't inspect the updated intervals __I_j, and
            // 2. We can't count how many of our elements actually lie within those updated
            //    intervals.
            //
            // So instead we use the fact that each round the number of samples must decrease,
            // because I_j gets tightened and we sample an increasingly smaller region. Therefore,
            // the high-water mark for the samples is the previous round's sample vector size.
            const auto __estimate = ::cuda::std::max(
              __j == 1 ? static_cast<::cuda::std::size_t>(::cuda::std::ceil(__n_local * __prob)) : __previous_sendcount,
              ::cuda::std::size_t{1});

            ::cuda::experimental::__detail::__resize_for_overwrite(__samples, __estimate);
            ::cuda::experimental::__detail::__sort_sample_probes(
              __stream, __input, __I_j, __prob, __cmp, &__samples, &__sample_size);
          }

          __gather_merge_broadcast(
            __comms,
            __envs,
            __streams,
            __local_samples,
            __cmp,
            &__local_samples_sizes,
            &__local_sample_sendcounts,
            &__local_probes,
            &__local_probe_counts,
            &__root_recvcounts,
            &__root_displs,
            &__root_all_samples);

          __local_histogram(__comms, __envs, __local_inputs, __local_probes, __cmp, &__local_hist);

          {
            auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

            for (auto&& [__comm, __stream, __hist] : ::cuda::std::ranges::views::zip(__comms, __streams, __local_hist))
            {
              auto* const __ptr = __hist.data();

              __comm.all_reduce(__guard, __ptr, __ptr, __hist.size(), ::cuda::std::plus<>{}, __stream);
            }
          }

          // ---- STEP 3: tighten brackets and rebuild intervals ----
          __update_intervals(
            __comms, __envs, __local_probes, __local_hist, __N, &__local_Ls, &__local_Us, &__local_I_js);
        }
      }

      __data_exchange(
        __comms, __envs, __streams, __resources, __local_Ls, __local_Us, __local_probes, __N, __cmp, __local_inputs);
    }

    __rebalance_to_original_counts(
      __comms, __envs, __streams, __resources, __local_original_sizes, __all_local_offsets, __N, __local_inputs);
  }
};
} // namespace __detail

_CCCL_TEMPLATE(
  class _Policy, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp = ::cuda::std::less<>)
_CCCL_REQUIRES(__range_of_communicators<_CommRange> _CCCL_AND ::cuda::std::ranges::forward_range<_EnvRange> _CCCL_AND
                 __detail::__range_of_sized_random_access_ranges<_InputRange>)
void sort(const __result_policy_base<_Policy>& __policy,
          _CommRange&& __comms,
          _EnvRange&& __envs,
          _InputRange&& __range_of_input_ranges,
          _BinaryOp __cmp = {})
{
  using __result_type =
    ::cuda::std::ranges::range_value_t<::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_InputRange>>>;
  using __env_type = ::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_EnvRange>>;

  _CCCL_NVTX_RANGE_SCOPE("cuda::experimental::sort");

  __detail::_Sorter<__result_type, __env_type, _BinaryOp>{}.__execute(
    __policy,
    ::cuda::std::forward<_CommRange>(__comms),
    ::cuda::std::forward<_EnvRange>(__envs),
    ::cuda::std::forward<_InputRange>(__range_of_input_ranges),
    ::cuda::std::move(__cmp));
}
} // namespace cuda::experimental

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_SORT_H
