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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_REBALANCE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_REBALANCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_scan.cuh>
#include <cub/device/device_transform.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Derive per-peer send/recv counts and displacements for the rebalance exchange.
//!
//! Invoked once per peer via `DeviceTransform`. `__current_offsets[i]` is the start of rank
//! `i`'s current (post-exchange) bucket and `__desired_offsets[i]` the start of its original
//! final bucket. Intersecting this rank's current global interval with the peer's desired
//! interval yields the send metadata; intersecting the peer's current interval with this rank's
//! desired interval yields the receive metadata.
struct __rebalance_counts_fn
{
  ::cuda::std::uint64_t __rank;
  ::cuda::std::uint64_t __comm_size;
  ::cuda::std::uint64_t __N;
  const ::cuda::std::uint64_t* __current_offsets;
  const ::cuda::std::uint64_t* __desired_offsets;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::
    tuple<::cuda::std::size_t, ::cuda::std::size_t, ::cuda::std::size_t, ::cuda::std::size_t>
    operator()(::cuda::std::uint64_t __peer) const noexcept
  {
    // current_offsets[i] is the start of rank i's current (post-exchange) bucket; desired
    // offsets are the original final buckets. Intersect the two global element intervals to
    // derive both send and receive metadata directly.
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
      __send_begin < __send_end ? static_cast<::cuda::std::size_t>(__send_end - __send_begin) : ::cuda::std::size_t{0};
    const auto __recv_count =
      __recv_begin < __recv_end ? static_cast<::cuda::std::size_t>(__recv_end - __recv_begin) : ::cuda::std::size_t{0};

    return ::cuda::std::tuple{
      __send_count,
      __send_count == 0 ? ::cuda::std::size_t{0} : static_cast<::cuda::std::size_t>(__send_begin - __my_src_begin),
      __recv_count,
      __recv_count == 0 ? ::cuda::std::size_t{0} : static_cast<::cuda::std::size_t>(__recv_begin - __my_dst_begin)};
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Redistribute the globally sorted keys back to each rank's original per-rank size.
//!
//! Final HSS phase. The data-exchange phase leaves a globally sorted but only approximately
//! balanced distribution; this phase corrects the rank ranges back to the exact original per-rank
//! sizes. Because duplicate splitter keys can route an unpredictable share of records to one
//! rank, the current distribution is measured rather than predicted: each rank's current input
//! size is all-gathered and exclusive-scanned into current offsets.
//!
//! The desired offsets are the setup's exclusive scan of the original sizes. One CUB
//! `DeviceTransform` intersects each rank's current global interval with each peer's desired
//! interval to derive the send/recv counts and displacements directly.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value and buffer types.
//!
//! @param[in] __setup The local-setup result supplying resources, desired offsets, original
//!            sizes, `N`, and comm size.
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[out] __local_inputs The range of per-rank local key ranges, rewritten at their
//!             original per-rank sizes. This is the only point in the algorithm at which the
//!             caller's ranges are resized, and it happens with no collective in flight.
//! @param[in] __local_exchanged The per-rank output of the data-exchange phase, used as the send
//!            buffer of the rebalance exchange.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__rebalance_to_original_counts(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const ::std::vector<__buffer_type<_Tp>>& __local_exchanged)
{
  const auto __comm_size        = __setup.__comm_size;
  const auto __N                = __setup.__N;
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  // The splitter exchange already produced a globally sorted, approximately balanced
  // distribution. Rebalance only corrects the rank ranges from that current distribution back
  // to the original per-rank sizes. Desired offsets are the exclusive scan of the original
  // sizes; the CURRENT offsets are measured here -- the actual post-exchange per-rank sizes,
  // all-gathered and exclusive-scanned -- exactly as the reference does. (Duplicate splitter
  // keys can route an unpredictable share of records to a single rank, so the current
  // distribution must be measured, not predicted from the splitter positions.)
  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __local_current_sizes;

  __local_current_sizes.reserve(__num_local_inputs);
  for (auto&& [__comm, __env, __resource, __exchanged] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources, __local_exchanged))
  {
    const auto __n_current = static_cast<::cuda::std::uint64_t>(__exchanged.size());
    auto& __sizes =
      __local_current_sizes.emplace_back(::cuda::get_stream(__env), __resource, __comm_size, ::cuda::no_init, __env);

    ::cuda::copy_bytes(
      __sizes.__get().stream(),
      ::cuda::std::span{&__n_current, ::cuda::std::size_t{1}},
      __sizes.__get().subspan(__comm.rank(), 1),
      ::cuda::copy_configuration{::cuda::host_memory_location,
                                 __comm.logical_device().underlying_device(),
                                 ::cuda::source_access_order::during_api_call});
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __sizes] : ::cuda::std::ranges::views::zip(__comms, __local_current_sizes))
    {
      auto* const __ptr = __sizes.data();

      __comm.all_gather(__guard, __ptr + __comm.rank(), __ptr, 1, __sizes.__get().stream());
    }
  }

  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __local_current_offsets;

  __local_current_offsets.reserve(__num_local_inputs);
  for (auto&& [__comm, __env, __resource, __sizes] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources, __local_current_sizes))
  {
    auto& __offsets =
      __local_current_offsets.emplace_back(__sizes.__get().stream(), __resource, __comm_size, ::cuda::no_init, __env);

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum,
      __sizes.begin(),
      __offsets.begin(),
      __comm_size,
      __env);
  }

  // The four count/displacement columns are all `__comm_size` elements of the same type, live on
  // the same device, and are used on the same stream, so each rank gets one flat device
  // allocation holding them back to back instead of four. `__column()` demarcates column `__col`
  // within a rank's device block; the same layout is mirrored on the host below.
  ::std::vector<__buffer_type<_Tp>> __local_rebalanced;

  // The column order is dictated by the tuple `__rebalance_counts_fn` returns: the
  // `DeviceTransform` writes its four outputs into these columns in this order, and the host
  // columns repeat it so the whole block transfers in one copy.
  constexpr ::cuda::std::size_t __send_counts_column = 0;
  constexpr ::cuda::std::size_t __send_displs_column = 1;
  constexpr ::cuda::std::size_t __recv_counts_column = 2;
  constexpr ::cuda::std::size_t __recv_displs_column = 3;
  constexpr ::cuda::std::size_t __num_columns        = 4;

  static_assert(
    __send_counts_column == 0 && __send_displs_column == 1 && __recv_counts_column == 2 && __recv_displs_column == 3,
    "The fused D2H copy requires the device and host columns to be contiguous and in "
    "the same order on both sides");

  // Every rank's host columns have the same lifetime too, so all ranks share one flat allocation
  // instead of one per rank. The layout is rank-major: rank `i`'s four columns occupy
  // `[i * __num_columns * __comm_size, (i + 1) * __num_columns * __comm_size)`, and `__h_column()`
  // demarcates column `__col` within rank `__rank_idx`'s block. Sized up front so that every
  // subspan below is valid immediately.
  ::std::vector<::cuda::std::size_t> __local_h_counts(__num_local_inputs * __num_columns * __comm_size);

  const auto __column = [__comm_size](auto& __counts, ::cuda::std::size_t __col) {
    return __counts.subspan(__col * __comm_size, __comm_size);
  };
  const auto __h_column =
    [__comm_size](
      ::std::vector<::cuda::std::size_t>& __h_counts, ::cuda::std::size_t __rank_idx, ::cuda::std::size_t __col) {
      return ::cuda::std::span<::cuda::std::size_t>{__h_counts}.subspan(
        ((__rank_idx * __num_columns) + __col) * __comm_size, __comm_size);
    };

  __local_rebalanced.reserve(__num_local_inputs);

  ::cuda::std::size_t __idx = 0;
  for (auto&& [__comm, __env, __resource, __current_offsets, __desired_offsets, __original_size] :
       ::cuda::std::ranges::views::zip(
         __comms,
         __envs,
         __setup.__resources,
         __local_current_offsets,
         __setup.__all_local_offsets,
         __setup.__local_original_sizes))
  {
    auto __counts = ::cuda::make_buffer<::cuda::std::size_t>(
      __current_offsets.__get().stream(),
      __resource,
      __num_columns * __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));

    auto __out = ::cuda::std::make_tuple(
      __column(__counts, __send_counts_column).data(),
      __column(__counts, __send_displs_column).data(),
      __column(__counts, __recv_counts_column).data(),
      __column(__counts, __recv_displs_column).data());

    auto __op = __rebalance_counts_fn{
      static_cast<::cuda::std::uint64_t>(__comm.rank()),
      static_cast<::cuda::std::uint64_t>(__comm_size),
      __N,
      __current_offsets.data(),
      __desired_offsets.data()};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      ::cuda::std::move(__out),
      __comm_size,
      __op,
      __env);

    // All four columns come back in one transfer: the device allocation holds them back to back
    // and this rank's four host columns are contiguous within the rank-major flat allocation, so
    // either side is a single contiguous `__num_columns * __comm_size` range starting at the
    // rank's first column.
    ::cuda::copy_bytes(
      __counts.stream(),
      __counts,
      ::cuda::std::span<::cuda::std::size_t>{__h_column(__local_h_counts, __idx, __send_counts_column).data(),
                                             __num_columns * static_cast<::cuda::std::size_t>(__comm_size)},
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});

    __local_rebalanced.emplace_back(__counts.stream(), __resource, __original_size, ::cuda::no_init, __env);

    ++__idx;
  }

  // The transfers above are device to host, and the all_to_all_v below reads the host columns as
  // its count and displacement arguments. Drain every stream first so those columns are populated
  // before any rank enters the collective.
  for (auto&& __local : __local_rebalanced)
  {
    __local.__get().stream().sync();
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    // The rebalance exchange is the only communication in this phase. It moves already
    // globally sorted contiguous rank intervals into the exact original per-rank sizes.
    __idx = 0;
    for (auto&& [__comm, __exchanged, __out] :
         ::cuda::std::ranges::views::zip(__comms, __local_exchanged, __local_rebalanced))
    {
      __comm.all_to_all_v(
        __guard,
        __exchanged.data(),
        __h_column(__local_h_counts, __idx, __send_counts_column).data(),
        __h_column(__local_h_counts, __idx, __send_displs_column).data(),
        __out.data(),
        __h_column(__local_h_counts, __idx, __recv_counts_column).data(),
        __h_column(__local_h_counts, __idx, __recv_displs_column).data(),
        __out.__get().stream());

      ++__idx;
    }
  }

  // Drain every rank's stream before touching the caller's ranges. The rebalance exchange above
  // is only submitted by closing the group guard, and resizing a caller range may reallocate
  // through an allocator that is neither stream-ordered nor able to make progress while a
  // collective is pending. Syncing first means the resize below runs with nothing in flight, so
  // it can neither alias the exchange's buffers nor block a peer that is waiting to join a
  // collective this rank has not yet reached.
  for (auto&& __out : __local_rebalanced)
  {
    __out.__get().stream().sync();
  }

  for (auto&& [__comm, __input, __out] : ::cuda::std::ranges::views::zip(__comms, __local_inputs, __local_rebalanced))
  {
    // This resize is safe only so long as the user promises to free their allocation on the
    // stream that they passed us. For thrust/cuda containers, this is vacuously true.
    ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(__input, __out.size());

    ::cuda::copy_bytes(
      __out.__get().stream(),
      __out.__get(),
      ::cuda::std::span<_Tp>{::cuda::std::to_address(::cuda::std::ranges::begin(__input)), __out.size()},
      ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                 __comm.logical_device().underlying_device(),
                                 ::cuda::source_access_order::stream});
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_REBALANCE_H
