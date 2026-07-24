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
#include <cuda/__iterator/zip_iterator.h>
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

  [[nodiscard]] _CCCL_DEVICE constexpr ::cuda::std::
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
//! @param[in,out] __local_inputs The range of per-rank local key ranges, rewritten at their
//!                original per-rank sizes.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__rebalance_to_original_counts(
  const __local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __local_inputs)
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
  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __local_current_offsets;

  __local_current_sizes.reserve(__num_local_inputs);
  __local_current_offsets.reserve(__num_local_inputs);

  for (auto&& [__comm, __env, __resource, __input] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources, __local_inputs))
  {
    const auto __n_current = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));
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

  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs;
  ::std::vector<__buffer_type<_Tp>> __local_rebalanced;

  __local_h_send_counts.reserve(__num_local_inputs);
  __local_h_send_displs.reserve(__num_local_inputs);
  __local_h_recv_counts.reserve(__num_local_inputs);
  __local_h_recv_displs.reserve(__num_local_inputs);
  __local_rebalanced.reserve(__num_local_inputs);
  for (auto&& [__comm, __env, __resource, __current_offsets, __desired_offsets, __original_size] :
       ::cuda::std::ranges::views::zip(
         __comms,
         __envs,
         __setup.__resources,
         __local_current_offsets,
         __setup.__all_local_offsets,
         __setup.__local_original_sizes))
  {
    auto __send_counts = ::cuda::make_buffer<::cuda::std::size_t>(
      __current_offsets.__get().stream(),
      __resource,
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
    auto __send_displs = ::cuda::make_buffer<::cuda::std::size_t>(
      __current_offsets.__get().stream(),
      __resource,
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
    auto __recv_counts = ::cuda::make_buffer<::cuda::std::size_t>(
      __current_offsets.__get().stream(),
      __resource,
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));
    auto __recv_displs = ::cuda::make_buffer<::cuda::std::size_t>(
      __current_offsets.__get().stream(),
      __resource,
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));

    auto __out = ::cuda::make_zip_iterator(
      __send_counts.begin(), __send_displs.begin(), __recv_counts.begin(), __recv_displs.begin());

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

    auto& __h_send_counts = __local_h_send_counts.emplace_back(__send_counts.size());
    auto& __h_send_displs = __local_h_send_displs.emplace_back(__send_displs.size());
    auto& __h_recv_counts = __local_h_recv_counts.emplace_back(__recv_counts.size());
    auto& __h_recv_displs = __local_h_recv_displs.emplace_back(__recv_displs.size());

    ::cuda::copy_bytes(
      __send_counts.stream(),
      __send_counts,
      __h_send_counts,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});
    ::cuda::copy_bytes(
      __send_displs.stream(),
      __send_displs,
      __h_send_displs,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});
    ::cuda::copy_bytes(
      __recv_counts.stream(),
      __recv_counts,
      __h_recv_counts,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});
    ::cuda::copy_bytes(
      __recv_displs.stream(),
      __recv_displs,
      __h_recv_displs,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});

    __local_rebalanced.emplace_back(__recv_displs.stream(), __resource, __original_size, ::cuda::no_init, __env);
  }

  // Sync for HtoD transfers
  for (auto&& __local : __local_rebalanced)
  {
    __local.__get().stream().sync();
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    // The rebalance exchange is the only communication in this phase. It moves already
    // globally sorted contiguous rank intervals into the exact original per-rank sizes.
    for (auto&& [__comm, __input, __out, __h_send_counts, __h_send_displs, __h_recv_counts, __h_recv_displs] :
         ::cuda::std::ranges::views::zip(
           __comms,
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
        __out.__get().stream());
    }
  }

  for (auto&& [__comm, __input, __out] : ::cuda::std::ranges::views::zip(__comms, __local_inputs, __local_rebalanced))
  {
    // This resize is safe only so long as the user promises to free their allocation on the
    // stream that they passed us. For thrust/cuda containers, this is vacuously true
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
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_REBALANCE_H
