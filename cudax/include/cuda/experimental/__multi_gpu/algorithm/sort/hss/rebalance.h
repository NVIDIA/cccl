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

#include <cub/device/device_transform.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__iterator/counting_iterator.h>
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
//! sizes. It needs each rank's current global offset, which the data-exchange phase derived from
//! the all-reduced probe histogram and hands in, so no collective measures the distribution here.
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
//! @param[in] __local_current_offsets The per-rank starts of the post-exchange rank intervals,
//!            derived by the data-exchange phase.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__rebalance_to_original_counts(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const __data_exchange_result_type& __exchange_results)
{
  // At this point we have sorted a huge pile of numbers across several GPUs. Near the end,
  // every GPU holds a correctly-sorted *chunk*, but the chunks are the wrong sizes. GPU 0
  // might be holding 1.2 million items when it's supposed to end up with exactly 1
  // million. So there's a final shuffle to hand the excess around until everyone's chunk is
  // the size it started as.
  //
  // To run that shuffle, each GPU needs to know where its chunk currently begins in the global
  // order - "I'm currently holding items 0 through 1,199,999." Equivalently: how much
  // *everyone else* is holding, so it can work out its own starting position.
  //
  // The naive solution is to just ask via an all-gather. Every GPU announces its current size
  // to every other GPU, then they add them up. It works, but it's a synchronization point -
  // all GPUs have to stop and wait for each other.
  //
  // But we don't actually need this because we can deduce everyone else's sizes based on what
  // we have already:
  //
  // Earlier in the algorithm the GPUs agreed on a set of splitters - dividing lines in the
  // sorted order. "Everything below 500 goes to GPU 0, 500-999 to GPU 1," etc. That agreement
  // is what drove the shuffle in the first place.
  //
  // It's tempting to think you'd need to track the actual traffic - who sent how many items to
  // whom - but this isn't necessary. We already have this information from... the splitters!
  // GPU N ends up holding *exactly the items that fall between splitter N-1 and splitter N*,
  // which is encoded in the histogram.
  //
  // Even better, the histogram is broadcasted equally to all ranks during the histogramming
  // phase so we know that everyone has the same information.
  const auto __comm_size        = __setup.__comm_size;
  const auto __N                = __setup.__N;
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  ::std::vector<__buffer_type<_Tp>> __local_rebalanced;

  constexpr ::cuda::std::size_t __send_counts_column = 0;
  constexpr ::cuda::std::size_t __send_displs_column = 1;
  constexpr ::cuda::std::size_t __recv_counts_column = 2;
  constexpr ::cuda::std::size_t __recv_displs_column = 3;
  constexpr ::cuda::std::size_t __num_columns        = 4;

  static_assert(
    __send_counts_column == 0 && __send_displs_column == 1 && __recv_counts_column == 2 && __recv_displs_column == 3,
    "The fused D2H copy requires the device and host columns to be contiguous and in "
    "the same order on both sides");

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

  {
    auto __idx = 0;

    for (auto&& [__comm, __env, __resource, __current_offsets, __desired_offsets, __original_size] :
         ::cuda::std::ranges::views::zip(
           __comms,
           __envs,
           __setup.__resources,
           __exchange_results.__local_current_offsets,
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

      ::cuda::copy_bytes(
        __counts.stream(),
        __counts,
        ::cuda::std::span<::cuda::std::size_t>{__h_column(__local_h_counts, __idx, __send_counts_column).data(),
                                               __num_columns * static_cast<::cuda::std::size_t>(__comm_size)},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __local_rebalanced.emplace_back(__counts.stream(), __resource, __original_size, ::cuda::no_init, __env);

      ++__idx;
    }
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    auto __idx     = 0;

    for (auto&& [__comm, __exchanged, __out] :
         ::cuda::std::ranges::views::zip(__comms, __exchange_results.__local_merged, __local_rebalanced))
    {
      // Wait for DtoH above to finish
      __out.__get().stream().sync();

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

  for (auto&& [__comm, __input, __out] : ::cuda::std::ranges::views::zip(__comms, __local_inputs, __local_rebalanced))
  {
    const ::cuda::device_ref __device = __comm.logical_device().underlying_device();
    // We cannot assume the user will set their device properly if they need to reallocate/move
    // elements in the resize, so we do this for them
    const auto _ = ::cuda::__ensure_current_context{__device};

    // This resize is safe only so long as the user promises to free their allocation on the
    // stream that they passed us. For thrust/cuda containers, this is vacuously true.
    ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(__input, __out.size());

    ::cuda::copy_bytes(
      __out.__get().stream(),
      __out.__get(),
      ::cuda::std::span<_Tp>{::cuda::std::to_address(::cuda::std::ranges::begin(__input)), __out.size()},
      ::cuda::copy_configuration{__device, __device, ::cuda::source_access_order::stream});
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_REBALANCE_H
