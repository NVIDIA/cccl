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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H

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
#include <cuda/__iterator/zip_transform_iterator.h>
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/bucket_count_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/ideal_rank_fn.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/merge_k_way.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Splitter-selection functor that realizes the final splitter key from an `[L, U]`
//!        bracket.
//!
//! Given a splitter's target rank `Ni/p` and its `L` / `U` brackets, it returns whichever
//! bracket endpoint key is closest to the target rank, realizing an unset (unbounded) endpoint
//! from the probe extrema (`__first_probe` / `__last_probe`).
//!
//! This is HSS Section 4.2.2 step (5): "Once the histogramming phase finishes, the key ranked
//! closest to `Ni/p` among the keys seen so far is set as the `i`-th splitter." The `(L, U)`
//! bracket is Table 1's `L(i)` / `U(i)` (ranks of the largest sample key below / smallest
//! sample key above `Ni/p`), whose realized keys delimit the splitter interval `I(i)`.
template <class _Tp>
struct __finalize_splitters_fn
{
  const _Tp* __first_probe;
  const _Tp* __last_probe;

  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp
  operator()(::cuda::std::uint64_t __target_rank, _Bracket<_Tp> __L_i, _Bracket<_Tp> __U_i) const noexcept
  {
    const bool __use_L = (__target_rank - __L_i.__rank) <= (__U_i.__rank - __target_rank);

    // Note that L_i and U_i might not have values if we never found any global splitters among
    // our values. In this case the "closest" is simply our extrema.
    if (__use_L)
    {
      // Lower bound is closer to target
      return __L_i.__key.has_value() ? *__L_i.__key : *__first_probe;
    }
    // Upper bound is closer to target
    return __U_i.__key.has_value() ? *__U_i.__key : *__last_probe;
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Route every rank's local keys to their destination ranks and merge the received runs.
//!
//! The HSS Data Exchange phase (Section 3.1 step (3), reused unchanged per Section 3.3): "a key
//! in range `[S(i), S(i + 1))` goes to processor `i`". For each communicator it counts, per
//! destination bucket, how many local keys fall between consecutive finalized splitters. The
//! finalized splitters are reconstructed lazily on the fly by fusing `__finalize_splitters_fn`
//! into a `transform_iterator` (`__splitter_it`) fed to a `__bucket_count_fn`, so one CUB
//! `DeviceTransform` produces the send counts without a separate splitter buffer or launch.
//!
//! `__local_inputs` must be locally sorted and `__local_splitters` carry finalized brackets
//! from `__histogramming_phase`.
//!
//! @tparam _Traits The `__hss_traits` instantiation carrying the value and buffer types.
//!
//! @param[in] __setup The local-setup result supplying resources, comm size, and `N`.
//! @param[in] __comms The range of per-rank communicators.
//! @param[in] __envs The range of per-rank execution environments (one stream each).
//! @param[in] __local_inputs The range of per-rank local key ranges, read as the send buffer of
//!            the exchange and left unmodified.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[in] __local_splitters The per-comm splitter state supplying the finalized brackets and
//!            probes.
//!
//! @returns The per-rank exchanged-and-merged keys, one buffer per communicator.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::template __buffer_type<_Tp>>
_HSSSorter<_Tp, _Env, _BinaryOp>::__data_exchange(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const _BinaryOp& __cmp,
  const ::std::vector<__per_comm_splitters_type>& __local_splitters)
{
  const auto __comm_size = __setup.__comm_size;
  const auto __N         = __setup.__N;

  // The send and recv counts are the same size, live on the same device, and are used on the
  // same stream, so they share one allocation per rank instead of two: the send counts occupy
  // `[0, __comm_size)` and the recv counts `[__comm_size, 2 * __comm_size)`. `__send_span()` /
  // `__recv_span()` below demarcate the halves.
  ::std::vector<__buffer_type<::cuda::std::size_t>> __local_counts;

  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  __local_counts.reserve(__num_local_inputs);

  const auto __send_span = [__comm_size](auto& __counts) {
    return __counts.__get().subspan(0, __comm_size);
  };
  const auto __recv_span = [__comm_size](auto& __counts) {
    return __counts.__get().subspan(__comm_size, __comm_size);
  };

  for (auto&& [__comm, __env, __resource, __input, __splitters] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources, __local_inputs, __local_splitters))
  {
    const auto& __Ls     = __splitters.__Ls;
    const auto& __Us     = __splitters.__Us;
    const auto& __probes = __splitters.__probes;

    auto& __counts =
      __local_counts.emplace_back(__Ls.__get().stream(), __resource, 2 * __comm_size, ::cuda::no_init, __env);

    const auto __input_begin = ::cuda::std::to_address(::cuda::std::ranges::begin(__input));

    // Lazily reconstruct the finalized splitters (HSS Section 4.2.2 step (5), "the key
    // ranked closest to Ni/p ... is set as the ith splitter") on the fly instead of
    // materializing them. Rather than a separate Transform launch writing a splitter buffer
    // that this kernel then reads, we fuse __finalize_splitters_fn into the data-exchange
    // kernel through a transform_iterator: each __splitter_it[d] evaluates finalize on
    // demand, eliminating one Transform launch and the splitter buffer. The ideal rank Ni/p
    // (the center of the Section 2 / Table 1 target range Ti) is supplied per-splitter by
    // __ideal_rank_fn.
    _CCCL_VERIFY(!__probes.__get().empty(), "Histogramming phase should have generated at least one probe");

    auto __splitter_it = ::cuda::make_zip_transform_iterator(
      __finalize_splitters_fn<_Tp>{__probes.data(), ::cuda::std::to_address(__probes.end() - 1)},
      ::cuda::make_transform_iterator(
        ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
      __Ls.data(),
      __Us.data());

    // Route this rank's local keys to destination ranks via the splitter keys: the Data
    // Exchange phase, HSS Section 3.1 step (3), "a key in range [S(i), S(i + 1)) goes to
    // processor i". HSS reuses this phase unchanged (Section 3.3), so bucket d receives the keys
    // in [S(d - 1), S(d)) and its count becomes the send metadata. The send displacements are the
    // exclusive prefix-sum of these counts (buckets are contiguous and non-overlapping), so we
    // recompute them on the host below instead of emitting a second device column here.
    auto __op = __bucket_count_fn<::cuda::std::remove_cvref_t<decltype(__input_begin)>,
                                  ::cuda::std::remove_cvref_t<decltype(__splitter_it)>,
                                  _BinaryOp>{
      __input_begin,
      ::cuda::std::to_address(::cuda::std::ranges::end(__input)),
      ::cuda::std::move(__splitter_it),
      __Ls.size(),
      __cmp};

    const auto __send_counts = __send_span(__counts);

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      __send_counts.data(),
      __send_counts.size(),
      ::cuda::std::move(__op),
      __env);
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __counts] : ::cuda::std::ranges::views::zip(__comms, __local_counts))
    {
      auto* const __send_ptr = __send_span(__counts).data();
      auto* const __recv_ptr = __recv_span(__counts).data();

      __comm.all_to_all(__guard, __send_ptr, __recv_ptr, /*__count=*/1, __counts.__get().stream());
    }
  }

  ::std::vector<__buffer_type<_Tp>> __local_recvd;

  // The count columns are adjacent and in the same order as the halves of the device-side
  // allocation (send then recv), so the two count columns together form one contiguous
  // destination that mirrors the device buffer exactly. The single D2H copy below relies on
  // that; the displacement columns follow and are filled on the host.
  constexpr ::cuda::std::size_t __h_send_counts_column = 0;
  constexpr ::cuda::std::size_t __h_recv_counts_column = 1;
  constexpr ::cuda::std::size_t __h_send_displs_column = 2;
  constexpr ::cuda::std::size_t __h_recv_displs_column = 3;
  constexpr ::cuda::std::size_t __h_num_columns        = 4;

  static_assert(__h_recv_counts_column == __h_send_counts_column + 1,
                "The fused counts copy requires the send and recv count columns to be adjacent");

  // Every rank's four columns have the same lifetime as well, so all ranks share one flat
  // allocation instead of one per rank. The layout is rank-major: rank `i`'s four columns occupy
  // `[i * __h_num_columns * __comm_size, (i + 1) * __h_num_columns * __comm_size)`, and
  // `__h_column()` demarcates column `__col` within rank `__rank_idx`'s block. Sized up front so
  // that every subspan below is valid immediately.
  ::std::vector<::cuda::std::size_t> __local_h_counts(__num_local_inputs * __h_num_columns * __comm_size);

  const auto __h_column =
    [__comm_size](
      ::std::vector<::cuda::std::size_t>& __h_counts, ::cuda::std::size_t __rank_idx, ::cuda::std::size_t __col) {
      return ::cuda::std::span<::cuda::std::size_t>{__h_counts}.subspan(
        ((__rank_idx * __h_num_columns) + __col) * __comm_size, __comm_size);
    };

  __local_recvd.reserve(__num_local_inputs);

  ::cuda::std::size_t __idx = 0;
  for (auto&& [__comm, __resource, __env, __counts] :
       ::cuda::std::ranges::views::zip(__comms, __setup.__resources, __envs, __local_counts))
  {
    const auto __h_send_counts = __h_column(__local_h_counts, __idx, __h_send_counts_column);
    const auto __h_recv_counts = __h_column(__local_h_counts, __idx, __h_recv_counts_column);
    const auto __h_send_displs = __h_column(__local_h_counts, __idx, __h_send_displs_column);
    const auto __h_recv_displs = __h_column(__local_h_counts, __idx, __h_recv_displs_column);

    // Both count halves come back in one transfer: the device allocation holds them back to back
    // and the two host count columns mirror that layout, so the send and recv counts are a single
    // contiguous `2 * __comm_size` range on either side.
    ::cuda::copy_bytes(
      __counts.__get().stream(),
      __counts.__get(),
      ::cuda::std::span<::cuda::std::size_t>{__h_send_counts.data(), 2 * static_cast<::cuda::std::size_t>(__comm_size)},
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});

    // All streams are the same, so any suffices
    __counts.__get().stream().sync();

    // The send/recv displacements are just the exclusive prefix-sums of the corresponding
    // counts, and both are consumed only on the host (below and in the all_to_all_v). counts is
    // small (O(ranks)), so we scan on the host after the sync instead of paying a device scan
    // plus a D2H copy of the result. Host counts are only valid post-sync, so scan here.
    ::cuda::std::exclusive_scan(
      __h_send_counts.begin(), __h_send_counts.end(), __h_send_displs.begin(), ::cuda::std::size_t{0});
    ::cuda::std::exclusive_scan(
      __h_recv_counts.begin(), __h_recv_counts.end(), __h_recv_displs.begin(), ::cuda::std::size_t{0});

    const auto __total_recv = __h_recv_displs.back() + __h_recv_counts.back();

    __local_recvd.emplace_back(__counts.__get().stream(), __resource, __total_recv, ::cuda::no_init, __env);

    ++__idx;
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    __idx = 0;
    for (auto&& [__comm, __input, __recvd] : ::cuda::std::ranges::views::zip(__comms, __local_inputs, __local_recvd))
    {
      __comm.all_to_all_v(
        __guard,
        ::cuda::std::to_address(::cuda::std::ranges::begin(__input)),
        __h_column(__local_h_counts, __idx, __h_send_counts_column).data(),
        __h_column(__local_h_counts, __idx, __h_send_displs_column).data(),
        __recvd.data(),
        __h_column(__local_h_counts, __idx, __h_recv_counts_column).data(),
        __h_column(__local_h_counts, __idx, __h_recv_displs_column).data(),
        __recvd.__get().stream());

      ++__idx;
    }
  }

  // Merge the p received sorted runs into this phase's output.
  //
  // The merged keys stay in a stream-ordered buffer rather than being written back into the
  // caller's range: `__local_inputs` is the send buffer of the `all_to_all_v` enqueued above, and
  // resizing it here would both alias that in-flight collective and, for a container whose growth
  // reallocates through a synchronous allocator, block this rank inside a region where its peers
  // are waiting on a collective it has not yet joined. The caller's range is written exactly once,
  // at the very end of the rebalance phase, when nothing is in flight.
  ::std::vector<__buffer_type<_Tp>> __local_merged;

  __local_merged.reserve(__num_local_inputs);

  __idx = 0;
  for (auto&& [__comm, __env, __recvd] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_recvd))
  {
    auto& __merged = __local_merged.emplace_back(__recvd.__make_empty_like(0));

    __merge_k_way(
      __comm,
      __env,
      __recvd,
      __h_column(__local_h_counts, __idx, __h_recv_counts_column),
      __h_column(__local_h_counts, __idx, __h_recv_displs_column),
      __cmp,
      &__merged);

    ++__idx;
  }

  return __local_merged;
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H
