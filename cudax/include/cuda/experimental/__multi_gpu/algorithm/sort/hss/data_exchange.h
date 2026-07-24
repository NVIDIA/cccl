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
#include <cuda/__iterator/zip_iterator.h>
#include <cuda/std/__iterator/back_insert_iterator.h>
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
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/traits.h>

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

  template <class _Tup>
  [[nodiscard]] _CCCL_DEVICE constexpr _Tp operator()(const _Tup& __tup) const noexcept
  {
    const auto [__target_rank, __L_i, __U_i] = __tup;
    const bool __use_L                       = (__target_rank - __L_i.__rank) <= (__U_i.__rank - __target_rank);

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
//! @param[in,out] __local_inputs The range of per-rank local key ranges, overwritten with the
//!                exchanged and merged keys.
//! @param[in] __cmp The comparator defining the sorted order.
//! @param[in] __local_splitters The per-comm splitter state supplying the finalized brackets and
//!            probes.
template <class _Traits, class _CommRange, class _EnvRange, class _InputRange, class _BinaryOp>
_CCCL_HOST_API void __data_exchange(
  const typename _Traits::__local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const _BinaryOp& __cmp,
  const ::std::vector<typename _Traits::__per_comm_splitters_type>& __local_splitters)
{
  using _Tp = typename _Traits::__value_type;

  const auto __comm_size = __setup.__comm_size;
  const auto __N         = __setup.__N;

  ::std::vector<__buffer_of<_Traits, ::cuda::std::size_t>> __local_send_counts;
  ::std::vector<__buffer_of<_Traits, ::cuda::std::size_t>> __local_recv_counts;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_counts;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_counts;

  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  __local_send_counts.reserve(__num_local_inputs);
  __local_recv_counts.reserve(__num_local_inputs);
  __local_h_send_counts.reserve(__num_local_inputs);
  __local_h_recv_counts.reserve(__num_local_inputs);

  for (auto&& [__comm, __env, __resource, __input, __splitters] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __setup.__resources, __local_inputs, __local_splitters))
  {
    const auto& __Ls     = __splitters.__Ls;
    const auto& __Us     = __splitters.__Us;
    const auto& __probes = __splitters.__probes;

    auto& __send_counts =
      __local_send_counts.emplace_back(__Ls.__get().stream(), __resource, __comm_size, ::cuda::no_init, __env);

    const auto* __input_begin = ::cuda::std::to_address(::cuda::std::ranges::begin(__input));

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
          ::cuda::counting_iterator<::cuda::std::uint64_t>{}, __ideal_rank_fn{__N, __comm_size}),
        __Ls.begin(),
        __Us.begin()),
      __finalize_splitters_fn<_Tp>{
        ::cuda::std::to_address(__probes.data()), ::cuda::std::to_address(__probes.end() - 1)});

    // Route this rank's local keys to destination ranks via the splitter keys: the Data
    // Exchange phase, HSS Section 3.1 step (3), "a key in range [S(i), S(i + 1)) goes to
    // processor i". HSS reuses this phase unchanged (Section 3.3), so bucket d receives the keys
    // in [S(d - 1), S(d)) and its count becomes the send metadata. The send displacements are the
    // exclusive prefix-sum of these counts (buckets are contiguous and non-overlapping), so we
    // recompute them on the host below instead of emitting a second device column here.
    auto __op = ::cuda::experimental::__detail::__hss_sort::__bucket_count_fn<
      ::cuda::std::remove_cvref_t<decltype(__input_begin)>,
      ::cuda::std::remove_cvref_t<decltype(__splitter_it)>,
      _BinaryOp>{
      __input_begin, ::cuda::std::to_address(::cuda::std::ranges::end(__input)), __splitter_it, __Ls.size(), __cmp};

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      __send_counts.data(),
      __comm_size,
      ::cuda::std::move(__op),
      __env);
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __send_counts] : ::cuda::std::ranges::views::zip(__comms, __local_send_counts))
    {
      auto& __recv_counts    = __local_recv_counts.emplace_back(__send_counts.__make_empty_like());
      auto* const __send_ptr = __send_counts.data();
      auto* const __recv_ptr = __recv_counts.data();

      __comm.all_to_all(__guard, __send_ptr, __recv_ptr, /*__count=*/1, __send_counts.__get().stream());
    }
  }

  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_send_displs;
  ::std::vector<::std::vector<::cuda::std::size_t>> __local_h_recv_displs;
  ::std::vector<__buffer_of<_Traits, _Tp>> __local_recvd;

  __local_recvd.reserve(__num_local_inputs);
  __local_h_send_displs.reserve(__num_local_inputs);
  __local_h_recv_displs.reserve(__num_local_inputs);
  for (auto&& [__comm, __resource, __env, __send_counts, __recv_counts] :
       ::cuda::std::ranges::views::zip(__comms, __setup.__resources, __envs, __local_send_counts, __local_recv_counts))
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
      __send_counts.__get().stream(),
      __send_counts.__get(),
      __h_send_counts,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});
    ::cuda::copy_bytes(
      __recv_counts.__get().stream(),
      __recv_counts.__get(),
      __h_recv_counts,
      ::cuda::copy_configuration{
        __comm.logical_device().underlying_device(), ::cuda::host_memory_location, ::cuda::source_access_order::stream});

    // All streams are the same, so any suffices
    __recv_counts.__get().stream().sync();

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

    __local_recvd.emplace_back(__recv_counts.__get().stream(), __resource, __total_recv, ::cuda::no_init, __env);
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __input, __recvd, __h_send_counts, __h_send_displs, __h_recv_counts, __h_recv_displs] :
         ::cuda::std::ranges::views::zip(
           __comms,
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
        __recvd.__get().stream());
    }
  }

  // --- Phase C: merge the p received sorted runs into the final local output
  // ---
  for (auto&& [__comm, __env, __recvd, __h_recv_counts, __h_recv_displs, __inputs] : ::cuda::std::ranges::views::zip(
         __comms, __envs, __local_recvd, __local_h_recv_counts, __local_h_recv_displs, __local_inputs))
  {
    // TODO(jfaibussowit):
    //
    // Don't use __tmp and instead write directly to __inputs
    auto __tmp = __buffer_of<_Traits, _Tp>{__recvd.__make_empty_like(0)};

    ::cuda::experimental::__detail::__hss_sort::__merge_k_way<_Traits>(
      __comm, __env, __recvd, __h_recv_counts, __h_recv_displs, __cmp, &__tmp);

    ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(__inputs, __tmp.size());

    ::cuda::copy_bytes(
      __tmp.__get().stream(),
      __tmp.__get(),
      ::cuda::std::span<_Tp>{::cuda::std::to_address(::cuda::std::ranges::begin(__inputs)), __tmp.size()},
      ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                 __comm.logical_device().underlying_device(),
                                 ::cuda::source_access_order::stream});
  }
}
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H
