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
#include <cuda/std/__numeric/exclusive_scan.h>
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
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
//! @brief Realizes finalized splitter `i` from its `[L, U]` bracket, as either a key or a rank.
//!
//! Picks whichever bracket endpoint is closest to the ideal rank `Ni/p`, ties to `L` (HSS
//! Section 4.2.2 step (5)). `operator()` and `__rank` report the key and the global rank of that
//! one choice.
//!
//! @tparam _Tp The key (value) type.
template <class _Tp>
struct __bucket_to_splitter_key_fn
{
  const _Splitter<_Tp>* const __I_j;
  const _Tp* const __probes;
  const ::cuda::std::uint64_t __num_probes;
  const ::cuda::std::uint64_t* const __hist;
  const __ideal_rank_fn __ideal_rank;

  // Indexable so that it can be handed to `__bucket_count_fn` as the search sequence directly,
  // with no `transform_iterator` wrapper. `__bucket_count_fn` only ever subscripts it.
  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp operator[](const ::cuda::std::uint64_t __i) const noexcept
  {
    return __key(__i);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr _Tp __key(const ::cuda::std::uint64_t __i) const noexcept
  {
    if (__use_lower(__i))
    {
      return __I_j[__i].__L.__key.value_or(__probes[0]);
    }
    return __I_j[__i].__U.__key.value_or(__probes[__num_probes - 1]);
  }

  // Returns the global rank of a bucket index
  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint64_t
  __rank(const ::cuda::std::uint64_t __bucket) const noexcept
  {
    // A keyless bracket means the splitter realizes to a probe extremum, whose rank is a histogram
    // bucket rather than the bracket's 0 / N placeholder. Reporting the placeholder here hangs the
    // exchange's all_to_all_v.
    const auto& [__L_i, __U_i] = __I_j[__bucket];

    if (__use_lower(__bucket))
    {
      return __L_i.__key.has_value() ? __L_i.__rank : __hist[0];
    }
    return __U_i.__key.has_value() ? __U_i.__rank : (__ideal_rank.__N - __hist[__num_probes]);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool __use_lower(const ::cuda::std::uint64_t __i) const noexcept
  {
    const auto __target_rank   = __ideal_rank(__i);
    const auto& [__L_i, __U_i] = __I_j[__i];

    return (__target_rank - __L_i.__rank) <= (__U_i.__rank - __target_rank);
  }
};

//! @brief Emits, for one destination rank, both its send count and its post-exchange start offset.
//!
//! The ranks are read back out of the very `__bucket_to_splitter_key_fn` that `__count_fn`
//! searches against rather than from a second copy of it, so both columns are guaranteed to
//! describe the same splitter realization.
//!
//! `__count_fn` caches a cursor across calls, so it must be invoked exactly once per index,
//! unconditionally. Do not branch around this call or repeat it.
template <class _BucketCountFn>
struct __send_count_and_offset_fn
{
  // some specialization of __bucket_count_fn
  const _BucketCountFn __count_fn;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::tuple<::cuda::std::size_t, ::cuda::std::uint64_t>
  operator()(const ::cuda::std::uint64_t __bucket) const noexcept
  {
    // Rank 0's interval starts at 0; there is no splitter -1 to take a rank from.
    const auto __offset = __bucket == 0 ? ::cuda::std::uint64_t{0} : __count_fn.__search_it.__rank(__bucket - 1);

    return {static_cast<::cuda::std::size_t>(__count_fn(__bucket)), __offset};
  }
};

// The host-side count bookkeeping for the exchange is one flat allocation holding, per local
// rank, a row of `__h_num_columns` `__comm_size`-wide columns.
inline constexpr ::cuda::std::size_t __h_send_counts_column = 0;
inline constexpr ::cuda::std::size_t __h_recv_counts_column = 1;
inline constexpr ::cuda::std::size_t __h_send_displs_column = 2;
inline constexpr ::cuda::std::size_t __h_recv_displs_column = 3;
inline constexpr ::cuda::std::size_t __h_num_columns        = 4;

//! @brief Returns local rank `__rank_idx`'s `__col` column of `__h_counts`.
[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::span<::cuda::std::size_t> __h_column(
  ::std::vector<::cuda::std::size_t>& __h_counts,
  ::cuda::std::size_t __comm_size,
  ::cuda::std::size_t __rank_idx,
  ::cuda::std::size_t __col) noexcept
{
  return {__h_counts.data() + (((__rank_idx * __h_num_columns) + __col) * __comm_size), __comm_size};
}

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Compute, per destination rank, how many local keys it is owed and where they land.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
_CCCL_HOST_API ::std::vector<
  typename _HSSSorter<_Tp, _Env, _BinaryOp>::template __resizable_buffer_type<::cuda::std::size_t>>
_HSSSorter<_Tp, _Env, _BinaryOp>::__compute_send_counts_and_offsets(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  const _BinaryOp& __cmp,
  const ::std::vector<__per_comm_histogramming_result_type>& __hist_results,
  ::std::vector<__resizable_buffer_type<::cuda::std::uint64_t>>* __local_current_offsets)
{
  const auto __comm_size = __setup.__comm_size;
  const auto __N         = __setup.__N;
  const auto __num_local = ::cuda::std::ranges::size(__comms);

  ::std::vector<__resizable_buffer_type<::cuda::std::size_t>> __local_counts;

  // The send and recv counts are the same size, live on the same device, and are used on the
  // same stream, so they share one allocation per rank instead of two: the send counts occupy
  // `[0, __comm_size)` and the recv counts `[__comm_size, 2 * __comm_size)`.
  const auto __send_span = [__comm_size](auto& __counts) {
    return __counts.subspan(0, __comm_size);
  };
  const auto __recv_span = [__comm_size](auto& __counts) {
    return __counts.subspan(__comm_size, __comm_size);
  };

  __local_counts.reserve(__num_local);
  __local_current_offsets->reserve(__num_local);

  {
    auto __comm_it      = ::cuda::std::ranges::begin(__comms);
    auto __env_it       = ::cuda::std::ranges::begin(__envs);
    auto __input_it     = ::cuda::std::ranges::begin(__input_iters);
    auto __num_items_it = ::cuda::std::ranges::begin(__num_items_range);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local;
         (void) ++__idx, (void) ++__comm_it, (void) ++__env_it, (void) ++__input_it, (void) ++__num_items_it)
    {
      const auto& __hist   = __hist_results[__idx].__hist;
      const auto& __I_j    = __hist_results[__idx].__splitters.__I_j;
      const auto& __probes = __hist_results[__idx].__splitters.__probes;

      auto& __counts = __local_counts.emplace_back(
        __I_j.stream(),
        __I_j.memory_resource(),
        2 * __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it));
      auto& __offsets = __local_current_offsets->emplace_back(
        __I_j.stream(),
        __I_j.memory_resource(),
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it));

      // A final round whose splitters all landed on an exact rank match narrows every sampling
      // interval to width zero, so it draws no samples and merges to zero probes. The probe set
      // from the preceding round is still the finalized one, and its allocation is what carries
      // it here, so this checks the allocation rather than the logical size.
      _CCCL_VERIFY(__probes.capacity() != 0, "Histogramming phase should have generated at least one probe");
      _CCCL_VERIFY(__hist.size() == __probes.size() + 1, "The probe histogram must still describe the final probe set");

      // Everyone is sorted locally but nobody holds a globally correct slice yet, so each rank
      // has to hand every key to whichever rank the splitters say owns it. Per destination d we
      // need:
      //
      // - send count: how many of our keys belong to d. Sizes are all we have to agree on, since
      //   we are sorted and d's keys are already a contiguous run of ours.
      // - offset: d's position in the final order. The splitters divide by key value, not
      //   evenly, so d ends up with more or fewer keys than it started with. The rebalance phase
      //   reads these to put the original sizes back (note we should consider dropping rebalance).
      //
      // Both fall out of splitter d - 1, because a bracket endpoint carries a key *and* that
      // key's global rank. The key bounds the search for the count; the rank is already the
      // offset, since it counts exactly the keys ordered before d. There is one splitter object,
      // held by the search iterator, so the two columns cannot describe different splitters.
      //
      // The reason for complexity here is that this is actually the fusion of 2 separate
      // kernels. It used to be that the counts and offsets were computed separately, but they
      // ultimately both ask the same question "for my particular bucket, how many keys and where
      // are they?".

      // TODO(jfaibussowit)
      //
      // This mess can probably be simplified greatly. __bucket_count_fn is shared with the
      // histogramming phase, but ultimately it does something very similar here. I wonder if we
      // can't just reuse the histogram directly here. There is a lot of repeated information
      // here.
      const auto* __input_begin = ::cuda::std::to_address(*__input_it);
      using __input_it_t        = ::cuda::std::remove_cvref_t<decltype(__input_begin)>;

      auto __op =
        __send_count_and_offset_fn<__bucket_count_fn<__input_it_t, __bucket_to_splitter_key_fn<_Tp>, _BinaryOp>>{
          {__input_begin,
           __input_begin + *__num_items_it,
           // This is doing double duty here. Not only do we use it to calculate the actual size
           // of each bin, but we also use the __rank() function to calculate the offsets.
           __bucket_to_splitter_key_fn<_Tp>{
             __I_j.data(),
             __probes.data(),
             static_cast<::cuda::std::uint64_t>(__probes.size()),
             __hist.data(),
             __ideal_rank_fn{__N, static_cast<::cuda::std::uint64_t>(__comm_size)}},
           __I_j.size(),
           __cmp}};

      const auto __send_counts = __send_span(__counts);

      auto __out = ::cuda::std::make_tuple(__send_counts.data(), __offsets.data());

      __CUDAX_MULTI_GPU_DISPATCH(
        __comm_it->logical_device(),
        CUB_NS_QUALIFIER::DeviceTransform::Transform,
        ::cuda::counting_iterator<::cuda::std::uint64_t>{},
        ::cuda::std::move(__out),
        __send_counts.size(),
        ::cuda::std::move(__op),
        *__env_it);
    }
  }

  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto&& __guard = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local; (void) ++__idx, (void) ++__comm_it)
    {
      auto* const __send_ptr = __send_span(__local_counts[__idx]).data();
      auto* const __recv_ptr = __recv_span(__local_counts[__idx]).data();

      __comm_it->all_to_all(__guard, __send_ptr, __recv_ptr, /*__count=*/1, __local_counts[__idx].stream());
    }
  }

  return __local_counts;
}

template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange>
_CCCL_HOST_API ::std::vector<typename _HSSSorter<_Tp, _Env, _BinaryOp>::template __resizable_buffer_type<_Tp>>
_HSSSorter<_Tp, _Env, _BinaryOp>::__make_recv_buffers(
  _CommRange&& __comms,
  _EnvRange&& __envs,
  ::cuda::std::size_t __comm_size,
  const ::std::vector<__resizable_buffer_type<::cuda::std::size_t>>& __local_counts,
  ::std::vector<::cuda::std::size_t>* __h_counts)
{
  const auto __num_local = ::cuda::std::ranges::size(__comms);
  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);

    // Queue the memcpys first
    for (::cuda::std::size_t __idx = 0; __idx < __num_local; (void) ++__idx, (void) ++__comm_it)
    {
      const auto __h_send_counts = __h_column(*__h_counts, __comm_size, __idx, __h_send_counts_column);

      static_assert(__h_recv_counts_column == __h_send_counts_column + 1,
                    "The fused counts copy requires the send and recv count columns to be adjacent");
      ::cuda::copy_bytes(
        __local_counts[__idx].stream(),
        __local_counts[__idx],
        ::cuda::std::span<::cuda::std::size_t>{__h_send_counts.data(), 2 * __h_send_counts.size()},
        ::cuda::copy_configuration{__comm_it->logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
    }
  }

  ::std::vector<__resizable_buffer_type<_Tp>> __local_recvd;

  __local_recvd.reserve(__num_local);

  {
    auto __env_it = ::cuda::std::ranges::begin(__envs);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local; (void) ++__idx, (void) ++__env_it)
    {
      // All streams are the same, so any suffices
      __local_counts[__idx].stream().sync();

      const auto __h_send_counts = __h_column(*__h_counts, __comm_size, __idx, __h_send_counts_column);
      const auto __h_recv_counts = __h_column(*__h_counts, __comm_size, __idx, __h_recv_counts_column);
      const auto __h_send_displs = __h_column(*__h_counts, __comm_size, __idx, __h_send_displs_column);
      const auto __h_recv_displs = __h_column(*__h_counts, __comm_size, __idx, __h_recv_displs_column);
      // The send/recv displacements are just the exclusive prefix-sums of the corresponding
      // counts, and both are consumed only on the host (below and in the all_to_all_v). counts
      // is small (O(ranks)), so we scan on the host after the sync instead of paying a device
      // scan plus a D2H copy of the result. Host counts are only valid post-sync, so scan
      // here.
      ::cuda::std::exclusive_scan(
        __h_send_counts.begin(), __h_send_counts.end(), __h_send_displs.begin(), ::cuda::std::size_t{0});
      ::cuda::std::exclusive_scan(
        __h_recv_counts.begin(), __h_recv_counts.end(), __h_recv_displs.begin(), ::cuda::std::size_t{0});

      const auto __total_recv = __h_recv_displs.back() + __h_recv_counts.back();

      __local_recvd.emplace_back(
        __local_counts[__idx].stream(),
        __local_counts[__idx].memory_resource(),
        __total_recv,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it));
    }
  }

  return __local_recvd;
}

//! @brief Send the keys in `[S(d - 1), S(d))` to rank `d`, then merge the runs each rank receives.
//!
//! The keys in `__input_iters` must be locally sorted and `__hist_results` carry finalized
//! brackets from `__histogramming_phase`.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
_CCCL_HOST_API typename _HSSSorter<_Tp, _Env, _BinaryOp>::__data_exchange_result_type
_HSSSorter<_Tp, _Env, _BinaryOp>::__data_exchange(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputIterRange&& __input_iters,
  _SizeTRange&& __num_items_range,
  const _BinaryOp& __cmp,
  const ::std::vector<__per_comm_histogramming_result_type>& __hist_results)
{
  const auto __comm_size = __setup.__comm_size;
  const auto __num_local = ::cuda::std::ranges::size(__comms);

  ::std::vector<__resizable_buffer_type<::cuda::std::uint64_t>> __local_current_offsets;
  ::std::vector<::cuda::std::size_t> __local_h_counts(__num_local * __h_num_columns * __comm_size);

  auto __local_recvd = [&] {
    const auto __local_counts = __compute_send_counts_and_offsets(
      __setup, __comms, __envs, __input_iters, __num_items_range, __cmp, __hist_results, &__local_current_offsets);

    return __make_recv_buffers(__comms, __envs, __comm_size, __local_counts, &__local_h_counts);
  }();

  {
    auto __comm_it  = ::cuda::std::ranges::begin(__comms);
    auto __input_it = ::cuda::std::ranges::begin(__input_iters);
    auto&& __guard  = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local; (void) ++__idx, (void) ++__comm_it, (void) ++__input_it)
    {
      __comm_it->all_to_all_v(
        __guard,
        ::cuda::std::to_address(*__input_it),
        __h_column(__local_h_counts, __comm_size, __idx, __h_send_counts_column).data(),
        __h_column(__local_h_counts, __comm_size, __idx, __h_send_displs_column).data(),
        __local_recvd[__idx].data(),
        __h_column(__local_h_counts, __comm_size, __idx, __h_recv_counts_column).data(),
        __h_column(__local_h_counts, __comm_size, __idx, __h_recv_displs_column).data(),
        __local_recvd[__idx].stream());
    }
  }

  // Merge the p received sorted runs into this phase's output.
  //
  // The merged keys stay in a stream-ordered buffer rather than being written back into the
  // caller's storage: the keys behind `__input_iters` are the send buffer of the `all_to_all_v`
  // enqueued above, and writing them here would alias that in-flight collective. The caller's
  // storage is written exactly once, at the very end of the rebalance phase, when nothing is in
  // flight.
  ::std::vector<__resizable_buffer_type<_Tp>> __local_merged;

  __local_merged.reserve(__num_local);

  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto __env_it  = ::cuda::std::ranges::begin(__envs);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local; (void) ++__idx, (void) ++__comm_it, (void) ++__env_it)
    {
      auto& __merged = __local_merged.emplace_back(
        __local_recvd[__idx].stream(),
        __local_recvd[__idx].memory_resource(),
        ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it));

      __merge_k_way(
        *__comm_it,
        *__env_it,
        __local_recvd[__idx],
        __h_column(__local_h_counts, __comm_size, __idx, __h_recv_counts_column),
        __h_column(__local_h_counts, __comm_size, __idx, __h_recv_displs_column),
        __cmp,
        &__merged);
    }
  }

  return __data_exchange_result_type{::cuda::std::move(__local_merged), ::cuda::std::move(__local_current_offsets)};
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H
