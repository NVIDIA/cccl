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
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/move.h>
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
  const _Bracket<_Tp>* const __Ls;
  const _Bracket<_Tp>* const __Us;
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
      return __Ls[__i].__key.value_or(__probes[0]);
    }
    return __Us[__i].__key.value_or(__probes[__num_probes - 1]);
  }

  // Returns the global rank of a bucket index
  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::uint64_t
  __rank(const ::cuda::std::uint64_t __bucket) const noexcept
  {
    // A keyless bracket means the splitter realizes to a probe extremum, whose rank is a histogram
    // bucket rather than the bracket's 0 / N placeholder. Reporting the placeholder here hangs the
    // exchange's all_to_all_v.
    if (__use_lower(__bucket))
    {
      return __Ls[__bucket].__key.has_value() ? __Ls[__bucket].__rank : __hist[0];
    }
    return __Us[__bucket].__key.has_value() ? __Us[__bucket].__rank : (__ideal_rank.__N - __hist[__num_probes]);
  }

  [[nodiscard]] _CCCL_DEVICE_API constexpr bool __use_lower(const ::cuda::std::uint64_t __i) const noexcept
  {
    const auto __target_rank = __ideal_rank(__i);

    return (__target_rank - __Ls[__i].__rank) <= (__Us[__i].__rank - __target_rank);
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
template <class _BucketCount>
struct __send_count_and_offset_fn
{
  // some specialization of __bucket_count_fn
  const _BucketCount __count_fn;

  [[nodiscard]] _CCCL_DEVICE_API constexpr ::cuda::std::tuple<::cuda::std::size_t, ::cuda::std::uint64_t>
  operator()(const ::cuda::std::uint64_t __bucket) const noexcept
  {
    // Rank 0's interval starts at 0; there is no splitter -1 to take a rank from.
    const auto __offset = __bucket == 0 ? ::cuda::std::uint64_t{0} : __count_fn.__search_it.__rank(__bucket - 1);

    return {static_cast<::cuda::std::size_t>(__count_fn(__bucket)), __offset};
  }
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Send the keys in `[S(d - 1), S(d))` to rank `d`, then merge the runs each rank receives.
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
//! @param[in] __local_hists The per-comm all-reduced probe histograms from the histogramming
//!            phase.
//!
//! @returns The per-rank exchanged-and-merged keys alongside the per-rank starts of the
//!          post-exchange intervals.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
_CCCL_HOST_API typename _HSSSorter<_Tp, _Env, _BinaryOp>::__data_exchange_result_type
_HSSSorter<_Tp, _Env, _BinaryOp>::__data_exchange(
  const __local_setup_result_type& __setup,
  _CommRange&& __comms,
  _EnvRange&& __envs,
  _InputRange&& __local_inputs,
  const _BinaryOp& __cmp,
  const __histogramming_result_type& __hist_results)
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

  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __local_current_offsets;

  __local_current_offsets.reserve(__num_local_inputs);
  for (auto&& [__comm, __env, __resource, __input, __splitters, __hist] : ::cuda::std::ranges::views::zip(
         __comms,
         __envs,
         __setup.__resources,
         __local_inputs,
         __hist_results.__local_splitters,
         __hist_results.__local_hists))
  {
    const auto& __Ls     = __splitters.__Ls;
    const auto& __Us     = __splitters.__Us;
    const auto& __probes = __splitters.__probes;

    auto& __counts =
      __local_counts.emplace_back(__Ls.__get().stream(), __resource, 2 * __comm_size, ::cuda::no_init, __env);
    auto& __offsets =
      __local_current_offsets.emplace_back(__Ls.__get().stream(), __resource, __comm_size, ::cuda::no_init, __env);

    _CCCL_VERIFY(!__probes.__get().empty(), "Histogramming phase should have generated at least one probe");
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
    const auto* __input_begin = ::cuda::std::to_address(::cuda::std::ranges::begin(__input));
    using __input_it_t        = ::cuda::std::remove_cvref_t<decltype(__input_begin)>;

    auto __op = __send_count_and_offset_fn<__bucket_count_fn<__input_it_t, __bucket_to_splitter_key_fn<_Tp>, _BinaryOp>>{
      {__input_begin,
       ::cuda::std::to_address(::cuda::std::ranges::end(__input)),
       // This is doing double duty here. Not only do we use it to calculate the actual size
       // of each bin, but we also use the __rank() function to calculate the offsets.
       __bucket_to_splitter_key_fn<_Tp>{
         __Ls.data(),
         __Us.data(),
         __probes.data(),
         static_cast<::cuda::std::uint64_t>(__probes.size()),
         __hist.data(),
         __ideal_rank_fn{__N, static_cast<::cuda::std::uint64_t>(__comm_size)}},
       __Ls.size(),
       __cmp}};

    const auto __send_counts = __send_span(__counts);

    auto __out = ::cuda::std::make_tuple(__send_counts.data(), __offsets.data());

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceTransform::Transform,
      ::cuda::counting_iterator<::cuda::std::uint64_t>{},
      ::cuda::std::move(__out),
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

  constexpr ::cuda::std::size_t __h_send_counts_column = 0;
  constexpr ::cuda::std::size_t __h_recv_counts_column = 1;
  constexpr ::cuda::std::size_t __h_send_displs_column = 2;
  constexpr ::cuda::std::size_t __h_recv_displs_column = 3;
  constexpr ::cuda::std::size_t __h_num_columns        = 4;

  ::std::vector<::cuda::std::size_t> __local_h_counts(__num_local_inputs * __h_num_columns * __comm_size);

  const auto __h_column =
    [__comm_size](
      ::std::vector<::cuda::std::size_t>& __h_counts, ::cuda::std::size_t __rank_idx, ::cuda::std::size_t __col) {
      return ::cuda::std::span<::cuda::std::size_t>{__h_counts}.subspan(
        ((__rank_idx * __h_num_columns) + __col) * __comm_size, __comm_size);
    };

  ::std::vector<__buffer_type<_Tp>> __local_recvd;

  __local_recvd.reserve(__num_local_inputs);
  {
    auto __idx = 0;

    for (auto&& [__comm, __env, __counts] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_counts))
    {
      const auto __h_send_counts = __h_column(__local_h_counts, __idx, __h_send_counts_column);

      static_assert(__h_recv_counts_column == __h_send_counts_column + 1,
                    "The fused counts copy requires the send and recv count columns to be adjacent");
      ::cuda::copy_bytes(
        __counts.__get().stream(),
        __counts.__get(),
        ::cuda::std::span<::cuda::std::size_t>{__h_send_counts.data(), 2 * __h_send_counts.size()},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      // All streams are the same, so any suffices
      __counts.__get().stream().sync();

      const auto __h_recv_counts = __h_column(__local_h_counts, __idx, __h_recv_counts_column);
      const auto __h_send_displs = __h_column(__local_h_counts, __idx, __h_send_displs_column);
      const auto __h_recv_displs = __h_column(__local_h_counts, __idx, __h_recv_displs_column);
      // The send/recv displacements are just the exclusive prefix-sums of the corresponding
      // counts, and both are consumed only on the host (below and in the all_to_all_v). counts
      // is small (O(ranks)), so we scan on the host after the sync instead of paying a device
      // scan plus a D2H copy of the result. Host counts are only valid post-sync, so scan here.
      ::cuda::std::exclusive_scan(
        __h_send_counts.begin(), __h_send_counts.end(), __h_send_displs.begin(), ::cuda::std::size_t{0});
      ::cuda::std::exclusive_scan(
        __h_recv_counts.begin(), __h_recv_counts.end(), __h_recv_displs.begin(), ::cuda::std::size_t{0});

      const auto __total_recv = __h_recv_displs.back() + __h_recv_counts.back();

      __local_recvd.emplace_back(
        __counts.__get().stream(), __counts.__get().memory_resource(), __total_recv, ::cuda::no_init, __env);

      ++__idx;
    }
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();
    auto __idx     = 0;

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

  {
    auto __idx = 0;

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
  }

  return __data_exchange_result_type{::cuda::std::move(__local_merged), ::cuda::std::move(__local_current_offsets)};
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_DATA_EXCHANGE_H
