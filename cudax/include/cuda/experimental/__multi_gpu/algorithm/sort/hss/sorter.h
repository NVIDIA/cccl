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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_TRAITS_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__container/resizable_buffer.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
template <class _Tp>
struct _Bracket
{
  ::cuda::std::uint64_t __rank; // < global rank of the key
  ::cuda::std::optional<_Tp> __key; // < the key, if found. If nullopt means either +/- inf
};

//! @brief One splitter's `[L, U]` rank bracket pair.
//!
//! `__L` and `__U` bracket the splitter's ideal rank from below and from above. The pair of
//! keys `(__L.__key, __U.__key)` is also that splitter's sampling interval for the following
//! round, so the histogramming phase rewrites these in place rather than projecting the keys
//! out into a separate interval array.
template <class _Tp>
struct _Splitter
{
  _Bracket<_Tp> __L;
  _Bracket<_Tp> __U;
};

template <class _Tp, template <class> class _Buffer>
struct _PerCommSplitters
{
  _Buffer<_Splitter<_Tp>> __I_j;
  _Buffer<_Tp> __probes;
};

template <class _Tp, template <class> class _Buffer>
struct _PerCommSamplingScratch
{
  // Holds every rank's samples in one fixed-capacity slot per rank, laid out in rank order.
  // This rank samples straight into its own slot and the all-gather fills the others, so no
  // separate per-rank sample buffer is needed. Slots are sized to an upper bound, so each one
  // is only partly filled; __samples_size says by how much.
  _Buffer<_Tp> __all_samples;
  // A __comm_size array that holds how many samples each rank actually drew, so that
  // __samples_size[rank] gives the filled length of that rank's slot. Written on the device by
  // the sampling kernel, then all-gathered.
  _Buffer<::cuda::std::size_t> __samples_size;
};

template <template <class> class _Buffer>
struct _LocalSetupResult
{
  ::std::vector<_Buffer<::cuda::std::uint64_t>> __all_local_offsets{};
  ::std::vector<::cuda::std::uint64_t> __all_local_sizes{};
  ::cuda::std::uint64_t __N{};
  ::cuda::std::int32_t __comm_size{};
};

template <class _Tp, template <class> class _Buffer>
struct _PerCommHistogrammingResult
{
  _PerCommSplitters<_Tp, _Buffer> __splitters{};
  _Buffer<::cuda::std::uint64_t> __hist{};
};

template <class _Tp, template <class> class _Buffer>
struct _DataExchangeResult
{
  ::std::vector<_Buffer<_Tp>> __local_merged{};
  ::std::vector<_Buffer<::cuda::std::uint64_t>> __local_current_offsets{};
};

_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <class _Tp, class _Env, class _BinaryOp>
class _HSSSorter
{
public:
  using __resource_type _CCCL_NODEBUG = ::cuda::experimental::__detail::__resource_type_for<_Env>;

  // The size/capacity-aware device buffer type for element type `_Up`.
  template <class _Up>
  using __resizable_buffer_type _CCCL_NODEBUG =
    typename __resource_type::default_queries::template rebind<::cuda::__resizable_buffer, _Up>;

  using __per_comm_splitters_type _CCCL_NODEBUG            = _PerCommSplitters<_Tp, __resizable_buffer_type>;
  using __per_comm_sampling_scratch_type _CCCL_NODEBUG     = _PerCommSamplingScratch<_Tp, __resizable_buffer_type>;
  using __local_setup_result_type _CCCL_NODEBUG            = _LocalSetupResult<__resizable_buffer_type>;
  using __per_comm_histogramming_result_type _CCCL_NODEBUG = _PerCommHistogrammingResult<_Tp, __resizable_buffer_type>;
  using __data_exchange_result_type _CCCL_NODEBUG          = _DataExchangeResult<_Tp, __resizable_buffer_type>;

private:
  template <class _CommRange, class _EnvRange, class _SizeTRange>
  [[nodiscard]] _CCCL_HOST_API static __local_setup_result_type __local_setup(
    _CommRange&& __comms, _EnvRange&& __envs, _SizeTRange&& __num_items_range, ::cuda::std::int32_t __comm_size);

  // Histogram helpers
  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  [[nodiscard]] _CCCL_HOST_API static ::std::vector<__per_comm_histogramming_result_type> __histogramming_phase(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    const _BinaryOp& __cmp);

  template <class _CommRange, class _InputIterRange, class _SizeTRange>
  _CCCL_HOST_API static void __local_sampling(
    _CommRange&& __comms,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    ::cuda::std::int32_t __j,
    double __sampling_probability,
    const _BinaryOp& __cmp,
    const ::std::vector<__per_comm_histogramming_result_type>& __local_hist_results,
    ::cuda::std::span<const ::cuda::std::size_t> __cap_displs,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch);

  template <class _CommRange, class _EnvRange>
  [[nodiscard]]
  _CCCL_HOST_API static ::cuda::std::pair<::std::vector<__per_comm_sampling_scratch_type>,
                                          ::std::vector<__per_comm_histogramming_result_type>>
  __allocate_histogramming_buffers(const __local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs);

  template <class _CommRange>
  _CCCL_HOST_API static void __exchange_sample_counts(
    _CommRange&& __comms,
    ::cuda::std::span<::cuda::std::size_t> __h_recvcounts,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch);

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __gather_and_merge_probes(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const _BinaryOp& __cmp,
    ::cuda::std::span<const ::cuda::std::size_t> __h_recvcounts,
    ::cuda::std::span<const ::cuda::std::size_t> __h_cap_displs,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch,
    ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results);

  template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  _CCCL_HOST_API static void __compute_histogram(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __key_iters,
    _SizeTRange&& __num_items_range,
    const _BinaryOp& __cmp,
    ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results);

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __update_intervals(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    ::cuda::std::uint64_t __N,
    ::std::vector<__per_comm_histogramming_result_type>* __local_hist_results);

  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  [[nodiscard]] _CCCL_HOST_API static ::std::vector<__resizable_buffer_type<::cuda::std::size_t>>
  __compute_send_counts_and_offsets(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    const _BinaryOp& __cmp,
    const ::std::vector<__per_comm_histogramming_result_type>& __hist_results,
    ::std::vector<__resizable_buffer_type<::cuda::std::uint64_t>>* __local_current_offsets);

  template <class _CommRange, class _EnvRange>
  [[nodiscard]] _CCCL_HOST_API static ::std::vector<__resizable_buffer_type<_Tp>> __make_recv_buffers(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    ::cuda::std::size_t __comm_size,
    const ::std::vector<__resizable_buffer_type<::cuda::std::size_t>>& __local_counts,
    ::std::vector<::cuda::std::size_t>* __h_counts);

  template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  [[nodiscard]] _CCCL_HOST_API static __data_exchange_result_type __data_exchange(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    const _BinaryOp& __cmp,
    const ::std::vector<__per_comm_histogramming_result_type>& __hist_results);

  // ------------------------------------------------------------------------------------------

  template <class _Comm>
  _CCCL_HOST_API static void __merge_k_way_tree(
    const _Comm& __comm,
    const _Env& __env,
    const __resizable_buffer_type<_Tp>& __data,
    ::cuda::std::span<const ::cuda::std::size_t> __counts,
    ::cuda::std::span<const ::cuda::std::size_t> __displs,
    const _BinaryOp& __cmp,
    __resizable_buffer_type<_Tp>* __ret);

  template <class _Comm>
  _CCCL_HOST_API static void __merge_k_way(
    const _Comm& __comm,
    const _Env& __env,
    const __resizable_buffer_type<_Tp>& __data,
    ::cuda::std::span<const ::cuda::std::size_t> __counts,
    ::cuda::std::span<const ::cuda::std::size_t> __displs,
    const _BinaryOp& __cmp,
    __resizable_buffer_type<_Tp>* __ret);

  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  _CCCL_HOST_API static void __rebalance_to_original_counts(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    const __data_exchange_result_type& __exchange_result);

public:
  template <class _Policy, class _CommRange, class _EnvRange, class _InputIterRange, class _SizeTRange>
  _CCCL_HOST_API static void __execute(
    const __result_policy_base<_Policy>&,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputIterRange&& __input_iters,
    _SizeTRange&& __num_items_range,
    _BinaryOp __cmp);
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_TRAITS_H
