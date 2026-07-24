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

#include <cuda/__container/buffer.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>
#include <cuda/experimental/__utility/result_policy.cuh>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Load-balance tolerance for the HSS sampling loop (2%).
//!
//! Bounds the acceptable deviation of each rank's final key count from the ideal `N/p`. It
//! feeds the paper's sample-count schedule in `__histogramming_phase`: both the number of
//! sampling rounds `K` and the per-round sample budget `s_j` scale with `1 / __eps`.
inline constexpr double __eps = 0.02; // 2% tolerance

template <class _Tp>
struct _Bracket
{
  ::cuda::std::uint64_t __rank; // < global rank of the key
  ::cuda::std::optional<_Tp> __key; // < the key, if found. If nullopt means either +/- inf
};

template <template <class> class _Buffer, class _Tp>
struct _PerCommSplitters
{
  _Buffer<_Bracket<_Tp>> __Ls;
  _Buffer<_Bracket<_Tp>> __Us;
  _Buffer<_Tp> __probes;
};

template <template <class> class _Buffer, class _Tp>
struct _PerCommSamplingScratch
{
  _Buffer<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>> __I_j;
  _Buffer<_Tp> __samples;
  _Buffer<::cuda::std::size_t> __samples_size;
  _Buffer<::cuda::std::uint64_t> __hist;
  ::cuda::buffer<::cuda::std::uint64_t, ::cuda::mr::device_accessible, ::cuda::mr::host_accessible> __probe_counts;
  ::cuda::std::size_t __sample_sendcount{};
};

template <class _Resource, template <class> class _Buffer>
struct _LocalSetupResult
{
  ::std::vector<_Resource> __resources{};
  ::std::vector<_Buffer<::cuda::std::uint64_t>> __all_local_offsets{};
  ::std::vector<::cuda::std::size_t> __local_original_sizes{};
  ::cuda::std::uint64_t __N{};
  ::cuda::std::int32_t __comm_size{};
};

template <class _Tp, class _Env, class _BinaryOp>
class _HSSSorter
{
public:
  using __resource_type _CCCL_NODEBUG = ::cuda::experimental::__detail::__resource_type_for<_Env>;

  // The size/capacity-aware device buffer type for element type `_Up`.
  template <class _Up>
  using __buffer_type _CCCL_NODEBUG = ::cuda::experimental::__detail::__hss_sort::__buffer<_Up, __resource_type>;

  using __bracket_type _CCCL_NODEBUG                   = _Bracket<_Tp>;
  using __per_comm_splitters_type _CCCL_NODEBUG        = _PerCommSplitters<__buffer_type, _Tp>;
  using __per_comm_sampling_scratch_type _CCCL_NODEBUG = _PerCommSamplingScratch<__buffer_type, _Tp>;
  using __local_setup_result_type _CCCL_NODEBUG        = _LocalSetupResult<__resource_type, __buffer_type>;

private:
  template <class _CommRange, class _EnvRange, class _InputRange>
  [[nodiscard]] _CCCL_HOST_API static __local_setup_result_type __local_setup(
    _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __local_inputs, ::cuda::std::int32_t __comm_size);

  // Histogram helpers
  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputRange>
  [[nodiscard]] _CCCL_HOST_API static ::std::vector<__per_comm_splitters_type> __histogramming_phase(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __local_inputs,
    const _BinaryOp& __cmp);

  template <class _InputRange>
  _CCCL_HOST_API static void __sample_probes(
    _InputRange&& __input,
    const __buffer_type<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>>& __I_j,
    double __sampling_probability,
    _BinaryOp __cmp,
    __buffer_type<_Tp>* __samples,
    __buffer_type<::cuda::std::size_t>* __sample_size);

  template <class _CommRange, class _EnvRange>
  [[nodiscard]]
  _CCCL_HOST_API static ::cuda::std::pair<::std::vector<__per_comm_splitters_type>,
                                          ::std::vector<__per_comm_sampling_scratch_type>>
  __allocate_histogramming_buffers(const __local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs);

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __gather_merge_broadcast(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    const _BinaryOp& __cmp,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch,
    ::std::vector<__per_comm_splitters_type>* __local_splitters,
    ::std::vector<::cuda::std::size_t>* __root_recvcounts,
    ::std::vector<::cuda::std::size_t>* __root_displs,
    ::cuda::std::optional<__buffer_type<_Tp>>* __root_all_samples);

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __compute_histogram(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __range_of_local_keys,
    const ::std::vector<__per_comm_splitters_type>& __local_splitters,
    const _BinaryOp& __cmp,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch);

  template <class _CommRange, class _EnvRange>
  _CCCL_HOST_API static void __update_intervals(
    _CommRange&& __comms,
    _EnvRange&& __envs,
    ::cuda::std::uint64_t __N,
    ::std::vector<__per_comm_splitters_type>* __local_splitters,
    ::std::vector<__per_comm_sampling_scratch_type>* __local_scratch);

  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __data_exchange(
    const __local_setup_result_type& __setup,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __local_inputs,
    const _BinaryOp& __cmp,
    const ::std::vector<__per_comm_splitters_type>& __local_splitters);

  // ------------------------------------------------------------------------------------------

  template <class _Comm>
  _CCCL_HOST_API static void __merge_k_way(
    const _Comm& __comm,
    const _Env& __env,
    const __buffer_type<_Tp>& __data,
    const ::std::vector<::cuda::std::size_t>& __counts,
    const ::std::vector<::cuda::std::size_t>& __displs,
    const _BinaryOp& __cmp,
    __buffer_type<_Tp>* __ret);

  // ------------------------------------------------------------------------------------------

  template <class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __rebalance_to_original_counts(
    const __local_setup_result_type& __setup, _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __local_inputs);

public:
  template <class _Policy, class _CommRange, class _EnvRange, class _InputRange>
  _CCCL_HOST_API static void __execute(
    const __result_policy_base<_Policy>&,
    _CommRange&& __comms,
    _EnvRange&& __envs,
    _InputRange&& __local_inputs,
    _BinaryOp __cmp);
};
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_TRAITS_H
