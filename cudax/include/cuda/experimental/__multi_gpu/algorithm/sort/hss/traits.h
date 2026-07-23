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
#include <cuda/std/__optional/optional.h>
#include <cuda/std/__utility/pair.h>
#include <cuda/std/cstdint>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
inline constexpr double __eps                     = 0.02; // 2% tolerance
inline constexpr ::cuda::std::int32_t __root_rank = 0;

// Shared type aliases and per-comm state for the HSS (histogram-sort-with-sampling) phases.
// Replaces the type-carrying role of the former _Sorter<_Tp, _Env> class: every free function
// below takes __hss_traits as its first (non-deduced) template argument and recovers the local
// aliases it needs from it.
template <class _Tp, class _Env, class _BinaryOp>
struct __hss_traits
{
  using __value_type _CCCL_NODEBUG     = _Tp;
  using __env_type _CCCL_NODEBUG       = _Env;
  using __resource_type _CCCL_NODEBUG  = ::cuda::experimental::__detail::__resource_type_for<_Env>;
  using __binary_op_type _CCCL_NODEBUG = _BinaryOp;

  template <class _Up>
  using __buffer_type _CCCL_NODEBUG = ::cuda::experimental::__detail::__hss_sort::__buffer<_Up, __resource_type>;

  // Persistent per-splitter bracket. rank = global rank of `key`.
  //
  // L: largest key proven to sit BELOW the ideal rank Ni/p.
  // U: smallest key proven to sit ABOVE  the ideal rank Ni/p.
  struct __bracket_type
  {
    ::cuda::std::uint64_t __rank; // < global rank of the key
    ::cuda::std::optional<__value_type> __key; // < the key, if found. If nullopt means either +/- inf
  };

  struct __per_comm_splitters_type
  {
    __buffer_type<__bracket_type> __Ls;
    __buffer_type<__bracket_type> __Us;
    __buffer_type<_Tp> __probes;
  };

  struct __per_comm_sampling_scratch_type
  {
    __buffer_type<::cuda::std::pair<::cuda::std::optional<_Tp>, ::cuda::std::optional<_Tp>>> __I_j;
    __buffer_type<_Tp> __samples;
    __buffer_type<::cuda::std::size_t> __samples_size;
    __buffer_type<::cuda::std::uint64_t> __hist;
    ::cuda::buffer<::cuda::std::uint64_t, ::cuda::mr::device_accessible, ::cuda::mr::host_accessible> __probe_counts;
    ::cuda::std::size_t __sample_sendcount{};
  };

  // Outputs of the HSS local setup phase (paper Section 6, "local sorting of input data").
  //
  // Besides the sorted local runs (produced in-place on the input), resources reused by every
  // later phase, the exclusive-scan of the original per-rank sizes (__all_local_offsets, the
  // desired final offsets consumed by rebalance), the original per-rank sizes, and the derived
  // global key count __N.
  struct __local_setup_result_type
  {
    ::std::vector<__resource_type> __resources{};
    ::std::vector<__buffer_type<::cuda::std::uint64_t>> __all_local_offsets{};
    ::std::vector<::cuda::std::size_t> __local_original_sizes{};
    ::cuda::std::uint64_t __N{};
    ::cuda::std::int32_t __comm_size{};
  };
};

// Convenience alias for the per-comm buffer type carried by a traits instantiation. Spells
// `typename _Traits::template __buffer_type<_Up>` without the `typename ... ::template` noise.
template <class _Traits, class _Up>
using __buffer_of = typename _Traits::template __buffer_type<_Up>;
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_TRAITS_H
