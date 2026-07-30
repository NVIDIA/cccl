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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_LOCAL_SETUP_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_LOCAL_SETUP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_scan.cuh>

#include <cuda/__algorithm/copy.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

//! @brief Measure the per-rank sizes, desired offsets, and global key count for the sort.
template <class _Tp, class _Env, class _BinaryOp>
template <class _CommRange, class _EnvRange, class _InputRange>
[[nodiscard]] _CCCL_HOST_API typename _HSSSorter<_Tp, _Env, _BinaryOp>::__local_setup_result_type
_HSSSorter<_Tp, _Env, _BinaryOp>::__local_setup(
  _CommRange&& __comms, _EnvRange&& __envs, _InputRange&& __local_inputs, ::cuda::std::int32_t __comm_size)
{
  const auto __num_local_inputs = ::cuda::std::ranges::size(__comms);

  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __all_local_offsets;
  ::std::vector<::cuda::std::size_t> __local_original_sizes;
  ::cuda::std::uint64_t __N = 0;

  // TODO (jfaibussowit): maybe can combine some of these
  __all_local_offsets.reserve(__num_local_inputs);
  __local_original_sizes.reserve(__num_local_inputs);

  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __all_local_sizes;

  __all_local_sizes.reserve(__num_local_inputs);

  for (auto&& [__comm, __env, __input] : ::cuda::std::ranges::views::zip(__comms, __envs, __local_inputs))
  {
    const auto __n_local = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(__input));

    auto& __sizes = __all_local_sizes.emplace_back(::cuda::make_buffer<::cuda::std::uint64_t>(
      ::cuda::get_stream(__env),
      ::cuda::experimental::__detail::__resource_from_env(__env, __comm.logical_device().underlying_device()),
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env)));

    __local_original_sizes.push_back(__n_local);
    ::cuda::copy_bytes(
      __sizes.stream(),
      ::cuda::std::span{&__n_local, ::cuda::std::size_t{1}},
      __sizes.subspan(__comm.rank(), 1),
      ::cuda::copy_configuration{::cuda::host_memory_location,
                                 __comm.logical_device().underlying_device(),
                                 ::cuda::source_access_order::during_api_call});
  }

  {
    auto&& __guard = ::cuda::std::ranges::begin(__comms)->group_guard();

    for (auto&& [__comm, __sizes] : ::cuda::std::ranges::views::zip(__comms, __all_local_sizes))
    {
      auto* const __ptr = __sizes.data();

      __comm.all_gather(__guard, __ptr + __comm.rank(), __ptr, /*__count=*/1, __sizes.stream());
    }
  }

  // TODO(jfaibussowit)
  //
  // Consider deferring this. We end up doing a very similar computation later on on the root
  // and could potentially merge it there.
  bool __N_computed = false;

  for (auto&& [__comm, __env, __input, __sizes] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __local_inputs, __all_local_sizes))
  {
    auto& __offsets = __all_local_offsets.emplace_back(
      __sizes.stream(),
      __sizes.memory_resource(),
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(__env));

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm.logical_device(),
      CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum,
      __sizes.begin(),
      __offsets.begin(),
      __sizes.size(),
      __env);

    if (!__N_computed)
    {
      ::cuda::std::uint64_t __last_offset = 0;
      ::cuda::std::uint64_t __last_size   = 0;

      // The desired-offset scan already encodes the global extent: N =
      // offset[p - 1] + size[p - 1].
      ::cuda::copy_bytes(
        __offsets.stream(),
        __offsets.subspan(__comm_size - 1, 1),
        ::cuda::std::span{&__last_offset, ::cuda::std::size_t{1}},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __sizes.stream(),
        __sizes.subspan(__comm_size - 1, 1),
        ::cuda::std::span{&__last_size, ::cuda::std::size_t{1}},
        ::cuda::copy_configuration{__comm.logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __sizes.stream().sync();
      __N          = __last_offset + __last_size;
      __N_computed = true;
    }
  }

  return __local_setup_result_type{
    ::cuda::std::move(__all_local_offsets), ::cuda::std::move(__local_original_sizes), __N, __comm_size};
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_LOCAL_SETUP_H
