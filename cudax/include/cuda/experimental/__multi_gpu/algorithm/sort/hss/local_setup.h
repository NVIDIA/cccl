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
#include <cuda/std/__ranges/access.h>
#include <cuda/std/__ranges/size.h>
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
  ::cuda::std::uint64_t __N = 0;

  // TODO (jfaibussowit): maybe can combine some of these
  __all_local_offsets.reserve(__num_local_inputs);

  ::std::vector<__buffer_type<::cuda::std::uint64_t>> __all_local_sizes;

  __all_local_sizes.reserve(__num_local_inputs);

  {
    auto __comm_it  = ::cuda::std::ranges::begin(__comms);
    auto __env_it   = ::cuda::std::ranges::begin(__envs);
    auto __input_it = ::cuda::std::ranges::begin(__local_inputs);

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs;
         (void) ++__idx, (void) ++__comm_it, (void) ++__env_it, (void) ++__input_it)
    {
      auto& __sizes = __all_local_sizes.emplace_back(::cuda::make_buffer<::cuda::std::uint64_t>(
        ::cuda::get_stream(*__env_it),
        ::cuda::experimental::__detail::__resource_from_env(*__env_it, __comm_it->logical_device().underlying_device()),
        __comm_size,
        ::cuda::no_init,
        ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it)));

      const auto __n_local = static_cast<::cuda::std::uint64_t>(::cuda::std::ranges::size(*__input_it));

      ::cuda::copy_bytes(
        __sizes.stream(),
        ::cuda::std::span{&__n_local, ::cuda::std::size_t{1}},
        __sizes.subspan(__comm_it->rank(), 1),
        ::cuda::copy_configuration{::cuda::host_memory_location,
                                   __comm_it->logical_device().underlying_device(),
                                   ::cuda::source_access_order::during_api_call});
    }
  }

  {
    auto __comm_it = ::cuda::std::ranges::begin(__comms);
    auto&& __guard = __comm_it->group_guard();

    for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it)
    {
      auto* const __ptr = __all_local_sizes[__idx].data();

      __comm_it->all_gather(__guard, __ptr + __comm_it->rank(), __ptr, /*__count=*/1, __all_local_sizes[__idx].stream());
    }
  }

  // TODO(jfaibussowit)
  //
  // Consider deferring this. We end up doing a very similar computation later on on the root
  // and could potentially merge it there.
  bool __N_computed = false;

  auto __comm_it = ::cuda::std::ranges::begin(__comms);
  auto __env_it  = ::cuda::std::ranges::begin(__envs);

  for (::cuda::std::size_t __idx = 0; __idx < __num_local_inputs; (void) ++__idx, (void) ++__comm_it, (void) ++__env_it)
  {
    auto& __offsets = __all_local_offsets.emplace_back(
      __all_local_sizes[__idx].stream(),
      __all_local_sizes[__idx].memory_resource(),
      __comm_size,
      ::cuda::no_init,
      ::cuda::experimental::__detail::__sanitize_buffer_env(*__env_it));

    __CUDAX_MULTI_GPU_DISPATCH(
      __comm_it->logical_device(),
      CUB_NS_QUALIFIER::DeviceScan::ExclusiveSum,
      __all_local_sizes[__idx].begin(),
      __offsets.begin(),
      __all_local_sizes[__idx].size(),
      *__env_it);

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
        ::cuda::copy_configuration{__comm_it->logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});
      ::cuda::copy_bytes(
        __all_local_sizes[__idx].stream(),
        __all_local_sizes[__idx].subspan(__comm_size - 1, 1),
        ::cuda::std::span{&__last_size, ::cuda::std::size_t{1}},
        ::cuda::copy_configuration{__comm_it->logical_device().underlying_device(),
                                   ::cuda::host_memory_location,
                                   ::cuda::source_access_order::stream});

      __all_local_sizes[__idx].stream().sync();
      __N          = __last_offset + __last_size;
      __N_computed = true;
    }
  }

  return __local_setup_result_type{::cuda::std::move(__all_local_offsets), __N, __comm_size};
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_LOCAL_SETUP_H
