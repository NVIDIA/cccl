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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_merge.cuh>

#include <cuda/std/__cstddef/types.h>

#include <cuda/experimental/__multi_gpu/algorithm/common.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/buffer.h>
#include <cuda/experimental/__multi_gpu/algorithm/sort/hss/sorter.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__hss_sort
{
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

// TODO(jfaibussowit):
//
// Horrifically inefficient, needs to be replaced by a proper CUB primitive!
//
// __data holds a series of sorted sequences. The offsets of the beginning of each such
// sequence is in __displs, while the count is in __counts. __dipls[0] should probably be 0,
// but it's not required.
template <class _Tp, class _Env, class _BinaryOp>
template <class _Comm>
_CCCL_HOST_API void _HSSSorter<_Tp, _Env, _BinaryOp>::__merge_k_way(
  const _Comm& __comm,
  const _Env& __env,
  const __buffer_type<_Tp>& __data,
  const ::std::vector<::cuda::std::size_t>& __counts,
  const ::std::vector<::cuda::std::size_t>& __displs,
  const _BinaryOp& __cmp,
  __buffer_type<_Tp>* __ret)
{
  _CCCL_VERIFY(__counts.size() > 1, "We should never get here for single-node. We should have exited earlier.");

  const auto __total = __counts.back() + __displs.back();

  ::cuda::experimental::__detail::__hss_sort::__resize_for_overwrite(*__ret, __total);

  // This staggered implementation is an optimization to avoid allocating a temporary
  // buffer. If we can just do a single merge we don't need the bounce buffer.

  __CUDAX_MULTI_GPU_DISPATCH(
    __comm.logical_device(),
    CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
    __data.data() + __displs[0],
    __counts[0],
    __data.data() + __displs[1],
    __counts[1],
    __ret->data(),
    __cmp,
    __env);

  if (__counts.size() > 2)
  {
    ::cuda::std::size_t __merged_size = __counts[0] + __counts[1];
    auto __tmp_buf                    = __ret->__make_empty_like(__total);

    for (::cuda::std::size_t __i = 2; __i < __counts.size(); ++__i)
    {
      __CUDAX_MULTI_GPU_DISPATCH(
        __comm.logical_device(),
        CUB_NS_QUALIFIER::DeviceMerge::MergeKeys,
        __ret->data(),
        __merged_size,
        __data.data() + __displs[__i],
        __counts[__i],
        __tmp_buf.data(),
        __cmp,
        __env);

      __ret->__get().swap(__tmp_buf);
      __merged_size += __counts[__i];
    }
  }
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_MERGE_K_WAY_H
