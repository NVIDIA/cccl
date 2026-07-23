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

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_BUCKET_COUNT_FN_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_BUCKET_COUNT_FN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/lower_bound.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail::__sort::__hss
{
// Per-bucket count functor shared by __local_histogram and __data_exchange. Both phases walk a
// sorted key range against a monotonically increasing bucket index, and for bucket b emit the
// number of keys landing in [split[b - 1], split[b]) (the boundary buckets are half-open at
// the range ends). Because CUB drives the Transform with a monotonically increasing bucket
// index and a fixed stride, we cache the previous upper bound in __lo: the next bucket's lower
// bound can never precede it, so each call only needs to search forward from __lo. This is why
// operator() is mutable -- it advances the __lo cursor across calls.
template <class _KeyIt, class _SplitIt, class _Cmp>
struct __bucket_count_fn
{
  mutable _KeyIt __lo; // mutable cursor: last computed upper bound, reused as the next lower bound
  _KeyIt __keys_last;
  _SplitIt __split_it;
  ::cuda::std::uint64_t __num_splitters;
  _Cmp __cmp;

  // operator() is intentionally non-const: it advances the __lo cursor across successive
  // monotonic bucket invocations.
  [[nodiscard]] _CCCL_DEVICE constexpr ::cuda::std::uint64_t operator()(::cuda::std::uint64_t __bucket) const noexcept
  {
    auto __hi = __keys_last;
    // This reordering takes advantage of the fact that we cached __lo previously. In
    // that case, __lo will already have raised the lower bound of the search. We now
    // just need to raise the *upper* bound of the search. We cannot do that generally by
    // caching (because the operators are called from left to right), but we *can* bound
    // the search for __lo.
    if (__bucket != __num_splitters)
    {
      __hi = ::cuda::std::lower_bound(__lo, __keys_last, __split_it[__bucket], __cmp);
    }

    if (__bucket != 0)
    {
      __lo = ::cuda::std::lower_bound(__lo, __hi, __split_it[__bucket - 1], __cmp);
    }

    const auto __ret = static_cast<::cuda::std::uint64_t>(__hi - __lo);
    // This caching of __lo relies on the fact that CUB calls these operators with
    // monotonically increasing bucket count (i.e. the stride is always adding blockIdx.x). If
    // that doesn't happen, then this caching is wrong.
    __lo = __hi;
    return __ret;
  }
};
} // namespace cuda::experimental::__detail::__sort::__hss

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_BUCKET_COUNT_FN_H
