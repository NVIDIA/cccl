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

namespace cuda::experimental::__detail::__hss_sort
{
//! @brief Per-bucket key-count functor for the histogramming and data-exchange phases.
//!
//! Counts, for a given bucket index `b`, how many keys of a sorted key range land in
//! `[split[b - 1], split[b])`, with the two boundary buckets half-open at the ends of the range
//! (bucket `0` covers `(-inf, split[0])` and bucket `__num_splitters` covers
//! `[split[__num_splitters - 1], +inf)`). It is shared by `__compute_histogram` in
//! `histogramming.h` (splitters = probe keys) and by `__data_exchange` in `data_exchange.h`
//! (splitters = the finalized splitter keys), and drives one CUB `DeviceTransform` in each.
//!
//! CUB invokes the operator with a monotonically increasing bucket index over a fixed stride,
//! so the previous call's upper bound is a valid lower bound for the next bucket. The functor
//! caches that upper bound in the `mutable` `__lo` cursor and searches forward from it via
//! `cuda::std::lower_bound`, turning the per-bucket scan into two bounded binary searches. The
//! operator is therefore declared `const` yet mutates `__lo`, which is correct only under CUB's
//! monotonic left-to-right invocation contract; any other call order would corrupt the cursor.
//!
//! @tparam _KeyIt The iterator type over the sorted key range.
//! @tparam _SearchIt The iterator type over the values to search for in the key range.
//! @tparam _Cmp The strict-weak-ordering comparator type used for the binary searches.
template <class _KeyIt, class _SearchIt, class _Cmp>
struct __bucket_count_fn
{
  mutable _KeyIt __lo; // mutable cursor: last computed upper bound, reused as the next lower bound
  _KeyIt __keys_last;
  _SearchIt __search_it;
  ::cuda::std::uint64_t __num_splitters;
  _Cmp __cmp;

  //! @brief Return the number of keys falling in bucket `__bucket`.
  //!
  //! @param[in] __bucket The bucket index in `[0, __num_splitters]`.
  //!
  //! @return The count of keys in the half-open interval delimited by the surrounding splitters.
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
      __hi = ::cuda::std::lower_bound(__lo, __keys_last, __search_it[__bucket], __cmp);
    }

    if (__bucket != 0)
    {
      __lo = ::cuda::std::lower_bound(__lo, __hi, __search_it[__bucket - 1], __cmp);
    }

    const auto __ret = static_cast<::cuda::std::uint64_t>(__hi - __lo);
    // This caching of __lo relies on the fact that CUB calls these operators with
    // monotonically increasing bucket count (i.e. the stride is always adding blockIdx.x). If
    // that doesn't happen, then this caching is wrong.
    __lo = __hi;
    return __ret;
  }
};
} // namespace cuda::experimental::__detail::__hss_sort

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_SORT_HSS_BUCKET_COUNT_FN_H
