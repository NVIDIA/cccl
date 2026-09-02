//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__UTILITY_UNSTABLE_UNIQUE
#define _CUDAX__UTILITY_UNSTABLE_UNIQUE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/next.h>
#include <cuda/std/__iterator/prev.h>
#include <cuda/std/__utility/move.h>

namespace cuda::experimental
{
//! @brief Removes duplicates from a sorted range using a custom predicate.
//!
//! Operates from both sides of the range, moving elements from the right-hand
//! side to the left to eliminate duplicates. The relative order of elements is
//! not preserved. Requires a bidirectional iterator.
//!
//! @tparam _Iterator The type of the iterator.
//! @tparam _BinaryPredicate The type of the predicate.
//!
//! @param[in] __first Iterator to the beginning of the range.
//! @param[in] __last Iterator past the end of the range.
//! @param[in] __pred The predicate used to compare adjacent elements.
//!
//! @return Iterator to the new end of the range after duplicates have been removed.
template <class _Iterator, class _BinaryPredicate>
_CCCL_HOST_API _Iterator unstable_unique(_Iterator __first, _Iterator __last, _BinaryPredicate __pred)
{
  static_assert(::cuda::std::bidirectional_iterator<_Iterator>, "unstable_unique requires a bidirectional iterator");
  if (__first == __last || ::cuda::std::next(__first) == __last)
  {
    return __last;
  }

  bool __first_is_known_duplicate = false;
  for (++__first; __first != __last; ++__first)
  {
    if (!__first_is_known_duplicate)
    {
      if (!__pred(*__first, *::cuda::std::prev(__first)))
      {
        continue;
      }
    }
    _CCCL_ASSERT(__first != __last, "unstable_unique: iterator out of range");
    for (--__last;; --__last)
    {
      if (__first == __last)
      {
        return __first;
      }
      _CCCL_ASSERT(__first != __last, "unstable_unique: iterator out of range");
      if (!__pred(*__last, *::cuda::std::prev(__last)))
      {
        break;
      }
    }
    _CCCL_ASSERT(!__pred(*__first, *__last), "unstable_unique: unexpected duplicate");
    __first_is_known_duplicate = __pred(*__first, *::cuda::std::next(__first));
    *__first                   = ::cuda::std::move(*__last);
  }

  return __first;
}

//! @brief Removes duplicates from a sorted range using `operator==`.
//!
//! Equivalent to calling the predicate overload with `operator==`.
//!
//! @tparam _Iterator The type of the iterator.
//!
//! @param[in] __first Iterator to the beginning of the range.
//! @param[in] __last Iterator past the end of the range.
//!
//! @return Iterator to the new end of the range after duplicates have been removed.
template <class _Iterator>
_CCCL_HOST_API _Iterator unstable_unique(_Iterator __first, _Iterator __last)
{
  return ::cuda::experimental::unstable_unique(__first, __last, ::cuda::std::equal_to<>{});
}
} // namespace cuda::experimental

#endif // _CUDAX__UTILITY_UNSTABLE_UNIQUE
