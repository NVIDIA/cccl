//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RANGES_OVERLAP_H
#define _CUDA___MEMORY_RANGES_OVERLAP_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

#if _CCCL_HOST_COMPILATION()
#  include <functional>
#endif // _CCCL_HOST_COMPILATION()

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_DEVICE_COMPILATION()

[[nodiscard]]
_CCCL_DEVICE_API inline bool __ptr_ranges_overlap_device(
  const void* __lhs_begin, const void* __lhs_end, const void* __rhs_begin, const void* __rhs_end) noexcept
{
  using uintptr_t            = ::cuda::std::uintptr_t;
  const auto __lhs_start_ptr = reinterpret_cast<uintptr_t>(__lhs_begin);
  const auto __lhs_end_ptr   = reinterpret_cast<uintptr_t>(__lhs_end);
  const auto __rhs_start_ptr = reinterpret_cast<uintptr_t>(__rhs_begin);
  const auto __rhs_end_ptr   = reinterpret_cast<uintptr_t>(__rhs_end);
  _CCCL_ASSERT(__lhs_start_ptr <= __lhs_end_ptr, "lhs range is invalid");
  _CCCL_ASSERT(__rhs_start_ptr <= __rhs_end_ptr, "rhs range is invalid");
  return __lhs_start_ptr < __rhs_end_ptr && __rhs_start_ptr < __lhs_end_ptr;
}

#else // ^^^^ _CCCL_DEVICE_COMPILATION() ^^^^ / vvvv _CCCL_HOST_COMPILATION() vvvv

template <typename _Tp>
[[nodiscard]]
_CCCL_HOST_API bool
__ptr_ranges_overlap_host(_Tp* __lhs_begin, _Tp* __lhs_end, _Tp* __rhs_begin, _Tp* __rhs_end) noexcept
{
  _CCCL_ASSERT(::std::less_equal<>{}(__lhs_begin, __lhs_end), "lhs range is invalid");
  _CCCL_ASSERT(::std::less_equal<>{}(__rhs_begin, __rhs_end), "rhs range is invalid");
  return ::std::less<>{}(__lhs_begin, __rhs_end) && ::std::less<>{}(__rhs_begin, __lhs_end);
}

#endif // ^^^^ _CCCL_HOST_COMPILATION() ^^^^

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(::cuda::std::forward_iterator<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool
ranges_overlap(_Tp __lhs_begin, _Tp __lhs_end, _Tp __rhs_begin, _Tp __rhs_end) noexcept
{
  if constexpr (::cuda::std::contiguous_iterator<_Tp>)
  {
    _CCCL_IF_CONSTEVAL_DEFAULT
    {
      // UB is not possible in a constant expression
      _CCCL_ASSERT(__lhs_begin <= __lhs_end, "lhs range is invalid");
      _CCCL_ASSERT(__rhs_begin <= __rhs_end, "rhs range is invalid");
      return __lhs_begin < __rhs_end && __rhs_begin < __lhs_end;
    }
    else
    {
      const auto __ptr_lhs_begin = ::cuda::std::to_address(__lhs_begin);
      const auto __ptr_lhs_end   = ::cuda::std::to_address(__lhs_end);
      const auto __ptr_rhs_begin = ::cuda::std::to_address(__rhs_begin);
      const auto __ptr_rhs_end   = ::cuda::std::to_address(__rhs_end);
      NV_IF_ELSE_TARGET(
        NV_IS_HOST,
        (return ::cuda::__ptr_ranges_overlap_host(__ptr_lhs_begin, __ptr_lhs_end, __ptr_rhs_begin, __ptr_rhs_end);),
        (return ::cuda::__ptr_ranges_overlap_device(__ptr_lhs_begin, __ptr_lhs_end, __ptr_rhs_begin, __ptr_rhs_end);));
    }
  }
  else if constexpr (::cuda::std::random_access_iterator<_Tp>)
  {
    _CCCL_ASSERT(__lhs_begin <= __lhs_end, "lhs range is invalid");
    _CCCL_ASSERT(__rhs_begin <= __rhs_end, "rhs range is invalid");
    return __lhs_begin < __rhs_end && __rhs_begin < __lhs_end;
  }
  else
  {
    // For forward iterators: if two ranges [A,B) and [C,D) overlap from the same sequence,
    // then either C is in [A,B) or A is in [C,D). We check both conditions.
    for (auto __lhs_it = __lhs_begin; __lhs_it != __lhs_end; ++__lhs_it)
    {
      if (__lhs_it == __rhs_begin)
      {
        return true;
      }
    }
    for (auto __rhs_it = __rhs_begin; __rhs_it != __rhs_end; ++__rhs_it)
    {
      if (__rhs_it == __lhs_begin)
      {
        return true;
      }
    }
    return false;
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_RANGES_OVERLAP_H
