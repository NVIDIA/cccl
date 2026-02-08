//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_POINTER_IN_RANGE_H
#define _CUDA___MEMORY_POINTER_IN_RANGE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>
#if _CCCL_HOST_COMPILATION()
#  include <functional>
#endif // _CCCL_HOST_COMPILATION()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// Pointers comparison <, <=, >=, > is undefined behavior in C++ (https://eel.is/c++draft/expr.rel#4) when pointers
// don't belong to the same object or array.
// - Even when a platform guarantees flat address space, the compiler can leverage UB for optimization purposes.
// - However, the compiler treats ::std::less<> other functional operators in a special way, ensuring a total ordering.
// - For device code, we can convert pointers to uintptr_t and compare them.
//
// References:
// - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2024/p3234r0.html
// - https://www.open-std.org/jtc1/sc22/wg21/docs/papers/2023/p2865r2.pdf
// - https://www.boost.org/doc/libs/develop/libs/core/doc/html/core/pointer_in_range.html
// - https://pvs-studio.com/en/blog/posts/cpp/1199/
// - https://releases.llvm.org/20.1.0/tools/clang/docs/ReleaseNotes.html#resolutions-to-c-defect-reports

#if _CCCL_HOST_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API bool __ptr_in_range_host(_Tp* __ptr, _Tp* __start, _Tp* __end) noexcept
{
  _CCCL_ASSERT(::std::greater_equal<>{}(__end, __start), "__ptr_in_range_host: __end must be greater than __start");
  return ::std::greater_equal<>{}(__ptr, __start) && ::std::less<>{}(__ptr, __end);
}

#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_DEVICE_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API bool __ptr_in_range_device(_Tp* __ptr, _Tp* __start, _Tp* __end) noexcept
{
  using uintptr_t  = ::cuda::std::uintptr_t;
  auto __end_ptr   = reinterpret_cast<uintptr_t>(__end);
  auto __start_ptr = reinterpret_cast<uintptr_t>(__start);
  auto __ptr_ptr   = reinterpret_cast<uintptr_t>(__ptr);
  _CCCL_ASSERT(__end_ptr >= __start_ptr, "__ptr_in_range_device: __end must be greater than __start");
  return __ptr_ptr >= __start_ptr && __ptr_ptr < __end_ptr;
}

#endif // _CCCL_DEVICE_COMPILATION()

template <typename _Tp>
[[nodiscard]] _CCCL_API constexpr bool ptr_in_range(_Tp* __ptr, _Tp* __start, _Tp* __end) noexcept
{
  _CCCL_IF_CONSTEVAL_DEFAULT
  {
    _CCCL_ASSERT(__end >= __start, "ptr_in_range: __end must be greater than __start");
    return __ptr >= __start && __ptr < __end; // UB is not possible in a constant expression
  }
  else
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      (return ::cuda::__ptr_in_range_host(__ptr, __start, __end);),
                      (return ::cuda::__ptr_in_range_device(__ptr, __start, __end);));
  }
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MEMORY_POINTER_IN_RANGE_H
