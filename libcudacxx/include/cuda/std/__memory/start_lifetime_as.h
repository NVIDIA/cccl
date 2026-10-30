// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_START_LIFETIME_AS_H
#define _CUDA_STD___MEMORY_START_LIFETIME_AS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_implicit_lifetime.h>
#include <cuda/std/cstddef> // size_t

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// start_lifetime_as implicitly creates objects whose object representation is the prior content of the storage, so all
// accesses through the returned pointer must observe the bytes as they are at the call and no access may be reordered
// across it. This cannot be expressed in the library, so we make the compiler forget what it knows about the storage
// using an empty asm statement that passes the pointer through and clobbers memory. libstdc++ does the same, but with
// memory operands for the storage instead of a memory clobber. nvcc does not support memory operands in device code.
#if _CCCL_HOST_COMPILATION()
[[nodiscard]] _CCCL_HOST_API inline void* __start_lifetime_as_barrier_host(void* __ptr) noexcept
{
#  if !_CCCL_COMPILER(MSVC) // MSVC does not perform type-based alias analysis and has no inline asm on x64
  asm volatile("" : "+r"(__ptr) : : "memory");
#  endif // !_CCCL_COMPILER(MSVC)
  return __ptr;
}
#endif // _CCCL_HOST_COMPILATION()

#if _CCCL_CUDA_COMPILATION()
[[nodiscard]] _CCCL_DEVICE_API inline void* __start_lifetime_as_barrier_device(void* __ptr) noexcept
{
  asm volatile("" : "+l"(__ptr) : : "memory");
  return __ptr;
}
#endif // _CCCL_CUDA_COMPILATION()

[[nodiscard]] _CCCL_API inline void* __start_lifetime_as_barrier(void* __ptr) noexcept
{
  NV_IF_ELSE_TARGET(NV_IS_HOST,
                    (return ::cuda::std::__start_lifetime_as_barrier_host(__ptr);),
                    (return ::cuda::std::__start_lifetime_as_barrier_device(__ptr);))
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* __start_lifetime_as_impl(void* __ptr) noexcept
{
#if defined(_CCCL_BUILTIN_IS_IMPLICIT_LIFETIME)
  static_assert(is_implicit_lifetime_v<_Tp>, "start_lifetime_as requires an implicit-lifetime type");
#endif // _CCCL_BUILTIN_IS_IMPLICIT_LIFETIME
  static_assert(sizeof(_Tp) > 0, "start_lifetime_as requires a complete type");
  return static_cast<_Tp*>(::cuda::std::__start_lifetime_as_barrier(__ptr));
}

// Arrays are always implicit-lifetime types, so only the element type's completeness is required
template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* __start_lifetime_as_array_impl(void* __ptr, size_t __count) noexcept
{
  static_assert(sizeof(_Tp) > 0, "start_lifetime_as_array requires a complete type");
  if (__count == 0)
  {
    return static_cast<_Tp*>(__ptr);
  }
  return static_cast<_Tp*>(::cuda::std::__start_lifetime_as_barrier(__ptr));
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* start_lifetime_as(void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(__ptr);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const _Tp* start_lifetime_as(const void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr));
}

template <class _Tp>
[[nodiscard]] _CCCL_API volatile _Tp* start_lifetime_as(volatile void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr));
}

template <class _Tp>
[[nodiscard]] _CCCL_API const volatile _Tp* start_lifetime_as(const volatile void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr));
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* start_lifetime_as_array(void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_array_impl<_Tp>(__ptr, __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const _Tp* start_lifetime_as_array(const void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_array_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API volatile _Tp* start_lifetime_as_array(volatile void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_array_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const volatile _Tp* start_lifetime_as_array(const volatile void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_array_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_START_LIFETIME_AS_H
