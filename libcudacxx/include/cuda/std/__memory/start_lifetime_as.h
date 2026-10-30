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

#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/is_implicit_lifetime.h>
#include <cuda/std/cstddef> // size_t

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// Creating an array of bytes in the storage implicitly creates objects of implicit-lifetime types in it, with the
// bytes of the storage as their value representation. This is the reference implementation of P2590R2.
template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* __start_lifetime_as_impl(void* __ptr, size_t __count) noexcept
{
#if defined(_CCCL_BUILTIN_IS_IMPLICIT_LIFETIME)
  static_assert(is_implicit_lifetime_v<_Tp>, "start_lifetime_as requires an implicit-lifetime type");
#endif // _CCCL_BUILTIN_IS_IMPLICIT_LIFETIME
  static_assert(sizeof(_Tp) > 0, "start_lifetime_as requires a complete type");
  if (__count == 0)
  {
    return static_cast<_Tp*>(__ptr);
  }
  const auto __bytes = ::new (__ptr) unsigned char[__count * sizeof(_Tp)];
  return reinterpret_cast<_Tp*>(__bytes);
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* start_lifetime_as(void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(__ptr, 1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const _Tp* start_lifetime_as(const void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), 1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API volatile _Tp* start_lifetime_as(volatile void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), 1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const volatile _Tp* start_lifetime_as(const volatile void* __ptr) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), 1);
}

template <class _Tp>
[[nodiscard]] _CCCL_API _Tp* start_lifetime_as_array(void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(__ptr, __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const _Tp* start_lifetime_as_array(const void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API volatile _Tp* start_lifetime_as_array(volatile void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

template <class _Tp>
[[nodiscard]] _CCCL_API const volatile _Tp* start_lifetime_as_array(const volatile void* __ptr, size_t __count) noexcept
{
  return ::cuda::std::__start_lifetime_as_impl<_Tp>(const_cast<void*>(__ptr), __count);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_START_LIFETIME_AS_H
