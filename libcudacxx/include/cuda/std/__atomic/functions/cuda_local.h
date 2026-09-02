//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef __CUDA_STD___ATOMIC_FUNCTIONS_CUDA_LOCAL_H
#define __CUDA_STD___ATOMIC_FUNCTIONS_CUDA_LOCAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory/address_space.h>
#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

// This file works around a bug in CUDA in which the compiler miscompiles
// atomics to automatic storage (local memory). This bug is not fixed on any
// CUDA version yet.
//
// CUDA compilers < 12.3 also miscompile __isLocal, such that the library cannot
// detect automatic storage and error. Therefore, in CUDA < 12.3 compilers this
// uses inline PTX to bypass __isLocal.

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CUDA_COMPILATION()

_CCCL_DEVICE_API inline bool __cuda_atomic_is_local(const volatile void* __ptr)
{
#  if defined(_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE) && !defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
  return false;
#  else // ^^^ _CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE && !defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH) ^^^
        // / vvv !_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE || defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
        // vvv
  return ::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::local);
#  endif // ^^^ !_CCCL_ATOMIC_UNSAFE_AUTOMATIC_STORAGE || defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
         // ^^^
}

template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_and(_Type __atom, _Type const& __v)
{
  return __atom & __v;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_or(_Type __atom, _Type const& __v)
{
  return __atom | __v;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_xor(_Type __atom, _Type const& __v)
{
  return __atom ^ __v;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_add(_Type __atom, _Type const& __v)
{
  return __atom + __v;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_sub(_Type __atom, _Type const& __v)
{
  return __atom - __v;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_max(_Type __atom, _Type const& __v)
{
  return ::cuda::std::__cuda_atomic_less(__atom, __v) ? __v : __atom;
}
template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_fetch_local_bop_min(_Type __atom, _Type const& __v)
{
  return ::cuda::std::__cuda_atomic_less(__v, __atom) ? __v : __atom;
}

template <class _Type>
_CCCL_DEVICE_API bool
__cuda_atomic_load_weak_if_local(const _Type* __ptr, __unv<_Type>* __ret, [[maybe_unused]] size_t __size)
{
  if (!::cuda::std::__cuda_atomic_is_local(__ptr))
  {
    return false;
  }
  ::cuda::std::__atomic_assign_volatile(__ret, *__ptr);
  // Required to workaround a compiler bug, see nvbug/4064730
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type>
_CCCL_DEVICE_API bool
__cuda_atomic_store_weak_if_local(_Type* __ptr, const __unv<_Type>* __val, [[maybe_unused]] size_t __size)
{
  if (!::cuda::std::__cuda_atomic_is_local(__ptr))
  {
    return false;
  }
  ::cuda::std::__atomic_assign_volatile(__ptr, *__val);
  return true;
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_compare_exchange_weak_if_local(
  _Type* __ptr, __unv<_Type>* __expected, const __unv<_Type>* __desired, bool* __success)
{
  if (!::cuda::std::__cuda_atomic_is_local(__ptr))
  {
    return false;
  }
  using _ValueType = __unv<_Type>;
  _ValueType __old{};
  ::cuda::std::__atomic_assign_volatile(&__old, *__ptr);
  if (::cuda::std::__atomic_memcmp(&__old, __expected, sizeof(_ValueType)) == 0)
  {
    ::cuda::std::__atomic_assign_volatile(__ptr, *__desired);
    *__success = true;
  }
  else
  {
    *__expected = __old;
    *__success  = false;
  }
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_exchange_weak_if_local(_Type* __ptr, __unv<_Type>* __val, __unv<_Type>* __ret)
{
  if (!::cuda::std::__cuda_atomic_is_local(__ptr))
  {
    return false;
  }
  ::cuda::std::__atomic_assign_volatile(__ret, *__ptr);
  ::cuda::std::__atomic_assign_volatile(__ptr, *__val);
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type, class _BOp>
_CCCL_DEVICE_API bool
__cuda_atomic_fetch_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret, _BOp&& __bop)
{
  if (!::cuda::std::__cuda_atomic_is_local(__ptr))
  {
    return false;
  }
  using _ValueType = __unv<_Type>;
  _ValueType __old{};
  ::cuda::std::__atomic_assign_volatile(&__old, *__ptr);
  *__ret                     = __old;
  const _ValueType __desired = __bop(__old, __val);
  ::cuda::std::__atomic_assign_volatile(__ptr, __desired);
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_and_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_and<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_or_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_or<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_xor_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_xor<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_add_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_add<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_sub_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_sub<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_max_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_max<_ValueType>);
}

template <class _Type>
_CCCL_DEVICE_API bool __cuda_atomic_fetch_min_weak_if_local(_Type* __ptr, __unv<_Type> __val, __unv<_Type>* __ret)
{
  using _ValueType = __unv<_Type>;
  return ::cuda::std::__cuda_atomic_fetch_weak_if_local(
    __ptr, __val, __ret, ::cuda::std::__cuda_atomic_fetch_local_bop_min<_ValueType>);
}

#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_FUNCTIONS_CUDA_LOCAL_H
