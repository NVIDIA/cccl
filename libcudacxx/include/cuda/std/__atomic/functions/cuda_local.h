//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef __LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_LOCAL_H
#define __LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_LOCAL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/cstdint>

// This file works around a bug in CUDA in which the compiler miscompiles
// atomics to automatic storage (local memory). This bug is not fixed on any
// CUDA version yet.
//
// CUDA compilers < 12.3 also miscompile __isLocal, such that the library cannot
// detect automatic storage and error. Therefore, in CUDA < 12.3 compilers this
// uses inline PTX to bypass __isLocal.
_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_DEVICE inline bool __cuda_is_local(const void* __ptr)
{
#if defined(_LIBCUDACXX_CUDACC_BELOW_12_3)
  int __tmp = 0;
  asm("{\n\t"
      "  .reg .pred p;\n\t"
      "  isspacep.local p, %1;\n\t"
      "  @p mov.s32 %0, 1;\n\t"
      "}\n\t"
      : "=r"(__tmp)
      : "l"(__ptr));
  return __tmp == 1;
#else
  return __isLocal(__ptr);
#endif
}

template <class _Type>
_CCCL_DEVICE bool __cuda_load_weak_if_local(const volatile _Type* __ptr, _Type* __ret)
{
  if (!__cuda_is_local((const void*) __ptr))
  {
    return false;
  }
  memcpy((void*) __ret, (void const*) __ptr, sizeof(_Type));
  // Required to workaround a compiler bug, see nvbug/4064730
  __nanosleep(0);
  return true;
}

template <class _Type>
_CCCL_DEVICE bool __cuda_store_weak_if_local(volatile _Type* __ptr, _Type __val)
{
  if (!__cuda_is_local((const void*) __ptr))
  {
    return false;
  }
  memcpy((void*) __ptr, (void const*) &__val, sizeof(_Type));
  return true;
}

template <class _Type>
_CCCL_DEVICE bool
__cuda_compare_exchange_weak_if_local(volatile _Type* __ptr, _Type* __expected, const _Type* __desired, bool* __success)
{
  if (!__cuda_is_local((const void*) __ptr))
  {
    return false;
  }
  if (0 == __atomic_memcmp((const void*) __ptr, (const void*) __expected, sizeof(_Type)))
  {
    memcpy((void*) __ptr, (void const*) __desired, sizeof(_Type));
    *__success = true;
  }
  else
  {
    memcpy((void*) __expected, (void const*) __ptr, sizeof(_Type));
    *__success = false;
  }
  __nanosleep(0);
  return true;
}

template <class _Type>
_CCCL_DEVICE bool __cuda_exchange_weak_if_local(volatile _Type* __ptr, _Type* __val, _Type* __ret)
{
  if (!__cuda_is_local((const void*) __ptr))
  {
    return false;
  }
  memcpy((void*) __ret, (void const*) __ptr, sizeof(_Type));
  memcpy((void*) __ptr, (void const*) __val, sizeof(_Type));
  __nanosleep(0);
  return true;
}

template <class _Type, class _BOp>
_CCCL_DEVICE bool __cuda_fetch_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret, _BOp&& __bop)
{
  if (!__cuda_is_local((const void*) __ptr))
  {
    return false;
  }
  memcpy((void*) __ret, (void const*) __ptr, sizeof(_Type));
  __bop(*__ptr, __val);
  __nanosleep(0);
  return true;
}

template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_and(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom & __v;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_or(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom | __v;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_xor(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom ^ __v;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_add(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom + __v;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_sub(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom - __v;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_max(volatile _Type& __atom, _Type const& __v)
{
  __atom = __atom < __v ? __v : __atom;
}
template <class _Type>
_CCCL_DEVICE void __cuda_fetch_local_bop_min(volatile _Type& __atom, _Type const& __v)
{
  __atom = __v < __atom ? __v : __atom;
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_and_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_and<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_or_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_or<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_xor_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_xor<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_add_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_add<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_sub_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_sub<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_max_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_max<_Type>);
}

template <class _Type>
_CCCL_DEVICE bool __cuda_fetch_min_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret)
{
  return __cuda_fetch_weak_if_local(__ptr, __val, __ret, __cuda_fetch_local_bop_min<_Type>);
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_LOCAL_H
