//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/cstring>

// This file works around a bug in CUDA in which the compiler miscompiles
// atomics to automatic storage (local memory). This bug is not fixed on any
// CUDA version yet.
//
// CUDA compilers < 12.3 also miscompile __isLocal, such that the library cannot
// detect automatic storage and error. Therefore, in CUDA < 12.3 compilers this
// uses inline PTX to bypass __isLocal.
_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_HAS_CUDA_COMPILER()

_CCCL_DEVICE inline bool __cuda_is_local(const volatile void* __ptr)
{
#  if defined(_LIBCUDACXX_ATOMIC_UNSAFE_AUTOMATIC_STORAGE) && !defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
  return false;
// Only NVCC+NVRTC define __isLocal, so drop to PTX
// Some tests require using the inline PTX path to ensure it is bug-free
#  elif _CCCL_CUDACC_BELOW(12, 3) || _CCCL_CUDA_COMPILER(NVHPC) || defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
  int __tmp = 0;
  asm("{\n\t"
      "  .reg .pred p;\n\t"
      "  isspacep.local p, %1;\n\t"
      "  selp.u32 %0, 1, 0, p;\n\t"
      "}\n\t"
      : "=r"(__tmp)
      : "l"(const_cast<const void*>(__ptr)));
  return __tmp == 1;
#  else // ^^^ _CCCL_CUDACC_BELOW(12, 3) || _CCCL_CUDA_COMPILER(NVHPC) ||
        // defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH) ^^^ / vvv other compiler vvv
  return __isLocal(const_cast<const void*>(__ptr));
#  endif // _CCCL_CUDACC_AT_LEAST(12, 3) && !_CCCL_CUDA_COMPILER(NVHPC) &&
         // !defined(_LIBCUDACXX_FORCE_PTX_AUTOMATIC_STORAGE_PATH)
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

_CCCL_DEVICE inline bool __cuda_load_weak_if_local(const volatile void* __ptr, void* __ret, size_t __size)
{
  if (!__cuda_is_local(__ptr))
  {
    return false;
  }
  _CUDA_VSTD::memcpy(__ret, const_cast<const void*>(__ptr), __size);
  // Required to workaround a compiler bug, see nvbug/4064730
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

_CCCL_DEVICE inline bool __cuda_store_weak_if_local(volatile void* __ptr, const void* __val, size_t __size)
{
  if (!__cuda_is_local(__ptr))
  {
    return false;
  }
  _CUDA_VSTD::memcpy(const_cast<void*>(__ptr), __val, __size);
  return true;
}

template <class _Type>
_CCCL_DEVICE bool
__cuda_compare_exchange_weak_if_local(volatile _Type* __ptr, _Type* __expected, const _Type* __desired, bool* __success)
{
  if (!__cuda_is_local(__ptr))
  {
    return false;
  }
  if (__atomic_memcmp(const_cast<const _Type*>(__ptr), const_cast<const _Type*>(__expected), sizeof(_Type)) == 0)
  {
    _CUDA_VSTD::memcpy(const_cast<_Type*>(__ptr), const_cast<_Type const*>(__desired), sizeof(_Type));
    *__success = true;
  }
  else
  {
    _CUDA_VSTD::memcpy(const_cast<_Type*>(__expected), const_cast<_Type const*>(__ptr), sizeof(_Type));
    *__success = false;
  }
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type>
_CCCL_DEVICE bool __cuda_exchange_weak_if_local(volatile _Type* __ptr, _Type* __val, _Type* __ret)
{
  if (!__cuda_is_local(__ptr))
  {
    return false;
  }
  _CUDA_VSTD::memcpy(const_cast<_Type*>(__ret), const_cast<const _Type*>(__ptr), sizeof(_Type));
  _CUDA_VSTD::memcpy(const_cast<_Type*>(__ptr), const_cast<const _Type*>(__val), sizeof(_Type));
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
}

template <class _Type, class _BOp>
_CCCL_DEVICE bool __cuda_fetch_weak_if_local(volatile _Type* __ptr, _Type __val, _Type* __ret, _BOp&& __bop)
{
  if (!__cuda_is_local(__ptr))
  {
    return false;
  }
  _CUDA_VSTD::memcpy(const_cast<_Type*>(__ret), const_cast<const _Type*>(__ptr), sizeof(_Type));
  __bop(*__ptr, __val);
  NV_IF_TARGET(NV_PROVIDES_SM_70, (__nanosleep(0);))
  return true;
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

#endif // _CCCL_HAS_CUDA_COMPILER()

_LIBCUDACXX_END_NAMESPACE_STD

#endif // __LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_LOCAL_H
