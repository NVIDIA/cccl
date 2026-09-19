//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_TYPES_COMMON_H
#define _CUDA_STD___ATOMIC_TYPES_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
_CCCL_HOST_DEVICE_API bool __cuda_atomic_less(_Tp __lhs, _Tp __rhs)
{
  if constexpr (__is_extended_floating_point_v<_Tp> && sizeof(_Tp) == 2)
  {
#if _CCCL_HAS_CTK() && _CCCL_CTK_BELOW(12, 2)
    // Before CTK 12.2, __hlt is device-only and its bfloat16 overload is unavailable before SM80.
#  if _CCCL_HAS_NVBF16()
    if constexpr (is_same_v<_Tp, __nv_bfloat16>)
    {
      // Intentionally unqualified to avoid including <cuda_bf16.h>.
      NV_IF_ELSE_TARGET(
        NV_PROVIDES_SM_80, (return __hlt(__lhs, __rhs);), (return __bfloat162float(__lhs) < __bfloat162float(__rhs);))
    }
    else
#  endif // _CCCL_HAS_NVBF16()
    {
      // Intentionally unqualified to avoid including <cuda_fp16.h>.
      NV_IF_ELSE_TARGET(NV_IS_DEVICE, (return __hlt(__lhs, __rhs);), (return __half2float(__lhs) < __half2float(__rhs);))
    }
#else // ^^^ CTK below 12.2 ^^^ / vvv CTK 12.2 or newer vvv
    // Intentionally unqualified to avoid including <cuda_fp16.h> and <cuda_bf16.h>.
    return __hlt(__lhs, __rhs);
#endif // CTK 12.2 or newer
  }
  else
  {
    return __lhs < __rhs;
  }
}

enum class __atomic_tag
{
  __atomic_base_tag,
  __atomic_locked_tag,
  __atomic_small_tag,
};

// Helpers to SFINAE on the tag inside the storage object
template <typename _Sto>
using __atomic_storage_is_base = enable_if_t<__atomic_tag::__atomic_base_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_locked = enable_if_t<__atomic_tag::__atomic_locked_tag == remove_cvref_t<_Sto>::__tag, int>;
template <typename _Sto>
using __atomic_storage_is_small = enable_if_t<__atomic_tag::__atomic_small_tag == remove_cvref_t<_Sto>::__tag, int>;

template <typename _Tp>
using __atomic_underlying_t = typename _Tp::__underlying_t;
template <typename _Tp>
using __atomic_underlying_remove_cv_t = remove_cv_t<typename _Tp::__underlying_t>;

// [atomics.types.generic]p1 guarantees _Tp is trivially copyable. Because
// the default operator= in an object is not volatile, a byte-by-byte copy
// is required.
template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE_API enable_if_t<is_assignable_v<_Tp&, _Tv>> __atomic_assign_volatile(_Tp* __a_value, _Tv const& __val)
{
  *__a_value = __val;
}

template <typename _Tp, typename _Tv>
_CCCL_HOST_DEVICE_API enable_if_t<is_assignable_v<_Tp&, _Tv>>
__atomic_assign_volatile(_Tp volatile* __a_value, _Tv volatile const& __val)
{
  volatile char* __to         = reinterpret_cast<volatile char*>(__a_value);
  volatile char* __end        = __to + sizeof(_Tp);
  volatile const char* __from = reinterpret_cast<volatile const char*>(&__val);
  while (__to != __end)
  {
    *__to++ = *__from++;
  }
}

_CCCL_HOST_DEVICE_API inline int __atomic_memcmp(void const* __lhs, void const* __rhs, size_t __count)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (unsigned char const* __lhs_c; unsigned char const* __rhs_c;
     // NVCC recommended laundering through inline asm to compare padding bytes.
     asm("mov.b64 %0, %2;\n mov.b64 %1, %3;" : "=l"(__lhs_c), "=l"(__rhs_c) : "l"(__lhs), "l"(__rhs));
     while (__count--) {
       auto const __lhs_v = *__lhs_c++;
       auto const __rhs_v = *__rhs_c++;
       if (__lhs_v < __rhs_v)
       {
         return -1;
       }
       if (__lhs_v > __rhs_v)
       {
         return 1;
       }
     } return 0;),
    NV_IS_HOST,
    (return ::cuda::std::memcmp(__lhs, __rhs, __count);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_TYPES_COMMON_H
