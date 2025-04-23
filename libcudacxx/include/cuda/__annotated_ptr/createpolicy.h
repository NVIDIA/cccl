//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ANNOTATED_PTR_CREATEPOLICY
#define _CUDA___ANNOTATED_PTR_CREATEPOLICY

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

enum class _L2_Policy
{
  __evict_last,
  __evict_normal,
  __evict_first,
  __evict_unchanged
};

/***********************************************************************************************************************
 * PTX MAPPING
 **********************************************************************************************************************/

template <typename = void>
[[nodiscard]] _CCCL_CONST _CCCL_HIDE_FROM_ABI _CCCL_DEVICE uint64_t __createpolicy_range_ptx(
  size_t __gmem_ptr, _L2_Policy __primary, _L2_Policy __secondary, uint32_t __primary_size, uint32_t __total_size)
{
  uint64_t __policy;
  if (__secondary == _L2_Policy::__evict_unchanged)
  {
    if (__primary == _L2_Policy::__evict_last)
    {
      asm("createpolicy.range.global.L2::evict_last.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_normal)
    {
      asm("createpolicy.range.global.L2::evict_normal.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_first)
    {
      asm("createpolicy.range.global.L2::evict_first.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_unchanged)
    {
      asm("createpolicy.range.global.L2::evict_unchanged.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
  }
  else // __secondary == __evict_first
  {
    if (__primary == _L2_Policy::__evict_last)
    {
      asm("createpolicy.range.global.L2::evict_last.L2::evict_first.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_normal)
    {
      asm("createpolicy.range.global.L2::evict_normal.L2::evict_first.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_first)
    {
      asm("createpolicy.range.global.L2::evict_first.L2::evict_first.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
    else if (__primary == _L2_Policy::__evict_unchanged)
    {
      asm("createpolicy.range.global.L2::evict_unchanged.L2::evict_first.b64 %0, [%1], %2, %3;"
          : "=l"(__policy)
          : "l"(__gmem_ptr), "r"(__primary_size), "r"(__total_size));
    }
  }
  return __policy;
}

template <typename = void>
[[nodiscard]] _CCCL_CONST _CCCL_HIDE_FROM_ABI _CCCL_DEVICE uint64_t
__createpolicy_fraction_ptx(_L2_Policy __primary, _L2_Policy __secondary, float __fraction)
{
  uint64_t __policy;
  if (__secondary == _L2_Policy::__evict_unchanged)
  {
    if (__primary == _L2_Policy::__evict_last)
    {
      asm("createpolicy.fractional.L2::evict_last.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_normal)
    {
      asm("createpolicy.fractional.L2::evict_normal.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_first)
    {
      asm("createpolicy.fractional.L2::evict_first.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_unchanged)
    {
      asm("createpolicy.fractional.L2::evict_unchanged.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
  }
  else // __secondary == __evict_first
  {
    if (__primary == _L2_Policy::__evict_last)
    {
      asm("createpolicy.fractional.L2::evict_last.L2::evict_first.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_normal)
    {
      asm("createpolicy.fractional.L2::evict_normal.L2::evict_first.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_first)
    {
      asm("createpolicy.fractional.L2::evict_first.L2::evict_first.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
    else if (__primary == _L2_Policy::__evict_unchanged)
    {
      asm("createpolicy.fractional.L2::evict_unchanged.L2::evict_first.b64 %0, %1;" : "=l"(__policy) : "f"(__fraction));
    }
  }
  return __policy;
}

/***********************************************************************************************************************
 * C++ API
 **********************************************************************************************************************/

extern "C" _CCCL_DEVICE void __createpolicy_is_not_supported_before_SM_80();

template <typename T = void>
[[nodiscard]] _CCCL_CONST _CCCL_HIDE_FROM_ABI _CCCL_DEVICE uint64_t __createpolicy_range(
  const void* __ptr,
  _L2_Policy __primary,
  uint32_t __primary_size,
  uint32_t __total_size,
  _L2_Policy __secondary = _L2_Policy::__evict_unchanged)
{
  _CCCL_ASSERT(__isGlobal(__ptr), "ptr must be global");
  _CCCL_ASSERT(__primary_size <= __total_size, "primary_size must be less than or equal to total_size");
  _CCCL_ASSERT(__secondary == _L2_Policy::__evict_first || __secondary == _L2_Policy::__evict_unchanged,
               "secondary policy must be evict_first or evict_unchanged");
  auto __gmem_ptr = __cvta_generic_to_global(__ptr);
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (return ::cuda::__createpolicy_range_ptx(__gmem_ptr, __primary, __secondary, __primary_size, __total_size);),
    (__createpolicy_is_not_supported_before_SM_80(); return 0;))
}

template <typename T = void>
[[nodiscard]] _CCCL_CONST _CCCL_HIDE_FROM_ABI _CCCL_DEVICE uint64_t __createpolicy_fraction(
  _L2_Policy __primary, float __fraction = 1.0f, _L2_Policy __secondary = _L2_Policy::__evict_unchanged)
{
  _CCCL_ASSERT(__fraction > 0.0f && __fraction <= 1.0f, "fraction must be between 0.0f and 1.0f");
  _CCCL_ASSERT(__secondary == _L2_Policy::__evict_first || __secondary == _L2_Policy::__evict_unchanged,
               "secondary policy must be evict_first or evict_unchanged");
  NV_IF_ELSE_TARGET(NV_PROVIDES_SM_80,
                    (return ::cuda::__createpolicy_fraction_ptx(__primary, __secondary, __fraction);),
                    (__createpolicy_is_not_supported_before_SM_80(); return 0;))
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___ANNOTATED_PTR_CREATEPOLICY
