// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CMATH_TRAITS_H
#define _CUDA_STD___CMATH_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cmath/isnan.h>
#include <cuda/std/__floating_point/format.h>
#include <cuda/std/__floating_point/mask.h>
#include <cuda/std/__floating_point/properties.h>
#include <cuda/std/__floating_point/storage.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__host_stdlib/math.h> // IWYU pragma: keep
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_signed.h>
#include <cuda/std/__type_traits/promote.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class _Cmp>
[[nodiscard]] _CCCL_API constexpr bool __fp_compare_generic(_Tp __lhs, _Tp __rhs, _Cmp __cmp) noexcept
{
  constexpr auto __fmt = __fp_format_of_v<_Tp>;

  if constexpr (__fp_has_nan_v<__fmt> || __fp_has_nans_v<__fmt>)
  {
    if (::cuda::std::isnan(__lhs) || ::cuda::std::isnan(__rhs))
    {
      return false;
    }
  }

  const auto __lhs_storage = ::cuda::std::__fp_get_storage(__lhs);
  const auto __rhs_storage = ::cuda::std::__fp_get_storage(__rhs);

  if constexpr (__fp_is_signed_v<__fmt>)
  {
    using _SignedStorage = ::cuda::std::make_signed_t<__fp_storage_t<__fmt>>;

    constexpr auto __sign_mask = __fp_sign_mask_v<__fmt>;
    const auto __lhs_magnitude = static_cast<_SignedStorage>(__lhs_storage & (__sign_mask - 1));
    const auto __rhs_magnitude = static_cast<_SignedStorage>(__rhs_storage & (__sign_mask - 1));

    // Use the format's sign bit, which need not be the storage's top bit, and map both signed zeros to zero.
    const auto __lhs_ordered = (__lhs_storage & __sign_mask) ? -__lhs_magnitude : __lhs_magnitude;
    const auto __rhs_ordered = (__rhs_storage & __sign_mask) ? -__rhs_magnitude : __rhs_magnitude;
    return __cmp(__lhs_ordered, __rhs_ordered);
  }
  else
  {
    return __cmp(__lhs_storage, __rhs_storage);
  }
}

// isgreater

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_fp_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isgreater(_Tp __x, _Tp __y) noexcept
{
  if constexpr (__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_same_v<_Tp, ::__half>
#  if _CCCL_HAS_NVBF16()
                    || is_same_v<_Tp, ::__nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16()
      )
      {
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
                       {
                         return false;
                       }
                       return ::__hgt(__x, __y);
                     }))
      }
#endif // _CCCL_HAS_NVFP16()
    }

    using _Cmp = ::cuda::std::greater<>;
    return ::cuda::std::__fp_compare_generic(__x, __y, _Cmp{});
  }
  else
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __fmt = __fp_format_of_v<_Tp>;

      if constexpr (__fmt == __fp_format::__binary64)
      {
        _CCCL_DISPATCH_SLOW_FP64_TARGET(return ::cuda::std::__fp_compare_generic(__x, __y, ::cuda::std::greater<>{});)
      }

      NV_IF_TARGET(NV_IS_HOST, (return ::isgreater(__x, __y);))
    }

    if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
    {
      return false;
    }
    return __x > __y;
  }
}

_CCCL_TEMPLATE(class _T1, class _T2, class _Promoted = __promote_t<_T1, _T2>)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool isgreater(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::isgreater((_Promoted) __x, (_Promoted) __y);
}

// isgreaterequal

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_fp_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isgreaterequal(_Tp __x, _Tp __y) noexcept
{
  if constexpr (__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_same_v<_Tp, ::__half>
#  if _CCCL_HAS_NVBF16()
                    || is_same_v<_Tp, ::__nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16()
      )
      {
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
                       {
                         return false;
                       }
                       return ::__hge(__x, __y);
                     }))
      }
#endif // _CCCL_HAS_NVFP16()
    }

    using _Cmp = ::cuda::std::greater_equal<>;
    return ::cuda::std::__fp_compare_generic(__x, __y, _Cmp{});
  }
  else
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __fmt = __fp_format_of_v<_Tp>;

      if constexpr (__fmt == __fp_format::__binary64)
      {
        _CCCL_DISPATCH_SLOW_FP64_TARGET(
          return ::cuda::std::__fp_compare_generic(__x, __y, ::cuda::std::greater_equal<>{});)
      }

      NV_IF_TARGET(NV_IS_HOST, (return ::isgreaterequal(__x, __y);))
    }

    if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
    {
      return false;
    }
    return __x >= __y;
  }
}

_CCCL_TEMPLATE(class _T1, class _T2, class _Promoted = __promote_t<_T1, _T2>)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool isgreaterequal(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::isgreaterequal((_Promoted) __x, (_Promoted) __y);
}

// isless

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_fp_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool isless(_Tp __x, _Tp __y) noexcept
{
  if constexpr (__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_same_v<_Tp, ::__half>
#  if _CCCL_HAS_NVBF16()
                    || is_same_v<_Tp, ::__nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16()
      )
      {
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
                       {
                         return false;
                       }
                       return ::__hlt(__x, __y);
                     }))
      }
#endif // _CCCL_HAS_NVFP16()
    }

    using _Cmp = ::cuda::std::less<>;
    return ::cuda::std::__fp_compare_generic(__x, __y, _Cmp{});
  }
  else
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __fmt = __fp_format_of_v<_Tp>;

      if constexpr (__fmt == __fp_format::__binary64)
      {
        _CCCL_DISPATCH_SLOW_FP64_TARGET(return ::cuda::std::__fp_compare_generic(__x, __y, ::cuda::std::less<>{});)
      }

      NV_IF_TARGET(NV_IS_HOST, (return ::isless(__x, __y);))
    }

    if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
    {
      return false;
    }
    return __x < __y;
  }
}

_CCCL_TEMPLATE(class _T1, class _T2, class _Promoted = __promote_t<_T1, _T2>)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool isless(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::isless((_Promoted) __x, (_Promoted) __y);
}

// islessequal

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_fp_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool islessequal(_Tp __x, _Tp __y) noexcept
{
  if constexpr (__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_same_v<_Tp, ::__half>
#  if _CCCL_HAS_NVBF16()
                    || is_same_v<_Tp, ::__nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16()
      )
      {
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
                       {
                         return false;
                       }
                       return ::__hle(__x, __y);
                     }))
      }
#endif // _CCCL_HAS_NVFP16()
    }

    using _Cmp = ::cuda::std::less_equal<>;
    return ::cuda::std::__fp_compare_generic(__x, __y, _Cmp{});
  }
  else
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __fmt = __fp_format_of_v<_Tp>;

      if constexpr (__fmt == __fp_format::__binary64)
      {
        _CCCL_DISPATCH_SLOW_FP64_TARGET(
          return ::cuda::std::__fp_compare_generic(__x, __y, ::cuda::std::less_equal<>{});)
      }

      NV_IF_TARGET(NV_IS_HOST, (return ::islessequal(__x, __y);))
    }

    if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
    {
      return false;
    }
    return __x <= __y;
  }
}

_CCCL_TEMPLATE(class _T1, class _T2, class _Promoted = __promote_t<_T1, _T2>)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool islessequal(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::islessequal((_Promoted) __x, (_Promoted) __y);
}

// islessgreater

_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES(__is_fp_v<_Tp>)
[[nodiscard]] _CCCL_API constexpr bool islessgreater(_Tp __x, _Tp __y) noexcept
{
  if constexpr (__is_ext_nv_fp_v<_Tp>)
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
#if _CCCL_HAS_NVFP16()
      if constexpr (is_same_v<_Tp, ::__half>
#  if _CCCL_HAS_NVBF16()
                    || is_same_v<_Tp, ::__nv_bfloat16>
#  endif // _CCCL_HAS_NVBF16()
      )
      {
        NV_IF_TARGET(NV_IS_DEVICE, ({
                       if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
                       {
                         return false;
                       }
                       return ::__hne(__x, __y);
                     }))
      }
#endif // _CCCL_HAS_NVFP16()
    }

    using _Cmp = ::cuda::std::not_equal_to<>;
    return ::cuda::std::__fp_compare_generic(__x, __y, _Cmp{});
  }
  else
  {
    _CCCL_IF_NOT_CONSTEVAL_DEFAULT
    {
      constexpr auto __fmt = __fp_format_of_v<_Tp>;

      if constexpr (__fmt == __fp_format::__binary64)
      {
        _CCCL_DISPATCH_SLOW_FP64_TARGET(
          return ::cuda::std::__fp_compare_generic(__x, __y, ::cuda::std::not_equal_to<>{});)
      }

      NV_IF_TARGET(NV_IS_HOST, (return ::islessgreater(__x, __y);))
    }

    if (::cuda::std::isnan(__x) || ::cuda::std::isnan(__y))
    {
      return false;
    }
    return __x != __y;
  }
}

_CCCL_TEMPLATE(class _T1, class _T2, class _Promoted = __promote_t<_T1, _T2>)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool islessgreater(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::islessgreater((_Promoted) __x, (_Promoted) __y);
}

// isunordered

_CCCL_TEMPLATE(class _T1, class _T2)
_CCCL_REQUIRES(__is_extended_arithmetic_v<_T1> _CCCL_AND __is_extended_arithmetic_v<_T2>)
[[nodiscard]] _CCCL_API constexpr bool isunordered(_T1 __x, _T2 __y) noexcept
{
  return ::cuda::std::isnan(__x) || ::cuda::std::isnan(__y);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CMATH_TRAITS_H
