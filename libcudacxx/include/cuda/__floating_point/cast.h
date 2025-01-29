//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_CAST_H
#define _CUDA___FLOATING_POINT_CAST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_STD_VER >= 2017

#  include <cuda/__floating_point/fp.h>
#  include <cuda/__floating_point/type_traits.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/is_integral.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_signed.h>
#  include <cuda/std/__type_traits/is_unsigned.h>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

struct __fp_cast
{
  template <class _Tp, class _Up>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __cast_generic(const _Up& __src)
  {
    _CCCL_ASSERT(false, "Unsupported floating point cast");
    // todo: implement generic cast
    //  - fp -> fp
    //  - fp -> integral
    //  - integral -> fp

    _Tp __dst{};

    // Copy sign
    // if constexpr (_Tp::__is_signed && _Up::__is_signed)
    // {
    //   __dst.__set_sign(__src.__get_sign());
    // }
    // else if constexpr (!_Tp::__is_signed && _Up::__is_signed)
    // {
    //   if (__src.__get_sign())
    //   {
    //     return _Tp::__nan();
    //   }
    // }

    // using _Sp =
    //   _CUDA_VSTD::make_signed_t<_CUDA_VSTD::common_type_t<typename _Tp::__storage_type, typename
    //   _Up::__storage_type>>;

    // // Convert exponent
    // constexpr _Sp __src_exp_bias = _Up::__exp_val_mask() / 2;
    // constexpr _Sp __dst_exp_bias = _Tp::__exp_val_mask() / 2;

    // _Sp __dst_exp = static_cast<_Sp>(__src.__get_exp()) - __src_exp_bias + __dst_exp_bias;

    // if (__dst_exp >= static_cast<_Sp>(_Tp::__exp_val_mask()))
    // {
    //   return _Tp::__inf();
    // }

    // __dst.__set_exp(static_cast<typename _Tp::__storage_type>(__dst_exp));

    // // Convert mantissa (todo: implement rounding)
    // constexpr ptrdiff_t __mant_diff =
    //   static_cast<ptrdiff_t>(_Tp::__mant_nbits) - static_cast<ptrdiff_t>(_Up::__mant_nbits);

    // _Sp __dst_mant{};

    // if constexpr (__mant_diff < 0)
    // {
    //   __dst_mant = static_cast<_Sp>(__src.__get_mant()) >> (-__mant_diff);
    // }
    // else
    // {
    //   __dst_mant = static_cast<_Sp>(__src.__get_mant()) << __mant_diff;
    // }

    // __dst.__set_mant(static_cast<typename _Tp::__storage_type>(__dst_mant));

    return __dst;
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_same_v<__fp_make_config_from_t<_Up>, __fp16_config>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    fp16 __fp_val{__fp_from_native, __val};
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp32>)
    {
      NV_IF_TARGET(NV_IS_DEVICE,
                   (asm("cvt.f32.f16 %0, %1;" : "=r"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp64>)
    {
      NV_IF_TARGET(NV_IS_DEVICE,
                   (asm("cvt.f64.f16 %0, %1;" : "=l"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.bf16.f16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_integral_v<_Tp> && _CUDA_VSTD::is_signed_v<_Tp>)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (int16_t __ret; asm("cvt.rni.s8.f16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_));
                      return static_cast<_Tp>(__ret);))
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.s16.f16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.s32.f16 %0, %1;" : "=r"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.s64.f16 %0, %1;" : "=l"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
    }
    else if constexpr (_CUDA_VSTD::is_integral_v<_Tp> && _CUDA_VSTD::is_unsigned_v<_Tp>)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (uint16_t __ret; asm("cvt.rni.u8.f16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_));
                      return static_cast<_Tp>(__ret);))
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.u16.f16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.u32.f16 %0, %1;" : "=r"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        NV_IF_TARGET(NV_IS_DEVICE,
                     (asm("cvt.rni.u64.f16 %0, %1;" : "=l"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
    }

    return __ret = __cast_generic<_Tp>(__fp_val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_same_v<__fp_make_config_from_t<_Up>, __fp32_config>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    fp32 __fp_val{__fp_from_native, __val};
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE,
                   (asm("cvt.rn.f16.f32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__fp_val.__storage_)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__fp_val.__storage_)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__fp_val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_same_v<__fp_make_config_from_t<_Up>, __fp64_config>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    fp64 __fp_val{__fp_from_native, __val};
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE,
                   (asm("cvt.rn.f16.f64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__fp_val.__storage_)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.f64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__fp_val.__storage_)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__fp_val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_same_v<__fp_make_config_from_t<_Up>, __bf16_config>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    bf16 __fp_val{__fp_from_native, __val};
    _Tp __ret{};

#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.f16.bf16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp32>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.f32.bf16 %0, %1;" : "=r"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp64>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.f64.bf16 %0, %1;" : "=l"(__ret.__storage_) : "h"(__fp_val.__storage_)); return __ret;))
    }
    else if constexpr (_CUDA_VSTD::is_integral_v<_Tp> && _CUDA_VSTD::is_signed_v<_Tp>)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (int16_t __ret; asm("cvt.s8.bf16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_));
                      return static_cast<_Tp>(__ret);))
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.s16.bf16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.s32.bf16 %0, %1;" : "=r"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.s64.bf16 %0, %1;" : "=l"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
    }
    else if constexpr (_CUDA_VSTD::is_integral_v<_Tp> && _CUDA_VSTD::is_unsigned_v<_Tp>)
    {
      if constexpr (sizeof(_Tp) == 1)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (uint16_t __ret; asm("cvt.u8.bf16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_));
                      return static_cast<_Tp>(__ret);))
      }
      else if constexpr (sizeof(_Tp) == 2)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.u16.bf16 %0, %1;" : "=h"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 4)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.u32.bf16 %0, %1;" : "=r"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
      else if constexpr (sizeof(_Tp) == 8)
      {
        NV_IF_TARGET(NV_PROVIDES_SM_90,
                     (asm("cvt.u64.bf16 %0, %1;" : "=l"(__ret) : "h"(__fp_val.__storage_)); return __ret;))
      }
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__fp_val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_signed_v<_Up> _CCCL_AND(sizeof(_Up) == 1))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(
        NV_IS_DEVICE,
        (asm("cvt.rn.f16.s8 %0, %1;" : "=h"(__ret.__storage_) : "h"(static_cast<int16_t>(__val))); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.s8 %0, %1;" : "=h"(__ret.__storage_) : "h"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_signed_v<_Up> _CCCL_AND(sizeof(_Up) == 2))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.s16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.s16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_signed_v<_Up> _CCCL_AND(sizeof(_Up) == 4))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.s32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.s32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_signed_v<_Up> _CCCL_AND(sizeof(_Up) == 8))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.s64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.s64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_unsigned_v<_Up> _CCCL_AND(sizeof(_Up) == 1))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(
        NV_IS_DEVICE,
        (asm("cvt.rn.f16.u8 %0, %1;" : "=h"(__ret.__storage_) : "h"(static_cast<int16_t>(__val))); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.u8 %0, %1;" : "=h"(__ret.__storage_) : "r"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_unsigned_v<_Up> _CCCL_AND(sizeof(_Up) == 2))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.u16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.u16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_unsigned_v<_Up> _CCCL_AND(sizeof(_Up) == 4))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.u32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.u32 %0, %1;" : "=h"(__ret.__storage_) : "r"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

  _CCCL_TEMPLATE(class _Tp, class _Up)
  _CCCL_REQUIRES(_CUDA_VSTD::is_integral_v<_Up> _CCCL_AND _CUDA_VSTD::is_unsigned_v<_Up> _CCCL_AND(sizeof(_Up) == 8))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __cast_impl_device(const _Up& __val) noexcept
  {
    _Tp __ret{};

    if constexpr (_CUDA_VSTD::is_same_v<_Tp, fp16>)
    {
      NV_IF_TARGET(NV_IS_DEVICE, (asm("cvt.rn.f16.u64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__val)); return __ret;))
    }
#  if __cccl_ptx_isa >= 780
    else if constexpr (_CUDA_VSTD::is_same_v<_Tp, bf16>)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (asm("cvt.rn.bf16.u64 %0, %1;" : "=h"(__ret.__storage_) : "l"(__val)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ret = __cast_generic<_Tp>(__val);
  }

public:
  template <class _Tp, class _Up>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __cast(const _Up& __src) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __cast_impl_device<_Tp>(__src);))
    }

    return __cast_generic<_Tp>(__src);
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_CAST_H
