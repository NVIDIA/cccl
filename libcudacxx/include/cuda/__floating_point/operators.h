//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FLOATING_POINT_OPERATORS_H
#define _CUDA___FLOATING_POINT_OPERATORS_H

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
#  include <cuda/std/__concepts/concept_macros.h>

#  if _CCCL_HAS_INCLUDE(<stdfloat>)
#    include <stdfloat>
#  endif

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <class _Lhs, typename _Rhs>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto __fp_make_common_type()
{
  constexpr auto __rank_order = __fp_make_conv_rank_order<_Lhs, _Rhs>();

  if constexpr (__rank_order == __fp_conv_rank_order::__equal)
  {
    // Extended floating point types have higher subrank, prefer cuda extended types over std extended types
    // Fixme: potentially will not work correctly for long double
    // auto val = 1.0f64 + 1.0l; // val will be of type long double, is this right?
    if constexpr (__is_standard_floating_point_v<_Lhs> || __is_std_extended_floating_point_v<_Lhs>)
    {
      return _Rhs{};
    }
    else
    {
      return _Lhs{};
    }
  }
  else if constexpr (__rank_order == __fp_conv_rank_order::__greater)
  {
    return _Lhs{};
  }
  else if constexpr (__rank_order == __fp_conv_rank_order::__less)
  {
    return _Rhs{};
  }
  else
  {
    static_assert(__always_false<sizeof(_Lhs)>(), "Cannot make a common fp type from the given types");
    _CCCL_UNREACHABLE();
  }
}

template <class _Lhs, typename _Rhs>
using __fp_common_type_t = decltype(__fp_make_common_type<_Lhs, _Rhs>());

// Implementations of the arithmetic operations. Usually an operation is in several parts:
//   1. __op() - the entry point for the operation which tries to implement the operation using the host & device native
//      types if available, otherwise calls the __op_impl() function.
//
//   2. __op_impl() - the non native implementation of the operation for the given type. This function dispatches to the
//      target specific implementation and falls back to the constexpr implementation.
//
//   3. __op_impl_TARGET() - the target specific implementation. If available, implements the operation via asm.
//
//   4. __op_impl_constexpr() - the constexpr implementation of the operation. Slow.
//
// The arguments may be heterogenous. In that case the implementation is chosen in the following order:
//   1. try to use the host & device native types
//   2. try to use host & device native instructions for mixed arithmetic
//   3. cast the arguments to the common type and use the implementation for homogeneous arguments
struct __fp_ops
{
  /********************************************************************************************************************/
  // Negation
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static _Tp __neg_impl_constexpr(const _Tp& __src) noexcept
  {
    auto __ret{__src};
    __ret.__set_sign(!__ret.__get_sign());
    return __ret;
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static _Tp __neg_impl_device(const _Tp& __src) noexcept
  {
    [[maybe_unused]] _Tp __ret;

#  if __cccl_ptx_isa >= 600
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (asm("neg.f16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__src.__storage_)); return __ret;))
    }
#  endif // __cccl_ptx_isa >= 600
#  if __cccl_ptx_isa >= 700
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (asm("neg.bf16 %0, %1;" : "=h"(__ret.__storage_) : "h"(__src.__storage_)); return __ret;))
    }
#  endif // ^^^ __cccl_ptx_isa < 700 ^^^

    return __neg_impl_constexpr(__src);
  }

  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr _Tp __neg_impl(const _Tp& __src) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __neg_impl_device(__src);))
    }

    return __neg_impl_constexpr(__src);
  }

  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __neg(const _Tp& __src) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Tp::__has_host_native_type) { return __fp{__fp_from_native, -__src.__host_native()}; } else {
          return __neg_impl(__src);
        }),
      (
        if constexpr (_Tp::__has_device_native_type) { return __fp{__fp_from_native, -__src.__device_native()}; } else {
          return __neg_impl(__src);
        }))
  }

  /********************************************************************************************************************/
  // Addition
  /********************************************************************************************************************/
  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static __fp_common_type_t<_Lhs, _Rhs>
  __add_impl_constexpr(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr ((_CUDA_VSTD::is_same_v<_Lhs, fp16> && _CUDA_VSTD::is_same_v<_Rhs, fp16>)
                  || (_CUDA_VSTD::is_same_v<_Lhs, bf16> && _CUDA_VSTD::is_same_v<_Rhs, bf16>) )
    {
      return _Lhs{static_cast<float>(__lhs) + static_cast<float>(__rhs)};
    }
    else
    {
      _CCCL_ASSERT(false, "Addition is not supported for the given type");
      return {};
    }

    // if (__lhs.__is_nan() && __rhs.__is_nan())
    // {
    //   return _Tp::__nan();
    // }

    // if (__lhs.__is_inf() && __rhs.__is_inf())
    // {
    //   return (__lhs.__get_sign() == __rhs.__get_sign()) ? __lhs : -_Tp::__nan();
    // }

    // auto __lhs_sign = __lhs.__get_sign();
    // auto __lhs_exp  = __lhs.__get_exp();
    // auto __lhs_mant = __lhs.__get_mant();
    // auto __rhs_sign = __rhs.__get_sign();
    // auto __rhs_exp  = __rhs.__get_exp();
    // auto __rhs_mant = __rhs.__get_mant();

    // if (__lhs_exp > __rhs_exp)
    // {
    //   __rhs_mant >>= (__lhs_exp - __rhs_exp);
    //   __rhs_exp = __lhs_exp;
    // }
    // else if (__rhs_exp > __lhs_exp)
    // {
    //   __lhs_mant >>= (__rhs_exp - __lhs_exp);
    //   __lhs_exp = __rhs_exp;
    // }

    // using _Sp = _CUDA_VSTD::make_signed_t<typename _Tp::__storage_type>;

    // bool __res_sign{};
    // _Sp __res_exp = __lhs_exp;
    // _Sp __res_mant{};

    // if (__lhs_sign == __rhs_sign)
    // {
    //   __res_mant = __lhs_mant + __rhs_mant;
    //   __res_sign = __lhs_sign;
    // }
    // else if (__lhs_mant >= __rhs_mant)
    // {
    //   __res_mant = __lhs_mant - __rhs_mant;
    //   __res_sign = __lhs_sign;
    // }
    // else
    // {
    //   __res_mant = __rhs_mant - __lhs_mant;
    //   __res_sign = __rhs_sign;
    // }

    // while (__res_mant > static_cast<_Sp>(_Tp::__mant_val_mask()))
    // {
    //   __res_mant >>= 1;
    //   __res_exp++;
    // }

    // while (__res_mant < (_Sp{1} << (_Tp::__mant_nbits - 1)) && __res_mant != 0)
    // {
    //   __res_mant <<= 1;
    //   __res_exp--;
    // }

    // todo: implement denormalized numbers

    // _Tp __ret{};

    // if (__res_exp > static_cast<_Sp>(_Tp::__exp_val_mask()) / 2)
    // {
    //   __ret = _Tp::__inf();
    //   __ret.__set_sign(__res_sign);
    // }
    // else if (__res_exp < -static_cast<_Sp>(_Tp::__exp_val_mask()) / 2 + 1)
    // {
    //   __ret = _Tp{};
    // }
    // else if (__res_mant == 0)
    // {
    //   __ret.__set_sign(__res_sign);
    //   __ret.__set_exp(__res_exp);
    //   __ret.__set_mant(__res_mant);
    // }

    // return __ret;
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static __fp_common_type_t<_Lhs, _Rhs>
  __add_impl_device(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    [[maybe_unused]] __fp_common_type_t<_Lhs, _Rhs> __ret;

    if constexpr (!_CUDA_VSTD::is_same_v<_Lhs, _Rhs>)
    {
#  if __cccl_ptx_isa >= 860
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, fp16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp32))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_100,
          (asm("add.f32.f16 %0, %1, %2;" : "=r"(__ret.__storage_) : "h"(__lhs.__storage_), "r"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 860
#  if __cccl_ptx_isa >= 860
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, bf16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp32))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_100,
          (asm("add.f32.bf16 %0, %1, %2;" : "=r"(__ret.__storage_) : "h"(__lhs.__storage_), "r"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 860
      return __add(__fp_common_type_t<_Lhs, _Rhs>{__lhs}, __fp_common_type_t<_Lhs, _Rhs>{__rhs});
    }
    else
    {
#  if __cccl_ptx_isa >= 420
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, fp16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_53,
          (asm("add.f16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 420
#  if __cccl_ptx_isa >= 780
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, bf16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, bf16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_90,
          (asm("add.bf16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 780
      return __add_impl_constexpr(__lhs, __rhs);
    }
    _CCCL_UNREACHABLE();
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __add_impl(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __add_impl_device(__lhs, __rhs);))
    }

    return __add_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __add(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Lhs::__has_host_native_type && _Rhs::__has_host_native_type) {
          return __fp{__fp_from_native, __lhs.__host_native() + __rhs.__host_native()};
        } else { return __add_impl(__lhs, __rhs); }),
      (
        if constexpr (_Lhs::__has_device_native_type && _Rhs::__has_device_native_type) {
          return __fp{__fp_from_native, __lhs.__device_native() + __rhs.__device_native()};
        } else { return __add_impl(__lhs, __rhs); }))
  }

  /********************************************************************************************************************/
  // Subtraction
  /********************************************************************************************************************/
  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static __fp_common_type_t<_Lhs, _Rhs>
  __sub_impl_constexpr(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr ((_CUDA_VSTD::is_same_v<_Lhs, fp16> && _CUDA_VSTD::is_same_v<_Rhs, fp16>)
                  || (_CUDA_VSTD::is_same_v<_Lhs, bf16> && _CUDA_VSTD::is_same_v<_Rhs, bf16>) )
    {
      return _Lhs{static_cast<float>(__lhs) - static_cast<float>(__rhs)};
    }
    else
    {
      _CCCL_ASSERT(false, "Subtraction is not supported for the given type");
      return {};
    }
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static __fp_common_type_t<_Lhs, _Rhs>
  __sub_impl_device(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    [[maybe_unused]] __fp_common_type_t<_Lhs, _Rhs> __ret;

    if constexpr (!_CUDA_VSTD::is_same_v<_Lhs, _Rhs>)
    {
#  if __cccl_ptx_isa >= 860
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, fp16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp32))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_100,
          (asm("sub.f32.f16 %0, %1, %2;" : "=r"(__ret.__storage_) : "h"(__lhs.__storage_), "r"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 860
#  if __cccl_ptx_isa >= 860
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, bf16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp32))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_100,
          (asm("sub.f32.bf16 %0, %1, %2;" : "=r"(__ret.__storage_) : "h"(__lhs.__storage_), "r"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 860
      return __sub(__fp_common_type_t<_Lhs, _Rhs>{__lhs}, __fp_common_type_t<_Lhs, _Rhs>{__rhs});
    }
    else
    {
#  if __cccl_ptx_isa >= 420
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, fp16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_53,
          (asm("sub.f16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 420
#  if __cccl_ptx_isa >= 780
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, bf16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, bf16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_90,
          (asm("sub.bf16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 780
      return __sub_impl_constexpr(__lhs, __rhs);
    }
    _CCCL_UNREACHABLE();
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __sub_impl(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __sub_impl_device(__lhs, __rhs);))
    }

    return __sub_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __sub(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Lhs::__has_host_native_type && _Rhs::__has_host_native_type) {
          return __fp{__fp_from_native, __lhs.__host_native() - __rhs.__host_native()};
        } else { return __sub_impl(__lhs, __rhs); }),
      (
        if constexpr (_Lhs::__has_device_native_type && _Rhs::__has_device_native_type) {
          return __fp{__fp_from_native, __lhs.__device_native() - __rhs.__device_native()};
        } else { return __sub_impl(__lhs, __rhs); }))
  }

  /********************************************************************************************************************/
  // Multiplication
  /********************************************************************************************************************/
  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static __fp_common_type_t<_Lhs, _Rhs>
  __mul_impl_constexpr(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr ((_CUDA_VSTD::is_same_v<_Lhs, fp16> && _CUDA_VSTD::is_same_v<_Rhs, fp16>)
                  || (_CUDA_VSTD::is_same_v<_Lhs, bf16> && _CUDA_VSTD::is_same_v<_Rhs, bf16>) )
    {
      return _Lhs{static_cast<float>(__lhs) * static_cast<float>(__rhs)};
    }
    else
    {
      _CCCL_ASSERT(false, "Multiplication is not supported for the given type");
      return {};
    }
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static __fp_common_type_t<_Lhs, _Rhs>
  __mul_impl_device(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    [[maybe_unused]] __fp_common_type_t<_Lhs, _Rhs> __ret;

    if constexpr (!_CUDA_VSTD::is_same_v<_Lhs, _Rhs>)
    {
      return __mul(__fp_common_type_t<_Lhs, _Rhs>{__lhs}, __fp_common_type_t<_Lhs, _Rhs>{__rhs});
    }
    else
    {
#  if __cccl_ptx_isa >= 420
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, fp16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, fp16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_53,
          (asm("mul.f16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 420
#  if __cccl_ptx_isa >= 780
      if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Lhs, bf16) && _CUDA_VSTD::_CCCL_TRAIT(is_same, _Rhs, bf16))
      {
        NV_IF_TARGET(
          NV_PROVIDES_SM_90,
          (asm("mul.bf16 %0, %1, %2;" : "=h"(__ret.__storage_) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
           return __ret;))
      }
#  endif // __cccl_ptx_isa >= 780
      return __mul_impl_constexpr(__lhs, __rhs);
    }
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __mul_impl(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __mul_impl_device(__lhs, __rhs);))
    }

    return __mul_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __mul(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Lhs::__has_host_native_type && _Rhs::__has_host_native_type) {
          return __fp{__fp_from_native, __lhs.__host_native() * __rhs.__host_native()};
        } else { return __mul_impl(__lhs, __rhs); }),
      (
        if constexpr (_Lhs::__has_device_native_type && _Rhs::__has_device_native_type) {
          return __fp{__fp_from_native, __lhs.__device_native() * __rhs.__device_native()};
        } else { return __mul_impl(__lhs, __rhs); }))
  }

  /********************************************************************************************************************/
  // Division
  /********************************************************************************************************************/
  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static __fp_common_type_t<_Lhs, _Rhs>
  __div_impl_constexpr(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr ((_CUDA_VSTD::is_same_v<_Lhs, fp16> && _CUDA_VSTD::is_same_v<_Rhs, fp16>)
                  || (_CUDA_VSTD::is_same_v<_Lhs, bf16> && _CUDA_VSTD::is_same_v<_Rhs, bf16>) )
    {
      return _Lhs{static_cast<float>(__lhs) / static_cast<float>(__rhs)};
    }
    else
    {
      _CCCL_ASSERT(false, "Division is not supported for the given type");
      return {};
    }
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static __fp_common_type_t<_Lhs, _Rhs>
  __div_impl_device(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (!_CUDA_VSTD::is_same_v<_Lhs, _Rhs>)
    {
      return __div(__fp_common_type_t<_Lhs, _Rhs>{__lhs}, __fp_common_type_t<_Lhs, _Rhs>{__rhs});
    }
    else
    {
      return __div_impl_constexpr(__lhs, __rhs);
    }
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __div_impl(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __div_impl_device(__lhs, __rhs);))
    }

    return __div_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr __fp_common_type_t<_Lhs, _Rhs>
  __div(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Lhs::__has_host_native_type && _Rhs::__has_host_native_type) {
          return __fp{__fp_from_native, __lhs.__host_native() / __rhs.__host_native()};
        } else { return __div_impl(__lhs, __rhs); }),
      (
        if constexpr (_Lhs::__has_device_native_type && _Rhs::__has_device_native_type) {
          return __fp{__fp_from_native, __lhs.__device_native() / __rhs.__device_native()};
        } else { return __div_impl(__lhs, __rhs); }))
  }

  /********************************************************************************************************************/
  // Equality
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __eq_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    if (__lhs.__is_nan() || __rhs.__is_nan())
    {
      return false;
    }

    return (__lhs.__storage_ & __lhs.__mask()) == (__rhs.__storage_ & __rhs.__mask());
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __eq_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.eq.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.eq.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __eq_impl_constexpr(__lhs, __rhs);
  }

  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool __eq_impl(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    using _CommonFp = __fp_common_type_t<_Tp, _Tp>;

    const auto __clhs = static_cast<_CommonFp>(__lhs);
    const auto __crhs = static_cast<_CommonFp>(__rhs);

    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_DEVICE, (return __eq_impl_device(__clhs, __crhs);))
    }

    return __eq_impl_constexpr(__clhs, __crhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __eq(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    NV_IF_ELSE_TARGET(
      NV_IS_HOST,
      (
        if constexpr (_Lhs::__has_host_native_type && _Rhs::__has_host_native_type) {
          return __lhs.__host_native() == __rhs.__host_native();
        } else { return __eq_impl(__lhs, __rhs); }),
      (
        if constexpr (_Lhs::__has_device_native_type && _Rhs::__has_device_native_type) {
          return __lhs.__device_native() == __rhs.__device_native();
        } else { return __eq_impl(__lhs, __rhs); }))
  }

  /********************************************************************************************************************/
  // Inequality
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __neq_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    return !__eq_impl_constexpr(__lhs, __rhs);
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __neq_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.ne.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.ne.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __eq_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __neq(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (_Lhs::__has_native_type() && _Rhs::__has_native_type())
    {
      NV_IF_ELSE_TARGET(NV_IS_HOST,
                        (return __lhs.__host_native() != __rhs.__host_native();),
                        (return __lhs.__device_native() != __rhs.__device_native();))
    }
    else
    {
      using _CommonFp = __fp_common_type_t<_Lhs, _Rhs>;

      const auto __l = static_cast<_CommonFp>(__lhs);
      const auto __r = static_cast<_CommonFp>(__rhs);

      if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return __neq_impl_device(__l, __r);))
      }

      return __neq_impl_constexpr(__l, __r);
    }
  }

  /********************************************************************************************************************/
  // Less than
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __lt_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    if (__lhs.__is_nan() || __rhs.__is_nan())
    {
      return false;
    }

    return (__lhs.__storage_ & __lhs.__mask()) < (__rhs.__storage_ & __rhs.__mask());
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __lt_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.lt.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.lt.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __lt_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __lt(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (_Lhs::__has_native_type() && _Rhs::__has_native_type())
    {
      NV_IF_ELSE_TARGET(NV_IS_HOST,
                        (return __lhs.__host_native() < __rhs.__host_native();),
                        (return __lhs.__device_native() < __rhs.__device_native();))
    }
    else
    {
      using _CommonFp = __fp_common_type_t<_Lhs, _Rhs>;

      const auto __l = static_cast<_CommonFp>(__lhs);
      const auto __r = static_cast<_CommonFp>(__rhs);

      if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return __lt_impl_device(__l, __r);))
      }

      return __lt_impl_constexpr(__l, __r);
    }
  }

  /********************************************************************************************************************/
  // Less than or equal
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __le_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    if (__lhs.__is_nan() || __rhs.__is_nan())
    {
      return false;
    }

    return (__lhs.__storage_ & __lhs.__mask()) <= (__rhs.__storage_ & __rhs.__mask());
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __le_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.le.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.le.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __le_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __le(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (_Lhs::__has_native_type() && _Rhs::__has_native_type())
    {
      NV_IF_ELSE_TARGET(NV_IS_HOST,
                        (return __lhs.__host_native() <= __rhs.__host_native();),
                        (return __lhs.__device_native() <= __rhs.__device_native();))
    }
    else
    {
      using _CommonFp = __fp_common_type_t<_Lhs, _Rhs>;

      const auto __l = static_cast<_CommonFp>(__lhs);
      const auto __r = static_cast<_CommonFp>(__rhs);

      if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return __le_impl_device(__l, __r);))
      }

      return __le_impl_constexpr(__l, __r);
    }
  }

  /********************************************************************************************************************/
  // Greater than
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __gt_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    if (__lhs.__is_nan() || __rhs.__is_nan())
    {
      return false;
    }

    return (__lhs.__storage_ & __lhs.__mask()) > (__rhs.__storage_ & __rhs.__mask());
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __gt_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.gt.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.gt.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __gt_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __gt(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (_Lhs::__has_native_type() && _Rhs::__has_native_type())
    {
      NV_IF_ELSE_TARGET(NV_IS_HOST,
                        (return __lhs.__host_native() > __rhs.__host_native();),
                        (return __lhs.__device_native() > __rhs.__device_native();))
    }
    else
    {
      using _CommonFp = __fp_common_type_t<_Lhs, _Rhs>;

      const auto __l = static_cast<_CommonFp>(__lhs);
      const auto __r = static_cast<_CommonFp>(__rhs);

      if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return __gt_impl_device(__l, __r);))
      }

      return __gt_impl_constexpr(__l, __r);
    }
  }

  /********************************************************************************************************************/
  // Greater than or equal
  /********************************************************************************************************************/
  template <class _Tp>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr static bool
  __ge_impl_constexpr(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
    if (__lhs.__is_nan() || __rhs.__is_nan())
    {
      return false;
    }

    return (__lhs.__storage_ & __lhs.__mask()) >= (__rhs.__storage_ & __rhs.__mask());
  }

  template <class _Tp>
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _CCCL_DEVICE static bool
  __ge_impl_device(const _Tp& __lhs, const _Tp& __rhs) noexcept
  {
#  if __cccl_ptx_isa >= 650
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, fp16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_53,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.ge.u16.f16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 650
#  if __cccl_ptx_isa >= 780
    if constexpr (_CUDA_VSTD::_CCCL_TRAIT(is_same, _Tp, bf16))
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (_CUDA_VSTD::uint16_t __ret;
                    asm("set.ge.u16.bf16 %0, %1, %2;" : "=h"(__ret) : "h"(__lhs.__storage_), "h"(__rhs.__storage_));
                    return static_cast<bool>(__ret);))
    }
#  endif // __cccl_ptx_isa >= 780

    return __ge_impl_constexpr(__lhs, __rhs);
  }

  template <class _Lhs, class _Rhs>
  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI static constexpr auto __ge(const _Lhs& __lhs, const _Rhs& __rhs) noexcept
  {
    if constexpr (_Lhs::__has_native_type() && _Rhs::__has_native_type())
    {
      NV_IF_ELSE_TARGET(NV_IS_HOST,
                        (return __lhs.__host_native() >= __rhs.__host_native();),
                        (return __lhs.__device_native() >= __rhs.__device_native();))
    }
    else
    {
      using _CommonFp = __fp_common_type_t<_Lhs, _Rhs>;

      const auto __l = static_cast<_CommonFp>(__lhs);
      const auto __r = static_cast<_CommonFp>(__rhs);

      if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
      {
        NV_IF_TARGET(NV_IS_DEVICE, (return __ge_impl_device(__l, __r);))
      }

      return __ge_impl_constexpr(__l, __r);
    }
  }
};

/**********************************************************************************************************************/
// Unary operators
/**********************************************************************************************************************/
template <class _Config>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_Config> operator+(const __fp<_Config>& __src) noexcept
{
  return __src;
}
template <class _Config>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_Config> operator-(const __fp<_Config>& __src) noexcept
{
  static_assert(_Config::__is_signed, "Unary minus is not allowed for unsigned floating point types");
  return __fp_ops::__neg(__src);
}

/**********************************************************************************************************************/
// Binary operators
/**********************************************************************************************************************/
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
operator+(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return static_cast<__fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(__fp_ops::__add(__lhs, __rhs));
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
operator-(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return static_cast<__fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(__fp_ops::__sub(__lhs, __rhs));
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
operator*(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return static_cast<__fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(__fp_ops::__mul(__lhs, __rhs));
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
operator/(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return static_cast<__fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(__fp_ops::__div(__lhs, __rhs));
}
template <class _LhsConfig, class _RhsConfig>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_LhsConfig>&
operator+=(__fp<_LhsConfig>& __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(),
                "Implicit narrow conversion from higher to lower rank is not allowed");
  return __lhs = __lhs + __rhs;
}
template <class _LhsConfig, class _RhsConfig>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_LhsConfig>&
operator-=(__fp<_LhsConfig>& __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(),
                "Implicit narrow conversion from higher to lower rank is not allowed");
  return __lhs = __lhs - __rhs;
}
template <class _LhsConfig, class _RhsConfig>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_LhsConfig>&
operator*=(__fp<_LhsConfig>& __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(),
                "Implicit narrow conversion from higher to lower rank is not allowed");
  return __lhs = __lhs * __rhs;
}
template <class _LhsConfig, class _RhsConfig>
_LIBCUDACXX_HIDE_FROM_ABI constexpr __fp<_LhsConfig>&
operator/=(__fp<_LhsConfig>& __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, __fp<_RhsConfig>>>(),
                "Implicit narrow conversion from higher to lower rank is not allowed");
  return __lhs = __lhs / __rhs;
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator==(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__eq(__lhs, __rhs);
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator!=(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__neq(__lhs, __rhs);
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__lt(__lhs, __rhs);
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator<=(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__le(__lhs, __rhs);
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__gt(__lhs, __rhs);
}
template <class _LhsConfig, class _RhsConfig>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr bool
operator>=(__fp<_LhsConfig> __lhs, __fp<_RhsConfig> __rhs) noexcept
{
  return __fp_ops::__ge(__lhs, __rhs);
}

#  define _LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(_TYPE, _EXSPACE)                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator+(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(__fp_ops::__add(__lhs, __fp{__rhs}));             \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator+(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<_TYPE, __fp<_RhsConfig>>>(__fp_ops::__add(__fp{__lhs}, __rhs));             \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator-(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(__fp_ops::__sub(__lhs, __fp{__rhs}));             \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator-(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<_TYPE, __fp<_RhsConfig>>>(__fp_ops::__sub(__fp{__lhs}, __rhs));             \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator*(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(__fp_ops::__mul(__lhs, __fp{__rhs}));             \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator*(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<_TYPE, __fp<_RhsConfig>>>(__fp_ops::__mul(__fp{__lhs}, __rhs));             \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator/(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(__fp_ops::__div(__lhs, __fp{__rhs}));             \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr auto operator/(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return static_cast<__fp_common_type_t<_TYPE, __fp<_RhsConfig>>>(__fp_ops::__div(__fp{__lhs}, __rhs));             \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_HIDE_FROM_ABI _EXSPACE constexpr __fp<_LhsConfig>& operator+=(__fp<_LhsConfig>& __lhs, _TYPE __rhs) noexcept  \
    {                                                                                                                   \
      static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(),             \
                    "Implicit narrow conversion from higher to lower rank is not allowed");                             \
      return __lhs = __lhs + __rhs;                                                                                     \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_HIDE_FROM_ABI _EXSPACE constexpr __fp<_LhsConfig>& operator-=(__fp<_LhsConfig>& __lhs, _TYPE __rhs) noexcept  \
    {                                                                                                                   \
      static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(),             \
                    "Implicit narrow conversion from higher to lower rank is not allowed");                             \
      return __lhs = __lhs - __rhs;                                                                                     \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_HIDE_FROM_ABI _EXSPACE constexpr __fp<_LhsConfig>& operator*=(__fp<_LhsConfig>& __lhs, _TYPE __rhs) noexcept  \
    {                                                                                                                   \
      static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(),             \
                    "Implicit narrow conversion from higher to lower rank is not allowed");                             \
      return __lhs = __lhs * __rhs;                                                                                     \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_HIDE_FROM_ABI _EXSPACE constexpr __fp<_LhsConfig>& operator/=(__fp<_LhsConfig>& __lhs, _TYPE __rhs) noexcept  \
    {                                                                                                                   \
      static_assert(__fp_cast_is_implicit<__fp<_LhsConfig>, __fp_common_type_t<__fp<_LhsConfig>, _TYPE>>(),             \
                    "Implicit narrow conversion from higher to lower rank is not allowed");                             \
      return __lhs = __lhs / __rhs;                                                                                     \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator==(                                             \
      __fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__eq(__lhs, __fp{__rhs});                                                                        \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator==(                                             \
      _TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__eq(__fp{__lhs}, __rhs);                                                                        \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator!=(                                             \
      __fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__neq(__lhs, __fp{__rhs});                                                                       \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator!=(                                             \
      _TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__neq(__fp{__lhs}, __rhs);                                                                       \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator<(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return __fp_ops::__lt(__lhs, __fp{__rhs});                                                                        \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator<(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return __fp_ops::__lt(__fp{__lhs}, __rhs);                                                                        \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator<=(                                             \
      __fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__le(__lhs, __fp{__rhs});                                                                        \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator<=(                                             \
      _TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__le(__fp{__lhs}, __rhs);                                                                        \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator>(__fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept \
    {                                                                                                                   \
      return __fp_ops::__gt(__lhs, __fp{__rhs});                                                                        \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator>(_TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept \
    {                                                                                                                   \
      return __fp_ops::__gt(__fp{__lhs}, __rhs);                                                                        \
    }                                                                                                                   \
    template <class _LhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator>=(                                             \
      __fp<_LhsConfig> __lhs, _TYPE __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__ge(__lhs, __fp{__rhs});                                                                        \
    }                                                                                                                   \
    template <class _RhsConfig>                                                                                         \
    _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI _EXSPACE constexpr bool operator>=(                                             \
      _TYPE __lhs, __fp<_RhsConfig> __rhs) noexcept                                                                     \
    {                                                                                                                   \
      return __fp_ops::__ge(__fp{__lhs}, __rhs);                                                                        \
    }

_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(float, _CCCL_HOST_DEVICE)
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(double, _CCCL_HOST_DEVICE)
#  if !_CCCL_COMPILER(NVRTC) && (!_CCCL_HAS_CUDA_COMPILER || _CCCL_CUDA_COMPILER(NVHPC))
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(long double, _CCCL_HOST)
#  endif // !_CCCL_COMPILER(NVRTC) && (!_CCCL_HAS_CUDA_COMPILER || _CCCL_CUDA_COMPILER(NVHPC))

#  if __STDCPP_FLOAT16_T__ == 1
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(::std::float16_t, _CCCL_HOST_DEVICE)
#  endif // __STDCPP_FLOAT16_T__ == 1
#  if __STDCPP_FLOAT32_T__ == 1
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(::std::float32_t, _CCCL_HOST_DEVICE)
#  endif // __STDCPP_FLOAT32_T__ == 1
#  if __STDCPP_FLOAT64_T__ == 1
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(::std::float64_t, _CCCL_HOST_DEVICE)
#  endif // __STDCPP_FLOAT64_T__ == 1
#  if __STDCPP_FLOAT128_T__ == 1
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(::std::float128_t, _CCCL_HOST_DEVICE)
#  endif // __STDCPP_FLOAT128_T__ == 1
#  if __STDCPP_BFLOAT16_T__ == 1
_LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR(::std::bfloat16_t, _CCCL_HOST_DEVICE)
#  endif // __STDCPP_BFLOAT16_T__ == 1

#  undef _LIBCUDACXX_FP_DEFINE_BINARY_OPERATORS_FOR

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CCCL_STD_VER >= 2017

#endif // _CUDA___FLOATING_POINT_OPERATORS_H
