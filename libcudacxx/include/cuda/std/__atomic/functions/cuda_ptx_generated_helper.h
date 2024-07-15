//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_HELPER_H
#define _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_HELPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __atomic_cuda_memorder
{
  _relaxed,
  _release,
  _acquire,
  _acq_rel,
  _seq_cst,
  _volatile,
};

template <__atomic_cuda_memorder _Order>
using __atomic_cuda_memorder_tag = integral_constant<__atomic_cuda_memorder, _Order>;

using __atomic_cuda_relaxed  = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_relaxed>;
using __atomic_cuda_release  = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_release>;
using __atomic_cuda_acquire  = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_acquire>;
using __atomic_cuda_acq_rel  = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_acq_rel>;
using __atomic_cuda_seq_cst  = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_seq_cst>;
using __atomic_cuda_volatile = __atomic_cuda_memorder_tag<__atomic_cuda_memorder::_volatile>;

template <bool _Volatile>
using __atomic_cuda_mmio_tag = integral_constant<bool, _Volatile>;

using __atomic_cuda_mmio_enable  = __atomic_cuda_mmio_tag<true>;
using __atomic_cuda_mmio_disable = __atomic_cuda_mmio_tag<false>;

enum class __atomic_cuda_operand
{
  _f,
  _s,
  _u,
  _b,
};

template <__atomic_cuda_operand _Op, size_t _Size>
struct __atomic_cuda_operand_tag
{};

using __atomic_cuda_operand_f16  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_f, 16>;
using __atomic_cuda_operand_s16  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_s, 16>;
using __atomic_cuda_operand_u16  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_u, 16>;
using __atomic_cuda_operand_b16  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, 16>;
using __atomic_cuda_operand_f32  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_f, 32>;
using __atomic_cuda_operand_s32  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_s, 32>;
using __atomic_cuda_operand_u32  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_u, 32>;
using __atomic_cuda_operand_b32  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, 32>;
using __atomic_cuda_operand_f64  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_f, 64>;
using __atomic_cuda_operand_s64  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_s, 64>;
using __atomic_cuda_operand_u64  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_u, 64>;
using __atomic_cuda_operand_b64  = __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, 64>;
using __atomic_cuda_operand_f128 = __atomic_cuda_operand_tag<__atomic_cuda_operand::_f, 128>;
using __atomic_cuda_operand_s128 = __atomic_cuda_operand_tag<__atomic_cuda_operand::_s, 128>;
using __atomic_cuda_operand_u128 = __atomic_cuda_operand_tag<__atomic_cuda_operand::_u, 128>;
using __atomic_cuda_operand_b128 = __atomic_cuda_operand_tag<__atomic_cuda_operand::_b, 128>;

template <class _AtomicType, class _OpTag>
struct __atomic_cuda_operand_deduction
{
  using __type = _AtomicType;
  using __tag  = _OpTag;
};

struct __atomic_longlong2
{
  uint64_t __x;
  uint64_t __y;
};

template <class _Type>
using __atomic_cuda_deduce_bitwise =
  _If<sizeof(_Type) == 2,
      __atomic_cuda_operand_deduction<uint16_t, __atomic_cuda_operand_b16>,
      _If<sizeof(_Type) == 4,
          __atomic_cuda_operand_deduction<uint32_t, __atomic_cuda_operand_b32>,
          _If<sizeof(_Type) == 8,
              __atomic_cuda_operand_deduction<uint64_t, __atomic_cuda_operand_b64>,
              __atomic_cuda_operand_deduction<__atomic_longlong2, __atomic_cuda_operand_b128>>>>;

template <class _Type>
using __atomic_cuda_deduce_arithmetic =
  _If<_CCCL_TRAIT(is_floating_point, _Type),
      _If<sizeof(_Type) == 4,
          __atomic_cuda_operand_deduction<float, __atomic_cuda_operand_f32>,
          __atomic_cuda_operand_deduction<double, __atomic_cuda_operand_f64>>,
      _If<_CCCL_TRAIT(is_signed, _Type),
          _If<sizeof(_Type) == 4,
              __atomic_cuda_operand_deduction<int32_t, __atomic_cuda_operand_s32>,
              __atomic_cuda_operand_deduction<int64_t, __atomic_cuda_operand_u64>>, // There is no atom.add.s64
          _If<sizeof(_Type) == 4,
              __atomic_cuda_operand_deduction<uint32_t, __atomic_cuda_operand_u32>,
              __atomic_cuda_operand_deduction<uint64_t, __atomic_cuda_operand_u64>>>>;

template <class _Type>
using __atomic_cuda_deduce_minmax =
  _If<_CCCL_TRAIT(is_signed, _Type),
      _If<sizeof(_Type) == 4,
          __atomic_cuda_operand_deduction<int32_t, __atomic_cuda_operand_s32>,
          __atomic_cuda_operand_deduction<int64_t, __atomic_cuda_operand_s64>>,
      _If<sizeof(_Type) == 4,
          __atomic_cuda_operand_deduction<uint32_t, __atomic_cuda_operand_u32>,
          __atomic_cuda_operand_deduction<uint64_t, __atomic_cuda_operand_u64>>>;

template <class _Type>
using __atomic_enable_if_native_bitwise = bool;

template <class _Type>
using __atomic_enable_if_native_arithmetic = typename enable_if<_CCCL_TRAIT(is_scalar, _Type), bool>::type;

template <class _Type>
using __atomic_enable_if_not_native_arithmetic = typename enable_if<!_CCCL_TRAIT(is_scalar, _Type), bool>::type;

template <class _Type>
using __atomic_enable_if_native_minmax = typename enable_if<_CCCL_TRAIT(is_integral, _Type), bool>::type;

template <class _Type>
using __atomic_enable_if_not_native_minmax = typename enable_if<!_CCCL_TRAIT(is_integral, _Type), bool>::type;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_H
