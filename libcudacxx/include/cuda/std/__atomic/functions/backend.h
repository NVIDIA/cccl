//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_BACKEND_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_BACKEND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_extended_floating_point.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/climits>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

enum class __cuda_atomic_order
{
  _relaxed,
  _release,
  _acquire,
  _acq_rel,
  _seq_cst,
  _volatile,
};

template <__cuda_atomic_order _Order>
using __cuda_atomic_order_tag _CCCL_NODEBUG = integral_constant<__cuda_atomic_order, _Order>;

using __cuda_atomic_order_relaxed _CCCL_NODEBUG  = __cuda_atomic_order_tag<__cuda_atomic_order::_relaxed>;
using __cuda_atomic_order_release _CCCL_NODEBUG  = __cuda_atomic_order_tag<__cuda_atomic_order::_release>;
using __cuda_atomic_order_acquire _CCCL_NODEBUG  = __cuda_atomic_order_tag<__cuda_atomic_order::_acquire>;
using __cuda_atomic_order_acq_rel _CCCL_NODEBUG  = __cuda_atomic_order_tag<__cuda_atomic_order::_acq_rel>;
using __cuda_atomic_order_seq_cst _CCCL_NODEBUG  = __cuda_atomic_order_tag<__cuda_atomic_order::_seq_cst>;
using __cuda_atomic_order_volatile _CCCL_NODEBUG = __cuda_atomic_order_tag<__cuda_atomic_order::_volatile>;

template <class _Order>
struct __cuda_atomic_ptx_order : _Order
{
  bool __was_seq_cst;

  _CCCL_HOST_DEVICE_API constexpr explicit __cuda_atomic_ptx_order(bool __was_seq_cst_) noexcept
      : __was_seq_cst(__was_seq_cst_)
  {}
};

using __cuda_atomic_ptx_order_relaxed _CCCL_NODEBUG = __cuda_atomic_ptx_order<__cuda_atomic_order_relaxed>;
using __cuda_atomic_ptx_order_release _CCCL_NODEBUG = __cuda_atomic_ptx_order<__cuda_atomic_order_release>;
using __cuda_atomic_ptx_order_acquire _CCCL_NODEBUG = __cuda_atomic_ptx_order<__cuda_atomic_order_acquire>;
using __cuda_atomic_ptx_order_acq_rel _CCCL_NODEBUG = __cuda_atomic_ptx_order<__cuda_atomic_order_acq_rel>;

struct __cuda_atomic_operation_load
{};

struct __cuda_atomic_operation_store
{};

struct __cuda_atomic_operation_rmw
{};

struct __cuda_atomic_runtime_cas_order
{
  memory_order __success;
  memory_order __failure;
};

template <class _Success, class _Failure>
struct __cuda_atomic_cas_order
{
  using __success _CCCL_NODEBUG = _Success;
  using __failure _CCCL_NODEBUG = _Failure;
};

struct __cuda_atomic_cas_strong
{};

struct __cuda_atomic_cas_weak : __cuda_atomic_cas_strong
{};

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool __cuda_atomic_cas_is_weak(__cuda_atomic_cas_strong)
{
  return false;
}

[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr bool __cuda_atomic_cas_is_weak(__cuda_atomic_cas_weak)
{
  return true;
}

template <class _Order>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto __cuda_atomic_initial_load_order(_Order)
{
  if constexpr (is_same_v<_Order, __cuda_atomic_order_volatile>)
  {
    return __cuda_atomic_order_volatile{};
  }
  else if constexpr (is_same_v<_Order, memory_order> || is_same_v<_Order, __cuda_atomic_runtime_cas_order>)
  {
    return memory_order_relaxed;
  }
  else
  {
    return __cuda_atomic_order_relaxed{};
  }
}

template <class _Order>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto __cuda_atomic_initial_load_order(__cuda_atomic_ptx_order<_Order>)
{
  return __cuda_atomic_ptx_order_relaxed{false};
}

template <class _Success, class _Failure>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
__cuda_atomic_compare_exchange_initial_load_order(__cuda_atomic_cas_order<_Success, _Failure>)
{
  return _Failure{};
}

// Compare-exchange may fail after the initial load without issuing a CAS, so that load must satisfy the failure order.
template <class _Order>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto __cuda_atomic_compare_exchange_initial_load_order(_Order __order)
{
  if constexpr (is_same_v<_Order, __cuda_atomic_runtime_cas_order>)
  {
    return __order.__failure;
  }
  else if constexpr (is_same_v<_Order, memory_order>)
  {
    return __order == memory_order_release
           ? memory_order_relaxed
           : (__order == memory_order_acq_rel ? memory_order_acquire : __order);
  }
  else if constexpr (is_same_v<_Order, __cuda_atomic_order_acquire> || is_same_v<_Order, __cuda_atomic_order_acq_rel>)
  {
    return __cuda_atomic_order_acquire{};
  }
  else
  {
    return ::cuda::std::__cuda_atomic_initial_load_order(__order);
  }
}

template <class _Order>
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto
__cuda_atomic_compare_exchange_initial_load_order(__cuda_atomic_ptx_order<_Order> __order)
{
  if constexpr (is_same_v<_Order, __cuda_atomic_order_acquire> || is_same_v<_Order, __cuda_atomic_order_acq_rel>)
  {
    return __cuda_atomic_ptx_order_acquire{__order.__was_seq_cst};
  }
  else
  {
    return __cuda_atomic_ptx_order_relaxed{false};
  }
}

template <bool _Volatile>
using __cuda_atomic_mmio_tag = integral_constant<bool, _Volatile>;

using __cuda_atomic_mmio_enable  = __cuda_atomic_mmio_tag<true>;
using __cuda_atomic_mmio_disable = __cuda_atomic_mmio_tag<false>;

enum class __cuda_atomic_operand
{
  _f,
  _s,
  _u,
  _b,
};

template <__cuda_atomic_operand _Op, size_t _Size>
struct __cuda_atomic_operand_tag
{
  static constexpr auto __op   = _Op;
  static constexpr auto __size = _Size;
};

using __cuda_atomic_operand_f8   = __cuda_atomic_operand_tag<__cuda_atomic_operand::_f, 8>;
using __cuda_atomic_operand_s8   = __cuda_atomic_operand_tag<__cuda_atomic_operand::_s, 8>;
using __cuda_atomic_operand_u8   = __cuda_atomic_operand_tag<__cuda_atomic_operand::_u, 8>;
using __cuda_atomic_operand_b8   = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, 8>;
using __cuda_atomic_operand_f16  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_f, 16>;
using __cuda_atomic_operand_s16  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_s, 16>;
using __cuda_atomic_operand_u16  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_u, 16>;
using __cuda_atomic_operand_b16  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, 16>;
using __cuda_atomic_operand_f32  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_f, 32>;
using __cuda_atomic_operand_s32  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_s, 32>;
using __cuda_atomic_operand_u32  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_u, 32>;
using __cuda_atomic_operand_b32  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, 32>;
using __cuda_atomic_operand_f64  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_f, 64>;
using __cuda_atomic_operand_s64  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_s, 64>;
using __cuda_atomic_operand_u64  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_u, 64>;
using __cuda_atomic_operand_b64  = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, 64>;
using __cuda_atomic_operand_f128 = __cuda_atomic_operand_tag<__cuda_atomic_operand::_f, 128>;
using __cuda_atomic_operand_s128 = __cuda_atomic_operand_tag<__cuda_atomic_operand::_s, 128>;
using __cuda_atomic_operand_u128 = __cuda_atomic_operand_tag<__cuda_atomic_operand::_u, 128>;
using __cuda_atomic_operand_b128 = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, 128>;

template <class _AtomicType, class _OpTag>
struct __cuda_atomic_operand_deduction
{
  using __type = _AtomicType;
  using __tag  = _OpTag;
};

struct _CCCL_ALIGNAS(16) __cuda_atomic_longlong2
{
  uint64_t __x;
  uint64_t __y;

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr __cuda_atomic_longlong2
  operator&(__cuda_atomic_longlong2 __lhs, __cuda_atomic_longlong2 __rhs) noexcept
  {
    return {__lhs.__x & __rhs.__x, __lhs.__y & __rhs.__y};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr __cuda_atomic_longlong2
  operator|(__cuda_atomic_longlong2 __lhs, __cuda_atomic_longlong2 __rhs) noexcept
  {
    return {__lhs.__x | __rhs.__x, __lhs.__y | __rhs.__y};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr __cuda_atomic_longlong2
  operator^(__cuda_atomic_longlong2 __lhs, __cuda_atomic_longlong2 __rhs) noexcept
  {
    return {__lhs.__x ^ __rhs.__x, __lhs.__y ^ __rhs.__y};
  }
};

template <class _Type>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __cuda_atomic_deduce_bitwise_impl() noexcept
{
  using __tag = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, sizeof(_Type) * CHAR_BIT>;
  if constexpr (sizeof(_Type) == 16)
  {
    return __cuda_atomic_operand_deduction<__cuda_atomic_longlong2, __tag>{};
  }
  else
  {
    return __cuda_atomic_operand_deduction<__make_nbit_uint_t<sizeof(_Type) * CHAR_BIT>, __tag>{};
  }
}

// TODO: Once CUDA 12.0 is no longer supported, factor the repeated decltype below into a common deduction alias.
// CUDA 12.0 cudafe can substitute an unrelated alias for an intermediate alias template in large translation units.
template <class _Type>
using __cuda_atomic_deduce_bitwise_t = typename decltype(__cuda_atomic_deduce_bitwise_impl<_Type>())::__type;

template <class _Type>
using __cuda_atomic_deduce_bitwise_tag_t = typename decltype(__cuda_atomic_deduce_bitwise_impl<_Type>())::__tag;

template <class _Type>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __cuda_atomic_deduce_arithmetic_impl() noexcept
{
  constexpr bool __is_floating = is_floating_point_v<_Type> || __is_extended_floating_point_v<_Type>;
  constexpr auto __op =
    __is_floating ? __cuda_atomic_operand::_f
                  : (is_signed_v<_Type> && sizeof(_Type) != 8 ? __cuda_atomic_operand::_s : __cuda_atomic_operand::_u);
  using __tag = __cuda_atomic_operand_tag<__op, sizeof(_Type) * CHAR_BIT>;
  if constexpr (__is_floating || sizeof(_Type) == 16)
  {
    return __cuda_atomic_operand_deduction<_Type, __tag>{};
  }
  else
  {
    return __cuda_atomic_operand_deduction<__make_nbit_int_t<sizeof(_Type) * CHAR_BIT, is_signed_v<_Type>>, __tag>{};
  }
}

template <class _Type>
using __cuda_atomic_deduce_arithmetic_t = typename decltype(__cuda_atomic_deduce_arithmetic_impl<_Type>())::__type;

template <class _Type>
using __cuda_atomic_deduce_arithmetic_tag_t = typename decltype(__cuda_atomic_deduce_arithmetic_impl<_Type>())::__tag;

template <class _Type>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __cuda_atomic_deduce_minmax_impl() noexcept
{
  constexpr bool __is_floating = is_floating_point_v<_Type> || __is_extended_floating_point_v<_Type>;
  constexpr auto __op = __is_floating ? __cuda_atomic_operand::_f
                                      : (is_signed_v<_Type> ? __cuda_atomic_operand::_s : __cuda_atomic_operand::_u);
  using __tag         = __cuda_atomic_operand_tag<__op, sizeof(_Type) * CHAR_BIT>;
  if constexpr (__is_floating || sizeof(_Type) == 16)
  {
    return __cuda_atomic_operand_deduction<_Type, __tag>{};
  }
  else
  {
    return __cuda_atomic_operand_deduction<__make_nbit_int_t<sizeof(_Type) * CHAR_BIT, is_signed_v<_Type>>, __tag>{};
  }
}

template <class _Type>
using __cuda_atomic_deduce_minmax_t = typename decltype(__cuda_atomic_deduce_minmax_impl<_Type>())::__type;

template <class _Type>
using __cuda_atomic_deduce_minmax_tag_t = typename decltype(__cuda_atomic_deduce_minmax_impl<_Type>())::__tag;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_BACKEND_H
