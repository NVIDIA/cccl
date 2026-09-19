//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_RMW_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_RMW_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/backend.h>
#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Backend, class _Operand>
using __cuda_atomic_enable_generic_rmw = enable_if_t<(_Operand::__size <= _Backend::__widest_cas), bool>;

template <size_t _Size>
[[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL auto __cuda_atomic_rmw_type_impl() noexcept
{
  if constexpr (_Size == 8)
  {
    return uint8_t{};
  }
  else if constexpr (_Size == 16)
  {
    return uint16_t{};
  }
  else if constexpr (_Size == 32)
  {
    return uint32_t{};
  }
  else if constexpr (_Size == 64)
  {
    return uint64_t{};
  }
  else
  {
    static_assert(_Size == 128, "invalid generic RMW size");
    return __cuda_atomic_longlong2{};
  }
}

template <size_t _Size>
using __cuda_atomic_rmw_type_t = decltype(__cuda_atomic_rmw_type_impl<_Size>());

template <class _Type, class _RmwType>
[[nodiscard]] _CCCL_HOST_DEVICE_API _RmwType
__cuda_atomic_rmw_replace([[maybe_unused]] _RmwType __old, _Type __op, uint8_t __offset)
{
  if constexpr (sizeof(_Type) == sizeof(_RmwType))
  {
    return ::cuda::std::bit_cast<_RmwType>(__op);
  }
  else if constexpr (is_same_v<_RmwType, __cuda_atomic_longlong2>)
  {
    static_assert(sizeof(_Type) == sizeof(uint64_t), "only 64-bit atomics can be widened to 128 bits");
    using __logical_type                    = __cuda_atomic_rmw_type_t<sizeof(_Type) * 8>;
    (__offset == 0 ? __old.__x : __old.__y) = ::cuda::std::bit_cast<__logical_type>(__op);
    return __old;
  }
  else
  {
    using __logical_type      = __cuda_atomic_rmw_type_t<sizeof(_Type) * 8>;
    constexpr auto __sizemask = (_RmwType{1} << (sizeof(_Type) * 8)) - 1;
    const auto __value_mask   = __sizemask << __offset;
    const auto __op_bits      = static_cast<_RmwType>(::cuda::std::bit_cast<__logical_type>(__op));
    return (__old & ~__value_mask) | ((__op_bits << __offset) & __value_mask);
  }
}

template <class _Type, class _RmwType>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_rmw_extract(_RmwType __old, uint8_t __offset)
{
  if constexpr (sizeof(_Type) == sizeof(_RmwType))
  {
    return ::cuda::std::bit_cast<_Type>(__old);
  }
  else if constexpr (is_same_v<_RmwType, __cuda_atomic_longlong2>)
  {
    static_assert(sizeof(_Type) == sizeof(uint64_t), "only 64-bit atomics can be widened to 128 bits");
    using __logical_type = __cuda_atomic_rmw_type_t<sizeof(_Type) * 8>;
    return ::cuda::std::bit_cast<_Type>(static_cast<__logical_type>(__offset == 0 ? __old.__x : __old.__y));
  }
  else
  {
    using __logical_type      = __cuda_atomic_rmw_type_t<sizeof(_Type) * 8>;
    constexpr auto __sizemask = (_RmwType{1} << (sizeof(_Type) * 8)) - 1;
    const auto __old_bits     = static_cast<__logical_type>((__old >> __offset) & __sizemask);
    return ::cuda::std::bit_cast<_Type>(__old_bits);
  }
}

template <class _Type>
struct __cuda_atomic_rmw_update
{
  _Type __value;
  bool __apply;
};

template <class _Type>
struct __cuda_atomic_rmw_result
{
  _Type __old;
  bool __applied;
};

template <class _Backend,
          class _Pointee,
          class _Fn,
          class _Order,
          class _InitialOrder,
          class _Operand,
          class _RmwOperand,
          class _Sco>
_CCCL_HOST_DEVICE_API __cuda_atomic_rmw_result<__unv<_Pointee>> __cuda_atomic_rmw(
  _Backend __backend,
  _Pointee* __ptr,
  const _Fn& __op,
  _Order __order,
  _InitialOrder __initial_order,
  _Operand,
  _RmwOperand,
  _Sco __scope)
{
  using _Type = __unv<_Pointee>;
  static_assert(_Operand::__op == __cuda_atomic_operand::_b, "generic RMW requires a bitwise operand");
  static_assert(_RmwOperand::__op == __cuda_atomic_operand::_b, "generic RMW requires a bitwise CAS operand");
  static_assert(_Operand::__size <= _RmwOperand::__size, "generic RMW cannot use a narrower CAS operand");

  using __rmw_type    = __cuda_atomic_rmw_type_t<_RmwOperand::__size>;
  using __rmw_pointee = __copy_cv_t<_Pointee, __rmw_type>;

  __rmw_pointee* __aligned;
  uint8_t __offset;
  if constexpr (sizeof(_Type) == sizeof(__rmw_type))
  {
    __aligned = reinterpret_cast<__rmw_pointee*>(__ptr);
    __offset  = 0;
  }
  else
  {
    constexpr uintptr_t __alignmask = sizeof(__rmw_type) - 1;
    __aligned = reinterpret_cast<__rmw_pointee*>(reinterpret_cast<uintptr_t>(__ptr) & ~__alignmask); // NOLINT
    __offset  = static_cast<uint8_t>((reinterpret_cast<uintptr_t>(__ptr) & __alignmask) * 8);
  }

  __rmw_type __old;
  __cuda_atomic_load(__backend, __aligned, __old, __initial_order, _RmwOperand{}, __scope, __cuda_atomic_mmio_disable{});

  while (true)
  {
    const _Type __logical_old                      = ::cuda::std::__cuda_atomic_rmw_extract<_Type>(__old, __offset);
    const __cuda_atomic_rmw_update<_Type> __update = __op(__logical_old);
    if (!__update.__apply)
    {
      return {__logical_old, false};
    }

    const __rmw_type __attempt = ::cuda::std::__cuda_atomic_rmw_replace(__old, __update.__value, __offset);
    if (__cuda_atomic_compare_exchange(
          __backend, __aligned, __old, __old, __attempt, __cuda_atomic_cas_weak{}, __order, _RmwOperand{}, __scope))
    {
      return {__logical_old, true};
    }
  }
}

template <class _Type, class _Fn>
struct __cuda_atomic_rmw_op
{
  _Fn __op;

  [[nodiscard]] _CCCL_HOST_DEVICE_API __cuda_atomic_rmw_update<_Type> operator()(_Type __old) const
  {
    return {__op(__old), true};
  }
};

template <class _Type>
struct __cuda_atomic_compare_exchange_op
{
  _Type __cmp;
  _Type __op;

  [[nodiscard]] _CCCL_HOST_DEVICE_API __cuda_atomic_rmw_update<_Type> operator()(_Type __old) const
  {
    return {__op, __old == __cmp};
  }
};

template <class _Type, class _Op>
struct __cuda_atomic_op_bind
{
  _Type __val;

  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type operator()(_Type __old) const
  {
    return _Op{}(__val, __old);
  }
};

struct __cuda_atomic_op_store
{
  template <class _Type>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type _CCCL_STATIC_CALL_OPERATOR(_Type __val, _Type)
  {
    return __val;
  }
};

struct __cuda_atomic_op_fetch_add
{
  template <class _Type>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type _CCCL_STATIC_CALL_OPERATOR(_Type __op, _Type __old)
  {
    if constexpr (is_integral_v<_Type>)
    {
      using __unsigned_type = make_unsigned_t<_Type>;
      const auto __op_bits  = ::cuda::std::bit_cast<__unsigned_type>(__op);
      const auto __old_bits = ::cuda::std::bit_cast<__unsigned_type>(__old);
      return ::cuda::std::bit_cast<_Type>(static_cast<__unsigned_type>(__old_bits + __op_bits));
    }
    else
    {
      return __old + __op;
    }
  }
};

struct __cuda_atomic_op_fetch_sub
{
  template <class _Type>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type _CCCL_STATIC_CALL_OPERATOR(_Type __op, _Type __old)
  {
    if constexpr (is_integral_v<_Type>)
    {
      using __unsigned_type = make_unsigned_t<_Type>;
      const auto __op_bits  = ::cuda::std::bit_cast<__unsigned_type>(__op);
      const auto __old_bits = ::cuda::std::bit_cast<__unsigned_type>(__old);
      return ::cuda::std::bit_cast<_Type>(static_cast<__unsigned_type>(__old_bits - __op_bits));
    }
    else
    {
      return __old - __op;
    }
  }
};

struct __cuda_atomic_op_fetch_min
{
  template <class _Type>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type _CCCL_STATIC_CALL_OPERATOR(_Type __op, _Type __old)
  {
    return ::cuda::std::__cuda_atomic_less(__op, __old) ? __op : __old;
  }
};

struct __cuda_atomic_op_fetch_max
{
  template <class _Type>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _Type _CCCL_STATIC_CALL_OPERATOR(_Type __op, _Type __old)
  {
    return ::cuda::std::__cuda_atomic_less(__old, __op) ? __op : __old;
  }
};

template <class _Backend, class _Pointee, class _Fn, class _Order, class _Operand, class _Sco>
_CCCL_HOST_DEVICE_API __unv<_Pointee>
__cuda_atomic_fetch_update(_Backend __backend, _Pointee* __ptr, const _Fn& __op, _Order __order, _Operand, _Sco __scope)
{
  using _Type = __unv<_Pointee>;
  static_assert(sizeof(_Type) * 8 == _Operand::__size, "generic RMW requires matching type and operand sizes");
  constexpr size_t __rmw_size =
    _Operand::__size < _Backend::__smallest_cas ? _Backend::__smallest_cas : _Operand::__size;
  static_assert(__rmw_size <= _Backend::__widest_cas, "generic RMW requires a supported CAS width");

  using __bitwise_operand = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, _Operand::__size>;
  using __rmw_operand     = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, __rmw_size>;
  return ::cuda::std::__cuda_atomic_rmw(
           __backend,
           __ptr,
           __cuda_atomic_rmw_op<_Type, _Fn>{__op},
           __order,
           ::cuda::std::__cuda_atomic_initial_load_order(__order),
           __bitwise_operand{},
           __rmw_operand{},
           __scope)
    .__old;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_RMW_H
