//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
#define _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/fold.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_object.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/auto_cast.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// clang-tidy warns about for example _LIBCUDACXX_AUTO_CAST(++_Tp::value) being repeated multiple times in the macro
// expansion.
// NOLINTBEGIN(bugprone-macro-repeated-side-effects)

template <auto _Xp, class = remove_cvref_t<decltype(_Xp)>>
struct __constant_wrapper;

template <class _Tp>
inline constexpr bool __is_cuda_std_constant_wrapper_v = false;
template <auto _Xp, class _Tp>
inline constexpr bool __is_cuda_std_constant_wrapper_v<__constant_wrapper<_Xp, _Tp>> = true;

// MSVC 2019 rejects constant pointer arguments in the partial-specialization probe.
#if _CCCL_COMPILER(MSVC, <, 19, 30)
template <class _Tp, auto = _Tp::value>
_CCCL_HOST_DEVICE_API true_type __cw_is_constexpr_param(int);
template <class>
_CCCL_HOST_DEVICE_API false_type __cw_is_constexpr_param(...);
template <class _Tp, class = void>
inline constexpr bool __is_constexpr_param_v = decltype(__cw_is_constexpr_param<_Tp>(0))::value;
#else // ^^^ _CCCL_COMPILER(MSVC, <, 19, 30) ^^^ / vvv !_CCCL_COMPILER(MSVC, <, 19, 30) vvv
template <class _Tp, class = void>
inline constexpr bool __is_constexpr_param_v = false;
template <class _Tp>
inline constexpr bool __is_constexpr_param_v<_Tp, void_t<__constant_wrapper<_Tp::value>>> = true;
#endif // ^^^ !_CCCL_COMPILER(MSVC, <, 19, 30) ^^^

template <auto _Xp>
inline constexpr __constant_wrapper<_Xp> __cw;

struct __cw_operators
{
  // unary operators
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(+_Tp::value)>{})
  operator+(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(-_Tp::value)>{})
  operator-(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(~_Tp::value)>{})
  operator~(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(!_Tp::value)>{})
  operator!(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(&_Tp::value)>{})
  operator&(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(*_Tp::value)>{})
  operator*(_Tp) noexcept
  {
    return {};
  }

  // pseudo-mutators
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(++_Tp::value)>{})
  operator++(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Tp::value++)>{})
  operator++(_Tp, int) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(--_Tp::value)>{})
  operator--(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_cuda_std_constant_wrapper_v<_Tp> _CCCL_AND __is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Tp::value--)>{})
  operator--(_Tp, int) noexcept
  {
    return {};
  }

  // binary operators
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value + _Rp::value)>{})
  operator+(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value - _Rp::value)>{})
  operator-(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Lp::value* _Rp::value)>{})
  operator*(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value / _Rp::value)>{})
  operator/(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value % _Rp::value)>{})
  operator%(_Lp, _Rp) noexcept
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value << _Rp::value)>{})
  operator<<(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value >> _Rp::value)>{})
  operator>>(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Lp::value& _Rp::value)>{})
  operator&(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value | _Rp::value)>{})
  operator|(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value ^ _Rp::value)>{})
  operator^(_Lp, _Rp) noexcept
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(
    (__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
      _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp> _CCCL_AND(
        !is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>))
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Lp::value&& _Rp::value)>{})
  operator&&(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(
    (__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
      _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp> _CCCL_AND(
        !is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>))
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value || _Rp::value)>{})
  operator||(_Lp, _Rp) noexcept
  {
    return {};
  }

  // comparisons
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value <=> _Rp::value)>{})
  operator<=>(_Lp, _Rp) noexcept
  {
    return {};
  }
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value < _Rp::value)>{})
  operator<(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value <= _Rp::value)>{})
  operator<=(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value == _Rp::value)>{})
  operator==(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value != _Rp::value)>{})
  operator!=(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value > _Rp::value)>{})
  operator>(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value >= _Rp::value)>{})
  operator>=(_Lp, _Rp) noexcept
  {
    return {};
  }

  // Use enable_if, because default template arguments may not be used in template friend declarations in C++17.
  template <class _Lp, class _Rp>
  friend enable_if_t<(__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                     && __is_constexpr_param_v<_Lp> && __is_constexpr_param_v<_Rp>>
  operator,(_Lp, _Rp) = delete;

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES((__is_cuda_std_constant_wrapper_v<_Lp> || __is_cuda_std_constant_wrapper_v<_Rp>)
                   _CCCL_AND __is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr decltype(__constant_wrapper<
                                                                _LIBCUDACXX_AUTO_CAST(_Lp::value->*_Rp::value)>{})
  operator->*(_Lp, _Rp) noexcept
  {
    return {};
  }

#if _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVRTC) || _CCCL_COMPILER(NVHPC)
  // EDG loses the pointee's cv-qualifiers when built-in operator->* uses a user-defined conversion. Apply the operator
  // to the stored pointer directly for runtime data-member pointers.
  _CCCL_TEMPLATE(class _Lp, class _Mp, class _Cp)
  _CCCL_REQUIRES(
    __is_cuda_std_constant_wrapper_v<_Lp> _CCCL_AND is_pointer_v<decltype(_Lp::value)> _CCCL_AND is_object_v<_Mp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend constexpr auto operator->*(_Lp, _Mp _Cp::* __pm) noexcept
    -> decltype(_Lp::value->*__pm)
  {
    return _Lp::value->*__pm;
  }
#endif // _CCCL_CUDA_COMPILER(NVCC) || _CCCL_COMPILER(NVRTC) || _CCCL_COMPILER(NVHPC)
};

// MSVC rejects some constant invocations in partial specializations. Probe a default template argument instead.
#if _CCCL_COMPILER(MSVC)
template <class _Fn, class... _Args, auto = _LIBCUDACXX_AUTO_CAST(::cuda::std::invoke(_Fn::value, _Args::value...))>
_CCCL_HOST_DEVICE_API true_type __cw_is_constexpr_callable(int);
template <class, class...>
_CCCL_HOST_DEVICE_API false_type __cw_is_constexpr_callable(...);
template <class _Fn, class _Void, class... _Args>
inline constexpr bool __cw_is_constexpr_callable_v = decltype(__cw_is_constexpr_callable<_Fn, _Args...>(0))::value;
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
template <class _Fn, class _Void, class... _Args>
inline constexpr bool __cw_is_constexpr_callable_v = false;
template <class _Fn, class... _Args>
inline constexpr bool __cw_is_constexpr_callable_v<
  _Fn,
  void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(::cuda::std::invoke(_Fn::value, _Args::value...))>>,
  _Args...> = true;
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

template <class _Vp, class _Void, class... _Args>
inline constexpr bool __cw_is_constexpr_indexable_v = false;
#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class _Vp, class... _Args>
inline constexpr bool
  __cw_is_constexpr_indexable_v<_Vp,
                                void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Vp::value[_Args::value...])>>,
                                _Args...> = true;
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <class _Vp, class _Arg>
inline constexpr bool
  __cw_is_constexpr_indexable_v<_Vp, void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Vp::value[_Arg::value])>>, _Arg> =
    true;
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

template <class _Vp, class _Void, class... _Args>
inline constexpr bool __cw_is_indexable_v = false;
#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class _Vp, class... _Args>
inline constexpr bool
  __cw_is_indexable_v<_Vp, void_t<decltype(_Vp::value[::cuda::std::declval<_Args>()...])>, _Args...> = true;
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <class _Vp, class _Arg>
inline constexpr bool __cw_is_indexable_v<_Vp, void_t<decltype(_Vp::value[::cuda::std::declval<_Arg>()])>, _Arg> = true;
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

template <auto _Xp, class _Tp>
struct __constant_wrapper : __cw_operators
{
  using type       = __constant_wrapper;
  using value_type = _Tp;

  // msvc doesn't evaluate correctly decltype(auto) nor decltype((_Xp)), so we need to set the type by hand.
#if _CCCL_COMPILER(MSVC)
  static constexpr conditional_t<is_class_v<_Tp>, const _Tp&, const _Tp> value = (_Xp);
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
  static constexpr decltype((_Xp)) value = (_Xp);
#endif // ^^^ !_CCCL_COMPILER(MSVC) ^^^

  // [const.wrap.class] mandates this signature: operator= is a constant-expression
  // operation that yields a new constant_wrapper, so it must not return *this.
  // NOLINTBEGIN(misc-unconventional-assign-operator)
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value = _Rp::value)>{})
  operator=(_Rp) const noexcept
  {
    return {};
  }
  // NOLINTEND(misc-unconventional-assign-operator)

  _CCCL_HOST_DEVICE_API constexpr operator decltype(value)() const noexcept
  {
    return (_Xp);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr decltype(value) __get() noexcept
  {
    return (_Xp);
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...> _CCCL_AND
                   __cw_is_constexpr_callable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>)
  _CCCL_HOST_DEVICE_API constexpr auto _CCCL_STATIC_CALL_OPERATOR(_Args&&...) noexcept
  {
    return __constant_wrapper<_LIBCUDACXX_AUTO_CAST(::cuda::std::invoke(__get(), remove_cvref_t<_Args>::value...))>{};
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...>
                    && __cw_is_constexpr_callable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>) )
                   _CCCL_AND is_invocable_v<const _Tp&, _Args&&...>)
  _CCCL_HOST_DEVICE_API constexpr decltype(auto)
  _CCCL_STATIC_CALL_OPERATOR(_Args&&... __args) noexcept(::cuda::std::is_nothrow_invocable_v<const _Tp&, _Args...>)
  {
    return ::cuda::std::invoke(__get(), ::cuda::std::forward<_Args>(__args)...);
  }

#if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...> _CCCL_AND
                   __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr __constant_wrapper<
    _LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Args>::value...])>
  _CCCL_STATIC_SUBSCRIPT_OPERATOR(_Args&&...) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...>
                    && __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>) )
                   _CCCL_AND __cw_is_indexable_v<__constant_wrapper, void, _Args...>)
  _CCCL_HOST_DEVICE_API constexpr decltype(auto)
  _CCCL_STATIC_SUBSCRIPT_OPERATOR(_Args&&... __args) noexcept(noexcept(value[::cuda::std::forward<_Args>(__args)...]))
  {
    return __get()[::cuda::std::forward<_Args>(__args)...];
  }
#else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(__is_constexpr_param_v<remove_cvref_t<_Arg>> _CCCL_AND
                   __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Arg>>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API constexpr __constant_wrapper<_LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Arg>::value])>
  _CCCL_STATIC_SUBSCRIPT_OPERATOR(_Arg&&) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES((!(__is_constexpr_param_v<remove_cvref_t<_Arg>>
                    && __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Arg>>) )
                   _CCCL_AND __cw_is_indexable_v<__constant_wrapper, void, _Arg>)
  _CCCL_HOST_DEVICE_API constexpr decltype(auto)
  _CCCL_STATIC_SUBSCRIPT_OPERATOR(_Arg&& __arg) noexcept(noexcept(value[::cuda::std::forward<_Arg>(__arg)]))
  {
    return __get()[::cuda::std::forward<_Arg>(__arg)];
  }
#endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value += _Rp::value)>{})
  operator+=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value -= _Rp::value)>{})
  operator-=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value *= _Rp::value)>{})
  operator*=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value /= _Rp::value)>{})
  operator/=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value %= _Rp::value)>{})
  operator%=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value &= _Rp::value)>{})
  operator&=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value |= _Rp::value)>{})
  operator|=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value ^= _Rp::value)>{})
  operator^=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                           value <<= _Rp::value)>{}) operator<<=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                           value >>= _Rp::value)>{}) operator>>=(_Rp) const noexcept
  {
    return {};
  }
};

// NOLINTEND(bugprone-macro-repeated-side-effects)

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
