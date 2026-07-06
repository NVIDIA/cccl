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

#if _CCCL_STD_VER >= 2020 && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__type_traits/fold.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/remove_const.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/__utility/auto_cast.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/__utility/integer_sequence.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// operator may not be a static member function
// a host member cannot be directly read in a __device__/__global__ function
_CCCL_BEGIN_NV_DIAG_SUPPRESS(342, 20094)

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_NVHPC(static_member_operator_not_allowed)

// clang-tidy warns about for example _LIBCUDACXX_AUTO_CAST(++_Tp::value) being repeated multiple times in the macro
// expansion.
// NOLINTBEGIN(bugprone-macro-repeated-side-effects)

template <auto _Xp, class = remove_cvref_t<decltype(_Xp)>>
struct __constant_wrapper;

template <class _Tp, class = void>
inline constexpr bool __is_constexpr_param_v = false;
template <class _Tp>
inline constexpr bool __is_constexpr_param_v<_Tp, void_t<__constant_wrapper<_Tp::value>>> = true;

template <auto _Xp>
inline constexpr __constant_wrapper<_Xp> __cw;

struct __cw_operators
{
  // unary operators
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<
                                                                      _LIBCUDACXX_AUTO_CAST(+_Tp::value)>{})
  operator+(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<
                                                                      _LIBCUDACXX_AUTO_CAST(-_Tp::value)>{})
  operator-(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<
                                                                      _LIBCUDACXX_AUTO_CAST(~_Tp::value)>{})
  operator~(_Tp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<
                                                                      _LIBCUDACXX_AUTO_CAST(!_Tp::value)>{})
  operator!(_Tp) noexcept
  {
    return {};
  }
  // todo(dabayer): nvcc 13.1-13.3 needs this to be concept, otherwise cudafe++ produces invalid input file for the host
  // compiler for code like `constant_wrapper<&v>`. Try to find a workaround that could work even in C++17.
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp> _CCCL_AND requires { typename __constant_wrapper<(&_Tp::value)>; })
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL auto operator&(_Tp) noexcept
  {
    return __constant_wrapper<(&_Tp::value)>{};
  }
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Tp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<
                                                                      _LIBCUDACXX_AUTO_CAST(*_Tp::value)>{})
  operator*(_Tp) noexcept
  {
    return {};
  }

  // binary operators
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value + _Rp::value)>{}) operator+(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value - _Rp::value)>{}) operator-(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value* _Rp::value)>{}) operator*(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value / _Rp::value)>{}) operator/(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value % _Rp::value)>{}) operator%(_Lp, _Rp) noexcept
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value << _Rp::value)>{}) operator<<(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value >> _Rp::value)>{}) operator>>(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value& _Rp::value)>{}) operator&(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value | _Rp::value)>{}) operator|(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value ^ _Rp::value)>{}) operator^(_Lp, _Rp) noexcept
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp> _CCCL_AND(
    !is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>))
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value&& _Rp::value)>{}) operator&&(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp> _CCCL_AND(
    !is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>))
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value || _Rp::value)>{}) operator||(_Lp, _Rp) noexcept
  {
    return {};
  }

  // comparisons
#  if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                                        _Lp::value <=> _Rp::value)>{})
  operator<=>(_Lp, _Rp) noexcept
  {
    return {};
  }
#  endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value < _Rp::value)>{}) operator<(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value <= _Rp::value)>{}) operator<=(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value == _Rp::value)>{}) operator==(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value != _Rp::value)>{}) operator!=(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value > _Rp::value)>{}) operator>(_Lp, _Rp) noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value >= _Rp::value)>{}) operator>=(_Lp, _Rp) noexcept
  {
    return {};
  }

  // Use enable_if, because default template arguments may not be used in template friend declarations in C++17.
  template <class _Lp, class _Rp>
  friend enable_if_t<__is_constexpr_param_v<_Lp> && __is_constexpr_param_v<_Rp>> operator,(_Lp, _Rp) = delete;

  _CCCL_TEMPLATE(class _Lp, class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Lp> _CCCL_AND __is_constexpr_param_v<_Rp>)
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API friend _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(
                                                          _Lp::value->*_Rp::value)>{}) operator->*(_Lp, _Rp) noexcept
  {
    return {};
  }
};

template <class _Fn, class _Void, class... _Args>
inline constexpr bool __cw_is_constexpr_callable_v = false;
template <class _Fn, class... _Args>
inline constexpr bool __cw_is_constexpr_callable_v<
  _Fn,
  void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(::cuda::std::invoke(_Fn::value, _Args::value...))>>,
  _Args...> = true;

template <class _Vp, class _Void, class... _Args>
inline constexpr bool __cw_is_constexpr_indexable_v = false;
#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class _Vp, class... _Args>
inline constexpr bool
  __cw_is_constexpr_indexable_v<_Vp,
                                void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Vp::value[_Args::value...])>>,
                                _Args...> = true;
#  else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <class _Vp, class _Arg>
inline constexpr bool
  __cw_is_constexpr_indexable_v<_Vp, void_t<__constant_wrapper<_LIBCUDACXX_AUTO_CAST(_Vp::value[_Arg::value])>>, _Arg> =
    true;
#  endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

template <class _Vp, class _Void, class... _Args>
inline constexpr bool __cw_is_indexable_v = false;
#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
template <class _Vp, class... _Args>
inline constexpr bool
  __cw_is_indexable_v<_Vp, void_t<decltype(_Vp::value[::cuda::std::declval<_Args>()...])>, _Args...> = true;
#  else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
template <class _Vp, class _Arg>
inline constexpr bool __cw_is_indexable_v<_Vp, void_t<decltype(_Vp::value[::cuda::std::declval<_Arg>()])>, _Arg> = true;
#  endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

template <auto _Xp, class _Tp>
struct __constant_wrapper : __cw_operators
{
  using type = __constant_wrapper;

  // This should be just `decltype(_Xp)`, but nvcc 12.0 adds `const` to the type.
  using value_type = remove_const_t<decltype(_Xp)>;

  static constexpr decltype((_Xp)) value = (_Xp);

  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value = _Rp::value)>{})
    operator=(_Rp) const noexcept
  {
    return {};
  }

  _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL operator decltype(value)() const noexcept
  {
    return value;
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...> _CCCL_AND
                   __cw_is_constexpr_callable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>)
#  if _CCCL_HAS_STATIC_CALL_OPERATOR()
  _CCCL_HOST_DEVICE_API static constexpr auto operator()(_Args&&...) noexcept
#  else // ^^^ _CCCL_HAS_STATIC_CALL_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_CALL_OPERATOR() vvv
  _CCCL_HOST_DEVICE_API constexpr auto operator()(_Args&&...) const noexcept
#  endif // ^^^ !_CCCL_HAS_STATIC_CALL_OPERATOR() ^^^
  {
    return __constant_wrapper<_LIBCUDACXX_AUTO_CAST(::cuda::std::invoke(value, remove_cvref_t<_Args>::value...))>{};
  }

  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...>
                    && __cw_is_constexpr_callable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>) )
                   _CCCL_AND is_invocable_v<const _Tp&, _Args&&...>)
#  if _CCCL_HAS_STATIC_CALL_OPERATOR()
  _CCCL_HOST_DEVICE_API static constexpr decltype(auto) operator()(_Args&&... __args)
#  else // ^^^ _CCCL_HAS_STATIC_CALL_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_CALL_OPERATOR() vvv
  _CCCL_HOST_DEVICE_API constexpr decltype(auto) operator()(_Args&&... __args) const
#  endif // ^^^ !_CCCL_HAS_STATIC_CALL_OPERATOR() ^^^
    noexcept(::cuda::std::is_nothrow_invocable_v<const _Tp&, _Args...>)
  {
    return ::cuda::std::invoke(value, ::cuda::std::forward<_Args>(__args)...);
  }

#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...> _CCCL_AND
                   __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>)
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL static __constant_wrapper<_LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Args>::value...])>
    operator[](_Args&&...) noexcept
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Args>::value...])>
    operator[](_Args&&...) const noexcept
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
  {
    return {};
  }
  _CCCL_TEMPLATE(class... _Args)
  _CCCL_REQUIRES((!(__fold_and_v<__is_constexpr_param_v<remove_cvref_t<_Args>>...>
                    && __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Args>...>) )
                   _CCCL_AND __cw_is_indexable_v<__constant_wrapper, void, _Args...>)
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  _CCCL_HOST_DEVICE_API static constexpr decltype(auto) operator[](_Args&&... __args)
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  _CCCL_HOST_DEVICE_API constexpr decltype(auto) operator[](_Args&&... __args) const
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
    noexcept(noexcept(value[::cuda::std::forward<_Args>(__args)...]))
  {
    return value[::cuda::std::forward<_Args>(__args)...];
  }
#  else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES(__is_constexpr_param_v<remove_cvref_t<_Arg>> _CCCL_AND
                   __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Arg>>)
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]]
  _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL static __constant_wrapper<_LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Arg>::value])>
    operator[](_Arg&&) noexcept
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(value[remove_cvref_t<_Arg>::value])>
    operator[](_Arg&&) const noexcept
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Arg)
  _CCCL_REQUIRES((!(__is_constexpr_param_v<remove_cvref_t<_Arg>>
                    && __cw_is_constexpr_indexable_v<__constant_wrapper, void, remove_cvref_t<_Arg>>) )
                   _CCCL_AND __cw_is_indexable_v<__constant_wrapper, void, _Arg>)
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  _CCCL_HOST_DEVICE_API static constexpr decltype(auto) operator[](_Arg&& __arg)
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  _CCCL_HOST_DEVICE_API constexpr decltype(auto) operator[](_Arg&& __arg) const
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
    noexcept(noexcept(value[::cuda::std::forward<_Arg>(__arg)]))
  {
    return value[::cuda::std::forward<_Arg>(__arg)];
  }
#  endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

  // pseudo-mutators
  template <class _This = __constant_wrapper>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(++_This::value)>
  operator++() const noexcept
  {
    return {};
  }
  template <class _This = __constant_wrapper>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(_This::value++)>
  operator++(int) const noexcept
  {
    return {};
  }
  template <class _This = __constant_wrapper>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(--_This::value)>
  operator--() const noexcept
  {
    return {};
  }
  template <class _This = __constant_wrapper>
  [[nodiscard]] _CCCL_HOST_DEVICE_API _CCCL_CONSTEVAL __constant_wrapper<_LIBCUDACXX_AUTO_CAST(_This::value--)>
  operator--(int) const noexcept
  {
    return {};
  }

  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value += _Rp::value)>{})
    operator+=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value -= _Rp::value)>{})
    operator-=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value *= _Rp::value)>{})
    operator*=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value /= _Rp::value)>{})
    operator/=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value %= _Rp::value)>{})
    operator%=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value &= _Rp::value)>{})
    operator&=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value |= _Rp::value)>{})
    operator|=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value ^= _Rp::value)>{})
    operator^=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value <<= _Rp::value)>{})
    operator<<=(_Rp) const noexcept
  {
    return {};
  }
  _CCCL_TEMPLATE(class _Rp)
  _CCCL_REQUIRES(__is_constexpr_param_v<_Rp>)
  [[nodiscard]] _CCCL_HOST_DEVICE_API
    _CCCL_CONSTEVAL decltype(__constant_wrapper<_LIBCUDACXX_AUTO_CAST(value >>= _Rp::value)>{})
    operator>>=(_Rp) const noexcept
  {
    return {};
  }
};

// NOLINTEND(bugprone-macro-repeated-side-effects)

_CCCL_DIAG_POP

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_STD_VER >= 2020 && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
