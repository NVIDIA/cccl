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

#if _CCCL_STD_VER >= 2020

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__functional/invoke.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
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

template <class _Tp>
struct __cw_fixed_value
{
  using __type _CCCL_NODEBUG_ALIAS = _Tp;
  _CCCL_HOST_DEVICE consteval __cw_fixed_value(__type __v) noexcept
      : __data(__v)
  {}
  _Tp __data;
};

template <class _Tp, size_t _Extent>
struct __cw_fixed_value<_Tp[_Extent]>
{
  using __type _CCCL_NODEBUG_ALIAS = _Tp[_Extent];
  _Tp __data[_Extent];

  _CCCL_HOST_DEVICE consteval __cw_fixed_value(_Tp (&__arr)[_Extent]) noexcept
      : __cw_fixed_value(__arr, make_index_sequence<_Extent>{})
  {}

private:
  template <size_t... _Idxs>
  _CCCL_HOST_DEVICE consteval __cw_fixed_value(_Tp (&__arr)[_Extent], index_sequence<_Idxs...>) noexcept
      : __data{__arr[_Idxs]...}
  {}
};

template <class _Tp, size_t _Extent>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES __cw_fixed_value(_Tp (&)[_Extent]) -> __cw_fixed_value<_Tp[_Extent]>;

template <__cw_fixed_value _Xp, class = typename decltype(__cw_fixed_value(_Xp))::__type>
struct __constant_wrapper;

template <class _Tp>
concept __constexpr_param = requires { typename __constant_wrapper<_Tp::value>; };

template <__cw_fixed_value _Xp>
constexpr auto __cw = __constant_wrapper<_Xp>{};

struct __cw_operators
{
  // unary operators
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(+_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator+(_Tp) noexcept
  {
    return __constant_wrapper<(+_Tp::value)>{};
  }
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(-_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator-(_Tp) noexcept
  {
    return __constant_wrapper<(-_Tp::value)>{};
  }
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(~_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator~(_Tp) noexcept
  {
    return __constant_wrapper<(~_Tp::value)>{};
  }
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(!_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator!(_Tp) noexcept
  {
    return __constant_wrapper<(!_Tp::value)>{};
  }
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(&_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator&(_Tp) noexcept
  {
    return __constant_wrapper<(&_Tp::value)>{};
  }
  template <__constexpr_param _Tp>
    requires requires { typename __constant_wrapper<(*_Tp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator*(_Tp) noexcept
  {
    return __constant_wrapper<(*_Tp::value)>{};
  }

  // binary operators
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value + _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator+(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value + _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value - _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator-(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value - _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value * _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator*(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value * _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value / _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator/(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value / _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value % _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator%(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value % _Rp::value)>{};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value << _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator<<(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value << _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value >> _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator>>(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value >> _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value & _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator&(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value & _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value | _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator|(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value | _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value ^ _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator^(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value ^ _Rp::value)>{};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value && _Rp::value)>; }
          && (!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator&&(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value && _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value || _Rp::value)>; }
          && (!is_constructible_v<bool, decltype(_Lp::value)> || !is_constructible_v<bool, decltype(_Rp::value)>)
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator||(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value || _Rp::value)>{};
  }

  // comparisons
  // template <__constexpr_param _Lp, __constexpr_param _Rp>
  //   requires requires { typename __constant_wrapper<(_Lp::value <=> _Rp::value)>; }
  // [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator<=>(_Lp, _Rp) noexcept
  // {
  //   return __constant_wrapper<(_Lp::value <=> _Rp::value)>{};
  // }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value < _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator<(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value < _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value <= _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator<=(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value <= _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value == _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator==(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value == _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value != _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator!=(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value != _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value > _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator>(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value > _Rp::value)>{};
  }
  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value >= _Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator>=(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value >= _Rp::value)>{};
  }

  template <__constexpr_param _Lp, __constexpr_param _Rp>
  friend auto operator,(_Lp, _Rp) = delete;

  template <__constexpr_param _Lp, __constexpr_param _Rp>
    requires requires { typename __constant_wrapper<(_Lp::value->*_Rp::value)>; }
  [[nodiscard]] _CCCL_HOST_DEVICE friend consteval auto operator->*(_Lp, _Rp) noexcept
  {
    return __constant_wrapper<(_Lp::value->*_Rp::value)>{};
  }
};

template <__cw_fixed_value _Xp, class _Tp>
struct __constant_wrapper : __cw_operators
{
  static constexpr const auto& value = _Xp.__data;
  using __cw_fixed_value_type        = remove_cvref_t<decltype(_Xp)>;
  using type                         = __constant_wrapper;
  using value_type                   = _Tp;

  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value = _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator=(_Rp) const noexcept
  {
    constexpr auto __ret = (value = _Rp::value);
    return __constant_wrapper<__ret>{};
  }

  _CCCL_API constexpr operator const _Tp&() const noexcept
  {
    return _Xp.__data;
  }

  template <class... _Args>
    requires(__constexpr_param<remove_cvref_t<_Args>> && ...)
         && requires { typename __constant_wrapper<::cuda::std::invoke(value, remove_cvref_t<_Args>::value...)>; }
#  if _CCCL_HAS_STATIC_CALL_OPERATOR()
  _CCCL_API static constexpr decltype(auto) operator()(_Args&&...) noexcept
#  else // ^^^ _CCCL_HAS_STATIC_CALL_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_CALL_OPERATOR() vvv
  _CCCL_API constexpr decltype(auto) operator()(_Args&&...) const noexcept
#  endif // ^^^ !_CCCL_HAS_STATIC_CALL_OPERATOR() ^^^
  {
    return __constant_wrapper<::cuda::std::invoke(value, remove_cvref_t<_Args>::value...)>{};
  }

  template <class... _Args>
    requires(!(
              (__constexpr_param<remove_cvref_t<_Args>> && ...)
              && requires { typename __constant_wrapper<::cuda::std::invoke(value, remove_cvref_t<_Args>::value...)>; }))
         && is_invocable_v<const _Tp&, _Args&&...>
#  if _CCCL_HAS_STATIC_CALL_OPERATOR()
  _CCCL_API static constexpr decltype(auto) operator()(_Args&&... __args)
#  else // ^^^ _CCCL_HAS_STATIC_CALL_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_CALL_OPERATOR() vvv
  _CCCL_API constexpr decltype(auto) operator()(_Args&&... __args) const
#  endif // ^^^ !_CCCL_HAS_STATIC_CALL_OPERATOR() ^^^
    noexcept(::cuda::std::is_nothrow_invocable_v<const _Tp&, _Args...>)
  {
    return ::cuda::std::invoke(value, ::cuda::std::forward<_Args>(__args)...);
  }

#  if _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS()
  template <class... _Args>
    requires(__constexpr_param<remove_cvref_t<_Args>> && ...)
         && requires { typename __constant_wrapper<value[remove_cvref_t<_Args>::value...]>; }
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]] _CCCL_HOST_DEVICE consteval static auto operator[](_Args&&...) noexcept
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator[](_Args&&...) const noexcept
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
  {
    return __constant_wrapper<value[remove_cvref_t<_Args>::value...]>{};
  }
  template <class... _Args>
    requires(!((__constexpr_param<remove_cvref_t<_Args>> && ...)
               && requires { typename __constant_wrapper<value[remove_cvref_t<_Args>::value...]>; }))
         && requires { value[::cuda::std::declval<_Args>()...]; }
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) operator[](_Args&&... __args)
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](_Args&&... __args) const
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
    noexcept(noexcept(value[::cuda::std::forward<_Args>(__args)...]))
  {
    return value[::cuda::std::forward<_Args>(__args)...];
  }
#  else // ^^^ _CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^ / vvv !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() vvv
  template <class _Arg>
    requires __constexpr_param<remove_cvref_t<_Arg>> && requires {
      typename __constant_wrapper<value[remove_cvref_t<_Arg>::value]>;
    }
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]] _CCCL_HOST_DEVICE consteval static auto operator[](_Arg&&) noexcept
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator[](_Arg&&) const noexcept
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
  {
    return __constant_wrapper<value[remove_cvref_t<_Arg>::value]>{};
  }
  template <class _Arg>
    requires(!(__constexpr_param<remove_cvref_t<_Arg>>
               && requires { typename __constant_wrapper<value[remove_cvref_t<_Arg>::value]>; }))
         && requires { value[::cuda::std::declval<_Arg>()]; }
#    if _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR()
  [[nodiscard]] _CCCL_API static constexpr decltype(auto) operator[](_Arg&& __arg)
#    else // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() vvv
  [[nodiscard]] _CCCL_API constexpr decltype(auto) operator[](_Arg&& __arg) const
#    endif // ^^^ _CCCL_HAS_STATIC_SUBSCRIPT_OPERATOR() ^^^
    noexcept(noexcept(value[::cuda::std::forward<_Arg>(__arg)]))
  {
    return value[::cuda::std::forward<_Arg>(__arg)];
  }
#  endif // ^^^ !_CCCL_HAS_MULTIARG_OPERATOR_BRACKETS() ^^^

  // pseudo-mutators
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator++() const noexcept
    requires requires { typename __constant_wrapper<(++value)>; }
  {
    return __constant_wrapper<(++value)>{};
  }
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator++(int) const noexcept
    requires requires { typename __constant_wrapper<(value++)>; }
  {
    return __constant_wrapper<(value++)>{};
  }
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator--() const noexcept
    requires requires { typename __constant_wrapper<(--value)>; }
  {
    return __constant_wrapper<(--value)>{};
  }
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator--(int) const noexcept
    requires requires { typename __constant_wrapper<(value--)>; }
  {
    return __constant_wrapper<(value--)>{};
  }

  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value += _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator+=(_Rp) const noexcept
  {
    constexpr auto __ret = (value += _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value -= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator-=(_Rp) const noexcept
  {
    constexpr auto __ret = (value -= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value *= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator*=(_Rp) const noexcept
  {
    constexpr auto __ret = (value *= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value /= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator/=(_Rp) const noexcept
  {
    constexpr auto __ret = (value /= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value %= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator%=(_Rp) const noexcept
  {
    constexpr auto __ret = (value %= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value &= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator&=(_Rp) const noexcept
  {
    constexpr auto __ret = (value &= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value |= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator|=(_Rp) const noexcept
  {
    constexpr auto __ret = (value |= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value ^= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator^=(_Rp) const noexcept
  {
    constexpr auto __ret = (value ^= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value <<= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator<<=(_Rp) const noexcept
  {
    constexpr auto __ret = (value <<= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
  template <__constexpr_param _Rp, class = decltype(__constant_wrapper<__cw_fixed_value(value >>= _Rp::value)>{})>
  [[nodiscard]] _CCCL_HOST_DEVICE consteval auto operator>>=(_Rp) const noexcept
  {
    constexpr auto __ret = (value >>= _Rp::value);
    return __constant_wrapper<__ret>{};
  }
};

_CCCL_DIAG_POP

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_STD_VER >= 2020

#endif // _CUDA_STD___UTILITY_CONSTANT_WRAPPER_H
