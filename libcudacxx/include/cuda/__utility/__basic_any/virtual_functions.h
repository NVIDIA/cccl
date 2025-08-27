//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_VIRUAL_FUNCTIONS_H
#define _CUDA___UTILITY_BASIC_ANY_VIRUAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/is_member_function_pointer.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//!
//! __override_tag
//!
template <class _Tp, auto _Override>
struct __override_tag_;

template <class _Tp, auto _Override>
using __override_tag _CCCL_NODEBUG_ALIAS = __override_tag_<_Tp, _Override>*;

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")

template <class _Fn, class _Cp>
_CCCL_API auto __class_of_(_Fn _Cp::*) -> _Cp;

template <class _Fn>
using __class_of _CCCL_NODEBUG_ALIAS = decltype(::cuda::__class_of_(_Fn()));

//! We use a C-style cast instead of a static_cast because a C-style cast will
//! ignore accessibility, letting us cast to a private base class.
template <class _DstPtr, class _Src>
_CCCL_NODEBUG_API auto __c_style_cast(_Src* __ptr) noexcept -> _DstPtr
{
  static_assert(::cuda::std::is_pointer_v<_DstPtr>, "");
  static_assert(::cuda::std::is_base_of_v<::cuda::std::remove_pointer_t<_DstPtr>, _Src>,
                "invalid C-style cast detected");
  return (_DstPtr) __ptr; // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
}

template <class _Tp, auto _Fn, class _Ret, bool _IsConst, bool _IsNothrow, class... _Args>
[[nodiscard]] _CCCL_API auto __override_fn_([[maybe_unused]] ::cuda::std::__maybe_const<_IsConst, void>* __pv,
                                            [[maybe_unused]] _Args... __args) noexcept(_IsNothrow) -> _Ret
{
  using __value_type _CCCL_NODEBUG_ALIAS = ::cuda::std::__maybe_const<_IsConst, _Tp>;

  if constexpr (::cuda::std::is_same_v<_Tp, void>)
  {
    // This instantiation is created only during the computation of the vtable
    // type. It is never actually called.
    _CCCL_UNREACHABLE();
  }
  else if constexpr (::cuda::std::is_member_function_pointer_v<decltype(_Fn)>)
  {
    // _Fn may be a pointer to a member function of a private base of _Tp. So
    // after static_cast-ing to _Tp*, we need to use a C-style cast to get a
    // pointer to the correct base class.
    using __class_type  = ::cuda::std::__maybe_const<_IsConst, __class_of<decltype(_Fn)>>;
    __class_type& __obj = *::cuda::__c_style_cast<__class_type*>(static_cast<__value_type*>(__pv));
    return (__obj.*_Fn)(static_cast<_Args&&>(__args)...);
  }
  else
  {
    __value_type& __obj = *static_cast<__value_type*>(__pv);
    return (*_Fn)(__obj, static_cast<_Args&&>(__args)...);
  }
  _CCCL_UNREACHABLE();
}

_CCCL_DIAG_POP

template <class _Fn, class _Tp = void, auto _Override = 0>
extern ::cuda::std::__undefined<_Fn> __virtual_override_fn;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...), _Tp, _Override> = //
  &__override_fn_<_Tp, _Override, _Ret, false, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void const*, _Args...)>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...), _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, false, true, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, true, _Args...>;

// TODO: Add support for member functions with reference qualifiers.

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...), _Tp, _Override> = //
  &__override_fn_<_Tp, _Override, _Ret, false, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void const*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, false, true, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr ::cuda::std::type_identity_t<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, true, _Args...>;

template <class _Ret, class... _Args>
_CCCL_API auto __get_virtual_result(_Ret (*)(_Args...)) -> _Ret;

template <class _Ret, class... _Args>
_CCCL_API auto __get_virtual_result(_Ret (*)(_Args...) noexcept) noexcept -> _Ret;

template <class _Ret, class... _Args>
_CCCL_API auto __is_virtual_const(_Ret (*)(void*, _Args...)) -> ::cuda::std::false_type;

template <class _Ret, class... _Args>
_CCCL_API auto __is_virtual_const(_Ret (*)(void const*, _Args...)) -> ::cuda::std::true_type;

//!
//! __virtual_fn
//!
template <auto _Fn>
struct __virtual_fn
{
  using __function_t _CCCL_NODEBUG_ALIAS = decltype(__virtual_override_fn<decltype(_Fn)>);
  using __result_t _CCCL_NODEBUG_ALIAS   = decltype(__get_virtual_result(__function_t{}));

  static constexpr bool __const_fn   = decltype(::cuda::__is_virtual_const(__function_t{}))::value;
  static constexpr bool __nothrow_fn = noexcept(::cuda::__get_virtual_result(__function_t{}));

  template <class _Tp, auto _Override>
  _CCCL_API constexpr __virtual_fn(__override_tag<_Tp, _Override>) noexcept
      : __fn_(__virtual_override_fn<decltype(_Fn), _Tp, _Override>)
  {}

  __function_t __fn_;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_VIRUAL_FUNCTIONS_H
