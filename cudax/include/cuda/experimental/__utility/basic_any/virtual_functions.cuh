//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_VIRUAL_FUNCTIONS_H
#define __CUDAX_DETAIL_BASIC_ANY_VIRUAL_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>

namespace cuda::experimental
{
///
/// __override_tag
///
template <class _Tp, auto _Override>
struct __override_tag_;

template <class _Tp, auto _Override>
using __override_tag _CCCL_NODEBUG_ALIAS = __override_tag_<_Tp, _Override>*;

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")

template <class _Fn, class _Cp>
_CUDAX_API auto __class_of_(_Fn _Cp::*) -> _Cp;

template <class _Fn>
using __class_of _CCCL_NODEBUG_ALIAS = decltype(__cudax::__class_of_(_Fn()));

/// We use a C-style cast instead of a static_cast because a C-style cast will
/// ignore accessibility, letting us cast to a private base class.
template <class _DstPtr, class _Src>
_CUDAX_TRIVIAL_API _DstPtr __c_style_cast(_Src* __ptr) noexcept
{
  static_assert(_CUDA_VSTD::is_pointer_v<_DstPtr>, "");
  static_assert(_CUDA_VSTD::is_base_of_v<_CUDA_VSTD::remove_pointer_t<_DstPtr>, _Src>, "invalid C-style cast detected");
  return (_DstPtr) __ptr; // NOLINT(cppcoreguidelines-pro-type-cstyle-cast)
}

template <class _Tp, auto _Fn, class _Ret, bool _IsConst, bool _IsNothrow, class... _Args>
_CCCL_NODISCARD _CUDAX_API _Ret __override_fn_([[maybe_unused]] _CUDA_VSTD::__maybe_const<_IsConst, void>* __pv,
                                               [[maybe_unused]] _Args... __args) noexcept(_IsNothrow)
{
  using __value_type _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>;

  if constexpr (_CUDA_VSTD::is_same_v<_Tp, void>)
  {
    // This instantiation is created only during the computation of the vtable
    // type. It is never actually called.
    _CCCL_UNREACHABLE();
  }
  else if constexpr (_CUDA_VSTD::is_member_function_pointer_v<decltype(_Fn)>)
  {
    // _Fn may be a pointer to a member function of a private base of _Tp. So
    // after static_cast-ing to _Tp*, we need to use a C-style cast to get a
    // pointer to the correct base class.
    using __class_type  = _CUDA_VSTD::__maybe_const<_IsConst, __class_of<decltype(_Fn)>>;
    __class_type& __obj = *__cudax::__c_style_cast<__class_type*>(static_cast<__value_type*>(__pv));
    return (__obj.*_Fn)(static_cast<_Args&&>(__args)...);
  }
  else
  {
    __value_type& __obj = *static_cast<__value_type*>(__pv);
    return (*_Fn)(__obj, static_cast<_Args&&>(__args)...);
  }
}

_CCCL_DIAG_POP

template <class _Fn, class _Tp = void, auto _Override = 0>
extern _CUDA_VSTD::__undefined<_Fn> __virtual_override_fn;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...), _Tp, _Override> = //
  &__override_fn_<_Tp, _Override, _Ret, false, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void const*, _Args...)>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...), _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, false, true, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, true, _Args...>;

// TODO: Add support for member functions with reference qualifiers.

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...), _Tp, _Override> = //
  &__override_fn_<_Tp, _Override, _Ret, false, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void const*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, false, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, false, true, _Args...>;

template <class _Tp, auto _Override, class _Ret, class _Cp, class... _Args>
inline constexpr __identity_t<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const noexcept, _Tp, _Override> =
    &__override_fn_<_Tp, _Override, _Ret, true, true, _Args...>;

template <class _Ret, class... _Args>
_CUDAX_API _Ret __get_virtual_result(_Ret (*)(_Args...));

template <class _Ret, class... _Args>
_CUDAX_API _Ret __get_virtual_result(_Ret (*)(_Args...) noexcept) noexcept;

template <class _Ret, class... _Args>
_CUDAX_API _CUDA_VSTD::false_type __is_virtual_const(_Ret (*)(void*, _Args...));

template <class _Ret, class... _Args>
_CUDAX_API _CUDA_VSTD::true_type __is_virtual_const(_Ret (*)(void const*, _Args...));

///
/// __virtual_fn
///
template <auto _Fn>
struct __virtual_fn
{
  using __function_t _CCCL_NODEBUG_ALIAS = decltype(__virtual_override_fn<decltype(_Fn)>);
  using __result_t _CCCL_NODEBUG_ALIAS   = decltype(__get_virtual_result(__function_t{}));

  static constexpr bool __const_fn   = decltype(__cudax::__is_virtual_const(__function_t{}))::value;
  static constexpr bool __nothrow_fn = noexcept(__cudax::__get_virtual_result(__function_t{}));

  template <class _Tp, auto _Override>
  _CUDAX_API constexpr __virtual_fn(__override_tag<_Tp, _Override>) noexcept
      : __fn_(__virtual_override_fn<decltype(_Fn), _Tp, _Override>)
  {}

  __function_t __fn_;
};

} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_BASIC_ANY_VIRUAL_FUNCTIONS_H
