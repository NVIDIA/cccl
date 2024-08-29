//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES_H
#define __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/exception.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
// A typelist for completion signatures
template <class... _Ts>
struct completion_signatures
{};

// A metafunction to determine if a type is a completion signature
template <class>
_CCCL_INLINE_VAR constexpr bool __is_valid_signature = false;

template <class... _Ts>
_CCCL_INLINE_VAR constexpr bool __is_valid_signature<set_value_t(_Ts...)> = true;

template <class _Error>
_CCCL_INLINE_VAR constexpr bool __is_valid_signature<set_error_t(_Error)> = true;

template <>
_CCCL_INLINE_VAR constexpr bool __is_valid_signature<set_stopped_t()> = true;

// The implementation of transform_completion_signatures starts here
template <class _Sig, template <class...> class _Vy, template <class...> class _Ey, class _Sy>
extern __undefined<_Sig> __transform_sig;

template <class... _Values, template <class...> class _Vy, template <class...> class _Ey, class _Sy>
extern __fn_t<_Vy<_Values...>>* __transform_sig<set_value_t(_Values...), _Vy, _Ey, _Sy>;

template <class _Error, template <class...> class _Vy, template <class...> class _Ey, class _Sy>
extern __fn_t<_Ey<_Error>>* __transform_sig<set_error_t(_Error), _Vy, _Ey, _Sy>;

template <template <class...> class _Vy, template <class...> class _Ey, class _Sy>
extern __fn_t<_Sy>* __transform_sig<set_stopped_t(), _Vy, _Ey, _Sy>;

template <class _Sig, template <class...> class _Vy, template <class...> class _Ey, class _Sy>
using __transform_sig_t = decltype(__transform_sig<_Sig, _Vy, _Ey, _Sy>());

template <class _Sigs,
          template <class...>
          class _Vy,
          template <class...>
          class _Ey,
          class _Sy,
          template <class...>
          class _Variant,
          class... _More>
extern _DIAGNOSTIC<_Sigs> __transform_completion_signatures_v;

template <class... _What,
          template <class...>
          class _Vy,
          template <class...>
          class _Ey,
          class _Sy,
          template <class...>
          class _Variant,
          class... _More>
extern __fn_t<_ERROR<_What...>>*
  __transform_completion_signatures_v<_ERROR<_What...>, _Vy, _Ey, _Sy, _Variant, _More...>;

template <class... _Sigs,
          template <class...>
          class _Vy,
          template <class...>
          class _Ey,
          class _Sy,
          template <class...>
          class _Variant,
          class... _More>
extern __fn_t<_Variant<__transform_sig_t<_Sigs, _Vy, _Ey, _Sy>..., _More...>>*
  __transform_completion_signatures_v<completion_signatures<_Sigs...>, _Vy, _Ey, _Sy, _Variant, _More...>;

template <class _Sigs,
          template <class...>
          class _Vy,
          template <class...>
          class _Ey,
          class _Sy,
          template <class...>
          class _Variant,
          class... _More>
using __transform_completion_signatures =
  decltype(__transform_completion_signatures_v<_Sigs, _Vy, _Ey, _Sy, _Variant, _More...>());

template <class _WantedTag>
struct __gather_sigs_fn;

template <>
struct __gather_sigs_fn<set_value_t>
{
  template <class _Sigs,
            template <class...>
            class _Then,
            template <class...>
            class _Else,
            template <class...>
            class _Variant,
            class... _More>
  using __f = __transform_completion_signatures<
    _Sigs,
    _Then,
    __mbind_front_q<_Else, set_error_t>::template __f,
    _Else<set_stopped_t>,
    _Variant,
    _More...>;
};

template <>
struct __gather_sigs_fn<set_error_t>
{
  template <class _Sigs,
            template <class...>
            class _Then,
            template <class...>
            class _Else,
            template <class...>
            class _Variant,
            class... _More>
  using __f = __transform_completion_signatures<
    _Sigs,
    __mbind_front_q<_Else, set_value_t>::template __f,
    _Then,
    _Else<set_stopped_t>,
    _Variant,
    _More...>;
};

template <>
struct __gather_sigs_fn<set_stopped_t>
{
  template <class _Sigs,
            template <class...>
            class _Then,
            template <class...>
            class _Else,
            template <class...>
            class _Variant,
            class... _More>
  using __f = __transform_completion_signatures<
    _Sigs,
    __mbind_front_q<_Else, set_value_t>::template __f,
    __mbind_front_q<_Else, set_error_t>::template __f,
    _Then<>,
    _Variant,
    _More...>;
};

template <class _Sigs,
          class _WantedTag,
          template <class...>
          class _Then,
          template <class...>
          class _Else,
          template <class...>
          class _Variant,
          class... _More>
using __gather_completion_signatures =
  typename __gather_sigs_fn<_WantedTag>::template __f<_Sigs, _Then, _Else, _Variant, _More...>;

template <class... _Ts>
using __set_value_transform_t = completion_signatures<set_value_t(_Ts...)>;

template <class _Ty>
using __set_error_transform_t = completion_signatures<set_error_t(_Ty)>;

template <class... _Ts, class... _Us>
auto operator*(__mset<_Ts...>&, __undefined<completion_signatures<_Us...>>&) -> __mset_insert<__mset<_Ts...>, _Us...>&;

template <class... _Ts, class... _What>
auto operator*(__mset<_Ts...>&, __undefined<_ERROR<_What...>>&) -> _ERROR<_What...>&;

template <class... _What, class... _Us>
auto operator*(_ERROR<_What...>&, __undefined<completion_signatures<_Us...>>&) -> _ERROR<_What...>&;

template <class... _Sigs>
using __concat_completion_signatures = //
  __mapply_q<completion_signatures, __mconcat_into_q<__mmake_set>::__f<_Sigs...>>;

template <class _Tag, class... _Ts>
using __default_completions = completion_signatures<_Tag(_Ts...)>;

template <class _Sigs,
          class _MoreSigs                           = completion_signatures<>,
          template <class...> class _ValueTransform = __set_value_transform_t,
          template <class> class _ErrorTransform    = __set_error_transform_t,
          class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
using transform_completion_signatures = //
  __transform_completion_signatures<_Sigs,
                                    _ValueTransform,
                                    _ErrorTransform,
                                    _StoppedSigs,
                                    __mtry_quote<__concat_completion_signatures>::__f,
                                    _MoreSigs>;

template <class _Sndr,
          class _Rcvr,
          class _MoreSigs                           = completion_signatures<>,
          template <class...> class _ValueTransform = __set_value_transform_t,
          template <class> class _ErrorTransform    = __set_error_transform_t,
          class _StoppedSigs                        = completion_signatures<set_stopped_t()>>
using transform_completion_signatures_of = //
  transform_completion_signatures<completion_signatures_of_t<_Sndr, _Rcvr>,
                                  _MoreSigs,
                                  _ValueTransform,
                                  _ErrorTransform,
                                  _StoppedSigs>;

template <class _Sigs,
          template <class...>
          class _Tuple,
          template <class...>
          class _Variant>
using __value_types = //
  __transform_completion_signatures<_Sigs,
                                    __mcompose_q<__mlist, _Tuple>::template __f,
                                    __malways<__mlist<>>::__f,
                                    __mlist<>,
                                    __mconcat_into_q<_Variant>::template __f>;

template <class _Sndr, class _Rcvr, template <class...> class _Tuple, template <class...> class _Variant>
using value_types_of_t =
  __value_types<completion_signatures_of_t<_Sndr, _Rcvr>, _Tuple, __mtry_quote<_Variant>::template __f>;

template <class _Sigs,
          template <class...>
          class _Variant>
using __error_types = //
  __transform_completion_signatures<_Sigs,
                                    __malways<__mlist<>>::__f,
                                    __mlist,
                                    __mlist<>,
                                    __mconcat_into_q<_Variant>::template __f>;

template <class _Sndr, class _Rcvr, template <class...> class _Variant>
using error_types_of_t = __error_types<completion_signatures_of_t<_Sndr, _Rcvr>, _Variant>;

template <class _Sigs>
_CCCL_INLINE_VAR constexpr bool __sends_stopped = //
  __transform_completion_signatures<_Sigs, __malways<__mfalse>::__f, __malways<__mfalse>::__f, __mtrue, __mor>::__value;

template <class _Sndr, class _Rcvr = receiver_archetype>
_CCCL_INLINE_VAR constexpr bool sends_stopped = //
  __sends_stopped<completion_signatures_of_t<_Sndr, _Rcvr>>;

using __eptr_completion = completion_signatures<set_error_t(::std::exception_ptr)>;

template <bool _NoExcept>
using __eptr_completion_if = __mif<_NoExcept, completion_signatures<>, __eptr_completion>;

template <class>
_CCCL_INLINE_VAR constexpr bool __is_completion_signatures = false;

template <class... _Sigs>
_CCCL_INLINE_VAR constexpr bool __is_completion_signatures<completion_signatures<_Sigs...>> = true;

template <class _Sndr>
using __is_non_dependent_detail_ = //
  __mif<__is_completion_signatures<completion_signatures_of_t<_Sndr>>>;

template <class _Sndr>
_CCCL_INLINE_VAR constexpr bool __is_non_dependent_sender = __mvalid_q<__is_non_dependent_detail_, _Sndr>;

namespace __csig
{
struct __dep
{};

template <class... _Sigs>
struct __sigs;

template <class... _As, class... _Bs>
auto operator+(__sigs<_As...>&, __sigs<_Bs...>&) -> __sigs<_As..., _Bs...>&;

template <class... _Sigs>
auto operator+(__sigs<_Sigs...>&) //
  -> __concat_completion_signatures<completion_signatures<_Sigs...>>;

template <class _Other>
auto __to_sigs(_Other&) -> _Other&;

template <class... _Sigs>
auto __to_sigs(completion_signatures<_Sigs...>&) -> __sigs<_Sigs...>&;
} // namespace __csig

using dependent_completions = __csig::__dep;

namespace meta
{
template <class... _Sigs>
using sigs = __csig::__sigs<_Sigs...>*;

template <class _Tag, class... _Args>
auto completion(_Tag, _Args&&...) -> __csig::__sigs<_Tag(_Args...)>&;

template <class _Sndr, class _Rcvr = receiver_archetype>
auto completions_of(_Sndr&&,
                    _Rcvr = {}) -> decltype(__csig::__to_sigs(DECLVAL(completion_signatures_of_t<_Sndr, _Rcvr>&)));

template <bool _PotentiallyThrowing>
auto eptr_completion_if()
  -> __mif<_PotentiallyThrowing, __csig::__sigs<set_error_t(::std::exception_ptr)>, __csig::__sigs<>>&;
} // namespace meta
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
