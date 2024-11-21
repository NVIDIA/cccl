//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES
#define __CUDAX_ASYNC_DETAIL_COMPLETION_SIGNATURES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/disjunction.h>
#include <cuda/std/__type_traits/enable_if.h>

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/type_traits.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
// A typelist for completion signatures
template <class... _Sigs>
struct completion_signatures
{
  struct __partitioned;
};

template <class _CompletionSignatures>
using __partitioned_completions_of = typename _CompletionSignatures::__partitioned;

constexpr int __invalid_disposition = -1;

// A metafunction to determine whether a type is a completion signature, and if
// so, what its disposition is.
template <class>
inline constexpr int __signature_disposition = __invalid_disposition;

template <class... _Ts>
inline constexpr __disposition_t __signature_disposition<set_value_t(_Ts...)> = __disposition_t::__value;

template <class _Error>
inline constexpr __disposition_t __signature_disposition<set_error_t(_Error)> = __disposition_t::__error;

template <>
inline constexpr __disposition_t __signature_disposition<set_stopped_t()> = __disposition_t::__stopped;

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
  using __call = __transform_completion_signatures<
    _Sigs,
    _Then,
    _CUDA_VSTD::__type_bind_front_quote<_Else, set_error_t>::template __call,
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
  using __call = __transform_completion_signatures<
    _Sigs,
    _CUDA_VSTD::__type_bind_front_quote<_Else, set_value_t>::template __call,
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
  using __call = __transform_completion_signatures<
    _Sigs,
    _CUDA_VSTD::__type_bind_front_quote<_Else, set_value_t>::template __call,
    _CUDA_VSTD::__type_bind_front_quote<_Else, set_error_t>::template __call,
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
  typename __gather_sigs_fn<_WantedTag>::template __call<_Sigs, _Then, _Else, _Variant, _More...>;

// __partitioned_completions is a cache of completion signatures for fast
// access. The completion_signatures<Sigs...>::__partitioned nested struct
// inherits from __partitioned_completions. If the cache is never accessed,
// it is never instantiated.
template <class _ValueTuplesList = _CUDA_VSTD::__type_list<>,
          class _ErrorsList      = _CUDA_VSTD::__type_list<>,
          bool _HasStopped       = false>
struct __partitioned_completions;

template <class... _ValueTuples, class... _Errors, bool _HasStopped>
struct __partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>,
                                 _CUDA_VSTD::__type_list<_Errors...>,
                                 _HasStopped>
{
  using __stopped_sigs =
    _CUDA_VSTD::conditional_t<_HasStopped, _CUDA_VSTD::__type_list<set_stopped_t()>, _CUDA_VSTD::__type_list<>>;

  template <template <class...> class _Tuple, template <class...> class _Variant>
  using __value_types = _Variant<_CUDA_VSTD::__type_call1<_ValueTuples, _CUDA_VSTD::__type_quote<_Tuple>>...>;

  template <template <class...> class _Variant>
  using __error_types = _Variant<_Errors...>;

  template <template <class...> class _Variant, class _Type>
  using __stopped_types = _CUDA_VSTD::__type_call1<
    _CUDA_VSTD::conditional_t<_HasStopped, _CUDA_VSTD::__type_list<_Type>, _CUDA_VSTD::__type_list<>>,
    _CUDA_VSTD::__type_quote<_Variant>>;

  using __count_values  = _CUDA_VSTD::integral_constant<size_t, sizeof...(_ValueTuples)>;
  using __count_errors  = _CUDA_VSTD::integral_constant<size_t, sizeof...(_Errors)>;
  using __count_stopped = _CUDA_VSTD::integral_constant<size_t, _HasStopped>;

  struct __nothrow_decay_copyable
  {
    // These aliases are placed in a separate struct to avoid computing them
    // if they are not needed.
    using __fn     = _CUDA_VSTD::__type_quote<__nothrow_decay_copyable_t>;
    using __values = _CUDA_VSTD::_And<_CUDA_VSTD::__type_call1<_ValueTuples, __fn>...>;
    using __errors = __nothrow_decay_copyable_t<_Errors...>;
    using __all    = _CUDA_VSTD::_And<__values, __errors>;
  };
};

// The following overload set of operator* is used to build up the cache of
// completion signatures. We fold over operator*, accumulating the completion
// signatures in the cache. `__undefined` is used here to prevent the
// instantiation of the intermediate types.

template <class... _ValueTuples, class... _Errors, bool _HasStopped, class... _Values>
auto operator*(__undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>,
                                                     _CUDA_VSTD::__type_list<_Errors...>,
                                                     _HasStopped>>&,
               set_value_t (*)(_Values...))
  -> __undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples..., _CUDA_VSTD::__type_list<_Values...>>,
                                           _CUDA_VSTD::__type_list<_Errors...>,
                                           _HasStopped>>&;

template <class... _ValueTuples, class... _Errors, bool _HasStopped, class _Error>
auto operator*(__undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>,
                                                     _CUDA_VSTD::__type_list<_Errors...>,
                                                     _HasStopped>>&,
               set_error_t (*)(_Error))
  -> __undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>,
                                           _CUDA_VSTD::__type_list<_Errors..., _Error>,
                                           _HasStopped>>&;

template <class... _ValueTuples, class... _Errors>
auto operator*(__undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>, //
                                                     _CUDA_VSTD::__type_list<_Errors...>,
                                                     false>>&,
               set_stopped_t (*)())
  -> __undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>, //
                                           _CUDA_VSTD::__type_list<_Errors...>,
                                           true>>&;

// This unary overload is used to extract the cache from the `__undefined` type.
template <class... _ValueTuples, class... _Errors, bool _HasStopped>
auto operator*(__undefined<__partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>,
                                                     _CUDA_VSTD::__type_list<_Errors...>,
                                                     _HasStopped>>&)
  -> __partitioned_completions<_CUDA_VSTD::__type_list<_ValueTuples...>, //
                               _CUDA_VSTD::__type_list<_Errors...>,
                               _HasStopped>;

template <class... _Sigs>
using __partition_completions =
  decltype(*(__declval<__undefined<__partitioned_completions<>>&>() * ... * static_cast<_Sigs*>(nullptr)));

// Here we give the completion_signatures<Sigs...> type a nested struct that
// contains the fast lookup cache of completion signatures.
template <class... _Sigs>
struct completion_signatures<_Sigs...>::__partitioned : __partition_completions<_Sigs...>
{};

template <class... _Ts>
using __set_value_transform_t = completion_signatures<set_value_t(_Ts...)>;

template <class _Ty>
using __set_error_transform_t = completion_signatures<set_error_t(_Ty)>;

template <class... _Sigs>
using __concat_completion_signatures = //
  _CUDA_VSTD::__type_apply<_CUDA_VSTD::__type_quote<completion_signatures>,
                           __type_concat_into_quote<_CUDA_VSTD::__make_type_set>::__call<_Sigs...>>;

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
                                    __type_try_quote<__concat_completion_signatures>::__call,
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

template <class _Sigs, template <class...> class _Tuple, template <class...> class _Variant>
using __value_types = typename _Sigs::__partitioned::template __value_types<_Tuple, _Variant>;

template <class _Sndr, class _Rcvr, template <class...> class _Tuple, template <class...> class _Variant>
using value_types_of_t =
  __value_types<completion_signatures_of_t<_Sndr, _Rcvr>, _Tuple, __type_try_quote<_Variant>::template __call>;

template <class _Sigs, template <class...> class _Variant>
using __error_types = typename _Sigs::__partitioned::template __error_types<_Variant>;

template <class _Sndr, class _Rcvr, template <class...> class _Variant>
using error_types_of_t = __error_types<completion_signatures_of_t<_Sndr, _Rcvr>, _Variant>;

template <class _Sigs, template <class...> class _Variant, class _Type>
using __stopped_types = typename _Sigs::__partitioned::template __stopped_types<_Variant, _Type>;

template <class _Sigs>
inline constexpr bool __sends_stopped = _Sigs::__partitioned::__count_stopped::value != 0;

template <class _Sndr, class _Rcvr = receiver_archetype>
inline constexpr bool sends_stopped = __sends_stopped<completion_signatures_of_t<_Sndr, _Rcvr>>;

using __eptr_completion = completion_signatures<set_error_t(::std::exception_ptr)>;

template <bool _NoExcept>
using __eptr_completion_unless = _CUDA_VSTD::conditional_t<_NoExcept, completion_signatures<>, __eptr_completion>;

template <class>
inline constexpr bool __is_completion_signatures = false;

template <class... _Sigs>
inline constexpr bool __is_completion_signatures<completion_signatures<_Sigs...>> = true;

template <class _Sndr>
using __is_non_dependent_detail_ = //
  _CUDA_VSTD::enable_if_t<__is_completion_signatures<completion_signatures_of_t<_Sndr>>>;

template <class _Sndr>
inline constexpr bool __is_non_dependent_sender = __type_valid_v<__is_non_dependent_detail_, _Sndr>;

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
                    _Rcvr = {}) -> decltype(__csig::__to_sigs(__declval<completion_signatures_of_t<_Sndr, _Rcvr>&>()));

template <bool _PotentiallyThrowing>
auto eptr_completion_if()
  -> _CUDA_VSTD::
    conditional_t<_PotentiallyThrowing, __csig::__sigs<set_error_t(::std::exception_ptr)>, __csig::__sigs<>>&;
} // namespace meta
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
