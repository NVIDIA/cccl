//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "cpos.cuh"
#include "exception.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
// A typelist for completion signatures
template <class... Ts>
struct completion_signatures
{};

// A metafunction to determine if a type is a completion signature
template <class>
_CCCL_INLINE_VAR constexpr bool _is_valid_signature = false;

template <class... Ts>
_CCCL_INLINE_VAR constexpr bool _is_valid_signature<set_value_t(Ts...)> = true;

template <class Error>
_CCCL_INLINE_VAR constexpr bool _is_valid_signature<set_error_t(Error)> = true;

template <>
_CCCL_INLINE_VAR constexpr bool _is_valid_signature<set_stopped_t()> = true;

// The implementation of transform_completion_signatures starts here
template <class Sig, template <class...> class V, template <class...> class E, class S>
extern _undefined<Sig> _transform_sig;

template <class... Values, template <class...> class V, template <class...> class E, class S>
extern _fn_t<V<Values...>>* _transform_sig<set_value_t(Values...), V, E, S>;

template <class Error, template <class...> class V, template <class...> class E, class S>
extern _fn_t<E<Error>>* _transform_sig<set_error_t(Error), V, E, S>;

template <template <class...> class V, template <class...> class E, class S>
extern _fn_t<S>* _transform_sig<set_stopped_t(), V, E, S>;

template <class Sig, template <class...> class V, template <class...> class E, class S>
using _transform_sig_t = decltype(_transform_sig<Sig, V, E, S>());

template <class Sigs,
          template <class...>
          class V,
          template <class...>
          class E,
          class S,
          template <class...>
          class Variant,
          class... More>
extern DIAGNOSTIC<Sigs> _transform_completion_signatures_v;

template <class... What,
          template <class...>
          class V,
          template <class...>
          class E,
          class S,
          template <class...>
          class Variant,
          class... More>
extern _fn_t<ERROR<What...>>* _transform_completion_signatures_v<ERROR<What...>, V, E, S, Variant, More...>;

template <class... Sigs,
          template <class...>
          class V,
          template <class...>
          class E,
          class S,
          template <class...>
          class Variant,
          class... More>
extern _fn_t<Variant<_transform_sig_t<Sigs, V, E, S>..., More...>>*
  _transform_completion_signatures_v<completion_signatures<Sigs...>, V, E, S, Variant, More...>;

template <class Sigs,
          template <class...>
          class V,
          template <class...>
          class E,
          class S,
          template <class...>
          class Variant,
          class... More>
using _transform_completion_signatures =
  decltype(_transform_completion_signatures_v<Sigs, V, E, S, Variant, More...>());

template <class WantedTag>
struct _gather_sigs_fn;

template <>
struct _gather_sigs_fn<set_value_t>
{
  template <class Sigs,
            template <class...>
            class Then,
            template <class...>
            class Else,
            template <class...>
            class Variant,
            class... More>
  using _f = _transform_completion_signatures<
    Sigs,
    Then,
    _mbind_front_q<Else, set_error_t>::template _f,
    Else<set_stopped_t>,
    Variant,
    More...>;
};

template <>
struct _gather_sigs_fn<set_error_t>
{
  template <class Sigs,
            template <class...>
            class Then,
            template <class...>
            class Else,
            template <class...>
            class Variant,
            class... More>
  using _f = _transform_completion_signatures<
    Sigs,
    _mbind_front_q<Else, set_value_t>::template _f,
    Then,
    Else<set_stopped_t>,
    Variant,
    More...>;
};

template <>
struct _gather_sigs_fn<set_stopped_t>
{
  template <class Sigs,
            template <class...>
            class Then,
            template <class...>
            class Else,
            template <class...>
            class Variant,
            class... More>
  using _f = _transform_completion_signatures<
    Sigs,
    _mbind_front_q<Else, set_value_t>::template _f,
    _mbind_front_q<Else, set_error_t>::template _f,
    Then<>,
    Variant,
    More...>;
};

template <class Sigs,
          class WantedTag,
          template <class...>
          class Then,
          template <class...>
          class Else,
          template <class...>
          class Variant,
          class... More>
using _gather_completion_signatures =
  typename _gather_sigs_fn<WantedTag>::template _f<Sigs, Then, Else, Variant, More...>;

template <class... Ts>
using _set_value_transform_t = completion_signatures<set_value_t(Ts...)>;

template <class Ty>
using _set_error_transform_t = completion_signatures<set_error_t(Ty)>;

template <class... Ts, class... Us>
auto operator*(_mset<Ts...>&, _undefined<completion_signatures<Us...>>&) -> _mset_insert<_mset<Ts...>, Us...>&;

template <class... Ts, class... What>
auto operator*(_mset<Ts...>&, _undefined<ERROR<What...>>&) -> ERROR<What...>&;

template <class... What, class... Us>
auto operator*(ERROR<What...>&, _undefined<completion_signatures<Us...>>&) -> ERROR<What...>&;

template <class... Sigs>
using _concat_completion_signatures = //
  _mapply_q<completion_signatures, _mconcat_into_q<_mmake_set>::_f<Sigs...>>;

template <class Tag, class... Ts>
using _default_completions = completion_signatures<Tag(Ts...)>;

template <class Sigs,
          class MoreSigs                           = completion_signatures<>,
          template <class...> class ValueTransform = _set_value_transform_t,
          template <class> class ErrorTransform    = _set_error_transform_t,
          class StoppedSigs                        = completion_signatures<set_stopped_t()>>
using transform_completion_signatures = //
  _transform_completion_signatures<Sigs,
                                   ValueTransform,
                                   ErrorTransform,
                                   StoppedSigs,
                                   _mtry_quote<_concat_completion_signatures>::_f,
                                   MoreSigs>;

template <class Sndr,
          class Rcvr,
          class MoreSigs                           = completion_signatures<>,
          template <class...> class ValueTransform = _set_value_transform_t,
          template <class> class ErrorTransform    = _set_error_transform_t,
          class StoppedSigs                        = completion_signatures<set_stopped_t()>>
using transform_completion_signatures_of = //
  transform_completion_signatures<completion_signatures_of_t<Sndr, Rcvr>,
                                  MoreSigs,
                                  ValueTransform,
                                  ErrorTransform,
                                  StoppedSigs>;

template <class Sigs,
          template <class...>
          class Tuple,
          template <class...>
          class Variant>
using _value_types = //
  _transform_completion_signatures<Sigs,
                                   _mcompose_q<_mlist, Tuple>::template _f,
                                   _malways<_mlist<>>::_f,
                                   _mlist<>,
                                   _mconcat_into_q<Variant>::template _f>;

template <class Sndr, class Rcvr, template <class...> class Tuple, template <class...> class Variant>
using value_types_of_t = _value_types<completion_signatures_of_t<Sndr, Rcvr>, Tuple, _mtry_quote<Variant>::template _f>;

template <class Sigs,
          template <class...>
          class Variant>
using _error_types = //
  _transform_completion_signatures<Sigs, _malways<_mlist<>>::_f, _mlist, _mlist<>, _mconcat_into_q<Variant>::template _f>;

template <class Sndr, class Rcvr, template <class...> class Variant>
using error_types_of_t = _error_types<completion_signatures_of_t<Sndr, Rcvr>, Variant>;

template <class Sigs>
_CCCL_INLINE_VAR constexpr bool _sends_stopped = //
  _transform_completion_signatures<Sigs, _malways<_mfalse>::_f, _malways<_mfalse>::_f, _mtrue, _mor>::value;

template <class Sndr, class Rcvr = receiver_archetype>
_CCCL_INLINE_VAR constexpr bool sends_stopped = //
  _sends_stopped<completion_signatures_of_t<Sndr, Rcvr>>;

using _eptr_completion = completion_signatures<set_error_t(::std::exception_ptr)>;

template <bool NoExcept>
using _eptr_completion_if = _mif<NoExcept, completion_signatures<>, _eptr_completion>;

template <class>
_CCCL_INLINE_VAR constexpr bool _is_completion_signatures = false;

template <class... Sigs>
_CCCL_INLINE_VAR constexpr bool _is_completion_signatures<completion_signatures<Sigs...>> = true;

template <class Sndr>
using _is_non_dependent_detail_ = //
  _mif<_is_completion_signatures<completion_signatures_of_t<Sndr>>>;

template <class Sndr>
_CCCL_INLINE_VAR constexpr bool _is_non_dependent_sender = _mvalid_q<_is_non_dependent_detail_, Sndr>;

namespace _csig
{
struct _dep
{};

template <class... Sigs>
struct _sigs;

template <class... As, class... Bs>
auto operator+(_sigs<As...>&, _sigs<Bs...>&) -> _sigs<As..., Bs...>&;

template <class... Sigs>
auto operator+(_sigs<Sigs...>&) //
  -> _concat_completion_signatures<completion_signatures<Sigs...>>;

template <class Other>
auto _to_sigs(Other&) -> Other&;

template <class... Sigs>
auto _to_sigs(completion_signatures<Sigs...>&) -> _sigs<Sigs...>&;
} // namespace _csig

using dependent_completions = _csig::_dep;

namespace meta
{
template <class... Sigs>
using sigs = _csig::_sigs<Sigs...>*;

template <class Tag, class... Args>
auto completion(Tag, Args&&...) -> _csig::_sigs<Tag(Args...)>&;

template <class Sndr, class Rcvr = receiver_archetype>
auto completions_of(Sndr&&, Rcvr = {}) -> decltype(_csig::_to_sigs(DECLVAL(completion_signatures_of_t<Sndr, Rcvr>&)));

template <bool PotentiallyThrowing>
auto eptr_completion_if() -> _mif<PotentiallyThrowing, _csig::_sigs<set_error_t(::std::exception_ptr)>, _csig::_sigs<>>&;
} // namespace meta
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
