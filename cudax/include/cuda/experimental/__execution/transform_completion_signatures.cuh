//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_TRANSFORM_COMPLETION_SIGNATURES
#define __CUDAX_EXECUTION_TRANSFORM_COMPLETION_SIGNATURES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>

// include this last:
#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Tag>
struct __default_transform_fn
{
  template <class... _Ts>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()() const noexcept -> completion_signatures<_Tag(_Ts...)>
  {
    return {};
  }
};

struct __swallow_transform
{
  template <class... _Ts>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()() const noexcept -> completion_signatures<>
  {
    return {};
  }
};

template <class _Tag>
struct __decay_transform
{
  template <class... _Ts>
  _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto operator()() const noexcept -> completion_signatures<_Tag(decay_t<_Ts>...)>
  {
    return {};
  }
};

template <class _Fn, class... _As>
using __meta_call_result_t _CCCL_NODEBUG_ALIAS = decltype(declval<_Fn>().template operator()<_As...>());

_CCCL_EXEC_CHECK_DISABLE
template <class _Ay, class... _As, class _Fn>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto __transform_expr(const _Fn& __fn)
  -> __meta_call_result_t<const _Fn&, _Ay, _As...>
{
  return __fn.template operator()<_Ay, _As...>();
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Fn>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto __transform_expr(const _Fn& __fn) -> __call_result_t<const _Fn&>
{
  return __fn();
}

template <class _Fn, class... _As>
using __transform_expr_t _CCCL_NODEBUG_ALIAS = decltype(execution::__transform_expr<_As...>(declval<const _Fn&>()));

struct _IN_TRANSFORM_COMPLETION_SIGNATURES;
struct _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION;
struct _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS;

// transform_completion_signatures:
template <class... _As, class _Fn>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __apply_transform(const _Fn& __fn)
{
  if constexpr (__is_instantiable_with<__transform_expr_t, _Fn, _As...>)
  {
    using __completions _CCCL_NODEBUG_ALIAS = __transform_expr_t<_Fn, _As...>;
    if constexpr (__valid_completion_signatures<__completions> || __type_is_error<__completions>
                  || ::cuda::std::is_base_of_v<dependent_sender_error, __completions>)
    {
      return execution::__transform_expr<_As...>(__fn);
    }
    else
    {
      (void) execution::__transform_expr<_As...>(__fn); // potentially throwing
      return invalid_completion_signature<
        _IN_TRANSFORM_COMPLETION_SIGNATURES,
        _A_TRANSFORM_FUNCTION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
        _WITH_FUNCTION(_Fn),
        _WITH_ARGUMENTS(_As...)>();
    }
  }
  else
  {
    return invalid_completion_signature< //
      _IN_TRANSFORM_COMPLETION_SIGNATURES,
      _COULD_NOT_CALL_THE_TRANSFORM_FUNCTION_WITH_THE_GIVEN_TEMPLATE_ARGUMENTS,
      _WITH_FUNCTION(_Fn),
      _WITH_ARGUMENTS(_As...)>();
  }
}

template <class _ValueFn, class _ErrorFn, class _StoppedFn>
struct __transform_one
{
  _ValueFn __value_fn;
  _ErrorFn __error_fn;
  _StoppedFn __stopped_fn;

  template <class _Tag, class... _Ts>
  [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()(_Tag (*)(_Ts...)) const
  {
    if constexpr (_Tag{} == set_value)
    {
      return __apply_transform<_Ts...>(__value_fn);
    }
    else if constexpr (_Tag{} == set_error)
    {
      return __apply_transform<_Ts...>(__error_fn);
    }
    else
    {
      return __apply_transform<_Ts...>(__stopped_fn);
    }
  }
};

template <class _TransformOne>
struct __transform_all_fn
{
  _TransformOne __tfx1;

  template <class... _Sigs>
  [[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto operator()(_Sigs*... __sigs) const
  {
    return concat_completion_signatures(__tfx1(__sigs)...);
  }
};

template <class _TransformOne>
__transform_all_fn(_TransformOne) -> __transform_all_fn<_TransformOne>;

template <class _Completions,
          class _ValueFn   = __default_transform_fn<set_value_t>,
          class _ErrorFn   = __default_transform_fn<set_error_t>,
          class _StoppedFn = __default_transform_fn<set_stopped_t>,
          class _ExtraSigs = completion_signatures<>>
_CCCL_API _CCCL_CONSTEVAL auto transform_completion_signatures(
  _Completions, //
  _ValueFn __value_fn     = {},
  _ErrorFn __error_fn     = {},
  _StoppedFn __stopped_fn = {},
  _ExtraSigs              = {})
{
  _CUDAX_LET_COMPLETIONS(auto(__completions) = _Completions{})
  {
    _CUDAX_LET_COMPLETIONS(auto(__extra) = _ExtraSigs{})
    {
      __transform_one<_ValueFn, _ErrorFn, _StoppedFn> __tfx1{__value_fn, __error_fn, __stopped_fn};
      return concat_completion_signatures(__completions.apply(__transform_all_fn{__tfx1}), __extra);
    }
  }
}

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TRANSFORM_COMPLETION_SIGNATURES
