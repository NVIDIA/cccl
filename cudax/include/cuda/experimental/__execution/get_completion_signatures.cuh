//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_GET_COMPLETION_SIGNATURES
#define __CUDAX_EXECUTION_GET_COMPLETION_SIGNATURES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/copy_cvref.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh> // IWYU pragma: export
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>

// include this last:
#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{

#if __cpp_lib_constexpr_exceptions >= 202502L // constexpr exception types, https://wg21.link/p3378

using __exception = ::std::exception;

#elif __cpp_constexpr >= 202411L // constexpr virtual functions

struct _CCCL_TYPE_VISIBILITY_DEFAULT __exception
{
  _CCCL_HIDE_FROM_ABI constexpr __exception() noexcept = default;
  _CCCL_HIDE_FROM_ABI virtual constexpr ~__exception() = default;

  [[nodiscard]] _CCCL_API virtual constexpr auto what() const noexcept -> const char*
  {
    return "<exception>";
  }
};

#else // no constexpr virtual functions:

struct _CCCL_TYPE_VISIBILITY_DEFAULT __exception
{
  _CCCL_HIDE_FROM_ABI constexpr __exception() noexcept = default;

  [[nodiscard]] _CCCL_API constexpr auto what() const noexcept -> const char*
  {
    return "<exception>";
  }
};

#endif // __cpp_lib_constexpr_exceptions >= 202502L

template <class _Derived>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __compile_time_error : __exception
{
  _CCCL_HIDE_FROM_ABI __compile_time_error() = default;

  [[nodiscard]] _CCCL_API constexpr auto what() const noexcept -> const char*
  {
    return static_cast<_Derived const*>(this)->__what();
  }
};

template <class _Data, class... _What>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sender_type_check_failure //
    : __compile_time_error<__sender_type_check_failure<_Data, _What...>>
{
  static_assert(__nothrow_movable<_Data>,
                "The data member of __sender_type_check_failure must be nothrow move constructible.");

  _CCCL_HIDE_FROM_ABI constexpr __sender_type_check_failure() noexcept = default;

  _CCCL_API constexpr explicit __sender_type_check_failure(_Data __data)
      : __data_(static_cast<_Data&&>(__data))
  {}

private:
  friend struct __compile_time_error<__sender_type_check_failure>;

  _CCCL_NODEBUG_API constexpr auto __what() const noexcept -> const char*
  {
    return "This sender is not well-formed. It does not meet the requirements of a sender type.";
  }

  _Data __data_{};
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT dependent_sender_error : __compile_time_error<dependent_sender_error>
{
  _CCCL_API constexpr explicit dependent_sender_error(char const* __what) noexcept
      : __what_(__what)
  {}

private:
  friend struct __compile_time_error<dependent_sender_error>;

  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto __what() const noexcept -> char const*
  {
    return __what_;
  }

  char const* __what_;
};

template <class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __dependent_sender_error : dependent_sender_error
{
  _CCCL_NODEBUG_API constexpr __dependent_sender_error() noexcept
      : dependent_sender_error{"This sender needs to know its execution " //
                               "environment before it can know how it will complete."}
  {}

  _CCCL_HOST_DEVICE auto operator+() -> __dependent_sender_error;

  template <class _Ty>
  _CCCL_HOST_DEVICE auto operator,(_Ty&) -> __dependent_sender_error&;

  template <class... _What>
  _CCCL_HOST_DEVICE auto operator,(_ERROR<_What...>&) -> _ERROR<_What...>&;
};

// Below is the definition of the _CUDAX_LET_COMPLETIONS portability macro. It
// is used to check that an expression's type is a valid completion_signature
// specialization.
//
// USAGE:
//
//   _CUDAX_LET_COMPLETIONS(auto(__cs) = <expression>)
//   {
//     // __cs is guaranteed to be a specialization of completion_signatures.
//   }
//
// When constexpr exceptions are available (C++26), the macro simply expands to
// the moral equivalent of:
//
//   // With constexpr exceptions:
//   auto __cs = <expression>; // throws if __cs is not a completion_signatures
//
// When constexpr exceptions are not available, the macro expands to:
//
//   // Without constexpr exceptions:
//   if constexpr (auto __cs = <expression>; !__valid_completion_signatures<decltype(__cs)>)
//   {
//     return __cs;
//   }
//   else

#if _CCCL_HAS_EXCEPTIONS() && __cpp_constexpr_exceptions >= 202411L // C++26, https://wg21.link/p3068

#  define _CUDAX_LET_COMPLETIONS(...)                  \
    if constexpr ([[maybe_unused]] __VA_ARGS__; false) \
    {                                                  \
    }                                                  \
    else

template <class... _Sndr>
[[noreturn, nodiscard]] _CCCL_API consteval auto __dependent_sender() -> completion_signatures<>
{
  throw __dependent_sender_error<_Sndr...>{};
}

#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv

#  define _CUDAX_PP_EAT_AUTO_auto(_ID)    _ID _CCCL_PP_EAT _CCCL_PP_LPAREN
#  define _CUDAX_PP_EXPAND_AUTO_auto(_ID) auto _ID
#  define _CUDAX_LET_COMPLETIONS_ID(...)  _CCCL_PP_EXPAND(_CCCL_PP_CAT(_CUDAX_PP_EAT_AUTO_, __VA_ARGS__) _CCCL_PP_RPAREN)

#  define _CUDAX_LET_COMPLETIONS(...)                                                                                 \
    if constexpr (_CCCL_PP_CAT(_CUDAX_PP_EXPAND_AUTO_, __VA_ARGS__);                                                  \
                  !::cuda::experimental::execution::__valid_completion_signatures<decltype(_CUDAX_LET_COMPLETIONS_ID( \
                    __VA_ARGS__))>)                                                                                   \
    {                                                                                                                 \
      return _CUDAX_LET_COMPLETIONS_ID(__VA_ARGS__);                                                                  \
    }                                                                                                                 \
    else

template <class... _Sndr>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto __dependent_sender() -> __dependent_sender_error<_Sndr...>
{
  return __dependent_sender_error<_Sndr...>{};
}

#endif // ^^^ no constexpr exceptions ^^^

////////////////////////////////////////////////////////////////////////////////////////////////////
// get_completion_signatures
_CCCL_DIAG_PUSH
// warning C4913: user defined binary operator ',' exists but no overload could convert all operands,
// default built-in binary operator ',' used
_CCCL_DIAG_SUPPRESS_MSVC(4913)

#define _CUDAX_GET_COMPLSIGS(...) \
  ::cuda::std::remove_reference_t<_Sndr>::template get_completion_signatures<__VA_ARGS__>()

#define _CUDAX_CHECKED_COMPLSIGS(...) \
  (static_cast<void>(__VA_ARGS__), void(), execution::__checked_complsigs<decltype(__VA_ARGS__)>())

struct _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION
{};

template <class _Completions>
_CCCL_NODEBUG_API _CCCL_CONSTEVAL auto __checked_complsigs()
{
  _CUDAX_LET_COMPLETIONS(auto(__cs) = _Completions())
  {
    if constexpr (__valid_completion_signatures<_Completions>)
    {
      return __cs;
    }
    else
    {
      return invalid_completion_signature<
        _A_GET_COMPLETION_SIGNATURES_CUSTOMIZATION_RETURNED_A_TYPE_THAT_IS_NOT_A_COMPLETION_SIGNATURES_SPECIALIZATION,
        _WITH_SIGNATURES(_Completions)>();
    }
  }
}

template <class _Sndr, class... _Env>
inline constexpr bool __has_get_completion_signatures = false;

// clang-format off
template <class _Sndr>
inline constexpr bool __has_get_completion_signatures<_Sndr> =
  _CCCL_REQUIRES_EXPR((_Sndr))
  (
    (_CUDAX_GET_COMPLSIGS(_Sndr))
  );

template <class _Sndr, class _Env>
inline constexpr bool __has_get_completion_signatures<_Sndr, _Env> =
  _CCCL_REQUIRES_EXPR((_Sndr, _Env))
  (
    (_CUDAX_GET_COMPLSIGS(_Sndr, _Env))
  );
// clang-format on

struct _COULD_NOT_DETERMINE_COMPLETION_SIGNATURES_FOR_THIS_SENDER
{};

_CCCL_EXEC_CHECK_DISABLE
template <class _Sndr, class... _Env>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto __get_completion_signatures_helper()
{
  if constexpr (__has_get_completion_signatures<_Sndr, _Env...>)
  {
    return _CUDAX_CHECKED_COMPLSIGS(_CUDAX_GET_COMPLSIGS(_Sndr, _Env...));
  }
  else if constexpr (__has_get_completion_signatures<_Sndr>)
  {
    return _CUDAX_CHECKED_COMPLSIGS(_CUDAX_GET_COMPLSIGS(_Sndr));
  }
  // else if constexpr (__is_awaitable<_Sndr, __env_promise<_Env>...>)
  // {
  //   using Result _CCCL_NODEBUG_ALIAS = __await_result_t<_Sndr, __env_promise<_Env>...>;
  //   return completion_signatures{__set_value_v<Result>, __set_error_v<>, __set_stopped_v};
  // }
  else if constexpr (sizeof...(_Env) == 0)
  {
    return __dependent_sender<_Sndr>();
  }
  else
  {
    return invalid_completion_signature<_COULD_NOT_DETERMINE_COMPLETION_SIGNATURES_FOR_THIS_SENDER,
                                        _WITH_SENDER(_Sndr),
                                        _WITH_ENVIRONMENT(_Env...)>();
  }
}

template <class _Sndr, class... _Env>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto get_completion_signatures()
{
  static_assert(sizeof...(_Env) <= 1, "At most one environment is allowed.");
  if constexpr (0 == sizeof...(_Env))
  {
    return execution::__get_completion_signatures_helper<_Sndr>();
  }
  else if constexpr (!__has_sender_transform<_Sndr, _Env...>)
  {
    return execution::__get_completion_signatures_helper<_Sndr, _Env...>();
  }
  else
  {
    // Apply a lazy sender transform if one exists before computing the completion signatures:
    using __new_sndr_t = __call_result_t<transform_sender_t, __domain_of_t<_Sndr, _Env...>, _Sndr, _Env...>;
    return execution::__get_completion_signatures_helper<__new_sndr_t, _Env...>();
  }
}

template <class _Parent, class _Child, class... _Env>
[[nodiscard]] _CCCL_NODEBUG_API _CCCL_CONSTEVAL auto get_child_completion_signatures()
{
  return get_completion_signatures<::cuda::std::__copy_cvref_t<_Parent, _Child>, __fwd_env_t<_Env>...>();
}

#undef _CUDAX_GET_COMPLSIGS
#undef _CUDAX_CHECKED_COMPLSIGS
_CCCL_DIAG_POP

#if _CCCL_HAS_EXCEPTIONS() && __cpp_constexpr_exceptions >= 202411L // C++26, https://wg21.link/p3068
// When asked for its completions without an envitonment, a dependent sender
// will throw an exception of a type derived from `dependent_sender_error`.
template <class _Sndr>
[[nodiscard]] _CCCL_API consteval bool __is_dependent_sender() noexcept
try
{
  (void) get_completion_signatures<_Sndr>();
  return false; // didn't throw, not a dependent sender
}
catch (dependent_sender_error&)
{
  return true;
}
catch (...)
{
  return false; // different kind of exception was thrown; not a dependent sender
}
#else // ^^^ constexpr exceptions ^^^ / vvv no constexpr exceptions vvv
template <class _Sndr>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __is_dependent_sender() noexcept -> bool
{
  using _Completions _CCCL_NODEBUG_ALIAS = decltype(get_completion_signatures<_Sndr>());
  return ::cuda::std::is_base_of_v<dependent_sender_error, _Completions>;
}
#endif // ^^^ no constexpr exceptions ^^^

template <class _SetTag, class _Sndr, class... _Env>
_CCCL_CONCEPT __has_completions_for = _CCCL_REQUIRES_EXPR((_SetTag, _Sndr, variadic _Env)) //
  ( //
    typename(completion_signatures_of_t<_Sndr, _Env...>),
    requires(completion_signatures_of_t<_Sndr, _Env...>::count(_SetTag{}) != 0) //
  );

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_GET_COMPLETION_SIGNATURES
