//===----------------------------------------------------------------------===//
//
// Copyright (c) Facebook, Inc. and its affiliates.
// Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___CONCEPTS_CONCEPT_MACROS_H
#define _CUDA___CONCEPTS_CONCEPT_MACROS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

////////////////////////////////////////////////////////////////////////////////
// _CCCL_TEMPLATE
// Usage:
//   _CCCL_TEMPLATE(typename A, typename _Bp)
//     _CCCL_REQUIRES( Concept1<A> _CCCL_AND Concept2<_Bp>)
//   void foo(A a, _Bp b)
//   {}

// Barebones enable if implementation to use outside of cuda::std
template <bool>
struct __cccl_select
{};

template <>
struct __cccl_select<true>
{
  template <class _Tp>
  using type = _Tp;
};

template <bool _Bp, class _Tp = void>
using __cccl_enable_if_t = typename __cccl_select<_Bp>::template type<_Tp>;

template <class _Tp, bool _Bp>
using __cccl_requires_t = typename __cccl_select<_Bp>::template type<_Tp>;

#if _CCCL_HAS_CONCEPTS() || defined(_CCCL_DOXYGEN_INVOKED)
#  define _CCCL_TEMPLATE(...)                template <__VA_ARGS__>
#  define _CCCL_REQUIRES(...)                requires __VA_ARGS__
#  define _CCCL_AND                          &&
#  define _CCCL_TRAILING_REQUIRES_IMPL_(...) requires __VA_ARGS__
#  define _CCCL_TRAILING_REQUIRES(...)       ->__VA_ARGS__ _CCCL_TRAILING_REQUIRES_IMPL_
#  define _CCCL_CONCEPT                      concept
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
#  define _CCCL_TEMPLATE(...)                template <__VA_ARGS__
#  define _CCCL_REQUIRES(...)                , bool __cccl_true_ = true, __cccl_enable_if_t < __VA_ARGS__ && __cccl_true_, int > = 0 >
#  define _CCCL_AND                          &&__cccl_true_, int > = 0, __cccl_enable_if_t <
#  define _CCCL_TRAILING_REQUIRES(...)       ->__cccl_requires_t < __VA_ARGS__ _CCCL_TRAILING_REQUIRES_IMPL_
#  define _CCCL_TRAILING_REQUIRES_IMPL_(...) , __VA_ARGS__ >
#  define _CCCL_CONCEPT                      inline constexpr bool
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

// The following concepts emulation macros need variable template support

template <class...>
struct __cccl_tag;

template <class>
_CCCL_API constexpr bool __cccl_is_true()
{
  return true;
}

#if _CCCL_COMPILER(MSVC)
template <bool _Bp>
_CCCL_API inline __cccl_enable_if_t<_Bp> __cccl_requires()
{}
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
template <bool _Bp, __cccl_enable_if_t<_Bp, int> = 0>
inline constexpr int __cccl_requires = 0;
#endif // !_CCCL_COMPILER(MSVC)

template <class _Tp, class... _Args>
extern _Tp __cccl_make_dependent;

template <class _Impl, class... _Args>
using __cccl_requires_expr_impl = decltype(__cccl_make_dependent<_Impl, _Args...>);

template <typename _Tp>
_CCCL_API constexpr void __cccl_unused(_Tp&&) noexcept
{}

// So that we can refer to the ::cuda::std namespace below
_CCCL_BEGIN_NAMESPACE_CUDA_STD
_CCCL_END_NAMESPACE_CUDA_STD

// We put an alias for ::cuda::std here because of a bug in nvcc <12.2
// where a requirement such as:
//
//  { expression } -> ::concept<type>
//
// where ::concept is a fully qualified name, would not compile. The
// ::cuda::std macro is fully qualified.
namespace __cccl_unqualified_cuda_std = ::cuda::std; // NOLINT(misc-unused-alias-decls)

#if _CCCL_CUDACC_BELOW(12, 2)
#  define _CCCL_CONCEPT_VSTD __cccl_unqualified_cuda_std // must not be fully qualified
#else
#  define _CCCL_CONCEPT_VSTD ::cuda::std
#endif

// GCC < 14 can't mangle noexcept expressions. See
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=70790.
#if _CCCL_COMPILER(GCC, <, 14)
#  define _CCCL_HAS_NOEXCEPT_MANGLING() 0
#else
#  define _CCCL_HAS_NOEXCEPT_MANGLING() 1
#endif

// We use this macro to ignore the result of required expressions. It is needed because
// gcc < 10 complains about ignored [[nodiscard]] expressions when emulating concepts.
#if _CCCL_COMPILER(GCC, <, 10)
#  define _CCCL_CONCEPT_IGNORE_RESULT_(...) static_cast<void>(__VA_ARGS__)
#else
#  define _CCCL_CONCEPT_IGNORE_RESULT_(...) __VA_ARGS__
#endif

// The "0" or "1" suffixes indicate whether _REQ is parenthesized or not.
#define _CCCL_CONCEPT_REQUIREMENT_0(_REQ) _CCCL_PP_SWITCH(_CCCL_CONCEPT_REQUIREMENT, _REQ)
#define _CCCL_CONCEPT_REQUIREMENT_1(_REQ) _CCCL_CONCEPT_IGNORE_RESULT_ _REQ

// Permissible requirements are of the form (where ... indicates that the pattern can
// contain commas):
//
// - EXPR
// - (EXPR...)
// - noexcept(EXPR...)
// - requires(BOOL-EXPR...)
// - typename(TYPE...)
// - _Same_as(TYPE...) EXPR...
// - _Satisfies(CONCEPT...) EXPR...
//
// The last 4 are handled below:
#define _CCCL_CONCEPT_REQUIREMENT_SWITCH_requires   _CCCL_PP_CASE(REQUIRES)
#define _CCCL_CONCEPT_REQUIREMENT_SWITCH_noexcept   _CCCL_PP_CASE(NOEXCEPT)
#define _CCCL_CONCEPT_REQUIREMENT_SWITCH_typename   _CCCL_PP_CASE(TYPENAME)
#define _CCCL_CONCEPT_REQUIREMENT_SWITCH__Same_as   _CCCL_PP_CASE(SAME_AS)
#define _CCCL_CONCEPT_REQUIREMENT_SWITCH__Satisfies _CCCL_PP_CASE(SATISFIES)

// Converts "requires(ARGS...)" to "ARGS..."
#define _CCCL_CONCEPT_EAT_REQUIRES_(...)         _CCCL_PP_CAT(_CCCL_CONCEPT_EAT_REQUIRES_, __VA_ARGS__)
#define _CCCL_CONCEPT_EAT_REQUIRES_requires(...) __VA_ARGS__

// Converts "noexcept(ARGS...)" to "ARGS..."
#define _CCCL_CONCEPT_EAT_NOEXCEPT_(...)         _CCCL_PP_CAT(_CCCL_CONCEPT_EAT_NOEXCEPT_, __VA_ARGS__)
#define _CCCL_CONCEPT_EAT_NOEXCEPT_noexcept(...) __VA_ARGS__

// Converts "typename(TYPE...)" to "TYPE..."
#define _CCCL_CONCEPT_EAT_TYPENAME_(_REQ)        _CCCL_PP_CAT2(_CCCL_CONCEPT_EAT_TYPENAME_, _REQ)
#define _CCCL_CONCEPT_EAT_TYPENAME_typename(...) __VA_ARGS__

// Converts "[typename]opt TYPE..." to "typename TYPE..."
#define _CCCL_CONCEPT_TRY_ADD_TYPENAME_(...)              _CCCL_PP_SWITCH2(_CCCL_CONCEPT_TRY_ADD_TYPENAME, __VA_ARGS__)
#define _CCCL_CONCEPT_TRY_ADD_TYPENAME_SWITCH_typename    _CCCL_PP_CASE(TYPENAME)
#define _CCCL_CONCEPT_TRY_ADD_TYPENAME_CASE_DEFAULT(...)  typename __VA_ARGS__
#define _CCCL_CONCEPT_TRY_ADD_TYPENAME_CASE_TYPENAME(...) __VA_ARGS__

// Converts "_Same_as(TYPE) EXPR..." to "EXPR..."
#define _CCCL_CONCEPT_EAT_SAME_AS_(...) _CCCL_PP_CAT(_CCCL_CONCEPT_EAT_SAME_AS_, __VA_ARGS__)
#define _CCCL_CONCEPT_EAT_SAME_AS__Same_as(...)

// Converts "_Same_as(TYPE) EXPR..." to "TYPE" (The ridiculous concatenation of _CCCL with
// _PP_EXPAND(__VA_ARGS__) is the only way to get MSVC's broken preprocessor to do macro
// expansion here.)
#define _CCCL_CONCEPT_GET_TYPE_FROM_SAME_AS_(...) \
  _CCCL_PP_CAT(_CCCL, _CCCL_PP_EVAL(_CCCL_PP_FIRST, _CCCL_PP_CAT(_CCCL_CONCEPT_GET_TYPE_FROM_SAME_AS_, __VA_ARGS__)))
#define _CCCL_CONCEPT_GET_TYPE_FROM_SAME_AS__Same_as(...) _PP_EXPAND(__VA_ARGS__),

// Converts "_Satisfies(TYPE) EXPR..." to "EXPR..."
#define _CCCL_CONCEPT_EAT_SATISFIES_(...) _CCCL_PP_CAT(_CCCL_CONCEPT_EAT_SATISFIES_, __VA_ARGS__)
#define _CCCL_CONCEPT_EAT_SATISFIES__Satisfies(...)

// Converts "_Satisfies(TYPE) EXPR..." to "TYPE" (The ridiculous concatenation of _CCCL_PP_
// with EXPAND(__VA_ARGS__) is the only way to get MSVC's broken preprocessor to do macro
// expansion here.)
#define _CCCL_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(...) \
  _CCCL_PP_CAT(_CCCL_PP_,                              \
               _CCCL_PP_EVAL(_CCCL_PP_FIRST, _CCCL_PP_CAT(_CCCL_CONCEPT_GET_CONCEPT_FROM_SATISFIES_, __VA_ARGS__)))
#define _CCCL_CONCEPT_GET_CONCEPT_FROM_SATISFIES__Satisfies(...) EXPAND(__VA_ARGS__),

// Here are the implementations of the internal macros, first for when concepts
// are available, and then for when they're not.
#if _CCCL_HAS_CONCEPTS() || defined(_CCCL_DOXYGEN_INVOKED)

// "_CCCL_CONCEPT_FRAGMENT(NAME, ARGS...)(REQS...)" expands into
// "concept NAME = requires(ARGS...) { _CCCL_CONCEPT_REQUIREMENT_(REQS)... }"
#  define _CCCL_CONCEPT_FRAGMENT(_NAME, ...)                concept _NAME = _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_##__VA_ARGS__
#  define _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_requires(...) requires(__VA_ARGS__) _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_
#  define _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_(...)         {_CCCL_PP_FOR_EACH(_CCCL_CONCEPT_REQUIREMENT_, __VA_ARGS__)}

// Converts "EXPR" to "_CCCL_CONCEPT_REQUIREMENT_0(EXPR)", and
// "(EXPR)" to "_CCCL_CONCEPT_REQUIREMENT_1((EXPR))"
#  define _CCCL_CONCEPT_REQUIREMENT_(_REQ)                            \
    _CCCL_PP_CAT(_CCCL_CONCEPT_REQUIREMENT_, _CCCL_PP_IS_PAREN(_REQ)) \
    (_REQ);

// The following macros handle the various special forms of requirements:
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_DEFAULT(_REQ)  _REQ
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_REQUIRES(_REQ) requires _CCCL_CONCEPT_EAT_REQUIRES_(_REQ)
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_NOEXCEPT(_REQ) _CCCL_PP_EXPAND({ _CCCL_CONCEPT_EAT_NOEXCEPT_(_REQ) } noexcept)
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_TYPENAME(_REQ) \
    _CCCL_CONCEPT_TRY_ADD_TYPENAME_(_CCCL_CONCEPT_EAT_TYPENAME_(_REQ))
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_SAME_AS(_REQ) \
    {_CCCL_CONCEPT_EAT_SAME_AS_(_REQ)}->_CCCL_CONCEPT_VSTD::same_as<_CCCL_CONCEPT_GET_TYPE_FROM_SAME_AS_(_REQ)>
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_SATISFIES(_REQ) \
    {_CCCL_CONCEPT_EAT_SATISFIES_(_REQ)}->_CCCL_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(_REQ)

#  define _CCCL_FRAGMENT(_NAME, ...) _NAME<__VA_ARGS__>

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

// "_CCCL_CONCEPT_FRAGMENT(Foo, ARGS...)(REQS...)" expands into:
//
// _CCCL_API inline auto Foo_CCCL_CONCEPT_FRAGMENT_impl_(ARGS...)
//   -> __cccl_enable_if_t<
//        ::__cccl_is_true<decltype(_CCCL_CONCEPT_REQUIREMENT_(REQS)..., void())>()>
// {}
//
// template <class... As>
// _CCCL_API inline auto Foo_CCCL_CONCEPT_FRAGMENT_(::__cccl_tag<As...>*,
//                                                  decltype(&Foo_CCCL_CONCEPT_FRAGMENT_impl_<As...>))
//   -> char(&)[1];
//
// template <class... As>
// _CCCL_API inline auto Foo_CCCL_CONCEPT_FRAGMENT_(...)
//   -> char(&)[2]
//
#  define _CCCL_CONCEPT_FRAGMENT(_NAME, ...)                                                                        \
    _CCCL_API inline auto _NAME##_CCCL_CONCEPT_FRAGMENT_impl_ _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_##__VA_ARGS__> {} \
    template <class... _As>                                                                                         \
    _CCCL_API inline auto _NAME##_CCCL_CONCEPT_FRAGMENT_(                                                           \
      ::__cccl_tag<_As...>*, decltype(&_NAME##_CCCL_CONCEPT_FRAGMENT_impl_<_As...>)) -> char (&)[1];                \
    _CCCL_API inline auto _NAME##_CCCL_CONCEPT_FRAGMENT_(...) -> char (&)[2]
#  define _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_requires(...) \
    (__VA_ARGS__)->__cccl_enable_if_t < _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_IMPL_
#  define _CCCL_CONCEPT_FRAGMENT_REQUIREMENTS_IMPL_(...) \
    ::__cccl_is_true<decltype(_CCCL_PP_FOR_EACH(_CCCL_CONCEPT_REQUIREMENT_, __VA_ARGS__) void())>()

// Called with each individual requirement in the list of requirements
#  define _CCCL_CONCEPT_REQUIREMENT_(_REQ) \
    void(), _CCCL_PP_CAT(_CCCL_CONCEPT_REQUIREMENT_, _CCCL_PP_IS_PAREN(_REQ))(_REQ),

// The following macros handle the various special forms of requirements:
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_DEFAULT(_REQ)  _CCCL_CONCEPT_IGNORE_RESULT_(_REQ)
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_REQUIRES(_REQ) ::__cccl_requires<_CCCL_CONCEPT_EAT_REQUIRES_(_REQ)>
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_NOEXCEPT(_REQ) _CCCL_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ)
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_TYPENAME(_REQ) \
    static_cast<::__cccl_tag<_CCCL_CONCEPT_EAT_TYPENAME_(_REQ)>*>(nullptr)
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_SAME_AS(_REQ) \
    ::__cccl_requires<::cuda::std::same_as<_CCCL_CONCEPT_SAME_AS_REQUIREMENT_(_REQ)>>
#  define _CCCL_CONCEPT_REQUIREMENT_CASE_SATISFIES(_REQ)                                                               \
    ::__cccl_requires < _CCCL_CONCEPT_GET_CONCEPT_FROM_SATISFIES_(_REQ) < decltype(_CCCL_CONCEPT_EAT_SATISFIES_(_REQ)) \
      >>

// Converts "_Same_as(TYPE) EXPR..." to "TYPE, decltype(EXPR...)"
#  define _CCCL_CONCEPT_SAME_AS_REQUIREMENT_(_REQ) \
    _CCCL_CONCEPT_GET_TYPE_FROM_SAME_AS_(_REQ), decltype(_CCCL_CONCEPT_EAT_SAME_AS_(_REQ))

#  if _CCCL_HAS_NOEXCEPT_MANGLING()
// Converts "noexcept(EXPR)" to "::__cccl_requires<noexcept(EXPR)>"
#    define _CCCL_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ) ::__cccl_requires<_REQ>
#  else
// If the compiler cannot mangle noexcept expressions, just check that the expression is
// well-formed. This converts "noexcept(EXPR)" to "static_cast<void>(EXPR)"
#    define _CCCL_CONCEPT_NOEXCEPT_REQUIREMENT_(_REQ) _CCCL_CONCEPT_IGNORE_RESULT_(_CCCL_CONCEPT_EAT_NOEXCEPT_(_REQ))
#  endif

// "_CCCL_FRAGMENT(Foo, Args...)" expands to
// "(1 == sizeof(Foo_CCCL_CONCEPT_FRAGMENT_(static_cast<::__cccl_tag<Args...>*>(nullptr), nullptr)))"
#  define _CCCL_FRAGMENT(_NAME, ...) \
    (1 == sizeof(_NAME##_CCCL_CONCEPT_FRAGMENT_(static_cast<::__cccl_tag<__VA_ARGS__>*>(nullptr), nullptr)))

#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

////////////////////////////////////////////////////////////////////////////////
// _CCCL_REQUIRES_EXPR
// Usage:
//   template <typename T>
//   _CCCL_CONCEPT equality_comparable =
//     _CCCL_REQUIRES_EXPR((T), T const& lhs, T const& rhs) (
//       lhs == rhs,
//       lhs != rhs
//     );
//
// Can only be used as the last requirement in a concept definition.
#if _CCCL_HAS_CONCEPTS() || defined(_CCCL_DOXYGEN_INVOKED)

#  define _CCCL_REQUIRES_EXPR(_TY, ...)  requires(__VA_ARGS__) _CCCL_REQUIRES_EXPR_IMPL_
#  define _CCCL_REQUIRES_EXPR_IMPL_(...) {_CCCL_PP_FOR_EACH(_CCCL_CONCEPT_REQUIREMENT_, __VA_ARGS__)}

#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv

#  define _CCCL_REQUIRES_EXPR(_TY, ...) _CCCL_REQUIRES_EXPR_IMPL(_TY, _CCCL_REQUIRES_EXPR_ID(_TY), __VA_ARGS__)
#  define _CCCL_REQUIRES_EXPR_IMPL(_TY, _ID, ...)                                                                    \
    ::__cccl_requires_expr_impl<                                                                                     \
      struct _CCCL_PP_CAT(__cccl_requires_expr_detail_, _ID) _CCCL_REQUIRES_EXPR_TPARAM_REFS                         \
        _TY>::__cccl_is_satisfied(static_cast<::__cccl_tag<void _CCCL_REQUIRES_EXPR_TPARAM_REFS _TY>*>(nullptr), 0); \
    struct _CCCL_PP_CAT(__cccl_requires_expr_detail_, _ID)                                                           \
    {                                                                                                                \
      using __cccl_self_t = _CCCL_PP_CAT(__cccl_requires_expr_detail_, _ID);                                         \
      template <class _CCCL_REQUIRES_EXPR_TPARAM_DEFNS _TY>                                                          \
      _CCCL_API inline static auto __cccl_well_formed(__VA_ARGS__) _CCCL_REQUIRES_EXPR_REQUIREMENTS_

// Expands "T1, T2, variadic T3" to ", class T1, class T2, class... T3"
#  define _CCCL_REQUIRES_EXPR_TPARAM_DEFNS(...)             _CCCL_PP_FOR_EACH(_CCCL_REQUIRES_EXPR_TPARAM_DEFN, __VA_ARGS__)

// Expands "TY" to ", class TY" and "variadic TY" to ", class... TY"
#  define _CCCL_REQUIRES_EXPR_TPARAM_DEFN(_TY)              , _CCCL_PP_SWITCH2(_CCCL_REQUIRES_EXPR_TPARAM_DEFN, _TY)
#  define _CCCL_REQUIRES_EXPR_TPARAM_DEFN_SWITCH_variadic   _CCCL_PP_CASE(VARIADIC)
#  define _CCCL_REQUIRES_EXPR_TPARAM_DEFN_CASE_DEFAULT(_TY) class _TY
#  define _CCCL_REQUIRES_EXPR_TPARAM_DEFN_CASE_VARIADIC(_TY) \
    class... _CCCL_PP_CAT(_CCCL_REQUIRES_EXPR_EAT_VARIADIC_, _TY)

// Expands "T1, T2, variadic T3" to ", T1, T2, T3..."
#  define _CCCL_REQUIRES_EXPR_TPARAM_REFS(...)              _CCCL_PP_FOR_EACH(_CCCL_REQUIRES_EXPR_TPARAM_REF, __VA_ARGS__)

// Expands "TY" to ", TY" and "variadic TY" to ", TY..."
#  define _CCCL_REQUIRES_EXPR_TPARAM_REF(_TY)               , _CCCL_PP_SWITCH2(_CCCL_REQUIRES_EXPR_TPARAM_REF, _TY)
#  define _CCCL_REQUIRES_EXPR_TPARAM_REF_SWITCH_variadic    _CCCL_PP_CASE(VARIADIC)
#  define _CCCL_REQUIRES_EXPR_TPARAM_REF_CASE_DEFAULT(_TY)  _TY
#  define _CCCL_REQUIRES_EXPR_TPARAM_REF_CASE_VARIADIC(_TY) _CCCL_PP_CAT(_CCCL_REQUIRES_EXPR_EAT_VARIADIC_, _TY)...

// NVRTC does not support __COUNTER__ so we need a better way of defining unique identifiers
#  if _CCCL_COMPILER(NVRTC)

// Expands ((Ty...), Ty...) into _CCCL_REQUIRES_EXPR_ID_NO_PAREN(Ty...)
#    define _CCCL_REQUIRES_EXPR_ID(_TY, ...) _CCCL_REQUIRES_EXPR_ID_NO_PAREN _TY

// Expands "T1, T2, variadic T3" to "T1_T2_T3_##__LINE__"
#    define _CCCL_REQUIRES_EXPR_ID_NO_PAREN(...) \
      _CCCL_REQUIRES_EXPR_ID_CONCAT_ALL(_CCCL_PP_FOR_EACH(_CCCL_REQUIRES_EXPR_ID_IMPL, __VA_ARGS__), _CCCL_COUNTER())

// Expands "T1, T2, T3" to "T1T2T3"
#    define _CCCL_REQUIRES_EXPR_ID_CONCAT_ALL_IMPL(_0, _1, _2, _3, _4, _5, _6, _7, _8, _9, ...) \
      _0##_1##_2##_3##_4##_5##_6##_7##_8##_9
#    define _CCCL_REQUIRES_EXPR_ID_CONCAT_ALL(...) \
      _CCCL_PP_EVAL(_CCCL_REQUIRES_EXPR_ID_CONCAT_ALL_IMPL, __VA_ARGS__, , , , , , , , , )

// Expands "TY" to "TY" and "variadic TY" to "TY"
#    define _CCCL_REQUIRES_EXPR_ID_IMPL(_TY)               , _CCCL_PP_SWITCH2(_CCCL_REQUIRES_EXPR_ID_IMPL, _TY)
#    define _CCCL_REQUIRES_EXPR_ID_IMPL_SWITCH_variadic    _CCCL_PP_CASE(VARIADIC)
#    define _CCCL_REQUIRES_EXPR_ID_IMPL_CASE_DEFAULT(_TY)  _TY
#    define _CCCL_REQUIRES_EXPR_ID_IMPL_CASE_VARIADIC(_TY) _CCCL_PP_CAT(_CCCL_REQUIRES_EXPR_EAT_VARIADIC_, _TY)

#  else // ^^^ _CCCL_COMPILER(NVRTC) ^^^^/ vvv !_CCCL_COMPILER(NVRTC)
#    define _CCCL_REQUIRES_EXPR_ID(...) _CCCL_COUNTER()
#  endif // !_CCCL_COMPILER(NVRTC)

#  define _CCCL_REQUIRES_EXPR_EAT_VARIADIC_variadic

#  define _CCCL_REQUIRES_EXPR_REQUIREMENTS_(...)                                              \
    ->decltype(_CCCL_PP_FOR_EACH(_CCCL_CONCEPT_REQUIREMENT_, __VA_ARGS__) void()) {}          \
    template <class... _Args, class = decltype(&__cccl_self_t::__cccl_well_formed<_Args...>)> \
    _CCCL_API static constexpr bool __cccl_is_satisfied(::__cccl_tag<_Args...>*, int)         \
    {                                                                                         \
      return true;                                                                            \
    }                                                                                         \
    _CCCL_API static constexpr bool __cccl_is_satisfied(void*, long)                          \
    {                                                                                         \
      return false;                                                                           \
    }                                                                                         \
    }
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDA___CONCEPTS_CONCEPT_MACROS_H
