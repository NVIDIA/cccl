//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_META
#define __CUDAX_ASYNC_DETAIL_META

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <cuda/experimental/__async/config.cuh>

#if __cpp_lib_three_way_comparison
#  include <compare>
#endif

#include <cuda/experimental/__async/prologue.cuh>

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-string-literal-operator-template")
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")

namespace cuda::experimental::__async
{
template <class _Ret, class... _Args>
using __fn_t = _Ret(_Args...);

template <class _Ret, class... _Args>
using __nothrow_fn_t = _Ret(_Args...) noexcept;

template <class _Ty>
_Ty&& __declval() noexcept;

template <class...>
using __mvoid = void;

template <class _Ty>
struct __mtype
{
  using type = _Ty;
};

template <class _Ty>
using __t = typename _Ty::type;

template <class... _Ts>
struct __mlist;

template <auto _Val>
struct __mvalue
{
  static constexpr auto __value = _Val;
};

// A separate __mbool template is needed in addition to __mvalue
// because of an EDG bug in the handling of auto template parameters.
template <bool _Val>
struct __mbool
{
  static constexpr auto __value = _Val;
};

using __mtrue  = __mbool<true>;
using __mfalse = __mbool<false>;

template <auto... _Vals>
struct __mvalues;

template <size_t... _Vals>
struct __moffsets;

template <class... _Bools>
using __mand = __mbool<(_Bools::__value && ...)>;

template <class... _Bools>
using __mor = __mbool<(_Bools::__value || ...)>;

template <size_t... _Idx>
using __mindices = _CUDA_VSTD::index_sequence<_Idx...>*;

template <size_t Count>
using __mmake_indices = _CUDA_VSTD::make_index_sequence<Count>*;

template <class... _Ts>
using __mmake_indices_for = _CUDA_VSTD::make_index_sequence<sizeof...(_Ts)>*;

constexpr size_t __mpow2(size_t __size) noexcept
{
  --__size;
  __size |= __size >> 1;
  __size |= __size >> 2;
  __size |= __size >> 4;
  __size |= __size >> 8;
  if constexpr (sizeof(__size) >= 4)
  {
    __size |= __size >> 16;
  }
  if constexpr (sizeof(__size) >= 8)
  {
    __size |= __size >> 32;
  }
  return ++__size;
}

template <class _Ty>
constexpr _Ty __mmin(_Ty __lhs, _Ty __rhs) noexcept
{
  return __lhs < __rhs ? __lhs : __rhs;
}

template <class _Ty>
constexpr int __mcompare(_Ty __lhs, _Ty __rhs) noexcept
{
  return __lhs < __rhs ? -1 : __lhs > __rhs ? 1 : 0;
}

template <size_t _Len>
struct __mstring
{
  template <size_t _Ny, size_t... _Is>
  constexpr __mstring(const char (&__str)[_Ny], __mindices<_Is...>) noexcept
      : __len_{_Ny}
      , __what_{(_Is < _Ny ? __str[_Is] : '\0')...}
  {}

  template <size_t _Ny>
  constexpr __mstring(const char (&__str)[_Ny], int = 0) noexcept
      : __mstring{__str, __mmake_indices<_Len>{}}
  {}

  constexpr auto length() const noexcept -> size_t
  {
    return __len_;
  }

  template <size_t _OtherLen>
  constexpr int compare(const __mstring<_OtherLen>& __other) const noexcept
  {
    size_t const len = __mmin(__len_, __other.__len_);
    for (size_t i = 0; i < len; ++i)
    {
      if (auto const cmp = __mcompare(__what_[i], __other.__what_[i]))
      {
        return cmp;
      }
    }
    return __mcompare(__len_, __other.__len_);
  }

  template <size_t _OtherLen>
  constexpr auto operator==(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return __len_ == __other.__len_ && compare(__other) == 0;
  }

  template <size_t _OtherLen>
  constexpr auto operator!=(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return !operator==(__other);
  }

  template <size_t _OtherLen>
  constexpr auto operator<(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return compare(__other) < 0;
  }

  template <size_t _OtherLen>
  constexpr auto operator>(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return compare(__other) > 0;
  }

  template <size_t _OtherLen>
  constexpr auto operator<=(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return compare(__other) <= 0;
  }

  template <size_t _OtherLen>
  constexpr auto operator>=(const __mstring<_OtherLen>& __other) const noexcept -> bool
  {
    return compare(__other) >= 0;
  }

  size_t __len_;
  char __what_[_Len];
};

template <size_t _Len>
__mstring(const char (&__str)[_Len]) -> __mstring<_Len>;

template <size_t _Len>
__mstring(const char (&__str)[_Len], int) -> __mstring<__mpow2(_Len)>;

template <class _Ty>
constexpr auto __mnameof() noexcept
{
#if defined(_CCCL_COMPILER_MSVC)
  return __mstring{__FUNCSIG__, 0};
#else
  return __mstring{__PRETTY_FUNCTION__, 0};
#endif
}

// The following must be left undefined
template <class...>
struct _DIAGNOSTIC;

struct _WHERE;

struct _IN_ALGORITHM;

struct _WHAT;

struct _WITH_FUNCTION;

struct _WITH_SENDER;

struct _WITH_ARGUMENTS;

struct _WITH_QUERY;

struct _WITH_ENVIRONMENT;

template <class>
struct _WITH_COMPLETION_SIGNATURE;

struct _FUNCTION_IS_NOT_CALLABLE;

struct _UNKNOWN;

struct _SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS;

template <class... _Sigs>
struct _WITH_COMPLETIONS
{};

struct __merror_base
{
  constexpr friend bool __ustdex_unhandled_error(void*) noexcept
  {
    return true;
  }
};

template <class... _What>
struct _ERROR : __merror_base
{
  template <class...>
  using __f = _ERROR;

  _ERROR operator+();

  template <class _Ty>
  _ERROR& operator,(_Ty&);

  template <class... _With>
  _ERROR<_What..., _With...>& with(_ERROR<_With...>&);
};

constexpr bool __ustdex_unhandled_error(...) noexcept
{
  return false;
}

template <class _Ty>
_CCCL_INLINE_VAR constexpr bool __is_error = false;

template <class... _What>
_CCCL_INLINE_VAR constexpr bool __is_error<_ERROR<_What...>> = true;

template <class... _What>
_CCCL_INLINE_VAR constexpr bool __is_error<_ERROR<_What...>&> = true;

// True if any of the types in _Ts... are errors; false otherwise.
template <class... _Ts>
_CCCL_INLINE_VAR constexpr bool __contains_error =
#if defined(_CCCL_COMPILER_MSVC)
  (__is_error<_Ts> || ...);
#else
  __ustdex_unhandled_error(static_cast<__mlist<_Ts...>*>(nullptr));
#endif

template <class... _Ts>
using __find_error = decltype(+(__declval<_Ts&>(), ..., __declval<_ERROR<_UNKNOWN>&>()));

template <template <class...> class _Fn, class... _Ts>
using __minvoke_q = _Fn<_Ts...>;

template <class _Fn, class... _Ts>
using __minvoke = typename _Fn::template __f<_Ts...>;

template <class _Fn, class _Ty>
using __minvoke1 = typename _Fn::template __f<_Ty>;

template <class _Fn, template <class...> class _Cy, class... _Ts, class _Ret = __minvoke<_Fn, _Ts...>>
auto __apply_fn(_Cy<_Ts...>*) -> _Ret;

template <template <class...> class _Fn, template <class...> class _Cy, class... _Ts, class _Ret = _Fn<_Ts...>>
auto __apply_fn_q(_Cy<_Ts...>*) -> _Ret;

template <class _Fn, class _List>
using __mapply = decltype(__async::__apply_fn<_Fn>(static_cast<_List*>(nullptr)));

template <template <class...> class _Fn, class _List>
using __mapply_q = decltype(__async::__apply_fn_q<_Fn>(static_cast<_List*>(nullptr)));

template <class _Ty, class...>
using __mfront = _Ty;

template <template <class...> class _Fn, class _List, class _Enable = void>
_CCCL_INLINE_VAR constexpr bool __mvalid_ = false;

template <template <class...> class _Fn, class... _Ts>
_CCCL_INLINE_VAR constexpr bool __mvalid_<_Fn, __mlist<_Ts...>, __mvoid<_Fn<_Ts...>>> = true;

template <template <class...> class _Fn, class... _Ts>
_CCCL_INLINE_VAR constexpr bool __mvalid_q = __mvalid_<_Fn, __mlist<_Ts...>>;

template <class _Fn, class... _Ts>
_CCCL_INLINE_VAR constexpr bool __mvalid = __mvalid_<_Fn::template __f, __mlist<_Ts...>>;

template <class _Tp>
_CCCL_INLINE_VAR constexpr auto __v = _Tp::__value;

template <auto _Value>
_CCCL_INLINE_VAR constexpr auto __v<__mvalue<_Value>> = _Value;

template <bool _Value>
_CCCL_INLINE_VAR constexpr auto __v<__mbool<_Value>> = _Value;

template <class _Tp, _Tp _Value>
_CCCL_INLINE_VAR constexpr auto __v<_CUDA_VSTD::integral_constant<_Tp, _Value>> = _Value;

struct __midentity
{
  template <class _Ty>
  using __f = _Ty;
};

template <class _Ty>
struct __malways
{
  template <class...>
  using __f = _Ty;
};

template <class _Ty>
struct __malways1
{
  template <class>
  using __f = _Ty;
};

template <bool>
struct __mif_
{
  template <class _Then, class...>
  using __f = _Then;
};

template <>
struct __mif_<false>
{
  template <class _Then, class _Else>
  using __f = _Else;
};

template <bool If, class _Then = void, class... _Else>
using __mif = typename __mif_<If>::template __f<_Then, _Else...>;

template <class If, class _Then = void, class... _Else>
using __mif_t = typename __mif_<__v<If>>::template __f<_Then, _Else...>;

template <bool _Error>
struct __midentity_or_error_with_
{
  template <class _Ty, class... _With>
  using __f = _Ty;
};

template <>
struct __midentity_or_error_with_<true>
{
  template <class _Ty, class... _With>
  using __f = decltype(__declval<_Ty&>().with(__declval<_ERROR<_With...>&>()));
};

template <class _Ty, class... _With>
using __midentity_or_error_with = __minvoke<__midentity_or_error_with_<__is_error<_Ty>>, _Ty, _With...>;

template <bool>
struct __mtry_;

template <>
struct __mtry_<false>
{
  template <template <class...> class _Fn, class... _Ts>
  using __g = _Fn<_Ts...>;

  template <class _Fn, class... _Ts>
  using __f = typename _Fn::template __f<_Ts...>;
};

template <>
struct __mtry_<true>
{
  template <template <class...> class _Fn, class... _Ts>
  using __g = __find_error<_Ts...>;

  template <class _Fn, class... _Ts>
  using __f = __find_error<_Fn, _Ts...>;
};

template <class _Fn, class... _Ts>
using __mtry_invoke = typename __mtry_<__contains_error<_Ts...>>::template __f<_Fn, _Ts...>;

template <template <class...> class _Fn, class... _Ts>
using __mtry_invoke_q = typename __mtry_<__contains_error<_Ts...>>::template __g<_Fn, _Ts...>;

template <class _Fn>
struct __mtry
{
  template <class... _Ts>
  using __f = __mtry_invoke<_Fn, _Ts...>;
};

template <class _Fn>
struct __mpoly
{
  template <class... _Ts>
  using __f = typename __mtry_<(sizeof...(_Ts) == ~0ul)>::template __f<_Fn, _Ts...>;
};

template <template <class...> class _Fn>
struct __mpoly_q
{
  template <class... _Ts>
  using __f = typename __mtry_<(sizeof...(_Ts) == ~0ul)>::template __g<_Fn, _Ts...>;
};

template <template <class...> class _Fn, class... _Default>
struct __mquote;

template <template <class...> class _Fn>
struct __mquote<_Fn>
{
  template <class... _Ts>
  using __f = _Fn<_Ts...>;
};

template <template <class...> class _Fn, class _Default>
struct __mquote<_Fn, _Default>
{
  template <class... _Ts>
  using __f = typename __mif<__mvalid_q<_Fn, _Ts...>, __mquote<_Fn>, __malways<_Default>>::template __f<_Ts...>;
};

template <template <class...> class _Fn, class... _Default>
struct __mtry_quote;

template <template <class...> class _Fn>
struct __mtry_quote<_Fn>
{
  template <class... _Ts>
  using __f = typename __mtry_<__contains_error<_Ts...>>::template __g<_Fn, _Ts...>;
};

template <template <class...> class _Fn, class _Default>
struct __mtry_quote<_Fn, _Default>
{
  template <class... _Ts>
  using __f = typename __mif<__mvalid_q<_Fn, _Ts...>, __mtry_quote<_Fn>, __malways<_Default>>::template __f<_Ts...>;
};

template <class _Fn, class... _Ts>
struct __mbind_front
{
  template <class... _Us>
  using __f = __minvoke<_Fn, _Ts..., _Us...>;
};

template <class _Fn, class _Ty>
struct __mbind_front1
{
  template <class... _Us>
  using __f = __minvoke<_Fn, _Ty, _Us...>;
};

template <template <class...> class _Fn, class... _Ts>
struct __mbind_front_q
{
  template <class... _Us>
  using __f = __minvoke_q<_Fn, _Ts..., _Us...>;
};

template <class _Fn, class... _Ts>
struct __mbind_back
{
  template <class... _Us>
  using __f = __minvoke<_Fn, _Us..., _Ts...>;
};

template <template <class...> class _Fn, class... _Ts>
struct __mbind_back_q
{
  template <class... _Us>
  using __f = __minvoke_q<_Fn, _Us..., _Ts...>;
};

#if defined(__cpp_pack_indexing)

template <class _Np, class... _Ts>
using __m_at = _Ts...[__v<_Np>];

template <size_t _Np, class... _Ts>
using __m_at_c = _Ts...[_Np];

#elif defined(_CCCL_BUILTIN_TYPE_PACK_ELEMENT)

template <bool>
struct __m_at_
{
  template <class _Np, class... _Ts>
  using __f = _CCCL_BUILTIN_TYPE_PACK_ELEMENT(__v<_Np>, _Ts...);
};

template <class _Np, class... _Ts>
using __m_at = __minvoke<__m_at_<__v<_Np> == ~0ul>, _Np, _Ts...>;

template <size_t _Np, class... _Ts>
using __m_at_c = __minvoke<__m_at_<_Np == ~0ul>, __mvalue<_Np>, _Ts...>;

template <size_t _Idx>
struct __mget
{
  template <class... _Ts>
  using __f = __m_at<__mvalue<_Idx>, _Ts...>;
};

#else

template <size_t _Idx>
struct __mget
{
  template <class, class, class, class, class... _Ts>
  using __f = __minvoke<__mtry_<sizeof...(_Ts) == ~0ull>, __mget<_Idx - 4>, _Ts...>;
};

template <>
struct __mget<0>
{
  template <class _Ty, class...>
  using __f = _Ty;
};

template <>
struct __mget<1>
{
  template <class, class _Ty, class...>
  using __f = _Ty;
};

template <>
struct __mget<2>
{
  template <class, class, class _Ty, class...>
  using __f = _Ty;
};

template <>
struct __mget<3>
{
  template <class, class, class, class _Ty, class...>
  using __f = _Ty;
};

template <class _Np, class... _Ts>
using __m_at = __minvoke<__mget<__v<_Np>>, _Ts...>;

template <size_t _Np, class... _Ts>
using __m_at_c = __minvoke<__mget<_Np>, _Ts...>;

#endif

template <class _First, class _Second>
struct __mpair
{
  using first  = _First;
  using second = _Second;
};

template <class _Pair>
using __mfirst = typename _Pair::first;

template <class _Pair>
using __msecond = typename _Pair::second;

template <template <class...> class _Second, template <class...> class _First>
struct __mcompose_q
{
  template <class... _Ts>
  using __f = _Second<_First<_Ts...>>;
};

struct __mcount
{
  template <class... _Ts>
  using __f = __mvalue<sizeof...(_Ts)>;
};

template <bool>
struct __mconcat_
{
  template <class... _Ts,
            template <class...> class _Ap = __mlist,
            class... _As,
            template <class...> class _Bp = __mlist,
            class... _Bs,
            template <class...> class _Cp = __mlist,
            class... _Cs,
            template <class...> class _Dp = __mlist,
            class... _Ds,
            class... _Tail>
  static auto
  __f(__mlist<_Ts...>*,
      _Ap<_As...>*,
      _Bp<_Bs...>* = nullptr,
      _Cp<_Cs...>* = nullptr,
      _Dp<_Ds...>* = nullptr,
      _Tail*... __tail)
    -> decltype(__mconcat_<(sizeof...(_Tail) == 0)>::__f(
      static_cast<__mlist<_Ts..., _As..., _Bs..., _Cs..., _Ds...>*>(nullptr), __tail...));
};

template <>
struct __mconcat_<true>
{
  template <class... _As>
  static auto __f(__mlist<_As...>*) -> __mlist<_As...>;
};

template <class _Continuation = __mquote<__mlist>>
struct __mconcat_into
{
  template <class... _Args>
  using __f =
    __mapply<_Continuation, decltype(__mconcat_<(sizeof...(_Args) == 0)>::__f({}, static_cast<_Args*>(nullptr)...))>;
};

template <template <class...> class _Continuation = __mlist>
struct __mconcat_into_q
{
  template <class... _Args>
  using __f =
    __mapply_q<_Continuation, decltype(__mconcat_<(sizeof...(_Args) == 0)>::__f({}, static_cast<_Args*>(nullptr)...))>;
};

// The following must be super-fast to compile, so use an intrinsic directly if it is available
template <class _Set, class... _Ty>
_CCCL_INLINE_VAR constexpr bool __mset_contains = (_CUDA_VSTD::is_base_of_v<__mtype<_Ty>, _Set> && ...);

namespace __set
{
template <class... _Ts>
struct __inherit
{};

template <class _Ty, class... _Ts>
struct __inherit<_Ty, _Ts...>
    : __mtype<_Ty>
    , __inherit<_Ts...>
{};

template <class... _Set>
auto operator+(__inherit<_Set...>&) -> __inherit<_Set...>;

template <class... _Set, class _Ty>
auto operator%(__inherit<_Set...>&, __mtype<_Ty>&) //
  -> __mif< //
    __mset_contains<__inherit<_Set...>, _Ty>,
    __inherit<_Set...>,
    __inherit<_Ty, _Set...>>&;

template <class _ExpectedSet>
struct __eq
{
  static constexpr size_t __count = __v<__mapply<__mcount, _ExpectedSet>>;

  template <class... _Ts>
  using __f = __mbool<sizeof...(_Ts) == __count && __mset_contains<_ExpectedSet, _Ts...>>;
};
} // namespace __set

template <class... _Ts>
using __mset = __set::__inherit<_Ts...>;

template <class _Set, class... _Ts>
using __mset_insert = decltype(+(__declval<_Set&>() % ... % __declval<__mtype<_Ts>&>()));

template <class... _Ts>
using __mmake_set = __mset_insert<__mset<>, _Ts...>;

template <class _Set1, class _Set2>
_CCCL_INLINE_VAR constexpr bool __mset_eq = __v<__mapply<__set::__eq<_Set1>, _Set2>>;

template <class _Fn>
struct __munique
{
  template <class... _Ts>
  using __f = __minvoke<__mmake_set<_Ts...>, _Fn>;
};

template <class _Ty>
struct __msingle_or
{
  template <class _Uy = _Ty>
  using __f = _Uy;
};
} // namespace cuda::experimental::__async

_CCCL_DIAG_POP

#include <cuda/experimental/__async/epilogue.cuh>

#endif
