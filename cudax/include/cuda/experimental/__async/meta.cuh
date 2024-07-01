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

#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_base_of.h>
#include <cuda/std/__utility/integer_sequence.h>

#include "config.cuh"
#if __cpp_lib_three_way_comparison
#  include <compare>
#endif

// This must be the last #include
#include "prologue.cuh"

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-string-literal-operator-template")
_CCCL_DIAG_SUPPRESS_GCC("-Wnon-template-friend")

namespace cuda::experimental::__async
{
template <class Ret, class... Args>
using _fn_t = Ret(Args...);

template <class Ret, class... Args>
using _nothrow_fn_t = Ret(Args...) noexcept;

template <class _Ty, class _Ret = _Ty&&>
extern _nothrow_fn_t<_Ret>* _declval;

#define DECLVAL(...) _declval<__VA_ARGS__>()

template <class...>
using _mvoid = void;

template <class Ty>
struct _mtype
{
  using type = Ty;
};

template <class Ty>
using _t = typename Ty::type;

template <class... Ts>
struct _mlist;

template <auto Val>
struct _mvalue
{
  static constexpr auto value = Val;
};

// A separate _mbool template is needed in addition to _mvalue
// because of an EDG bug in the handling of auto template parameters.
template <bool Val>
struct _mbool
{
  static constexpr auto value = Val;
};

using _mtrue  = _mbool<true>;
using _mfalse = _mbool<false>;

template <auto... Vals>
struct _mvalues;

template <size_t... Vals>
struct _moffsets;

template <class... Bools>
using _mand = _mbool<(Bools::value && ...)>;

template <class... Bools>
using _mor = _mbool<(Bools::value || ...)>;

template <size_t... Idx>
using _mindices = _CUDA_VSTD::index_sequence<Idx...>*;

template <size_t Count>
using _mmake_indices = _CUDA_VSTD::make_index_sequence<Count>*;

template <class... Ts>
using _mmake_indices_for = _CUDA_VSTD::make_index_sequence<sizeof...(Ts)>*;

constexpr size_t _mpow2(size_t _size) noexcept
{
  --_size;
  _size |= _size >> 1;
  _size |= _size >> 2;
  _size |= _size >> 4;
  _size |= _size >> 8;
  if constexpr (sizeof(_size) >= 4)
  {
    _size |= _size >> 16;
  }
  if constexpr (sizeof(_size) >= 8)
  {
    _size |= _size >> 32;
  }
  return ++_size;
}

template <class Ty>
constexpr Ty _mmin(Ty _lhs, Ty _rhs) noexcept
{
  return _lhs < _rhs ? _lhs : _rhs;
}

template <class Ty>
constexpr int _mcompare(Ty _lhs, Ty _rhs) noexcept
{
  return _lhs < _rhs ? -1 : _lhs > _rhs ? 1 : 0;
}

template <size_t Len>
struct _mstring
{
  template <size_t Ny, size_t... Is>
  constexpr _mstring(const char (&_str)[Ny], _mindices<Is...>) noexcept
      : len_{Ny}
      , what_{(Is < Ny ? _str[Is] : '\0')...}
  {}

  template <size_t Ny>
  constexpr _mstring(const char (&_str)[Ny], int = 0) noexcept
      : _mstring{_str, _mmake_indices<Len>{}}
  {}

  constexpr auto length() const noexcept -> size_t
  {
    return len_;
  }

  template <size_t OtherLen>
  constexpr int compare(const _mstring<OtherLen>& other) const noexcept
  {
    size_t const len = _mmin(len_, other.len_);
    for (size_t i = 0; i < len; ++i)
    {
      if (auto const cmp = _mcompare(what_[i], other.what_[i]))
      {
        return cmp;
      }
    }
    return _mcompare(len_, other.len_);
  }

  template <size_t OtherLen>
  constexpr auto operator==(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return len_ == other.len_ && compare(other) == 0;
  }

  template <size_t OtherLen>
  constexpr auto operator!=(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return !operator==(other);
  }

  template <size_t OtherLen>
  constexpr auto operator<(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return compare(other) < 0;
  }

  template <size_t OtherLen>
  constexpr auto operator>(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return compare(other) > 0;
  }

  template <size_t OtherLen>
  constexpr auto operator<=(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return compare(other) <= 0;
  }

  template <size_t OtherLen>
  constexpr auto operator>=(const _mstring<OtherLen>& other) const noexcept -> bool
  {
    return compare(other) >= 0;
  }

  size_t len_;
  char what_[Len];
};

template <size_t Len>
_mstring(const char (&_str)[Len]) -> _mstring<Len>;

template <size_t Len>
_mstring(const char (&_str)[Len], int) -> _mstring<_mpow2(Len)>;

template <class T>
constexpr auto _mnameof() noexcept
{
#if defined(_CCCL_COMPILER_MSVC)
  return _mstring{__FUNCSIG__, 0};
#else
  return _mstring{__PRETTY_FUNCTION__, 0};
#endif
}

// The following must be left undefined
template <class...>
struct DIAGNOSTIC;

struct WHERE;

struct IN_ALGORITHM;

struct WHAT;

struct WITH_FUNCTION;

struct WITH_SENDER;

struct WITH_ARGUMENTS;

struct WITH_QUERY;

struct WITH_ENVIRONMENT;

template <class>
struct WITH_COMPLETION_SIGNATURE;

struct FUNCTION_IS_NOT_CALLABLE;

struct UNKNOWN;

struct SENDER_HAS_TOO_MANY_SUCCESS_COMPLETIONS;

template <class... Sigs>
struct WITH_COMPLETIONS
{};

struct _merror_base
{
  constexpr friend bool _ustdex_unhandled_error(void*) noexcept
  {
    return true;
  }
};

template <class... What>
struct ERROR : _merror_base
{
  template <class...>
  using _f = ERROR;

  ERROR operator+();

  template <class Ty>
  ERROR& operator,(Ty&);

  template <class... With>
  ERROR<What..., With...>& with(ERROR<With...>&);
};

constexpr bool _ustdex_unhandled_error(...) noexcept
{
  return false;
}

template <class Ty>
_CCCL_INLINE_VAR constexpr bool _is_error = false;

template <class... What>
_CCCL_INLINE_VAR constexpr bool _is_error<ERROR<What...>> = true;

template <class... What>
_CCCL_INLINE_VAR constexpr bool _is_error<ERROR<What...>&> = true;

// True if any of the types in Ts... are errors; false otherwise.
template <class... Ts>
_CCCL_INLINE_VAR constexpr bool _contains_error =
#if defined(_CCCL_COMPILER_MSVC)
  (_is_error<Ts> || ...);
#else
  _ustdex_unhandled_error(static_cast<_mlist<Ts...>*>(nullptr));
#endif

template <class... Ts>
using _find_error = decltype(+(DECLVAL(Ts&), ..., DECLVAL(ERROR<UNKNOWN>&)));

template <template <class...> class Fn, class... Ts>
using _minvoke_q = Fn<Ts...>;

template <class Fn, class... Ts>
using _minvoke = typename Fn::template _f<Ts...>;

template <class Fn, class Ty>
using _minvoke1 = typename Fn::template _f<Ty>;

template <class Fn, template <class...> class C, class... Ts>
_minvoke<Fn, Ts...> _apply_fn(C<Ts...>*);

template <template <class...> class Fn, template <class...> class C, class... Ts>
Fn<Ts...> _apply_fn_q(C<Ts...>*);

template <class Fn, class List>
using _mapply = decltype(__async::_apply_fn<Fn>(static_cast<List*>(nullptr)));

template <template <class...> class Fn, class List>
using _mapply_q = decltype(__async::_apply_fn_q<Fn>(static_cast<List*>(nullptr)));

template <class Ty, class...>
using _mfront = Ty;

template <template <class...> class Fn, class List, class Enable = void>
_CCCL_INLINE_VAR constexpr bool _mvalid_ = false;

template <template <class...> class Fn, class... Ts>
_CCCL_INLINE_VAR constexpr bool _mvalid_<Fn, _mlist<Ts...>, _mvoid<Fn<Ts...>>> = true;

template <template <class...> class Fn, class... Ts>
_CCCL_INLINE_VAR constexpr bool _mvalid_q = _mvalid_<Fn, _mlist<Ts...>>;

template <class Fn, class... Ts>
_CCCL_INLINE_VAR constexpr bool _mvalid = _mvalid_<Fn::template _f, _mlist<Ts...>>;

template <class Tp>
_CCCL_INLINE_VAR constexpr auto _v = Tp::value;

template <auto Value>
_CCCL_INLINE_VAR constexpr auto _v<_mvalue<Value>> = Value;

template <bool Value>
_CCCL_INLINE_VAR constexpr auto _v<_mbool<Value>> = Value;

template <class Tp, Tp Value>
_CCCL_INLINE_VAR constexpr auto _v<_CUDA_VSTD::integral_constant<Tp, Value>> = Value;

struct _midentity
{
  template <class Ty>
  using _f = Ty;
};

template <class Ty>
struct _malways
{
  template <class...>
  using _f = Ty;
};

template <class Ty>
struct _malways1
{
  template <class>
  using _f = Ty;
};

template <bool>
struct _mif_
{
  template <class Then, class...>
  using _f = Then;
};

template <>
struct _mif_<false>
{
  template <class Then, class Else>
  using _f = Else;
};

template <bool If, class Then = void, class... Else>
using _mif = typename _mif_<If>::template _f<Then, Else...>;

template <class If, class Then = void, class... Else>
using _mif_t = typename _mif_<_v<If>>::template _f<Then, Else...>;

template <bool Error>
struct _midentity_or_error_with_
{
  template <class Ty, class... With>
  using _f = Ty;
};

template <>
struct _midentity_or_error_with_<true>
{
  template <class Ty, class... With>
  using _f = decltype(DECLVAL(Ty&).with(DECLVAL(ERROR<With...>&)));
};

template <class Ty, class... With>
using _midentity_or_error_with = _minvoke<_midentity_or_error_with_<_is_error<Ty>>, Ty, With...>;

template <bool>
struct _mtry_;

template <>
struct _mtry_<false>
{
  template <template <class...> class Fn, class... Ts>
  using _g = Fn<Ts...>;

  template <class Fn, class... Ts>
  using _f = typename Fn::template _f<Ts...>;
};

template <>
struct _mtry_<true>
{
  template <template <class...> class Fn, class... Ts>
  using _g = _find_error<Ts...>;

  template <class Fn, class... Ts>
  using _f = _find_error<Fn, Ts...>;
};

template <class Fn, class... Ts>
using _mtry_invoke = typename _mtry_<_contains_error<Ts...>>::template _f<Fn, Ts...>;

template <template <class...> class Fn, class... Ts>
using _mtry_invoke_q = typename _mtry_<_contains_error<Ts...>>::template _g<Fn, Ts...>;

template <class Fn>
struct _mtry
{
  template <class... Ts>
  using _f = _mtry_invoke<Fn, Ts...>;
};

template <class Fn>
struct _mpoly
{
  template <class... Ts>
  using _f = typename _mtry_<(sizeof...(Ts) == ~0ul)>::template _f<Fn, Ts...>;
};

template <template <class...> class Fn>
struct _mpoly_q
{
  template <class... Ts>
  using _f = typename _mtry_<(sizeof...(Ts) == ~0ul)>::template _g<Fn, Ts...>;
};

template <template <class...> class Fn, class... Default>
struct _mquote;

template <template <class...> class Fn>
struct _mquote<Fn>
{
  template <class... Ts>
  using _f = Fn<Ts...>;
};

template <template <class...> class Fn, class Default>
struct _mquote<Fn, Default>
{
  template <class... Ts>
  using _f = typename _mif<_mvalid_q<Fn, Ts...>, _mquote<Fn>, _malways<Default>>::template _f<Ts...>;
};

template <template <class...> class Fn, class... Default>
struct _mtry_quote;

template <template <class...> class Fn>
struct _mtry_quote<Fn>
{
  template <class... Ts>
  using _f = typename _mtry_<_contains_error<Ts...>>::template _g<Fn, Ts...>;
};

template <template <class...> class Fn, class Default>
struct _mtry_quote<Fn, Default>
{
  template <class... Ts>
  using _f = typename _mif<_mvalid_q<Fn, Ts...>, _mtry_quote<Fn>, _malways<Default>>::template _f<Ts...>;
};

template <class Fn, class... Ts>
struct _mbind_front
{
  template <class... Us>
  using _f = _minvoke<Fn, Ts..., Us...>;
};

template <class Fn, class Ty>
struct _mbind_front1
{
  template <class... Us>
  using _f = _minvoke<Fn, Ty, Us...>;
};

template <template <class...> class Fn, class... Ts>
struct _mbind_front_q
{
  template <class... Us>
  using _f = _minvoke_q<Fn, Ts..., Us...>;
};

template <class Fn, class... Ts>
struct _mbind_back
{
  template <class... Us>
  using _f = _minvoke<Fn, Us..., Ts...>;
};

template <template <class...> class Fn, class... Ts>
struct _mbind_back_q
{
  template <class... Us>
  using _f = _minvoke_q<Fn, Us..., Ts...>;
};

#if defined(__cpp_pack_indexing)

template <class _Np, class... _Ts>
using _m_at = _Ts...[_mvalue<_Np>];

template <size_t _Np, class... _Ts>
using _m_at_c = _Ts...[_Np];

#elif __has_builtin(__type_pack_element)

template <bool>
struct _m_at_
{
  template <class Np, class... Ts>
  using _f = __type_pack_element<_v<Np>, Ts...>;
};

template <class Np, class... Ts>
using _m_at = _minvoke<_m_at_<_v<Np> == ~0ul>, Np, Ts...>;

template <size_t Np, class... Ts>
using _m_at_c = _minvoke<_m_at_<Np == ~0ul>, _mvalue<Np>, Ts...>;

template <size_t Idx>
struct _mget
{
  template <class... Ts>
  using _f = _m_at<_mvalue<Idx>, Ts...>;
};

#else

template <size_t Idx>
struct _mget
{
  template <class, class, class, class, class... Ts>
  using _f = _minvoke<_mtry_<sizeof...(Ts) == ~0ull>, _mget<Idx - 4>, Ts...>;
};

template <>
struct _mget<0>
{
  template <class T, class...>
  using _f = T;
};

template <>
struct _mget<1>
{
  template <class, class T, class...>
  using _f = T;
};

template <>
struct _mget<2>
{
  template <class, class, class T, class...>
  using _f = T;
};

template <>
struct _mget<3>
{
  template <class, class, class, class T, class...>
  using _f = T;
};

template <class Np, class... Ts>
using _m_at = _minvoke<_mget<_v<Np>>, Ts...>;

template <size_t Np, class... Ts>
using _m_at_c = _minvoke<_mget<Np>, Ts...>;

#endif

template <class First, class Second>
struct _mpair
{
  using first  = First;
  using second = Second;
};

template <class Pair>
using _mfirst = typename Pair::first;

template <class Pair>
using _msecond = typename Pair::second;

template <template <class...> class Second, template <class...> class First>
struct _mcompose_q
{
  template <class... Ts>
  using _f = Second<First<Ts...>>;
};

struct _mcount
{
  template <class... Ts>
  using _f = _mvalue<sizeof...(Ts)>;
};

template <bool>
struct _mconcat_
{
  template <class... Ts,
            template <class...> class Ap = _mlist,
            class... As,
            template <class...> class Bp = _mlist,
            class... Bs,
            template <class...> class Cp = _mlist,
            class... Cs,
            template <class...> class Dp = _mlist,
            class... Ds,
            class... Tail>
  static auto
  _f(_mlist<Ts...>*, Ap<As...>*, Bp<Bs...>* = nullptr, Cp<Cs...>* = nullptr, Dp<Ds...>* = nullptr, Tail*... _tail)
    -> decltype(_mconcat_<(sizeof...(Tail) == 0)>::_f(static_cast<_mlist<Ts..., As..., Bs..., Cs..., Ds...>*>(nullptr),
                                                      _tail...));
};

template <>
struct _mconcat_<true>
{
  template <class... As>
  static auto _f(_mlist<As...>*) -> _mlist<As...>;
};

template <class Continuation = _mquote<_mlist>>
struct _mconcat_into
{
  template <class... Args>
  using _f = _mapply<Continuation, decltype(_mconcat_<(sizeof...(Args) == 0)>::_f({}, static_cast<Args*>(nullptr)...))>;
};

template <template <class...> class Continuation = _mlist>
struct _mconcat_into_q
{
  template <class... Args>
  using _f =
    _mapply_q<Continuation, decltype(_mconcat_<(sizeof...(Args) == 0)>::_f({}, static_cast<Args*>(nullptr)...))>;
};

// The following must be super-fast to compile, so use an intrinsic directly if it is available
#if defined(_LIBCUDACXX_IS_BASE_OF) && !defined(_LIBCUDACXX_USE_IS_BASE_OF_FALLBACK)

template <class Set, class... Ty>
_CCCL_INLINE_VAR constexpr bool _mset_contains = (_LIBCUDACXX_IS_BASE_OF(_mtype<Ty>, Set) && ...);

#else

template <class Set, class... Ty>
_CCCL_INLINE_VAR constexpr bool _mset_contains = (_CUDA_VSTD::is_base_of_v<_mtype<Ty>, Set> && ...);

#endif

namespace _set
{
template <class... Ts>
struct _inherit
{};

template <class Ty, class... Ts>
struct _inherit<Ty, Ts...>
    : _mtype<Ty>
    , _inherit<Ts...>
{};

template <class... Set>
auto operator+(_inherit<Set...>&) -> _inherit<Set...>;

template <class... Set, class Ty>
auto operator%(_inherit<Set...>&, _mtype<Ty>&) //
  -> _mif< //
    _mset_contains<_inherit<Set...>, Ty>,
    _inherit<Set...>,
    _inherit<Ty, Set...>>&;

template <class ExpectedSet>
struct _eq
{
  static constexpr size_t count = _v<_mapply<_mcount, ExpectedSet>>;

  template <class... Ts>
  using _f = _mbool<sizeof...(Ts) == count && _mset_contains<ExpectedSet, Ts...>>;
};
} // namespace _set

template <class... Ts>
using _mset = _set::_inherit<Ts...>;

template <class Set, class... Ts>
using _mset_insert = decltype(+(DECLVAL(Set&) % ... % DECLVAL(_mtype<Ts>&)));

template <class... Ts>
using _mmake_set = _mset_insert<_mset<>, Ts...>;

template <class Set1, class Set2>
_CCCL_INLINE_VAR constexpr bool _mset_eq = _v<_mapply<_set::_eq<Set1>, Set2>>;

template <class Fn>
struct _munique
{
  template <class... Ts>
  using _f = _minvoke<_mmake_set<Ts...>, Fn>;
};

template <class Ty>
struct _msingle_or
{
  template <class Uy = Ty>
  using _f = Uy;
};
} // namespace cuda::experimental::__async

_CCCL_DIAG_POP

#include "epilogue.cuh"
