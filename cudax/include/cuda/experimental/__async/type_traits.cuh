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

#include <cuda/std/__type_traits/remove_reference.h>

#include "config.cuh"
#include "meta.cuh"

// This must be the last #include
#include "prologue.cuh"

namespace cuda::experimental::__async
{
#if __has_builtin(__remove_reference)

template <class Ty>
using _remove_reference_t = __remove_reference(Ty);

#elif __has_builtin(__remove_reference_t)

template <class Ty>
using _remove_reference_t = __remove_reference_t(Ty);

#else

template <class Ty>
using _remove_reference_t = _CUDA_VSTD::remove_reference_t<Ty>;

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
// __decay_t: An efficient implementation for ::std::decay
#if __has_builtin(__decay)

template <class Ty>
using _decay_t = __decay(Ty);

// #elif defined(_CCCL_COMPILER_NVHPC)

//   template <class Ty>
//   using _decay_t = _CUDA_VSTD::decay_t<Ty>;

#else

struct _decay_object
{
  template <class Ty>
  static Ty _g(Ty const&);
  template <class Ty>
  using _f = decltype(_g(DECLVAL(Ty)));
};

struct _decay_default
{
  template <class Ty>
  static Ty _g(Ty);
  template <class Ty>
  using _f = decltype(_g(DECLVAL(Ty)));
};

// I don't care to support abominable function types,
// but if that's needed, this is the way to do it:
// struct _decay_abominable {
//   template <class Ty>
//   using _f = Ty;
// };

struct _decay_void
{
  template <class Ty>
  using _f = void;
};

template <class Ty>
extern _decay_object _mdecay;

template <class Ty, class... Us>
extern _decay_default _mdecay<Ty(Us...)>;

template <class Ty, class... Us>
extern _decay_default _mdecay<Ty(Us...) noexcept>;

template <class Ty, class... Us>
extern _decay_default _mdecay<Ty (&)(Us...)>;

template <class Ty, class... Us>
extern _decay_default _mdecay<Ty (&)(Us...) noexcept>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const noexcept>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const &>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const & noexcept>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const &&>;

// template <class Ty, class... Us>
// extern _decay_abominable _mdecay<Ty(Us...) const && noexcept>;

template <class Ty>
extern _decay_default _mdecay<Ty[]>;

template <class Ty, size_t N>
extern _decay_default _mdecay<Ty[N]>;

template <class Ty, size_t N>
extern _decay_default _mdecay<Ty (&)[N]>;

template <>
inline _decay_void _mdecay<void>;

template <>
inline _decay_void _mdecay<void const>;

template <class Ty>
using _decay_t = typename decltype(_mdecay<Ty>)::template _f<Ty>;

#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
// _copy_cvref_t: For copying cvref from one type to another
struct _cp
{
  template <class Tp>
  using _f = Tp;
};

struct _cpc
{
  template <class Tp>
  using _f = const Tp;
};

struct _cplr
{
  template <class Tp>
  using _f = Tp&;
};

struct _cprr
{
  template <class Tp>
  using _f = Tp&&;
};

struct _cpclr
{
  template <class Tp>
  using _f = const Tp&;
};

struct _cpcrr
{
  template <class Tp>
  using _f = const Tp&&;
};

template <class>
extern _cp _cpcvr;
template <class Tp>
extern _cpc _cpcvr<const Tp>;
template <class Tp>
extern _cplr _cpcvr<Tp&>;
template <class Tp>
extern _cprr _cpcvr<Tp&&>;
template <class Tp>
extern _cpclr _cpcvr<const Tp&>;
template <class Tp>
extern _cpcrr _cpcvr<const Tp&&>;
template <class Tp>
using _copy_cvref_fn = decltype(_cpcvr<Tp>);

template <class From, class To>
using _copy_cvref_t = typename _copy_cvref_fn<From>::template _f<To>;

template <class Fn, class... As>
using _call_result_t = decltype(DECLVAL(Fn)(DECLVAL(As)...));

template <class Fn, class... As>
_CCCL_INLINE_VAR constexpr bool _callable = _mvalid_q<_call_result_t, Fn, As...>;

#if defined(__CUDA_ARCH__)
template <class Fn, class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_callable = true;

template <class Ty, class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_constructible = true;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_decay_copyable = true;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_movable = true;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_copyable = true;
#else
template <class Fn, class... As>
using _nothrow_callable_ = _mif<noexcept(DECLVAL(Fn)(DECLVAL(As)...))>;

template <class Fn, class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_callable = _mvalid_q<_nothrow_callable_, Fn, As...>;

template <class Ty, class... As>
using _nothrow_constructible_ = _mif<noexcept(Ty{DECLVAL(As)...})>;

template <class Ty, class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_constructible = _mvalid_q<_nothrow_constructible_, Ty, As...>;

template <class Ty>
using _nothrow_decay_copyable_ = _mif<noexcept(_decay_t<Ty>(DECLVAL(Ty)))>;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_decay_copyable = (_mvalid_q<_nothrow_decay_copyable_, As> && ...);

template <class Ty>
using _nothrow_movable_ = _mif<noexcept(Ty(DECLVAL(Ty&&)))>;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_movable = (_mvalid_q<_nothrow_movable_, As> && ...);

template <class Ty>
using _nothrow_copyable_ = _mif<noexcept(Ty(DECLVAL(const Ty&)))>;

template <class... As>
_CCCL_INLINE_VAR constexpr bool _nothrow_copyable = (_mvalid_q<_nothrow_copyable_, As> && ...);
#endif
} // namespace cuda::experimental::__async

#include "epilogue.cuh"
