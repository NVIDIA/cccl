//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_TYPE_TRAITS
#define __CUDAX_ASYNC_DETAIL_TYPE_TRAITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/remove_reference.h>

#include <cuda/experimental/__async/config.cuh>
#include <cuda/experimental/__async/meta.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{

template <class _Ty>
using __remove_ref_t = _CUDA_VSTD::__libcpp_remove_reference_t<_Ty>;

//////////////////////////////////////////////////////////////////////////////////////////////////
// __decay_t: An efficient implementation for ::std::decay
#if defined(_CCCL_BUILTIN_DECAY)

template <class _Ty>
using __decay_t = __decay(_Ty);

// #elif defined(_CCCL_COMPILER_NVHPC)

//   template <class _Ty>
//   using __decay_t = _CUDA_VSTD::decay_t<_Ty>;

#else // ^^^ _CCCL_BUILTIN_DECAY ^^^ / vvv !_CCCL_BUILTIN_DECAY vvv

struct __decay_object
{
  template <class _Ty>
  static _Ty __g(_Ty const&);
  template <class _Ty>
  using __f = decltype(__g(__declval<_Ty>()));
};

struct __decay_default
{
  template <class _Ty>
  static _Ty __g(_Ty);
  template <class _Ty>
  using __f = decltype(__g(__declval<_Ty>()));
};

// I don't care to support abominable function types,
// but if that's needed, this is the way to do it:
// struct __decay_abominable {
//   template <class _Ty>
//   using __f = _Ty;
// };

struct __decay_void
{
  template <class _Ty>
  using __f = void;
};

template <class _Ty>
extern __decay_object __mdecay;

template <class _Ty, class... _Us>
extern __decay_default __mdecay<_Ty(_Us...)>;

template <class _Ty, class... _Us>
extern __decay_default __mdecay<_Ty(_Us...) noexcept>;

template <class _Ty, class... _Us>
extern __decay_default __mdecay<_Ty (&)(_Us...)>;

template <class _Ty, class... _Us>
extern __decay_default __mdecay<_Ty (&)(_Us...) noexcept>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const noexcept>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const &>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const & noexcept>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const &&>;

// template <class _Ty, class... _Us>
// extern __decay_abominable __mdecay<_Ty(_Us...) const && noexcept>;

template <class _Ty>
extern __decay_default __mdecay<_Ty[]>;

template <class _Ty, size_t _Ny>
extern __decay_default __mdecay<_Ty[_Ny]>;

template <class _Ty, size_t _Ny>
extern __decay_default __mdecay<_Ty (&)[_Ny]>;

template <>
inline __decay_void __mdecay<void>;

template <>
inline __decay_void __mdecay<void const>;

template <class _Ty>
using __decay_t = typename decltype(__mdecay<_Ty>)::template __f<_Ty>;

#endif // _CCCL_BUILTIN_DECAY

//////////////////////////////////////////////////////////////////////////////////////////////////
// __copy_cvref_t: For copying cvref from one type to another
struct __cp
{
  template <class _Tp>
  using __f = _Tp;
};

struct __cpc
{
  template <class _Tp>
  using __f = const _Tp;
};

struct __cplr
{
  template <class _Tp>
  using __f = _Tp&;
};

struct __cprr
{
  template <class _Tp>
  using __f = _Tp&&;
};

struct __cpclr
{
  template <class _Tp>
  using __f = const _Tp&;
};

struct __cpcrr
{
  template <class _Tp>
  using __f = const _Tp&&;
};

template <class>
extern __cp __cpcvr;
template <class _Tp>
extern __cpc __cpcvr<const _Tp>;
template <class _Tp>
extern __cplr __cpcvr<_Tp&>;
template <class _Tp>
extern __cprr __cpcvr<_Tp&&>;
template <class _Tp>
extern __cpclr __cpcvr<const _Tp&>;
template <class _Tp>
extern __cpcrr __cpcvr<const _Tp&&>;
template <class _Tp>
using __copy_cvref_fn = decltype(__cpcvr<_Tp>);

template <class _From, class _To>
using __copy_cvref_t = typename __copy_cvref_fn<_From>::template __f<_To>;

template <class _Fn, class... _As>
using __call_result_t = decltype(__declval<_Fn>()(__declval<_As>()...));

template <class _Fn, class... _As>
_CCCL_INLINE_VAR constexpr bool __callable = __mvalid_q<__call_result_t, _Fn, _As...>;

#if defined(__CUDA_ARCH__)
template <class _Fn, class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_callable = true;

template <class _Ty, class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_constructible = true;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_decay_copyable = true;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_movable = true;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_copyable = true;
#else
template <class _Fn, class... _As>
using __nothrow_callable_ = __mif<noexcept(__declval<_Fn>()(__declval<_As>()...))>;

template <class _Fn, class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_callable = __mvalid_q<__nothrow_callable_, _Fn, _As...>;

template <class _Ty, class... _As>
using __nothrow_constructible_ = __mif<noexcept(_Ty{__declval<_As>()...})>;

template <class _Ty, class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_constructible = __mvalid_q<__nothrow_constructible_, _Ty, _As...>;

template <class _Ty>
using __nothrow_decay_copyable_ = __mif<noexcept(__decay_t<_Ty>(__declval<_Ty>()))>;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_decay_copyable = (__mvalid_q<__nothrow_decay_copyable_, _As> && ...);

template <class _Ty>
using __nothrow_movable_ = __mif<noexcept(_Ty(__declval<_Ty>()))>;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_movable = (__mvalid_q<__nothrow_movable_, _As> && ...);

template <class _Ty>
using __nothrow_copyable_ = __mif<noexcept(_Ty(__declval<const _Ty&>()))>;

template <class... _As>
_CCCL_INLINE_VAR constexpr bool __nothrow_copyable = (__mvalid_q<__nothrow_copyable_, _As> && ...);
#endif
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
