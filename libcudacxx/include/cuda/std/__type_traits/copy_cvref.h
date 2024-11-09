//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
#define _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __apply_cvref_
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Tp;
};

struct __apply_cvref_c
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const _Tp;
};

struct __apply_cvref_v
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = volatile _Tp;
};

struct __apply_cvref_cv
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const volatile _Tp;
};

struct __apply_cvref_lr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Tp&;
};

struct __apply_cvref_clr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const _Tp&;
};

struct __apply_cvref_vlr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = volatile _Tp&;
};

struct __apply_cvref_cvlr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const volatile _Tp&;
};

struct __apply_cvref_rr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = _Tp&&;
};

struct __apply_cvref_crr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const _Tp&&;
};

struct __apply_cvref_vrr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = volatile _Tp&&;
};

struct __apply_cvref_cvrr
{
  template <class _Tp>
  using __call _LIBCUDACXX_NODEBUG_TYPE = const volatile _Tp&&;
};

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class>
extern __apply_cvref_ __apply_cvref;
template <class _Tp>
extern __apply_cvref_c __apply_cvref<const _Tp>;
template <class _Tp>
extern __apply_cvref_v __apply_cvref<volatile _Tp>;
template <class _Tp>
extern __apply_cvref_cv __apply_cvref<const volatile _Tp>;
template <class _Tp>
extern __apply_cvref_lr __apply_cvref<_Tp&>;
template <class _Tp>
extern __apply_cvref_clr __apply_cvref<const _Tp&>;
template <class _Tp>
extern __apply_cvref_vlr __apply_cvref<volatile _Tp&>;
template <class _Tp>
extern __apply_cvref_cvlr __apply_cvref<const volatile _Tp&>;
template <class _Tp>
extern __apply_cvref_rr __apply_cvref<_Tp&&>;
template <class _Tp>
extern __apply_cvref_crr __apply_cvref<const _Tp&&>;
template <class _Tp>
extern __apply_cvref_vrr __apply_cvref<volatile _Tp&&>;
template <class _Tp>
extern __apply_cvref_cvrr __apply_cvref<const volatile _Tp&&>;

template <class _Tp>
using __apply_cvref_fn = decltype(__apply_cvref<_Tp>);
#else // ^^^ !_CCCL_NO_VARIABLE_TEMPLATES / _CCCL_NO_VARIABLE_TEMPLATES vvv
template <class>
struct __apply_cvref
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_;
};
template <class _Tp>
struct __apply_cvref<const _Tp>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_c;
};
template <class _Tp>
struct __apply_cvref<volatile _Tp>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_v;
};
template <class _Tp>
struct __apply_cvref<const volatile _Tp>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_cv;
};
template <class _Tp>
struct __apply_cvref<_Tp&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_lr;
};
template <class _Tp>
struct __apply_cvref<const _Tp&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_clr;
};
template <class _Tp>
struct __apply_cvref<volatile _Tp&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_vlr;
};
template <class _Tp>
struct __apply_cvref<const volatile _Tp&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_cvlr;
};
template <class _Tp>
struct __apply_cvref<_Tp&&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_rr;
};
template <class _Tp>
struct __apply_cvref<const _Tp&&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_crr;
};
template <class _Tp>
struct __apply_cvref<volatile _Tp&&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_vrr;
};
template <class _Tp>
struct __apply_cvref<const volatile _Tp&&>
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __apply_cvref_cvrr;
};

template <class _Tp>
using __apply_cvref_fn _LIBCUDACXX_NODEBUG_TYPE = typename __apply_cvref<_Tp>::type;
#endif // _CCCL_NO_VARIABLE_TEMPLATES

template <class _From, class _To>
using __copy_cvref_t = typename __apply_cvref_fn<_From>::template __call<_To>;

template <class _From, class _To>
struct __copy_cvref
{
  using type _LIBCUDACXX_NODEBUG_TYPE = __copy_cvref_t<_From, _To>;
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_COPY_CVREF_H
