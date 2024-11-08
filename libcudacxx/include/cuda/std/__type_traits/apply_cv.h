//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H
#define _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

struct __apply_cv_
{
  template <class _Tp>
  using __call = _Tp;
};

struct __apply_cv_c
{
  template <class _Tp>
  using __call = const _Tp;
};

struct __apply_cv_v
{
  template <class _Tp>
  using __call = volatile _Tp;
};

struct __apply_cv_cv
{
  template <class _Tp>
  using __call = const volatile _Tp;
};

struct __apply_cv_lr
{
  template <class _Tp>
  using __call = _Tp&;
};

struct __apply_cv_clr
{
  template <class _Tp>
  using __call = const _Tp&;
};

struct __apply_cv_vlr
{
  template <class _Tp>
  using __call = volatile _Tp&;
};

struct __apply_cv_cvlr
{
  template <class _Tp>
  using __call = const volatile _Tp&;
};

struct __apply_cv_rr
{
  template <class _Tp>
  using __call = _Tp&&;
};

struct __apply_cv_crr
{
  template <class _Tp>
  using __call = const _Tp&&;
};

struct __apply_cv_vrr
{
  template <class _Tp>
  using __call = volatile _Tp&&;
};

struct __apply_cv_cvrr
{
  template <class _Tp>
  using __call = const volatile _Tp&&;
};

#ifndef _CCCL_NO_VARIABLE_TEMPLATES
template <class>
extern __apply_cv_ __apply_cvr;
template <class _Tp>
extern __apply_cv_c __apply_cvr<const _Tp>;
template <class _Tp>
extern __apply_cv_v __apply_cvr<volatile _Tp>;
template <class _Tp>
extern __apply_cv_cv __apply_cvr<const volatile _Tp>;
template <class _Tp>
extern __apply_cv_lr __apply_cvr<_Tp&>;
template <class _Tp>
extern __apply_cv_clr __apply_cvr<const _Tp&>;
template <class _Tp>
extern __apply_cv_vlr __apply_cvr<volatile _Tp&>;
template <class _Tp>
extern __apply_cv_cvlr __apply_cvr<const volatile _Tp&>;
template <class _Tp>
extern __apply_cv_rr __apply_cvr<_Tp&&>;
template <class _Tp>
extern __apply_cv_crr __apply_cvr<const _Tp&&>;
template <class _Tp>
extern __apply_cv_vrr __apply_cvr<volatile _Tp&&>;
template <class _Tp>
extern __apply_cv_cvrr __apply_cvr<const volatile _Tp&&>;

template <class _Tp>
using __apply_cv_fn = decltype(__apply_cpcvr<_Tp>);
#else // ^^^ !_CCCL_NO_VARIABLE_TEMPLATES / _CCCL_NO_VARIABLE_TEMPLATES vvv
template <class>
struct __apply_cvr
{
  using type = __apply_cv_;
};
template <class _Tp>
struct __apply_cvr<const _Tp>
{
  using type = __apply_cv_c;
};
template <class _Tp>
struct __apply_cvr<volatile _Tp>
{
  using type = __apply_cv_v;
};
template <class _Tp>
struct __apply_cvr<const volatile _Tp>
{
  using type = __apply_cv_cv;
};
template <class _Tp>
struct __apply_cvr<_Tp&>
{
  using type = __apply_cv_lr;
};
template <class _Tp>
struct __apply_cvr<const _Tp&>
{
  using type = __apply_cv_clr;
};
template <class _Tp>
struct __apply_cvr<volatile _Tp&>
{
  using type = __apply_cv_vlr;
};
template <class _Tp>
struct __apply_cvr<const volatile _Tp&>
{
  using type = __apply_cv_cvlr;
};
template <class _Tp>
struct __apply_cvr<_Tp&&>
{
  using type = __apply_cv_rr;
};
template <class _Tp>
struct __apply_cvr<const _Tp&&>
{
  using type = __apply_cv_crr;
};
template <class _Tp>
struct __apply_cvr<volatile _Tp&&>
{
  using type = __apply_cv_vrr;
};
template <class _Tp>
struct __apply_cvr<const volatile _Tp&&>
{
  using type = __apply_cv_cvrr;
};

template <class _Tp>
using __apply_cv_fn = typename __apply_cvr<_Tp>::type;
#endif // _CCCL_NO_VARIABLE_TEMPLATES

template <class _From, class _To>
using __apply_cv_t = typename __apply_cv_fn<_From>::template __call<_To>;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_APPLY_CV_H
