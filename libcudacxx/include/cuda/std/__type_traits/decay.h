//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_DECAY_H
#define _LIBCUDACXX___TYPE_TRAITS_DECAY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_referenceable.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_extent.h>
#include <cuda/std/__type_traits/remove_reference.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if defined(_CCCL_BUILTIN_DECAY) && !defined(_LIBCUDACXX_USE_DECAY_FALLBACK)
template <class _Tp>
struct decay
{
  using type _LIBCUDACXX_NODEBUG_TYPE = _CCCL_BUILTIN_DECAY(_Tp);
};

#else

template <class _Up, bool>
struct __decay_impl
{
  typedef _LIBCUDACXX_NODEBUG_TYPE __remove_cv_t<_Up> type;
};

template <class _Up>
struct __decay_impl<_Up, true>
{
public:
  typedef _LIBCUDACXX_NODEBUG_TYPE
    __conditional_t<is_array<_Up>::value,
                    __remove_extent_t<_Up>*,
                    __conditional_t<is_function<_Up>::value, __add_pointer_t<_Up>, __remove_cv_t<_Up>>>
      type;
};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT decay
{
private:
  typedef _LIBCUDACXX_NODEBUG_TYPE __libcpp_remove_reference_t<_Tp> _Up;

public:
  typedef _LIBCUDACXX_NODEBUG_TYPE typename __decay_impl<_Up, __libcpp_is_referenceable<_Up>::value>::type type;
};
#endif // defined(_CCCL_BUILTIN_DECAY) && !defined(_LIBCUDACXX_USE_DECAY_FALLBACK)

template <class _Tp>
using __decay_t = typename decay<_Tp>::type;

#if _CCCL_STD_VER >= 2014
template <class _Tp>
using decay_t = typename decay<_Tp>::type;
#endif // _CCCL_STD_VER >= 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_DECAY_H
