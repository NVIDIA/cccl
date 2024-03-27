//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H
#define _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H

#ifndef __cuda_std__
#include <__config>
#endif // __cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__type_traits/integral_constant.h"
#include "../__type_traits/remove_all_extents.h"
#include "../__type_traits/remove_cv.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011 && defined(_LIBCUDACXX_HAS_UNIQUE_OBJECT_REPRESENTATIONS)

template <class _Tp> struct _LIBCUDACXX_TEMPLATE_VIS has_unique_object_representations
    : public integral_constant<bool,
       __has_unique_object_representations(remove_cv_t<remove_all_extents_t<_Tp>>)> {};

#if !defined(_LIBCUDACXX_HAS_NO_VARIABLE_TEMPLATES)
template <class _Tp>
_LIBCUDACXX_INLINE_VAR constexpr bool has_unique_object_representations_v = has_unique_object_representations<_Tp>::value;
#endif

#endif

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_HAS_UNIQUE_OBJECT_REPRESENTATION_H
