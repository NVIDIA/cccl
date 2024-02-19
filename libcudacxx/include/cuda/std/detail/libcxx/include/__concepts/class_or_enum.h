//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CONCEPTS_CLASS_OR_ENUM_H
#define _LIBCUDACXX___CONCEPTS_CLASS_OR_ENUM_H

#ifndef __cuda_std__
#include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__concepts/__concept_macros.h"
#include "../__type_traits/is_class.h"
#include "../__type_traits/is_enum.h"
#include "../__type_traits/is_union.h"
#include "../__type_traits/remove_cvref.h"

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

template<class _Tp>
_LIBCUDACXX_CONCEPT __class_or_enum = _LIBCUDACXX_TRAIT(is_class, _Tp) || _LIBCUDACXX_TRAIT(is_union, _Tp) || _LIBCUDACXX_TRAIT(is_enum, _Tp);

// Work around Clang bug https://llvm.org/PR52970
// TODO: remove this workaround once libc++ no longer has to support Clang 13 (it was fixed in Clang 14).
template<class _Tp>
_LIBCUDACXX_CONCEPT __workaround_52970 = _LIBCUDACXX_TRAIT(is_class, remove_cvref_t<_Tp>) || _LIBCUDACXX_TRAIT(is_union, remove_cvref_t<_Tp>);

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___CONCEPTS_CLASS_OR_ENUM_H
