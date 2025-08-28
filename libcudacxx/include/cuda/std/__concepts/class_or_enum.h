//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___CONCEPTS_CLASS_OR_ENUM_H
#define _CUDA_STD___CONCEPTS_CLASS_OR_ENUM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_union.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp>
_CCCL_CONCEPT __class_or_enum = is_class_v<_Tp> || is_union_v<_Tp> || is_enum_v<_Tp>;

// Work around Clang bug https://llvm.org/PR52970
// TODO: remove this workaround once libc++ no longer has to support Clang 13 (it was fixed in Clang 14).
template <class _Tp>
_CCCL_CONCEPT __workaround_52970 = is_class_v<remove_cvref_t<_Tp>> || is_union_v<remove_cvref_t<_Tp>>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___CONCEPTS_CLASS_OR_ENUM_H
