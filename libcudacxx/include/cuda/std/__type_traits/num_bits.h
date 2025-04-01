//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___TYPE_TRAITS_NUM_BITS
#define _LIBCUDACXX___TYPE_TRAITS_NUM_BITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/has_unique_object_representation.h>
#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/climits>
#include <cuda/std/complex>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_TEMPLATE(typename _Tp)
_CCCL_REQUIRES(_CCCL_TRAIT(has_unique_object_representations, _Tp) || _CCCL_TRAIT(__is_extended_arithmetic, _Tp)
               || _CCCL_TRAIT(__is_complex, _Tp))
inline constexpr int __num_bits_v = sizeof(_Tp) * CHAR_BIT;

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___TYPE_TRAITS_NUM_BITS
