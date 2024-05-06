//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___RANDOM_IS_SEED_SEQUENCE_H
#define _LIBCUDACXX___RANDOM_IS_SEED_SEQUENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _Sseq, class _Engine>
struct __is_seed_sequence
{
  static constexpr const bool value = !_CCCL_TRAIT(is_convertible, _Sseq, typename _Engine::result_type)
                                   && !_CCCL_TRAIT(is_same, remove_cv_t<_Sseq>, _Engine);
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___RANDOM_IS_SEED_SEQUENCE_H
