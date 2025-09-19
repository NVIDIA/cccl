//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___FORMAT_FORMATTER_H
#define _CUDA_STD___FORMAT_FORMATTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

//! @brief A disabled formatter.
//!
//! This is used to disable formatting for types that are not supported.
struct __fmt_disabled_formatter
{
  __fmt_disabled_formatter()                                           = delete;
  __fmt_disabled_formatter(const __fmt_disabled_formatter&)            = delete;
  __fmt_disabled_formatter(__fmt_disabled_formatter&&)                 = delete;
  __fmt_disabled_formatter& operator=(const __fmt_disabled_formatter&) = delete;
  __fmt_disabled_formatter& operator=(__fmt_disabled_formatter&&)      = delete;
};

//! @brief The default formatter template.
//!
//! [format.formatter.spec]/5
//! If F is a disabled specialization of formatter, these values are false:
//! - is_default_constructible_v<F>,
//! - is_copy_constructible_v<F>,
//! - is_move_constructible_v<F>,
//! - is_copy_assignable_v<F>, and
//! - is_move_assignable_v<F>.
template <class _Tp, class _CharT = char>
struct _CCCL_TYPE_VISIBILITY_DEFAULT formatter : __fmt_disabled_formatter
{};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___FORMAT_FORMATTER_H
