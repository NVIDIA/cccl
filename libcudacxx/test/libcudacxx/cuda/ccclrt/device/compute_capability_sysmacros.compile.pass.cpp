//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

// <sys/sysmacros.h> defines `major` and `minor` as function-like macros, which would otherwise rename the deprecated
// cuda::compute_capability::major() and cuda::compute_capability::minor().
#if __has_include(<sys/sysmacros.h>)
#  include <sys/sysmacros.h>
#endif // __has_include(<sys/sysmacros.h>)

#include <cuda/devices>

#if __has_include(<sys/sysmacros.h>)
#  if !defined(major) || !defined(minor)
#    error "<cuda/devices> must restore the macros from <sys/sysmacros.h>"
#  endif // !defined(major) || !defined(minor)
#endif // __has_include(<sys/sysmacros.h>)

// The parentheses prevent the expansion of the function-like macros.
static_assert((cuda::compute_capability{8, 6}.major)() == 8);
static_assert((cuda::compute_capability{8, 6}.minor)() == 6);

int main(int, char**)
{
  return 0;
}
