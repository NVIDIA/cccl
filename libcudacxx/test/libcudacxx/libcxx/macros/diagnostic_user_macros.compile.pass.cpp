//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// FORCE_ALL_WARNINGS.
// TREAT_WARNINGS_AS_ERRORS.

#include <cuda/std/__cccl/diagnostic.h>

// A diagnostic pragma that expanded these macros would be an unknown pragma.
#define clang "clang must not be expanded in a diagnostic pragma"
#define GCC   "GCC must not be expanded in a diagnostic pragma"

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Wdeprecated")
_CCCL_DIAG_SUPPRESS_GCC("-Wdeprecated")
_CCCL_DIAG_POP

int main(int, char**)
{
  return 0;
}
