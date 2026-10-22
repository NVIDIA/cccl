// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Verifies that CCCL_IGNORE_DEPRECATED_COMPILER silences the deprecated host
// compiler warning in thrust/detail/config/config.h.
//
// The compiler detection is faked below so that the suppression is tested on
// every configuration, and not only on the deprecated compilers themselves,
// which CI does not build with. Compiling this file is the test: warnings are
// errors for CCCL targets, so a warning that escapes the suppression lets the
// build fail. See NVIDIA/cccl#1620.

#define CCCL_IGNORE_DEPRECATED_COMPILER

// See deprecated_compiler_warning.cpp: CCCL builds its own tests with
// _CCCL_NO_SYSTEM_HEADER, so drop it again to match how users see the header.
#undef _CCCL_NO_SYSTEM_HEADER

#include <cuda/__cccl_config>

#undef _CCCL_COMPILER_GCC
#define _CCCL_COMPILER_GCC() (6, 0)

#include <thrust/detail/config.h>

int main()
{
  return 0;
}
