//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/version>

// Define these macros to true values. This tests that CCCL_HOST_COMPILER(FOO) is resilient against
// macro expansion in case the user defines FOO, because if CCCL_HOST_COMPILER() expands the macro,
// then the below assertions should fire.
#define NVHPC    1
#define CLANG    1
#define GCC      1
#define MSVC     1
#define MSVC2019 1
#define MSVC2022 1
#define MSVC2026 1

#if CCCL_HOST_COMPILER(NVHPC) != _CCCL_COMPILER(NVHPC)
#  error "CCCL_HOST_COMPILER(NVHPC) does not match _CCCL_COMPILER(NVHPC)"
#endif

#if CCCL_HOST_COMPILER(CLANG) != _CCCL_COMPILER(CLANG)
#  error "CCCL_HOST_COMPILER(CLANG) does not match _CCCL_COMPILER(CLANG)"
#endif

#if CCCL_HOST_COMPILER(GCC) != _CCCL_COMPILER(GCC)
#  error "CCCL_HOST_COMPILER(GCC) does not match _CCCL_COMPILER(GCC)"
#endif

#if CCCL_HOST_COMPILER(GCC, >=, 0) != _CCCL_COMPILER(GCC, >=, 0)
#  error "CCCL_HOST_COMPILER(GCC, >=, 0) does not match _CCCL_COMPILER(GCC, >=, 0)"
#endif

#if CCCL_HOST_COMPILER(CLANG, >=, 0, 0) != _CCCL_COMPILER(CLANG, >=, 0, 0)
#  error "CCCL_HOST_COMPILER(CLANG, >=, 0, 0) does not match _CCCL_COMPILER(CLANG, >=, 0, 0)"
#endif

#if CCCL_HOST_COMPILER(MSVC) != _CCCL_COMPILER(MSVC)
#  error "CCCL_HOST_COMPILER(MSVC) does not match _CCCL_COMPILER(MSVC)"
#endif

#if CCCL_HOST_COMPILER(MSVC2019) != _CCCL_COMPILER(MSVC2019)
#  error "CCCL_HOST_COMPILER(MSVC2019) does not match _CCCL_COMPILER(MSVC2019)"
#endif

#if CCCL_HOST_COMPILER(MSVC2022) != _CCCL_COMPILER(MSVC2022)
#  error "CCCL_HOST_COMPILER(MSVC2022) does not match _CCCL_COMPILER(MSVC2022)"
#endif

#if CCCL_HOST_COMPILER(MSVC2026) != _CCCL_COMPILER(MSVC2026)
#  error "CCCL_HOST_COMPILER(MSVC2026) does not match _CCCL_COMPILER(MSVC2026)"
#endif

#ifdef _CCCL_HOST_COMPILER_NVRTC
#  error "NVRTC must not be exposed as a host compiler"
#endif

#undef NVHPC
#undef CLANG
#undef GCC
#undef MSVC
#undef MSVC2019
#undef MSVC2022
#undef MSVC2026

int main(int, char**)
{
  static_assert(CCCL_HOST_COMPILER(NVHPC) == _CCCL_COMPILER(NVHPC));
  static_assert(CCCL_HOST_COMPILER(CLANG) == _CCCL_COMPILER(CLANG));
  static_assert(CCCL_HOST_COMPILER(GCC) == _CCCL_COMPILER(GCC));
  static_assert(CCCL_HOST_COMPILER(MSVC) == _CCCL_COMPILER(MSVC));
  static_assert(CCCL_HOST_COMPILER(MSVC2019) == _CCCL_COMPILER(MSVC2019));
  static_assert(CCCL_HOST_COMPILER(MSVC2022) == _CCCL_COMPILER(MSVC2022));
  static_assert(CCCL_HOST_COMPILER(MSVC2026) == _CCCL_COMPILER(MSVC2026));

  static_assert(CCCL_HOST_COMPILER(NVHPC, >=, 0) == _CCCL_COMPILER(NVHPC, >=, 0));
  static_assert(CCCL_HOST_COMPILER(CLANG, >=, 0) == _CCCL_COMPILER(CLANG, >=, 0));
  static_assert(CCCL_HOST_COMPILER(GCC, >=, 0) == _CCCL_COMPILER(GCC, >=, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC, >=, 0) == _CCCL_COMPILER(MSVC, >=, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2019, >=, 0) == _CCCL_COMPILER(MSVC2019, >=, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2022, >=, 0) == _CCCL_COMPILER(MSVC2022, >=, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2026, >=, 0) == _CCCL_COMPILER(MSVC2026, >=, 0));

  static_assert(CCCL_HOST_COMPILER(NVHPC, >=, 0, 0) == _CCCL_COMPILER(NVHPC, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(CLANG, >=, 0, 0) == _CCCL_COMPILER(CLANG, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(GCC, >=, 0, 0) == _CCCL_COMPILER(GCC, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC, >=, 0, 0) == _CCCL_COMPILER(MSVC, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2019, >=, 0, 0) == _CCCL_COMPILER(MSVC2019, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2022, >=, 0, 0) == _CCCL_COMPILER(MSVC2022, >=, 0, 0));
  static_assert(CCCL_HOST_COMPILER(MSVC2026, >=, 0, 0) == _CCCL_COMPILER(MSVC2026, >=, 0, 0));
  return 0;
}
