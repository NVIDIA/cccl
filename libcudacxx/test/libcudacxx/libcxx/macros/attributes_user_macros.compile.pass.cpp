//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// NVRTC does not support #pragma push_macro and #pragma pop_macro.
// UNSUPPORTED: nvrtc

// Users may define the names of vendor attributes as macros before or after including CCCL headers.

// Host standard libraries do not all protect themselves against these macros, so include the host headers that the
// CCCL headers below use first.
#if __has_include(<version>)
#  include <version>
#endif // __has_include(<version>)
#if __has_include(<utility>)
#  include <cfloat>
#  include <climits>
#  include <cstddef>
#  include <cstdint>
#  include <cstring>
#  include <iterator>
#  include <memory>
#  include <type_traits>
#  include <utility>
#endif // __has_include(<utility>)

#define lifetimebound

#include <cuda/std/__cccl/attributes.h>

// `[[clang::lifetimebound]]` and `[[msvc::lifetimebound]]` are optional, so they are not used if users define
// `clang`, `msvc` or `lifetimebound` as macros before including CCCL headers.
constexpr const int& lifetimebound_test(const int& x _CCCL_LIFETIMEBOUND) noexcept
{
  return x;
}

// The headers that use _CCCL_INTRINSIC and _CCCL_NO_SPECIALIZATIONS push and pop the names of their attributes.
#define clang
#define intrinsic
#define msvc
#define no_specializations

#include <cuda/std/type_traits>
#include <cuda/std/utility>

#if !defined(clang) || !defined(intrinsic) || !defined(lifetimebound) || !defined(msvc) || !defined(no_specializations)
#  error "CCCL headers must restore the macros"
#endif

int main(int, char**)
{
  return 0;
}
