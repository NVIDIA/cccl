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

// Older versions of the MSVC STL do not protect [[msvc::known_semantics]] in <utility> and [[msvc::noop_dtor]] in
// <system_error> against these macros, so include them before defining them.
#if __has_include(<utility>)
#  include <system_error>
#  include <utility>
#endif // __has_include(<utility>)

// Some code bases define `clang`, `msvc` or the names of vendor attributes as macros. They must break neither the
// feature checks in <cuda/std/__cccl/attributes.h> nor the attribute macros, like _CCCL_INTRINSIC. These headers
// include every libcu++ header that uses them.
#define clang              "clang must not be expanded inside CCCL headers"
#define msvc               "msvc must not be expanded inside CCCL headers"
#define intrinsic          "intrinsic must not be expanded inside CCCL headers"
#define lifetimebound      "lifetimebound must not be expanded inside CCCL headers"
#define noalias            "noalias must not be expanded inside CCCL headers"
#define no_specializations "no_specializations must not be expanded inside CCCL headers"

#include <cuda/annotated_ptr>
#include <cuda/bit>
#include <cuda/cmath>
#include <cuda/mdspan>
#include <cuda/memory_resource>
#include <cuda/numeric>
#include <cuda/simd>
#include <cuda/std/__format_>
#include <cuda/std/__simd_>
#include <cuda/std/algorithm>
#include <cuda/std/execution>
#include <cuda/std/numeric>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#if !defined(clang) || !defined(msvc) || !defined(intrinsic) || !defined(lifetimebound) || !defined(noalias) \
  || !defined(no_specializations)
#  error "libcu++ headers must restore the macros"
#endif

int main(int, char**)
{
  return 0;
}
