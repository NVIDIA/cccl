//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/__simd_>

// [simd.expos] __is_vectorizable_v: true for all standard integer types,
// character types, and float/double; false for bool, const/volatile types.

#include <cuda/std/__simd_>
#include <cuda/std/cstdint>

#include "test_macros.h"

namespace simd = cuda::std::simd;

// positive cases: integer types
static_assert(simd::__is_vectorizable_v<int8_t>);
static_assert(simd::__is_vectorizable_v<int16_t>);
static_assert(simd::__is_vectorizable_v<int32_t>);
static_assert(simd::__is_vectorizable_v<int64_t>);
static_assert(simd::__is_vectorizable_v<uint8_t>);
static_assert(simd::__is_vectorizable_v<uint16_t>);
static_assert(simd::__is_vectorizable_v<uint32_t>);
static_assert(simd::__is_vectorizable_v<uint64_t>);

// positive cases: character types
static_assert(simd::__is_vectorizable_v<char>);
static_assert(simd::__is_vectorizable_v<char16_t>);
static_assert(simd::__is_vectorizable_v<char32_t>);
static_assert(simd::__is_vectorizable_v<wchar_t>);
#if defined(__cccl_lib_char8_t)
static_assert(simd::__is_vectorizable_v<char8_t>);
#endif

#if _CCCL_HAS_INT128()
static_assert(simd::__is_vectorizable_v<__int128_t>);
#endif

// floating-point types
static_assert(simd::__is_vectorizable_v<float>);
static_assert(simd::__is_vectorizable_v<double>);
#if _LIBCUDACXX_HAS_NVFP16()
static_assert(simd::__is_vectorizable_v<__half>);
#endif
#if _LIBCUDACXX_HAS_NVBF16()
static_assert(simd::__is_vectorizable_v<__nv_bfloat16>);
#endif

// negative cases
static_assert(!simd::__is_vectorizable_v<bool>);
static_assert(!simd::__is_vectorizable_v<const int>);
static_assert(!simd::__is_vectorizable_v<volatile int>);
static_assert(!simd::__is_vectorizable_v<const volatile int>);
static_assert(!simd::__is_vectorizable_v<void>);

struct user_type
{};
static_assert(!simd::__is_vectorizable_v<user_type>);
static_assert(!simd::__is_vectorizable_v<int*>);
static_assert(!simd::__is_vectorizable_v<int&>);

int main(int, char**)
{
  return 0;
}
