//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <random>
//
// Verify [rand.req.genl]/1.6 and /1.7 (C++26, P4037R1) for the set of integer
// types accepted as IntType (for distributions) and UIntType (for engines).
//
// - signed char / unsigned char must be accepted (P4037R1).
// - char / bool / char8_t / char16_t / char32_t / wchar_t must be rejected.
// - __int128_t / __uint128_t are part of CCCL's implementation-defined subset.

#include <cuda/std/random>
#include <cuda/std/type_traits>

#include "test_macros.h"

// -----------------------------------------------------------------------------
// IntType trait (__cccl_random_is_valid_inttype)
// -----------------------------------------------------------------------------

// Accepted: standard signed/unsigned integer types (including signed/unsigned char).
static_assert(cuda::std::__cccl_random_is_valid_inttype<signed char>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<unsigned char>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<short>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<unsigned short>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<int>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<unsigned int>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<long>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<unsigned long>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<long long>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<unsigned long long>);

#if _CCCL_HAS_INT128()
// CCCL implementation defined extended integer types.
static_assert(cuda::std::__cccl_random_is_valid_inttype<__int128_t>);
static_assert(cuda::std::__cccl_random_is_valid_inttype<__uint128_t>);
#endif // _CCCL_HAS_INT128()

// Rejected: non-integer types and character/boolean types
static_assert(!cuda::std::__cccl_random_is_valid_inttype<bool>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<char>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<char16_t>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<char32_t>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<wchar_t>);
#if _CCCL_HAS_CHAR8_T()
static_assert(!cuda::std::__cccl_random_is_valid_inttype<char8_t>);
#endif // _CCCL_HAS_CHAR8_T()
static_assert(!cuda::std::__cccl_random_is_valid_inttype<float>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<double>);
static_assert(!cuda::std::__cccl_random_is_valid_inttype<void*>);

// -----------------------------------------------------------------------------
// UIntType trait (__cccl_random_is_valid_uinttype)
// -----------------------------------------------------------------------------

// Accepted: standard unsigned integer types
static_assert(cuda::std::__cccl_random_is_valid_uinttype<unsigned char>);
static_assert(cuda::std::__cccl_random_is_valid_uinttype<unsigned short>);
static_assert(cuda::std::__cccl_random_is_valid_uinttype<unsigned int>);
static_assert(cuda::std::__cccl_random_is_valid_uinttype<unsigned long>);
static_assert(cuda::std::__cccl_random_is_valid_uinttype<unsigned long long>);

#if _CCCL_HAS_INT128()
// CCCL implementation-defined extended unsigned integer type.
static_assert(cuda::std::__cccl_random_is_valid_uinttype<__uint128_t>);
// Signed 128-bit must be rejected (UIntType requires unsigned).
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<__int128_t>);
#endif // _CCCL_HAS_INT128()

// Rejected: signed integer types (UIntType requires unsigned).
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<signed char>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<short>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<int>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<long>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<long long>);

// Rejected: bool and character types.
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<bool>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<char>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<char16_t>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<char32_t>);
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<wchar_t>);
#if _CCCL_HAS_CHAR8_T()
static_assert(!cuda::std::__cccl_random_is_valid_uinttype<char8_t>);
#endif // _CCCL_HAS_CHAR8_T()

int main(int, char**)
{
  return 0;
}
