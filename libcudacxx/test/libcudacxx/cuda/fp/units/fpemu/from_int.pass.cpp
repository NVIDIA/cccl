// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
//  Unit test: integer -> fp64emu -> double conversions (bit-exact).
//
//  Validates that fpemu integer-to-double conversions produce bit-identical
//  results to native casts for int32_t / uint32_t / int64_t / uint64_t as well as
//  long long / unsigned long long (which route through the constrained integer
//  constructor template). int32/uint32 are always exact; the 64-bit types require
//  round-to-nearest-even for values with more than 53 significant bits.
//
//===----------------------------------------------------------------------===//

#include <cuda/fpemu>
#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace cudax = cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

#if _CCCL_HAS_INT128()
// 128-bit integer construction is deliberately deleted: it would silently truncate
// to 64 bits. Verify no emulated type is constructible from __int128 while the
// standard integer widths remain constructible.
static_assert(!cuda::std::is_constructible_v<cudax::fpemu<double>, __int128_t>);
static_assert(!cuda::std::is_constructible_v<cudax::fpemu<double>, __uint128_t>);
static_assert(!cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, __int128_t>);
static_assert(!cuda::std::is_constructible_v<cudax::fpemu_unpacked<double>, __uint128_t>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, int64_t>);
static_assert(cuda::std::is_constructible_v<cudax::fpemu<double>, uint64_t>);
#endif // _CCCL_HAS_INT128()

// Convert one integer through fp64emu and compare bit-for-bit against the native
// cast to double.
template <class T>
TEST_HOST_DEVICE_FUNC bool int_ok(T v)
{
  cudax::fp64emu e(v);
  return cuda::std::bit_cast<uint64_t>((double) e) == cuda::std::bit_cast<uint64_t>((double) v);
}

template <class T, int N>
TEST_HOST_DEVICE_FUNC void check_all(const T (&vals)[N])
{
  for (int i = 0; i < N; i++)
  {
    assert(int_ok(vals[i]));
  }
}

// Converts a spread of boundary integers (type limits, 2^53 rounding thresholds,
// power-of-two edges) of every supported width and checks each against the native
// cast to double.
TEST_FUNC void test()
{
  const int32_t int32_vals[] = {
    0,
    1,
    -1,
    2,
    -2,
    100,
    -100,
    1000000,
    -1000000,
    INT32_MAX,
    INT32_MIN,
    INT32_MIN + 1,
    INT32_MAX - 1,
    0x7FFFFFFF,
    (int32_t) 0x80000000,
    12345678,
    -12345678,
  };
  check_all(int32_vals);

  const uint32_t uint32_vals[] = {
    0,
    1,
    2,
    100,
    1000000,
    0x7FFFFFFFu,
    0x80000000u,
    0xFFFFFFFFu,
    0xFFFFFFFEu,
    42,
    999999999,
    0x12345678u,
    0xDEADBEEFu,
  };
  check_all(uint32_vals);

  const int64_t int64_vals[] = {
    0,
    1,
    -1,
    2,
    -2,
    INT32_MAX,
    INT32_MIN,
    (int64_t) INT32_MAX + 1,
    (int64_t) INT32_MIN - 1,
    INT64_MAX,
    INT64_MIN,
    INT64_MIN + 1,
    INT64_MAX - 1,
    (1LL << 53),
    -(1LL << 53),
    (1LL << 53) + 1,
    (1LL << 53) - 1,
    (1LL << 53) + 2,
    (1LL << 53) + 3,
    (1LL << 54) - 1,
    -(1LL << 54) + 1,
    (1LL << 62),
    -(1LL << 62),
    0x100000000LL,
    -0x100000000LL,
    123456789012345LL,
    -123456789012345LL,
  };
  check_all(int64_vals);

  const uint64_t uint64_vals[] = {
    0,
    1,
    2,
    (uint64_t) UINT32_MAX,
    (uint64_t) UINT32_MAX + 1,
    UINT64_MAX,
    UINT64_MAX - 1,
    (1ULL << 53),
    (1ULL << 53) + 1,
    (1ULL << 53) - 1,
    (1ULL << 53) + 2,
    (1ULL << 53) + 3,
    (1ULL << 54) - 1,
    (1ULL << 63),
    (1ULL << 63) + 1,
    0x8000000000000000ULL,
    0xFFFFFFFF00000000ULL,
    123456789012345ULL,
    9999999999999999ULL,
  };
  check_all(uint64_vals);

  const long long longlong_vals[] = {
    0,
    1,
    -1,
    2,
    -2,
    INT32_MAX,
    INT32_MIN,
    (long long) INT32_MAX + 1,
    (long long) INT32_MIN - 1,
    INT64_MAX,
    INT64_MIN,
    INT64_MIN + 1,
    INT64_MAX - 1,
    (1LL << 53),
    -(1LL << 53),
    (1LL << 53) + 1,
    (1LL << 53) - 1,
    (1LL << 62),
    -(1LL << 62),
    123456789012345LL,
    -123456789012345LL,
  };
  check_all(longlong_vals);

  const unsigned long long ulonglong_vals[] = {
    0,
    1,
    2,
    (unsigned long long) UINT32_MAX,
    (unsigned long long) UINT32_MAX + 1,
    UINT64_MAX,
    UINT64_MAX - 1,
    (1ULL << 53),
    (1ULL << 53) + 1,
    (1ULL << 53) - 1,
    (1ULL << 63),
    0x8000000000000000ULL,
    0xFFFFFFFF00000000ULL,
    123456789012345ULL,
    9999999999999999ULL,
  };
  check_all(ulonglong_vals);
}

int main(int, char**)
{
  test();

  return 0;
}
