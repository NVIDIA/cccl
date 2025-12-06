//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// int ffs(T x) noexcept;
//
// Returns: The 1-based index of the first (least significant) set bit, or 0 if x == 0.

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "literal.h"
#include "test_macros.h"

using namespace test_integer_literals;

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
  assert(cuda::ffs(T(0)) == 0);
  assert(cuda::ffs(T(1)) == 1);
  assert(cuda::ffs(T(2)) == 2);
  assert(cuda::ffs(T(3)) == 1);
  assert(cuda::ffs(T(4)) == 3);
  assert(cuda::ffs(T(5)) == 1);
  assert(cuda::ffs(T(6)) == 2);
  assert(cuda::ffs(T(7)) == 1);
  assert(cuda::ffs(T(8)) == 4);
  assert(cuda::ffs(T(9)) == 1);
  assert(cuda::ffs(T(127)) == 1);
  assert(cuda::ffs(T(128)) == 8);

  // Test relationship with countr_zero: ffs(x) == countr_zero(x) + 1 for x != 0
  assert(cuda::ffs(T(1)) == cuda::std::countr_zero(T(1)) + 1);
  assert(cuda::ffs(T(2)) == cuda::std::countr_zero(T(2)) + 1);
  assert(cuda::ffs(T(4)) == cuda::std::countr_zero(T(4)) + 1);
  assert(cuda::ffs(T(8)) == cuda::std::countr_zero(T(8)) + 1);

  // Test MSB for different sizes (compile-time)
  if constexpr (sizeof(T) >= 4)
  {
    assert(cuda::ffs(T(0x80000000u)) == 32);
  }
  if constexpr (sizeof(T) >= 8)
  {
    assert(cuda::ffs(T(0x8000000000000000ull)) == 64);
  }

  return true;
}

template <typename T>
__host__ __device__ inline void assert_ffs(T val, int expected)
{
  volatile auto v = val;
  assert(cuda::ffs(v) == expected);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  static_assert(cuda::std::is_same_v<int, decltype(cuda::ffs(T(0)))>, "");
  static_assert(noexcept(cuda::ffs(T(0))), "");

  assert_ffs(T(0), 0);
  assert_ffs(T(1), 1);
  assert_ffs(T(121), 1);
  assert_ffs(T(122), 2);
  assert_ffs(T(124), 3);

  if constexpr (sizeof(T) > 1)
  {
    assert_ffs(T(128), 8);
    assert_ffs(T(256), 9);
    assert_ffs(T(512), 10);
    assert_ffs(T(1024), 11);
  }

  if constexpr (sizeof(T) >= 4)
  {
    assert_ffs(T(0x80000000u), 32);
  }

  if constexpr (sizeof(T) >= 8)
  {
    assert_ffs(T(0x8000000000000000ull), 64);
  }
}

int main(int, char**)
{
  static_assert(constexpr_test<unsigned char>(), "");
  static_assert(constexpr_test<unsigned short>(), "");
  static_assert(constexpr_test<unsigned>(), "");
  static_assert(constexpr_test<unsigned long>(), "");
  static_assert(constexpr_test<unsigned long long>(), "");

#if _CCCL_HAS_INT128()
  static_assert(constexpr_test<__uint128_t>(), "");
#endif // _CCCL_HAS_INT128()

  runtime_test<unsigned char>();
  runtime_test<unsigned short>();
  runtime_test<unsigned>();
  runtime_test<unsigned long>();
  runtime_test<unsigned long long>();

#if _CCCL_HAS_INT128()
  runtime_test<__uint128_t>();

  // Additional 128-bit tests with literals
  assert_ffs(0_u128, 0);
  assert_ffs(1_u128, 1);
  assert_ffs(0x8000000000000000_u128, 64);
  assert_ffs((1_u128 << 64), 65);
  assert_ffs((1_u128 << 65), 66);
  assert_ffs((1_u128 << 100), 101);
  assert_ffs((1_u128 << 127), 128);
#endif // _CCCL_HAS_INT128()

  return 0;
}
