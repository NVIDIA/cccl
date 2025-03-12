//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_log2()
{
  int i = 0;
  for (T value = 1; value <= cuda::std::numeric_limits<T>::max() / 2; value *= 2)
  {
    if (i >= 1)
    {
      assert(cuda::ilog2(static_cast<T>(value - 1)) == i - 1);
      assert(cuda::ilog2(static_cast<T>(value + 1)) == i); // not true if value == 1
    }
    assert(cuda::ilog2(value) == i);
    i++;
  }
  assert(cuda::ilog2(T{1}) == 0);
  assert(cuda::ilog2(cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::digits - 1);
}

template <class T>
__host__ __device__ constexpr void test_log10()
{
  int i = 0;
  for (T value = 1; value <= cuda::std::numeric_limits<T>::max() / 10; value *= 10)
  {
    if (i >= 1)
    {
      assert(cuda::ilog10(static_cast<T>(value - 1)) == i - 1);
      assert(cuda::ilog10(static_cast<T>(value + 1)) == i);
    }
    assert(cuda::ilog10(value) == i);
    i++;
  }
  static_assert(cuda::ilog10(T{1}) == 0);
  static_assert(cuda::ilog10(T{9}) == 0);
  static_assert(cuda::ilog10(T{10}) == 1);
  static_assert(cuda::ilog10(T{100}) == 2);
  static_assert(cuda::ilog10(T{10}) == 1);
  static_assert(cuda::ilog10(cuda::std::numeric_limits<T>::max()) <= cuda::std::numeric_limits<T>::digits / 3);
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_log2<T>();
  test_log10<T>();
}

__host__ __device__ constexpr bool test()
{
  // Builtin integer types:
  test<signed char>();
  test<unsigned char>();

  test<short>();
  test<unsigned short>();

  test<int>();
  test<unsigned int>();

  test<long>();
  test<unsigned long>();

  test<long long>();
  test<unsigned long long>();

#if !defined(TEST_COMPILER_NVRTC)
  // cstdint types:
  test<std::size_t>();
  test<std::ptrdiff_t>();
  test<std::intptr_t>();
  test<std::uintptr_t>();

  test<std::int8_t>();
  test<std::int16_t>();
  test<std::int32_t>();
  test<std::int64_t>();

  test<std::uint8_t>();
  test<std::uint16_t>();
  test<std::uint32_t>();
  test<std::uint64_t>();
#endif // !TEST_COMPILER_NVRTC

  // #if _CCCL_HAS_INT128()
  //   test<__int128_t>();
  //   test<__uint128_t>();
  // #endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
