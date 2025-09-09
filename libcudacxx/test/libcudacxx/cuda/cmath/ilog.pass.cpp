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
#include <cuda/std/cmath>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ constexpr void test_log2()
{
  int i = 0;
  for (T value = 1; value <= cuda::std::numeric_limits<T>::max() / 2; value *= 2)
  {
    assert(cuda::ilog2(value) == i);
    if (value > 1)
    {
      assert(cuda::ilog2(static_cast<T>(value - 1)) == i - 1);
      assert(cuda::ilog2(static_cast<T>(value + 1)) == i);
    }
    i++;
  }
  assert(cuda::ilog2(cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::digits - 1);
}

template <class T>
__host__ __device__ constexpr void test_ceil_log2()
{
  int i = 0;
  for (T value = 1; value <= cuda::std::numeric_limits<T>::max() / 2; value *= 2)
  {
    assert(cuda::ceil_ilog2(value) == i);
    assert(cuda::ceil_ilog2(static_cast<T>(value + 1)) == i + 1);
    if (value > 2)
    {
      assert(cuda::ceil_ilog2(static_cast<T>(value - 1)) == i);
    }
    i++;
  }
  assert(cuda::ceil_ilog2(cuda::std::numeric_limits<T>::max()) == cuda::std::numeric_limits<T>::digits);
}

template <class T>
__host__ __device__ constexpr void test_log10()
{
  int i = 0;
  for (T value = 1; value <= cuda::std::numeric_limits<T>::max() / 10; value *= 10)
  {
    assert(cuda::ilog10(value) == i);
    assert(cuda::ilog10(value + value / 2) == i);
    if (i >= 1)
    {
      assert(cuda::ilog10(static_cast<T>(value - 1)) == i - 1);
      assert(cuda::ilog10(static_cast<T>(value + 1)) == i);
      assert(cuda::ilog10(value - value / 2) == i - 1);
      assert(cuda::ilog10(value - value / 2 - 1) == i - 1);
    }
    i++;
  }
#if !TEST_COMPILER(MSVC)
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    constexpr auto max_v = cuda::std::numeric_limits<T>::max();
    assert(cuda::ilog10(max_v) == static_cast<int>(cuda::std::floor(cuda::std::log10(max_v))));
  }
#endif // !TEST_COMPILER(MSVC)
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_log2<T>();
  test_ceil_log2<T>();
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

#if !TEST_COMPILER(NVRTC)
  // cstdint types:
  test<cuda::std::size_t>();
  test<cuda::std::ptrdiff_t>();
  test<cuda::std::intptr_t>();
  test<cuda::std::uintptr_t>();

  test<cuda::std::int8_t>();
  test<cuda::std::int16_t>();
  test<cuda::std::int32_t>();
  test<cuda::std::int64_t>();

  test<cuda::std::uint8_t>();
  test<cuda::std::uint16_t>();
  test<cuda::std::uint32_t>();
  test<cuda::std::uint64_t>();
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
