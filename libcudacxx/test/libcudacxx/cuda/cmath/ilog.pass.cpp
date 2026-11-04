//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Expanded boundary coverage across all integer types exceeds the default Clang constexpr step limit.
// CONSTEXPR_STEPS: 15000000

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include "test_macros.h"

template <class T>
TEST_FUNC constexpr void test_log2()
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
TEST_FUNC constexpr void test_ceil_log2()
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
TEST_FUNC constexpr int reference_log10(T value)
{
  int result = 0;
  while (value >= 10)
  {
    value /= 10;
    ++result;
  }
  return result;
}

template <class T>
TEST_FUNC constexpr void test_log10()
{
  int i = 0;
  for (T value = 1;; value *= 10)
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
    if (value > cuda::std::numeric_limits<T>::max() / 10)
    {
      break;
    }
    i++;
  }
  constexpr auto max_v = cuda::std::numeric_limits<T>::max();
  assert(cuda::ilog10(max_v) == reference_log10(max_v));
  assert(cuda::ilog10<const T>(max_v) == cuda::std::numeric_limits<T>::digits10);
  assert(cuda::ilog10(static_cast<T>(max_v - 1)) == reference_log10(static_cast<T>(max_v - 1)));
}

template <class T>
TEST_FUNC constexpr void test_ceil_log10()
{
  int i = 0;
  for (T value = 1;; value *= 10)
  {
    assert(cuda::ceil_ilog10(value) == i);
    assert(cuda::ceil_ilog10(static_cast<T>(value + 1)) == i + 1);
    if (i >= 1)
    {
      assert(cuda::ceil_ilog10(static_cast<T>(value - 1)) == i);
      assert(cuda::ceil_ilog10(value + value / 2) == i + 1);
      assert(cuda::ceil_ilog10(value - value / 2) == i);
      assert(cuda::ceil_ilog10(value - value / 2 - 1) == i);
    }
    if (value > cuda::std::numeric_limits<T>::max() / 10)
    {
      break;
    }
    i++;
  }
  constexpr auto max_v = cuda::std::numeric_limits<T>::max();
  assert(cuda::ceil_ilog10(max_v) == reference_log10(static_cast<T>(max_v - 1)) + 1);
}

template <class T>
TEST_FUNC constexpr void test_log10_binary_boundaries()
{
  // The approximation changes at powers of two; the exact correction changes at powers of ten.
  for (T value = 1;; value *= 2)
  {
    assert(cuda::ilog10(value) == reference_log10(value));
    assert(cuda::ceil_ilog10(value) == (value == 1 ? 0 : reference_log10(static_cast<T>(value - 1)) + 1));
    if (value > 1)
    {
      auto before = static_cast<T>(value - 1);
      auto after  = static_cast<T>(value + 1);
      assert(cuda::ilog10(before) == reference_log10(before));
      assert(cuda::ilog10(after) == reference_log10(after));
      assert(cuda::ceil_ilog10(before) == (before == 1 ? 0 : reference_log10(static_cast<T>(before - 1)) + 1));
      assert(cuda::ceil_ilog10(after) == reference_log10(value) + 1);
    }
    if (value > cuda::std::numeric_limits<T>::max() / 2)
    {
      break;
    }
  }
}

template <class T>
TEST_FUNC constexpr void test()
{
  test_log2<T>();
  test_ceil_log2<T>();
  test_log10<T>();
  test_ceil_log10<T>();
  test_log10_binary_boundaries<T>();
}

TEST_FUNC constexpr bool test()
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
