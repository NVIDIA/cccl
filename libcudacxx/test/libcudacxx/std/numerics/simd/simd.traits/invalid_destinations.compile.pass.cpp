//===----------------------------------------------------------------------===//
//
// Part of libcu++ in the CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: calling a host device function in tile mode

// [simd.traits], invalid destination types

#include <cuda/std/__simd_>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

namespace simd = cuda::std::simd;

template <typename T, typename = void>
inline constexpr bool has_type_field = false;

template <typename T>
inline constexpr bool has_type_field<T, cuda::std::void_t<typename T::type>> = true;

template <typename T, typename = void>
inline constexpr bool has_value_field = false;

template <typename T>
inline constexpr bool has_value_field<T, cuda::std::void_t<decltype(T::value)>> = true;

struct unrelated_type
{};

struct incomplete_type;

template <typename T>
TEST_HOST_DEVICE_FUNC constexpr void test_invalid_destination()
{
  using V = simd::vec<int, 4>;
  using M = simd::mask<int, 4>;

  static_cast<void>(sizeof(simd::rebind<T, V>)); // is a complete type
  static_cast<void>(sizeof(simd::rebind<T, M>)); // is a complete type
  static_assert(!has_type_field<simd::rebind<T, V>>);
  static_assert(!has_type_field<simd::rebind<T, M>>);

  static_cast<void>(sizeof(simd::alignment<V, T>)); // is a complete type
  static_assert(!has_value_field<simd::alignment<V, T>>);
}

TEST_HOST_DEVICE_FUNC constexpr bool test_invalid_destinations()
{
  test_invalid_destination<bool>();
  test_invalid_destination<const int>();
  test_invalid_destination<volatile int>();
  test_invalid_destination<void>();
  test_invalid_destination<int*>();
  test_invalid_destination<int&>();
  test_invalid_destination<void (&)()>();
  test_invalid_destination<unrelated_type>();
  test_invalid_destination<incomplete_type>();
  test_invalid_destination<cuda::std::complex<int>>();
  return true;
}

int main(int, char**)
{
  static_assert(test_invalid_destinations());
  return 0;
}
