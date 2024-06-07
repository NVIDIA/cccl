//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/utility>
#include "test_macros.h"

template <class T>
void test()
{
  constexpr T maxv = cuda::std::numeric_limits<T>::max();

  assert(ceil_div(T(0), T(1)) == T(0));
  assert(ceil_div(T(1), T(1)) == T(1));
  assert(ceil_div(maxv, T(1)) == maxv);
  assert(ceil_div(maxv, maxv) == T(1));
}

#include <cstdint>

int main(int arg, char** argv)
{
  // Builtin integer types:
  test<char>();
  test<signed char>();
  test<unsigned char>();

  test<int>();
  test<unsigned int>();

  test<long>();
  test<unsigned long>();

  test<long long>();
  test<unsigned long long>();

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

  // Other
  test<__int128>();

  return 0;
}
