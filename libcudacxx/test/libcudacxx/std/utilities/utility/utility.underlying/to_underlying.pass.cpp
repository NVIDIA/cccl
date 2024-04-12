//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// [utility.underlying], to_underlying
// template <class T>
//     constexpr underlying_type_t<T> to_underlying( T value ) noexcept; // C++2b

#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include "test_macros.h"

enum class e_default
{
  a = 0,
  b = 1,
  c = 2
};
enum class e_ushort : unsigned short
{
  d = 10,
  e = 25,
  f = 50
};
enum class e_longlong : long long
{
  low  = cuda::std::numeric_limits<long long>::min(),
  high = cuda::std::numeric_limits<long long>::max()
};
enum e_non_class
{
  enum_a = 10,
  enum_b = 11,
  enum_c = 12
};
enum e_int : int
{
  enum_min = cuda::std::numeric_limits<int>::min(),
  enum_max = cuda::std::numeric_limits<int>::max()
};
enum class e_bool : cuda::std::uint8_t
{
  f = 0,
  t = 1
};

struct WithBitfieldEnums
{
  e_default e1 : 3;
  e_ushort e2  : 6;
  e_bool e3    : 1;
};

__host__ __device__ constexpr bool test()
{
  ASSERT_NOEXCEPT(cuda::std::to_underlying(e_default::a));
  ASSERT_SAME_TYPE(int, decltype(cuda::std::to_underlying(e_default::a)));
  ASSERT_SAME_TYPE(unsigned short, decltype(cuda::std::to_underlying(e_ushort::d)));
  ASSERT_SAME_TYPE(long long, decltype(cuda::std::to_underlying(e_longlong::low)));
  ASSERT_SAME_TYPE(int, decltype(cuda::std::to_underlying(enum_min)));
  ASSERT_SAME_TYPE(int, decltype(cuda::std::to_underlying(enum_max)));

  assert(0 == cuda::std::to_underlying(e_default::a));
  assert(1 == cuda::std::to_underlying(e_default::b));
  assert(2 == cuda::std::to_underlying(e_default::c));

  assert(10 == cuda::std::to_underlying(e_ushort::d));
  assert(25 == cuda::std::to_underlying(e_ushort::e));
  assert(50 == cuda::std::to_underlying(e_ushort::f));

  // Check no truncating.
  assert(cuda::std::numeric_limits<long long>::min() == cuda::std::to_underlying(e_longlong::low));
  assert(cuda::std::numeric_limits<long long>::max() == cuda::std::to_underlying(e_longlong::high));

  assert(10 == cuda::std::to_underlying(enum_a));
  assert(11 == cuda::std::to_underlying(enum_b));
  assert(12 == cuda::std::to_underlying(enum_c));
  assert(cuda::std::numeric_limits<int>::min() == cuda::std::to_underlying(enum_min));
  assert(cuda::std::numeric_limits<int>::max() == cuda::std::to_underlying(enum_max));

  WithBitfieldEnums bf;
  bf.e1 = static_cast<e_default>(3);
  bf.e2 = e_ushort::e;
  bf.e3 = e_bool::t;
  assert(3 == cuda::std::to_underlying(bf.e1));
  assert(25 == cuda::std::to_underlying(bf.e2));
  assert(1 == cuda::std::to_underlying(bf.e3));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
