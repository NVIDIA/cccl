//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03, c++11, c++14

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include "cuda/std/__type_traits/conditional.h"
#include "cuda/std/__type_traits/is_enum.h"
#include "cuda/std/__type_traits/underlying_type.h"
#include "test_macros.h"

#if !defined(TEST_COMPILER_NVRTC)
#  include <cstdint>
#endif // !TEST_COMPILER_NVRTC

template <class T, class U>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  constexpr auto maxv = cuda::std::numeric_limits<T>::max();
  using CommonType    = cuda::std::common_type_t<T, U>;
  // ensure that we return the right type
  static_assert(cuda::std::is_same<decltype(cuda::round_up(T(0), U(1))), CommonType>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::round_down(T(0), U(1))), CommonType>::value, "");

  assert(cuda::round_up(T(0), U(1)) == CommonType(0));
  assert(cuda::round_up(T(1), U(1)) == CommonType(1));
  assert(cuda::round_up(T(45), U(32)) == CommonType(64));
  // ensure that we are resilient against overflow
  assert(cuda::round_up(maxv, U(1)) == maxv);
  assert(cuda::round_up(maxv, maxv) == maxv);

  assert(cuda::round_down(T(0), U(1)) == CommonType(0));
  assert(cuda::round_down(T(1), U(1)) == CommonType(1));
  assert(cuda::round_down(T(78), U(64)) == CommonType(64));
  // ensure that we are resilient against overflow
  assert(cuda::round_down(maxv, U(1)) == maxv);
  assert(cuda::round_down(maxv, maxv) == maxv);
}

template <class T, class = void>
struct relaxed_underlying_type
{
  using type = T;
};

template <class T>
struct relaxed_underlying_type<T, cuda::std::void_t<cuda::std::underlying_type_t<T>>>
{
  using type = cuda::std::underlying_type_t<T>;
};

template <class T1, class U1>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test_enum()
{
  using T          = typename relaxed_underlying_type<T1>::type;
  using U          = typename relaxed_underlying_type<U1>::type;
  using CommonType = cuda::std::common_type_t<T, U>;
  // ensure that we return the right type
  static_assert(cuda::std::is_same<decltype(cuda::round_up(T(0), U(1))), CommonType>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::round_down(T(0), U(1))), CommonType>::value, "");

  assert(cuda::round_up(T(0), U(1)) == CommonType(0));
  assert(cuda::round_up(T(1), U(1)) == CommonType(1));
  assert(cuda::round_up(T(46), U(32)) == CommonType(64));

  assert(cuda::round_down(T(0), U(1)) == CommonType(0));
  assert(cuda::round_down(T(1), U(1)) == CommonType(1));
  assert(cuda::round_down(T(78), U(64)) == CommonType(64));
}

enum Enum1 : short
{
  E1,
  E2,
  E3 = 64,
  E4 = 78
};
enum class Enum2 : long
{
  C1,
  C2,
  C3 = 64,
  C4 = 78
};

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test()
{
  // Builtin integer types:
  test<T, char>();
  test<T, signed char>();
  test<T, unsigned char>();

  test<T, short>();
  test<T, unsigned short>();

  test<T, int>();
  test<T, unsigned int>();

  test<T, long>();
  test<T, unsigned long>();

  test<T, long long>();
  test<T, unsigned long long>();

  test_enum<T, Enum1>();
  test_enum<T, Enum2>();
  test_enum<Enum1, T>();
  test_enum<Enum2, T>();

#if !defined(TEST_COMPILER_NVRTC)
  // cstdint types:
  test<T, std::size_t>();
  test<T, std::ptrdiff_t>();
  test<T, std::intptr_t>();
  test<T, std::uintptr_t>();

  test<T, std::int8_t>();
  test<T, std::int16_t>();
  test<T, std::int32_t>();
  test<T, std::int64_t>();

  test<T, std::uint8_t>();
  test<T, std::uint16_t>();
  test<T, std::uint32_t>();
  test<T, std::uint64_t>();
#endif // !TEST_COMPILER_NVRTC

#if !defined(TEST_HAS_NO_INT128_T)
  test<T, __int128_t>();
  test<T, __uint128_t>();
#endif // !TEST_HAS_NO_INT128_T
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  // Builtin integer types:
  test<char>();
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

#if !defined(TEST_HAS_NO_INT128_T)
  test<__int128_t>();
  test<__uint128_t>();
#endif // !TEST_HAS_NO_INT128_T

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test(), "");
  return 0;
}
