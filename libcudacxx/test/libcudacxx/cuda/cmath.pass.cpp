//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/utility>

#include "test_macros.h"

#if !TEST_COMPILER(NVRTC)
#  include <cstdint>
#endif // !TEST_COMPILER(NVRTC)

template <class T, class U>
__host__ __device__ constexpr void test()
{
  constexpr T maxv = cuda::std::numeric_limits<T>::max();

  // ensure that we return the right type
  using Common = ::cuda::std::common_type_t<T, U>;
  static_assert(cuda::std::is_same<decltype(cuda::ceil_div(T(0), U(1))), Common>::value);
  assert(cuda::ceil_div(T(0), U(1)) == Common(0));
  assert(cuda::ceil_div(T(1), U(1)) == Common(1));
  assert(cuda::ceil_div(T(126), U(64)) == Common(2));

  // ensure that we are resilient against overflow
  assert(cuda::ceil_div(maxv, U(1)) == maxv);
  assert(cuda::ceil_div(maxv, maxv) == Common(1));
}

template <class T>
__host__ __device__ constexpr void test()
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

#if !TEST_COMPILER(NVRTC)
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
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
  test<T, __int128_t>();
  test<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
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

#if !TEST_COMPILER(NVRTC)
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
#endif // !TEST_COMPILER(NVRTC)

#if _CCCL_HAS_INT128()
  test<__int128_t>();
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test(), "");
  return 0;
}
