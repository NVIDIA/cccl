//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool test()
{
  using nl              = cuda::std::numeric_limits<T>;
  constexpr T all_ones  = static_cast<T>(~T{0});
  constexpr T half_low  = all_ones >> (nl::digits / 2u);
  constexpr T half_high = static_cast<T>(all_ones << (nl::digits / 2u));
  static_assert(cuda::bit_reverse(all_ones) == all_ones);
  static_assert(cuda::bit_reverse(T{0}) == T{0});
  static_assert(cuda::bit_reverse(half_low) == half_high);
  static_assert(cuda::bit_reverse(T{0b11001001}) == (T{0b10010011} << (nl::digits - 8u)));
  static_assert(cuda::bit_reverse(T{T{0b10010011} << (nl::digits - 8u)}) == T{0b11001001});
  unused(all_ones);
  unused(half_low);
  unused(half_high);
  return true;
}

__host__ __device__ constexpr bool test()
{
  test<unsigned char>();
  test<unsigned short>();
  test<unsigned>();
  test<unsigned long>();
  test<unsigned long long>();

  test<uint8_t>();
  test<uint16_t>();
  test<uint32_t>();
  test<uint64_t>();
  test<size_t>();
  test<uintmax_t>();
  test<uintptr_t>();

#if _CCCL_HAS_INT128()
  test<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  assert(test());
  static_assert(test());
  return 0;
}
