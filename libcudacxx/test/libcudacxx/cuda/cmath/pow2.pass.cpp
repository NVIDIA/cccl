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
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename R>
__host__ __device__ void constexpr test()
{
  using T = cuda::std::make_unsigned_t<R>;
  static_assert(cuda::std::is_same_v<R, decltype(cuda::next_power_of_two(R(0)))>);
  static_assert(cuda::std::is_same_v<R, decltype(cuda::prev_power_of_two(R(0)))>);
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::is_power_of_two(R(0)))>);
  static_assert(noexcept(cuda::next_power_of_two(R(0))));
  static_assert(noexcept(cuda::prev_power_of_two(R(0))));
  static_assert(noexcept(cuda::is_power_of_two(R(0))));
  assert(static_cast<R>(cuda::std::bit_ceil(T(0))) == cuda::next_power_of_two(R{0}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(1))) == cuda::next_power_of_two(R{1}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(2))) == cuda::next_power_of_two(R{2}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(3))) == cuda::next_power_of_two(R{3}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(4))) == cuda::next_power_of_two(R{4}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(5))) == cuda::next_power_of_two(R{5}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(10))) == cuda::next_power_of_two(R{10}));
  assert(static_cast<R>(cuda::std::bit_ceil(T(16))) == cuda::next_power_of_two(R{16}));

  assert(static_cast<R>(cuda::std::bit_floor(T(0))) == cuda::prev_power_of_two(R{0}));
  assert(static_cast<R>(cuda::std::bit_floor(T(1))) == cuda::prev_power_of_two(R{1}));
  assert(static_cast<R>(cuda::std::bit_floor(T(2))) == cuda::prev_power_of_two(R{2}));
  assert(static_cast<R>(cuda::std::bit_floor(T(3))) == cuda::prev_power_of_two(R{3}));
  assert(static_cast<R>(cuda::std::bit_floor(T(4))) == cuda::prev_power_of_two(R{4}));
  assert(static_cast<R>(cuda::std::bit_floor(T(5))) == cuda::prev_power_of_two(R{5}));
  assert(static_cast<R>(cuda::std::bit_floor(T(10))) == cuda::prev_power_of_two(R{10}));
  assert(static_cast<R>(cuda::std::bit_floor(T(16))) == cuda::prev_power_of_two(R{16}));

  assert(cuda::std::has_single_bit(T(0)) == cuda::is_power_of_two(R{0}));
  assert(cuda::std::has_single_bit(T(1)) == cuda::is_power_of_two(R{1}));
  assert(cuda::std::has_single_bit(T(2)) == cuda::is_power_of_two(R{2}));
  assert(cuda::std::has_single_bit(T(3)) == cuda::is_power_of_two(R{3}));
  assert(cuda::std::has_single_bit(T(4)) == cuda::is_power_of_two(R{4}));
  assert(cuda::std::has_single_bit(T(5)) == cuda::is_power_of_two(R{5}));
  assert(cuda::std::has_single_bit(T(10)) == cuda::is_power_of_two(R{10}));
  assert(cuda::std::has_single_bit(T(16)) == cuda::is_power_of_two(R{16}));
}

__host__ __device__ bool constexpr test()
{
  test<signed char>();
  test<int>();
  test<short>();
  test<long>();
  test<long long>();

  test<unsigned char>();
  test<unsigned>();
  test<unsigned short>();
  test<unsigned long>();
  test<unsigned long long>();

  test<int8_t>();
  test<int16_t>();
  test<int32_t>();
  test<int64_t>();
  test<intmax_t>();
  test<ptrdiff_t>();
#if _CCCL_HAS_INT128()
  test<__int128_t>();
#endif // _CCCL_HAS_INT128()

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
  test();
  static_assert(test());
  return 0;
}
