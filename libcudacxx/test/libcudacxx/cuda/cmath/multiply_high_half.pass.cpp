//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/cassert>
#include <cuda/std/climits>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "literal.h"

template <typename T>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::multiply_half_high(T{}, T{}))>);
  static_assert(noexcept(cuda::multiply_half_high(T{}, T{})));
  constexpr int bits       = static_cast<int>(sizeof(T) * CHAR_BIT);
  constexpr auto max_value = cuda::std::numeric_limits<T>::max();
  using U                  = cuda::std::make_unsigned_t<T>;

  // trivial cases
  assert(cuda::multiply_half_high(T{0}, T{0}) == T{0});
  assert(cuda::multiply_half_high(max_value, T{0}) == T{0});
  assert(cuda::multiply_half_high(max_value, T{1}) == T{0});
  assert(cuda::multiply_half_high(T{0}, max_value) == T{0});
  assert(cuda::multiply_half_high(T{1}, max_value) == T{0});

  // non-trivial cases
  constexpr auto mask1 = T{T{0x7B} << (bits - 8)};
  assert(cuda::multiply_half_high(mask1, T{16}) == T{0x7});
  assert(cuda::multiply_half_high(T{16}, mask1) == T{0x7});

  constexpr auto mask2 = static_cast<U>(-1);
  assert(cuda::multiply_half_high(mask2, U{64}) == mask2 >> (bits - 6));
  assert(cuda::multiply_half_high(U{64}, mask2) == mask2 >> (bits - 6));

  if constexpr (sizeof(T) >= 2)
  {
    constexpr auto mask3 = static_cast<T>(0x7BCD);
    assert(cuda::multiply_half_high(mask3, T{4096}) == (mask3 >> (bits - 12)));
    assert(cuda::multiply_half_high(T{4096}, mask3) == (mask3 >> (bits - 12)));
  }
  if constexpr (sizeof(T) >= 4)
  {
    constexpr auto mask3 = static_cast<T>(0x7ABCABCD);
    assert(cuda::multiply_half_high(mask3, T{1 << 24}) == mask3 >> (bits - 24));
    assert(cuda::multiply_half_high(T{1 << 24}, mask3) == mask3 >> (bits - 24));
  }
  if constexpr (sizeof(T) >= 8)
  {
    constexpr auto mask3 = static_cast<T>(0x7ABCABCD12345678);
    assert(cuda::multiply_half_high(mask3, T{1} << 48) == mask3 >> (bits - 48));
    assert(cuda::multiply_half_high(T{1} << 48, mask3) == mask3 >> (bits - 48));
  }
#if _CCCL_HAS_INT128()
  if constexpr (sizeof(T) == 16)
  {
    using namespace test_integer_literals;
    constexpr auto mask4 = T{0x7ABCABCD12345678_i128};
    assert(cuda::multiply_half_high(mask4, T{1} << 96) == mask4 >> (bits - 96));
    assert(cuda::multiply_half_high(T{1} << 96, mask4) == mask4 >> (bits - 96));
  }
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_type<unsigned char>();
  test_type<unsigned short>();
  test_type<unsigned int>();
  test_type<unsigned long>();
  test_type<unsigned long long>();

  test_type<signed char>();
  test_type<short>();
  test_type<int>();
  test_type<long>();
  test_type<long long>();

#if _CCCL_HAS_INT128()
  test_type<__uint128_t>();
  test_type<__int128_t>();
#endif // _CCCL_HAS_INT128()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
