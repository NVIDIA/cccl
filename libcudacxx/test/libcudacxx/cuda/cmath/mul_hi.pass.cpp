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

_CCCL_DIAG_SUPPRESS_NVCC(23)

template <typename T>
__host__ __device__ constexpr void test_type()
{
  static_assert(cuda::std::is_same_v<T, decltype(cuda::mul_hi(T{}, T{}))>);
  static_assert(noexcept(cuda::mul_hi(T{}, T{})));
  constexpr int bits       = static_cast<int>(sizeof(T) * CHAR_BIT);
  constexpr auto max_value = cuda::std::numeric_limits<T>::max();
  using U                  = cuda::std::make_unsigned_t<T>;

  // trivial cases
  assert(cuda::mul_hi(T{0}, T{0}) == T{0});
  assert(cuda::mul_hi(max_value, T{0}) == T{0});
  assert(cuda::mul_hi(max_value, T{1}) == T{0});
  assert(cuda::mul_hi(T{0}, max_value) == T{0});
  assert(cuda::mul_hi(T{1}, max_value) == T{0});
  if constexpr (cuda::std::is_signed_v<T>)
  {
    assert(cuda::mul_hi(T{-1}, T{0}) == T{0});
    assert(cuda::mul_hi(T{-1}, T{-1}) == T{0});
    assert(cuda::mul_hi(T{-1}, T{1}) == T{-1});
    assert(cuda::mul_hi(T{1}, T{-1}) == T{-1});
  }

  // non-trivial cases
  constexpr auto mask1 = T{T{0x7B} << (bits - 8)};
  assert(cuda::mul_hi(mask1, T{16}) == T{0x7});
  assert(cuda::mul_hi(T{16}, mask1) == T{0x7});

  constexpr auto mask2 = static_cast<U>(-1);
  assert(cuda::mul_hi(mask2, U{64}) == mask2 >> (bits - 6));
  assert(cuda::mul_hi(U{64}, mask2) == mask2 >> (bits - 6));

  // power of two cases
  if constexpr (sizeof(T) >= 2)
  {
    constexpr auto mask3 = static_cast<T>(0x7BCD);
    assert(cuda::mul_hi(mask3, T{4096}) == (mask3 >> (bits - 12)));
    assert(cuda::mul_hi(T{4096}, mask3) == (mask3 >> (bits - 12)));
  }
  if constexpr (sizeof(T) >= 4)
  {
    constexpr auto mask3 = static_cast<T>(0x7ABCABCD);
    assert(cuda::mul_hi(mask3, T{1 << 24}) == mask3 >> (bits - 24));
    assert(cuda::mul_hi(T{1 << 24}, mask3) == mask3 >> (bits - 24));
  }
  if constexpr (sizeof(T) >= 8)
  {
    constexpr auto mask3 = static_cast<T>(0x7ABCABCD12345678);
    assert(cuda::mul_hi(mask3, T{1} << 48) == mask3 >> (bits - 48));
    assert(cuda::mul_hi(T{1} << 48, mask3) == mask3 >> (bits - 48));
  }
#if _CCCL_HAS_INT128()
  if constexpr (sizeof(T) == 16)
  {
    using namespace test_integer_literals;
    constexpr auto mask4 = T{0x7ABCABCD12345678_i128};
    assert(cuda::mul_hi(mask4, T{1} << 96) == mask4 >> (bits - 96));
    assert(cuda::mul_hi(T{1} << 96, mask4) == mask4 >> (bits - 96));
  }
#endif // _CCCL_HAS_INT128()

  // random numbers cases
  if constexpr (sizeof(T) == 2)
  {
    assert(cuda::mul_hi(T{23058}, T{31852}) == T{0x2BC6});
    if constexpr (cuda::std::is_same_v<T, short>)
    {
      assert(cuda::mul_hi(T{-23058}, T{-31852}) == T{0x2BC6});
      assert(cuda::mul_hi(T{23058}, T{-31852}) == T{-0x2BC7});
      assert(cuda::mul_hi(T{-23058}, T{31852}) == T{-0x2BC7});
    }
  }
  else if constexpr (sizeof(T) == 4)
  {
    assert(cuda::mul_hi(T{2305878}, T{31852876}) == T{0x42CD});
    if constexpr (cuda::std::is_signed_v<T>)
    {
      assert(cuda::mul_hi(T{-2305878}, T{-31852876}) == T{0x42CD});
      assert(cuda::mul_hi(T{-2305878}, T{31852876}) == T{-0x42CE});
      assert(cuda::mul_hi(T{2305878}, T{-31852876}) == T{-0x42CE});
    }
  }
  else if constexpr (sizeof(T) == 8)
  {
    assert(cuda::mul_hi(T{2'305'878'454'534}, T{31'852'876'355'863}) == T{0x3CC166});
    if constexpr (cuda::std::is_signed_v<T>)
    {
      assert(cuda::mul_hi(T{-2'305'878'454'534}, T{-31'852'876'355'863}) == T{0x3CC166});
      assert(cuda::mul_hi(T{-2'305'878'454'534}, T{31'852'876'355'863}) == T{-0x3CC167});
      assert(cuda::mul_hi(T{2'305'878'454'534}, T{-31'852'876'355'863}) == T{-0x3CC167});
    }
  }
#if _CCCL_HAS_INT128()
  if constexpr (sizeof(T) == 16)
  {
    using namespace test_integer_literals;
    assert(cuda::mul_hi(T{204'446'744'073'709'551'616_i128}, T{48'433'654'723'709'533'982_i128}) == T{29});
    if constexpr (cuda::std::is_signed_v<T>)
    {
      assert(cuda::mul_hi(T{-204'446'744'073'709'551'616_i128}, T{-48'433'654'723'709'533'982_i128}) == T{29});
      assert(cuda::mul_hi(T{204'446'744'073'709'551'616_i128}, T{-48'433'654'723'709'533'982_i128}) == T{-30});
      assert(cuda::mul_hi(T{-204'446'744'073'709'551'616_i128}, T{48'433'654'723'709'533'982_i128}) == T{-30});
    }
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
