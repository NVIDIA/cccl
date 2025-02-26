//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
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
  using nl             = cuda::std::numeric_limits<T>;
  constexpr auto max_v = nl::max();
  static_assert(cuda::bitfield_insert(T{0}, 0) == 1);
  static_assert(cuda::bitfield_insert(T{0}, 1) == 0b10);
  static_assert(cuda::bitfield_insert(T{0b10}, 0) == 0b11);
  static_assert(cuda::bitfield_insert(max_v, 0) == max_v);
  static_assert(cuda::bitfield_insert(max_v, 2) == max_v);

  static_assert(cuda::bitfield_insert(T{0}, 0, 2) == 0b11);
  static_assert(cuda::bitfield_insert(T{0}, 3, 2) == 0b11000);
  static_assert(cuda::bitfield_insert(T{0b10100000}, 3, 2) == 0b10111000);
  static_assert(cuda::bitfield_insert(T{0}, nl::digits - 1, 1) == (T{1} << (nl::digits - 1u)));

  static_assert(cuda::bitfield_extract(T{0}, 3, 4) == 0);
  static_assert(cuda::bitfield_extract(T{0b1011}, 0) == 0b001);
  static_assert(cuda::bitfield_extract(T{0b1011}, 1) == 0b010);
  static_assert(cuda::bitfield_extract(T{0b1011}, 2) == 0);
  static_assert(cuda::bitfield_extract(max_v, 0, 4) == 0b1111);
  static_assert(cuda::bitfield_extract(max_v, 2, 4) == 0b111100);

  static_assert(cuda::bitfield_extract(T{0b1010010}, 0, 2) == 0b10);
  static_assert(cuda::bitfield_extract(T{0b10101100}, 3, 2) == 0b01000);
  static_assert(cuda::bitfield_extract(T{0b10100000}, 3, 3) == 0b00100000);

  static_assert(cuda::bitfield_extract(T{max_v}, nl::digits - 1, 1) == (T{1} << (nl::digits - 1u)));
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
