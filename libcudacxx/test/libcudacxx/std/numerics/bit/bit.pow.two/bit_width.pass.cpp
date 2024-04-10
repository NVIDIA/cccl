//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++98, c++03

// template <class T>
//   constexpr T log2p1(T x) noexcept;

// If x == 0, 0; otherwise one plus the base-2 logarithm of x, with any fractional part discarded.

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

class A
{};
enum E1 : unsigned char
{
  rEd
};
enum class E2 : unsigned char
{
  red
};

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
  return cuda::std::bit_width(T(0)) == T(0) && cuda::std::bit_width(T(1)) == T(1) && cuda::std::bit_width(T(2)) == T(2)
      && cuda::std::bit_width(T(3)) == T(2) && cuda::std::bit_width(T(4)) == T(3) && cuda::std::bit_width(T(5)) == T(3)
      && cuda::std::bit_width(T(6)) == T(3) && cuda::std::bit_width(T(7)) == T(3) && cuda::std::bit_width(T(8)) == T(4)
      && cuda::std::bit_width(T(9)) == T(4);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(T, decltype(cuda::std::bit_width(T(0))));
  ASSERT_NOEXCEPT(cuda::std::bit_width(T(0)));

  assert(cuda::std::bit_width(T(0)) == T(0));
  assert(cuda::std::bit_width(T(1)) == T(1));
  assert(cuda::std::bit_width(T(2)) == T(2));
  assert(cuda::std::bit_width(T(3)) == T(2));
  assert(cuda::std::bit_width(T(4)) == T(3));
  assert(cuda::std::bit_width(T(5)) == T(3));
  assert(cuda::std::bit_width(T(6)) == T(3));
  assert(cuda::std::bit_width(T(7)) == T(3));
  assert(cuda::std::bit_width(T(8)) == T(4));
  assert(cuda::std::bit_width(T(9)) == T(4));

  assert(cuda::std::bit_width(T(121)) == T(7));
  assert(cuda::std::bit_width(T(122)) == T(7));
  assert(cuda::std::bit_width(T(123)) == T(7));
  assert(cuda::std::bit_width(T(124)) == T(7));
  assert(cuda::std::bit_width(T(125)) == T(7));
  assert(cuda::std::bit_width(T(126)) == T(7));
  assert(cuda::std::bit_width(T(127)) == T(7));
  assert(cuda::std::bit_width(T(128)) == T(8));
  assert(cuda::std::bit_width(T(129)) == T(8));
  assert(cuda::std::bit_width(T(130)) == T(8));
}

int main(int, char**)
{
  static_assert(constexpr_test<unsigned char>(), "");
  static_assert(constexpr_test<unsigned short>(), "");
  static_assert(constexpr_test<unsigned>(), "");
  static_assert(constexpr_test<unsigned long>(), "");
  static_assert(constexpr_test<unsigned long long>(), "");

  static_assert(constexpr_test<uint8_t>(), "");
  static_assert(constexpr_test<uint16_t>(), "");
  static_assert(constexpr_test<uint32_t>(), "");
  static_assert(constexpr_test<uint64_t>(), "");
  static_assert(constexpr_test<size_t>(), "");
  static_assert(constexpr_test<uintmax_t>(), "");
  static_assert(constexpr_test<uintptr_t>(), "");

#ifndef _LIBCUDACXX_HAS_NO_INT128
  static_assert(constexpr_test<__uint128_t>(), "");
#endif

  runtime_test<unsigned char>();
  runtime_test<unsigned>();
  runtime_test<unsigned short>();
  runtime_test<unsigned long>();
  runtime_test<unsigned long long>();

  runtime_test<uint8_t>();
  runtime_test<uint16_t>();
  runtime_test<uint32_t>();
  runtime_test<uint64_t>();
  runtime_test<size_t>();
  runtime_test<uintmax_t>();
  runtime_test<uintptr_t>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
  runtime_test<__uint128_t>();

  {
    __uint128_t val = 128;
    val <<= 32;
    assert(cuda::std::bit_width(val - 1) == 39);
    assert(cuda::std::bit_width(val) == 40);
    assert(cuda::std::bit_width(val + 1) == 40);
    val <<= 2;
    assert(cuda::std::bit_width(val - 1) == 41);
    assert(cuda::std::bit_width(val) == 42);
    assert(cuda::std::bit_width(val + 1) == 42);
    val <<= 3;
    assert(cuda::std::bit_width(val - 1) == 44);
    assert(cuda::std::bit_width(val) == 45);
    assert(cuda::std::bit_width(val + 1) == 45);
  }
#endif

  return 0;
}
