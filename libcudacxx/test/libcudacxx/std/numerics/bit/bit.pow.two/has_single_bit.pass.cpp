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
//   constexpr bool has_single_bit(T x) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//	T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

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
  return cuda::std::has_single_bit(T(1)) && cuda::std::has_single_bit(T(2)) && !cuda::std::has_single_bit(T(3))
      && cuda::std::has_single_bit(T(4)) && !cuda::std::has_single_bit(T(5)) && !cuda::std::has_single_bit(T(6))
      && !cuda::std::has_single_bit(T(7)) && cuda::std::has_single_bit(T(8)) && !cuda::std::has_single_bit(T(9));
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(bool, decltype(cuda::std::has_single_bit(T(0))));
  ASSERT_NOEXCEPT(cuda::std::has_single_bit(T(0)));

  assert(!cuda::std::has_single_bit(T(121)));
  assert(!cuda::std::has_single_bit(T(122)));
  assert(!cuda::std::has_single_bit(T(123)));
  assert(!cuda::std::has_single_bit(T(124)));
  assert(!cuda::std::has_single_bit(T(125)));
  assert(!cuda::std::has_single_bit(T(126)));
  assert(!cuda::std::has_single_bit(T(127)));
  assert(cuda::std::has_single_bit(T(128)));
  assert(!cuda::std::has_single_bit(T(129)));
  assert(!cuda::std::has_single_bit(T(130)));
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
    assert(!cuda::std::has_single_bit(val - 1));
    assert(cuda::std::has_single_bit(val));
    assert(!cuda::std::has_single_bit(val + 1));
    val <<= 2;
    assert(!cuda::std::has_single_bit(val - 1));
    assert(cuda::std::has_single_bit(val));
    assert(!cuda::std::has_single_bit(val + 1));
    val <<= 3;
    assert(!cuda::std::has_single_bit(val - 1));
    assert(cuda::std::has_single_bit(val));
    assert(!cuda::std::has_single_bit(val + 1));
  }
#endif

  return 0;
}
