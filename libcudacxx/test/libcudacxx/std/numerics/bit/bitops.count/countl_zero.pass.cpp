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
//   constexpr int countl_zero(T x) noexcept;

// Returns: The number of consecutive 0 bits, starting from the most significant bit.
//   [ Note: Returns N if x == 0. ]
//
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
  static_assert(cuda::std::countl_zero(T(0)) == cuda::std::numeric_limits<T>::digits, "");
  static_assert(cuda::std::countl_zero(T(1)) == cuda::std::numeric_limits<T>::digits - 1, "");
  static_assert(cuda::std::countl_zero(T(2)) == cuda::std::numeric_limits<T>::digits - 2, "");
  static_assert(cuda::std::countl_zero(T(3)) == cuda::std::numeric_limits<T>::digits - 2, "");
  static_assert(cuda::std::countl_zero(T(4)) == cuda::std::numeric_limits<T>::digits - 3, "");
  static_assert(cuda::std::countl_zero(T(5)) == cuda::std::numeric_limits<T>::digits - 3, "");
  static_assert(cuda::std::countl_zero(T(6)) == cuda::std::numeric_limits<T>::digits - 3, "");
  static_assert(cuda::std::countl_zero(T(7)) == cuda::std::numeric_limits<T>::digits - 3, "");
  static_assert(cuda::std::countl_zero(T(8)) == cuda::std::numeric_limits<T>::digits - 4, "");
  static_assert(cuda::std::countl_zero(T(9)) == cuda::std::numeric_limits<T>::digits - 4, "");
  static_assert(cuda::std::countl_zero(cuda::std::numeric_limits<T>::max()) == 0, "");

  return true;
}

template <typename T>
__host__ __device__ inline void assert_countl_zero(T val, int expected)
{
  volatile auto v = val;
  assert(cuda::std::countl_zero(v) == expected);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::countl_zero(T(0))));
  ASSERT_NOEXCEPT(cuda::std::countl_zero(T(0)));
  const int dig = cuda::std::numeric_limits<T>::digits;

  assert_countl_zero(T(121), dig - 7);
  assert_countl_zero(T(122), dig - 7);
  assert_countl_zero(T(123), dig - 7);
  assert_countl_zero(T(124), dig - 7);
  assert_countl_zero(T(125), dig - 7);
  assert_countl_zero(T(126), dig - 7);
  assert_countl_zero(T(127), dig - 7);
  assert_countl_zero(T(128), dig - 8);
  assert_countl_zero(T(129), dig - 8);
  assert_countl_zero(T(130), dig - 8);
}

int main(int, char**)
{
  constexpr_test<unsigned char>();
  constexpr_test<unsigned short>();
  constexpr_test<unsigned>();
  constexpr_test<unsigned long>();
  constexpr_test<unsigned long long>();

  constexpr_test<uint8_t>();
  constexpr_test<uint16_t>();
  constexpr_test<uint32_t>();
  constexpr_test<uint64_t>();
  constexpr_test<size_t>();
  constexpr_test<uintmax_t>();
  constexpr_test<uintptr_t>();

#ifndef _LIBCUDACXX_HAS_NO_INT128
  constexpr_test<__uint128_t>();
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
    const int dig   = cuda::std::numeric_limits<__uint128_t>::digits;
    __uint128_t val = 128;

    val <<= 32;
    assert(cuda::std::countl_zero(val - 1) == dig - 39);
    assert(cuda::std::countl_zero(val) == dig - 40);
    assert(cuda::std::countl_zero(val + 1) == dig - 40);
    val <<= 2;
    assert(cuda::std::countl_zero(val - 1) == dig - 41);
    assert(cuda::std::countl_zero(val) == dig - 42);
    assert(cuda::std::countl_zero(val + 1) == dig - 42);
    val <<= 3;
    assert(cuda::std::countl_zero(val - 1) == dig - 44);
    assert(cuda::std::countl_zero(val) == dig - 45);
    assert(cuda::std::countl_zero(val + 1) == dig - 45);
  }
#endif

  return 0;
}
