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
//   constexpr int countl_one(T x) noexcept;

// The number of consecutive 1 bits, starting from the most significant bit.
//   [ Note: Returns N if x == cuda::std::numeric_limits<T>::max(). ]
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
  using nl = cuda::std::numeric_limits<T>;

  static_assert(cuda::std::countl_one(nl::max()) == nl::digits, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 1)) == nl::digits - 1, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 2)) == nl::digits - 2, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 3)) == nl::digits - 2, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 4)) == nl::digits - 3, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 5)) == nl::digits - 3, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 6)) == nl::digits - 3, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 7)) == nl::digits - 3, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 8)) == nl::digits - 4, "");
  static_assert(cuda::std::countl_one(T(nl::max() - 9)) == nl::digits - 4, "");

  return true;
}

template <typename T>
__host__ __device__ inline void assert_countl_one(T val, int expected)
{
  volatile auto v = val;
  assert(cuda::std::countl_one(v) == expected);
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(int, decltype(cuda::std::countl_one(T(0))));
  ASSERT_NOEXCEPT(cuda::std::countl_one(T(0)));
  const int dig = cuda::std::numeric_limits<T>::digits;

  assert_countl_one(T(~121), dig - 7);
  assert_countl_one(T(~122), dig - 7);
  assert_countl_one(T(~123), dig - 7);
  assert_countl_one(T(~124), dig - 7);
  assert_countl_one(T(~125), dig - 7);
  assert_countl_one(T(~126), dig - 7);
  assert_countl_one(T(~127), dig - 7);
  assert_countl_one(T(~128), dig - 8);
  assert_countl_one(T(~129), dig - 8);
  assert_countl_one(T(~130), dig - 8);
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
    assert(cuda::std::countl_one(~val) == dig - 40);
    val <<= 2;
    assert(cuda::std::countl_one(~val) == dig - 42);
    val <<= 3;
    assert(cuda::std::countl_one(~val) == dig - 45);
  }
#endif

  return 0;
}
