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
//   constexpr int rotl(T x, unsigned int s) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//  T is an unsigned integer type

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
  return cuda::std::rotl(T(1), 0) == T(1) && cuda::std::rotl(T(1), 1) == T(2) && cuda::std::rotl(T(1), 2) == T(4)
      && cuda::std::rotl(T(1), 3) == T(8) && cuda::std::rotl(T(1), 4) == T(16) && cuda::std::rotl(T(1), 5) == T(32)
      && cuda::std::rotl(T(1), 6) == T(64) && cuda::std::rotl(T(1), 7) == T(128)
      && cuda::std::rotl(nl::max(), 0) == nl::max() && cuda::std::rotl(nl::max(), 1) == nl::max()
      && cuda::std::rotl(nl::max(), 2) == nl::max() && cuda::std::rotl(nl::max(), 3) == nl::max()
      && cuda::std::rotl(nl::max(), 4) == nl::max() && cuda::std::rotl(nl::max(), 5) == nl::max()
      && cuda::std::rotl(nl::max(), 6) == nl::max() && cuda::std::rotl(nl::max(), 7) == nl::max();
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(T, decltype(cuda::std::rotl(T(0), 0)));
  ASSERT_NOEXCEPT(cuda::std::rotl(T(0), 0));
  const T val = cuda::std::numeric_limits<T>::max() - 1;

  assert(cuda::std::rotl(val, 0) == val);
  assert(cuda::std::rotl(val, 1) == static_cast<T>((val << 1) + 1));
  assert(cuda::std::rotl(val, 2) == static_cast<T>((val << 2) + 3));
  assert(cuda::std::rotl(val, 3) == static_cast<T>((val << 3) + 7));
  assert(cuda::std::rotl(val, 4) == static_cast<T>((val << 4) + 15));
  assert(cuda::std::rotl(val, 5) == static_cast<T>((val << 5) + 31));
  assert(cuda::std::rotl(val, 6) == static_cast<T>((val << 6) + 63));
  assert(cuda::std::rotl(val, 7) == static_cast<T>((val << 7) + 127));
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
  runtime_test<unsigned short>();
  runtime_test<unsigned>();
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
    __uint128_t val = 168; // 0xA8 (aka 10101000)

    assert(cuda::std::rotl(val, 128) == 168);
    val <<= 32;
    assert(cuda::std::rotl(val, 96) == 168);
    val <<= 2;
    assert(cuda::std::rotl(val, 95) == 336);
    val <<= 3;
    assert(cuda::std::rotl(val, 90) == 84);
    assert(cuda::std::rotl(val, 218) == 84);
  }
#endif

  return 0;
}
