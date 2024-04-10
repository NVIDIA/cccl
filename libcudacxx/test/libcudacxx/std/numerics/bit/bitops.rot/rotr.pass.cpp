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
//   constexpr int rotr(T x, unsigned int s) noexcept;

// Remarks: This function shall not participate in overload resolution unless
//  T is an unsigned integer type

#include <cuda/std/bit>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename T>
__host__ __device__ constexpr bool constexpr_test()
{
  using nl = cuda::std::numeric_limits<T>;

  return cuda::std::rotr(T(128), 0) == T(128) && cuda::std::rotr(T(128), 1) == T(64)
      && cuda::std::rotr(T(128), 2) == T(32) && cuda::std::rotr(T(128), 3) == T(16)
      && cuda::std::rotr(T(128), 4) == T(8) && cuda::std::rotr(T(128), 5) == T(4) && cuda::std::rotr(T(128), 6) == T(2)
      && cuda::std::rotr(T(128), 7) == T(1) && cuda::std::rotr(nl::max(), 0) == nl::max()
      && cuda::std::rotr(nl::max(), 1) == nl::max() && cuda::std::rotr(nl::max(), 2) == nl::max()
      && cuda::std::rotr(nl::max(), 3) == nl::max() && cuda::std::rotr(nl::max(), 4) == nl::max()
      && cuda::std::rotr(nl::max(), 5) == nl::max() && cuda::std::rotr(nl::max(), 6) == nl::max()
      && cuda::std::rotr(nl::max(), 7) == nl::max();
}

template <typename T>
__host__ __device__ void runtime_test()
{
  ASSERT_SAME_TYPE(T, decltype(cuda::std::rotr(T(0), 0)));
  ASSERT_NOEXCEPT(cuda::std::rotr(T(0), 0));
  const T max = cuda::std::numeric_limits<T>::max();
  const T val = cuda::std::numeric_limits<T>::max() - 1;

  const T uppers[] = {
    max, // not used
    max - max, // 000 .. 0
    max - (max >> 1), // 800 .. 0
    max - (max >> 2), // C00 .. 0
    max - (max >> 3), // E00 .. 0
    max - (max >> 4), // F00 .. 0
    max - (max >> 5), // F80 .. 0
    max - (max >> 6), // FC0 .. 0
    max - (max >> 7), // FE0 .. 0
  };

  assert(cuda::std::rotr(val, 0) == val);
  assert(cuda::std::rotr(val, 1) == T((val >> 1) + uppers[1]));
  assert(cuda::std::rotr(val, 2) == T((val >> 2) + uppers[2]));
  assert(cuda::std::rotr(val, 3) == T((val >> 3) + uppers[3]));
  assert(cuda::std::rotr(val, 4) == T((val >> 4) + uppers[4]));
  assert(cuda::std::rotr(val, 5) == T((val >> 5) + uppers[5]));
  assert(cuda::std::rotr(val, 6) == T((val >> 6) + uppers[6]));
  assert(cuda::std::rotr(val, 7) == T((val >> 7) + uppers[7]));
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
    __uint128_t val = 168; // 0xA8 (aka 10101000)

    assert(cuda::std::rotr(val, 128) == 168);
    val <<= 32;
    assert(cuda::std::rotr(val, 32) == 168);
    val <<= 2;
    assert(cuda::std::rotr(val, 33) == 336);
    val <<= 3;
    assert(cuda::std::rotr(val, 38) == 84);
    assert(cuda::std::rotr(val, 166) == 84);
  }
#endif

  return 0;
}
