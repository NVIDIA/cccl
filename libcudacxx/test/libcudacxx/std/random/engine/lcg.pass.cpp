//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: dynamic allocation is not supported in tile mode

#include <cuda/std/random>

#include "random_utilities/test_engine.h"

#if _CCCL_HOSTED()
// The state is written in decimal and the format flags and fill character of the stream are unchanged
void test_stream_format()
{
  cuda::std::minstd_rand e(1000);
  (void) e();
  std::stringstream ss;
  ss.flags(std::ios_base::hex | std::ios_base::showbase);
  ss.fill('*');
  const auto flags = ss.flags();
  ss << e;
  assert(ss.str() == "48271000");
  assert(ss.flags() == flags);
  assert(ss.fill() == '*');

  cuda::std::minstd_rand e2;
  ss >> e2;
  assert(!ss.fail());
  assert(e == e2);
}
#endif // _CCCL_HOSTED()

TEST_HOST_DEVICE_FUNC void test()
{
  test_engine<cuda::std::minstd_rand0, 1043618065u>();
  test_engine<cuda::std::minstd_rand, 399268537ull>();
}

// Steps the engine and discards the same number of values, the values were obtained from libstdc++ and a big integer
// implementation of [rand.eng.lcong]
template <typename UIntType, UIntType A, UIntType C, UIntType M>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_sequence(UIntType seed, int z, UIntType value_after_z)
{
  using Engine = cuda::std::linear_congruential_engine<UIntType, A, C, M>;
  Engine e1(seed);
  Engine e2(seed);
  for (int i = 0; i < z; ++i)
  {
    (void) e1();
  }
  e2.discard(z);
  assert(e1 == e2);
  assert(e1() == value_after_z);
  assert(e2() == value_after_z);
  return true;
}

template <typename UIntType, UIntType A, UIntType C, UIntType M>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_seed_seq(UIntType first_value)
{
  cuda::std::seed_seq seq{1, 2, 3, 4, 5};
  cuda::std::linear_congruential_engine<UIntType, A, C, M> e(seq);
  assert(e() == first_value);
  return true;
}

// If the template parameter m is 0, the modulus is numeric_limits<result_type>::max() + 1
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_modulus_zero()
{
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  test_sequence<unsigned char, 5, 1, 0>(1, 1000, 222);
  test_sequence<uint16_t, 25173, 13849, 0>(1, 1000, 38470);
  test_sequence<uint32_t, 1103515245u, 12345u, 0>(1, 1000, 2582971166u);
  test_sequence<uint64_t, 6364136223846793005ull, 1442695040888963407ull, 0>(1, 1000, 610409228822633476ull);
  test_seed_seq<uint16_t, 25173, 13849, 0>(36654);
  test_seed_seq<uint32_t, 1103515245u, 12345u, 0>(3630829414u);
  test_seed_seq<uint64_t, 6364136223846793005ull, 1442695040888963407ull, 0>(1094635726686565209ull);
  return true;
}

// discard must not overflow when squaring the multiplier, even though a * (m - 1) + c fits
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_large_modulus()
{
  using cuda::std::uint64_t;
  test_sequence<uint64_t, 16807, 0, (1ull << 40) - 87>(1, 1000, 300634309918ull);
  test_sequence<uint64_t, 16807, 11, (1ull << 40) - 87>(1, 1000, 388956881613ull);
  test_seed_seq<uint64_t, 16807, 0, (1ull << 40) - 87>(1028154845247ull);

  // m = 2^n keeps the logarithmic discard, stepping 10^12 times would not finish
  test_sequence<uint64_t, 5, 1, (1ull << 40)>(1, 1000, 1085124145118ull);
  cuda::std::linear_congruential_engine<uint64_t, 5, 1, (1ull << 40)> e;
  e.discard(1000000000000ull);
  assert(e() == 725246242822ull);
  return true;
}

// a * x + c overflows and Schrage's algorithm does not apply, since m % a > m / a
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_promotion()
{
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  test_sequence<uint32_t, 1664525u, 1013904223u, 4294967291u>(1, 1, 1587973397u);
  test_sequence<uint32_t, 1664525u, 1013904223u, 4294967291u>(1, 1000, 3122718224u);
#if _CCCL_HAS_INT128()
  test_sequence<uint64_t, 2806196910506780709ull, 1, (1ull << 63) - 25>(1, 1, 598720870811554157ull);
  test_sequence<uint64_t, 2806196910506780709ull, 1, (1ull << 63) - 25>(1, 1000, 985508913947959965ull);
#endif // _CCCL_HAS_INT128()
  return true;
}

// Wrapping around is only correct if m is a power of two, not for any even m
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_even_modulus()
{
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  // Schrage's algorithm applies
  test_sequence<uint32_t, 3, 0, 4294967294u>(4294967293u, 0, 4294967291u);
  test_sequence<uint32_t, 3, 0, 4294967294u>(1, 1000, 2805970877u);
  test_sequence<uint64_t, 3, 0, ~0ull - 1>(~0ull - 2, 0, ~0ull - 4);
  test_sequence<uint64_t, 3, 0, ~0ull - 1>(1, 1000, 16423488893119839011ull);
  // Schrage's algorithm does not apply
  test_sequence<uint32_t, 1664525u, 1013904223u, 4294967294u>(1, 1, 1586792639u);
  test_sequence<uint32_t, 1664525u, 1013904223u, 4294967294u>(1, 1000, 2340327564u);
#if _CCCL_HAS_INT128()
  test_sequence<uint64_t, 2806196910506780709ull, 1, ~0ull - 1>(1, 1, 17001461636717115033ull);
  test_sequence<uint64_t, 2806196910506780709ull, 1, ~0ull - 1>(1, 1000, 17645910041383950886ull);
#endif // _CCCL_HAS_INT128()
  return true;
}

// P4037R1: linear_congruential_engine must accept the unsigned char UIntType
TEST_HOST_DEVICE_FUNC void test_p4037r1_small_uinttype()
{
  using E = cuda::std::linear_congruential_engine<unsigned char, 5u, 1u, 13u>;
  static_assert(cuda::std::is_same_v<E::result_type, unsigned char>);
  static_assert(E::min() == 0u);
  static_assert(E::max() == 12u);
  static_assert(E::multiplier == 5u);
  static_assert(E::increment == 1u);
  static_assert(E::modulus == 13u);
}

int main(int, char**)
{
  test();
  test_p4037r1_small_uinttype();
  NV_IF_TARGET(NV_IS_HOST, ({ test_stream_format(); }));
  test_modulus_zero();
  test_large_modulus();
  test_promotion();
  test_even_modulus();
#if TEST_STD_VER >= 2020
  static_assert(test_modulus_zero());
  static_assert(test_large_modulus());
  static_assert(test_promotion());
  static_assert(test_even_modulus());
#endif // TEST_STD_VER >= 2020
  return 0;
}
