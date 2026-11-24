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

template <typename Engine>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_set_counter()
{
  Engine e1(7);
  Engine e2(7);
  e1.discard(100);
  static_assert(::cuda::std::is_void_v<decltype(e2.set_counter({0, 0, 0, 100}))>);
  e2.set_counter({0, 0, 0, 100 / 4});
  assert(e1 == e2);

  // Set counter can go "back" to where it was before
  e1 = Engine(7);
  e2.set_counter({0, 0, 0, 0});
  assert(e1 == e2);
  // Overflow the counter
  e1 = Engine(7);
  e2 = Engine(7);
  e1.set_counter({0, 0, 1, 0});
  if constexpr (::cuda::std::is_same_v<Engine, typename ::cuda::std::philox4x32>)
  {
    e2.set_counter({0, 0, 0, ::cuda::std::numeric_limits<::cuda::std::uint32_t>::max()});
  }
  else
  {
    e2.set_counter({0, 0, 0, ::cuda::std::numeric_limits<::cuda::std::uint64_t>::max()});
  }
  e2.discard(4);
  assert(e1 == e2);
  return true;
}

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_against_reference()
{
  // reference values obtained from other standard library implementations
  const int seeds[]                               = {10823018, 0, 23};
  const int discards[]                            = {0, 5, 100};
  const cuda::std::uint64_t reference_values_64[] = {
    597860052874975753ull,
    16480731654955167298ull,
    4676222276634405366ull,
    1609277786247541068ull,
    4455796210202625458ull,
    12591023382997339072ull,
    13350274857560636235ull,
    5430341342746607840ull,
    4983921884708484958ull};
  const cuda::std::uint32_t reference_values_32[] = {
    1514423753u, 254961463u, 4167151386u, 1713891541u, 1555169499u, 444026393u, 1368340107u, 2016696101u, 1090885419u};

  int ref_index = 0;
  for (auto seed : seeds)
  {
    for (auto discard : discards)
    {
      cuda::std::philox4x64 rng(seed);
      cuda::std::philox4x32 rng32(seed);
      rng.discard(discard);
      rng32.discard(discard);
      assert(rng() == reference_values_64[ref_index]);
      assert(rng32() == reference_values_32[ref_index]);
      ref_index++;
    }
  }
  return true;
}

// discard must carry into the next counter word when the word of the increment plus the carry is 2^w. Starting from
// counter 2^w - 1, discarding n * (2^(2w) - 2^w + 1) values gives counter 2^(2w), which is exactly {0, 1, 0, 0}.
template <typename UIntType, cuda::std::size_t W, UIntType M0, UIntType C0, UIntType M1, UIntType C1>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_discard_carry(unsigned long long z, UIntType next)
{
  using Engine = cuda::std::philox_engine<UIntType, W, 4, 10, M0, C0, M1, C1>;
  Engine e1;
  e1.set_counter({0, 0, 0, Engine::max()});
  e1.discard(z);
  Engine e2;
  e2.set_counter({0, 1, 0, 0});
  assert(e1 == e2);
  assert(e1() == next);
  return true;
}

// The sequence is defined by w: mulhi(a, b, w) is floor(a * b / 2^w), whatever the width of UIntType is.
template <typename UIntType, cuda::std::size_t W, cuda::std::size_t N, UIntType... Constants>
TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_word_size(UIntType value_3, UIntType value_10000)
{
  using Engine = cuda::std::philox_engine<UIntType, W, N, 10, Constants...>;
  Engine e;
  e.discard(2);
  assert(e() == value_3);
  e = Engine();
  e.discard(9999);
  assert(e() == value_10000);
  return true;
}

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_word_sizes()
{
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  using cuda::std::uint64_t;
  using cuda::std::uint8_t;
  // reference values obtained from a big integer implementation of [rand.eng.philox] and libstdc++
  test_word_size<uint8_t, 8, 4, 0xD2, 0x9E, 0xCA, 0xBB>(8, 72);
  test_word_size<uint32_t, 8, 4, 0xD2, 0x9E, 0xCA, 0xBB>(8, 72);
  test_word_size<uint16_t, 12, 4, 0xD25, 0x9E3, 0xCA5, 0xBB6>(3971, 1747);
  test_word_size<uint32_t, 12, 4, 0xD25, 0x9E3, 0xCA5, 0xBB6>(3971, 1747);
  test_word_size<uint16_t, 15, 4, 0x5256, 0x1E37, 0x4A50, 0x3B67>(26842, 6240);
  test_word_size<uint64_t, 15, 4, 0x5256, 0x1E37, 0x4A50, 0x3B67>(26842, 6240);
  test_word_size<uint16_t, 16, 4, 0xD256, 0x9E37, 0xCA50, 0xBB67>(26065, 19920);
  test_word_size<uint32_t, 16, 4, 0xD256, 0x9E37, 0xCA50, 0xBB67>(26065, 19920);
  test_word_size<uint64_t, 16, 4, 0xD256, 0x9E37, 0xCA50, 0xBB67>(26065, 19920);
  test_word_size<uint32_t, 16, 2, 0xD256, 0x9E37>(57017, 34504);
  test_word_size<uint32_t, 24, 4, 0xD25651, 0x9E3779, 0xCA5A82, 0xBB67AE>(16651119, 10459244);
  test_word_size<uint64_t, 48, 4, 0xCA5A82639512, 0x9E3779B97F4A, 0xD2E7470EE14C, 0xBB67AE8584CA>(
    192901316765185, 152776453925548);
  test_word_size<uint64_t, 63, 4, 0x4A5A826395121157, 0x1E3779B97F4A7C15, 0x52E7470EE14C6C93, 0x3B67AE8584CAA73B>(
    2623682490625152343, 3545397186678796316);
  return true;
}

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test_discard_carries()
{
  using cuda::std::uint16_t;
  using cuda::std::uint32_t;
  // reference values obtained from a big integer implementation of [rand.eng.philox] and libstdc++
  test_discard_carry<uint32_t, 8, 0xD2, 0x9E, 0xCA, 0xBB>(261124ull, 33);
  test_discard_carry<uint16_t, 16, 0xD256, 0x9E37, 0xCA50, 0xBB67>(17179607044ull, 50295);
  test_discard_carry<uint32_t, 31, 0x4D9E8D57, 0x1E3779B9, 0x52511F53, 0x3B67AE85>(18446744065119617028ull, 1942294099);
  return true;
}

TEST_HOST_DEVICE_FUNC TEST_CONSTEXPR_CXX20 bool test()
{
  test_engine<cuda::std::philox4x32, 1955073260u>();
  test_engine<cuda::std::philox4x64, 3409172418970261260ull>();
  test_set_counter<cuda::std::philox4x32>();
  test_set_counter<cuda::std::philox4x64>();
  test_against_reference();
  test_discard_carries();
  test_word_sizes();
#if TEST_STD_VER >= 2020
  static_assert(test_set_counter<cuda::std::philox4x32>());
  static_assert(test_set_counter<cuda::std::philox4x64>());
  static_assert(test_against_reference());
  static_assert(test_discard_carries());
  static_assert(test_word_sizes());
#endif
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
