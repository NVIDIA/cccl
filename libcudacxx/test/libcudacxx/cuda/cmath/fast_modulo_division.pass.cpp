//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// error: dynamic memory allocation is unsupported in tile code

#include <cuda/cmath>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include "test_macros.h"

// test all power of 2 and maximum values
template <typename value_t, typename divisor_t>
TEST_FUNC void test_power_of_2(value_t value)
{
  constexpr auto max_divisor = cuda::std::numeric_limits<divisor_t>::max();
  constexpr auto max_value   = cuda::std::numeric_limits<value_t>::max();
  auto range                 = cuda::ceil_div(max_divisor, divisor_t{2}); // 2^(N/2)
  using div_op               = cuda::fast_mod_div<divisor_t>;
  using common_t             = cuda::std::common_type_t<value_t, divisor_t>;
  for (divisor_t i = 1; i < range; i *= 2)
  {
    assert(value / div_op{i} == value / i);
  }
  assert(value / div_op{range} == value / range);
  assert(value_t{0} / div_op{range} == value_t{0} / range);
  assert(value / div_op{max_divisor} == value / max_divisor);
  assert(max_divisor / div_op{max_divisor} == 1);
  assert(max_value / div_op(max_value) == 1);
  assert((max_value / div_op(value)) == (common_t) (max_value / value));
}

template <typename value_t, typename divisor_t>
TEST_FUNC void test_sequence(value_t value)
{
  constexpr auto max_value  = cuda::std::numeric_limits<divisor_t>::max();
  constexpr divisor_t range = max_value < 10000 ? max_value : 10000;
  using div_op              = cuda::fast_mod_div<divisor_t>;
  for (divisor_t i = 1; i < range; i++)
  {
    assert(value / div_op{i} == value / i);
  }
}

template <typename value_t, typename divisor_t, typename gen_t>
TEST_FUNC void test_random(value_t value, gen_t& gen)
{
  cuda::std::uniform_int_distribution<divisor_t> distrib_div;
  constexpr auto max_value  = cuda::std::numeric_limits<divisor_t>::max();
  constexpr divisor_t range = max_value < 10000 ? max_value : 10000;
  using div_op              = cuda::fast_mod_div<divisor_t>;
  for (divisor_t i = 1; i < range; i++)
  {
    auto divisor = gen(distrib_div);
    assert(value / div_op{divisor} == value / divisor);
  }
}

template <typename value_t, typename divisor_t>
TEST_FUNC void test_boundary_divisors(value_t value)
{
  using div_op               = cuda::fast_mod_div<divisor_t>;
  using unsigned_divisor_t   = cuda::std::make_unsigned_t<divisor_t>;
  constexpr int digits       = cuda::std::numeric_limits<divisor_t>::digits;
  constexpr auto max_divisor = cuda::std::numeric_limits<divisor_t>::max();
  constexpr auto medium_pow2 = static_cast<unsigned_divisor_t>(unsigned_divisor_t{1} << (digits / 2));
  constexpr auto high_pow2   = static_cast<unsigned_divisor_t>(unsigned_divisor_t{1} << (digits - 2));
  const divisor_t divisors[] = {
    divisor_t{1},
    divisor_t{2},
    divisor_t{3},
    divisor_t{5},
    static_cast<divisor_t>(medium_pow2 - 1),
    static_cast<divisor_t>(medium_pow2),
    static_cast<divisor_t>(medium_pow2 + 1),
    static_cast<divisor_t>(high_pow2 - 1),
    static_cast<divisor_t>(high_pow2),
    static_cast<divisor_t>(high_pow2 + 1),
    static_cast<divisor_t>(max_divisor - 1),
    max_divisor,
  };
  for (const auto divisor : divisors)
  {
    assert(value / div_op{divisor} == value / divisor);
  }
}

struct PositiveDistribution
{
  cuda::std::minstd_rand0 rng;

  template <typename T>
  TEST_FUNC T operator()(cuda::std::uniform_int_distribution<T>& distrib)
  {
    auto value = distrib(rng);
    return cuda::std::max(T{1}, static_cast<T>(cuda::uabs(value)));
  }
};

template <typename value_t, typename divisor_t>
TEST_FUNC void test()
{
  auto seed = cuda::std::chrono::system_clock::now().time_since_epoch().count();
  printf("%s: seed: %lld\n", (_CCCL_BUILTIN_PRETTY_FUNCTION()), (long long int) seed);
  cuda::std::uniform_int_distribution<value_t> distrib;
  cuda::std::minstd_rand0 rng(static_cast<uint32_t>(seed));
  PositiveDistribution positive_distr{rng};
  value_t value = positive_distr(distrib);
  test_power_of_2<value_t, divisor_t>(value);
  test_sequence<value_t, divisor_t>(value);
  test_boundary_divisors<value_t, divisor_t>(value);
  test_random<value_t, divisor_t>(value, positive_distr);
}

#if _CCCL_HAS_INT128()
TEST_FUNC void test_int128()
{
  constexpr __uint128_t unsigned_value = (__uint128_t{1} << 127) + 123456789;
  constexpr __int128_t signed_value    = static_cast<__int128_t>((__uint128_t{1} << 126) + 123456789);

  test_power_of_2<__int128_t, __int128_t>(signed_value);
  test_sequence<__int128_t, __int128_t>(signed_value);
  test_boundary_divisors<__int128_t, __int128_t>(signed_value);

  test_power_of_2<__uint128_t, __uint128_t>(unsigned_value);
  test_sequence<__uint128_t, __uint128_t>(unsigned_value);
  test_boundary_divisors<__uint128_t, __uint128_t>(unsigned_value);

  test_power_of_2<__int128_t, __uint128_t>(signed_value);
  test_sequence<__int128_t, __uint128_t>(signed_value);
  test_boundary_divisors<__int128_t, __uint128_t>(signed_value);
}
#endif // _CCCL_HAS_INT128()

TEST_FUNC void test_cpp_semantic()
{
  cuda::fast_mod_div<int> div{3};
  assert(div == 3);
  assert(3 % div == 0);
  assert(div + 1 == 4);
  assert(cuda::div(5, div) == (cuda::std::pair<int, int>{1, 2}));
}

TEST_FUNC void test_divisor_is_never_one()
{
  cuda::fast_mod_div<int, true> div{3};
  assert(div == 3);
  assert(3 % div == 0);
  assert(div + 1 == 4);
  assert(cuda::div(5, div) == (cuda::std::pair<int, int>{1, 2}));
}

TEST_FUNC bool test()
{
  test<int8_t, int8_t>();
  test<uint8_t, uint8_t>();
  test<int16_t, int16_t>();
  test<uint16_t, uint16_t>();
  test<int, int>();
  test<unsigned, unsigned>();
  test<int64_t, int64_t>();
  test<uint64_t, uint64_t>();
  //
  test<int16_t, int>();
  test<int8_t, unsigned>();
  test<uint8_t, unsigned>();
  test<int, unsigned>();
  test<int64_t, uint64_t>();

#if _CCCL_HAS_INT128()
  test_int128();

  test<int, __int128_t>();
  test<int64_t, __uint128_t>();
#endif // _CCCL_HAS_INT128()

  test_cpp_semantic();
  test_divisor_is_never_one();

  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
