//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/cmath>
#include <cuda/std/__random_>
#include <cuda/std/chrono>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

// test all power of 2 and maximum values
template <typename value_t, typename divisor_t>
__host__ __device__ void test_power_of_2(value_t value)
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
__host__ __device__ void test_sequence(value_t value)
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
__host__ __device__ void test_random(value_t value, gen_t& gen)
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

struct PositiveDistribution
{
  cuda::std::minstd_rand0 rng;

  template <typename T>
  __host__ __device__ T operator()(cuda::std::uniform_int_distribution<T>& distrib)
  {
    auto value = distrib(rng);
    return cuda::std::max(T{1}, static_cast<T>(cuda::uabs(value)));
  }
};

template <typename value_t, typename divisor_t>
__host__ __device__ void test()
{
  auto seed = cuda::std::chrono::system_clock::now().time_since_epoch().count();
  printf("%s: seed: %lld\n", (_CCCL_BUILTIN_PRETTY_FUNCTION()), (long long int) seed);
  cuda::std::uniform_int_distribution<value_t> distrib;
  cuda::std::minstd_rand0 rng(static_cast<uint32_t>(seed));
  PositiveDistribution positive_distr{rng};
  value_t value = positive_distr(distrib);
  test_power_of_2<value_t, divisor_t>(value);
  test_sequence<value_t, divisor_t>(value);
  test_random<value_t, divisor_t>(value, positive_distr);
}

__host__ __device__ void test_cpp_semantic()
{
  cuda::fast_mod_div<int> div{3};
  assert(div == 3);
  assert(3 % div == 0);
  assert(div + 1 == 4);
  assert(cuda::div(5, div) == (cuda::std::pair<int, int>{1, 2}));
}

__host__ __device__ bool test()
{
  test<int8_t, int8_t>();
  test<uint8_t, uint8_t>();
  test<int16_t, int16_t>();
  test<uint16_t, uint16_t>();
  test<int, int>();
  test<unsigned, unsigned>();
#if _CCCL_HAS_INT128()
  test<int64_t, int64_t>();
  test<uint64_t, uint64_t>();
#endif // _CCCL_HAS_INT128()
  //
  test<int16_t, int>();
  test<int8_t, unsigned>();
  test<uint8_t, unsigned>();
  test<int, unsigned>();
#if _CCCL_HAS_INT128()
  test<int64_t, uint64_t>();
#endif // _CCCL_HAS_INT128()
  //
  test_cpp_semantic();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
