//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/numeric>
#include <cuda/std/cassert>
#include <cuda/std/limits>

#include "test_macros.h"

#define CHECK_NARROWING_ERROR(expr, throw_cond)                                                                    \
  NV_IF_TARGET(NV_IS_HOST,                                                                                         \
               (                                                                                                   \
                 try {                                                                                             \
                   assert((expr));                                                                                 \
                   assert(!(throw_cond));                                                                          \
                 } catch (const cuda::narrowing_error&) { assert((throw_cond)); } catch (...) { assert(false); }), \
               (if (!(throw_cond)) { assert((expr)); }))

template <class To, class From>
__host__ __device__ void test_type()
{
  // 1. Casting zero should always work
  assert(cuda::narrow<To>(From{0}) == To{0});

  // 2. Casting positive one should always work
  assert(cuda::narrow<To>(From{1}) == To{1});

  // 3. Casting negative one should overflow if the destination type is unsigned
  if constexpr (cuda::std::is_signed_v<From>)
  {
    CHECK_NARROWING_ERROR((cuda::narrow<To>(From{-1}) == (To) -1), (cuda::std::is_unsigned_v<To>) );
  }

  // 4. Casting the minimum value of From type
  if constexpr (cuda::std::is_integral_v<From> && cuda::std::is_integral_v<To>)
  {
    constexpr auto min = cuda::std::numeric_limits<From>::min();
    CHECK_NARROWING_ERROR((cuda::narrow<To>(min) == (To) min),
                          (cuda::std::cmp_less(min, cuda::std::numeric_limits<To>::min())));
  }

  // 5. Casting the maximum value of From type
  if constexpr (cuda::std::is_integral_v<From> && cuda::std::is_integral_v<To>)
  {
    constexpr auto max = cuda::std::numeric_limits<From>::max();
    CHECK_NARROWING_ERROR((cuda::narrow<To>(max) == (To) max),
                          (cuda::std::cmp_greater(max, cuda::std::numeric_limits<To>::max())));
  }
}

template <class T>
__host__ __device__ void test_type()
{
  test_type<T, signed char>();
  test_type<T, unsigned char>();
  test_type<T, short>();
  test_type<T, unsigned short>();
  test_type<T, int>();
  test_type<T, unsigned int>();
  test_type<T, long>();
  test_type<T, unsigned long>();
  test_type<T, long long>();
  test_type<T, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<T, __int128_t>();
  test_type<T, __uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_type<T, float>();
  test_type<T, double>();
}

__host__ __device__ bool test()
{
  test_type<signed char>();
  test_type<unsigned char>();
  test_type<short>();
  test_type<unsigned short>();
  test_type<int>();
  test_type<unsigned int>();
  test_type<long>();
  test_type<unsigned long>();
  test_type<long long>();
  test_type<unsigned long long>();
#if _CCCL_HAS_INT128()
  test_type<__int128_t>();
  test_type<__uint128_t>();
#endif // _CCCL_HAS_INT128()
  test_type<float>();
  test_type<double>();

  return true;
}

int main(int arg, char** argv)
{
  test();

  assert(cuda::narrow<float>(2 << (23 + 1)) == float{2 << (23 + 1)});
  CHECK_NARROWING_ERROR((cuda::narrow<float>((2 << (23 + 1)) + 1)), true);
  assert(cuda::narrow<double>(2ll << (52 + 1)) == float{2ll << (52 + 1)});
  CHECK_NARROWING_ERROR((cuda::narrow<double>((2ll << (52 + 1)) + 1)), true);

  return 0;
}
