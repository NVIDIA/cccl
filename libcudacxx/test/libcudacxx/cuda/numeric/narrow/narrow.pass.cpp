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
#include <cuda/std/type_traits>

#include "test_macros.h"

#define CHECK_NARROWING_ERROR(expr, throw_cond)                                                                    \
  NV_IF_TARGET(NV_IS_HOST,                                                                                         \
               (                                                                                                   \
                 try {                                                                                             \
                   assert((expr));                                                                                 \
                   assert(!(throw_cond));                                                                          \
                 } catch (const cuda::narrowing_error&) { assert((throw_cond)); } catch (...) { assert(false); }), \
               (if (!(throw_cond)) { assert((expr)); }))

struct my_float
{
  explicit _CCCL_HOST_DEVICE my_float(float value)
      : value(value)
  {}

  _CCCL_HOST_DEVICE operator float() const
  {
    return value;
  }

private:
  float value;
};
static_assert(!cuda::std::is_arithmetic_v<my_float>);

template <class To, class From>
__host__ __device__ void test_type()
{
  // 1. Casting zero should always work
  assert(cuda::narrow<To>(From{0}) == To{0});

  // 2. Casting positive one should always work
  assert(cuda::narrow<To>(From{1}) == To{1});

  // 3. Casting negative one should overflow if the destination type is not signed
  if constexpr (cuda::std::is_signed_v<From>)
  {
    CHECK_NARROWING_ERROR((cuda::narrow<To>(From{-1}) == (To) -1), (!cuda::std::is_signed_v<To>) );
  }

  // 4. Casting the minimum value of From type
  if constexpr (cuda::std::is_integral_v<From> && cuda::std::is_integral_v<To>)
  {
    constexpr auto min = cuda::std::numeric_limits<From>::min();
    CHECK_NARROWING_ERROR((cuda::narrow<To>(min) == static_cast<To>(min)),
                          (cuda::std::cmp_less(min, cuda::std::numeric_limits<To>::min())));
  }

  // 5. Casting the maximum value of From type
  if constexpr (cuda::std::is_integral_v<From> && cuda::std::is_integral_v<To>)
  {
    constexpr auto max = cuda::std::numeric_limits<From>::max();
    CHECK_NARROWING_ERROR((cuda::narrow<To>(max) == static_cast<To>(max)),
                          (cuda::std::cmp_greater(max, cuda::std::numeric_limits<To>::max())));
  }
}

template <class To>
__host__ __device__ void test_type()
{
  test_type<To, signed char>();
  test_type<To, unsigned char>();
  test_type<To, short>();
  test_type<To, unsigned short>();
  test_type<To, int>();
  test_type<To, unsigned int>();
  test_type<To, long>();
  test_type<To, unsigned long>();
  test_type<To, long long>();
  test_type<To, unsigned long long>();
#if _CCCL_HAS_INT128()
#  if _LIBCUDACXX_HAS_NVFP16()
  // __int128_t and __uint128_t are not convertible to __half or __nv_bfloat16
  if constexpr (!::cuda::std::is_same_v<To, __half> && !::cuda::std::is_same_v<To, __nv_bfloat16>)
#  endif // _LIBCUDACXX_HAS_NVFP16()
  {
    test_type<To, __int128_t>();
    test_type<To, __uint128_t>();
  }
#endif // _CCCL_HAS_INT128()
  test_type<To, float>();
  test_type<To, double>();
#if _CCCL_HAS_INT128()
  // __half and __nv_bfloat16 are not convertible to __int128_t or __uint128_t
  if constexpr (!::cuda::std::is_same_v<To, __int128_t> && !::cuda::std::is_same_v<To, __uint128_t>)
#endif // _CCCL_HAS_INT128()
  {
#if _LIBCUDACXX_HAS_NVFP16()
    test_type<To, __half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
    test_type<To, __nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
  }
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
#if _LIBCUDACXX_HAS_NVFP16()
  test_type<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test_type<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()

  return true;
}

int main(int arg, char** argv)
{
  test();

  test_type<float, my_float>();
  test_type<my_float, float>();

  assert(cuda::narrow<float>(2 << (23 + 1)) == float{2 << (23 + 1)});
  CHECK_NARROWING_ERROR((cuda::narrow<float>((2 << (23 + 1)) + 1)), true);
  assert(cuda::narrow<double>(2ll << (52 + 1)) == float{2ll << (52 + 1)});
  CHECK_NARROWING_ERROR((cuda::narrow<double>((2ll << (52 + 1)) + 1)), true);

  return 0;
}
