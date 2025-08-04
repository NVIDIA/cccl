//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_fp_set_exp(T val, int exponent)
{
  assert(cuda::std::__fp_get_exp(val) != exponent);
  assert(cuda::std::__fp_get_exp(cuda::std::__fp_set_exp(val, exponent)) == exponent);
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test_fp_set_exp(T val)
{
  constexpr auto fmt = cuda::std::__fp_format_of_v<T>;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::__fp_set_exp(cuda::std::declval<T>(), int{0})), T>);
  static_assert(noexcept(cuda::std::__fp_set_exp(cuda::std::declval<T>(), 0)));

  test_fp_set_exp(val, 2);
  if constexpr (cuda::std::__fp_exp_nbits_v<fmt> > 3)
  {
    test_fp_set_exp(T{32}, 0);
  }

  { // set all exponent bits to zero
    if constexpr (fmt != cuda::std::__fp_format::__fp8_nv_e8m0) // All zero counts as NaN for __fp8_nv_e8m0
    { // 2 has no mantissa bit set, so it will be a zero
      const T res = cuda::std::__fp_set_exp(T{2}, cuda::std::__fp_exp_min_v<fmt> - 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(!cuda::std::isinf(res));
      assert(cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_ZERO);
    }

    if constexpr (cuda::std::__fp_is_signed_v<fmt>)
    { // -2 has no mantissa bit set, so it will be a zero
      const T res = cuda::std::__fp_set_exp(T{-2}, cuda::std::__fp_exp_min_v<fmt> - 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(!cuda::std::isinf(res));
      assert(cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_ZERO);
    }

    if constexpr (cuda::std::__fp_has_denorm_v<fmt>)
    { // 3 has a mantissa bit set, so it will be a denormal
      const T res = cuda::std::__fp_set_exp(T{3}, cuda::std::__fp_exp_min_v<fmt> - 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(!cuda::std::isinf(res));
      assert(cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_SUBNORMAL);
    }

    if constexpr (cuda::std::__fp_has_denorm_v<fmt> && cuda::std::__fp_is_signed_v<fmt>)
    { // -3 has a mantissa bit set, so it will be a denormal
      const T res = cuda::std::__fp_set_exp(T{-3}, cuda::std::__fp_exp_min_v<fmt> - 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(!cuda::std::isinf(res));
      assert(cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_SUBNORMAL);
    }
  }

  { // set all exponent bits to one
    if constexpr (cuda::std::__fp_has_inf_v<fmt>)
    { // 2 has no mantissa bit set, so it will be an infinity
      const T res = cuda::std::__fp_set_exp(T{2}, cuda::std::__fp_exp_max_v<fmt> + 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(cuda::std::isinf(res));
      assert(!cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_INFINITE);
    }

    if constexpr (cuda::std::__fp_has_inf_v<fmt> && cuda::std::__fp_is_signed_v<fmt>)
    { // -2 has no mantissa bit set, so it will be an infinity
      const T res = cuda::std::__fp_set_exp(T{-2}, cuda::std::__fp_exp_max_v<fmt> + 1);
      assert(!cuda::std::isnormal(res));
      assert(!cuda::std::isnan(res));
      assert(cuda::std::isinf(res));
      assert(!cuda::std::isfinite(res));
      assert(cuda::std::fpclassify(res) == FP_INFINITE);
    }

    if constexpr (fmt != cuda::std::__fp_format::__fp8_nv_e4m3)
    { // __fp8_nv_e4m3 has special NaN values
      if constexpr (cuda::std::__fp_has_nan_v<fmt>)
      { // 3 has no mantissa bit set, so it will be a NaN
        const T res = cuda::std::__fp_set_exp(T{3}, cuda::std::__fp_exp_max_v<fmt> + 1);
        assert(!cuda::std::isnormal(res));
        assert(cuda::std::isnan(res));
        assert(!cuda::std::isinf(res));
        assert(!cuda::std::isfinite(res));
        assert(cuda::std::fpclassify(res) == FP_NAN);
      }

      if constexpr (cuda::std::__fp_has_nan_v<fmt> && cuda::std::__fp_is_signed_v<fmt>)
      { // -3 has no mantissa bit set, so it will be a NaN
        const T res = cuda::std::__fp_set_exp(T{-3}, cuda::std::__fp_exp_max_v<fmt> + 1);
        assert(!cuda::std::isnormal(res));
        assert(cuda::std::isnan(res));
        assert(!cuda::std::isinf(res));
        assert(!cuda::std::isfinite(res));
        assert(cuda::std::fpclassify(res) == FP_NAN);
      }
    }
  }
}

template <class T>
__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST void test(T val = T{1})
{
  test_fp_set_exp<T>(val);
}

__host__ __device__ bool test(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>(val);
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>(val);
#endif // _LIBCUDACXX_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3
#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2
#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0
#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3
#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2
#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1

  return true;
}

__host__ __device__ _CCCL_CONSTEXPR_BIT_CAST bool test_constexpr(float val)
{
  test<float>(val);
  test<double>(val);
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>(val);
#endif // _CCCL_HAS_LONG_DOUBLE()

  return true;
}

int main(int, char**)
{
  volatile float value = 1.0f;
  test(value);

#if _CCCL_HAS_CONSTEXPR_BIT_CAST()
  static_assert(test_constexpr(1.0f));
#endif // _CCCL_HAS_CONSTEXPR_BIT_CAST()
  return 0;
}
