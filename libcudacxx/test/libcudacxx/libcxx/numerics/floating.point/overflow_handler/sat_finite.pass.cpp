//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_OPTIONS_HOST: -fext-numeric-literals
// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_GCC_HAS_EXTENDED_NUMERIC_LITERALS

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cmath>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_fp_sat_finite_overflow_handler()
{
  using Handler = cuda::std::__fp_overflow_handler<cuda::std::__fp_overflow_handler_kind::__sat_finite>;

  static_assert(cuda::std::__fp_is_overflow_handler_v<Handler>);

  static_assert(cuda::std::is_same_v<decltype(Handler::__handle_overflow<T>()), T>);
  static_assert(cuda::std::is_same_v<decltype(Handler::__handle_underflow<T>()), T>);

  static_assert(noexcept(Handler::__handle_overflow<T>()));
  static_assert(noexcept(Handler::__handle_underflow<T>()));

  // test __handle_overflow
  {
    const auto val = Handler::__handle_overflow<T>();
    const auto ref = cuda::std::numeric_limits<T>::max();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }

  // test __handle_underflow
  {
    const auto val = Handler::__handle_underflow<T>();
    const auto ref = cuda::std::numeric_limits<T>::lowest();
    assert(cuda::std::memcmp(&val, &ref, sizeof(T)) == 0);
  }
}

int main(int, char**)
{
  test_fp_sat_finite_overflow_handler<float>();
  test_fp_sat_finite_overflow_handler<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_sat_finite_overflow_handler<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_sat_finite_overflow_handler<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_sat_finite_overflow_handler<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_sat_finite_overflow_handler<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_sat_finite_overflow_handler<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_sat_finite_overflow_handler<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_sat_finite_overflow_handler<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_sat_finite_overflow_handler<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_sat_finite_overflow_handler<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()
#if _CCCL_HAS_FLOAT128()
  test_fp_sat_finite_overflow_handler<__float128>();
#endif // _CCCL_HAS_FLOAT128()

  return 0;
}
