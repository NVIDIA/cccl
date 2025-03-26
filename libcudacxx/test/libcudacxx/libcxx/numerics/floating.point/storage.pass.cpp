//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__floating_point/fp.h>
#include <cuda/std/cassert>
#include <cuda/std/cstring>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_fp_storage()
{
  using Storage = cuda::std::__fp_storage_of_t<T>;
  static_assert(cuda::std::is_integral_v<Storage>);
  static_assert(sizeof(Storage) == sizeof(T));
  static_assert(alignof(Storage) == alignof(T));

  static_assert(cuda::std::is_same_v<Storage, decltype(cuda::std::__fp_get_storage(T{}))>);
  const T max            = cuda::std::numeric_limits<T>::max();
  const Storage max_bits = cuda::std::__fp_get_storage(max);
  assert(cuda::std::memcmp(&max, &max_bits, sizeof(T)) == 0);

  static_assert(cuda::std::is_same_v<T, decltype(cuda::std::__fp_from_storage<T>(Storage{}))>);
  const T max_back = cuda::std::__fp_from_storage<T>(max_bits);
  assert(cuda::std::memcmp(&max, &max_back, sizeof(T)) == 0);
}

int main(int, char**)
{
  test_fp_storage<float>();
  test_fp_storage<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test_fp_storage<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _CCCL_HAS_NVFP16()
  test_fp_storage<__half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_fp_storage<__nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_fp_storage<__nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
#if _CCCL_HAS_NVFP8_E5M2()
  test_fp_storage<__nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()
#if _CCCL_HAS_NVFP8_E8M0()
  test_fp_storage<__nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()
#if _CCCL_HAS_NVFP6_E2M3()
  test_fp_storage<__nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()
#if _CCCL_HAS_NVFP6_E3M2()
  test_fp_storage<__nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()
#if _CCCL_HAS_NVFP4_E2M1()
  test_fp_storage<__nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  return 0;
}
