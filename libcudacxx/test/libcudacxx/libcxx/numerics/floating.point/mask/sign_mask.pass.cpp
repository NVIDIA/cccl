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
#include <cuda/std/type_traits>

#include "test_macros.h"

template <cuda::std::__fp_format Fmt>
__host__ __device__ constexpr bool test_fp_sign_mask(cuda::std::__fp_storage_t<Fmt> expected)
{
  using T = cuda::std::__cccl_fp<Fmt>;

  static_assert(cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(cuda::std::__fp_sign_mask_v<Fmt>)>,
                                     cuda::std::__fp_storage_t<Fmt>>);
  static_assert(cuda::std::is_same_v<cuda::std::remove_cv_t<decltype(cuda::std::__fp_sign_mask_of_v<T>)>,
                                     cuda::std::__fp_storage_t<Fmt>>);

  assert(cuda::std::__fp_sign_mask_v<Fmt> == expected);
  assert(cuda::std::__fp_sign_mask_of_v<T> == expected);

  return true;
}

static_assert(test_fp_sign_mask<cuda::std::__fp_format::__binary16>(0x8000u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__binary32>(0x80000000u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__binary64>(0x8000000000000000ull));
#if _CCCL_HAS_INT128()
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__binary128>(__uint128_t(0x8000000000000000ull) << 64));
#endif // _CCCL_HAS_INT128()
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__bfloat16>(0x8000u));
#if _CCCL_HAS_INT128()
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp80_x86>(__uint128_t(0x8000ull) << 64));
#endif // _CCCL_HAS_INT128()
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp8_nv_e4m3>(0x80u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp8_nv_e5m2>(0x80u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp8_nv_e8m0>(0x00u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp6_nv_e2m3>(0x20u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp6_nv_e3m2>(0x20u));
static_assert(test_fp_sign_mask<cuda::std::__fp_format::__fp4_nv_e2m1>(0x8u));

int main(int, char**)
{
  return 0;
}
