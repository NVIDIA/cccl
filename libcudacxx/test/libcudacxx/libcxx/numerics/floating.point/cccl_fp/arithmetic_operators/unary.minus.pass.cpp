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

template <typename T, typename = void>
struct has_unary_minus : cuda::std::false_type
{};

template <typename T>
struct has_unary_minus<T, cuda::std::void_t<decltype(-T{})>> : cuda::std::true_type
{};

template <cuda::std::__fp_format Fmt>
__host__ __device__ constexpr void test_fp_unary_minus()
{
  using T = cuda::std::__cccl_fp<Fmt>;

  if constexpr (cuda::std::__fp_is_signed_v<Fmt>)
  {
    static_assert(cuda::std::is_same_v<decltype(-T{}), T>);
    static_assert(noexcept(-T{}));

    // todo: implement test once __fp_cast is implemented
  }
  else
  {
    static_assert(!has_unary_minus<T>::value);
  }
}

__host__ __device__ constexpr bool test()
{
  test_fp_unary_minus<cuda::std::__fp_format::__binary16>();
  test_fp_unary_minus<cuda::std::__fp_format::__binary32>();
  test_fp_unary_minus<cuda::std::__fp_format::__binary64>();
#if _CCCL_HAS_INT128()
  test_fp_unary_minus<cuda::std::__fp_format::__binary128>();
#endif // _CCCL_HAS_INT128()
  test_fp_unary_minus<cuda::std::__fp_format::__bfloat16>();
#if _CCCL_HAS_INT128()
  test_fp_unary_minus<cuda::std::__fp_format::__fp80_x86>();
#endif // _CCCL_HAS_INT128()
  test_fp_unary_minus<cuda::std::__fp_format::__fp8_nv_e4m3>();
  test_fp_unary_minus<cuda::std::__fp_format::__fp8_nv_e5m2>();
  test_fp_unary_minus<cuda::std::__fp_format::__fp8_nv_e8m0>();
  test_fp_unary_minus<cuda::std::__fp_format::__fp6_nv_e2m3>();
  test_fp_unary_minus<cuda::std::__fp_format::__fp6_nv_e3m2>();
  test_fp_unary_minus<cuda::std::__fp_format::__fp4_nv_e2m1>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
