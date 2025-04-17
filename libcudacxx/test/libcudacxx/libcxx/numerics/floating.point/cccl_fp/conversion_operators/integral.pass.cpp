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
#include <cuda/std/cstring>
#include <cuda/std/type_traits>

template <cuda::std::__fp_format Fmt, class Int>
__host__ __device__ constexpr void test_int_conversion_operator()
{
  using T = cuda::std::__cccl_fp<Fmt>;

  // Construction from an integral type is always noexcept
  static_assert(cuda::std::is_nothrow_constructible_v<T, Int>);

  // Construction from an integral type is always implicit
  static_assert(cuda::std::is_convertible_v<T, Int>);

  // TODO: check conversion to an integral type
  [[maybe_unused]] Int val(T{});
}

template <cuda::std::__fp_format Fmt>
__host__ __device__ constexpr void test_format()
{
  test_int_conversion_operator<Fmt, bool>();

  test_int_conversion_operator<Fmt, char>();
#if _CCCL_HAS_CHAR8_T()
  test_int_conversion_operator<Fmt, char8_t>();
#endif // _CCCL_HAS_CHAR8_T()
  test_int_conversion_operator<Fmt, char16_t>();
  test_int_conversion_operator<Fmt, char32_t>();
  test_int_conversion_operator<Fmt, wchar_t>();

  test_int_conversion_operator<Fmt, signed char>();
  test_int_conversion_operator<Fmt, signed short>();
  test_int_conversion_operator<Fmt, signed int>();
  test_int_conversion_operator<Fmt, signed long>();
  test_int_conversion_operator<Fmt, signed long long>();
#if _CCCL_HAS_INT128()
  test_int_conversion_operator<Fmt, __int128_t>();
#endif // _CCCL_HAS_INT128()

  test_int_conversion_operator<Fmt, unsigned char>();
  test_int_conversion_operator<Fmt, unsigned short>();
  test_int_conversion_operator<Fmt, unsigned int>();
  test_int_conversion_operator<Fmt, unsigned long>();
  test_int_conversion_operator<Fmt, unsigned long long>();
#if _CCCL_HAS_INT128()
  test_int_conversion_operator<Fmt, __uint128_t>();
#endif // _CCCL_HAS_INT128()
}

__host__ __device__ constexpr bool test()
{
  test_format<cuda::std::__fp_format::__binary16>();
  test_format<cuda::std::__fp_format::__binary32>();
  test_format<cuda::std::__fp_format::__binary64>();
#if _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__binary128>();
#endif // _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__bfloat16>();
#if _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__fp80_x86>();
#endif // _CCCL_HAS_INT128()
  test_format<cuda::std::__fp_format::__fp8_nv_e4m3>();
  test_format<cuda::std::__fp_format::__fp8_nv_e5m2>();
  test_format<cuda::std::__fp_format::__fp8_nv_e8m0>();
  test_format<cuda::std::__fp_format::__fp6_nv_e2m3>();
  test_format<cuda::std::__fp_format::__fp6_nv_e3m2>();
  test_format<cuda::std::__fp_format::__fp4_nv_e2m1>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
