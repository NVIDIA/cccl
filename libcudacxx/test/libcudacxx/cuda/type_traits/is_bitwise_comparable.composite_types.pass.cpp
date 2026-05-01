//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__complex_>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#include <cuda/type_traits>

#include "test_macros.h"

struct WithNoPadding
{
  int x;
  int y;
};

struct WithPadding
{
  int x;
  char y;
};

struct UserSpecialization
{
  double value;
};

template <>
constexpr bool cuda::is_bitwise_comparable_v<UserSpecialization> = true;

__host__ __device__ void test_composite_types()
{
  // padding-free cuda::std::array, pair, tuple of bitwise comparable types
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::array<int, 4>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::pair<int, unsigned>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::tuple<int, unsigned>>);
  static_assert(cuda::is_bitwise_comparable_v<cuda::std::tuple<>>);

  // composites with padding are not bitwise comparable
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<int, char>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<int, unsigned, char>>);

  // complex types are explicitly not bitwise comparable
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::complex<float>>);
  static_assert(!cuda::is_bitwise_comparable_v<const cuda::std::complex<float>>);

  static_assert(cuda::is_bitwise_comparable_v<WithNoPadding>);
  static_assert(!cuda::is_bitwise_comparable_v<WithPadding>);

  // user specialization of the variable template
  static_assert(cuda::is_bitwise_comparable_v<UserSpecialization>);
  static_assert(cuda::is_bitwise_comparable_v<const UserSpecialization>);
  static_assert(cuda::is_bitwise_comparable_v<volatile UserSpecialization>);
  static_assert(cuda::is_bitwise_comparable_v<const volatile UserSpecialization>);
  static_assert(cuda::is_bitwise_comparable<UserSpecialization>::value);
  static_assert(cuda::is_bitwise_comparable<const UserSpecialization>::value);
  static_assert(cuda::is_bitwise_comparable<volatile UserSpecialization>::value);
  static_assert(cuda::is_bitwise_comparable<const volatile UserSpecialization>::value);
}

#if _CCCL_HAS_NVFP16()

struct NoPaddingExtendedFloatingPoint
{
  unsigned short x;
  __half y;
};

#endif // _CCCL_HAS_NVFP16()

__host__ __device__ void test_extended_floating_point_types()
{
  // compositions of extended floating-point types
#if _CCCL_HAS_NVFP16()
  static_assert(!cuda::is_bitwise_comparable_v<__half[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<const __half[4]>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<__half, 4>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<__half, int>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<__half, float>>);
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<__nv_bfloat16, 2>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::pair<__nv_bfloat16, int>>);
#endif // _CCCL_HAS_NVBF16()

  // nested compositions
#if _CCCL_HAS_NVFP16()
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::array<cuda::std::pair<__half, int>, 2>>);
  static_assert(!cuda::is_bitwise_comparable_v<cuda::std::tuple<cuda::std::array<__half, 4>, int>>);
  static_assert(!cuda::is_bitwise_comparable_v<NoPaddingExtendedFloatingPoint>);
#endif // _CCCL_HAS_NVFP16()
}

int main(int, char**)
{
  test_composite_types();
  test_extended_floating_point_types();
  return 0;
}
