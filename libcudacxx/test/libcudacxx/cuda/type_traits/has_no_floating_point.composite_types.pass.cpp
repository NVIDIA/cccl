//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__complex_>
#include <cuda/__type_traits/has_no_floating_point.h>
#include <cuda/std/array>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "cuda_fp_types.h"
#include "test_macros.h"

template <class T>
TEST_FUNC void test_has_no_floating_point()
{
  static_assert(cuda::__has_no_floating_point_v<T>);
  static_assert(cuda::__has_no_floating_point_v<const T>);
  static_assert(cuda::__has_no_floating_point_v<volatile T>);
  static_assert(cuda::__has_no_floating_point_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_has_floating_point()
{
  static_assert(!cuda::__has_no_floating_point_v<T>);
  static_assert(!cuda::__has_no_floating_point_v<const T>);
  static_assert(!cuda::__has_no_floating_point_v<volatile T>);
  static_assert(!cuda::__has_no_floating_point_v<const volatile T>);
}

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

struct WithFloatingPoint
{
  int x;
  float y;
};

struct NestedWithoutFloatingPoint
{
  WithPadding x;
  cuda::std::array<int, 4> y;
};

struct UserFloatingPoint
{};

template <>
constexpr bool cuda::is_floating_point_v<UserFloatingPoint> = true;

struct ContainsUserFloatingPoint
{
  UserFloatingPoint value;
};

TEST_FUNC void test_composite_types()
{
  // padding does not affect whether a type contains floating-point members
  test_has_no_floating_point<WithNoPadding>();
  test_has_no_floating_point<WithPadding>();
  test_has_no_floating_point<NestedWithoutFloatingPoint>();

  test_has_floating_point<WithFloatingPoint>();

  // cuda::std::array, pair, and tuple compositions
  test_has_no_floating_point<cuda::std::array<int, 4>>();
  test_has_no_floating_point<cuda::std::pair<int, char>>();
  test_has_no_floating_point<cuda::std::tuple<int, unsigned, char>>();
  test_has_no_floating_point<cuda::std::tuple<>>();

  test_has_floating_point<cuda::std::array<float, 4>>();
  test_has_floating_point<cuda::std::pair<int, float>>();
  test_has_floating_point<cuda::std::tuple<int, double, char>>();
  test_has_floating_point<cuda::std::tuple<cuda::std::array<float, 4>, int>>();

  test_has_floating_point<cuda::std::complex<float>>();
  test_has_floating_point<cuda::complex<double>>();

  // cuda::is_floating_point_v user specializations are respected
  test_has_floating_point<UserFloatingPoint>();
  test_has_floating_point<ContainsUserFloatingPoint>();
}

#if _CCCL_HAS_NVFP16()

struct NoPaddingExtendedFloatingPoint
{
  unsigned short x;
  __half y;
};

#endif // _CCCL_HAS_NVFP16()

TEST_FUNC void test_extended_floating_point_types()
{
#if _CCCL_HAS_NVFP16()
  test_has_floating_point<__half[4]>();
  test_has_floating_point<cuda::std::array<__half, 4>>();
  test_has_floating_point<cuda::std::pair<__half, int>>();
  test_has_floating_point<cuda::std::tuple<__half, float>>();
  test_has_floating_point<cuda::std::array<cuda::std::pair<__half, int>, 2>>();
  test_has_floating_point<cuda::std::tuple<cuda::std::array<__half, 4>, int>>();
  test_has_floating_point<NoPaddingExtendedFloatingPoint>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test_has_floating_point<cuda::std::array<__nv_bfloat16, 2>>();
  test_has_floating_point<cuda::std::pair<__nv_bfloat16, int>>();
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_CTK()
  test_has_no_floating_point<int2>();
  test_has_floating_point<float2>();
  test_has_floating_point<cuda::std::array<double2, 2>>();
#endif // _CCCL_HAS_CTK()
}

int main(int, char**)
{
  test_composite_types();
  test_extended_floating_point_types();
  return 0;
}
