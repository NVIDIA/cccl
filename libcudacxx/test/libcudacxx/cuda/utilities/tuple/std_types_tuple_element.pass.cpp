//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include <array>
#include <complex>
#include <tuple>
#include <utility>

#include "test_macros.h"

template <class STD_TYPE, size_t Index, typename Expected>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<Index, STD_TYPE>, Expected>);
  static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<Index, const STD_TYPE>, const Expected>);
  static_assert(cuda::std::is_same_v<cuda::std::tuple_element_t<Index, volatile STD_TYPE>, volatile Expected>);
  static_assert(
    cuda::std::is_same_v<cuda::std::tuple_element_t<Index, const volatile STD_TYPE>, const volatile Expected>);
}

__host__ __device__ constexpr bool test()
{
  // complex has both elements as the template argument
  test<::std::complex<float>, 0, float>();
  test<::std::complex<float>, 1, float>();
  test<::std::complex<double>, 0, double>();
  test<::std::complex<double>, 1, double>();
#if _CCCL_HAS_NVFP16()
  test<::std::complex<__half>, 0, __half>();
  test<::std::complex<__half>, 1, __half>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<::std::complex<__nv_bfloat16>, 0, __nv_bfloat16>();
  test<::std::complex<__nv_bfloat16>, 1, __nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()

  test<::std::pair<int, float>, 0, int>();
  test<::std::pair<int, float>, 1, float>();

  // tuple has the size of the number of template arguments
  test<::std::tuple<int>, 0, int>();
  test<::std::tuple<int, float>, 0, int>();
  test<::std::tuple<int, float>, 1, float>();
  test<::std::tuple<int, float, short>, 0, int>();
  test<::std::tuple<int, float, short>, 1, float>();
  test<::std::tuple<int, float, short>, 2, short>();

  // array has always the same element
  test<::std::array<int, 4>, 0, int>();
  test<::std::array<int, 4>, 1, int>();
  test<::std::array<int, 4>, 2, int>();
  test<::std::array<int, 4>, 3, int>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
