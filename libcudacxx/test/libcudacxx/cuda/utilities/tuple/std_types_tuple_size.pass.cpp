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

template <class STD_TYPE, size_t Size>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::tuple_size<STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<const STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<volatile STD_TYPE>::value == Size);
  static_assert(cuda::std::tuple_size<const volatile STD_TYPE>::value == Size);
}

__host__ __device__ constexpr bool test()
{
  // complex has a size of 2
  test<::std::complex<float>, 2>();
  test<::std::complex<double>, 2>();
#if _CCCL_HAS_NVFP16()
  test<::std::complex<__half>, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<::std::complex<__nv_bfloat16>, 2>();
#endif // _CCCL_HAS_NVBF16()

  // pair always has a size of 2
  test<::std::pair<int, float>, 2>();

  // tuple has the size of the number of template arguments
  test<::std::tuple<int>, 1>();
  test<::std::tuple<int, int>, 2>();
  test<::std::tuple<int, int, int>, 3>();

  // array has the size of the number of elements
  test<::std::array<int, 4>, 4>();
  test<::std::array<int, 1337>, 1337>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
