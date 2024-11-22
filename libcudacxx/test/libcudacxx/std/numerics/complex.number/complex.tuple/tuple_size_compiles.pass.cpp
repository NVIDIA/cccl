//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

//   template<class T> struct tuple_size;

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <typename C, typename = void>
struct HasTupleSize : cuda::std::false_type
{};

template <typename C>
struct HasTupleSize<C, cuda::std::void_t<decltype(cuda::std::tuple_size<C>{})>> : cuda::std::true_type
{};

struct SomeObject
{};

static_assert(!HasTupleSize<SomeObject>::value, "");

template <typename T>
__host__ __device__ void test()
{
  using C = cuda::std::complex<T>;

  static_assert(HasTupleSize<C>::value, "");
  ASSERT_SAME_TYPE(size_t, typename cuda::std::tuple_size<C>::value_type);
  static_assert(cuda::std::tuple_size<C>() == 2, "");
}

__host__ __device__ void test()
{
  test<float>();
  test<double>();

  // CUDA treats long double as double
  // test<long double>();

#ifdef _LIBCUDACXX_HAS_NVFP16
  test<__half>();
#endif
#ifdef _LIBCUDACXX_HAS_NVBF16
  test<__nv_bfloat16>();
#endif
}

int main(int, char**)
{
  return 0;
}
