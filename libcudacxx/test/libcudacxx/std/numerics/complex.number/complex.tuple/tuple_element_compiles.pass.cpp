//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/complex>

//   template<size_t I, class T> struct tuple_element;

#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <size_t I, typename C, typename = void>
struct HasTupleElement : cuda::std::false_type
{};

template <size_t I, typename C>
struct HasTupleElement<I, C, cuda::std::void_t<decltype(cuda::std::tuple_element<I, C>{})>> : cuda::std::true_type
{};

struct SomeObject
{};

static_assert(!HasTupleElement<0, SomeObject>::value, "");
static_assert(!HasTupleElement<1, SomeObject>::value, "");
static_assert(!HasTupleElement<3, SomeObject>::value, "");

template <typename T>
__host__ __device__ void test()
{
  using C = cuda::std::complex<T>;

  static_assert(HasTupleElement<0, C>::value, "");
  static_assert(HasTupleElement<1, C>::value, "");

  ASSERT_SAME_TYPE(T, typename cuda::std::tuple_element<0, C>::type);
  ASSERT_SAME_TYPE(T, typename cuda::std::tuple_element<1, C>::type);
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
