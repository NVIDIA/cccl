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
  static_assert(cuda::std::is_same_v<size_t, typename cuda::std::tuple_size<C>::value_type>);
  static_assert(cuda::std::tuple_size<C>() == 2, "");
}

__host__ __device__ void test()
{
  test<float>();
  test<double>();
#if _CCCL_HAS_LONG_DOUBLE()
  test<long double>();
#endif // _CCCL_HAS_LONG_DOUBLE()
#if _LIBCUDACXX_HAS_NVFP16()
  test<__half>();
#endif // _LIBCUDACXX_HAS_NVFP16()
#if _LIBCUDACXX_HAS_NVBF16()
  test<__nv_bfloat16>();
#endif // _LIBCUDACXX_HAS_NVBF16()
}

int main(int, char**)
{
  return 0;
}
