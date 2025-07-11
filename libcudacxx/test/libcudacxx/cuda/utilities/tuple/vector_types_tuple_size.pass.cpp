//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

_CCCL_SUPPRESS_DEPRECATED_PUSH

template <class VType, size_t Size>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::tuple_size<VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<volatile VType>::value == Size, "");
  static_assert(cuda::std::tuple_size<const volatile VType>::value == Size, "");
}

#define EXPAND_VECTOR_TYPE(Type) \
  test<Type##1, 1>();            \
  test<Type##2, 2>();            \
  test<Type##3, 3>();            \
  test<Type##4, 4>();

__host__ __device__ constexpr bool test()
{
  EXPAND_VECTOR_TYPE(char);
  EXPAND_VECTOR_TYPE(uchar);
  EXPAND_VECTOR_TYPE(short);
  EXPAND_VECTOR_TYPE(ushort);
  EXPAND_VECTOR_TYPE(int);
  EXPAND_VECTOR_TYPE(uint);
  EXPAND_VECTOR_TYPE(long);
  EXPAND_VECTOR_TYPE(ulong);
  EXPAND_VECTOR_TYPE(longlong);
  EXPAND_VECTOR_TYPE(ulonglong);
  EXPAND_VECTOR_TYPE(float);
  EXPAND_VECTOR_TYPE(double);

#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, 4>();
  test<long4_32a, 4>();
  test<ulong4_16a, 4>();
  test<ulong4_32a, 4>();
  test<longlong4_16a, 4>();
  test<longlong4_32a, 4>();
  test<ulonglong4_16a, 4>();
  test<ulonglong4_32a, 4>();
  test<double4_16a, 4>();
  test<double4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

#if _CCCL_HAS_NVFP16()
  test<__half2, 2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, 2>();
#endif // _CCCL_HAS_NVBF16()

  test<dim3, 3>();

  return true;
}

int main(int arg, char** argv)
{
  test();
  static_assert(test());
  return 0;
}
