//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__type_traits/vector_type.h>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

template <class T, cuda::std::size_t Size, class Ref>
__host__ __device__ void test()
{
  using Vec = cuda::__vector_type_t<T, Size>;
  static_assert(cuda::std::is_same_v<Vec, Ref>);
  static_assert(cuda::__has_vector_type_v<T, Size> == !cuda::std::is_same_v<Ref, void>);
}

__host__ __device__ void test()
{
  // 1. Test valid combinations

  test<signed char, 1, char1>();
  test<signed char, 2, char2>();
  test<signed char, 3, char3>();
  test<signed char, 4, char4>();

  test<unsigned char, 1, uchar1>();
  test<unsigned char, 2, uchar2>();
  test<unsigned char, 3, uchar3>();
  test<unsigned char, 4, uchar4>();

  test<signed short, 1, short1>();
  test<signed short, 2, short2>();
  test<signed short, 3, short3>();
  test<signed short, 4, short4>();

  test<unsigned short, 1, ushort1>();
  test<unsigned short, 2, ushort2>();
  test<unsigned short, 3, ushort3>();
  test<unsigned short, 4, ushort4>();

  test<signed int, 1, int1>();
  test<signed int, 2, int2>();
  test<signed int, 3, int3>();
  test<signed int, 4, int4>();

  test<unsigned int, 1, uint1>();
  test<unsigned int, 2, uint2>();
  test<unsigned int, 3, uint3>();
  test<unsigned int, 4, uint4>();

  test<signed long, 1, long1>();
  test<signed long, 2, long2>();
  test<signed long, 3, long3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<signed long, 4, long4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<signed long, 4, long4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<unsigned long, 1, ulong1>();
  test<unsigned long, 2, ulong2>();
  test<unsigned long, 3, ulong3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<unsigned long, 4, ulong4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<unsigned long, 4, ulong4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<signed long long, 1, longlong1>();
  test<signed long long, 2, longlong2>();
  test<signed long long, 3, longlong3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<signed long long, 4, longlong4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<signed long long, 4, longlong4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<unsigned long long, 1, ulonglong1>();
  test<unsigned long long, 2, ulonglong2>();
  test<unsigned long long, 3, ulonglong3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<unsigned long long, 4, ulonglong4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<unsigned long long, 4, ulonglong4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<float, 1, float1>();
  test<float, 2, float2>();
  test<float, 3, float3>();
  test<float, 4, float4>();

  test<double, 1, double1>();
  test<double, 2, double2>();
  test<double, 3, double3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<double, 4, double4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<double, 4, double4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  // 2. Test invalid combinations

  test<float, 0, void>();
  test<char, 1, void>();
  test<long, 5, void>();
}

int main(int, char**)
{
  return 0;
}
