//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cstddef>
#include <cuda/type_traits>

#include <test_macros.h>

template <class T, cuda::std::size_t Size, class Ref>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::std::is_same_v<typename cuda::vector_type<T, Size>::type, Ref>);
  static_assert(cuda::std::is_same_v<cuda::vector_type_t<T, Size>, Ref>);
}

TEST_FUNC void test()
{
  test<signed char, 1, char1>();
  test<signed char, 2, char2>();
  test<signed char, 3, char3>();
  test<signed char, 4, char4>();

  test<unsigned char, 1, uchar1>();
  test<unsigned char, 2, uchar2>();
  test<unsigned char, 3, uchar3>();
  test<unsigned char, 4, uchar4>();

  test<short, 1, short1>();
  test<short, 2, short2>();
  test<short, 3, short3>();
  test<short, 4, short4>();

  test<unsigned short, 1, ushort1>();
  test<unsigned short, 2, ushort2>();
  test<unsigned short, 3, ushort3>();
  test<unsigned short, 4, ushort4>();

  test<int, 1, int1>();
  test<int, 2, int2>();
  test<int, 3, int3>();
  test<int, 4, int4>();

  test<unsigned, 1, uint1>();
  test<unsigned, 2, uint2>();
  test<unsigned, 3, uint3>();
  test<unsigned, 4, uint4>();

  test<long, 1, long1>();
  test<long, 2, long2>();
  test<long, 3, long3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long, 4, long4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<long, 4, long4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<unsigned long, 1, ulong1>();
  test<unsigned long, 2, ulong2>();
  test<unsigned long, 3, ulong3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<unsigned long, 4, ulong4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<unsigned long, 4, ulong4>();
#endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

  test<long long, 1, longlong1>();
  test<long long, 2, longlong2>();
  test<long long, 3, longlong3>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long long, 4, longlong4_32a>();
#else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  test<long long, 4, longlong4>();
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

#if _CCCL_HAS_NVFP16()
  test<__half, 2, __half2>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat16, 2, __nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8_e4m3, 2, __nv_fp8x2_e4m3>();
  test<__nv_fp8_e4m3, 4, __nv_fp8x4_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8_e5m2, 2, __nv_fp8x2_e5m2>();
  test<__nv_fp8_e5m2, 4, __nv_fp8x4_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8_e8m0, 2, __nv_fp8x2_e8m0>();
  test<__nv_fp8_e8m0, 4, __nv_fp8x4_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6_e2m3, 2, __nv_fp6x2_e2m3>();
  test<__nv_fp6_e2m3, 4, __nv_fp6x4_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6_e3m2, 2, __nv_fp6x2_e3m2>();
  test<__nv_fp6_e3m2, 4, __nv_fp6x4_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4_e2m1, 2, __nv_fp4x2_e2m1>();
  test<__nv_fp4_e2m1, 4, __nv_fp4x4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  test<float, 0, void>();
  test<float, 5, void>();
  struct MyStruct
  {};
  test<MyStruct, 1, void>();
}

int main(int, char**)
{
  test();
  return 0;
}
