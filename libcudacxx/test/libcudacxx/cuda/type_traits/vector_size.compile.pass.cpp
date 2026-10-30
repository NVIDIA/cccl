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

template <class Vec, cuda::std::size_t Size>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::vector_size<Vec>::value == Size);
  static_assert(cuda::vector_size<const Vec>::value == Size);
  static_assert(cuda::vector_size<volatile Vec>::value == Size);
  static_assert(cuda::vector_size<const volatile Vec>::value == Size);

  static_assert(cuda::vector_size_v<Vec> == Size);
  static_assert(cuda::vector_size_v<const Vec> == Size);
  static_assert(cuda::vector_size_v<volatile Vec> == Size);
  static_assert(cuda::vector_size_v<const volatile Vec> == Size);
}

TEST_FUNC void test()
{
  test<char1, 1>();
  test<char2, 2>();
  test<char3, 3>();
  test<char4, 4>();

  test<uchar1, 1>();
  test<uchar2, 2>();
  test<uchar3, 3>();
  test<uchar4, 4>();

  test<short1, 1>();
  test<short2, 2>();
  test<short3, 3>();
  test<short4, 4>();

  test<ushort1, 1>();
  test<ushort2, 2>();
  test<ushort3, 3>();
  test<ushort4, 4>();

  test<int1, 1>();
  test<int2, 2>();
  test<int3, 3>();
  test<int4, 4>();

  test<uint1, 1>();
  test<uint2, 2>();
  test<uint3, 3>();
  test<uint4, 4>();

  test<long1, 1>();
  test<long2, 2>();
  test<long3, 3>();
  test<long4, 4>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, 4>();
  test<long4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulong1, 1>();
  test<ulong2, 2>();
  test<ulong3, 3>();
  test<ulong4, 4>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulong4_16a, 4>();
  test<ulong4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<longlong1, 1>();
  test<longlong2, 2>();
  test<longlong3, 3>();
  test<longlong4, 4>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<longlong4_16a, 4>();
  test<longlong4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulonglong1, 1>();
  test<ulonglong2, 2>();
  test<ulonglong3, 3>();
  test<ulonglong4, 4>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulonglong4_16a, 4>();
  test<ulonglong4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<float1, 1>();
  test<float2, 2>();
  test<float3, 3>();
  test<float4, 4>();

  test<double1, 1>();
  test<double2, 2>();
  test<double3, 3>();
  test<double4, 4>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<double4_16a, 4>();
  test<double4_32a, 4>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<dim3, 3>();

#if _CCCL_HAS_NVFP16()
  test<__half2, 2>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, 2>();
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8x2_e4m3, 2>();
  test<__nv_fp8x4_e4m3, 4>();
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8x2_e5m2, 2>();
  test<__nv_fp8x4_e5m2, 4>();
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8x2_e8m0, 2>();
  test<__nv_fp8x4_e8m0, 4>();
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6x2_e2m3, 2>();
  test<__nv_fp6x4_e2m3, 4>();
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6x2_e3m2, 2>();
  test<__nv_fp6x4_e3m2, 4>();
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4x2_e2m1, 2>();
  test<__nv_fp4x4_e2m1, 4>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  test<float, 0>();
  struct MyStruct
  {};
  test<MyStruct, 0>();
}

int main(int, char**)
{
  test();
  return 0;
}
