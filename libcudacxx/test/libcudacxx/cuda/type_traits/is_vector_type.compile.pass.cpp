//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/type_traits>

#include <test_macros.h>

template <class Vec, bool Ref>
TEST_FUNC constexpr void test()
{
  static_assert(cuda::is_vector_type<Vec>::value == Ref);
  static_assert(cuda::is_vector_type<const Vec>::value == Ref);
  static_assert(cuda::is_vector_type<volatile Vec>::value == Ref);
  static_assert(cuda::is_vector_type<const volatile Vec>::value == Ref);

  static_assert(cuda::is_vector_type_v<Vec> == Ref);
  static_assert(cuda::is_vector_type_v<const Vec> == Ref);
  static_assert(cuda::is_vector_type_v<volatile Vec> == Ref);
  static_assert(cuda::is_vector_type_v<const volatile Vec> == Ref);
}

TEST_FUNC void test()
{
  test<char1, true>();
  test<char2, true>();
  test<char3, true>();
  test<char4, true>();

  test<uchar1, true>();
  test<uchar2, true>();
  test<uchar3, true>();
  test<uchar4, true>();

  test<short1, true>();
  test<short2, true>();
  test<short3, true>();
  test<short4, true>();

  test<ushort1, true>();
  test<ushort2, true>();
  test<ushort3, true>();
  test<ushort4, true>();

  test<int1, true>();
  test<int2, true>();
  test<int3, true>();
  test<int4, true>();

  test<uint1, true>();
  test<uint2, true>();
  test<uint3, true>();
  test<uint4, true>();

  test<long1, true>();
  test<long2, true>();
  test<long3, true>();
  test<long4, true>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, true>();
  test<long4_32a, true>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulong1, true>();
  test<ulong2, true>();
  test<ulong3, true>();
  test<ulong4, true>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulong4_16a, true>();
  test<ulong4_32a, true>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<longlong1, true>();
  test<longlong2, true>();
  test<longlong3, true>();
  test<longlong4, true>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<longlong4_16a, true>();
  test<longlong4_32a, true>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulonglong1, true>();
  test<ulonglong2, true>();
  test<ulonglong3, true>();
  test<ulonglong4, true>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulonglong4_16a, true>();
  test<ulonglong4_32a, true>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<float1, true>();
  test<float2, true>();
  test<float3, true>();
  test<float4, true>();

  test<double1, true>();
  test<double2, true>();
  test<double3, true>();
  test<double4, true>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<double4_16a, true>();
  test<double4_32a, true>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<dim3, true>();

#if _CCCL_HAS_NVFP16()
  test<__half2, true>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, true>();
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8x2_e4m3, true>();
  test<__nv_fp8x4_e4m3, true>();
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8x2_e5m2, true>();
  test<__nv_fp8x4_e5m2, true>();
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8x2_e8m0, true>();
  test<__nv_fp8x4_e8m0, true>();
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6x2_e2m3, true>();
  test<__nv_fp6x4_e2m3, true>();
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6x2_e3m2, true>();
  test<__nv_fp6x4_e3m2, true>();
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4x2_e2m1, true>();
  test<__nv_fp4x4_e2m1, true>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  test<float, false>();
  struct MyStruct
  {};
  test<MyStruct, false>();
}

int main(int, char**)
{
  test();
  return 0;
}
