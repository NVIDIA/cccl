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
  static_assert(cuda::is_extended_fp_vector_type<Vec>::value == Ref);
  static_assert(cuda::is_extended_fp_vector_type<const Vec>::value == Ref);
  static_assert(cuda::is_extended_fp_vector_type<volatile Vec>::value == Ref);
  static_assert(cuda::is_extended_fp_vector_type<const volatile Vec>::value == Ref);

  static_assert(cuda::is_extended_fp_vector_type_v<Vec> == Ref);
  static_assert(cuda::is_extended_fp_vector_type_v<const Vec> == Ref);
  static_assert(cuda::is_extended_fp_vector_type_v<volatile Vec> == Ref);
  static_assert(cuda::is_extended_fp_vector_type_v<const volatile Vec> == Ref);
}

TEST_FUNC void test()
{
  test<char1, false>();
  test<char2, false>();
  test<char3, false>();
  test<char4, false>();

  test<uchar1, false>();
  test<uchar2, false>();
  test<uchar3, false>();
  test<uchar4, false>();

  test<short1, false>();
  test<short2, false>();
  test<short3, false>();
  test<short4, false>();

  test<ushort1, false>();
  test<ushort2, false>();
  test<ushort3, false>();
  test<ushort4, false>();

  test<int1, false>();
  test<int2, false>();
  test<int3, false>();
  test<int4, false>();

  test<uint1, false>();
  test<uint2, false>();
  test<uint3, false>();
  test<uint4, false>();

  test<long1, false>();
  test<long2, false>();
  test<long3, false>();
  test<long4, false>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, false>();
  test<long4_32a, false>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulong1, false>();
  test<ulong2, false>();
  test<ulong3, false>();
  test<ulong4, false>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulong4_16a, false>();
  test<ulong4_32a, false>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<longlong1, false>();
  test<longlong2, false>();
  test<longlong3, false>();
  test<longlong4, false>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<longlong4_16a, false>();
  test<longlong4_32a, false>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulonglong1, false>();
  test<ulonglong2, false>();
  test<ulonglong3, false>();
  test<ulonglong4, false>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulonglong4_16a, false>();
  test<ulonglong4_32a, false>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<float1, false>();
  test<float2, false>();
  test<float3, false>();
  test<float4, false>();

  test<double1, false>();
  test<double2, false>();
  test<double3, false>();
  test<double4, false>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<double4_16a, false>();
  test<double4_32a, false>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<dim3, false>();

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
