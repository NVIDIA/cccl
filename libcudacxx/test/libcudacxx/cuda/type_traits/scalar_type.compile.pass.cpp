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

template <class T, class = void>
inline constexpr bool has_type_alias = false;
template <class T>
inline constexpr bool has_type_alias<T, cuda::std::void_t<typename T::type>> = true;

template <class Vec, class T>
TEST_FUNC constexpr void test()
{
  static_assert(has_type_alias<cuda::scalar_type<Vec>>);
  static_assert(has_type_alias<cuda::scalar_type<const Vec>>);
  static_assert(has_type_alias<cuda::scalar_type<volatile Vec>>);
  static_assert(has_type_alias<cuda::scalar_type<const volatile Vec>>);

  static_assert(cuda::std::is_same_v<typename cuda::scalar_type<Vec>::type, T>);
  static_assert(cuda::std::is_same_v<typename cuda::scalar_type<const Vec>::type, T>);
  static_assert(cuda::std::is_same_v<typename cuda::scalar_type<volatile Vec>::type, T>);
  static_assert(cuda::std::is_same_v<typename cuda::scalar_type<const volatile Vec>::type, T>);

  static_assert(cuda::std::is_same_v<cuda::scalar_type_t<Vec>, T>);
  static_assert(cuda::std::is_same_v<cuda::scalar_type_t<const Vec>, T>);
  static_assert(cuda::std::is_same_v<cuda::scalar_type_t<volatile Vec>, T>);
  static_assert(cuda::std::is_same_v<cuda::scalar_type_t<const volatile Vec>, T>);
}

TEST_FUNC void test()
{
  test<char1, signed char>();
  test<char2, signed char>();
  test<char3, signed char>();
  test<char4, signed char>();

  test<uchar1, unsigned char>();
  test<uchar2, unsigned char>();
  test<uchar3, unsigned char>();
  test<uchar4, unsigned char>();

  test<short1, short>();
  test<short2, short>();
  test<short3, short>();
  test<short4, short>();

  test<ushort1, unsigned short>();
  test<ushort2, unsigned short>();
  test<ushort3, unsigned short>();
  test<ushort4, unsigned short>();

  test<int1, int>();
  test<int2, int>();
  test<int3, int>();
  test<int4, int>();

  test<uint1, unsigned>();
  test<uint2, unsigned>();
  test<uint3, unsigned>();
  test<uint4, unsigned>();

  test<long1, long>();
  test<long2, long>();
  test<long3, long>();
  test<long4, long>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<long4_16a, long>();
  test<long4_32a, long>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulong1, unsigned long>();
  test<ulong2, unsigned long>();
  test<ulong3, unsigned long>();
  test<ulong4, unsigned long>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulong4_16a, unsigned long>();
  test<ulong4_32a, unsigned long>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<longlong1, long long>();
  test<longlong2, long long>();
  test<longlong3, long long>();
  test<longlong4, long long>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<longlong4_16a, long long>();
  test<longlong4_32a, long long>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<ulonglong1, unsigned long long>();
  test<ulonglong2, unsigned long long>();
  test<ulonglong3, unsigned long long>();
  test<ulonglong4, unsigned long long>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<ulonglong4_16a, unsigned long long>();
  test<ulonglong4_32a, unsigned long long>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<float1, float>();
  test<float2, float>();
  test<float3, float>();
  test<float4, float>();

  test<double1, double>();
  test<double2, double>();
  test<double3, double>();
  test<double4, double>();
#if _CCCL_CTK_AT_LEAST(13, 0)
  test<double4_16a, double>();
  test<double4_32a, double>();
#endif // _CCCL_CTK_AT_LEAST(13, 0)

  test<dim3, unsigned>();

#if _CCCL_HAS_NVFP16()
  test<__half2, __half>();
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  test<__nv_bfloat162, __nv_bfloat16>();
#endif // _CCCL_HAS_NVBF16()

#if _CCCL_HAS_NVFP8_E4M3()
  test<__nv_fp8x2_e4m3, __nv_fp8_e4m3>();
  test<__nv_fp8x4_e4m3, __nv_fp8_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()

#if _CCCL_HAS_NVFP8_E5M2()
  test<__nv_fp8x2_e5m2, __nv_fp8_e5m2>();
  test<__nv_fp8x4_e5m2, __nv_fp8_e5m2>();
#endif // _CCCL_HAS_NVFP8_E5M2()

#if _CCCL_HAS_NVFP8_E8M0()
  test<__nv_fp8x2_e8m0, __nv_fp8_e8m0>();
  test<__nv_fp8x4_e8m0, __nv_fp8_e8m0>();
#endif // _CCCL_HAS_NVFP8_E8M0()

#if _CCCL_HAS_NVFP6_E2M3()
  test<__nv_fp6x2_e2m3, __nv_fp6_e2m3>();
  test<__nv_fp6x4_e2m3, __nv_fp6_e2m3>();
#endif // _CCCL_HAS_NVFP6_E2M3()

#if _CCCL_HAS_NVFP6_E3M2()
  test<__nv_fp6x2_e3m2, __nv_fp6_e3m2>();
  test<__nv_fp6x4_e3m2, __nv_fp6_e3m2>();
#endif // _CCCL_HAS_NVFP6_E3M2()

#if _CCCL_HAS_NVFP4_E2M1()
  test<__nv_fp4x2_e2m1, __nv_fp4_e2m1>();
  test<__nv_fp4x4_e2m1, __nv_fp4_e2m1>();
#endif // _CCCL_HAS_NVFP4_E2M1()

  static_assert(!has_type_alias<cuda::scalar_type<float>>);
  static_assert(!has_type_alias<cuda::scalar_type<const float>>);
  static_assert(!has_type_alias<cuda::scalar_type<volatile float>>);
  static_assert(!has_type_alias<cuda::scalar_type<const volatile float>>);

  struct MyStruct
  {};
  static_assert(!has_type_alias<cuda::scalar_type<MyStruct>>);
  static_assert(!has_type_alias<cuda::scalar_type<const MyStruct>>);
  static_assert(!has_type_alias<cuda::scalar_type<volatile MyStruct>>);
  static_assert(!has_type_alias<cuda::scalar_type<const volatile MyStruct>>);
}

int main(int, char**)
{
  test();
  return 0;
}
