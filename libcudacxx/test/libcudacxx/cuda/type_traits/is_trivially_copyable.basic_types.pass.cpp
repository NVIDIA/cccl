//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/type_traits>

#include "cuda_fp_types.h"
#include "test_macros.h"

template <class T>
TEST_FUNC void test_is_trivially_copyable()
{
  static_assert(cuda::is_trivially_copyable<T>::value);
  static_assert(cuda::is_trivially_copyable<const T>::value);
  static_assert(cuda::is_trivially_copyable_v<T>);
  static_assert(cuda::is_trivially_copyable_v<const T>);
}

TEST_FUNC void test_single_types()
{
  // standard trivially copyable types
  test_is_trivially_copyable<int>();
  test_is_trivially_copyable<float>();
  test_is_trivially_copyable<double>();

#if _CCCL_HAS_CTK()
  // CUDA vector types
  test_is_trivially_copyable<char1>();
  test_is_trivially_copyable<char2>();
  test_is_trivially_copyable<char3>();
  test_is_trivially_copyable<char4>();
  test_is_trivially_copyable<uchar1>();
  test_is_trivially_copyable<uchar2>();
  test_is_trivially_copyable<uchar3>();
  test_is_trivially_copyable<uchar4>();
  test_is_trivially_copyable<short1>();
  test_is_trivially_copyable<short2>();
  test_is_trivially_copyable<short3>();
  test_is_trivially_copyable<short4>();
  test_is_trivially_copyable<ushort1>();
  test_is_trivially_copyable<ushort2>();
  test_is_trivially_copyable<ushort3>();
  test_is_trivially_copyable<ushort4>();
  test_is_trivially_copyable<int1>();
  test_is_trivially_copyable<int2>();
  test_is_trivially_copyable<int3>();
  test_is_trivially_copyable<int4>();
  test_is_trivially_copyable<uint1>();
  test_is_trivially_copyable<uint2>();
  test_is_trivially_copyable<uint3>();
  test_is_trivially_copyable<uint4>();
  test_is_trivially_copyable<long1>();
  test_is_trivially_copyable<long2>();
  test_is_trivially_copyable<long3>();
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  test_is_trivially_copyable<long4>();
  _CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<long4_16a>();
  test_is_trivially_copyable<long4_32a>();
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<ulong1>();
  test_is_trivially_copyable<ulong2>();
  test_is_trivially_copyable<ulong3>();
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  test_is_trivially_copyable<ulong4>();
  _CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<ulong4_16a>();
  test_is_trivially_copyable<ulong4_32a>();
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<longlong1>();
  test_is_trivially_copyable<longlong2>();
  test_is_trivially_copyable<longlong3>();
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  test_is_trivially_copyable<longlong4>();
  _CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<longlong4_16a>();
  test_is_trivially_copyable<longlong4_32a>();
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<ulonglong1>();
  test_is_trivially_copyable<ulonglong2>();
  test_is_trivially_copyable<ulonglong3>();
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  test_is_trivially_copyable<ulonglong4>();
  _CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<ulonglong4_16a>();
  test_is_trivially_copyable<ulonglong4_32a>();
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<float1>();
  test_is_trivially_copyable<float2>();
  test_is_trivially_copyable<float3>();
  test_is_trivially_copyable<float4>();
  test_is_trivially_copyable<double1>();
  test_is_trivially_copyable<double2>();
  test_is_trivially_copyable<double3>();
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  test_is_trivially_copyable<double4>();
  _CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
  test_is_trivially_copyable<double4_16a>();
  test_is_trivially_copyable<double4_32a>();
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
#endif // _CCCL_HAS_CTK()

  // extended floating point scalar types
#if _CCCL_HAS_NVFP16()
  test_is_trivially_copyable<__half>();
  test_is_trivially_copyable<__half2>();
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  test_is_trivially_copyable<__nv_bfloat16>();
  test_is_trivially_copyable<__nv_bfloat162>();
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8_E4M3()
  test_is_trivially_copyable<__nv_fp8_e4m3>();
  test_is_trivially_copyable<__nv_fp8x2_e4m3>();
#endif // _CCCL_HAS_NVFP8_E4M3()
}

int main(int, char**)
{
  test_single_types();
  return 0;
}
