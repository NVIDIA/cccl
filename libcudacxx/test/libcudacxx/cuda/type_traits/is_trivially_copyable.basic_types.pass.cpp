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
  test_is_trivially_copyable<int4>();
  test_is_trivially_copyable<double2>();
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
