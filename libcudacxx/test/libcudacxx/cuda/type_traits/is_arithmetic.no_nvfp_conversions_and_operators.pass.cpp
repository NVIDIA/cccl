//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/type_traits>

int main(int, char**)
{
#if _CCCL_HAS_NVFP16()
  static_assert(!cuda::is_arithmetic_v<__half>);
  static_assert(!cuda::is_arithmetic<__half>::value);
#endif // _CCCL_HAS_NVFP16()

#if _CCCL_HAS_NVBF16()
  static_assert(!cuda::is_arithmetic_v<__nv_bfloat16>);
  static_assert(!cuda::is_arithmetic<__nv_bfloat16>::value);
#endif // _CCCL_HAS_NVBF16()

  return 0;
}
