//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/attributes.h>

#if _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS()

// 1. Attribute applied to a template class

template <class T>
struct _CCCL_NO_SPECIALIZATIONS Struct
{
  static constexpr bool value = false;
};

// 2. Attribute applied to a template variable

template <class T>
_CCCL_NO_SPECIALIZATIONS inline constexpr bool variable = false;

// 3. Attribute applied to a template function

template <class T>
_CCCL_NO_SPECIALIZATIONS __host__ __device__ T function()
{
  return T{0};
}

#endif // _CCCL_HAS_ATTRIBUTE_NO_SPECIALIZATIONS()

int main(int, char**)
{
  return 0;
}
