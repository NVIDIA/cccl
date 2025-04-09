//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "generated/cp_reduce_async_bulk.h"

#if _LIBCUDACXX_HAS_NVFP16()
#  include "generated/cp_reduce_async_bulk_f16.h"
#endif // _LIBCUDACXX_HAS_NVFP16()

#if _LIBCUDACXX_HAS_NVBF16()
#  include "generated/cp_reduce_async_bulk_bf16.h"
#endif // _LIBCUDACXX_HAS_NVBF16()

int main(int, char**)
{
  return 0;
}
