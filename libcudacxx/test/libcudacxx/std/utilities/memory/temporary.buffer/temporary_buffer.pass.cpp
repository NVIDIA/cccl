//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

// template <class T>
//   pair<T*, ptrdiff_t>
//   get_temporary_buffer(ptrdiff_t n);
//
// template <class T>
//   void
//   return_temporary_buffer(T* p);

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/utility>

#include "test_macros.h"

int main(int, char**)
{
  cuda::std::pair<int*, cuda::std::ptrdiff_t> ip = cuda::std::get_temporary_buffer<int>(5);
  assert(ip.first);
  assert(ip.second == 5);
  cuda::std::return_temporary_buffer(ip.first);

  return 0;
}
