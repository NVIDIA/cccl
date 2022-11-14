//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <ObjectType T> constexpr T* addressof(T& r);

#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

#if defined(_LIBCUDACXX_ADDRESSOF) || defined(__NVCOMPILER)
struct Pointer {
  __host__ __device__ constexpr Pointer(void* v) : value(v) {}
  void* value;
};

struct A
{
    __host__ __device__ constexpr A() : n(42) {}
    __host__ __device__ void operator&() const { }
    int n; 
};

__device__ constexpr int i = 0;
static_assert(cuda::std::addressof(i) == &i, "");

__device__ constexpr double d = 0.0;
static_assert(cuda::std::addressof(d) == &d, "");
 
#ifndef __CUDA_ARCH__ // fails in __cudaRegisterVariable
__device__ constexpr A a{};
__device__ constexpr const A* ap = cuda::std::addressof(a);
static_assert(&(ap->n) == &(a.n), "");
#endif
#endif

int main(int, char**)
{
  return 0;
}
