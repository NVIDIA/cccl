//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// <memory>

// template <ObjectType T> constexpr T* addressof(T& r);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

#if defined(_CCCL_BUILTIN_ADDRESSOF) || defined(__NVCOMPILER)
struct Pointer
{
  __host__ __device__ constexpr Pointer(void* v)
      : value(v)
  {}
  void* value;
};

struct A
{
  __host__ __device__ constexpr A()
      : n(42)
  {}
  __host__ __device__ void operator&() const {}
  int n;
};

constexpr int global_integer = 0;
static_assert(cuda::std::addressof(global_integer) == &global_integer, "");

constexpr double global_double = 0.0;
static_assert(cuda::std::addressof(global_double) == &global_double, "");

#  ifndef __CUDA_ARCH__ // fails in __cudaRegisterVariable
constexpr A global_struct{};
constexpr const A* address = cuda::std::addressof(global_struct);
static_assert(&(address->n) == &(global_struct.n), "");
#  endif
#endif

int main(int, char**)
{
  return 0;
}
