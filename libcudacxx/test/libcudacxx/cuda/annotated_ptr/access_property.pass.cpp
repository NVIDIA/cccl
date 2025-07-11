//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "utils.h"

template <typename T>
__host__ __device__ __noinline__ void test_global_implicit_property(T ap, cudaAccessProperty cp)
{
  // Test implicit conversions
  cudaAccessProperty v = ap;
  assert(cp == v);

  // Test default, copy constructor, and copy-assignent
  cuda::access_property o(ap);
  cuda::access_property d;
  d = ap;

  // Test explicit conversion to i64
  uint64_t x = (uint64_t) o;
  uint64_t y = (uint64_t) d;
  assert(x == y);
}

__host__ __device__ __noinline__ void test_global()
{
  cuda::access_property o(cuda::access_property::global{});
  uint64_t x = (uint64_t) o;
  unused(x);
}

__host__ __device__ __noinline__ void test_shared()
{
  (void) cuda::access_property::shared{};
}

static_assert(sizeof(cuda::access_property::shared) == 1);
static_assert(sizeof(cuda::access_property::global) == 1);
static_assert(sizeof(cuda::access_property::persisting) == 1);
static_assert(sizeof(cuda::access_property::normal) == 1);
static_assert(sizeof(cuda::access_property::streaming) == 1);
static_assert(sizeof(cuda::access_property) == 8);

static_assert(alignof(cuda::access_property::shared) == 1);
static_assert(alignof(cuda::access_property::global) == 1);
static_assert(alignof(cuda::access_property::persisting) == 1);
static_assert(alignof(cuda::access_property::normal) == 1);
static_assert(alignof(cuda::access_property::streaming) == 1);
static_assert(alignof(cuda::access_property) == 8);

int main(int argc, char** argv)
{
  test_global_implicit_property(cuda::access_property::normal{}, cudaAccessProperty::cudaAccessPropertyNormal);
  test_global_implicit_property(cuda::access_property::streaming{}, cudaAccessProperty::cudaAccessPropertyStreaming);
  test_global_implicit_property(cuda::access_property::persisting{}, cudaAccessProperty::cudaAccessPropertyPersisting);

  test_global();
  test_shared();
  return 0;
}
