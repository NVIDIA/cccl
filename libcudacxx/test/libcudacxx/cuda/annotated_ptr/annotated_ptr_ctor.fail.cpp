//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: !nvcc

#include "utils.h"

template <typename T, typename P>
__host__ __device__ __noinline__ void test_ctor()
{
  // default ctor, cpy and cpy assignment
  cuda::annotated_ptr<T, P> def;
  def = def;
  cuda::annotated_ptr<T, P> other(def);

  // from ptr
  T* rp = nullptr;
  cuda::annotated_ptr<T, P> a(rp);
  assert(!a);

  // cpy ctor & asign to cv
  cuda::annotated_ptr<const T, P> c(def);
  cuda::annotated_ptr<volatile T, P> d(def);
  cuda::annotated_ptr<const volatile T, P> e(def);
  c = e; // FAIL
  d = d; // FAIL
}

template <typename T, typename P>
__host__ __device__ __noinline__ void test_global_ctor()
{
  test_ctor<T, P>();
}

__host__ __device__ __noinline__ void test_global_ctors()
{
  test_global_ctor<int, cuda::access_property::normal>();
}

int main(int argc, char** argv)
{
  test_global_ctors();
  return 0;
}
