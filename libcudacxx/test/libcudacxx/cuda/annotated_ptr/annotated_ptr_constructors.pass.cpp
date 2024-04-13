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
// UNSUPPORTED: nvrtc

#include "utils.h"

template <typename T, typename P>
__host__ __device__ __noinline__ void test_ctor()
{
  // default ctor, cpy and cpy assignment
  cuda::annotated_ptr<T, P> def;
  {
    cuda::annotated_ptr<T, P> temp;
    temp = def;
    unused(temp);
  }
  cuda::annotated_ptr<T, P> other(def);
  unused(other);

  // from ptr
  T* rp = nullptr;
  cuda::annotated_ptr<T, P> a(rp);
  assert(!a);

  // cpy ctor & asign to cv
  cuda::annotated_ptr<const T, P> c(def);
  cuda::annotated_ptr<volatile T, P> d(def);
  cuda::annotated_ptr<const volatile T, P> e(def);
  c = def;
  d = def;
  e = def;

  // from c|v to c|v|cv
  cuda::annotated_ptr<const T, P> f(c);
  cuda::annotated_ptr<volatile T, P> g(d);
  cuda::annotated_ptr<const volatile T, P> h(e);
  f = c;
  g = d;
  h = e;
  unused(f, g, h);

  // to cv
  cuda::annotated_ptr<const volatile T, P> i(c);
  cuda::annotated_ptr<const volatile T, P> j(d);
  i = c;
  j = d;
}

template <typename T, typename P>
__host__ __device__ __noinline__ void test_global_ctor()
{
  test_ctor<T, P>();

  // from ptr + prop
  T* rp = nullptr;
  P p;
  cuda::annotated_ptr<T, cuda::access_property> a(rp, p);
  cuda::annotated_ptr<const T, cuda::access_property> b(rp, p);
  cuda::annotated_ptr<volatile T, cuda::access_property> c(rp, p);
  cuda::annotated_ptr<const volatile T, cuda::access_property> d(rp, p);
}

__host__ __device__ __noinline__ void test_global_ctors()
{
  test_global_ctor<int, cuda::access_property::normal>();
  test_global_ctor<int, cuda::access_property::streaming>();
  test_global_ctor<int, cuda::access_property::persisting>();
  test_global_ctor<int, cuda::access_property::global>();
  test_global_ctor<int, cuda::access_property>();
  test_ctor<int, cuda::access_property::shared>();
}

int main(int argc, char** argv)
{
  test_global_ctors();
  return 0;
}
