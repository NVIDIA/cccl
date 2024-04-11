//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// #include <cuda/std/functional>
#include <cuda/std/utility>

#include "test_macros.h"

struct Incomplete;
template <class T>
struct Holder
{
  T t;
};
typedef Holder<Incomplete>* Ptr;

__host__ __device__ Ptr no_args()
{
  return nullptr;
}
__host__ __device__ Ptr one_arg(Ptr p)
{
  return p;
}
__host__ __device__ Ptr two_args(Ptr p, Ptr)
{
  return p;
}
__host__ __device__ Ptr three_args(Ptr p, Ptr, Ptr)
{
  return p;
}

__host__ __device__ void one_arg_void(Ptr) {}

int main(int, char**)
{
  Ptr x        = nullptr;
  const Ptr cx = nullptr;
  cuda::std::ref(no_args)();
  cuda::std::ref(one_arg)(x);
  cuda::std::ref(one_arg)(cx);
  cuda::std::ref(two_args)(x, x);
  cuda::std::ref(two_args)(x, cx);
  cuda::std::ref(two_args)(cx, x);
  cuda::std::ref(two_args)(cx, cx);
  cuda::std::ref(three_args)(x, x, x);
  cuda::std::ref(three_args)(x, x, cx);
  cuda::std::ref(three_args)(x, cx, x);
  cuda::std::ref(three_args)(cx, x, x);
  cuda::std::ref(three_args)(x, cx, cx);
  cuda::std::ref(three_args)(cx, x, cx);
  cuda::std::ref(three_args)(cx, cx, x);
  cuda::std::ref(three_args)(cx, cx, cx);
  cuda::std::ref(one_arg_void)(x);
  cuda::std::ref(one_arg_void)(cx);

  return 0;
}
