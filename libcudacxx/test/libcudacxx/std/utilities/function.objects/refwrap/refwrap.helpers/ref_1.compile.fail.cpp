//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <functional>

// reference_wrapper

// template <ObjectType T> reference_wrapper<T> ref(T& t);

// Don't allow binding to a temp

// #include <cuda/std/functional>
#include <cuda/std/utility>

struct A
{};

__host__ __device__ const A source()
{
  return A();
}

int main(int, char**)
{
  cuda::std::reference_wrapper<const A> r = cuda::std::ref(source());
  (void) r;

  return 0;
}
