//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++11, c++14

#include <cuda/std/__mdspan/aligned_accessor.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include <test_macros.h>

template <class ElementType>
__host__ __device__ void take_default_accessor_generic(cuda::std::default_accessor<ElementType>)
{}

template <class ElementType>
cuda::std::enable_if_t<cuda::std::is_const_v<ElementType>>
  __host__ __device__ take_default_accessor_generic_const(cuda::std::default_accessor<ElementType>)
{}

__host__ __device__ void take_default_accessor(cuda::std::default_accessor<int>) {}

__host__ __device__ void take_default_accessor_const(cuda::std::default_accessor<const int>) {}

__host__ __device__ bool test()
{
  using T = int;
  using E = cuda::std::extents<size_t, 2>;
  using L = cuda::std::layout_right;
  using A = cuda::std::aligned_accessor<T, sizeof(T)>;
  cuda::std::array<T, 2> d{42, 43};
  cuda::std::mdspan<T, E, L, A> md(d.data(), 2);
  assert(md(0) == 42);
  assert(md(1) == 43);

  A aligned_non_const;
  cuda::std::default_accessor<int> acc1{aligned_non_const};
  cuda::std::default_accessor<int> acc2 = aligned_non_const;
  unused(acc1);
  unused(acc2);
  take_default_accessor(aligned_non_const);

  cuda::std::aligned_accessor<const T, sizeof(T)> aligned_const;
  cuda::std::default_accessor<const T> acc3{aligned_const};
  cuda::std::default_accessor<const T> acc4 = aligned_const;
  unused(acc3);
  unused(acc4);
  take_default_accessor_const(aligned_const);
  return true;
}

int main(int, char**)
{
  test();
  return 0;
}
