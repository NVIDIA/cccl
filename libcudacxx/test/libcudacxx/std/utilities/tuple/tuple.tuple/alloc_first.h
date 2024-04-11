//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ALLOC_FIRST_H
#define ALLOC_FIRST_H

#include <cuda/std/cassert>

#include "allocators.h"

struct alloc_first
{
  STATIC_MEMBER_VAR(allocator_constructed, bool);

  typedef A1<int> allocator_type;

  int data_;

  __host__ __device__ alloc_first()
      : data_(0)
  {}
  __host__ __device__ alloc_first(int d)
      : data_(d)
  {}
  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a)
      : data_(0)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a, int d)
      : data_(d)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ alloc_first(cuda::std::allocator_arg_t, const A1<int>& a, const alloc_first& d)
      : data_(d.data_)
  {
    assert(a.id() == 5);
    allocator_constructed() = true;
  }

  __host__ __device__ ~alloc_first()
  {
    data_ = -1;
  }

  __host__ __device__ friend bool operator==(const alloc_first& x, const alloc_first& y)
  {
    return x.data_ == y.data_;
  }
  __host__ __device__ friend bool operator<(const alloc_first& x, const alloc_first& y)
  {
    return x.data_ < y.data_;
  }
};

#endif // ALLOC_FIRST_H
