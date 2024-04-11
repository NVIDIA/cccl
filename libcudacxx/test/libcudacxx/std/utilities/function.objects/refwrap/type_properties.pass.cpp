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

// Test that reference wrapper meets the requirements of CopyConstructible and
// CopyAssignable, and TriviallyCopyable (starting in C++14).

// #include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#ifdef _LIBCUDACXX_HAS_
#  include <cuda/std/string>
#endif

#include "test_macros.h"

class MoveOnly
{
  __host__ __device__ MoveOnly(const MoveOnly&);
  __host__ __device__ MoveOnly& operator=(const MoveOnly&);

  int data_;

public:
  __host__ __device__ MoveOnly(int data = 1)
      : data_(data)
  {}
  __host__ __device__ MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  __host__ __device__ MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  __host__ __device__ int get() const
  {
    return data_;
  }
};

template <class T>
__host__ __device__ void test()
{
  typedef cuda::std::reference_wrapper<T> Wrap;
  static_assert(cuda::std::is_copy_constructible<Wrap>::value, "");
  static_assert(cuda::std::is_copy_assignable<Wrap>::value, "");
#if TEST_STD_VER >= 2014
  static_assert(cuda::std::is_trivially_copyable<Wrap>::value, "");
#endif
}

int main(int, char**)
{
  test<int>();
  test<double>();
#ifdef _LIBCUDACXX_HAS_
  test<cuda::std::string>();
#endif
  test<MoveOnly>();

  return 0;
}
