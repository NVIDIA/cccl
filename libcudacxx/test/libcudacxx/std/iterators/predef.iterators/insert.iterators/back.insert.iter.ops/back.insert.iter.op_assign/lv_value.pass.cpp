//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error calling a __host__ __device__ function from a __host__ __device__ __tile__ function is not allowed

// <cuda/std/iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/cassert>
#include <cuda/std/inplace_vector>
#include <cuda/std/iterator>

#include "test_macros.h"

template <class C>
TEST_HOST_DEVICE_FUNC void test(C c)
{
  const typename C::value_type v = typename C::value_type();
  cuda::std::back_insert_iterator<C> i(c);
  i = v;
  assert(c.back() == v);
}

class Copyable
{
  int data_;

public:
  TEST_HOST_DEVICE_FUNC Copyable()
      : data_(0)
  {}
  TEST_HOST_DEVICE_FUNC ~Copyable()
  {
    data_ = -1;
  }

  TEST_HOST_DEVICE_FUNC friend bool operator==(const Copyable& x, const Copyable& y)
  {
    return x.data_ == y.data_;
  }
};

int main(int, char**)
{
  test(cuda::std::inplace_vector<Copyable, 3>());

  return 0;
}
