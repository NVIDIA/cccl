//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/cassert>
#  include <cuda/std/vector>

#  include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
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
  __host__ __device__ Copyable()
      : data_(0)
  {}
  __host__ __device__ ~Copyable()
  {
    data_ = -1;
  }

  __host__ __device__ friend bool operator==(const Copyable& x, const Copyable& y)
  {
    return x.data_ == y.data_;
  }
};

int main(int, char**)
{
  test(cuda::std::vector<Copyable>());

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
