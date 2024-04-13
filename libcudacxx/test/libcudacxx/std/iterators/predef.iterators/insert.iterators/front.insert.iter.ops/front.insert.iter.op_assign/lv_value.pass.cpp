//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#  include <cuda/std/list>

#  include "nasty_containers.h"
#endif // _LIBCUDACXX_HAS_LIST

#include "test_macros.h"

template <class C>
__host__ __device__ void test(C c)
{
  const typename C::value_type v = typename C::value_type();
  cuda::std::front_insert_iterator<C> i(c);
  i = v;
  assert(c.front() == v);
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
#if defined(_LIBCUDACXX_HAS_LIST)
  test(cuda::std::list<Copyable>());
  test(nasty_list<Copyable>());
#endif

  return 0;
}
