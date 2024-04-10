//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/iterator>

// insert_iterator

// requires CopyConstructible<Cont::value_type>
//   insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/iterator>
#include <cuda/std/utility>
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/cassert>
#  include <cuda/std/memory>
#  include <cuda/std/vector>

#  include "test_macros.h"

template <class C>
__host__ __device__ void
test(C c1,
     typename C::difference_type j,
     typename C::value_type x1,
     typename C::value_type x2,
     typename C::value_type x3,
     const C& c2)
{
  cuda::std::insert_iterator<C> q(c1, c1.begin() + j);
  q = cuda::std::move(x1);
  q = cuda::std::move(x2);
  q = cuda::std::move(x3);
  assert(c1 == c2);
}

template <class C>
__host__ __device__ void
insert3at(C& c, typename C::iterator i, typename C::value_type x1, typename C::value_type x2, typename C::value_type x3)
{
  i = c.insert(i, cuda::std::move(x1));
  i = c.insert(++i, cuda::std::move(x2));
  c.insert(++i, cuda::std::move(x3));
}

struct do_nothing
{
  __host__ __device__ void operator()(void*) const {}
};

int main(int, char**)
{
  {
    typedef cuda::std::unique_ptr<int, do_nothing> Ptr;
    typedef cuda::std::vector<Ptr> C;
    C c1;
    int x[6] = {0};
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(Ptr(x + i));
    }
    C c2;
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(Ptr(x + i));
    }
    insert3at(c2, c2.begin(), Ptr(x + 3), Ptr(x + 4), Ptr(x + 5));
    test(cuda::std::move(c1), 0, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(Ptr(x + i));
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(Ptr(x + i));
    }
    insert3at(c2, c2.begin() + 1, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5));
    test(cuda::std::move(c1), 1, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(Ptr(x + i));
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(Ptr(x + i));
    }
    insert3at(c2, c2.begin() + 2, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5));
    test(cuda::std::move(c1), 2, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5), c2);
    c1.clear();
    for (int i = 0; i < 3; ++i)
    {
      c1.push_back(Ptr(x + i));
    }
    c2.clear();
    for (int i = 0; i < 3; ++i)
    {
      c2.push_back(Ptr(x + i));
    }
    insert3at(c2, c2.begin() + 3, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5));
    test(cuda::std::move(c1), 3, Ptr(x + 3), Ptr(x + 4), Ptr(x + 5), c2);
  }

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
