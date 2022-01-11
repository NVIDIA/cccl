//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// back_insert_iterator

// requires CopyConstructible<Cont::value_type>
//   back_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/iterator>
#include <cuda/std/vector>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class C>
void
test(C c)
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
    Copyable() : data_(0) {}
    ~Copyable() {data_ = -1;}

    friend bool operator==(const Copyable& x, const Copyable& y)
        {return x.data_ == y.data_;}
};

int main(int, char**)
{
    test(cuda::std::vector<Copyable>());

  return 0;
}
