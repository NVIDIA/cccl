//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// front_insert_iterator

// front_insert_iterator<Cont>&
//   operator=(const Cont::value_type& value);

#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_LIST)
#include <cuda/std/list>
#endif
#include <cuda/std/cassert>
#include "nasty_containers.h"

#include "test_macros.h"

template <class C>
void
test(C c)
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
    Copyable() : data_(0) {}
    ~Copyable() {data_ = -1;}

    friend bool operator==(const Copyable& x, const Copyable& y)
        {return x.data_ == y.data_;}
};

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_LIST)
    test(cuda::std::list<Copyable>());
    test(nasty_list<Copyable>());
#endif

  return 0;
}
