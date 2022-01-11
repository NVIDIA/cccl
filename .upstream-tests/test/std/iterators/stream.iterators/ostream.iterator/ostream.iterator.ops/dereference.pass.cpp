//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// class ostream_iterator

// ostream_iterator& operator*() const;

#include <cuda/std/iterator>
#include <cuda/std/sstream>
#include <cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    cuda::std::ostringstream os;
    cuda::std::ostream_iterator<int> i(os);
    cuda::std::ostream_iterator<int>& iref = *i;
    assert(&iref == &i);

  return 0;
}
