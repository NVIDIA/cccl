//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   explicit tuple(UTypes&&... u);

/*
    This is testing an extension whereby only Types having an explicit conversion
    from UTypes are bound by the explicit tuple constructor.
*/

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "test_macros.h"

class MoveOnly
{
  MoveOnly(const MoveOnly&);
  MoveOnly& operator=(const MoveOnly&);

  int data_;

public:
  TEST_FUNC explicit MoveOnly(int data = 1)
      : data_(data)
  {}
  TEST_FUNC MoveOnly(MoveOnly&& x)
      : data_(x.data_)
  {
    x.data_ = 0;
  }
  TEST_FUNC MoveOnly& operator=(MoveOnly&& x)
  {
    data_   = x.data_;
    x.data_ = 0;
    return *this;
  }

  TEST_FUNC int get() const
  {
    return data_;
  }

  TEST_FUNC bool operator==(const MoveOnly& x) const
  {
    return data_ == x.data_;
  }
  TEST_FUNC bool operator<(const MoveOnly& x) const
  {
    return data_ < x.data_;
  }
};

int main(int, char**)
{
  {
    cuda::std::tuple<MoveOnly> t = 1;
    unused(t);
  }

  return 0;
}
