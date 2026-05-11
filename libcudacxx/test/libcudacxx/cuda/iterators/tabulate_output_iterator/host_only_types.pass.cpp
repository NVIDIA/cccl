//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

struct host_functor
{
  // nvbug6163849: if the struct is empty NVCC fails with a host device access warning
  int val_ = 0;

  void operator()(const cuda::std::ptrdiff_t val, const int expected) const noexcept
  {
    assert(val == expected); // asserts that the assigned value matches the index
  }

  host_functor() {}
  host_functor(const host_functor&) {}
  host_functor(host_functor&&) {}
  host_functor& operator=(const host_functor&)
  {
    return *this;
  }
  host_functor& operator=(host_functor&&)
  {
    return *this;
  }
  ~host_functor() {}
};

void test()
{
  {
    using tabulate_output_iterator = cuda::tabulate_output_iterator<host_functor, int>;

    const tabulate_output_iterator default_constructed{};
    tabulate_output_iterator value_constructed{host_functor{}};

    tabulate_output_iterator copy_constructed{default_constructed};
    tabulate_output_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] tabulate_output_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] tabulate_output_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);

    [[maybe_unused]] tabulate_output_iterator func_index_constructed{host_functor{}, 42};
  }

  cuda::tabulate_output_iterator iter1{host_functor{}, 100};
  const cuda::tabulate_output_iterator iter2{host_functor{}, 101};
  assert(iter1 != iter2);

  {
    assert(++iter1 == iter2);
    assert(--iter1 != iter2);
  }

  {
    assert(iter1++ != iter2);
    assert(iter1-- == iter2);
  }

  {
    assert(iter1 + 1 == iter2);
    assert(1 + iter1 == iter2);
    assert(iter1 - 1 != iter2);
    assert(iter2 - iter1 == 1);
  }

  {
    iter1 += 1;
    assert(iter1 == iter2);
    iter1 -= 1;
    assert(iter1 != iter2);
  }

  {
    iter1[1] = 1 + 100;
    *iter1   = 0 + 100;

    iter2[1] = 1 + 101;
    *iter2   = 0 + 101;
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
