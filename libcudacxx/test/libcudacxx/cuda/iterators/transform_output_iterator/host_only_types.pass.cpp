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

#include "host_device_types.h"
#include "test_macros.h"

struct host_functor
{
  // nvbug6163849: if the struct is empty NVCC fails with a host device access warning
  int val_ = 0;

  int operator()(const int val) const noexcept
  {
    return val + 42;
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
  host_only_container vec{};

  {
    using Iter                      = typename host_only_container::iterator;
    using transform_output_iterator = cuda::transform_output_iterator<host_functor, Iter>;

    const transform_output_iterator default_constructed{};
    transform_output_iterator value_constructed{vec.begin(), host_functor{}};

    transform_output_iterator copy_constructed{default_constructed};
    transform_output_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] transform_output_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] transform_output_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);
  }

  cuda::transform_output_iterator iter1{vec.begin(), host_functor{}};
  const cuda::transform_output_iterator iter2{vec.begin() + 1, host_functor{}};
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
    iter1[1] = 1337;
    assert(vec[1] == 1337 + 42);

    *iter1 = 1337;
    assert(vec[0] == 1337 + 42);

    iter2[1] = 1;
    assert(vec[2] == 1 + 42);

    *iter2 = 1;
    assert(vec[1] == 1 + 42);
  }

  {
    assert(iter1.base() == vec.begin());
    assert(iter2.base() == vec.begin() + 1);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
