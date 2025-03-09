//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "host_device_types.h"
#include "test_macros.h"

__device__ void test()
{
  using tuple = cuda::std::tuple<device_only_type>;
  { // default construction
    tuple default_constructed{};
    assert(cuda::std::get<0>(default_constructed) == 0);
  }

  { // value initialization
    tuple value_initialization{device_only_type{42}};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // value initialization
    tuple value_initialization{42};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // copy construction
    tuple input{42};
    tuple dest{input};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // move construction
    tuple input{42};
    tuple dest{cuda::std::move(input)};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // assignment, value to value
    tuple input{42};
    tuple dest{1337};
    dest = input;
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // comparison with tuple
    tuple lhs{42};
    tuple rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // swap
    tuple lhs{42};
    tuple rhs{1337};
    lhs.swap(rhs);
    assert(cuda::std::get<0>(lhs) == 1337);
    assert(cuda::std::get<0>(rhs) == 42);

    swap(lhs, rhs);
    assert(cuda::std::get<0>(lhs) == 42);
    assert(cuda::std::get<0>(rhs) == 1337);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
