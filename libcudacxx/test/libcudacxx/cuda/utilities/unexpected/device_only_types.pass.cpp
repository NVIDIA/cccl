//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/expected>

#include "host_device_types.h"
#include "test_macros.h"

__device__ void test()
{
  using unexpected = cuda::std::unexpected<device_only_type>;
  { // in_place zero initialization
    unexpected in_place_zero_initialization{cuda::std::in_place};
    assert(in_place_zero_initialization.error() == 0);
  }

  { // in_place initialization
    unexpected in_place_initialization{cuda::std::in_place, 42};
    assert(in_place_initialization.error() == 42);
  }

  { // value initialization
    unexpected value_initialization{42};
    assert(value_initialization.error() == 42);
  }

  { // initializer_list initialization
    unexpected init_list_initialization{cuda::std::in_place, cuda::std::initializer_list<int>{}, 42};
    assert(init_list_initialization.error() == 42);
  }

  { // copy construction
    unexpected input{42};
    unexpected dest{input};
    assert(dest.error() == 42);
  }

  { // move construction
    unexpected input{42};
    unexpected dest{cuda::std::move(input)};
    assert(dest.error() == 42);
  }

  { // assignment
    unexpected input{42};
    unexpected dest{1337};
    dest = input;
    assert(dest.error() == 42);
  }

  { // comparison with unexpected
    unexpected lhs{42};
    unexpected rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }

  { // swap
    unexpected lhs{42};
    unexpected rhs{1337};
    lhs.swap(rhs);
    assert(lhs.error() == 1337);
    assert(rhs.error() == 42);

    swap(lhs, rhs);
    assert(lhs.error() == 42);
    assert(rhs.error() == 1337);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
