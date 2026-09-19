//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "host_device_types.h"
#include "test_macros.h"

void test()
{
  using pair = cuda::std::pair<host_only_type, host_only_type>;
  { // default construction
    pair default_constructed{};
    assert(default_constructed.first == 0);
    assert(default_constructed.second == 0);
  }

  { // value initialization
    pair value_initialization{host_only_type{42}, host_only_type{1337}};
    assert(value_initialization.first == 42);
    assert(value_initialization.second == 1337);
  }

  { // value initialization
    pair value_initialization{42, 1337};
    assert(value_initialization.first == 42);
    assert(value_initialization.second == 1337);
  }

  { // copy construction
    pair input{42, 1337};
    pair dest{input};
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // move construction
    pair input{42, 1337};
    pair dest{cuda::std::move(input)};
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // assignment, value to value
    pair input{42, 1337};
    pair dest{1337, 42};
    dest = input;
    assert(dest.first == 42);
    assert(dest.second == 1337);
  }

  { // comparison with pair
    pair lhs{42, 1337};
    pair rhs{1337, 42};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // swap
    pair lhs{42, 1337};
    pair rhs{1337, 42};
    lhs.swap(rhs);
    assert(lhs.first == 1337);
    assert(lhs.second == 42);
    assert(rhs.first == 42);
    assert(rhs.second == 1337);

    swap(lhs, rhs);
    assert(lhs.first == 42);
    assert(lhs.second == 1337);
    assert(rhs.first == 1337);
    assert(rhs.second == 42);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
