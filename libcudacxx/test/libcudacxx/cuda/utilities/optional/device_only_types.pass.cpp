//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// We cannot suppress execution checks in cuda::std::construct_at
// XFAIL: c++20 && !nvrtc && nvcc && !msvc
// UNSUPPORTED: clang-14

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "host_device_types.h"
#include "test_macros.h"

__device__ void test()
{
  using optional = cuda::std::optional<device_only_type>;
  { // default construction
    optional default_constructed{};
    assert(!default_constructed.has_value());
  }

  { // in_place zero initialization
    optional in_place_zero_initialization{cuda::std::in_place};
    assert(in_place_zero_initialization.has_value());
    assert(*in_place_zero_initialization == 0);
  }

  { // in_place initialization
    optional in_place_initialization{cuda::std::in_place, 42};
    assert(in_place_initialization.has_value());
    assert(*in_place_initialization == 42);
  }

  { // value initialization
    optional value_initialization{42};
    assert(value_initialization.has_value());
    assert(*value_initialization == 42);
  }

  { // copy construction
    optional input{42};
    optional dest{input};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // move construction
    optional input{42};
    optional dest{cuda::std::move(input)};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to value
    optional input{42};
    optional dest{1337};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to empty
    optional input{42};
    optional dest{};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, empty to value
    optional input{};
    optional dest{1337};
    dest = input;
    assert(!dest.has_value());
  }

  { // assignment, empty to empty
    optional input{};
    optional dest{};
    dest = input;
    assert(!dest.has_value());
  }

  { // comparison with optional
    optional lhs{42};
    optional rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // comparison with type
    optional opt{42};
    assert(opt == device_only_type{42});
    assert(device_only_type{42} == opt);
    assert(opt != device_only_type{1337});
    assert(device_only_type{1337} != opt);

    assert(opt < device_only_type{1337});
    assert(device_only_type{7} < opt);
    assert(opt <= device_only_type{1337});
    assert(device_only_type{7} <= opt);

    assert(opt > device_only_type{7});
    assert(device_only_type{1337} > opt);
    assert(opt >= device_only_type{7});
    assert(device_only_type{1337} >= opt);
  }

  { // swap
    optional lhs{42};
    optional rhs{1337};
    lhs.swap(rhs);
    assert(*lhs == 1337);
    assert(*rhs == 42);

    swap(lhs, rhs);
    assert(*lhs == 42);
    assert(*rhs == 1337);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
  return 0;
}
