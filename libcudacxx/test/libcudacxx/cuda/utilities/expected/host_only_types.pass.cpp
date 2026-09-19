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
#include <cuda/std/expected>
#include <cuda/std/initializer_list>

#include "host_device_types.h"
#include "test_macros.h"

void test()
{
  using expected = cuda::std::expected<host_only_type, host_only_type>;
  { // default construction
    expected default_constructed{};
    assert(default_constructed.has_value());
    assert(*default_constructed == 0);
  }

  { // in_place zero initialization
    expected in_place_zero_initialization{cuda::std::in_place};
    assert(in_place_zero_initialization.has_value());
    assert(*in_place_zero_initialization == 0);
  }

  { // in_place initialization
    expected in_place_initialization{cuda::std::in_place, 42};
    assert(in_place_initialization.has_value());
    assert(*in_place_initialization == 42);
  }

  { // initializer_list initialization
    expected init_list_initialization{cuda::std::in_place, cuda::std::initializer_list<int>{}, 42};
    assert(init_list_initialization.has_value());
    assert(*init_list_initialization == 42);
  }

  { // unexpect zero initialization
    expected in_place_zero_initialization{cuda::std::unexpect};
    assert(!in_place_zero_initialization.has_value());
    assert(in_place_zero_initialization.error() == 0);
  }

  { // unexpect initialization
    expected in_place_initialization{cuda::std::unexpect, 42};
    assert(!in_place_initialization.has_value());
    assert(in_place_initialization.error() == 42);
  }

  { // initializer_list initialization
    expected init_list_initialization{cuda::std::unexpect, cuda::std::initializer_list<int>{}, 42};
    assert(!init_list_initialization.has_value());
    assert(init_list_initialization.error() == 42);
  }

  { // value initialization
    expected value_initialization{42};
    assert(value_initialization.has_value());
    assert(*value_initialization == 42);
  }

  { // copy construction
    expected input{42};
    expected dest{input};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // move construction
    expected input{42};
    expected dest{cuda::std::move(input)};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to value
    expected input{42};
    expected dest{1337};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to empty
    expected input{42};
    expected dest{};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, empty to value
    expected input{};
    expected dest{1337};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 0);
  }

  { // assignment, empty to empty
    expected input{};
    expected dest{};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 0);
  }

  { // assignment, error to value
    expected input{cuda::std::unexpect, 42};
    expected dest{1337};
    dest = input;
    assert(!dest.has_value());
    assert(dest.error() == 42);
  }

  { // assignment, value to error
    expected input{42};
    expected dest{cuda::std::unexpect, 1337};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, error to error
    expected input{cuda::std::unexpect, 42};
    expected dest{cuda::std::unexpect, 1337};
    dest = input;
    assert(!dest.has_value());
    assert(dest.error() == 42);
  }

  { // comparison with expected with value
    expected lhs{42};
    expected rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }

  { // comparison with expected with error
    expected lhs{cuda::std::unexpect, 42};
    expected rhs{cuda::std::unexpect, 1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
  }

  { // comparison with type and value
    expected expect{42};
    assert(expect == host_only_type{42});
    assert(host_only_type{42} == expect);
    assert(expect != host_only_type{1337});
    assert(host_only_type{1337} != expect);
  }

  { // comparison with type and error
    expected expect{cuda::std::unexpect, 42};
    assert(expect == cuda::std::unexpected<host_only_type>{42});
    assert(cuda::std::unexpected<host_only_type>{42} == expect);
    assert(expect != cuda::std::unexpected<host_only_type>{1337});
    assert(cuda::std::unexpected<host_only_type>{1337} != expect);
  }

  { // swap
    expected lhs{42};
    expected rhs{1337};
    lhs.swap(rhs);
    assert(*lhs == 1337);
    assert(*rhs == 42);

    swap(lhs, rhs);
    assert(*lhs == 42);
    assert(*rhs == 1337);
  }

  { // swap cross error
    expected lhs{42};
    expected rhs{cuda::std::unexpect, 1337};
    lhs.swap(rhs);
    assert(lhs.error() == 1337);
    assert(*rhs == 42);

    swap(lhs, rhs);
    assert(*lhs == 42);
    assert(rhs.error() == 1337);
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
