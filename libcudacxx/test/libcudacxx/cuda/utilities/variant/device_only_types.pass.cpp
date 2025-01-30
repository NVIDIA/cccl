//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/variant>

#include "host_device_types.h"
#include "test_macros.h"

__device__ void test()
{
  using variant = cuda::std::variant<device_only_type>;
  { // default construction
    variant default_constructed{};
    assert(cuda::std::get<0>(default_constructed) == 0);
  }

  { // value initialization
    variant value_initialization{device_only_type{42}};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // value initialization
    variant value_initialization{42};
    assert(cuda::std::get<0>(value_initialization) == 42);
  }

  { // in_place_type_t initialization
    variant in_place_initialization{cuda::std::in_place_type_t<device_only_type>{}, 42};
    assert(cuda::std::get<0>(in_place_initialization) == 42);
  }

  { // in_place_index_t initialization
    variant in_place_initialization{cuda::std::in_place_index_t<0>{}, 42};
    assert(cuda::std::get<0>(in_place_initialization) == 42);
  }

  { // in_place_type_t initializer_list initialization
    variant init_list_initialization{
      cuda::std::in_place_type_t<device_only_type>{}, cuda::std::initializer_list<int>{}, 42};
    assert(cuda::std::get<0>(init_list_initialization) == 42);
  }

  { // in_place_type_t initializer_list initialization
    variant init_list_initialization{cuda::std::in_place_index_t<0>{}, cuda::std::initializer_list<int>{}, 42};
    assert(cuda::std::get<0>(init_list_initialization) == 42);
  }

  { // copy construction
    variant input{42};
    variant dest{input};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // move construction
    variant input{42};
    variant dest{cuda::std::move(input)};
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // assignment, value to value
    variant input{42};
    variant dest{1337};
    dest = input;
    assert(cuda::std::get<0>(dest) == 42);
  }

  { // emplace
    variant var{42};
    var.emplace<device_only_type>(42);
    assert(cuda::std::get<0>(var) == 42);
  }

  { // emplace
    variant var{42};
    var.emplace<0>(42);
    assert(cuda::std::get<0>(var) == 42);
  }

  { // emplace init list
    variant var{42};
    var.emplace<device_only_type>(cuda::std::initializer_list<int>{}, 42);
    assert(cuda::std::get<0>(var) == 42);
  }

  { // comparison with variant
    variant lhs{42};
    variant rhs{1337};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // swap
    variant lhs{42};
    variant rhs{1337};
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
