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
#include <cuda/std/optional>

#include "host_device_types.h"
#include "test_macros.h"

template <class T>
void test()
{
  using optional = cuda::std::optional<T>;
  { // default construction
    optional default_constructed{};
    assert(!default_constructed.has_value());
  }

  if constexpr (!cuda::std::is_reference_v<T>)
  { // in_place zero initialization
    optional in_place_zero_initialization{cuda::std::in_place};
    assert(in_place_zero_initialization.has_value());
    assert(*in_place_zero_initialization == 0);
  }

  cuda::std::remove_reference_t<T> val{42};
  { // in_place initialization
    optional in_place_initialization{cuda::std::in_place, val};
    assert(in_place_initialization.has_value());
    assert(*in_place_initialization == 42);
  }

  { // value initialization
    optional value_initialization{val};
    assert(value_initialization.has_value());
    assert(*value_initialization == 42);
  }

  { // copy construction
    optional input{val};
    optional dest{input};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // move construction
    optional input{val};
    optional dest{cuda::std::move(input)};
    assert(dest.has_value());
    assert(*dest == 42);
  }

  cuda::std::remove_reference_t<T> other_val{1337};
  { // assignment, value to value
    optional input{val};
    optional dest{other_val};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, value to empty
    optional input{val};
    optional dest{};
    dest = input;
    assert(dest.has_value());
    assert(*dest == 42);
  }

  { // assignment, empty to value
    optional input{};
    optional dest{other_val};
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
    optional lhs{val};
    optional rhs{other_val};
    assert(!(lhs == rhs));
    assert(lhs != rhs);
    assert(lhs < rhs);
    assert(lhs <= rhs);
    assert(!(lhs > rhs));
    assert(!(lhs >= rhs));
  }

  { // comparison with type
    optional opt{val};
    assert(opt == host_only_type{val});
    assert(host_only_type{val} == opt);
    assert(opt != host_only_type{other_val});
    assert(host_only_type{other_val} != opt);

    assert(opt < host_only_type{other_val});
    assert(host_only_type{7} < opt);
    assert(opt <= host_only_type{other_val});
    assert(host_only_type{7} <= opt);

    assert(opt > host_only_type{7});
    assert(host_only_type{other_val} > opt);
    assert(opt >= host_only_type{7});
    assert(host_only_type{other_val} >= opt);
  }

  { // swap
    optional lhs{val};
    optional rhs{other_val};
    lhs.swap(rhs);
    assert(*lhs == 1337);
    assert(*rhs == 42);

    swap(lhs, rhs);
    assert(*lhs == 42);
    assert(*rhs == 1337);
  }
}

void test()
{
  test<host_only_type>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<host_only_type&>();
#endif // CCCL_ENABLE_OPTIONAL_REF
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
