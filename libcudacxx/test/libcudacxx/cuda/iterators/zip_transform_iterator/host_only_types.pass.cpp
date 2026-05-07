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

#include <vector>

#include "test_macros.h"

struct host_plus_value
{
  int value_;

  host_plus_value(const int value = 42) noexcept
      : value_(value)
  {}

  host_plus_value(const host_plus_value& other) noexcept
      : value_(other.value_)
  {}

  host_plus_value(host_plus_value&& other) noexcept
      : value_(other.value_)
  {}

  host_plus_value& operator=(const host_plus_value& other) noexcept
  {
    value_ = other.value_;
    return *this;
  }

  host_plus_value& operator=(host_plus_value&& other) noexcept
  {
    value_ = other.value_;
    return *this;
  }

  int operator()(const int& val) const noexcept
  {
    return val + value_;
  }

  ~host_plus_value() noexcept {}
};

void test()
{
  std::vector<int> vec{1, 2, 3, 4};
  cuda::zip_transform_iterator iter{host_plus_value{42}, vec.begin()};
  assert(iter[1] == vec[1] + 42);
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
